"""
Combined model for ablation study.
Trains all three stages simultaneously with all three losses:
  - CrossEntropy loss  (from Stage 1: LLM token prediction)
  - Diffusion loss     (from Stage 2: flow-matching on SANA)
  - VAE MSE loss       (from Stage 3: VAE encoder alignment)

Trainable modules (union of all stages):
  - Qwen LM layers + embeddings      (Stage 1)
  - diffusion_connector              (Stage 1)
  - sana.caption_projection          (Stage 1)
  - sana.transformer_blocks          (Stage 2)
  - vae_connector                    (Stage 2)
  - sana_vae.encoder                 (Stage 3)
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from diffusers import AutoencoderDC

from blip3o.model.blip3o_arch import blip3oMetaForCausalLM, blip3oMetaModel
from blip3o.model.language_model.blip3o_qwen_vae import blip3oQwenForCausalLMVAE, blip3oQwenConfigVAE
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from blip3o.utils import rank0_print


class blip3oQwenForCausalLMCombined(blip3oQwenForCausalLMVAE):
    """
    Ablation model: merges Stage 1 + Stage 2 + Stage 3 into a single training pass.
    Loss = w_ce * CE  +  w_diff * Diffusion  +  w_vae * VAE_MSE
    Default weights are all 1.0; override via config attributes
    `ce_loss_weight`, `diff_loss_weight`, `vae_loss_weight` if needed.
    """

    config_class = blip3oQwenConfigVAE

    def get_or_create_fixed_vae(self):
        """
        Override parent's lazy loader to use object.__setattr__, which bypasses
        nn.Module's __setattr__ and prevents fixed_vae from being registered as
        a submodule. This avoids the DeepSpeed ZeRO checkpoint error:
          'failed to find frozen {param} in named params'
        which occurs when a module is registered *after* DeepSpeed builds its
        internal parameter map (i.e. after the first forward pass).
        """
        if not hasattr(self, '_fixed_vae') or self._fixed_vae is None:
            vae = AutoencoderDC.from_pretrained(
                self.fixed_vae_path,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
            )
            for param in vae.parameters():
                param.requires_grad = False
            vae.eval()
            # Use object.__setattr__ so PyTorch does NOT register this as a
            # submodule — it won't appear in named_parameters() or state_dict().
            object.__setattr__(self, '_fixed_vae', vae)

        model_device = next(self.model.parameters()).device
        if next(self._fixed_vae.parameters()).device != model_device:
            object.__setattr__(self, '_fixed_vae', self._fixed_vae.to(model_device))
        return self._fixed_vae

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        target_images: Optional[torch.FloatTensor] = None,
        detailed_conditions: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # ------------------------------------------------------------------ #
        # 1. LLM forward pass
        # ------------------------------------------------------------------ #
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values,
             inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values,
                labels, images, modalities, image_sizes)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # ------------------------------------------------------------------ #
        # 2. Cross-Entropy loss  (Stage 1)
        # ------------------------------------------------------------------ #
        loss = None
        ce_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            ce_loss = loss_fct(shift_logits, shift_labels)
            loss = ce_loss

        # ------------------------------------------------------------------ #
        # 3. VAE MSE loss + Diffusion loss  (Stage 2 & 3)
        # ------------------------------------------------------------------ #
        vae_mse_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        diff_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        if detailed_conditions is not None and target_images is not None:
            vae = self.model.get_sana_vae()
            sana = self.model.get_sana()

            # --- encode degraded image with *trainable* VAE encoder --- #
            degraded_latents = vae.encode(detailed_conditions).latent
            if "shift_factor" in vae.config and vae.config.shift_factor is not None:
                degraded_latents = degraded_latents - vae.config.shift_factor
            degraded_latents = degraded_latents * vae.config.scaling_factor
            degraded_latents_for_mse = degraded_latents.clone()  # for VAE MSE loss

            # patch embed + vae_connector -> detailed condition for SANA
            degraded_latents = sana.patch_embed(degraded_latents)
            degraded_latents = self.model.vae_connector(degraded_latents)

            # --- encode target image with *fixed* VAE (no_grad) --- #
            fixed_vae = self.get_or_create_fixed_vae()
            with torch.no_grad():
                latents = fixed_vae.encode(target_images).latent
            if "shift_factor" in fixed_vae.config and fixed_vae.config.shift_factor is not None:
                latents = latents - fixed_vae.config.shift_factor
            latents = latents * fixed_vae.config.scaling_factor

            # VAE MSE loss (Stage 3): align degraded latent toward clean latent
            vae_mse_loss = torch.mean((latents - degraded_latents_for_mse) ** 2)

            # --- Diffusion (flow matching) loss (Stage 2) --- #
            noise = torch.randn_like(latents, device=latents.device)
            weighting_scheme = "uniform"
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=latents.shape[0],
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.model.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.model.noise_scheduler.timesteps[indices].to(device=latents.device)
            sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            start_pos = (labels == self.config.image_start_tag_id).float().argmax(dim=1)
            end_pos = (labels == self.config.image_end_tag_id).float().argmax(dim=1)

            selected_hidden_states = []
            for b in range(hidden_states.size(0)):
                start = start_pos[b].item() + 1
                end = end_pos[b].item()
                hidden_states_filter = hidden_states[b, start:end, :]
                if hidden_states_filter.size(1) != 730:
                    hidden_states_filter = hidden_states[b, -730:, :]
                selected_hidden_states.append(hidden_states_filter)
            selected_hidden_states = torch.stack(selected_hidden_states, dim=0)
            selected_hidden_states = self.model.diffusion_connector(selected_hidden_states)

            # concat high-level text features and detailed degraded image latents
            encoder_hidden_states = torch.cat([selected_hidden_states, degraded_latents], dim=1)

            diffusion_pred = sana(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=self.mask_drop(encoder_hidden_states),
                encoder_attention_mask=None,
            ).sample

            target = noise - latents
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
            diff_loss = torch.mean(
                (weighting.float() * (diffusion_pred.float() - target.float()) ** 2
                 ).reshape(target.shape[0], -1),
                1,
            ).mean()

            # --- loss weights (default 1.0, can be overridden in config) --- #
            w_ce   = getattr(self.config, "ce_loss_weight",   1.0)
            w_diff = getattr(self.config, "diff_loss_weight", 1.0)
            w_vae  = getattr(self.config, "vae_loss_weight",  1.0)

            rank0_print(
                f" [Combined] CE loss {ce_loss:.4f}  "
                f"Diffusion loss {diff_loss:.4f}  "
                f"VAE MSE loss {vae_mse_loss:.4f}"
            )

            loss = w_ce * ce_loss + w_diff * diff_loss + w_vae * vae_mse_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
