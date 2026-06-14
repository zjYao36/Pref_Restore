from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from diffusers import AutoencoderDC
from blip3o.model.blip3o_arch import blip3oMetaForCausalLM, blip3oMetaModel
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from blip3o.utils import rank0_print


class blip3oQwenConfigVAE(Qwen3Config):
    model_type = "blip3o_qwen"

class blip3oQwenModel(blip3oMetaModel, Qwen3Model):
    config_class = blip3oQwenConfigVAE

    def __init__(self, config: Qwen3Config):
        super(blip3oQwenModel, self).__init__(config)

class blip3oQwenForCausalLMVAE(Qwen3ForCausalLM, blip3oMetaForCausalLM):
    config_class = blip3oQwenConfigVAE

    def __init__(self, config):
        Qwen3ForCausalLM.__init__(self, config)
        config.model_type = "blip3o_qwen"
        config.rope_scaling = None

        self.model = blip3oQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # 在初始化完成后，创建固定的VAE副本
        self.fixed_vae_path = config.diffusion_name_or_path


    def get_or_create_fixed_vae(self):
        """懒加载：获取或创建固定的VAE"""
        if not hasattr(self, 'fixed_vae') or self.fixed_vae is None:
            self.fixed_vae = AutoencoderDC.from_pretrained(
                self.fixed_vae_path, 
                subfolder="vae", 
                torch_dtype=torch.bfloat16
            )
            # 冻结fixed_vae的所有参数
            for param in self.fixed_vae.parameters():
                param.requires_grad = False
            self.fixed_vae.eval()
        # 确保fixed_vae在正确的设备上
        model_device = next(self.model.parameters()).device
        if self.fixed_vae.device != model_device:
            self.fixed_vae = self.fixed_vae.to(model_device)
        return self.fixed_vae
    
    def get_model(self):
        return self.model

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.model.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.model.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def mask_drop(self, latents, drop_prob=0.1):
        if drop_prob <= 0:
            return latents
        mask = torch.bernoulli(torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype) + drop_prob)
        while len(mask.shape) < len(latents.shape):
            mask = mask.unsqueeze(-1)
        mask = 1 - mask  # need to flip 0 <-> 1
        return latents * mask


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


        if inputs_embeds is None: # 主要获得position_ids, attention_mask, new_input_embeds, new_labels  （返回的input_ids=None）
            # import ipdb; ipdb.set_trace()
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values, # kv cache
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions, # attentions - 所有层的注意力权重（可选）
            output_hidden_states=output_hidden_states, # hidden_states - 所有层的隐藏状态
            return_dict=return_dict,
        ) # BaseModelOutputWithPast(last_hidden_state=, past_key_values=None, hidden_states=None, attentions=None)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states) # torch.Size([bsz, len, 217210])
        
        '''
        LLM freeze了, 所以不用算 cross-entropy loss
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        '''

        
        vae = self.model.get_sana_vae()
        sana = self.model.get_sana()

        
        if detailed_conditions is not None:
            degraded_latents = vae.encode(detailed_conditions).latent #  ([bsz, 32, 16, 16])
            if "shift_factor" in vae.config and vae.config.shift_factor is not None:
                degraded_latents = degraded_latents - vae.config.shift_factor
            degraded_latents = degraded_latents * vae.config.scaling_factor #  ([bsz, 32, 16, 16])
            degraded_latents_for_mse = degraded_latents.clone() # step3: mse loss
            degraded_latents = sana.patch_embed(degraded_latents) #  ([bsz, 256, 2240])
            degraded_latents = self.model.vae_connector(degraded_latents) #  ([bsz, 256, 2304])

        if target_images is not None:    
            # latents = vae.encode(target_images).latent
            # if "shift_factor" in vae.config and vae.config.shift_factor is not None:
            #     latents = latents - vae.config.shift_factor
            # latents = latents * vae.config.scaling_factor

            ####### step3: mse loss #######
            fixed_vae = self.get_or_create_fixed_vae()  # 懒加载
            with torch.no_grad():
                latents = fixed_vae.encode(target_images).latent
            if "shift_factor" in fixed_vae.config and fixed_vae.config.shift_factor is not None:
                latents = latents - fixed_vae.config.shift_factor
            latents = latents * fixed_vae.config.scaling_factor

            latents_for_mse = latents  # 不需要detach，已经在no_grad下
            vae_mse_loss = torch.mean((latents_for_mse - degraded_latents_for_mse) ** 2)
            rank0_print(f" VAE MSE loss {vae_mse_loss} ")
            ###############################
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
            selected_hidden_states = self.model.diffusion_connector(selected_hidden_states) # ([bsz, 369, 2304])
            
            # concate high-level text features and detailed degraded image latents
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
                (weighting.float() * (diffusion_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            diff_loss = diff_loss.mean()

            # rank0_print(f" Cross-entropy loss {loss}, Diffusion loss {diff_loss} ")
            # loss += diff_loss
            rank0_print(f" Diffusion loss {diff_loss} ")
            loss = diff_loss

            ####### step3: mse loss #######
            loss += vae_mse_loss
            ###############################



        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)



    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("blip3o_qwen", blip3oQwenConfigVAE)
AutoModelForCausalLM.register(blip3oQwenConfigVAE, blip3oQwenForCausalLMVAE)
