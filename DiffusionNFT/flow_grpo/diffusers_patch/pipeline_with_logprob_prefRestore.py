# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .solver import run_sampling
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers import DDPMScheduler, DDIMScheduler, LCMScheduler, FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.image_processor import PipelineImageInput

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# images, latents, _ = pipeline_with_logprob( # images是输出的图片， latents是每个图片对应的去噪过程中间变量. nft用不到中间的东西，所以第三个不要了！
#     pipeline,
#     image=ref_images,
#     prompt=prompts,
#     negative_prompt=[""] * len(prompts),
#     num_inference_steps=config.sample.num_steps, # 40
#     guidance_scale=config.sample.guidance_scale, # 1.0
#     output_type="pt",
#     height=config.resolution,
#     width=config.resolution,
#     noise_level=config.sample.noise_level, # 0.7
#     deterministic=config.sample.deterministic, # True
#     solver=config.sample.solver, # "dpm2"
# )

@torch.no_grad()
def pipeline_with_logprob(
    self,
    image: Optional[PipelineImageInput] = None,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 2.0,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    noise_level: float = 0.7,
    deterministic: bool = False,
    solver: str = "flow",
    max_new_tokens: Optional[int] = 729, # for blip-3o Next
    top_p: float = 0.95,
    top_k: int = 1200,
):
    height = height
    width = width
    

    # 1. Check inputs. Raise error if not correct
    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs # None
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # device = self._execution_device

    # lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    
    
    # 3. Encode input prompt and condition images
    # 处理图像
    processed_image, image_size, original_image = self._process_image(image) # processed_image [tensor(3, 384, 384), tensor(3, 384, 384), tensor(3, 384, 384), ...] batch_size 个元素
    detailed_condition = self._process_target_image(original_image) # detailed_condition [tensor(3, 512, 512), tensor(3, 512, 512), tensor(3, 512, 512), ...] batch_size 个元素
    # 准备消息
    messages = [
        {"from": "human", "value": "<image>\nPlease reconstruct the given image."},
        {"from": "gpt", "value": f"<im_start><S{self.config.scale}>"}
    ]
    # 预处理输入
    data_dict = self._preprocess_qwen([messages], has_image=True)
    inputs = data_dict['input_ids']
    # inputs 只代表一个数据，但是processed_image和detailed_condition是一个batch的数据
    # 需要扩展inputs以匹配batch大小
    inputs = inputs.repeat(batch_size, 1) # torch.Size([8, 29]) 所有都用固定的提示词.加入提示词之后，这里就需要参考Dataset里面collator的方法进行padding了！！！！！！！！！！！！！！！！！

    
    # 4. Prepare latent variables
    latents, pred_latents, pred_latents_cond = self.model.prepare_latents(
        inputs.to(self.device),
        images=processed_image, # image for Qwen
        detailed_conditions=detailed_condition, # image for VAE
        modalities=["image"] * batch_size,
        max_new_tokens=self.config.seq_len,
        do_sample=False,
        top_p=top_p,
        top_k=top_k,
        num_inference_steps=num_inference_steps,
    )

    # num_channels_latents = self.transformer.config.in_channels
    # latents = self.prepare_latents(
    #     batch_size * num_images_per_prompt,
    #     num_channels_latents,
    #     height,
    #     width,
    #     prompt_embeds.dtype,
    #     device,
    #     generator,
    #     latents,
    # )

    

    # 5. Prepare timesteps
    # set step values
    if isinstance(self.model.model.noise_scheduler, FlowMatchEulerDiscreteScheduler):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        self.model.model.noise_scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        self.model.model.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
    else:
        self.model.model.noise_scheduler.set_timesteps(num_inference_steps)

    # if not flux:
    #     timesteps, num_inference_steps = retrieve_timesteps(
    #         self.scheduler,
    #         num_inference_steps,
    #         device,
    #         sigmas=None,
    #     )
    #     self._num_timesteps = len(timesteps)
    # else:
    #     sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    #     if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
    #         sigmas = None
    #     image_seq_len = latents.shape[1]
    #     mu = calculate_shift(
    #         image_seq_len,
    #         self.scheduler.config.get("base_image_seq_len", 256),
    #         self.scheduler.config.get("max_image_seq_len", 4096),
    #         self.scheduler.config.get("base_shift", 0.5),
    #         self.scheduler.config.get("max_shift", 1.15),
    #     )
    #     timesteps, num_inference_steps = retrieve_timesteps(
    #         self.scheduler,
    #         num_inference_steps,
    #         device,
    #         sigmas=sigmas,
    #         mu=mu,
    #     )
    #     self._num_timesteps = len(timesteps)

    # sigmas = self.scheduler.sigmas.float()



    def v_pred_fn(z, sigma, pred_latents):
        latent_model_input = torch.cat([z] * 2)
        latent_model_input = latent_model_input.to(pred_latents.dtype)

        timesteps = torch.full([latent_model_input.shape[0]], sigma * 1000, device=z.device, dtype=torch.long)
        t = timesteps

        if hasattr(self.model.model.noise_scheduler.timesteps, "scale_model_input"): # 没过这个逻辑，确定正常不正常
            latent_model_input = self.model.model.noise_scheduler.scale_model_input(latent_model_input, t)
        # predict noise model_output
        noise_pred = self.model.model.sana(
            hidden_states=latent_model_input,
            # encoder_hidden_states=self.model.diffusion_connector(pred_latents),
            encoder_hidden_states=pred_latents,
            timestep=t.to(latents.device),
            encoder_attention_mask=None
        ).sample


        noise_pred_uncond, noise_pred = noise_pred.chunk(2)

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

        return noise_pred

        # latent_model_input = torch.cat([z] * 2) if self.do_classifier_free_guidance else z
        # # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps = torch.full([latent_model_input.shape[0]], sigma * 1000, device=z.device, dtype=torch.long)
        # noise_pred = self.transformer(
        #     hidden_states=latent_model_input,
        #     timestep=timesteps,
        #     encoder_hidden_states=prompt_embeds,
        #     pooled_projections=pooled_prompt_embeds,
        #     joint_attention_kwargs=self.joint_attention_kwargs,
        #     return_dict=False,
        # )[0]
        # noise_pred = noise_pred.to(prompt_embeds.dtype)
        # # perform guidance
        # if self.do_classifier_free_guidance:
        #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # return noise_pred

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []

    # 7. Denoising loop
    latents, all_latents, all_log_probs = run_sampling(v_pred_fn, latents, sigmas, solver, deterministic, noise_level, pred_latents)
    
    image = self.model.decode_latents_nft(latents.to(self.model.model.sana_vae.dtype), return_tensor=True)

    return image, all_latents, pred_latents_cond, all_log_probs

    if flux:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not flux:
        return image, all_latents, all_log_probs
    else:
        return image, all_latents, latent_image_ids, text_ids, all_log_probs
