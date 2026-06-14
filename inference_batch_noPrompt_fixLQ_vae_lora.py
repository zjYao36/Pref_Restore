import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
# === 这里保持你原来的 import ===
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer
from blip3o.model import *
from blip3o.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from blip3o.data.image_degradation import degrade_image
from torchvision.transforms import v2
# LoRA imports
from peft import LoraConfig, get_peft_model

# === 保持你定义的参数 ===
degradation_params = {
    'gt_size': 512,
    'in_size': 512,
    'use_motion_kernel': False,
    'blur_kernel_size': 41,
    'blur_sigma': [1, 15],
    'downsample_range': [4, 30],
    'noise_range': [0, 20],
    'jpeg_range': [30, 80]
}

## target transform for sana
target_transform = v2.Compose(
    [
        v2.Resize(512),
        v2.CenterCrop(512),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ]
    )

@dataclass
class T2IConfig:
    model_path: str = "/data/phd/yaozhengjian/zjYao_Exprs/BLIP-3o-next/Face-Restore_restoration-FFHQ+CelebA/checkpoint-30800"
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    # generation config
    scale: int = 0  
    seq_len: int = 729  
    top_p: float = 0.95
    top_k: int = 1200
    # LoRA config
    lora_path: str = None
    use_lora: bool = False


class TextToImageInference:
    def __init__(self, config: T2IConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._load_models()
        self.processor = self.model.get_vision_tower().image_processor
        
    def _load_models(self):
        self.model = blip3oQwenForInferenceLMVAE.from_pretrained(
            self.config.model_path, torch_dtype=self.config.dtype
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        # Add LoRA if specified
        if self.config.use_lora and self.config.lora_path:
            self._add_lora()
    
    def _add_lora(self):
        """Add LoRA to the sana model"""
        target_modules = [
            "attn1.to_k",
            "attn1.to_out.0", 
            "attn1.to_q",
            "attn1.to_v",
            "attn2.to_k",
            "attn2.to_out.0",
            "attn2.to_q", 
            "attn2.to_v",
        ]
        
        transformer_lora_config = LoraConfig(
            r=32, 
            lora_alpha=64, 
            init_lora_weights="gaussian", 
            target_modules=target_modules
        )
        
        # Apply LoRA to the sana model
        self.model.model.sana = get_peft_model(self.model.model.sana, transformer_lora_config)
        
        # Load LoRA weights if path exists
        lora_path = os.path.join(self.config.lora_path, "lora")
        if os.path.exists(lora_path):
            self.model.model.sana.load_adapter(lora_path, adapter_name="default")
            print(f"LoRA weights loaded from: {lora_path}")
        else:
            print(f"LoRA path not found: {lora_path}, using initialized weights")

    def process_image(self, image):
        image_size = image.size
        image = self.processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size
    
    def preprocess_qwen(self, sources, tokenizer, has_image: bool = True, max_len=2048,
                        system_message: str = "You are a helpful assistant."):
        roles = {"human": "user", "gpt": "assistant"}

        if 'image_token_index' not in globals():
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            global image_token_index
            image_token_index = tokenizer.convert_tokens_to_ids("<image>") 

        im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
        unmask_tokens_idx = [198, im_start, im_end]
        chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        tokenizer.chat_template = chat_template

        input_ids, targets = [], []
        for source in sources:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id, target = [], []
            input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
            target += input_id

            for conv in source:
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]
                role = roles.get(role, role)
                conv = [{"role": role, "content": content}]
                encode_id = tokenizer.apply_chat_template(conv)

                if role == roles["human"]:
                    input_id += encode_id
                    target += encode_id
                else:
                    input_id += encode_id[:-2]
                    target += encode_id[:-2]

            assert len(input_id) == len(target)
            for idx, encode_id in enumerate(input_id):
                if encode_id in unmask_tokens_idx:
                    target[idx] = encode_id
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX

            input_ids.append(input_id)
            targets.append(target)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        return dict(input_ids=input_ids, labels=targets)

    def process_target_image(self, image):
        image = target_transform(image)
        return image
    
    def generate_image(self, prompt: str, image_file: str) -> Image.Image:
        image = Image.open(image_file).convert("RGB")
        degraded_image = image
        image, _ = self.process_image(degraded_image)
        detailed_condition = self.process_target_image(degraded_image)

        messages = [
            {"from": "human", "value": "<image>\nPlease reconstruct the given image."},
            {"from": "gpt", "value": f"<im_start><S{self.config.scale}>"}
        ]

        data_dict = self.preprocess_qwen([messages], self.tokenizer, has_image=True)
        inputs = data_dict['input_ids']

        output_image = self.model.generate_images_from_image(
            inputs.to(self.device),
            images=[image],
            detailed_conditions=[detailed_condition],
            max_new_tokens=self.config.seq_len,
            do_sample=False,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )
        return degraded_image, output_image[0]


def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Batch image generation with degradation")
    parser.add_argument("--model_path", type=str, 
                       default='/data/phd/yaozhengjian/zjYao_Exprs/BLIP-3o-next/Face-Restore_restoration/checkpoint-34640',
                       help="Path to the model checkpoint")
    parser.add_argument("--json_path", type=str,
                       default="/data/zgq/yaozhengjian/Datasets/FFHQ_val/CelebA_HQ/captions.json",
                       help="Path to the JSON dataset file")
    parser.add_argument("--output_dir", type=str,
                       default="/data/phd/yaozhengjian/zjYao_Exprs/BLIP-3o-next/Eval/FR-FFHQ-heavy",
                       help="Output directory for generated images")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="Path to LoRA weights directory")
    parser.add_argument("--use_lora", action="store_true",
                       help="Whether to use LoRA")
    
    args = parser.parse_args()
    
    config = T2IConfig()
    config.model_path = args.model_path
    config.lora_path = args.lora_path
    config.use_lora = args.use_lora
    inference = TextToImageInference(config)

    # === 读取 JSON 文件 ===
    with open(args.json_path, "r") as f:
        dataset = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # tqdm 进度条
    for idx, sample in enumerate(tqdm(dataset, desc="Generating images")):
        image_file = sample["image"]
        prompt = sample["caption"]

        try:
            degraded_image, image_sana = inference.generate_image(prompt, image_file)
            base_name = os.path.splitext(os.path.basename(sample["image"]))[0]
            # 保存复原后的图像
            save_path = os.path.join(args.output_dir, "restored", f"{base_name}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image_sana.save(save_path)
            # 保存降质后的图像
            # degraded_save_path = os.path.join(args.output_dir, "degraded", f"{base_name}.png")
            # os.makedirs(os.path.dirname(degraded_save_path), exist_ok=True)
            # degraded_image.save(degraded_save_path)
            # 打印保存路径
            tqdm.write(f"Saved: {save_path}")  # 不打乱进度条
            # tqdm.write(f"Saved degraded: {degraded_save_path}")
        except Exception as e:
            tqdm.write(f"Error processing {image_file}: {e}")


if __name__ == "__main__":
    main()
