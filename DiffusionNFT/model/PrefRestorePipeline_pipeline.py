"""
Text-to-Image Pipeline for ART-FR Model
类似于 diffusers pipeline 的接口，方便调用 TextToImageInference
"""
import os
import sys
import torch
from PIL import Image
from dataclasses import dataclass
from typing import Union, Optional, List
from torchvision.transforms import v2
from transformers import AutoTokenizer

# 假设这些是您项目中的模块，需要根据实际路径调整
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from blip3o.model.language_model.blip3o_qwen_inference_vae import blip3oQwenForInferenceLMVAE


IMAGE_TOKEN_INDEX = -200


@dataclass
class PipelineConfig:
    """Pipeline 配置类"""
    model_path: str = "/data/phd/yaozhengjian/zjYao_Exprs/BLIP-3o-next/Face-Restore_restoration-FFHQ+CelebA/checkpoint-30800"
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    # 生成配置
    scale: int = 0  
    seq_len: int = 729  
    top_p: float = 0.95
    top_k: int = 1200
    # 图像处理配置
    image_size: int = 512


class PrefRestorePipeline:
    """
    类似 diffusers pipeline 的文本到图像生成管道
    
    使用示例:
    ```python
    from DiffusionNFT.model.text_to_image_pipeline import PrefRestorePipeline
    
    # 初始化管道
    pipeline = PrefRestorePipeline.from_pretrained(model_path="your_model_path")
    
    # 生成图像
    result = pipeline(
        prompt="Please reconstruct the given image.",
        image="path/to/image.jpg",
        num_inference_steps=50
    )
    
    # 保存结果
    result.images[0].save("output.jpg")
    ```
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._setup_transforms()
        self._load_models()
        
    @classmethod
    def from_pretrained(
        cls, 
        model_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):
        """从预训练模型创建管道"""
        config = PipelineConfig(
            model_path=model_path,
            device=device,
            dtype=dtype,
            **kwargs
        )
        return cls(config)
    
    def _setup_transforms(self):
        """设置图像变换"""
        self.target_transform = v2.Compose([
            v2.Resize(self.config.image_size),
            v2.CenterCrop(self.config.image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ])
        
    def _load_models(self):
        """加载模型和分词器"""
        self.model = blip3oQwenForInferenceLMVAE.from_pretrained(
            self.config.model_path, 
            torch_dtype=self.config.dtype
        ).to(self.device)
        self.transformer = self.model.model.sana.transformer_blocks
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.processor = self.model.get_vision_tower().image_processor
        
    def _process_image(self, image: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> tuple:
        """处理输入图像（支持单张图片或图片列表）"""
        if isinstance(image, list):
            # 处理图片列表
            processed_images = []
            image_sizes = []
            original_images = []
            
            for img in image:
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                elif not isinstance(img, Image.Image):
                    raise ValueError("Each image must be a PIL Image or file path")
                
                image_size = img.size
                processed_image = self.processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
                
                processed_images.append(processed_image)
                image_sizes.append(image_size)
                original_images.append(img)
            
            return processed_images, image_sizes, original_images
        else:
            # 处理单张图片（保持原有逻辑）
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError("image must be a PIL Image or file path")
                
            image_size = image.size
            processed_image = self.processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            return processed_image, image_size, image
    
    def _process_target_image(self, image: Union[Image.Image, List[Image.Image]]):
        """处理目标图像（支持单张图片或图片列表）"""
        if isinstance(image, list):
            # 处理图片列表
            processed_images = []
            for img in image:
                if not isinstance(img, Image.Image):
                    raise ValueError("Each target image must be a PIL Image")
                processed_img = self.target_transform(img)
                processed_images.append(processed_img)
            return processed_images
        else:
            # 处理单张图片
            if not isinstance(image, Image.Image):
                raise ValueError("Target image must be a PIL Image")
            return self.target_transform(image)
    
    def _preprocess_qwen(self, sources, has_image: bool = True, max_len=2048,
                        system_message: str = "You are a helpful assistant."):
        """预处理 Qwen 输入"""
        roles = {"human": "user", "gpt": "assistant"}

        if 'image_token_index' not in globals():
            self.tokenizer.add_tokens(["<image>"], special_tokens=True)
            global image_token_index
            image_token_index = self.tokenizer.convert_tokens_to_ids("<image>") 

        im_start, im_end = self.tokenizer.additional_special_tokens_ids[:2]
        unmask_tokens_idx = [198, im_start, im_end]
        chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        self.tokenizer.chat_template = chat_template

        input_ids, targets = [], []
        for source in sources:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id, target = [], []
            input_id += self.tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
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
                encode_id = self.tokenizer.apply_chat_template(conv)

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
    
    def __call__(
        self,
        prompt: str = "Please reconstruct the given image.",
        image: Union[str, Image.Image] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs
    ):
        """
        生成图像的主要接口
        
        Args:
            prompt: 文本提示
            image: 输入图像（路径或PIL Image对象）
            num_inference_steps: 推理步数（暂未使用，为了兼容性保留）
            guidance_scale: 引导尺度（暂未使用，为了兼容性保留）
            **kwargs: 其他参数
            
        Returns:
            包含生成图像的结果对象
        """
        if image is None:
            raise ValueError("image is required for this pipeline")
            
        # 处理图像
        processed_image, image_size, original_image = self._process_image(image)
        detailed_condition = self._process_target_image(original_image)
        
        # 准备消息
        messages = [
            {"from": "human", "value": "<image>\nPlease reconstruct the given image."},
            {"from": "gpt", "value": f"<im_start><S{self.config.scale}>"}
        ]
        
        # 预处理输入
        data_dict = self._preprocess_qwen([messages], has_image=True)
        inputs = data_dict['input_ids']
        
        # 生成图像
        with torch.no_grad():
            output_images = self.model.generate_images_from_image(
                inputs.to(self.device),
                images=[processed_image],
                detailed_conditions=[detailed_condition],
                max_new_tokens=self.config.seq_len,
                do_sample=True,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        
        # 返回结果
        return PipelineResult(
            images=output_images,
            original_image=original_image,
            prompt=prompt
        )


class PipelineResult:
    """管道结果类"""
    def __init__(self, images: List[Image.Image], original_image: Image.Image, prompt: str):
        self.images = images
        self.original_image = original_image
        self.prompt = prompt
        
    def save(self, output_path: str, index: int = 0):
        """保存生成的图像"""
        if index < len(self.images):
            self.images[index].save(output_path)
        else:
            raise IndexError(f"Index {index} out of range. Only {len(self.images)} images available.")


# 便捷函数
def load_pipeline(model_path: str, **kwargs) -> PrefRestorePipeline:
    """加载管道的便捷函数"""
    return PrefRestorePipeline.from_pretrained(model_path, **kwargs)