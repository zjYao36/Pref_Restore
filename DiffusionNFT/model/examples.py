"""
使用示例：TextToImagePipeline 的各种调用方式
"""

from DiffusionNFT.model import TextToImagePipeline, load_pipeline
from PIL import Image

def example_basic_usage():
    """基础使用示例"""
    # 方法1：直接初始化
    pipeline = TextToImagePipeline.from_pretrained(
        model_path="/data/phd/yaozhengjian/zjYao_Exprs/BLIP-3o-next/Face-Restore_restoration-FFHQ+CelebA/checkpoint-30800",
        device="cuda:0"
    )
    
    # 生成图像
    result = pipeline(
        prompt="Please reconstruct the given image.",
        image="path/to/your/image.jpg"
    )
    
    # 保存结果
    result.save("output.jpg")
    print(f"Generated {len(result.images)} images")


def example_with_pil_image():
    """使用PIL Image对象的示例"""
    # 加载管道
    pipeline = load_pipeline(
        model_path="your_model_path",
        device="cuda:0",
        dtype="bfloat16"
    )
    
    # 使用PIL Image对象
    input_image = Image.open("input.jpg")
    result = pipeline(
        prompt="Please enhance and reconstruct this image.",
        image=input_image
    )
    
    # 访问原始图像和生成图像
    original = result.original_image
    generated = result.images[0]
    
    # 保存
    generated.save("enhanced_output.jpg")


def example_batch_processing():
    """批量处理示例"""
    import os
    from pathlib import Path
    
    # 初始化管道
    pipeline = TextToImagePipeline.from_pretrained("your_model_path")
    
    # 批量处理文件夹中的图像
    input_folder = Path("input_images")
    output_folder = Path("output_images")
    output_folder.mkdir(exist_ok=True)
    
    for image_file in input_folder.glob("*.jpg"):
        try:
            result = pipeline(
                prompt="Please reconstruct the given image.",
                image=str(image_file)
            )
            
            output_path = output_folder / f"restored_{image_file.name}"
            result.save(str(output_path))
            print(f"Processed: {image_file.name}")
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")


def example_custom_config():
    """自定义配置示例"""
    from DiffusionNFT.model import PipelineConfig
    
    # 创建自定义配置
    config = PipelineConfig(
        model_path="your_model_path",
        device="cuda:1",  # 使用不同的GPU
        scale=1,          # 调整scale参数
        seq_len=1024,     # 调整序列长度
        top_p=0.9,        # 调整采样参数
        top_k=1000,
        image_size=1024   # 使用更大的图像尺寸
    )
    
    pipeline = TextToImagePipeline(config)
    
    result = pipeline(
        prompt="Please reconstruct and enhance this image with high quality.",
        image="high_res_input.jpg"
    )
    
    result.save("high_quality_output.jpg")


if __name__ == "__main__":
    # 运行示例（请根据实际情况修改路径）
    print("TextToImagePipeline 使用示例")
    print("请根据您的实际模型路径和图像路径修改代码中的路径")
    
    # example_basic_usage()
    # example_with_pil_image() 
    # example_batch_processing()
    # example_custom_config()