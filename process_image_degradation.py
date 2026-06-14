#!/usr/bin/env python3
"""
图像退化处理脚本
读取包含图像路径和描述的JSON文件，对图像进行退化处理，并生成JSONL输出文件
"""

import json
import jsonlines
import os
import sys
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

# 添加blip3o模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'blip3o'))

from blip3o.data.image_degradation import degrade_image, ImageDegradationConfig


def load_json_data(json_file_path):
    """
    加载JSON数据文件
    
    Args:
        json_file_path (str): JSON文件路径
    
    Returns:
        list: 包含图像信息的列表
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载 {len(data)} 条数据")
        return data
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return []


def create_output_directory(output_dir):
    """
    创建输出目录
    
    Args:
        output_dir (str): 输出目录路径
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"输出目录已创建: {output_dir}")


def process_single_image(image_path, caption, output_dir, degradation_config, index):
    """
    处理单个图像
    
    Args:
        image_path (str): 原始图像路径
        caption (str): 图像描述
        output_dir (str): 输出目录
        degradation_config (ImageDegradationConfig): 退化配置
        index (int): 图像索引
    
    Returns:
        dict or None: 处理结果字典，包含prompt、image和requirement字段
    """
    try:
        # 检查原始图像是否存在
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在 - {image_path}")
            return None
        
        # 加载原始图像
        img_gt = Image.open(image_path).convert('RGB')
        
        # 对图像进行退化处理
        img_degraded = degradation_config.degrade(img_gt)
        
        # 生成输出文件名
        original_filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(original_filename)[0]
        output_filename = f"{name_without_ext}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存退化后的图像
        img_degraded.save(output_path)
        
        # 简化caption作为prompt（可以根据需要调整）
        prompt = caption.strip()
        
        # 返回JSONL格式的数据
        return {
            "prompt": prompt,
            "image": output_path,
            "requirement": "Restore"
        }
        
    except Exception as e:
        print(f"处理图像失败 {image_path}: {e}")
        return None


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='图像退化处理脚本')
    parser.add_argument('--input_json', type=str, 
                       default='/data/zgq/yaozhengjian/Datasets/FFHQ/caption/long_captions.json',
                       help='输入JSON文件路径')
    parser.add_argument('--output_dir', type=str, 
                       default='/data/zgq/yaozhengjian/Datasets/FFHQ/images_degraded',
                       help='退化图像输出目录')
    parser.add_argument('--output_jsonl', type=str, 
                       default='degraded_images.jsonl',
                       help='输出JSONL文件路径')
    parser.add_argument('--max_images', type=int, default=None,
                       help='最大处理图像数量（用于测试）')
    
    args = parser.parse_args()
    
    print("开始图像退化处理...")
    print(f"输入JSON文件: {args.input_json}")
    print(f"输出目录: {args.output_dir}")
    print(f"输出JSONL文件: {args.output_jsonl}")
    
    # 加载JSON数据
    data = load_json_data(args.input_json)
    if not data:
        print("没有数据可处理，退出程序")
        return
    
    # 限制处理数量（用于测试）
    if args.max_images:
        data = data[:args.max_images]
        print(f"限制处理数量为: {args.max_images}")
    
    # 创建输出目录
    create_output_directory(args.output_dir)
    
#     degradation_params = {
#     'gt_size': 512,
#     'in_size': 512,
#     'use_motion_kernel': False,
#     'blur_kernel_size': 41,
#     'blur_sigma': [1, 15],
#     'downsample_range': [4, 30],
#     'noise_range': [0, 20],
#     'jpeg_range': [30, 80]
# }
    
    # 创建图像退化配置
    degradation_config = ImageDegradationConfig(
        gt_size=512,
        in_size=512,
        use_motion_kernel=False,
        kernel_list=['iso', 'aniso'],
        kernel_prob=[0.5, 0.5],
        blur_kernel_size=41,
        blur_sigma=[1, 15],
        downsample_range=[4, 30],
        noise_range=[0, 20],
        jpeg_range=[30, 80]
    )
    
    # 处理图像并收集结果
    results = []
    failed_count = 0
    
    print("开始处理图像...")
    for i, item in enumerate(tqdm(data, desc="处理图像")):
        image_path = item.get('image', '')
        caption = item.get('caption', '')
        
        if not image_path or not caption:
            print(f"跳过无效数据项 {i}: 缺少image或caption字段")
            failed_count += 1
            continue
        
        result = process_single_image(
            image_path=image_path,
            caption=caption,
            output_dir=args.output_dir,
            degradation_config=degradation_config,
            index=i
        )
        
        if result:
            results.append(result)
        else:
            failed_count += 1
    
    # 保存JSONL文件
    print(f"保存JSONL文件到: {args.output_jsonl}")
    try:
        with jsonlines.open(args.output_jsonl, 'w') as writer:
            for result in results:
                writer.write(result)
        
        print(f"处理完成!")
        print(f"成功处理: {len(results)} 张图像")
        print(f"失败: {failed_count} 张图像")
        print(f"JSONL文件已保存: {args.output_jsonl}")
        
    except Exception as e:
        print(f"保存JSONL文件失败: {e}")


if __name__ == "__main__":
    main()