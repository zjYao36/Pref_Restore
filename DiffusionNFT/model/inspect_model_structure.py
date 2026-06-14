"""
测试脚本：加载 PrefRestorePipeline 并打印模型结构
"""

import sys
import os
import torch

# 添加项目路径
project_root = "/data/phd/yaozhengjian/Code/RL/ART-FRv2/DiffusionNFT"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# from DiffusionNFT.model.PrefRestorePipeline_pipeline import PrefRestorePipeline
from PrefRestorePipeline_pipeline import PrefRestorePipeline

def print_model_structure():
    """打印模型结构到文件"""
    
    # 配置模型路径（请根据实际情况修改）
    model_path = "/data/phd/yaozhengjian/zjYao_Exprs/BLIP-3o-next/Face-Restoration_FFHQ_VAE_Step3_scaling/checkpoint-108000"
    
    print("Loading PrefRestorePipeline...")
    pipeline = PrefRestorePipeline.from_pretrained(
        model_path=model_path,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    print("Pipeline loaded successfully!")
    
    # 创建详细的模型结构报告
    output_file = "PrefRestorePipeline_model_structure.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("PrefRestorePipeline 模型结构详细分析\n")
        f.write("=" * 100 + "\n\n")
        
        # 1. 基本信息
        f.write("1. 基本信息\n")
        f.write("-" * 50 + "\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"设备: {pipeline.device}\n")
        f.write(f"数据类型: {pipeline.config.dtype}\n")
        f.write(f"主模型类型: {type(pipeline.model).__name__}\n")
        f.write(f"分词器类型: {type(pipeline.tokenizer).__name__}\n")
        f.write(f"处理器类型: {type(pipeline.processor).__name__}\n\n")
        
        # 2. 模型层级结构
        f.write("2. 模型层级结构\n")
        f.write("-" * 50 + "\n")
        for name, module in pipeline.model.named_modules():
            f.write(f"{name}: {type(module).__name__}\n")
        f.write("\n")
        
        # 3. 参数统计
        f.write("3. 参数统计\n")
        f.write("-" * 50 + "\n")
        total_params = sum(p.numel() for p in pipeline.model.parameters())
        trainable_params = sum(p.numel() for p in pipeline.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        f.write(f"总参数量: {total_params:,}\n")
        f.write(f"可训练参数: {trainable_params:,}\n")
        f.write(f"冻结参数: {frozen_params:,}\n")
        f.write(f"可训练参数比例: {trainable_params/total_params*100:.2f}%\n\n")
        
        # 4. 各模块参数详情
        f.write("4. 各模块参数详情\n")
        f.write("-" * 50 + "\n")
        
        module_stats = {}
        for name, module in pipeline.model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if total > 0:
                    module_stats[name] = {
                        'total': total,
                        'trainable': trainable,
                        'frozen': total - trainable,
                        'trainable_ratio': trainable / total * 100 if total > 0 else 0
                    }
        
        # 按参数量排序
        sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for name, stats in sorted_modules:
            f.write(f"\n{name}:\n")
            f.write(f"  总参数: {stats['total']:,}\n")
            f.write(f"  可训练: {stats['trainable']:,}\n")
            f.write(f"  冻结: {stats['frozen']:,}\n")
            f.write(f"  可训练比例: {stats['trainable_ratio']:.2f}%\n")
        
        # 5. 建议的冻结策略
        f.write("\n\n5. 建议的冻结策略\n")
        f.write("-" * 50 + "\n")
        f.write("# 基于模块名称的冻结建议:\n\n")
        
        freeze_suggestions = []
        for name, module in pipeline.model.named_modules():
            name_lower = name.lower()
            # 常见的需要冻结的模块
            if any(keyword in name_lower for keyword in [
                'vision_tower', 'image_processor', 'vision_model',
                'embeddings', 'encoder', 'layernorm', 'norm',
                'position_embedding', 'patch_embedding'
            ]):
                freeze_suggestions.append(name)
        
        if freeze_suggestions:
            f.write("# 建议冻结的模块:\n")
            for suggestion in freeze_suggestions:
                f.write(f"pipeline.model.{suggestion}.requires_grad_(False)\n")
        else:
            f.write("# 未找到明显需要冻结的模块，请根据具体需求手动设置\n")
        
        # 6. 训练建议
        f.write("\n\n6. 训练建议\n")
        f.write("-" * 50 + "\n")
        f.write("# 基于模型结构的训练建议:\n\n")
        
        if 'vision' in str(pipeline.model).lower():
            f.write("- 检测到视觉模块，建议冻结预训练的视觉编码器\n")
        if 'language' in str(pipeline.model).lower() or 'llm' in str(pipeline.model).lower():
            f.write("- 检测到语言模块，建议冻结预训练的语言模型主体\n")
        if 'vae' in str(pipeline.model).lower():
            f.write("- 检测到VAE模块，建议冻结VAE编码器和解码器\n")
        
        f.write(f"\n建议只训练特定的适配层或新增的模块\n")
        f.write(f"总参数量较大({total_params:,})，建议使用LoRA等参数高效训练方法\n")
        
    print(f"模型结构分析已保存到: {output_file}")
    
    # 7. 打印关键信息到控制台
    print("\n" + "="*60)
    print("模型结构概览:")
    print("="*60)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {frozen_params:,}")
    print(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")
    
    print(f"\n详细分析已保存到: {output_file}")
        


if __name__ == "__main__":
    print_model_structure()