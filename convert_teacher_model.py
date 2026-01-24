#!/usr/bin/env python3
"""
转换教师模型格式

将从 Hugging Face 下载的 TinyLLaVA 模型转换为训练脚本期望的目录结构
"""

import os
import torch
from pathlib import Path
from safetensors.torch import load_file
import json
import shutil
from transformers import AutoConfig


def convert_teacher_model(
    source_dir: str = "./pretrained_checkpoints/LLaVA_KD_ckpts/tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune",
    target_dir: str = None
):
    """
    转换教师模型格式
    
    参数:
        source_dir: 下载的模型目录
        target_dir: 目标目录 (如果为 None,则在原地转换)
    """
    
    source_path = Path(source_dir)
    if target_dir is None:
        target_path = source_path
    else:
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("🔄 转换教师模型格式")
    print("="*70)
    print(f"📂 源目录: {source_path}")
    print(f"📂 目标目录: {target_path}")
    print()
    
    # 检查是否已经是正确的格式
    if (target_path / "vision_tower" / "pytorch_model.bin").exists():
        print("✅ 模型已经是正确的格式,无需转换")
        return True
    
    # 加载 safetensors 模型
    print("📥 加载模型权重...")
    model_files = list(source_path.glob("model-*.safetensors"))
    
    if not model_files:
        print("❌ 未找到 safetensors 文件")
        return False
    
    # 加载所有权重
    state_dict = {}
    for model_file in sorted(model_files):
        print(f"   加载 {model_file.name}...")
        state_dict.update(load_file(str(model_file)))
    
    print(f"✅ 加载了 {len(state_dict)} 个权重张量")
    print()
    
    # 分离不同组件的权重
    print("🔍 分离模型组件...")
    
    vision_tower_weights = {}
    connector_weights = {}
    language_model_weights = {}
    
    for key, value in state_dict.items():
        if key.startswith('vision_tower.'):
            # 移除 'vision_tower.' 前缀
            new_key = key.replace('vision_tower._vision_tower.', '')
            vision_tower_weights[new_key] = value
            print(f"   Vision Tower: {key} -> {new_key}")
        elif key.startswith('connector.'):
            # 保留完整的 connector 键名
            connector_weights[key] = value
            print(f"   Connector: {key}")
        elif key.startswith('language_model.'):
            # 移除 'language_model.' 前缀
            new_key = key.replace('language_model.', '')
            language_model_weights[new_key] = value
        else:
            # 其他权重归入 language model
            language_model_weights[key] = value
    
    print()
    print(f"📊 组件统计:")
    print(f"   Vision Tower: {len(vision_tower_weights)} 个权重")
    print(f"   Connector: {len(connector_weights)} 个权重")
    print(f"   Language Model: {len(language_model_weights)} 个权重")
    print()
    
    # 创建目录结构
    print("📁 创建目录结构...")
    vision_tower_dir = target_path / "vision_tower"
    connector_dir = target_path / "connector"
    language_model_dir = target_path / "language_model"
    
    vision_tower_dir.mkdir(parents=True, exist_ok=True)
    connector_dir.mkdir(parents=True, exist_ok=True)
    language_model_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 vision_tower 权重
    if vision_tower_weights:
        print("💾 保存 Vision Tower 权重...")
        vision_tower_path = vision_tower_dir / "pytorch_model.bin"
        torch.save(vision_tower_weights, vision_tower_path)
        print(f"   ✅ 已保存到 {vision_tower_path}")
    else:
        print("⚠️  警告: 未找到 Vision Tower 权重")
    
    # 保存 connector 权重
    if connector_weights:
        print("💾 保存 Connector 权重...")
        connector_path = connector_dir / "pytorch_model.bin"
        torch.save(connector_weights, connector_path)
        print(f"   ✅ 已保存到 {connector_path}")
    else:
        print("⚠️  警告: 未找到 Connector 权重")
    
    # 保存 language_model 权重
    if language_model_weights:
        print("💾 保存 Language Model 权重...")
        language_model_path = language_model_dir / "pytorch_model.bin"
        torch.save(language_model_weights, language_model_path)
        print(f"   ✅ 已保存到 {language_model_path}")
        
        # 复制配置文件到 language_model 目录
        config_files = ['config.json', 'generation_config.json', 'tokenizer_config.json', 
                       'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json']
        
        for config_file in config_files:
            src = source_path / config_file
            if src.exists():
                dst = language_model_dir / config_file
                shutil.copy2(src, dst)
                print(f"   📄 复制 {config_file}")
    else:
        print("⚠️  警告: 未找到 Language Model 权重")
    
    # 复制主配置文件
    print()
    print("📄 复制配置文件...")
    for config_file in ['config.json', 'generation_config.json']:
        src = source_path / config_file
        if src.exists() and target_path != source_path:
            dst = target_path / config_file
            shutil.copy2(src, dst)
            print(f"   ✅ 复制 {config_file}")
    
    print()
    print("="*70)
    print("✨ 转换完成!")
    print("="*70)
    print()
    print("📂 目录结构:")
    print(f"{target_path}/")
    print("  ├── vision_tower/")
    print("  │   └── pytorch_model.bin")
    print("  ├── connector/")
    print("  │   └── pytorch_model.bin")
    print("  ├── language_model/")
    print("  │   ├── pytorch_model.bin")
    print("  │   ├── config.json")
    print("  │   └── [其他配置文件]")
    print("  └── config.json")
    print()
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="转换教师模型格式",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--source-dir',
        type=str,
        default='./pretrained_checkpoints/LLaVA_KD_ckpts/tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune',
        help='源模型目录'
    )
    
    parser.add_argument(
        '--target-dir',
        type=str,
        default=None,
        help='目标目录 (默认: 在源目录原地转换)'
    )
    
    args = parser.parse_args()
    
    success = convert_teacher_model(
        source_dir=args.source_dir,
        target_dir=args.target_dir
    )
    
    if success:
        print("✅ 转换成功! 现在可以运行训练脚本了。")
    else:
        print("❌ 转换失败,请检查错误信息。")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
