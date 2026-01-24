#!/usr/bin/env python3
"""
下载 LLaVA-KD 教师模型

这个脚本用于下载训练所需的教师模型。
根据你的需求,可以从 Hugging Face 或其他来源下载。
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import argparse


def download_teacher_model(
    model_name: str = "tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune",
    save_dir: str = "./pretrained_checkpoints/LLaVA_KD_ckpts",
    repo_id: str = "Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP"
):
    """
    下载教师模型
    
    参数:
        model_name: 模型名称
        save_dir: 保存目录
        repo_id: Hugging Face 仓库 ID (默认使用 Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP)
    """
    
    # 创建保存目录
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 准备下载教师模型...")
    print(f"📁 保存路径: {save_path}")
    
    if repo_id:
        # 从 Hugging Face 下载
        print(f"🤗 从 Hugging Face 下载: {repo_id}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(save_path),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"✅ 模型下载成功!")
            print(f"📍 模型位置: {save_path}")
            return True
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False
    else:
        # 提供手动下载说明
        print("\n" + "="*70)
        print("⚠️  未提供 Hugging Face 仓库 ID")
        print("="*70)
        print("\n请按照以下步骤手动下载教师模型:\n")
        print("方案 1: 从 Hugging Face 下载")
        print("-" * 70)
        print("1. 访问 Hugging Face 模型页面")
        print("   例如: https://huggingface.co/[YOUR_MODEL_REPO]")
        print("2. 使用以下命令下载:")
        print(f"   huggingface-cli download [REPO_ID] --local-dir {save_path}")
        print()
        
        print("方案 2: 从 Google Drive 或其他云盘下载")
        print("-" * 70)
        print("1. 获取下载链接")
        print("2. 下载模型文件")
        print(f"3. 解压到: {save_path}")
        print()
        
        print("方案 3: 使用已有的教师模型")
        print("-" * 70)
        print("如果你已经有教师模型,请确保目录结构如下:")
        print(f"{save_path}/")
        print("  ├── vision_tower/")
        print("  │   └── pytorch_model.bin")
        print("  ├── connector/")
        print("  │   └── pytorch_model.bin")
        print("  ├── language_model/")
        print("  │   └── [模型文件]")
        print("  └── config.json")
        print()
        
        print("方案 4: 训练自己的教师模型")
        print("-" * 70)
        print("如果你想训练自己的教师模型,可以:")
        print("1. 先训练一个较大的模型作为教师模型")
        print("2. 使用该模型进行知识蒸馏")
        print()
        
        print("="*70)
        print("💡 提示: 如果你有 Hugging Face 仓库 ID,可以运行:")
        print(f"   python {sys.argv[0]} --repo-id YOUR_REPO_ID")
        print("="*70)
        
        return False


def download_components(save_dir: str = "./pretrained_checkpoints"):
    """
    下载训练所需的其他组件 (Vision Encoder, LLM)
    """
    print("\n" + "="*70)
    print("📦 下载训练所需的基础模型组件")
    print("="*70)
    
    components = {
        "Vision Encoder": {
            "repo_id": "google/siglip-so400m-patch14-384",
            "local_dir": f"{save_dir}/siglip-so400m-patch14-384"
        },
        "LLM (Qwen2.5-0.5B)": {
            "repo_id": "Qwen/Qwen2.5-0.5B",
            "local_dir": f"{save_dir}/Qwen2.5-0.5B"
        },
        "LLM (Qwen2.5-3B) - Teacher": {
            "repo_id": "Qwen/Qwen2.5-3B",
            "local_dir": f"{save_dir}/Qwen2.5-3B"
        }
    }
    
    for name, info in components.items():
        print(f"\n📥 下载 {name}...")
        print(f"   仓库: {info['repo_id']}")
        print(f"   路径: {info['local_dir']}")
        
        choice = input(f"   是否下载? [y/N]: ").strip().lower()
        if choice == 'y':
            try:
                Path(info['local_dir']).mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=info['repo_id'],
                    local_dir=info['local_dir'],
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"   ✅ {name} 下载成功!")
            except Exception as e:
                print(f"   ❌ 下载失败: {e}")
        else:
            print(f"   ⏭️  跳过 {name}")


def main():
    parser = argparse.ArgumentParser(
        description="下载 LLaVA-KD 教师模型和相关组件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 显示下载说明
  python download_teacher_model.py
  
  # 从 Hugging Face 下载教师模型
  python download_teacher_model.py --repo-id YOUR_REPO_ID
  
  # 下载基础组件 (Vision Encoder, LLM)
  python download_teacher_model.py --components
  
  # 指定自定义保存路径
  python download_teacher_model.py --save-dir /path/to/save
        """
    )
    
    parser.add_argument(
        '--repo-id',
        type=str,
        default='Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP',
        help='Hugging Face 仓库 ID (默认: Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune',
        help='模型名称 (默认: tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune)'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./pretrained_checkpoints/LLaVA_KD_ckpts',
        help='保存目录 (默认: ./pretrained_checkpoints/LLaVA_KD_ckpts)'
    )
    
    parser.add_argument(
        '--components',
        action='store_true',
        help='下载基础组件 (Vision Encoder, LLM)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 LLaVA-KD 教师模型下载工具")
    print("="*70)
    
    if args.components:
        download_components(save_dir='./pretrained_checkpoints')
    else:
        download_teacher_model(
            model_name=args.model_name,
            save_dir=args.save_dir,
            repo_id=args.repo_id
        )
    
    print("\n" + "="*70)
    print("✨ 完成!")
    print("="*70)


if __name__ == "__main__":
    main()
