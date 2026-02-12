"""
评估专用的模型加载模块

功能：
1. 从 HuggingFace 官方加载模型（如 Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP）
2. 从本地完整 checkpoint 加载模型（包含 vision_tower, language_model, connector 三个组件）

不影响现有的训练流程和 load_model.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import SiglipVisionModel, SiglipImageProcessor

from .modeling_LLaVA_KD import LLaVAKD
from .configuration_tinyllava import TinyLlavaConfig


def load_hf_model(
    model_id,
    cache_dir="./pretrained_checkpoints",
    device="cuda",
    torch_dtype=torch.float16,
    load_8bit=False,
    load_4bit=False,
    **kwargs
):
    """
    从 HuggingFace Hub 加载官方发布的多模态模型
    
    Args:
        model_id: HuggingFace 模型 ID，例如 "Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP"
        cache_dir: 模型缓存目录
        device: 设备 (cuda/cpu)
        torch_dtype: 模型精度
        load_8bit: 是否使用 8bit 量化
        load_4bit: 是否使用 4bit 量化
        
    Returns:
        model, tokenizer, image_processor, context_len
        
    Example:
        >>> model, tokenizer, image_processor, context_len = load_hf_model(
        ...     "Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP"
        ... )
    """
    print(f"[HuggingFace 加载] 正在加载模型: {model_id}")
    
    # 配置量化参数
    kwargs_dict = {"low_cpu_mem_usage": True, "cache_dir": cache_dir}
    
    if load_8bit:
        kwargs_dict['load_in_8bit'] = True
        print("  - 使用 8bit 量化")
    elif load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs_dict['load_in_4bit'] = True
        kwargs_dict['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        print("  - 使用 4bit 量化")
    else:
        kwargs_dict['torch_dtype'] = torch_dtype
        print(f"  - 使用 {torch_dtype} 精度")
    
    # 从 HuggingFace 加载完整模型
    print("  - 正在下载/加载模型...")
    model = LLaVAKD.from_pretrained(
        model_id,
        **kwargs_dict
    )
    
    # 移动到指定设备
    if not (load_8bit or load_4bit):
        model = model.to(device)
    
    model.eval()
    print("  - 模型加载完成！")
    
    # 获取相关组件
    image_processor = model.vision_tower._image_processor
    tokenizer = model.tokenizer
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    
    print(f"  - Context Length: {context_len}")
    print(f"  - Tokenizer: {tokenizer.__class__.__name__}")
    print(f"  - Image Processor: {image_processor.__class__.__name__}")
    
    return model, tokenizer, image_processor, context_len


def load_local_full_checkpoint(
    checkpoint_path,
    device="cuda",
    torch_dtype=torch.float16,
    load_8bit=False,
    load_4bit=False,
    **kwargs
):
    """
    从本地完整 checkpoint 加载模型
    
    支持的目录结构：
    checkpoint_path/
    ├── config.json
    ├── vision_tower/
    │   └── pytorch_model.bin
    ├── language_model/
    │   └── pytorch_model.bin
    ├── connector/
    │   └── pytorch_model.bin
    └── tokenizer相关文件
    
    Args:
        checkpoint_path: checkpoint 目录路径
        device: 设备 (cuda/cpu)
        torch_dtype: 模型精度
        load_8bit: 是否使用 8bit 量化
        load_4bit: 是否使用 4bit 量化
        
    Returns:
        model, tokenizer, image_processor, context_len
        
    Example:
        >>> model, tokenizer, image_processor, context_len = load_local_full_checkpoint(
        ...     "/home/jsun/llava-kd/checkpoints/qwen25-0_5b-distill-after-sft"
        ... )
    """
    print(f"[本地加载] 正在加载模型: {checkpoint_path}")
    checkpoint_path = os.path.expanduser(checkpoint_path)
    
    # 检查必要的文件是否存在
    required_files = {
        'config': os.path.join(checkpoint_path, 'config.json'),
        'vision_tower': os.path.join(checkpoint_path, 'vision_tower', 'pytorch_model.bin'),
        'language_model': os.path.join(checkpoint_path, 'language_model', 'pytorch_model.bin'),
        'connector': os.path.join(checkpoint_path, 'connector', 'pytorch_model.bin'),
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(
            f"缺少必要的模型文件:\n" + "\n".join(f"  - {f}" for f in missing_files)
        )
    
    print("  - 所有必要文件检查通过")
    
    # 1. 加载模型配置
    print("  - 正在加载配置文件...")
    model_config = TinyLlavaConfig.from_pretrained(checkpoint_path)
    
    # 2. 创建模型架构
    print("  - 正在创建模型架构...")
    model = LLaVAKD(model_config)
    
    # 3. 加载 Vision Tower 权重
    print("  - 正在加载 Vision Tower 权重...")
    vision_tower_ckp = torch.load(
        required_files['vision_tower'],
        map_location='cpu'
    )
    model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
    print(f"    ✓ Vision Tower 加载完成 ({len(vision_tower_ckp)} 个参数)")
    
    # 4. 加载 Language Model 权重
    print("  - 正在加载 Language Model 权重...")
    language_model_ckp = torch.load(
        required_files['language_model'],
        map_location='cpu'
    )
    model.language_model.load_state_dict(language_model_ckp)
    print(f"    ✓ Language Model 加载完成 ({len(language_model_ckp)} 个参数)")
    
    # 5. 加载 Connector 权重
    print("  - 正在加载 Connector 权重...")
    connector_ckp = torch.load(
        required_files['connector'],
        map_location='cpu'
    )
    model.connector.load_state_dict(connector_ckp)
    print(f"    ✓ Connector 加载完成 ({len(connector_ckp)} 个参数)")
    
    # 6. 处理精度和量化
    if load_8bit:
        print("  - 应用 8bit 量化...")
        # 注意：8bit量化需要在加载时应用，这里提供一个简化版本
        # 如果需要完整的8bit支持，建议使用 from_pretrained 方法
        print("    ⚠ 警告：从本地checkpoint加载时，8bit量化支持有限")
        print("    ⚠ 建议：如需量化，请使用 load_hf_model 或 load_pretrained_model")
    elif load_4bit:
        print("  - 应用 4bit 量化...")
        print("    ⚠ 警告：从本地checkpoint加载时，4bit量化支持有限")
        print("    ⚠ 建议：如需量化，请使用 load_hf_model 或 load_pretrained_model")
    else:
        model = model.to(torch_dtype)
        print(f"  - 模型转换为 {torch_dtype} 精度")
    
    # 7. 移动到设备
    model = model.to(device)
    model.eval()
    print(f"  - 模型移动到设备: {device}")
    
    # 清理 CPU 内存
    del vision_tower_ckp, language_model_ckp, connector_ckp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 8. 获取相关组件
    image_processor = model.vision_tower._image_processor
    tokenizer = model.tokenizer
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    
    print("  - 模型加载完成！")
    print(f"  - Context Length: {context_len}")
    print(f"  - Tokenizer: {tokenizer.__class__.__name__}")
    print(f"  - Image Processor: {image_processor.__class__.__name__}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    
    return model, tokenizer, image_processor, context_len


def load_model_for_eval(
    model_path,
    source="auto",
    device="cuda",
    torch_dtype=torch.float16,
    load_8bit=False,
    load_4bit=False,
    cache_dir="./pretrained_checkpoints",
    **kwargs
):
    """
    统一的评估模型加载接口（自动判断来源）
    
    Args:
        model_path: 模型路径或 HuggingFace ID
        source: 加载来源，可选:
            - "auto": 自动判断（默认）
            - "huggingface": 从 HuggingFace Hub 加载
            - "local": 从本地 checkpoint 加载
        device: 设备
        torch_dtype: 精度
        load_8bit: 8bit 量化
        load_4bit: 4bit 量化
        cache_dir: 缓存目录
        
    Returns:
        model, tokenizer, image_processor, context_len
        
    Examples:
        >>> # 从 HuggingFace 加载
        >>> model, tok, proc, ctx = load_model_for_eval(
        ...     "Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP",
        ...     source="huggingface"
        ... )
        
        >>> # 从本地加载
        >>> model, tok, proc, ctx = load_model_for_eval(
        ...     "/path/to/checkpoint",
        ...     source="local"
        ... )
        
        >>> # 自动判断
        >>> model, tok, proc, ctx = load_model_for_eval(model_path)
    """
    if source == "auto":
        # 自动判断：如果路径存在且是目录，则为本地；否则为 HuggingFace
        if os.path.exists(os.path.expanduser(model_path)) and os.path.isdir(os.path.expanduser(model_path)):
            # 进一步检查是否包含 vision_tower, language_model, connector 子目录
            checkpoint_path = os.path.expanduser(model_path)
            has_components = (
                os.path.exists(os.path.join(checkpoint_path, 'vision_tower')) and
                os.path.exists(os.path.join(checkpoint_path, 'language_model')) and
                os.path.exists(os.path.join(checkpoint_path, 'connector'))
            )
            if has_components:
                source = "local"
                print("[自动判断] 检测到本地完整 checkpoint，使用本地加载模式")
            else:
                # 可能是 HuggingFace 下载后的缓存目录
                source = "huggingface"
                print("[自动判断] 检测到标准模型目录，使用 HuggingFace 加载模式")
        else:
            source = "huggingface"
            print("[自动判断] 使用 HuggingFace 加载模式")
    
    # 根据来源选择加载方法
    if source == "huggingface":
        return load_hf_model(
            model_path,
            cache_dir=cache_dir,
            device=device,
            torch_dtype=torch_dtype,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            **kwargs
        )
    elif source == "local":
        return load_local_full_checkpoint(
            model_path,
            device=device,
            torch_dtype=torch_dtype,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的加载来源: {source}，请使用 'auto', 'huggingface' 或 'local'")


# 为了方便使用，提供简短的别名
load_for_eval = load_model_for_eval
load_hf = load_hf_model
load_local = load_local_full_checkpoint


if __name__ == "__main__":
    """
    测试用例
    """
    import sys
    
    print("=" * 80)
    print("评估模型加载器测试")
    print("=" * 80)
    
    # 测试1: 从 HuggingFace 加载
    if "--test-hf" in sys.argv:
        print("\n[测试 1] 从 HuggingFace 加载模型")
        print("-" * 80)
        try:
            model, tokenizer, image_processor, context_len = load_hf_model(
                "Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP"
            )
            print("\n✓ HuggingFace 加载测试成功！")
        except Exception as e:
            print(f"\n✗ HuggingFace 加载测试失败: {e}")
    
    # 测试2: 从本地加载
    if "--test-local" in sys.argv:
        print("\n[测试 2] 从本地 checkpoint 加载模型")
        print("-" * 80)
        local_path = sys.argv[sys.argv.index("--test-local") + 1] if len(sys.argv) > sys.argv.index("--test-local") + 1 else None
        if local_path:
            try:
                model, tokenizer, image_processor, context_len = load_local_full_checkpoint(
                    local_path
                )
                print("\n✓ 本地加载测试成功！")
            except Exception as e:
                print(f"\n✗ 本地加载测试失败: {e}")
        else:
            print("请提供本地 checkpoint 路径")
    
    # 测试3: 自动判断
    if "--test-auto" in sys.argv:
        print("\n[测试 3] 自动判断加载模式")
        print("-" * 80)
        test_path = sys.argv[sys.argv.index("--test-auto") + 1] if len(sys.argv) > sys.argv.index("--test-auto") + 1 else None
        if test_path:
            try:
                model, tokenizer, image_processor, context_len = load_model_for_eval(
                    test_path
                )
                print("\n✓ 自动判断加载测试成功！")
            except Exception as e:
                print(f"\n✗ 自动判断加载测试失败: {e}")
        else:
            print("请提供模型路径")
    
    if len(sys.argv) == 1:
        print("\n使用方法:")
        print("  python eval_model_load.py --test-hf")
        print("  python eval_model_load.py --test-local /path/to/checkpoint")
        print("  python eval_model_load.py --test-auto /path/or/model_id")
