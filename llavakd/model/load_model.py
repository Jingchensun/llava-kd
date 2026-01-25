import os
import torch
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers import SiglipVisionModel

from .modeling_LLaVA_KD import LLaVAKD
from .configuration_tinyllava import TinyLlavaConfig

# ============== 默认模型配置 ==============
DEFAULT_STUDENT_LLM_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_STUDENT_VT_ID = "google/siglip-so400m-patch14-384"
DEFAULT_CACHE_DIR = "./pretrained_checkpoints"


def load_base_ckp_for_lora(ckp_path):
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    new_ckp = OrderedDict()
    for k, v in ckp.items():
        new_k = k.replace('.base_layer', '')
        new_ckp[new_k] = v
    return new_ckp


def load_pretrained_model(model_name_or_path, load_type='hf', load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    if model_name_or_path is not None and 'lora' not in model_name_or_path:
        
        # for qwen1.5 and qwen2.5
        model = LLaVAKD.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16)

    elif model_name_or_path is not None and 'lora' in model_name_or_path:
        if os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')):
            model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
            model = LLaVAKD(model_config)
            language_model_ckp_path = os.path.join(model_name_or_path, 'language_model/pytorch_model.bin')
            language_model_ckp = load_base_ckp_for_lora(language_model_ckp_path)
            model.language_model.load_state_dict(language_model_ckp)
            vision_tower_ckp_path = os.path.join(model_name_or_path, 'vision_tower/pytorch_model.bin')
            vision_tower_ckp = load_base_ckp_for_lora(vision_tower_ckp_path)
            model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
            connector_ckp_path = os.path.join(model_name_or_path, 'connector/pytorch_model.bin')
            connector_ckp = load_base_ckp_for_lora(connector_ckp_path)
            model.connector.load_state_dict(connector_ckp)
            model.to(torch.float16)
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_name_or_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        
    image_processor = model.vision_tower._image_processor
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    tokenizer = model.tokenizer
    return model, tokenizer, image_processor, context_len


def load_distill_model(
    connector_checkpoint_path,
    llm_model_id=DEFAULT_STUDENT_LLM_ID,
    vt_model_id=DEFAULT_STUDENT_VT_ID,
    cache_dir=DEFAULT_CACHE_DIR,
    device="cuda",
    torch_dtype=torch.float16,
    **kwargs
):
    """
    加载蒸馏训练后的模型（仅加载 Connector 权重）。
    Vision Tower 和 LLM 从 HuggingFace 重新加载。
    
    Args:
        connector_checkpoint_path: 训练保存的 checkpoint 目录路径
        llm_model_id: HuggingFace LLM 模型 ID
        vt_model_id: HuggingFace Vision Tower 模型 ID
        cache_dir: 模型缓存目录
        device: 设备 (cuda/cpu)
        torch_dtype: 模型精度
        
    Returns:
        model, tokenizer, image_processor, context_len
    """
    print(f"Loading distilled model from: {connector_checkpoint_path}")
    
    # 1. 加载模型配置
    model_config = TinyLlavaConfig.from_pretrained(connector_checkpoint_path)
    
    # 2. 创建模型框架
    model = LLaVAKD(model_config)
    
    # 3. 从 HuggingFace 加载 Vision Tower
    print(f"Loading Vision Tower from HuggingFace: {vt_model_id}")
    pre_vision_tower = SiglipVisionModel.from_pretrained(
        vt_model_id,
        cache_dir=cache_dir
    )
    model.vision_tower._vision_tower.load_state_dict(pre_vision_tower.state_dict())
    del pre_vision_tower
    print("Vision Tower loaded successfully")
    
    # 4. 从 HuggingFace 加载 LLM
    print(f"Loading LLM from HuggingFace: {llm_model_id}")
    pre_language = AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    for key, value in pre_language.state_dict().items():
        model.language_model.state_dict()[key].copy_(value)
    del pre_language
    print("LLM loaded successfully")
    
    # 5. 加载训练好的 Connector 权重
    connector_ckp_path = os.path.join(connector_checkpoint_path, 'connector', 'pytorch_model.bin')
    if os.path.exists(connector_ckp_path):
        print(f"Loading Connector from: {connector_ckp_path}")
        connector_ckp = torch.load(connector_ckp_path, map_location='cpu')
        model.connector.load_state_dict(connector_ckp)
        print("Connector loaded successfully")
    else:
        raise FileNotFoundError(f"Connector checkpoint not found: {connector_ckp_path}")
    
    # 6. 转换精度并移动到设备
    model = model.to(torch_dtype)
    model = model.to(device)
    model.eval()
    
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    
    image_processor = model.vision_tower._image_processor
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    tokenizer = model.tokenizer
    
    print("Model loaded successfully!")
    return model, tokenizer, image_processor, context_len