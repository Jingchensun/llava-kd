"""
Distill After SFT 训练脚本
- Teacher 模型: 从 HuggingFace 直接加载预训练 checkpoint
- Student 模型:
  - Vision Tower: 从 HuggingFace 加载预训练权重 (frozen)
  - LLM + Connector: 从 SFT checkpoint 加载权重 (full tuning)
- 中间 checkpoint: 保存 LLM + Connector 权重
- 最终保存: 保存完整模型 (VT + Connector + LLM)
"""

from packaging import version
import pathlib
import os

import tokenizers
import transformers
from transformers import SiglipVisionModel, AutoModelForCausalLM

import sys
sys.path.append('/mnt/data/LLaVA_KD/LLaVA_KD')
from llavakd.train.tinyllava_distill_trainer import DistillLLaVATrainer
from llavakd.training_recipe import TrainingRecipeFactory
from llavakd.utils import *
from llavakd.model import *
from llavakd.data.dataset import make_supervised_data_module


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

import wandb

# ============== 模型配置 ==============
# HuggingFace 模型路径
TEACHER_MODEL_ID = "Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP"
STUDENT_LLM_ID = "Qwen/Qwen2.5-0.5B"
STUDENT_VT_ID = "google/siglip-so400m-patch14-384"

# 本地缓存目录
CACHE_DIR = "./pretrained_checkpoints"


def load_teacher_model(local_rank):
    """
    从 HuggingFace 加载 Teacher 模型，每个进程都加载一份到自己的GPU上
    """
    from llavakd.model.modeling_LLaVA_KD import LLaVAKD
    
    print(f"[Rank {local_rank}] Loading teacher model from HuggingFace: {TEACHER_MODEL_ID}")
    teacher_model = LLaVAKD.from_pretrained(
        TEACHER_MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    
    # 将 teacher 模型放到对应 GPU 上
    device = f'cuda:{local_rank}' if local_rank >= 0 else 'cuda:0'
    teacher_model = teacher_model.to(device)
    print(f"[Rank {local_rank}] Teacher model loaded to {device}")
    
    return teacher_model


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments) 
    return model_args


def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation
    return llm_args


def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args


def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args


def wandb_log(model_arguments, training_arguments):
    wandb_dir = "./wandb/"
    os.makedirs(wandb_dir, exist_ok=True)

    run_name = getattr(training_arguments, 'run_name', None) or \
               f"distill-after-sft-{model_arguments.model_name_or_path.split('/')[-1]}"

    config = {
        "model_name": f"{model_arguments.model_name_or_path.split('/')[-1]}-{model_arguments.vision_tower.split('/')[-1]}",
        'total_epochs': training_arguments.num_train_epochs,
        'learning_rate': training_arguments.learning_rate,
        'distil_ratio_type': getattr(model_arguments, 'distil_ratio_type', 'unknown'),
    }
    if int(os.getenv('LOCAL_RANK', '0')) == 0:
        wandb.init(project='DistillTinyLLaVA',
                   name=run_name,
                   config=config,
                   dir=wandb_dir)


def get_connector_weights(weights, keyword):
    """从 checkpoint 中提取 connector 权重"""
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}


def train():
    from llavakd.model.modeling_LLaVA_KD import LLaVAKD
    from llavakd.model.configuration_tinyllava import TinyLlavaConfig
    
    # ============== 0. 获取分布式环境信息 ==============
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    
    # ============== 1. 解析训练参数 ==============
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    
    logger_setting(getattr(training_arguments, 'output_dir', None))
    wandb_log(model_arguments, training_arguments)
    
    # ============== 2. 加载 Teacher 模型 (从 HuggingFace) ==============
    teacher_model = load_teacher_model(local_rank)
    
    # ============== 3. 加载 Student 模型组件 ==============
    # 3.1 从 HuggingFace 加载 Vision Tower (SigLIP)
    print(f"Loading student vision tower from HuggingFace: {STUDENT_VT_ID}")
    pre_vision_tower = SiglipVisionModel.from_pretrained(
        STUDENT_VT_ID,
        cache_dir=CACHE_DIR
    )
    print("Student vision tower loaded successfully")
    
    # 3.2 从 SFT checkpoint 加载 LLM 和 Connector 权重
    sft_checkpoint_path = training_arguments.pretrained_model_path
    if sft_checkpoint_path is None:
        raise ValueError("必须指定 --pretrained_model_path 来加载 SFT 阶段的 checkpoint")
    
    # 加载 LLM 权重
    llm_ckp_path = os.path.join(sft_checkpoint_path, 'language_model', 'pytorch_model.bin')
    if not os.path.exists(llm_ckp_path):
        raise FileNotFoundError(f"LLM checkpoint not found: {llm_ckp_path}")
    print(f"Loading LLM from SFT checkpoint: {llm_ckp_path}")
    pre_llm_weights = torch.load(llm_ckp_path, map_location='cpu')
    
    # 加载 Connector 权重
    connector_ckp_path = os.path.join(sft_checkpoint_path, 'connector', 'pytorch_model.bin')
    if not os.path.exists(connector_ckp_path):
        raise FileNotFoundError(f"Connector checkpoint not found: {connector_ckp_path}")
    print(f"Loading Connector from SFT checkpoint: {connector_ckp_path}")
    pre_connector_weights = torch.load(connector_ckp_path, map_location='cpu')
    
    # ============== 4. 配置训练参数 ==============
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    
    # ============== 5. 构建 Student 模型 ==============
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = LLaVAKD(model_config)
    
    # 5.1 加载 Vision Tower 权重 (从 HuggingFace)
    model.vision_tower._vision_tower.load_state_dict(pre_vision_tower.state_dict())
    print(f"✓ Vision tower weights loaded from {STUDENT_VT_ID}")
    
    # 5.2 加载 LLM 权重 (从 SFT checkpoint)
    # 处理 tied weights: Qwen2 的 lm_head.weight 和 embed_tokens.weight 是共享的
    # 如果保存时只保存了 embed_tokens，需要补充 lm_head.weight
    if 'lm_head.weight' not in pre_llm_weights and 'model.embed_tokens.weight' in pre_llm_weights:
        pre_llm_weights['lm_head.weight'] = pre_llm_weights['model.embed_tokens.weight'].clone()
        print("  Note: Added lm_head.weight from embed_tokens.weight (tied weights)")
    model.language_model.load_state_dict(pre_llm_weights)
    print(f"✓ LLM weights loaded from {llm_ckp_path}")
    
    # 5.3 加载 Connector 权重 (从 SFT checkpoint)
    # 处理可能的 key 格式差异
    if '_connector' in list(pre_connector_weights.keys())[0]:
        connector_weights = get_connector_weights(pre_connector_weights, '_connector')
        model.connector._connector.load_state_dict(connector_weights)
    else:
        model.connector.load_state_dict(pre_connector_weights)
    print(f"✓ Connector weights loaded from {connector_ckp_path}")
    
    # 释放预加载模型的内存
    del pre_vision_tower, pre_llm_weights, pre_connector_weights
    torch.cuda.empty_cache()
    
    # ============== 6. 配置训练 ==============
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)

    log_trainable_params(model)
    
    # ============== 7. 开始训练 ==============
    trainer = DistillLLaVATrainer(
        teacher_model=teacher_model,
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module
    )
    
    print('Training Start')
    trainer.train()
    
    # ============== 8. 保存最终完整模型 ==============
    training_recipe.save(model, trainer)

    wandb.finish()


if __name__ == "__main__":
    transformers.set_seed(1234)
    train()