"""
SFT训练脚本 - 仅训练Student模型
- Vision Tower: 从 HuggingFace 加载预训练权重
- LLM: 从 HuggingFace 加载预训练权重  
- Connector: 从 pretrain 阶段的 checkpoint 加载权重
- 中间checkpoint: 保存 connector + llm 权重
- 最终保存: 保存完整模型
"""

from packaging import version
import pathlib

import tokenizers
import transformers
from transformers import SiglipVisionModel, AutoModelForCausalLM

import sys
import os
sys.path.append('/mnt/data/LLaVA_KD/LLaVA_KD')
from llavakd.train.tinyllava_sft_trainer import SFTStudentTrainer
from llavakd.training_recipe import TrainingRecipeFactory
from llavakd.utils import *
from llavakd.model import *
from llavakd.data.dataset import make_supervised_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


# ============== 模型配置 ==============
STUDENT_LLM_ID = "Qwen/Qwen2.5-0.5B"
STUDENT_VT_ID = "google/siglip-so400m-patch14-384"
CACHE_DIR = "./pretrained_checkpoints"


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


def get_connector_weights(weights, keyword):
    """从checkpoint中提取connector权重"""
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}


def train():
    from llavakd.model.modeling_LLaVA_KD import LLaVAKD
    from llavakd.model.configuration_tinyllava import TinyLlavaConfig
    
    # ============== 1. 解析训练参数 ==============
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    
    logger_setting(getattr(training_arguments, 'output_dir', None))
    
    # ============== 2. 加载 Student 模型组件 ==============
    # 2.1 从 HuggingFace 加载 Vision Tower (SigLIP)
    print(f"Loading student vision tower from HuggingFace: {STUDENT_VT_ID}")
    pre_vision_tower = SiglipVisionModel.from_pretrained(
        STUDENT_VT_ID,
        cache_dir=CACHE_DIR
    )
    print("Student vision tower loaded successfully")
    
    # 2.2 从 HuggingFace 加载 LLM (Qwen2.5-0.5B)
    print(f"Loading student LLM from HuggingFace: {STUDENT_LLM_ID}")
    pre_language = AutoModelForCausalLM.from_pretrained(
        STUDENT_LLM_ID,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    print("Student LLM loaded successfully")
    
    # 2.3 从 pretrain checkpoint 加载 Connector 权重
    pretrain_connector_path = training_arguments.pretrained_model_path
    if pretrain_connector_path is None:
        raise ValueError("必须指定 --pretrained_model_path 来加载 pretrain 阶段的 connector 权重")
    
    connector_ckp_path = os.path.join(pretrain_connector_path, 'connector', 'pytorch_model.bin')
    if not os.path.exists(connector_ckp_path):
        raise FileNotFoundError(f"Connector checkpoint not found: {connector_ckp_path}")
    
    print(f"Loading connector from pretrain checkpoint: {connector_ckp_path}")
    pre_connector = torch.load(connector_ckp_path, map_location='cpu')
    print("Connector weights loaded successfully")
    
    # ============== 3. 配置训练参数 ==============
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    
    # ============== 4. 构建 Student 模型 ==============
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = LLaVAKD(model_config)
    
    # 4.1 加载 Vision Tower 权重 (从 HuggingFace)
    model.vision_tower._vision_tower.load_state_dict(pre_vision_tower.state_dict())
    print(f"✓ Vision tower weights loaded from {STUDENT_VT_ID}")
    
    # 4.2 加载 LLM 权重 (从 HuggingFace)
    for key, value in pre_language.state_dict().items():
        model.language_model.state_dict()[key].copy_(value)
    # 验证加载成功
    for key in pre_language.state_dict().keys():
        assert torch.equal(pre_language.state_dict()[key],
                          model.language_model.state_dict()[key]), f"Mismatch found in parameter: {key}"
    print(f"✓ LLM weights loaded from {STUDENT_LLM_ID}")
    
    # 4.3 加载 Connector 权重 (从 pretrain checkpoint)
    # 处理可能的key格式差异
    if '_connector' in list(pre_connector.keys())[0]:
        connector_weights = get_connector_weights(pre_connector, '_connector')
        model.connector._connector.load_state_dict(connector_weights)
    else:
        model.connector.load_state_dict(pre_connector)
    print(f"✓ Connector weights loaded from {pretrain_connector_path}")
    
    # 释放预加载模型的内存
    del pre_vision_tower, pre_language, pre_connector
    torch.cuda.empty_cache()
    
    # ============== 5. 配置训练 ==============
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)

    log_trainable_params(model)
    
    # ============== 6. 开始训练 ==============
    trainer = SFTStudentTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module
    )
    
    print('Training Start')
    trainer.train()
    
    # ============== 7. 保存最终完整模型 ==============
    training_recipe.save(model, trainer)


if __name__ == "__main__":
    transformers.set_seed(1234)
    train()
