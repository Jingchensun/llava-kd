from packaging import version
import pathlib

import tokenizers
import transformers
from transformers import SiglipVisionModel, SiglipImageProcessor

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
    加载Teacher模型，每个进程都加载一份到自己的GPU上
    """
    from llavakd.model.modeling_LLaVA_KD import LLaVAKD
    
    print(f"[Rank {local_rank}] Loading teacher model: {TEACHER_MODEL_ID}")
    teacher_model = LLaVAKD.from_pretrained(
        TEACHER_MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    
    # 将teacher模型放到对应GPU上
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
               f"{model_arguments.model_name_or_path.split('/')[-1]}-{model_arguments.vision_tower.split('/')[-1]}"

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


def train():
    from llavakd.model.modeling_LLaVA_KD import LLaVAKD
    from llavakd.model.configuration_tinyllava import TinyLlavaConfig
    
    # ============== 0. 获取分布式环境信息 ==============
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    
    # ============== 1. 加载 Teacher 模型 ==============
    teacher_model = load_teacher_model(local_rank)

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

    # ============== 3. 解析训练参数 ==============
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()

    logger_setting(getattr(training_arguments, 'output_dir', None))
    wandb_log(model_arguments, training_arguments)

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    
    # ============== 4. 构建 Student 模型 ==============
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = LLaVAKD(model_config)
    
    # 4.1 加载 Vision Tower 权重
    model.vision_tower._vision_tower.load_state_dict(pre_vision_tower.state_dict())
    print(f"Vision tower weights loaded from {STUDENT_VT_ID}")
    
    # 4.2 加载 LLM 权重
    for key, value in pre_language.state_dict().items():
        model.language_model.state_dict()[key].copy_(value)
    # 验证加载成功
    for key in pre_language.state_dict().keys():
        assert torch.equal(pre_language.state_dict()[key],
                          model.language_model.state_dict()[key]), f"Mismatch found in parameter: {key}"
    print(f"LLM weights loaded from {STUDENT_LLM_ID}")

    # 4.3 Connector 随机初始化 (由 LLaVAKD 构造函数完成)
    model.load_connector(**model_args['connector'])
    print("Connector initialized randomly")

    # 释放预加载模型的内存
    del pre_vision_tower, pre_language
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
    trainer = DistillLLaVATrainer(
        teacher_model=teacher_model,
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module)
    
    trainer.train()
    
    # ============== 7. 保存最终模型 (仅 Connector) ==============
    training_recipe.save(model, trainer)

    wandb.finish()


if __name__ == "__main__":
    transformers.set_seed(1234)
    train()
