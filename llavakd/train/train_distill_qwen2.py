from packaging import version
import pathlib

import tokenizers
import transformers

import sys
sys.path.append('/mnt/data/LLaVA_KD/LLaVA_KD')
from llavakd.train.tinyllava_distill_trainer import DistillLLaVATrainer
from llavakd.training_recipe import TrainingRecipeFactory
from llavakd.utils import *
from llavakd.model import *
from llavakd.data.dataset import make_supervised_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

import wandb
# WandB配置：在sbatch脚本中设置环境变量
# 或使用: wandb login (保存到 ~/.netrc)
# os.environ["WANDB_API_KEY"] = 'YOUR_API_KEY'  # 不建议硬编码
# os.environ["WANDB_MODE"] = "online"  # 在sbatch中设置

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
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
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


def wandb_log(model_arguments):

    wandb_dir = "./wandb/"
    os.makedirs(wandb_dir, exist_ok=True)

    config = {
        "model_name": f"{model_arguments.model_name_or_path.split('/')[-1]}-{model_arguments.vision_tower.split('/')[-1]}",
        'total_epochs': 1,
    }
    if int(os.getenv('LOCAL_RANK', '0')) == 0:
        wandb.init(project='DistillTinyLLaVA',
                   name=f"{model_arguments.model_name_or_path.split('/')[-1]}-{model_arguments.vision_tower.split('/')[-1]}",
                   config=config,
                   dir=wandb_dir)
                

def get_conncet_w(weights, keyword):
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}


def train():
    
    # load teacher model
    # teacher_dir = "Your Teacher ckpts dir"
    teacher_dir = './pretrained_checkpoints/LLaVA_KD_ckpts/tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune'

    # 直接从 Hugging Face 格式加载完整的教师模型
    print(f"Loading teacher model from {teacher_dir}")
    from llavakd.model.modeling_LLaVA_KD import LLaVAKD
    from llavakd.model.configuration_tinyllava import TinyLlavaConfig
    
    teacher_model_config = TinyLlavaConfig.from_pretrained(teacher_dir)
    teacher_model = LLaVAKD.from_pretrained(
        teacher_dir,
        config=teacher_model_config,
        torch_dtype=torch.float16
    )
    print(f"Teacher model loaded successfully")
    
    teacher_model.eval()

    # stu_pre_vision_tower_path = "We use the weights from S-MLLM training with PT-SFT scheme"
    # 使用教师模型的 vision tower 作为学生模型的初始权重
    stu_pre_vision_tower_path = os.path.join('./pretrained_checkpoints/LLaVA_KD_ckpts/tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune', 'vision_tower/pytorch_model.bin')
    pre_vision = torch.load(stu_pre_vision_tower_path, map_location='cpu')
    
    # 从 Hugging Face 加载学生模型的语言模型 (Qwen2.5-0.5B)
    # 如果本地有缓存会自动使用,否则会从 HF 下载
    print("Loading student language model: Qwen/Qwen2.5-0.5B")
    pre_language = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
    print("Student language model loaded successfully")

    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()

    logger_setting(getattr(training_arguments, 'output_dir', None))
    wandb_log(model_arguments)

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    # model_args contain arguements for huggingface model .from_pretrained function
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = LLaVAKD(model_config)    
    

    model.vision_tower._vision_tower.load_state_dict(pre_vision)
    print(f"Loading vision from {stu_pre_vision_tower_path}")

    # load from pretrained
    for key, Value in pre_language.state_dict().items():
        model.language_model.state_dict()[key].copy_(Value)
    for key in pre_language.state_dict().keys():
        assert torch.equal(pre_language.state_dict()[key],
                        model.language_model.state_dict()[key]), f"Mismatch found in parameter: {key}"

    print(f"Loading language from Huggingface Pretrained Model")

    model.load_connector(**model_args['connector'])


    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)

    log_trainable_params(model)  # not work well with zero3
    trainer = DistillLLaVATrainer(
        teacher_model=teacher_model,
        model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module)
    
    trainer.train()
    
    training_recipe.save(model, trainer)

    wandb.finish()

if __name__ == "__main__":
    transformers.set_seed(1234)
    train()
