"""
支持新加载方式的评估脚本

新增功能:
1. 支持从 HuggingFace 直接加载模型 (--load-source huggingface)
2. 支持从本地完整checkpoint加载 (--load-source local)
3. 向后兼容原有的加载方式 (--load-source original)
"""

import argparse
import time

import torch
import os
import json
from tqdm import tqdm
import shortuuid

# 只导入评估所需的模块，避免导入 train_utils (包含 deepspeed)
from llavakd.utils.constants import DEFAULT_IMAGE_TOKEN
from llavakd.utils.message import Message
from llavakd.utils.eval_utils import disable_torch_init
from llavakd.data.text_preprocess import TextPreprocess
from llavakd.data.image_preprocess import ImagePreprocess
from llavakd.model import load_pretrained_model, load_distill_model
# 导入新的加载函数
from llavakd.model.eval_model_load import load_model_for_eval, load_hf_model, load_local_full_checkpoint
import transformers

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, text_processor, image_processor):
        self.questions = questions
        self.image_folder = image_folder
        self.text_processor = text_processor
        self.image_processor = image_processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = self.image_processor(image)
        
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        msg = Message()
        msg.add_message(qs)
        #print(prompt)
        result = self.text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, text_processor, image_processor, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, text_processor, image_processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    print("=" * 80)
    print(f"模型加载配置:")
    print(f"  - 加载来源: {args.load_source}")
    print(f"  - 模型路径: {model_path}")
    print("=" * 80)
    
    # 根据加载来源选择不同的加载方式
    if args.load_source == "huggingface":
        print("\n[使用新方式] 从 HuggingFace 加载模型...")
        model, tokenizer, image_processor, context_len = load_hf_model(
            model_path,
            cache_dir=args.cache_dir,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit
        )
    elif args.load_source == "local":
        print("\n[使用新方式] 从本地完整 checkpoint 加载模型...")
        model, tokenizer, image_processor, context_len = load_local_full_checkpoint(
            model_path,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit
        )
    elif args.load_source == "auto":
        print("\n[使用新方式] 自动判断加载来源...")
        model, tokenizer, image_processor, context_len = load_model_for_eval(
            model_path,
            cache_dir=args.cache_dir,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit
        )
    else:  # original
        print("\n[使用原始方式] 兼容原有的加载方式...")
        # 兼容原有的加载方式
        if args.load_distill:
            model, tokenizer, image_processor, context_len = load_distill_model(
                connector_checkpoint_path=model_path,
                llm_model_id=args.llm_model_id,
                vt_model_id=args.vt_model_id,
                cache_dir=args.cache_dir
            )
        else:
            model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    
    print("\n✓ 模型加载完成！")
    print("=" * 80)
    
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    data_loader = create_data_loader(questions, args.image_folder, text_processor, image_processor)
    # print("Tokenizer's eos token: ", tokenizer.eos_token)
    model.to(device='cuda')
    
    print(f"\n开始评估 {len(questions)} 个样本...")
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        # keywords = [tokenizer.eos_token]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            # import pdb;pdb.set_trace()
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                # stopping_criteria=[stopping_criteria],
                image_sizes=image_sizes,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print("Printing outputs")
        # print(outputs)
        # time.sleep(5)
        # import pdb;pdb.set_trace()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": args.model_base,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()
    print(f"\n✓ 评估完成！结果保存到: {answers_file}")

if __name__ == "__main__":
    transformers.set_seed(42)
    parser = argparse.ArgumentParser()
    
    # 基础参数
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    
    # 新增：加载来源参数
    parser.add_argument("--load-source", type=str, default="auto",
                        choices=["auto", "huggingface", "local", "original"],
                        help="模型加载来源: auto(自动判断), huggingface(HF Hub), local(本地完整checkpoint), original(原始方式)")
    
    # 量化参数
    parser.add_argument("--load-8bit", action="store_true",
                        help="使用 8bit 量化加载模型")
    parser.add_argument("--load-4bit", action="store_true",
                        help="使用 4bit 量化加载模型")
    
    # 缓存目录
    parser.add_argument("--cache-dir", type=str, default="./pretrained_checkpoints",
                        help="模型缓存目录")
    
    # 兼容原有的蒸馏模型加载参数
    parser.add_argument("--load-distill", action="store_true", 
                        help="(原始方式) 加载蒸馏训练的模型（仅 Connector 权重）")
    parser.add_argument("--llm-model-id", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="(原始方式) HuggingFace LLM 模型 ID")
    parser.add_argument("--vt-model-id", type=str, default="google/siglip-so400m-patch14-384",
                        help="(原始方式) HuggingFace Vision Tower 模型 ID")
    
    args = parser.parse_args()

    eval_model(args)
