import copy
from dataclasses import dataclass
import json
from typing import Dict, Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import os

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *


import transformers
import torch
from torch.utils.data import Dataset



ImageFile.LOAD_TRUNCATED_IMAGES = True


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        # 预筛选：检查图片是否存在
        image_folder = data_args.image_folder
        valid_data = []
        missing_images = []
        
        print(f"[Dataset] 开始预筛选，检查图片是否存在...")
        for sample in list_data_dict:
            if 'image' in sample:
                image_path = os.path.join(image_folder, sample['image'])
                if os.path.exists(image_path):
                    valid_data.append(sample)
                else:
                    missing_images.append(sample['image'])
            else:
                # 纯文本样本，保留
                valid_data.append(sample)
        
        # 保存缺失图片列表
        if missing_images:
            data_dir = os.path.dirname(data_path)
            missing_file = os.path.join(data_dir, 'missing_data.txt')
            with open(missing_file, 'w') as f:
                for img in missing_images:
                    f.write(f"{img}\n")
            print(f"[Dataset] 发现 {len(missing_images)} 个缺失图片，已保存到: {missing_file}")
        
        print(f"[Dataset] 预筛选完成: 原始样本 {len(list_data_dict)}, 有效样本 {len(valid_data)}, 缺失 {len(missing_images)}")

        self.tokenizer = tokenizer
        self.list_data_dict = valid_data  # 使用筛选后的数据
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)
        self._text_cache = {}  # 缓存文本预处理结果

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        # 优化: 缓存text预处理结果
        if i in self._text_cache:
            data_dict = copy.deepcopy(self._text_cache[i])
        else:
            data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
            # 只缓存一部分，避免内存爆炸
            if len(self._text_cache) < 10000:
                self._text_cache[i] = copy.deepcopy(data_dict)
        
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image_path = os.path.join(image_folder, image_file)
            try:
                # 优化: 使用更高效的图片加载方式
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = self.image_preprocess(image)
                data_dict['image'] = image
            except Exception as e:
                print(f"[Warning] Failed to load image {image_path}: {e}")
                crop_size = getattr(self.data_args.image_processor, 'crop_size', 
                                   getattr(self.data_args.image_processor, 'size'))
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        elif self.data_args.is_multimodal:
            crop_size = getattr(self.data_args.image_processor, 'crop_size', 
                               getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
