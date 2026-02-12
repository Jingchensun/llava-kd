# 新的模型加载方式使用指南

## 概述

新增的 `eval_model_load.py` 模块提供了更灵活的模型加载方式，支持：

1. **从 HuggingFace Hub 加载模型**（如官方发布的模型）
2. **从本地完整 checkpoint 加载**（包含 vision_tower, language_model, connector 三个组件的完整权重）

## 特点

- ✅ 不修改现有的 `load_model.py`
- ✅ 不影响训练流程
- ✅ 向后兼容原有的评估脚本
- ✅ 支持自动判断加载来源
- ✅ 支持 8bit/4bit 量化（HuggingFace 模式）

## 使用方法

### 方式1: 从 HuggingFace 加载官方模型

#### Python 代码

```python
from llavakd.model.eval_model_load import load_hf_model

# 加载 HuggingFace 上的官方模型
model, tokenizer, image_processor, context_len = load_hf_model(
    "Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP",
    cache_dir="./pretrained_checkpoints"
)
```

#### Shell 脚本

```bash
# 使用新的评估脚本
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.12 -m llavakd.eval.model_vqa_loader_v2 \
    --model-path "Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP" \
    --load-source huggingface \
    --question-file ./eval_dataset/gqa/llava_gqa_testdev_balanced.jsonl \
    --image-folder ./eval_dataset/gqa/images \
    --answers-file ./results/answers.jsonl \
    --temperature 0 \
    --conv-mode phi
```

### 方式2: 从本地完整 checkpoint 加载

#### Python 代码

```python
from llavakd.model.eval_model_load import load_local_full_checkpoint

# 加载本地训练的完整 checkpoint
model, tokenizer, image_processor, context_len = load_local_full_checkpoint(
    "/home/jsun/llava-kd/checkpoints/qwen25-0_5b-distill-after-sft"
)
```

#### Shell 脚本

```bash
# 使用新的评估脚本
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.12 -m llavakd.eval.model_vqa_loader_v2 \
    --model-path "/home/jsun/llava-kd/checkpoints/qwen25-0_5b-distill-after-sft" \
    --load-source local \
    --question-file ./eval_dataset/gqa/llava_gqa_testdev_balanced.jsonl \
    --image-folder ./eval_dataset/gqa/images \
    --answers-file ./results/answers.jsonl \
    --temperature 0 \
    --conv-mode phi
```

### 方式3: 自动判断（推荐）

```python
from llavakd.model.eval_model_load import load_model_for_eval

# 自动判断是 HuggingFace ID 还是本地路径
model, tokenizer, image_processor, context_len = load_model_for_eval(
    model_path,  # 可以是 HF ID 或本地路径
    source="auto"
)
```

```bash
# Shell 脚本中使用 auto 模式
python3.12 -m llavakd.eval.model_vqa_loader_v2 \
    --model-path "$MODEL_PATH" \
    --load-source auto \
    ...
```

## 完整评估流程

### 评估 HuggingFace 模型

```bash
cd /home/jsun/llava-kd

# 方法1: 使用 eval_all_v3.sh（修改配置）
# 编辑 scripts/eval/eval_all_v3.sh，设置:
#   MODE="huggingface"
#   MODEL_PATH="Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP"
bash scripts/eval/eval_all_v3.sh

# 方法2: 使用单个数据集脚本
bash scripts/eval/gqa_v2.sh \
    "Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP" \
    "TinyLLaVA_3B" \
    "huggingface"
```

### 评估本地训练模型

```bash
cd /home/jsun/llava-kd

# 方法1: 使用 eval_all_v3.sh（修改配置）
# 编辑 scripts/eval/eval_all_v3.sh，设置:
#   MODE="local"
#   MODEL_PATH="/home/jsun/llava-kd/checkpoints/qwen25-0_5b-distill-after-sft"
bash scripts/eval/eval_all_v3.sh

# 方法2: 使用单个数据集脚本
bash scripts/eval/gqa_v2.sh \
    "/home/jsun/llava-kd/checkpoints/qwen25-0_5b-distill-after-sft" \
    "Qwen25_0.5B_Local" \
    "local"
```

## 本地 Checkpoint 目录结构

确保你的本地 checkpoint 包含以下文件：

```
checkpoint_path/
├── config.json                    # 模型配置
├── vision_tower/
│   └── pytorch_model.bin         # Vision Tower 权重
├── language_model/
│   └── pytorch_model.bin         # Language Model 权重
├── connector/
│   └── pytorch_model.bin         # Connector 权重
├── tokenizer_config.json         # Tokenizer 配置
├── special_tokens_map.json
├── vocab.json
└── merges.txt
```

## API 参考

### `load_hf_model()`

从 HuggingFace Hub 加载模型。

**参数：**
- `model_id` (str): HuggingFace 模型 ID
- `cache_dir` (str): 缓存目录，默认 `"./pretrained_checkpoints"`
- `device` (str): 设备，默认 `"cuda"`
- `torch_dtype`: 精度，默认 `torch.float16`
- `load_8bit` (bool): 是否使用 8bit 量化
- `load_4bit` (bool): 是否使用 4bit 量化

**返回：**
- `model`: 加载的模型
- `tokenizer`: 分词器
- `image_processor`: 图像处理器
- `context_len`: 上下文长度

### `load_local_full_checkpoint()`

从本地完整 checkpoint 加载模型。

**参数：**
- `checkpoint_path` (str): checkpoint 目录路径
- `device` (str): 设备，默认 `"cuda"`
- `torch_dtype`: 精度，默认 `torch.float16`
- `load_8bit` (bool): 8bit 量化（支持有限）
- `load_4bit` (bool): 4bit 量化（支持有限）

**返回：**
- 同 `load_hf_model()`

### `load_model_for_eval()`

统一接口，自动判断加载来源。

**参数：**
- `model_path` (str): 模型路径或 HuggingFace ID
- `source` (str): 加载来源，可选：
  - `"auto"`: 自动判断（默认）
  - `"huggingface"`: 强制从 HuggingFace 加载
  - `"local"`: 强制从本地加载
- 其他参数同上

**返回：**
- 同 `load_hf_model()`

## 测试

### 测试新的加载函数

```bash
cd /home/jsun/llava-kd

# 测试 HuggingFace 加载
python3.12 llavakd/model/eval_model_load.py --test-hf

# 测试本地加载
python3.12 llavakd/model/eval_model_load.py \
    --test-local /home/jsun/llava-kd/checkpoints/qwen25-0_5b-distill-after-sft

# 测试自动判断
python3.12 llavakd/model/eval_model_load.py \
    --test-auto Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP
```

### 快速评估测试

```bash
# 使用测试脚本
bash scripts/eval/test_new_loading.sh
```

## 与原有方式的比较

| 特性 | 原有方式 | 新方式 (HuggingFace) | 新方式 (Local) |
|------|---------|---------------------|----------------|
| HF 官方模型 | ✅ 支持 | ✅ 支持 | ❌ 不适用 |
| 本地完整 checkpoint | ✅ 支持 | ❌ 不适用 | ✅ 支持 |
| 蒸馏训练模型 | ✅ 支持 (`--load-distill`) | ❌ 不适用 | ❌ 不适用 |
| 量化加载 | ✅ 支持 | ✅ 完整支持 | ⚠️ 有限支持 |
| 代码修改 | 无需修改 | 使用 `_v2` 脚本 | 使用 `_v2` 脚本 |

## 注意事项

1. **本地加载的量化支持有限**：如需量化，建议使用 HuggingFace 加载方式
2. **确保 checkpoint 完整**：本地加载需要所有三个组件的权重文件
3. **向后兼容**：原有的评估脚本和训练流程不受影响
4. **显存管理**：大模型建议使用量化或串行评估

## 常见问题

### Q: 如何判断我的模型应该用哪种加载方式？

A: 
- 如果是从 HuggingFace 下载的官方模型 → 使用 `huggingface` 模式
- 如果是自己训练的完整 checkpoint（包含三个组件）→ 使用 `local` 模式
- 如果是蒸馏训练的 checkpoint（仅 connector）→ 使用原有方式 `--load-distill`
- 不确定 → 使用 `auto` 模式

### Q: 新方式会影响训练吗？

A: 不会。新的加载函数仅用于评估，训练流程使用原有的 `load_model.py`。

### Q: 可以在原有脚本中使用新方式吗？

A: 可以。只需在调用时添加参数：
```bash
python3.12 -m llavakd.eval.model_vqa_loader_v2 --load-source huggingface ...
```

## 更多示例

查看以下文件获取更多示例：
- `llavakd/model/eval_model_load.py` - 加载函数实现
- `llavakd/eval/model_vqa_loader_v2.py` - 支持新加载方式的评估脚本
- `scripts/eval/gqa_v2.sh` - GQA 评估示例
- `scripts/eval/eval_all_v3.sh` - 完整评估示例
