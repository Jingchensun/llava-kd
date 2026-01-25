# LLaVA-KD 知识蒸馏训练流程详细分析

本文档详细追踪 LLaVA-KD 从数据加载到损失计算的完整训练流程，包括所有中间维度变化和关键代码实现。

## 📋 目录

1. [输入数据格式与维度](#1-输入数据格式与维度)
2. [Teacher模型结构与维度变化](#2-teacher模型结构与维度变化)
3. [Student模型结构与维度变化](#3-student模型结构与维度变化)
4. [损失函数详细分析](#4-损失函数详细分析)
5. [反向传播与参数更新](#5-反向传播与参数更新)
6. [完整训练流程图](#6-完整训练流程图)
7. [数据维度流动总表](#7-数据维度流动总表)
8. [关键代码位置索引](#8-关键代码位置索引)

---

## 1. 输入数据格式与维度

### 1.1 原始数据格式

**数据集**: BLIP-LAION-CC-SBU-558K (558,128 样本)

**单条样本结构**:

```json
{
  "id": "004539375",
  "image": "00453/004539375.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "Render a clear and concise summary of the photo.\n<image>"
    },
    {
      "from": "gpt",
      "value": "select luxury furniture 3 - inch gel memory foam mattress topper"
    }
  ]
}
```

**关键代码**: `llavakd/data/dataset.py:22-71`

---

### 1.2 数据加载流程

#### **Step 1: Dataset.__getitem__**

```python
def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    sources = self.list_data_dict[i]
    
    # 文本预处理: conversations → token IDs
    data_dict = self.text_preprocess(sources["conversations"])
    
    # 图像预处理: PIL Image → Tensor
    if 'image' in sources:
        image = Image.open(os.path.join(image_folder, image_file))
        image = self.image_preprocess(image)  # [3, 384, 384]
        data_dict['image'] = image
    
    return data_dict
```

**输出**: 
```python
{
    'input_ids': [seq_len],        # Token IDs
    'labels': [seq_len],           # Target tokens (问题部分=-100)
    'attention_mask': [seq_len],   # 1=有效, 0=padding
    'image': [3, 384, 384]         # 归一化的RGB图像
}
```

---

#### **Step 2: 文本预处理**

**关键代码**: `llavakd/data/text_preprocess.py:11-12`

```python
def __call__(self, messages, mode='train'):
    return self.template.encode(messages, self.tokenizer, mode)
```

**处理流程**:
1. 对话转文本: `"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"`
2. Tokenization: 文本 → Token IDs
3. 标签生成: 问题部分标记为 `-100` (不计算loss)

**示例**:
```
原始对话:
  User: "Describe the image. <image>"
  Assistant: "A cat sitting on a sofa."

Token IDs:
  [151644, 872, 198, 75885, 279, 2168, 13, 151652, ...]
  
Labels:
  [-100, -100, -100, ..., 32, 8280, 11162, 389, 264, 33564, 13, 151645]
  └─────问题部分(不计loss)─────┘  └────答案部分(计算loss)────┘
```

---

#### **Step 3: 图像预处理**

**关键代码**: `llavakd/data/image_preprocess.py:18-25`

```python
def __call__(self, image):
    # SigLIP Preprocessor
    image = self.image_processor(image, return_tensors='pt')
    return image['pixel_values'][0]  # [3, 384, 384]
```

**处理步骤**:
1. **Resize**: 原始图像 → 384×384
2. **Normalize**: RGB → `(pixel - mean) / std`
   - Mean: `[0.5, 0.5, 0.5]`
   - Std: `[0.5, 0.5, 0.5]`
3. **Tensor转换**: `numpy.ndarray` → `torch.Tensor`

**输出维度**: `[3, 384, 384]`

---

### 1.3 Batch Collation

**DataLoader** 自动将多个样本拼接成batch:

```python
# 假设 batch_size = 1 (per_device)
batch = {
    'input_ids': [B, seq_len],      # [1, 150]
    'labels': [B, seq_len],         # [1, 150]
    'attention_mask': [B, seq_len], # [1, 150]
    'images': [B, 3, 384, 384]      # [1, 3, 384, 384]
}
```

**注意**: 
- `seq_len` 动态变化 (padding到batch最大长度)
- `<image>` 占位符的 token_id = `-200` (`IMAGE_TOKEN_INDEX`)

---

## 2. Teacher模型结构与维度变化

### 2.1 Teacher模型配置

**模型**: TinyLLaVA-Qwen2.5-3B

| 组件 | 模型 | 参数量 | Hidden Size |
|------|------|--------|-------------|
| **Vision Tower** | SigLIP-SO400M | 400M | 1152 |
| **Connector** | MLP2x_GELU | 2.4M | 1152→2048 |
| **Language Model** | Qwen2.5-3B | 3.0B | 2048 |

**总参数**: ~3.4B

---

### 2.2 Teacher前向传播

#### **阶段1: Vision Encoding**

**关键代码**: `llavakd/model/modeling_LLaVA_KD.py:195-203`

```python
def encode_images(self, images):
    # images: [B, 3, 384, 384]
    image_features = self.vision_tower(images, 
                                       vision_feature_layer=-2,
                                       vision_feature_select_strategy='default')
    # -> [B, num_patches, vision_hidden_size]
    image_features = self.connector(image_features)
    # -> [B, num_patches, llm_hidden_size]
    return image_features
```

**维度变化**:

```
输入: images [B, 3, 384, 384]

  ↓ Patch Embedding (Conv2d 3→1152, kernel=14, stride=14)
  
[B, 729, 1152]
  - 729 = (384 / 14) × (384 / 14) = 27 × 27
  - 每个patch: 14×14像素

  ↓ SigLIP Transformer (27层, hidden=1152)
  
[B, 729, 1152]
  - 提取第 -2 层 (倒数第二层) 的输出

  ↓ Connector: MLP2x_GELU
     Linear(1152 → 2048) + GELU + Linear(2048 → 2048)
  
输出: [B, 729, 2048]
  - 对齐到 Teacher LLM 的 hidden_size
```

---

#### **阶段2: 多模态融合**

**关键代码**: `llavakd/model/modeling_LLaVA_KD.py:220-370`

```python
def prepare_inputs_labels_for_multimodal(
    self, input_ids, position_ids, attention_mask, 
    past_key_values, labels, images, image_sizes=None
):
    # 1. 编码图像
    image_features = self.encode_images(images)  # [B, 729, 2048]
    
    # 2. 文本embedding
    text_embeds = self.language_model.get_input_embeddings()(input_ids)
    # [B, text_len, 2048]
    
    # 3. 找到 <image> 占位符 (token_id = -200)
    image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
    
    # 4. 替换占位符为图像特征
    new_input_embeds = torch.cat([
        text_embeds[:image_pos],      # 图像前的文本
        image_features,               # 729个图像tokens
        text_embeds[image_pos+1:]     # 图像后的文本
    ], dim=0)
    
    return new_input_embeds, new_labels, ...
```

**融合示例**:

```
原始序列 (text_len=150):
[system_prompt: 30 tokens] + [<image>: 1 token] + [question: 70 tokens] + [answer: 49 tokens]

融合后序列 (seq_len=878):
[system_prompt: 30 tokens] + [image_features: 729 tokens] + [question: 70 tokens] + [answer: 49 tokens]

维度: [B, 878, 2048]
```

**Label对应**:
```
labels: [-100 × 30] + [-100 × 729] + [-100 × 70] + [real_token_ids × 49]
         └─系统提示─┘   └─图像特征─┘   └─问题部分─┘   └─答案部分(计loss)─┘
```

---

#### **阶段3: Language Model**

**关键代码**: `llavakd/model/modeling_LLaVA_KD.py:142-153`

```python
return self.language_model.forward(
    input_ids=None,               # 不使用token IDs
    inputs_embeds=inputs_embeds,  # 直接使用embeddings
    attention_mask=attention_mask,
    labels=labels,
    ...
)
```

**Qwen2.5-3B 内部流程**:

```
输入: inputs_embeds [B, 878, 2048]

  ↓ 36 × Decoder Layer:
       - Multi-Head Attention (32 heads, head_dim=64)
       - Feed-Forward Network (intermediate_size=11008)
       - Layer Norm + Residual
  
[B, 878, 2048]  # hidden_states

  ↓ LM Head: Linear(2048 → 151936)
     151936 = Qwen2.5 词表大小
  
输出: logits [B, 878, 151936]
```

**Teacher输出**:

```python
teacher_outputs = {
    'logits': torch.FloatTensor([B, 878, 151936]),  # 每个位置的词表分布
    'loss': torch.FloatTensor([]),                  # CE loss (scalar)
    'hidden_states': [...],                         # 各层隐藏状态 (可选)
    'attentions': [...]                             # 注意力权重 (可选)
}

# 同时返回额外信息:
multimodal_labels: [B, 878]           # 融合后的完整labels
image_rela: [B, 729, 729]             # 图像特征相关矩阵 (用于蒸馏)
num_images: int                       # 本batch的图像数量
split_sizes: List[List[int]]          # 文本分段信息
```

---

## 3. Student模型结构与维度变化

### 3.1 Student模型配置

**模型**: TinyLLaVA-Qwen2.5-0.5B

| 组件 | 模型 | 参数量 | Hidden Size | 是否训练 |
|------|------|--------|-------------|---------|
| **Vision Tower** | SigLIP-SO400M (复制) | 400M | 1152 | ❌ 冻结 |
| **Connector** | MLP2x_GELU (新建) | 1.8M | 1152→**896** | ✅ **训练** |
| **Language Model** | Qwen2.5-0.5B | 500M | **896** | ❌ 冻结 |

**总参数**: ~902M  
**可训练参数**: **1.8M (0.2%)**

---

### 3.2 关键区别

与 Teacher 的主要差异：

| 属性 | Teacher | Student |
|------|---------|---------|
| **LLM层数** | 36层 | 24层 |
| **Hidden Size** | 2048 | **896** |
| **注意力头数** | 32 | 14 |
| **FFN中间维度** | 11008 | 4864 |
| **Connector输出** | 2048 | **896** |

---

### 3.3 Student前向传播

**完全相同的流程**，但维度不同：

```
images [B, 3, 384, 384]

  ↓ Vision Tower (从Teacher复制，冻结)
  
[B, 729, 1152]  # 与Teacher相同

  ↓ Connector (新训练的!)
     Linear(1152 → 896) + GELU + Linear(896 → 896)
  
[B, 729, 896]  # 对齐到 Student LLM 的 hidden_size

  ↓ 融合文本
  
[B, 878, 896]  # 混合embeddings

  ↓ Qwen2.5-0.5B (24层Decoder)
  
[B, 878, 896]  # hidden_states

  ↓ LM Head: Linear(896 → 151936)
  
输出: logits [B, 878, 151936]  # 词表大小与Teacher相同!
```

**Student输出**:
```python
student_outputs = {
    'logits': [B, 878, 151936],  # 与Teacher维度完全相同
    'loss': scalar,
    ...
}
```

---

## 4. 损失函数详细分析

### 4.1 总损失公式

**关键代码**: `llavakd/train/tinyllava_distill_trainer.py:259-359`

```python
def compute_loss(self, model, inputs, return_outputs=False):
    # 1. Teacher前向 (无梯度)
    with torch.no_grad():
        teacher_outputs, multimodal_teacher_labels, ... = self.teacher_model(**inputs)
    
    # 2. Student前向 (有梯度)
    student_outputs, multimodal_labels, ... = model(**inputs)
    
    # 3. 计算损失
    loss = L1_ce_loss + L2_kl_distill + L3_visual_distill
    
    return loss
```

**损失组成**:
```
Total Loss = L1: Ground Truth CE Loss
           + L2: KL Divergence (全文本)
           + L3a: Visual KL Divergence (图像部分)
           + L3b: Visual Relation Loss (图像相关性)
```

---

### 4.2 Loss 1: 标准交叉熵损失

**目的**: 确保Student生成正确答案

**计算**:
```python
loss = student_outputs['loss']  # 模型内置的CrossEntropyLoss
```

**公式**:
```
L_CE = - Σ log P_student(y_true | x)
     = CrossEntropy(student_logits, ground_truth_labels)
```

**维度**:
```
student_logits: [B, 878, 151936]
labels: [B, 878]
  - 只有 answer 部分有效 (其他=-100)
  - 有效tokens数: ~49

L_CE: scalar
```

**代码位置**: `modeling_LLaVA_KD.py:142-153` (LLM内部)

---

### 4.3 Loss 2: 知识蒸馏 (KL散度)

**目的**: 让Student学习Teacher的**软标签分布**

#### **计算流程**

**关键代码**: `tinyllava_distill_trainer.py:299-329`

```python
# 1. 提取logits (去掉最后一个位置，因为没有next token)
shift_student_logits = outputs['logits'][..., :-1, :].contiguous()
shift_teacher_logits = teacher_outputs['logits'][..., :-1, :].contiguous()
# [B, 877, 151936]

# 2. 对应的labels (右移1位)
shift_labels = multimodal_labels[..., 1:].contiguous()
shift_labels = shift_labels.view(-1)  # [B*877]

# 3. Mask: 只在 answer 部分计算KL
mask = torch.ne(shift_labels, -100).int()  # [B*877]

# 4. 提取有效位置的logits
masked_student_logits = shift_student_logits.view(-1, voc_size)[mask.bool()]
masked_teacher_logits = shift_teacher_logits.view(-1, voc_size)[teacher_mask.bool()]
# [N, 151936], N ≈ 48 (answer部分的token数)

# 5. KL散度
forward_distillation_loss = nn.KLDivLoss(reduction="batchmean")(
    F.log_softmax(masked_student_logits, dim=-1),  # log P_student
    F.softmax(masked_teacher_logits, dim=-1)       # P_teacher
)
```

#### **数学公式**

```
L_KL = KL(P_teacher || P_student)
     = Σ P_teacher(w) × log(P_teacher(w) / P_student(w))
     
其中:
- P_teacher: Teacher的概率分布 (softmax后)
- P_student: Student的概率分布 (softmax后)
- w: 词表中的每个token
```

#### **为什么用KL散度？**

**硬标签** vs **软标签**:

```
示例: "A cat is sitting."

Ground Truth (硬标签):
  P(cat) = 1.0
  P(dog) = 0.0
  P(kitten) = 0.0

Teacher Output (软标签):
  P(cat) = 0.7
  P(kitten) = 0.2    ← 包含额外知识!
  P(dog) = 0.05
  P(feline) = 0.03
  
KL散度让Student学习这些"dark knowledge"
```

#### **维度总结**

```
输入:
  student_logits: [B, 877, 151936]
  teacher_logits: [B, 877, 151936]
  
  ↓ Apply mask (只保留answer部分)
  
  masked_student: [48, 151936]
  masked_teacher: [48, 151936]
  
  ↓ Softmax + KL Divergence
  
输出: L_KL (scalar)
```

---

### 4.4 Loss 3: 视觉蒸馏

**目的**: 保持图像表示的结构和关系

#### **3a. 视觉Token KL散度**

**关键代码**: `tinyllava_distill_trainer.py:332-338`

```python
if num_images == 1:
    # 提取图像部分的logits (729个tokens)
    # split_sizes[0][0] = 图像前文本长度
    start = split_sizes[0][0]
    
    shift_STU_image_logits = outputs['logits'][:, start:start+729]
    shift_Tea_image_logits = teacher_outputs['logits'][:, start:start+729, :]
    # [B, 729, 151936]
    
    # 计算KL散度
    forward_visual_distillation = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(shift_STU_image_logits.view(-1, voc_size), dim=-1),
        F.softmax(shift_Tea_image_logits.view(-1, voc_size), dim=-1)
    )
```

**维度**:
```
image_logits: [B, 729, 151936]
  ↓ Reshape
[B*729, 151936] = [729, 151936]
  ↓ KL Divergence
scalar
```

---

#### **3b. 视觉关系矩阵蒸馏**

**目的**: 保持图像tokens之间的**关系结构**

**关键代码**: `tinyllava_distill_trainer.py:340-346`

```python
# 1. 计算自相关矩阵
student_img_rela = torch.matmul(
    shift_STU_image_logits,              # [B, 729, 151936]
    shift_STU_image_logits.permute(0,2,1) # [B, 151936, 729]
)  # -> [B, 729, 729]

teacher_img_rela = torch.matmul(
    shift_Tea_image_logits,
    shift_Tea_image_logits.permute(0,2,1)
)  # -> [B, 729, 729]

# 2. Cosine相似度
llm_visual_rela_distill_loss = 1 - F.cosine_similarity(
    teacher_img_rela.view(-1).unsqueeze(0),  # [1, 729*729]
    student_img_rela.view(-1).unsqueeze(0)   # [1, 729*729]
).mean()
```

**数学解释**:

```
关系矩阵 R[i,j] 表示 token_i 和 token_j 的相关性

R = Image_logits × Image_logits^T
  = [729, 151936] × [151936, 729]
  = [729, 729]

R[i,j] = Σ logit_i[k] × logit_j[k]
         k=1...151936

目标: Student的R 应该接近 Teacher的R
Loss = 1 - cosine_sim(R_teacher, R_student)
```

**为什么要这样做？**

```
保持token间关系有助于学习图像的结构信息:
- R[0,1]: "左上角patch" 和 "中上角patch" 的关系
- R[0,728]: "左上角patch" 和 "右下角patch" 的关系

即使Student的hidden_size更小(896 vs 2048)，
也要保持这些patch之间的相对关系!
```

---

### 4.5 最终损失

**组合代码**: `tinyllava_distill_trainer.py:329, 346`

```python
# 初始loss (CE)
loss = outputs['loss']

# 加上文本KL
loss = loss + forward_distillation_loss

# 如果有图像，加上视觉蒸馏
if num_images == 1:
    loss = loss + forward_visual_distillation + llm_visual_rela_distill_loss
```

**权重分配**:
```
Total Loss = 1.0 × L_CE
           + 1.0 × L_KL_text
           + 1.0 × L_KL_visual
           + 1.0 × L_relation

(所有权重默认为1，未使用额外系数)
```

---

## 5. 反向传播与参数更新

### 5.1 梯度流动路径

```
Loss (scalar)
  │
  ├─ ∂L/∂L_CE
  ├─ ∂L/∂L_KL_text
  ├─ ∂L/∂L_KL_visual
  └─ ∂L/∂L_relation
  │
  ↓ 反向传播
  
student_logits [B, 878, 151936]
  │
  ↓ ∂L/∂logits
  
LM Head [B, 878, 896]
  │ (冻结，不更新参数)
  ↓ 梯度继续传播
  
hidden_states [B, 878, 896]
  │
  ↓ 反向传播通过24层Decoder (冻结)
  
inputs_embeds [B, 878, 896]
  │
  ↓ 只在图像tokens位置有梯度
  
image_embeds [B, 729, 896]
  │
  ↓ 反向传播到Connector (可训练!)
  
Connector:
  ├─ Linear(896 → 896).bias:    ∂L/∂bias
  ├─ Linear(896 → 896).weight:  ∂L/∂weight
  ├─ Linear(1152 → 896).bias:   ∂L/∂bias
  └─ Linear(1152 → 896).weight: ∂L/∂weight
  │
  ↓ 停止! Vision Tower冻结
  
[不再继续传播]
```

---

### 5.2 可训练参数统计

**关键代码**: `llavakd/model/connector/mlp.py:25-30`

```python
self._connector = nn.Sequential(
    nn.Linear(1152, 896),  # Layer 0
    nn.GELU(),             # 无参数
    nn.Linear(896, 896)    # Layer 2
)
```

**参数量**:

| 参数 | 维度 | 数量 |
|------|------|------|
| `_connector.0.weight` | [896, 1152] | 1,032,192 |
| `_connector.0.bias` | [896] | 896 |
| `_connector.2.weight` | [896, 896] | 802,816 |
| `_connector.2.bias` | [896] | 896 |
| **总计** | - | **1,836,800** |

**占比**: 1.8M / 4.3B (所有参数) = **0.042%**

---

### 5.3 优化器配置

**关键代码**: `tinyllava_distill_trainer.py:166-256`

```python
def create_optimizer(self):
    # 只选择requires_grad=True的参数
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                       if "connector" in n and p.requires_grad],
            "weight_decay": 0.0,
            "lr": 1e-3
        }
    ]
    
    # AdamW优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3)
    return optimizer
```

**优化器状态**:
```
每个参数需要存储:
- 参数本身: 1.8M × 4B = 7.2 MB (FP32)
- 一阶动量: 1.8M × 4B = 7.2 MB
- 二阶动量: 1.8M × 4B = 7.2 MB

总计: 21.6 MB (单GPU)

使用DeepSpeed ZeRO-2分片 (4-GPU):
每个GPU只存储: 21.6 MB / 4 = 5.4 MB ✅
```

---

### 5.4 参数更新

**每个训练步**:

```python
# 1. 梯度累积 (16步)
for micro_step in range(16):
    loss = compute_loss(model, batch[micro_step])
    loss = loss / 16  # 归一化
    loss.backward()   # 梯度累加
    
# 2. 梯度归约 (多GPU同步)
# DeepSpeed自动处理: All-Reduce梯度

# 3. 参数更新
optimizer.step()  # 更新1.8M参数
optimizer.zero_grad()

# 4. 学习率调度
scheduler.step()
```

**单步耗时**:
```
Forward (Teacher + Student): ~1.5s
Backward (仅Connector):     ~0.3s
Optimizer Step:              ~0.01s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计:                        ~1.8s/step
```

---

## 6. 完整训练流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT BATCH                              │
│  Image: [1, 3, 384, 384]                                    │
│  Text IDs: [1, 150]  (含<image>占位符)                      │
│  Labels: [1, 150]  (问题部分=-100)                          │
└─────────────────────────────────────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
         ┌──────▼─────┐        ┌─────▼──────┐
         │  TEACHER   │        │  STUDENT   │
         │  (冻结)    │        │ (部分训练) │
         └──────┬─────┘        └─────┬──────┘
                │                     │
        ┌───────▼────────┐    ┌──────▼───────┐
        │ Vision Tower   │    │ Vision Tower │
        │ SigLIP-SO400M  │    │ SigLIP-SO400M│
        │ [1,729,1152]   │    │ [1,729,1152] │
        │   ❌冻结       │    │   ❌冻结     │
        └───────┬────────┘    └──────┬───────┘
                │                     │
        ┌───────▼────────┐    ┌──────▼───────┐
        │  Connector     │    │  Connector   │
        │  MLP2x_GELU    │    │  MLP2x_GELU  │
        │ [1,729,2048]   │    │ [1,729,896]  │
        │   ❌冻结       │    │  ✅可训练!   │← 唯一训练部分
        └───────┬────────┘    └──────┬───────┘
                │                     │
        ┌───────▼────────┐    ┌──────▼───────┐
        │ 融合文本embeddings │ │ 融合文本embeddings│
        │ [1,878,2048]   │    │ [1,878,896]  │
        │ 30(sys)+729(img)│    │ 30(sys)+729(img)│
        │ +70(q)+49(a)   │    │ +70(q)+49(a) │
        └───────┬────────┘    └──────┬───────┘
                │                     │
        ┌───────▼────────┐    ┌──────▼───────┐
        │  Qwen2.5-3B    │    │ Qwen2.5-0.5B │
        │  36层Decoder   │    │ 24层Decoder  │
        │ [1,878,2048]   │    │ [1,878,896]  │
        │   ❌冻结       │    │   ❌冻结     │
        └───────┬────────┘    └──────┬───────┘
                │                     │
        ┌───────▼────────┐    ┌──────▼───────┐
        │  LM Head       │    │  LM Head     │
        │ [1,878,151936] │    │ [1,878,151936]│
        └───────┬────────┘    └──────┬───────┘
                │                     │
                └──────────┬──────────┘
                           │
                    ┌──────▼──────────────────────┐
                    │       LOSS COMPUTATION      │
                    │                             │
                    │ L1: CE Loss (answer部分)    │
                    │    = -log P(y_true)         │
                    │                             │
                    │ L2: KL Divergence (text)    │
                    │    = KL(P_T || P_S)         │
                    │    只在answer部分计算        │
                    │                             │
                    │ L3a: KL Divergence (visual) │
                    │     = KL on 729 img tokens  │
                    │                             │
                    │ L3b: Relation Loss          │
                    │     = 1 - cos_sim(R_T, R_S) │
                    │     R = [729, 729] 相关矩阵  │
                    │                             │
                    │ Total = L1 + L2 + L3a + L3b │
                    └──────┬──────────────────────┘
                           │
                    ┌──────▼──────────────────────┐
                    │    BACKWARD PROPAGATION     │
                    │                             │
                    │ 梯度流动:                    │
                    │ Loss → Logits → Embeddings  │
                    │      → Connector ✅          │
                    │      → Vision Tower ❌停止   │
                    │                             │
                    │ 更新参数:                    │
                    │ - Connector: 1.8M params    │
                    │ - 其他: 冻结                │
                    └──────┬──────────────────────┘
                           │
                    ┌──────▼──────────────────────┐
                    │   OPTIMIZER STEP (AdamW)    │
                    │   Learning Rate: 1e-3       │
                    │   Weight Decay: 0.0         │
                    │   DeepSpeed ZeRO-2 分片     │
                    └─────────────────────────────┘
```

---

## 7. 数据维度流动总表

| 阶段 | Teacher维度 | Student维度 | 是否可训练 | 说明 |
|------|------------|------------|-----------|------|
| **原始输入** | | | | |
| - Image | `[1, 3, 384, 384]` | `[1, 3, 384, 384]` | - | RGB图像 |
| - Text IDs | `[1, 150]` | `[1, 150]` | - | Token序列 |
| **Vision Encoding** | | | | |
| - Patches | `[1, 729, 1152]` | `[1, 729, 1152]` | ❌ 冻结 | 27×27 patches |
| **Connector** | | | | |
| - Projected | `[1, 729, 2048]` | `[1, 729, 896]` | ✅ **训练** | 对齐LLM维度 |
| **Multimodal Fusion** | | | | |
| - Text Embed | `[1, 150, 2048]` | `[1, 150, 896]` | ❌ 冻结 | LLM Embedding |
| - Mixed Embed | `[1, 878, 2048]` | `[1, 878, 896]` | - | 文本+图像 |
| **Language Model** | | | | |
| - Hidden | `[1, 878, 2048]` | `[1, 878, 896]` | ❌ 冻结 | Decoder输出 |
| - Logits | `[1, 878, 151936]` | `[1, 878, 151936]` | - | 词表分布 |
| **Loss Computation** | | | | |
| - Answer Mask | `[48]` | `[48]` | - | 有效token数 |
| - Masked Logits | `[48, 151936]` | `[48, 151936]` | - | 用于KL |
| - Image Logits | `[729, 151936]` | `[729, 151936]` | - | 视觉蒸馏 |
| - Relation Matrix | `[729, 729]` | `[729, 729]` | - | 相关性矩阵 |

**注释**:
- `B=1`: batch_size per device
- `878 = 30(system) + 729(image) + 70(question) + 49(answer)`
- `48 ≈ answer部分的token数`
- `151936`: Qwen2.5词表大小

---

## 8. 关键代码位置索引

### 8.1 数据处理

| 功能 | 文件 | 行数 | 说明 |
|------|------|------|------|
| **Dataset类** | `llavakd/data/dataset.py` | 22-71 | LazySupervisedDataset |
| **数据加载** | `llavakd/data/dataset.py` | 57-71 | `__getitem__` |
| **图像预处理** | `llavakd/data/image_preprocess.py` | 18-25 | Resize + Normalize |
| **文本预处理** | `llavakd/data/text_preprocess.py` | 6-12 | Template + Tokenize |
| **DataCollator** | `llavakd/data/dataset.py` | 74-128 | Padding + Batching |

---

### 8.2 模型结构

| 功能 | 文件 | 行数 | 说明 |
|------|------|------|------|
| **模型定义** | `llavakd/model/modeling_LLaVA_KD.py` | 57-432 | LLaVAKD类 |
| **Forward方法** | `llavakd/model/modeling_LLaVA_KD.py` | 106-153 | 主前向传播 |
| **图像编码** | `llavakd/model/modeling_LLaVA_KD.py` | 195-203 | `encode_images` |
| **多模态融合** | `llavakd/model/modeling_LLaVA_KD.py` | 220-370 | `prepare_inputs_labels_for_multimodal` |
| **Connector** | `llavakd/model/connector/mlp.py` | 18-40 | MLP2x_GELU |

---

### 8.3 训练流程

| 功能 | 文件 | 行数 | 说明 |
|------|------|------|------|
| **Trainer类** | `llavakd/train/tinyllava_distill_trainer.py` | 129-362 | DistillLLaVATrainer |
| **初始化** | `llavakd/train/tinyllava_distill_trainer.py` | 131-148 | Teacher冻结 |
| **损失计算** | `llavakd/train/tinyllava_distill_trainer.py` | 259-359 | `compute_loss` |
| **优化器创建** | `llavakd/train/tinyllava_distill_trainer.py` | 166-256 | `create_optimizer` |
| **Sampler** | `llavakd/train/tinyllava_distill_trainer.py` | 150-164 | 长度分组采样 |

---

### 8.4 损失函数

| 功能 | 文件 | 行数 | 说明 |
|------|------|------|------|
| **CE Loss** | `modeling_LLaVA_KD.py` | 142-153 | LLM内置 |
| **KL Divergence (文本)** | `tinyllava_distill_trainer.py` | 312-329 | 答案部分蒸馏 |
| **KL Divergence (视觉)** | `tinyllava_distill_trainer.py` | 332-338 | 图像tokens蒸馏 |
| **Relation Loss** | `tinyllava_distill_trainer.py` | 340-346 | 相关矩阵蒸馏 |
| **Loss工具函数** | `llavakd/utils/distill_loss_utils.py` | 1-130 | KL变体 |

---

### 8.5 训练脚本

| 功能 | 文件 | 说明 |
|------|------|------|
| **主训练脚本** | `scripts/train/llava_kd_qwen2/train_qwen2_distill.sh` | 调用预训练脚本 |
| **DeepSpeed配置** | `scripts/train/llava_kd_qwen2/pretrain_qwen2_distill.sh` | 所有训练参数 |
| **ZeRO-2配置** | `scripts/zero2.json` | DeepSpeed优化 |
| **训练主程序** | `llavakd/train/train_distill_qwen2.py` | Python入口 |

---

## 📊 附录: 实验数据

### 显存占用 (per GPU, 4×H100 80GB)

| 组件 | 显存占用 |
|------|---------|
| Teacher模型 (冻结) | ~7 GB |
| Student模型 (冻结LLM+VT) | ~3 GB |
| Connector (训练) | ~0.01 GB |
| 激活值 (FP16) | ~10 GB |
| 优化器状态 (ZeRO-2) | ~1 GB |
| 梯度 (ZeRO-2) | ~1 GB |
| DeepSpeed开销 | ~2 GB |
| **总计** | **~24 GB** |

### 训练速度

```
配置: 4-GPU, batch=1, grad_accum=16
- Samples/s: ~1.8
- Seconds/iter: ~2.2
- 训练558K样本预计: ~86小时
```

### 参数分布

```
总参数: 4.3B
- Teacher: 3.4B (冻结)
- Student Vision: 400M (冻结)
- Student LLM: 500M (冻结)
- Student Connector: 1.8M (✅训练)

可训练比例: 0.04%
```

---

---

## ❓ 常见疑问 FAQ

### Q1: Vision Encoder都冻结了，L3a和L3b损失还有必要吗？

**A**: 有必要！虽然Teacher和Student的Vision Encoder相同且冻结，但：

1. **损失计算位置**: L3a/L3b是在**LLM输出的logits**上计算，不是在Vision Encoder的输出上
   ```python
   # 提取的是 LLM 的输出，不是 Vision Features
   shift_STU_image_logits = outputs['logits'][:, start:start+729]
   # 维度: [729, 151936] ← 词表概率分布
   ```

2. **差异来源**: 
   - Vision Features相同: `[729, 1152]` ✅
   - 但Connector不同: Teacher `1152→2048`, Student `1152→896` ❌
   - LLM架构不同: Teacher 3B vs Student 0.5B ❌
   - 最终logits不同: 对图像的"理解"不同 ❌

3. **L3a/L3b的作用**:
   - L3a: 让Student学会像Teacher一样"理解"图像
   - L3b: 保持图像patches之间的关系结构
   - 这些约束帮助Student更好地利用视觉信息

4. **实验验证**: 可以通过消融实验验证是否真的有用

---

**文档更新**: 2026-01-24  
**训练环境**: 4×NVIDIA H100 80GB, CUDA 12.4, PyTorch 2.6.0, DeepSpeed 0.16.2
