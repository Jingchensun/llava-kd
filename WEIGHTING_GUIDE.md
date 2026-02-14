# 损失加权策略 (Loss Weighting) 完整指南

## 📚 概述

本项目实现了三种知识蒸馏的损失加权策略，用于优化多任务学习中的损失平衡。

## 🎯 三种策略对比

### Type1: 等权重 (Equal Weighting)
- **原理**: 简单地将所有损失相加
- **参数量**: 0
- **学习率**: 不涉及
- **公式**: `L_total = L1 + L2 + L3 + L4`
- **适用场景**: 基线方法，快速实验

### Type2: 异方差不确定性 (Heteroscedastic Uncertainty)
- **原理**: 基于 Kendall et al. 2018，学习任务级的不确定性参数
- **参数量**: num_tasks 个标量参数（本项目中为 4 个）
- **学习率**: 基础学习率 × 1000
- **公式**: `L_total = Σ (1/(2σ²) * L_i + log(σ_i))`
- **适用场景**: 任务间权重比例相对稳定的场景
- **优势**: 
  - 参数极少，训练开销小
  - 理论基础扎实
  - 收敛快速

### Type3: 实例条件权重 (Instance-Conditional Weighting)
- **原理**: 使用 MLP 网络根据 teacher 特征动态预测每个样本的权重
- **参数量**: ~524K (4096→128→4 的 MLP)
- **学习率**: 基础学习率 × 10
- **公式**: MLP 预测每个样本的 log_vars，然后应用类似 Type2 的公式
- **适用场景**: 样本难度差异大，需要自适应权重
- **优势**:
  - 样本级动态权重
  - 可以学习复杂的权重模式
  - 对异构数据更友好

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活环境
conda activate llava-kd

# 进入项目目录
cd /home/jsun/llava-kd
```

### 2. 训练命令

#### 方法 A: 使用便捷脚本

```bash
# Type1 (默认)
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft.sh

# Type2 (推荐)
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft_type2.sh

# Type3 (高级)
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft_type3.sh
```

#### 方法 B: 使用测试脚本

```bash
# Type2 快速测试（包含环境检查）
bash scripts/train/llava_kd_qwen2/RUN_TYPE2_TEST.sh
```

#### 方法 C: 手动指定参数

```bash
# 传递参数给主脚本
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft.sh type2
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft.sh type3
```

## 📊 训练输出说明

### Type2 输出示例

```
🔧 Distillation Weighting Strategy: type2
  Feature dim: 1536
  Num tasks: 4
  Hidden dim: 128

🔧 Distillation Weighting Parameters:
  - distill_weighting_strategy.log_vars: shape=torch.Size([4]), dtype=torch.float32
  Learning Rate: 2.00e-02 (1000.0x base LR for type2)

[Step 100] Weights - Main Loss: 0.5234, Logits Distill Loss: 0.3421, Forward Visual Distillation: 0.0892, Llm Visual Rela Distill Loss: 0.0453
[Step 100] Weighted - Main Loss: 1.2345, Logits Distill Loss: 0.8765, Forward Visual Distillation: 0.2345, Llm Visual Rela Distill Loss: 0.1234
[Step 100] Log_vars - [0.6543, -0.3421, 2.1234, 3.4567]
```

**解释**:
- `Weights`: 各任务的当前权重（precision = exp(-log_vars)）
- `Weighted`: 加权后的各任务损失值
- `Log_vars`: 学习的不确定性参数（越大表示该任务不确定性越高，权重越低）

### Type3 输出示例

```
🔧 Distillation Weighting Strategy: type3
  Feature dim: 1536
  Num tasks: 4
  Hidden dim: 128

🔧 Distillation Weighting Parameters:
  - distill_weighting_strategy.uncertainty_predictor.0.weight: shape=torch.Size([128, 1536])
  - distill_weighting_strategy.uncertainty_predictor.0.bias: shape=torch.Size([128])
  - distill_weighting_strategy.uncertainty_predictor.2.weight: shape=torch.Size([4, 128])
  - distill_weighting_strategy.uncertainty_predictor.2.bias: shape=torch.Size([4])
  Learning Rate: 2.00e-04 (10.0x base LR for type3)

[Step 100] Weights - Main Loss: 0.4123, Logits Distill Loss: 0.4567, ...
```

## 💾 Checkpoint 保存

### 保存内容

所有策略都会保存：
1. **LLM 权重**: `language_model/pytorch_model.bin`
2. **Connector 权重**: `connector/pytorch_model.bin`
3. **Weighting 权重**: `weighting_strategy.bin` (仅 type2/type3)

### 保存路径

```
./checkpoints/
├── qwen25-0_5b-distill-after-sft-type1/    # Type1
├── qwen25-0_5b-distill-after-sft-type2/    # Type2
└── qwen25-0_5b-distill-after-sft-type3/    # Type3
```

### 中间 Checkpoint

每 2000 步保存一次：
```
./checkpoints/qwen25-0_5b-distill-after-sft-type2/
├── checkpoint-2000/
├── checkpoint-4000/
└── checkpoint-6000/
```

## 📈 Wandb 监控

训练指标会自动记录到 Wandb：

### 通用指标
- `train/llm_loss`: 基础语言模型损失
- `train/total_loss`: 加权后的总损失
- `train/forward_distillation_loss`: 前向 KL 散度损失
- `train/forward_visual_distillation`: 视觉蒸馏损失
- `train/llm_visual_rela_distill_loss`: 视觉关系蒸馏损失

### Type2/Type3 特有指标
- `weights/main_loss_weight`: 主损失权重
- `weights/logits_distill_loss_weight`: 逻辑蒸馏损失权重
- `weights/forward_visual_distillation_weight`: 视觉蒸馏权重
- `weights/llm_visual_rela_distill_loss_weight`: 关系蒸馏权重
- `weighted_losses/*`: 加权后的各项损失

### Type2 特有指标
- `log_vars/task_0`: 主损失的 log_vars
- `log_vars/task_1`: 逻辑蒸馏的 log_vars
- `log_vars/task_2`: 视觉蒸馏的 log_vars
- `log_vars/task_3`: 关系蒸馏的 log_vars

## 🔧 实现细节

### 代码修改总结

1. **Trainer 修改** (`llavakd/train/tinyllava_distill_trainer.py`):
   - 添加 weighting_strategy 初始化
   - 修改 create_optimizer 支持不同学习率
   - 修改 compute_loss 使用加权策略
   - 修改 _save_checkpoint 保存 weighting 权重

2. **参数定义** (`llavakd/utils/arguments.py`):
   - 添加 `distil_ratio_type` 参数

3. **训练脚本** (`scripts/train/llava_kd_qwen2/`):
   - 更新主脚本支持 weighting 参数
   - 创建便捷脚本 (type2/type3)
   - 创建测试脚本

### 学习率设置

```python
# Type2: 基础 LR × 1000
if ratio_strategy == 'type2':
    distill_weighting_lr = base_lr * 1000.0
    
# Type3: 基础 LR × 10
elif ratio_strategy == 'type3':
    distill_weighting_lr = base_lr * 10.0
```

**原因**:
- Type2 参数极少（4个标量），需要更大的学习率快速收敛
- Type3 参数较多（MLP网络），使用适中的学习率避免不稳定

## 🎓 推荐使用策略

### 初次实验
推荐使用 **Type2**：
- 参数少，训练稳定
- 理论基础扎实
- 通常能获得较好效果

### 数据异构性强
推荐使用 **Type3**：
- 可以学习样本级权重
- 适应不同难度的样本
- 需要更长训练时间

### 快速基线
使用 **Type1**：
- 无额外参数
- 训练最快
- 作为对比基线

## 📝 常见问题

### Q1: Type2 的 log_vars 一直为负数正常吗？
**A**: 正常。log_vars 可以是负数，重要的是相对大小。负值对应 σ < 1（高精度/高权重）。

### Q2: Type3 的权重变化很大正常吗？
**A**: 正常。Type3 是样本级权重，不同样本的权重本就不同。观察平均权重的变化趋势即可。

### Q3: 如何选择合适的策略？
**A**: 
1. 先用 Type2 作为基线
2. 如果效果不理想，尝试 Type3
3. 对比 Type1 看是否有提升

### Q4: Checkpoint 加载时会自动加载 weighting 权重吗？
**A**: 当前版本保存了 weighting 权重，但加载需要额外实现。通常蒸馏训练不需要恢复 weighting 权重。

## 📚 参考文献

- **Type2**: Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. CVPR 2018.

## 🆘 获取帮助

如遇问题，请检查：
1. SFT checkpoint 是否存在
2. Conda 环境是否正确激活
3. 查看训练日志中的错误信息
4. 检查 Wandb 日志确认权重是否在更新

---

**开始训练 Type2：**
```bash
conda activate llava-kd
bash scripts/train/llava_kd_qwen2/RUN_TYPE2_TEST.sh
```
