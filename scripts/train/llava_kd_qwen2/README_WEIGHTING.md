# 损失加权策略使用说明

## 快速开始

### 1. 激活环境
```bash
conda activate llava-kd
cd /home/jsun/llava-kd
```

### 2. 运行训练

#### Type1 (默认 - 等权重)
```bash
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft.sh
# 或者显式指定
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft.sh type1
```

#### Type2 (异方差不确定性 - Kendall et al. 2018)
```bash
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft_type2.sh
# 或者
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft.sh type2
```

#### Type3 (实例条件权重 - MLP网络)
```bash
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft_type3.sh
# 或者
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft.sh type3
```

## 三种策略对比

| 策略 | 原理 | 参数量 | 学习率倍数 | 适用场景 |
|------|------|--------|-----------|----------|
| **Type1** | 简单等权重相加 | 0 | 1x | 基线方法 |
| **Type2** | 任务级不确定性学习 | 4个标量 | 1000x | 任务权重稳定 |
| **Type3** | 实例级动态权重 | ~524K (MLP) | 10x | 样本难度差异大 |

## 输出说明

### Type2 训练日志示例
```
🔧 Distillation Weighting Strategy: type2
  Feature dim: 1536
  Num tasks: 4
  Hidden dim: 128

[Step 100] Weights - Main Loss: 0.5234, Logits Distill Loss: 0.3421, Forward Visual Distillation: 0.0892, Llm Visual Rela Distill Loss: 0.0453
[Step 100] Log_vars - [0.6543, -0.3421, 2.1234, 3.4567]
```

### Type3 训练日志示例
```
🔧 Distillation Weighting Strategy: type3
  Feature dim: 1536
  Num tasks: 4
  Hidden dim: 128

🔧 Distillation Weighting Parameters:
  - distill_weighting_strategy.uncertainty_predictor.0.weight: shape=torch.Size([128, 1536])
  - distill_weighting_strategy.uncertainty_predictor.2.weight: shape=torch.Size([4, 128])
  Learning Rate: 2.00e-04 (10.0x base LR for type3)

[Step 100] Weights - Main Loss: 0.4123, Logits Distill Loss: 0.4567, ...
```

## Checkpoint 保存

所有策略的 checkpoint 都会保存：
- LLM 权重
- Connector 权重
- Weighting Strategy 权重 (type2/type3 会额外保存 `weighting_strategy.bin`)

保存路径：`./checkpoints/qwen25-0_5b-distill-after-sft-{type}/`

## Wandb 监控

训练过程会自动记录到 wandb：
- `weights/*`: 各任务的权重
- `weighted_losses/*`: 各任务的加权损失
- `log_vars/*`: Type2 的 log_vars 参数
- `train/total_loss`: 总损失
- `train/llm_loss`: 基础语言模型损失

## 注意事项

1. **Type2** 收敛较快，参数少，适合大多数场景
2. **Type3** 需要更多训练步数，但可以学习到样本级的自适应权重
3. 确保 SFT checkpoint 路径正确：`./checkpoints/qwen25-0_5b-sft`
4. Type2/Type3 的权重参数会自动使用更高的学习率以加快收敛
