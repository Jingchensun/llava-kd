# 🚀 快速开始 - Type2 Weighting

## 一键运行 Type2

```bash
# 1. 激活环境
conda activate llava-kd

# 2. 进入项目目录
cd /home/jsun/llava-kd

# 3. 运行训练（推荐使用测试脚本，包含环境检查）
bash scripts/train/llava_kd_qwen2/RUN_TYPE2_TEST.sh
```

## 或者使用便捷脚本

```bash
# Type2 (异方差不确定性 - 推荐)
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft_type2.sh

# Type3 (实例条件权重 - 高级)
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft_type3.sh

# Type1 (等权重 - 基线)
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft.sh
```

## 查看训练日志

```bash
# 实时查看日志
tail -f ./checkpoints/qwen25-0_5b-distill-after-sft-type2/train_*.log

# 查看权重变化
grep "Weights -" ./checkpoints/qwen25-0_5b-distill-after-sft-type2/train_*.log | tail -20

# 查看 log_vars 变化 (Type2)
grep "Log_vars -" ./checkpoints/qwen25-0_5b-distill-after-sft-type2/train_*.log | tail -20
```

## 前置条件

✅ 确保 SFT checkpoint 存在：`./checkpoints/qwen25-0_5b-sft`

## 预期输出

训练开始后，您会看到：
```
🔧 Distillation Weighting Strategy: type2
  Feature dim: 1536
  Num tasks: 4
  
[Step 100] Weights - Main Loss: 0.52, Logits Distill Loss: 0.34, ...
[Step 100] Log_vars - [0.65, -0.34, 2.12, 3.46]
```

## 详细文档

- 完整指南: `./WEIGHTING_GUIDE.md`
- Weighting 说明: `./scripts/train/llava_kd_qwen2/README_WEIGHTING.md`

---

**现在就开始训练吧！** 🎉
