#!/bin/bash
# ============================================================
# Distill After SFT 训练入口脚本 - Type2 Weighting
#
# 使用 Type2 (HeteroscedasticUncertainty) 加权策略:
#   - 基于 Kendall et al. 2018 的异方差不确定性理论
#   - 学习任务级的不确定性参数（仅 num_tasks 个标量参数）
#   - 学习率设置为基础学习率的 1000 倍
#   - 适用于任务间权重差异稳定的场景
# ============================================================

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 调用主训练脚本，传递 type2 参数
bash 3_train_qwen2_distill_after_sft.sh type2

echo "Training with Type2 (Heteroscedastic Uncertainty) weighting completed!"
