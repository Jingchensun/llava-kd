#!/bin/bash
# ============================================================
# Distill After SFT 训练入口脚本 - Type3 Weighting
#
# 使用 Type3 (InstanceConditionalWeighting) 加权策略:
#   - 基于实例特征的条件权重学习
#   - 使用 MLP 网络根据 teacher 特征动态预测权重
#   - 不同样本可以有不同的任务权重分配
#   - 学习率设置为基础学习率的 10 倍
#   - 适用于样本难度差异大、需要自适应权重的场景
# ============================================================

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 调用主训练脚本，传递 type3 参数
bash 3_train_qwen2_distill_after_sft.sh type3

echo "Training with Type3 (Instance-Conditional) weighting completed!"
