#!/bin/bash
# ============================================================
# Type2 快速测试脚本
# 
# 运行前请确保：
# 1. conda activate llava-kd
# 2. SFT checkpoint 存在于 ./checkpoints/qwen25-0_5b-sft
# ============================================================

echo "=================================================="
echo "🚀 启动 Type2 (异方差不确定性) 训练测试"
echo "=================================================="
echo ""
echo "📋 配置信息："
echo "  - 加权策略: Type2 (HeteroscedasticUncertainty)"
echo "  - 参数量: 仅 4 个标量参数"
echo "  - 学习率: 基础 LR × 1000"
echo "  - 输出目录: ./checkpoints/qwen25-0_5b-distill-after-sft-type2"
echo ""
echo "=================================================="
echo ""

# 检查环境
if [[ "$CONDA_DEFAULT_ENV" != "llava-kd" ]]; then
    echo "⚠️  警告: 当前不在 llava-kd 环境中"
    echo "请运行: conda activate llava-kd"
    exit 1
fi

# 检查 SFT checkpoint
SFT_CKPT="./checkpoints/qwen25-0_5b-sft"
if [ ! -d "$SFT_CKPT" ]; then
    echo "❌ 错误: SFT checkpoint 不存在: $SFT_CKPT"
    echo "请先运行 SFT 训练或检查路径"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""
echo "开始训练..."
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/../../.."

# 运行训练
bash scripts/train/llava_kd_qwen2/3_train_qwen2_distill_after_sft.sh type2

echo ""
echo "=================================================="
echo "✅ Type2 训练完成！"
echo "=================================================="
