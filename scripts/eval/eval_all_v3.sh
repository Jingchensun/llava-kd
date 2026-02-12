#!/bin/bash
# 使用新的加载方式评估所有数据集
# 支持: HuggingFace 模型 和 本地完整 checkpoint

# ==================== 配置区 ====================

# 选择评估模式（二选一）
MODE="huggingface"  # 或 "huggingface"

if [ "$MODE" == "huggingface" ]; then
    # 方式1: 从 HuggingFace 加载
    MODEL_PATH="Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP"
    MODEL_NAME="TinyLLaVA_Qwen25_3B_HF"
    LOAD_SOURCE="huggingface"
elif [ "$MODE" == "local" ]; then
    # 方式2: 从本地完整 checkpoint 加载
    MODEL_PATH="/home/jsun/llava-kd/checkpoints/qwen25-0_5b-distill-after-sft"
    MODEL_NAME="Qwen25_0.5B_Local"
    LOAD_SOURCE="local"
fi

# GPU 配置
GPU_IDS="0,1,2,3"

# ================================================

cd /home/jsun/llava-kd

echo "================================================"
echo "评估配置"
echo "  模式: $MODE"
echo "  模型路径: $MODEL_PATH"
echo "  模型名称: $MODEL_NAME"
echo "  加载来源: $LOAD_SOURCE"
echo "  GPU 设备: $GPU_IDS"
echo "================================================"
echo ""
echo "开始串行评估 - $(date)"
echo ""

# 定义评估函数
run_eval() {
    local dataset=$1
    local script=$2
    
    echo "=== 运行 $dataset 评估 ==="
    echo "  开始时间: $(date)"
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS bash $script "$MODEL_PATH" "$MODEL_NAME" "$LOAD_SOURCE"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ $dataset 评估完成"
    else
        echo "  ✗ $dataset 评估失败 (exit code: $exit_code)"
    fi
    echo "  结束时间: $(date)"
    echo ""
}

# 串行执行每个评估任务，避免OOM
run_eval "GQA" "scripts/eval/gqa_v2.sh"

# 其他数据集可以使用类似的 v2 版本脚本，或者创建通用的评估脚本
# 这里先展示如何使用新的加载方式

echo "================================================"
echo "所有评估完成 - $(date)"
echo "================================================"
echo ""
echo "结果保存在: eval/results/${MODEL_NAME}_eval.txt"

# 显示评估结果
if [ -f "eval/results/${MODEL_NAME}_eval.txt" ]; then
    echo ""
    echo "评估结果摘要:"
    echo "------------------------------------------------"
    cat "eval/results/${MODEL_NAME}_eval.txt"
    echo "------------------------------------------------"
fi
