#!/bin/bash
# Type2 & Type3 Weighting 学习率扫描: base(2e-5), 10x, 100x, 1000x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for type in type2 type3; do
    for lr_name in "lr-0.1x:2e-6"; do
        name="${lr_name%%:*}"
        lr="${lr_name##*:}"

        echo "=== 实验: ${type}-${name} (lr=$lr) ==="

        # 训练
        cd "$SCRIPT_DIR"
        bash 3_train_qwen2_distill_after_sft.sh "$type" "$lr" "$name"

        # 评估
        cd /home/jsun/llava-kd
        MODEL_PATH="./checkpoints/qwen25-0_5b-distill-after-sft-${type}-${name}"
        MODEL_NAME="Qwen25_0.5B_Local-last-${type}-${name}"
        #CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/sqa.sh "$MODEL_PATH" "$MODEL_NAME" || echo "评估失败: ${type}-${name}"
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/eval_all_v2.sh "$MODEL_PATH" "$MODEL_NAME"
    done
done

echo "=== 所有实验完成 ==="
