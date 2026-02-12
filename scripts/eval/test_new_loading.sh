#!/bin/bash
# 快速测试新的模型加载方式

echo "================================================"
echo "测试新的模型加载方式"
echo "================================================"
echo ""

cd /home/jsun/llava-kd

# 测试1: 测试加载函数本身
echo "[测试 1] 测试本地 checkpoint 加载"
echo "------------------------------------------------"
python3.12 llavakd/model/eval_model_load.py \
    --test-local /home/jsun/llava-kd/checkpoints/qwen25-0_5b-distill-after-sft

if [ $? -eq 0 ]; then
    echo "✓ 本地加载测试通过"
else
    echo "✗ 本地加载测试失败"
    exit 1
fi

echo ""
echo "================================================"
echo "测试完成！"
echo "================================================"
echo ""
echo "下一步:"
echo "1. 评估本地模型: bash scripts/eval/gqa_v2.sh \\"
echo "     '/home/jsun/llava-kd/checkpoints/qwen25-0_5b-distill-after-sft' \\"
echo "     'Qwen25_0.5B_Local' 'local'"
echo ""
echo "2. 评估 HF 模型: bash scripts/eval/gqa_v2.sh \\"
echo "     'Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP' \\"
echo "     'TinyLLaVA_3B_HF' 'huggingface'"
echo ""
echo "3. 完整评估: 编辑 scripts/eval/eval_all_v3.sh 并运行"
