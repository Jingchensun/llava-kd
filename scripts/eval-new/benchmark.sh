#!/usr/bin/env bash

WORK_DIR=$(cd "$(dirname "$0")/..";pwd)
export PYTHONPATH=${WORK_DIR}
CHCEKPOINT_PATH="/home/jsun/llava-kd/pretrained_checkpoints/LLaVA_KD_ckpts/tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune"
MODEL_NAME="LLaVA_KD_Qwen25_3B"
OUTPUT_DIR_EVAL=${WORK_DIR}/eval/results/${MODEL_NAME}
mkdir -p ${OUTPUT_DIR_EVAL}
CONV_MODE=v1
cd ${WORK_DIR}

DATASET_NAME=mme
MODEL_GENERATOR=mobilevlm.eval.model_vqa_loader
DATA_ROOT=${WORK_DIR}/dataset/benchmark_data/mme
SPLIT_NAME=llava_mme
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=gqa
MODEL_GENERATOR=mobilevlm.eval.model_vqa_loader
DATA_ROOT=${WORK_DIR}/dataset/benchmark_data/gqa
SPLIT_NAME=llava_gqa_testdev_balanced
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=textvqa
MODEL_GENERATOR=mobilevlm.eval.model_vqa_loader
DATA_ROOT=${WORK_DIR}/dataset/benchmark_data/textvqa
SPLIT_NAME=llava_textvqa_val_v051_ocr
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=pope
MODEL_GENERATOR=mobilevlm.eval.model_vqa_loader
DATA_ROOT=${WORK_DIR}/dataset/benchmark_data/pope
SPLIT_NAME=llava_pope_test
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=mmbench
MODEL_GENERATOR=mobilevlm.eval.model_vqa_mmbench
DATA_ROOT=${WORK_DIR}/dataset/benchmark_data/mmbench
SPLIT_NAME=mmbench_dev_en_20231003
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=sqa
MODEL_GENERATOR=mobilevlm.eval.model_vqa_science
DATA_ROOT=${WORK_DIR}/dataset/benchmark_data/sqa
SPLIT_NAME=llava_test_CQM-A
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHCEKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}
