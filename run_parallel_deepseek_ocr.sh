#!/bin/bash
# ============================================================
# DeepSeek-OCR Zero-Shot Evaluation on BRIGHT
# Multimodal model (text-only mode for reranking)
# Model: deepseek-ai/DeepSeek-OCR (~3B params)
# TP=1 per instance, 8 instances on 8 GPUs
#
# Note: DeepSeek-OCR requires:
#   - trust_remote_code=True (handled in code)
#   - disable prefix caching (--disable_prefix_caching True)
#   - flash-attn >= 2.7.3
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export REASONRANK_WORKSPACE_DIR="${REASONRANK_WORKSPACE_DIR:-$(dirname ${SCRIPT_DIR})/workspace}"
export REASONRANK_PROJECT_DIR="${REASONRANK_PROJECT_DIR:-${SCRIPT_DIR}}"
export HF_HOME="${HF_HOME:-${REASONRANK_WORKSPACE_DIR}/hf_cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="${WANDB_PROJECT:-ReasonRank}"

MODEL="deepseek-ai/DeepSeek-OCR"
PROMPT_MODE="rank_GPT"
TP=1
TOTAL_GPUS="${NUM_GPUS:-8}"
NUM_INSTANCES=$((TOTAL_GPUS / TP))

ALL_DATASETS=(
    economics earth_science robotics biology
    psychology stackoverflow sustainable_living leetcode
    pony aops theoremqa_questions theoremqa_theorems
)
NUM_DATASETS=${#ALL_DATASETS[@]}

echo "============================================"
echo "  DeepSeek-OCR Zero-Shot Evaluation"
echo "============================================"
echo "Model:          ${MODEL}"
echo "Prompt mode:    ${PROMPT_MODE}"
echo "Total GPUs:     ${TOTAL_GPUS}"
echo "TP per instance:${TP}"
echo "Instances:      ${NUM_INSTANCES}"
echo "Datasets:       ${NUM_DATASETS}"
echo "============================================"

WANDB_ENTITY_ARG=""
if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ENTITY_ARG="--wandb_entity ${WANDB_ENTITY}"
fi

LOG_DIR="${SCRIPT_DIR}/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

PIDS=()
for ((i=0; i<NUM_INSTANCES; i++)); do
    GPU_IDS="${i}"

    INSTANCE_DATASETS=()
    for ((d=0; d<NUM_DATASETS; d++)); do
        if [ $((d % NUM_INSTANCES)) -eq $i ]; then
            INSTANCE_DATASETS+=("${ALL_DATASETS[$d]}")
        fi
    done

    if [ ${#INSTANCE_DATASETS[@]} -eq 0 ]; then
        continue
    fi

    DATASETS_STR="${INSTANCE_DATASETS[*]}"
    LOG_FILE="${LOG_DIR}/gpu${GPU_IDS}.log"

    echo "[Instance ${i}] GPU=${GPU_IDS} datasets=[${DATASETS_STR}] log=${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${GPU_IDS} python run_rank_llm.py \
        --model_path ${MODEL} \
        --window_size 20 \
        --step_size 10 \
        --retrieval_num 100 \
        --num_passes 1 \
        --reasoning_maxlen 3072 \
        --retrieval_method reasonir \
        --use_gpt4cot_retrieval True \
        --datasets ${DATASETS_STR} \
        --shuffle_candidates False \
        --prompt_mode ${PROMPT_MODE} \
        --context_size 32768 \
        --vllm_batched True \
        --batch_size 512 \
        --output "DeepSeek-OCR.txt" \
        --num_gpus ${TP} \
        --disable_prefix_caching True \
        --wandb_project ${WANDB_PROJECT} \
        ${WANDB_ENTITY_ARG} \
        --wandb_run_name "DeepSeek-OCR_gpu${GPU_IDS}" \
        --notes "B200: DeepSeek-OCR zero-shot GPU=${GPU_IDS} datasets=[${DATASETS_STR}]" \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} instances launched. Logs in: ${LOG_DIR}"
echo "Waiting for all instances to finish..."
echo "  Monitor: tail -f ${LOG_DIR}/*.log"

FAILED=0
for ((i=0; i<${#PIDS[@]}; i++)); do
    PID=${PIDS[$i]}
    if wait ${PID}; then
        echo "[Instance ${i}] PID=${PID} completed successfully"
    else
        echo "[Instance ${i}] PID=${PID} FAILED (exit code: $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ ${FAILED} -gt 0 ]; then
    echo "WARNING: ${FAILED} instance(s) failed. Check logs: ${LOG_DIR}"
fi

echo "============================================"
echo "  Collecting results into unified wandb run"
echo "============================================"

python wandb_summary.py \
    --results_dir results \
    --datasets ${ALL_DATASETS[@]} \
    --model_path ${MODEL} \
    --prompt_mode ${PROMPT_MODE} \
    --num_gpus ${TOTAL_GPUS} \
    --tp ${TP} \
    --wandb_project ${WANDB_PROJECT} \
    ${WANDB_ENTITY_ARG}

echo "============================================"
echo "  All done! Results in: results/"
echo "============================================"
