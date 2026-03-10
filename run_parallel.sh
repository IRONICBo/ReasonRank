#!/bin/bash
# ============================================================
# ReasonRank Data-Parallel Evaluation
# Distributes datasets across GPUs, 1 model instance per GPU
#
# Usage:
#   bash run_parallel.sh                          # 8 GPUs, reasonrank-7B
#   NUM_GPUS=4 bash run_parallel.sh               # 4 GPUs
#   MODEL=liuwenhan/reasonrank-32B TP=2 bash run_parallel.sh  # 32B with TP=2
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ===================== Config =====================
export REASONRANK_WORKSPACE_DIR="${REASONRANK_WORKSPACE_DIR:-$(dirname ${SCRIPT_DIR})/workspace}"
export REASONRANK_PROJECT_DIR="${REASONRANK_PROJECT_DIR:-${SCRIPT_DIR}}"
export HF_HOME="${HF_HOME:-${REASONRANK_WORKSPACE_DIR}/hf_cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="${WANDB_PROJECT:-ReasonRank}"

# Model config
MODEL="${MODEL:-liuwenhan/reasonrank-7B}"
PROMPT_MODE="${PROMPT_MODE:-rank_GPT_reasoning}"
TP="${TP:-1}"  # tensor_parallel per instance, 1 for 7B/8B, 2~4 for 32B

# Total available GPUs
TOTAL_GPUS="${NUM_GPUS:-8}"

# GPUs per model instance
GPUS_PER_INSTANCE=${TP}
NUM_INSTANCES=$((TOTAL_GPUS / GPUS_PER_INSTANCE))

# All BRIGHT datasets
ALL_DATASETS=(
    economics earth_science robotics biology
    psychology stackoverflow sustainable_living leetcode
    pony aops theoremqa_questions theoremqa_theorems
)
NUM_DATASETS=${#ALL_DATASETS[@]}

echo "============================================"
echo "  ReasonRank Data-Parallel Evaluation"
echo "============================================"
echo "Model:          ${MODEL}"
echo "Prompt mode:    ${PROMPT_MODE}"
echo "Total GPUs:     ${TOTAL_GPUS}"
echo "TP per instance:${TP}"
echo "Instances:      ${NUM_INSTANCES}"
echo "Datasets:       ${NUM_DATASETS}"
echo "============================================"

# wandb entity arg
WANDB_ENTITY_ARG=""
if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ENTITY_ARG="--wandb_entity ${WANDB_ENTITY}"
fi

# ===================== Distribute datasets to instances =====================
LOG_DIR="${SCRIPT_DIR}/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

PIDS=()
for ((i=0; i<NUM_INSTANCES; i++)); do
    # Calculate GPU IDs for this instance
    GPU_START=$((i * GPUS_PER_INSTANCE))
    GPU_IDS=""
    for ((g=0; g<GPUS_PER_INSTANCE; g++)); do
        if [ -n "${GPU_IDS}" ]; then GPU_IDS="${GPU_IDS},"; fi
        GPU_IDS="${GPU_IDS}$((GPU_START + g))"
    done

    # Distribute datasets round-robin
    INSTANCE_DATASETS=()
    for ((d=0; d<NUM_DATASETS; d++)); do
        if [ $((d % NUM_INSTANCES)) -eq $i ]; then
            INSTANCE_DATASETS+=("${ALL_DATASETS[$d]}")
        fi
    done

    # Skip if no datasets assigned
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
        --output "${MODEL}.txt" \
        --num_gpus ${TP} \
        --wandb_project ${WANDB_PROJECT} \
        ${WANDB_ENTITY_ARG} \
        --wandb_run_name "$(basename ${MODEL})_gpu${GPU_IDS}" \
        --notes "B200 parallel: GPU=${GPU_IDS} datasets=[${DATASETS_STR}]" \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} instances launched. Logs in: ${LOG_DIR}"
echo "Waiting for all instances to finish..."
echo "  Monitor: tail -f ${LOG_DIR}/*.log"

# ===================== Wait and collect results =====================
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
echo "============================================"
if [ ${FAILED} -eq 0 ]; then
    echo "  All instances completed successfully!"
else
    echo "  ${FAILED} instance(s) failed. Check logs: ${LOG_DIR}"
fi
echo "  Results saved in: results/"
echo "============================================"
