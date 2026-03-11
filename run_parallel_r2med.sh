#!/bin/bash
# ============================================================
# ReasonRank R2MED Benchmark Evaluation
# 8 R2MED medical datasets on 8 GPUs (data parallel)
#
# Usage:
#   MODEL=liuwenhan/reasonrank-7B bash run_parallel_r2med.sh
#   MODEL=liuwenhan/reasonrank-32B bash run_parallel_r2med.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export REASONRANK_WORKSPACE_DIR="${REASONRANK_WORKSPACE_DIR:-$(dirname ${SCRIPT_DIR})/workspace}"
export REASONRANK_PROJECT_DIR="${REASONRANK_PROJECT_DIR:-${SCRIPT_DIR}}"
export HF_HOME="${HF_HOME:-${REASONRANK_WORKSPACE_DIR}/hf_cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="${WANDB_PROJECT:-ReasonRank}"

MODEL="${MODEL:-liuwenhan/reasonrank-7B}"
PROMPT_MODE="${PROMPT_MODE:-rank_GPT_reasoning}"
TP="${TP:-1}"
TOTAL_GPUS="${NUM_GPUS:-8}"
GPUS_PER_INSTANCE=${TP}
NUM_INSTANCES=$((TOTAL_GPUS / GPUS_PER_INSTANCE))

ALL_DATASETS=(
    r2med_Medical-Sciences r2med_Biology r2med_MedQA-Diag r2med_PMC-Clinical
    r2med_Bioinformatics r2med_MedXpertQA-Exam r2med_PMC-Treatment r2med_IIYi-Clinical
)
NUM_DATASETS=${#ALL_DATASETS[@]}

echo "============================================"
echo "  ReasonRank R2MED Benchmark Evaluation"
echo "============================================"
echo "Model:          ${MODEL}"
echo "Prompt mode:    ${PROMPT_MODE}"
echo "Total GPUs:     ${TOTAL_GPUS}"
echo "TP per instance:${TP}"
echo "Instances:      ${NUM_INSTANCES}"
echo "Datasets:       ${NUM_DATASETS} (R2MED)"
echo "============================================"

WANDB_ENTITY_ARG=""
if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ENTITY_ARG="--wandb_entity ${WANDB_ENTITY}"
fi

LOG_DIR="${SCRIPT_DIR}/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

PIDS=()
for ((i=0; i<NUM_INSTANCES; i++)); do
    GPU_START=$((i * GPUS_PER_INSTANCE))
    GPU_IDS=""
    for ((g=0; g<GPUS_PER_INSTANCE; g++)); do
        if [ -n "${GPU_IDS}" ]; then GPU_IDS="${GPU_IDS},"; fi
        GPU_IDS="${GPU_IDS}$((GPU_START + g))"
    done

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
        --retrieval_method e5-mistral-7b-instruct \
        --datasets ${DATASETS_STR} \
        --shuffle_candidates False \
        --prompt_mode ${PROMPT_MODE} \
        --context_size 32768 \
        --vllm_batched True \
        --batch_size 512 \
        --output "$(basename ${MODEL}).txt" \
        --num_gpus ${TP} \
        --wandb_project ${WANDB_PROJECT} \
        ${WANDB_ENTITY_ARG} \
        --wandb_run_name "$(basename ${MODEL})_r2med_gpu${GPU_IDS}" \
        --notes "B200 R2MED: GPU=${GPU_IDS} datasets=[${DATASETS_STR}]" \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} instances launched. Logs in: ${LOG_DIR}"
echo "Waiting for all instances to finish..."

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
echo "  Collecting R2MED results into wandb"
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
echo "  R2MED evaluation done! Results in: results/"
echo "============================================"
