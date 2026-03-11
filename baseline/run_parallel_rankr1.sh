#!/bin/bash
# Rank-R1 Setwise Reranker - Data Parallel on 8 GPUs
# Model: ielabgroup/Rank-R1-7B-v0.1 or ielabgroup/Rank-R1-14B-v0.1
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

export REASONRANK_WORKSPACE_DIR="${REASONRANK_WORKSPACE_DIR:-$(dirname ${PROJECT_DIR})/workspace}"
export REASONRANK_PROJECT_DIR="${REASONRANK_PROJECT_DIR:-${PROJECT_DIR}}"
export HF_HOME="${HF_HOME:-${REASONRANK_WORKSPACE_DIR}/hf_cache}"

MODEL="${MODEL:-ielabgroup/Rank-R1-7B-v0.1}"
RETRIEVAL="${RETRIEVAL:-reasonir}"
TOTAL_GPUS="${NUM_GPUS:-8}"
BENCHMARK="${BENCHMARK:-bright}"

if [ "${BENCHMARK}" == "bright" ]; then
    ALL_DATASETS=(economics earth_science robotics biology psychology stackoverflow sustainable_living leetcode pony aops theoremqa_questions theoremqa_theorems)
elif [ "${BENCHMARK}" == "r2med" ]; then
    RETRIEVAL="e5-mistral-7b-instruct"
    ALL_DATASETS=(r2med_Medical-Sciences r2med_Biology r2med_MedQA-Diag r2med_PMC-Clinical r2med_Bioinformatics r2med_MedXpertQA-Exam r2med_PMC-Treatment r2med_IIYi-Clinical)
fi

NUM_DATASETS=${#ALL_DATASETS[@]}
MODEL_SHORT=$(basename ${MODEL})

echo "============================================"
echo "  Rank-R1 Setwise: ${MODEL_SHORT} on ${BENCHMARK}"
echo "  GPUs: ${TOTAL_GPUS}, Datasets: ${NUM_DATASETS}"
echo "============================================"

LOG_DIR="${PROJECT_DIR}/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

PIDS=()
for ((i=0; i<TOTAL_GPUS; i++)); do
    INSTANCE_DATASETS=()
    for ((d=0; d<NUM_DATASETS; d++)); do
        if [ $((d % TOTAL_GPUS)) -eq $i ]; then
            INSTANCE_DATASETS+=("${ALL_DATASETS[$d]}")
        fi
    done
    [ ${#INSTANCE_DATASETS[@]} -eq 0 ] && continue
    DATASETS_STR="${INSTANCE_DATASETS[*]}"
    LOG_FILE="${LOG_DIR}/gpu${i}.log"
    echo "[GPU ${i}] datasets=[${DATASETS_STR}]"

    CUDA_VISIBLE_DEVICES=${i} python baseline/run_rankr1.py \
        --model_path ${MODEL} \
        --datasets ${DATASETS_STR} \
        --retrieval_method ${RETRIEVAL} \
        --set_size 20 \
        --num_gpus 1 \
        --wandb_project ReasonRank \
        --wandb_run_name "${MODEL_SHORT}_gpu${i}" \
        > "${LOG_FILE}" 2>&1 &
    PIDS+=($!)
done

echo "All ${#PIDS[@]} instances launched. Logs: ${LOG_DIR}"
FAILED=0
for ((i=0; i<${#PIDS[@]}; i++)); do
    if ! wait ${PIDS[$i]}; then FAILED=$((FAILED+1)); fi
done
[ ${FAILED} -gt 0 ] && echo "WARNING: ${FAILED} failed"
echo "Done! Results in: results/"
