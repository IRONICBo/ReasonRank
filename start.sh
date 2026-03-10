#!/bin/bash
# ============================================================
# ReasonRank Evaluation Entrypoint
#
# Usage:
#   bash start.sh                          # run all models
#   bash start.sh reasonrank-7B            # run specific model
#   bash start.sh reasonrank-8B --dry-run  # print commands only
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===================== Config =====================
# WORKSPACE_DIR: root directory for data/models cache
# Defaults to a 'workspace' folder alongside this project
export REASONRANK_WORKSPACE_DIR="${REASONRANK_WORKSPACE_DIR:-$(dirname ${SCRIPT_DIR})/workspace}"
export REASONRANK_PROJECT_DIR="${REASONRANK_PROJECT_DIR:-${SCRIPT_DIR}}"

# HuggingFace cache (models download here)
export HF_HOME="${HF_HOME:-${REASONRANK_WORKSPACE_DIR}/hf_cache}"

# GPU settings
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# wandb settings
export WANDB_PROJECT="${WANDB_PROJECT:-ReasonRank}"
# export WANDB_ENTITY="your_team"  # uncomment and set

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l | tr -d ' ')

# ===================== Parse Args =====================
MODEL_FILTER="${1:-all}"
DRY_RUN=false
if [[ "$*" == *"--dry-run"* ]]; then
    DRY_RUN=true
fi

# ===================== Derived paths =====================
echo "============================================"
echo "  ReasonRank Evaluation"
echo "============================================"
echo "PROJECT_DIR:   ${REASONRANK_PROJECT_DIR}"
echo "WORKSPACE_DIR: ${REASONRANK_WORKSPACE_DIR}"
echo "HF_HOME:       ${HF_HOME}"
echo "GPUs:          ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} total)"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "Model filter:  ${MODEL_FILTER}"
echo "============================================"

# Create workspace dirs
mkdir -p "${REASONRANK_WORKSPACE_DIR}/data/bright"
mkdir -p "${REASONRANK_WORKSPACE_DIR}/data/r2med"
mkdir -p "${HF_HOME}"

# ===================== Download Models =====================
download_model() {
    local model_id=$1
    echo "[Download] Checking model: ${model_id} ..."
    # huggingface-cli download caches to HF_HOME, skips if already present
    python -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download('${model_id}', cache_dir=os.environ.get('HF_HOME'))
print(f'  Model ready at: {path}')
"
}

MODELS_7B="liuwenhan/reasonrank-7B"
MODELS_32B="liuwenhan/reasonrank-32B"
MODELS_8B="liuwenhan/reasonrank-8B"

if [ "${MODEL_FILTER}" = "all" ] || [ "${MODEL_FILTER}" = "reasonrank-7B" ]; then
    download_model "${MODELS_7B}"
fi
if [ "${MODEL_FILTER}" = "all" ] || [ "${MODEL_FILTER}" = "reasonrank-32B" ]; then
    download_model "${MODELS_32B}"
fi
if [ "${MODEL_FILTER}" = "all" ] || [ "${MODEL_FILTER}" = "reasonrank-8B" ]; then
    download_model "${MODELS_8B}"
fi

# ===================== Common Settings =====================
BRIGHT_DATASETS='economics earth_science robotics biology psychology stackoverflow sustainable_living leetcode pony aops theoremqa_questions theoremqa_theorems'

WANDB_ENTITY_ARG=""
if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ENTITY_ARG="--wandb_entity ${WANDB_ENTITY}"
fi

run_eval() {
    local model_name=$1
    local prompt_mode=$2
    local notes=$3

    echo ""
    echo ">>> Evaluating: ${model_name} (prompt_mode=${prompt_mode})"

    CMD="python run_rank_llm.py \
        --model_path ${model_name} \
        --window_size 20 \
        --step_size 10 \
        --retrieval_num 100 \
        --num_passes 1 \
        --reasoning_maxlen 3072 \
        --retrieval_method reasonir \
        --use_gpt4cot_retrieval True \
        --datasets ${BRIGHT_DATASETS} \
        --shuffle_candidates False \
        --prompt_mode ${prompt_mode} \
        --context_size 32768 \
        --vllm_batched True \
        --batch_size 512 \
        --output ${model_name}.txt \
        --num_gpus ${NUM_GPUS} \
        --wandb_project ${WANDB_PROJECT} \
        ${WANDB_ENTITY_ARG} \
        --notes \"${notes}\""

    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY RUN] ${CMD}"
    else
        eval ${CMD}
    fi
}

# ===================== Run Evaluations =====================
if [ "${MODEL_FILTER}" = "all" ] || [ "${MODEL_FILTER}" = "reasonrank-7B" ]; then
    run_eval "${MODELS_7B}" "rank_GPT_reasoning" "B200 eval: reasonrank-7B + ReasonIR"
fi

if [ "${MODEL_FILTER}" = "all" ] || [ "${MODEL_FILTER}" = "reasonrank-32B" ]; then
    run_eval "${MODELS_32B}" "rank_GPT_reasoning" "B200 eval: reasonrank-32B + ReasonIR"
fi

if [ "${MODEL_FILTER}" = "all" ] || [ "${MODEL_FILTER}" = "reasonrank-8B" ]; then
    run_eval "${MODELS_8B}" "rank_GPT_qwen3" "B200 eval: reasonrank-8B (Qwen3) + ReasonIR"
fi

echo ""
echo "All evaluations complete. Check wandb: https://wandb.ai/${WANDB_ENTITY:-your_entity}/${WANDB_PROJECT}"
