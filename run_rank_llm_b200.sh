#!/bin/bash
# ============================================================
# ReasonRank Evaluation on B200 with wandb logging
# ============================================================
set -e

# B200 environment settings
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# wandb settings (override via environment variables)
WANDB_PROJECT=${WANDB_PROJECT:-"ReasonRank"}
WANDB_ENTITY=${WANDB_ENTITY:-""}  # set to your team/user

workspace_dir=$(grep "WORKSPACE_DIR" config.py | cut -d "'" -f 2)

# Detect number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l | tr -d ' ')
echo "Using ${NUM_GPUS} GPUs: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Common datasets
BRIGHT_DATASETS=('economics' 'earth_science' 'robotics' 'biology' 'psychology' 'stackoverflow' 'sustainable_living' 'leetcode' 'pony' 'aops' 'theoremqa_questions' 'theoremqa_theorems')
TREC_BEIR_DATASETS=('dl19' 'dl20' 'covid' 'dbpedia' 'scifact' 'nfcorpus' 'signal' 'robust04' 'news')

# wandb entity arg
WANDB_ENTITY_ARG=""
if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ENTITY_ARG="--wandb_entity ${WANDB_ENTITY}"
fi

################ evaluate reasonrank-7B using ReasonIR retrieval #################
window_size=20
model_name=liuwenhan/reasonrank-7B
DATASETS=("${BRIGHT_DATASETS[@]}")
python run_rank_llm.py \
    --model_path ${model_name} \
    --window_size $window_size \
    --step_size 10 \
    --retrieval_num 100 \
    --num_passes 1 \
    --reasoning_maxlen 3072 \
    --retrieval_method reasonir \
    --use_gpt4cot_retrieval True \
    --datasets ${DATASETS[@]} \
    --shuffle_candidates False \
    --prompt_mode rank_GPT_reasoning \
    --context_size 32768 \
    --vllm_batched True \
    --batch_size 512 \
    --output "${model_name}.txt" \
    --num_gpus ${NUM_GPUS} \
    --wandb_project ${WANDB_PROJECT} \
    ${WANDB_ENTITY_ARG} \
    --notes "B200 eval: reasonrank-7B + ReasonIR retrieval"

################ evaluate reasonrank-32B using ReasonIR retrieval #################
window_size=20
model_name=liuwenhan/reasonrank-32B
DATASETS=("${BRIGHT_DATASETS[@]}")
python run_rank_llm.py \
    --model_path ${model_name} \
    --window_size $window_size \
    --step_size 10 \
    --retrieval_num 100 \
    --num_passes 1 \
    --reasoning_maxlen 3072 \
    --retrieval_method reasonir \
    --use_gpt4cot_retrieval True \
    --datasets ${DATASETS[@]} \
    --shuffle_candidates False \
    --prompt_mode rank_GPT_reasoning \
    --context_size 32768 \
    --vllm_batched True \
    --batch_size 512 \
    --output "${model_name}.txt" \
    --num_gpus ${NUM_GPUS} \
    --wandb_project ${WANDB_PROJECT} \
    ${WANDB_ENTITY_ARG} \
    --notes "B200 eval: reasonrank-32B + ReasonIR retrieval"

################ evaluate reasonrank-8B (Qwen3-8B) using ReasonIR retrieval #################
window_size=20
model_name=liuwenhan/reasonrank-8B
DATASETS=("${BRIGHT_DATASETS[@]}")
python run_rank_llm.py \
    --model_path ${model_name} \
    --window_size $window_size \
    --step_size 10 \
    --retrieval_num 100 \
    --num_passes 1 \
    --reasoning_maxlen 3072 \
    --retrieval_method reasonir \
    --use_gpt4cot_retrieval True \
    --datasets ${DATASETS[@]} \
    --shuffle_candidates False \
    --prompt_mode rank_GPT_qwen3 \
    --context_size 32768 \
    --vllm_batched True \
    --batch_size 512 \
    --output "${model_name}.txt" \
    --num_gpus ${NUM_GPUS} \
    --wandb_project ${WANDB_PROJECT} \
    ${WANDB_ENTITY_ARG} \
    --notes "B200 eval: reasonrank-8B (Qwen3) + ReasonIR retrieval"

echo "All evaluations complete. Check wandb project: ${WANDB_PROJECT}"
