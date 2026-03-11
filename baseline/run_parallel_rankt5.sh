#!/bin/bash
# RankT5/monoT5 Pointwise Reranker - Single GPU (T5 encoder-decoder)
# Model: castorini/monot5-3b-msmarco
# Note: T5 cannot use vLLM, runs on single GPU with Transformers
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

export REASONRANK_WORKSPACE_DIR="${REASONRANK_WORKSPACE_DIR:-$(dirname ${PROJECT_DIR})/workspace}"
export REASONRANK_PROJECT_DIR="${REASONRANK_PROJECT_DIR:-${PROJECT_DIR}}"
export HF_HOME="${HF_HOME:-${REASONRANK_WORKSPACE_DIR}/hf_cache}"

MODEL="${MODEL:-castorini/monot5-3b-msmarco}"
RETRIEVAL="${RETRIEVAL:-reasonir}"
BENCHMARK="${BENCHMARK:-bright}"

if [ "${BENCHMARK}" == "bright" ]; then
    ALL_DATASETS=(economics earth_science robotics biology psychology stackoverflow sustainable_living leetcode pony aops theoremqa_questions theoremqa_theorems)
elif [ "${BENCHMARK}" == "r2med" ]; then
    RETRIEVAL="e5-mistral-7b-instruct"
    ALL_DATASETS=(r2med_Medical-Sciences r2med_Biology r2med_MedQA-Diag r2med_PMC-Clinical r2med_Bioinformatics r2med_MedXpertQA-Exam r2med_PMC-Treatment r2med_IIYi-Clinical)
fi

MODEL_SHORT=$(basename ${MODEL})

echo "============================================"
echo "  monoT5 Pointwise: ${MODEL_SHORT} on ${BENCHMARK}"
echo "  Datasets: ${#ALL_DATASETS[@]}"
echo "  Note: T5 runs on single GPU (encoder-decoder)"
echo "============================================"

# T5 is encoder-decoder, can only run on 1 GPU per process
# Run sequentially or split manually
CUDA_VISIBLE_DEVICES=0 python baseline/run_rankt5.py \
    --model_path ${MODEL} \
    --datasets ${ALL_DATASETS[@]} \
    --retrieval_method ${RETRIEVAL} \
    --batch_size 32 \
    --wandb_project ReasonRank

echo "Done! Results in: results/"
