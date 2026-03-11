#!/bin/bash
# ============================================================
# ReasonRank-7B Evaluation (Qwen2.5-7B-Instruct)
# TP=1 per instance, 8 instances on 8 GPUs
# ============================================================

export MODEL="liuwenhan/reasonrank-7B"
export PROMPT_MODE="rank_GPT_reasoning"
export TP=1          # 7B fits on 1 GPU
export NUM_GPUS=8    # total GPUs -> 8 instances

bash "$(dirname $0)/run_parallel.sh"
