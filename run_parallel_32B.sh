#!/bin/bash
# ============================================================
# ReasonRank-32B Evaluation (Qwen2.5-32B-Instruct)
# TP=4 per instance, 2 instances on 8 GPUs
# ============================================================

export MODEL="liuwenhan/reasonrank-32B"
export PROMPT_MODE="rank_GPT_reasoning"
export TP=4          # 32B needs 4 GPUs per instance
export NUM_GPUS=8    # total GPUs -> 2 instances

bash "$(dirname $0)/run_parallel.sh"
