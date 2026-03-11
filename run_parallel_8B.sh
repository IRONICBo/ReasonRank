#!/bin/bash
# ============================================================
# ReasonRank-8B Evaluation (Qwen3-8B)
# TP=1 per instance, 8 instances on 8 GPUs
# Note: prompt_mode is rank_GPT_qwen3 for Qwen3 based model
# ============================================================

export MODEL="liuwenhan/reasonrank-8B"
export PROMPT_MODE="rank_GPT_qwen3"
export TP=1          # 8B fits on 1 GPU
export NUM_GPUS=8    # total GPUs -> 8 instances

bash "$(dirname $0)/run_parallel.sh"
