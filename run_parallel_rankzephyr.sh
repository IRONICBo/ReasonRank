#!/bin/bash
# ============================================================
# RankZephyr (7B) Baseline - Listwise Reranker (Non-reasoning)
# Model: castorini/rank_zephyr_7b_v1_full (Mistral-7B based)
# Uses rank_GPT prompt mode (standard listwise, no CoT)
# TP=1 per instance, 8 instances on 8 GPUs
# ============================================================

export MODEL="castorini/rank_zephyr_7b_v1_full"
export PROMPT_MODE="rank_GPT"
export TP=1
export NUM_GPUS=8

bash "$(dirname $0)/run_parallel.sh"
