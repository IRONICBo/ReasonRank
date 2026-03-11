#!/bin/bash
# ============================================================
# Rank-K (32B) Baseline - Listwise Reasoning Reranker
# Model: hltcoe/Rank-K-32B (QwQ-32B based)
# Uses rank_GPT_rankk prompt mode (with reasoning)
# TP=1 on B200 (192GB), 8 instances on 8 GPUs
# ============================================================

export MODEL="hltcoe/Rank-K-32B"
export PROMPT_MODE="rank_GPT_rankk"
export TP=1
export NUM_GPUS=8

bash "$(dirname $0)/run_parallel.sh"
