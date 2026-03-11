#!/bin/bash
# ============================================================
# Qwen2.5-7B-Instruct Zero-Shot Evaluation on BRIGHT
# Base model without ReasonRank fine-tuning
# TP=1 per instance, 8 instances on 8 GPUs
# ============================================================

export MODEL="Qwen/Qwen2.5-7B-Instruct"
export PROMPT_MODE="rank_GPT"
export TP=1
export NUM_GPUS=8

bash "$(dirname $0)/run_parallel.sh"
