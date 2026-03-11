#!/bin/bash
# ============================================================
# Qwen2.5-32B-Instruct Zero-Shot Evaluation on BRIGHT
# Base model without ReasonRank fine-tuning
# TP=1 per instance (B200 192GB fits 32B on single GPU)
# 8 instances on 8 GPUs
# ============================================================

export MODEL="Qwen/Qwen2.5-32B-Instruct"
export PROMPT_MODE="rank_GPT"
export TP=1
export NUM_GPUS=8

bash "$(dirname $0)/run_parallel.sh"
