#!/bin/bash
# ============================================================
# ReasonRank Full R2MED Evaluation Pipeline
# Runs ALL models on R2MED benchmark (8 medical datasets)
#
# Models:
#   1. ReasonRank-7B   (reasonrank fine-tuned, rank_GPT_reasoning)
#   2. ReasonRank-8B   (reasonrank fine-tuned, rank_GPT_qwen3)
#   3. ReasonRank-32B  (reasonrank fine-tuned, rank_GPT_reasoning)
#   4. Qwen2.5-7B-Instruct   (base, zero-shot rank_GPT)
#   5. Qwen3-8B              (base, zero-shot rank_GPT)
#   6. Qwen2.5-32B-Instruct  (base, zero-shot rank_GPT)
#   7. DeepSeek-OCR           (VLM, zero-shot rank_GPT)
#
# Usage:
#   bash run_all_r2med.sh                    # run all models
#   bash run_all_r2med.sh --skip-to 4       # skip to model 4
#   bash run_all_r2med.sh --only 1 3 7      # only run models 1, 3, 7
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Parse arguments
SKIP_TO=0
ONLY_MODELS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-to)
            SKIP_TO=$2
            shift 2
            ;;
        --only)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ONLY_MODELS+=($1)
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_all_r2med.sh [--skip-to N] [--only N1 N2 ...]"
            exit 1
            ;;
    esac
done

should_run() {
    local model_num=$1
    if [ ${#ONLY_MODELS[@]} -gt 0 ]; then
        for m in "${ONLY_MODELS[@]}"; do
            if [ "$m" -eq "$model_num" ]; then
                return 0
            fi
        done
        return 1
    fi
    if [ "$model_num" -ge "$SKIP_TO" ]; then
        return 0
    fi
    return 1
}

TOTAL_START=$(date +%s)

echo "============================================================"
echo "  ReasonRank Full R2MED Evaluation Pipeline"
echo "  Started at: $(date)"
echo "============================================================"
echo ""

# ======================== 1. ReasonRank-7B ========================
if should_run 1; then
    echo "========================================"
    echo "  [1/7] ReasonRank-7B on R2MED"
    echo "  $(date)"
    echo "========================================"
    MODEL=liuwenhan/reasonrank-7B PROMPT_MODE=rank_GPT_reasoning \
        bash "${SCRIPT_DIR}/run_parallel_r2med.sh"
    echo ""
fi

# ======================== 2. ReasonRank-8B ========================
if should_run 2; then
    echo "========================================"
    echo "  [2/7] ReasonRank-8B on R2MED"
    echo "  $(date)"
    echo "========================================"
    MODEL=liuwenhan/reasonrank-8B PROMPT_MODE=rank_GPT_qwen3 \
        bash "${SCRIPT_DIR}/run_parallel_r2med.sh"
    echo ""
fi

# ======================== 3. ReasonRank-32B ========================
if should_run 3; then
    echo "========================================"
    echo "  [3/7] ReasonRank-32B on R2MED"
    echo "  $(date)"
    echo "========================================"
    MODEL=liuwenhan/reasonrank-32B PROMPT_MODE=rank_GPT_reasoning \
        bash "${SCRIPT_DIR}/run_parallel_r2med.sh"
    echo ""
fi

# ======================== 4. Qwen2.5-7B-Instruct (base) ========================
if should_run 4; then
    echo "========================================"
    echo "  [4/7] Qwen2.5-7B-Instruct zero-shot on R2MED"
    echo "  $(date)"
    echo "========================================"
    MODEL=Qwen/Qwen2.5-7B-Instruct PROMPT_MODE=rank_GPT \
        bash "${SCRIPT_DIR}/run_parallel_r2med.sh"
    echo ""
fi

# ======================== 5. Qwen3-8B (base) ========================
if should_run 5; then
    echo "========================================"
    echo "  [5/7] Qwen3-8B zero-shot on R2MED"
    echo "  $(date)"
    echo "========================================"
    MODEL=Qwen/Qwen3-8B PROMPT_MODE=rank_GPT \
        bash "${SCRIPT_DIR}/run_parallel_r2med.sh"
    echo ""
fi

# ======================== 6. Qwen2.5-32B-Instruct (base) ========================
if should_run 6; then
    echo "========================================"
    echo "  [6/7] Qwen2.5-32B-Instruct zero-shot on R2MED"
    echo "  $(date)"
    echo "========================================"
    MODEL=Qwen/Qwen2.5-32B-Instruct PROMPT_MODE=rank_GPT \
        bash "${SCRIPT_DIR}/run_parallel_r2med.sh"
    echo ""
fi

# ======================== 7. DeepSeek-OCR ========================
if should_run 7; then
    echo "========================================"
    echo "  [7/7] DeepSeek-OCR zero-shot on R2MED"
    echo "  $(date)"
    echo "========================================"
    MODEL=deepseek-ai/DeepSeek-OCR PROMPT_MODE=rank_GPT \
        bash "${SCRIPT_DIR}/run_parallel_r2med.sh"
    echo ""
fi

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
HOURS=$((TOTAL_ELAPSED / 3600))
MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))

echo "============================================================"
echo "  R2MED evaluations complete!"
echo "  Total time: ${HOURS}h ${MINUTES}m"
echo "  Finished at: $(date)"
echo "  Results in: results/r2med_*.json"
echo "  Details in: runs/r2med_*/*_details.jsonl"
echo "============================================================"
