#!/bin/bash
# ============================================================
# Run ALL Baseline Rerankers on BRIGHT Benchmark
#
# Models:
#   1. RankT5 (3B)        - Pointwise, monoT5, encoder-decoder
#   2. RankZephyr (7B)    - Listwise, non-reasoning
#   3. Rank-R1 (7B)       - Setwise, reasoning
#   4. Rank-R1 (14B)      - Setwise, reasoning
#   5. Rank1 (7B)         - Pointwise, reasoning
#   6. Rank1 (32B)        - Pointwise, reasoning
#   7. Rank-K (32B)       - Listwise, reasoning
#
# Usage:
#   bash run_all_baselines.sh                  # all baselines on BRIGHT
#   bash run_all_baselines.sh --skip-to 3     # skip to model 3
#   bash run_all_baselines.sh --only 2 7      # only RankZephyr + Rank-K
#   BENCHMARK=r2med bash run_all_baselines.sh  # on R2MED
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export BENCHMARK="${BENCHMARK:-bright}"

SKIP_TO=0
ONLY_MODELS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-to) SKIP_TO=$2; shift 2 ;;
        --only)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ONLY_MODELS+=($1); shift
            done ;;
        *) echo "Usage: bash run_all_baselines.sh [--skip-to N] [--only N1 N2 ...]"; exit 1 ;;
    esac
done

should_run() {
    local n=$1
    if [ ${#ONLY_MODELS[@]} -gt 0 ]; then
        for m in "${ONLY_MODELS[@]}"; do [ "$m" -eq "$n" ] && return 0; done
        return 1
    fi
    [ "$n" -ge "$SKIP_TO" ]
}

TOTAL_START=$(date +%s)
echo "============================================================"
echo "  Baseline Rerankers Evaluation (${BENCHMARK})"
echo "  Started at: $(date)"
echo "============================================================"

# ---- 1. RankT5 (3B) ----
if should_run 1; then
    echo -e "\n[1/7] RankT5 (3B) - monoT5 pointwise  $(date)"
    bash baseline/run_parallel_rankt5.sh
fi

# ---- 2. RankZephyr (7B) ----
if should_run 2; then
    echo -e "\n[2/7] RankZephyr (7B) - listwise  $(date)"
    if [ "${BENCHMARK}" == "r2med" ]; then
        MODEL=castorini/rank_zephyr_7b_v1_full PROMPT_MODE=rank_GPT \
            bash run_parallel_r2med.sh
    else
        bash run_parallel_rankzephyr.sh
    fi
fi

# ---- 3. Rank-R1 (7B) ----
if should_run 3; then
    echo -e "\n[3/7] Rank-R1 (7B) - setwise reasoning  $(date)"
    MODEL=ielabgroup/Rank-R1-7B-v0.1 BENCHMARK=${BENCHMARK} \
        bash baseline/run_parallel_rankr1.sh
fi

# ---- 4. Rank-R1 (14B) ----
if should_run 4; then
    echo -e "\n[4/7] Rank-R1 (14B) - setwise reasoning  $(date)"
    MODEL=ielabgroup/Rank-R1-14B-v0.1 BENCHMARK=${BENCHMARK} \
        bash baseline/run_parallel_rankr1.sh
fi

# ---- 5. Rank1 (7B) ----
if should_run 5; then
    echo -e "\n[5/7] Rank1 (7B) - pointwise reasoning  $(date)"
    MODEL=jhu-clsp/rank1-7b BENCHMARK=${BENCHMARK} \
        bash baseline/run_parallel_rank1.sh
fi

# ---- 6. Rank1 (32B) ----
if should_run 6; then
    echo -e "\n[6/7] Rank1 (32B) - pointwise reasoning  $(date)"
    MODEL=jhu-clsp/rank1-32b BENCHMARK=${BENCHMARK} \
        bash baseline/run_parallel_rank1.sh
fi

# ---- 7. Rank-K (32B) ----
if should_run 7; then
    echo -e "\n[7/7] Rank-K (32B) - listwise reasoning  $(date)"
    if [ "${BENCHMARK}" == "r2med" ]; then
        MODEL=hltcoe/Rank-K-32B PROMPT_MODE=rank_GPT_rankk \
            bash run_parallel_r2med.sh
    else
        bash run_parallel_rankk.sh
    fi
fi

TOTAL_END=$(date +%s)
ELAPSED=$((TOTAL_END - TOTAL_START))
echo ""
echo "============================================================"
echo "  All baselines complete! ${ELAPSED}s total"
echo "  Finished at: $(date)"
echo "  Results: results/"
echo "============================================================"
