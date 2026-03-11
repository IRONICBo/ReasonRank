# ReasonRank Evaluation Results on B200

## Environment

- **GPU**: 8x NVIDIA B200 (CUDA 13.0, Driver 580.82.07)
- **PyTorch**: 2.10.0+cu128
- **Inference**: vLLM batched, TP=1, 8-way data parallel
- **Benchmark**: BRIGHT (12 datasets)
- **Retrieval**: ReasonIR (top-100)
- **Reranking**: sliding window (window=20, step=10, 1 pass)

## reasonrank-7B (Qwen2.5-7B-Instruct)

| Dataset | NDCG@1 | NDCG@5 | NDCG@10 | Time (s) |
|---------|--------|--------|---------|----------|
| economics | 31.07 | 34.12 | 34.96 | 147.0 |
| earth_science | 55.17 | 46.83 | 46.48 | 120.3 |
| robotics | 26.73 | 28.22 | 29.32 | 113.0 |
| biology | 57.28 | 53.92 | 56.66 | 102.6 |
| psychology | 40.59 | 45.73 | 46.55 | 129.8 |
| stackoverflow | 23.93 | 27.97 | 30.66 | 260.1 |
| sustainable_living | 33.33 | 36.60 | 40.05 | 109.8 |
| leetcode | 14.79 | 19.82 | 23.36 | 356.4 |
| pony | 36.61 | 28.22 | 25.43 | 118.3 |
| aops | 9.01 | 7.33 | 7.73 | 214.9 |
| theoremqa_questions | 38.14 | 37.48 | 38.63 | 339.7 |
| theoremqa_theorems | 34.21 | 39.76 | 43.33 | 147.1 |
| **Average** | **33.40** | **33.83** | **35.26** | **2159.1 (total)** |

- Total tokens: 80,864,264

## reasonrank-32B (Qwen2.5-32B-Instruct)

> Pending evaluation.

## reasonrank-8B (Qwen3-8B)

> Pending evaluation.
