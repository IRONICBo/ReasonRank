"""
Rank1 Pointwise Reranker Evaluation
====================================
Model: jhu-clsp/rank1-7b, jhu-clsp/rank1-32b
Method: Pointwise scoring with reasoning. For each (query, doc) pair,
        the model generates <think>reasoning</think> then "true"/"false".
        Score = P(true) / (P(true) + P(false)) from logprobs.

Usage:
    python baseline/run_rank1.py \
        --model_path jhu-clsp/rank1-7b \
        --datasets economics earth_science \
        --retrieval_method reasonir \
        --num_gpus 1
"""
import argparse
import os
import sys
import time
import json
import math
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import LLM, SamplingParams
from baseline.eval_utils import (
    load_datasets_and_candidates, write_trec_run,
    save_details_jsonl, evaluate_and_save, init_wandb_run,
)
from index_and_topics import DOC_MAXLEN
from utils import convert_doc_to_prompt_content
from transformers import AutoTokenizer


RANK1_PROMPT_TEMPLATE = (
    "Determine if the following passage is relevant to the query. "
    "Answer only with 'true' or 'false'.\n"
    "Query: {query}\n"
    "Passage: {passage}\n"
)


def build_prompts(tokenizer, query, candidates, max_passage_length):
    """Build pointwise prompts for all candidates."""
    prompts = []
    for cand in candidates:
        content = cand.doc.get('contents', cand.doc.get('text', ''))
        # Truncate passage
        tokens = tokenizer.encode(content, add_special_tokens=False)
        if len(tokens) > max_passage_length:
            content = tokenizer.decode(tokens[:max_passage_length])

        user_msg = RANK1_PROMPT_TEMPLATE.format(query=query, passage=content)
        messages = [{"role": "user", "content": user_msg}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
        # Append <think> to trigger reasoning mode
        prompt += "<think>\n"
        prompts.append(prompt)
    return prompts


def score_from_logprobs(output):
    """Extract P(true)/(P(true)+P(false)) from the output logprobs."""
    # The output should end with </think> followed by "true" or "false"
    # We look at the logprobs of the final generated token(s) for true/false
    text = output.outputs[0].text.lower().strip()

    # Check cumulative logprobs for the entire output as a simple fallback
    # The main scoring is from the final token logprobs
    logprobs_list = output.outputs[0].logprobs
    if not logprobs_list:
        # Fallback: check if output contains "true" or "false"
        if 'true' in text.split('</think>')[-1]:
            return 1.0
        return 0.0

    # Find the token after </think> - that's where true/false appears
    # We look at the last few logprob entries for true/false probabilities
    true_logprob = -float('inf')
    false_logprob = -float('inf')

    for lp_dict in reversed(logprobs_list[-5:]):
        if lp_dict is None:
            continue
        for token_id, lp_info in lp_dict.items():
            token_text = lp_info.decoded_token.lower().strip() if hasattr(lp_info, 'decoded_token') else ''
            if token_text == 'true':
                true_logprob = max(true_logprob, lp_info.logprob)
            elif token_text == 'false':
                false_logprob = max(false_logprob, lp_info.logprob)

    if true_logprob == -float('inf') and false_logprob == -float('inf'):
        # Fallback: parse text
        final_text = text.split('</think>')[-1].strip() if '</think>' in text else text
        if 'true' in final_text:
            return 1.0
        return 0.0

    # Softmax over true/false
    max_lp = max(true_logprob, false_logprob)
    true_prob = math.exp(true_logprob - max_lp)
    false_prob = math.exp(false_logprob - max_lp)
    score = true_prob / (true_prob + false_prob)
    return score


def rerank_pointwise(llm, tokenizer, requests, max_passage_length, batch_size=256):
    """Rerank all requests using pointwise scoring."""
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=8192,
        logprobs=20,
        stop=["</think> true", "</think> false", "</think>\ntrue", "</think>\nfalse"],
    )

    all_details = []
    for req in tqdm(requests, desc="Reranking queries"):
        query = req.query.text
        prompts = build_prompts(tokenizer, query, req.candidates, max_passage_length)

        # Score all candidates for this query
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        scored = []
        for i, (cand, output) in enumerate(zip(req.candidates, outputs)):
            score = score_from_logprobs(output)
            response_text = output.outputs[0].text
            scored.append((cand, score, response_text))
            all_details.append({
                'qid': req.query.qid,
                'query': query,
                'docid': cand.docid,
                'prompt': prompts[i],
                'response': response_text,
                'score': score,
                'input_tokens': len(output.prompt_token_ids),
                'output_tokens': len(output.outputs[0].token_ids),
            })

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        for rank, (cand, score, _) in enumerate(scored):
            cand.score = 1.0 / (rank + 1)  # Use reciprocal rank as score
        req.candidates = [c for c, _, _ in scored]

    return requests, all_details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='jhu-clsp/rank1-7b')
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--retrieval_method', type=str, default='reasonir')
    parser.add_argument('--retrieval_num', type=int, default=100)
    parser.add_argument('--use_gpt4cot_retrieval', type=bool, default=True)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--wandb_project', type=str, default='ReasonRank')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_disabled', action='store_true')
    parser.add_argument('--notes', type=str, default='')
    args = parser.parse_args()

    model_short = args.model_path.split('/')[-1]

    # Load data
    dataset_data = load_datasets_and_candidates(
        args.datasets, args.retrieval_method, args.retrieval_num,
        args.use_gpt4cot_retrieval)

    # Load model
    print(f"Loading {args.model_path} with vLLM ...")
    llm = LLM(args.model_path,
              download_dir=os.getenv("HF_HOME"),
              gpu_memory_utilization=0.9,
              tensor_parallel_size=args.num_gpus,
              trust_remote_code=True)
    tokenizer = llm.get_tokenizer()

    # Init wandb
    wandb_run = None
    if not args.wandb_disabled:
        run_name = args.wandb_run_name or f"{model_short}_pointwise"
        wandb_run = init_wandb_run(args.model_path, args.datasets,
                                    project=args.wandb_project,
                                    entity=args.wandb_entity,
                                    extra_config={'reranker_type': 'pointwise',
                                                  'method': 'rank1'})

    # Evaluate each dataset
    for dataset in args.datasets:
        if dataset not in dataset_data:
            print(f"Skipping {dataset} (no data loaded)")
            continue
        topics, qrels, requests = dataset_data[dataset]
        max_pass_len = DOC_MAXLEN.get(dataset, 512)

        print(f"\n{'='*60}")
        print(f"  Rank1 pointwise reranking: {dataset}")
        print(f"  Queries: {len(requests)}, Max passage len: {max_pass_len}")
        print(f"{'='*60}")

        t_start = time.time()
        reranked, details = rerank_pointwise(llm, tokenizer, requests, max_pass_len)
        t_cost = time.time() - t_start

        # Write TREC run
        out_path = f'runs/{dataset}/{model_short}.txt'
        write_trec_run(reranked, out_path, tag=model_short)

        # Save details
        details_path = f'runs/{dataset}/{model_short}_details.jsonl'
        save_details_jsonl(details, details_path)
        print(f"  Saved {len(details)} detail records to {details_path}")

        # Evaluate
        evaluate_and_save(dataset, out_path, qrels, args.model_path,
                          t_cost, extra_info=args.notes, wandb_run=wandb_run)

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
