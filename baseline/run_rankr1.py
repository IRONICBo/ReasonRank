"""
Rank-R1 Setwise Reranker Evaluation
=====================================
Model: ielabgroup/Rank-R1-7B-v0.1, ielabgroup/Rank-R1-14B-v0.1
Method: Setwise ranking with reasoning. Given a query and a set of
        candidate documents, the model selects the MOST relevant one.
        Uses a heapsort-like algorithm to produce full ranking.

Usage:
    python baseline/run_rankr1.py \
        --model_path ielabgroup/Rank-R1-7B-v0.1 \
        --datasets economics earth_science \
        --retrieval_method reasonir \
        --num_gpus 1
"""
import argparse
import os
import sys
import time
import json
import re
import copy
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import LLM, SamplingParams
from baseline.eval_utils import (
    load_datasets_and_candidates, write_trec_run,
    save_details_jsonl, evaluate_and_save, init_wandb_run,
)
from index_and_topics import DOC_MAXLEN
from utils import convert_doc_to_prompt_content


RANKR1_SYSTEM = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>."
)

RANKR1_USER_TEMPLATE = (
    'Given the query "{query}", which of the following documents is most relevant?\n'
    '{docs}\n'
    'After completing the reasoning process, please provide only the label of '
    'the most relevant document to the query, enclosed in square brackets, '
    'within the answer tags.'
)


def build_setwise_prompt(tokenizer, query, candidates, max_passage_length):
    """Build a setwise prompt asking the model to pick the most relevant doc."""
    docs_text = ""
    for i, cand in enumerate(candidates):
        content = cand.doc.get('contents', cand.doc.get('text', ''))
        tokens = tokenizer.encode(content, add_special_tokens=False)
        if len(tokens) > max_passage_length:
            content = tokenizer.decode(tokens[:max_passage_length])
        docs_text += f"[{i+1}] {content}\n"

    user_msg = RANKR1_USER_TEMPLATE.format(query=query, docs=docs_text)
    messages = [
        {"role": "system", "content": RANKR1_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
    return prompt


def parse_selected_idx(response):
    """Extract the selected document index from the response."""
    # Look for [N] in <answer> tags
    answer_match = re.search(r'<answer>\s*\[(\d+)\]\s*</answer>', response)
    if answer_match:
        return int(answer_match.group(1)) - 1  # 0-indexed

    # Fallback: look for [N] anywhere
    bracket_matches = re.findall(r'\[(\d+)\]', response)
    if bracket_matches:
        return int(bracket_matches[-1]) - 1

    return 0  # Default to first


def heapsort_rerank(llm, tokenizer, query, candidates, max_passage_length,
                     set_size=20, sampling_params=None):
    """Use heapsort-like approach to rank candidates.

    For each step, pick a set of `set_size` candidates, ask the model which is
    most relevant, move it to the top, repeat with remaining candidates.
    """
    remaining = list(range(len(candidates)))
    ranked_indices = []
    details = []

    while len(remaining) > 1:
        # Take up to set_size candidates from remaining
        current_set = remaining[:set_size]
        current_cands = [candidates[i] for i in current_set]

        prompt = build_setwise_prompt(tokenizer, query, current_cands,
                                       max_passage_length)
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text

        selected = parse_selected_idx(response)
        selected = min(selected, len(current_set) - 1)
        selected = max(selected, 0)

        # Move selected to ranked list
        winner_idx = current_set[selected]
        ranked_indices.append(winner_idx)
        remaining.remove(winner_idx)

        details.append({
            'prompt': prompt,
            'response': response,
            'set_size': len(current_set),
            'selected': selected,
            'winner_docid': candidates[winner_idx].docid,
            'input_tokens': len(outputs[0].prompt_token_ids),
            'output_tokens': len(outputs[0].outputs[0].token_ids),
        })

    # Last remaining goes at the end
    if remaining:
        ranked_indices.append(remaining[0])

    return ranked_indices, details


def rerank_setwise(llm, tokenizer, requests, max_passage_length, set_size=20):
    """Rerank all queries using setwise approach."""
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=4096,
    )

    all_details = []
    for req in tqdm(requests, desc="Reranking queries"):
        query = req.query.text
        ranked_indices, details = heapsort_rerank(
            llm, tokenizer, query, req.candidates,
            max_passage_length, set_size=set_size,
            sampling_params=sampling_params)

        # Reorder candidates
        new_candidates = []
        for rank, idx in enumerate(ranked_indices):
            cand = req.candidates[idx]
            cand.score = 1.0 / (rank + 1)
            new_candidates.append(cand)
        req.candidates = new_candidates

        for d in details:
            d['qid'] = req.query.qid
            d['query'] = query
        all_details.extend(details)

    return requests, all_details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='ielabgroup/Rank-R1-7B-v0.1')
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--retrieval_method', type=str, default='reasonir')
    parser.add_argument('--retrieval_num', type=int, default=100)
    parser.add_argument('--use_gpt4cot_retrieval', type=bool, default=True)
    parser.add_argument('--set_size', type=int, default=20,
                        help='Number of candidates per setwise comparison')
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
        wandb_run = init_wandb_run(args.model_path, args.datasets,
                                    project=args.wandb_project,
                                    entity=args.wandb_entity,
                                    extra_config={'reranker_type': 'setwise',
                                                  'method': 'rank-r1',
                                                  'set_size': args.set_size})

    for dataset in args.datasets:
        if dataset not in dataset_data:
            print(f"Skipping {dataset}")
            continue
        topics, qrels, requests = dataset_data[dataset]
        max_pass_len = DOC_MAXLEN.get(dataset, 512)

        print(f"\n{'='*60}")
        print(f"  Rank-R1 setwise reranking: {dataset}")
        print(f"  Queries: {len(requests)}, Set size: {args.set_size}")
        print(f"{'='*60}")

        t_start = time.time()
        reranked, details = rerank_setwise(llm, tokenizer, requests,
                                            max_pass_len, set_size=args.set_size)
        t_cost = time.time() - t_start

        out_path = f'runs/{dataset}/{model_short}.txt'
        write_trec_run(reranked, out_path, tag=model_short)

        details_path = f'runs/{dataset}/{model_short}_details.jsonl'
        save_details_jsonl(details, details_path)
        print(f"  Saved {len(details)} detail records to {details_path}")

        evaluate_and_save(dataset, out_path, qrels, args.model_path,
                          t_cost, extra_info=args.notes, wandb_run=wandb_run)

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
