"""
RankT5 / monoT5 Pointwise Reranker Evaluation
================================================
Model: castorini/monot5-3b-msmarco
Method: Pointwise scoring. For each (query, doc) pair, the model generates
        "true" or "false". Score = logit("true") - logit("false").

Uses HuggingFace Transformers (T5 is encoder-decoder, not vLLM compatible).

Usage:
    python baseline/run_rankt5.py \
        --model_path castorini/monot5-3b-msmarco \
        --datasets economics earth_science \
        --retrieval_method reasonir
"""
import argparse
import os
import sys
import time
import json
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import T5Tokenizer, T5ForConditionalGeneration
from baseline.eval_utils import (
    load_datasets_and_candidates, write_trec_run,
    save_details_jsonl, evaluate_and_save, init_wandb_run,
)
from index_and_topics import DOC_MAXLEN


MONOT5_PROMPT = "Query: {query} Document: {document} Relevant:"


def score_batch(model, tokenizer, prompts, device, batch_size=32):
    """Score a batch of prompts. Returns list of scores."""
    scores = []
    true_id = tokenizer.encode("true", add_special_tokens=False)[0]
    false_id = tokenizer.encode("false", add_special_tokens=False)[0]

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            # Generate decoder input (just the start token)
            decoder_input_ids = torch.full(
                (len(batch), 1),
                model.config.decoder_start_token_id,
                dtype=torch.long, device=device)
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[:, 0, :]  # First generated token logits
            true_logits = logits[:, true_id]
            false_logits = logits[:, false_id]
            batch_scores = (true_logits - false_logits).cpu().tolist()
            scores.extend(batch_scores)

    return scores


def rerank_pointwise_t5(model, tokenizer, requests, max_passage_length,
                         device, batch_size=32):
    """Rerank all requests using monoT5 pointwise scoring."""
    all_details = []

    for req in tqdm(requests, desc="Reranking queries"):
        query = req.query.text
        prompts = []
        for cand in req.candidates:
            content = cand.doc.get('contents', cand.doc.get('text', ''))
            # Truncate by words (T5 tokenizer handles differently)
            words = content.split()
            if len(words) > max_passage_length:
                content = ' '.join(words[:max_passage_length])
            prompt = MONOT5_PROMPT.format(query=query, document=content)
            prompts.append(prompt)

        scores = score_batch(model, tokenizer, prompts, device, batch_size)

        # Pair and sort
        scored = list(zip(req.candidates, scores, prompts))
        scored.sort(key=lambda x: x[1], reverse=True)

        for rank, (cand, score, prompt) in enumerate(scored):
            cand.score = 1.0 / (rank + 1)
            all_details.append({
                'qid': req.query.qid,
                'query': query,
                'docid': cand.docid,
                'prompt': prompt,
                'response': f'score={score:.4f}',
                'score': score,
            })

        req.candidates = [c for c, _, _ in scored]

    return requests, all_details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='castorini/monot5-3b-msmarco')
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--retrieval_method', type=str, default='reasonir')
    parser.add_argument('--retrieval_num', type=int, default=100)
    parser.add_argument('--use_gpt4cot_retrieval', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--wandb_project', type=str, default='ReasonRank')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_disabled', action='store_true')
    parser.add_argument('--notes', type=str, default='')
    args = parser.parse_args()

    model_short = args.model_path.split('/')[-1]

    dataset_data = load_datasets_and_candidates(
        args.datasets, args.retrieval_method, args.retrieval_num,
        args.use_gpt4cot_retrieval)

    print(f"Loading {args.model_path} with Transformers ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    wandb_run = None
    if not args.wandb_disabled:
        wandb_run = init_wandb_run(args.model_path, args.datasets,
                                    project=args.wandb_project,
                                    entity=args.wandb_entity,
                                    extra_config={'reranker_type': 'pointwise',
                                                  'method': 'monot5'})

    for dataset in args.datasets:
        if dataset not in dataset_data:
            print(f"Skipping {dataset}")
            continue
        topics, qrels, requests = dataset_data[dataset]
        max_pass_len = DOC_MAXLEN.get(dataset, 512)

        print(f"\n{'='*60}")
        print(f"  monoT5 pointwise reranking: {dataset}")
        print(f"  Queries: {len(requests)}")
        print(f"{'='*60}")

        t_start = time.time()
        reranked, details = rerank_pointwise_t5(
            model, tokenizer, requests, max_pass_len, device, args.batch_size)
        t_cost = time.time() - t_start

        out_path = f'runs/{dataset}/{model_short}.txt'
        write_trec_run(reranked, out_path, tag=model_short)

        details_path = f'runs/{dataset}/{model_short}_details.jsonl'
        save_details_jsonl(details, details_path)

        evaluate_and_save(dataset, out_path, qrels, args.model_path,
                          t_cost, extra_info=args.notes, wandb_run=wandb_run)

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
