import argparse
import os
import sys
import copy
import json
import time
import datetime
from typing import Any, Dict, List, Union, Optional, Sequence
from data import Query, Request, Candidate
import torch
from enum import Enum
from rerank.api_keys import get_openai_api_key, get_api_key_and_base
from rerank.rank_gpt import SafeOpenai
from rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rerank.rankllm import PromptMode, RankLLM
from rerank.reranker import Reranker
from pyserini.search.lucene import LuceneSearcher
try:
    from pyserini.search import FaissSearcher
except ImportError:
    FaissSearcher = None
from pyserini.search._base import get_topics, get_qrels
from utils import OutputFormat, get_output_writer, get_qrels_dl22, get_topics_dl22, get_topics_qrels_excluded_ids_for_bright, get_topics_qrels_for_r2med
from dataclasses import dataclass, field
from trec_eval import Eval
from index_and_topics import THE_TOPICS, THE_INDEX, THE_QRELS, BRIGHT_DATASETS, r2med_DATASETS, DOC_MAXLEN
from enum import Enum
from transformers import HfArgumentParser
from tqdm import tqdm
from config import WORKSPACE_DIR, PROJECT_DIR
import random
from datasets import load_dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Run `pip install wandb` to enable logging.")

import re
import numpy as np


def parse_cot_answer_lengths(response: str):
    """Parse a response to extract CoT (think) and answer character/word lengths."""
    cot_text = ""
    answer_text = ""

    # Extract <think>...</think>
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        cot_text = think_match.group(1).strip()

    # Extract <answer>...</answer>
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()

    return {
        'cot_chars': len(cot_text),
        'cot_words': len(cot_text.split()) if cot_text else 0,
        'answer_chars': len(answer_text),
        'answer_words': len(answer_text.split()) if answer_text else 0,
        'total_chars': len(response),
    }


def compute_generation_stats(reranked_results):
    """Compute input/output/cot/answer token and length statistics from reranked results."""
    total_input_tokens = 0
    total_output_tokens = 0
    cot_chars_list = []
    cot_words_list = []
    answer_chars_list = []
    answer_words_list = []
    input_token_list = []
    output_token_list = []

    for result in reranked_results:
        for summary in result.ranking_exec_summary:
            total_input_tokens += summary.input_token_count
            total_output_tokens += summary.output_token_count
            input_token_list.append(summary.input_token_count)
            output_token_list.append(summary.output_token_count)

            lengths = parse_cot_answer_lengths(summary.response)
            cot_chars_list.append(lengths['cot_chars'])
            cot_words_list.append(lengths['cot_words'])
            answer_chars_list.append(lengths['answer_chars'])
            answer_words_list.append(lengths['answer_words'])

    stats = {
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'avg_input_tokens': np.mean(input_token_list) if input_token_list else 0,
        'avg_output_tokens': np.mean(output_token_list) if output_token_list else 0,
        'avg_cot_chars': np.mean(cot_chars_list) if cot_chars_list else 0,
        'avg_cot_words': np.mean(cot_words_list) if cot_words_list else 0,
        'max_cot_words': max(cot_words_list) if cot_words_list else 0,
        'min_cot_words': min(cot_words_list) if cot_words_list else 0,
        'avg_answer_chars': np.mean(answer_chars_list) if answer_chars_list else 0,
        'avg_answer_words': np.mean(answer_words_list) if answer_words_list else 0,
        'num_prompts': len(input_token_list),
    }
    return stats

os.environ["PYSERINI_CACHE"] = "../../cache"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

class RetrievalMode(Enum):
    DATASET = "dataset"
    CUSTOM = "custom"

    def __str__(self):
        return self.value

class RetrievalMethod(Enum):
    UNSPECIFIED = "unspecified"
    BM25 = "bm25"
    BM25_RM3 = "bm25_rm3"
    SPLADE_P_P_ENSEMBLE_DISTIL = "SPLADE++_EnsembleDistil_ONNX"
    D_BERT_KD_TASB = "distilbert_tas_b"
    OPEN_AI_ADA2 = "openai-ada2"
    REP_LLAMA = "rep-llama"
    CUSTOM_INDEX = "custom_index"
    RESONIR = "reasonir"
    E5_MISTRAL = "e5-mistral-7b-instruct"
    RaDeR = "RaDeR-gte-Qwen2-LLMq_CoT_lexical_hybrid_BM25_Rader"

    def __str__(self):
        return self.value

@dataclass
class Arguments:
    # retrieval arguments
    datasets: List[str] = field(metadata={'help': 'List of test datasets.'})
    output: str = field(metadata={'help': 'Path to output file.'})
    output_format: Optional[str] = field(default='trec', metadata={'help': 'Output format.'})
    retrieval_method: RetrievalMethod = field(default=RetrievalMethod.BM25, metadata={'help': 'Method of retrieval.'})
    retrieval_results_name: str = field(default=None, metadata={'help': 'the filename of custom retrieval results'})
    retrieval_num: int = field(default=100, metadata={'help': 'retrieval number'})
    rerank_topk: int = field(default=None, metadata={'help': 'only need to rerank top-k candidates'})
    threads: int = field(default=30, metadata={'help': 'Number of threads.'})
    batchsize_retrieval: int = field(default=32, metadata={'help': 'batchsize for dense retrieval'})
    reasoning_maxlen: int = field(default=1500, metadata={'help': 'the max length of reasoning chain to generate'})
    remove_query: Optional[bool] = field(default=True, metadata={'help': 'Remove query from output.'})
    save_first_stage_run: Optional[bool] = field(default=True, metadata={'help': 'Save first-stage run.'})
    remove_duplicates: Optional[bool] = field(default=False, metadata={'help': 'Remove duplicates from output.'})
    shuffle_candidates: bool = field(default=False, metadata={'help': 'Whether to shuffle the candidates before reranking.'})
    api_resource: str = field(default='baidu', metadata={'help': ''})
    use_gpt4cot_retrieval: bool = field(default=True, metadata={'help': 'Whether to use the retrieval results of gpt4 cot for BRIGHT datasets (otherwise use the original query).'})

    # llm arguments
    model_path: str = field(default=f'{WORKSPACE_DIR}/llm/rank_vicuna_7b_v1', metadata={'help': 'Path to the model. If `use_azure_ai`, pass your deployment name.'})
    lora_path: str = field(default=None, metadata={'help': 'path of lora'})
    max_lora_rank: int = field(default=32, metadata={'help': 'one parameter of vllm initialization'})
    use_azure_openai: bool = field(default=False, metadata={'help': 'If True, use Azure OpenAI. Requires env var to be set: `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`'})
    context_size: int = field(default=4096, metadata={'help': 'context size used for model.'})
    num_gpus: int = field(default=1, metadata={'help': 'the number of GPUs to use'})
    cache_dir: Optional[str] = field(default='../../cache', metadata={'help': 'Path to cache directory.'})
    llm_dtype: str = field(default='bf16', metadata={'help': 'Data type of llm.'})
    batch_size: int = field(default=32, metadata={'help': 'inference batchsize'})
    num_passes: int = field(default=1, metadata={'help': 'Number of passes to run the model.'})
    window_size: int = field(default=20, metadata={'help': 'Window size for the sliding window approach.'})
    step_size: int = field(default=10, metadata={'help': 'Step size for the sliding window approach.'})
    vllm_batched: bool = field(default=False, metadata={'help': 'Whether to run the model in batches.'})
    prompt_mode: PromptMode = field(default=PromptMode.RANK_GPT, metadata={'required': True, 'choices': list(PromptMode)})
    prompt_info_path: str = field(default=f'{PROJECT_DIR}/listwise_prompt_r1.toml')
    disable_prefix_caching: bool = field(default=False, metadata={'help': 'Disable vLLM prefix caching (needed for some multimodal models like DeepSeek-OCR)'})
    notes: str = field(default='', metadata={'help': 'notes for code running'})

    # wandb arguments
    wandb_project: str = field(default='ReasonRank', metadata={'help': 'wandb project name'})
    wandb_entity: Optional[str] = field(default=None, metadata={'help': 'wandb entity (team/user)'})
    wandb_run_name: Optional[str] = field(default=None, metadata={'help': 'wandb run name, auto-generated if not set'})
    wandb_disabled: bool = field(default=False, metadata={'help': 'disable wandb logging'})


def init_wandb(args):
    """Initialize wandb run with full config."""
    if not WANDB_AVAILABLE or args.wandb_disabled:
        return None

    run_name = args.wandb_run_name
    if run_name is None:
        model_short = args.model_path.split('/')[-1]
        datasets_str = '_'.join(args.datasets[:3])
        if len(args.datasets) > 3:
            datasets_str += f'_+{len(args.datasets)-3}more'
        run_name = f"{model_short}_{datasets_str}_{datetime.datetime.now().strftime('%m%d_%H%M')}"

    # Collect GPU info
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info['gpu_count'] = torch.cuda.device_count()
        gpu_info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        gpu_info['gpu_memory_gb'] = [round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1) for i in range(torch.cuda.device_count())]
        gpu_info['cuda_version'] = torch.version.cuda

    config = {
        # model config
        'model_path': args.model_path,
        'lora_path': args.lora_path,
        'context_size': args.context_size,
        'llm_dtype': args.llm_dtype,
        'num_gpus': args.num_gpus,
        'prompt_mode': str(args.prompt_mode),
        # retrieval config
        'retrieval_method': str(args.retrieval_method),
        'retrieval_num': args.retrieval_num,
        'rerank_topk': args.rerank_topk,
        # reranking config
        'window_size': args.window_size,
        'step_size': args.step_size,
        'num_passes': args.num_passes,
        'batch_size': args.batch_size,
        'vllm_batched': args.vllm_batched,
        'reasoning_maxlen': args.reasoning_maxlen,
        'shuffle_candidates': args.shuffle_candidates,
        # datasets
        'datasets': args.datasets,
        'num_datasets': len(args.datasets),
        # misc
        'notes': args.notes,
        **gpu_info,
    }

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config,
        tags=[args.model_path.split('/')[-1], str(args.retrieval_method)],
    )
    return run


def log_dataset_results_to_wandb(wandb_run, args, dataset, all_metrics, time_cost, current_pass, gen_stats):
    """Log per-dataset evaluation results to wandb."""
    if wandb_run is None:
        return

    total_input_tokens = gen_stats['total_input_tokens']
    total_output_tokens = gen_stats['total_output_tokens']

    # Log per-dataset metrics with dataset prefix
    metrics = {}
    for metric_name, metric_value in all_metrics.items():
        metrics[f"{dataset}/{metric_name}"] = float(metric_value)

    # Log timing and token usage per dataset
    metrics[f"{dataset}/time_cost_s"] = time_cost
    metrics[f"{dataset}/total_input_tokens"] = total_input_tokens
    metrics[f"{dataset}/total_output_tokens"] = total_output_tokens
    metrics[f"{dataset}/total_tokens"] = total_input_tokens + total_output_tokens
    metrics[f"{dataset}/pass"] = current_pass

    # Log CoT and answer length stats
    metrics[f"{dataset}/avg_input_tokens"] = gen_stats['avg_input_tokens']
    metrics[f"{dataset}/avg_output_tokens"] = gen_stats['avg_output_tokens']
    metrics[f"{dataset}/avg_cot_words"] = gen_stats['avg_cot_words']
    metrics[f"{dataset}/max_cot_words"] = gen_stats['max_cot_words']
    metrics[f"{dataset}/min_cot_words"] = gen_stats['min_cot_words']
    metrics[f"{dataset}/avg_answer_words"] = gen_stats['avg_answer_words']
    metrics[f"{dataset}/num_prompts"] = gen_stats['num_prompts']

    wandb_run.log(metrics)


def log_summary_to_wandb(wandb_run, all_dataset_metrics):
    """Log aggregated summary across all datasets."""
    if wandb_run is None or not all_dataset_metrics:
        return

    # Build a wandb Table for the full results
    columns = ["dataset", "NDCG@1", "NDCG@5", "NDCG@10", "time_cost_s",
               "input_tokens", "output_tokens", "avg_input_tok", "avg_output_tok",
               "avg_cot_words", "max_cot_words", "avg_answer_words", "num_prompts"]
    table = wandb.Table(columns=columns)

    ndcg1_values, ndcg5_values, ndcg10_values = [], [], []
    total_time, total_input, total_output = 0, 0, 0
    all_cot_words, all_answer_words = [], []

    for ds_name, ds_info in all_dataset_metrics.items():
        metrics = ds_info['metrics']
        gs = ds_info.get('gen_stats', {})
        ndcg1 = float(metrics.get('NDCG@1', 0))
        ndcg5 = float(metrics.get('NDCG@5', 0))
        ndcg10 = float(metrics.get('NDCG@10', 0))
        t_cost = ds_info['time_cost']
        in_tok = ds_info['input_tokens']
        out_tok = ds_info['output_tokens']

        table.add_data(ds_name, ndcg1, ndcg5, ndcg10, t_cost, in_tok, out_tok,
                       round(gs.get('avg_input_tokens', 0), 1),
                       round(gs.get('avg_output_tokens', 0), 1),
                       round(gs.get('avg_cot_words', 0), 1),
                       gs.get('max_cot_words', 0),
                       round(gs.get('avg_answer_words', 0), 1),
                       gs.get('num_prompts', 0))

        ndcg1_values.append(ndcg1)
        ndcg5_values.append(ndcg5)
        ndcg10_values.append(ndcg10)
        total_time += t_cost
        total_input += in_tok
        total_output += out_tok
        if gs.get('avg_cot_words', 0) > 0:
            all_cot_words.append(gs['avg_cot_words'])
            all_answer_words.append(gs['avg_answer_words'])

    # Log the table
    wandb_run.log({"results_table": table})

    # Log aggregated summary metrics
    summary = {
        "summary/avg_NDCG@1": np.mean(ndcg1_values),
        "summary/avg_NDCG@5": np.mean(ndcg5_values),
        "summary/avg_NDCG@10": np.mean(ndcg10_values),
        "summary/total_time_s": total_time,
        "summary/total_input_tokens": total_input,
        "summary/total_output_tokens": total_output,
        "summary/total_tokens": total_input + total_output,
        "summary/num_datasets": len(all_dataset_metrics),
        "summary/avg_cot_words": np.mean(all_cot_words) if all_cot_words else 0,
        "summary/avg_answer_words": np.mean(all_answer_words) if all_answer_words else 0,
    }
    wandb_run.log(summary)

    # Also set wandb summary for easy comparison
    for k, v in summary.items():
        wandb_run.summary[k] = v


def write_run(output_writer, results, args):
    with output_writer:
        for request in results:
            qid = request.query.qid
            hits = request.candidates
            if args.remove_duplicates:
                seen_docids = set()
                dedup_hits = []
                for hit in hits:
                    if hit.docid.strip() in seen_docids:
                        continue
                    seen_docids.add(hit.docid.strip())
                    dedup_hits.append(hit)
                hits = dedup_hits

            # For some test collections, a query is doc from the corpus (e.g., arguana in BEIR).
            # We want to remove the query from the results.
            if args.remove_query:
                hits = [hit for hit in hits if hit.docid != qid]

            # write results
            output_writer.write(qid, hits)

def evaluate_results(args, dataset, out_path, qrels, time_cost, current_pass, gen_stats, wandb_run=None):
    all_metrics = Eval(out_path, qrels)
    print(f'###################### {dataset} ######################')
    print(all_metrics)
    print(f'time_cost: {time_cost}')
    print(f'avg_input_tokens: {gen_stats["avg_input_tokens"]:.0f}, avg_output_tokens: {gen_stats["avg_output_tokens"]:.0f}')
    print(f'avg_cot_words: {gen_stats["avg_cot_words"]:.0f} (min={gen_stats["min_cot_words"]}, max={gen_stats["max_cot_words"]}), avg_answer_words: {gen_stats["avg_answer_words"]:.0f}')
    result = {'model_path': args.model_path,
              'lora_path': args.lora_path,
              'datetime': str(datetime.datetime.now()),
              'retrieval_method': args.retrieval_method,
              'retrieval_num': args.retrieval_num,
              'rerank_topk': args.rerank_topk if args.rerank_topk is not None else 100,
              'current_pass': current_pass,
              'window_size': args.window_size,
              'step_size': args.step_size,
              'shuffle_candidates': args.shuffle_candidates,
              'time_cost': time_cost,
              'total_input_tokens': gen_stats['total_input_tokens'],
              'total_output_tokens': gen_stats['total_output_tokens'],
              'avg_input_tokens': round(gen_stats['avg_input_tokens'], 1),
              'avg_output_tokens': round(gen_stats['avg_output_tokens'], 1),
              'avg_cot_words': round(gen_stats['avg_cot_words'], 1),
              'max_cot_words': gen_stats['max_cot_words'],
              'min_cot_words': gen_stats['min_cot_words'],
              'avg_answer_words': round(gen_stats['avg_answer_words'], 1),
              'num_prompts': gen_stats['num_prompts'],
              'notes': args.notes,
              **all_metrics}
    os.makedirs('results/', exist_ok=True)
    result_path = f'results/{dataset}.json'
    lock_path = f'{result_path}.lock'
    import fcntl
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            if not os.path.exists(result_path):
                json_data = []
            else:
                with open(result_path, 'r') as f:
                    try:
                        json_data = json.load(f)
                    except json.JSONDecodeError:
                        json_data = []
            json_data.append(result)
            with open(result_path, 'w') as f:
                json.dump(json_data, f, indent=4)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)

    # Log to wandb
    log_dataset_results_to_wandb(wandb_run, args, dataset, all_metrics, time_cost, current_pass, gen_stats)

    return all_metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(Arguments)
    _args, *_ = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(_args))

    # Initialize wandb
    wandb_run = init_wandb(args)

    ############################## retrieval for each dataset ##############################
    dataset_qrels = {}
    dataset_topics = {}
    dataset_results = {}
    ### load bright datasets if needed
    if len(set(args.datasets) & set(BRIGHT_DATASETS)) >= 1:
        all_bright_examples = load_dataset('xlangai/bright', 'examples', cache_dir=f'{WORKSPACE_DIR}/data/bright')
    for dataset in args.datasets: 
        if dataset in BRIGHT_DATASETS: # BRIGHT
            topics, qrels, excluded_ids = get_topics_qrels_excluded_ids_for_bright(dataset, all_bright_examples, is_long_doc=False)
            id2doc = {}
            with open(f'{WORKSPACE_DIR}/data/bright/{dataset}/corpus.jsonl', 'r') as f:
                for line in f:
                    data = json.loads(line)
                    id2doc[data['id']] = data['contents']
        elif dataset in r2med_DATASETS: # R2MED
            topics, qrels = get_topics_qrels_for_r2med(dataset)
            excluded_ids = {qid: [] for qid in list(topics.keys())}
            id2doc = {}
            with open(f'{WORKSPACE_DIR}/data/r2med/{dataset}/corpus.jsonl', 'r') as f:
                for line in f:
                    data = json.loads(line)
                    id2doc[data['id']] = data['contents']
        else: # TREC and BEIR
            index_path = THE_INDEX[dataset]
            topics_path = THE_TOPICS[dataset]
            qrels_path = THE_QRELS[dataset]

            searcher = LuceneSearcher.from_prebuilt_index(index_path)
            topics = {str(qid): content['title'] for qid, content in get_topics(topics_path).items()} if dataset != 'dl22' else get_topics_dl22() # dl22 is not supported by pyserini 0.20.0
            qrels = {str(qid): {str(docid): int(score) for docid, score in docs.items()} 
                    for qid, docs in get_qrels(qrels_path).items()} if dataset != 'dl22' else get_qrels_dl22()
            excluded_ids = {qid: [] for qid in list(topics.keys())}
        dataset_qrels[dataset] = qrels
        dataset_topics[dataset] = topics
        batch_topic_ids = []
        batch_topics = []
        for topic_id in list(topics.keys()):
            if topic_id in qrels:
                batch_topic_ids.append(topic_id)
                batch_topics.append(topics[topic_id])
        if args.retrieval_results_name is not None:
            first_state_run_path = f'runs/{dataset}/{args.retrieval_results_name}'
        else:
            if args.retrieval_method in ['bm25', 'reasonir']:
                if args.use_gpt4cot_retrieval and dataset in BRIGHT_DATASETS:
                    first_state_run_path = f'runs/{dataset}/{args.retrieval_method}_gpt4cot_top{args.retrieval_num}.txt'
                else:
                    first_state_run_path = f'runs/{dataset}/{args.retrieval_method}_top{args.retrieval_num}.txt'
            else:
                first_state_run_path = f'runs/{dataset}/{args.retrieval_method}_top{args.retrieval_num}.txt'
        if os.path.exists(first_state_run_path):
            print(f'Loading first stage run from {first_state_run_path}.')
            results = []
            with open(first_state_run_path, 'r') as f:
                current_qid = None
                current_ranking = []
                for line in f:
                    if len(line.strip().split()) != 6:  # for stackoverflow dataset, there are several docids containing white space (e.g., pytorch_torch_tensor_functions/Memory Management_2_0.txt), we just filter these docids
                        continue
                    qid, _, docid, _, score, _ = line.strip().split()
                    if qid != current_qid:
                        if current_qid is not None:
                            current_query = Query(qid=current_qid, text=topics[current_qid])
                            results.append(Request(query=current_query, candidates=current_ranking[:args.retrieval_num]))
                        current_ranking = []
                        current_qid = qid
                    current_ranking.append(Candidate(
                                            docid=docid, 
                                            score=float(score), 
                                            doc={'contents': id2doc[docid]} if dataset in BRIGHT_DATASETS + r2med_DATASETS else json.loads(searcher.doc(docid).raw()))
                                            )
                current_query = Query(qid=current_qid, text=topics[current_qid])
                results.append(Request(query=current_query, candidates=current_ranking[:args.retrieval_num]))
        else:
            if args.retrieval_method != 'bm25':
                raise ValueError("the runs of retrieval_method are not found!")
            print(f'First stage run on {dataset} based on bm25...')
            max_excluded_ids_num = max([len(excluded_ids[qid]) for qid in batch_topic_ids])
            _results = searcher.batch_search(batch_topics, batch_topic_ids, k=args.retrieval_num + max_excluded_ids_num, threads=args.threads) # we need to filter the excluded_ids while keeping the retrieved number as retrieval_num
            results = []
            for topic_id in batch_topic_ids:
                candidates = [Candidate(docid=result.docid, score=result.score, doc=json.loads(searcher.doc(result.docid).raw())) 
                                    for result in _results[topic_id] 
                                    if result.docid not in excluded_ids[topic_id]][:args.retrieval_num]
                results.append(Request(query=Query(qid=topic_id, text=topics[topic_id]), candidates=candidates))

            if args.save_first_stage_run:
                output_writer = get_output_writer(first_state_run_path, OutputFormat(args.output_format), 'w',
                                                max_hits=args.retrieval_num, tag=args.retrieval_method, topics=topics, )
                write_run(output_writer, results, args)
        dataset_results[dataset] = results
    ############################# load LLM reranker #############################
    API_MODELS = ['gpt-4o-mini-2024-07-18', 'gpt-4o-mini', 'gpt-4o-2024-08-06', 'gpt-4o-2024-05-13', 'gpt-4o', 'deepseek-chat', 'deepseek-reasoner']
    if args.model_path in API_MODELS:
        api_key, api_base_url = get_api_key_and_base(args.model_path)
        agent = SafeOpenai(
            args=args,
            model=args.model_path,
            context_size=args.context_size,
            resource=args.api_resource,
            prompt_mode=args.prompt_mode,
            window_size=args.window_size,
            prompt_info_path=args.prompt_info_path,
            keys=api_key,
            api_base_url=api_base_url,
        )
    else:
        print(f"Loading {args.model_path} ...")
        agent = RankListwiseOSLLM(
            args=args,
            model=args.model_path,
            context_size=args.context_size,
            prompt_mode=args.prompt_mode,
            num_gpus=args.num_gpus,
            window_size=args.window_size,
            prompt_info_path=args.prompt_info_path,
            vllm_batched=args.vllm_batched,
        )
    reranker = Reranker(agent)
    ###################################### Reranking ######################################
    all_dataset_metrics = {}  # collect for wandb summary
    for dataset in args.datasets:
        print(f"########################## Reranking on {dataset} ##########################")
        qrels = dataset_qrels[dataset]
        topics = dataset_topics[dataset]
        results = dataset_results[dataset]
        agent.max_passage_length = DOC_MAXLEN[dataset]

        total_time_cost = 0
        for pass_ct in range(args.num_passes):
            print(f"Pass {pass_ct + 1} of {args.num_passes}:")
            # in case of the number of candidate passages < 100 (for sliding windows strategy)
            results_with_100_passages = []
            results_less_100_passages = []
            for result in results:
                if len(result.candidates) < args.retrieval_num:
                    results_less_100_passages.append([result])
                else:
                    results_with_100_passages.append(result)
            results_grouped = [results_with_100_passages[i:i+args.batch_size] for i in range(0, len(results_with_100_passages), args.batch_size)] + results_less_100_passages
            reranked_results = []
            rerank_details = []
            for batch in results_grouped:
                reranked_batch, time_cost, rerank_details_batch = reranker.rerank_batch(
                                            batch,
                                            rank_start=args.step_size * pass_ct,
                                            rank_end=args.retrieval_num if args.rerank_topk is None else args.rerank_topk,
                                            window_size=min(args.window_size, len(batch[0].candidates)),
                                            shuffle_candidates=args.shuffle_candidates,
                                            step=args.step_size,
                                            vllm_batched=args.vllm_batched,
                                        )
                reranked_results.extend(reranked_batch)
                total_time_cost += time_cost
                rerank_details.extend(rerank_details_batch)
            # save results and evaluate
            out_path = os.path.join(f'runs/{dataset}', args.output)
            output_writer = get_output_writer(out_path, OutputFormat(args.output_format), 'w',
                                                max_hits=args.retrieval_num, tag=args.retrieval_method, topics=topics, )
            write_run(output_writer, reranked_results, args)
            gen_stats = compute_generation_stats(reranked_results)
            ds_metrics = evaluate_results(args, dataset, out_path, qrels, time_cost=total_time_cost, current_pass=pass_ct+1,
                                          gen_stats=gen_stats, wandb_run=wandb_run)
            if args.num_passes > 1:
                results = [
                    Request(copy.deepcopy(r.query), copy.deepcopy(r.candidates))
                    for r in reranked_results
                ]
        # Store final pass metrics for summary
        all_dataset_metrics[dataset] = {
            'metrics': ds_metrics,
            'time_cost': total_time_cost,
            'input_tokens': gen_stats['total_input_tokens'],
            'output_tokens': gen_stats['total_output_tokens'],
            'gen_stats': gen_stats,
        }
        print(f"Reranking with {args.num_passes} passes complete!")

    # Log aggregated summary to wandb and finish
    log_summary_to_wandb(wandb_run, all_dataset_metrics)
    if wandb_run is not None:
        wandb_run.finish()
        print(f"wandb run finished: {wandb_run.url}")