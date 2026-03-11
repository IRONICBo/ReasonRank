"""Shared utilities for baseline model evaluation.
Reuses data loading / eval from the main ReasonRank pipeline.
"""
import os, sys, json, datetime, fcntl
import numpy as np

# Make sure the project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import WORKSPACE_DIR, PROJECT_DIR
from index_and_topics import BRIGHT_DATASETS, r2med_DATASETS, DOC_MAXLEN
from data import Query, Request, Candidate
from trec_eval import Eval
from datasets import load_dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ========== Data loading ==========

def load_datasets_and_candidates(datasets, retrieval_method, retrieval_num=100,
                                  use_gpt4cot_retrieval=True):
    """Load topics, qrels, and first-stage retrieval results for each dataset.
    Returns: dict[dataset] -> (topics, qrels, results_list[Request])
    """
    all_bright_examples = None
    if len(set(datasets) & set(BRIGHT_DATASETS)) >= 1:
        all_bright_examples = load_dataset('xlangai/bright', 'examples',
                                           cache_dir=f'{WORKSPACE_DIR}/data/bright')

    from utils import get_topics_qrels_excluded_ids_for_bright, get_topics_qrels_for_r2med

    dataset_data = {}
    for dataset in datasets:
        if dataset in BRIGHT_DATASETS:
            topics, qrels, excluded_ids = get_topics_qrels_excluded_ids_for_bright(
                dataset, all_bright_examples, is_long_doc=False)
            id2doc = {}
            with open(f'{WORKSPACE_DIR}/data/bright/{dataset}/corpus.jsonl', 'r') as f:
                for line in f:
                    data = json.loads(line)
                    id2doc[data['id']] = data['contents']
        elif dataset in r2med_DATASETS:
            topics, qrels = get_topics_qrels_for_r2med(dataset)
            id2doc = {}
            with open(f'{WORKSPACE_DIR}/data/r2med/{dataset}/corpus.jsonl', 'r') as f:
                for line in f:
                    data = json.loads(line)
                    id2doc[data['id']] = data['contents']
        else:
            raise NotImplementedError(f"Dataset {dataset} not supported in baseline eval")

        # Determine first-stage run file
        if retrieval_method in ['bm25', 'reasonir']:
            if use_gpt4cot_retrieval and dataset in BRIGHT_DATASETS:
                run_path = f'runs/{dataset}/{retrieval_method}_gpt4cot_top{retrieval_num}.txt'
            else:
                run_path = f'runs/{dataset}/{retrieval_method}_top{retrieval_num}.txt'
        else:
            run_path = f'runs/{dataset}/{retrieval_method}_top{retrieval_num}.txt'

        if not os.path.exists(run_path):
            print(f"WARNING: First stage run not found: {run_path}, skipping {dataset}")
            continue

        print(f'Loading first stage run from {run_path}')
        results = []
        with open(run_path, 'r') as f:
            current_qid = None
            current_ranking = []
            for line in f:
                if len(line.strip().split()) != 6:
                    continue
                qid, _, docid, _, score, _ = line.strip().split()
                if qid not in qrels:
                    continue
                if qid != current_qid:
                    if current_qid is not None:
                        current_query = Query(qid=current_qid, text=topics[current_qid])
                        results.append(Request(query=current_query,
                                               candidates=current_ranking[:retrieval_num]))
                    current_ranking = []
                    current_qid = qid
                if docid in id2doc:
                    current_ranking.append(Candidate(
                        docid=docid, score=float(score),
                        doc={'contents': id2doc[docid]}))
            if current_qid is not None:
                current_query = Query(qid=current_qid, text=topics[current_qid])
                results.append(Request(query=current_query,
                                       candidates=current_ranking[:retrieval_num]))

        dataset_data[dataset] = (topics, qrels, results)
        print(f"  {dataset}: {len(results)} queries loaded")

    return dataset_data


# ========== Output writing ==========

def write_trec_run(results, out_path, tag="baseline"):
    """Write reranked results in TREC format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for req in results:
            qid = req.query.qid
            for rank, cand in enumerate(req.candidates, start=1):
                f.write(f"{qid} Q0 {cand.docid} {rank} {cand.score} {tag}\n")


def save_details_jsonl(records, out_path):
    """Save detailed records to JSONL."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


# ========== Evaluation ==========

def evaluate_and_save(dataset, out_path, qrels, model_name, time_cost,
                      extra_info=None, wandb_run=None):
    """Evaluate TREC run, save metrics JSON, optionally log to wandb."""
    all_metrics = Eval(out_path, qrels)
    print(f'###################### {dataset} ######################')
    print(all_metrics)
    print(f'time_cost: {time_cost:.1f}s')

    result = {
        'model_path': model_name,
        'datetime': str(datetime.datetime.now()),
        'time_cost': time_cost,
        'notes': extra_info or '',
        **all_metrics,
    }

    os.makedirs('results/', exist_ok=True)
    result_path = f'results/{dataset}.json'
    lock_path = f'{result_path}.lock'
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

    if wandb_run is not None:
        metrics = {}
        for k, v in all_metrics.items():
            metrics[f"{dataset}/{k}"] = float(v)
        metrics[f"{dataset}/time_cost_s"] = time_cost
        wandb_run.log(metrics)

    return all_metrics


def init_wandb_run(model_name, datasets, project="ReasonRank", entity=None,
                    extra_config=None):
    """Initialize a wandb run for baseline evaluation."""
    if not WANDB_AVAILABLE:
        return None
    model_short = model_name.split('/')[-1]
    config = {
        'model_path': model_name,
        'datasets': datasets,
        'reranker_type': 'baseline',
        **(extra_config or {}),
    }
    run = wandb.init(project=project, entity=entity,
                     name=f"{model_short}_baseline",
                     config=config,
                     tags=[model_short, 'baseline'])
    return run
