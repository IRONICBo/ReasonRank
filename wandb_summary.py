"""
Collect results from all datasets and log a unified summary to wandb.
Called after all parallel GPU instances finish.
"""
import argparse
import json
import os
import numpy as np

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Run `pip install wandb`.")
    exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_mode", type=str, default="rank_GPT_reasoning")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="ReasonRank")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    model_short = args.model_path.split("/")[-1]

    # Collect per-dataset results
    all_results = {}
    missing = []
    for dataset in args.datasets:
        result_path = os.path.join(args.results_dir, f"{dataset}.json")
        if not os.path.exists(result_path):
            missing.append(dataset)
            continue
        try:
            with open(result_path, "r") as f:
                records = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"WARNING: Failed to parse {result_path}: {e}, skipping")
            missing.append(dataset)
            continue
        # Take the latest record matching this model
        matched = [r for r in records if r.get("model_path") == args.model_path]
        if not matched:
            matched = records  # fallback: take latest regardless
        if matched:
            all_results[dataset] = matched[-1]

    if missing:
        print(f"WARNING: Missing results for datasets: {missing}")

    if not all_results:
        print("ERROR: No results found. Nothing to log.")
        return

    # Init wandb summary run
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{model_short}_summary_{args.num_gpus}gpu",
        config={
            "model_path": args.model_path,
            "prompt_mode": args.prompt_mode,
            "num_gpus": args.num_gpus,
            "tp_per_instance": args.tp,
            "num_instances": args.num_gpus // args.tp,
            "datasets": args.datasets,
            "num_datasets": len(args.datasets),
        },
        tags=[model_short, "summary", f"{args.num_gpus}gpu"],
    )

    # Log per-dataset metrics
    ndcg1_list, ndcg5_list, ndcg10_list = [], [], []
    total_time, total_input_tokens, total_output_tokens = 0, 0, 0
    all_cot_words, all_answer_words = [], []

    columns = ["dataset", "NDCG@1", "NDCG@5", "NDCG@10", "time_cost_s",
               "input_tokens", "output_tokens", "avg_input_tok", "avg_output_tok",
               "avg_cot_words", "max_cot_words", "avg_answer_words"]
    table = wandb.Table(columns=columns)

    for dataset, result in all_results.items():
        ndcg1 = float(result.get("NDCG@1", 0))
        ndcg5 = float(result.get("NDCG@5", 0))
        ndcg10 = float(result.get("NDCG@10", 0))
        t_cost = result.get("time_cost", 0)
        in_tok = result.get("total_input_tokens", 0)
        out_tok = result.get("total_output_tokens", 0)
        avg_in = result.get("avg_input_tokens", 0)
        avg_out = result.get("avg_output_tokens", 0)
        avg_cot = result.get("avg_cot_words", 0)
        max_cot = result.get("max_cot_words", 0)
        avg_ans = result.get("avg_answer_words", 0)

        # Per-dataset metrics
        run.log({
            f"{dataset}/NDCG@1": ndcg1,
            f"{dataset}/NDCG@5": ndcg5,
            f"{dataset}/NDCG@10": ndcg10,
            f"{dataset}/time_cost_s": t_cost,
            f"{dataset}/input_tokens": in_tok,
            f"{dataset}/output_tokens": out_tok,
            f"{dataset}/avg_input_tokens": avg_in,
            f"{dataset}/avg_output_tokens": avg_out,
            f"{dataset}/avg_cot_words": avg_cot,
            f"{dataset}/max_cot_words": max_cot,
            f"{dataset}/avg_answer_words": avg_ans,
        })

        table.add_data(dataset, ndcg1, ndcg5, ndcg10, t_cost, in_tok, out_tok,
                       avg_in, avg_out, avg_cot, max_cot, avg_ans)
        ndcg1_list.append(ndcg1)
        ndcg5_list.append(ndcg5)
        ndcg10_list.append(ndcg10)
        total_time += t_cost
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        if avg_cot > 0:
            all_cot_words.append(avg_cot)
            all_answer_words.append(avg_ans)

        print(f"  {dataset}: NDCG@1={ndcg1} NDCG@5={ndcg5} NDCG@10={ndcg10} time={t_cost:.1f}s cot={avg_cot:.0f}w ans={avg_ans:.0f}w")

    # Log table
    run.log({"results_table": table})

    # Summary metrics
    summary = {
        "summary/avg_NDCG@1": np.mean(ndcg1_list),
        "summary/avg_NDCG@5": np.mean(ndcg5_list),
        "summary/avg_NDCG@10": np.mean(ndcg10_list),
        "summary/total_time_s": total_time,
        "summary/total_input_tokens": total_input_tokens,
        "summary/total_output_tokens": total_output_tokens,
        "summary/total_tokens": total_input_tokens + total_output_tokens,
        "summary/num_datasets_completed": len(all_results),
        "summary/avg_cot_words": np.mean(all_cot_words) if all_cot_words else 0,
        "summary/avg_answer_words": np.mean(all_answer_words) if all_answer_words else 0,
    }
    run.log(summary)
    for k, v in summary.items():
        run.summary[k] = v

    print(f"\n  avg NDCG@1={summary['summary/avg_NDCG@1']:.2f}  "
          f"NDCG@5={summary['summary/avg_NDCG@5']:.2f}  "
          f"NDCG@10={summary['summary/avg_NDCG@10']:.2f}")
    print(f"  total time={total_time:.1f}s  total tokens={total_input_tokens + total_output_tokens}")

    run.finish()
    print(f"\nwandb summary run: {run.url}")


if __name__ == "__main__":
    main()
