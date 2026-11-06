"""
Grid sparsity diagnostic: how many of the 10 thresholds in Theta actually
contribute unique admissible responses per question?

Reviewer question Q6 (NeurIPS): if only 1-2 thresholds do the work, the
'union over Theta' framing is cosmetic.

We report three metrics per question, aggregated by dataset:

  num_distinct_Ostar : number of distinct argmax-cluster sets O_theta*
                      observed across theta (upper bound of 10)
  num_essential     : number of thresholds theta such that some r in O_theta*
                      is in NO other O_{theta'}*  (strictly necessary)
  num_sweep_contrib : order-dependent count -- thresholds (low-to-high) that
                      expand the running union of O_theta*

Reuses cached similarity matrices from experiments/sim_cache/. No new NLI
calls; pure post-hoc analysis.

Usage:
    ~/ip_llm/bin/python experiments/grid_sparsity.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from kle_snne_comparison import THRESHOLDS, _cluster


def argmax_cluster_set(sim_matrix, theta):
    """Set of response indices in any max-probability cluster at theta."""
    m = sim_matrix.shape[0]
    clusters = _cluster(sim_matrix, theta)
    sizes = [len(c) for c in clusters]
    max_size = max(sizes)
    out = set()
    for c, s in zip(clusters, sizes):
        if s == max_size:
            out |= c
    return frozenset(out)


def per_question_metrics(sim_matrix):
    m = sim_matrix.shape[0]
    O_sets = [argmax_cluster_set(sim_matrix, theta) for theta in THRESHOLDS]
    full_union = set().union(*O_sets)

    # 1. Distinct O_theta* sets
    num_distinct = len(set(O_sets))

    # 2. Essential thresholds: theta is essential iff some r in O_theta* is in
    #    no other O_{theta'}*.
    essential = 0
    for i, O in enumerate(O_sets):
        others = set().union(*(Oj for j, Oj in enumerate(O_sets) if j != i))
        exclusive = O - others
        if exclusive:
            essential += 1

    # 3. Sweep contribution: low-to-high, count thresholds that grow the running union
    sweep = 0
    running = set()
    for O in O_sets:
        before = len(running)
        running |= set(O)
        if len(running) > before:
            sweep += 1

    return {
        "num_distinct_Ostar": num_distinct,
        "num_essential": essential,
        "num_sweep_contrib": sweep,
        "size_Csem": len(full_union),
        "M": m,
    }


def analyze_dataset(dataset, model_keys, sim_cache_dir):
    per_q = []
    for model in model_keys:
        sim_file = sim_cache_dir / f"sim_matrices_{dataset}_{model}.json"
        if not sim_file.exists():
            print(f"  SKIP {dataset}/{model}: no cached sim matrices")
            continue
        with open(sim_file) as f:
            sims = json.load(f)
        for qid, sim_list in sims.items():
            sim_matrix = np.array(sim_list)
            m = per_question_metrics(sim_matrix)
            per_q.append({"dataset": dataset, "model": model, "qid": qid, **m})
        print(f"  {dataset}/{model}: {len(sims)} questions")
    return per_q


def summarize(per_q, datasets):
    print(f"\n{'='*78}")
    print("  GRID SPARSITY METRICS (pooled across models per dataset)")
    print(f"{'='*78}\n")

    for d in datasets:
        subset = [r for r in per_q if r["dataset"] == d]
        if not subset:
            continue
        print(f"  === {d.upper()}  (n={len(subset)} questions, M=6) ===")
        for metric in ("num_distinct_Ostar", "num_essential", "num_sweep_contrib"):
            vals = [r[metric] for r in subset]
            arr = np.array(vals)
            hist = Counter(vals)
            hist_str = " ".join(f"{k}:{hist[k]}" for k in sorted(hist))
            print(f"    {metric:22s}  mean={arr.mean():.2f}  median={np.median(arr):.1f}  "
                  f"p10={np.percentile(arr,10):.0f}  p90={np.percentile(arr,90):.0f}")
            print(f"    {' '*22}  histogram (value:count)  {hist_str}")
        # Also: size of C_sem
        csem_sizes = [r["size_Csem"] for r in subset]
        size_hist = Counter(csem_sizes)
        print(f"    {'size_Csem (of M=6)':22s}  mean={np.mean(csem_sizes):.2f}  "
              f"histogram  " + " ".join(f"{k}:{size_hist[k]}" for k in sorted(size_hist)))
        print()


def main():
    script_dir = Path(__file__).parent
    sim_cache_dir = script_dir / "sim_cache"

    all_per_q = []
    for ds in ("triviaqa", "arc"):
        print(f"--- {ds.upper()} ---")
        all_per_q.extend(analyze_dataset(ds, ["Qwen", "Mistral", "Llama", "Gemma"], sim_cache_dir))
    print("--- MMLU ---")
    all_per_q.extend(analyze_dataset("mmlu", ["Gemma"], sim_cache_dir))

    summarize(all_per_q, ["triviaqa", "arc", "mmlu"])

    out = script_dir / "grid_sparsity_results.json"
    with open(out, "w") as f:
        json.dump(all_per_q, f, default=str)
    print(f"\nRaw per-question data saved to {out}")


if __name__ == "__main__":
    main()
