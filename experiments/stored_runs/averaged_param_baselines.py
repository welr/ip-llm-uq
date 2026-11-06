"""
Averaged-parameter baselines: does the choice-function structure deliver value
beyond generic parameter aggregation?

For each question we compute:
  - SE_avg: mean of SE(theta) over theta in {0.50, 0.55, ..., 0.95}
  - KLE_avg: mean of KLE over Gaussian bandwidths {0.1, 0.2, 0.3, 0.5, 1.0}
  - 1-maxCP: 1 - max over theta of max cluster probability (parameter-free choice
    function signal; Section 4.6 of the paper)

and evaluate each by AUROC for predicting majority-vote incorrectness, on the
same cached similarity matrices used by kle_snne_comparison.py.

Reviewer's concern (NeurIPS W5): if averaged SE / averaged KLE matches
1-maxCP, then the 'union of per-threshold argmax clusters' structure is doing
less work than claimed. This script tests that.

Output: experiments/averaged_param_baselines_results.json, plus a printed
per-dataset table suitable for copy-paste into the appendix.

Usage:
    ~/ip_llm/bin/python experiments/averaged_param_baselines.py
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from kle_snne_comparison import (
    THRESHOLDS, _entropy_bits, _cluster, fixed_threshold_se,
    kle_bandwidth_sweep, kernel_language_entropy,
)


BANDWIDTHS = [0.1, 0.2, 0.3, 0.5, 1.0]


def compute_signals(sim_matrix):
    """Return dict of all per-question signals derivable from a sim matrix."""
    m = sim_matrix.shape[0]

    # SE at each threshold
    se_by_theta = {}
    max_cluster_probs = []
    for theta in THRESHOLDS:
        clusters = _cluster(sim_matrix, theta)
        probs = [len(c) / m for c in clusters]
        se_by_theta[theta] = _entropy_bits(probs)
        max_cluster_probs.append(max(probs))

    # Averaged SE and SE at best fixed (best = max AUROC, computed over dataset later
    # but here we also output SE(0.75) for the paper's reference signal).
    se_avg = float(np.mean(list(se_by_theta.values())))

    # KLE averaged over bandwidths
    kle_bw = kle_bandwidth_sweep(sim_matrix, BANDWIDTHS)
    kle_avg = float(np.mean(list(kle_bw.values())))
    kle_direct = kernel_language_entropy(sim_matrix)  # paper's default KLE

    # 1 - maxCP
    one_minus_maxcp = 1.0 - max(max_cluster_probs)

    return {
        "se_avg": se_avg,
        "se_075": se_by_theta[0.75],
        "se_085": se_by_theta[0.85],
        "se_090": se_by_theta[0.90],
        "kle": kle_direct,
        "kle_avg": kle_avg,
        "kle_bw": {str(b): float(v) for b, v in kle_bw.items()},
        "one_minus_maxcp": one_minus_maxcp,
    }


def auroc(scores, labels):
    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def analyze(dataset, model_keys, cached_results, sim_cache_dir):
    """Pool per-question signals from sim cache and compute AUROCs."""
    results_by_model = {}
    per_q = []

    for model in model_keys:
        sim_file = sim_cache_dir / f"sim_matrices_{dataset}_{model}.json"
        if not sim_file.exists():
            print(f"  SKIP {dataset}/{model}: no cached sim matrices at {sim_file}")
            continue

        with open(sim_file) as f:
            sims = json.load(f)

        # Cross-reference to cached_results for mv_correct labels
        model_records = [r for r in cached_results
                         if r.get("dataset") == dataset and r.get("model") == model]
        qid_to_record = {r["qid"]: r for r in model_records}

        matched = 0
        for qid, sim_list in sims.items():
            if qid not in qid_to_record:
                continue
            sim_matrix = np.array(sim_list)
            signals = compute_signals(sim_matrix)
            per_q.append({
                "dataset": dataset,
                "model": model,
                "qid": qid,
                "mv_correct": qid_to_record[qid]["mv_correct"],
                **signals,
            })
            matched += 1
        print(f"  {dataset}/{model}: {matched}/{len(sims)} questions matched in cache")

    return per_q


def print_table(per_q, datasets, signals_to_show):
    """Print AUROC table per dataset, pooled across models."""
    print(f"\n{'='*78}")
    print(f"  AUROC for majority-vote incorrectness (pooled across models per dataset)")
    print(f"{'='*78}")
    hdr = f"  {'Dataset':<12}" + "".join(f" {s:>14}" for s in signals_to_show)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for d in datasets:
        subset = [r for r in per_q if r["dataset"] == d]
        if not subset:
            continue
        labels = [1 - r["mv_correct"] for r in subset]
        row = f"  {d:<12}"
        for s in signals_to_show:
            a = auroc([r[s] for r in subset], labels)
            row += f" {a:>14.4f}"
        row += f"   (n={len(subset)})"
        print(row)

    print(f"\n{'='*78}")
    print(f"  Per-model breakdown")
    print(f"{'='*78}")
    for d in datasets:
        print(f"\n  {d.upper()}")
        hdr = f"  {'Model':<10}" + "".join(f" {s:>14}" for s in signals_to_show)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        models = sorted(set(r["model"] for r in per_q if r["dataset"] == d))
        for m in models:
            subset = [r for r in per_q if r["dataset"] == d and r["model"] == m]
            labels = [1 - r["mv_correct"] for r in subset]
            if len(set(labels)) < 2:
                print(f"  {m:<10}   (all same label — AUROC undefined)")
                continue
            row = f"  {m:<10}"
            for s in signals_to_show:
                a = auroc([r[s] for r in subset], labels)
                row += f" {a:>14.4f}"
            print(row)


def main():
    script_dir = Path(__file__).parent
    sim_cache_dir = script_dir / "sim_cache"

    # Load existing per-question records (to get mv_correct labels)
    cached = []
    for f in ["kle_snne_results.json", "kle_snne_triviaqa_results.json"]:
        path = script_dir / f
        if path.exists():
            cached.extend(json.load(open(path)))
    print(f"Loaded {len(cached)} cached per-question records")

    model_keys = ["Qwen", "Mistral", "Llama", "Gemma"]

    all_per_q = []
    for ds in ("triviaqa", "arc"):
        print(f"\n--- {ds.upper()} ---")
        all_per_q.extend(analyze(ds, model_keys, cached, sim_cache_dir))
    # MMLU: only Gemma has checkpoints
    print(f"\n--- MMLU ---")
    all_per_q.extend(analyze("mmlu", ["Gemma"], cached, sim_cache_dir))

    print(f"\nTotal per-question records: {len(all_per_q)}")

    signals = ["se_075", "se_085", "se_avg", "kle", "kle_avg", "one_minus_maxcp"]
    print_table(all_per_q, ["triviaqa", "arc", "mmlu"], signals)

    # Save
    out = script_dir / "averaged_param_baselines_results.json"
    with open(out, "w") as f:
        json.dump(all_per_q, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
