"""
KLE/SNNE comparison on TriviaQA using pred1 data from the JMLR project.

The pred1 files contain 500 TriviaQA questions with 6 base-model responses each.
We compute NLI similarity matrices, then evaluate KLE, SNNE, fixed-threshold SE,
and admissibility as uncertainty estimators.

Usage:
    ~/ip_llm/bin/python experiments/kle_snne_triviaqa.py
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

# Reuse functions from the ARC/MMLU comparison
sys.path.insert(0, str(Path(__file__).parent))
from kle_snne_comparison import (
    compute_similarity_matrix,
    fixed_threshold_se,
    admissibility_result,
    kernel_language_entropy,
    kle_bandwidth_sweep,
    snne_entropy,
    normalize_answer,
    check_answer,
    compute_auroc,
    THRESHOLDS,
)


PRED1_FILES = {
    "Qwen": "pred1_qwen.json",  # renamed from pred1_v2_20260109_053417.json
    "Mistral": "pred1_mistral_20260110_071956.json",
    "Llama": "pred1_llama8b_20260112_074432.json",
    "Gemma": "pred1_gemma9b_20260112_074222.json",
}


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / "modal_data"
    checkpoint_dir = script_dir / "sim_cache"
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"{'='*70}")
    print(f"  TRIVIAQA COMPARISON: Admissibility vs KLE vs SNNE")
    print(f"{'='*70}")

    all_results = []

    for model_key, filename in PRED1_FILES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"\n  SKIP {model_key}: {filepath} not found")
            continue

        print(f"\n  Processing {model_key}...")
        with open(filepath) as f:
            d = json.load(f)

        pq = d["per_question_results"]
        print(f"    {len(pq)} questions loaded")

        # Sim matrix cache
        sim_checkpoint = checkpoint_dir / f"sim_matrices_triviaqa_{model_key}.json"
        cached_sims = {}
        if sim_checkpoint.exists():
            with open(sim_checkpoint) as f:
                cached_sims = json.load(f)
            print(f"    Loaded {len(cached_sims)} cached similarity matrices")

        n_computed = 0
        t0 = time.time()

        for idx, (qid, qdata) in enumerate(pq.items()):
            gold_answers = qdata["gold_answers"]
            responses = qdata["base_responses"][:6]

            if len(responses) < 2:
                continue

            accuracies = [check_answer(r, gold_answers) for r in responses]

            # Majority vote
            normalized = [normalize_answer(r) for r in responses]
            counts = Counter(normalized)
            mv_winner = counts.most_common(1)[0][0]
            mv_idx = next(i for i, n in enumerate(normalized) if n == mv_winner)
            mv_correct = accuracies[mv_idx]

            # Similarity matrix
            if qid in cached_sims:
                sim_matrix = np.array(cached_sims[qid])
            else:
                sim_matrix = compute_similarity_matrix(responses)
                cached_sims[qid] = sim_matrix.tolist()
                n_computed += 1

                if n_computed % 50 == 0:
                    elapsed = time.time() - t0
                    rate = n_computed / elapsed
                    remaining = (len(pq) - idx - 1) / rate if rate > 0 else 0
                    print(f"    [{idx+1}/{len(pq)}] {n_computed} sim matrices computed "
                          f"({rate:.1f}/s, ~{remaining/60:.0f}m remaining)")
                    with open(sim_checkpoint, 'w') as f:
                        json.dump(cached_sims, f)

            # Compute all methods
            adm, rej, se_lo, se_hi = admissibility_result(sim_matrix)
            se_width = se_hi - se_lo
            se_085 = fixed_threshold_se(sim_matrix, 0.85)
            se_075 = fixed_threshold_se(sim_matrix, 0.75)
            se_090 = fixed_threshold_se(sim_matrix, 0.90)
            kle = kernel_language_entropy(sim_matrix)
            kle_bw = kle_bandwidth_sweep(sim_matrix, [0.1, 0.2, 0.3, 0.5, 1.0])
            snne = snne_entropy(sim_matrix, temperature=1.0)

            # Admissible-set majority vote
            adm_normalized = [normalized[i] for i in sorted(adm)]
            if adm_normalized:
                adm_counts = Counter(adm_normalized)
                adm_mv_winner = adm_counts.most_common(1)[0][0]
                adm_mv_idx = next(i for i in sorted(adm)
                                  if normalized[i] == adm_mv_winner)
                adm_mv_correct = accuracies[adm_mv_idx]
            else:
                adm_mv_correct = mv_correct

            all_results.append({
                "dataset": "triviaqa",
                "model": model_key,
                "qid": qid,
                "mv_correct": mv_correct,
                "adm_mv_correct": adm_mv_correct,
                "n_admissible": len(adm),
                "n_rejected": len(rej),
                "se_width": se_width,
                "se_085": se_085,
                "se_075": se_075,
                "se_090": se_090,
                "kle": kle,
                "kle_bw": kle_bw,
                "snne": snne,
                "has_rejection": len(rej) > 0,
                "accuracies": accuracies,
            })

        # Save sim cache
        if n_computed > 0:
            with open(sim_checkpoint, 'w') as f:
                json.dump(cached_sims, f)
            print(f"    Saved {len(cached_sims)} similarity matrices")

        elapsed = time.time() - t0
        print(f"    Done: {n_computed} new sim matrices in {elapsed:.0f}s")

    # Save results
    results_file = script_dir / "kle_snne_triviaqa_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # Print tables
    print(f"\n{'='*70}")
    print(f"  AUROC for predicting majority-vote INCORRECTNESS (TriviaQA)")
    print(f"  {'-'*65}")

    models = sorted(set(r["model"] for r in all_results))
    print(f"  {'Model':<10} {'SE(.85)':>8} {'SE(.75)':>8} {'SE(.90)':>8} "
          f"{'KLE':>8} {'SNNE':>8} {'SE width':>8}")
    print(f"  {'-'*60}")

    for model in models:
        subset = [r for r in all_results if r["model"] == model]
        labels = [1 - r["mv_correct"] for r in subset]
        if len(set(labels)) < 2:
            print(f"  {model:<10} (insufficient variation)")
            continue
        se85 = compute_auroc([r["se_085"] for r in subset], labels)
        se75 = compute_auroc([r["se_075"] for r in subset], labels)
        se90 = compute_auroc([r["se_090"] for r in subset], labels)
        kle_a = compute_auroc([r["kle"] for r in subset], labels)
        snne_a = compute_auroc([r["snne"] for r in subset], labels)
        sew_a = compute_auroc([r["se_width"] for r in subset], labels)
        print(f"  {model:<10} {se85:>8.3f} {se75:>8.3f} {se90:>8.3f} "
              f"{kle_a:>8.3f} {snne_a:>8.3f} {sew_a:>8.3f}")

    # Pooled
    labels = [1 - r["mv_correct"] for r in all_results]
    if len(set(labels)) >= 2:
        se85 = compute_auroc([r["se_085"] for r in all_results], labels)
        se75 = compute_auroc([r["se_075"] for r in all_results], labels)
        se90 = compute_auroc([r["se_090"] for r in all_results], labels)
        kle_a = compute_auroc([r["kle"] for r in all_results], labels)
        snne_a = compute_auroc([r["snne"] for r in all_results], labels)
        sew_a = compute_auroc([r["se_width"] for r in all_results], labels)
        print(f"  {'-'*60}")
        print(f"  {'Pooled':<10} {se85:>8.3f} {se75:>8.3f} {se90:>8.3f} "
              f"{kle_a:>8.3f} {snne_a:>8.3f} {sew_a:>8.3f}")

    # KLE bandwidth sensitivity
    print(f"\n  KLE BANDWIDTH SENSITIVITY (TriviaQA)")
    print(f"  {'-'*55}")
    bandwidths = [0.1, 0.2, 0.3, 0.5, 1.0]
    print(f"  {'Model':<8}", end="")
    for bw in bandwidths:
        print(f" {bw:>8}", end="")
    print(f" {'Range':>8}")
    print(f"  {'-'*55}")

    for model in models:
        subset = [r for r in all_results if r["model"] == model]
        means = []
        print(f"  {model:<8}", end="")
        for bw in bandwidths:
            vals = [r["kle_bw"].get(bw, 0) for r in subset]
            m = np.mean(vals)
            means.append(m)
            print(f" {m:>8.3f}", end="")
        print(f" {max(means)-min(means):>8.3f}")

    # Downstream: majority vote
    print(f"\n  MAJORITY VOTE: Full vs Admissible-filtered (TriviaQA)")
    print(f"  {'-'*50}")
    print(f"  {'Model':<10} {'Full MV':>8} {'Adm MV':>8} {'Delta':>8} {'% w/ rej':>8}")
    print(f"  {'-'*45}")

    for model in models:
        subset = [r for r in all_results if r["model"] == model]
        full = np.mean([r["mv_correct"] for r in subset])
        adm = np.mean([r["adm_mv_correct"] for r in subset])
        pct = np.mean([r["has_rejection"] for r in subset]) * 100
        print(f"  {model:<10} {full:>7.1%} {adm:>7.1%} {adm-full:>+7.1%} {pct:>7.1f}%")

    subset = all_results
    full = np.mean([r["mv_correct"] for r in subset])
    adm = np.mean([r["adm_mv_correct"] for r in subset])
    pct = np.mean([r["has_rejection"] for r in subset]) * 100
    print(f"  {'-'*45}")
    print(f"  {'Pooled':<10} {full:>7.1%} {adm:>7.1%} {adm-full:>+7.1%} {pct:>7.1f}%")


if __name__ == "__main__":
    main()
