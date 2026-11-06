"""
Extended selective prediction analysis.

Tries additional choice-function-derived signals beyond SE width and rejection
count, using the cached similarity matrices:

  - max_cluster_prob: maximum P(C*) observed across any threshold
  - min_cluster_prob: minimum P(C*) observed across any threshold
  - num_distinct_clusterings: number of distinct cluster structures observed
  - cluster_prob_075: cluster prob at the best single threshold (same as SE input)
  - confident_threshold: binary, 1 if some theta has max cluster >= 5/6
  - cluster_at_worst: the smallest largest-cluster observed across thresholds

Also runs:
  - Stratified AUROC by interval width quartile (is SE(0.75) less reliable
    when the interval is wide?)
  - Best combined signal via simple convex combination

Usage:
    ~/ip_llm/bin/python experiments/selective_prediction_v2.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from kle_snne_comparison import _cluster, _entropy_bits, normalize_answer, check_answer
from selective_prediction import (
    risk_coverage_curve, aurc, accuracy_at_coverage,
)

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def extract_signals_from_sim(sim_matrix):
    """Compute choice-function signals directly from similarity matrix."""
    m = sim_matrix.shape[0]
    max_probs = []
    all_probs_flat = []
    cluster_signatures = set()

    for theta in THRESHOLDS:
        clusters = _cluster(sim_matrix, theta)
        probs = [len(c) / m for c in clusters]
        max_probs.append(max(probs))
        all_probs_flat.extend(probs)
        # Signature = sorted tuple of cluster sizes
        sig = tuple(sorted([len(c) for c in clusters]))
        cluster_signatures.add(sig)

    return {
        "max_cluster_prob": max(max_probs),       # best-case confidence
        "min_cluster_prob": min(max_probs),       # worst-case confidence
        "mean_max_cluster_prob": float(np.mean(max_probs)),
        "num_distinct_clusterings": len(cluster_signatures),
        "confident_threshold": int(any(p >= 5/6 for p in max_probs)),
        "cluster_at_worst": min(max_probs),       # worst-case plurality
    }


def load_data_with_sims(dataset, model_keys, data_dir, cache_dir,
                       results_path):
    """Load selective prediction data and merge with cached sim matrices."""
    with open(results_path) as f:
        all_results = json.load(f)

    # Filter by dataset
    subset = [r for r in all_results if r["dataset"] == dataset]

    # Group by model
    by_model = defaultdict(list)
    for r in subset:
        by_model[r["model"]].append(r)

    merged = []
    for model in model_keys:
        if model not in by_model:
            continue
        model_results = by_model[model]

        # Load cached sim matrices
        cache_file = cache_dir / f"sim_matrices_{dataset}_{model}.json"
        if not cache_file.exists():
            print(f"  WARN: no sim cache for {dataset}/{model}, skipping")
            continue

        with open(cache_file) as f:
            sims = json.load(f)

        for r in model_results:
            qid = r["qid"]
            if qid not in sims:
                continue
            sim = np.array(sims[qid])
            extra = extract_signals_from_sim(sim)
            r2 = dict(r)
            r2.update(extra)
            merged.append(r2)

    return merged


def compute_all_signals(r):
    """Compute all candidate signals for one enriched result."""
    # Convert to uncertainty (higher = more uncertain)
    max_cp = r.get("max_cluster_prob", 1.0)
    min_cp = r.get("min_cluster_prob", 1.0)
    mean_max_cp = r.get("mean_max_cluster_prob", 1.0)
    n_clustering = r.get("num_distinct_clusterings", 1)
    conf_th = r.get("confident_threshold", 1)

    sigs = {
        "SE(0.75)":         r["se_075"],
        "SE(0.85)":         r["se_085"],
        "KLE":              r["kle"],
        "SE_width":         r["se_width"],
        # New signals (converted to uncertainty: higher = more uncertain)
        "1-max_cprob":      1.0 - max_cp,    # low max cluster = uncertain
        "1-min_cprob":      1.0 - min_cp,    # low min cluster = uncertain
        "1-mean_maxcp":     1.0 - mean_max_cp,
        "num_clusterings":  float(n_clustering),
        "not_confident":    float(1 - conf_th),  # 0 if any theta has big cluster
        # Combined signals
        "SE(0.75)+width":   r["se_075"] + 0.5 * r["se_width"],
        "SE(0.75)*width":   r["se_075"] * (1 + r["se_width"]),
        "SE(0.75)+1-max":   r["se_075"] + 2.0 * (1.0 - max_cp),
    }
    return sigs


def analyze(results, name):
    print(f"\n{'='*75}")
    print(f"  {name}")
    print(f"{'='*75}")

    correct = np.array([r["mv_correct"] for r in results])
    n = len(correct)
    base_acc = correct.mean()
    print(f"  n = {n}, base MV accuracy = {base_acc:.3f}")

    signal_names = [
        "SE(0.75)", "SE(0.85)", "KLE", "SE_width",
        "1-max_cprob", "1-min_cprob", "1-mean_maxcp",
        "num_clusterings", "not_confident",
        "SE(0.75)+width", "SE(0.75)*width", "SE(0.75)+1-max",
    ]
    signals = {name: [] for name in signal_names}
    for r in results:
        sigs = compute_all_signals(r)
        for k in signal_names:
            signals[k].append(sigs[k])

    print(f"\n  {'Signal':<18} {'AUROC':>7} {'AURC':>7} "
          f"{'Acc@90%':>8} {'Acc@75%':>8} {'Acc@50%':>8}")
    print(f"  {'-'*65}")

    for name in signal_names:
        scores = np.array(signals[name])
        labels = 1 - correct

        if len(set(labels)) >= 2:
            try:
                auroc = roc_auc_score(labels, scores)
            except Exception:
                auroc = float('nan')
        else:
            auroc = float('nan')

        au = aurc(scores, correct)
        a90 = accuracy_at_coverage(scores, correct, 0.90)
        a75 = accuracy_at_coverage(scores, correct, 0.75)
        a50 = accuracy_at_coverage(scores, correct, 0.50)

        print(f"  {name:<18} {auroc:>7.3f} {au:>7.3f} "
              f"{a90:>7.1%} {a75:>7.1%} {a50:>7.1%}")


def stratified_analysis(results, name):
    """Stratify by SE interval width quartile, evaluate SE(0.75) within each."""
    print(f"\n{'='*75}")
    print(f"  STRATIFIED: {name}")
    print(f"  Is SE(0.75) less reliable when the interval is wide?")
    print(f"{'='*75}")

    widths = np.array([r["se_width"] for r in results])
    se075 = np.array([r["se_075"] for r in results])
    correct = np.array([r["mv_correct"] for r in results])

    # Quartiles
    q25, q50, q75 = np.percentile(widths, [25, 50, 75])
    print(f"  SE width quartiles: Q25={q25:.3f}, Q50={q50:.3f}, Q75={q75:.3f}")

    strata = [
        ("Q1 (narrowest)", widths <= q25),
        ("Q2",            (widths > q25) & (widths <= q50)),
        ("Q3",            (widths > q50) & (widths <= q75)),
        ("Q4 (widest)",    widths > q75),
    ]

    print(f"\n  {'Stratum':<20} {'n':>5} {'base acc':>9} "
          f"{'SE(.75) AUROC':>14} {'SE(.75) AURC':>14}")
    print(f"  {'-'*70}")

    for sname, mask in strata:
        n_s = mask.sum()
        if n_s < 10:
            continue
        s_correct = correct[mask]
        s_scores = se075[mask]
        s_base = s_correct.mean()
        labels = 1 - s_correct
        if len(set(labels)) >= 2:
            try:
                auroc = roc_auc_score(labels, s_scores)
            except Exception:
                auroc = float('nan')
        else:
            auroc = float('nan')
        au = aurc(s_scores, s_correct)
        print(f"  {sname:<20} {n_s:>5} {s_base:>8.1%} "
              f"{auroc:>14.3f} {au:>14.3f}")


def width_as_uncertainty_in_uncertainty(results, name):
    """Test if interval width is a useful 'trust SE' signal.

    Hypothesis: when width is wide, SE(0.75) may be unreliable. Test by
    splitting into narrow-width and wide-width halves and comparing
    SE(0.75)'s selective prediction performance in each.
    """
    print(f"\n{'='*75}")
    print(f"  WIDTH-CONDITIONAL: {name}")
    print(f"  Does 'use SE(0.75), abstain when width>median' work?")
    print(f"{'='*75}")

    widths = np.array([r["se_width"] for r in results])
    se075 = np.array([r["se_075"] for r in results])
    correct = np.array([r["mv_correct"] for r in results])
    n = len(correct)
    med_width = np.median(widths)

    # Narrow-width mask
    narrow = widths <= med_width
    n_narrow = narrow.sum()

    # Strategy: answer top-X% most confident questions as ranked by
    # combined signal: wide-width questions go to "abstain" pile, narrow-width
    # sorted by SE(0.75).
    # Equivalent: create a composite score where wide-width gets +inf.

    # Simple composite: SE(0.75) + huge penalty if width > median
    composite = se075.copy()
    composite[~narrow] += 10.0  # push wide to abstain first

    cov, risks, acc = risk_coverage_curve(composite, correct)
    au_comp = aurc(composite, correct)

    # Compare to plain SE(0.75)
    au_se = aurc(se075, correct)

    # Accuracies at coverage levels (limited by narrow-width set size)
    max_cov = n_narrow / n
    print(f"  Narrow-width fraction: {max_cov:.1%}")
    print(f"  Plain SE(0.75) AURC:       {au_se:.3f}")
    print(f"  Width-gated SE(0.75) AURC: {au_comp:.3f}")
    for c in [0.25, 0.50, 0.75]:
        if c <= max_cov:
            a_plain = accuracy_at_coverage(se075, correct, c)
            a_comp = accuracy_at_coverage(composite, correct, c)
            print(f"  Acc@{int(c*100)}%: plain={a_plain:.1%}, width-gated={a_comp:.1%}, "
                  f"delta={a_comp - a_plain:+.1%}")


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / "modal_data"
    cache_dir = script_dir / "sim_cache"

    models = ["Qwen", "Mistral", "Llama", "Gemma"]

    print("Loading enriched data with cached similarity matrices...")
    triviaqa = load_data_with_sims(
        "triviaqa", models, data_dir, cache_dir,
        script_dir / "kle_snne_triviaqa_results.json",
    )
    arc = load_data_with_sims(
        "arc", models, data_dir, cache_dir,
        script_dir / "kle_snne_results.json",
    )

    print(f"\nLoaded {len(triviaqa)} TriviaQA and {len(arc)} ARC questions")

    analyze(triviaqa, "TRIVIAQA — EXTENDED SIGNALS")
    analyze(arc, "ARC — EXTENDED SIGNALS")

    stratified_analysis(triviaqa, "TRIVIAQA")
    stratified_analysis(arc, "ARC")

    width_as_uncertainty_in_uncertainty(triviaqa, "TRIVIAQA")
    width_as_uncertainty_in_uncertainty(arc, "ARC")


if __name__ == "__main__":
    main()
