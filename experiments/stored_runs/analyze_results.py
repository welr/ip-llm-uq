"""
Analyze pre-registered experiment results.

Computes:
- Per-model and pooled accuracy ratios with bootstrap 95% CIs
- Fisher's exact tests
- Heuristic filter comparisons
- SE interval width vs. difficulty correlation
- Per-subject breakdown (MMLU only)

Usage:
    ~/ip_llm/bin/python analyze_results.py results_dir/
"""

import json
import sys
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load_results(results_dir: str, dataset: str) -> dict:
    """Load per-model result files for a dataset."""
    model_results = {}
    for f in Path(results_dir).glob(f"prereg_{dataset}_*.json"):
        if "aggregate" in f.name:
            continue
        with open(f) as fh:
            data = json.load(fh)
        model = data["summary"]["model"]
        model_results[model] = data
    return model_results


def bootstrap_ratio(adm_correct, adm_total, rej_correct, rej_total,
                    n_boot=10000, seed=42):
    """Bootstrap 95% CI for accuracy ratio."""
    rng = np.random.RandomState(seed)

    adm_arr = np.array([1]*adm_correct + [0]*(adm_total - adm_correct))
    rej_arr = np.array([1]*rej_correct + [0]*(rej_total - rej_correct))

    ratios = []
    for _ in range(n_boot):
        adm_sample = rng.choice(adm_arr, size=adm_total, replace=True)
        rej_sample = rng.choice(rej_arr, size=max(rej_total, 1), replace=True)
        adm_acc = adm_sample.mean()
        rej_acc = rej_sample.mean()
        if rej_acc > 0:
            ratios.append(adm_acc / rej_acc)
        else:
            ratios.append(np.inf)

    finite_ratios = [r for r in ratios if np.isfinite(r)]
    if len(finite_ratios) < n_boot * 0.5:
        return float('inf'), float('inf')
    return np.percentile(finite_ratios, 2.5), np.percentile(finite_ratios, 97.5)


def analyze_dataset(model_results: dict, dataset: str):
    """Full analysis for one dataset."""

    print(f"\n{'='*70}")
    print(f"  PRE-REGISTERED ANALYSIS: {dataset.upper()}")
    print(f"{'='*70}")

    # Aggregate across models
    total = {"nli_adm_c": 0, "nli_adm_t": 0, "nli_rej_c": 0, "nli_rej_t": 0,
             "str_adm_c": 0, "str_adm_t": 0, "str_rej_c": 0, "str_rej_t": 0,
             "freq_ret_c": 0, "freq_ret_t": 0, "freq_rej_c": 0, "freq_rej_t": 0,
             "cons_ret_c": 0, "cons_ret_t": 0, "cons_rej_c": 0, "cons_rej_t": 0,
             "len_ret_c": 0, "len_ret_t": 0, "len_rej_c": 0, "len_rej_t": 0}

    se_widths = []
    question_accs = []

    per_model = {}
    for model_key, data in model_results.items():
        pq = data["per_question"]
        m_adm_c = m_adm_t = m_rej_c = m_rej_t = 0

        for qid, r in pq.items():
            accs = r["accuracies"]
            nli_adm = set(r["nli_admissible"])
            nli_rej = set(r["nli_rejected"])
            str_adm = set(r["str_admissible"])
            str_rej = set(r["str_rejected"])
            freq_rej = set(r["freq_rejected"])
            cons_rej = set(r["cons_rejected"])
            len_rej = set(r["len_rejected"])

            for i in nli_adm:
                total["nli_adm_c"] += accs[i]; m_adm_c += accs[i]
                total["nli_adm_t"] += 1; m_adm_t += 1
            for i in nli_rej:
                total["nli_rej_c"] += accs[i]; m_rej_c += accs[i]
                total["nli_rej_t"] += 1; m_rej_t += 1
            for i in str_adm:
                total["str_adm_c"] += accs[i]
                total["str_adm_t"] += 1
            for i in str_rej:
                total["str_rej_c"] += accs[i]
                total["str_rej_t"] += 1
            n = len(accs)
            for i in range(n):
                if i not in freq_rej:
                    total["freq_ret_c"] += accs[i]; total["freq_ret_t"] += 1
                else:
                    total["freq_rej_c"] += accs[i]; total["freq_rej_t"] += 1
                if i not in cons_rej:
                    total["cons_ret_c"] += accs[i]; total["cons_ret_t"] += 1
                else:
                    total["cons_rej_c"] += accs[i]; total["cons_rej_t"] += 1
                if i not in len_rej:
                    total["len_ret_c"] += accs[i]; total["len_ret_t"] += 1
                else:
                    total["len_rej_c"] += accs[i]; total["len_rej_t"] += 1

            se_widths.append(r["se_upper"] - r["se_lower"])
            question_accs.append(np.mean(accs))

        # Per-model stats
        m_adm_acc = m_adm_c / max(m_adm_t, 1)
        m_rej_acc = m_rej_c / max(m_rej_t, 1)
        m_ratio = m_adm_acc / max(m_rej_acc, 0.001) if m_rej_t > 0 else float('inf')

        if m_rej_t > 0:
            table = [[m_adm_c, m_adm_t - m_adm_c],
                     [m_rej_c, m_rej_t - m_rej_c]]
            _, p = stats.fisher_exact(table, alternative='two-sided')
        else:
            p = None

        per_model[model_key] = {
            "adm_acc": m_adm_acc, "rej_acc": m_rej_acc, "ratio": m_ratio,
            "n_adm": m_adm_t, "n_rej": m_rej_t, "p": p,
        }

    # ---- H1a/H2a: Primary hypothesis ----
    print(f"\n  H1a/H2a: Admissible acc > Rejected acc")
    print(f"  {'-'*60}")
    print(f"  {'Model':<10} {'Adm Acc':>8} {'Rej Acc':>8} {'Ratio':>8} "
          f"{'n_rej':>6} {'p-value':>10}")
    print(f"  {'-'*60}")

    models_with_ratio_gt_1 = 0
    for model_key, pm in per_model.items():
        ratio_str = f"{pm['ratio']:.2f}x" if pm['ratio'] < 100 else "inf"
        p_str = f"{pm['p']:.6f}" if pm['p'] is not None else "N/A"
        print(f"  {model_key:<10} {pm['adm_acc']:>7.1%} {pm['rej_acc']:>7.1%} "
              f"{ratio_str:>8} {pm['n_rej']:>6} {p_str:>10}")
        if pm['ratio'] > 1.0:
            models_with_ratio_gt_1 += 1

    # Pooled
    pool_adm_acc = total["nli_adm_c"] / max(total["nli_adm_t"], 1)
    pool_rej_acc = total["nli_rej_c"] / max(total["nli_rej_t"], 1)
    pool_ratio = pool_adm_acc / max(pool_rej_acc, 0.001)

    if total["nli_rej_t"] > 0:
        table = [[total["nli_adm_c"], total["nli_adm_t"] - total["nli_adm_c"]],
                 [total["nli_rej_c"], total["nli_rej_t"] - total["nli_rej_c"]]]
        _, pool_p = stats.fisher_exact(table, alternative='two-sided')
    else:
        pool_p = None

    print(f"  {'-'*60}")
    p_str = f"{pool_p:.6f}" if pool_p is not None else "N/A"
    print(f"  {'Pooled':<10} {pool_adm_acc:>7.1%} {pool_rej_acc:>7.1%} "
          f"{pool_ratio:>7.2f}x {total['nli_rej_t']:>6} {p_str:>10}")

    # Bootstrap CI
    if total["nli_rej_t"] >= 30:
        ci_lo, ci_hi = bootstrap_ratio(
            total["nli_adm_c"], total["nli_adm_t"],
            total["nli_rej_c"], total["nli_rej_t"],
        )
        print(f"  Bootstrap 95% CI for ratio: [{ci_lo:.2f}, {ci_hi:.2f}]")
    else:
        print(f"  WARNING: Only {total['nli_rej_t']} rejected responses -- underpowered")

    # Verdict
    h1a = pool_ratio > 1.0 and pool_p is not None and pool_p < 0.05
    h1b = total["nli_rej_t"] >= 30 and ci_lo >= 1.5 if total["nli_rej_t"] >= 30 else False
    h1c = models_with_ratio_gt_1 >= 3
    print(f"\n  H1a (ratio > 1.0, p < 0.05): {'SUPPORTED' if h1a else 'NOT SUPPORTED'}")
    print(f"  H1b (CI lower >= 1.5):        {'SUPPORTED' if h1b else 'NOT SUPPORTED'}")
    print(f"  H1c (>= 3/4 models > 1.0):    {'SUPPORTED' if h1c else 'NOT SUPPORTED'} "
          f"({models_with_ratio_gt_1}/4)")

    # ---- H3: Heuristic comparison ----
    print(f"\n  H3: Admissibility vs. heuristic filters (string-based)")
    print(f"  {'-'*60}")

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    str_adm_acc = safe_div(total["str_adm_c"], total["str_adm_t"])
    str_rej_acc = safe_div(total["str_rej_c"], total["str_rej_t"])
    str_ratio = safe_div(str_adm_acc, str_rej_acc) if total["str_rej_t"] > 0 else float('inf')

    freq_ret_acc = safe_div(total["freq_ret_c"], total["freq_ret_t"])
    freq_rej_acc = safe_div(total["freq_rej_c"], total["freq_rej_t"])
    freq_ratio = safe_div(freq_ret_acc, freq_rej_acc) if total["freq_rej_t"] > 0 else float('inf')

    cons_ret_acc = safe_div(total["cons_ret_c"], total["cons_ret_t"])
    cons_rej_acc = safe_div(total["cons_rej_c"], total["cons_rej_t"])
    cons_ratio = safe_div(cons_ret_acc, cons_rej_acc) if total["cons_rej_t"] > 0 else float('inf')

    n_total = total["nli_adm_t"] + total["nli_rej_t"]
    str_pct = 100 * total["str_rej_t"] / max(n_total, 1)
    freq_pct = 100 * total["freq_rej_t"] / max(n_total, 1)
    cons_pct = 100 * total["cons_rej_t"] / max(n_total, 1)

    print(f"  {'Filter':<16} {'Ratio':>8} {'% Rejected':>10}")
    print(f"  {'-'*40}")
    print(f"  {'Admissibility':<16} {str_ratio:>7.2f}x {str_pct:>9.1f}%")
    print(f"  {'Frequency':<16} {freq_ratio:>7.2f}x {freq_pct:>9.1f}%")
    print(f"  {'Consistency':<16} {cons_ratio:>7.2f}x {cons_pct:>9.1f}%")

    h3 = str_ratio > max(freq_ratio, cons_ratio)
    print(f"\n  H3: {'SUPPORTED' if h3 else 'NOT SUPPORTED'}")

    # ---- H4: SE interval width vs difficulty ----
    if se_widths and question_accs:
        difficulties = [1 - a for a in question_accs]
        rho, p_val = stats.spearmanr(se_widths, difficulties)
        print(f"\n  H4: SE interval width vs. difficulty")
        print(f"  Spearman rho = {rho:.3f}, p = {p_val:.6f}")
        h4 = rho > 0 and p_val < 0.05
        print(f"  H4: {'SUPPORTED' if h4 else 'NOT SUPPORTED'}")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {dataset.upper()}")
    print(f"{'='*70}")
    print(f"  H1a: {'PASS' if h1a else 'FAIL'}")
    print(f"  H1b: {'PASS' if h1b else 'FAIL'}")
    print(f"  H1c: {'PASS' if h1c else 'FAIL'}")
    print(f"  H3:  {'PASS' if h3 else 'FAIL'}")
    if se_widths:
        print(f"  H4:  {'PASS' if h4 else 'FAIL'}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_dir> [dataset]")
        print("  dataset: mmlu or arc (default: both)")
        sys.exit(1)

    results_dir = sys.argv[1]
    datasets = [sys.argv[2]] if len(sys.argv) > 2 else ["mmlu", "arc"]

    for dataset in datasets:
        model_results = load_results(results_dir, dataset)
        if not model_results:
            print(f"No results found for {dataset} in {results_dir}")
            continue
        analyze_dataset(model_results, dataset)


if __name__ == "__main__":
    main()
