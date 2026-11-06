"""
Selective prediction / abstention analysis.

For each question, compute a confidence score, sort questions by confidence,
and measure majority-vote accuracy as a function of coverage (the fraction
of questions we choose to answer). This is the standard selective prediction
evaluation: does an uncertainty signal help us abstain on the hard questions?

Compares multiple signals:
  - SE at fixed thresholds (0.75, 0.85, 0.90)
  - KLE
  - SE width (choice function native)
  - Number of rejected responses
  - Binary: any rejections
  - SE within admissible set at 0.75
  - Combined: SE(0.75) + rejection-aware adjustment

Reports:
  - AUROC (for reference)
  - AURC (area under risk-coverage curve, lower = better)
  - Accuracy at 50%, 75%, 90% coverage
  - Cross-domain transfer: validated on one dataset, tested on another

Usage:
    ~/ip_llm/bin/python experiments/selective_prediction.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_signals(result):
    """Compute all candidate confidence signals for one question."""
    se_width = result["se_width"]
    se_075 = result["se_075"]
    se_085 = result["se_085"]
    se_090 = result["se_090"]
    kle = result["kle"]
    n_rejected = result["n_rejected"]
    n_admissible = result["n_admissible"]
    accs = result["accuracies"]

    # SE within the admissible set requires knowing which responses are admissible,
    # which we don't store explicitly -- but we do have the counts. Instead,
    # we use the ratio of admissible to total as a proxy for "cluster concentration".
    adm_ratio = n_admissible / (n_admissible + n_rejected) if (n_admissible + n_rejected) > 0 else 1.0

    signals = {
        "SE(0.75)": se_075,
        "SE(0.85)": se_085,
        "SE(0.90)": se_090,
        "KLE": kle,
        "SE_width": se_width,
        "n_rejected": -n_rejected,  # fewer rejections = more confident, so negate
        "has_rejection": -float(n_rejected > 0),
        "SE(0.75)+rej": se_075 + 0.5 * float(n_rejected > 0),  # combined signal
    }
    return signals


def risk_coverage_curve(scores, correct):
    """Compute risk-coverage curve.

    scores: higher = more uncertain (sort ascending to answer confident first)
    correct: 1 if majority vote was correct, 0 otherwise

    Returns (coverages, risks, accuracies) arrays.
    """
    n = len(scores)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    # Sort by confidence (ascending uncertainty score)
    order = np.argsort(scores)
    sorted_correct = np.array(correct)[order]

    coverages = []
    accuracies = []
    cum_correct = 0
    for i in range(n):
        cum_correct += sorted_correct[i]
        coverage = (i + 1) / n
        acc = cum_correct / (i + 1)
        coverages.append(coverage)
        accuracies.append(acc)

    coverages = np.array(coverages)
    accuracies = np.array(accuracies)
    risks = 1.0 - accuracies
    return coverages, risks, accuracies


def aurc(scores, correct):
    """Area under risk-coverage curve (lower is better)."""
    cov, risks, _ = risk_coverage_curve(scores, correct)
    if len(cov) == 0:
        return float('nan')
    # Trapezoidal integration
    return np.trapezoid(risks, cov)


def accuracy_at_coverage(scores, correct, target_coverage):
    """Accuracy achieved when answering the top target_coverage fraction."""
    cov, _, acc = risk_coverage_curve(scores, correct)
    if len(cov) == 0:
        return float('nan')
    idx = np.searchsorted(cov, target_coverage, side='right') - 1
    idx = max(0, min(idx, len(acc) - 1))
    return acc[idx]


def analyze_dataset(results, dataset_name):
    """Full selective prediction analysis for one dataset."""
    print(f"\n{'='*75}")
    print(f"  SELECTIVE PREDICTION: {dataset_name.upper()}")
    print(f"{'='*75}")

    if not results:
        print("  No data")
        return

    # Build signal arrays
    correct = np.array([r["mv_correct"] for r in results])
    n_total = len(correct)
    base_acc = correct.mean()
    print(f"  n = {n_total}, base majority-vote accuracy = {base_acc:.3f}")

    signals_by_name = defaultdict(list)
    for r in results:
        sigs = compute_signals(r)
        for k, v in sigs.items():
            signals_by_name[k].append(v)

    signal_names = [
        "SE(0.75)", "SE(0.85)", "SE(0.90)", "KLE",
        "SE_width", "n_rejected", "has_rejection", "SE(0.75)+rej",
    ]

    print(f"\n  {'Signal':<16} {'AUROC':>7} {'AURC':>7} "
          f"{'Acc@90%':>8} {'Acc@75%':>8} {'Acc@50%':>8}")
    print(f"  {'-'*65}")

    results_table = {}
    for name in signal_names:
        scores = np.array(signals_by_name[name])
        labels_incorrect = (1 - correct)  # 1 = incorrect

        # AUROC (higher = better at predicting incorrectness)
        if len(set(labels_incorrect)) >= 2:
            try:
                auroc = roc_auc_score(labels_incorrect, scores)
            except Exception:
                auroc = float('nan')
        else:
            auroc = float('nan')

        # AURC
        au = aurc(scores, correct)

        # Accuracy at coverages
        acc_90 = accuracy_at_coverage(scores, correct, 0.90)
        acc_75 = accuracy_at_coverage(scores, correct, 0.75)
        acc_50 = accuracy_at_coverage(scores, correct, 0.50)

        results_table[name] = {
            "auroc": auroc, "aurc": au,
            "acc_90": acc_90, "acc_75": acc_75, "acc_50": acc_50,
        }

        print(f"  {name:<16} {auroc:>7.3f} {au:>7.3f} "
              f"{acc_90:>7.1%} {acc_75:>7.1%} {acc_50:>7.1%}")

    # Baseline: answer all (no abstention)
    print(f"  {'-'*65}")
    print(f"  {'(answer all)':<16} {'---':>7} {'---':>7} "
          f"{'---':>8} {'---':>8} {base_acc:>7.1%}")
    print(f"  {'(random abst.)':<16} {0.5:>7.3f} {1-base_acc:>7.3f} "
          f"{base_acc:>7.1%} {base_acc:>7.1%} {base_acc:>7.1%}")

    return results_table


def cross_domain_transfer(results_by_dataset):
    """Cross-domain test: select best threshold on source, evaluate on target.

    Tests the claim that the choice function has a parameter-free advantage:
    if you can't validate on the target domain, what's the cost of
    picking the wrong fixed threshold?
    """
    print(f"\n{'='*75}")
    print(f"  CROSS-DOMAIN TRANSFER: threshold selected on one dataset,")
    print(f"  applied to another")
    print(f"{'='*75}")

    datasets = list(results_by_dataset.keys())
    if len(datasets) < 2:
        print("  Need at least 2 datasets for transfer test")
        return

    fixed_threshold_signals = ["SE(0.75)", "SE(0.85)", "SE(0.90)"]
    adaptive_signals = ["SE_width", "SE(0.75)+rej"]

    # For each (source, target) pair:
    #   1. Find the best fixed-threshold SE on source (by AURC)
    #   2. Apply that signal to target, measure AURC
    #   3. Compare to: (a) best fixed SE on target (oracle),
    #                  (b) choice-function-based signal on target (no tuning)
    print(f"\n  {'Source':<10} {'Target':<10} {'Best on src':>14} "
          f"{'Transfer AURC':>15} {'Oracle AURC':>13} {'SE_width AURC':>15}")
    print(f"  {'-'*80}")

    for source in datasets:
        for target in datasets:
            if source == target:
                continue

            src = results_by_dataset[source]
            tgt = results_by_dataset[target]
            tgt_correct = np.array([r["mv_correct"] for r in tgt])

            # Best fixed threshold on source
            src_correct = np.array([r["mv_correct"] for r in src])
            best_src_signal = None
            best_src_aurc = float('inf')
            for sig in fixed_threshold_signals:
                scores = np.array([compute_signals(r)[sig] for r in src])
                au = aurc(scores, src_correct)
                if au < best_src_aurc:
                    best_src_aurc = au
                    best_src_signal = sig

            # Apply to target
            tgt_scores = np.array([compute_signals(r)[best_src_signal] for r in tgt])
            transfer_aurc = aurc(tgt_scores, tgt_correct)

            # Oracle (best fixed on target)
            best_tgt_aurc = float('inf')
            best_tgt_signal = None
            for sig in fixed_threshold_signals:
                scores = np.array([compute_signals(r)[sig] for r in tgt])
                au = aurc(scores, tgt_correct)
                if au < best_tgt_aurc:
                    best_tgt_aurc = au
                    best_tgt_signal = sig

            # SE width (parameter-free) on target
            width_scores = np.array([compute_signals(r)["SE_width"] for r in tgt])
            width_aurc = aurc(width_scores, tgt_correct)

            print(f"  {source:<10} {target:<10} {best_src_signal:>14} "
                  f"{transfer_aurc:>15.3f} {best_tgt_aurc:>13.3f} "
                  f"{width_aurc:>15.3f}")


def main():
    script_dir = Path(__file__).parent

    # Load all results
    with open(script_dir / "kle_snne_triviaqa_results.json") as f:
        triviaqa = json.load(f)
    with open(script_dir / "kle_snne_results.json") as f:
        arc_mmlu = json.load(f)

    arc = [r for r in arc_mmlu if r["dataset"] == "arc"]
    mmlu = [r for r in arc_mmlu if r["dataset"] == "mmlu"]

    # Per-dataset analysis
    all_tables = {}
    for name, results in [("TriviaQA", triviaqa), ("ARC", arc), ("MMLU-Gemma", mmlu)]:
        all_tables[name] = analyze_dataset(results, name)

    # Cross-domain transfer
    cross_domain_transfer({"TriviaQA": triviaqa, "ARC": arc})

    # Save summary
    summary_file = script_dir / "selective_prediction_results.json"
    with open(summary_file, 'w') as f:
        json.dump(all_tables, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
