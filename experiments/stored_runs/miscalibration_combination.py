"""
Miscalibration-detector combination (reviewer Q5 / deployment-guidance gap).

The 1-2/6 inversion failure mode: when the model's sampling mode is a
confident hallucination, admissibility rejects the correct minority.
1-maxCP alone gets this wrong (it says "confident, trust it") precisely
when the confidence is misplaced. A usable deployment signal needs a
second term that's *high* in the 1-2/6 regime.

We evaluate, on existing cached data (no new API calls, no new NLI):

  - baseline signals individually (1-maxCP, SE(0.75), SE width, KLE)
  - pairwise combinations via standardized linear stack
  - logistic regression stack of all features (via held-out cross-val)

and report AUROC for predicting majority-vote incorrectness, pooled
and stratified by Goldilocks bin (n_correct in {0, 1-2, 3-4, 5, 6}).

Usage:
    ~/ip_llm/bin/python experiments/miscalibration_combination.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


SIGNALS = ["one_minus_maxcp", "se_075", "se_085", "se_width", "kle"]


def standardize(xs):
    xs = np.asarray(xs, dtype=float)
    mu = xs.mean()
    sd = xs.std() + 1e-8
    return (xs - mu) / sd


def safe_auroc(scores, labels):
    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def logistic_stack_cv(X, y, n_splits=5, seed=42):
    """Out-of-fold logistic-regression AUROC."""
    if len(set(y)) < 2:
        return float("nan")
    X = np.asarray(X)
    y = np.asarray(y)
    # Use StratifiedKFold to keep both classes in each train fold.
    # Fall back to NaN if any class has fewer than n_splits samples.
    from sklearn.model_selection import StratifiedKFold
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if min(n_pos, n_neg) < n_splits:
        return float("nan")
    preds = np.zeros_like(y, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if len(set(y[tr])) < 2:
            return float("nan")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, preds)


def bin_label(n_correct):
    if n_correct == 0:
        return "0/6"
    if n_correct <= 2:
        return "1-2/6"
    if n_correct <= 4:
        return "3-4/6"
    if n_correct == 5:
        return "5/6"
    return "6/6"


def build_dataset(per_q_records, per_q_kle):
    """Join per-q signals from averaged_param_baselines with SE width and
    bin labels from kle_snne_* records."""
    # Index kle records by (dataset, model, qid)
    kle_idx = {(r["dataset"], r["model"], r["qid"]): r for r in per_q_kle}
    rows = []
    for r in per_q_records:
        key = (r["dataset"], r["model"], r["qid"])
        if key not in kle_idx:
            continue
        k = kle_idx[key]
        # accuracies is a list of 6 ints; n_correct determines bin
        accs = k["accuracies"]
        n_correct = int(sum(accs))
        rows.append({
            "dataset": r["dataset"],
            "model": r["model"],
            "qid": r["qid"],
            "one_minus_maxcp": r["one_minus_maxcp"],
            "se_075": r["se_075"],
            "se_085": r["se_085"],
            "se_avg": r["se_avg"],
            "se_width": k["se_width"],
            "kle": r["kle"],
            "kle_avg": r["kle_avg"],
            "mv_correct": k["mv_correct"],
            "n_correct": n_correct,
            "bin": bin_label(n_correct),
        })
    return rows


def evaluate(rows, dataset=None, bin_filter=None):
    """Return AUROC table for a subset of rows."""
    sub = rows
    if dataset is not None:
        sub = [r for r in sub if r["dataset"] == dataset]
    if bin_filter is not None:
        sub = [r for r in sub if r["bin"] == bin_filter]
    if not sub:
        return None

    # Labels: 1 = incorrect majority vote
    y = np.array([1 - r["mv_correct"] for r in sub])
    n = len(sub)
    if len(set(y)) < 2:
        return {"n": n, "note": "all-same-label"}

    # Individual signals
    out = {"n": n, "frac_incorrect": float(y.mean())}
    for sig in SIGNALS:
        out[sig] = safe_auroc([r[sig] for r in sub], y)

    # Sign heuristic: we expect larger signal = more incorrect. 1-maxCP,
    # SE(0.75), SE width, KLE all follow this (higher uncertainty -> more
    # likely wrong).
    Xall = np.column_stack([standardize([r[s] for r in sub]) for s in SIGNALS])

    # Simple pair combinations (standardized average)
    for a, b in [
        ("one_minus_maxcp", "se_width"),
        ("one_minus_maxcp", "se_075"),
        ("one_minus_maxcp", "kle"),
        ("se_075", "se_width"),
    ]:
        xa = standardize([r[a] for r in sub])
        xb = standardize([r[b] for r in sub])
        out[f"{a}+{b}"] = safe_auroc(0.5 * (xa + xb), y)

    # Logistic regression stack (5-fold CV)
    out["logistic_stack_cv"] = logistic_stack_cv(Xall, y)

    return out


def print_main_table(rows, datasets):
    print(f"\n{'='*100}")
    print("  AUROC pooled across models")
    print(f"{'='*100}\n")
    sigs = [
        "one_minus_maxcp", "se_075", "se_width", "kle",
        "one_minus_maxcp+se_width", "one_minus_maxcp+se_075",
        "logistic_stack_cv",
    ]
    header = f"  {'Dataset':<12} " + " ".join(f"{s[:14]:>14}" for s in sigs) + f"  {'n':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for d in datasets:
        r = evaluate(rows, dataset=d)
        if r is None:
            continue
        row = f"  {d:<12}"
        for s in sigs:
            v = r.get(s, float("nan"))
            row += f" {v:>14.4f}" if not np.isnan(v) else f" {'N/A':>14}"
        row += f"  {r['n']:>6}"
        print(row)


def print_stratified(rows, datasets):
    print(f"\n{'='*100}")
    print("  AUROC stratified by Goldilocks bin (base-accuracy-per-question)")
    print(f"{'='*100}")
    bins = ["0/6", "1-2/6", "3-4/6", "5/6", "6/6"]
    sigs = ["one_minus_maxcp", "se_075", "se_width",
            "one_minus_maxcp+se_width", "logistic_stack_cv"]
    for d in datasets:
        print(f"\n  {d.upper()}")
        print(f"  {'Bin':<8} {'n':>6} {'%inc':>6}  " +
              " ".join(f"{s[:18]:>18}" for s in sigs))
        print("  " + "-" * 100)
        for b in bins:
            r = evaluate(rows, dataset=d, bin_filter=b)
            if r is None:
                print(f"  {b:<8} (no data)")
                continue
            if "note" in r:
                print(f"  {b:<8} {r['n']:>6}  (all-same-label)")
                continue
            line = f"  {b:<8} {r['n']:>6} {r['frac_incorrect']:>6.1%}  "
            for s in sigs:
                v = r.get(s, float("nan"))
                line += f" {v:>18.3f}" if not np.isnan(v) else f" {'N/A':>18}"
            print(line)


def main():
    script_dir = Path(__file__).parent
    # Load per-q records with all signals (computed in averaged_param_baselines)
    avg_file = script_dir / "averaged_param_baselines_results.json"
    per_q = json.load(open(avg_file))

    # Load kle-snne records for se_width + accuracies
    kle_records = []
    for f in ["kle_snne_results.json", "kle_snne_triviaqa_results.json"]:
        p = script_dir / f
        if p.exists():
            kle_records.extend(json.load(open(p)))
    print(f"Loaded {len(per_q)} per-q averaged-param records "
          f"+ {len(kle_records)} kle-snne records")

    rows = build_dataset(per_q, kle_records)
    print(f"Joined {len(rows)} rows")

    datasets = ["triviaqa", "arc", "mmlu"]
    print_main_table(rows, datasets)
    print_stratified(rows, datasets)

    out = script_dir / "miscalibration_combination_results.json"
    with open(out, "w") as f:
        json.dump(rows, f, default=str)
    print(f"\nPer-q joined data saved to {out}")


if __name__ == "__main__":
    main()
