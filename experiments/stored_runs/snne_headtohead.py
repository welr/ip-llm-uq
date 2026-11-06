"""
SNNE head-to-head selective-abstention comparison (reviewer response, R3).

Per-question scalar signals: SNNE, KLE, SE(theta=0.75), 1-maxCP.
At matched coverage, which signal gives higher retained accuracy?

Also: how does the admissibility PARTITION (filter responses, recompute
majority vote over admissible set) compare to these scalars at 100%
question-level coverage? The partition is the choice-function's distinctive
contribution beyond any scalar derivative.

Usage:
    ~/ip_llm/bin/python experiments/snne_headtohead.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from kle_snne_comparison import _cluster
from selective_prediction import (
    risk_coverage_curve,
    aurc,
    accuracy_at_coverage,
)

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
RNG_SEED = 42


def compute_maxcp(sim_matrix):
    """1 - max_theta max_k P_theta(C_k^theta)."""
    m = sim_matrix.shape[0]
    max_prob = 0.0
    for t in THRESHOLDS:
        clusters = _cluster(sim_matrix, t)
        if not clusters:
            continue
        max_prob = max(max_prob, max(len(c) for c in clusters) / m)
    return 1.0 - max_prob


def load_triviaqa(script_dir):
    with open(script_dir / "kle_snne_triviaqa_results.json") as f:
        records = json.load(f)
    records = [r for r in records if r["dataset"] == "triviaqa"]

    sim_cache_dir = script_dir / "sim_cache"
    model_keys = ["Qwen", "Mistral", "Llama", "Gemma"]
    sims_by_key = {}
    for model in model_keys:
        sim_file = sim_cache_dir / f"sim_matrices_triviaqa_{model}.json"
        if not sim_file.exists():
            print(f"  WARN: no sim cache for {model}")
            continue
        with open(sim_file) as f:
            sims = json.load(f)
        for qid, sim_list in sims.items():
            sims_by_key[(model, qid)] = np.array(sim_list)

    for r in records:
        key = (r["model"], r["qid"])
        if key in sims_by_key:
            r["maxcp"] = compute_maxcp(sims_by_key[key])
        else:
            r["maxcp"] = None
    records = [r for r in records if r["maxcp"] is not None]
    return records


def load_arc_or_mmlu(script_dir, dataset_name):
    with open(script_dir / "kle_snne_results.json") as f:
        records = json.load(f)
    records = [r for r in records if r["dataset"] == dataset_name]

    sim_cache_dir = script_dir / "sim_cache"
    sims_by_key = {}
    for sim_file in sim_cache_dir.glob(f"sim_matrices_{dataset_name}_*.json"):
        model = sim_file.stem.split("_")[-1]
        with open(sim_file) as f:
            sims = json.load(f)
        for qid, sim_list in sims.items():
            sims_by_key[(model, qid)] = np.array(sim_list)

    for r in records:
        key = (r["model"], r["qid"])
        if key in sims_by_key:
            r["maxcp"] = compute_maxcp(sims_by_key[key])
        else:
            r["maxcp"] = None
    records = [r for r in records if r["maxcp"] is not None]
    return records


def signal_values(records, signal_name):
    """Return array of per-question scores. Higher = more uncertain."""
    if signal_name == "SNNE":
        # codebase convention: use -SNNE so that higher score = more uncertain
        return np.array([-r["snne"] for r in records])
    if signal_name == "KLE":
        return np.array([r["kle"] for r in records])
    if signal_name == "SE(0.75)":
        return np.array([r["se_075"] for r in records])
    if signal_name == "1-maxCP":
        return np.array([r["maxcp"] for r in records])
    if signal_name == "n_rejected":
        return np.array([r["n_rejected"] for r in records], dtype=float)
    raise ValueError(signal_name)


def paired_bootstrap_contrast(records, signal_a, signal_b, n_boot=10000, seed=RNG_SEED):
    """Bootstrap CI on AURC(signal_a) - AURC(signal_b) contrast (negative = a better).

    Question-level bootstrap: resample questions with replacement, recompute both.
    """
    rng = np.random.default_rng(seed)
    n = len(records)
    correct = np.array([r["mv_correct"] for r in records])
    sa = signal_values(records, signal_a)
    sb = signal_values(records, signal_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(aurc(sa[idx], correct[idx]) - aurc(sb[idx], correct[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(np.mean(diffs))


def analyze(records, dataset_name):
    print("=" * 86)
    print(f"  SELECTIVE ABSTENTION HEAD-TO-HEAD: {dataset_name}  (n_Q={len(records)})")
    print("=" * 86)

    correct = np.array([r["mv_correct"] for r in records])
    base_acc = float(correct.mean())
    print(f"  Base (no abstention) MV accuracy:            {base_acc:.3%}")

    # Admissibility-filter-then-MV: per-question majority vote over admissible set only
    adm_correct = np.array([r.get("adm_mv_correct", r["mv_correct"]) for r in records])
    adm_acc = float(adm_correct.mean())
    print(f"  Admissibility-filter MV (100% Q-coverage):   {adm_acc:.3%}  "
          f"[delta vs base: {adm_acc - base_acc:+.3%}]")
    print()

    signals = ["SNNE", "KLE", "SE(0.75)", "1-maxCP", "n_rejected"]
    table = {}
    print(f"  {'Signal':<12} {'AUROC':>7} {'AURC':>7} "
          f"{'Acc@95%':>8} {'Acc@90%':>8} {'Acc@80%':>8} {'Acc@70%':>8}")
    print("  " + "-" * 70)

    for name in signals:
        scores = signal_values(records, name)
        labels_incorrect = (1 - correct).astype(int)
        try:
            auroc_v = float(roc_auc_score(labels_incorrect, scores)) if len(set(labels_incorrect)) >= 2 else float("nan")
        except Exception:
            auroc_v = float("nan")
        au = aurc(scores, correct)
        acc_95 = accuracy_at_coverage(scores, correct, 0.95)
        acc_90 = accuracy_at_coverage(scores, correct, 0.90)
        acc_80 = accuracy_at_coverage(scores, correct, 0.80)
        acc_70 = accuracy_at_coverage(scores, correct, 0.70)
        table[name] = {
            "auroc": auroc_v, "aurc": au,
            "acc_95": acc_95, "acc_90": acc_90,
            "acc_80": acc_80, "acc_70": acc_70,
        }
        print(f"  {name:<12} {auroc_v:>7.3f} {au:>7.3f} "
              f"{acc_95:>7.1%} {acc_90:>7.1%} {acc_80:>7.1%} {acc_70:>7.1%}")

    # Bootstrap contrasts
    print()
    print("  Paired-bootstrap AURC contrasts (negative = row method better):")
    print(f"    {'Contrast':<32} {'Mean delta':>11} {'95% CI':>22}")
    print("  " + "-" * 70)
    for (a, b) in [
        ("1-maxCP", "SNNE"),
        ("1-maxCP", "KLE"),
        ("1-maxCP", "SE(0.75)"),
        ("SNNE", "SE(0.75)"),
    ]:
        lo, hi, mean = paired_bootstrap_contrast(records, a, b)
        excl_zero = " *" if (lo > 0 or hi < 0) else ""
        print(f"    AURC({a}) - AURC({b})       {mean:>+11.4f}  [{lo:>+.4f}, {hi:>+.4f}]{excl_zero}")

    # Admissibility filter as a point method overlay
    # At 100% question coverage, we normally get base_acc. If instead we use
    # admissibility-filter MV, we get adm_acc. The "effective" comparison is:
    # "at what coverage does each scalar signal achieve adm_acc?"
    print()
    print(f"  Admissibility-filter MV gives {adm_acc:.3%} at 100% Q-coverage.")
    print(f"  Coverage at which each scalar signal matches this accuracy:")
    for name in signals:
        scores = signal_values(records, name)
        covs, _, accs = risk_coverage_curve(scores, correct)
        # Find smallest coverage at which acc >= adm_acc
        if accs.max() >= adm_acc:
            idx = np.argmax(accs >= adm_acc)
            # argmax returns first True; if never true then idx=0 but we gated above
            match_cov = float(covs[idx])
            print(f"    {name:<12}  coverage {match_cov:.1%}  (acc {accs[idx]:.3%})")
        else:
            print(f"    {name:<12}  never reaches {adm_acc:.3%} (max {accs.max():.3%})")

    return {"base_acc": base_acc, "adm_acc": adm_acc, "signals": table}


def main():
    script_dir = Path(__file__).parent
    tqa = load_triviaqa(script_dir)
    arc = load_arc_or_mmlu(script_dir, "arc")
    mmlu = load_arc_or_mmlu(script_dir, "mmlu")

    out = {}
    for name, recs in [("TriviaQA", tqa), ("ARC", arc), ("MMLU", mmlu)]:
        if recs:
            out[name] = analyze(recs, name)
        print()

    with open(script_dir / "snne_headtohead_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to experiments/snne_headtohead_results.json")


if __name__ == "__main__":
    main()
