"""
Self-consistency baseline for selective abstention (R3's Q1, R1 implicit).

Self-consistency confidence = fraction of responses in the largest normalized-string
group (Wang et al. 2023 "Self-Consistency Improves Chain of Thought"). This is
the baseline most production UQ stacks already deploy; comparing against it is
the deployment-relevant question R3 asked.

Computes:
  - Per-question self-consistency score (1 - max_fraction) used as uncertainty.
  - AUROC for predicting majority-vote incorrectness.
  - AURC and accuracy at 80%/90% coverage.
  - Paired-bootstrap CI on contrast vs 1-maxCP and -SNNE.

Usage:
    ~/ip_llm/bin/python experiments/self_consistency_baseline.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from kle_snne_comparison import normalize_answer
from selective_prediction import risk_coverage_curve, aurc, accuracy_at_coverage
from snne_headtohead import load_triviaqa, signal_values

PRED_FILES = {
    "Qwen": "pred1_qwen.json",
    "Mistral": "pred1_mistral_20260110_071956.json",
    "Llama": "pred1_llama8b_20260112_074432.json",
    "Gemma": "pred1_gemma9b_20260112_074222.json",
}


def compute_self_consistency_scores(script_dir):
    """For each (model, qid), compute self-consistency score = 1 - max_fraction
    where max_fraction is (size of largest normalized-string group) / M."""
    data_dir = script_dir / "modal_data"
    scores = {}
    for model, fn in PRED_FILES.items():
        path = data_dir / fn
        if not path.exists():
            print(f"  SKIP {model}: {path} not found")
            continue
        with open(path) as f:
            d = json.load(f)
        pq = d["per_question_results"]
        for qid, qdata in pq.items():
            responses = qdata["base_responses"][:6]
            if len(responses) < 2:
                continue
            normalized = [normalize_answer(r) for r in responses]
            counts = Counter(normalized)
            max_count = counts.most_common(1)[0][1]
            max_frac = max_count / len(responses)
            scores[(model, qid)] = 1.0 - max_frac  # higher = more uncertain
    return scores


def main():
    script_dir = Path(__file__).parent

    tqa = load_triviaqa(script_dir)
    print(f"Loaded {len(tqa)} TriviaQA records")

    sc_scores = compute_self_consistency_scores(script_dir)
    print(f"Computed self-consistency scores for {len(sc_scores)} (model, qid) pairs")

    # Merge: filter tqa to records that have SC scores + attach SC score
    merged = []
    for r in tqa:
        key = (r["model"], r["qid"])
        if key in sc_scores:
            r2 = dict(r)
            r2["self_consistency"] = sc_scores[key]
            merged.append(r2)
    print(f"Merged: {len(merged)} records with self-consistency scores\n")

    correct = np.array([r["mv_correct"] for r in merged])
    sc = np.array([r["self_consistency"] for r in merged])
    snne = signal_values(merged, "SNNE")
    maxcp = signal_values(merged, "1-maxCP")
    se075 = signal_values(merged, "SE(0.75)")
    kle = signal_values(merged, "KLE")

    # AUROC / AURC for each
    labels_incorrect = (1 - correct).astype(int)

    def metrics(scores, name):
        au_roc = float(roc_auc_score(labels_incorrect, scores)) if len(set(labels_incorrect)) > 1 else float("nan")
        au_rc = aurc(scores, correct)
        a90 = accuracy_at_coverage(scores, correct, 0.90)
        a80 = accuracy_at_coverage(scores, correct, 0.80)
        return {"name": name, "AUROC": au_roc, "AURC": au_rc, "Acc@90": a90, "Acc@80": a80}

    print("=" * 86)
    print(f"  SELF-CONSISTENCY HEAD-TO-HEAD on TriviaQA  (n_Q={len(merged)})")
    print("=" * 86)
    print(f"  {'Signal':<20} {'AUROC':>7} {'AURC':>7} {'Acc@90':>8} {'Acc@80':>8}")
    print("  " + "-" * 60)

    table = []
    for sc_arr, name in [
        (sc, "Self-consistency"),
        (snne, "-SNNE"),
        (maxcp, "1-maxCP"),
        (se075, "SE(0.75)"),
        (kle, "KLE"),
    ]:
        m = metrics(sc_arr, name)
        table.append(m)
        print(f"  {name:<20} {m['AUROC']:>7.3f} {m['AURC']:>7.3f} {m['Acc@90']:>7.1%} {m['Acc@80']:>7.1%}")

    # Paired bootstrap on AURC contrasts
    print()
    print("  Paired-bootstrap AURC contrasts (10k resamples, seed 42).")
    print("  Negative sign in 'delta' means ROW is better (lower AURC).")
    print(f"    {'Contrast':<40} {'Mean delta':>11} {'95% CI':>22}")
    print("  " + "-" * 78)

    rng = np.random.default_rng(42)
    n_boot = 10000
    n = len(correct)

    def paired(a, b, label):
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            diffs.append(aurc(a[idx], correct[idx]) - aurc(b[idx], correct[idx]))
        lo = float(np.percentile(diffs, 2.5))
        hi = float(np.percentile(diffs, 97.5))
        mean = float(np.mean(diffs))
        excl = " *" if (lo > 0 or hi < 0) else ""
        print(f"    {label:<40} {mean:>+11.4f}  [{lo:>+.4f}, {hi:>+.4f}]{excl}")
        return {"contrast": label, "mean": mean, "ci_lo": lo, "ci_hi": hi}

    paired(sc, maxcp, "AURC(SC) - AURC(1-maxCP)")
    paired(sc, snne, "AURC(SC) - AURC(-SNNE)")
    paired(sc, se075, "AURC(SC) - AURC(SE(0.75))")
    paired(sc, kle, "AURC(SC) - AURC(KLE)")

    out = {
        "n_Q": len(merged),
        "metrics": {m["name"]: m for m in table},
    }
    with open(script_dir / "self_consistency_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to experiments/self_consistency_results.json")


if __name__ == "__main__":
    main()
