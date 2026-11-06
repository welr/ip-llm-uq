"""
Stratified-by-difficulty analysis.

Tests the paper's ceiling-effect claim: "the lower the base accuracy, the
larger the admissibility ratio." Within each dataset, bin questions by
per-question base accuracy (fraction of responses correct out of 6), then
compute admissible vs rejected accuracy and their ratio within each bin.

If the ceiling-effect story is right, we should see:
  - Hard bins (low base accuracy): ratio ~ 1.0 (nothing to discriminate)
  - Medium bins: highest ratio (admissibility helps most)
  - Easy bins: ratio ~ 1.0 (everything is correct anyway)

Uses per-question data from:
  - kle_snne_triviaqa_results.json (TriviaQA, 4 models)
  - kle_snne_results.json (ARC, 4 models; MMLU, Gemma only)

Also loads checkpoint_analysis files to get per-question admissibility labels
where available (needed to compute per-response admissibility within bins).

Usage:
    ~/ip_llm/bin/python experiments/stratified_difficulty.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def bootstrap_ratio(adm_c, adm_t, rej_c, rej_t, n_boot=5000, seed=42):
    """Bootstrap 95% CI for accuracy ratio."""
    if rej_t == 0 or adm_t == 0:
        return float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    adm_arr = np.array([1]*adm_c + [0]*(adm_t - adm_c))
    rej_arr = np.array([1]*rej_c + [0]*(rej_t - rej_c))
    ratios = []
    for _ in range(n_boot):
        a_sample = rng.choice(adm_arr, size=adm_t, replace=True)
        r_sample = rng.choice(rej_arr, size=rej_t, replace=True)
        a_acc = a_sample.mean()
        r_acc = r_sample.mean()
        if r_acc > 0:
            ratios.append(a_acc / r_acc)
    if len(ratios) < 100:
        return float('nan'), float('nan')
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def load_analysis_checkpoint(path):
    """Load checkpoint_analysis file: maps qid -> per-question dict."""
    with open(path) as f:
        data = json.load(f)
    return data  # already a qid -> dict mapping


def load_triviaqa_pred1(path):
    """Load pred1 TriviaQA file: maps qid -> response texts + gold answers."""
    with open(path) as f:
        d = json.load(f)
    return d["per_question_results"]


def load_triviaqa_results(path):
    """Load kle_snne_triviaqa_results.json; return list of per-question records."""
    with open(path) as f:
        return json.load(f)


def bin_difficulty(n_correct, n_total=6):
    """Map number of correct responses to a difficulty bin."""
    frac = n_correct / n_total
    if n_correct == 0:
        return "0/6"
    elif n_correct <= 2:
        return "1-2/6"
    elif n_correct <= 4:
        return "3-4/6"
    elif n_correct == 5:
        return "5/6"
    else:  # 6/6
        return "6/6"


BIN_ORDER = ["0/6", "1-2/6", "3-4/6", "5/6", "6/6"]
BIN_ACC_MIDPOINTS = {
    "0/6": 0.0, "1-2/6": 0.25, "3-4/6": 0.58, "5/6": 0.83, "6/6": 1.0,
}


def stratify_from_perresult(per_question_results, label):
    """Given list of per-question dicts with 'accuracies', 'n_admissible', 'n_rejected',
    and ideally admissible/rejected indices, compute bin-stratified stats.

    This version requires the *indices* of admissible/rejected responses, not
    just counts. If we only have counts we can't compute bin-level accuracies
    correctly.
    """
    bins = defaultdict(lambda: {"adm_c": 0, "adm_t": 0, "rej_c": 0, "rej_t": 0,
                                "n_questions": 0, "n_q_with_rej": 0,
                                "base_accs": []})
    skipped = 0
    for r in per_question_results:
        accuracies = r["accuracies"]
        n = len(accuracies)
        n_correct = sum(accuracies)
        b = bin_difficulty(n_correct, n_total=n)

        # We need admissible/rejected indices
        if "nli_admissible" in r:
            adm_idx = set(r["nli_admissible"])
            rej_idx = set(r["nli_rejected"])
        elif "admissible_indices" in r:
            adm_idx = set(r["admissible_indices"])
            rej_idx = set(r["rejected_indices"])
        else:
            # Fall back: reconstruct from counts if we trust the ordering
            # Not possible — skip
            skipped += 1
            continue

        bins[b]["n_questions"] += 1
        bins[b]["base_accs"].append(n_correct / n)
        if len(rej_idx) > 0:
            bins[b]["n_q_with_rej"] += 1

        for i in adm_idx:
            bins[b]["adm_t"] += 1
            bins[b]["adm_c"] += accuracies[i]
        for i in rej_idx:
            bins[b]["rej_t"] += 1
            bins[b]["rej_c"] += accuracies[i]

    if skipped:
        print(f"  WARN: skipped {skipped} questions (no admissibility indices)")

    print(f"\n  {label}")
    print(f"  {'-'*85}")
    print(f"  {'Difficulty':<12} {'n_Q':>5} {'n_adm':>6} {'n_rej':>6} "
          f"{'Adm Acc':>8} {'Rej Acc':>8} {'Ratio':>8} {'[95% CI]':>18}")
    print(f"  {'-'*85}")

    # Also compute correlation: per-bin ratio vs per-bin mean base accuracy
    bin_ratios = []
    bin_base_accs = []

    for b in BIN_ORDER:
        if b not in bins or bins[b]["n_questions"] == 0:
            continue
        d = bins[b]
        mean_base = np.mean(d["base_accs"])
        adm_acc = d["adm_c"] / max(d["adm_t"], 1)
        rej_acc = d["rej_c"] / max(d["rej_t"], 1) if d["rej_t"] > 0 else float('nan')
        if d["rej_t"] > 0 and rej_acc > 0:
            ratio = adm_acc / rej_acc
            ci_lo, ci_hi = bootstrap_ratio(d["adm_c"], d["adm_t"],
                                            d["rej_c"], d["rej_t"])
            ci_str = f"[{ci_lo:.2f},{ci_hi:.2f}]"
            bin_ratios.append(ratio)
            bin_base_accs.append(mean_base)
        elif d["rej_t"] > 0:
            ratio = float('inf')
            ci_str = "[inf]"
            bin_ratios.append(ratio)
            bin_base_accs.append(mean_base)
        else:
            ratio = float('nan')
            ci_str = "(no rej)"

        print(f"  {b:<12} {d['n_questions']:>5} {d['adm_t']:>6} {d['rej_t']:>6} "
              f"{adm_acc:>7.1%} {rej_acc:>7.1%} "
              f"{ratio:>7.2f}x {ci_str:>18}")

    # Pooled (all bins)
    total_adm_c = sum(bins[b]["adm_c"] for b in bins)
    total_adm_t = sum(bins[b]["adm_t"] for b in bins)
    total_rej_c = sum(bins[b]["rej_c"] for b in bins)
    total_rej_t = sum(bins[b]["rej_t"] for b in bins)
    total_q = sum(bins[b]["n_questions"] for b in bins)
    if total_rej_t > 0 and total_adm_t > 0:
        pool_adm = total_adm_c / total_adm_t
        pool_rej = total_rej_c / total_rej_t
        pool_ratio = pool_adm / max(pool_rej, 0.001)
        ci_lo, ci_hi = bootstrap_ratio(total_adm_c, total_adm_t,
                                        total_rej_c, total_rej_t)
        print(f"  {'-'*85}")
        print(f"  {'Pooled':<12} {total_q:>5} {total_adm_t:>6} {total_rej_t:>6} "
              f"{pool_adm:>7.1%} {pool_rej:>7.1%} "
              f"{pool_ratio:>7.2f}x [{ci_lo:.2f},{ci_hi:.2f}]")

    # Compute Spearman correlation between bin mean accuracy and bin ratio
    # (excluding degenerate bins)
    if len(bin_ratios) >= 3:
        finite = [(b, r) for b, r in zip(bin_base_accs, bin_ratios)
                  if np.isfinite(r)]
        if len(finite) >= 3:
            ba, br = zip(*finite)
            rho, pval = stats.spearmanr(ba, br)
            print(f"\n  Spearman(bin base accuracy, bin ratio) = {rho:.3f} (p={pval:.3f})")
            if rho < 0:
                print(f"  => As base accuracy increases, ratio DECREASES "
                      f"(ceiling-effect prediction confirmed)")
            else:
                print(f"  => As base accuracy increases, ratio INCREASES "
                      f"(ceiling-effect prediction NOT confirmed)")


def stratify_from_checkpoint(checkpoint_path, label):
    """Load checkpoint_analysis file and stratify."""
    with open(checkpoint_path) as f:
        data = json.load(f)

    # Data is a qid -> {accuracies, nli_admissible, nli_rejected, ...} dict
    results = list(data.values())

    # Check structure of first result
    if not results:
        print(f"  {label}: empty")
        return

    first = results[0]
    if "nli_admissible" not in first:
        print(f"  {label}: no nli_admissible field; keys: {list(first.keys())[:10]}")
        return

    stratify_from_perresult(results, label)


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / "modal_data"

    print(f"{'='*85}")
    print(f"  STRATIFIED-BY-DIFFICULTY ANALYSIS")
    print(f"  Tests: does admissibility ratio decrease as base accuracy increases?")
    print(f"{'='*85}")

    # ARC: 4 models via checkpoint_analysis files
    print(f"\n\n=== ARC-Challenge ===")
    for model in ["Qwen", "Mistral", "Llama", "Gemma"]:
        cp = data_dir / f"checkpoint_analysis_arc_{model}.json"
        if cp.exists():
            stratify_from_checkpoint(cp, f"ARC / {model}")

    # Pool ARC across models
    print(f"\n--- ARC POOLED ACROSS MODELS ---")
    all_arc = []
    for model in ["Qwen", "Mistral", "Llama", "Gemma"]:
        cp = data_dir / f"checkpoint_analysis_arc_{model}.json"
        if cp.exists():
            with open(cp) as f:
                data = json.load(f)
            all_arc.extend(list(data.values()))
    stratify_from_perresult(all_arc, "ARC (pooled, 4 models)")

    # MMLU: only Gemma has per-question data
    print(f"\n\n=== MMLU ===")
    cp = data_dir / f"checkpoint_analysis_mmlu_Gemma.json"
    if cp.exists():
        stratify_from_checkpoint(cp, f"MMLU / Gemma")

    # TriviaQA: we don't have checkpoint_analysis files. The kle_snne_triviaqa_results
    # has n_admissible/n_rejected counts but not the indices. We can compute base
    # accuracy from "accuracies" but can't compute adm/rej accuracy without indices.
    # For TriviaQA we need to recompute admissibility from the cached sim matrices.

    print(f"\n\n=== TriviaQA (recomputing admissibility from cached sim matrices) ===")
    from kle_snne_comparison import _cluster
    THRESHOLDS_T = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    cache_dir = script_dir / "sim_cache"
    all_triviaqa = []
    for model in ["Qwen", "Mistral", "Llama", "Gemma"]:
        cache_file = cache_dir / f"sim_matrices_triviaqa_{model}.json"
        if not cache_file.exists():
            continue
        with open(cache_file) as f:
            sims = json.load(f)

        # Load per-question data from kle_snne_triviaqa_results.json for accuracies
        with open(script_dir / "kle_snne_triviaqa_results.json") as f:
            all_results = json.load(f)
        model_results = [r for r in all_results if r["model"] == model
                         and r["dataset"] == "triviaqa"]

        for r in model_results:
            qid = r["qid"]
            if qid not in sims:
                continue
            sim = np.array(sims[qid])
            m = sim.shape[0]
            # Recompute admissibility
            admissible = set()
            for theta in THRESHOLDS_T:
                clusters = _cluster(sim, theta)
                probs = [len(c) / m for c in clusters]
                max_p = max(probs)
                for c, p in zip(clusters, probs):
                    if p == max_p:
                        admissible |= c
            rejected = set(range(m)) - admissible
            all_triviaqa.append({
                "accuracies": r["accuracies"],
                "nli_admissible": sorted(admissible),
                "nli_rejected": sorted(rejected),
            })

    if all_triviaqa:
        stratify_from_perresult(all_triviaqa, "TriviaQA (pooled, 4 models)")

    # TruthfulQA: per-model prereg JSONs contain per_question with
    # nli_admissible/nli_rejected indices and accuracies — drop them straight
    # into stratify_from_perresult.
    print(f"\n\n=== TruthfulQA ===")
    tqa_files = {
        "Qwen":    "prereg_truthfulqa_Qwen_20260416_123936.json",
        "Mistral": "prereg_truthfulqa_Mistral_20260416_173916.json",
        "Llama":   "prereg_truthfulqa_Llama_20260416_123942.json",
        "Gemma":   "prereg_truthfulqa_Gemma_20260416_211849.json",
    }
    all_tqa = []
    for model, fname in tqa_files.items():
        path = data_dir / fname
        if not path.exists():
            print(f"  TruthfulQA / {model}: missing {fname}")
            continue
        with open(path) as f:
            data = json.load(f)
        records = list(data["per_question"].values())
        stratify_from_perresult(records, f"TruthfulQA / {model}")
        all_tqa.extend(records)

    print(f"\n--- TruthfulQA POOLED ACROSS MODELS ---")
    if all_tqa:
        stratify_from_perresult(all_tqa, "TruthfulQA (pooled, 4 models)")


if __name__ == "__main__":
    main()
