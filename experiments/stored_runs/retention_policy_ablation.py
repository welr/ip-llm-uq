"""
Retention-policy ablation (Reviewer W3 / Q2).

Question: Is the admissibility effect (2.94x TriviaQA ratio) sensitive to the
retention policy — top-k by log-probability — or does it persist under other
retention rules at matched count?

We cannot directly compare to all-15 (raw 15-response NLI matrices are not
stored; only the 6x6 matrices corresponding to the top-6-by-log-prob
responses are persisted). What we CAN do is verify the effect is stable
under:
    (1) retention count: k in {3, 4, 5, 6} using top-k-by-log-prob
        (response index 0..k-1, since indices are log-prob-descending);
    (2) retention order: for each k < 6, random k-of-6 subsampling
        (50 reps per (model, qid)) versus top-k-of-6.

If the ratio collapses under random-k-of-6 but holds under top-k-of-6, the
effect would be an artifact of log-prob ranking. If both hold, the effect is
not a sampling-order artifact at the level we can probe.

Datasets:
    - TriviaQA, 4 base models (12,000 responses total under k=6)

Usage:
    ~/ip_llm/bin/python experiments/retention_policy_ablation.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MODELS = ["Qwen", "Mistral", "Llama", "Gemma"]
RNG_SEED = 42
N_RANDOM_REPS = 50


def cluster_cc_subset(sim, theta, idx_subset):
    """Connected-component clustering over a subset of responses."""
    m = len(idx_subset)
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(m):
        for b in range(a + 1, m):
            if sim[idx_subset[a]][idx_subset[b]] >= theta:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for i in range(m):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def admissibility(sim, idx_subset, thresholds=THRESHOLDS):
    """Return list[bool] of length len(idx_subset): admissible flag per retained response."""
    m = len(idx_subset)
    admissible = [False] * m
    for t in thresholds:
        clusters = cluster_cc_subset(sim, t, idx_subset)
        max_size = max(len(c) for c in clusters)
        for c in clusters:
            if len(c) == max_size:
                for r in c:
                    admissible[r] = True
    return admissible


def compute_ratio(retained_acc, retained_adm):
    """Pooled admissibility ratio (acc|admissible / acc|rejected) with bootstrap CI."""
    retained_acc = np.asarray(retained_acc, dtype=float)
    retained_adm = np.asarray(retained_adm, dtype=bool)
    adm_acc = retained_acc[retained_adm]
    rej_acc = retained_acc[~retained_adm]
    if len(rej_acc) == 0 or rej_acc.mean() == 0:
        ratio = float("inf") if adm_acc.mean() > 0 else float("nan")
    else:
        ratio = adm_acc.mean() / rej_acc.mean()
    rng = np.random.default_rng(RNG_SEED)
    ratios = []
    n = len(retained_acc)
    for _ in range(5000):
        idx = rng.integers(0, n, size=n)
        adm_b = retained_adm[idx]
        acc_b = retained_acc[idx]
        a_acc = acc_b[adm_b].mean() if adm_b.any() else 0
        r_acc = acc_b[~adm_b].mean() if (~adm_b).any() else 0
        if r_acc == 0:
            continue
        ratios.append(a_acc / r_acc)
    ratios = np.array(ratios)
    lo, hi = (np.quantile(ratios, [0.025, 0.975]) if len(ratios) else (float("nan"), float("nan")))
    return {
        "ratio": float(ratio),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_admissible": int(retained_adm.sum()),
        "n_rejected": int((~retained_adm).sum()),
        "adm_acc": float(adm_acc.mean()) if len(adm_acc) else float("nan"),
        "rej_acc": float(rej_acc.mean()) if len(rej_acc) else float("nan"),
        "rejection_rate": float((~retained_adm).sum() / n) if n else float("nan"),
    }


def run_policy(sim_by_qm, acc_by_qm, subset_fn, n_reps=1, seed_base=0):
    """
    sim_by_qm: {(model, qid): 6x6 similarity matrix (list of lists)}
    acc_by_qm: {(model, qid): list of 6 ints (0/1) accuracy per response}
    subset_fn: (rng, m) -> list of indices to retain (len = k).
    n_reps: number of random repetitions (for deterministic policies use 1).
    """
    rep_results = []
    for rep in range(n_reps):
        rng = np.random.default_rng(seed_base + rep)
        retained_acc = []
        retained_adm = []
        for key, sim in sim_by_qm.items():
            acc = acc_by_qm.get(key)
            if acc is None:
                continue
            idx = subset_fn(rng, len(sim))
            if len(idx) < 2:
                continue
            adm = admissibility(sim, idx)
            for local_i, global_i in enumerate(idx):
                retained_acc.append(acc[global_i])
                retained_adm.append(adm[local_i])
        rep_results.append(compute_ratio(retained_acc, retained_adm))

    if n_reps == 1:
        return rep_results[0]

    ratios = [r["ratio"] for r in rep_results if np.isfinite(r["ratio"])]
    rej_rates = [r["rejection_rate"] for r in rep_results]
    return {
        "ratio_mean": float(np.mean(ratios)) if ratios else float("nan"),
        "ratio_std": float(np.std(ratios)) if ratios else float("nan"),
        "ratio_min": float(np.min(ratios)) if ratios else float("nan"),
        "ratio_max": float(np.max(ratios)) if ratios else float("nan"),
        "rejection_rate_mean": float(np.mean(rej_rates)),
        "n_reps": len(rep_results),
        "first_rep_detail": rep_results[0],
    }


def main():
    script_dir = Path(__file__).parent
    sim_cache_dir = script_dir / "sim_cache"

    # Load accuracies (top-6 order, log-prob-descending).
    with open(script_dir / "kle_snne_triviaqa_results.json") as f:
        records = json.load(f)
    records = [r for r in records if r["dataset"] == "triviaqa"]
    acc_by_qm = {(r["model"], r["qid"]): r["accuracies"] for r in records}

    # Load sim matrices and index by (model, qid).
    sim_by_qm = {}
    for model in MODELS:
        with open(sim_cache_dir / f"sim_matrices_triviaqa_{model}.json") as f:
            sims = json.load(f)
        for qid, matrix in sims.items():
            sim_by_qm[(model, qid)] = matrix

    print("=== TriviaQA retention-policy ablation ===")
    print(f"Questions: {len(sim_by_qm)}  (4 models x 500 questions expected = 2000)")
    print(f"Thresholds: {THRESHOLDS}")
    print()

    rows_out = []

    # --- Policy: top-k by log-prob (deterministic, indices 0..k-1) ---
    print("--- Top-k-by-log-prob (deterministic; indices 0..k-1) ---")
    header = f"{'k':>3}  {'ratio':>6}  {'95% CI':>14}  {'adm_acc':>7}  {'rej_acc':>7}  {'rej_rate':>8}  {'n_adm':>6}  {'n_rej':>6}"
    print(header)
    for k in [3, 4, 5, 6]:
        def top_k(rng, m, k=k):
            return list(range(min(k, m)))
        r = run_policy(sim_by_qm, acc_by_qm, top_k, n_reps=1)
        print(f"{k:>3}  {r['ratio']:>6.3f}  [{r['ci_lo']:>.2f},{r['ci_hi']:>5.2f}]  "
              f"{r['adm_acc']*100:>6.2f}%  {r['rej_acc']*100:>6.2f}%  "
              f"{r['rejection_rate']*100:>7.2f}%  {r['n_admissible']:>6}  {r['n_rejected']:>6}")
        rows_out.append({"policy": f"top_{k}", "k": k, **r})

    print()
    print("--- Random k-of-6 (50 reps; mean +/- std across reps) ---")
    header2 = f"{'k':>3}  {'ratio_mean':>10}  {'std':>5}  {'min':>5}  {'max':>5}  {'rej_rate':>8}"
    print(header2)
    for k in [3, 4, 5]:
        def rand_k(rng, m, k=k):
            return sorted(rng.choice(m, size=k, replace=False).tolist())
        r = run_policy(sim_by_qm, acc_by_qm, rand_k, n_reps=N_RANDOM_REPS, seed_base=RNG_SEED)
        print(f"{k:>3}  {r['ratio_mean']:>10.3f}  {r['ratio_std']:>5.3f}  "
              f"{r['ratio_min']:>5.2f}  {r['ratio_max']:>5.2f}  "
              f"{r['rejection_rate_mean']*100:>7.2f}%")
        rows_out.append({"policy": f"random_{k}_of_6", "k": k, **r})

    print()
    print("--- Bottom-k-by-log-prob (k lowest of top-6; indices (6-k)..5) ---")
    print(header)
    for k in [3, 4, 5]:
        def bot_k(rng, m, k=k):
            return list(range(m - k, m))
        r = run_policy(sim_by_qm, acc_by_qm, bot_k, n_reps=1)
        print(f"{k:>3}  {r['ratio']:>6.3f}  [{r['ci_lo']:>.2f},{r['ci_hi']:>5.2f}]  "
              f"{r['adm_acc']*100:>6.2f}%  {r['rej_acc']*100:>6.2f}%  "
              f"{r['rejection_rate']*100:>7.2f}%  {r['n_admissible']:>6}  {r['n_rejected']:>6}")
        rows_out.append({"policy": f"bottom_{k}", "k": k, **r})

    print()
    print("--- Full-6 reference (paper default) ---")
    def full6(rng, m):
        return list(range(m))
    r6 = run_policy(sim_by_qm, acc_by_qm, full6, n_reps=1)
    print(f"  k=6  ratio={r6['ratio']:.3f} [{r6['ci_lo']:.2f}, {r6['ci_hi']:.2f}]  "
          f"adm_acc={r6['adm_acc']*100:.2f}%  rej_acc={r6['rej_acc']*100:.2f}%  "
          f"rej_rate={r6['rejection_rate']*100:.2f}%")
    rows_out.append({"policy": "full_6", "k": 6, **r6})

    out = {
        "dataset": "triviaqa",
        "models": MODELS,
        "thresholds": THRESHOLDS,
        "n_random_reps": N_RANDOM_REPS,
        "results": rows_out,
    }
    with open(script_dir / "retention_policy_ablation_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print()
    print(f"Saved to {script_dir/'retention_policy_ablation_results.json'}")


if __name__ == "__main__":
    main()
