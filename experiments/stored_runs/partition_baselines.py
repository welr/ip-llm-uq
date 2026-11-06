"""
Partition-level baselines: what does the IP union structure buy over averaging
and over single-theta plurality?

Two analyses on TriviaQA (the headline benchmark, 4.1% admissibility rejection):

(1) Averaged-signal partition: for each response r, compute
        score(r) = 1 - (1/|Theta|) sum_theta (cluster_size(r, theta) / M)
    i.e., 1 minus the response's mean cluster-fraction across thresholds.
    Reject the bottom 4.1% of responses by this score (matched to admissibility).
    If this partition's discrimination ratio approaches 2.94x, averaging alone
    recovers admissibility's gain on the partition side (not just the signal side).

(2) Single-theta plurality Pareto frontier: for each theta in {0.50, ..., 0.95},
    treat "in the argmax cluster at theta" as admissible and the rest as rejected.
    Plot (rejection_rate, accuracy_ratio) for each theta and overlay admissibility.

Usage:
    ~/ip_llm/bin/python experiments/partition_baselines.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

from kle_snne_comparison import _cluster

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_triviaqa(script_dir):
    with open(script_dir / "kle_snne_triviaqa_results.json") as f:
        records = json.load(f)
    return [r for r in records if r["dataset"] == "triviaqa"]


def load_sim_caches(script_dir):
    cache_dir = script_dir / "sim_cache"
    out = {}
    for fn in cache_dir.glob("sim_matrices_triviaqa_*.json"):
        model = fn.stem.split("_")[-1]
        with open(fn) as f:
            sims = json.load(f)
        for qid, sim_list in sims.items():
            out[(model, qid)] = np.array(sim_list)
    return out


def admissibility_partition(sim_matrix):
    m = sim_matrix.shape[0]
    adm = set()
    for t in THRESHOLDS:
        clusters = _cluster(sim_matrix, t)
        if not clusters:
            continue
        sizes = [len(c) for c in clusters]
        mx = max(sizes)
        for c, s in zip(clusters, sizes):
            if s == mx:
                adm |= c
    return sorted(adm), sorted(set(range(m)) - adm)


def single_theta_partition(sim_matrix, theta):
    m = sim_matrix.shape[0]
    clusters = _cluster(sim_matrix, theta)
    if not clusters:
        return list(range(m)), []
    sizes = [len(c) for c in clusters]
    mx = max(sizes)
    adm = set()
    for c, s in zip(clusters, sizes):
        if s == mx:
            adm |= c
    return sorted(adm), sorted(set(range(m)) - adm)


def averaged_cluster_fraction_score(sim_matrix):
    """Per-response score: 1 - mean across theta of (response's cluster size / M).
    Low cluster fraction -> less "central" -> more likely incorrect."""
    m = sim_matrix.shape[0]
    cluster_frac = np.zeros(m)
    for t in THRESHOLDS:
        clusters = _cluster(sim_matrix, t)
        if not clusters:
            continue
        for c in clusters:
            frac = len(c) / m
            for i in c:
                cluster_frac[i] += frac
    cluster_frac /= len(THRESHOLDS)
    return 1.0 - cluster_frac  # higher = more rejected-like


def averaged_argmax_fraction_score(sim_matrix):
    """Per-response score: 1 - mean across theta of (response is in argmax cluster).
    Admissibility rejects iff score = 1 (never in argmax).
    This baseline tests: does thresholding on mean_theta(in_argmax) at matched
    rejection rate match admissibility's discrimination?"""
    m = sim_matrix.shape[0]
    in_argmax = np.zeros(m)
    for t in THRESHOLDS:
        clusters = _cluster(sim_matrix, t)
        if not clusters:
            continue
        sizes = [len(c) for c in clusters]
        mx = max(sizes)
        argmax_members = set()
        for c, s in zip(clusters, sizes):
            if s == mx:
                argmax_members |= c
        for i in argmax_members:
            in_argmax[i] += 1
    in_argmax /= len(THRESHOLDS)
    return 1.0 - in_argmax


def ratio_stats(ac, at, rc, rt):
    if at == 0:
        adm_acc = float("nan")
    else:
        adm_acc = ac / at
    if rt == 0:
        rej_acc = float("nan")
        ratio = float("inf")
    elif rc == 0:
        rej_acc = 0.0
        ratio = float("inf") if ac > 0 else float("nan")
    else:
        rej_acc = rc / rt
        ratio = adm_acc / rej_acc
    if at > 0 and rt > 0:
        try:
            _, p = stats.fisher_exact([[ac, at - ac], [rc, rt - rc]])
        except Exception:
            p = float("nan")
    else:
        p = float("nan")
    return adm_acc, rej_acc, ratio, p


def boot_ci(ac, at, rc, rt, n=10000, seed=42):
    if at == 0 or rt == 0 or rc == 0:
        return None, None
    rng = np.random.default_rng(seed)
    adm = np.array([1] * ac + [0] * (at - ac))
    rej = np.array([1] * rc + [0] * (rt - rc))
    ratios = []
    for _ in range(n):
        a = rng.choice(adm, size=len(adm), replace=True).mean()
        r = rng.choice(rej, size=len(rej), replace=True).mean()
        if r > 0:
            ratios.append(a / r)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def evaluate_partition(records, sims_by_key, partition_fn, label):
    """partition_fn(record, sim_matrix) -> (adm_indices, rej_indices)."""
    ac = at = rc = rt = 0
    for r in records:
        key = (r["model"], r["qid"])
        if key not in sims_by_key:
            continue
        sim = sims_by_key[key]
        adm_idx, rej_idx = partition_fn(r, sim)
        accs = r["accuracies"]
        for i in adm_idx:
            at += 1
            ac += accs[i]
        for i in rej_idx:
            rt += 1
            rc += accs[i]
    aa, ra, ratio, p = ratio_stats(ac, at, rc, rt)
    lo, hi = boot_ci(ac, at, rc, rt)
    total = at + rt
    rej_rate = rt / total if total else 0.0
    return {
        "label": label, "ac": ac, "at": at, "rc": rc, "rt": rt,
        "adm_acc": aa, "rej_acc": ra, "ratio": ratio, "p": p,
        "ci_lo": lo, "ci_hi": hi, "rej_rate": rej_rate,
    }


def print_row(res):
    ratio_s = f"{res['ratio']:.2f}x" if np.isfinite(res['ratio']) else "  inf"
    ci_s = f"[{res['ci_lo']:.2f}, {res['ci_hi']:.2f}]" if res.get('ci_lo') else "---"
    p_s = f"{res['p']:.2g}" if not np.isnan(res['p']) else "---"
    print(f"  {res['label']:<45} {res['at']:>6} {res['rt']:>6} "
          f"{res['rej_rate']*100:>5.1f}% {res['adm_acc']:>6.1%} {res['rej_acc']:>6.1%} "
          f"{ratio_s:>7} {ci_s:>17} {p_s:>8}")


def main():
    script_dir = Path(__file__).parent
    tqa = load_triviaqa(script_dir)
    sims_by_key = load_sim_caches(script_dir)
    print(f"Loaded {len(tqa)} records; {len(sims_by_key)} sim matrices.")
    print()

    # Reference: admissibility itself
    print("=" * 110)
    print("  REFERENCE: admissibility partition on TriviaQA")
    print("=" * 110)
    print(f"  {'Label':<45} {'n_adm':>6} {'n_rej':>6} {'%rej':>6} "
          f"{'AdmAcc':>6} {'RejAcc':>6} {'Ratio':>7} {'95% CI':>17} {'p':>8}")
    print("  " + "-" * 105)

    adm_res = evaluate_partition(
        tqa, sims_by_key,
        lambda r, s: admissibility_partition(s),
        "Admissibility (multi-theta union)"
    )
    print_row(adm_res)
    target_rej_rate = adm_res["rej_rate"]
    print(f"  -> target rejection rate for matched-baseline comparison: {target_rej_rate*100:.2f}%")
    print()

    # (1) Averaged-signal partitions at matched rejection rate
    def run_averaged_baseline(score_fn, label):
        all_rows = []
        for r in tqa:
            key = (r["model"], r["qid"])
            if key not in sims_by_key:
                continue
            sim = sims_by_key[key]
            scores = score_fn(sim)
            accs = r["accuracies"]
            for i in range(len(scores)):
                all_rows.append({"score": scores[i], "acc": accs[i]})
        total = len(all_rows)
        target_n_rej = int(round(target_rej_rate * total))
        all_rows.sort(key=lambda x: -x["score"])
        ac = at = rc = rt = 0
        for i, row in enumerate(all_rows):
            if i < target_n_rej:
                rt += 1
                rc += row["acc"]
            else:
                at += 1
                ac += row["acc"]
        aa, ra, ratio, p = ratio_stats(ac, at, rc, rt)
        lo, hi = boot_ci(ac, at, rc, rt)
        return {
            "label": label, "ac": ac, "at": at, "rc": rc, "rt": rt,
            "adm_acc": aa, "rej_acc": ra, "ratio": ratio, "p": p,
            "ci_lo": lo, "ci_hi": hi, "rej_rate": rt / (at + rt),
        }

    print("=" * 110)
    print("  (1) AVERAGED-SIGNAL PARTITIONS at matched rejection rate")
    print("=" * 110)
    print(f"  {'Label':<45} {'n_adm':>6} {'n_rej':>6} {'%rej':>6} "
          f"{'AdmAcc':>6} {'RejAcc':>6} {'Ratio':>7} {'95% CI':>17} {'p':>8}")
    print("  " + "-" * 105)

    avg_res = run_averaged_baseline(
        averaged_cluster_fraction_score,
        "Avg cluster-fraction (1 - mean_theta |C_r|/M)")
    print_row(avg_res)

    argmax_res = run_averaged_baseline(
        averaged_argmax_fraction_score,
        "Avg argmax-fraction (1 - mean_theta [r in O*_theta])")
    print_row(argmax_res)
    print()
    print(f"  Admissibility ratio: {adm_res['ratio']:.2f}x")
    print(f"  Avg cluster-fraction: {avg_res['ratio']:.2f}x  ({(avg_res['ratio']/adm_res['ratio']-1)*100:+.0f}% vs admissibility)")
    print(f"  Avg argmax-fraction:  {argmax_res['ratio']:.2f}x  ({(argmax_res['ratio']/adm_res['ratio']-1)*100:+.0f}% vs admissibility)")
    print()

    # (2) Single-theta Pareto frontier
    print("=" * 110)
    print("  (2) SINGLE-THETA PLURALITY PARETO FRONTIER")
    print("  For each theta in {0.50, ..., 0.95}: partition = in argmax cluster at theta.")
    print("=" * 110)
    print(f"  {'theta':<8} {'%rej':>6} {'AdmAcc':>6} {'RejAcc':>6} {'Ratio':>7} {'95% CI':>17} {'p':>8}")
    print("  " + "-" * 85)
    frontier_data = []
    for t in THRESHOLDS:
        res = evaluate_partition(
            tqa, sims_by_key,
            lambda r, s, tt=t: single_theta_partition(s, tt),
            f"theta={t:.2f}"
        )
        ratio_s = f"{res['ratio']:.2f}x" if np.isfinite(res['ratio']) else "  inf"
        ci_s = f"[{res['ci_lo']:.2f}, {res['ci_hi']:.2f}]" if res.get('ci_lo') else "---"
        p_s = f"{res['p']:.2g}" if not np.isnan(res['p']) else "---"
        print(f"  {t:<8.2f} {res['rej_rate']*100:>5.1f}% {res['adm_acc']:>6.1%} {res['rej_acc']:>6.1%} "
              f"{ratio_s:>7} {ci_s:>17} {p_s:>8}")
        frontier_data.append({"theta": t, **{k: v for k, v in res.items() if k != "label"}})

    print()
    print(f"  Admissibility operating point:  rej_rate = {adm_res['rej_rate']*100:.2f}%, ratio = {adm_res['ratio']:.2f}x")
    print(f"  Max single-theta ratio:         ", end="")
    best = max(frontier_data, key=lambda x: x["ratio"] if np.isfinite(x["ratio"]) else -1)
    print(f"theta = {best['theta']:.2f}, rej_rate = {best['rej_rate']*100:.2f}%, ratio = {best['ratio']:.2f}x")

    # Save
    def _clean(x):
        if isinstance(x, dict):
            return {str(k): _clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x

    out = {
        "admissibility": adm_res,
        "averaged_cluster_fraction": avg_res,
        "averaged_argmax_fraction": argmax_res,
        "single_theta_frontier": frontier_data,
    }
    with open(script_dir / "partition_baselines_results.json", "w") as f:
        json.dump(_clean(out), f, indent=2)
    print(f"\nSaved to experiments/partition_baselines_results.json")


if __name__ == "__main__":
    main()
