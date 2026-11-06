"""
Goldilocks robustness check under external-to-accuracy difficulty proxies.

R1's concern (W7): stratifying by per-question correct-count X/6 conditions on
the outcome, so the "3-4/6 peak, 1-2/6 inversion" pattern is partly tautological.

This script restratifies TriviaQA questions by proxies that do NOT use K/6
correctness counts:

  1. SNNE quintile (soft nearest-neighbor entropy; uses same similarity matrix
     as admissibility but aggregates differently)
  2. SE(theta=0.75) quintile (semantic entropy at a fixed threshold)
  3. Mean pairwise NLI similarity (purely structural, not accuracy-based)

For each stratifier, we compute admissibility ratio (adm_acc / rej_acc) within
each bin and check whether the Goldilocks mid-bin peak and extreme-bin
attenuation/inversion pattern replicates.

Usage:
    ~/ip_llm/bin/python experiments/goldilocks_external_proxy.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load_triviaqa(script_dir):
    with open(script_dir / "kle_snne_triviaqa_results.json") as f:
        records = json.load(f)
    return [r for r in records if r["dataset"] == "triviaqa"]


def mean_pairwise_similarity(sim_matrix):
    """Mean off-diagonal similarity of the response set."""
    n = sim_matrix.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    return float(sim_matrix[iu].mean())


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


def quintile_bin(values, n_bins=5):
    """Rank-based equal-frequency binning; ties broken by stable sort order."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    # Stable ranks, random tiebreak so identical values spread across bins
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-1e-9, 1e-9, n)
    ranks = np.argsort(values + jitter)
    bins = np.empty(n, dtype=int)
    for rank_pos, idx in enumerate(ranks):
        bins[idx] = min(rank_pos * n_bins // n, n_bins - 1)
    return bins


def per_response_ratio(records_in_bin):
    """Admissible vs rejected accuracy ratio in a bin, with bootstrap CI."""
    ac = at = rc = rt = 0
    for r in records_in_bin:
        accs = r["accuracies"]
        n_adm = r["n_admissible"]
        # admissible responses are first n_adm entries? Actually we don't have
        # explicit indices. However the fields n_admissible and n_rejected
        # sum to M=6 and adm_mv_correct tells us MV over admissible set. To get
        # per-response admissible accuracy, we need per-question admissibility
        # labels which aren't in this file. Approximation: use adm_mv_correct
        # as proxy for admissible-set quality. BUT we want per-response, so
        # we need the partition itself.
        # Use the fact that per-response accuracies + knowledge of partition
        # indices gives us acc. We don't have partition indices, so approximate.
        # Instead use: assume admissible set contains the n_adm most-accurate
        # responses (this is an upper bound). For a fair analysis, we need
        # the actual partition.
        # SIMPLER: use counts. admissible_correct = min(n_adm, sum(accs));
        # this is not correct either.
        # Correct approach: load similarity matrix and recompute partition.
        pass
    # This approach is broken. Use the approach below instead.
    raise NotImplementedError


def recompute_partition(sim_matrix, thresholds=None):
    """Return (admissible_indices, rejected_indices) for a similarity matrix."""
    from kle_snne_comparison import _cluster
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    m = sim_matrix.shape[0]
    admissible = set()
    for t in thresholds:
        clusters = _cluster(sim_matrix, t)
        if not clusters:
            continue
        sizes = [len(c) for c in clusters]
        mx = max(sizes)
        for c, s in zip(clusters, sizes):
            if s == mx:
                admissible |= c
    rejected = set(range(m)) - admissible
    return sorted(admissible), sorted(rejected)


def compute_ratio(records, sims_by_key):
    """For a list of records, compute pooled admissible vs rejected accuracy ratio."""
    ac = at = rc = rt = 0
    for r in records:
        key = (r["model"], r["qid"])
        if key not in sims_by_key:
            continue
        sim = sims_by_key[key]
        adm_idx, rej_idx = recompute_partition(sim)
        accs = r["accuracies"]
        for i in adm_idx:
            at += 1
            ac += accs[i]
        for i in rej_idx:
            rt += 1
            rc += accs[i]
    adm_acc = ac / at if at > 0 else float("nan")
    rej_acc = rc / rt if rt > 0 else float("nan")
    if rt == 0:
        ratio = float("inf")  # no rejected responses in this bin
    elif rej_acc == 0:
        ratio = float("inf") if ac > 0 else float("nan")
    else:
        ratio = adm_acc / rej_acc
    # Fisher's exact
    if at > 0 and rt > 0:
        try:
            _, p = stats.fisher_exact([[ac, at - ac], [rc, rt - rc]])
        except Exception:
            p = float("nan")
    else:
        p = float("nan")
    return {"ac": ac, "at": at, "rc": rc, "rt": rt,
            "adm_acc": adm_acc, "rej_acc": rej_acc, "ratio": ratio, "p": p}


def stratify_by_proxy(records, sims_by_key, proxy_key, n_bins=5):
    """Bin records by proxy, return per-bin ratios."""
    # Compute proxy value per record
    proxied = []
    for r in records:
        if proxy_key == "snne":
            proxied.append((r["snne"], r))
        elif proxy_key == "se_075":
            proxied.append((r["se_075"], r))
        elif proxy_key == "mean_sim":
            key = (r["model"], r["qid"])
            if key in sims_by_key:
                proxied.append((mean_pairwise_similarity(sims_by_key[key]), r))
        elif proxy_key == "k_correct":
            proxied.append((sum(r["accuracies"]), r))
        else:
            raise ValueError(proxy_key)

    values = np.array([p[0] for p in proxied])
    bins = quintile_bin(values, n_bins)

    out = {}
    for b in range(n_bins):
        bin_recs = [rec for (val, rec), bi in zip(proxied, bins) if bi == b]
        if not bin_recs:
            out[b] = None
            continue
        res = compute_ratio(bin_recs, sims_by_key)
        if res is not None:
            res["n_q"] = len(bin_recs)
            res["bin_range"] = (float(min(p[0] for p, bi in zip(proxied, bins) if bi == b)),
                                float(max(p[0] for p, bi in zip(proxied, bins) if bi == b)))
        out[b] = res
    return out


def print_table(stratified, proxy_name, direction="high=hard"):
    print(f"\n  Stratifier: {proxy_name}  ({direction})")
    print(f"  {'Bin':<5} {'Range':<20} {'n_Q':>5} {'n_adm':>6} {'n_rej':>6} "
          f"{'AdmAcc':>7} {'RejAcc':>7} {'Ratio':>8} {'Fisher p':>10}")
    print("  " + "-" * 90)
    for b in sorted(stratified.keys()):
        r = stratified[b]
        if r is None:
            print(f"  {b:<5}  (empty)")
            continue
        rng = f"[{r['bin_range'][0]:.2f}, {r['bin_range'][1]:.2f}]"
        adm_acc_s = f"{r['adm_acc']:.1%}" if not np.isnan(r['adm_acc']) else "---"
        rej_acc_s = f"{r['rej_acc']:.1%}" if not np.isnan(r['rej_acc']) else "---"
        if np.isinf(r['ratio']):
            ratio_s = "  inf"
        elif np.isnan(r['ratio']):
            ratio_s = "  nan"
        else:
            ratio_s = f"{r['ratio']:.2f}x"
        p_s = f"{r['p']:.3g}" if not np.isnan(r['p']) else "---"
        print(f"  {b:<5} {rng:<20} {r['n_q']:>5} {r['at']:>6} {r['rt']:>6} "
              f"{adm_acc_s:>7} {rej_acc_s:>7} {ratio_s:>8} {p_s:>10}")


def main():
    script_dir = Path(__file__).parent
    tqa = load_triviaqa(script_dir)
    print(f"Loaded {len(tqa)} TriviaQA records")

    sims_by_key = load_sim_caches(script_dir)
    print(f"Loaded similarity caches for {len(sims_by_key)} (model, qid) pairs")

    # Sanity: reference stratification by K/6 (the paper's existing approach)
    print("\n" + "=" * 95)
    print("  BASELINE: stratify by K/6 correct-count (the paper's existing Goldilocks stratification)")
    print("=" * 95)

    # Custom binning by K = 0,1,2,3,4,5,6 with 3-4 merged as the 'Goldilocks' peak
    k_bins = {"0/6": [], "1-2/6": [], "3-4/6": [], "5-6/6": []}
    for r in tqa:
        k = sum(r["accuracies"])
        if k == 0:
            k_bins["0/6"].append(r)
        elif k <= 2:
            k_bins["1-2/6"].append(r)
        elif k <= 4:
            k_bins["3-4/6"].append(r)
        else:
            k_bins["5-6/6"].append(r)

    print(f"  {'Bin':<8} {'n_Q':>5} {'n_adm':>6} {'n_rej':>6} "
          f"{'AdmAcc':>7} {'RejAcc':>7} {'Ratio':>8}")
    print("  " + "-" * 65)
    for bin_name in ["0/6", "1-2/6", "3-4/6", "5-6/6"]:
        res = compute_ratio(k_bins[bin_name], sims_by_key)
        if res is None:
            print(f"  {bin_name:<8} {len(k_bins[bin_name]):>5}  (no data)")
            continue
        print(f"  {bin_name:<8} {len(k_bins[bin_name]):>5} {res['at']:>6} {res['rt']:>6} "
              f"{res['adm_acc']:>6.1%} {res['rej_acc']:>6.1%} {res['ratio']:>7.2f}x")

    # External proxies
    print("\n" + "=" * 95)
    print("  EXTERNAL PROXIES: restratify by signals not derived from per-question correctness")
    print("=" * 95)

    for proxy, direction in [
        ("snne", "high=high-entropy=hard"),
        ("se_075", "high=high-entropy=hard"),
        ("mean_sim", "high=responses-agree=easy"),
    ]:
        out = stratify_by_proxy(tqa, sims_by_key, proxy, n_bins=5)
        print_table(out, proxy, direction)

    # Save
    all_out = {"k_6": {}, "snne": {}, "se_075": {}, "mean_sim": {}}
    for bin_name, recs in k_bins.items():
        all_out["k_6"][bin_name] = compute_ratio(recs, sims_by_key)
    for proxy in ["snne", "se_075", "mean_sim"]:
        all_out[proxy] = stratify_by_proxy(tqa, sims_by_key, proxy, n_bins=5)

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

    with open(script_dir / "goldilocks_external_proxy_results.json", "w") as f:
        json.dump(_clean(all_out), f, indent=2)
    print(f"\nSaved to experiments/goldilocks_external_proxy_results.json")


if __name__ == "__main__":
    main()
