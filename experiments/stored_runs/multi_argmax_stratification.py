"""
Multi-argmax stratification (reviewer Q1, peer-review response).

The grid-sparsity diagnostic (grid_sparsity.py, Appendix C) shows that on
TriviaQA, 83.6% of questions have every threshold in {0.50, 0.55, ..., 0.95}
give the same argmax cluster O*_theta; admissibility on those questions
reduces exactly to single-threshold argmax.

The peer-review question: on the 16.4% of TriviaQA questions where at
least two distinct O*_theta sets appear across the grid (i.e., where the
multi-threshold sweep is actually doing work), does admissibility still
discriminate correct from incorrect responses? This isolates the
choice-function contribution from single-threshold plurality.

Outputs:
  - admissible vs rejected accuracy ratio, stratified by
    num_distinct_Ostar in {1, >=2}
  - comparison to single-theta (theta=0.75) plurality on the same
    stratification
  - per-model breakdown

Usage:
    ~/ip_llm/bin/python experiments/multi_argmax_stratification.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from kle_snne_comparison import THRESHOLDS, _cluster


def argmax_cluster_set(sim_matrix, theta):
    m = sim_matrix.shape[0]
    clusters = _cluster(sim_matrix, theta)
    sizes = [len(c) for c in clusters]
    mx = max(sizes)
    out = set()
    for c, s in zip(clusters, sizes):
        if s == mx:
            out |= c
    return frozenset(out)


def single_threshold_plurality(sim_matrix, theta):
    """Return the single-theta argmax cluster as admissible, rest rejected."""
    return argmax_cluster_set(sim_matrix, theta)


def compute_per_question(sim_matrix, accuracies):
    """Return dict with:
      num_distinct_Ostar
      admissible_indices, rejected_indices   (for multi-theta union)
      single075_admissible, single075_rejected  (for theta=0.75 only)
    """
    m = sim_matrix.shape[0]
    O_sets = [argmax_cluster_set(sim_matrix, t) for t in THRESHOLDS]
    num_distinct = len(set(O_sets))
    admissible = set().union(*O_sets)
    rejected = set(range(m)) - admissible

    # Single-threshold plurality at theta=0.75 (the paper's reference theta)
    s075 = single_threshold_plurality(sim_matrix, 0.75)
    s075_rej = set(range(m)) - s075

    return {
        "num_distinct_Ostar": num_distinct,
        "admissible": sorted(admissible),
        "rejected": sorted(rejected),
        "single075_admissible": sorted(s075),
        "single075_rejected": sorted(s075_rej),
    }


def boot_ci(ac, at, rc, rt, n=10000, seed=42):
    if at == 0 or rt == 0 or rc == 0:
        return None, None
    rng = np.random.default_rng(seed)
    adm = np.array([1]*ac + [0]*(at-ac))
    rej = np.array([1]*rc + [0]*(rt-rc))
    ratios = []
    for _ in range(n):
        a = rng.choice(adm, size=len(adm), replace=True).mean()
        r = rng.choice(rej, size=len(rej), replace=True).mean()
        if r > 0:
            ratios.append(a/r)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def accumulate(per_q, ac_acc, at, rc_acc, rt):
    return ac_acc + sum(per_q["accuracies"][i] for i in per_q["adm_idx"]), \
           at + len(per_q["adm_idx"]), \
           rc_acc + sum(per_q["accuracies"][i] for i in per_q["rej_idx"]), \
           rt + len(per_q["rej_idx"])


def main():
    script_dir = Path(__file__).parent
    sim_cache_dir = script_dir / "sim_cache"

    # Load TriviaQA per-question accuracy records (from kle_snne comparison)
    records = json.load(open(script_dir / "kle_snne_triviaqa_results.json"))
    records = [r for r in records if r["dataset"] == "triviaqa"]

    by_key = {(r["model"], r["qid"]): r for r in records}

    # For each TriviaQA model, load sim cache and compute per-question admissibility
    model_keys = ["Qwen", "Mistral", "Llama", "Gemma"]
    # Bins: {1 distinct Ostar, >=2 distinct Ostar}
    strata = {
        "single_argmax":  {"ac":0,"at":0,"rc":0,"rt":0,"n_q":0,
                           "s075_ac":0,"s075_at":0,"s075_rc":0,"s075_rt":0},
        "multi_argmax":   {"ac":0,"at":0,"rc":0,"rt":0,"n_q":0,
                           "s075_ac":0,"s075_at":0,"s075_rc":0,"s075_rt":0},
    }
    per_model = {m: {"single":{"ac":0,"at":0,"rc":0,"rt":0,"n_q":0},
                     "multi": {"ac":0,"at":0,"rc":0,"rt":0,"n_q":0}} for m in model_keys}

    for model in model_keys:
        sim_file = sim_cache_dir / f"sim_matrices_triviaqa_{model}.json"
        sims = json.load(open(sim_file))
        for qid, sim_list in sims.items():
            rec = by_key.get((model, qid))
            if rec is None:
                continue
            sim_matrix = np.array(sim_list)
            accs = rec["accuracies"]
            res = compute_per_question(sim_matrix, accs)
            stratum = "single_argmax" if res["num_distinct_Ostar"] == 1 else "multi_argmax"
            adm_idx = res["admissible"]
            rej_idx = res["rejected"]
            s075_adm = res["single075_admissible"]
            s075_rej = res["single075_rejected"]

            strata[stratum]["n_q"] += 1
            strata[stratum]["ac"] += sum(accs[i] for i in adm_idx)
            strata[stratum]["at"] += len(adm_idx)
            strata[stratum]["rc"] += sum(accs[i] for i in rej_idx)
            strata[stratum]["rt"] += len(rej_idx)
            strata[stratum]["s075_ac"] += sum(accs[i] for i in s075_adm)
            strata[stratum]["s075_at"] += len(s075_adm)
            strata[stratum]["s075_rc"] += sum(accs[i] for i in s075_rej)
            strata[stratum]["s075_rt"] += len(s075_rej)

            # Per-model
            pm = per_model[model]["single" if stratum == "single_argmax" else "multi"]
            pm["n_q"] += 1
            pm["ac"] += sum(accs[i] for i in adm_idx)
            pm["at"] += len(adm_idx)
            pm["rc"] += sum(accs[i] for i in rej_idx)
            pm["rt"] += len(rej_idx)

    # Report
    print("="*110)
    print("  TRIVIAQA MULTI-ARGMAX STRATIFICATION (pooled across 4 base models)")
    print("="*110)
    print(f"  {'Stratum':<16} {'n_Q':>6} {'n_adm':>7} {'n_rej':>7} {'AdmAcc':>7} {'RejAcc':>7} {'Ratio':>8} {'95% CI':>15} {'Fisher p':>12}")
    print("  " + "-"*100)
    for name, label in [("single_argmax","1 distinct O*"),("multi_argmax",">=2 distinct O*")]:
        v = strata[name]
        if v["at"] == 0 or v["rt"] == 0:
            print(f"  {label:<16} {v['n_q']:>6} {v['at']:>7} {v['rt']:>7}   (undefined)")
            continue
        aa = v["ac"]/v["at"]; ra = v["rc"]/v["rt"]
        ratio = aa/ra if ra > 0 else float("inf")
        lo, hi = boot_ci(v["ac"], v["at"], v["rc"], v["rt"])
        _, p = stats.fisher_exact([[v["ac"], v["at"]-v["ac"]],[v["rc"], v["rt"]-v["rc"]]])
        ci_str = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "---"
        print(f"  {label:<16} {v['n_q']:>6} {v['at']:>7} {v['rt']:>7} {aa:>6.1%} {ra:>6.1%} {ratio:>7.2f}x {ci_str:>15} {p:>12.3g}")

    # Compare to single-theta=0.75 plurality on SAME stratification
    print()
    print("="*110)
    print("  COMPARISON: admissibility vs single-theta=0.75 plurality, same stratification")
    print("="*110)
    for name, label in [("single_argmax","1 distinct O*"),("multi_argmax",">=2 distinct O*")]:
        v = strata[name]
        print(f"\n  Stratum: {label}  (n_Q={v['n_q']})")
        # Multi-theta
        aa = v["ac"]/v["at"]; ra = v["rc"]/v["rt"] if v["rt"]>0 else 0
        r_multi = aa/ra if ra > 0 else float("inf")
        # Single theta=0.75
        s075_aa = v["s075_ac"]/v["s075_at"] if v["s075_at"]>0 else 0
        s075_ra = v["s075_rc"]/v["s075_rt"] if v["s075_rt"]>0 else 0
        r_s075 = s075_aa/s075_ra if s075_ra > 0 else float("inf")
        print(f"    multi-theta admissibility:  adm {v['at']:>5}, rej {v['rt']:>4}, "
              f"{aa:>6.1%} / {ra:>6.1%} = {r_multi:>5.2f}x")
        print(f"    single-theta=0.75 plurality: adm {v['s075_at']:>5}, rej {v['s075_rt']:>4}, "
              f"{s075_aa:>6.1%} / {s075_ra:>6.1%} = {r_s075:>5.2f}x")

    # Per-model breakdown on multi-argmax stratum (where the story lives)
    print()
    print("="*110)
    print("  PER-MODEL: multi-argmax stratum only (the ~16% where grid actually sweeps)")
    print("="*110)
    print(f"  {'Model':<10} {'n_Q':>6} {'AdmAcc':>7} {'RejAcc':>7} {'Ratio':>8} {'Fisher p':>12}")
    print("  " + "-"*60)
    for model in model_keys:
        pm = per_model[model]["multi"]
        if pm["at"] == 0 or pm["rt"] == 0:
            print(f"  {model:<10} {pm['n_q']:>6}   (insufficient data)")
            continue
        aa = pm["ac"]/pm["at"]; ra = pm["rc"]/pm["rt"]
        ratio = aa/ra if ra > 0 else float("inf")
        _, p = stats.fisher_exact([[pm["ac"], pm["at"]-pm["ac"]],[pm["rc"], pm["rt"]-pm["rc"]]])
        print(f"  {model:<10} {pm['n_q']:>6} {aa:>6.1%} {ra:>6.1%} {ratio:>7.2f}x {p:>12.3g}")

    # Save
    out = {"strata": strata, "per_model": per_model}
    # Convert int types for JSON
    def clean(x):
        if isinstance(x, dict): return {k: clean(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        return x
    with open(script_dir / "multi_argmax_stratification_results.json", "w") as f:
        json.dump(clean(out), f, indent=2)
    print(f"\nSaved to experiments/multi_argmax_stratification_results.json")


if __name__ == "__main__":
    main()
