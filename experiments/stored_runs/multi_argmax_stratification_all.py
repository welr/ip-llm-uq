"""
Multi-argmax stratification, extended to ARC and MMLU (beyond TriviaQA).

For each dataset x model, we use cached similarity matrices and per-question
accuracy records to compute:

  1. Stratify questions by num_distinct O*_theta across the threshold grid
     (single-argmax: every theta gives same cluster; multi-argmax: >=2)
  2. On each stratum, compute admissibility adm/rej ratio AND single-theta=0.75
     plurality adm/rej ratio on the same question subset
  3. Report per-dataset pooled result and the multi-threshold margin
     (admissibility_ratio - single_theta_ratio on the multi-argmax stratum)

Datasets available with cached sim matrices:
  - triviaqa  (4 models)
  - arc       (4 models)
  - mmlu      (Gemma only)

TruthfulQA and NQ-Open would require per-question sim matrices not in sim_cache/.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

from kle_snne_comparison import THRESHOLDS, _cluster


def argmax_cluster_set(sim_matrix, theta):
    clusters = _cluster(sim_matrix, theta)
    sizes = [len(c) for c in clusters]
    mx = max(sizes)
    out = set()
    for c, s in zip(clusters, sizes):
        if s == mx:
            out |= c
    return frozenset(out)


def compute_per_question(sim_matrix):
    m = sim_matrix.shape[0]
    O_sets = [argmax_cluster_set(sim_matrix, t) for t in THRESHOLDS]
    num_distinct = len(set(O_sets))
    admissible = set().union(*O_sets)
    rejected = set(range(m)) - admissible
    s075 = argmax_cluster_set(sim_matrix, 0.75)
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
    if not ratios:
        return None, None
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


DATASETS = [
    ("triviaqa", "kle_snne_triviaqa_results.json", ["Qwen","Mistral","Llama","Gemma"]),
    ("arc",      "kle_snne_results.json",          ["Qwen","Mistral","Llama","Gemma"]),
    ("mmlu",     "kle_snne_results.json",          ["Gemma"]),
]


def run_dataset(dataset, records_file, models, sim_cache_dir):
    records = json.load(open(records_file))
    records = [r for r in records if r["dataset"] == dataset]
    by_key = {(r["model"], r["qid"]): r for r in records}

    strata = {
        s: {"ac":0,"at":0,"rc":0,"rt":0,"n_q":0,
            "s075_ac":0,"s075_at":0,"s075_rc":0,"s075_rt":0}
        for s in ["single_argmax","multi_argmax"]
    }

    for model in models:
        sim_file = sim_cache_dir / f"sim_matrices_{dataset}_{model}.json"
        if not sim_file.exists():
            print(f"  skipping {dataset}/{model}: no sim cache")
            continue
        sims = json.load(open(sim_file))
        for qid, sim_list in sims.items():
            rec = by_key.get((model, qid))
            if rec is None:
                continue
            sim_matrix = np.array(sim_list)
            accs = rec["accuracies"]
            res = compute_per_question(sim_matrix)
            stratum = "single_argmax" if res["num_distinct_Ostar"] == 1 else "multi_argmax"
            v = strata[stratum]
            v["n_q"] += 1
            v["ac"] += sum(accs[i] for i in res["admissible"])
            v["at"] += len(res["admissible"])
            v["rc"] += sum(accs[i] for i in res["rejected"])
            v["rt"] += len(res["rejected"])
            v["s075_ac"] += sum(accs[i] for i in res["single075_admissible"])
            v["s075_at"] += len(res["single075_admissible"])
            v["s075_rc"] += sum(accs[i] for i in res["single075_rejected"])
            v["s075_rt"] += len(res["single075_rejected"])

    return strata


def main():
    script_dir = Path(__file__).parent
    sim_cache_dir = script_dir / "sim_cache"

    print("="*110)
    print("  MULTI-ARGMAX STRATIFICATION ACROSS DATASETS")
    print("  Admissibility vs single-theta=0.75 plurality on the multi-argmax stratum")
    print("="*110)

    all_results = {}
    for dataset, rfile, models in DATASETS:
        print(f"\n{'#'*70}\n### {dataset.upper()}  (models: {', '.join(models)})\n{'#'*70}")
        strata = run_dataset(dataset, script_dir/rfile, models, sim_cache_dir)
        all_results[dataset] = strata

        print(f"\n  {'Stratum':<20} {'n_Q':>5} {'frac':>6} "
              f"{'Adm ratio':>12} {'SE(0.75) ratio':>16} {'Margin':>8}")
        print("  " + "-"*85)
        total_q = sum(v["n_q"] for v in strata.values())
        for name, label in [("single_argmax","1 distinct O*"),("multi_argmax",">=2 distinct O*")]:
            v = strata[name]
            if v["at"] == 0 or v["rt"] == 0 or v["n_q"] == 0:
                print(f"  {label:<20} {v['n_q']:>5}   (insufficient data)")
                continue
            frac = v["n_q"]/total_q
            aa, ra = v["ac"]/v["at"], v["rc"]/v["rt"] if v["rt"]>0 else 0
            r_multi = aa/ra if ra > 0 else float("inf")
            lo, hi = boot_ci(v["ac"], v["at"], v["rc"], v["rt"])
            ci_str = f"[{lo:.2f},{hi:.2f}]" if lo is not None else ""
            s_aa = v["s075_ac"]/v["s075_at"] if v["s075_at"]>0 else 0
            s_ra = v["s075_rc"]/v["s075_rt"] if v["s075_rt"]>0 else 0
            r_s075 = s_aa/s_ra if s_ra > 0 else float("inf")
            margin = (r_multi - r_s075) / r_s075 if r_s075 and np.isfinite(r_s075) else float("nan")
            r_multi_str = f"{r_multi:.2f}x {ci_str}"
            r_s075_str = f"{r_s075:.2f}x" if np.isfinite(r_s075) else "undef"
            margin_str = f"{margin:+.1%}" if np.isfinite(margin) else "---"
            print(f"  {label:<20} {v['n_q']:>5} {frac:>5.1%} "
                  f"{r_multi_str:>12} {r_s075_str:>16} {margin_str:>8}")

            if name == "multi_argmax":
                # Fisher on admissibility
                _, p = stats.fisher_exact([[v["ac"], v["at"]-v["ac"]],
                                           [v["rc"], v["rt"]-v["rc"]]])
                print(f"    Admissibility Fisher p = {p:.3g};  "
                      f"n_adm={v['at']}, n_rej={v['rt']}")

    # Save
    def clean(x):
        if isinstance(x, dict): return {k: clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [clean(v) for v in x]
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        return x
    out_file = script_dir / "multi_argmax_stratification_all_results.json"
    with open(out_file, "w") as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"\nSaved to {out_file.name}")


if __name__ == "__main__":
    main()
