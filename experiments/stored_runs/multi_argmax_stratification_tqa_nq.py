"""
Multi-argmax stratification for TruthfulQA and NQ-Open.

Uses sim_cache (built by build_sim_cache_tqa_nq.py) + modal_data accuracies.
Mirrors multi_argmax_stratification_all.py for TriviaQA/ARC/MMLU, then combines
all 5 datasets into a single summary table.

For TruthfulQA, uses the pre-registered SUBSTRING correctness labels (the
primary scorer for H1-H6). Judge rescoring is scope-dependent and lives in
Appendix H.
"""

import glob
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


SCRIPT_DIR = Path(__file__).parent
SIM_CACHE = SCRIPT_DIR / "sim_cache"
MODAL = SCRIPT_DIR / "modal_data"

BASE_MODELS = ["Qwen", "Mistral", "Llama", "Gemma"]


def modal_data_file(dataset, model):
    pattern = f"prereg_{dataset}_{model}_*.json"
    candidates = sorted(MODAL.glob(pattern))
    # glob safely excludes LlamaInstruct / Gemma4MoE (no underscore after model name in those)
    return candidates[0] if candidates else None


def accuracies_from_modal(dataset, model):
    f = modal_data_file(dataset, model)
    if f is None:
        return None
    d = json.load(open(f))
    return {qid: [int(a) for a in v["accuracies"]]
            for qid, v in d["per_question"].items()}


def run_dataset(dataset):
    strata = {
        s: {"ac":0,"at":0,"rc":0,"rt":0,"n_q":0,
            "s075_ac":0,"s075_at":0,"s075_rc":0,"s075_rt":0}
        for s in ["single_argmax","multi_argmax"]
    }
    for model in BASE_MODELS:
        sim_file = SIM_CACHE / f"sim_matrices_{dataset}_{model}.json"
        if not sim_file.exists():
            print(f"  skip {dataset}/{model}: no sim cache")
            continue
        acc_by_q = accuracies_from_modal(dataset, model)
        if acc_by_q is None:
            print(f"  skip {dataset}/{model}: no modal data")
            continue
        sims = json.load(open(sim_file))
        for qid, sim_list in sims.items():
            accs = acc_by_q.get(qid)
            if accs is None:
                continue
            sim_matrix = np.array(sim_list)
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


def report(dataset, strata):
    print(f"\n{'#'*70}\n### {dataset.upper()}\n{'#'*70}")
    print(f"  {'Stratum':<20} {'n_Q':>5} {'frac':>6} "
          f"{'Adm ratio':>20} {'SE(0.75)':>10} {'Margin':>8} {'Fisher p':>10}")
    print("  " + "-"*90)
    total = sum(v["n_q"] for v in strata.values())
    out_rows = {}
    for name, label in [("single_argmax","1 distinct O*"),("multi_argmax",">=2 distinct O*")]:
        v = strata[name]
        if v["at"] == 0 or v["rt"] == 0 or v["n_q"] == 0:
            print(f"  {label:<20} {v['n_q']:>5}   (insufficient data)")
            continue
        frac = v["n_q"]/total
        aa, ra = v["ac"]/v["at"], v["rc"]/v["rt"] if v["rt"]>0 else 0
        r_multi = aa/ra if ra > 0 else float("inf")
        lo, hi = boot_ci(v["ac"], v["at"], v["rc"], v["rt"])
        ci = f"[{lo:.2f},{hi:.2f}]" if lo is not None else ""
        s_aa = v["s075_ac"]/v["s075_at"] if v["s075_at"]>0 else 0
        s_ra = v["s075_rc"]/v["s075_rt"] if v["s075_rt"]>0 else 0
        r_s075 = s_aa/s_ra if s_ra > 0 else float("inf")
        margin = (r_multi - r_s075) / r_s075 if r_s075 and np.isfinite(r_s075) else float("nan")
        _, p = stats.fisher_exact([[v["ac"], v["at"]-v["ac"]],
                                   [v["rc"], v["rt"]-v["rc"]]])
        m_str = f"{r_multi:.2f}x {ci}"
        print(f"  {label:<20} {v['n_q']:>5} {frac:>5.1%} "
              f"{m_str:>20} {r_s075:>9.2f}x {margin:>+7.1%} {p:>10.3g}")
        out_rows[name] = {
            "n_Q": v["n_q"], "frac": frac,
            "adm_ratio": r_multi, "adm_ratio_CI": [lo, hi],
            "se075_ratio": r_s075, "margin": margin, "fisher_p": p,
            "n_adm": v["at"], "n_rej": v["rt"],
        }
    return out_rows


def main():
    print("="*100)
    print("  MULTI-ARGMAX STRATIFICATION: TruthfulQA + NQ-Open")
    print("  (TruthfulQA uses pre-registered substring scorer)")
    print("="*100)

    all_out = {}
    for dataset in ["truthfulqa", "nq_open"]:
        strata = run_dataset(dataset)
        all_out[dataset] = report(dataset, strata)

    out_file = SCRIPT_DIR / "multi_argmax_stratification_tqa_nq_results.json"
    with open(out_file, "w") as f:
        json.dump(all_out, f, indent=2, default=float)
    print(f"\nSaved to {out_file.name}")


if __name__ == "__main__":
    main()
