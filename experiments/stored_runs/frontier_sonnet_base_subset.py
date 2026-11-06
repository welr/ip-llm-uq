"""
Compute admissibility ratio on base-model TruthfulQA data restricted to the
same 100 seed-42 question indices as the Sonnet 4.6 exploratory run, so
Sonnet vs base-model numbers are directly comparable on the identical subset.

Uses pre-registered substring-scored accuracies from
  experiments/modal_data/prereg_truthfulqa_{Qwen,Mistral,Llama,Gemma}_*.json

Outputs:
  experiments/results/frontier_sonnet_base_subset.json
  - per-model: adm_acc, rej_acc, ratio, CI, Fisher p on 100-question subset
  - pooled across 4 base models: same stats

Usage:
    ~/ip_llm/bin/python experiments/frontier_sonnet_base_subset.py
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats


SCRIPT_DIR = Path(__file__).parent
SONNET_PATH = SCRIPT_DIR / "modal_data" / "frontier_sonnet_truthfulqa.json"
OUT_PATH = SCRIPT_DIR / "results" / "frontier_sonnet_base_subset.json"

BASE_MODELS = {
    "Qwen":    "prereg_truthfulqa_Qwen_*.json",
    "Mistral": "prereg_truthfulqa_Mistral_*.json",
    "Llama":   "prereg_truthfulqa_Llama_2*.json",
    "Gemma":   "prereg_truthfulqa_Gemma_*.json",
}


def boot_ratio_ci(ac, at, rc, rt, n_boot=10000, seed=42):
    if at == 0 or rt == 0 or rc == 0:
        return None, None
    rng = np.random.default_rng(seed)
    adm = np.array([1]*ac + [0]*(at-ac))
    rej = np.array([1]*rc + [0]*(rt-rc))
    ratios = []
    for _ in range(n_boot):
        a = rng.choice(adm, size=len(adm), replace=True).mean()
        r = rng.choice(rej, size=len(rej), replace=True).mean()
        if r > 0:
            ratios.append(a/r)
    return float(np.percentile(ratios,2.5)), float(np.percentile(ratios,97.5))


def compute_subset(pq, qids, adm_key="nli_admissible", rej_key="nli_rejected"):
    ac = at = rc = rt = 0
    missing = 0
    for qid in qids:
        if qid not in pq:
            missing += 1
            continue
        r = pq[qid]
        accs = r["accuracies"]
        for i in r[adm_key]:
            ac += int(accs[i]); at += 1
        for i in r[rej_key]:
            rc += int(accs[i]); rt += 1
    return ac, at, rc, rt, missing


def main():
    sonnet = json.load(open(SONNET_PATH))
    qids = sorted(sonnet.keys())
    print(f"Sonnet run qids: {len(qids)} (first 3: {qids[:3]})")

    out = {"qids_used": qids, "per_model_subset": {}, "pooled_subset": {}}

    # Per-model
    pooled_ac = pooled_at = pooled_rc = pooled_rt = 0
    print()
    print(f"  {'Model':<10} {'n_adm':>5} {'n_rej':>5} {'AdmAcc':>7} {'RejAcc':>7} {'Ratio':>7} {'95%CI':>14} {'p':>10}")
    print("  " + "-"*75)
    for model, pattern in BASE_MODELS.items():
        files = sorted(glob.glob(str(SCRIPT_DIR / "modal_data" / pattern)))
        if not files:
            print(f"  {model}: NO FILE FOUND FOR PATTERN {pattern}")
            continue
        f = files[-1]  # latest
        d = json.load(open(f))
        pq = d["per_question"]
        ac, at, rc, rt, missing = compute_subset(pq, qids)
        pooled_ac += ac; pooled_at += at; pooled_rc += rc; pooled_rt += rt

        adm_acc = ac/at if at else 0
        rej_acc = rc/rt if rt else 0
        ratio = adm_acc/rej_acc if rej_acc > 0 else float("inf")
        lo, hi = boot_ratio_ci(ac, at, rc, rt)
        _, p = stats.fisher_exact([[ac, at-ac], [rc, max(rt-rc,0)]])
        ci = f"[{lo:.2f},{hi:.2f}]" if lo is not None else "---"
        print(f"  {model:<10} {at:>5} {rt:>5} {adm_acc:>6.1%} {rej_acc:>6.1%} {ratio:>6.2f}x {ci:>14} {p:>10.3g}")
        out["per_model_subset"][model] = {
            "source_file": Path(f).name, "missing_qids": missing,
            "n_adm": at, "n_rej": rt, "ac": ac, "rc": rc,
            "adm_acc": adm_acc, "rej_acc": rej_acc, "ratio": ratio,
            "ci95": [lo, hi], "fisher_p": float(p),
        }

    adm_acc = pooled_ac/pooled_at if pooled_at else 0
    rej_acc = pooled_rc/pooled_rt if pooled_rt else 0
    ratio = adm_acc/rej_acc if rej_acc > 0 else float("inf")
    lo, hi = boot_ratio_ci(pooled_ac, pooled_at, pooled_rc, pooled_rt)
    _, p = stats.fisher_exact([[pooled_ac, pooled_at-pooled_ac],
                                [pooled_rc, max(pooled_rt-pooled_rc,0)]])
    print("  " + "-"*75)
    print(f"  {'POOLED':<10} {pooled_at:>5} {pooled_rt:>5} {adm_acc:>6.1%} {rej_acc:>6.1%} {ratio:>6.2f}x "
          f"[{lo:.2f},{hi:.2f}]  {p:>10.3g}")

    out["pooled_subset"] = {
        "n_adm": pooled_at, "n_rej": pooled_rt,
        "ac": pooled_ac, "rc": pooled_rc,
        "adm_acc": adm_acc, "rej_acc": rej_acc, "ratio": ratio,
        "ci95": [lo, hi], "fisher_p": float(p),
    }

    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
