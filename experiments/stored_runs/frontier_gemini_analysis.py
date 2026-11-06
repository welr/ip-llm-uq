"""
NLI admissibility analysis for the Gemini 2.5 Flash frontier run.

Identical pipeline to frontier_sonnet_analysis.py (same NLI model, thresholds,
scoring, Goldilocks bins) applied to the Gemini output.

Usage:
    ~/ip_llm/bin/python experiments/frontier_gemini_analysis.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


NLI_MODEL = "microsoft/deberta-large-mnli"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
N_RESPONSES = 6

SCRIPT_DIR = Path(__file__).parent
IN_PATH = SCRIPT_DIR / "modal_data" / "frontier_gemini_truthfulqa.json"
OUT_PATH = SCRIPT_DIR / "results" / "frontier_gemini_admissibility.json"


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    a = answer.lower().strip()
    a = re.sub(r"[^\w\s]", "", a)
    a = re.sub(r"\s+", " ", a)
    return a[:500]


def check_answer(generated: str, gold_answers) -> bool:
    if not generated:
        return False
    g = normalize_answer(generated)
    if not g:
        return False
    for gold in gold_answers:
        gn = normalize_answer(gold)
        if gn and (gn in g or g in gn):
            return True
    return False


def admissible_at_theta(n, sim, theta):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py: parent[px] = py
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] >= theta: union(i, j)
    cl = defaultdict(set)
    for i in range(n): cl[find(i)].add(i)
    cl = list(cl.values())
    mx = max(len(c) for c in cl)
    return set().union(*[c for c in cl if len(c) == mx]), cl


def boot_ratio_ci(ac, at, rc, rt, n_boot=10000, seed=42):
    if at == 0 or rt == 0 or rc == 0:
        return None, None
    rng = np.random.default_rng(seed)
    adm = np.array([1]*ac + [0]*(at-ac)); rej = np.array([1]*rc + [0]*(rt-rc))
    rs = []
    for _ in range(n_boot):
        a = rng.choice(adm, size=len(adm), replace=True).mean()
        r = rng.choice(rej, size=len(rej), replace=True).mean()
        if r > 0: rs.append(a/r)
    return float(np.percentile(rs,2.5)), float(np.percentile(rs,97.5))


def main():
    print(f"Loading: {IN_PATH}")
    data = json.load(open(IN_PATH))
    print(f"  {len(data)} questions, {sum(len(v['responses']) for v in data.values())} responses")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading NLI model: {NLI_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()

    @torch.no_grad()
    def nli_sim(t1, t2):
        if not t1.strip() or not t2.strip():
            return 0.0
        sims = []
        for prem, hyp in [(t1, t2), (t2, t1)]:
            inputs = tokenizer(prem, hyp, return_tensors="pt",
                               truncation=True, max_length=256).to(device)
            probs = torch.softmax(nli(**inputs).logits, dim=-1)
            sims.append(probs[0, 2].item())
        return float(np.mean(sims))

    per_question = {}
    dispersion_hist = Counter()

    for qid, v in tqdm(data.items(), desc="NLI"):
        texts = [r["text"] for r in v["responses"][:N_RESPONSES]]
        gold = v["gold_answers"]
        accs = [check_answer(t, gold) for t in texts]
        n = len(texts)

        sim_matrix = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                s = nli_sim(texts[i], texts[j])
                sim_matrix[i, j] = s; sim_matrix[j, i] = s

        admissible = set()
        ostar_sets = []
        se_values = []
        for theta in THRESHOLDS:
            adm_t, clusters = admissible_at_theta(n, sim_matrix, theta)
            ostar_sets.append(frozenset(adm_t))
            admissible.update(adm_t)
            cp = [len(c)/n for c in clusters]
            se = -sum(p*np.log2(p) for p in cp if p>0)
            se_values.append(se)
        rejected = set(range(n)) - admissible
        nd = len(set(ostar_sets))
        dispersion_hist[nd] += 1

        per_question[qid] = {
            "accuracies": [bool(a) for a in accs],
            "admissible": sorted(admissible),
            "rejected": sorted(rejected),
            "num_distinct_Ostar": nd,
            "se_lower": float(min(se_values)),
            "se_upper": float(max(se_values)),
            "sim_matrix": sim_matrix.tolist(),
        }

    # Aggregate
    ac = at = rc = rt = 0
    for r in per_question.values():
        accs = r["accuracies"]
        for i in r["admissible"]: ac += int(accs[i]); at += 1
        for i in r["rejected"]:   rc += int(accs[i]); rt += 1
    adm_acc = ac/at if at else 0
    rej_acc = rc/rt if rt else 0
    ratio = (adm_acc/rej_acc) if rej_acc>0 else float("inf")
    lo, hi = boot_ratio_ci(ac, at, rc, rt)
    _, p = stats.fisher_exact([[ac, at-ac],[rc, max(rt-rc,0)]])

    # Goldilocks
    bins = {"0-2":[0,0],"3-4":[0,0],"5-6":[0,0],"7-8":[0,0],"9-10":[0,0]}
    for qid, r in per_question.items():
        sim = np.array(r["sim_matrix"]); n = sim.shape[0]
        tc = [0]*n
        for theta in THRESHOLDS:
            adm, _ = admissible_at_theta(n, sim, theta)
            for i in adm: tc[i] += 1
        accs = r["accuracies"]
        for i,c in enumerate(tc):
            b = "0-2" if c<=2 else "3-4" if c<=4 else "5-6" if c<=6 else "7-8" if c<=8 else "9-10"
            bins[b][0] += int(accs[i]); bins[b][1] += 1
    goldilocks = {b: {"n":t,"n_correct":c,"acc":(c/t if t else 0)} for b,(c,t) in bins.items()}

    dispersion = {
        "histogram_num_distinct_Ostar": dict(dispersion_hist),
        "mean": float(np.mean([v["num_distinct_Ostar"] for v in per_question.values()])),
        "median": float(np.median([v["num_distinct_Ostar"] for v in per_question.values()])),
    }

    aggregate = {
        "n_questions": len(per_question),
        "n_adm": at, "n_rej": rt,
        "adm_acc": adm_acc, "rej_acc": rej_acc,
        "ratio": ratio, "ci95": [lo, hi],
        "fisher_p": float(p), "ac": ac, "rc": rc,
    }

    out = {
        "protocol": {
            "model": "gemini-2.5-flash",
            "nli_model": NLI_MODEL,
            "thresholds": THRESHOLDS,
            "n_responses_used": N_RESPONSES,
            "deviations_from_base": [
                "temperature range clipped to [0.3, 1.0] (base: [0.3, 1.3])",
                "top_p unset",
                "max_output_tokens=1500 (base: 50); Flash essay-style answers extend ~700 tok",
                "thinking_budget=0 (Gemini 2.5 Flash thinking disabled for parity)",
            ],
        },
        "dispersion": dispersion,
        "aggregate": aggregate,
        "goldilocks": goldilocks,
        "per_question": {qid: {k: v for k, v in r.items() if k != "sim_matrix"}
                         for qid, r in per_question.items()},
        "sim_matrices": {qid: r["sim_matrix"] for qid, r in per_question.items()},
    }
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "="*70)
    print(f"  GEMINI 2.5 FLASH - TRUTHFULQA (100 seed-42 questions)")
    print("="*70)
    print(f"  Dispersion mean={dispersion['mean']:.2f} median={dispersion['median']:.0f}")
    print(f"  Hist: {dispersion['histogram_num_distinct_Ostar']}")
    print(f"  Admissible {at}/{adm_acc:.1%}  correct={ac}")
    print(f"  Rejected   {rt}/{rej_acc:.1%}  correct={rc}")
    if lo is not None:
        print(f"  Ratio {ratio:.2f}x  95% CI [{lo:.2f}, {hi:.2f}]  Fisher p={p:.3g}")
    else:
        print(f"  Ratio {ratio:.2f}x  Fisher p={p:.3g}")
    print()
    print("  Goldilocks:")
    for b, v in goldilocks.items():
        if v["n"]>0:
            print(f"    {b:>5}: n={v['n']:>4}  acc={v['acc']:.1%}")
    print()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
