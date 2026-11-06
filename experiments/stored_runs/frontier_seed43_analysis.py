"""
Seed-43 NLI admissibility analysis for both frontier models.

Runs on seed-43 raw data, then combines with seed-42 results to produce
merged n=200 admissibility statistics. Pipeline identical to
frontier_sonnet_analysis.py and frontier_gemini_analysis.py (NLI
microsoft/deberta-large-mnli; thresholds 0.50-0.95 step 0.05; N=6 responses).

Outputs (all new seed-43-only first, then merged n=200):
  results/frontier_sonnet_admissibility_seed43.json
  results/frontier_gemini_admissibility_seed43.json
  results/frontier_n200_admissibility.json (combined)

Usage:
    ~/ip_llm/bin/python experiments/frontier_seed43_analysis.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


NLI_MODEL = "microsoft/deberta-large-mnli"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
N_RESPONSES = 6
SCRIPT_DIR = Path(__file__).parent
RES = SCRIPT_DIR / "results"


def normalize_answer(a):
    if not a: return ""
    a = a.lower().strip()
    a = re.sub(r"[^\w\s]", "", a)
    a = re.sub(r"\s+", " ", a)
    return a[:500]


def check_answer(gen, gold):
    if not gen: return False
    g = normalize_answer(gen)
    if not g: return False
    for x in gold:
        xn = normalize_answer(x)
        if xn and (xn in g or g in xn):
            return True
    return False


def admissible_at_theta(n, sim, theta):
    parent = list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(x,y):
        px,py=find(x),find(y)
        if px!=py: parent[px]=py
    for i in range(n):
        for j in range(i+1,n):
            if sim[i][j]>=theta: union(i,j)
    cl=defaultdict(set)
    for i in range(n): cl[find(i)].add(i)
    cl=list(cl.values())
    mx=max(len(c) for c in cl)
    return set().union(*[c for c in cl if len(c)==mx]), cl


def analyze_one_model(raw_path, out_path, model_name, nli, tokenizer, device):
    print(f"\n=== {model_name}: {raw_path.name} ===")
    data = json.load(open(raw_path))
    print(f"  {len(data)} questions, {sum(len(v['responses']) for v in data.values())} responses")

    @torch.no_grad()
    def nli_sim(t1, t2):
        if not t1.strip() or not t2.strip(): return 0.0
        sims = []
        for prem, hyp in [(t1,t2),(t2,t1)]:
            inputs = tokenizer(prem, hyp, return_tensors="pt",
                               truncation=True, max_length=256).to(device)
            probs = torch.softmax(nli(**inputs).logits, dim=-1)
            sims.append(probs[0, 2].item())
        return float(np.mean(sims))

    per_question = {}
    dispersion_hist = Counter()

    for qid, v in tqdm(data.items(), desc=f"  {model_name} NLI"):
        texts = [r["text"] for r in v["responses"][:N_RESPONSES]]
        gold = v["gold_answers"]
        accs = [check_answer(t, gold) for t in texts]
        n = len(texts)
        sim_matrix = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                s = nli_sim(texts[i], texts[j])
                sim_matrix[i,j] = s; sim_matrix[j,i] = s
        admissible = set(); ostar_sets = []; se_vals = []
        for theta in THRESHOLDS:
            adm_t, clusters = admissible_at_theta(n, sim_matrix, theta)
            ostar_sets.append(frozenset(adm_t))
            admissible.update(adm_t)
            cp = [len(c)/n for c in clusters]
            se = -sum(p*np.log2(p) for p in cp if p>0)
            se_vals.append(se)
        rejected = set(range(n)) - admissible
        nd = len(set(ostar_sets))
        dispersion_hist[nd] += 1
        per_question[qid] = {
            "accuracies": [bool(a) for a in accs],
            "admissible": sorted(admissible),
            "rejected": sorted(rejected),
            "num_distinct_Ostar": nd,
            "se_lower": float(min(se_vals)),
            "se_upper": float(max(se_vals)),
            "sim_matrix": sim_matrix.tolist(),
        }

    # Aggregate
    ac = at = rc = rt = 0
    for r in per_question.values():
        accs = r["accuracies"]
        for i in r["admissible"]: ac += int(accs[i]); at += 1
        for i in r["rejected"]: rc += int(accs[i]); rt += 1
    adm_acc = ac/at if at else 0
    rej_acc = rc/rt if rt else 0
    ratio = (adm_acc/rej_acc) if rej_acc>0 else float("inf")

    out = {
        "protocol": {"model": model_name, "seed": 43, "thresholds": THRESHOLDS, "n_responses_used": N_RESPONSES},
        "dispersion": {
            "histogram_num_distinct_Ostar": dict(dispersion_hist),
            "mean": float(np.mean([v["num_distinct_Ostar"] for v in per_question.values()])),
            "median": float(np.median([v["num_distinct_Ostar"] for v in per_question.values()])),
        },
        "aggregate": {
            "n_questions": len(per_question),
            "n_adm": at, "n_rej": rt,
            "adm_acc": adm_acc, "rej_acc": rej_acc,
            "ratio": ratio, "ac": ac, "rc": rc,
        },
        "per_question": {qid: {k:v for k,v in r.items() if k != "sim_matrix"}
                         for qid, r in per_question.items()},
        "sim_matrices": {qid: r["sim_matrix"] for qid, r in per_question.items()},
    }
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  n_adm={at} n_rej={rt} adm_acc={adm_acc:.1%} rej_acc={rej_acc:.1%} ratio={ratio:.2f}x")
    print(f"  Saved: {out_path}")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading NLI on {device}")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()

    # Sonnet seed-43
    analyze_one_model(
        SCRIPT_DIR / "modal_data" / "frontier_sonnet_truthfulqa_seed43.json",
        RES / "frontier_sonnet_admissibility_seed43.json",
        "claude-sonnet-4-6 (seed-43)",
        nli, tokenizer, device,
    )
    # Gemini seed-43
    analyze_one_model(
        SCRIPT_DIR / "modal_data" / "frontier_gemini_truthfulqa_seed43.json",
        RES / "frontier_gemini_admissibility_seed43.json",
        "gemini-2.5-flash (seed-43)",
        nli, tokenizer, device,
    )
    print("\nDone. Next step: judge rescore on seed-43 (frontier_seed43_judge_rescore.py)")


if __name__ == "__main__":
    main()
