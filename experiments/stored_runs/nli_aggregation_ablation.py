"""
NLI aggregation ablation (Reviewer W5 / Q5).

Question: how sensitive is the admissibility ratio (2.94x pooled TriviaQA) to the
choice of NLI aggregator for symmetrizing the pairwise entailment score?

Aggregators (for directional entailment probs p(a->b), p(b->a) in [0,1]):
    mean:    (p_ab + p_ba) / 2                    # used by the paper
    min:     min(p_ab, p_ba)                       # strict bidirectional entailment
    max:     max(p_ab, p_ba)                       # one-way entailment
    product: p_ab * p_ba                           # joint probability under independence

For each aggregator, recompute pairwise similarities, run the admissibility
pipeline at the paper's threshold grid {0.50, ..., 0.95}, and report
(n_adm, n_rej, adm_acc, rej_acc, ratio).

Protocol:
- 4 base models (Qwen, Mistral, Llama, Gemma); TriviaQA, n=500 questions each.
- Top-6 responses per question per model (12000 responses total).
- Correctness = fuzzy string match to gold (same as main-text Table 1).
- MPS / CUDA / CPU device autodetect.

Usage:
    ~/ip_llm/bin/python experiments/nli_aggregation_ablation.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

NLI_MODEL = "microsoft/deberta-large-mnli"
MAX_LEN = 256
N_RESPONSES = 6
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

SCRIPT_DIR = Path(__file__).parent
MODAL = SCRIPT_DIR / "modal_data"

PRED1_FILES = {
    "Qwen":    "pred1_qwen.json",
    "Mistral": "pred1_mistral_20260110_071956.json",
    "Llama":   "pred1_llama8b_20260112_074432.json",
    "Gemma":   "pred1_gemma9b_20260112_074222.json",
}


def normalize(s: str) -> str:
    """Match kle_snne_comparison.normalize_answer exactly."""
    import re
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def fuzzy_correct(response: str, gold_answers: list[str]) -> int:
    r = normalize(response)
    for g in gold_answers:
        gn = normalize(g)
        if gn and (gn in r or r in gn):
            return 1
    return 0


def cluster_cc(sim, theta):
    m = len(sim)
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i in range(m):
        for j in range(i + 1, m):
            if sim[i][j] >= theta:
                union(i, j)
    groups = {}
    for i in range(m):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def admissibility(sim, thresholds=THRESHOLDS):
    m = len(sim)
    adm = [False] * m
    for t in thresholds:
        clusters = cluster_cc(sim, t)
        max_size = max(len(c) for c in clusters)
        for c in clusters:
            if len(c) == max_size:
                for r in c:
                    adm[r] = True
    return adm


def compute_raw_nli_matrix(texts, tokenizer, nli, device):
    """Return 6x6x2 array: [i,j,0] = p(i->j entailment), [i,j,1] = p(j->i entailment)."""
    m = len(texts)
    raw = np.zeros((m, m, 2), dtype=np.float32)
    # Build all (i,j) pairs with i<j.
    pairs_ij = []
    for i in range(m):
        for j in range(i + 1, m):
            pairs_ij.append((i, j))
    # Batch directional entailment.
    premises, hypotheses, which = [], [], []  # which direction: 0=i->j, 1=j->i
    for (i, j) in pairs_ij:
        if not texts[i].strip() or not texts[j].strip():
            raw[i, j, 0] = 0.0
            raw[i, j, 1] = 0.0
            raw[j, i, 0] = 0.0
            raw[j, i, 1] = 0.0
            continue
        premises.extend([texts[i], texts[j]])
        hypotheses.extend([texts[j], texts[i]])
        which.extend([(i, j, 0), (i, j, 1)])

    if premises:
        # Process in batches to avoid OOM.
        bs = 32
        ent_probs = []
        for s in range(0, len(premises), bs):
            inp = tokenizer(premises[s:s+bs], hypotheses[s:s+bs],
                            return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True).to(device)
            with torch.no_grad():
                logits = nli(**inp).logits
            probs = torch.softmax(logits, dim=-1)[:, 2].cpu().numpy()  # entailment index 2
            ent_probs.extend(probs.tolist())
        for p, (i, j, d) in zip(ent_probs, which):
            if d == 0:
                raw[i, j, 0] = p
                raw[j, i, 1] = p  # mirror for symmetry downstream if needed
            else:
                raw[i, j, 1] = p
                raw[j, i, 0] = p
    # Diagonal = 1.0
    for i in range(m):
        raw[i, i, 0] = 1.0
        raw[i, i, 1] = 1.0
    return raw


def aggregate(raw, mode):
    """raw: 6x6x2. Returns symmetric 6x6 similarity matrix."""
    m = raw.shape[0]
    sim = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        sim[i, i] = 1.0
        for j in range(i + 1, m):
            a, b = raw[i, j, 0], raw[i, j, 1]
            if mode == "mean":    s = (a + b) / 2
            elif mode == "min":   s = min(a, b)
            elif mode == "max":   s = max(a, b)
            elif mode == "product": s = a * b
            else: raise ValueError(mode)
            sim[i, j] = s
            sim[j, i] = s
    return sim


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()

    # Load all 4 models' response texts + correctness labels.
    all_rows = []  # dicts with model, qid, responses, accuracies
    for model, filename in PRED1_FILES.items():
        with open(MODAL / filename) as f:
            d = json.load(f)
        pqr = d["per_question_results"]
        for qid, rec in pqr.items():
            responses = rec["base_responses"][:N_RESPONSES]
            gold = rec["gold_answers"]
            if len(responses) < N_RESPONSES:
                continue
            accs = [fuzzy_correct(r, gold) for r in responses]
            all_rows.append({"model": model, "qid": qid, "responses": responses, "accuracies": accs})

    print(f"Loaded {len(all_rows)} (model, question) pairs")

    # Compute raw NLI matrices.
    raw_matrices = []
    t0 = time.time()
    for row in tqdm(all_rows, desc="NLI"):
        raw = compute_raw_nli_matrix(row["responses"], tokenizer, nli, device)
        raw_matrices.append(raw)
    print(f"NLI elapsed: {time.time()-t0:.1f}s")

    # For each aggregator, compute admissibility and accuracy breakdown.
    aggregators = ["mean", "min", "max", "product"]
    results = {}
    for agg in aggregators:
        n_adm, n_rej = 0, 0
        n_adm_correct, n_rej_correct = 0, 0
        for row, raw in zip(all_rows, raw_matrices):
            sim = aggregate(raw, agg)
            adm = admissibility(sim)
            for r_idx, is_adm in enumerate(adm):
                c = row["accuracies"][r_idx]
                if is_adm:
                    n_adm += 1
                    n_adm_correct += c
                else:
                    n_rej += 1
                    n_rej_correct += c
        adm_acc = n_adm_correct / n_adm if n_adm > 0 else float("nan")
        rej_acc = n_rej_correct / n_rej if n_rej > 0 else float("nan")
        ratio = adm_acc / rej_acc if rej_acc > 0 else float("inf")
        results[agg] = {
            "n_adm": n_adm, "n_rej": n_rej,
            "adm_acc": adm_acc, "rej_acc": rej_acc, "ratio": ratio,
        }
        print(f"{agg:8s}: n_adm={n_adm:>5} ({n_adm/(n_adm+n_rej)*100:.1f}%)  n_rej={n_rej:>4}  adm={adm_acc*100:.1f}%  rej={rej_acc*100:.1f}%  ratio={ratio:.2f}x")

    out = {
        "n_total": len(all_rows) * N_RESPONSES,
        "results": results,
    }
    with open(SCRIPT_DIR / "nli_aggregation_ablation_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved to {SCRIPT_DIR / 'nli_aggregation_ablation_results.json'}")


if __name__ == "__main__":
    main()
