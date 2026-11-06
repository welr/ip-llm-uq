"""
Embedding-similarity ablation (Reviewer W1, follow-up to Appendix G NLI
aggregator ablation).

Question: does the 2.94x admissibility ratio on TriviaQA survive when we
swap DeBERTa-large-MNLI for a non-NLI similarity --- specifically
sentence-embedding cosine via sentence-transformers (all-mpnet-base-v2)?
This closes the same-model loophole left by the NLI aggregator ablation
(min/max/mean/product all use the same DeBERTa NLI head).

Protocol:
- Cached top-6 responses for the 4 base models on TriviaQA primary
  (n=500 q each, 12000 responses total). Same responses, same correctness
  labels (fuzzy substring match) as Table 1 in the main paper.
- Encoder: all-mpnet-base-v2 (768-dim, well-known sentence-encoder).
- For each question, compute 6x6 cosine similarity matrix on the
  encoded responses.
- Admissibility: union of per-threshold argmax clusters using the paper's
  fixed grid {0.50, 0.55, ..., 0.95}, applied verbatim, for an apples-to-
  apples comparison with the NLI ratio in Table 1 of the paper.
- Pool and per-model ratios with bootstrap CIs.

Usage:
    ~/ip_llm/bin/python experiments/embedding_similarity_ablation.py
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"
N_RESPONSES = 6
FIXED_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
RNG_SEED = 42
N_BOOTSTRAP = 10000

SCRIPT_DIR = Path(__file__).parent
MODAL = SCRIPT_DIR / "modal_data"

PRED1_FILES = {
    "Qwen":    "pred1_qwen.json",
    "Mistral": "pred1_mistral_20260110_071956.json",
    "Llama":   "pred1_llama8b_20260112_074432.json",
    "Gemma":   "pred1_gemma9b_20260112_074222.json",
}


def normalize(s: str) -> str:
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
    for i in range(m):
        for j in range(i + 1, m):
            if sim[i][j] >= theta:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for i in range(m):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def admissibility_at_grid(sim, thresholds, weights=None):
    """Union of argmax clusters across thresholds. weights=None -> uniform 1/M."""
    m = len(sim)
    if weights is None:
        weights = np.ones(m, dtype=float) / m
    adm = [False] * m
    for t in thresholds:
        clusters = cluster_cc(sim, t)
        # Cluster probability under given weights (sum of member weights).
        probs = [sum(weights[i] for i in c) for c in clusters]
        max_p = max(probs)
        for c, p in zip(clusters, probs):
            if p == max_p:
                for r in c:
                    adm[r] = True
    return adm


def breakpoint_grid(sim):
    """Distinct upper-triangle similarities define piecewise-constant intervals.
    Use one threshold strictly inside each interval (mid-point of consecutive
    breakpoints, plus a value below the smallest and at/above the largest).
    """
    m = len(sim)
    sims = sorted({sim[i][j] for i in range(m) for j in range(i + 1, m)})
    # Always include endpoints just below smallest (collapse all into one cluster
    # only if smallest >= 0; we use -inf-equivalent) and just above largest
    # (split everything into singletons).
    # In practice we want to land strictly inside each interval (b_k, b_{k+1}].
    grid = []
    if sims:
        # Below smallest: every edge satisfies sim >= theta (trivial; all merged)
        # is uninformative. Above largest: no edges satisfy sim >= theta (all
        # singletons; trivial). Skip those degenerate ends; admissibility is
        # determined by interior intervals.
        for k in range(len(sims) - 1):
            mid = (sims[k] + sims[k + 1]) / 2
            grid.append(mid)
        # Plus one threshold equal to the smallest similarity (so smallest edge
        # is included) and just above the largest (so no edges) for completeness.
        grid = [sims[0]] + grid + [sims[-1] + 1e-9]
    else:
        grid = [0.5]
    return grid


def bootstrap_ratio_ci(adm_correct_flags, rej_correct_flags, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Bootstrap a 95% CI on (adm_acc / rej_acc)."""
    rng = np.random.default_rng(seed)
    a = np.array(adm_correct_flags)
    r = np.array(rej_correct_flags)
    if len(a) == 0 or len(r) == 0:
        return (float("nan"), float("nan"))
    boot_ratios = []
    for _ in range(n_boot):
        a_idx = rng.integers(0, len(a), size=len(a))
        r_idx = rng.integers(0, len(r), size=len(r))
        a_acc = a[a_idx].mean()
        r_acc = r[r_idx].mean()
        if r_acc > 0:
            boot_ratios.append(a_acc / r_acc)
    boot_ratios = np.array(boot_ratios)
    lo, hi = np.quantile(boot_ratios, [0.025, 0.975])
    return float(lo), float(hi)


def main():
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Encoder: {ENCODER_NAME}")
    encoder = SentenceTransformer(ENCODER_NAME, device=device)

    # Load all responses + accuracies.
    rows_by_model = {}  # {model: [(qid, [resp..6], [acc..6])]}
    for model, filename in PRED1_FILES.items():
        with open(MODAL / filename) as f:
            d = json.load(f)
        pqr = d["per_question_results"]
        out = []
        for qid, rec in pqr.items():
            responses = rec["base_responses"][:N_RESPONSES]
            gold = rec["gold_answers"]
            if len(responses) < N_RESPONSES:
                continue
            accs = [fuzzy_correct(r, gold) for r in responses]
            out.append((qid, responses, accs))
        rows_by_model[model] = out
        print(f"  {model}: {len(out)} questions loaded")

    # Encode all responses per model. Embeddings are independent across
    # questions, so we batch-encode the entire flat list per model.
    print("\nEncoding responses...")
    embs_by_model = {}
    for model, rows in rows_by_model.items():
        flat = []
        for qid, resp, _ in rows:
            flat.extend(resp)
        t0 = time.time()
        embeddings = encoder.encode(flat, batch_size=64, show_progress_bar=False,
                                    normalize_embeddings=True, convert_to_numpy=True)
        # normalize_embeddings=True so dot product = cosine.
        print(f"  {model}: {len(flat)} responses in {time.time()-t0:.1f}s")
        # Reshape into per-question 6x768 arrays.
        e = embeddings.reshape(len(rows), N_RESPONSES, -1)
        embs_by_model[model] = e

    # For each model + question, compute admissibility on fixed grid.
    pooled = {"adm": [], "rej": []}
    per_model = {}

    for model, rows in rows_by_model.items():
        e_m = embs_by_model[model]
        per_model[model] = {"adm": [], "rej": []}
        for q_idx, (qid, resp, accs) in enumerate(tqdm(rows, desc=f"{model} admissibility")):
            E = e_m[q_idx]                # (6, 768) unit-norm
            sim = (E @ E.T).astype(float) # (6, 6) cosine similarity in [-1, 1]
            sim_list = sim.tolist()
            adm_flags = admissibility_at_grid(sim_list, FIXED_GRID, weights=None)
            for r_idx, is_adm in enumerate(adm_flags):
                if is_adm:
                    pooled["adm"].append(accs[r_idx])
                    per_model[model]["adm"].append(accs[r_idx])
                else:
                    pooled["rej"].append(accs[r_idx])
                    per_model[model]["rej"].append(accs[r_idx])

    # Summarize.
    def summarize(adm, rej):
        n_adm, n_rej = len(adm), len(rej)
        adm_acc = float(np.mean(adm)) if adm else float("nan")
        rej_acc = float(np.mean(rej)) if rej else float("nan")
        ratio = adm_acc / rej_acc if rej_acc > 0 else float("nan")
        ci_lo, ci_hi = bootstrap_ratio_ci(adm, rej)
        return {
            "n_adm": n_adm, "n_rej": n_rej,
            "adm_acc": adm_acc, "rej_acc": rej_acc,
            "ratio": ratio, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "rejection_rate": n_rej / max(1, n_adm + n_rej),
        }

    print("\n=== Embedding-similarity admissibility (TriviaQA, 4 base models) ===\n")
    print(f"Encoder: {ENCODER_NAME}")
    print(f"Fixed grid: {FIXED_GRID}")
    print()

    summary = {"encoder": ENCODER_NAME, "fixed_grid": FIXED_GRID,
               "pooled": {}, "per_model": {}}
    s = summarize(pooled["adm"], pooled["rej"])
    summary["pooled"] = s
    print(f"POOLED: n_adm={s['n_adm']}, n_rej={s['n_rej']} ({s['rejection_rate']*100:.1f}%), "
          f"adm={s['adm_acc']*100:.1f}%, rej={s['rej_acc']*100:.1f}%, "
          f"ratio={s['ratio']:.2f}x [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]")
    for model in PRED1_FILES:
        sm = summarize(per_model[model]["adm"], per_model[model]["rej"])
        summary["per_model"][model] = sm
        print(f"  {model:7s}: n_adm={sm['n_adm']}, n_rej={sm['n_rej']:>3d} ({sm['rejection_rate']*100:.1f}%), "
              f"adm={sm['adm_acc']*100:.1f}%, rej={sm['rej_acc']*100:.1f}%, "
              f"ratio={sm['ratio']:.2f}x [{sm['ci_lo']:.2f}, {sm['ci_hi']:.2f}]")
    print()

    out_path = SCRIPT_DIR / "embedding_similarity_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
