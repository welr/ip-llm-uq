"""
Sentence-embedding similarity ablation on TruthfulQA (Reviewer W1, follow-up).

The TriviaQA embedding ablation in App. F shows the partition signal
survives a different similarity model class on free-form factual QA.
This script extends the same probe to TruthfulQA, the adversarial
benchmark where a reviewer might reasonably suspect that NLI's
substring->judge collapse (1.88x -> 1.08x) is itself an NLI artefact
rather than a genuine scorer-dependence story.

Protocol:
- Cached top-6 responses for the 4 base models on the full TruthfulQA
  validation split (817 questions, 4 * 817 * 6 = 19608 responses).
- Encoder: sentence-transformers/all-mpnet-base-v2 (768-dim, no NLI head).
- For each question, compute 6x6 cosine similarity matrix.
- Admissibility: union of per-threshold argmax clusters under the paper's
  fixed grid {0.50, ..., 0.95} applied verbatim.
- Two correctness labellings:
    (a) substring (the pre-registered scorer; from prereg_truthfulqa_*.json)
    (b) judge (post-hoc Claude Haiku 4.5 rescore; from rescore_*_judge.json)
- Pool and per-model ratios with 95% bootstrap CIs.

Usage: ~/ip_llm/bin/python experiments/embedding_similarity_truthfulqa.py
"""

import glob
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from scipy import stats

ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"
N_RESPONSES = 6
FIXED_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
RNG_SEED = 42
N_BOOTSTRAP = 10000

SCRIPT_DIR = Path(__file__).parent
MODAL = SCRIPT_DIR / "modal_data"
SUPP_RESULTS = SCRIPT_DIR.parent / "supplementary" / "data" / "results"

# Base models for TruthfulQA pre-registered evaluation.
MODELS = ["Qwen", "Mistral", "Llama", "Gemma"]


def find_prereg(model: str) -> Path:
    """Find the canonical prereg_truthfulqa_<model>_*.json (excluding instruct/MoE variants)."""
    candidates = sorted(MODAL.glob(f"prereg_truthfulqa_{model}_*.json"))
    # Filter out LlamaInstruct / Gemma4MoE which would also match Llama_/Gemma_
    candidates = [c for c in candidates if not any(
        x in c.name for x in ("LlamaInstruct", "Gemma4MoE")
    )]
    if not candidates:
        raise FileNotFoundError(f"no prereg_truthfulqa_{model}_*.json")
    return candidates[0]


def find_judge(model: str) -> Path:
    """Find the judge rescore file for a base model."""
    candidates = [
        SUPP_RESULTS / f"rescore_{model}_judge.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"no rescore_{model}_judge.json under {SUPP_RESULTS}")


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
    m = len(sim)
    if weights is None:
        weights = np.ones(m, dtype=float) / m
    adm = [False] * m
    for t in thresholds:
        clusters = cluster_cc(sim, t)
        probs = [sum(weights[i] for i in c) for c in clusters]
        max_p = max(probs)
        for c, p in zip(clusters, probs):
            if p == max_p:
                for r in c:
                    adm[r] = True
    return adm


def bootstrap_ratio_ci(adm, rej, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    a = np.array(adm); r = np.array(rej)
    if len(a) == 0 or len(r) == 0:
        return float("nan"), float("nan")
    rats = []
    for _ in range(n_boot):
        ai = rng.integers(0, len(a), size=len(a))
        ri = rng.integers(0, len(r), size=len(r))
        ra = a[ai].mean(); rr = r[ri].mean()
        if rr > 0:
            rats.append(ra / rr)
    return float(np.quantile(rats, 0.025)), float(np.quantile(rats, 0.975))


def fisher_p(adm_correct, adm_total, rej_correct, rej_total):
    table = [[adm_correct, adm_total - adm_correct],
             [rej_correct, rej_total - rej_correct]]
    try:
        _, p = stats.fisher_exact(table, alternative="two-sided")
    except Exception:
        p = float("nan")
    return float(p)


def summarize(adm, rej):
    n_adm, n_rej = len(adm), len(rej)
    adm_acc = float(np.mean(adm)) if adm else float("nan")
    rej_acc = float(np.mean(rej)) if rej else float("nan")
    ratio = adm_acc / rej_acc if rej_acc > 0 else float("nan")
    ci_lo, ci_hi = bootstrap_ratio_ci(adm, rej)
    p = fisher_p(int(np.sum(adm)), n_adm, int(np.sum(rej)), n_rej)
    return {
        "n_adm": n_adm, "n_rej": n_rej,
        "adm_acc": adm_acc, "rej_acc": rej_acc,
        "ratio": ratio, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "rejection_rate": n_rej / max(1, n_adm + n_rej),
        "fisher_p": p,
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Encoder: {ENCODER_NAME}")
    encoder = SentenceTransformer(ENCODER_NAME, device=device)

    # --------------------------------------------------------------
    # Load substring labels + texts (from prereg) and judge labels
    # (from rescore_*_judge), keyed by qid.
    # --------------------------------------------------------------
    rows_by_model = {}  # model -> [(qid, [text..6], [substring..6], [judge..6])]
    for model in MODELS:
        prereg = json.load(open(find_prereg(model)))
        judge = json.load(open(find_judge(model)))
        sub_pq = prereg["per_question"]
        jud_pq = judge["per_question"]
        rows = []
        skipped = 0
        for qid, sub_rec in sub_pq.items():
            texts = sub_rec.get("top6_texts", [])
            sub_acc = sub_rec.get("accuracies", [])
            if len(texts) < N_RESPONSES or len(sub_acc) < N_RESPONSES:
                skipped += 1
                continue
            jud_rec = jud_pq.get(qid)
            if jud_rec is None:
                skipped += 1
                continue
            jud_acc = jud_rec.get("accuracies", [])
            if len(jud_acc) < N_RESPONSES:
                skipped += 1
                continue
            # accuracies stored as bool/int/0-1 floats; coerce.
            sub_acc = [int(bool(a)) for a in sub_acc[:N_RESPONSES]]
            jud_acc = [int(bool(a)) for a in jud_acc[:N_RESPONSES]]
            rows.append((qid, texts[:N_RESPONSES], sub_acc, jud_acc))
        print(f"  {model}: {len(rows)} questions  ({skipped} skipped)")
        rows_by_model[model] = rows

    # --------------------------------------------------------------
    # Encode all responses per model (batch over the flat list).
    # --------------------------------------------------------------
    print("\nEncoding responses ...")
    embs_by_model = {}
    for model, rows in rows_by_model.items():
        flat = []
        for qid, resp, _, _ in rows:
            flat.extend(resp)
        t0 = time.time()
        E = encoder.encode(flat, batch_size=64, show_progress_bar=False,
                           normalize_embeddings=True, convert_to_numpy=True)
        print(f"  {model}: {len(flat)} responses in {time.time()-t0:.1f}s")
        embs_by_model[model] = E.reshape(len(rows), N_RESPONSES, -1)

    # --------------------------------------------------------------
    # For each (model, question), compute embedding-cosine similarity,
    # run admissibility, and accumulate per-(scorer, model) outcomes.
    # --------------------------------------------------------------
    pooled = {"substring": {"adm": [], "rej": []},
              "judge":     {"adm": [], "rej": []}}
    per_model = {m: {"substring": {"adm": [], "rej": []},
                     "judge":     {"adm": [], "rej": []}} for m in MODELS}

    for model, rows in rows_by_model.items():
        E = embs_by_model[model]
        for q_idx, (qid, resp, sub_acc, jud_acc) in enumerate(
            tqdm(rows, desc=f"{model} admissibility")
        ):
            Eq = E[q_idx]
            sim = (Eq @ Eq.T).astype(float).tolist()
            adm_flags = admissibility_at_grid(sim, FIXED_GRID, weights=None)
            for r_idx, is_adm in enumerate(adm_flags):
                bucket = "adm" if is_adm else "rej"
                pooled["substring"][bucket].append(sub_acc[r_idx])
                pooled["judge"][bucket].append(jud_acc[r_idx])
                per_model[model]["substring"][bucket].append(sub_acc[r_idx])
                per_model[model]["judge"][bucket].append(jud_acc[r_idx])

    # --------------------------------------------------------------
    # Summarize.
    # --------------------------------------------------------------
    print("\n=== Embedding-similarity admissibility on TruthfulQA "
          "(4 base models, 817-question full validation split) ===\n")
    print(f"Encoder: {ENCODER_NAME}\nFixed grid: {FIXED_GRID}\n")

    summary = {"encoder": ENCODER_NAME, "fixed_grid": FIXED_GRID,
               "pooled": {}, "per_model": {}}
    for scorer in ("substring", "judge"):
        pool = summarize(pooled[scorer]["adm"], pooled[scorer]["rej"])
        summary["pooled"][scorer] = pool
        print(f"[{scorer:>9s}] POOLED: n_adm={pool['n_adm']}, n_rej={pool['n_rej']} "
              f"({pool['rejection_rate']*100:.1f}%), "
              f"adm={pool['adm_acc']*100:.1f}%, rej={pool['rej_acc']*100:.1f}%, "
              f"ratio={pool['ratio']:.2f}x [{pool['ci_lo']:.2f}, {pool['ci_hi']:.2f}], "
              f"p={pool['fisher_p']:.2e}")
        for model in MODELS:
            sm = summarize(per_model[model][scorer]["adm"],
                           per_model[model][scorer]["rej"])
            summary["per_model"].setdefault(model, {})[scorer] = sm
            print(f"  [{scorer:>9s}] {model:7s}: n_adm={sm['n_adm']}, "
                  f"n_rej={sm['n_rej']:>4d} ({sm['rejection_rate']*100:.1f}%), "
                  f"adm={sm['adm_acc']*100:.1f}%, rej={sm['rej_acc']*100:.1f}%, "
                  f"ratio={sm['ratio']:.2f}x [{sm['ci_lo']:.2f}, {sm['ci_hi']:.2f}], "
                  f"p={sm['fisher_p']:.2e}")
        print()

    out_path = SCRIPT_DIR / "embedding_similarity_truthfulqa_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
