"""
Generation-probability reweighting (Reviewer W3, Edit 1c).

Question: the paper claims "incorporating generation probabilities would
likely strengthen discrimination." We test that. Concretely: holding the
NLI similarity matrix and threshold grid fixed, recompute admissibility
with cluster probabilities weighted by softmax-normalized exp(mean-token
logprob) instead of uniform 1/M, and compare the resulting ratio against
the published 2.94x.

Inputs:
- experiments/sim_cache/sim_matrices_triviaqa_{Qwen,Mistral,Llama,Gemma}.json
  (cached 6x6 NLI similarity matrices, unchanged from the paper protocol)
- experiments/modal_data/logprobs_triviaqa_{Qwen,Mistral,Llama,Gemma}.json
  (mean per-token logprobs from score_logprobs_modal.py; downloaded after
  Modal scoring completes)
- experiments/modal_data/pred1_*.json (for response text + gold answers,
  for fuzzy correctness labels matching the paper)

Outputs:
- experiments/genprob_reweighting_results.json: per-model and pooled
  ratios under (a) uniform 1/M (sanity check vs. paper 2.94x) and
  (b) softmax-exp(logprob) weights.

Usage:
    ~/ip_llm/bin/python experiments/genprob_reweighting.py
    [optionally, to download logprobs from Modal volume first:]
    ~/ip_llm/bin/modal volume get choice-fn-results logprobs_triviaqa_{model}.json experiments/modal_data/

Notes:
- The "uniform" run here is a sanity check: it should reproduce ~2.94x
  on the same data. (Fuzzy matching, threshold grid, and similarity
  matrices are byte-identical to the paper protocol.)
- "logprob" weights: w_r = softmax(mean_token_logprob_r) over r in
  {1..M}. We do NOT length-normalize beyond mean-per-token (the
  scoring already mean-pools across tokens), and we do NOT temperature-
  scale the softmax. Both are documented choices.
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
N_RESPONSES = 6
RNG_SEED = 42
N_BOOTSTRAP = 10000

SCRIPT_DIR = Path(__file__).parent
MODAL = SCRIPT_DIR / "modal_data"
SIM_CACHE = SCRIPT_DIR / "sim_cache"

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


def admissibility(sim, weights, thresholds=THRESHOLDS):
    m = len(sim)
    adm = [False] * m
    for t in thresholds:
        clusters = cluster_cc(sim, t)
        probs = [sum(weights[i] for i in c) for c in clusters]
        max_p = max(probs)
        for c, p in zip(clusters, probs):
            # Tied clusters all admissible (matches paper definition).
            if p == max_p:
                for r in c:
                    adm[r] = True
    return adm


def softmax_weights(logprobs):
    """Numerically-stable softmax over mean-token logprobs.

    Treats NaN logprobs (empty responses) as -inf so they get zero weight.
    If all logprobs are NaN, falls back to uniform.
    """
    lp = np.asarray(logprobs, dtype=float)
    valid = ~np.isnan(lp)
    if not valid.any():
        return np.ones(len(lp)) / len(lp)
    lp_clean = np.where(valid, lp, -np.inf)
    lp_max = np.max(lp_clean[valid])
    e = np.exp(lp_clean - lp_max)
    e = np.where(valid, e, 0.0)
    Z = e.sum()
    return e / Z


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
    rats = np.array(rats)
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


def run_one_setting(rows_by_model, sim_by_model, lp_by_model, weight_fn):
    """weight_fn(qid, logprobs_for_q) -> length-M weight vector."""
    per_model = {}
    for model, rows in rows_by_model.items():
        sims = sim_by_model[model]
        lps = lp_by_model[model]
        m_adm, m_rej = [], []
        for qid, resp, accs in rows:
            sim = sims.get(qid)
            if sim is None or len(sim) < N_RESPONSES:
                continue
            lp_for_q = lps.get(qid) if lps is not None else None
            weights = weight_fn(qid, lp_for_q)
            adm_flags = admissibility(sim, weights)
            for r_idx, is_adm in enumerate(adm_flags):
                (m_adm if is_adm else m_rej).append(accs[r_idx])
        per_model[model] = {"adm": m_adm, "rej": m_rej}
    pooled_adm, pooled_rej = [], []
    for d in per_model.values():
        pooled_adm.extend(d["adm"])
        pooled_rej.extend(d["rej"])
    return pooled_adm, pooled_rej, per_model


def main():
    print("Loading sim matrices and pred1 (responses + accuracies)...")
    rows_by_model = {}   # {model: [(qid, [resp..6], [acc..6])]}
    sim_by_model = {}    # {model: {qid: 6x6 list}}
    lp_by_model = {}     # {model: {qid: [lp..6]}}

    for model, fname in PRED1_FILES.items():
        with open(MODAL / fname) as f:
            d = json.load(f)
        rows = []
        for qid, rec in d["per_question_results"].items():
            resp = rec["base_responses"][:N_RESPONSES]
            if len(resp) < N_RESPONSES:
                continue
            accs = [fuzzy_correct(r, rec["gold_answers"]) for r in resp]
            rows.append((qid, resp, accs))
        rows_by_model[model] = rows

        sim_path = SIM_CACHE / f"sim_matrices_triviaqa_{model}.json"
        with open(sim_path) as f:
            sim_by_model[model] = json.load(f)

        lp_path = MODAL / f"logprobs_triviaqa_{model}.json"
        if not lp_path.exists():
            print(f"  WARNING: {lp_path.name} not found; skipping logprob run for {model}")
            lp_by_model[model] = {}
        else:
            with open(lp_path) as f:
                lp_obj = json.load(f)
            # Format: {qid: {"logprobs": [..6], "n_tokens": [..6]}}
            lp_by_model[model] = {qid: v["logprobs"] for qid, v in lp_obj.items()}
        print(f"  {model}: {len(rows)} q | sim {len(sim_by_model[model])} | logprobs {len(lp_by_model[model])}")

    # Sanity check (a): uniform weights -> should reproduce paper's 2.94x.
    print("\n=== (a) Uniform 1/M weights (sanity check) ===")
    def w_uniform(qid, lp): return np.ones(N_RESPONSES) / N_RESPONSES
    a_pool, r_pool, a_pm = run_one_setting(rows_by_model, sim_by_model, lp_by_model, w_uniform)
    s_uniform = summarize(a_pool, r_pool)
    print(f"  POOLED: n_adm={s_uniform['n_adm']}, n_rej={s_uniform['n_rej']} ({s_uniform['rejection_rate']*100:.2f}%), "
          f"adm={s_uniform['adm_acc']*100:.1f}%, rej={s_uniform['rej_acc']*100:.1f}%, "
          f"ratio={s_uniform['ratio']:.2f}x [{s_uniform['ci_lo']:.2f}, {s_uniform['ci_hi']:.2f}], p={s_uniform['fisher_p']:.2e}")
    per_model_uniform = {}
    for model in PRED1_FILES:
        sm = summarize(a_pm[model]["adm"], a_pm[model]["rej"])
        per_model_uniform[model] = sm
        print(f"  {model:7s}: n_adm={sm['n_adm']}, n_rej={sm['n_rej']:>3d} ({sm['rejection_rate']*100:.2f}%), "
              f"adm={sm['adm_acc']*100:.1f}%, rej={sm['rej_acc']*100:.1f}%, ratio={sm['ratio']:.2f}x, p={sm['fisher_p']:.2e}")

    # (b) Softmax-exp(logprob) weights.
    have_logprobs = all(len(lp_by_model[m]) > 0 for m in PRED1_FILES)
    if not have_logprobs:
        print("\nSkipping (b): logprobs not available for all models.")
        out = {"uniform": {"pooled": s_uniform, "per_model": per_model_uniform}}
        with open(SCRIPT_DIR / "genprob_reweighting_results.json", "w") as f:
            json.dump(out, f, indent=2)
        return

    print("\n=== (b) Softmax-exp(mean-token-logprob) weights ===")
    def w_logprob(qid, lp):
        if lp is None or len(lp) < N_RESPONSES:
            return np.ones(N_RESPONSES) / N_RESPONSES
        return softmax_weights(lp[:N_RESPONSES])
    a_pool_b, r_pool_b, a_pm_b = run_one_setting(rows_by_model, sim_by_model, lp_by_model, w_logprob)
    s_logprob = summarize(a_pool_b, r_pool_b)
    print(f"  POOLED: n_adm={s_logprob['n_adm']}, n_rej={s_logprob['n_rej']} ({s_logprob['rejection_rate']*100:.2f}%), "
          f"adm={s_logprob['adm_acc']*100:.1f}%, rej={s_logprob['rej_acc']*100:.1f}%, "
          f"ratio={s_logprob['ratio']:.2f}x [{s_logprob['ci_lo']:.2f}, {s_logprob['ci_hi']:.2f}], p={s_logprob['fisher_p']:.2e}")
    per_model_logprob = {}
    for model in PRED1_FILES:
        sm = summarize(a_pm_b[model]["adm"], a_pm_b[model]["rej"])
        per_model_logprob[model] = sm
        print(f"  {model:7s}: n_adm={sm['n_adm']}, n_rej={sm['n_rej']:>3d} ({sm['rejection_rate']*100:.2f}%), "
              f"adm={sm['adm_acc']*100:.1f}%, rej={sm['rej_acc']*100:.1f}%, ratio={sm['ratio']:.2f}x, p={sm['fisher_p']:.2e}")

    out = {
        "thresholds": THRESHOLDS,
        "n_responses": N_RESPONSES,
        "uniform": {"pooled": s_uniform, "per_model": per_model_uniform},
        "logprob": {"pooled": s_logprob, "per_model": per_model_logprob},
    }
    with open(SCRIPT_DIR / "genprob_reweighting_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to genprob_reweighting_results.json")


if __name__ == "__main__":
    main()
