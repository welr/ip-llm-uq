"""
Matched-retention pool-accuracy experiment (Reviewer W1).

Question: At a matched retention fraction f (= admissible-fraction), does the
admissibility partition yield higher retained-pool accuracy than scalar-based
filtering?

Compared methods (same retention rate 1 - n_rejected / n_total):
    (A) Admissibility partition: keep responses where admissible = True.
    (B) Response-level top-by-s(r): s(r) = max_theta P_theta(C_k(r)^theta).
        Keep highest-s responses until retention = f.
    (C) Question-level keep-by-(low 1-maxCP): drop questions with highest
        1-maxCP (lowest confidence) until retention = f; keep all 6 responses
        of the rest.
    (D) Question-level keep-by-(low SNNE): same as (C) with SNNE.

Dataset: TriviaQA, 4 base models (Qwen, Mistral, Llama, Gemma), 500 questions
each, 6 responses per question, 12000 total responses.

Usage:
    ~/ip_llm/bin/python experiments/matched_retention_w1.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MODELS = ["Qwen", "Mistral", "Llama", "Gemma"]
RNG_SEED = 42


def cluster_connected_components(sim, theta):
    """Return list of clusters (each a list of indices) at threshold theta."""
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


def admissibility_and_scalar(sim_matrix, thresholds=THRESHOLDS):
    """
    For one question (6x6 sim matrix), return:
        admissible: list[bool] of length M
        s_resp: list[float] response-level max cluster prob across thetas
        q_maxcp: float, 1 - max over thetas of max cluster prob (= 1 - maxCP)
    """
    m = len(sim_matrix)
    admissible = [False] * m
    s_resp = [0.0] * m
    q_max_prob = 0.0
    for t in thresholds:
        clusters = cluster_connected_components(sim_matrix, t)
        sizes = [len(c) for c in clusters]
        max_size = max(sizes)
        q_max_prob = max(q_max_prob, max_size / m)
        for c in clusters:
            p = len(c) / m
            for r in c:
                s_resp[r] = max(s_resp[r], p)
            if len(c) == max_size:
                for r in c:
                    admissible[r] = True
    return admissible, s_resp, 1.0 - q_max_prob


def bootstrap_accuracy(correct_mask, n_boot=5000, rng=None):
    """Return (mean, lo, hi) bootstrap estimates of accuracy."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    correct_mask = np.asarray(correct_mask, dtype=float)
    n = len(correct_mask)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = correct_mask[idx].mean(axis=1)
    return float(correct_mask.mean()), float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


def main():
    script_dir = Path(__file__).parent
    sim_cache_dir = script_dir / "sim_cache"

    # Load scalar signals + accuracies from kle_snne_triviaqa_results.
    with open(script_dir / "kle_snne_triviaqa_results.json") as f:
        records = json.load(f)
    records = [r for r in records if r["dataset"] == "triviaqa"]
    # Index by (model, qid)
    by_qm = {(r["model"], r["qid"]): r for r in records}

    # Per-response table.
    rows = []  # dicts with model, qid, resp_idx, correct, admissible, s_resp, q_maxcp, snne
    for model in MODELS:
        with open(sim_cache_dir / f"sim_matrices_triviaqa_{model}.json") as f:
            sims = json.load(f)
        for qid, sim_matrix in sims.items():
            rec = by_qm.get((model, qid))
            if rec is None:
                continue
            adm, s_resp, q_maxcp = admissibility_and_scalar(sim_matrix)
            for i, (a, s, c) in enumerate(zip(adm, s_resp, rec["accuracies"])):
                rows.append({
                    "model": model,
                    "qid": qid,
                    "resp_idx": i,
                    "correct": int(c),
                    "admissible": bool(a),
                    "s_resp": float(s),
                    "q_maxcp": float(q_maxcp),
                    "snne": float(rec["snne"]),
                    "kle": float(rec["kle"]),
                    "se075": float(rec["se_075"]),
                })

    # Build arrays
    n_total = len(rows)
    correct = np.array([r["correct"] for r in rows])
    adm = np.array([r["admissible"] for r in rows])
    s_resp = np.array([r["s_resp"] for r in rows])
    q_maxcp = np.array([r["q_maxcp"] for r in rows])
    snne = np.array([r["snne"] for r in rows])
    kle = np.array([r["kle"] for r in rows])
    se075 = np.array([r["se075"] for r in rows])

    n_admissible = int(adm.sum())
    n_rejected = n_total - n_admissible
    retention_rate = n_admissible / n_total

    print(f"=== TriviaQA matched-retention comparison ===")
    print(f"Total responses: {n_total}")
    print(f"Admissible: {n_admissible} ({retention_rate*100:.2f}%)")
    print(f"Rejected:   {n_rejected} ({(1-retention_rate)*100:.2f}%)")
    print()

    # Baseline: retain all
    base_acc, base_lo, base_hi = bootstrap_accuracy(correct)
    print(f"(baseline)   retain-all:         N={n_total:>5}  acc={base_acc*100:.2f}% [{base_lo*100:.2f}, {base_hi*100:.2f}]")

    # (A) admissibility partition
    a_acc, a_lo, a_hi = bootstrap_accuracy(correct[adm])
    print(f"(A) partition: admissibility    N={n_admissible:>5}  acc={a_acc*100:.2f}% [{a_lo*100:.2f}, {a_hi*100:.2f}]")

    # (B) response-level top-by-s_resp: keep n_admissible responses with highest s_resp.
    # Break ties randomly (seeded).
    rng = np.random.default_rng(RNG_SEED)
    jitter = rng.normal(0, 1e-9, size=n_total)
    s_with_tie = s_resp + jitter
    order = np.argsort(-s_with_tie)  # descending
    keep_mask_B = np.zeros(n_total, dtype=bool)
    keep_mask_B[order[:n_admissible]] = True
    b_acc, b_lo, b_hi = bootstrap_accuracy(correct[keep_mask_B])
    # Overlap with admissibility
    overlap_B = int((keep_mask_B & adm).sum())
    print(f"(B) scalar (response-level s): N={keep_mask_B.sum():>5}  acc={b_acc*100:.2f}% [{b_lo*100:.2f}, {b_hi*100:.2f}]  overlap with (A): {overlap_B}/{n_admissible}")

    # (C) question-level drop-by-(high 1-maxCP), match retention.
    # Group by (model, qid). Drop questions with highest q_maxcp until retention ≤ target.
    by_qm_indices = {}
    for i, r in enumerate(rows):
        by_qm_indices.setdefault((r["model"], r["qid"]), []).append(i)
    qm_scores = [(k, rows[v[0]]["q_maxcp"]) for k, v in by_qm_indices.items()]
    # sort by q_maxcp descending (highest = least confident)
    qm_scores.sort(key=lambda x: -x[1])
    keep_mask_C = np.ones(n_total, dtype=bool)
    dropped = 0
    for (k, _) in qm_scores:
        idxs = by_qm_indices[k]
        if n_total - (dropped + len(idxs)) < n_admissible:
            break
        for i in idxs:
            keep_mask_C[i] = False
        dropped += len(idxs)
    # Now drop within the first question not yet dropped, as needed, randomly.
    extra_needed = keep_mask_C.sum() - n_admissible
    # But we can't partially drop within the question-level scalar framework.
    # So report at *closest* retention above target.
    c_acc, c_lo, c_hi = bootstrap_accuracy(correct[keep_mask_C])
    print(f"(C) scalar (question-level 1-maxCP): N={keep_mask_C.sum():>5}  acc={c_acc*100:.2f}% [{c_lo*100:.2f}, {c_hi*100:.2f}]  (closest retention above target)")

    # (D) question-level drop-by-(high SNNE). Note: -SNNE in paper is the score (higher = more confident); so we drop by high SNNE (more uncertain).
    qm_scores_snne = [(k, rows[v[0]]["snne"]) for k, v in by_qm_indices.items()]
    qm_scores_snne.sort(key=lambda x: -x[1])
    keep_mask_D = np.ones(n_total, dtype=bool)
    dropped = 0
    for (k, _) in qm_scores_snne:
        idxs = by_qm_indices[k]
        if n_total - (dropped + len(idxs)) < n_admissible:
            break
        for i in idxs:
            keep_mask_D[i] = False
        dropped += len(idxs)
    d_acc, d_lo, d_hi = bootstrap_accuracy(correct[keep_mask_D])
    print(f"(D) scalar (question-level SNNE):    N={keep_mask_D.sum():>5}  acc={d_acc*100:.2f}% [{d_lo*100:.2f}, {d_hi*100:.2f}]  (closest retention above target)")

    # (E) question-level drop-by-(high SE(0.75)).
    qm_scores_se = [(k, rows[v[0]]["se075"]) for k, v in by_qm_indices.items()]
    qm_scores_se.sort(key=lambda x: -x[1])
    keep_mask_E = np.ones(n_total, dtype=bool)
    dropped = 0
    for (k, _) in qm_scores_se:
        idxs = by_qm_indices[k]
        if n_total - (dropped + len(idxs)) < n_admissible:
            break
        for i in idxs:
            keep_mask_E[i] = False
        dropped += len(idxs)
    e_acc, e_lo, e_hi = bootstrap_accuracy(correct[keep_mask_E])
    print(f"(E) scalar (question-level SE(.75)):  N={keep_mask_E.sum():>5}  acc={e_acc*100:.2f}% [{e_lo*100:.2f}, {e_hi*100:.2f}]  (closest retention above target)")

    # (F) question-level drop-by-(high KLE).
    qm_scores_kle = [(k, rows[v[0]]["kle"]) for k, v in by_qm_indices.items()]
    qm_scores_kle.sort(key=lambda x: -x[1])
    keep_mask_F = np.ones(n_total, dtype=bool)
    dropped = 0
    for (k, _) in qm_scores_kle:
        idxs = by_qm_indices[k]
        if n_total - (dropped + len(idxs)) < n_admissible:
            break
        for i in idxs:
            keep_mask_F[i] = False
        dropped += len(idxs)
    f_acc, f_lo, f_hi = bootstrap_accuracy(correct[keep_mask_F])
    print(f"(F) scalar (question-level KLE):      N={keep_mask_F.sum():>5}  acc={f_acc*100:.2f}% [{f_lo*100:.2f}, {f_hi*100:.2f}]  (closest retention above target)")

    # -------- Rejection-precision at matched budget --------
    # Question: at matched rejection budget (same count as admissibility rejects),
    # which filter catches the most hallucinations? Precision = fraction of rejected
    # responses that are actually wrong.
    n_rej_target = n_rejected
    print()
    print(f"--- Rejection-precision at matched budget (n_rejected={n_rej_target}) ---")

    # (A) admissibility rejection
    rej_A = ~adm
    prec_A = 1 - correct[rej_A].mean()
    print(f"(A) admissibility rejected:     N={rej_A.sum():>5}  precision={prec_A*100:.2f}% (fraction wrong)")

    # (B) response-level bottom-by-s_resp: reject n_rej_target lowest-s responses
    rej_B_idx = np.argsort(s_with_tie)[:n_rej_target]
    rej_B = np.zeros(n_total, dtype=bool)
    rej_B[rej_B_idx] = True
    prec_B = 1 - correct[rej_B].mean()
    overlap_reject_B = int((rej_B & rej_A).sum())
    print(f"(B) bottom-by-s_resp:           N={rej_B.sum():>5}  precision={prec_B*100:.2f}%  overlap with (A): {overlap_reject_B}/{n_rej_target}")

    # (C-F) question-level: drop questions with worst scalar until n_rejected matches, pad with remaining.
    # For fair matched-budget comparison, at question-level we can only drop whole questions.
    # Closest-to-target = drop floor(n_rej_target / 6) worst questions, then include as many responses of the next-worst question as fit.
    def q_level_reject(qscores_desc):
        # qscores_desc: list of (qm_key, score) sorted descending (worst first)
        rej = np.zeros(n_total, dtype=bool)
        dropped = 0
        for (k, _) in qscores_desc:
            idxs = by_qm_indices[k]
            if dropped + len(idxs) <= n_rej_target:
                for i in idxs:
                    rej[i] = True
                dropped += len(idxs)
            else:
                remaining = n_rej_target - dropped
                for i in idxs[:remaining]:
                    rej[i] = True
                dropped += remaining
                break
        return rej

    for name, scores in [
        ("1-maxCP", [(k, rows[v[0]]["q_maxcp"]) for k, v in by_qm_indices.items()]),
        ("SNNE",    [(k, rows[v[0]]["snne"])    for k, v in by_qm_indices.items()]),
        ("SE(.75)", [(k, rows[v[0]]["se075"])   for k, v in by_qm_indices.items()]),
        ("KLE",     [(k, rows[v[0]]["kle"])     for k, v in by_qm_indices.items()]),
    ]:
        scores.sort(key=lambda x: -x[1])  # worst first
        rej_Q = q_level_reject(scores)
        prec_Q = 1 - correct[rej_Q].mean()
        print(f"({name.ljust(8)}) q-level bottom:    N={rej_Q.sum():>5}  precision={prec_Q*100:.2f}%")

    print()
    # Paired-bootstrap delta (A) vs (B) — only test legitimate response-level baseline.
    # (C-F) operate at question granularity, different N, so delta is not apples-to-apples; we report them as context only.
    rng2 = np.random.default_rng(RNG_SEED + 1)
    a_mask = np.asarray(adm, dtype=bool)
    b_mask = keep_mask_B
    # Joint bootstrap at response level
    n_boot = 10000
    idx = rng2.integers(0, n_total, size=(n_boot, n_total))
    deltas = []
    for b in range(n_boot):
        boot_correct = correct[idx[b]]
        boot_a = a_mask[idx[b]]
        boot_b = b_mask[idx[b]]
        acc_a = boot_correct[boot_a].mean() if boot_a.any() else 0
        acc_b = boot_correct[boot_b].mean() if boot_b.any() else 0
        deltas.append(acc_a - acc_b)
    deltas = np.array(deltas)
    delta_mean = deltas.mean()
    delta_lo, delta_hi = np.quantile(deltas, [0.025, 0.975])
    print()
    print(f"Paired bootstrap Δ(A - B) = {delta_mean*100:+.2f}pp [{delta_lo*100:+.2f}, {delta_hi*100:+.2f}] (10k resamples)")

    # Save JSON.
    out = {
        "n_total": n_total,
        "n_admissible": n_admissible,
        "n_rejected": n_rejected,
        "retention_rate": retention_rate,
        "baseline_acc": base_acc,
        "admissibility": {"n": n_admissible, "acc": a_acc, "ci_lo": a_lo, "ci_hi": a_hi},
        "response_level_s": {"n": int(keep_mask_B.sum()), "acc": b_acc, "ci_lo": b_lo, "ci_hi": b_hi, "overlap_with_admissibility": overlap_B},
        "question_level_1_maxcp": {"n": int(keep_mask_C.sum()), "acc": c_acc, "ci_lo": c_lo, "ci_hi": c_hi},
        "question_level_snne": {"n": int(keep_mask_D.sum()), "acc": d_acc, "ci_lo": d_lo, "ci_hi": d_hi},
        "question_level_se075": {"n": int(keep_mask_E.sum()), "acc": e_acc, "ci_lo": e_lo, "ci_hi": e_hi},
        "question_level_kle": {"n": int(keep_mask_F.sum()), "acc": f_acc, "ci_lo": f_lo, "ci_hi": f_hi},
        "paired_bootstrap_A_minus_B": {"mean": delta_mean, "ci_lo": float(delta_lo), "ci_hi": float(delta_hi)},
    }
    with open(script_dir / "matched_retention_w1_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print()
    print(f"Saved results to {script_dir/'matched_retention_w1_results.json'}")


if __name__ == "__main__":
    main()
