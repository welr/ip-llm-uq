"""Post hoc, 2026-11-07. Why the TruthfulQA SE-width AUROC differs between the registered runs and
the cached matrices (.385 against .490).

The registered runs stored SE at theta = 0.50 and 0.95 for every question-model pair
(modal_data/prereg_truthfulqa_*.json; audit_p1_p2/audit_registered_form.py computes the stored
AUROC from them). The cached matrices (sim_cache/, built by build_sim_cache_tqa_nq.py with the same
NLI model, averaging and 256-token limit) are recomputed here at the same two thresholds, in bits.
Reports, per model and pooled: the AUROC of width for majority-vote error under each source, the
share of pairs whose stored SE differs from the recomputed SE at each endpoint, and the share of
pair scores within 0.02 of 0.95. Writes width_fragility.json.
"""
import glob, json, os
import numpy as np

SR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stored_runs')


def clusters(S, th):
    n = len(S)
    lab = list(range(n))

    def find(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]
            x = lab[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if S[i][j] >= th:
                lab[find(i)] = find(j)
    return [find(i) for i in range(n)]


def bits(c):
    _, k = np.unique(c, return_counts=True)
    p = k / k.sum()
    return float(-(p * np.log2(p)).sum())


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    o = s.argsort(kind='mergesort'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    _, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    t = np.zeros(len(c)); np.add.at(t, inv, r); r = (t / c)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


out, Y, WS, WC, UP, LO = {}, [], [], [], [], []
for m in ['Qwen', 'Mistral', 'Llama', 'Gemma']:
    P = json.load(open(sorted(glob.glob(os.path.join(SR, 'modal_data', f'prereg_truthfulqa_{m}_*.json')))[0]))['per_question']
    C = json.load(open(os.path.join(SR, 'sim_cache', f'sim_matrices_truthfulqa_{m}.json')))
    y, ws, wc, up, lo, sc = [], [], [], [], [], []
    for q, S in C.items():
        x = P[q]
        hi_c, lo_c = bits(clusters(S, .95)), bits(clusters(S, .50))
        y.append(int(not x['mv_correct'])); ws.append(round(x['se_upper'] - x['se_lower'], 9)); wc.append(round(hi_c - lo_c, 9))  # rounding makes equal widths tie exactly
        up.append(abs(x['se_upper'] - hi_c) > 1e-6); lo.append(abs(x['se_lower'] - lo_c) > 1e-6)
        S = np.array(S); sc.extend(S[np.triu_indices(len(S), 1)])
    sc = np.array(sc)
    out[m] = dict(pairs=len(y), auroc_stored=auroc(ws, y), auroc_recomputed=auroc(wc, y),
                  share_se095_differs=float(np.mean(up)), share_se050_differs=float(np.mean(lo)),
                  share_scores_within_002_of_095=float(np.mean((sc >= .93) & (sc < .97))))
    Y += y; WS += ws; WC += wc; UP += up; LO += lo
out['pooled'] = dict(pairs=len(Y), auroc_stored=auroc(WS, Y), auroc_recomputed=auroc(WC, Y),
                     share_se095_differs=float(np.mean(UP)), share_se050_differs=float(np.mean(LO)))
for k, v in out.items():
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'width_fragility.json'), 'w'), indent=1)
