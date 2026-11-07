"""Post hoc, 2026-11-07. Does the agreement profile across thresholds say more than one threshold?

For every retained response r (six per question-model pair, equal weights), u_theta(r) is the share
of the six responses in r's cluster at threshold theta, on the registered grid {0.50, ..., 0.95}.

1. Check: pooled admissibility ratios of plurality at 0.50 (Gamma-maximax), plurality at 0.95
   (Gamma-maximin) and C_sem, to be compared with Table 7.
2. Threshold choice: log ratio of Gamma-maximax minus log ratio of Gamma-maximin, with a bootstrap
   interval over question-model pairs (2,000 resamples), the unit used for the paper's other
   differences between rules.
3. Direct test: response-level AUROC for correctness, out of fold, with folds formed by question
   (all models' responses to a question share a fold; 5 folds). "One threshold" scores a test
   response by u_theta*(r), with theta* chosen on the training folds to maximize AUROC. "Profile"
   scores it by an L2-regularized logistic regression on the ten u_theta(r), fit on the training
   folds. Gain = AUROC(profile) - AUROC(one threshold), with a bootstrap interval over questions
   (1,000 resamples) holding the out-of-fold scores fixed.

Labels: TriviaQA full-alias rescoring (triviaqa_rescore/results_full); NQ-Open and TruthfulQA from
the registered runs (substring scorer); TruthfulQA (judge) from the Claude Haiku 4.5 rescoring;
ARC-Challenge and MMLU (Gemma) from the registered runs (registered scorer). Labels are chosen as in
criteria_comparison.py, so step 1 reproduces Table 7. Similarity: the cached matrices, which
reproduce the registered partitions except on 23 of 2,000 ARC pairs, 304 of 3,268 TruthfulQA pairs,
and 1 of 500 MMLU pairs. Writes direct_test.json.
"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SR = os.path.join(HERE, '..', 'stored_runs')
TQ = os.path.join(HERE, '..', 'triviaqa_rescore', 'results_full')
GRID = [round(0.50 + 0.05 * i, 2) for i in range(10)]
MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']


def as_list(a):
    return [int(bool(x)) for x in (json.loads(a) if isinstance(a, str) else a)]


def from_records(path, dataset):
    out = {}
    for r in json.load(open(path)):
        if r['dataset'] == dataset:
            out[(r['model'], r['qid'])] = as_list(r['accuracies'])
    return out


def from_prereg(dataset):
    out = {}
    # Same file selection as criteria_comparison.py, which builds Table 7.
    for fn in sorted(glob.glob(os.path.join(SR, 'modal_data', f'prereg_{dataset}_*.json'))):
        m = next((m for m in MODELS if f'_{m}_' in fn or fn.endswith(f'_{m}.json')), None)
        if m:
            for q, v in json.load(open(fn))['per_question'].items():
                out[(m, q)] = as_list(v['accuracies'])
    return out


def from_judge():
    out = {}
    for m in MODELS:
        for q, v in json.load(open(os.path.join(SR, 'results', f'rescore_{m}_judge.json')))['per_question'].items():
            out[(m, q)] = as_list(v['accuracies'])
    return out


LABELS = {
    'triviaqa': ('triviaqa', lambda: from_records(os.path.join(TQ, 'kle_snne_triviaqa_results.json'), 'triviaqa')),
    'nq_open': ('nq_open', lambda: from_prereg('nq_open')),
    'truthfulqa': ('truthfulqa', lambda: from_prereg('truthfulqa')),
    'truthfulqa_judge': ('truthfulqa', from_judge),
    'arc': ('arc', lambda: from_prereg('arc')),
    'mmlu': ('mmlu', lambda: from_prereg('mmlu')),
}


def cluster_share(S, th):
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
    roots = [find(i) for i in range(n)]
    return [roots.count(r) / n for r in roots]


def auroc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, int)
    order = score.argsort(kind='mergesort'); r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, r); r = (s / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic_fit(X, y, lam=1e-3, iters=50):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + lam * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + lam * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return lambda Xn: np.hstack([np.ones((len(Xn), 1)), (Xn - mu) / sd]) @ w


def ratio(adm, y):
    a, r = y[adm].mean(), y[~adm].mean()
    return a / r


out = {}
rng = np.random.default_rng(0)
for name, (dataset, loader) in LABELS.items():
    labels = loader()
    U, Y, PAIR, QID = [], [], [], []
    for m in MODELS:
        f = os.path.join(SR, 'sim_cache', f'sim_matrices_{dataset}_{m}.json')
        if not os.path.exists(f):
            continue
        for q, S in json.load(open(f)).items():
            acc = labels.get((m, q))
            if acc is None or len(acc) != len(S):
                continue
            u = np.array([cluster_share(S, t) for t in GRID]).T  # responses x thresholds
            U.append(u); Y.extend(acc); PAIR.extend([f'{m}|{q}'] * len(S)); QID.extend([q] * len(S))
    U = np.vstack(U); Y = np.array(Y); PAIR = np.array(PAIR); QID = np.array(QID)
    res = {'responses': int(len(Y)), 'question_model_pairs': int(len(set(PAIR)))}

    # 1. Table 7 check, and 2. the endpoint comparison with a pair bootstrap
    pairs, pidx = np.unique(PAIR, return_inverse=True)
    def adm_at(k):
        mx = np.zeros(len(pairs)); np.maximum.at(mx, pidx, U[:, k]); return U[:, k] == mx[pidx]
    a_max, a_min = adm_at(0), adm_at(len(GRID) - 1)
    a_sem = np.zeros(len(Y), bool)
    for k in range(len(GRID)):
        a_sem |= adm_at(k)
    res['ratio_gamma_maximax'], res['ratio_gamma_maximin'], res['ratio_csem'] = ratio(a_max, Y), ratio(a_min, Y), ratio(a_sem, Y)
    res['rejection_gamma_maximax'], res['rejection_gamma_maximin'] = float(1 - a_max.mean()), float(1 - a_min.mean())
    rows = [np.flatnonzero(pidx == i) for i in range(len(pairs))]
    diffs = []
    for _ in range(2000):
        ix = np.concatenate([rows[i] for i in rng.integers(0, len(pairs), len(pairs))])
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.log(ratio(a_max[ix], Y[ix])) - np.log(ratio(a_min[ix], Y[ix]))
        if np.isfinite(d):
            diffs.append(d)
    d0 = np.log(res['ratio_gamma_maximax']) - np.log(res['ratio_gamma_maximin'])
    res['log_ratio_maximax_minus_maximin'] = [float(d0), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

    # 3. Direct test, folds by question
    qs = np.array(sorted(set(QID))); rng.shuffle(qs)
    fold_of = {q: i % 5 for i, q in enumerate(qs)}
    fold = np.array([fold_of[q] for q in QID])
    s_one, s_prof = np.zeros(len(Y)), np.zeros(len(Y))
    for k in range(5):
        tr, te = fold != k, fold == k
        best = max(range(len(GRID)), key=lambda j: auroc(U[tr, j], Y[tr]))
        s_one[te] = U[te, best]
        s_prof[te] = logistic_fit(U[tr], Y[tr].astype(float))(U[te])
    res['auroc_one_threshold'], res['auroc_profile'] = auroc(s_one, Y), auroc(s_prof, Y)
    qrows = {q: np.flatnonzero(QID == q) for q in qs}
    gains = []
    for _ in range(1000):
        ix = np.concatenate([qrows[q] for q in rng.choice(qs, len(qs))])
        gains.append(auroc(s_prof[ix], Y[ix]) - auroc(s_one[ix], Y[ix]))
    g0 = res['auroc_profile'] - res['auroc_one_threshold']
    res['auroc_gain'] = [float(g0), float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]
    out[name] = res
    print(name, {k: (round(v, 3) if isinstance(v, float) else ([round(x, 3) for x in v] if isinstance(v, list) else v)) for k, v in res.items()})
json.dump(out, open(os.path.join(HERE, 'direct_test.json'), 'w'), indent=1)
