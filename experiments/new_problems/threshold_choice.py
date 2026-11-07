"""Post hoc, 2026-11-07. Which threshold, and does a richer model of the profile help?

Extends direct_test.py on the same responses, labels, and folds (its random sequence is replayed, so
the chosen thresholds and the profile gain reproduce direct_test.json exactly). Per label set:

1. auroc_by_theta: response-level AUROC of the agreement share u_theta(r) at each grid threshold.
2. chosen_thresholds, auroc_chosen: the threshold chosen on the training folds, as in direct_test.py.
3. d50_minus_chosen, d50_minus_85, d50_minus_95: AUROC at a fixed 0.50 minus the chosen threshold,
   0.85, and 0.95, with 95% intervals from 1,000 resamples of questions.
4. plurality_d50_minus_85: the same for the binary plurality indicator at 0.50 and 0.85.
5. boosted_gain: a gradient-boosted classifier on the ten u_theta(r) in place of the logistic
   regression, out of fold, minus the chosen threshold; boosted_by_setting repeats it for the
   settings in BOOST.
6. changing_*: on question-model pairs whose plurality set changes across the grid, the logistic
   profile minus one threshold, with both fit on those pairs only (fit_on_subset) or on all pairs.
Writes threshold_choice.json. Optional arguments name the label sets to compute (the rest are only
replayed for the random sequence); results merge into an existing threshold_choice.json. Loaded
agreement shares are cached in the system temp folder.
"""
import json, os, sys, tempfile
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'direct_test.py')
MARK = '\nout = {}\n'
code = open(SRC).read()
assert code.count(MARK) == 1
g = {'__file__': SRC, '__name__': 'direct_test_defs'}
exec(compile(code.split(MARK)[0], SRC, 'exec'), g)  # definitions only: loaders, cluster_share, auroc, logistic_fit
LABELS, MODELS, SR, GRID = g['LABELS'], g['MODELS'], g['SR'], g['GRID']
cluster_share, auroc, logistic_fit = g['cluster_share'], g['auroc'], g['logistic_fit']

rng = np.random.default_rng(0)    # replays direct_test.py's draws, so folds match
BOOST = {'default': dict(max_iter=200, learning_rate=0.05),
         'depth2': dict(max_depth=2, max_iter=200, learning_rate=0.05),
         'depth3': dict(max_depth=3, max_iter=100, learning_rate=0.1),
         'stumps': dict(max_depth=1, max_iter=300, learning_rate=0.05)}
OUTF = os.path.join(HERE, 'threshold_choice.json')
WANT = sys.argv[1:] or list(LABELS)
out = json.load(open(OUTF)) if sys.argv[1:] and os.path.exists(OUTF) else {}
CACHE = os.path.join(tempfile.gettempdir(), 'threshold_choice_cache')
os.makedirs(CACHE, exist_ok=True)


def load(name, dataset, loader):
    c = os.path.join(CACHE, f'{name}.npz')
    if os.path.exists(c):
        z = np.load(c); return z['U'], z['Y'], z['PAIR'], z['QID']
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
            U.append(np.array([cluster_share(S, t) for t in GRID]).T)
            Y.extend(acc); PAIR.extend([f'{m}|{q}'] * len(S)); QID.extend([q] * len(S))
    U = np.vstack(U); Y = np.array(Y); PAIR = np.array(PAIR); QID = np.array(QID)
    np.savez(c, U=U, Y=Y, PAIR=PAIR, QID=QID)
    return U, Y, PAIR, QID


for name, (dataset, loader) in LABELS.items():
    U, Y, PAIR, QID = load(name, dataset, loader)
    pairs, pidx = np.unique(PAIR, return_inverse=True)
    for _ in range(2000):
        rng.integers(0, len(pairs), len(pairs))
    qs = np.array(sorted(set(QID))); rng.shuffle(qs)
    fo = {q: i % 5 for i, q in enumerate(qs)}; fold = np.array([fo[q] for q in QID])
    for _ in range(1000):
        rng.choice(qs, len(qs))
    rng2 = np.random.default_rng([1, list(LABELS).index(name)])  # per label set, so chunked runs agree
    if name not in WANT:
        continue

    mx = np.zeros((len(pairs), len(GRID)))
    for j in range(len(GRID)):
        np.maximum.at(mx[:, j], pidx, U[:, j])
    plur = np.stack([U[:, j] == mx[pidx, j] for j in range(len(GRID))], 1)
    changes = np.array([len({tuple(plur[pidx == i, j]) for j in range(len(GRID))}) > 1 for i in range(len(pairs))])
    sub = changes[pidx]

    def oof(mask, kind):
        s, chosen = np.full(len(Y), np.nan), []
        for k in range(5):
            tr, te = (fold != k) & mask, (fold == k) & mask
            if kind == 'one':
                b = max(range(len(GRID)), key=lambda j: auroc(U[tr, j], Y[tr])); chosen.append(GRID[b]); s[te] = U[te, b]
            elif kind == 'logit':
                s[te] = logistic_fit(U[tr], Y[tr].astype(float))(U[te])
            else:
                clf = HistGradientBoostingClassifier(random_state=0, **BOOST[kind]).fit(U[tr], Y[tr])
                s[te] = clf.predict_proba(U[te])[:, 1]
        return s, chosen

    allm = np.ones(len(Y), bool)
    (one, chosen), (prof, _) = oof(allm, 'one'), oof(allm, 'logit')
    gbs = {k: oof(allm, k)[0] for k in BOOST}
    one_s, _ = oof(sub, 'one'); prof_s, _ = oof(sub, 'logit')
    qrows = {q: np.flatnonzero(QID == q) for q in qs}
    draws = [np.concatenate([qrows[q] for q in rng2.choice(qs, len(qs))]) for _ in range(1000)]

    def diff(a, b, rows=None):
        idx = np.arange(len(Y)) if rows is None else rows
        pt = auroc(a[idx], Y[idx]) - auroc(b[idx], Y[idx])
        ds = []
        for ix in draws:
            ix = ix if rows is None else ix[sub[ix]]
            ds.append(auroc(a[ix], Y[ix]) - auroc(b[ix], Y[ix]))
        return [float(pt), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]

    j50, j85, j95 = GRID.index(0.5), GRID.index(0.85), GRID.index(0.95)
    S = np.flatnonzero(sub)
    res = {
        'auroc_by_theta': {f'{t:.2f}': auroc(U[:, j], Y) for j, t in enumerate(GRID)},
        'chosen_thresholds': chosen, 'auroc_chosen': auroc(one, Y),
        'profile_gain': auroc(prof, Y) - auroc(one, Y),
        'd50_minus_chosen': diff(U[:, j50], one), 'd50_minus_85': diff(U[:, j50], U[:, j85]), 'd50_minus_95': diff(U[:, j50], U[:, j95]),
        'plurality_d50_minus_85': diff(plur[:, j50].astype(float), plur[:, j85].astype(float)),
        'boosted_gain': diff(gbs['default'], one),
        'boosted_by_setting': {k: diff(v, one) for k, v in gbs.items()},
        'changing_pairs_share': float(changes.mean()),
        'changing_gain_fit_on_subset': diff(prof_s, one_s, S),
        'changing_gain_fit_on_all': diff(prof, one, S),
    }
    out[name] = res
    r = lambda v: [round(x, 3) for x in v]
    print(name, 'chosen', chosen, 'auroc .50 %.3f chosen %.3f' % (res['auroc_by_theta']['0.50'], res['auroc_chosen']),
          '| 50-chosen', r(res['d50_minus_chosen']), '50-85', r(res['d50_minus_85']), '50-95', r(res['d50_minus_95']),
          'plur50-85', r(res['plurality_d50_minus_85']), '| boosted', r(res['boosted_gain']),
          '| changing %.2f' % res['changing_pairs_share'], r(res['changing_gain_fit_on_subset']), r(res['changing_gain_fit_on_all']), flush=True)
    json.dump(out, open(OUTF, 'w'), indent=1)
