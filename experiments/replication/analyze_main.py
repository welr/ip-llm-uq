"""Frozen confirmatory analysis for the PopQA replication. Run with ~/ip_llm/bin/python.

  --validate triviaqa|nq_open   run on stored discovery data, to test this code against known values
  --data DIR                    run on the confirmatory files; refuses without REGISTRATION_LOCK.txt

Primary arm: registered cut, word-bounded scorer, first six responses. Other arms and retention rules are pre-specified sensitivity analyses.
"""
import argparse, collections, glob, json, math, os, re
import numpy as np
from scipy import stats
from scoring import registered_rule, word_bounded_rule
MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']; GRID = [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95]
CF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', '')
def comps(S, t):
    n = len(S); p = list(range(n))
    def f(a):
        while p[a] != a: p[a] = p[p[a]]; a = p[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if S[i][j] >= t: p[f(i)] = f(j)
    g = collections.defaultdict(list)
    for i in range(n): g[f(i)].append(i)
    return list(g.values())
def rules(S):
    """Union (registered rule), the variant without all-singleton ties above theta_min, plurality at theta_min, plurality at theta_max."""
    C = [comps(S, t) for t in GRID]; union = set(); base = set(); top = set()
    for c in C[0]:
        if len(c) == max(map(len, C[0])): base |= set(c)
    for c in C[-1]:
        if len(c) == max(map(len, C[-1])): top |= set(c)
    nt = set(base)
    for cs in C:
        mx = max(map(len, cs))
        for c in cs:
            if len(c) == mx:
                union |= set(c)
                if mx >= 2: nt |= set(c)
    return dict(union=union, no_ties=nt, gmax=base, gmin=top)
FIELDS = ['union', 'no_ties', 'gmax', 'gmin']
def row(S, a):
    R = rules(S); out = []
    for k in FIELDS:
        A = R[k]; out += [sum(a[i] for i in A), len(A), sum(a[i] for i in range(len(a)) if i not in A), len(a) - len(A)]
    tie = R['union'] - R['no_ties']; rj = set(range(len(a))) - R['union']
    return out + [sum(a[i] for i in tie), len(tie), sum(a[i] for i in rj), len(rj)]
IDX = {k: 4 * i for i, k in enumerate(FIELDS)}
def lr(t, k):
    c = t[IDX[k]:IDX[k] + 4]
    return math.log((c[0] / c[1]) / (c[2] / c[3])) if c[1] and c[2] and c[3] else float('nan')
def stat(t):
    tg = t[16] / t[17] - t[18] / t[19] if t[17] and t[19] else float('nan')
    return dict(H1_tie_gap=tg, H2_logratio_no_ties_minus_union=lr(t, 'no_ties') - lr(t, 'union'), S1_logratio_union=lr(t, 'union'), S2_logratio_no_ties_minus_gmax=lr(t, 'no_ties') - lr(t, 'gmax'))
def analyze(by_q, B=10000, seed=42):
    x = np.array(list(by_q.values()), float); obs = stat(x.sum(axis=0)); rng = np.random.default_rng(seed); bs = collections.defaultdict(list)
    for _ in range(B):
        for k, v in stat(x[rng.integers(0, len(x), len(x))].sum(axis=0)).items(): bs[k].append(v)
    res = {}
    for k in obs:
        v = np.array(bs[k]); v = v[~np.isnan(v)]; se = float(v.std()) if len(v) else 0.0
        if math.isnan(obs[k]) or se == 0.0:  # a small stratum can leave a statistic undefined; report it as such
            res[k] = dict(estimate=None if math.isnan(obs[k]) else float(obs[k]), se=None, ci=None, p_one_sided=None); continue
        res[k] = dict(estimate=float(obs[k]), se=se, ci=[float(np.quantile(v, .025)), float(np.quantile(v, .975))], p_one_sided=float(stats.norm.sf(abs(obs[k]) / se)) if (obs[k] < 0) == (k == 'H1_tie_gap') else 1 - float(stats.norm.sf(abs(obs[k]) / se)))
    # Holm over the two primary hypotheses
    ps = sorted([('H1_tie_gap', res['H1_tie_gap']['p_one_sided']), ('H2_logratio_no_ties_minus_union', res['H2_logratio_no_ties_minus_union']['p_one_sided'])], key=lambda z: (z[1] is None, z[1] or 0))
    alive = True
    for i, (k, p) in enumerate(ps):
        alive = alive and p is not None and p <= 0.05 / (2 - i); res[k]['holm_reject'] = alive
    t = x.sum(axis=0)
    res['counts'] = {k: dict(rej_rate=t[IDX[k] + 3] / (t[IDX[k] + 1] + t[IDX[k] + 3]), kept_acc=t[IDX[k]] / t[IDX[k] + 1], rej_acc=t[IDX[k] + 2] / t[IDX[k] + 3] if t[IDX[k] + 3] else None, ratio=math.exp(lr(t, k))) for k in FIELDS}
    res['partition'] = dict(responses=t[IDX['union']+1]+t[IDX['union']+3], tie_readmitted=t[17], rejected_by_union=t[19], tie_share=t[17]/(t[IDX['union']+1]+t[IDX['union']+3]), acc_tie_readmitted=t[16]/t[17] if t[17] else None, acc_rejected_by_union=t[18]/t[19] if t[19] else None)
    u = t[IDX['union']:IDX['union'] + 4]; res['fisher_union_two_sided'] = float(stats.fisher_exact([[u[0], u[1] - u[0]], [u[2], u[3] - u[2]]])[1])  # ignores question-level dependence; for comparison with earlier registrations
    res['n_questions'] = len(x); c2 = res['S2_logratio_no_ties_minus_gmax']['ci']; res['S2_within_margin_0.10'] = c2 is not None and abs(c2[0]) <= .10 and abs(c2[1]) <= .10
    return res
SUBSETS = dict(first6=list(range(6)), one_per_temp_plus_one=[0, 3, 6, 9, 12, 1], all15=list(range(15)))
ARMS = dict(primary=('registered', word_bounded_rule), direct=('registered', registered_rule), repaired_cut=('repaired', word_bounded_rule), repaired_cut_registered_scorer=('repaired', registered_rule))
def load_main(d, order_file, n, arm='primary', subset='first6', models=MODELS, keep=None):
    cut, rule = ARMS[arm]; idx = SUBSETS[subset]; Q = json.load(open(order_file))['questions'][:n]; by = collections.defaultdict(lambda: np.zeros(20))
    for m in models:
        R = json.load(open(os.path.join(d, f'main_{m}.json')))['questions']
        missing = [q['id'] for q in Q if q['id'] not in R]
        assert not missing, f'{m}: {len(missing)} questions missing; resume the run or report a deviation'
        for q in Q:
            if keep and not keep(q): continue
            e = R[q['id']]; S = e['sim_' + cut]; S = [[S[i][j] for j in idx] for i in idx]
            by[q['id']] += np.array(row(S, [int(rule(e['responses'][i]['text_' + cut], q['gold_answers'])) for i in idx]), float)
    return by
def run_all(d, order_file, n, B=10000):
    out = dict(arms={}, retention={}, per_model={}, per_property={})
    for a in ARMS: out['arms'][a] = analyze(load_main(d, order_file, n, arm=a), B=B)
    for s in ['one_per_temp_plus_one', 'all15']: out['retention'][s] = analyze(load_main(d, order_file, n, subset=s), B=B)
    for m in MODELS: out['per_model'][m] = analyze(load_main(d, order_file, n, models=[m]), B=2000)
    for p in sorted({q['prop'] for q in json.load(open(order_file))['questions'][:n]}):
        out['per_property'][p] = analyze(load_main(d, order_file, n, keep=lambda q: q['prop'] == p), B=2000)
    return out
def load_validate(ds):
    acc = {}
    if ds == 'triviaqa':
        for r in json.load(open(CF + 'kle_snne_triviaqa_results.json')):
            if r['dataset'] == 'triviaqa': acc[(r['model'], r['qid'])] = [int(x) for x in r['accuracies']]
    else:
        for fn in sorted(glob.glob(CF + f'modal_data/prereg_{ds}_*.json')):
            m = next((m for m in MODELS if f'_{m}_' in fn or fn.endswith(f'_{m}.json')), None)
            if m:
                for q, e in json.load(open(fn))['per_question'].items(): acc[(m, q)] = [int(bool(x)) for x in e['accuracies']]
    by = collections.defaultdict(lambda: np.zeros(20))
    for m in MODELS:
        for q, S in json.load(open(CF + f'sim_cache/sim_matrices_{ds}_{m}.json')).items():
            a = acc.get((m, q))
            if a is not None and len(a) == len(S): by[q] += np.array(row(S, a), float)
    return by
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--validate'); ap.add_argument('--data'); a = ap.parse_args()
    if a.data:
        assert os.path.exists('REGISTRATION_LOCK.txt'), 'The confirmatory analysis needs a timestamped registration.'
        R = run_all(a.data, 'popqa_main_order.json', 3000); json.dump(R, open('main_results.json', 'w'), indent=1); p = R['arms']['primary']
        print(json.dumps({k: p[k] for k in ['H1_tie_gap', 'H2_logratio_no_ties_minus_union', 'S1_logratio_union', 'S2_logratio_no_ties_minus_gmax', 'S2_within_margin_0.10', 'partition', 'n_questions']}, indent=1)); raise SystemExit
    r = analyze(load_validate(a.validate), B=2000)
    print(a.validate, 'questions', r['n_questions'])
    for k in ['H1_tie_gap', 'H2_logratio_no_ties_minus_union', 'S1_logratio_union', 'S2_logratio_no_ties_minus_gmax']:
        print(' ', k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in r[k].items() if x != 'ci'}, 'ci', [round(c, 3) for c in r[k]['ci']])
    print('  ratios', {k: (round(v['ratio'], 3), round(100 * v['rej_rate'], 2)) for k, v in r['counts'].items()}, 'within margin', r['S2_within_margin_0.10'])
