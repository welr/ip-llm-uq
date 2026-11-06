"""Prediction 2 under a neutral prompt and a strict parser, against the original run on the same statements. Exploratory; writes audit_p2_neutral.json.

Coherence follows the original run: a pair is incoherent when the two confidences sum to more than 1.05. Intervals are bootstrap over statements.
The neutral run covers the first 400 stored statements, a mixed sample of subjects, so the comparison with the original run is like with like."""
import collections, json
import numpy as np
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
O = json.load(open(R + 'pred2_local_llama70b_20260407_161230.json'))['results']; N = json.load(open('neutral_prompt_llama70b.json'))
def ece(c, y, n=10):
    c = np.asarray(c, float); y = np.asarray(y, float); e = 0.0
    if not len(c): return float('nan')
    for i in range(n):
        lo, hi = i / n, (i + 1) / n; m = (c >= lo) & (c < hi) if hi < 1 else (c >= lo) & (c <= hi)
        if m.sum(): e += m.mean() * abs(y[m].mean() - c[m].mean())
    return float(e)
def auc(c, y):
    c = np.asarray(c, float); y = np.asarray(y, int); p, n = c[y == 1], c[y == 0]
    if not len(p) or not len(n): return float('nan')
    return float(np.mean((p[:, None] > n[None, :]) + .5 * (p[:, None] == n[None, :])))
def summarize(rows, label):
    v = [r for r in rows if r[0] is not None and r[1] is not None]; a = np.array([(x, y, int(t)) for x, y, t in v], float); s = a[:, 0] + a[:, 1]; inc = s > 1.05; coh = ~inc
    def g(m): return dict(n=int(m.sum()), ece=ece(a[m, 0], a[m, 2]), auroc=auc(a[m, 0], a[m, 2]), mean_c=float(a[m, 0].mean()) if m.sum() else None, share_true=float(a[m, 2].mean()) if m.sum() else None)
    both1 = (a[:, 0] == 1) & (a[:, 1] == 1); o = dict(label=label, statements=len(rows), valid=len(v), extraction=len(v) / len(rows), incoherent_share=float(inc.mean()), both_one=int(both1.sum()), mean_sum=float(s.mean()), median_sum=float(np.median(s)), all=g(np.ones(len(a), bool)), coherent=g(coh), incoherent=g(inc), incoherent_without_both_one=g(inc & ~both1))
    rng = np.random.default_rng(42); B = 5000; rs = []; rs2 = []; da = []
    for _ in range(B):
        i = rng.integers(0, len(a), len(a)); x = a[i]; ii = x[:, 0] + x[:, 1] > 1.05; bb = (x[:, 0] == 1) & (x[:, 1] == 1); ec = ece(x[~ii, 0], x[~ii, 2]) if (~ii).sum() else np.nan
        if ec and ec > 0: rs.append(ece(x[ii, 0], x[ii, 2]) / ec); rs2.append(ece(x[ii & ~bb, 0], x[ii & ~bb, 2]) / ec if (ii & ~bb).sum() else np.nan)
        da.append(auc(x[~ii, 0], x[~ii, 2]) - auc(x[ii, 0], x[ii, 2]))
    o['ece_ratio'] = o['incoherent']['ece'] / o['coherent']['ece'] if o['coherent']['ece'] else None; o['ece_ratio_ci'] = [float(np.nanquantile(rs, q)) for q in (.025, .975)] if rs else None
    o['ece_ratio_without_both_one'] = o['incoherent_without_both_one']['ece'] / o['coherent']['ece'] if o['coherent']['ece'] else None; o['ece_ratio_without_both_one_ci'] = [float(np.nanquantile(rs2, q)) for q in (.025, .975)] if rs2 else None
    o['auroc_coherent_minus_incoherent_ci'] = [float(np.nanquantile(da, q)) for q in (.025, .975)]
    return o
K = list(N); sub = collections.Counter(N[k]['subset'] for k in K)
orig = [(O[k].get('c_true'), O[k].get('c_false'), O[k]['is_true']) if O[k].get('c_true') is not None and O[k].get('c_false') is not None else (None, None, O[k]['is_true']) for k in K]
neut = [(N[k]['c_true'], N[k]['c_false'], N[k]['is_true']) for k in K]
both = [k for k in K if N[k]['c_true'] is not None and N[k]['c_false'] is not None and O[k].get('c_true') is not None and O[k].get('c_false') is not None]
OUT = dict(subsets=dict(sub), original_same_statements=summarize(orig, 'original'), neutral=summarize(neut, 'neutral'), original_where_neutral_parsed=summarize([(O[k]['c_true'], O[k]['c_false'], O[k]['is_true']) for k in both], 'original on shared'), neutral_where_both_parsed=summarize([(N[k]['c_true'], N[k]['c_false'], N[k]['is_true']) for k in both], 'neutral on shared'))
json.dump(OUT, open('audit_p2_neutral.json', 'w'), indent=1)
print('subjects', len(sub), 'largest', sub.most_common(3))
for nm in ['original_same_statements', 'neutral', 'original_where_neutral_parsed', 'neutral_where_both_parsed']:
    o = OUT[nm]; f = lambda x: 'nan' if x is None or x != x else f'{x:.3f}'
    print('=====', nm, 'valid', o['valid'], 'of', o['statements'], 'extraction', f(o['extraction']), 'incoherent', f(o['incoherent_share']), 'both 1.0:', o['both_one'], 'mean sum', f(o['mean_sum']), 'median', f(o['median_sum']))
    for g in ['all', 'coherent', 'incoherent', 'incoherent_without_both_one']: print('   ', g, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in o[g].items()})
    print('    ECE ratio', f(o['ece_ratio']), o['ece_ratio_ci'] and [round(x, 2) for x in o['ece_ratio_ci']], '| without both-1.0', f(o['ece_ratio_without_both_one']), o['ece_ratio_without_both_one_ci'] and [round(x, 2) for x in o['ece_ratio_without_both_one_ci']], '| AUROC coh-inc CI', [round(x, 3) for x in o['auroc_coherent_minus_incoherent_ci']])
ex = collections.Counter((N[k]['response_true'] or '').strip()[:18] for k in K); print('top raw true responses', ex.most_common(10)); bad = [N[k]['response_true'][:50] for k in K if N[k]['c_true'] is None][:6]; print('unparsed examples', bad)
