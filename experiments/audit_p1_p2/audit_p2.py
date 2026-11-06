"""Audit of Prediction 2 from stored responses. Reads only; generates nothing."""
import collections, json, re, sys
import numpy as np
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
FILES = dict(Llama='pred2_local_llama70b_20260407_161230.json', Gemma='pred2_gemma_20260111_213527.json', Mistral='pred2_mistral_gpu_20260111_223736.json')
def ece(c, y, n=10):
    c = np.asarray(c, float); y = np.asarray(y, float); e = 0.0
    if not len(c): return float('nan')
    for i in range(n):
        lo, hi = i / n, (i + 1) / n; m = (c >= lo) & (c < hi) if hi < 1 else (c >= lo) & (c <= hi)
        if m.sum(): e += m.mean() * abs(y[m].mean() - c[m].mean())
    return float(e)
def auc(c, y):
    c = np.asarray(c, float); y = np.asarray(y, int); p = c[y == 1]; n = c[y == 0]
    if not len(p) or not len(n): return float('nan')
    return float(np.mean((p[:, None] > n[None, :]) + .5 * (p[:, None] == n[None, :])))
OUT = {}
for m, f in FILES.items():
    d = json.load(open(R + f)); print('=====', m, f, list(d)[:8]); print(' config', {k: (v if not isinstance(v, list) else len(v)) for k, v in d.get('config', {}).items()})
    rs = d.get('results') or d.get('per_question_results'); rs = list(rs.values()) if isinstance(rs, dict) else rs
    print(' n', len(rs), 'fields', list(rs[0])); print(' sample', {k: (str(v)[:110]) for k, v in rs[0].items()})
    V = [r for r in rs if r.get('is_coherent') is not None]; C = [r for r in V if r['is_coherent']]; I = [r for r in V if not r['is_coherent']]
    print(' valid', len(V), 'coh', len(C), 'inc', len(I), 'fail rate', round(1 - len(V) / len(rs), 3), 'share true all', round(np.mean([r['is_true'] for r in rs]), 3), 'valid', round(np.mean([r['is_true'] for r in V]), 3))
    OUT[m] = dict(n=len(rs), valid=len(V), coh=len(C), inc=len(I))
    print(' ECE coh/inc', round(ece([r['c_true'] for r in C], [r['is_true'] for r in C]), 4), round(ece([r['c_true'] for r in I], [r['is_true'] for r in I]), 4))
json.dump(OUT, open('audit_p2_counts.json', 'w'), indent=1)
