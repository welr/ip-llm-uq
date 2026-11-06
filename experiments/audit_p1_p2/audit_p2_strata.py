"""Prediction 2: is the Llama signal a difference between responses, or between question types? Reads only.
The manuscript calls this a within-model comparison on the same questions. Each statement sits in one group, so the check is whether the groups differ in subject and truth, and whether discrimination differs inside subjects."""
import collections, json
import numpy as np
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
def auc(c, y):
    c = np.asarray(c, float); y = np.asarray(y, int); p, n = c[y == 1], c[y == 0]
    if not len(p) or not len(n): return float('nan')
    return float(np.mean((p[:, None] > n[None, :]) + .5 * (p[:, None] == n[None, :])))
OUT = {}
for name, f in dict(Llama='pred2_local_llama70b_20260407_161230.json', Gemma='pred2_gemma_20260111_213527.json').items():
    d = json.load(open(R + f)); rs = list(d['results'].values()); V = [r for r in rs if r['is_coherent'] is not None]
    # Do paired true/false versions of one question both appear? The statement prefix before 'The answer is:' identifies the question.
    Q = collections.defaultdict(list)
    for r in V: Q[r['statement'].split('. The answer is:')[0]].append(r)
    both = {k: v for k, v in Q.items() if len({x['is_true'] for x in v}) == 2}
    split = sum(len({x['is_coherent'] for x in v}) == 2 for v in both.values())
    s = {}
    for sub in sorted({r['subset'] for r in V}):
        C = [r for r in V if r['subset'] == sub and r['is_coherent']]; I = [r for r in V if r['subset'] == sub and not r['is_coherent']]
        s[sub] = dict(n_coh=len(C), n_inc=len(I), auc_coh=auc([r['c_true'] for r in C], [int(r['is_true']) for r in C]), auc_inc=auc([r['c_true'] for r in I], [int(r['is_true']) for r in I]))
    ok = {k: v for k, v in s.items() if v['n_coh'] >= 20 and v['n_inc'] >= 20 and not np.isnan(v['auc_coh']) and not np.isnan(v['auc_inc'])}
    w = np.array([min(v['n_coh'], v['n_inc']) for v in ok.values()], float); dif = np.array([v['auc_coh'] - v['auc_inc'] for v in ok.values()]) if ok else np.array([])
    OUT[name] = dict(questions_with_both_versions=len(both), of_which_split_across_groups=split, within_subject=s, subjects_compared=len(ok), subjects_coh_better=int((dif > 0).sum()), weighted_auc_gap=float((w * dif).sum() / w.sum()) if ok else None)
    print(name, {k: v for k, v in OUT[name].items() if k != 'within_subject'}); print('  ', {k: (v['n_coh'], v['n_inc'], round(v['auc_coh'], 2), round(v['auc_inc'], 2)) for k, v in ok.items()})
json.dump(OUT, open('audit_p2_strata.json', 'w'), indent=1)
