"""TruthfulQA strata the registration (vze74, section 4.3) says will be reported, absent from the manuscript. Exploratory by that section.

Stored labels and partitions as run; registered substring scorer; interval resamples question-model pairs. Writes audit_registered_strata.json.
"""
import collections, glob, json, os
import numpy as np
C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_data', '')
MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']
def boot(P, B=10000, seed=42):
    rng = np.random.default_rng(seed); X = np.array(P, float); s = X[rng.integers(0, len(X), (B, len(X)))].sum(axis=1)
    k = (s[:, 3] > 0) & (s[:, 2] > 0); r = (s[k, 0] / s[k, 1]) / (s[k, 2] / s[k, 3]); return [float(x) for x in np.percentile(r, [2.5, 97.5])]
G = collections.defaultdict(list)
for m in MODELS:
    f = sorted(x for x in glob.glob(C + f'prereg_truthfulqa_{m}_*') if f'_{m}_' in x)[-1]
    for k, r in json.load(open(f))['per_question'].items():
        A = r['accuracies']; adm = set(r['nli_admissible']); c = [sum(bool(A[i]) for i in adm), len(adm), sum(bool(A[i]) for i in range(6) if i not in adm), 6 - len(adm)]
        G['all', 'all'].append(c); G['type', r['tqa_type']].append(c); G['category', r['tqa_category']].append(c)
OUT = {}
for (kind, name), P in G.items():
    s = np.array(P).sum(axis=0); ac, at, rc, rt = map(int, s)
    OUT[f'{kind}:{name}'] = dict(pairs=len(P), adm_acc=ac / at, rej_acc=rc / rt if rt else None, rejected=rt, rejection_rate=rt / (at + rt), ratio=(ac / at) / (rc / rt) if rt and rc else None, ci_pairs=boot(P) if rt and rc else None)
json.dump(OUT, open('audit_registered_strata.json', 'w'), indent=1)
for k in ['all:all'] + sorted(x for x in OUT if x.startswith('type')):
    v = OUT[k]; print(f"{k:28s} pairs {v['pairs']} rejected {v['rejected']} ({100*v['rejection_rate']:.1f}%) adm {v['adm_acc']:.3f} rej {v['rej_acc']:.3f} ratio {v['ratio']:.2f} {[round(x,2) for x in v['ci_pairs']]}")
c = [v for k, v in OUT.items() if k.startswith('category')]; d = [v for v in c if v['ratio'] is not None and v['rejected'] >= 30]
print('categories', len(c), 'with at least 30 rejected', len(d), 'ratio range', round(min(v['ratio'] for v in d), 2), round(max(v['ratio'] for v in d), 2), 'below 1:', sum(v['ratio'] < 1 for v in d))
