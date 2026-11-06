"""MMLU/ARC under the tie rule in the OSF text (ady3k/preregistration.md, section 3.5): largest cluster, ties broken by lowest cluster index.

The run marks every tied cluster. Post hoc; reads stored labels and similarity matrices; writes audit_registered_tie.json.
"Lowest cluster index" is not defined further in the registration. Here clusters are ordered by smallest member, so the first response in generation order wins a tie.
The registered string arm (H3) is not recomputed: three MMLU models stored no texts.
"""
import collections, glob, json, os
import numpy as np
from scipy import stats
C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', '')
TH = [round(.5 + .05 * i, 2) for i in range(10)]; MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']
def load(ds, m):
    c = [f for f in glob.glob(C + f'modal_data/prereg_{ds}_*.json') if f'_{m}_' in f or f.endswith(f'_{m}.json')]
    return json.load(open(sorted(c)[-1]))['per_question']
def partition(S, ties):
    adm = set(); n = 6
    for t in TH:
        p = list(range(n))
        def f(x):
            while p[x] != x: p[x] = p[p[x]]; x = p[x]
            return x
        for a in range(n):
            for b in range(a + 1, n):
                if S[a][b] >= t: p[f(a)] = f(b)
        cl = collections.defaultdict(list)
        for a in range(n): cl[f(a)].append(a)
        M = max(len(v) for v in cl.values()); top = [v for v in cl.values() if len(v) == M]
        if ties:
            for v in top: adm.update(v)
        else: adm.update(min(top, key=min))
    return adm
def boot_groups(ac, at, rc, rt, B=10000, seed=42):
    """Appendix E's interval: resample independently within the admissible and rejected groups."""
    rng = np.random.default_rng(seed); a = rng.binomial(at, ac / at, B) / at; r = rng.binomial(rt, rc / rt, B) / rt; k = r > 0
    return [float(x) for x in np.percentile(a[k] / r[k], [2.5, 97.5])]
def boot_pairs(P, B=10000, seed=42):
    rng = np.random.default_rng(seed); X = np.array(P, float); s = X[rng.integers(0, len(X), (B, len(X)))].sum(axis=1)
    k = (s[:, 3] > 0) & (s[:, 2] > 0); r = (s[k, 0] / s[k, 1]) / (s[k, 2] / s[k, 3]); return [float(x) for x in np.percentile(r, [2.5, 97.5])]
OUT = {}
for ds in ['mmlu', 'arc']:
    OUT[ds] = {}
    for rule, ties in [('ties_included_as_run', True), ('registered_lowest_index', False)]:
        T = collections.Counter(); per = {}; pairs = []; same = chk = missing = 0
        for m in MODELS:
            pq = load(ds, m); sp = C + f'sim_cache/sim_matrices_{ds}_{m}.json'; SM = json.load(open(sp)) if os.path.exists(sp) else {}
            t = collections.Counter()
            for k, r in pq.items():
                if k not in SM: missing += 1; continue
                adm = partition(SM[k], ties); A = r['accuracies']; c = collections.Counter()
                if ties: chk += 1; same += adm == set(r['nli_admissible'])
                for i in range(6): g = 'a' if i in adm else 'r'; c[g + 't'] += 1; c[g + 'c'] += bool(A[i])
                t.update(c); pairs.append((c['ac'], c['at'], c['rc'], c['rt']))
            T.update(t)
            per[m] = dict(ratio=(t['ac'] / t['at']) / (t['rc'] / t['rt']) if t['rt'] and t['rc'] else None, rejected=t['rt'], adm_acc=t['ac'] / t['at'] if t['at'] else None, rej_acc=t['rc'] / t['rt'] if t['rt'] else None, p=float(stats.fisher_exact([[t['ac'], t['at'] - t['ac']], [t['rc'], t['rt'] - t['rc']]])[1]) if t['rt'] else None)
        n = T['at'] + T['rt']; ra = (T['ac'] / T['at']) / (T['rc'] / T['rt'])
        OUT[ds][rule] = dict(total=n, rejected=T['rt'], rejection_rate=T['rt'] / n, adm_acc=T['ac'] / T['at'], rej_acc=T['rc'] / T['rt'], ratio=ra, ci_groups=boot_groups(T['ac'], T['at'], T['rc'], T['rt']), ci_pairs=boot_pairs(pairs), fisher_p=float(stats.fisher_exact([[T['ac'], T['at'] - T['ac']], [T['rc'], T['rt'] - T['rc']]])[1]), models_above_1=sum(1 for v in per.values() if v['ratio'] and v['ratio'] > 1), per_model=per, stored_partition_reproduced=[same, chk] if ties else None, questions_without_matrix=missing)
json.dump(OUT, open('audit_registered_tie.json', 'w'), indent=1)
for ds, o in OUT.items():
    for rule, v in o.items():
        print(f"{ds:5s} {rule:26s} n={v['total']} rej={v['rejected']} ({100*v['rejection_rate']:.1f}%) adm {v['adm_acc']:.3f} rej {v['rej_acc']:.3f} ratio {v['ratio']:.3f} groups {[round(x,2) for x in v['ci_groups']]} pairs {[round(x,2) for x in v['ci_pairs']]} p {v['fisher_p']:.2e} above1 {v['models_above_1']}/4 repro {v['stored_partition_reproduced']} missing {v['questions_without_matrix']}")
        print('      ', {m: (round(x['ratio'], 2) if x['ratio'] else None, x['rejected']) for m, x in v['per_model'].items()})
