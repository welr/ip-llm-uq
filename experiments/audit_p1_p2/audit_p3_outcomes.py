"""Recompute registered Goldilocks bins from stored per-question data, and the cached-matrix baseline for TruthfulQA, against which the identical-string test must be read. Writes audit_p3_outcomes.json."""
import collections, glob, json, os, re
import numpy as np
C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', ''); MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']; TH = [round(.5 + .05 * i, 2) for i in range(10)]
def load(ds, m): c = [f for f in glob.glob(C + f'modal_data/prereg_{ds}_*.json') if f'_{m}_' in f or f.endswith(f'_{m}.json')]; return json.load(open(sorted(c)[-1]))['per_question']
def part(S):
    n = len(S); A = set()
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
        M = max(map(len, cl.values()))
        for v in cl.values():
            if len(v) == M: A.update(v)
    return A
BIN = lambda k: '0/6' if k == 0 else '1-2/6' if k <= 2 else '3-4/6' if k <= 4 else '5/6' if k == 5 else '6/6'
OUT = {}; rng = np.random.default_rng(42)
for ds in ['nq_open', 'truthfulqa']:
    B = collections.defaultdict(collections.Counter)
    for m in MODELS:
        for r in load(ds, m).values():
            A = r['accuracies']; adm = set(r['nli_admissible']); c = B[BIN(sum(map(bool, A)))]
            for i in range(6): g = 'a' if i in adm else 'r'; c[g + 't'] += 1; c[g + 'c'] += bool(A[i])
    o = {}
    for b in ['1-2/6', '3-4/6', '5/6']:
        c = B[b]; x = rng.binomial(c['at'], c['ac'] / c['at'], 10000) / c['at']; y = rng.binomial(c['rt'], c['rc'] / c['rt'], 10000) / c['rt']; y = np.where(y > 0, y, np.nan); r = x / y
        o[b] = dict(ratio=(c['ac'] / c['at']) / (c['rc'] / c['rt']), parametric_ci=[float(np.nanquantile(r, .025)), float(np.nanquantile(r, .975))], rejected=c['rt'])
    OUT[ds] = o; print(ds, {k: {a: (round(b, 3) if isinstance(b, float) else [round(z, 2) for z in b] if isinstance(b, list) else b) for a, b in v.items()} for k, v in o.items()})
T = collections.Counter(); F = collections.Counter(); per = {}
norm = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (s or '').lower().strip()))
for m in MODELS:
    SM = json.load(open(C + f'sim_cache/sim_matrices_truthfulqa_{m}.json')); pq = load('truthfulqa', m)
    for k, S in SM.items():
        if k not in pq: continue
        A, X = pq[k]['accuracies'], pq[k]['top6_texts']; ad = part(S); S2 = [list(r) for r in S]
        for a in range(6):
            for b in range(a + 1, 6):
                if norm(X[a]) and norm(X[a]) == norm(X[b]): S2[a][b] = S2[b][a] = 1.0
        ad2 = part(S2)
        for i in range(6):
            g = 'a' if i in ad else 'r'; T[g + 't'] += 1; T[g + 'c'] += bool(A[i]); g = 'a' if i in ad2 else 'r'; F[g + 't'] += 1; F[g + 'c'] += bool(A[i])
rr = lambda c: (c['ac'] / c['at']) / (c['rc'] / c['rt'])
OUT['truthfulqa_cached'] = dict(n_pairs_in_cache=sum(len(json.load(open(C + f'sim_cache/sim_matrices_truthfulqa_{m}.json'))) for m in MODELS), baseline_ratio=rr(T), baseline_rejected=T['rt'], forced_ratio=rr(F), forced_rejected=F['rt']); print(OUT['truthfulqa_cached'])
json.dump(OUT, open('audit_p3_outcomes.json', 'w'), indent=1)
