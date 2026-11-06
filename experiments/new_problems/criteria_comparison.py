"""Exploratory. Read-only over stored results.
Tests: (1) cluster score u_theta(r) is non-increasing in theta for every response;
(2) distinct partitions per question are at most M; (3) Gamma-maximax = plurality at theta_min, Gamma-maximin = plurality at theta_max;
(4) frozen-score choice C(A) = union of argmax over A satisfies Sen's alpha, and equals the K-rejection rule;
(5) partitions under each IP criterion on every benchmark, with question-cluster bootstrap.
Run with ~/ip_llm/bin/python."""
import json, os, glob, collections, itertools, random
import numpy as np
from scipy.optimize import linprog
CF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', '')
MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']
GRID = [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95]
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
def scores(S):
    n = len(S); U = np.zeros((len(GRID), n)); parts = set()
    for k, t in enumerate(GRID):
        c = comps(S, t); parts.add(frozenset(frozenset(x) for x in c))
        for x in c:
            for i in x: U[k, i] = len(x) / n
    return U, len(parts)
def sets(U):
    n = U.shape[1]; R = range(n); mx = U.max(axis=1)
    csem = {i for i in R if any(U[k, i] == mx[k] for k in range(len(GRID)))}
    maxi = {i for i in R if not any(all(U[k, j] > U[k, i] for k in range(len(GRID))) for j in R if j != i)}
    idom = {i for i in R if not any(U[:, j].min() > U[:, i].max() for j in R if j != i)}
    gmax = {i for i in R if U[:, i].max() == U.max()}
    lo = U.min(axis=0); gmin = {i for i in R if lo[i] == lo.max()}
    # E-admissible over the convex hull of the family: exists lambda making r optimal
    hull = set(csem)
    for i in maxi - csem:
        A = [[-(U[k, i] - U[k, j]) for k in range(len(GRID))] for j in R if j != i]
        r = linprog(np.zeros(len(GRID)), A_ub=A, b_ub=np.zeros(len(A)), A_eq=[np.ones(len(GRID))], b_eq=[1], bounds=(0, 1), method='highs')
        if r.status == 0: hull.add(i)
    return dict(C_sem=csem, hull_E=hull, Max=maxi, ID=idom, Gmaximax=gmax, Gmaximin=gmin)
def fixed(U, t):
    k = GRID.index(t); return {i for i in range(U.shape[1]) if U[k, i] == U[k].max()}
def load(ds):
    acc = {}
    if ds == 'triviaqa':
        for r in json.load(open('../triviaqa_rescore/results_full/kle_snne_triviaqa_results.json')):
            if r['dataset'] == 'triviaqa': acc[(r['model'], r['qid'])] = [int(x) for x in r['accuracies']]
    else:
        for fn in sorted(glob.glob(CF + f'modal_data/prereg_{ds}_*.json')):
            m = next((m for m in MODELS if f'_{m}_' in fn or fn.endswith(f'_{m}.json')), None)
            if m:
                for q, e in json.load(open(fn))['per_question'].items(): acc[(m, q)] = [int(bool(x)) for x in e['accuracies']]
    return acc
viol = 0; tot = 0; maxparts = 0; gmchk = [0, 0]; OUT = {}
rng = np.random.default_rng(42)
for ds in ['triviaqa', 'nq_open', 'truthfulqa', 'arc', 'mmlu']:
    acc = load(ds); rows = collections.defaultdict(list)
    for m in MODELS:
        p = CF + f'sim_cache/sim_matrices_{ds}_{m}.json'
        if not os.path.exists(p): continue
        for q, S in json.load(open(p)).items():
            a = acc.get((m, q))
            if a is None or len(a) != len(S): continue
            U, np_ = scores(np.array(S)); tot += 1; maxparts = max(maxparts, np_ - len(a))
            viol += int((np.diff(U, axis=0) > 1e-12).any())
            s = sets(U)
            gmchk[0] += s['Gmaximax'] != fixed(U, GRID[0]); gmchk[1] += s['Gmaximin'] != fixed(U, GRID[-1])
            s['theta_.75'] = fixed(U, .75)
            for name, A in s.items():
                ac = sum(a[i] for i in A); rc = sum(a) - ac
                rows[name].append((ac, len(A), rc, len(a) - len(A)))
    res = {}
    for name, v in rows.items():
        x = np.array(v, float); t = x.sum(axis=0)
        n = len(x); r = []
        for _ in range(2000):
            b = x[rng.integers(0, n, n)].sum(axis=0)
            if b[2] > 0 and b[3] > 0: r.append((b[0] / b[1]) / (b[2] / b[3]))
        res[name] = dict(rej_rate=t[3] / (t[1] + t[3]), kept_acc=t[0] / t[1], rej_acc=(t[2] / t[3] if t[3] else None), ratio=((t[0] / t[1]) / (t[2] / t[3]) if t[3] and t[2] else None), ci=[float(np.quantile(r, .025)), float(np.quantile(r, .975))] if r else None, n_rej=int(t[3]), n_q=n)
    OUT[ds] = res
    print('##', ds, 'question-model pairs', rows and len(rows['C_sem']))
    for name in ['Gmaximin', 'Gmaximax', 'theta_.75', 'C_sem', 'hull_E', 'Max', 'ID']:
        r = res[name]
        print(f"  {name:10s} rej {100*r['rej_rate']:6.2f}%  n_rej {r['n_rej']:5d}  kept {r['kept_acc']:.4f}  rej_acc {('%.4f'%r['rej_acc']) if r['rej_acc'] is not None else '  - '}  ratio {('%.2f'%r['ratio']) if r['ratio'] else ' - '} {[round(c,2) for c in r['ci']] if r['ci'] else ''}")
print('pairs', tot, '| u_theta increases with theta in', viol, '| partitions exceeding M', maxparts, '| Gmaximax != plurality(theta_min)', gmchk[0], '| Gmaximin != plurality(theta_max)', gmchk[1])
# frozen-score choice function: alpha and K-rejection, random score matrices
random.seed(1); bad_alpha = bad_K = 0; trials = 3000
for _ in range(trials):
    M, K = random.randint(2, 6), random.randint(1, 4); U = [[random.randint(0, 5) for _ in range(M)] for _ in range(K)]
    C = lambda A: {i for i in A if any(U[k][i] == max(U[k][j] for j in A) for k in range(K))}
    R = lambda A: {i for i in A if all(any(U[k][j] - U[k][i] > 0 for j in A) for k in range(K))}
    for z in range(1, 2 ** M):
        A = {i for i in range(M) if z >> i & 1}
        bad_K += C(A) != A - R(A)
        for y in range(1, 2 ** M):
            B = {i for i in range(M) if y >> i & 1}
            if B <= A: bad_alpha += not (C(A) & B <= C(B))
print('frozen scores: alpha violations', bad_alpha, '| C(A) != A minus K-rejection', bad_K, 'over', trials, 'random matrices')
json.dump(OUT, open('criteria_comparison.json', 'w'), indent=1, default=float)
