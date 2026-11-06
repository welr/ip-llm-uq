"""Compare the reported 10-point grid with the exact continuum partition on [0.50, 0.95]."""
import json, os, glob, collections
CF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', '')
MODELS = ['Qwen','Mistral','Llama','Gemma']
GRID = [.50,.55,.60,.65,.70,.75,.80,.85,.90,.95]
def comps(S, t):
    n = len(S); p = list(range(n))
    def f(a):
        while p[a] != a: p[a] = p[p[a]]; a = p[a]
        return a
    for i in range(n):
        for j in range(i+1, n):
            if S[i][j] >= t: p[f(i)] = f(j)
    g = collections.defaultdict(list)
    for i in range(n): g[f(i)].append(i)
    return list(g.values())
def adm(S, Th):
    A = set()
    for t in Th:
        c = comps(S, t); m = max(len(x) for x in c)
        for x in c:
            if len(x) == m: A |= set(x)
    return A
def thresholds(S, lo=.5, hi=.95):
    n = len(S); v = {lo, hi}
    for i in range(n):
        for j in range(n):
            if i != j and lo <= S[i][j] <= hi: v.add(S[i][j])
    return sorted(v)
def load(ds):
    acc = {}
    if ds == 'triviaqa':
        for r in json.load(open('../triviaqa_rescore/results_full/kle_snne_triviaqa_results.json')):
            acc[(r['model'], r['qid'])] = [int(x) for x in r['accuracies']]
    else:
        for fn in sorted(glob.glob(CF+f'modal_data/prereg_{ds}_*.json')):
            m = next((m for m in MODELS if f'_{m}_' in fn or fn.endswith(f'_{m}.json')), None)
            if not m: continue
            for q, e in json.load(open(fn))['per_question'].items():
                acc[(m, q)] = [int(bool(x)) for x in e['accuracies']]
    return acc
RES = {}
for ds in ['triviaqa','nq_open','truthfulqa']:
    acc = load(ds); T = {'grid':[0,0,0,0], 'cont':[0,0,0,0]}; asymm = 0; changed = 0; nq = 0
    for m in MODELS:
        S_all = json.load(open(CF+f'sim_cache/sim_matrices_{ds}_{m}.json'))
        for q, S in S_all.items():
            if (m, q) not in acc: continue
            a = acc[(m, q)]
            if len(a) != len(S): continue
            nq += 1
            g, c = adm(S, GRID), adm(S, thresholds(S))
            assert g <= c
            changed += g != c
            for name, A in [('grid', g), ('cont', c)]:
                for i, x in enumerate(a):
                    k = 0 if i in A else 2
                    T[name][k] += x; T[name][k+1] += 1
    print(ds, 'n_question_model', nq, 'with different partition', changed)
    RES[ds] = dict(n=nq, changed=changed, **{n: dict(ratio=(v[0]/v[1])/(v[2]/v[3]), rej=v[3], rej_rate=v[3]/(v[1]+v[3])) for n, v in T.items()})
    for n, (ac, at, rc, rt) in T.items():
        print(f"  {n}: adm {ac}/{at}={ac/at:.4f} rej {rc}/{rt}={rc/rt:.4f} ratio={(ac/at)/(rc/rt):.4f} rej_rate={rt/(at+rt):.4f}")
json.dump(RES, open('continuum_gap.json', 'w'), indent=1)
