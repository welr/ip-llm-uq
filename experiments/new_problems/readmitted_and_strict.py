"""Exploratory. (a) Who does the union admit beyond plurality at theta_min, and why?
(b) MMLU/ARC admissibility ratio under a strict letter rule, reconstructing the retained six."""
import json, os, glob, re, collections, sys
import numpy as np
sys.path.insert(0, '.')
CF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', '')
MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']; GRID = [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95]
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
OUT = {}
for ds in ['triviaqa', 'nq_open', 'truthfulqa', 'arc', 'mmlu']:
    acc = load(ds); added = [0, 0]; via_tie = [0, 0]; via_real = [0, 0]; rej = [0, 0]
    notie = [0, 0, 0, 0]; pairs = []
    for m in MODELS:
        p = CF + f'sim_cache/sim_matrices_{ds}_{m}.json'
        if not os.path.exists(p): continue
        for q, S in json.load(open(p)).items():
            a = acc.get((m, q))
            if a is None or len(a) != len(S): continue
            C = [comps(S, t) for t in GRID]
            base = set(); [base.update(x) for x in C[0] if len(x) == max(map(len, C[0]))]
            full = set(); nosing = set(base)
            for c in C:
                mx = max(map(len, c))
                for x in c:
                    if len(x) == mx: full.update(x)
                    if len(x) == mx and mx >= 2: nosing.update(x)
            for i in full - base:
                added[0] += a[i]; added[1] += 1
                t = via_real if i in nosing else via_tie
                t[0] += a[i]; t[1] += 1
            pr = [0, 0, 0, 0]
            for i in range(len(a)):
                if i not in full: rej[0] += a[i]; rej[1] += 1
                k = 0 if i in nosing else 2
                notie[k] += a[i]; notie[k + 1] += 1; pr[k] += a[i]; pr[k + 1] += 1
            pairs.append(pr)
    r = lambda x: (x[0] / x[1]) if x[1] else None
    ratio = (notie[0] / notie[1]) / (notie[2] / notie[3]) if notie[3] and notie[2] else None
    x = np.array(pairs, float); rng = np.random.default_rng(42); rs = []
    for _ in range(2000):
        b = x[rng.integers(0, len(x), len(x))].sum(axis=0)
        if b[2] > 0 and b[3] > 0: rs.append((b[0] / b[1]) / (b[2] / b[3]))
    ci_nt = [float(np.quantile(rs, .025)), float(np.quantile(rs, .975))]
    OUT[ds] = dict(ci_excl_ties=ci_nt, added=added, added_acc=r(added), via_singleton_tie=via_tie, tie_acc=r(via_tie), via_real_cluster=via_real, real_acc=r(via_real), rejected=rej, rej_acc=r(rej), excl_singleton_ties=dict(rej_rate=notie[3] / (notie[1] + notie[3]), kept=notie[0] / notie[1], rej=notie[2] / notie[3] if notie[3] else None, ratio=ratio))
    f = lambda x: 'n/a' if x is None else f'{x:.3f}'
    print(f"{ds:10s} admitted beyond theta_min plurality {added[1]:5d} acc {f(r(added))} | via singleton tie {via_tie[1]:5d} acc {f(r(via_tie))} | via cluster>=2 {via_real[1]:5d} acc {f(r(via_real))} | rejected {rej[1]:5d} acc {f(r(rej))} | union without singleton ties: rej {100*OUT[ds]['excl_singleton_ties']['rej_rate']:.2f}% ratio {f(ratio)}")
# strict scorer, MMLU/ARC
def norm(s, cap=100): return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (s or '').lower().strip()))[:cap]
def reg(r, G):
    # Registered scorer as run on MMLU/ARC: 100-character cap, substring test both ways, gold letter included.
    x = norm(r); return bool(x) and any(norm(g) and (norm(g) in x or x in norm(g)) for g in G)
def strict(r, G):
    # Post hoc. Letter: case-sensitive, alone, as a leading label, or after "answer/option/choice". Text: word-bounded containment of the gold text.
    raw = (r or '').strip(); t, L = norm(G[0], 10**6), G[1]
    if re.fullmatch(r'\(?%s\)?[.:)]?' % L, raw) or re.match(r'\(?%s\)?[.:)]\s' % L, raw): return True
    if re.search(r'(?i:answer|option|choice)\s*(?i:is)?\s*:?\s*\(?%s\)?(?![A-Za-z0-9])' % L, raw): return True
    x = norm(r, 10**6)
    return bool(x) and bool(t) and (' %s ' % t) in (' %s ' % x)
S = {}
for ds, ms in [('arc', MODELS), ('mmlu', ['Gemma'])]:
    tot_r = [0, 0, 0, 0]; tot_s = [0, 0, 0, 0]; match = chk = 0; prs = []
    for m in ms:
        g = json.load(open(CF + f'modal_data/checkpoint_gen_{ds}_{m}.json')); pq = json.load(open(sorted(glob.glob(CF + f'modal_data/prereg_{ds}_{m}*.json'))[-1]))['per_question']
        rm = rs = None; c_r = [0, 0, 0, 0]; c_s = [0, 0, 0, 0]
        for q, it in g.items():
            e = pq.get(q)
            if not e: continue
            six = [x['text'] for x in it['responses'][:6]]
            A = set(e['nli_admissible']); pp = [0, 0, 0, 0]
            for i, t in enumerate(six):
                R_, S_ = reg(t, it['gold_answers']), strict(t, it['gold_answers']); chk += 1; match += R_ == bool(e['accuracies'][i])
                k = 0 if i in A else 2
                c_r[k] += R_; c_r[k + 1] += 1; c_s[k] += S_; c_s[k + 1] += 1; pp[k] += S_; pp[k + 1] += 1
            prs.append(pp)
        for i in range(4): tot_r[i] += c_r[i]; tot_s[i] += c_s[i]
        fr = lambda c: (c[0] / c[1]) / (c[2] / c[3]) if c[3] and c[2] else None
        print(f'  {ds} {m}: registered ratio {fr(c_r)}  strict ratio {fr(c_s)}  kept acc {c_r[0]/c_r[1]:.3f} -> {c_s[0]/c_s[1]:.3f}')
        S[f'{ds}_{m}'] = dict(registered=fr(c_r), strict=fr(c_s))
    fr = lambda c: (c[0] / c[1]) / (c[2] / c[3]) if c[3] and c[2] else None
    print(f'{ds} POOLED reproduces stored labels {match}/{chk}; registered {fr(tot_r):.3f}; strict {fr(tot_s):.3f}; strict adm {tot_s[0]/tot_s[1]:.3f} rej {tot_s[2]/tot_s[3]:.3f}')
    x = np.array(prs, float); rng = np.random.default_rng(42); rs = []
    for _ in range(2000):
        b = x[rng.integers(0, len(x), len(x))].sum(axis=0)
        if b[2] > 0 and b[3] > 0: rs.append((b[0] / b[1]) / (b[2] / b[3]))
    S[ds] = dict(match=match, chk=chk, registered=fr(tot_r), strict=fr(tot_s), strict_ci=[float(np.quantile(rs, .025)), float(np.quantile(rs, .975))], strict_adm=tot_s[0] / tot_s[1], strict_rej=tot_s[2] / tot_s[3])
json.dump(dict(readmitted=OUT, strict=S), open('readmitted_and_strict.json', 'w'), indent=1)
