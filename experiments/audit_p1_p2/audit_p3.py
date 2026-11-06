"""Audit of Prediction 3's registered runs from stored data. Reads only; writes audit_p3.json.

The runner cuts each base response at the first newline, then the first period. A continuation that opens with a newline therefore becomes ''. The scorer marks '' wrong, and NLI similarity with '' is defined as 0, so '' never joins a cluster. Such a response is wrong and, unless every response ties as a singleton, rejected. Both labels follow from the cut, not from what the model said.
Three tests: how many responses are empty; the ratio with those responses out of the table; the partition recomputed on the remaining responses, where stored similarity matrices reproduce the registered partition."""
import collections, glob, json, os, re
import numpy as np
from scipy import stats
C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', '')
TH = [round(.5 + .05 * i, 2) for i in range(10)]; MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']; DS = ['nq_open', 'truthfulqa', 'mmlu', 'arc']
def load(ds, m):
    c = [f for f in glob.glob(C + f'modal_data/prereg_{ds}_*.json') if f'_{m}_' in f or f.endswith(f'_{m}.json')]; return json.load(open(sorted(c)[-1]))['per_question']
def fisher(ac, at, rc, rt): return float(stats.fisher_exact([[ac, at - ac], [rc, rt - rc]])[1]) if rt else None
def ratio(ac, at, rc, rt): return (ac / at) / (rc / rt) if at and rt and rc else None
def partition(S, keep):
    n = len(keep); adm = set()
    for t in TH:
        p = list(range(n))
        def f(x):
            while p[x] != x: p[x] = p[p[x]]; x = p[x]
            return x
        for a in range(n):
            for b in range(a + 1, n):
                if S[keep[a]][keep[b]] >= t: p[f(a)] = f(b)
        cl = collections.defaultdict(list)
        for a in range(n): cl[f(a)].append(a)
        M = max(len(v) for v in cl.values())
        for v in cl.values():
            if len(v) == M: adm.update(v)
    return {keep[a] for a in adm}
def cluster_boot(pairs, B=10000, seed=42):
    """Ratio over question-model pairs. pairs: array of (ac, at, rc, rt)."""
    rng = np.random.default_rng(seed); X = np.array(pairs, float); i = rng.integers(0, len(X), (B, len(X))); s = X[i].sum(axis=1); r = (s[:, 0] / s[:, 1]) / (s[:, 2] / s[:, 3]); return [float(np.nanquantile(r, .025)), float(np.nanquantile(r, .975))]
def rows(ds, m):
    """Yield (qid, labels, admissible set, texts or None, similarity or None)."""
    sp = C + f'sim_cache/sim_matrices_{ds}_{m}.json'; SM = json.load(open(sp)) if os.path.exists(sp) else {}
    if ds == 'triviaqa':
        for k, S in SM.items():
            if (m, k) in ACC: yield k, ACC[(m, k)], partition(S, list(range(6))), P1[m][k]['base_responses'][:6], S
        return
    pq = load(ds, m); gp = C + f'modal_data/checkpoint_gen_{ds}_{m}.json'; G = json.load(open(gp)) if os.path.exists(gp) else {}
    for k, r in pq.items():
        X = r.get('top6_texts') or ([x['text'] for x in G[k]['responses'][:6]] if k in G else None)
        yield k, r['accuracies'], set(r['nli_admissible']), X, SM.get(k)
R = '../stored_runs/modal_results/'; F = dict(Qwen='pred1_v2_20260109_053417.json', Mistral='pred1_mistral_20260110_071956.json', Llama='pred1_llama8b_20260112_074432.json', Gemma='pred1_gemma9b_20260112_074222.json')
P1 = {m: json.load(open(R + f))['per_question_results'] for m, f in F.items()}
ACC = {(r['model'], r['qid']): r['accuracies'] for r in json.load(open('../triviaqa_rescore/results_full/kle_snne_triviaqa_results.json')) if r['dataset'] == 'triviaqa'}
emp = lambda t: not re.sub(r'[^\w]', '', t or '')
OUT = {}
for ds in ['triviaqa'] + DS:
    T = collections.Counter(); per = {}; pairs_reg, pairs_ne, pairs2 = [], [], []; T2 = collections.Counter(); rep = dict(checked=0, same=0); notext = []
    for m in MODELS:
        t = collections.Counter(); seen_text = False
        for k, A, adm, X, S in rows(ds, m):
            c = collections.Counter()
            for i in range(6):
                g = 'a' if i in adm else 'r'; c[g + 't'] += 1; c[g + 'c'] += bool(A[i])
                if X is not None:
                    seen_text = True
                    if emp(X[i]): c['e' + g] += 1; c['ec'] += bool(A[i])
                    else: c['n' + g + 't'] += 1; c['n' + g + 'c'] += bool(A[i])
            t.update(c); pairs_reg.append((c['ac'], c['at'], c['rc'], c['rt']))
            if X is not None: pairs_ne.append((c['nac'], c['nat'], c['nrc'], c['nrt']))
            if X is not None and S is not None:
                rep['checked'] += 1; rep['same'] += partition(S, list(range(6))) == adm
                keep = [i for i in range(6) if not emp(X[i])]
                if keep:
                    a2 = partition(S, keep); d = collections.Counter()
                    for i in keep: g = 'a' if i in a2 else 'r'; d[g + 't'] += 1; d[g + 'c'] += bool(A[i])
                    T2.update(d); pairs2.append((d['ac'], d['at'], d['rc'], d['rt']))
        if not seen_text: notext.append(m)
        per[m] = dict(ratio=ratio(t['ac'], t['at'], t['rc'], t['rt']), ratio_no_empty=ratio(t['nac'], t['nat'], t['nrc'], t['nrt']), empty=t['ea'] + t['er'], empty_rejected=t['er'], rejected=t['rt'], rejected_acc=t['rc'] / t['rt'] if t['rt'] else None); T.update(t)
    o = dict(total=T['at'] + T['rt'], admissible=T['at'], rejected=T['rt'], adm_acc=T['ac'] / T['at'], rej_acc=T['rc'] / T['rt'], ratio=ratio(T['ac'], T['at'], T['rc'], T['rt']), fisher_p=fisher(T['ac'], T['at'], T['rc'], T['rt']), ci_question_reg=cluster_boot(pairs_reg), models_without_text=notext, empty=T['ea'] + T['er'], empty_admissible=T['ea'], empty_rejected=T['er'], empty_scored_correct=T['ec'], per_model=per, sim_reproduces=rep)
    if T['nrt']: o.update(ratio_no_empty=ratio(T['nac'], T['nat'], T['nrc'], T['nrt']), rej_acc_no_empty=T['nrc'] / T['nrt'], adm_acc_no_empty=T['nac'] / T['nat'], rejected_no_empty=T['nrt'], fisher_p_no_empty=fisher(T['nac'], T['nat'], T['nrc'], T['nrt']), ci_question_no_empty=cluster_boot(pairs_ne))
    if T2['rt']: o['repartition'] = dict(admissible=T2['at'], rejected=T2['rt'], adm_acc=T2['ac'] / T2['at'], rej_acc=T2['rc'] / T2['rt'], ratio=ratio(T2['ac'], T2['at'], T2['rc'], T2['rt']), ci_question=cluster_boot(pairs2))
    OUT[ds] = o
json.dump(OUT, open('audit_p3.json', 'w'), indent=1)
for ds in ['triviaqa'] + DS:
    o = OUT[ds]; print(f"===== {ds}: {o['total']} responses | registered {o['ratio']:.3f}x adm {o['adm_acc']:.3f} rej {o['rej_acc']:.3f} rejected {o['rejected']} CI(q) {[round(x,2) for x in o['ci_question_reg']]} | no texts for {o['models_without_text']}")
    print(f"   empty {o['empty']} (admissible {o['empty_admissible']}, rejected {o['empty_rejected']}, scored correct {o['empty_scored_correct']}) = {100*o['empty_rejected']/o['rejected']:.1f}% of rejected")
    if 'ratio_no_empty' in o: print(f"   empties out of the table: {o['ratio_no_empty']:.3f}x adm {o['adm_acc_no_empty']:.3f} rej {o['rej_acc_no_empty']:.3f} rejected {o['rejected_no_empty']} p {o['fisher_p_no_empty']:.2e} CI(q) {[round(x,2) for x in o['ci_question_no_empty']]}")
    if 'repartition' in o: print(f"   repartitioned on non-empty: {o['repartition']['ratio']:.3f}x adm {o['repartition']['adm_acc']:.3f} rej {o['repartition']['rej_acc']:.3f} rejected {o['repartition']['rejected']} CI(q) {[round(x,2) for x in o['repartition']['ci_question']]}")
    print('   stored matrices reproduce partition:', o['sim_reproduces']); print('   per model (ratio, no-empty, empty, rejected):', {m: (round(p['ratio'], 2) if p['ratio'] else None, round(p['ratio_no_empty'], 2) if p['ratio_no_empty'] else None, p['empty'], p['rejected']) for m, p in o['per_model'].items()})
