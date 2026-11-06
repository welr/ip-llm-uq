"""Audit of Prediction 4 from stored responses. Reads only; writes audit_p4.json.

The runner gave an instruct model a plain completion prompt with no chat template, did not cut the response, and scored all 50 tokens (stored to 200 characters). A continuation that invents further questions can therefore earn credit for text after the first answer. These tests rescore the same responses under stricter cuts and put question-bootstrap intervals on the quartile comparisons."""
import json, re, statistics as st, sys
import numpy as np
from datasets import load_dataset
sys.path.insert(0, '../replication')
from scoring import word_bounded_rule
from cuts import repaired_cut
P = '../stored_runs/modal_results/pred4_local_llama8b_checkpoint.json'; d = json.load(open(P)); print('complete flag', d.get('complete'), 'n', len(d['results'])); d = d['results']
gold = {}
for sp in ['validation', 'train']:
    for r in load_dataset('trivia_qa', 'rc.nocontext', split=sp): gold.setdefault(r['question'], r['answer']['aliases'] + r['answer']['normalized_aliases'])
def norm(s):
    s = s.lower().strip()
    for a in ['a ', 'an ', 'the ']:
        if s.startswith(a): s = s[len(a):]
    return re.sub(r'[^\w\s]', '', s).strip()
def guarded(r, G):
    x = norm(r)
    return bool(x) and any(norm(g) and (norm(g) in x or x in norm(g)) for g in G)
first_line = lambda r: r.lstrip().split('\n')[0]
RULES = dict(A_manuscript_full_text=lambda r, G: guarded(r, G), B_first_line=lambda r, G: guarded(first_line(r), G), C_repaired_cut=lambda r, G: guarded(repaired_cut(r), G), D_repaired_word_bounded=lambda r, G: word_bounded_rule(repaired_cut(r), G))
Q = ['Q1', 'Q2', 'Q3', 'Q4']; T = sorted(next(iter(d.values()))['temperature_results'], key=float); ks = list(d); qt = np.array([d[k]['quartile'] for k in ks])
txt = {k: {t: [(x['text'] if isinstance(x, dict) else x) for x in d[k]['temperature_results'][t]['responses']] for t in T} for k in ks}
shape = {t: dict(invented_question=100 * np.mean([bool(re.search(r'\n\s*(Question|Q)\s*:', r)) for k in ks for r in txt[k][t]]), has_newline=100 * np.mean(['\n' in r.strip() for k in ks for r in txt[k][t]]), empty=100 * np.mean([not norm(r) for k in ks for r in txt[k][t]]), median_words=float(np.median([len(r.split()) for k in ks for r in txt[k][t]]))) for t in T}
OUT = dict(shape=shape, rules={})
rng = np.random.default_rng(42)
for nm, fun in RULES.items():
    M = np.array([[np.mean([fun(r, gold[d[k]['question']]) for r in txt[k][t]]) for t in T] for k in ks])
    def summ(idx): m = M[idx].mean(axis=0); return m
    prof = {q: summ(np.where(qt == q)[0]) for q in Q}; sd = {q: float(np.std(prof[q])) for q in Q}; keep = {q: float(prof[q][T.index('1.0')] / prof[q].max()) for q in Q}; best = {q: T[int(np.argmax(prof[q]))] for q in Q}
    B = 10000; bs = {q: np.zeros((B, len(T))) for q in Q}
    for q in Q:
        ix = np.where(qt == q)[0]; bs[q] = M[ix[rng.integers(0, len(ix), (B, len(ix)))]].mean(axis=1)
    sdb = {q: bs[q].std(axis=1) for q in Q}; kpb = {q: bs[q][:, T.index('1.0')] / bs[q].max(axis=1) for q in Q}
    cmp = {f'{a}-{b}': [float(np.quantile(sdb[a] - sdb[b], .025)), float(np.quantile(sdb[a] - sdb[b], .975))] for a, b in [('Q1', 'Q2'), ('Q2', 'Q3'), ('Q3', 'Q4'), ('Q1', 'Q4')]}; kcmp = {f'{a}-{b}': [float(np.quantile(kpb[a] - kpb[b], .025)), float(np.quantile(kpb[a] - kpb[b], .975))] for a, b in [('Q1', 'Q4'), ('Q2', 'Q3')]}
    # Normalized sensitivity: profile divided by that quartile's own peak, so the floor effect is removed
    nsd = {q: float(np.std(prof[q] / prof[q].max())) for q in Q}; nb = {q: (bs[q] / bs[q].max(axis=1, keepdims=True)).std(axis=1) for q in Q}; ncmp = {'Q1-Q4': [float(np.quantile(nb['Q1'] - nb['Q4'], .025)), float(np.quantile(nb['Q1'] - nb['Q4'], .975))]}
    OUT['rules'][nm] = dict(acc={q: {t: 100 * float(prof[q][i]) for i, t in enumerate(T)} for q in Q}, sd=sd, sd_diff_ci=cmp, keep_at_1=keep, keep_diff_ci=kcmp, best_t=best, normalized_sd=nsd, normalized_sd_diff_ci=ncmp)
    print('=====', nm); print('   T      ', ' '.join(f'{t:>5s}' for t in T))
    for q in Q: print('  ', q, '   ', ' '.join(f'{100*x:5.1f}' for x in prof[q]), f"| sd {sd[q]:.3f} keep@1 {100*keep[q]:.0f}% best T {best[q]} | norm sd {nsd[q]:.3f}")
    print('   sd differences (95% CI):', {k: [round(x, 3) for x in v] for k, v in cmp.items()}); print('   keep@1 differences:', {k: [round(x, 3) for x in v] for k, v in kcmp.items()}, '| normalized sd Q1-Q4:', [round(x, 3) for x in ncmp['Q1-Q4']])
json.dump(OUT, open('audit_p4.json', 'w'), indent=1)
print('--- response shape by T'); [print(' ', t, {k: round(v, 1) for k, v in s.items()}) for t, s in shape.items()]
print('--- sample at T=0.5 and T=1.0');
import random; random.seed(1)
for k in random.sample(ks, 4): print(' ', d[k]['question'][:70], '| gold', gold[d[k]['question']][0][:25], '\n     .5:', repr(txt[k]['0.5'][0][:150]), '\n    1.0:', repr(txt[k]['1.0'][0][:150]))
