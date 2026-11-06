"""Prediction 4 with the instruct model's chat template, against the stored plain-completion run on the same questions. Exploratory; writes audit_p4_template.json.

Scoring uses the manuscript's guarded rule on full text and the repaired-cut word-bounded rule. Intervals are bootstrap over questions within quartile; the sensitivity comparison is Q1 minus Q4."""
import json, re, sys
import numpy as np
from datasets import load_dataset
sys.path.insert(0, '../replication')
from scoring import word_bounded_rule
from cuts import repaired_cut
d = json.load(open('p4_template.json')); QQ = d['questions'] if 'questions' in d else d
S = json.load(open('../stored_runs/modal_results/pred4_local_llama8b_checkpoint.json'))['results']
gold = {}
for sp in ['validation', 'train']:
    for r in load_dataset('trivia_qa', 'rc.nocontext', split=sp): gold.setdefault(r['question'], r['answer']['aliases'] + r['answer']['normalized_aliases'])
def norm(s):
    s = s.lower().strip()
    for a in ['a ', 'an ', 'the ']:
        if s.startswith(a): s = s[len(a):]
    return re.sub(r'[^\w\s]', '', s).strip()
def guarded(r, G):
    x = norm(r); return bool(x) and any(norm(g) and (norm(g) in x or x in norm(g)) for g in G)
RULES = dict(full_text=guarded, repaired_word_bounded=lambda r, G: word_bounded_rule(repaired_cut(r), G))
ks = [k for k in QQ if k in S]; T = sorted(next(iter(QQ.values()))['responses'], key=float); Q = ['Q1', 'Q2', 'Q3', 'Q4']; qt = np.array([QQ[k]['quartile'] for k in ks])
def texts_new(k, t): return QQ[k]['responses'][t]
def texts_old(k, t): return [(x['text'] if isinstance(x, dict) else x) for x in S[k]['temperature_results'][t]['responses']]
OUT = dict(n=len(ks), counts={q: int((qt == q).sum()) for q in Q}, temperatures=T); rng = np.random.default_rng(42)
for run, tx in [('chat_template', texts_new), ('stored_plain', texts_old)]:
    OUT[run] = {}
    for nm, fn in RULES.items():
        M = np.array([[np.mean([fn(r, gold[QQ[k]['question']]) for r in tx(k, t)]) for t in T] for k in ks]); prof = {q: M[qt == q].mean(axis=0) for q in Q}; i1 = T.index('1.0')
        B = 10000; bs = {q: M[np.where(qt == q)[0][rng.integers(0, (qt == q).sum(), (B, (qt == q).sum()))]].mean(axis=1) for q in Q}
        sd = {q: float(prof[q].std()) for q in Q}; keep = {q: float(prof[q][i1] / prof[q].max()) for q in Q}; ns = {q: float((prof[q] / prof[q].max()).std()) for q in Q}; sdb = {q: bs[q].std(axis=1) for q in Q}; kb = {q: bs[q][:, i1] / bs[q].max(axis=1) for q in Q}; nb = {q: (bs[q] / bs[q].max(axis=1, keepdims=True)).std(axis=1) for q in Q}
        ci = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .975))]
        OUT[run][nm] = dict(acc={q: {t: 100 * float(prof[q][i]) for i, t in enumerate(T)} for q in Q}, overall={t: 100 * float(M[:, i].mean()) for i, t in enumerate(T)}, best_t={q: T[int(prof[q].argmax())] for q in Q}, sd=sd, keep_at_1=keep, normalized_sd=ns, sd_q1_q4_ci=ci(sdb['Q1'] - sdb['Q4']), sd_q1_q2_ci=ci(sdb['Q1'] - sdb['Q2']), keep_q1_q4_ci=ci(kb['Q1'] - kb['Q4']), normalized_q1_q4_ci=ci(nb['Q1'] - nb['Q4']))
json.dump(OUT, open('audit_p4_template.json', 'w'), indent=1)
print('n', OUT['n'], OUT['counts']); print('T', T)
for run in ['chat_template', 'stored_plain']:
    for nm in RULES:
        o = OUT[run][nm]; print('=====', run, nm)
        for q in Q: print('  ', q, ' '.join(f"{o['acc'][q][t]:5.1f}" for t in T), f"| best {o['best_t'][q]} sd {o['sd'][q]:.3f} keep@1 {100*o['keep_at_1'][q]:.0f}% norm {o['normalized_sd'][q]:.3f}")
        print('   SD Q1-Q4', [round(x, 3) for x in o['sd_q1_q4_ci']], 'SD Q1-Q2', [round(x, 3) for x in o['sd_q1_q2_ci']], 'keep Q1-Q4', [round(x, 3) for x in o['keep_q1_q4_ci']], 'norm Q1-Q4', [round(x, 3) for x in o['normalized_q1_q4_ci']])
