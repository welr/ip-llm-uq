"""Audit of Prediction 1 from stored data. Reads only; writes audit_p1.json.

Only six of the 15 base responses are stored (the T=0.3 and T=0.5 samples), but base_accuracy covers all 15, so correct counts over the last nine (T=0.7, 1.0, 1.3) follow by subtraction.
That gives a comparison with the instruct sample (T=1.0) on base samples at the same mean temperature that played no part in a first-six stratification."""
import collections, json, re, sys
import numpy as np
from datasets import load_dataset
sys.path.insert(0, '../replication')
from scoring import registered_rule, word_bounded_rule
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
FILES = dict(Qwen='pred1_v2_20260109_053417.json', Mistral='pred1_mistral_20260110_071956.json', Llama='pred1_llama8b_20260112_074432.json', Gemma='pred1_gemma9b_20260112_074222.json')
tq = {r['question']: r['answer'] for r in load_dataset('trivia_qa', 'rc.nocontext', split='validation')}
def full(q): A = tq[q]; return list(dict.fromkeys(A['aliases'] + [A['value']]))
def norm(s): return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', s.lower().strip()))
def diversity(T):
    N = [norm(t) for t in T if t]
    return (len(set(N)) - 1) / (len(N) - 1) if len(N) > 1 else 0.0
def quartile(x):
    a, b, c = np.percentile(x, [25, 50, 75]); return np.where(x <= a, 0, np.where(x <= b, 1, np.where(x <= c, 2, 3)))
def boot(v, B=10000, seed=42):
    rng = np.random.default_rng(seed); v = np.asarray(v, float); m = v[rng.integers(0, len(v), (B, len(v)))].mean(axis=1)
    return dict(mean=float(v.mean()), ci=[float(np.quantile(m, .025)), float(np.quantile(m, .975))], p_one_sided=float((m >= 0).mean()), p_two_sided=float(2 * min((m >= 0).mean(), (m <= 0).mean())))
ABBR = re.compile(r'(^|\s)(St|Mt|Dr|Mr|Mrs|Ms|Jr|Sr|Prof|Gen|Col|Capt|Lt|Rev|No|vs|U|D|[A-Z])$')
D = {}; OUT = {}
for m, f in FILES.items():
    pq = json.load(open(R + f))['per_question_results']; rows = []
    for k, v in pq.items():
        g = full(v['question']); T = v['base_responses'][:6]; a6 = [registered_rule(t, g) for t in T]; b6 = [word_bounded_rule(t, g) for t in T]; c15 = int(round(v['base_accuracy'] * 15))
        rows.append(dict(id=k, q=v['question'], g=g, resp=v['instruct_response'], base=v['base_accuracy'], inst=v['instruct_accuracy'], div=v['semantic_diversity'], base6=np.mean(a6), base6wb=np.mean(b6), base9=(c15 - sum(a6)) / 9, inst_reg=float(registered_rule(v['instruct_response'], g)), inst_wb=float(word_bounded_rule(v['instruct_response'], g)), div6=diversity(T), T=T))
    D[m] = rows; A = {k: np.array([r[k] for r in rows], float) for k in ['base', 'inst', 'div', 'base6', 'base6wb', 'base9', 'inst_reg', 'inst_wb', 'div6']}; Q = quartile(A['div']); o = {}
    o['inst_label_mismatch'] = int((A['inst'] != A['inst_reg']).sum()); o['base9_out_of_range'] = int(((A['base9'] < -1e-9) | (A['base9'] > 1 + 1e-9)).sum())
    o['table'] = [dict(q=i + 1, n=int((Q == i).sum()), base=100 * A['base'][Q == i].mean(), inst=100 * A['inst'][Q == i].mean(), delta=100 * (A['inst'] - A['base'])[Q == i].mean(), **{k: v for k, v in boot((A['inst'] - A['base'])[Q == i]).items() if k != 'mean'}) for i in range(4)]
    o['overall'] = dict(base=100 * A['base'].mean(), inst=100 * A['inst'].mean())
    # Scorer: change in the delta when both sides move to word-bounded, on the comparison the stored data allow (first six base vs instruct)
    o['scorer'] = [dict(q=i + 1, delta_reg=100 * (A['inst_reg'] - A['base6'])[Q == i].mean(), delta_wb=100 * (A['inst_wb'] - A['base6wb'])[Q == i].mean(), inst_lost=100 * (A['inst_reg'] - A['inst_wb'])[Q == i].mean(), base_lost=100 * (A['base6'] - A['base6wb'])[Q == i].mean()) for i in range(4)]
    # Placebo: base against itself, and instruct against the held-out base samples
    Q6 = quartile(A['div6']); o['q6_n'] = [int((Q6 == i).sum()) for i in range(4)]
    o['placebo_q6'] = [dict(q=i + 1, n=int((Q6 == i).sum()), base6=100 * A['base6'][Q6 == i].mean(), base9=100 * A['base9'][Q6 == i].mean(), inst=100 * A['inst'][Q6 == i].mean(), placebo_base9_minus_base6=boot((A['base9'] - A['base6'])[Q6 == i]), net_inst_minus_base9=boot((A['inst'] - A['base9'])[Q6 == i]), raw_inst_minus_base6=boot((A['inst'] - A['base6'])[Q6 == i])) for i in range(4)]
    o['paper_quartiles_inst_minus_base9'] = [dict(q=i + 1, **boot((A['inst'] - A['base9'])[Q == i])) for i in range(4)]
    w = [len(r['resp'].split()) for r in rows]; o['instruct_shape'] = dict(median_words=float(np.median(w)), max_words=max(w), empty=sum(not r['resp'].strip() for r in rows), cut_at_abbrev_or_initial=sum(bool(ABBR.search(r['resp'].strip())) for r in rows))
    OUT[m] = o
# External stratifier: quartiles of the other three models' mean diversity, which this model's sampling noise cannot touch
ids = [r['id'] for r in D['Qwen']]
for m in FILES:
    by = {r['id']: r for r in D[m]}; other = np.array([np.mean([next(x for x in D[o] if x['id'] == i)['div'] for o in FILES if o != m]) for i in ids]); Q = quartile(other); dl = np.array([by[i]['inst'] - by[i]['base'] for i in ids])
    OUT[m]['external'] = [dict(q=i + 1, n=int((Q == i).sum()), base=100 * np.mean([by[x]['base'] for x, z in zip(ids, Q) if z == i]), **boot(dl[Q == i])) for i in range(4)]
# Flips at Q1
for m in FILES:
    rows = D[m]; div = np.array([r['div'] for r in rows]); Q = quartile(div); Q1 = [r for r, z in zip(rows, Q) if z == 0]
    bc = [r for r in Q1 if r['base'] >= .5]; fl = [r for r in bc if r['inst'] == 0]; OUT[m]['q1_flips'] = dict(q1=len(Q1), base_correct=len(bc), flips=len(fl), rate=100 * len(fl) / max(1, len(bc)), examples=[dict(q=r['q'][:70], gold=r['g'][:2], value=tq[r['q']]['value'], resp=r['resp'][:110], modal=collections.Counter(r['T']).most_common(1)[0][0][:40]) for r in fl])
# Holm over the 16 quartile tests, two-sided
ps = sorted([(OUT[m]['table'][i]['p_two_sided'], m, i + 1) for m in FILES for i in range(4)]); alive = True; holm = []
for j, (p, m, q) in enumerate(ps): alive = alive and p <= .05 / (16 - j); holm.append(dict(model=m, q=q, p=p, holm_threshold=.05 / (16 - j), reject=alive))
OUT['holm'] = holm
json.dump(OUT, open('audit_p1.json', 'w'), indent=1)
F = lambda x: round(x, 1)
for m in FILES:
    o = OUT[m]; print('=====', m, 'instruct label mismatches', o['inst_label_mismatch'], '| base9 out of range', o['base9_out_of_range'], '| overall', F(o['overall']['base']), F(o['overall']['inst']), '| instruct', o['instruct_shape'])
    print(' table   ', [(t['n'], F(t['base']), F(t['inst']), F(t['delta']), round(t['p_one_sided'], 4), round(t['p_two_sided'], 4)) for t in o['table']])
    print(' scorer  ', [(F(s['delta_reg']), F(s['delta_wb']), 'instlost', F(s['inst_lost']), 'baselost', F(s['base_lost'])) for s in o['scorer']])
    print(' placebo q6 n', o['q6_n']); 
    for p in o['placebo_q6']: print(f"   Q{p['q']} base6 {p['base6']:.1f} base9 {p['base9']:.1f} inst {p['inst']:.1f} | placebo base9-base6 {100*p['placebo_base9_minus_base6']['mean']:+.1f} | raw inst-base6 {100*p['raw_inst_minus_base6']['mean']:+.1f} | NET inst-base9 {100*p['net_inst_minus_base9']['mean']:+.1f} [{100*p['net_inst_minus_base9']['ci'][0]:+.1f},{100*p['net_inst_minus_base9']['ci'][1]:+.1f}]")
    print(' paper Q, inst-base9', [(f"{100*x['mean']:+.1f}", [round(100*c,1) for c in x['ci']]) for x in o['paper_quartiles_inst_minus_base9']])
    print(' external', [(x['n'], F(x['base']), f"{100*x['mean']:+.1f}", [round(100*c,1) for c in x['ci']]) for x in o['external']])
    print(' Q1 flips', {k: v for k, v in o['q1_flips'].items() if k != 'examples'})
print('HOLM', [(h['model'], h['q'], round(h['p'], 4), h['reject']) for h in holm[:6]])
