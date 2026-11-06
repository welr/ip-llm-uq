"""Instruct against base on regenerated responses: same temperature, same number of draws, same cut, same scorer. Writes audit_p1_matched.json.

Strata are the manuscript's quartiles, computed from the stored base responses, so the regenerated draws played no part in them. The prompt still differs between arms: base two-shot, instruct direct.
A base continuation runs on into an invented next question, so full-text scoring on the base arm stops at the first '\\nQ:'. Without that stop, any later invented answer would count.
Label disagreement (three draws not all scored alike) is a first look at spread. It compares correctness, not wording, because instruct responses are longer and differ more in wording."""
import json, os, re, sys
import numpy as np
from datasets import load_dataset
sys.path.insert(0, '../replication')
from scoring import registered_rule, word_bounded_rule
from cuts import registered_cut, repaired_cut
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
FILES = dict(Qwen='pred1_v2_20260109_053417.json', Mistral='pred1_mistral_20260110_071956.json', Llama='pred1_llama8b_20260112_074432.json', Gemma='pred1_gemma9b_20260112_074222.json')
tq = {r['question']: r['answer'] for r in load_dataset('trivia_qa', 'rc.nocontext', split='validation')}
def full(q): A = tq[q]; return list(dict.fromkeys(A['aliases'] + [A['value']]))
def cap100(t, g):
    n = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (s or '').lower().strip()))[:100]; x = n(t)
    return bool(x) and any(n(a) and (n(a) in x or x in n(a)) for a in g)
def whole(raw): return re.sub(r'\s+', ' ', raw.split('\nQ:')[0]).strip()
SCORE = dict(A_as_stored=lambda r, g: cap100(registered_cut(r), g), B_no_cap=lambda r, g: registered_rule(registered_cut(r), g), C_repaired_cut=lambda r, g: registered_rule(repaired_cut(r), g), E_repaired_word_bounded=lambda r, g: word_bounded_rule(repaired_cut(r), g), D_full_text=lambda r, g: registered_rule(whole(r), g), F_full_text_word_bounded=lambda r, g: word_bounded_rule(whole(r), g))
def boot(v, B=10000, seed=42):
    rng = np.random.default_rng(seed); v = np.asarray(v, float); m = v[rng.integers(0, len(v), (B, len(v)))].mean(axis=1)
    return dict(mean=100 * float(v.mean()), ci=[100 * float(np.quantile(m, .025)), 100 * float(np.quantile(m, .975))], p_two_sided=float(min(1, 2 * min((m >= 0).mean(), (m <= 0).mean()))))
OUT = {}
for fam, f in FILES.items():
    bp = f'regen/{fam}Base_two_shot.json'
    if not os.path.exists(bp): continue
    pq = json.load(open(R + f))['per_question_results']; B = json.load(open(bp))['questions']; d = np.array([pq[k]['semantic_diversity'] for k in pq]); a, b, c = np.percentile(d, [25, 50, 75])
    for fn in sorted(x for x in os.listdir('regen') if x.startswith(fam + '_') and x.endswith('.json')):
        I = json.load(open('regen/' + fn))['questions']; ks = [k for k in pq if k in I and k in B]; Q = np.array([0 if pq[k]['semantic_diversity'] <= a else 1 if pq[k]['semantic_diversity'] <= b else 2 if pq[k]['semantic_diversity'] <= c else 3 for k in ks])
        st = np.array([pq[k]['base_accuracy'] for k in ks]); si = np.array([pq[k]['instruct_accuracy'] for k in ks]); o = dict(n=len(ks), n_by_q=[int((Q == i).sum()) for i in range(4)], stored_base15=100 * float(st.mean()), stored_inst=100 * float(si.mean()), scores={})
        for nm, fun in SCORE.items():
            LI = np.array([[fun(r, full(pq[k]['question'])) for r in I[k]['sampled']] for k in ks], float); LB = np.array([[fun(r, full(pq[k]['question'])) for r in B[k]['sampled']] for k in ks], float); mi, mb = LI.mean(1), LB.mean(1); di, db = (LI.std(1) > 0).astype(float), (LB.std(1) > 0).astype(float)
            o['scores'][nm] = dict(inst=100 * float(mi.mean()), base=100 * float(mb.mean()), overall=boot(mi - mb), by_q=[dict(q=i + 1, inst=100 * float(mi[Q == i].mean()), base=100 * float(mb[Q == i].mean()), delta=boot((mi - mb)[Q == i]), disagree_inst=100 * float(di[Q == i].mean()), disagree_base=100 * float(db[Q == i].mean()), disagree_delta=boot((di - db)[Q == i])) for i in range(4)])
        o['stored_vs_regen_base_A'] = [dict(q=i + 1, stored_base15=100 * float(st[Q == i].mean()), regen_base=o['scores']['A_as_stored']['by_q'][i]['base']) for i in range(4)]
        OUT[fn[:-5]] = o
json.dump(OUT, open('audit_p1_matched.json', 'w'), indent=1)
for k, o in OUT.items():
    print('=====', k, 'n', o['n'], 'n_by_q', o['n_by_q'], 'stored base', round(o['stored_base15'], 1), 'inst', round(o['stored_inst'], 1))
    print('  stored base15 by Q', [round(x['stored_base15'], 1) for x in o['stored_vs_regen_base_A']], '| regen base T=1 by Q', [round(x['regen_base'], 1) for x in o['stored_vs_regen_base_A']])
    for nm, s in o['scores'].items():
        print(f"  {nm:26s} inst {s['inst']:.1f} base {s['base']:.1f} | overall {s['overall']['mean']:+.1f} | " + ' '.join(f"Q{x['q']} {x['delta']['mean']:+.1f}[{x['delta']['ci'][0]:+.0f},{x['delta']['ci'][1]:+.0f}]" for x in s['by_q']))
    s = o['scores']['E_repaired_word_bounded']; print('  label disagreement inst/base, scorer E:', ' '.join(f"Q{x['q']} {x['disagree_inst']:.0f}/{x['disagree_base']:.0f} ({x['disagree_delta']['mean']:+.0f}[{x['disagree_delta']['ci'][0]:+.0f},{x['disagree_delta']['ci'][1]:+.0f}])" for x in s['by_q']))
