"""Score the regenerated responses under each cut and scorer, against the manuscript's base accuracy and quartiles. Writes audit_p1_regen.json.

Variant A repeats the stored pipeline (registered cut, substring rule, 100-character cap), so its mean should sit near the stored instruct accuracy; that is the check on the rerun."""
import collections, json, os, re, sys
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
def whole(raw): return re.sub(r'\s+', ' ', raw).strip()
SCORE = dict(A_as_stored=lambda r, g: cap100(registered_cut(r), g), B_no_cap=lambda r, g: registered_rule(registered_cut(r), g), C_repaired_cut=lambda r, g: registered_rule(repaired_cut(r), g), D_full_text=lambda r, g: registered_rule(whole(r), g), E_repaired_word_bounded=lambda r, g: word_bounded_rule(repaired_cut(r), g), F_full_text_word_bounded=lambda r, g: word_bounded_rule(whole(r), g))
def boot(v, B=10000, seed=42):
    rng = np.random.default_rng(seed); v = np.asarray(v, float); m = v[rng.integers(0, len(v), (B, len(v)))].mean(axis=1)
    return dict(mean=100 * float(v.mean()), ci=[100 * float(np.quantile(m, .025)), 100 * float(np.quantile(m, .975))], p_two_sided=float(min(1, 2 * min((m >= 0).mean(), (m <= 0).mean()))))
OUT = {}; files = sorted(f for f in os.listdir('regen') if f.endswith('.json'))
for fn in files:
    key, var = fn[:-5].split('_', 1); base_run = key.endswith('Base'); fam = key.replace('Base', ''); res = json.load(open('regen/' + fn))['questions']; pq = json.load(open(R + FILES[fam]))['per_question_results']
    ks = [k for k in pq if k in res]; d = np.array([pq[k]['semantic_diversity'] for k in pq]); a, b, c = np.percentile(d, [25, 50, 75])
    Q = np.array([0 if pq[k]['semantic_diversity'] <= a else 1 if pq[k]['semantic_diversity'] <= b else 2 if pq[k]['semantic_diversity'] <= c else 3 for k in ks]); base = np.array([pq[k]['base_accuracy'] for k in ks]); st = np.array([pq[k]['instruct_accuracy'] for k in ks]); base9 = np.array([(int(round(pq[k]['base_accuracy'] * 15)) - sum(registered_rule(t, full(pq[k]['question'])) for t in pq[k]['base_responses'][:6])) / 9 for k in ks]); o = dict(stored_base9=100 * float(base9.mean()), n=len(ks), stored_inst=100 * float(st.mean()) , stored_base=100 * float(base.mean()), scores={})
    for nm, fun in SCORE.items():
        sm = np.array([np.mean([fun(r, full(pq[k]['question'])) for r in res[k]['sampled']]) for k in ks]); gr = np.array([float(fun(res[k]['greedy'], full(pq[k]['question']))) for k in ks])
        o['scores'][nm] = dict(sampled_acc=100 * float(sm.mean()), greedy_acc=100 * float(gr.mean()), by_q=[dict(q=i + 1, n=int((Q == i).sum()), sampled=100 * float(sm[Q == i].mean()), greedy=100 * float(gr[Q == i].mean()), delta_vs_base15=boot((sm - base)[Q == i]), delta_vs_base9=boot((sm - base9)[Q == i])) for i in range(4)], overall_vs_base15=boot(sm - base), overall_vs_base9=boot(sm - base9))
    w = [len(whole(r).split()) for k in ks for r in res[k]['sampled']]; o['raw_words_median'] = float(np.median(w))
    OUT[fn[:-5]] = o
json.dump(OUT, open('audit_p1_regen.json', 'w'), indent=1)
for k, o in OUT.items():
    print('=====', k, 'n', o['n'], 'stored inst', round(o['stored_inst'], 1), 'base', round(o['stored_base'], 1), 'raw median words', o['raw_words_median'])
    for nm, s in o['scores'].items():
        print(f"  {nm:26s} acc {s['sampled_acc']:.1f} greedy {s['greedy_acc']:.1f} | overall {s['overall_vs_base15']['mean']:+.1f} | " + ' '.join(f"Q{x['q']} {x['delta_vs_base15']['mean']:+.1f}[{x['delta_vs_base15']['ci'][0]:+.0f},{x['delta_vs_base15']['ci'][1]:+.0f}]" for x in s['by_q']))
        print(f"  {'   vs base9':26s} {'':24s} | overall {s['overall_vs_base9']['mean']:+.1f} | " + ' '.join(f"Q{x['q']} {x['delta_vs_base9']['mean']:+.1f}[{x['delta_vs_base9']['ci'][0]:+.0f},{x['delta_vs_base9']['ci'][1]:+.0f}]" for x in s['by_q']))
