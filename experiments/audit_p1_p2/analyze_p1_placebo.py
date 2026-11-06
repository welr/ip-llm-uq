"""Does the matched Q1-loss, Q4-gain pattern need instruction tuning? Writes audit_p1_placebo.json. Run from this folder.

A question on which model X has low diversity is one X knows. Any second model that knows partly different things will do worse than X there and better than X where X has high diversity, with no tuning involved.
Check 1 replaces the instruct model with another base model, same prompt, same draws: other base minus X base, on X's quartiles.
Check 2 keeps instruct minus base and stratifies on the other three models' mean diversity, so the strata do not favor X.
Check 3 applies both checks to three-draw label disagreement, the crude spread measure, which could show the same stratification effect.
Uses the corrected prompt for each instruct model. Accuracy uses scorer C (repaired cut, two-way substring rule); disagreement uses scorer E (repaired cut, word-bounded rule), as in analyze_matched.py."""
import json, sys
import numpy as np
from datasets import load_dataset
sys.path.insert(0, '../replication')
from scoring import registered_rule, word_bounded_rule
from cuts import repaired_cut
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
FILES = dict(Qwen='pred1_v2_20260109_053417.json', Mistral='pred1_mistral_20260110_071956.json', Llama='pred1_llama8b_20260112_074432.json', Gemma='pred1_gemma9b_20260112_074222.json')
INST = dict(Qwen='Qwen_chat', Mistral='Mistral_as_run', Llama='Llama_single_bos', Gemma='Gemma_as_run')
tq = {r['question']: r['answer'] for r in load_dataset('trivia_qa', 'rc.nocontext', split='validation')}
def full(q): A = tq[q]; return list(dict.fromkeys(A['aliases'] + [A['value']]))
def acc(path, keys):
    Q = json.load(open(path))['questions']
    return {keys[k]: np.mean([registered_rule(repaired_cut(r), full(keys[k])) for r in Q[k]['sampled']]) for k in Q}
def dis(path, keys):
    Q = json.load(open(path))['questions']
    return {keys[k]: float(len({bool(word_bounded_rule(repaired_cut(r), full(keys[k]))) for r in Q[k]['sampled']}) > 1) for k in Q}
def boot(v, B=10000, seed=42):
    rng = np.random.default_rng(seed); v = np.asarray(v, float); m = v[rng.integers(0, len(v), (B, len(v)))].mean(axis=1)
    return dict(mean=100 * float(v.mean()), ci=[100 * float(np.quantile(m, .025)), 100 * float(np.quantile(m, .975))])
def quart(d, qs):
    x = np.array([d[q] for q in qs]); a, b, c = np.percentile(x, [25, 50, 75]); return np.where(x <= a, 0, np.where(x <= b, 1, np.where(x <= c, 2, 3)))
DIV, KEY = {}, {}
for m, f in FILES.items():
    pq = json.load(open(R + f))['per_question_results']; KEY[m] = {k: v['question'] for k, v in pq.items()}; DIV[m] = {v['question']: v['semantic_diversity'] for v in pq.values()}
qs = sorted(set.intersection(*[set(d) for d in DIV.values()])); print('questions shared by all four', len(qs))
BASE = {m: acc(f'regen/{m}Base_two_shot.json', KEY[m]) for m in FILES}; IN = {m: acc(f'regen/{INST[m]}.json', KEY[m]) for m in FILES}
OUT = dict(n_shared=len(qs), other_base={}, instruct_on_other_strata={})
for x in FILES:
    Q = quart(DIV[x], qs); bx = np.array([BASE[x][q] for q in qs])
    for y in FILES:
        if y == x: continue
        d = np.array([BASE[y][q] for q in qs]) - bx
        OUT['other_base'][f'{y}_minus_{x}'] = dict(overall=boot(d), by_q=[boot(d[Q == i]) for i in range(4)])
    others = {q: np.mean([DIV[m][q] for m in FILES if m != x]) for q in qs}; O = quart(others, qs); d = np.array([IN[x][q] for q in qs]) - bx
    OUT['instruct_on_other_strata'][x] = dict(variant=INST[x], n_by_q=[int((O == i).sum()) for i in range(4)], overall=boot(d), by_q=[boot(d[O == i]) for i in range(4)], own_strata=[boot(d[Q == i]) for i in range(4)], base_by_q=[100 * float(bx[O == i].mean()) for i in range(4)])
DB = {m: dis(f'regen/{m}Base_two_shot.json', KEY[m]) for m in FILES}; DI = {m: dis(f'regen/{INST[m]}.json', KEY[m]) for m in FILES}; OUT['disagreement'] = {}
for x in FILES:
    Q = quart(DIV[x], qs); O = quart({q: np.mean([DIV[m][q] for m in FILES if m != x]) for q in qs}, qs); b = np.array([DB[x][q] for q in qs]); i_ = np.array([DI[x][q] for q in qs])
    OUT['disagreement'][x] = dict(own=dict(base=[100 * float(b[Q == i].mean()) for i in range(4)], inst=[100 * float(i_[Q == i].mean()) for i in range(4)], delta=[boot((i_ - b)[Q == i]) for i in range(4)]), other_strata=dict(base=[100 * float(b[O == i].mean()) for i in range(4)], inst=[100 * float(i_[O == i].mean()) for i in range(4)], delta=[boot((i_ - b)[O == i]) for i in range(4)]), other_base_on_own={y: [boot((np.array([DB[y][q] for q in qs]) - b)[Q == i]) for i in range(4)] for y in FILES if y != x})
json.dump(OUT, open('audit_p1_placebo.json', 'w'), indent=1)
f = lambda s: f"{s['mean']:+.1f}[{s['ci'][0]:+.0f},{s['ci'][1]:+.0f}]"
print('--- check 1: other base minus X base, on X quartiles')
for k, v in OUT['other_base'].items(): print(f'{k:24s} overall {f(v["overall"])} | ' + ' '.join(f'Q{i + 1} {f(s)}' for i, s in enumerate(v['by_q'])))
print('--- check 2: instruct minus base, own strata then other-model strata')
for k, v in OUT['instruct_on_other_strata'].items():
    print(k, 'own  ', ' '.join(f'Q{i + 1} {f(s)}' for i, s in enumerate(v['own_strata']))); print(k, 'other', ' '.join(f'Q{i + 1} {f(s)}' for i, s in enumerate(v['by_q'])), 'n', v['n_by_q'], 'base', [round(b, 1) for b in v['base_by_q']])
print('--- check 3: label disagreement, instruct minus base')
for k, v in OUT['disagreement'].items():
    print(k, 'own strata  ', ' '.join(f'Q{i + 1} {v["own"]["inst"][i]:.0f}/{v["own"]["base"][i]:.0f} {f(v["own"]["delta"][i])}' for i in range(4)))
    print(k, 'other strata', ' '.join(f'Q{i + 1} {v["other_strata"]["inst"][i]:.0f}/{v["other_strata"]["base"][i]:.0f} {f(v["other_strata"]["delta"][i])}' for i in range(4)))
    for y, r in v['other_base_on_own'].items(): print('   ', y, 'base minus', k, 'base', ' '.join(f'Q{i + 1} {f(r[i])}' for i in range(4)))
