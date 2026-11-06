"""Hand-check material for the Mistral regeneration: does the rerun match the stored labels, and what do the responses that move between scorers say?"""
import json, random, re, sys
import numpy as np
from datasets import load_dataset
sys.path.insert(0, '../replication')
from scoring import registered_rule, word_bounded_rule
from cuts import registered_cut, repaired_cut
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
pq = json.load(open(R + 'pred1_mistral_20260110_071956.json'))['per_question_results']; res = json.load(open('regen/Mistral_as_run.json'))['questions']
tq = {r['question']: r['answer'] for r in load_dataset('trivia_qa', 'rc.nocontext', split='validation')}
def full(q): A = tq[q]; return list(dict.fromkeys(A['aliases'] + [A['value']]))
def cap100(t, g):
    n = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (s or '').lower().strip()))[:100]; x = n(t)
    return bool(x) and any(n(a) and (n(a) in x or x in n(a)) for a in g)
def whole(r): return re.sub(r'\s+', ' ', r).strip()
ks = list(res); st = np.array([pq[k]['instruct_accuracy'] for k in ks]); A = np.array([np.mean([cap100(registered_cut(r), full(pq[k]['question'])) for r in res[k]['sampled']]) for k in ks]); A1 = np.array([float(cap100(registered_cut(res[k]['sampled'][0]), full(pq[k]['question']))) for k in ks])
rng = np.random.default_rng(42); d = A - st; m = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
print('regen minus stored accuracy, mean of 3', round(100 * d.mean(), 1), [round(100 * float(np.quantile(m, q)), 1) for q in (.025, .975)], '| agreement of one regen draw with stored label', round(float((A1 == st).mean()), 3))
sm = [len(whole(pq[k]['instruct_response']).split()) for k in ks]; rg = [len(registered_cut(r).split()) for k in ks for r in res[k]['sampled']]
print('median words after cut: stored', np.median(sm), 'regen', np.median(rg))
# identical text check: how often does a regen draw exactly equal the stored response?
same = sum(any(registered_cut(r).strip() == pq[k]['instruct_response'].strip() for r in res[k]['sampled']) for k in ks); print('questions where a regen draw equals the stored response', same, 'of', len(ks))
base = np.array([pq[k]['base_accuracy'] for k in ks]); hi = {k for k, b in zip(ks, base) if b >= .9}
rows = {'B_to_C': [], 'C_to_D': [], 'D_only_wb_rejects': []}
for k in ks:
    g = full(pq[k]['question'])
    for r in res[k]['sampled']:
        b, c, dd, f = registered_rule(registered_cut(r), g), registered_rule(repaired_cut(r), g), registered_rule(whole(r), g), word_bounded_rule(whole(r), g)
        if c and not b: rows['B_to_C'].append((k, r))
        if dd and not c: rows['C_to_D'].append((k, r))
        if dd and not f: rows['D_only_wb_rejects'].append((k, r))
print({k: len(v) for k, v in rows.items()}, 'of', 3 * len(ks), 'responses')
random.seed(42)
for nm, N in (('B_to_C', 14), ('C_to_D', 26)):
    print('=====', nm)
    for k, r in random.sample(rows[nm], min(N, len(rows[nm]))):
        print(f"[base {pq[k]['base_accuracy']:.2f}] Q: {pq[k]['question'][:80]}\n   gold: {tq[pq[k]['question']]['value']}\n   cut: {registered_cut(r)[:90]!r}\n   raw: {whole(r)[:230]!r}")
