"""Prediction 1: check the remaining prose claims against stored data. Reads only."""
import collections, json, re, sys
import numpy as np
from datasets import load_dataset
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
L = lambda f: json.load(open(R + f))['per_question_results']
M = L('pred1_mistral_20260110_071956.json'); G = L('pred1_gemma9b_20260112_074222.json'); W = L('pred1_v2_20260109_053417.json')
tq = {r['question']: r['answer'] for r in load_dataset('trivia_qa', 'rc.nocontext', split='validation')}
# '54 questions where both base models answer correctly but only Gemma's instruction-tuned variant preserves accuracy'
for thr in [.5, .8, 1.0]:
    S = [k for k in M if k in G and M[k]['base_accuracy'] >= thr and G[k]['base_accuracy'] >= thr and M[k]['instruct_accuracy'] == 0 and G[k]['instruct_accuracy'] == 1]
    print('threshold', thr, 'n', len(S), 'Mistral median words', np.median([len(M[k]['instruct_response'].split()) for k in S]) if S else None, 'Gemma median/max', (np.median([len(G[k]['instruct_response'].split()) for k in S]), max(len(G[k]['instruct_response'].split()) for k in S)) if S else None)
# Qwen Q3 modal-answer claim
d = np.array([v['semantic_diversity'] for v in W.values()]); a, b, c = np.percentile(d, [25, 50, 75]); ks = list(W)
Q3 = [k for k in ks if b < W[k]['semantic_diversity'] <= c]
def n(s): return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', s.lower().strip()))
for thr in [.5, 1e-9]:
    fl = [k for k in Q3 if W[k]['base_accuracy'] >= thr and W[k]['instruct_accuracy'] == 0]
    def modal_ok(k):
        T = W[k]['base_responses'][:6]; m = collections.Counter(n(t) for t in T).most_common(1)[0][0]; g = [n(x) for x in tq[W[k]['question']]['aliases'] + [tq[W[k]['question']]['value']]]
        return bool(m) and any(x and (x in m or m in x) for x in g)
    print('Qwen Q3 base>=', thr, 'flips', len(fl), 'modal correct', round(np.mean([modal_ok(k) for k in fl]), 3) if fl else None, 'mean base', round(np.mean([W[k]['base_accuracy'] for k in fl]), 3) if fl else None)
for ex in ['cathedral', 'mukluk', 'Bob Dylan']:
    for k, v in M.items():
        if ex.lower() in v['instruct_response'].lower() or ex.lower() in v['question'].lower():
            print(ex, '|', v['question'][:60], '|', v['instruct_accuracy'], '|', repr(v['instruct_response'][:90])); break
