"""Prediction 1, Mistral: the stored-label anomaly and the Q1 flips. Reads only."""
import json, re, sys
import numpy as np
from datasets import load_dataset
sys.path.insert(0, '../replication')
from scoring import registered_rule, word_bounded_rule
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
tq = {r['question']: r['answer'] for r in load_dataset('trivia_qa', 'rc.nocontext', split='validation')}
pq = json.load(open(R + 'pred1_mistral_20260110_071956.json'))['per_question_results']
def full(q): A = tq[q]; return list(dict.fromkeys(A['aliases'] + [A['value']]))
print('LABEL MISMATCHES: stored instruct_accuracy vs rule on stored response, full aliases')
n = [0, 0]
for k, v in pq.items():
    g = full(v['question']); a = registered_rule(v['instruct_response'], g)
    if a != bool(v['instruct_accuracy']):
        n[int(a)] += 1; hit = [x for x in g if x and re.sub(r'[^\w\s]', '', x.lower()).strip() in re.sub(r'[^\w\s]', '', v['instruct_response'].lower())][:2]
        print(f"  stored={v['instruct_accuracy']} rule={a} | {v['question'][:52]!r} | value={tq[v['question']]['value']!r} | hits={hit} | resp={v['instruct_response'][:120]!r}")
print('stored 1 rule 0:', n[0], '| stored 0 rule 1:', n[1])
