"""Prediction 1: what the 100-character cap and the first-sentence cut do to the stored tables. Reads only.
All four runners capped normalized text at 100 characters. Stored labels change only for Mistral, the one model whose responses run that long."""
import json, re, sys
import numpy as np
from datasets import load_dataset
sys.path.insert(0, '../replication')
from scoring import registered_rule
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
tq = {r['question']: r['answer'] for r in load_dataset('trivia_qa', 'rc.nocontext', split='validation')}
pq = json.load(open(R + 'pred1_mistral_20260110_071956.json'))['per_question_results']
def full(q): A = tq[q]; return list(dict.fromkeys(A['aliases'] + [A['value']]))
ABBR = re.compile(r'(^|\s)(St|Mt|Dr|Mr|Mrs|Ms|Jr|Sr|Prof|Gen|Col|Capt|Lt|Rev|No|vs|[A-Z])$')
rows = [dict(base=v['base_accuracy'], inst=v['instruct_accuracy'], fix=float(registered_rule(v['instruct_response'], full(v['question']))), div=v['semantic_diversity'], resp=v['instruct_response'], abbr=bool(ABBR.search(v['instruct_response'].strip()))) for v in pq.values()]
d = np.array([r['div'] for r in rows]); a, b, c = np.percentile(d, [25, 50, 75]); Q = np.where(d <= a, 0, np.where(d <= b, 1, np.where(d <= c, 2, 3)))
B = np.array([r['base'] for r in rows]); I = np.array([r['inst'] for r in rows]); F = np.array([r['fix'] for r in rows]); AB = np.array([r['abbr'] for r in rows])
out = dict(overall=dict(stored=100 * (I - B).mean(), cap_fixed=100 * (F - B).mean()), by_q=[dict(q=i + 1, n=int((Q == i).sum()), stored=100 * (I - B)[Q == i].mean(), cap_fixed=100 * (F - B)[Q == i].mean(), wrong_cut_at_abbrev=int((AB & (F == 0) & (Q == i)).sum())) for i in range(4)])
fl = (Q == 0) & (B >= .5) & (I == 0); out['q1_flips'] = dict(total=int(fl.sum()), correct_after_cap_fix=int((fl & (F == 1)).sum()), wrong_and_cut_at_abbrev=int((fl & (F == 0) & AB).sum()))
print(json.dumps(out, indent=0)); json.dump(out, open('audit_p1_cap.json', 'w'), indent=1)
