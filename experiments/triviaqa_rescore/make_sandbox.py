"""Build a sandbox holding copies of the scripts and data in stored_runs/, leaving stored_runs/ untouched.

  control: data as stored. Running the scripts here should reproduce the stored results.
  full:    the same, except `gold_answers` in the four Prediction 1 files is the benchmark's full alias list (aliases + value, as the Prediction 1 runners
           used in generation), minus aliases that normalize to the empty string, which an unguarded substring test would match against everything.

The stored records keep only `gold_answers[:3]`, and every TriviaQA label downstream was scored against those.
Usage: make_sandbox.py control|full DEST
"""
import json, os, re, shutil, sys
from datasets import load_dataset
mode, dest = sys.argv[1], sys.argv[2]
CF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs')
PRED1 = ['pred1_qwen.json', 'pred1_mistral_20260110_071956.json', 'pred1_llama8b_20260112_074432.json', 'pred1_gemma9b_20260112_074222.json']
os.makedirs(dest)
for n in os.listdir(CF):
    p = os.path.join(CF, n)
    if n == '__pycache__': continue
    if os.path.isdir(p): shutil.copytree(p, os.path.join(dest, n), ignore=shutil.ignore_patterns('__pycache__'))
    else: shutil.copy2(p, dest)
if mode == 'full':
    def norm(s): return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', s.lower().strip()))
    tq = {r['question']: r['answer'] for r in load_dataset('trivia_qa', 'rc.nocontext', split='validation')}
    n_alias = n_q = dropped = 0
    for f in PRED1:
        fn = os.path.join(dest, 'modal_data', f); d = json.load(open(fn))
        for q, v in d['per_question_results'].items():
            A = tq[v['question']]; full = list(dict.fromkeys(A['aliases'] + [A['value']])); keep = [a for a in full if norm(a)]
            assert v['gold_answers'][0] in full, q
            dropped += len(full) - len(keep); n_alias += len(keep); n_q += 1; v['gold_answers'] = keep
        json.dump(d, open(fn, 'w'))
    print('full lists written; mean aliases per record', round(n_alias / n_q, 2), 'dropped as empty after normalization', dropped)
print(mode, 'sandbox at', dest)
