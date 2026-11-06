"""Rerun Prediction 4 with the chat template the stored run omitted. Exploratory; audit only.

Same MLX model, same eight temperatures and three samples, same sampler and token limit. The stored run sent a plain completion prompt with no chat template and kept 200 characters. This run keeps full text and fixes the MLX seed per question.
Dependency: mlx_lm (MIT). Model is a study subject under the Llama 3.1 Community License."""
import json, os, sys, time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
R = '../stored_runs/modal_results/'; TEMPS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]; PATH = 'p4_template.json'
TPL = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the following question concisely.\n\nQuestion: {q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
d = json.load(open(R + 'pred4_local_llama8b_checkpoint.json'))['results']; model, tok = load('mlx-community/Meta-Llama-3.1-8B-Instruct-4bit')
ids = tok.encode(TPL.format(q='x'), add_special_tokens=False); n_bos = ids.count(tok.bos_token_id); assert n_bos == 1, n_bos
res = json.load(open(PATH)) if os.path.exists(PATH) else dict(model='mlx-community/Meta-Llama-3.1-8B-Instruct-4bit', template=TPL, n_bos=n_bos, questions={})
t0 = time.time(); done = 0
for k, v in list(d.items())[:int(sys.argv[1])]:
    if k in res['questions']: continue
    mx.random.seed(42); res['questions'][k] = dict(question=v['question'], quartile=v['quartile'], responses={str(t): [generate(model, tok, prompt=TPL.format(q=v['question']), max_tokens=50, sampler=make_sampler(temp=t)) for _ in range(3)] for t in TEMPS}); done += 1
    if done % 25 == 0: json.dump(res, open(PATH, 'w')); print(len(res['questions']), round((time.time() - t0) / done, 2), 's/q', flush=True)
json.dump(res, open(PATH, 'w')); print('done', len(res['questions']), flush=True)
