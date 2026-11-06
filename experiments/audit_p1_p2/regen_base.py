"""Regenerate Prediction 1's base responses locally, raw text kept, at the temperature and sample count of regen_instruct.py.

The stored runs kept six of 15 base responses, all at T=0.3 and 0.5, and only the cut text. This run gives each base model three draws at T=1.0 scored by the same rules as the instruct draws.
These draws played no part in the stored diversity quartiles, so stratifying on those quartiles is free of the selection in the manuscript's comparison. The prompt difference remains: base two-shot, instruct direct.

Dependencies: torch (BSD-3), transformers (Apache-2.0). Models are study subjects: Qwen and Mistral Apache-2.0, Llama 3.1 Community, Gemma Terms.
"""
import json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
TPL = 'Q: What is the capital of France?\nA: Paris\n\nQ: Who wrote Romeo and Juliet?\nA: William Shakespeare\n\nQ: {q}\nA:'
SPEC = dict(Mistral=dict(name='mistralai/Mistral-7B-v0.1', file='pred1_mistral_20260110_071956.json', dtype=torch.float16), Llama=dict(name='meta-llama/Llama-3.1-8B', file='pred1_llama8b_20260112_074432.json', dtype=torch.float16), Qwen=dict(name='Qwen/Qwen2.5-7B', file='pred1_v2_20260109_053417.json', dtype=torch.float16), Gemma=dict(name='google/gemma-2-9b', file='pred1_gemma9b_20260112_074222.json', dtype=torch.bfloat16))
for key in sys.argv[1:]:
    S = SPEC[key]; t0 = time.time(); pq = json.load(open(R + S['file']))['per_question_results']; qs = [(k, v['question']) for k, v in pq.items()]
    tok = AutoTokenizer.from_pretrained(S['name']); model = AutoModelForCausalLM.from_pretrained(S['name'], dtype=S['dtype']).to('mps').eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    sha = getattr(model.config, '_commit_hash', None); print(key, 'loaded', round(time.time() - t0), 's', sha, flush=True)
    path = f'regen/{key}Base_two_shot.json'; res = json.load(open(path)) if os.path.exists(path) else dict(model=S['name'], revision=sha, variant='two_shot', template=TPL, questions={})
    t1 = time.time(); done = 0
    for k, q in qs:
        if k in res['questions']: continue
        torch.manual_seed(42); x = tok(TPL.format(q=q), return_tensors='pt').to('mps'); L = x['input_ids'].shape[1]
        with torch.no_grad():
            s = model.generate(**x, max_new_tokens=50, do_sample=True, temperature=1.0, top_p=.95, num_return_sequences=3, pad_token_id=tok.pad_token_id)
            g = model.generate(**x, max_new_tokens=50, do_sample=False, pad_token_id=tok.pad_token_id, temperature=None, top_p=None)
        res['questions'][k] = dict(sampled=[tok.decode(o[L:], skip_special_tokens=True) for o in s], greedy=tok.decode(g[0][L:], skip_special_tokens=True)); done += 1
        if done % 25 == 0:
            json.dump(res, open(path, 'w')); print(key, 'base', len(res['questions']), round((time.time() - t1) / done, 2), 's/q', flush=True)
    json.dump(res, open(path, 'w')); print(key, 'base done', len(res['questions']), flush=True)
    del model; torch.mps.empty_cache()
