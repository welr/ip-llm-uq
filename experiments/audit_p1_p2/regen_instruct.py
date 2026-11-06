"""Regenerate Prediction 1's instruct responses locally, keeping raw text, so the cut and the scorer can be tested. No Modal spend.

Same 500 questions, prompts, top_p, and token limit as the stored runs. Three samples at T=1.0 (the stored runs drew one) and one greedy.
Variants test two prompt differences between the runners: Qwen-Instruct ran with no chat template, and Llama's prompt carries its own begin-of-text token on top of the tokenizer's.

Dependencies: torch (BSD-3), transformers (Apache-2.0). Models are study subjects: Qwen and Mistral Apache-2.0, Llama 3.1 Community, Gemma Terms.
"""
import json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
TXT = 'Answer the following trivia question with a brief, direct answer.\n\nQuestion: {q}'
SPEC = dict(
    Mistral=dict(name='mistralai/Mistral-7B-Instruct-v0.2', file='pred1_mistral_20260110_071956.json', dtype=torch.float16, variants=dict(as_run=dict(tpl='[INST] ' + TXT + ' [/INST]', special=True))),
    Llama=dict(name='meta-llama/Llama-3.1-8B-Instruct', file='pred1_llama8b_20260112_074432.json', dtype=torch.float16, variants=dict(as_run=dict(tpl='<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' + TXT + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', special=True), single_bos=dict(tpl='<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' + TXT + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', special=False))),
    Qwen=dict(name='Qwen/Qwen2.5-7B-Instruct', file='pred1_v2_20260109_053417.json', dtype=torch.float16, variants=dict(as_run=dict(tpl=TXT + '\n\nAnswer:', special=True), chat=dict(tpl='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + TXT + '<|im_end|>\n<|im_start|>assistant\n', special=False))),
    Gemma=dict(name='google/gemma-2-9b-it', file='pred1_gemma9b_20260112_074222.json', dtype=torch.bfloat16, variants=dict(as_run=dict(tpl='<start_of_turn>user\n' + TXT + '<end_of_turn>\n<start_of_turn>model\n', special=True))),
)
os.makedirs('regen', exist_ok=True)
for key in sys.argv[1:]:
    S = SPEC[key]; t0 = time.time(); pq = json.load(open(R + S['file']))['per_question_results']; qs = [(k, v['question']) for k, v in pq.items()]
    tok = AutoTokenizer.from_pretrained(S['name']); model = AutoModelForCausalLM.from_pretrained(S['name'], dtype=S['dtype']).to('mps').eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    sha = getattr(model.config, '_commit_hash', None); print(key, 'loaded', round(time.time() - t0), 's', sha, flush=True)
    for var, V in S['variants'].items():
        path = f'regen/{key}_{var}.json'; res = json.load(open(path)) if os.path.exists(path) else dict(model=S['name'], revision=sha, variant=var, template=V['tpl'], add_special_tokens=V['special'], questions={})
        t1 = time.time(); done = 0
        for k, q in qs:
            if k in res['questions']: continue
            torch.manual_seed(42); x = tok(V['tpl'].format(q=q), return_tensors='pt', add_special_tokens=V['special']).to('mps'); L = x['input_ids'].shape[1]
            if not res.get('n_bos'): res['n_bos'] = int((x['input_ids'][0] == tok.bos_token_id).sum()) if tok.bos_token_id is not None else None
            with torch.no_grad():
                s = model.generate(**x, max_new_tokens=50, do_sample=True, temperature=1.0, top_p=.95, num_return_sequences=3, pad_token_id=tok.pad_token_id)
                g = model.generate(**x, max_new_tokens=50, do_sample=False, pad_token_id=tok.pad_token_id, temperature=None, top_p=None)
            res['questions'][k] = dict(sampled=[tok.decode(o[L:], skip_special_tokens=True) for o in s], greedy=tok.decode(g[0][L:], skip_special_tokens=True)); done += 1
            if done % 25 == 0:
                json.dump(res, open(path, 'w')); print(key, var, len(res['questions']), round((time.time() - t1) / done, 2), 's/q', flush=True)
        json.dump(res, open(path, 'w')); print(key, var, 'done', len(res['questions']), 'n_bos', res.get('n_bos'), flush=True)
    del model; torch.mps.empty_cache()
