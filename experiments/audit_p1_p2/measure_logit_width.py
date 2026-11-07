"""Measures the temperature credal set of Definition 8 directly, from next-token logits, on the 500 Prediction 1 questions under the two-shot prompt of the base runs.

Writes logit_width.json: per question, the width of the top token's probability over the temperature range, the widest singleton event, the total-variation diameter over the grid, and the entropy at T=1. Also records next-token probabilities for "The capital of France is", which the manuscript's worked example reports.
One forward pass per question, so the widths carry no sampling noise. They describe the first answer token only, not whole responses. Needs local weights, so reproduce.sh does not run it; analyze_logit_width.py reads the stored output.

Dependencies: torch (BSD-3), transformers (Apache-2.0). Models are study subjects: Qwen and Mistral Apache-2.0, Llama 3.1 Community, Gemma Terms.
"""
import json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
TPL = 'Q: What is the capital of France?\nA: Paris\n\nQ: Who wrote Romeo and Juliet?\nA: William Shakespeare\n\nQ: {q}\nA:'
TAUS = [.3, .5, .7, 1.0, 1.3, 1.5]; EXAMPLE_TAUS = [.3, .7, 1.0, 1.3, 1.5]; EXAMPLE = 'The capital of France is'; CITIES = ['Paris', 'Lyon', 'Marseille']
SPEC = dict(Mistral=dict(name='mistralai/Mistral-7B-v0.1', file='pred1_mistral_20260110_071956.json', dtype=torch.float16), Llama=dict(name='meta-llama/Llama-3.1-8B', file='pred1_llama8b_20260112_074432.json', dtype=torch.float16), Qwen=dict(name='Qwen/Qwen2.5-7B', file='pred1_v2_20260109_053417.json', dtype=torch.float16), Gemma=dict(name='google/gemma-2-9b', file='pred1_gemma9b_20260112_074222.json', dtype=torch.bfloat16))
def last_logits(tok, model, text):
    x = tok(text, return_tensors='pt').to('mps')
    with torch.no_grad(): return model(**x).logits[0, -1].to(torch.float32).cpu().to(torch.float64)  # MPS has no float64
def probs(z): return torch.stack([torch.softmax(z / t, 0) for t in TAUS])
def city_ids(tok, z):  # token spelling differs by tokenizer, so match decoded text among the likely tokens
    top = torch.topk(z, 2000).indices.tolist(); out = {}
    for c in CITIES:
        hit = [i for i in top if tok.decode([i]).strip() == c]
        out[c] = hit[0] if hit else None
    return out
OUT = dict(taus=TAUS, template=TPL, example_prompt=EXAMPLE, models={})
for key in sys.argv[1:] or list(SPEC):
    S = SPEC[key]; t0 = time.time(); pq = json.load(open(R + S['file']))['per_question_results']
    tok = AutoTokenizer.from_pretrained(S['name'], local_files_only=True); model = AutoModelForCausalLM.from_pretrained(S['name'], dtype=S['dtype'], local_files_only=True).to('mps').eval()
    rows = {}
    for k, v in pq.items():
        z = last_logits(tok, model, TPL.format(q=v['question'])); P = probs(z); i = int(z.argmax()); w = P.max(0).values - P.min(0).values
        tv = max(float(.5 * (P[a] - P[b]).abs().sum()) for a in range(len(TAUS)) for b in range(a + 1, len(TAUS)))
        p1 = torch.softmax(z, 0); rows[k] = dict(top_token=tok.decode([i]), p_top_t1=float(p1[i]), width_top=float(P[:, i].max() - P[:, i].min()), width_max_singleton=float(w.max()), diam_tv=tv, entropy_t1=float(-(p1 * p1.clamp_min(1e-300).log()).sum()))
    z = last_logits(tok, model, EXAMPLE); ids = city_ids(tok, z)
    ex = {str(t): {**{c: (float(torch.softmax(z / t, 0)[ids[c]]) if ids[c] is not None else None) for c in CITIES}, 'top_token': tok.decode([int(z.argmax())])} for t in EXAMPLE_TAUS}
    OUT['models'][key] = dict(name=S['name'], revision=getattr(model.config, '_commit_hash', None), questions=rows, example=ex); print(key, len(rows), 'questions', round(time.time() - t0), 's', ex['1.0'], flush=True)
    json.dump(OUT, open('logit_width.json', 'w'), indent=0)
    del model; torch.mps.empty_cache()
print('done')
