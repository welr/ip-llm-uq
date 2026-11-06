"""Count begin-of-text tokens each Prediction 3 and 4 prompt receives, using the runners' own tokenization calls. Tokenizers only; no model weights or remote code."""
import json
from transformers import AutoTokenizer
Q = 'What is the capital of France?'; OUT = {}
BASE = f"Answer the following question concisely.\n\nQ: {Q}\nA:"
for n, name in dict(Qwen='Qwen/Qwen2.5-7B', Mistral='mistralai/Mistral-7B-v0.1', Llama='meta-llama/Llama-3.1-8B', Gemma='google/gemma-2-9b').items():
    t = AutoTokenizer.from_pretrained(name); ids = t(BASE)['input_ids']; OUT[f'P3 base {n}'] = dict(bos_id=t.bos_token_id, n_bos=ids.count(t.bos_token_id) if t.bos_token_id is not None else 0, first=t.convert_ids_to_tokens(ids[:2]))
# run_preregistered.py: apply_chat_template(tokenize=False), then tokenizer(prompt) with special tokens on
for n, name in dict(LlamaInstruct='meta-llama/Llama-3.1-8B-Instruct', Gemma4MoE='google/gemma-4-26B-A4B-it').items():
    try:
        t = AutoTokenizer.from_pretrained(name); p = t.apply_chat_template([{'role': 'user', 'content': f'Answer the following question concisely:\n\n{Q}'}], tokenize=False, add_generation_prompt=True); ids = t(p)['input_ids']; right = t(p, add_special_tokens=False)['input_ids']
        OUT[f'P3 instruct {n}'] = dict(n_bos=ids.count(t.bos_token_id), n_bos_without_special=right.count(t.bos_token_id), first=t.convert_ids_to_tokens(ids[:3]), template_head=p[:60])
    except Exception as e: OUT[f'P3 instruct {n}'] = dict(error=type(e).__name__ + ': ' + str(e)[:160])
# run_pred4_local_mlx.py: plain string, no chat template, on the MLX tokenizer (its encode adds BOS by default)
t = AutoTokenizer.from_pretrained('mlx-community/Meta-Llama-3.1-8B-Instruct-4bit'); p = f"Answer the following question concisely.\n\nQuestion: {Q}\nAnswer:"; ids = t.encode(p); OUT['P4 Llama instruct, no template'] = dict(n_bos=ids.count(t.bos_token_id), first=t.convert_ids_to_tokens(ids[:3]), has_template_tokens='<|start_header_id|>' in t.decode(ids))
json.dump(OUT, open('audit_tokens.json', 'w'), indent=1)
for k, v in OUT.items(): print(k, v)
