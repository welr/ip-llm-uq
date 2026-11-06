"""Prediction 2: does the result depend on the prompt and the parser? Exploratory; not part of any registered analysis.

The original prompt states 'This statement is TRUE.' before asking for confidence that it is true, and the same for FALSE, and the parser takes the first number it finds.
Here the prompt asserts nothing, asks for a bare number, and a response counts only if it is a bare number. Same model (Llama-3.1-70B-Instruct 4-bit, MLX), same statements, first N in stored order.

Dependency: mlx-lm (MIT)."""
import json, os, re, sys, time
from mlx_lm import load, generate
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
rs = list(json.load(open(R + 'pred2_local_llama70b_20260407_161230.json'))['results'].items())[:N]
TPL = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
def prompt(s, side): return f'Consider this statement: "{s}"\n\nWhat is the probability, from 0 to 100, that the statement is {side}? Reply with only the number.'
STRICT = re.compile(r'^\**\s*(\d{1,3}(?:\.\d+)?)\s*%?\s*\**\.?$')
def strict(t):
    m = STRICT.match(t.strip())
    if not m: return None
    v = float(m.group(1)); return v / 100 if v <= 100 else None
model, tok = load('mlx-community/Meta-Llama-3.1-70B-Instruct-4bit'); path = 'neutral_prompt_llama70b.json'; out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()
for i, (k, r) in enumerate(rs):
    if k in out: continue
    a = generate(model, tok, prompt=TPL.format(p=prompt(r['statement'], 'true')), max_tokens=12); b = generate(model, tok, prompt=TPL.format(p=prompt(r['statement'], 'false')), max_tokens=12)
    out[k] = dict(statement=r['statement'], is_true=r['is_true'], subset=r['subset'], response_true=a, response_false=b, c_true=strict(a), c_false=strict(b), original_is_coherent=r['is_coherent'], original_c_true=r['c_true'], original_c_false=r['c_false'])
    if (i + 1) % 25 == 0: json.dump(out, open(path, 'w')); print(i + 1, round((time.time() - t0) / (i + 1), 2), 's/statement', flush=True)
json.dump(out, open(path, 'w')); print('done', len(out))
