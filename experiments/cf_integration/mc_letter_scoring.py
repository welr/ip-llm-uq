"""MMLU/ARC: share of all sampled responses credited only because the gold letter occurs inside a word.
Registered scorer accepts the gold letter by substring. Strict variant: see strict() below.
Generation checkpoints exist for ARC (4 models) and MMLU (Gemma only); the retained six are not stored, so this covers all 15 samples."""
import json, os, re
CF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_data', '')
def norm(s, cap=100): return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (s or '').lower().strip()))[:cap]
def reg(r, G):
    # Registered scorer as run on MMLU/ARC: 100-character cap, substring test both ways, gold letter included.
    x = norm(r); return bool(x) and any(norm(g) and (norm(g) in x or x in norm(g)) for g in G)
def strict(r, G):
    # Post hoc. Letter: case-sensitive, alone, as a leading label, or after "answer/option/choice". Text: word-bounded containment of the gold text.
    raw = (r or '').strip(); t, L = norm(G[0], 10**6), G[1]
    if re.fullmatch(r'\(?%s\)?[.:)]?' % L, raw) or re.match(r'\(?%s\)?[.:)]\s' % L, raw): return True
    if re.search(r'(?i:answer|option|choice)\s*(?i:is)?\s*:?\s*\(?%s\)?(?![A-Za-z0-9])' % L, raw): return True
    x = norm(r, 10**6)
    return bool(x) and bool(t) and (' %s ' % t) in (' %s ' % x)
OUT = {}
for ds, models in [('arc', ['Qwen','Mistral','Llama','Gemma']), ('mmlu', ['Gemma'])]:
    for m in models:
        g = json.load(open(CF + f'checkpoint_gen_{ds}_{m}.json')); n = a = b = 0
        for q in g.values():
            for r in q['responses']:
                x, y = reg(r['text'], q['gold_answers']), strict(r['text'], q['gold_answers'])
                n += 1; a += x; b += y
        OUT[f'{ds}_{m}'] = dict(n=n, registered_acc=a/n, strict_acc=b/n)
        print(ds, m, 'n', n, f'registered {a/n:.3f} strict {b/n:.3f}')
json.dump(OUT, open('mc_letter_scoring.json', 'w'), indent=1)
