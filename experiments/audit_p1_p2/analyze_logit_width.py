"""Relates Definition 8 width, measured from logits by measure_logit_width.py, to the semantic diversity that Predictions 1 and 4 use as their stratifier. Writes logit_width_analysis.json.

Quartiles are those of the stored runs, per model. Intervals resample questions.
"""
import json
import numpy as np
from scipy import stats
T = json.load(open('logit_width.json')); R = '../stored_runs/modal_results/'
FILES = dict(Qwen='pred1_v2_20260109_053417.json', Mistral='pred1_mistral_20260110_071956.json', Llama='pred1_llama8b_20260112_074432.json', Gemma='pred1_gemma9b_20260112_074222.json')
def quartile(x):
    a, b, c = np.percentile(x, [25, 50, 75]); return np.where(x <= a, 0, np.where(x <= b, 1, np.where(x <= c, 2, 3)))
def rho_ci(x, y, B=5000, seed=42):
    rng = np.random.default_rng(seed); n = len(x); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) and np.ptp(y[i]): v.append(stats.spearmanr(x[i], y[i])[0])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def gap_ci(a, b, B=10000, seed=42):
    rng = np.random.default_rng(seed); d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
OUT = dict(models={})
for m, f in FILES.items():
    pq = json.load(open(R + f))['per_question_results']; q = T['models'][m]['questions']; ids = list(pq)
    div = np.array([pq[k]['semantic_diversity'] for k in ids], float); Q = quartile(div); o = dict(n=len(ids), quartile_n=[int((Q == i).sum()) for i in range(4)])
    for name in ['width_top', 'diam_tv', 'entropy_t1', 'p_top_t1']:
        y = np.array([q[k][name] for k in ids], float); r = stats.spearmanr(div, y)[0]
        o[name] = dict(rho=float(r), rho_ci=rho_ci(div, y), by_quartile=[float(y[Q == i].mean()) for i in range(4)], q1_minus_q4=float(y[Q == 0].mean() - y[Q == 3].mean()), q1_minus_q4_ci=gap_ci(y[Q == 0], y[Q == 3]))
    OUT['models'][m] = o
OUT['example'] = {m: T['models'][m]['example'] for m in FILES}
json.dump(OUT, open('logit_width_analysis.json', 'w'), indent=1)
for m, o in OUT['models'].items():
    print(m, o['quartile_n'])
    for name in ['width_top', 'diam_tv', 'entropy_t1', 'p_top_t1']:
        v = o[name]; print('  ', name, 'rho', round(v['rho'], 3), [round(x, 3) for x in v['rho_ci']], 'Q1..Q4', [round(x, 3) for x in v['by_quartile']], 'Q1-Q4', round(v['q1_minus_q4'], 3), [round(x, 3) for x in v['q1_minus_q4_ci']])

# Whole-response check on the Prediction 4 run: three samples per temperature, so noisy, and an instruct model without its chat template.
import collections, re
d = json.load(open(R + 'pred4_local_llama8b_checkpoint.json'))['results']; TS = ['0.3', '0.5', '0.7', '1.0', '1.3', '1.5']
def norm(s):
    s = s.lower().strip().split('\n')[0]; s = re.split(r'(?<=[a-z0-9]{3})\.', s)[0]; s = re.sub(r'[^\w\s]', '', s).strip()
    for a in ['a ', 'an ', 'the ']:
        if s.startswith(a): s = s[len(a):]
    return s
W = collections.defaultdict(list)
for e in d.values():
    txt = {t: [norm(x['text'] if isinstance(x, dict) else x) for x in e['temperature_results'][t]['responses']] for t in TS}
    mode = collections.Counter(r for t in TS for r in txt[t]).most_common(1)[0][0]; f = [np.mean([r == mode for r in txt[t]]) for t in TS]; W[e['quartile']].append(max(f) - min(f))
OUT['p4_modal_answer'] = dict(by_quartile=[float(np.mean(W[q])) for q in ['Q1', 'Q2', 'Q3', 'Q4']], q1_minus_q4_ci=gap_ci(np.array(W['Q1']), np.array(W['Q4'])), n=[len(W[q]) for q in ['Q1', 'Q2', 'Q3', 'Q4']])
OUT['the_as_top_token'] = {m: sum(v['top_token'].strip() == 'The' for v in T['models'][m]['questions'].values()) for m in FILES}
json.dump(OUT, open('logit_width_analysis.json', 'w'), indent=1); print('modal answer', OUT['p4_modal_answer'], OUT['the_as_top_token'])

# Text and tables for the manuscript.
MO = OUT['models']; rng = lambda k: (min(MO[m][k]['rho'] for m in MO), max(MO[m][k]['rho'] for m in MO)); PM = OUT['p4_modal_answer']
assert all(MO[m][k]['rho_ci'][0] > 0 and MO[m][k]['q1_minus_q4_ci'][1] < 0 for m in MO for k in ['width_top', 'diam_tv']) and PM['q1_minus_q4_ci'][0] > 0
qs = ', '.join(f"{MO[m]['width_top']['by_quartile'][0]:.2f} to {MO[m]['width_top']['by_quartile'][3]:.2f} on {m}" for m in FILES); a, b = rng('width_top'); c, e_ = rng('diam_tv')
open('../../generated/width_check.tex', 'w').write(rf"""\paragraph{{Does diversity track credal width?}} \label{{sec:width-check}} Prediction 1 uses semantic diversity as a proxy for credal width, which we had not checked, and Prediction 4 stratifies by it. Measuring the temperature credal set of Definition~\ref{{def:temp-credal}} directly from next-token logits on the 500 questions (post hoc), the width of the top token's probability over $\tau \in [0.3, 1.5]$ rises with diversity on all four base models (Spearman $\rho$ from {a:.2f} to {b:.2f}), and the total-variation diameter of the set is more strongly associated with it ($\rho$ from {c:.2f} to {e_:.2f}). Mean top-token width goes from Q1 to Q4 as follows: {qs}. This covers the first token only, which need not be the answer (it is ``The'' on up to {max(OUT['the_as_top_token'].values())} of 500 questions). For whole responses the relation can reverse: on the Prediction 4 run, with one instruct model and three samples per temperature, so noisily, the width of the modal answer's frequency falls from {PM['by_quartile'][0]:.2f} in Q1 to {PM['by_quartile'][3]:.2f} in Q4 (Q1 minus Q4 [{PM['q1_minus_q4_ci'][0]:.2f}, {PM['q1_minus_q4_ci'][1]:.2f}]). So Prediction 1 tests diversity, which is related to credal width and not the same thing.
""")
EX = T['models']['Qwen']['example']; EL = T['models']['Llama']['example']; tb = lambda v: '$<$0.001' if v < .001 else f'{v:.3f}'
assert all(EX[t]['top_token'].strip() == 'Paris' for t in EX) and all(EX[t][c] < .002 for t in EX for c in ['Lyon', 'Marseille'])
open('../../generated/logit_example.tex', 'w').write("\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccc}\n\\toprule\nTemperature $\\tau$ & $P(\\text{Paris})$ & $P(\\text{Lyon})$ & $P(\\text{Marseille})$ \\\\\n\\midrule\n" + ''.join(f"{t} & {tb(EX[t]['Paris'])} & {tb(EX[t]['Lyon'])} & {tb(EX[t]['Marseille'])} \\\\\n" for t in EX) + "\\bottomrule\n\\end{tabular}\n\\caption{Next-token probabilities for ``The capital of France is'' on Qwen2.5-7B, measured from its logits at each temperature.}\n\\label{tab:token-example}\n\\end{table}\n")
P = [EX[t]['Paris'] for t in ['0.3', '1.0', '1.5']]
open('../../generated/logit_macros.tex', 'w').write(f"\\newcommand{{\\TokParisA}}{{{P[0]:.3f}}}\n\\newcommand{{\\TokParisB}}{{{P[1]:.3f}}}\n\\newcommand{{\\TokParisC}}{{{P[2]:.3f}}}\n\\newcommand{{\\TokParisWidth}}{{{P[0] - P[2]:.3f}}}\n\\newcommand{{\\TokLlamaParis}}{{{EL['1.0']['Paris']:.2f}}}\n")
