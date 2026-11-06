"""Writes the Prediction 2 tolerance table from the stored responses, with the ECE of audit_p2.py, so the 5% row matches audit_p2.json. Run from this folder."""
import json
import numpy as np
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
FILES = dict(Llama='pred2_local_llama70b_20260407_161230.json', Gemma='pred2_gemma_20260111_213527.json')
def ece(c, y, n=10):
    c = np.asarray(c, float); y = np.asarray(y, float); e = 0.0
    for i in range(n):
        lo, hi = i / n, (i + 1) / n; m = (c >= lo) & (c < hi) if hi < 1 else (c >= lo) & (c <= hi)
        if m.sum(): e += m.mean() * abs(y[m].mean() - c[m].mean())
    return float(e)
OUT = {}
for k, f in FILES.items():
    d = json.load(open(R + f)); rs = d.get('results') or d.get('per_question_results'); rs = list(rs.values()) if isinstance(rs, dict) else rs
    V = [r for r in rs if r.get('is_coherent') is not None and r.get('c_true') is not None and r.get('c_false') is not None]
    for t in (0, 2, 5, 10, 15, 20):
        C = [r for r in V if r['c_true'] + r['c_false'] <= 1 + t / 100 + 1e-12]; I = [r for r in V if r['c_true'] + r['c_false'] > 1 + t / 100 + 1e-12]
        OUT.setdefault(k, {})[t] = dict(coh=len(C), inc=len(I), ratio=ece([r['c_true'] for r in I], [r['is_true'] for r in I]) / ece([r['c_true'] for r in C], [r['is_true'] for r in C]))
A = json.load(open('audit_p2.json'))
for k in FILES:
    assert OUT[k][5]['coh'] == A[k]['coherent']['n'] and OUT[k][5]['inc'] == A[k]['incoherent']['n'], (k, OUT[k][5])
    assert abs(OUT[k][5]['ratio'] - A[k]['ratio']) < .005, (k, OUT[k][5]['ratio'], A[k]['ratio'])
json.dump(OUT, open('audit_p2_tolerance.json', 'w'), indent=1)
r = lambda k: [v['ratio'] for v in OUT[k].values()]
rows = ''.join(f"{t}\\% & {OUT['Llama'][t]['coh']} & {OUT['Llama'][t]['inc']} & {OUT['Llama'][t]['ratio']:.2f}$\\times$ & {OUT['Gemma'][t]['coh']} & {OUT['Gemma'][t]['inc']} & {OUT['Gemma'][t]['ratio']:.2f}$\\times$ \\\\\n" for t in OUT['Llama'])
open('../../generated/pred2_tolerance.tex', 'w').write(r"""\begin{table}[h]
\centering
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ccccccc}
\toprule
& \multicolumn{3}{c}{Llama-3.1-70B} & \multicolumn{3}{c}{Gemma-2-9B} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
Tolerance & $n_{\text{coh}}$ & $n_{\text{inc}}$ & Ratio & $n_{\text{coh}}$ & $n_{\text{inc}}$ & Ratio \\
\midrule
""" + rows + rf"""\bottomrule
\end{{tabular}}
\caption{{ECE ratio (binned, 10 bins) as a function of coherence tolerance. The ratio ranges from {min(r('Gemma')):.2f}$\times$ to {max(r('Gemma')):.2f}$\times$ for Gemma and {min(r('Llama')):.2f}$\times$ to {max(r('Llama')):.2f}$\times$ for Llama, so it does not depend on the 5\% tolerance. This table does not address the limits in \S\ref{{sec:pred2}}.}}
\label{{tab:pred2-robustness}}
\end{{table}}
""")
print(rows)
