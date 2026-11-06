"""Writes the Prediction 1 findings for the manuscript from audit results, so no number is typed. Run from this folder.

Sources: audit_p1.json (stored runs), audit_p1_matched.json (matched rerun), audit_p1_placebo.json (stratification and spread checks), audit_p1_regen.json (response length)."""
import json
L = lambda f: json.load(open(f)); ST = L('audit_p1.json'); MA = L('audit_p1_matched.json'); PL = L('audit_p1_placebo.json'); RG = L('audit_p1_regen.json')
FAM = ['Qwen', 'Mistral', 'Llama', 'Gemma']; FIX = dict(Qwen='Qwen_chat', Mistral='Mistral_as_run', Llama='Llama_single_bos', Gemma='Gemma_as_run')
neg = lambda t: t.replace('-', '$-$').replace('+', '$+$')
def v1(x):
    r = round(x, 1); return '0.0' if r == 0 else neg(f'{r:+.1f}')
def i0(x):
    r = int(abs(x) + .5) * (1 if x > 0 else -1); return '0' if r == 0 else neg(f'{r:+d}')
ci = lambda c: f'[{i0(c[0])}, {i0(c[1])}]'
cell = lambda s: f"{v1(s['mean'])} {ci(s['ci'])}"
pc = lambda x: f'{x:.1f}\\%'
C = 'C_repaired_cut'; E = 'E_repaired_word_bounded'; D = 'D_full_text'; A = 'A_as_stored'
m = lambda var, q, sc=C: MA[var]['scores'][sc]['by_q'][q - 1]['delta']
ov = lambda var, sc=C: MA[var]['scores'][sc]['overall']
st = lambda fam, q: ST[fam]['table'][q - 1]
st_cell = lambda fam, q: f"{v1(st(fam, q)['delta'])} {ci([100 * x for x in st(fam, q)['ci']])}"
st_ov = lambda fam: v1(ST[fam]['overall']['inst'] - ST[fam]['overall']['base'])
excl = lambda s: s['ci'][0] > 0 or s['ci'][1] < 0
holm_pass = [h for h in ST['holm'] if h['reject']]; assert {(h['model'], h['q']) for h in holm_pass} == {('Mistral', 1), ('Llama', 1)}
n_tests = len(ST['holm']); q3p = next(h['p'] for h in ST['holm'] if h['model'] == 'Qwen' and h['q'] == 3)
fixed = [FIX[f] for f in FAM]
q3_neg = [v for v in fixed if m(v, 3)['ci'][1] < 0]; q4_gain = [v for v in fixed if m(v, 4)['ci'][0] > 0]; q4_pos = [v for v in fixed if m(v, 4)['mean'] > 0]; assert not q3_neg
OB = PL['other_base']; OS = PL['instruct_on_other_strata']
trio = ['Mistral', 'Llama', 'Gemma']; pairs = [f'{y}_minus_{x}' for x in trio for y in trio if y != x]; q1_low = [p for p in pairs if OB[p]['by_q'][0]['ci'][1] < 0]; q4_hi = [p for p in pairs if OB[p]['by_q'][3]['ci'][0] > 0]
for f in FAM:
    assert OS[f]['by_q'][3]['ci'][1] >= 0 and m(FIX[f], 4)['ci'][1] >= 0, f
    assert OS[f]['by_q'][2]['ci'][1] >= 0 and OS[f]['by_q'][1]['ci'][1] >= 0, f
    assert (OS[f]['by_q'][0]['ci'][1] < 0) == (f == 'Qwen'), f
OV_ALL_COVER = all(ov(FIX[f])['ci'][0] <= 0 <= ov(FIX[f])['ci'][1] for f in FAM)
OVERALL_NOTE = r', with every interval covering zero, which is compatible with the assumption in \S\ref{sec:rlhf-diagnosis} that accuracy is unchanged' if OV_ALL_COVER else ''
print('overall', {f: (round(ov(FIX[f])['mean'], 1), [round(x, 1) for x in ov(FIX[f])['ci']]) for f in FAM})
WORD = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}

rows = []
def row(fam, label, cells, last): rows.append(f"{fam} & {label} & " + ' & '.join(cells) + f" & {last} \\\\\n")
def mrow(fam, label, var): row(fam, label, [cell(m(var, q)) for q in range(1, 5)], cell(ov(var)))
def srow(fam, label): row(fam, label, [st_cell(fam, q) for q in range(1, 5)], st_ov(fam))
srow('Qwen', 'Stored, no template'); mrow('', 'Matched, no template', 'Qwen_as_run'); mrow('', 'Matched, chat', 'Qwen_chat'); rows.append('\\midrule\n')
srow('Mistral', 'Stored'); mrow('', 'Matched', 'Mistral_as_run'); rows.append('\\midrule\n')
srow('Llama', 'Stored, two tokens'); mrow('', 'Matched, two tokens', 'Llama_as_run'); mrow('', 'Matched, one token', 'Llama_single_bos'); rows.append('\\midrule\n')
srow('Gemma', 'Stored'); mrow('', 'Matched', 'Gemma_as_run')
table = r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{llccccc}
\toprule
Model & Run & Q1 (low) & Q2 & Q3 & Q4 (high) & Overall \\
\midrule
""" + ''.join(rows) + r"""\bottomrule
\end{tabular}
\caption{Instruct minus base accuracy (percentage points), by base-model diversity quartile, with 95\% bootstrap intervals over questions. Exploratory. Stored rows use the stored responses and scorer. Matched rows use regenerated responses, three draws per arm at $T=1.0$, and the repaired scorer. Quartiles come from the stored runs throughout.}
\label{tab:pred1-matched}
\end{table}
"""

dis = PL['disagreement']
body = rf"""\textbf{{Evidence status:}} Not registered, and all results below are exploratory. We first stated the prediction as linear, with damage growing with base-model diversity. The first run contradicted that, and we restated it after seeing the results, so the restated form is post hoc. The pairs compare base and instruction-tuned checkpoints and do not isolate RLHF from other tuning steps.

\textbf{{Protocol:}} Four base and instruction-tuned pairs (Qwen2.5-7B \citep{{qwen2024}}, Mistral-7B \citep{{jiang2023}}, Llama-3.1-8B \citep{{dubey2024}}, and Gemma-2-9B \citep{{gemma2024}}) on the same $n=500$ TriviaQA questions \citep{{joshi2017}}. Semantic diversity, our proxy for credal width, is the proportion of unique normalized responses among 15 from the base model (3 samples $\times$ 5 temperatures: 0.3, 0.5, 0.7, 1.0, 1.3). Quartiles are computed per model.

\textbf{{Results as run:}} The stored runs compare the mean accuracy of those 15 base responses with one instruct response at $T=1.0$. Table~\ref{{tab:pred1-matched}} gives the differences. They suggest two patterns: Qwen loses at moderate diversity (Q3), and Mistral and Llama lose at low diversity (Q1). Across the {n_tests} quartile tests, only Mistral Q1 and Llama Q1 pass Holm correction at 0.05; Qwen Q3 ($p={q3p:.3f}$) does not. Neither survives the checks below.

{table}
\paragraph{{Defects in the stored runs.}} There are four. The arms are unmatched: base accuracy averages over five temperatures and instruct accuracy rests on one draw, and the quartiles are formed from the same base responses whose accuracy is measured. Qwen-Instruct ran with no chat template. Llama-Instruct's prompt carried two begin-of-text tokens. The scorer cut each response at its first period, which truncates at abbreviations and initials, and capped normalized text at 100 characters, which mislabels long responses. On regenerated Mistral responses, the cap and the cut alone move instruct accuracy from {pc(MA['Mistral_as_run']['scores'][A]['inst'])} to {pc(MA['Mistral_as_run']['scores'][C]['inst'])}.

\paragraph{{Matched rerun.}} We regenerated both arms locally: three responses per question at $T=1.0$ from each base and instruct model, on all 500 questions, scored by one rule with the cut repaired and no cap. Quartiles stay those of the stored runs, so the new draws played no part in forming them. One difference remains: base models take a two-shot prompt and instruct models a direct one.

Qwen's Q3 loss goes with matching alone: {v1(st('Qwen', 3)['delta'])} as stored, {cell(m('Qwen_as_run', 3))} matched on the same prompt, and {cell(m('Qwen_chat', 3))} with the chat template. Llama's Q1 loss falls from {v1(st('Llama', 1)['delta'])} to {cell(m('Llama_as_run', 1))} when matched and to {cell(m('Llama_single_bos', 1))} with one begin-of-text token. Mistral's Q1 loss is {v1(m('Mistral_as_run', 1, A)['mean'])} on matched responses under the stored scorer and {cell(m('Mistral_as_run', 1))} under the repaired one. Qwen shows a Q1 loss under its chat template and none without it ({cell(m('Qwen_chat', 1))} against {cell(m('Qwen_as_run', 1))}). Gemma shows none ({cell(m('Gemma_as_run', 1))}). With prompts corrected, no model's Q3 interval lies below zero, and accuracy rises at Q4 in {WORD[len(q4_pos)]} of four models, with intervals above zero in {WORD[len(q4_gain)]}. Overall, accuracy moves by {', '.join(cell(ov(FIX[f])) for f in FAM)} points for Qwen, Mistral, Llama, and Gemma{OVERALL_NOTE}.

\paragraph{{Stratification check.}} A question on which a model has low diversity is a question that model knows. A second model that knows partly different things will do worse there and better at high diversity, with no tuning involved. We tested this two ways on the same draws. First, we replaced the instruct model with another base model under the same prompt. Llama base minus Mistral base, on Mistral's quartiles, gives {cell(OB['Llama_minus_Mistral']['by_q'][0])} at Q1 and {cell(OB['Llama_minus_Mistral']['by_q'][3])} at Q4, close to Mistral-Instruct's {v1(m('Mistral_as_run', 1)['mean'])} and {v1(m('Mistral_as_run', 4)['mean'])}. Among the six ordered pairs of Mistral, Llama, and Gemma base models, {WORD[len(q1_low)]} have a Q1 interval below zero and {WORD[len(q4_hi)]} a Q4 interval above zero. Second, we kept instruct minus base and stratified on the mean diversity of the other three models, which does not favor either arm of the pair. Mistral's Q1 loss falls to {cell(OS['Mistral']['by_q'][0])}, and Llama and Gemma stay near zero ({v1(OS['Llama']['by_q'][0]['mean'])} and {v1(OS['Gemma']['by_q'][0]['mean'])}). Qwen's persists ({cell(OS['Qwen']['by_q'][0])}). Mistral's intervals at Q2 and Q3 on these strata reach zero ({cell(OS['Mistral']['by_q'][1])} and {cell(OS['Mistral']['by_q'][2])}), so a moderate-diversity loss in Mistral is not excluded. Table~\ref{{tab:pred1-strata}} gives the full results.

@@SPREAD@@

\paragraph{{Status.}} The first-stated prediction is not supported: with prompts corrected, loss does not grow with diversity, and no model's Q4 interval lies below zero on either stratification. The restated form was post hoc, and its two patterns do not survive. Qwen's moderate-diversity loss came from unmatched arms, Llama's low-diversity loss largely from the token defect, and Mistral's is reproduced by a different base model and disappears on other models' strata. The split between concise and verbose tuning fails as well: under its chat template Qwen answers in a median of {RG['Qwen_chat']['raw_words_median']:.0f} words and loses at Q1, where the restated form assigned concise tuning to moderate diversity. One effect remains, a single exploratory result: Qwen-Instruct under its chat template loses accuracy on questions the other models find easy.
"""

cells = [(f, q, dis[f]['other_strata']['delta'][q - 1]) for f in FAM for q in range(1, 5)]; lower = [c for c in cells if c[2]['mean'] < 0]; below = [c for c in cells if c[2]['ci'][1] < 0]
named = ', '.join(f"{f} Q{q} ({cell(d)})" for f, q, d in below)
spread = rf"""\paragraph{{Spread.}} The stored runs keep one instruct response per question, so they do not measure narrowing. The rerun allows a crude measure: the share of questions whose three draws are not all scored alike. On the other models' strata, instruct draws disagree less than base draws in {len(lower)} of {len(cells)} cells, with intervals below zero in {WORD.get(len(below), len(below))}: {named}. On a model's own strata the drops look larger, but another base model reproduces much of them (Mistral base minus Qwen base at Q2 is {cell(dis['Qwen']['other_base_on_own']['Mistral'][1])}, against Qwen-Instruct's {cell(dis['Qwen']['own']['delta'][1])}), so we do not use those figures. This points toward narrowing without aggregate accuracy loss, as \S\ref{{sec:rlhf-diagnosis}} assumes. It does not measure the credal set: it compares correctness labels across three draws, and the prompt differs between arms."""
body = body.replace('@@SPREAD@@', spread)
open('../../generated/pred1_results.tex', 'w').write(body)

# Appendix: stored accuracies, strata and scorer sensitivity
ap = []
for pair in (('Qwen', 'Mistral'), ('Llama', 'Gemma')):
    ap.append(' & \\multicolumn{4}{c}{' + pair[0] + '} & \\multicolumn{4}{c}{' + pair[1] + '} \\\\\n\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}\nQuartile & $n$ & Base & Inst & $\\Delta$ & $n$ & Base & Inst & $\\Delta$ \\\\\n\\midrule\n')
    for q, nm in enumerate(['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'], 1):
        ap.append(nm + ' & ' + ' & '.join(f"{st(f, q)['n']} & {st(f, q)['base']:.1f} & {st(f, q)['inst']:.1f} & {v1(st(f, q)['delta'])}" for f in pair) + ' \\\\\n')
    ap.append('Overall & ' + ' & '.join(f"500 & {ST[f]['overall']['base']:.1f} & {ST[f]['overall']['inst']:.1f} & {st_ov(f)}" for f in pair) + ' \\\\\n')
    if pair[0] == 'Qwen': ap.append('\\midrule\n')
stored = r"""\begin{table}[h]
\centering
\small
\begin{tabular}{lcccc|cccc}
\toprule
""" + ''.join(ap) + r"""\bottomrule
\end{tabular}
\caption{Accuracy (\%) by base-model diversity quartile in the stored runs, before the corrections in \S\ref{sec:pred1}. Base averages 15 responses over five temperatures; instruct is one response at $T=1.0$.}
\label{tab:pred1-stored}
\end{table}
"""
s_rows = []
for f in FAM:
    s_rows.append(f"{f} & Own quartiles & " + ' & '.join(cell(OS[f]['own_strata'][i]) for i in range(4)) + " \\\\\n")
    s_rows.append(f" & Other models' quartiles & " + ' & '.join(cell(OS[f]['by_q'][i]) for i in range(4)) + " \\\\\n")
pair_rows = ''.join(f"{p.replace('_minus_', ' minus ')} & " + ' & '.join(cell(OB[p]['by_q'][i]) for i in range(4)) + " \\\\\n" for p in pairs)
strata = r"""\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccc}
\toprule
Model & Strata & Q1 (low) & Q2 & Q3 & Q4 (high) \\
\midrule
""" + ''.join(s_rows) + r"""\bottomrule
\end{tabular}

\medskip
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
Base models & Q1 (low) & Q2 & Q3 & Q4 (high) \\
\midrule
""" + pair_rows + rf"""\bottomrule
\end{{tabular}}
\caption{{Stratification checks, in percentage points with 95\% bootstrap intervals (exploratory). Top: instruct minus base with prompts corrected, on the model's own quartiles and on quartiles of the other three models' mean diversity. Bottom: one base model minus another, with no tuning involved, on the second model's quartiles. Questions are the {PL['n_shared']} distinct ones among the 500, so own-quartile values can differ slightly from Table~\ref{{tab:pred1-matched}}.}}
\label{{tab:pred1-strata}}
\end{{table}}
"""
sc_rows = ''.join(f"{f} & {nm} & " + ' & '.join(v1(m(FIX[f], q, sc)['mean']) for q in range(1, 5)) + f" & {v1(ov(FIX[f], sc)['mean'])} \\\\\n" for f in FAM for sc, nm in ((C, 'Cut, two-way'), (E, 'Cut, word-bounded'), (D, 'Full text, two-way')))
scorers = r"""\begin{table}[h]
\centering
\footnotesize
\begin{tabular}{llccccc}
\toprule
Model & Scorer & Q1 (low) & Q2 & Q3 & Q4 (high) & Overall \\
\midrule
""" + sc_rows + r"""\bottomrule
\end{tabular}
\caption{Matched instruct minus base accuracy (percentage points) under three scorers, with prompts corrected (exploratory). Cut is the repaired first-sentence cut. The two-way rule is the substring rule of the stored runs, the word-bounded rule accepts a gold answer only at word boundaries, and the full-text scorer drops the cut and stops at an invented next question.}
\label{tab:pred1-scorers}
\end{table}
"""
appendix = rf"""\paragraph{{Protocol.}} For each question: (1) generate 15 base-model responses (3 samples $\times$ 5 temperatures); (2) compute semantic diversity as proportion of unique normalized responses; (3) measure base accuracy (average over 15 samples) and instruct accuracy (one response sampled at $T=1.0$, top-p 0.95); (4) stratify by diversity quartile. All results are exploratory. Section~\ref{{sec:pred1}} gives the defects in these runs and the corrected comparisons.

{stored}
\paragraph{{Matched rerun.}} Both arms were regenerated locally on an Apple GPU at 16-bit precision (Gemma at bfloat16), with raw text kept: three responses per question at $T=1.0$, top-p 0.95, up to 50 new tokens. Instruct models took the stored direct prompt (\texttt{{Answer the following trivia question with a brief, direct answer.}}) in each model's chat format. Base models took the stored two-shot prompt. Qwen-Instruct was run with and without its chat template, and Llama-Instruct with one and two begin-of-text tokens. Table~\ref{{tab:pred1-scorers}} gives the scorer sensitivity and Table~\ref{{tab:pred1-strata}} the stratification checks.

{scorers}
{strata}"""
open('../../generated/pred1_appendix.tex', 'w').write(appendix)

macros = open('../../generated/audit_macros.tex').read()
open('generated_p1_row.json', 'w').write(json.dumps(dict(P1=f"Matched reruns: Qwen Q3 {v1(st('Qwen', 3)['delta'])} becomes {v1(m('Qwen_chat', 3)['mean'])}, Llama Q1 {v1(st('Llama', 1)['delta'])} becomes {v1(m('Llama_single_bos', 1)['mean'])}, Mistral Q1 {v1(st('Mistral', 1)['delta'])} becomes {v1(m('Mistral_as_run', 1)['mean'])} and is reproduced by another base model; Qwen's Q1 loss under its template remains (\\S\\ref{{sec:pred1}})")))
print(body)
