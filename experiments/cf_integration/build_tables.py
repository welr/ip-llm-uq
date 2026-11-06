"""Write LaTeX tables in ../../generated/ from cf_results.json. Tables are never typed by hand."""
import json, os
R = json.load(open('cf_results.json')); OUTDIR = '../../generated'; os.makedirs(OUTDIR, exist_ok=True)
MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']
pct = lambda x: f"{100*x:.1f}\\%"
def pv(p):
    if p is None: return '---'
    if p < 1e-10: return '$<10^{-10}$'
    if p < 1e-3:
        m, e = f"{p:.1e}".split('e'); return f"${m}\\times 10^{{{int(e)}}}$"
    return f"{p:.3f}"
def rt(r, ci=True):
    if r['ratio'] is None: return 'undef.'
    t = f"{r['ratio']:.2f}$\\times$"
    if ci and r['ci'][0] is not None and r['rc'] < r['rt']: t += f" [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
    return t
def w(name, body): open(f'{OUTDIR}/{name}.tex', 'w').write(body)
T = [('TriviaQA', 'triviaqa', 'NLI; substring', 'Discovery set, post hoc'), ('MMLU', 'mmlu', 'NLI; registered', 'Pre-registered*'), ('ARC-Challenge', 'arc', 'NLI; registered', 'Pre-registered*'),
     ('TruthfulQA', 'truthfulqa', 'NLI; substring', 'Pre-registered'), ('TruthfulQA', 'truthfulqa_judge', 'NLI; LLM judge', 'Post hoc rescore'), ('NQ-Open', 'nq_open', 'NLI; substring', 'Pre-registered')]
SHORT = {'Discovery set, post hoc': 'Discovery', 'Pre-registered': 'Registered', 'Pre-registered*': 'Registered*', 'Post hoc rescore': 'Post hoc'}
b = "\\begin{table}[t]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llcccccc}\n\\toprule\nBenchmark & Status & $n_{\\text{rej}}$ & Rejected & Adm.\\ Acc & Rej.\\ Acc & Ratio [response CI] & Question CI \\\\\n\\midrule\n"
for n, k, sc, st in T:
    p = R[k]['pooled']; q = R[k]['ci_q']; lab = n + (' (judge)' if k == 'truthfulqa_judge' else '')
    b += f"{lab} & {SHORT[st]} & {p['rt']} & {pct(p['rej_rate'])} & {pct(p['adm_acc'])} & {pct(p['rej_acc'])} & {rt(p)} & [{q[0]:.2f}, {q[1]:.2f}] \\\\\n"
b += "\\bottomrule\n\\end{tabular}\n\\caption{Admissible versus rejected accuracy, pooled over four base models (Qwen2.5-7B, Mistral-7B, Llama-3.1-8B, Gemma-2-9B) with similarity from bidirectional NLI entailment. The ratio is admissible accuracy over rejected accuracy. Response CI: bootstrap over responses, the registered method (10,000 resamples). Question CI: bootstrap over question-model pairs (2,000 resamples, post hoc), which respects dependence among responses to one question. *With deviations from the registered prompt format and retention rule, and a lenient letter rule in the scorer; see the text. Per-model values: Table~\\ref{tab:pred3-permodel}.}\n\\label{tab:pred3-results}\n\\end{table}\n"
w('pred3_results', b)
# single hypothesis table
yes, no = 'Supported', 'Not supported'
m, a, t, n, tr = R['mmlu'], R['arc'], R['truthfulqa'], R['nq_open'], R['triviaqa']
models_gt1 = lambda d: sum(d['per_model'][x]['ratio'] > 1 for x in MODELS)
sig = lambda d: sum(d['per_model'][x]['p'] < .05 for x in MODELS)
def r2(x): return f"{x:.2f}$\\times$"
rows = []
def G(title): rows.append(f"\\midrule\n\\multicolumn{{4}}{{l}}{{\\textit{{{title}}}}} \\\\\n")
def H(l, c, res, o): rows.append(f"{l} & {c} & {res} & {o} \\\\\n")
G('Not pre-registered')
AUD = json.load(open('../audit_p1_p2/generated_rows.json')); AUD.update(json.load(open('../audit_p1_p2/generated_p1_row.json')))
H('P1', 'Accuracy loss grows with base-model diversity; restated after the data as non-linear, with location set by tuning style (TriviaQA)', AUD['P1'], 'Not supported')
H('P2', 'ECE higher for incoherent responses (MMLU statements)', AUD['P2'], 'Weak: one model, prompt-dependent')
H('P3', 'Admissible more accurate than rejected (TriviaQA)', rt(tr['pooled']), 'Holds (discovery set)')
H('P4', 'Optimal $\\tau$ varies with difficulty; restated after the data as sensitivity falling with diversity (TriviaQA, one model)', AUD['P4'], 'Not supported')
G('MMLU, registration osf.io/ady3k, registered scorer; prompt, retention, and tie-rule deviations')
H('H1a', 'Pooled ratio $>1$', f"{rt(m['pooled'])}, {pv(m['pooled']['p'])}", yes)
H('H1b', 'Pooled ratio $\\geq 1.5$', rt(m['pooled'], False), no)
H('H1c', 'Ratio $>1$ for $\\geq 3$ of 4 models', f"{models_gt1(m)} of 4", yes)
G('ARC-Challenge, registration osf.io/ady3k, registered scorer; prompt, retention, and tie-rule deviations')
H('H2a', 'Pooled ratio $>1$', f"{rt(a['pooled'])}, {pv(a['pooled']['p'])}", yes)
H('H2b', 'Pooled ratio $\\geq 1.5$', rt(a['pooled'], False), no)
H('H2c', 'Ratio $>1$ for $\\geq 3$ of 4 models', f"{models_gt1(a)} of 4", yes)
G('MMLU and ARC-Challenge, secondary, registration osf.io/ady3k')
def h3(d): return f"{r2(d['H3']['str']['ratio'])} vs.\\ {r2(d['H3']['freq']['ratio'])}, {r2(d['H3']['cons']['ratio'])}; rejection gap {d['H3_rej_gap_pp']['freq']:.1f}, {d['H3_rej_gap_pp']['cons']:.1f} pp"
H('H3', 'String admissibility beats frequency and consistency filters, rejection rates within 3 pp', 'MMLU ' + h3(m) + '. ARC ' + h3(a), no + ' (rates not comparable)')
H('H4', 'SE interval width correlates positively with difficulty', f"Spearman $\\rho={m['H4']['rho']:.2f}$ MMLU, ${a['H4']['rho']:.2f}$ ARC", no)
G('TruthfulQA, registration osf.io/dqbmg, substring scorer')
H('H1', 'Pooled ratio $>1$, $p<0.05$', f"{rt(t['pooled'])}, {pv(t['pooled']['p'])}", yes)
H('H2', 'Pooled ratio $\\geq 1.5$', rt(t['pooled'], False), yes)
H('H3', 'Ratio $>1$ for $\\geq 3$ of 4 models, $p<0.05$ for $\\geq 2$', f"{models_gt1(t)} of 4; {sig(t)} of 4", yes)
H('H4', '3--4/6 bin: lower CI $\\geq 2.0$', rt(t['bins']['3-4/6']), yes)
H('H5', '1--2/6 bin: lower CI $\\geq 0.5$', rt(t['bins']['1-2/6']), no)
H('H6', 'Pooled ratio $\\geq 1.71$', rt(t['pooled'], False), yes)
G('NQ-Open, registration osf.io/78pb6, substring scorer')
H('H1', 'Pooled ratio $>1$, $p<0.05$', f"{rt(n['pooled'])}, {pv(n['pooled']['p'])}", yes)
H('H2', 'Pooled ratio $\\geq 2.0$', rt(n['pooled'], False), no)
H('H3', 'Ratio $>1$ for $\\geq 3$ of 4 models, $p<0.05$ for $\\geq 2$', f"{models_gt1(n)} of 4; {sig(n)} of 4", yes)
H('H4', '3--4/6 bin: lower CI $\\geq 2.0$', rt(n['bins']['3-4/6']), yes)
H('H5', '1--2/6 bin: lower CI $\\geq 0.5$', rt(n['bins']['1-2/6']), no)
H('H6', 'Pooled ratio $\\geq 1.71$', rt(n['pooled'], False), no)
cut = next(i for i, r in enumerate(rows) if 'TruthfulQA, registration' in r)
HEAD = "\\begin{table}[p]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}lp{5.0cm}p{5.9cm}>{\\raggedright\\arraybackslash}p{2.2cm}@{}}\n\\toprule\nLabel & Criterion & Result & Outcome \\\\\n"
b = HEAD + ''.join(rows[:cut]) + "\\bottomrule\n\\end{tabular}\n\\caption{All hypotheses in this paper. P1--P4 are this paper's predictions and were not registered; P1 and P4 were restated after their data were seen (\\S\\ref{sec:validation}). H-labels follow each registration. Implications: the direction holds on free-form factual QA; the registered format test (H6) fails on NQ-Open; registered pooled size criteria are met only on TruthfulQA under substring scoring; the bin peak recurs on two registered benchmarks; the bounded-inversion criterion fails on both. Bins count correct responses among six.}\n\\label{tab:hypotheses}\n\\end{table}\n"
b += HEAD.replace('[p]', '[tbp]') + ''.join(rows[cut:]) + "\\bottomrule\n\\end{tabular}\n\\caption{All hypotheses in this paper, continued from Table~\\ref{tab:hypotheses}: TruthfulQA and NQ-Open registrations.}\n\\label{tab:hypotheses-2}\n\\end{table}\n"
w('hypotheses', b)
b = "\\begin{table}[h]\n\\centering\n\\footnotesize\n\\begin{tabular}{llcccrc}\n\\toprule\nBenchmark & Model & Adm.\\ Acc & Rej.\\ Acc & Ratio [95\\% CI] & $n_{\\text{rej}}$ & Fisher $p$ \\\\\n\\midrule\n"
for i, (nm, k, sc, st) in enumerate(T):
    lab = nm + (' (judge)' if k == 'truthfulqa_judge' else '')
    if i: b += "\\midrule\n"
    for x in MODELS:
        r = R[k]['per_model'][x]; b += f"{lab} & {x} & {pct(r['adm_acc'])} & {pct(r['rej_acc'])} & {rt(r)} & {r['rt']} & {pv(r['p'])} \\\\\n"; lab = ''
b += "\\bottomrule\n\\end{tabular}\n\\caption{Per-model admissibility results. Gemma on ARC-Challenge rejected 7 responses, all correct, so its ratio carries no interval.}\n\\label{tab:pred3-permodel}\n\\end{table}\n"
w('pred3_permodel', b)
BN = ['1-2/6', '3-4/6', '5/6']; b = "\\begin{table}[h]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lccc}\n\\toprule\nBenchmark & 1--2/6 & 3--4/6 & 5/6 \\\\\n\\midrule\n"
for nm, k, st in [('TriviaQA (post hoc)', 'triviaqa', ''), ('TruthfulQA, substring (registered)', 'truthfulqa', ''), ('TruthfulQA, judge (post hoc)', 'truthfulqa_judge', ''), ('NQ-Open (registered)', 'nq_open', ''), ('MMLU (registered, with deviations)', 'mmlu', ''), ('ARC-Challenge (registered, with deviations)', 'arc', '')]:
    b += nm + ''.join(f" & {rt(R[k]['bins'][x])}" for x in BN) + " \\\\\n"
b += "\\bottomrule\n\\end{tabular}\n\\caption{Admissibility ratio by the number of correct responses among six. Bins 0/6 and 6/6 are trivial. Binning conditions on the outcome, so the peak and the inversion should not be read as effect sizes.}\n\\label{tab:pred3-bins}\n\\end{table}\n"
w('pred3_bins', b)
MA = R['multi_argmax']; b = "\\begin{table}[h]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccc}\n\\toprule\nBenchmark & In stratum & Admissibility & Plurality at $\\theta=0.75$ & $n_{\\text{rej}}$ adm.\\ / plur. \\\\\n\\midrule\n"
for nm, k in [('TriviaQA', 'triviaqa'), ('ARC-Challenge', 'arc'), ('MMLU (Gemma)', 'mmlu')]:
    x = MA[k]['multi_argmax']; s = MA[k]['single_argmax']; tot = x['n_q'] + s['n_q']
    b += f"{nm} & {100*x['n_q']/tot:.1f}\\% & {rt(x['adm'])} & {r2(x['s075']['ratio'])} & {x['adm']['rt']} / {x['s075']['rt']} \\\\\n"
for nm, k in [('TruthfulQA$^\\dagger$', 'truthfulqa'), ('NQ-Open', 'nq_open')]:
    x = MA[k]['multi_argmax']; b += f"{nm} & {100*x['frac']:.1f}\\% & {x['adm_ratio']:.2f}$\\times$ [{x['adm_ratio_CI'][0]:.2f}, {x['adm_ratio_CI'][1]:.2f}] & {r2(x['se075_ratio'])} & {x['n_rej']} / --- \\\\\n"
b += "\\bottomrule\n\\end{tabular}\n\\caption{Post hoc: questions where the argmax cluster changes with $\\theta$. Off this stratum, admissibility and plurality at $\\theta=0.75$ coincide. The two columns compare different operating points, since rejected pools differ in size. $^\\dagger$Computed on rebuilt similarity matrices that do not reproduce the registered partition (pooled ratio " + f"" + r2(json.load(open('continuum_gap.json'))['truthfulqa']['grid']['ratio']) + "" + " rather than " + r2(t['pooled']['ratio']) + ").}\n\\label{tab:multi-argmax}\n\\end{table}\n"
w('multi_argmax', b)
F = R['frontier']; b = "\\begin{table}[h]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccccc}\n\\toprule\nModel & Rejection & Adm.\\ Acc & Rej.\\ Acc & Ratio [95\\% CI] & $n_{\\text{rej}}$ & Fisher $p$ \\\\\n\\midrule\n"
for nm, r in [('Claude Sonnet 4.6', F['sonnet']['combined']), ('Gemini 2.5 Flash', F['gemini']['combined']), ('Base models, subset', F['base_seed42']), ('Llama-3.1-8B-Instruct', R['instruct']['Llama-Instruct']['judge']), ('Gemma-4-26B-it', R['instruct']['Gemma-4-26B']['judge'])]:
    b += f"{nm} & {pct(r['rej_rate'])} & {pct(r['adm_acc'])} & {pct(r['rej_acc'])} & {rt(r)} & {r['rt']} & {pv(r['p'])} \\\\\n"
b += "\\bottomrule\n\\end{tabular}\n\\caption{Exploratory, not pre-registered: TruthfulQA under LLM-judge scoring. Frontier rows use $n=200$ questions and their directions disagree. Bootstrap intervals and Fisher tests can disagree at this rejected-pool size; both are reported.}\n\\label{tab:scope-exploratory}\n\\end{table}\n"
w('scope_exploratory', b)

# Prediction 4, rescored with full aliases and the empty-gold guard (rescore_pred4.py)
P4 = json.load(open('pred4_rescored.json')); TS = ['0.1', '0.3', '0.5', '0.7', '1.0', '1.3', '1.5', '2.0']
b = "\\begin{table}[h]\n\\centering\n\\small\n\\begin{tabular}{lccccccccc}\n\\toprule\n& " + ' & '.join(f"$\\tau\\!=\\!{t}$" for t in TS) + " & $n$ \\\\\n\\midrule\n"
for q, v in P4.items():
    mx = max(round(v['acc'][t], 1) for t in TS)
    b += q + ''.join(' & ' + (f"\\textbf{{{v['acc'][t]:.1f}}}" if round(v['acc'][t], 1) == mx else f"{v['acc'][t]:.1f}") for t in TS) + f" & {v['n']} \\\\\n"
b += "\\bottomrule\n\\end{tabular}\n\\caption{Accuracy (\\%) by base-model diversity quartile and temperature (Llama-3.1-8B-Instruct, $n=500$ TriviaQA questions, 3 samples per temperature), scored against the full TriviaQA alias lists. Bold indicates best temperature per quartile. All quartiles peak at low temperatures and differ in sensitivity to increases.}\n\\label{tab:pred4-results}\n\\end{table}\n"
w('pred4_results', b)

# Decision rules and readmitted responses (post hoc)
C = json.load(open('../new_problems/criteria_comparison.json')); RS = json.load(open('../new_problems/readmitted_and_strict.json')); RA = RS['readmitted']
NM = [('TriviaQA', 'triviaqa'), ('NQ-Open', 'nq_open'), ('TruthfulQA$^\\dagger$', 'truthfulqa'), ('ARC-Challenge$^\\dagger$', 'arc'), ('MMLU (Gemma)$^\\dagger$', 'mmlu')]
cell = lambda x: f"{x['ratio']:.2f}$\\times$ ({100*x['rej_rate']:.1f}\\%)"
b = "\\begin{table}[h]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccc}\n\\toprule\nBenchmark & $\\Gamma$-maximin ($\\theta=0.95$) & $\\Gamma$-maximax ($\\theta=0.50$) & $C_{\\text{sem}}$ & Without singleton ties \\\\\n\\midrule\n"
for nm, k in NM:
    e = RA[k]['excl_singleton_ties']
    b += f"{nm} & {cell(C[k]['Gmaximin'])} & {cell(C[k]['Gmaximax'])} & {cell(C[k]['C_sem'])} & {e['ratio']:.2f}$\\times$ ({100*e['rej_rate']:.1f}\\%) \\\\\n"
b += "\\bottomrule\n\\end{tabular}\n\\caption{Post hoc comparison of decision rules on the threshold credal set: admissibility ratio (rejection rate). The text gives intervals on the main differences. $^\\dagger$Stored similarity matrices that do not exactly reproduce the registered partition.}\n\\label{tab:decision-rules}\n\\end{table}\n"
w('decision_rules', b)
f3 = lambda x: f"{x:.3f}"
b = "\\begin{table}[h]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lccc}\n\\toprule\nBenchmark & Rejected by $C_{\\text{sem}}$ & Readmitted: singleton tie & Readmitted: cluster of $\\geq 2$ \\\\\n\\midrule\n"
for nm, k in NM:
    v = RA[k]
    b += f"{nm} & {f3(v['rej_acc'])} & {f3(v['tie_acc'])} ($n={v['via_singleton_tie'][1]}$) & {f3(v['real_acc'])} ($n={v['via_real_cluster'][1]}$) \\\\\n"
b += "\\bottomrule\n\\end{tabular}\n\\caption{Post hoc: accuracy of responses that plurality at $\\theta_{\\min}$ rejects and the union over thresholds readmits, against accuracy of the responses the union rejects. $^\\dagger$As in Table~\\ref{tab:decision-rules}.}\n\\label{tab:readmitted}\n\\end{table}\n"
w('readmitted', b)
# Rescoring check on the discovery data (post hoc)
DS = json.load(open('../replication/discovery_scorer_check.json')); NQw = DS['nq_open']['word_bounded']; TQw = DS['triviaqa']['word_bounded']
def sg(x, d): return (f"{x:.{d}f}" if x >= 0 and f"{x:.{d}f}"[0] != '-' else f"$-${abs(x):.{d}f}")
def eci(o, d): return f"{sg(o['estimate'], d)} [{sg(o['ci'][0], d)}, {sg(o['ci'][1], d)}]"
TG, LG = 'H1_tie_gap', 'H2_logratio_no_ties_minus_union'
NQs, TQs = DS['nq_open']['stored'], DS['triviaqa']['registered_full']
w('tie_scorer_check', f"Singleton-tie accuracy minus rejected accuracy is {eci(TQs[TG], 3)} on TriviaQA and {eci(NQs[TG], 3)} on NQ-Open. With word boundaries, which refuse bare fragments the substring rule passes, it is {eci(TQw[TG], 3)} and {eci(NQw[TG], 3)}, and the variant's log-ratio gain is {eci(TQw[LG], 3)} and {eci(NQw[LG], 3)}.")
# String-based admissibility and heuristic filters, TriviaQA (post hoc)
SF = R['string_filters']; pc1 = lambda x: f"{100*x:.1f}\\%"
b = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lcccc}\n\\toprule\nFilter & Retained Acc & Rejected Acc & \\% Rejected & Ratio \\\\\n\\midrule\n"
for lab, k in [('Admissibility (string)', 'string'), ('Consistency', 'consistency'), ('Frequency', 'frequency'), ('Length ($<$2 words)', 'length')]:
    o = SF[k]; b += f"{lab} & {pc1(o['retained_acc'])} & {pc1(o['rejected_acc'])} & {pc1(o['rej_rate'])} & {o['ratio']:.2f}$\\times$ \\\\\n"
b += f"\\midrule\n\\multicolumn{{5}}{{l}}{{\\footnotesize Random baseline accuracy: {pc1(SF['random_baseline'])}; Majority vote accuracy: {pc1(R['mv_triviaqa']['overall'])}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
b += "\\caption{String-based admissibility vs.\\ heuristic filters on TriviaQA (post hoc; aggregate across 4 models, 12,000 responses). At comparable rejection rates, string-based admissibility has the highest ratio. The length filter is counterproductive: short responses are typically correct direct answers.}\n\\label{tab:filter-comparison}\n\\end{table}\n"
w('filter_comparison', b)
print(sorted(os.listdir(OUTDIR)))
