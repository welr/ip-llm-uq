"""Single source for every choice-function number in the manuscript.

Reads the stored result files of the choice-function study, recomputes pooled
counts, ratios, CIs, and registered hypothesis outcomes, and writes cf_results.json.
Run with ~/ip_llm/bin/python (numpy, scipy).
"""
import json, os, glob, collections
import numpy as np
from scipy import stats
CF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', '')
MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']
GRID = [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95]
N_BOOT = 10000

def L(f): return json.load(open(CF + f))
# TriviaQA analyses rerun on labels scored against the full alias lists; the stored files used the first three aliases (see ../triviaqa_rescore).
TQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../triviaqa_rescore/results_full/')
def LT(f): return json.load(open(TQ + f))
def boot(ac, at, rc, rt, seed=42):
    if not (at and rt) or rc == 0: return (None, None)
    rng = np.random.RandomState(seed)
    a = rng.binomial(at, ac / at, N_BOOT) / at
    r = rng.binomial(rt, rc / rt, N_BOOT) / rt
    keep = r > 0
    return tuple(float(x) for x in np.quantile(a[keep] / r[keep], [.025, .975]))
def fisher(ac, at, rc, rt): return float(stats.fisher_exact([[ac, at - ac], [rc, rt - rc]])[1])
def row(ac, at, rc, rt):
    lo, hi = boot(ac, at, rc, rt)
    return dict(ac=ac, at=at, rc=rc, rt=rt, adm_acc=ac / at if at else None, rej_acc=rc / rt if rt else None,
                ratio=(ac / at) / (rc / rt) if at and rt and rc else None, ci=[lo, hi],
                p=fisher(ac, at, rc, rt) if at and rt else None, rej_rate=rt / (at + rt) if at + rt else None)
def add(c, accs, adm):
    for i, x in enumerate(accs):
        k = 0 if i in adm else 2
        c[k] += int(bool(x)); c[k + 1] += 1
BINS = {'0/6': [0], '1-2/6': [1, 2], '3-4/6': [3, 4], '5/6': [5], '6/6': [6]}
def bin_of(n):
    return next(b for b, v in BINS.items() if n in v)
def comps(S, t):
    n = len(S); p = list(range(n))
    def f(a):
        while p[a] != a: p[a] = p[p[a]]; a = p[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if S[i][j] >= t: p[f(i)] = f(j)
    g = collections.defaultdict(list)
    for i in range(n): g[f(i)].append(i)
    return list(g.values())
def adm_set(S):
    A = set()
    for t in GRID:
        c = comps(S, t); m = max(map(len, c))
        for x in c:
            if len(x) == m: A |= set(x)
    return A
def qci(pairs, seed=42, B=2000):
    x = np.array(pairs, float); rng = np.random.default_rng(seed); r = []
    for _ in range(B):
        b = x[rng.integers(0, len(x), len(x))].sum(axis=0)
        if b[2] > 0 and b[3] > 0: r.append((b[0] / b[1]) / (b[2] / b[3]))
    return [float(np.quantile(r, .025)), float(np.quantile(r, .975))]
OUT = {}
# TriviaQA (discovery set, no registration)
acc = {(r['model'], r['qid']): r['accuracies'] for r in LT('kle_snne_triviaqa_results.json') if r['dataset'] == 'triviaqa'}
tri = {m: [0, 0, 0, 0] for m in MODELS}; tb = {b: [0, 0, 0, 0, 0] for b in BINS}; prs = []
for m in MODELS:
    for q, S in L(f'sim_cache/sim_matrices_triviaqa_{m}.json').items():
        if (m, q) not in acc: continue
        a = acc[(m, q)]; A = adm_set(S); add(tri[m], a, A)
        pc = [0, 0, 0, 0]; add(pc, a, A); prs.append(pc)
        b = bin_of(sum(map(int, a))); c = tb[b]; c[4] += 1
        add(c, a, A)
def pool(d): return [sum(v[i] for v in d.values()) for i in range(4)]
OUT['triviaqa'] = dict(ci_q=qci(prs), per_model={m: row(*v) for m, v in tri.items()}, pooled=row(*pool(tri)),
                       bins={b: dict(row(*v[:4]), n_q=v[4]) for b, v in tb.items()})
OUT['triviaqa']['jan_run'] = {k: L('modal_data/pred3_choice_functions.json')['aggregate'][k] for k in
                              ['admissible_correct', 'admissible_total', 'rejected_correct', 'rejected_total', 'accuracy_ratio']}
# Registered benchmarks
def files(ds):
    r = {}
    for m in MODELS:
        c = [f for f in glob.glob(CF + f'modal_data/prereg_{ds}_*.json') if f'_{m}_' in f or f.endswith(f'_{m}.json')]
        r[m] = json.load(open(sorted(c)[-1]))
    return r
for ds in ['mmlu', 'arc', 'truthfulqa', 'nq_open']:
    F = files(ds); pm = {}; bn = {b: [0, 0, 0, 0, 0] for b in BINS}; st = {m: [0, 0, 0, 0] for m in MODELS}
    H3 = dict(str=[0, 0, 0, 0], freq=[0, 0, 0, 0], cons=[0, 0, 0, 0]); qa = collections.defaultdict(list); qw = collections.defaultdict(list); prs = []
    for m, d in F.items():
        c = [0, 0, 0, 0]
        for q, e in d['per_question'].items():
            a = e['accuracies']; A = set(e['nli_admissible']); add(c, a, A)
            pc = [0, 0, 0, 0]; add(pc, a, A); prs.append(pc)
            b = bn[bin_of(sum(map(bool, a)))]; b[4] += 1; add(b, a, A)
            add(st[m], a, set(e['str_admissible']))
            for key, name in [('str_rejected', 'str'), ('freq_rejected', 'freq'), ('cons_rejected', 'cons')]:
                add(H3[name], a, set(range(len(a))) - set(e[key]))
            qa[q].append(np.mean(a)); qw[q].append(e['se_upper'] - e['se_lower'])
        pm[m] = row(*c)
    p = pool({m: [pm[m][k] for k in ['ac', 'at', 'rc', 'rt']] for m in MODELS})
    keys = sorted(qa); rho = stats.spearmanr([np.mean(qw[k]) for k in keys], [1 - np.mean(qa[k]) for k in keys])
    OUT[ds] = dict(ci_q=qci(prs), per_model=pm, pooled=row(*p), bins={b: dict(row(*v[:4]), n_q=v[4]) for b, v in bn.items()},
                   str_per_model={m: row(*v) for m, v in st.items()},
                   H3={k: row(*v) for k, v in H3.items()}, H4=dict(rho=float(rho[0]), p=float(rho[1])))
# TruthfulQA judge rescore (post hoc)
J = {}; jb = {b: [0, 0, 0, 0, 0] for b in BINS}; prs = []
for m in MODELS:
    d = L(f'results/rescore_{m}_judge.json'); c = [0, 0, 0, 0]; pq = d['per_question']
    for q, e in (pq.items() if isinstance(pq, dict) else enumerate(pq)):
        a = e['accuracies']; A = set(e['nli_admissible']); add(c, a, A)
        pc = [0, 0, 0, 0]; add(pc, a, A); prs.append(pc)
        b = jb[bin_of(sum(map(bool, a)))]; b[4] += 1; add(b, a, A)
    J[m] = row(*c)
OUT['truthfulqa_judge'] = dict(ci_q=qci(prs), per_model=J, pooled=row(*pool({m: [J[m][k] for k in ['ac', 'at', 'rc', 'rt']] for m in MODELS})),
                               bins={b: dict(row(*v[:4]), n_q=v[4]) for b, v in jb.items()})
# Frontier, exploratory
FR = {}
for name in ['sonnet', 'gemini']:
    s = [L(f'results/frontier_{name}_rescore_judge{x}.json')['summary'] for x in ['', '_seed43']]
    r = row(sum(x['ac'] for x in s), sum(x['n_adm'] for x in s), sum(x['rc'] for x in s), sum(x['n_rej'] for x in s))
    FR[name] = dict(combined=r, seeds=[row(x['ac'], x['n_adm'], x['rc'], x['n_rej']) for x in s])
bs = L('results/frontier_sonnet_base_subset_judge.json')['pooled_judge']
FR['base_seed42'] = row(bs['ac'], bs['n_adm'], bs['rc'], bs['n_rej'])
OUT['frontier'] = FR
# Instruction-tuned, post hoc
INS = {}
for name, tag, rs in [('Llama-Instruct', 'LlamaInstruct', 'rescore_LlamaInstruct_judge_v2'), ('Gemma-4-26B', 'Gemma4MoE', 'rescore_Gemma4MoE_judge_v2')]:
    c = [0, 0, 0, 0]
    for q, e in json.load(open(glob.glob(CF + f'modal_data/prereg_truthfulqa_{tag}_*.json')[0]))['per_question'].items(): add(c, e['accuracies'], set(e['nli_admissible']))
    k = [0, 0, 0, 0]; pq = L(f'results/{rs}.json')['per_question']
    for q, e in (pq.items() if isinstance(pq, dict) else enumerate(pq)): add(k, e['accuracies'], set(e['nli_admissible']))
    INS[name] = dict(substring=row(*c), judge=row(*k))
OUT['instruct'] = INS
# Multi-argmax strata
MA = {}
d = LT('multi_argmax_stratification_all_results.json')
for ds in d:
    MA[ds] = {s: dict(adm=row(x['ac'], x['at'], x['rc'], x['rt']), s075=row(x['s075_ac'], x['s075_at'], x['s075_rc'], x['s075_rt']), n_q=x['n_q']) for s, x in d[ds].items()}
d = L('multi_argmax_stratification_tqa_nq_results.json')
for ds in d: MA[ds] = {s: x for s, x in d[ds].items()}
OUT['multi_argmax'] = MA
# Stored post-hoc analyses, passed through
p = LT('partition_baselines_results.json'); OUT['single_theta'] = p['single_theta_frontier']; OUT['avg_cluster'] = p['averaged_cluster_fraction']
SF = LT('string_filters.json'); sc = SF['self_consistency']; OUT['string_filters'] = SF
OUT['self_consistency_partition'] = dict(row(*[sc['pooled'][k] for k in ['ac', 'at', 'rc', 'rt']]), per_model={m: sc[m]['ratio'] for m in MODELS})
OUT['matched_retention'] = LT('matched_retention_w1_results.json')
OUT['embedding_triviaqa'] = LT('embedding_similarity_ablation_results.json')['pooled']; OUT['embedding_truthfulqa'] = L('embedding_similarity_truthfulqa_results.json')['pooled']
OUT['logprob'] = LT('genprob_reweighting_results.json')['logprob']['pooled']
OUT['nli_agg'] = LT('nli_aggregation_ablation_results.json')['results']
OUT['scalar_triviaqa'] = LT('self_consistency_results.json')['metrics']
s = L('snne_headtohead_results.json'); OUT['scalar_arc'] = s['ARC']['signals']; OUT['scalar_mmlu'] = s['MMLU']['signals']
sw = LT('interval_endpoint_sweep_results.json')['datasets']
OUT['endpoint_sweep'] = {ds: dict(ratio=[min(c['ratio'] for c in v), max(c['ratio'] for c in v)], rej=[min(c['rejection_rate'] for c in v), max(c['rejection_rate'] for c in v)],
    ci_lo_min=min(c['ratio_ci_lo'] for c in v), auroc=[min(c['auroc_one_minus_maxcp'] for c in v), max(c['auroc_one_minus_maxcp'] for c in v)], n_cells=len(v)) for ds, v in sw.items()}
gs = collections.defaultdict(list)
for r in L('grid_sparsity_results.json'): gs[r['dataset']].append(r['num_distinct_Ostar'] == 1)
OUT['pct_single_argmax'] = {k: 100 * float(np.mean(v)) for k, v in gs.items()}
# Majority vote over all responses vs over the admissible set, TriviaQA (post-guard file)
rs = [r for r in LT('kle_snne_triviaqa_results.json') if r['dataset'] == 'triviaqa']
OUT['mv_triviaqa'] = dict(overall=float(np.mean([r['mv_correct'] for r in rs])), admissible=float(np.mean([r['adm_mv_correct'] for r in rs])), n=len(rs))
ret = [x.get('ratio', x.get('ratio_mean')) for x in LT('retention_policy_ablation_results.json')['results']]; OUT['retention_range'] = [min(ret), max(ret)]
gx = LT('goldilocks_external_proxy_results.json'); bins = [b for k in ['snne', 'se_075', 'mean_sim'] for b in gx[k].values() if b['rt'] > 0]
OUT['external_strat'] = dict(min=min(b['ratio'] for b in bins), max=max(b['ratio'] for b in bins), n_bins=len(bins))
for ds in ['mmlu', 'arc']:
    h = OUT[ds]['H3']; OUT[ds]['H3_rej_gap_pp'] = {k: 100 * abs(h['str']['rej_rate'] - h[k]['rej_rate']) for k in ['freq', 'cons']}
json.dump(OUT, open('cf_results.json', 'w'), indent=1)
print('CI_Q', {k: [round(c, 2) for c in OUT[k]['ci_q']] for k in ['triviaqa', 'mmlu', 'arc', 'truthfulqa', 'truthfulqa_judge', 'nq_open']})
print('MV', OUT['mv_triviaqa'], OUT['mmlu']['H3_rej_gap_pp'], OUT['arc']['H3_rej_gap_pp'])
def s(r): return f"{r['ratio']:.3f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}] adm {r['adm_acc']:.3f} rej {r['rej_acc']:.3f} nrej {r['rt']} rr {r['rej_rate']:.4f} p {r['p']:.2e}" if r['ratio'] else f"undef nrej {r['rt']} adm {r['adm_acc']}"
for ds in ['triviaqa', 'mmlu', 'arc', 'truthfulqa', 'truthfulqa_judge', 'nq_open']:
    print('##', ds, 'POOLED', s(OUT[ds]['pooled']))
    for m in MODELS: print('  ', m, s(OUT[ds]['per_model'][m]))
    for b in BINS: print('   bin', b, 'nQ', OUT[ds]['bins'][b]['n_q'], s(OUT[ds]['bins'][b]))
    if 'H3' in OUT[ds]: print('   H3', {k: (round(v['ratio'], 3), round(v['rej_rate'], 4)) for k, v in OUT[ds]['H3'].items()}, 'H4', OUT[ds]['H4'])
for k in ['sonnet', 'gemini']:
    print('##', k, s(FR[k]['combined'])); [print('   seed', s(x)) for x in FR[k]['seeds']]
print('## base seed42', s(FR['base_seed42']))
for k, v in INS.items(): print('##', k, 'substring', s(v['substring']), '| judge', s(v['judge']))
print('## sweep', OUT['endpoint_sweep']); print('## single', OUT['pct_single_argmax'])
