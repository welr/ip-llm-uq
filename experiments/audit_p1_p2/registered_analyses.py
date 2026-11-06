"""Secondary and exploratory analyses the registrations say will be reported, which no earlier script computed for these benchmarks.

All of it is run after the fact on stored responses, labels, partitions and similarity matrices. Writes registered_analyses.json.
Metric code is imported from the choice-function experiments so definitions match the TriviaQA and ARC-Challenge values in the manuscript; ARC-Challenge runs as the control.
TruthfulQA signals use cached matrices that reproduce 2964 of 3268 partitions, so they are indicative. Its stored partition still supplies the admissible and rejected groups.
The majority-vote label picks the same winner as the TriviaQA and ARC-Challenge pipeline and takes its accuracy from the stored registered labels; registered_analyses_control.py shows that pipeline's signals return the manuscript's AUROCs when joined to its labels.
The MMLU domain assignment below is ours, made now: the registration names the three domains and no mapping.
Dependencies already in the environment: numpy, scipy (BSD-3-Clause), scikit-learn (BSD-3-Clause), datasets (Apache-2.0).
"""
import collections, glob, json, os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score
C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', '')
sys.path.insert(0, C)
from kle_snne_comparison import kernel_language_entropy as KLE, kle_bandwidth_sweep as KBW, snne_entropy as SNNE, fixed_threshold_se as SE, _cluster, THRESHOLDS as TH, normalize_answer as NA
MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']; BW = [0.1, 0.2, 0.3, 0.5, 1.0]
STEM = set('abstract_algebra anatomy astronomy clinical_knowledge college_biology college_chemistry college_computer_science college_mathematics college_medicine college_physics computer_security conceptual_physics electrical_engineering elementary_mathematics high_school_biology high_school_chemistry high_school_computer_science high_school_mathematics high_school_physics high_school_statistics human_aging machine_learning medical_genetics nutrition professional_medicine virology'.split())
SOC = set('business_ethics econometrics global_facts high_school_geography high_school_government_and_politics high_school_macroeconomics high_school_microeconomics high_school_psychology human_sexuality international_law jurisprudence management marketing professional_accounting professional_law professional_psychology public_relations security_studies sociology us_foreign_policy'.split())
HUM = set('formal_logic high_school_european_history high_school_us_history high_school_world_history logical_fallacies miscellaneous moral_disputes moral_scenarios philosophy prehistory world_religions'.split())
def load(ds, m):
    c = [f for f in glob.glob(C + f'modal_data/prereg_{ds}_*.json') if f'_{m}_' in f or f.endswith(f'_{m}.json')]; return json.load(open(sorted(c)[-1]))['per_question']
def sims(ds, m):
    p = C + f'sim_cache/sim_matrices_{ds}_{m}.json'; return json.load(open(p)) if os.path.exists(p) else {}
def bin_label(n): return '0/6' if n == 0 else '1-2/6' if n <= 2 else '3-4/6' if n <= 4 else '5/6' if n == 5 else '6/6'
def signals(S):
    S = np.array(S); se = [SE(S, t) for t in TH]; cp = [max(len(c) for c in _cluster(S, t)) / S.shape[0] for t in TH]
    return dict(one_minus_maxcp=1 - max(cp), se_075=SE(S, .75), se_avg=float(np.mean(se)), se_width=max(se) - min(se), kle=float(KLE(S)), kle_avg=float(np.mean(list(KBW(S, BW).values()))), neg_snne=-float(SNNE(S)), n_distinct=len({frozenset().union(*[c for c in _cluster(S, t) if len(c) == max(len(x) for x in _cluster(S, t))]) for t in TH}))
def mv(texts, acc, keep=None):
    idx = sorted(keep) if keep is not None else list(range(len(texts)))
    if not idx: idx = list(range(len(texts)))
    n = [NA(texts[i]) for i in idx]; w = collections.Counter(n).most_common(1)[0][0]; return bool(acc[idx[n.index(w)]])
def ratio(ac, at, rc, rt): return (ac / at) / (rc / rt) if at and rt and rc else None
def group(rows):
    s = np.array(rows, float).sum(axis=0); ac, at, rc, rt = s; return dict(ratio=ratio(ac, at, rc, rt), rejected=int(rt), rejection_rate=float(rt / (at + rt)), adm_acc=float(ac / at), rej_acc=float(rc / rt) if rt else None, pairs=len(rows))
def boot(rows, B=10000, seed=42):
    rng = np.random.default_rng(seed); X = np.array(rows, float); s = X[rng.integers(0, len(X), (B, len(X)))].sum(axis=1); k = (s[:, 2] > 0) & (s[:, 3] > 0) & (s[:, 1] > 0); r = (s[k, 0] / s[k, 1]) / (s[k, 2] / s[k, 3]); return [float(x) for x in np.percentile(r, [2.5, 97.5])]
def counts(A, adm, n=6):
    return [sum(bool(A[i]) for i in adm), len(adm), sum(bool(A[i]) for i in range(n) if i not in adm), n - len(adm)]
OUT = {}
for ds in ['nq_open', 'truthfulqa', 'arc']:
    sig = []; mv_all = []; mv_adm = []
    for m in MODELS:
        SM = sims(ds, m); pq = load(ds, m); gp = C + f'modal_data/checkpoint_gen_{ds}_{m}.json'; G = json.load(open(gp)) if os.path.exists(gp) else {}
        for k, r in pq.items():
            X = r.get('top6_texts') or ([x['text'] for x in G[k]['responses'][:6]] if k in G else None)
            if X is not None: mv_all.append(mv(X, r['accuracies'])); mv_adm.append(mv(X, r['accuracies'], set(r['nli_admissible'])))
            if k not in SM: continue
            s = signals(SM[k]); s['mv_wrong'] = int(not mv(X, r['accuracies'])); s['stored_mv_wrong'] = int(not r['mv_correct']); s['bin'] = bin_label(sum(bool(a) for a in r['accuracies'])); sig.append(s)
    y = [s['mv_wrong'] for s in sig]; y2 = [s['stored_mv_wrong'] for s in sig]
    o = dict(n=len(sig), auroc={k: float(roc_auc_score(y, [s[k] for s in sig])) for k in ['one_minus_maxcp', 'se_075', 'se_avg', 'kle', 'kle_avg', 'neg_snne']}, auroc_runner_label={k: float(roc_auc_score(y2, [s[k] for s in sig])) for k in ['one_minus_maxcp', 'se_075', 'kle', 'neg_snne']}, mv_overall=float(np.mean(mv_all)), mv_admissible=float(np.mean(mv_adm)), mv_n=len(mv_all))
    o['se_width_detects_1_2'] = float(roc_auc_score([int(s['bin'] == '1-2/6') for s in sig], [s['se_width'] for s in sig]))
    nd = [s['n_distinct'] for s in sig]; o['grid'] = dict(mean_distinct=float(np.mean(nd)), share_one=float(np.mean(np.array(nd) == 1)), maximum=int(max(nd)))
    OUT[ds] = o
# TruthfulQA strata and overlap with the benchmark's incorrect answers.
from datasets import load_dataset
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
tq = load_dataset('truthful_qa', 'generation', split='validation')
by_q = {r['question']: r for r in tq}
def norm(t):
    t = re.sub(r'[^\w\s]', '', (t or '').lower()); return re.sub(r'\s+', ' ', t).strip()[:100]
def hit(t, L):
    g = norm(t)
    return bool(g) and any(norm(x) and (norm(x) in g or g in norm(x)) for x in L)
S = collections.defaultdict(list); ov = collections.Counter(); idx_miss = 0
for m in MODELS:
    pq = load('truthfulqa', m)
    for k, r in pq.items():
        A = r['accuracies']; adm = set(r['nli_admissible']); c = counts(A, adm)
        S['type', r['tqa_type']].append(c); S['category', r['tqa_category']].append(c)
        i = int(re.search(r'(\d+)$', k).group(1)); q = tq[i] if i < len(tq) else None
        if q is None or q.get('category') != r['tqa_category']: idx_miss += 1; continue
        for j in range(6):
            g = 'adm' if j in adm else 'rej'; ov[g + '_n'] += 1; ov[g + '_incorrect'] += hit(r['top6_texts'][j], q['incorrect_answers'])
OUT['truthfulqa']['strata'] = {f'{a}:{b}': dict(group(v), ci_pairs=boot(v) if group(v)['ratio'] else None) for (a, b), v in S.items() if a == 'type'}
cat = {b: group(v) for (a, b), v in S.items() if a == 'category'}; big = {b: v for b, v in cat.items() if v['ratio'] and v['rejected'] >= 30}
OUT['truthfulqa']['categories'] = dict(n=len(cat), with_30_rejected=len(big), low=min(v['ratio'] for v in big.values()), high=max(v['ratio'] for v in big.values()), below_1=sum(v['ratio'] < 1 for v in big.values()), per_category=cat)
OUT['truthfulqa']['overlap_incorrect'] = dict(rejected=ov['rej_incorrect'] / ov['rej_n'], admissible=ov['adm_incorrect'] / ov['adm_n'], rejected_n=ov['rej_n'], admissible_n=ov['adm_n'], pairs_unmatched=idx_miss)
# MMLU domains, Jaccard overlap with the heuristic filters, and range width.
dom = collections.defaultdict(list); J = {}
for ds in ['mmlu', 'arc']:
    jac = collections.defaultdict(lambda: [0, 0])
    for m in MODELS:
        for k, r in load(ds, m).items():
            adm = set(r['nli_admissible']); rej = set(r['nli_rejected'])
            if ds == 'mmlu':
                s = r['subject']; d = 'STEM' if s in STEM else 'Social sciences' if s in SOC else 'Humanities' if s in HUM else None
                if d is None: raise SystemExit('unassigned subject ' + s)
                dom[d].append(counts(r['accuracies'], adm))
            for f in ['freq_rejected', 'cons_rejected', 'len_rejected']:
                b = set(r[f]); jac[f][0] += len(rej & b); jac[f][1] += len(rej | b)
    J[ds] = {f: v[0] / v[1] for f, v in jac.items()}
OUT['mmlu_domains'] = {d: dict(group(v), ci_pairs=boot(v)) for d, v in dom.items()}; OUT['jaccard'] = J
E0 = json.load(open(C + 'interval_endpoint_sweep_results.json')); mins, maxs, ng = E0['config']['theta_mins'], E0['config']['theta_maxs'], E0['config']['n_grid']
def sweep(ds, models):
    res = []
    D = [(load(ds, m), sims(ds, m)) for m in models]
    for a in mins:
        for b in maxs:
            if a >= b: continue
            T = np.linspace(a, b, ng); tot = np.zeros(4)
            for pq, SM in D:
                for k, r in pq.items():
                    if k not in SM: continue
                    M = np.array(SM[k]); adm = set()
                    for t in T:
                        cl = _cluster(M, t); q = max(len(c) for c in cl)
                        for c in cl:
                            if len(c) == q: adm |= c
                    tot += counts(r['accuracies'], adm)
            res.append(dict(theta_min=a, theta_max=b, ratio=ratio(*tot), rejection_rate=float(tot[3] / (tot[1] + tot[3]))))
    v = [x for x in res if x['ratio']]
    return dict(pairs=len(res), ratio_low=min(x['ratio'] for x in v), ratio_high=max(x['ratio'] for x in v), rej_low=min(x['rejection_rate'] for x in res), rej_high=max(x['rejection_rate'] for x in res), below_1=sum(x['ratio'] < 1 for x in v), detail=res)
OUT['range_width'] = dict(nq_open_control=sweep('nq_open', MODELS), arc=sweep('arc', MODELS), mmlu_gemma=sweep('mmlu', ['Gemma']))
d = json.load(open(C + 'kle_snne_results.json')); arc = [r for r in d if r['dataset'] == 'arc']; gm = [r for r in d if r['dataset'] == 'mmlu']
OUT['combined_stored'] = dict(arc=dict(mv=float(np.mean([r['mv_correct'] for r in arc])), adm=float(np.mean([r['adm_mv_correct'] for r in arc])), n=len(arc)), mmlu_gemma=dict(mv=float(np.mean([r['mv_correct'] for r in gm])), adm=float(np.mean([r['adm_mv_correct'] for r in gm])), n=len(gm)))
json.dump(OUT, open('registered_analyses.json', 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
for ds in ['arc', 'nq_open', 'truthfulqa']:
    o = OUT[ds]; print(ds, 'n', o['n'], {k: round(v, 3) for k, v in o['auroc'].items()}, 'MV', round(o['mv_overall'], 4), round(o['mv_admissible'], 4), 'SE width 1-2/6', round(o['se_width_detects_1_2'], 3), 'grid', {k: round(v, 3) for k, v in o['grid'].items()})
print('runner label', {d: {k: round(v, 3) for k, v in OUT[d]['auroc_runner_label'].items()} for d in ['nq_open', 'truthfulqa']})
print('strata', {k: (round(v['ratio'], 2), [round(x, 2) for x in v['ci_pairs']], v['rejected']) for k, v in OUT['truthfulqa']['strata'].items()})
print('categories', {k: v for k, v in OUT['truthfulqa']['categories'].items() if k != 'per_category'})
print('overlap', OUT['truthfulqa']['overlap_incorrect'])
print('domains', {k: (round(v['ratio'], 2), [round(x, 2) for x in v['ci_pairs']], v['rejected'], v['pairs']) for k, v in OUT['mmlu_domains'].items()})
print('jaccard', {k: {f: round(x, 3) for f, x in v.items()} for k, v in J.items()})
print('range', {k: {f: (round(x, 4) if isinstance(x, float) else x) for f, x in v.items() if f != 'detail'} for k, v in OUT['range_width'].items()})
print('combined', OUT['combined_stored'])
