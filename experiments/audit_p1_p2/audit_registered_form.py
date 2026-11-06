"""What the filed MMLU/ARC form (osf.io/ady3k) states and its attached preregistration.md does not: two exploratory analyses and an exclusion rule.

Best-of-N is undefined in the form, the attached file and every stored script. The reading here is ours, made now: a question-model pair counts if at least one of its responses is correct, over all six and over the admissible ones.
Difficulty follows H4: one minus accuracy over all models' six responses to the question.
Stored response text is cut before storage, so empties that the model produced cannot be separated from empties the cut produced. MMLU samples exist for Gemma only.
Labels are the stored registered labels. All of it runs after the fact on stored data. Writes audit_registered_form.json. Run from this folder.

The TruthfulQA and NQ-Open forms (osf.io/dqbmg, osf.io/78pb6) each state two rules for empty responses (kept and noted; re-sampled once, then the pair dropped) and two bootstrap units (responses; questions). The runs kept every pair and never re-sampled, and re-sampling cannot be reproduced from stored text, so only pair dropping and question resampling are tested here. The forms also list string admissibility and the heuristic filters as measured variables, SE width among the AUROC signals, and, on NQ-Open, overlap with substring-incorrect responses.
"""
import collections, glob, json, os
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_data', '')
MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']; B = 10000
def load(ds, m): return json.load(open(sorted(f for f in glob.glob(C + f'prereg_{ds}_*.json') if f'_{m}_' in f or f.endswith(f'_{m}.json'))[-1]))['per_question']
def ratio(ac, at, rc, rt): return (ac / at) / (rc / rt)
def fisher(t): return float(stats.fisher_exact([[t[0], t[1] - t[0]], [t[2], t[3] - t[2]]])[1])
def tally(pq, keep=None):
    ac = at = rc = rt = 0
    for k, r in pq.items():
        if keep is not None and k not in keep: continue
        A = r['accuracies']; ac += sum(A[i] for i in r['nli_admissible']); at += len(r['nli_admissible']); rc += sum(A[i] for i in r['nli_rejected']); rt += len(r['nli_rejected'])
    return ac, at, rc, rt
OUT = {}
for ds in ['mmlu', 'arc']:
    P = {m: load(ds, m) for m in MODELS}; rate = collections.defaultdict(list); acc = collections.defaultdict(list); any_all = any_adm = pairs = 0
    for m in MODELS:
        for k, r in P[m].items():
            A = r['accuracies']; adm = r['nli_admissible']; pairs += 1; any_all += any(A); any_adm += any(A[i] for i in adm); rate[k].append(len(adm) / len(A)); acc[k] += [float(a) for a in A]
    keys = sorted(rate); x = np.array([np.mean(rate[k]) for k in keys]); y = np.array([1 - np.mean(acc[k]) for k in keys]); rho = stats.spearmanr(x, y); rng = np.random.default_rng(42); bs = []
    for _ in range(B):
        i = rng.integers(0, len(keys), len(keys))
        if np.ptp(x[i]) and np.ptp(y[i]): bs.append(stats.spearmanr(x[i], y[i])[0])
    OUT[ds] = dict(pairs=pairs, best_of_six_all=any_all / pairs, best_of_six_admissible=any_adm / pairs, questions=len(keys), rate_vs_difficulty=dict(rho=float(rho[0]), p=float(rho[1]), ci_questions=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], mean_rate=float(x.mean())))
    E = {}; base = [0, 0, 0, 0]; kept = [0, 0, 0, 0]; dropped = empty = checked = 0
    for m in MODELS:
        gp = C + f'checkpoint_gen_{ds}_{m}.json'
        if not os.path.exists(gp): continue
        G = json.load(open(gp)); keep = set(); e = 0
        for k, v in G.items():
            n = sum(x_['text'].strip() == '' for x_ in v['responses'][:6]); e += n
            if n == 0: keep.add(k)
        t = tally(P[m]); u = tally(P[m], keep); base = [a + b for a, b in zip(base, t)]; kept = [a + b for a, b in zip(kept, u)]; empty += e; dropped += len(G) - len(keep); checked += len(G); E[m] = dict(empty_in_six=e, pairs_with_empty=len(G) - len(keep), ratio_as_run=ratio(*t), ratio_pairs_dropped=ratio(*u))
    OUT[ds]['empty_rule'] = dict(models_with_samples=list(E), pairs_checked=checked, per_model=E, empty_in_six=empty, pairs_with_empty=dropped, ratio_as_run=ratio(*base), ratio_pairs_dropped=ratio(*kept), pooled_p_as_run=fisher(base), pooled_p_dropped=fisher(kept), models_above_1_as_run=sum(v['ratio_as_run'] > 1 for v in E.values()), models_above_1_dropped=sum(v['ratio_pairs_dropped'] > 1 for v in E.values()))
def response_ci(t):  # within-group resampling, the method behind the reported intervals
    rng = np.random.default_rng(42); a = rng.binomial(t[1], t[0] / t[1], B) / t[1]; r = rng.binomial(t[3], t[2] / t[3], B) / t[3]; ok = r > 0
    return [float(v) for v in np.percentile(a[ok] / r[ok], [2.5, 97.5])]
def question_ci(P, match):
    keys = sorted({k for m in P for k in P[m]}); ix = {k: i for i, k in enumerate(keys)}; rows = [(ix[k], *tally({k: x})) for m in P for k, x in P[m].items() if match(x)]
    q = np.array([r[0] for r in rows]); d = np.array([r[1:] for r in rows], float); rng = np.random.default_rng(42); out = []
    for _ in range(B):
        s = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))[q] @ d
        if s[1] and s[3] and s[2]: out.append(ratio(*s))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
def gap_ci(P):  # string minus NLI ratio; both sides of one draw share the resampled questions
    def t(x, lab): A = x['accuracies']; R = set(x[lab]); return [sum(A[i] for i in range(len(A)) if i not in R), len(A) - len(R), sum(A[i] for i in R), len(R)]
    keys = sorted({k for m in P for k in P[m]}); ix = {k: i for i, k in enumerate(keys)}; rows = [(ix[k], *t(x, 'str_rejected'), *t(x, 'nli_rejected')) for m in P for k, x in P[m].items()]
    q = np.array([r[0] for r in rows]); d = np.array([r[1:] for r in rows], float); rng = np.random.default_rng(42); out = []
    for _ in range(B):
        s = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))[q] @ d
        if s[2] and s[3] and s[6] and s[7]: out.append(ratio(*s[:4]) - ratio(*s[4:]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
BINS = dict(pooled=lambda x: True, bin_1_2=lambda x: sum(x['accuracies']) in (1, 2), bin_3_4=lambda x: sum(x['accuracies']) in (3, 4))
def summed(P, match, keep=None):
    T = [0, 0, 0, 0]
    for m in P:
        for k, x in P[m].items():
            if match(x) and (keep is None or k in keep[m]): T = [a + b for a, b in zip(T, tally({k: x}))]
    return T
def outcomes(pooled, above, sig, low34, low12, h2): return dict(H1=ratio(*pooled) > 1 and fisher(pooled) < .05, H2=ratio(*pooled) >= h2, H3=above >= 3 and sig >= 2, H4=low34 >= 2, H5=low12 >= .5, H6=ratio(*pooled) >= 1.71)
for ds, h2 in [('truthfulqa', 1.5), ('nq_open', 2.0)]:
    P = {m: load(ds, m) for m in MODELS}; keep = {m: {k for k, x in P[m].items() if not any(t.strip() == '' for t in x['top6_texts'])} for m in MODELS}
    def per_model(kp): return [tally(P[m], kp[m] if kp else None) for m in MODELS]
    def count(kp): T = per_model(kp); return sum(ratio(*t) > 1 for t in T), sum(ratio(*t) > 1 and fisher(t) < .05 for t in T)
    run = {b: summed(P, f) for b, f in BINS.items()}; drop = {b: summed(P, f, keep) for b, f in BINS.items()}; qci = {b: question_ci(P, f) for b, f in BINS.items()}
    ar = dict(as_run={b: dict(ratio=ratio(*run[b]), ci_responses=response_ci(run[b])) for b in BINS}, pairs_dropped={b: dict(ratio=ratio(*drop[b]), ci_responses=response_ci(drop[b])) for b in BINS})
    O = dict(registered=outcomes(run['pooled'], *count(None), ar['as_run']['bin_3_4']['ci_responses'][0], ar['as_run']['bin_1_2']['ci_responses'][0], h2), pairs_dropped=outcomes(drop['pooled'], *count(keep), ar['pairs_dropped']['bin_3_4']['ci_responses'][0], ar['pairs_dropped']['bin_1_2']['ci_responses'][0], h2), resample_questions=outcomes(run['pooled'], *count(None), qci['bin_3_4'][0], qci['bin_1_2'][0], h2))
    M = {}
    for lab in ['nli_rejected', 'str_rejected', 'freq_rejected', 'cons_rejected', 'len_rejected']:
        ac = at = rc = rt = 0
        for m in MODELS:
            for x in P[m].values():
                A = x['accuracies']; R = set(x[lab]); rc += sum(A[i] for i in R); rt += len(R); ac += sum(A[i] for i in range(len(A)) if i not in R); at += len(A) - len(R)
        M[lab] = dict(ratio=ratio(ac, at, rc, rt), rejected=rt, rejection_rate=rt / (at + rt), rejected_accuracy=rc / rt, kept_accuracy=ac / at)
    y = [int(not x['mv_correct']) for m in MODELS for x in P[m].values()]; w = [x['se_upper'] - x['se_lower'] for m in MODELS for x in P[m].values()]
    OUT[ds] = dict(pairs=sum(len(P[m]) for m in MODELS), pairs_with_empty=sum(len(P[m]) - len(keep[m]) for m in MODELS), per_model_pairs_with_empty={m: len(P[m]) - len(keep[m]) for m in MODELS}, intervals=ar, ci_resampling_questions=qci, outcomes=O, measured=M, str_minus_nli_ci_questions=gap_ci(P), se_width_auroc_majority_vote_error=float(roc_auc_score(y, w)))
json.dump(OUT, open('audit_registered_form.json', 'w'), indent=1); print(json.dumps(OUT, indent=1))
