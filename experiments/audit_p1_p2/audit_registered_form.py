"""What the filed MMLU/ARC form (osf.io/ady3k) states and its attached preregistration.md does not: two exploratory analyses and an exclusion rule.

Best-of-N is undefined in the form, the attached file and every stored script. The reading here is ours, made now: a question-model pair counts if at least one of its responses is correct, over all six and over the admissible ones.
Difficulty follows H4: one minus accuracy over all models' six responses to the question.
Stored response text is cut before storage, so empties that the model produced cannot be separated from empties the cut produced. MMLU samples exist for Gemma only.
Labels are the stored registered labels. All of it runs after the fact on stored data. Writes audit_registered_form.json. Run from this folder.
"""
import collections, glob, json, os
import numpy as np
from scipy import stats
C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_data', '')
MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']; B = 10000
def load(ds, m): return json.load(open(sorted(f for f in glob.glob(C + f'prereg_{ds}_*.json') if f'_{m}_' in f or f.endswith(f'_{m}.json'))[-1]))['per_question']
def ratio(ac, at, rc, rt): return (ac / at) / (rc / rt)
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
        t = tally(P[m]); u = tally(P[m], keep); base = [a + b for a, b in zip(base, t)]; kept = [a + b for a, b in zip(kept, u)]; empty += e; dropped += len(G) - len(keep); checked += len(G); E[m] = dict(empty_in_six=e, pairs_with_empty=len(G) - len(keep))
    OUT[ds]['empty_rule'] = dict(models_with_samples=list(E), pairs_checked=checked, per_model=E, empty_in_six=empty, pairs_with_empty=dropped, ratio_as_run=ratio(*base), ratio_pairs_dropped=ratio(*kept))
json.dump(OUT, open('audit_registered_form.json', 'w'), indent=1); print(json.dumps(OUT, indent=1))
