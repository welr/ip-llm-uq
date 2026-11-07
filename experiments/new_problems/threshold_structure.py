"""Post hoc, 2026-11-07. How much is there for the threshold sweep to sweep, and why does SE width
fall with difficulty on the multiple-choice benchmarks?

Reads the cached similarity matrices in ../stored_runs/sim_cache (six retained responses per
question-model pair; mean of the two directed entailment probabilities). The cached matrices
reproduce the registered partitions on TriviaQA and NQ-Open, on 1,977 of 2,000 ARC-Challenge pairs
and 2,964 of 3,268 TruthfulQA pairs; on MMLU only Gemma's are stored (499 of 500 reproduce).
For each benchmark:

- share of response pairs whose score lies in [0.55, 0.85), the interior of the registered range,
  and in [0.49, 0.50) and [0.50, 0.51), around its lower end;
- share of question-model pairs for which every threshold in the registered grid {0.50, ..., 0.95}
  gives the same clustering;
- share of question-model pairs in which every response is a singleton at theta = 0.95;
- share of response pairs whose score is at least 0.95, the strict end of the grid (added
  2026-11-07 to support the statement that even matching responses seldom reach it);
- Spearman correlation, over question-model pairs, between SE width, SE(0.95) - SE(0.50), and
  SE(0.50). Where every response is a singleton at 0.95, width is log 6 minus SE(0.50).

Writes threshold_structure.json.
"""
import glob, json, os
import numpy as np

GRID = [round(0.50 + 0.05 * i, 2) for i in range(10)]
CACHE = os.path.join('..', 'stored_runs', 'sim_cache')


def clusters(S, th):
    n = len(S)
    lab = list(range(n))

    def find(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]
            x = lab[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if S[i][j] >= th:
                lab[find(i)] = find(j)
    seen = {}
    return tuple(seen.setdefault(find(i), len(seen)) for i in range(n))


def entropy(c):
    _, cnt = np.unique(c, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum())


def ranks(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind='mergesort')
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, r)
    return (s / cnt)[inv]


out = {}
for bench in ['triviaqa', 'nq_open', 'truthfulqa', 'arc', 'mmlu']:
    scores, n_pairs, one_clustering, all_singletons, width, se_min = [], 0, 0, 0, [], []
    files = sorted(glob.glob(os.path.join(CACHE, f'sim_matrices_{bench}_*.json')))
    for f in files:
        for S in json.load(open(f)).values():
            S = np.array(S)
            iu = np.triu_indices(len(S), 1)
            scores.extend(S[iu].tolist())
            n_pairs += 1
            cl = [clusters(S, t) for t in GRID]
            one_clustering += len(set(cl)) == 1
            all_singletons += len(set(cl[-1])) == len(S)
            width.append(entropy(cl[-1]) - entropy(cl[0]))
            se_min.append(entropy(cl[0]))
    p = np.array(scores)
    out[bench] = {
        'models': [os.path.basename(f).rsplit('_', 1)[1][:-5] for f in files],
        'question_model_pairs': n_pairs,
        'share_scores_055_085': float(np.mean((p >= .55) & (p < .85))),
        'share_scores_049_050': float(np.mean((p >= .49) & (p < .50))),
        'share_scores_050_051': float(np.mean((p >= .50) & (p < .51))),
        'share_scores_ge_095': float(np.mean(p >= .95)),
        'share_one_clustering_over_grid': one_clustering / n_pairs,
        'share_all_singletons_at_095': all_singletons / n_pairs,
        'spearman_width_vs_se_min': float(np.corrcoef(ranks(width), ranks(se_min))[0, 1]),
    }
    print(bench, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[bench].items()})
json.dump(out, open('threshold_structure.json', 'w'), indent=1)
