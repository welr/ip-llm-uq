"""Post hoc, 2026-11-07. Do the decision rules on the threshold credal set have to coincide?

With M equally weighted responses, enumerate every chain of nested clusterings (every family of
distinct set partitions of M responses that is totally ordered by refinement, which is what a
threshold grid can produce) and compare C_sem (union of pluralities over thresholds) with
maximality (no single response beats r at every threshold). Since C_sem <= E-admissibility <= Max
(Proposition on decision rules), C_sem == Max forces all three to coincide. E-admissibility depends
only on the set of distinct clusterings, since weights on repeated clusterings merge.

Writes m6_enumeration.json: counts for M = 3..6, and the first counterexample found for M = 7.
"""
import json, sys


def partitions(s):
    if not s:
        yield []
        return
    first, rest = s[0], s[1:]
    for p in partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def run(M, stop_at_first=False):
    parts = [tuple(sorted(tuple(sorted(b)) for b in p)) for p in partitions(list(range(M)))]
    lab = [[0] * M for _ in parts]
    for k, p in enumerate(parts):
        for bi, b in enumerate(p):
            for x in b:
                lab[k][x] = bi

    def coarsens(a, b):  # b is strictly coarser than a
        return a != b and all(lab[b][x] == lab[b][y] for blk in parts[a] for x in blk for y in blk)

    coarser = {a: [b for b in range(len(parts)) if coarsens(a, b)] for a in range(len(parts))}
    size = [[len(p[lab[k][x]]) for x in range(M)] for k, p in enumerate(parts)]
    out = {'M': M, 'partitions': len(parts), 'chains': 0, 'csem_ne_max': 0, 'first_counterexample': None}

    class Done(Exception):
        pass

    def check(chain):
        out['chains'] += 1
        U = [size[k] for k in chain]
        csem = {r for u in U for r in range(M) if u[r] == max(u)}
        mx = {r for r in range(M) if not any(all(u[s] > u[r] for u in U) for s in range(M))}
        if csem != mx:
            out['csem_ne_max'] += 1
            if out['first_counterexample'] is None:
                out['first_counterexample'] = {'clusterings_finest_first': [list(map(list, parts[k])) for k in chain],
                                               'C_sem': sorted(csem), 'Max': sorted(mx)}
                if stop_at_first:
                    raise Done

    def dfs(chain):
        check(chain)
        for b in coarser[chain[-1]]:
            dfs(chain + [b])

    try:
        for a in range(len(parts)):
            dfs([a])
    except Done:
        out['chains'] = None  # enumeration stopped early
    return out


if __name__ == '__main__':
    res = {f'M={m}': run(m) for m in (3, 4, 5, 6)}
    res['M=7'] = run(7, stop_at_first=True)
    for k, v in res.items():
        print(k, v['partitions'], 'partitions,', v['chains'], 'chains,', v['csem_ne_max'], 'with C_sem != Max')
    assert res['M=6']['csem_ne_max'] == 0 and res['M=7']['first_counterexample'] is not None
    json.dump(res, open('m6_enumeration.json', 'w'), indent=1)
