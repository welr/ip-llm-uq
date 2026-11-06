"""Recompute the 'What causes the seasons?' example from the scores stated in Section 3."""
import math, itertools
freq = dict(A=3, B=2, C=2, D=2, E=1); M = sum(freq.values())
s = {('A','B'):.92, ('B','C'):.83, ('A','C'):.71, ('D','E'):.88}
def clusters(t):
    p = {k:k for k in freq}
    def f(x):
        while p[x] != x: x = p[x]
        return x
    for (a,b),v in s.items():
        if v >= t: p[f(a)] = f(b)
    g = {}
    for k in freq: g.setdefault(f(k), []).append(k)
    return sorted(''.join(sorted(v)) for v in g.values())
adm = set()
for t in [.70,.75,.80,.85,.90,.95]:
    c = clusters(t); P = [sum(freq[x] for x in k)/M for k in c]; mx = max(P)
    for k,q in zip(c,P):
        if q == mx: adm |= set(k)
    print(f"theta={t:.2f} clusters={c} n={len(c)} maxP={mx:.2f} SE_nats={-sum(q*math.log(q) for q in P):.3f} SE_bits={-sum(q*math.log2(q) for q in P):.3f}")
print('admissible', sorted(adm))
for r in 'ABCDE':
    u = [sum(freq[x] for x in next(k for k in clusters(t) if r in k)) / M for t in [.70, .90]]
    print('interval over .70-.90', r, [min(u), max(u)])
