"""Checks of the examples in Appendix A and Section 4.2. Exact arithmetic with Fractions."""
from fractions import Fraction as Fr
import math, collections, itertools
def comps(names, s, t):
    p = {n: n for n in names}
    def f(x):
        while p[x] != x: x = p[x]
        return x
    for a, b in itertools.combinations(names, 2):
        if s.get((a, b), s.get((b, a), s.get('other'))) >= t: p[f(a)] = f(b)
    g = collections.defaultdict(list)
    for n in names: g[f(n)].append(n)
    return list(g.values())
def score(names, w, s, Th):
    return {n: tuple(sum(w[m] for m in next(c for c in comps(names, s, t) if n in c)) for t in Th) for n in names}
def report(tag, names, w, s, Th):
    U = score(names, w, s, Th); K = range(len(Th))
    C = {n for n in names if any(U[n][k] == max(U[m][k] for m in names) for k in K)}
    Max = {n for n in names if not any(all(U[m][k] > U[n][k] for k in K) for m in names)}
    Par = {n for n in names if not any(all(U[m][k] >= U[n][k] for k in K) and U[m] != U[n] for m in names)}
    print(tag, 'scores', {n: tuple(map(float, v)) for n, v in U.items()}); print('  C_sem', sorted(C), 'Max', sorted(Max), 'Pareto', sorted(Par))
    if len(Th) == 2:
        EA = set()
        for n in names:
            lo, hi = Fr(0), Fr(1); ok = True
            for m in names:
                # lam*U[n][0]+(1-lam)*U[n][1] >= same for m
                a = (U[n][0] - U[n][1]) - (U[m][0] - U[m][1]); b = U[n][1] - U[m][1]
                if a == 0:
                    ok = ok and b >= 0
                elif a > 0: lo = max(lo, -b / a)
                else: hi = min(hi, -b / a)
            if ok and lo <= hi: EA.add(n); print('   ', n, 'optimal for lambda in', [str(lo), str(hi)])
        print('  EA', sorted(EA))
w1 = dict(zip('axbrw', map(Fr, ['.2', '.2', '.25', '.22', '.13'])))
s1 = {('a', 'x'): .6, ('r', 'w'): .6, 'other': .1}
report('W1', list('axbrw'), w1, s1, [.5, .9])
w2 = dict(zip('axbrwz', map(Fr, ['.2', '.2', '.25', '.22', '.08', '.05'])))
s2 = {('a', 'x'): .6, ('r', 'w'): .6, 'other': .1}
report('W2', list('axbrwz'), w2, s2, [.5, .9])
freq = dict(A=Fr(3, 10), B=Fr(2, 10), C=Fr(2, 10), D=Fr(2, 10), E=Fr(1, 10))
s3 = {('A', 'B'): .92, ('B', 'C'): .83, ('A', 'C'): .71, ('D', 'E'): .88, 'other': .1}
report('Worked', list('ABCDE'), freq, s3, [.70, .80, .85, .90])
# SE as expectation of the surprisal gamble
for t in [.70, .85, .90]:
    cs = comps(list('ABCDE'), s3, t); P = {n: float(sum(freq[m] for m in next(c for c in cs if n in c))) for n in 'ABCDE'}
    print('theta', t, 'E[-log2 u]', round(sum(float(freq[n]) * -math.log2(P[n]) for n in 'ABCDE'), 3), 'SE', round(-sum(float(sum(freq[m] for m in c)) * math.log2(float(sum(freq[m] for m in c))) for c in cs), 3))
