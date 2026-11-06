"""Prediction 2: does the ECE result survive once stray parses and the mechanical part are separated out? Reads stored results only; writes audit_p2.json."""
import collections, json, re
import numpy as np
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
FILES = dict(Llama='pred2_local_llama70b_20260407_161230.json', Gemma='pred2_gemma_20260111_213527.json', Mistral='pred2_mistral_gpu_20260111_223736.json')
def ece(c, y, n=10):
    c = np.asarray(c, float); y = np.asarray(y, float); e = 0.0
    if not len(c): return float('nan')
    for i in range(n):
        lo, hi = i / n, (i + 1) / n; m = (c >= lo) & (c < hi) if hi < 1 else (c >= lo) & (c <= hi)
        if m.sum(): e += m.mean() * abs(y[m].mean() - c[m].mean())
    return float(e)
def auc(c, y):
    c = np.asarray(c, float); y = np.asarray(y, int); p, n = c[y == 1], c[y == 0]
    if not len(p) or not len(n): return float('nan')
    return float(np.mean((p[:, None] > n[None, :]) + .5 * (p[:, None] == n[None, :])))
def desc(S):
    c = [r['c_true'] for r in S]; y = [int(r['is_true']) for r in S]; rn = [r['c_true'] / r['confidence_sum'] if r['confidence_sum'] > 0 else .5 for r in S]
    return dict(n=len(S), ece=ece(c, y), mean_c_true=float(np.mean(c)) if S else None, share_true=float(np.mean(y)) if S else None, auroc=auc(c, y), ece_renorm=ece(rn, y), auroc_renorm=auc(rn, y), brier_renorm=float(np.mean((np.array(rn) - np.array(y)) ** 2)) if S else None, const_half_ece=abs(float(np.mean(y)) - .5) if S else None)
def boot_ratio(C, I, B=5000, seed=42):
    rng = np.random.default_rng(seed); rs = []
    cc, cy = np.array([r['c_true'] for r in C]), np.array([int(r['is_true']) for r in C]); ic, iy = np.array([r['c_true'] for r in I]), np.array([int(r['is_true']) for r in I])
    if len(C) < 2 or len(I) < 2: return [None, None]
    for _ in range(B):
        a = rng.integers(0, len(C), len(C)); b = rng.integers(0, len(I), len(I)); e = ece(cc[a], cy[a])
        if e > 0: rs.append(ece(ic[b], iy[b]) / e)
    return [float(np.quantile(rs, .025)), float(np.quantile(rs, .975))]
CONF = re.compile(r'(confiden|certain|sure|rate|likelihood|probabil)', re.I)
def locate(resp, val):
    """Trace the parser on the stored text. Returns stated, stray, or unknown when the value came from past the stored 200 characters."""
    t = resp.lower().strip(); m = None; v = None
    x = re.search(r'(\d+(?:\.\d+)?)\s*%', t)
    if x: v = float(x.group(1)); v = v / 100 if v > 1 else v; m = x
    else:
        x = re.search(r'(?:^|[^\d])0\.(\d+)', t)
        if x: v = float(f'0.{x.group(1)}'); m = x
        else:
            x = re.search(r'(?:^|[\s:])(\d+(?:\.\d+)?)\b', t)
            if x:
                w = float(x.group(1))
                if 0 <= w <= 1: v, m = w, x
                elif 1 < w <= 100: v, m = w / 100, x
    if m is None or abs(v - val) > 1e-9: return 'unknown'
    if len(t) <= 8 or m.start() <= 3: return 'stated'
    return 'stated' if CONF.search(t[max(0, m.start() - 60):m.end() + 40]) else 'stray'
OUT = {}
for name, f in FILES.items():
    rs = list(json.load(open(R + f))['results'].values()); V = [r for r in rs if r['is_coherent'] is not None]; C = [r for r in V if r['is_coherent']]; I = [r for r in V if not r['is_coherent']]
    both1 = [r for r in I if r['c_true'] == 1 and r['c_false'] == 1]; I_rest = [r for r in I if not (r['c_true'] == 1 and r['c_false'] == 1)]
    o = dict(total=len(rs), valid=len(V), coherent=desc(C), incoherent=desc(I), ratio=ece([r['c_true'] for r in I], [int(r['is_true']) for r in I]) / ece([r['c_true'] for r in C], [int(r['is_true']) for r in C]), ratio_ci=boot_ratio(C, I), incoherent_both_one=len(both1), incoherent_excl_both_one=desc(I_rest))
    o['ratio_excl_both_one'] = o['incoherent_excl_both_one']['ece'] / o['coherent']['ece'] if I_rest else None
    o['ratio_excl_both_one_ci'] = boot_ratio(C, I_rest)
    o['extreme_share_c_true'] = float(np.mean([r['c_true'] in (0, 1) for r in V]))
    o['sum_exact_one'] = sum(abs(r['confidence_sum'] - 1) < 1e-9 for r in V)
    S = collections.defaultdict(lambda: [0, 0])
    for r in V: S[r['subset']][0 if r['is_coherent'] else 1] += 1
    o['pct_coherent_by_subset'] = {k: round(v[0] / sum(v), 3) for k, v in sorted(S.items())}
    if 'response_true' in rs[0]:
        for r in V: r['_t'] = locate(r['response_true'], r['c_true']); r['_f'] = locate(r['response_false'], r['c_false'])
        kinds = collections.Counter(); 
        for r in V: kinds[('coh' if r['is_coherent'] else 'inc', 'both_stated' if r['_t'] == r['_f'] == 'stated' else 'any_stray' if 'stray' in (r['_t'], r['_f']) else 'unknown')] += 1
        o['pair_kinds'] = {'|'.join(k): v for k, v in sorted(kinds.items())}
        CS = [r for r in C if r['_t'] == r['_f'] == 'stated']; IS = [r for r in I if r['_t'] == r['_f'] == 'stated']
        o['both_stated'] = dict(coherent=desc(CS), incoherent=desc(IS), ratio=desc(IS)['ece'] / desc(CS)['ece'], ratio_ci=boot_ratio(CS, IS), incoherent_both_one=sum(r['c_true'] == 1 and r['c_false'] == 1 for r in IS))
        IS2 = [r for r in IS if not (r['c_true'] == 1 and r['c_false'] == 1)]; o['both_stated']['incoherent_excl_both_one'] = desc(IS2)
        ack = re.compile(r'statement is (indeed |actually |in fact )?(true|false)|(indeed|actually) (true|false)', re.I)
        o['stated_verdict_follows_prompt'] = dict(true_prompt_says_true=sum(bool(re.search(r'(is|indeed|answer is)\W+(indeed\W+)?true', r['response_true'], re.I)) for r in rs), true_prompt_says_false=sum(bool(re.search(r'(is|actually)\W+(actually\W+)?false', r['response_true'], re.I)) for r in rs), false_prompt_says_false=sum(bool(re.search(r'(is|indeed|actually)\W+(indeed\W+|actually\W+)?false', r['response_false'], re.I)) for r in rs), false_prompt_says_true=sum(bool(re.search(r'(is|actually)\W+(actually\W+)?true', r['response_false'], re.I)) for r in rs))
    OUT[name] = o
json.dump(OUT, open('audit_p2.json', 'w'), indent=1)
def show(d): return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}
for k, o in OUT.items():
    print('=====', k, 'total', o['total'], 'valid', o['valid'], 'ratio', round(o['ratio'], 2), 'CI', [round(x, 2) for x in o['ratio_ci']], '| c_true extreme share', round(o['extreme_share_c_true'], 3), '| sum exactly 1:', o['sum_exact_one'])
    print(' coherent ', show(o['coherent'])); print(' incoherent', show(o['incoherent'])); print(' both=1 among incoherent', o['incoherent_both_one'], '| rest', show(o['incoherent_excl_both_one'])); print(' ratio excl both=1', o['ratio_excl_both_one'] and round(o['ratio_excl_both_one'], 2), o['ratio_excl_both_one_ci'] and [round(x, 2) if x else x for x in o['ratio_excl_both_one_ci']])
    print(' pct coherent by subset', o['pct_coherent_by_subset'])
    if 'pair_kinds' in o:
        print(' pair kinds', o['pair_kinds']); b = o['both_stated']; print(' BOTH STATED ratio', round(b['ratio'], 2), [round(x, 2) for x in b['ratio_ci']], 'both=1 among incoh', b['incoherent_both_one']); print('   coh', show(b['coherent'])); print('   inc', show(b['incoherent'])); print('   inc excl both=1', show(b['incoherent_excl_both_one'])); print(' verdicts', o['stated_verdict_follows_prompt'])
