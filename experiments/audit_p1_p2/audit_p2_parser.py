"""Prediction 2: what did the parser read? Only the Llama run stored raw responses. Reads only."""
import collections, json, re, random
import numpy as np
import os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', 'modal_results', '')
rs = list(json.load(open(R + 'pred2_local_llama70b_20260407_161230.json'))['results'].values())
def branch(resp):
    """Which branch of extract_confidence fired, and on what text."""
    if not resp: return 'none', None, None
    t = resp.lower().strip()
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', t)
    if m: v = float(m.group(1)); return 'percent', (v / 100 if v > 1 else v), m
    m = re.search(r'(?:^|[^\d])0\.(\d+)', t)
    if m: return 'decimal', float(f'0.{m.group(1)}'), m
    m = re.search(r'(?:^|[\s:])(\d+(?:\.\d+)?)\b', t)
    if m:
        v = float(m.group(1))
        if 0 <= v <= 1: return 'bare', v, m
        if 1 < v <= 100: return 'bare', v / 100, m
    return 'none', None, None
CONF = re.compile(r'(confiden|certain|likelihood|probabilit|sure)', re.I)
def stated(resp, m):
    """Is the parsed number presented as a confidence: near a confidence word, or the entire response?"""
    t = resp.lower().strip()
    if len(t) <= 8: return True
    w = t[max(0, m.start() - 60):m.end() + 40]
    return bool(CONF.search(w))
C = collections.Counter(); rows = []
for r in rs:
    for k, rk, ck in [('T', 'response_true', 'c_true'), ('F', 'response_false', 'c_false')]:
        b, v, m = branch(r[rk]); 
        C['parser_agrees_on_stored_text' if (v is not None and r[ck] is not None and abs(v - r[ck]) < 1e-9) or (v is None and r[ck] is None) else 'parser_differs_stored_text_truncated'] += 1
        C[b] += 1
        if m is not None:
            s = stated(r[rk], m); C[b + ('_stated' if s else '_stray')] += 1
            rows.append(dict(side=k, branch=b, stated=s, value=v, text=r[rk], num=m.group(0), coh=r['is_coherent']))
        if b == 'percent' and re.search(r'(?<![\d.])1\s*%', r[rk].lower()) and v == 1.0: C['one_percent_read_as_100'] += 1
        if b == 'percent' and m and float(m.group(1)) < 1 and v == float(m.group(1)) and v > 0: C['sub1_percent_read_as_fraction'] += 1
print(dict(C)); print('valid pairs', sum(r['is_coherent'] is not None for r in rs))
V = [r for r in rs if r['is_coherent'] is not None]
print('both identical text', sum(r['response_true'] == r['response_false'] for r in V), '| c_true=c_false=1', sum(r['c_true'] == 1 and r['c_false'] == 1 for r in V), '| sum==2 among incoherent', sum(r['confidence_sum'] == 2 for r in V if not r['is_coherent']), 'of', sum(not r['is_coherent'] for r in V))
print('value counts c_true', collections.Counter(round(r['c_true'], 2) for r in V).most_common(10)); print('value counts c_false', collections.Counter(round(r['c_false'], 2) for r in V).most_common(10))
print('sum histogram', np.histogram([r['confidence_sum'] for r in V], bins=[0, .5, .9, 1.0, 1.0001, 1.05, 1.2, 1.5, 1.99, 2.01])[0].tolist())
random.seed(3)
for tag, sel in [('INCOHERENT with both==1', [r for r in V if r['c_true'] == 1 and r['c_false'] == 1]), ('COHERENT', [r for r in V if r['is_coherent']])]:
    print('---', tag, len(sel))
    for r in random.sample(sel, 6):
        print(f" c_true={r['c_true']} c_false={r['c_false']} is_true={r['is_true']} | T: {r['response_true'][:150]!r} | F: {r['response_false'][:150]!r}")
print('--- percent-branch stray examples')
for x in random.sample([x for x in rows if x['branch'] == 'percent' and not x['stated']], 5): print('  ', x['side'], x['value'], repr(x['num']), repr(x['text'][:170]))
print('--- bare-branch examples')
for x in random.sample([x for x in rows if x['branch'] == 'bare'], 8): print('  ', x['side'], x['value'], repr(x['num']), 'stated' if x['stated'] else 'stray', repr(x['text'][:170]))
print('--- length of responses, chars: median', np.median([len(r['response_true']) for r in rs]), 'short (<=8 chars)', sum(len(r['response_true'].strip()) <= 8 for r in rs))
