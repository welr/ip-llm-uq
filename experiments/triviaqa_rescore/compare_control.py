"""Compare each result file in a sandbox with the stored original, leaf by leaf. Usage: compare_control.py SANDBOX FILE..."""
import json, math, os, sys
CF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', '')
sb = sys.argv[1]
def leaves(x, p=''):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, f'{p}/{k}')
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, f'{p}/{i}')
    else: yield p, x
for f in sys.argv[2:]:
    a = dict(leaves(json.load(open(CF + f)))); b = dict(leaves(json.load(open(os.path.join(sb, f)))))
    common = set(a) & set(b); diff = []
    for k in common:
        x, y = a[k], b[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if not (x == y or (math.isnan(x) and math.isnan(y)) or abs(x - y) <= 1e-9 * max(1, abs(x))): diff.append((k, x, y))
        elif x != y: diff.append((k, x, y))
    print(f"{f}: leaves {len(a)} vs {len(b)}, common {len(common)}, differ {len(diff)}", [(k[:44], (round(x, 5) if isinstance(x, float) else x), (round(y, 5) if isinstance(y, float) else y)) for k, x, y in sorted(diff, key=lambda t: t[0])[:4]])
