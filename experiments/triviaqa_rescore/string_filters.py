"""String-based admissibility and heuristic filters on TriviaQA, with the definitions in the registered runner (run_preregistered.py).
Also the self-consistency partition of verify_self_consistency_partition.py, which only prints its result.
No stored result file holds the string and heuristic figures for TriviaQA, so the control run against stored labels is the check on this code.  Usage: string_filters.py SANDBOX [OUT.json]"""
import collections, json, os, re, sys
sb = sys.argv[1]
MODELS = dict(Qwen='pred1_qwen.json', Mistral='pred1_mistral_20260110_071956.json', Llama='pred1_llama8b_20260112_074432.json', Gemma='pred1_gemma9b_20260112_074222.json')
def norm(a):
    if not a: return ''
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', a.lower().strip()))[:100]  # the cap of the original analysis, which keeps its partitions
def agg(a):
    n = norm(a); n = re.sub(r'^(the|a|an)\s+', '', n); n = re.sub(r'^(it is|its|they are|he is|she is|this is)\s+', '', n); return n.strip()
def sim(a, b):
    if norm(a) == norm(b): return 'exact'
    x, y = agg(a), agg(b)
    if x == y: return 'normalized'
    if x and y and (x in y or y in x): return 'substring'
    return 'distinct'
LEVELS = dict(exact={'exact'}, normalized={'exact', 'normalized'}, substring={'exact', 'normalized', 'substring'})
def clusters(T, lev):
    n = len(T); p = list(range(n))
    def f(x):
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if sim(T[i], T[j]) in LEVELS[lev]:
                a, b = f(i), f(j)
                if a != b: p[a] = b
    g = collections.defaultdict(set)
    for i in range(n): g[f(i)].add(i)
    return list(g.values())
lab = {(r['model'], r['qid']): [int(x) for x in r['accuracies']] for r in json.load(open(os.path.join(sb, 'kle_snne_triviaqa_results.json'))) if r['dataset'] == 'triviaqa'}
C = collections.defaultdict(lambda: [0, 0, 0, 0]); per = collections.defaultdict(lambda: [0, 0, 0, 0]); sc = collections.defaultdict(lambda: [0, 0, 0, 0])
for m, fn in MODELS.items():
    for q, v in json.load(open(os.path.join(sb, 'modal_data', fn)))['per_question_results'].items():
        T = v['base_responses'][:6]; a = lab[(m, q)]; n = len(T); adm = set()
        for lev in LEVELS:
            cs = clusters(T, lev); mx = max(map(len, cs))
            for c in cs:
                if len(c) == mx: adm |= c
        N = [norm(t) for t in T]; A = [agg(t) for t in T]; fc = collections.Counter(N); cc = collections.Counter(A)
        rej = dict(string=set(range(n)) - adm, consistency={i for i in range(n) if cc[A[i]] <= 1}, frequency={i for i in range(n) if fc[N[i]] <= 1}, length={i for i in range(n) if len(T[i].strip().split()) < 2})
        Nm = [re.sub(r'[^\w\s]', '', re.sub(r'\s+', ' ', t.lower().strip())) for t in T]; fm = collections.Counter(Nm); top = max(fm.values()); mode = {i for i in range(n) if fm[Nm[i]] == top}  # normalize_answer there: no cap, whitespace collapsed before punctuation goes
        for i in range(n):
            for t in (sc['pooled'], sc[m]):
                j = 0 if i in mode else 2; t[j] += a[i]; t[j + 1] += 1
        for k, R in rej.items():
            for i in range(n):
                t = C[k]; j = 2 if i in R else 0; t[j] += a[i]; t[j + 1] += 1
                if k == 'string': t = per[m]; t[j] += a[i]; t[j + 1] += 1
out = {}
for k, (ac, at, rc, rt) in C.items(): out[k] = dict(ac=ac, at=at, rc=rc, rt=rt, retained_acc=ac / at, rejected_acc=rc / rt, rej_rate=rt / (at + rt), ratio=(ac / at) / (rc / rt))
out['per_model_string_ratio'] = {m: (x[0] / x[1]) / (x[2] / x[3]) for m, x in per.items()}
out['self_consistency'] = {k: dict(ac=v[0], at=v[1], rc=v[2], rt=v[3], ratio=(v[0] / v[1]) / (v[2] / v[3]), rej_rate=v[3] / (v[1] + v[3])) for k, v in sc.items()}
out['random_baseline'] = sum(map(sum, lab.values())) / sum(map(len, lab.values()))
if len(sys.argv) > 2: json.dump(out, open(sys.argv[2], 'w'), indent=1)
for k in ['string', 'consistency', 'frequency', 'length']:
    o = out[k]; print(k, f"{100*o['retained_acc']:.1f} {100*o['rejected_acc']:.1f} {100*o['rej_rate']:.1f} {o['ratio']:.2f}")
print('self-consistency', out['self_consistency']['pooled'])
print('random baseline', round(100 * out['random_baseline'], 1), 'per model', {m: round(x, 2) for m, x in out['per_model_string_ratio'].items()})
