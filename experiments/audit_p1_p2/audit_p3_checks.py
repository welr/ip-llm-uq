"""Further stored-data checks on Prediction 3. Reads only; writes audit_p3_checks.json.

Fragments: the first-period cut truncates at abbreviations and initials, so a stored response ending in one is a fragment; tested by removing such responses from the table.
Identical strings: NLI similarity is computed even for identical responses, so identical text can fail to merge; tested by setting those pairs to 1 and recomputing the partition.
Labels: the stored labels are recomputed from stored text and dataset gold, at caps 100 and 500.
Sampling: frozen NQ-Open indices against the run, and the 500 TriviaQA questions across the four Prediction 1 files."""
import collections, glob, json, os, re, sys
import numpy as np
from datasets import load_dataset
sys.path.insert(0, '.')
import importlib.util as iu
C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', ''); TH = [round(.5 + .05 * i, 2) for i in range(10)]; MODELS = ['Qwen', 'Mistral', 'Llama', 'Gemma']
R = '../stored_runs/modal_results/'; F = dict(Qwen='pred1_v2_20260109_053417.json', Mistral='pred1_mistral_20260110_071956.json', Llama='pred1_llama8b_20260112_074432.json', Gemma='pred1_gemma9b_20260112_074222.json'); P1 = {m: json.load(open(R + f))['per_question_results'] for m, f in F.items()}
ACC = {(r['model'], r['qid']): r['accuracies'] for r in json.load(open('../triviaqa_rescore/results_full/kle_snne_triviaqa_results.json')) if r['dataset'] == 'triviaqa'}
ABBR = re.compile(r"(^|[\s\-(])(st|mt|dr|mr|mrs|ms|jr|sr|prof|gen|col|capt|lt|rev|no|vs|ft|sgt|gov|sen|inc|co|ltd|[a-z])$", re.I)
norm = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (s or '').lower().strip()))
def load(ds, m): c = [f for f in glob.glob(C + f'modal_data/prereg_{ds}_*.json') if f'_{m}_' in f or f.endswith(f'_{m}.json')]; return json.load(open(sorted(c)[-1]))['per_question']
def part(S):
    n = len(S); A = set()
    for t in TH:
        p = list(range(n))
        def f(x):
            while p[x] != x: p[x] = p[p[x]]; x = p[x]
            return x
        for a in range(n):
            for b in range(a + 1, n):
                if S[a][b] >= t: p[f(a)] = f(b)
        cl = collections.defaultdict(list)
        for a in range(n): cl[f(a)].append(a)
        M = max(map(len, cl.values()))
        for v in cl.values():
            if len(v) == M: A.update(v)
    return A
def tab(c): return dict(ratio=(c['ac'] / c['at']) / (c['rc'] / c['rt']) if c['rt'] and c['rc'] else None, adm=c['ac'] / c['at'], rej=c['rc'] / c['rt'] if c['rt'] else None, rejected=c['rt'], admissible=c['at'])
OUT = {}
for ds in ['triviaqa', 'nq_open', 'truthfulqa']:
    frag = collections.Counter(); nofrag = collections.Counter(); forced = collections.Counter(); id_pairs = []; ex = []; base = collections.Counter(); changed = 0; n_q = 0
    for m in MODELS:
        sp = C + f'sim_cache/sim_matrices_{ds}_{m}.json'; SM = json.load(open(sp)); pq = None if ds == 'triviaqa' else load(ds, m)
        for k, S in SM.items():
            if ds == 'triviaqa':
                if (m, k) not in ACC: continue
                A, X = ACC[(m, k)], P1[m][k]['base_responses'][:6]; adm = part(S)
            else:
                if k not in pq: continue
                A, X, adm = pq[k]['accuracies'], pq[k]['top6_texts'], set(pq[k]['nli_admissible'])
            n_q += 1; S2 = [list(r) for r in S]
            for a in range(6):
                for b in range(a + 1, 6):
                    if norm(X[a]) and norm(X[a]) == norm(X[b]): id_pairs.append(S[a][b]); S2[a][b] = S2[b][a] = 1.0
            ad2 = part(S2); changed += ad2 != adm
            for i in range(6):
                g = 'a' if i in adm else 'r'; base[g + 't'] += 1; base[g + 'c'] += bool(A[i]); isf = bool(ABBR.search((X[i] or '').strip()))
                if isf:
                    frag[g + 't'] += 1; frag[g + 'c'] += bool(A[i])
                    if len(ex) < 400: ex.append(X[i])
                else: nofrag[g + 't'] += 1; nofrag[g + 'c'] += bool(A[i])
                h = 'a' if i in ad2 else 'r'; forced[h + 't'] += 1; forced[h + 'c'] += bool(A[i])
    ip = np.array(id_pairs); OUT[ds] = dict(base=tab(base), fragments=dict(n=frag['at'] + frag['rt'], admissible=frag['at'], rejected=frag['rt'], correct=frag['ac'] + frag['rc']), without_fragments=tab(nofrag), identical_pairs=dict(n=len(ip), below_95=float((ip < .95).mean()), below_50=float((ip < .5).mean()), median=float(np.median(ip))), identical_forced=tab(forced), questions_whose_partition_changes=changed, n_q=n_q)
    o = OUT[ds]; print('=====', ds, {k: round(v, 3) if isinstance(v, float) else v for k, v in o['base'].items()}); print('  fragments', o['fragments'], 'examples', collections.Counter(ex).most_common(8)); print('  without fragments', {k: round(v, 3) if isinstance(v, float) else v for k, v in o['without_fragments'].items()}); print('  identical pairs', o['identical_pairs']); print('  identical forced to 1', {k: round(v, 3) if isinstance(v, float) else v for k, v in o['identical_forced'].items()}, '| question-model pairs changed', changed, 'of', n_q)
# Labels from text and gold
def check(g, G, cap):
    n = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', (s or '').lower().strip()))[:cap]; x = n(g)
    return bool(x) and any(n(a) and (n(a) in x or x in n(a)) for a in G)
nq = load_dataset('google-research-datasets/nq_open', split='validation'); tq = load_dataset('truthful_qa', 'generation', split='validation')
idx = json.load(open(C + 'nq_open_indices.json'))['indices']; L = collections.defaultdict(dict)
for ds, gold in [('nq_open', lambda k: list(nq[int(k.split('_')[1])]['answer']) or ['']), ('truthfulqa', lambda k: list(tq[int(k.split('_')[1])]['correct_answers']) or [tq[int(k.split('_')[1])]['best_answer']])]:
    for m in MODELS:
        pq = load(ds, m); d = {c: 0 for c in (100, 500)}; tot = 0
        for k, r in pq.items():
            for t, a in zip(r['top6_texts'], r['accuracies']):
                tot += 1
                for c in d: d[c] += check(t, gold(k), c) != bool(a)
        L[ds][m] = dict(total=tot, mismatches_cap100=d[100], mismatches_cap500=d[500])
OUT['labels'] = L; print('labels recomputed from text:', json.dumps(L))
ids = {m: sorted(int(k.split('_')[1]) for k in load('nq_open', m)) for m in MODELS}; OUT['nq_indices_match_frozen'] = {m: ids[m] == sorted(idx) for m in MODELS}; print('NQ indices match frozen list:', OUT['nq_indices_match_frozen'])
Q = {m: {v['question'] for v in P1[m].values()} for m in MODELS}; OUT['triviaqa_same_questions'] = dict(sizes={m: len(Q[m]) for m in MODELS}, intersection=len(set.intersection(*Q.values()))); print('TriviaQA question sets:', OUT['triviaqa_same_questions'])
json.dump(OUT, open('audit_p3_checks.json', 'w'), indent=1, default=str)
