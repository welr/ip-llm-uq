"""Prediction 4 was scored against 3 stored aliases by an unguarded test. Rescore with the full alias list and the guard."""
import json, re, collections, statistics as st
from datasets import load_dataset
P = '../stored_runs/modal_results/pred4_local_llama8b_checkpoint.json'
d = json.load(open(P))['results']
ds = load_dataset('trivia_qa', 'rc.nocontext', split='validation')
gold = {r['question']: r['answer']['aliases'] + r['answer']['normalized_aliases'] for r in ds}
extra = load_dataset('trivia_qa', 'rc.nocontext', split='train')
for r in extra:
    gold.setdefault(r['question'], r['answer']['aliases'] + r['answer']['normalized_aliases'])
def norm(s):
    s = s.lower().strip()
    for a in ['a ', 'an ', 'the ']:
        if s.startswith(a): s = s[len(a):]
    return re.sub(r'[^\w\s]', '', s).strip()
def unguarded(r, G):
    x = norm(r)
    return any(norm(g) in x or x in norm(g) for g in G)
def guarded(r, G):
    x = norm(r)
    if not x: return False
    return any(norm(g) and (norm(g) in x or x in norm(g)) for g in G)
v = next(iter(d.values())); tr = v['temperature_results']; print('temp entry keys', list(next(iter(tr.values()))))
Q = ['Q1','Q2','Q3','Q4']; T = sorted(tr, key=float); miss = 0; emptyS = 0
stored = {q:{t:[] for t in T} for q in Q}; full = {q:{t:[] for t in T} for q in Q}; full_u = {q:{t:[] for t in T} for q in Q}; n = collections.Counter()
for k, e in d.items():
    G = gold.get(e['question'])
    if G is None: miss += 1; continue
    n[e['quartile']] += 1
    if any(not norm(g) for g in e['gold_answers']): emptyS += 1
    for t in T:
        rs = e['temperature_results'][t]['responses']
        rs = [r['text'] if isinstance(r, dict) else r for r in rs]
        stored[e['quartile']][t] += [unguarded(r, e['gold_answers']) for r in rs]
        full[e['quartile']][t] += [guarded(r, G) for r in rs]
        full_u[e['quartile']][t] += [unguarded(r, G) for r in rs]
print('missing questions', miss, 'n by quartile', dict(n), 'questions with an empty stored alias', emptyS)
for name, tab in [('as run (3 aliases, unguarded)', stored), ('full aliases, unguarded', full_u), ('full aliases, guarded', full)]:
    print(name)
    for q in Q:
        row = [100*st.mean(tab[q][t]) for t in T]
        print(' ', q, ' '.join(f'{x:5.1f}' for x in row), 'sd(acc)=%.3f' % st.pstdev([x/100 for x in row]), 'keep@1.0=%.0f%%' % (100*row[T.index('1.0')]/max(row)))
json.dump({q: dict(acc={t: 100*st.mean(full[q][t]) for t in T}, n=n[q], sd=st.pstdev([st.mean(full[q][t]) for t in T]), keep_at_1=st.mean(full[q]['1.0'])/max(st.mean(full[q][t]) for t in T)) for q in Q}, open('pred4_rescored.json', 'w'), indent=1)
