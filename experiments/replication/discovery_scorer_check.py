"""Post hoc check on stored discovery data, prompted by the PopQA pilot: the word-bounded scorer, and on TriviaQA the alias list, against the estimates being replicated.

The stored TriviaQA records keep only the first three aliases of each question (`gold_answers[:3]` in the Prediction 1 runners), and the stored labels were scored against those.
Here the same six responses and similarity matrices are rescored against the benchmark's full alias lists, matched by question text. Generates nothing."""
import collections, glob, json, re
import numpy as np
from datasets import load_dataset
from scipy import stats
from analyze_main import row, analyze, MODELS, CF
from scoring import registered_rule, word_bounded_rule
INIT = re.compile(r'(^|\s)[A-Z]$'); Z = stats.norm.ppf(1 - .05 / 2)  # first Holm step, one-sided; the second step is laxer
PRED1 = dict(Qwen='pred1_qwen.json', Mistral='pred1_mistral_20260110_071956.json', Llama='pred1_llama8b_20260112_074432.json', Gemma='pred1_gemma9b_20260112_074222.json')
KEYS = ['H1_tie_gap', 'H2_logratio_no_ties_minus_union', 'S1_logratio_union', 'S2_logratio_no_ties_minus_gmax']
def zeros(): return collections.defaultdict(lambda: np.zeros(20))
def summarize(by, model_by=None):
    r = analyze(by, B=10000); o = {k: r[k] for k in KEYS}; o.update(partition=r['partition'], counts=r['counts'], n_questions=r['n_questions'], fisher_union_two_sided=r['fisher_union_two_sided'])
    return o
def power(o):
    pw = {}
    for k in [KEYS[0], KEYS[1], KEYS[3]]:
        e = abs(o[k]['estimate']); se = o[k]['se']; pw[k] = {}
        for N in [1000, 2000, 3000, 4000]:
            s = se * (500 / N) ** .5; pw[k][N] = dict(se=s, power_observed=float(stats.norm.cdf(e / s - Z)), power_half=float(stats.norm.cdf(e / 2 / s - Z)))
    return pw
OUT = {}
# NQ-Open: stored labels used the benchmark's full answer lists.
ds = load_dataset('google-research-datasets/nq_open', split='validation'); C = collections.Counter(); by = dict(stored=zeros(), word_bounded=zeros())
for m in MODELS:
    pq = json.load(open(sorted(glob.glob(CF + f'modal_data/prereg_nq_open_{m}_*.json'))[-1]))['per_question']; Sm = json.load(open(CF + f'sim_cache/sim_matrices_nq_open_{m}.json'))
    for q, e in pq.items():
        T = e['top6_texts']; gold = ds[int(q.split('_')[1])]['answer']; st = [int(bool(x)) for x in e['accuracies']]; a = [int(registered_rule(t, gold)) for t in T]; b = [int(word_bounded_rule(t, gold)) for t in T]
        C['responses'] += len(T); C['stored_correct'] += sum(st); C['word_bounded_correct'] += sum(b); C['stored_ne_registered_rule'] += sum(x != y for x, y in zip(st, a)); C['registered_only'] += sum(x and not y for x, y in zip(a, b)); C['word_bounded_only'] += sum(y and not x for x, y in zip(a, b))
        C['ends_in_initial'] += sum(bool(INIT.search(t.strip())) for t in T); C['empty'] += sum(not t.strip() for t in T)
        by['stored'][q] += np.array(row(Sm[q], st), float); by['word_bounded'][q] += np.array(row(Sm[q], b), float)
OUT['nq_open'] = dict(counts=dict(C), **{k: summarize(v) for k, v in by.items()}); OUT['nq_open']['power_word_bounded'] = power(OUT['nq_open']['word_bounded'])
# TriviaQA
tq = {r['question']: r['answer'] for r in load_dataset('trivia_qa', 'rc.nocontext', split='validation')}
st_all = {(r['model'], r['qid']): [int(x) for x in r['accuracies']] for r in json.load(open(CF + 'kle_snne_triviaqa_results.json')) if r['dataset'] == 'triviaqa'}
C = collections.Counter(); by = dict(stored=zeros(), registered_full=zeros(), word_bounded_first3=zeros(), word_bounded=zeros()); pm = {m: dict(stored=zeros(), registered_full=zeros()) for m in MODELS}; ex = []; alias_n = []
for m in MODELS:
    P = json.load(open(CF + 'modal_data/' + PRED1[m]))['per_question_results']; Sm = json.load(open(CF + f'sim_cache/sim_matrices_triviaqa_{m}.json'))
    for q, S in Sm.items():
        T = P[q]['base_responses'][:6]; g3 = P[q]['gold_answers']; A = tq[P[q]['question']]; full = sorted(set(A['aliases']) | {A['value']}); st = st_all[(m, q)]
        assert all(x in A['aliases'] or x == A['value'] for x in g3), q
        a3 = [int(registered_rule(t, g3)) for t in T]; af = [int(registered_rule(t, full)) for t in T]; b3 = [int(word_bounded_rule(t, g3)) for t in T]; bf = [int(word_bounded_rule(t, full)) for t in T]
        C['responses'] += len(T); C['stored_correct'] += sum(st); C['stored_ne_first3_rule'] += sum(x != y for x, y in zip(st, a3)); C['registered_full_correct'] += sum(af); C['word_bounded_first3_correct'] += sum(b3); C['word_bounded_correct'] += sum(bf)
        C['gained_by_full_aliases'] += sum(y and not x for x, y in zip(a3, af)); C['lost_by_full_aliases'] += sum(x and not y for x, y in zip(a3, af))
        C['registered_only'] += sum(x and not y for x, y in zip(af, bf)); C['word_bounded_only'] += sum(y and not x for x, y in zip(af, bf)); C['ends_in_initial'] += sum(bool(INIT.search(t.strip())) for t in T); C['empty'] += sum(not t.strip() for t in T)
        if m == MODELS[0]: alias_n.append(len(full)); C['value_outside_first3'] += int(A['value'] not in g3)
        for t, x, y in zip(T, a3, af):
            if y and not x and len(ex) < 14: ex.append((t[:40], g3[:2], A['value']))
        for nm, lab in [('stored', st), ('registered_full', af), ('word_bounded_first3', b3), ('word_bounded', bf)]: by[nm][q] += np.array(row(S, lab), float)
        pm[m]['stored'][q] += np.array(row(S, st), float); pm[m]['registered_full'][q] += np.array(row(S, af), float)
OUT['triviaqa'] = dict(counts=dict(C), alias_list_median=float(np.median(alias_n)), examples_gained=ex, **{k: summarize(v) for k, v in by.items()}); OUT['triviaqa']['power_word_bounded'] = power(OUT['triviaqa']['word_bounded'])
OUT['triviaqa']['per_model'] = {m: {k: {a: b for a, b in summarize(v).items() if a in ['counts', 'partition']} for k, v in d.items()} for m, d in pm.items()}
json.dump(OUT, open('discovery_scorer_check.json', 'w'), indent=1)
for n in OUT:
    v = OUT[n]; print(n, v['counts'])
    for a in [k for k in ['stored', 'registered_full', 'word_bounded_first3', 'word_bounded'] if k in v]:
        print(' ', a, {k: (round(v[a][k]['estimate'], 4), [round(c, 3) for c in v[a][k]['ci']]) for k in KEYS}, '| union ratio', round(v[a]['counts']['union']['ratio'], 3), 'kept/rej acc', round(v[a]['counts']['union']['kept_acc'], 4), round(v[a]['counts']['union']['rej_acc'], 4), '| no-ties ratio', round(v[a]['counts']['no_ties']['ratio'], 3), '| gmax', round(v[a]['counts']['gmax']['ratio'], 3))
    print('  power at 3000', {k: {a: round(b, 3) for a, b in p[3000].items()} for k, p in v['power_word_bounded'].items()})
print('per model union ratio', {m: {k: round(v[k]['counts']['union']['ratio'], 3) for k in v} for m, v in OUT['triviaqa']['per_model'].items()})
print('alias median', OUT['triviaqa']['alias_list_median'], 'examples', ex)
