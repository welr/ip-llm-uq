"""Control for registered_analyses.py: do its signals, joined to the labels behind the manuscript's AUROCs, return the manuscript's values?

Manuscript, 1-maxCP, SE(0.75), KLE, SNNE: TriviaQA .773 .774 .786 .788; ARC-Challenge .712 .702 .695 .722."""
import json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', ''); sys.path.insert(0, C)
from kle_snne_comparison import kernel_language_entropy as KLE, snne_entropy as SNNE, fixed_threshold_se as SE, _cluster, THRESHOLDS as TH
S = {}
def sig(M):
    M = np.array(M); cp = [max(len(c) for c in _cluster(M, t)) / M.shape[0] for t in TH]
    return dict(one_minus_maxcp=1 - max(cp), se_075=SE(M, .75), kle=float(KLE(M)), neg_snne=-float(SNNE(M)))
for name, rows, ds in [('arc', [r for r in json.load(open(C + 'kle_snne_results.json')) if r['dataset'] == 'arc'], 'arc'), ('triviaqa', [r for r in json.load(open('../triviaqa_rescore/results_full/kle_snne_triviaqa_results.json')) if r['dataset'] == 'triviaqa'], 'triviaqa')]:
    X = []; y = []; own = []
    for m in ['Qwen', 'Mistral', 'Llama', 'Gemma']:
        SM = json.load(open(C + f'sim_cache/sim_matrices_{ds}_{m}.json')); L = {r['qid']: r for r in rows if r['model'] == m}
        for k, r in L.items():
            if k in SM: X.append(sig(SM[k])); y.append(int(not r['mv_correct'])); own.append(r)
    print(name, 'n', len(y), 'mv wrong', round(float(np.mean(y)), 4), 'AUROC', {k: round(float(roc_auc_score(y, [s[k] for s in X])), 3) for k in X[0]}, 'stored-signal KLE', round(float(roc_auc_score(y, [r['kle'] for r in own])), 3), 'stored -SNNE', round(float(roc_auc_score(y, [-r['snne'] for r in own])), 3), 'max |kle diff|', round(float(np.max(np.abs(np.array([s['kle'] for s in X]) - np.array([r['kle'] for r in own])))), 6))
s = json.load(open(C + 'snne_headtohead_results.json')); print('headtohead keys', list(s)[:6], {k: s['ARC'].get(k) for k in list(s['ARC'])[:4] if k != 'signals'})
print(open(C + 'snne_headtohead.py').read()[:0] if os.path.exists(C + 'snne_headtohead.py') else 'no snne_headtohead.py')
