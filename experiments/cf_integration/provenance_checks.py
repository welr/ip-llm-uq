"""What separates 2.94x from 3.16x: scorer or partition? Reads stored results only."""
import json, os
CF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stored_runs', '')
p3 = json.load(open(CF + 'modal_data/pred3_choice_functions.json'))
u = json.load(open(CF + 'partition_baselines_results.json'))['admissibility']
a = p3['aggregate']
print('Jan run (pred3):', a['admissible_correct'], '/', a['admissible_total'], '|', a['rejected_correct'], '/', a['rejected_total'], 'ratio', round(a['accuracy_ratio'], 4))
print('Current       :', u['ac'], '/', u['at'], '|', u['rc'], '/', u['rt'], 'ratio', round(u['ratio'], 4))
print('total correct: Jan', a['admissible_correct'] + a['rejected_correct'], 'current', u['ac'] + u['rc'])
pj = {m: v['per_question'] for m, v in p3['per_model'].items()}
print('Jan per-model correct', {m: v['admissible_correct'] + v['rejected_correct'] for m, v in p3['per_model'].items()})
g = json.load(open(CF + 'genprob_reweighting_results.json'))['uniform']['per_model']
print('Cur per-model correct', {m: round(v['adm_acc'] * v['n_adm']) + round(v['rej_acc'] * v['n_rej']) for m, v in g.items()})
