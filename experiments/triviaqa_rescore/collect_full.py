"""Copy the full-alias outputs out of the sandbox into results_full/, which the manuscript pipeline reads.  Usage: collect_full.py FULL_SANDBOX CONTROL_SANDBOX"""
import json, os, shutil, subprocess, sys
full, control = sys.argv[1], sys.argv[2]; os.makedirs('results_full', exist_ok=True)
FILES = ['kle_snne_triviaqa_results.json', 'multi_argmax_stratification_all_results.json', 'partition_baselines_results.json', 'matched_retention_w1_results.json', 'genprob_reweighting_results.json', 'self_consistency_results.json', 'interval_endpoint_sweep_results.json', 'retention_policy_ablation_results.json', 'goldilocks_external_proxy_results.json', 'embedding_similarity_ablation_results.json', 'nli_aggregation_ablation_results.json']
for f in FILES: shutil.copy2(os.path.join(full, f), 'results_full')
subprocess.run([sys.executable, 'string_filters.py', full, 'results_full/string_filters.json'], check=True, stdout=subprocess.DEVNULL)
r = subprocess.run([sys.executable, 'compare_control.py', control] + FILES, check=True, capture_output=True, text=True).stdout
s = subprocess.run([sys.executable, 'string_filters.py', control], check=True, capture_output=True, text=True).stdout
open('results_full/control_report.txt', 'w').write('Control sandbox against stored results; every file should report differ 0.\n' + r + '\nString filters and self-consistency on stored labels. Paper table: string 63.9 14.2 7.1 4.49; consistency 63.4 18.1 6.9 3.51; frequency 63.6 20.8 7.6 3.06; length 52.8 70.1 43.5 0.75. Stored self-consistency counts: 6889/10715, 349/1285.\n' + s)
print(open('results_full/control_report.txt').read()[:1400])
