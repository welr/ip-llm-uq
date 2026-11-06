# Stored runs

Copied unchanged on 2026-11-05 from the author's earlier project (working tree at commit 72e7f16). `MANIFEST.sha256` lists each file and its hash.

`modal_results/` is the exception: its files are copied unchanged from `paper/experiments/modal_results/` in this repository. They are the Prediction 1, 2 and 4 runs that `audit_p1_p2/` and `cf_integration/rescore_pred4.py` read.

Holds the stored model runs, similarity caches and result files the analyses in `experiments/` read, chosen by tracing those reads, plus the raw frontier-model outputs. The Python files are that project's experiment scripts, including the generation runners (`run_preregistered.py`, `frontier_*_run*.py`) and the scripts `triviaqa_rescore/make_sandbox.py` needs to regenerate `triviaqa_rescore/results_full/`. Their comments are as first written, including references to that project's reviewers.

Not carried: backups and intermediate results that no analysis here reads. Benchmark text in these files stays under each benchmark's own license.
