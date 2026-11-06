# LLM Uncertainty Methods as Imprecise Probabilities: code and data

Code and stored runs behind the paper's empirical results.

## Rerunning the analyses

```
pip install -r requirements.txt
./reproduce.sh
```

`reproduce.sh` reruns the analyses on the stored runs and rewrites `generated/`. It needs no GPU, API key or model weights. It downloads public benchmarks through `datasets` on the first run. Set `PYTHON` to use another interpreter. To run one script, run it from its own folder.

## What it does not rerun

These need model weights or API access, and regenerate the inputs rather than the analyses:

- `experiments/stored_runs/run_preregistered.py` and `frontier_*_run*.py`: the registered and frontier-model runs.
- `experiments/audit_p1_p2/regen_base.py`, `regen_instruct.py`, `neutral_prompt_test.py`, `rerun_p4_template.py`: the local reruns. Their outputs are in `regen/` and beside the scripts.

`experiments/triviaqa_rescore/` regenerates `results_full/` from the stored runs through a sandbox (`make_sandbox.py`, `run_sandbox.sh`, `collect_full.py`). It takes longer, so `reproduce.sh` leaves it out.

## Registrations

[MMLU and ARC-Challenge](https://osf.io/ady3k), [TruthfulQA](https://osf.io/dqbmg), [NQ-Open](https://osf.io/78pb6). The paper reports the deviations from them.

## Notes

- `experiments/stored_runs/` is copied unchanged from an earlier choice-function project, per `MANIFEST.sha256`, so its comments refer to that project and its reviewers.
- Docstrings and `run_sandbox.sh` name `~/ip_llm/bin/python`, the author's environment. Substitute any environment built from `requirements.txt`.
- See `DATA_NOTICE.md` for the terms on benchmark text and model outputs, and `LICENSE` for the code.
