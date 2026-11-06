# Data notice

The MIT license in `LICENSE` covers the code (`*.py`, `*.sh`). It does not cover the following.

**Benchmark text.** Files under `experiments/` hold questions and answers from these benchmarks, which stay under their own terms.

| Benchmark | Terms | Source |
|---|---|---|
| TriviaQA (Joshi et al., 2017) | The project README says Apache-2.0 "applies to both the code and the data". The project page says the University of Washington "does not own the copyright of the questions and documents". The Hugging Face card lists the license as unknown. We cannot resolve this conflict. | [README](https://github.com/mandarjoshi90/triviaqa), [project page](http://nlp.cs.washington.edu/triviaqa/) |
| NQ-Open (Kwiatkowski et al., 2019; Roberts et al., 2020) | CC BY-SA 3.0 | [Natural Questions](https://github.com/google-research-datasets/natural-questions) |
| ARC-Challenge (Clark et al., 2018) | CC BY-SA 4.0 | [dataset card](https://huggingface.co/datasets/allenai/ai2_arc) |
| TruthfulQA (Lin et al., 2022) | Apache-2.0 | [repository](https://github.com/sylinrl/TruthfulQA) |
| MMLU (Hendrycks et al., 2021) | MIT | [repository](https://github.com/hendrycks/test) |

**Model outputs.** `experiments/stored_runs/` and `experiments/audit_p1_p2/regen/` hold responses from Qwen2.5 and Mistral (Apache-2.0), Llama 3.1 (Llama 3.1 Community License), Gemma 2 (Gemma Terms of Use), Claude and Gemini (provider terms). No model weights are included. Model outputs can be false, and factual assertions in them should not be relied upon without independent checking.

**Paper text.** `generated/` holds the tables and paragraphs the scripts write for the paper. That text is part of the paper and is not offered under the code license.

This notice summarizes the terms as read on 2026-11-05 and 2026-11-06. It is not legal advice.
