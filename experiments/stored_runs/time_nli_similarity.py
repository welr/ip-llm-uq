"""
Quick wall-clock benchmark for the per-question NLI similarity computation
that the choice-function method uses as preprocessing. Reports per-question
ms over a representative sample of TriviaQA responses with the same NLI model
(DeBERTa-large-MNLI) and bidirectional symmetrization used in the paper.

Usage: ~/ip_llm/bin/python experiments/time_nli_similarity.py
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_MODEL = "microsoft/deberta-large-mnli"
MAX_LEN = 256
N_RESPONSES = 6
N_BENCH_QUESTIONS = 50

SCRIPT_DIR = Path(__file__).parent
MODAL = SCRIPT_DIR / "modal_data"


def main():
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()

    with open(MODAL / "pred1_qwen.json") as f:
        d = json.load(f)
    qids = list(d["per_question_results"].keys())[:N_BENCH_QUESTIONS]

    # Warmup.
    sample_texts = d["per_question_results"][qids[0]]["base_responses"][:N_RESPONSES]
    with torch.no_grad():
        for _ in range(3):
            inp = tokenizer(sample_texts[0], sample_texts[1], return_tensors="pt",
                            truncation=True, max_length=MAX_LEN).to(device)
            nli(**inp).logits

    # Time M(M-1) directed forward passes per question for N_BENCH_QUESTIONS.
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n_calls = 0
    for qid in qids:
        texts = d["per_question_results"][qid]["base_responses"][:N_RESPONSES]
        if len(texts) < N_RESPONSES:
            continue
        with torch.no_grad():
            for i in range(N_RESPONSES):
                for j in range(N_RESPONSES):
                    if i == j: continue
                    if not texts[i].strip() or not texts[j].strip(): continue
                    inp = tokenizer(texts[i], texts[j], return_tensors="pt",
                                    truncation=True, max_length=MAX_LEN).to(device)
                    nli(**inp).logits
                    n_calls += 1
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    per_q = elapsed / N_BENCH_QUESTIONS * 1000
    per_call = elapsed / n_calls * 1000
    print(f"\nElapsed: {elapsed:.2f}s for {N_BENCH_QUESTIONS} questions ({n_calls} calls)")
    print(f"Per-question: {per_q:.1f} ms")
    print(f"Per-call    : {per_call:.2f} ms")
    print(f"Device      : {device}")

    out = {
        "device": device,
        "nli_model": NLI_MODEL,
        "max_len": MAX_LEN,
        "n_responses": N_RESPONSES,
        "n_bench_questions": N_BENCH_QUESTIONS,
        "n_calls": n_calls,
        "elapsed_s": elapsed,
        "per_question_ms": per_q,
        "per_call_ms": per_call,
    }
    out_path = SCRIPT_DIR / "time_nli_similarity_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
