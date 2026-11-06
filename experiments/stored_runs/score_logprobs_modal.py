"""
Score cached TriviaQA top-6 responses under each base model to obtain mean
token log-probabilities. Used to reweight the admissibility partition (Edit
1c, Reviewer W3): does using model confidence instead of uniform 1/M weights
change the rejected/admissible split?

We do NOT regenerate. For each cached response string r given the prompt
"Answer the following question concisely.\n\nQ: {q}\nA:", we run a single
teacher-forced forward pass over (prompt + r) and compute the mean per-token
log-probability of the response tokens conditional on the prompt and earlier
response tokens.

Inputs:
  experiments/modal_data/pred1_{qwen,mistral_*,llama8b_*,gemma9b_*}.json

Outputs (per model):
  /results/logprobs_triviaqa_{model_key}.json
    {qid: {"logprobs": [float; 6], "n_tokens": [int; 6]}}

Run on Modal:
    modal run experiments/score_logprobs_modal.py

Compute envelope: 4 models * 500 q * 6 responses = 12000 forward passes,
~50 tokens each. On A100 80GB with bf16 and per-pair batching this runs
in roughly 30-60 minutes total wall-clock.
"""

import modal

app = modal.App("score-logprobs-triviaqa")
volume = modal.Volume.from_name("choice-fn-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.50.0",
        "accelerate",
        "numpy",
        "tqdm",
        "sentencepiece",
        "protobuf",
    )
)

MODELS = {
    "Qwen":    {"name": "Qwen/Qwen2.5-7B"},
    "Mistral": {"name": "mistralai/Mistral-7B-v0.1"},
    "Llama":   {"name": "meta-llama/Llama-3.1-8B"},
    "Gemma":   {"name": "google/gemma-2-9b", "dtype": "bfloat16"},
}

PRED1_FILES = {
    "Qwen":    "pred1_qwen.json",
    "Mistral": "pred1_mistral_20260110_071956.json",
    "Llama":   "pred1_llama8b_20260112_074432.json",
    "Gemma":   "pred1_gemma9b_20260112_074222.json",
}

PROMPT_TEMPLATE = "Answer the following question concisely.\n\nQ: {question}\nA:"

N_RESPONSES = 6
MAX_PROMPT_TOKENS = 256
MAX_RESPONSE_TOKENS = 64


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/results": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def score_one_model(model_key: str):
    import os
    import json
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    HF_TOKEN = os.environ["HF_TOKEN"]
    info = MODELS[model_key]
    model_name = info["name"]
    dtype = torch.bfloat16 if info.get("dtype") == "bfloat16" else torch.float16

    pred1_path = f"/results/{PRED1_FILES[model_key]}"
    out_path = f"/results/logprobs_triviaqa_{model_key}.json"

    print(f"[{model_key}] loading pred1: {pred1_path}")
    with open(pred1_path) as f:
        d = json.load(f)
    pqr = d["per_question_results"]
    qids = sorted(pqr.keys(), key=lambda x: int(x))
    print(f"[{model_key}] {len(qids)} questions")

    print(f"[{model_key}] loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=HF_TOKEN, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    # Resume support: skip already-scored qids.
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"[{model_key}] resuming with {len(out)} qids already scored")

    t0 = time.time()
    pbar = tqdm(qids, desc=f"{model_key} score")
    for qid in pbar:
        if qid in out:
            continue
        rec = pqr[qid]
        question = rec["question"]
        responses = rec["base_responses"][:N_RESPONSES]
        if len(responses) < N_RESPONSES:
            continue
        prompt = PROMPT_TEMPLATE.format(question=question)

        # Tokenize prompt once.
        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
            add_special_tokens=True,
        ).input_ids.to(device)
        n_prompt = prompt_ids.shape[1]

        per_response_logprobs = []
        per_response_ntokens = []

        for r in responses:
            # Tokenize response (no special tokens at the start; we want pure
            # continuation of the prompt). Allow a leading space so subword
            # tokenizers produce the right opening token for a continuation.
            resp_text = r if r.startswith((" ", "\n")) else " " + r
            resp_ids = tokenizer(
                resp_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_RESPONSE_TOKENS,
                add_special_tokens=False,
            ).input_ids.to(device)
            n_resp = resp_ids.shape[1]

            if n_resp == 0:
                # Empty response -> log p = -inf. Use NaN sentinel; downstream
                # falls back to uniform weight for that response.
                per_response_logprobs.append(float("nan"))
                per_response_ntokens.append(0)
                continue

            # Concatenate prompt + response. Forward pass over the full
            # sequence; logits at position t-1 give next-token distribution
            # for position t. We want log p(resp_token_t | prompt + resp_<t).
            full_ids = torch.cat([prompt_ids, resp_ids], dim=1)
            with torch.no_grad():
                logits = model(full_ids).logits  # (1, T, V)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            # Positions [n_prompt-1 ... n_prompt+n_resp-2] predict the
            # response tokens [0 ... n_resp-1].
            target_positions = torch.arange(n_prompt - 1, n_prompt + n_resp - 1, device=device)
            target_tokens = resp_ids[0]
            picked = log_probs[0, target_positions, target_tokens]
            mean_lp = picked.mean().item()

            per_response_logprobs.append(mean_lp)
            per_response_ntokens.append(n_resp)

        out[qid] = {
            "logprobs": per_response_logprobs,
            "n_tokens": per_response_ntokens,
        }

        # Periodic checkpoint.
        if len(out) % 50 == 0:
            with open(out_path, "w") as f:
                json.dump(out, f)
            volume.commit()
            elapsed = time.time() - t0
            pbar.set_postfix(saved=len(out), s=f"{elapsed:.0f}")

    with open(out_path, "w") as f:
        json.dump(out, f)
    volume.commit()
    print(f"[{model_key}] done. saved {len(out)} qids to {out_path}")
    return {"model": model_key, "n_qids": len(out), "elapsed_s": time.time() - t0}


@app.local_entrypoint()
def main(model_key: str = ""):
    """If --model-key is given, score only that model. Otherwise score all four."""
    if model_key:
        keys = [model_key]
    else:
        keys = ["Qwen", "Mistral", "Llama", "Gemma"]
    for k in keys:
        print(f"\n=== scoring {k} ===")
        result = score_one_model.remote(k)
        print(f"=== {k}: {result} ===")
