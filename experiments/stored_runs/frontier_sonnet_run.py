"""
Exploratory frontier-scale scope extension: Claude Sonnet 4.6 on TruthfulQA.

Not pre-registered. Closes the reviewer's specific ask about GPT-4-class
frontier evaluation on TruthfulQA (~100 questions).

Protocol (matching the existing base-model protocol as closely as the
Anthropic API allows):
  - 100 questions: seed-42 random sub-sample of TruthfulQA validation split
  - 15 responses per question: 5 temperatures * 3 samples
      Temperatures {0.3, 0.5, 0.7, 0.9, 1.0}  (API caps at 1.0; base
      models used {0.3, 0.5, 0.7, 1.0, 1.3} -- document the deviation)
  - top_p: unset (Anthropic API rejects simultaneous temperature+top_p;
      deviates from base protocol's top_p=0.95)
  - max_tokens = 500 (base protocol was 50; probe showed Sonnet's natural
      essay-style answers reach ~310 tokens, so 500 catches the tail with
      margin. Document deviation.)
  - No system prompt
  - Model: claude-sonnet-4-6 (current version as of run)

Output:
  experiments/modal_data/frontier_sonnet_truthfulqa.json
    Same structure as prereg_truthfulqa_*.json so downstream analysis
    code works unchanged.

Usage:
    export ANTHROPIC_API_KEY=...
    ~/ip_llm/bin/python experiments/frontier_sonnet_run.py
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from anthropic import Anthropic
from datasets import load_dataset


MODEL = "claude-sonnet-4-6"
N_QUESTIONS = 100
SEED = 42
TEMPERATURES = [0.3, 0.5, 0.7, 0.9, 1.0]  # API-clipped from base {0.3,0.5,0.7,1.0,1.3}
SAMPLES_PER_TEMP = 3
MAX_TOKENS = 500  # probe showed Sonnet essay-answers reach ~310 tok; 500 catches tail
WORKERS = 8  # parallel concurrency (Tier 2+ Anthropic rate limits)


def sample_question(client, question_text, max_retries=8):
    """Generate 15 responses (5 temps * 3 samples)."""
    responses = []
    for temp in TEMPERATURES:
        for _ in range(SAMPLES_PER_TEMP):
            last_err = None
            for attempt in range(max_retries):
                try:
                    r = client.messages.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        temperature=temp,
                        messages=[{"role": "user", "content": question_text}],
                    )
                    text = r.content[0].text if r.content else ""
                    responses.append({
                        "text": text,
                        "temp": temp,
                        "stop_reason": r.stop_reason,
                    })
                    break
                except Exception as e:
                    err_str = str(e)
                    last_err = err_str
                    if "429" in err_str or "rate_limit" in err_str.lower():
                        wait = 30 + 10 * attempt
                        print(f"  rate limit, sleep {wait}s", file=sys.stderr, flush=True)
                        time.sleep(wait)
                    elif "529" in err_str or "overloaded" in err_str.lower():
                        wait = 20 + 10 * attempt
                        print(f"  overloaded, sleep {wait}s", file=sys.stderr, flush=True)
                        time.sleep(wait)
                    else:
                        time.sleep(min(2 ** attempt, 30))
            else:
                # All retries failed; record a failed response to preserve index alignment
                responses.append({
                    "text": "",
                    "temp": temp,
                    "stop_reason": "error",
                    "error": last_err,
                })
    return responses


def process_one_question(args):
    client, qid, row = args
    q_text = row["question"]
    gold = list(row.get("correct_answers", [])) or [row.get("best_answer", "")]
    incorrect = list(row.get("incorrect_answers", []))
    responses = sample_question(client, q_text)
    return qid, {
        "question": q_text,
        "question_raw": q_text,
        "gold_answers": gold,
        "tqa_type": row.get("type", ""),
        "tqa_category": row.get("category", ""),
        "tqa_incorrect_answers": incorrect,
        "responses": responses,
    }


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY environment variable not set.")
    client = Anthropic(api_key=api_key)

    script_dir = Path(__file__).parent
    out_path = script_dir / "modal_data" / "frontier_sonnet_truthfulqa.json"
    checkpoint_every = 10

    print(f"Loading TruthfulQA generation/validation...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    rng = np.random.default_rng(SEED)
    all_indices = rng.choice(len(ds), size=N_QUESTIONS, replace=False).tolist()
    all_indices = sorted(all_indices)
    print(f"  Seed {SEED}: {len(all_indices)} questions sampled from {len(ds)}")

    # Resume from checkpoint if present
    question_data = {}
    if out_path.exists():
        question_data = json.load(open(out_path))
        print(f"  Resumed with {len(question_data)} already-done questions")

    # Pending = not yet in question_data
    pending = []
    for idx in all_indices:
        qid = f"tqa_{idx}"
        if qid in question_data and len(question_data[qid].get("responses", [])) >= 15:
            continue
        pending.append((qid, ds[int(idx)]))

    print(f"  {len(pending)} questions to generate")
    print(f"  Model: {MODEL}  Temps: {TEMPERATURES}  Samples/temp: {SAMPLES_PER_TEMP}")
    print(f"  max_tokens={MAX_TOKENS}  Expected cost: ~$6-11 (1500 calls, ~100 in + ~250 avg out tokens)")
    print()

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {
            ex.submit(process_one_question, (client, qid, row)): qid
            for qid, row in pending
        }
        for fut in as_completed(futures):
            qid, data = fut.result()
            question_data[qid] = data
            done += 1
            if done % checkpoint_every == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(pending) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(pending)}] "
                      f"{done * 15 / elapsed:.1f} gen/s | ETA {eta/60:.1f} min", flush=True)
                with open(out_path, "w") as f:
                    json.dump(question_data, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(question_data, f, indent=2)

    total_responses = sum(len(v.get("responses", [])) for v in question_data.values())
    total_errors = sum(
        1 for v in question_data.values()
        for r in v.get("responses", [])
        if r.get("stop_reason") == "error"
    )
    print()
    print(f"Saved to {out_path}")
    print(f"  {len(question_data)} questions, {total_responses} total responses, {total_errors} errors")

    # Spot-check output sanity
    sample_qid = list(question_data.keys())[0]
    sample = question_data[sample_qid]
    print(f"\n  First question (id {sample_qid}):")
    print(f"    Q: {sample['question'][:120]}")
    print(f"    First response: {sample['responses'][0]['text'][:150]!r}")
    print(f"    Stop reasons: {set(r.get('stop_reason') for r in sample['responses'])}")


if __name__ == "__main__":
    main()
