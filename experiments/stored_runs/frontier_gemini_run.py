"""
Exploratory frontier-scale scope extension: Gemini 2.5 Flash on TruthfulQA.

Companion to frontier_sonnet_run.py. Addresses the peer-reviewer ask:
"distinguish Sonnet-specific from frontier-RLHF-generic."

Protocol (matches Sonnet run as closely as the Google API allows):
  - Exactly the same 100 seed-42 TruthfulQA validation questions as Sonnet
  - 15 responses per question: 5 temperatures * 3 samples
  - Temperatures {0.3, 0.5, 0.7, 0.9, 1.0}
  - top_p: unset (match Sonnet)
  - max_output_tokens = 1500 (probe showed Flash's natural essay length is
      ~700 tokens with thinking disabled; 1500 gives ~2x margin)
  - thinking_budget = 0 (Gemini 2.5 Flash "thinking" consumes the token
      budget invisibly; disabled for parity with Sonnet/base protocol)
  - No system prompt
  - Model: gemini-2.5-flash

Output:
  experiments/modal_data/frontier_gemini_truthfulqa.json
    Same schema as frontier_sonnet_truthfulqa.json so downstream analysis
    scripts work unchanged.

Usage:
    ~/ip_llm/bin/python experiments/frontier_gemini_run.py
    (reads GEMINI_API_KEY from ~/.gemini-env; never printed)
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import load_dataset
from google import genai
from google.genai import types as gtypes


MODEL = "gemini-2.5-flash"
N_QUESTIONS = 100
SEED = 42
TEMPERATURES = [0.3, 0.5, 0.7, 0.9, 1.0]
SAMPLES_PER_TEMP = 3
MAX_OUTPUT_TOKENS = 1500
WORKERS = 16  # paid tier supports 2000 RPM; 16 * ~4 call/s = well under quota


def _load_env():
    """Read GEMINI_API_KEY from ~/.gemini-env without shell interpolation."""
    envfile = Path(os.path.expanduser("~/.gemini-env"))
    if not envfile.exists():
        sys.exit("ERROR: ~/.gemini-env not found")
    for line in envfile.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ[k] = v.strip().strip('"').strip("'")


def sample_question(client, question_text, max_retries=8):
    """Generate 15 responses (5 temps * 3 samples)."""
    responses = []
    for temp in TEMPERATURES:
        for _ in range(SAMPLES_PER_TEMP):
            last_err = None
            for attempt in range(max_retries):
                try:
                    cfg = gtypes.GenerateContentConfig(
                        temperature=temp,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
                    )
                    r = client.models.generate_content(
                        model=MODEL,
                        contents=question_text,
                        config=cfg,
                    )
                    text = r.text or ""
                    fr = str(r.candidates[0].finish_reason).split(".")[-1] if r.candidates else "UNKNOWN"
                    responses.append({
                        "text": text,
                        "temp": temp,
                        "stop_reason": fr,
                    })
                    break
                except Exception as e:
                    err = str(e)
                    last_err = err
                    if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate" in err.lower():
                        wait = 30 + 5 * attempt
                        print(f"  rate-limit, sleep {wait}s", file=sys.stderr, flush=True)
                        time.sleep(wait)
                    elif "503" in err or "UNAVAILABLE" in err or "overloaded" in err.lower():
                        wait = 20 + 5 * attempt
                        print(f"  API overloaded, sleep {wait}s", file=sys.stderr, flush=True)
                        time.sleep(wait)
                    else:
                        time.sleep(min(2 ** attempt, 30))
            else:
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
    _load_env()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

    script_dir = Path(__file__).parent
    out_path = script_dir / "modal_data" / "frontier_gemini_truthfulqa.json"
    checkpoint_every = 10

    # Load TruthfulQA and sample the same 100 indices as the Sonnet run
    print(f"Loading TruthfulQA generation/validation...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    rng = np.random.default_rng(SEED)
    all_indices = rng.choice(len(ds), size=N_QUESTIONS, replace=False).tolist()
    all_indices = sorted(all_indices)
    print(f"  Seed {SEED}: {len(all_indices)} questions sampled from {len(ds)}")

    # Sanity check: the same sampling as Sonnet
    sonnet_path = script_dir / "modal_data" / "frontier_sonnet_truthfulqa.json"
    if sonnet_path.exists():
        sonnet = json.load(open(sonnet_path))
        sonnet_qids = sorted(sonnet.keys())
        my_qids = sorted(f"tqa_{i}" for i in all_indices)
        if sonnet_qids != my_qids:
            print(f"  WARN: qid mismatch vs Sonnet run ({len(set(my_qids) ^ set(sonnet_qids))} differ)")
        else:
            print(f"  qids match Sonnet run exactly")

    # Resume
    question_data = {}
    if out_path.exists():
        question_data = json.load(open(out_path))
        print(f"  Resumed with {len(question_data)} already-done questions")

    pending = []
    for idx in all_indices:
        qid = f"tqa_{idx}"
        if qid in question_data and len(question_data[qid].get("responses", [])) >= 15:
            continue
        pending.append((qid, ds[int(idx)]))

    print(f"  {len(pending)} questions to generate")
    print(f"  Model: {MODEL}  Temps: {TEMPERATURES}  Samples/temp: {SAMPLES_PER_TEMP}")
    print(f"  max_output_tokens={MAX_OUTPUT_TOKENS}  thinking=disabled")
    print(f"  Expected cost: ~$0.50-1.00 (1500 calls, ~10 in + ~700 out tokens)")
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

    # Spot-check
    sample_qid = list(question_data.keys())[0]
    sample = question_data[sample_qid]
    print(f"\n  First question ({sample_qid}):")
    print(f"    Q: {sample['question'][:120]}")
    print(f"    First response: {sample['responses'][0]['text'][:150]!r}")
    print(f"    Stop reasons: {set(r.get('stop_reason') for r in sample['responses'])}")


if __name__ == "__main__":
    main()
