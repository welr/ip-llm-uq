"""
Seed-43 extension of the Sonnet 4.6 frontier run.

Reads frozen seed-43 indices from experiments/frontier_seed43_indices.json
(100 TruthfulQA validation indices disjoint from seed-42). Otherwise
identical protocol to frontier_sonnet_run.py (max_tokens=500,
top_p unset, temps {0.3, 0.5, 0.7, 0.9, 1.0}, 3 samples per temp).

Output:
  experiments/modal_data/frontier_sonnet_truthfulqa_seed43.json

Usage:
    ~/ip_llm/bin/python experiments/frontier_sonnet_run_seed43.py
    (reads ANTHROPIC_API_KEY via ~/.anthropic-env; never printed)
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from anthropic import Anthropic
from datasets import load_dataset


MODEL = "claude-sonnet-4-6"
TEMPERATURES = [0.3, 0.5, 0.7, 0.9, 1.0]
SAMPLES_PER_TEMP = 3
MAX_TOKENS = 500
WORKERS = 8


def _load_env():
    envfile = Path(os.path.expanduser("~/.anthropic-env"))
    if not envfile.exists():
        sys.exit("ERROR: ~/.anthropic-env not found")
    for line in envfile.read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ[k] = v.strip().strip('"').strip("'")


def sample_question(client, question_text, max_retries=8):
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
                    err = str(e); last_err = err
                    if "429" in err or "rate_limit" in err.lower():
                        time.sleep(30 + 10 * attempt)
                    elif "529" in err or "overloaded" in err.lower():
                        time.sleep(20 + 10 * attempt)
                    else:
                        time.sleep(min(2 ** attempt, 30))
            else:
                responses.append({"text": "", "temp": temp, "stop_reason": "error", "error": last_err})
    return responses


def process_one(args):
    client, qid, row = args
    responses = sample_question(client, row["question"])
    return qid, {
        "question": row["question"],
        "question_raw": row["question"],
        "gold_answers": list(row.get("correct_answers", [])) or [row.get("best_answer", "")],
        "tqa_type": row.get("type", ""),
        "tqa_category": row.get("category", ""),
        "tqa_incorrect_answers": list(row.get("incorrect_answers", [])),
        "responses": responses,
    }


def main():
    _load_env()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=api_key)

    script_dir = Path(__file__).parent
    idx_path = script_dir / "frontier_seed43_indices.json"
    out_path = script_dir / "modal_data" / "frontier_sonnet_truthfulqa_seed43.json"

    idx_data = json.load(open(idx_path))
    indices = idx_data["indices"]
    print(f"Loaded {len(indices)} seed-43 indices (disjoint from seed-42)")

    ds = load_dataset("truthful_qa", "generation", split="validation")

    question_data = {}
    if out_path.exists():
        question_data = json.load(open(out_path))
        print(f"  Resumed with {len(question_data)} done")

    pending = []
    for i in indices:
        qid = f"tqa_{i}"
        if qid in question_data and len(question_data[qid].get("responses", [])) >= 15:
            continue
        pending.append((qid, ds[int(i)]))

    print(f"  Pending: {len(pending)}")
    print(f"  Model: {MODEL}  max_tokens={MAX_TOKENS}")
    print(f"  Expected cost: ~$5 (1500 calls)")

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_one, (client, qid, row)): qid for qid, row in pending}
        for fut in as_completed(futs):
            qid, data = fut.result()
            question_data[qid] = data
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t0; rate = done / elapsed
                eta = (len(pending) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(pending)}] {done*15/elapsed:.1f} gen/s | ETA {eta/60:.1f} min", flush=True)
                with open(out_path, "w") as f:
                    json.dump(question_data, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(question_data, f, indent=2)

    total = sum(len(v.get("responses", [])) for v in question_data.values())
    errs = sum(1 for v in question_data.values() for r in v.get("responses", []) if r.get("stop_reason") == "error")
    print(f"\nSaved to {out_path}")
    print(f"  {len(question_data)} questions, {total} responses, {errs} errors")


if __name__ == "__main__":
    main()
