"""
Build NLI sim_cache for TruthfulQA and NQ-Open from cached response texts.

Inputs: modal_data/prereg_{truthfulqa,nq_open}_{Qwen,Mistral,Llama,Gemma}_*.json
Outputs: sim_cache/sim_matrices_{truthfulqa,nq_open}_{Model}.json
         (6x6 float matrices keyed by qid, matching existing format)

This does NOT regenerate LLM responses: those are cached in modal_data.
It only recomputes pairwise bidirectional DeBERTa-large-MNLI entailment.
"""

import glob
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

NLI_MODEL = "microsoft/deberta-large-mnli"
MAX_LEN = 256
N_RESPONSES = 6

SCRIPT_DIR = Path(__file__).parent
MODAL = SCRIPT_DIR / "modal_data"
SIM_CACHE = SCRIPT_DIR / "sim_cache"
SIM_CACHE.mkdir(exist_ok=True)

# Only the four base models reported in the main pre-registrations.
BASE_MODELS = ["Qwen", "Mistral", "Llama", "Gemma"]


def find_file(dataset, model):
    """Pick the file matching dataset + base model (exclude instruct/MoE variants)."""
    pattern = f"prereg_{dataset}_{model}_*.json"
    candidates = sorted(MODAL.glob(pattern))
    # Exclude LlamaInstruct, Gemma4MoE etc. (they'd match the glob for Llama/Gemma
    # because the glob uses _{model}_, but instruct suffixes aren't followed by _).
    # But e.g. 'Llama' glob matches 'Llama_' not 'LlamaInstruct_', so we're safe.
    return candidates[0] if candidates else None


def load_texts(dataset, model):
    f = find_file(dataset, model)
    if f is None:
        return None
    d = json.load(open(f))
    pq = d["per_question"]
    out = {}
    for qid, v in pq.items():
        texts = v.get("top6_texts", [])
        if len(texts) >= N_RESPONSES:
            out[qid] = texts[:N_RESPONSES]
    return out


def make_nli_sim(device):
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()

    @torch.no_grad()
    def nli_sim(text1, text2):
        if not text1.strip() or not text2.strip():
            return 0.0
        sims = []
        for premise, hypothesis in [(text1, text2), (text2, text1)]:
            inputs = tokenizer(
                premise, hypothesis,
                return_tensors="pt", truncation=True, max_length=MAX_LEN,
            ).to(device)
            logits = nli(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            sims.append(probs[0, 2].item())  # index 2 = entailment
        return float(np.mean(sims))

    return nli_sim


def sim_matrix(texts, nli_sim):
    n = len(texts)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s = nli_sim(texts[i], texts[j])
            M[i, j] = s
            M[j, i] = s
    return M


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading NLI model: {NLI_MODEL}")
    nli_sim = make_nli_sim(device)

    total_qs = 0
    for dataset in ["truthfulqa", "nq_open"]:
        for model in BASE_MODELS:
            out_file = SIM_CACHE / f"sim_matrices_{dataset}_{model}.json"
            if out_file.exists():
                print(f"  skip (exists): {out_file.name}")
                continue
            texts_by_q = load_texts(dataset, model)
            if texts_by_q is None:
                print(f"  no modal_data for {dataset}/{model}")
                continue
            print(f"  {dataset}/{model}: {len(texts_by_q)} questions")
            mats = {}
            for qid, texts in tqdm(texts_by_q.items(), desc=f"{dataset}/{model}"):
                M = sim_matrix(texts, nli_sim)
                mats[qid] = M.tolist()
            with open(out_file, "w") as f:
                json.dump(mats, f)
            print(f"  wrote {out_file.name}")
            total_qs += len(texts_by_q)

    print(f"\nDone. Total questions processed: {total_qs}")


if __name__ == "__main__":
    main()
