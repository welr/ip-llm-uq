"""
Pre-registered experiments: Choice function admissibility on MMLU,
ARC-Challenge, TruthfulQA, and Natural Questions (NQ-Open).

OSF Registrations:
    - MMLU + ARC:       osf.io/ady3k (Apr 9, 2026)
    - TruthfulQA:       osf.io/vze74 (Apr 15, 2026)
    - Natural Questions: osf.io/78pb6 (Apr 18, 2026)

Runs all 4 base models on the pre-registered question sample for each
dataset, generates responses, computes NLI-based and string-based
admissibility, evaluates accuracy, and compares against heuristic filters.

Run on Modal:
    modal run run_preregistered.py --dataset mmlu
    modal run run_preregistered.py --dataset arc
    modal run run_preregistered.py --dataset truthfulqa
    modal run run_preregistered.py --dataset nq_open
"""

import modal

app = modal.App("choice-functions-preregistered")
volume = modal.Volume.from_name("choice-fn-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.50.0",  # Gemma-4 MoE support (released Apr 2026)
        "accelerate",
        "datasets",
        "numpy",
        "scipy",
        "tqdm",
        "sentencepiece",
        "protobuf",
    )
)

# Base models are the pre-registered four. Instruct / Gemma-4 entries are
# scope-extension runs added post-hoc in response to reviewer feedback and
# are clearly labelled as exploratory (not pre-registered) in the paper.
MODELS = {
    "Qwen": {
        "name": "Qwen/Qwen2.5-7B",
    },
    "Mistral": {
        "name": "mistralai/Mistral-7B-v0.1",
    },
    "Llama": {
        "name": "meta-llama/Llama-3.1-8B",
    },
    "Gemma": {
        "name": "google/gemma-2-9b",
        "dtype": "bfloat16",
    },
    "LlamaInstruct": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "instruct": True,
        "max_new_tokens": 100,
    },
    "Gemma4MoE": {
        # Gemma-4 26B MoE, instruction-tuned. "A4B" = ~4B active parameters
        # per token (26B total). Per-token compute is close to a 4B dense
        # model; full 26B in memory. Released 2026-04-02, Apache 2.0.
        "name": "google/gemma-4-26B-A4B-it",
        "dtype": "bfloat16",
        "instruct": True,
        "max_new_tokens": 100,
        "trust_remote_code": True,  # new architecture; code may not yet be in transformers
    },
}

NLI_MODEL = "microsoft/deberta-large-mnli"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TEMPERATURES = [0.3, 0.5, 0.7, 1.0, 1.3]
SAMPLES_PER_TEMP = 3
MAX_NEW_TOKENS = 50
N_RESPONSES = 6
SEED = 42
QUESTION_SEED = 123
# NQ-Open pre-registration uses a separate seed (42) via numpy's default_rng,
# frozen to experiments/nq_open_indices.json at registration time. This is
# kept distinct from QUESTION_SEED (=123, np.random.RandomState) used by the
# earlier MMLU/ARC pre-registration to preserve the reproducibility of those
# prior runs and match the NQ-Open pre-registration (osf.io/[NQ]) protocol.
NQ_QUESTION_SEED = 42

# Per-dataset question counts. TruthfulQA uses the full 817-question
# validation split (no sub-sampling); MMLU, ARC, and NQ-Open use
# pre-registered samples.
N_QUESTIONS_DEFAULT = 500
N_QUESTIONS_BY_DATASET = {
    "mmlu": 500,
    "arc": 500,
    "truthfulqa": 817,   # full validation split
    "nq_open": 500,      # random sub-sample with seed 42, indices frozen
}


def sample_mmlu(n: int, seed: int):
    """Sample n questions from MMLU test split, stratified across subjects."""
    from datasets import load_dataset
    import numpy as np

    rng = np.random.RandomState(seed)
    ds = load_dataset("cais/mmlu", "all", split="test")

    # Group by subject
    by_subject = {}
    for i, row in enumerate(ds):
        subj = row["subject"]
        by_subject.setdefault(subj, []).append(i)

    subjects = sorted(by_subject.keys())
    n_subjects = len(subjects)
    per_subject = max(1, n // n_subjects)

    sampled_indices = []
    for subj in subjects:
        indices = by_subject[subj]
        k = min(per_subject, len(indices))
        sampled_indices.extend(rng.choice(indices, size=k, replace=False).tolist())

    # If we need more to reach n, sample from remaining
    remaining = set(range(len(ds))) - set(sampled_indices)
    if len(sampled_indices) < n and remaining:
        extra = rng.choice(list(remaining), size=n - len(sampled_indices), replace=False)
        sampled_indices.extend(extra.tolist())

    sampled_indices = sampled_indices[:n]
    rng.shuffle(sampled_indices)

    questions = []
    choices_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    for idx in sampled_indices:
        row = ds[int(idx)]
        # Format question as open-ended
        q_text = row["question"]
        # Include choices for context but ask for the answer
        choice_strs = [f"{choices_map[i]}. {row['choices'][i]}" for i in range(4)]
        full_q = q_text + "\n" + "\n".join(choice_strs)
        correct_idx = row["answer"]
        gold_answer = row["choices"][correct_idx]
        gold_letter = choices_map[correct_idx]
        questions.append({
            "id": f"mmlu_{idx}",
            "question": full_q,
            "question_raw": q_text,
            "gold_answers": [gold_answer, gold_letter],
            "subject": row["subject"],
        })

    return questions


def sample_arc(n: int, seed: int):
    """Sample n questions from ARC-Challenge test split."""
    from datasets import load_dataset
    import numpy as np

    rng = np.random.RandomState(seed)
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist()

    questions = []
    for idx in indices:
        row = ds[idx]
        q_text = row["question"]
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        choice_strs = [f"{labels[i]}. {texts[i]}" for i in range(len(labels))]
        full_q = q_text + "\n" + "\n".join(choice_strs)

        answer_key = row["answerKey"]
        gold_text = ""
        for i, lbl in enumerate(labels):
            if lbl == answer_key:
                gold_text = texts[i]
                break
        questions.append({
            "id": f"arc_{row['id']}",
            "question": full_q,
            "question_raw": q_text,
            "gold_answers": [gold_text, answer_key],
        })

    return questions[:n]


def sample_truthfulqa(n: int, seed: int):
    """Return TruthfulQA questions (generation / validation split).

    TruthfulQA is free-form by design: each item has a question and lists of
    correct and incorrect answers. We present the question directly (no
    choices), use ``correct_answers`` as the gold set, and preserve ``type``
    (Adversarial vs Non-Adversarial), ``category``, and ``incorrect_answers``
    for the exploratory analyses pre-registered in
    ``preregistration_truthfulqa.md``.

    The pre-registration specifies the **full validation split** (n = 817,
    no sub-sampling). If ``n >= len(ds)``, we use every question in the
    original order and the ``seed`` argument is ignored. If ``n < len(ds)``
    (only used for testing), we fall back to a seeded random sample.
    """
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset("truthful_qa", "generation", split="validation")

    if n >= len(ds):
        indices = list(range(len(ds)))
    else:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(ds), size=n, replace=False).tolist()
        indices = sorted(indices)  # stable ordering for reproducibility

    questions = []
    for idx in indices:
        row = ds[int(idx)]
        q_text = row["question"]
        # Use correct_answers as gold set; fall back to best_answer if empty
        gold = list(row.get("correct_answers", []))
        if not gold:
            gold = [row.get("best_answer", "")]
        questions.append({
            "id": f"tqa_{idx}",
            "question": q_text,       # TruthfulQA is already free-form
            "question_raw": q_text,
            "gold_answers": gold,
            "tqa_type": row.get("type", ""),
            "tqa_category": row.get("category", ""),
            "tqa_incorrect_answers": list(row.get("incorrect_answers", [])),
        })

    return questions


def sample_nq_open(n: int, seed: int):
    """Return Natural Questions (NQ-Open) questions.

    NQ-Open (Roberts et al., 2020; underlying dataset from Kwiatkowski et al.,
    2019) is an open-domain short-answer factual QA benchmark derived from
    Google search queries. Each item has a question and a list of acceptable
    short answers.

    The pre-registration (osf.io/78pb6) specifies a 500-question random
    sub-sample of the 3,610-question validation split, drawn with
    numpy.random.default_rng(42). The canonical index list is frozen in
    experiments/nq_open_indices.json, committed before the OSF timestamp.

    This function regenerates those indices deterministically from the same
    (dataset, seed) pair. On a local run with the indices file available, it
    optionally asserts identity against the frozen list.
    """
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset("google-research-datasets/nq_open", split="validation")

    # Regenerate the pre-registered index list with the fixed seed.
    # This must match experiments/nq_open_indices.json exactly; if it ever
    # fails to, the pre-registration timestamp is the authoritative artefact.
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist())

    # Optional verification against the frozen canonical list when present
    # (primarily for local dev runs; on Modal the file is not mounted).
    try:
        import os
        canonical_path = os.path.join(
            os.path.dirname(__file__), "nq_open_indices.json"
        )
        if os.path.exists(canonical_path):
            with open(canonical_path) as f:
                canonical = json.load(f)
            if canonical.get("indices") != indices:
                raise RuntimeError(
                    "NQ-Open indices do not match the frozen pre-registered "
                    f"list at {canonical_path}. The pre-registered sample is "
                    "authoritative; investigate before proceeding."
                )
    except (ImportError, OSError):
        pass  # file not accessible at Modal runtime; rely on seed determinism

    questions = []
    for idx in indices:
        row = ds[int(idx)]
        q_text = row["question"]
        # NQ-Open gold answers: `answer` is a list of acceptable short strings
        gold = list(row.get("answer", []))
        if not gold:
            gold = [""]
        questions.append({
            "id": f"nq_{idx}",
            "question": q_text,       # NQ-Open is already free-form
            "question_raw": q_text,
            "gold_answers": gold,
        })

    return questions


@app.function(
    image=image,
    volumes={"/results": volume},
    # H100 (always 80GB) required for Gemma-4-26B-A4B-it, which does not fit
    # in the 40GB A100 the "A100" default resolves to on Modal. The 7-9B
    # base models and Llama-3.1-8B-Instruct fit in 40GB but also run fine on
    # H100 (faster). Using H100 uniformly keeps the pipeline simple.
    gpu="H100",
    timeout=28800,  # 8 hours
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def run_model(model_key: str, dataset: str):
    """Run inference + NLI analysis for one model on one dataset."""
    import os
    import json
    import re
    from datetime import datetime
    from collections import Counter, defaultdict

    import numpy as np
    from scipy import stats
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        AutoModelForSequenceClassification,
    )
    from tqdm import tqdm

    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    model_info = MODELS[model_key]
    model_name = model_info["name"]
    dtype = getattr(torch, model_info.get("dtype", "float16"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"{'='*60}")
    print(f"Model: {model_key} ({model_name})")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")

    # ============================================================
    # HELPERS
    # ============================================================

    def normalize_answer(answer):
        if not answer:
            return ""
        answer = answer.lower().strip()
        answer = re.sub(r'[^\w\s]', '', answer)
        answer = re.sub(r'\s+', ' ', answer)
        # 500-char cap (was 100). Base-model responses are post-processed
        # to a single sentence and rarely exceed ~40 chars, so the pre-
        # registered four-model results are unaffected. Instruct-tuned
        # models return whole-paragraph replies up to ~400 chars where the
        # actual answer can appear past char 100; the looser cap ensures
        # the substring match in check_answer still finds it.
        return answer[:500]

    def normalize_aggressive(answer):
        norm = normalize_answer(answer)
        norm = re.sub(r'^(the|a|an)\s+', '', norm)
        norm = re.sub(r'^(it is|its|they are|he is|she is|this is)\s+', '', norm)
        return norm.strip()

    def check_answer(generated, gold_answers):
        if not generated:
            return False
        gen_norm = normalize_answer(generated)
        if not gen_norm:
            return False
        for gold in gold_answers:
            gold_norm = normalize_answer(gold)
            if gold_norm and (gold_norm in gen_norm or gen_norm in gold_norm):
                return True
        return False

    def string_similarity(s1, s2):
        n1, n2 = normalize_answer(s1), normalize_answer(s2)
        if n1 == n2:
            return "exact"
        a1, a2 = normalize_aggressive(s1), normalize_aggressive(s2)
        if a1 == a2:
            return "normalized"
        if a1 and a2 and (a1 in a2 or a2 in a1):
            return "substring"
        return "distinct"

    STRING_THRESHOLDS = ["exact", "normalized", "substring"]

    def string_cluster(responses, threshold):
        n = len(responses)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        levels = {
            "exact": {"exact"},
            "normalized": {"exact", "normalized"},
            "substring": {"exact", "normalized", "substring"},
        }
        merge_at = levels[threshold]
        for i in range(n):
            for j in range(i + 1, n):
                if string_similarity(responses[i], responses[j]) in merge_at:
                    union(i, j)
        clusters = defaultdict(set)
        for i in range(n):
            clusters[find(i)].add(i)
        return list(clusters.values())

    # ============================================================
    # SAMPLE QUESTIONS
    # ============================================================

    n_questions = N_QUESTIONS_BY_DATASET.get(dataset, N_QUESTIONS_DEFAULT)
    print(f"Sampling {n_questions} questions (seed={QUESTION_SEED})...")
    if dataset == "mmlu":
        questions = sample_mmlu(n_questions, QUESTION_SEED)
    elif dataset == "arc":
        questions = sample_arc(n_questions, QUESTION_SEED)
    elif dataset == "truthfulqa":
        questions = sample_truthfulqa(n_questions, QUESTION_SEED)
    elif dataset == "nq_open":
        questions = sample_nq_open(n_questions, NQ_QUESTION_SEED)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    print(f"  Sampled {len(questions)} questions")

    # ============================================================
    # CHECKPOINT: load existing partial results if any
    # ============================================================

    CHECKPOINT_FILE = f"/results/checkpoint_gen_{dataset}_{model_key}.json"
    # Generation checkpoint frequency. At ~5-7s/question on H100, 10
    # questions = ~1 min of work at risk on a crash. Volume.commit()
    # overhead (~1-2s) is negligible relative to the safety gain.
    CHECKPOINT_EVERY = 10  # questions

    question_data = {}
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Found checkpoint: {CHECKPOINT_FILE}")
        with open(CHECKPOINT_FILE) as f:
            question_data = json.load(f)
        print(f"  Resumed with {len(question_data)} questions already generated")
    else:
        print("No checkpoint found; starting fresh")

    questions_remaining = [q for q in questions if q["id"] not in question_data]
    print(f"  {len(questions_remaining)} questions remaining to generate")

    # ============================================================
    # LOAD LLM (skip if all questions already generated)
    # ============================================================

    is_instruct = bool(model_info.get("instruct", False))
    trust_remote = bool(model_info.get("trust_remote_code", False))
    gen_max_new_tokens = int(model_info.get("max_new_tokens", MAX_NEW_TOKENS))

    if questions_remaining:
        print(f"Loading {model_name}... (instruct={is_instruct}, max_new_tokens={gen_max_new_tokens})")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=HF_TOKEN, trust_remote_code=trust_remote,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=HF_TOKEN, torch_dtype=dtype, device_map="auto",
            trust_remote_code=trust_remote,
        )
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def generate_response(prompt, temperature):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_max_new_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            input_len = inputs['input_ids'].shape[1]
            response = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
            if is_instruct:
                # Chat-tuned models return clean replies; don't chop at first
                # period (loses cases like "The answer is Astana.").
                response = response.strip()
            else:
                # Base models ramble; keep the first line / first sentence as
                # in the pre-registered protocol for the original four models.
                response = response.split('\n')[0].split('.')[0].strip()
            return response

        # ============================================================
        # GENERATE RESPONSES (with periodic checkpointing)
        # ============================================================

        print("Generating responses...")

        def make_prompt(question_text):
            if is_instruct:
                user_msg = f"Answer the following question concisely:\n\n{question_text}"
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return f"""Answer the following question concisely.

Q: {question_text}
A:"""

        for idx, q in enumerate(tqdm(questions_remaining, desc=f"  {model_key} generation")):
            qid = q["id"]
            prompt = make_prompt(q["question"])

            responses = []
            torch.manual_seed(SEED)
            for temp in TEMPERATURES:
                for _ in range(SAMPLES_PER_TEMP):
                    resp = generate_response(prompt, temp)
                    responses.append({"text": resp, "temp": temp})

            question_data[qid] = {
                "question": q["question"],
                "question_raw": q["question_raw"],
                "gold_answers": q["gold_answers"],
                "subject": q.get("subject", ""),
                "tqa_type": q.get("tqa_type", ""),
                "tqa_category": q.get("tqa_category", ""),
                "tqa_incorrect_answers": q.get("tqa_incorrect_answers", []),
                "responses": responses,
            }

            # Checkpoint every N questions
            if (idx + 1) % CHECKPOINT_EVERY == 0:
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(question_data, f)
                volume.commit()

        # Final checkpoint after generation
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(question_data, f)
        volume.commit()
        print(f"Generation complete: {len(question_data)} questions")
    else:
        print("All questions already generated from checkpoint; skipping LLM load")

    # ============================================================
    # FREE LLM, LOAD NLI
    # ============================================================

    if questions_remaining:
        del model
        del tokenizer
        torch.cuda.empty_cache()

    print(f"\nLoading NLI model ({NLI_MODEL})...")
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    nli_model = nli_model.cuda().eval()

    def compute_nli_similarity(text1, text2):
        if not text1.strip() or not text2.strip():
            return 0.0
        sims = []
        for premise, hypothesis in [(text1, text2), (text2, text1)]:
            inputs = nli_tokenizer(
                premise, hypothesis,
                return_tensors="pt", truncation=True, max_length=256,
            ).to("cuda")
            with torch.no_grad():
                outputs = nli_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                sims.append(probs[0, 2].item())
        return np.mean(sims)

    # ============================================================
    # COMPUTE ADMISSIBILITY + FILTERS (with checkpointing)
    # ============================================================

    print("Computing NLI similarity, admissibility, and filters...")

    ANALYSIS_CHECKPOINT_FILE = f"/results/checkpoint_analysis_{dataset}_{model_key}.json"
    # Analysis is I/O and CPU-light relative to generation; checkpoint more
    # frequently so a crash loses at most ~25 questions of NLI work.
    ANALYSIS_CHECKPOINT_EVERY = 25  # questions

    results = {}
    if os.path.exists(ANALYSIS_CHECKPOINT_FILE):
        print(f"Found analysis checkpoint: {ANALYSIS_CHECKPOINT_FILE}")
        with open(ANALYSIS_CHECKPOINT_FILE) as f:
            results = json.load(f)
        print(f"  Resumed with {len(results)} questions already analyzed")

    qids_to_analyze = [q for q in question_data.keys() if q not in results]
    print(f"  {len(qids_to_analyze)} questions remaining to analyze")

    for idx, qid in enumerate(tqdm(qids_to_analyze, desc=f"  {model_key} analysis")):
        qd = question_data[qid]
        gold_answers = qd["gold_answers"]
        all_responses = qd["responses"]

        # Select top 6 (first sample from each temp + one extra)
        top6_texts = [r["text"] for r in all_responses[:N_RESPONSES]]
        accuracies = [check_answer(t, gold_answers) for t in top6_texts]
        n = len(top6_texts)

        # --- NLI-based admissibility ---
        sim_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                sim = compute_nli_similarity(top6_texts[i], top6_texts[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        nli_admissible = set()
        se_values = []
        for theta in THRESHOLDS:
            parent = list(range(n))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i, j] >= theta:
                        union(i, j)

            clusters = defaultdict(set)
            for i in range(n):
                clusters[find(i)].add(i)
            clusters = list(clusters.values())

            cluster_probs = [len(c) / n for c in clusters]
            max_prob = max(cluster_probs)
            for c, p in zip(clusters, cluster_probs):
                if p == max_prob:
                    nli_admissible.update(c)

            se = -sum(p * np.log2(p) for p in cluster_probs if p > 0)
            se_values.append(se)

        nli_rejected = set(range(n)) - nli_admissible

        # --- String-based admissibility ---
        str_admissible = set()
        for threshold in STRING_THRESHOLDS:
            clusters = string_cluster(top6_texts, threshold)
            max_size = max(len(c) for c in clusters)
            for c in clusters:
                if len(c) == max_size:
                    str_admissible.update(c)
        str_rejected = set(range(n)) - str_admissible

        # --- Heuristic filters ---
        normalized = [normalize_answer(t) for t in top6_texts]
        aggressive = [normalize_aggressive(t) for t in top6_texts]

        freq_counts = Counter(normalized)
        freq_rejected = {i for i, nm in enumerate(normalized) if freq_counts[nm] <= 1}

        cons_counts = Counter(aggressive)
        cons_rejected = {i for i, ag in enumerate(aggressive) if cons_counts[ag] <= 1}

        len_rejected = {i for i, t in enumerate(top6_texts) if len(t.strip().split()) < 2}

        # --- Majority vote ---
        counts = Counter(normalized)
        mv_winner = counts.most_common(1)[0][0]
        mv_correct = False
        for i, t in enumerate(top6_texts):
            if normalize_answer(t) == mv_winner:
                mv_correct = accuracies[i]
                break

        results[qid] = {
            "accuracies": accuracies,
            "nli_admissible": sorted(nli_admissible),
            "nli_rejected": sorted(nli_rejected),
            "str_admissible": sorted(str_admissible),
            "str_rejected": sorted(str_rejected),
            "freq_rejected": sorted(freq_rejected),
            "cons_rejected": sorted(cons_rejected),
            "len_rejected": sorted(len_rejected),
            "mv_correct": mv_correct,
            "se_lower": min(se_values),
            "se_upper": max(se_values),
            "subject": qd.get("subject", ""),
            "tqa_type": qd.get("tqa_type", ""),
            "tqa_category": qd.get("tqa_category", ""),
            "top6_texts": top6_texts,
        }

        # Checkpoint analysis every N questions
        if (idx + 1) % ANALYSIS_CHECKPOINT_EVERY == 0:
            with open(ANALYSIS_CHECKPOINT_FILE, 'w') as f:
                json.dump(results, f, default=str)
            volume.commit()

    # Final analysis checkpoint
    with open(ANALYSIS_CHECKPOINT_FILE, 'w') as f:
        json.dump(results, f, default=str)
    volume.commit()

    # ============================================================
    # AGGREGATE
    # ============================================================

    def compute_filter_stats(results, adm_key, rej_key):
        adm_correct = sum(
            r["accuracies"][i]
            for r in results.values()
            for i in r[adm_key]
        )
        adm_total = sum(len(r[adm_key]) for r in results.values())
        rej_correct = sum(
            r["accuracies"][i]
            for r in results.values()
            for i in r[rej_key]
        )
        rej_total = sum(len(r[rej_key]) for r in results.values())
        return adm_correct, adm_total, rej_correct, rej_total

    def compute_heuristic_stats(results, rejected_key):
        n_total = sum(len(r["accuracies"]) for r in results.values())
        retained_correct = sum(
            r["accuracies"][i]
            for r in results.values()
            for i in range(len(r["accuracies"]))
            if i not in set(r[rejected_key])
        )
        retained_total = n_total - sum(len(r[rejected_key]) for r in results.values())
        rej_correct = sum(
            r["accuracies"][i]
            for r in results.values()
            for i in r[rejected_key]
        )
        rej_total = sum(len(r[rejected_key]) for r in results.values())
        return retained_correct, retained_total, rej_correct, rej_total

    nli_ac, nli_at, nli_rc, nli_rt = compute_filter_stats(
        results, "nli_admissible", "nli_rejected"
    )
    str_ac, str_at, str_rc, str_rt = compute_filter_stats(
        results, "str_admissible", "str_rejected"
    )
    freq_rc_c, freq_rt_t, freq_rj_c, freq_rj_t = compute_heuristic_stats(
        results, "freq_rejected"
    )
    cons_rc_c, cons_rt_t, cons_rj_c, cons_rj_t = compute_heuristic_stats(
        results, "cons_rejected"
    )
    len_rc_c, len_rt_t, len_rj_c, len_rj_t = compute_heuristic_stats(
        results, "len_rejected"
    )

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    summary = {
        "model": model_key,
        "dataset": dataset,
        "n_questions": len(results),
        "nli_admissible_acc": safe_div(nli_ac, nli_at),
        "nli_rejected_acc": safe_div(nli_rc, nli_rt),
        "nli_n_admissible": nli_at,
        "nli_n_rejected": nli_rt,
        "nli_ratio": safe_div(safe_div(nli_ac, nli_at), safe_div(nli_rc, nli_rt))
            if nli_rt > 0 else float('inf'),
        "str_admissible_acc": safe_div(str_ac, str_at),
        "str_rejected_acc": safe_div(str_rc, str_rt),
        "str_n_rejected": str_rt,
        "str_ratio": safe_div(safe_div(str_ac, str_at), safe_div(str_rc, str_rt))
            if str_rt > 0 else float('inf'),
        "freq_rejected_acc": safe_div(freq_rj_c, freq_rj_t),
        "freq_n_rejected": freq_rj_t,
        "cons_rejected_acc": safe_div(cons_rj_c, cons_rj_t),
        "cons_n_rejected": cons_rj_t,
        "mv_accuracy": np.mean([r["mv_correct"] for r in results.values()]),
        "random_accuracy": np.mean(
            [np.mean(r["accuracies"]) for r in results.values()]
        ),
        "mean_se_width": np.mean(
            [r["se_upper"] - r["se_lower"] for r in results.values()]
        ),
    }

    # Fisher's exact test for NLI admissibility
    table = np.array([
        [nli_ac, nli_at - nli_ac],
        [nli_rc, max(nli_rt - nli_rc, 0)],
    ])
    if nli_rt > 0:
        _, p_value = stats.fisher_exact(table, alternative='two-sided')
        summary["fisher_p"] = p_value
    else:
        summary["fisher_p"] = None

    print(f"\n{'='*60}")
    print(f"RESULTS: {model_key} on {dataset}")
    print(f"{'='*60}")
    print(f"  NLI admissible acc:  {summary['nli_admissible_acc']:.1%}")
    print(f"  NLI rejected acc:    {summary['nli_rejected_acc']:.1%}")
    print(f"  NLI ratio:           {summary['nli_ratio']:.2f}x")
    print(f"  NLI rejected count:  {summary['nli_n_rejected']}")
    print(f"  Fisher p:            {summary['fisher_p']}")
    print(f"  String ratio:        {summary['str_ratio']:.2f}x")
    print(f"  Random accuracy:     {summary['random_accuracy']:.1%}")
    print(f"  Majority vote:       {summary['mv_accuracy']:.1%}")
    print(f"  Mean SE width:       {summary['mean_se_width']:.3f} bits")

    # Save
    output = {
        "timestamp": timestamp,
        "summary": {k: float(v) if isinstance(v, (np.floating, float)) else v
                     for k, v in summary.items()},
        "per_question": {
            qid: {k: (float(v) if isinstance(v, (np.floating, float))
                       else [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
                       if isinstance(v, list) else v)
                   for k, v in r.items()}
            for qid, r in results.items()
        },
    }
    outfile = f"/results/prereg_{dataset}_{model_key}_{timestamp}.json"
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    volume.commit()
    print(f"  Saved to {outfile}")

    return summary


@app.local_entrypoint()
def main(dataset: str = "mmlu", models: str = ""):
    import json

    if dataset not in ("mmlu", "arc", "truthfulqa", "nq_open"):
        raise ValueError(
            f"Dataset must be 'mmlu', 'arc', 'truthfulqa', or 'nq_open', "
            f"got '{dataset}'"
        )

    # Pre-registered base models (Qwen/Mistral/Llama/Gemma) are the default
    # for all pre-registered datasets. Instruct-tuned models (LlamaInstruct,
    # Gemma4MoE) were added post-hoc for the TruthfulQA scope-extension
    # analysis and must be opted into explicitly via --models.
    BASE_MODELS = ["Qwen", "Mistral", "Llama", "Gemma"]

    # Filter models if specified (comma-separated list)
    if models:
        model_keys = [m.strip() for m in models.split(",")]
        for m in model_keys:
            if m not in MODELS:
                raise ValueError(f"Unknown model '{m}'. Available: {list(MODELS.keys())}")
    else:
        model_keys = BASE_MODELS

    print(f"Pre-registered experiment: {dataset.upper()}")
    print(f"Models: {model_keys}")
    print()

    print(f"Spawning {len(model_keys)} models in parallel...")
    handles = {
        model_key: run_model.spawn(model_key, dataset)
        for model_key in model_keys
    }

    all_summaries = {}
    for model_key, handle in handles.items():
        print(f"\n>>> Waiting for {model_key}...")
        all_summaries[model_key] = handle.get()

    # Aggregate results
    print("\n" + "=" * 80)
    print(f"AGGREGATE RESULTS: {dataset.upper()}")
    print("=" * 80)

    total_nli_adm_correct = 0
    total_nli_adm = 0
    total_nli_rej_correct = 0
    total_nli_rej = 0

    print(f"{'Model':<10} {'NLI Adm':>8} {'NLI Rej':>8} {'Ratio':>8} "
          f"{'n_rej':>6} {'p-value':>10}")
    print("-" * 60)
    for model_key, s in all_summaries.items():
        ratio_str = f"{s['nli_ratio']:.2f}x" if s['nli_ratio'] < 100 else "inf"
        p_str = f"{s['fisher_p']:.4f}" if s['fisher_p'] is not None else "N/A"
        print(f"{model_key:<10} {s['nli_admissible_acc']:>7.1%} "
              f"{s['nli_rejected_acc']:>7.1%} {ratio_str:>8} "
              f"{s['nli_n_rejected']:>6} {p_str:>10}")

        total_nli_adm_correct += int(s['nli_admissible_acc'] * s['nli_n_admissible'])
        total_nli_adm += s['nli_n_admissible']
        total_nli_rej_correct += int(s['nli_rejected_acc'] * s['nli_n_rejected'])
        total_nli_rej += s['nli_n_rejected']

    print("-" * 60)
    overall_adm_acc = total_nli_adm_correct / max(total_nli_adm, 1)
    overall_rej_acc = total_nli_rej_correct / max(total_nli_rej, 1)
    overall_ratio = overall_adm_acc / max(overall_rej_acc, 0.001)
    print(f"{'Overall':<10} {overall_adm_acc:>7.1%} {overall_rej_acc:>7.1%} "
          f"{overall_ratio:>7.2f}x {total_nli_rej:>6}")

    # Save aggregate to a local path (main() runs locally, /results is a
    # Modal volume mount that only exists inside containers).
    import numpy as np
    import os as _os
    local_results_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results")
    _os.makedirs(local_results_dir, exist_ok=True)
    agg_file = _os.path.join(local_results_dir, f"prereg_{dataset}_aggregate.json")
    agg = {
        "dataset": dataset,
        "models": {k: {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv)
                        for kk, vv in v.items()}
                   for k, v in all_summaries.items()},
        "overall": {
            "nli_admissible_acc": overall_adm_acc,
            "nli_rejected_acc": overall_rej_acc,
            "nli_ratio": overall_ratio,
            "total_admissible": total_nli_adm,
            "total_rejected": total_nli_rej,
        },
    }
    try:
        with open(agg_file, 'w') as f:
            json.dump(agg, f, indent=2)
        print(f"\nAggregate saved to {agg_file}")
    except Exception as e:
        print(f"\nWarning: could not save aggregate file: {e}")
        print("Per-model results are saved on the Modal volume; aggregate can be reconstructed.")
