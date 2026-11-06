"""
KLE/SNNE comparison with admissibility filtering.

Computes Kernel Language Entropy (KLE), Soft Nearest-Neighbor Entropy (SNNE),
fixed-threshold SE, and admissibility on the same data. Evaluates all methods
as uncertainty estimators via AUROC for predicting majority-vote correctness.

Requires generation checkpoint files (response texts + gold answers) in
experiments/modal_data/.

Usage:
    ~/ip_llm/bin/python experiments/kle_snne_comparison.py
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# NLI similarity (DeBERTa-large-MNLI on MPS)
# ---------------------------------------------------------------------------

_NLI_MODEL = None
_NLI_TOKENIZER = None
_DEVICE = None


def _load_nli():
    global _NLI_MODEL, _NLI_TOKENIZER, _DEVICE
    if _NLI_MODEL is not None:
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "microsoft/deberta-large-mnli"
    print(f"Loading NLI model: {model_name} ...")
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _NLI_TOKENIZER = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(
        model_name, local_files_only=True
    )

    if torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
        print("  Using MPS (Apple Silicon)")
    else:
        _DEVICE = torch.device("cpu")
        print("  Using CPU")

    _NLI_MODEL = _NLI_MODEL.to(_DEVICE)
    _NLI_MODEL.eval()
    print("  Model loaded.")


def nli_similarity(text1: str, text2: str) -> float:
    """Bidirectional NLI entailment score in [0, 1]."""
    import torch

    _load_nli()

    scores = []
    for premise, hypothesis in [(text1, text2), (text2, text1)]:
        inputs = _NLI_TOKENIZER(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=256,
        ).to(_DEVICE)
        with torch.no_grad():
            logits = _NLI_MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        # DeBERTa-MNLI labels: 0=contradiction, 1=neutral, 2=entailment
        scores.append(probs[2].item())
    return sum(scores) / 2.0


def compute_similarity_matrix(responses: list[str]) -> np.ndarray:
    """Compute M x M bidirectional NLI similarity matrix."""
    m = len(responses)
    sim = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            s = nli_similarity(responses[i], responses[j])
            sim[i, j] = s
            sim[j, i] = s
    return sim


# ---------------------------------------------------------------------------
# Uncertainty methods
# ---------------------------------------------------------------------------

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def _entropy_bits(probs):
    """Shannon entropy in bits."""
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * np.log2(p)
    return h


def _cluster(sim_matrix, theta):
    """Connected-component clustering at threshold theta. Returns list of sets."""
    m = sim_matrix.shape[0]
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(m):
        for j in range(i + 1, m):
            if sim_matrix[i, j] >= theta:
                union(i, j)

    clusters = defaultdict(set)
    for i in range(m):
        clusters[find(i)].add(i)
    return list(clusters.values())


def fixed_threshold_se(sim_matrix, theta=0.85):
    """Standard semantic entropy at a single threshold."""
    clusters = _cluster(sim_matrix, theta)
    m = sim_matrix.shape[0]
    probs = [len(c) / m for c in clusters]
    return _entropy_bits(probs)


def admissibility_result(sim_matrix, thresholds=THRESHOLDS):
    """Compute admissible set and SE bounds."""
    m = sim_matrix.shape[0]
    admissible = set()
    se_values = []

    for theta in thresholds:
        clusters = _cluster(sim_matrix, theta)
        probs = [len(c) / m for c in clusters]
        max_prob = max(probs)
        for c, p in zip(clusters, probs):
            if p == max_prob:
                admissible |= c
        se_values.append(_entropy_bits(probs))

    rejected = set(range(m)) - admissible
    return admissible, rejected, min(se_values), max(se_values)


def kernel_language_entropy(sim_matrix):
    """KLE: von Neumann entropy of the normalized similarity kernel.

    Nikitin et al. (2024). Computes eigenvalues of the normalized kernel
    matrix and returns -sum(lambda_i * log2(lambda_i)).
    """
    m = sim_matrix.shape[0]
    # Normalize so eigenvalues sum to 1 (density matrix)
    trace = np.trace(sim_matrix)
    if trace < 1e-12:
        return 0.0
    rho = sim_matrix / trace
    eigenvalues = np.linalg.eigvalsh(rho)
    # Clip small negatives from numerical error
    eigenvalues = np.clip(eigenvalues, 1e-15, None)
    eigenvalues = eigenvalues / eigenvalues.sum()  # re-normalize
    return -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))


def kle_bandwidth_sweep(sim_matrix, bandwidths):
    """KLE with Gaussian kernel at different bandwidths, showing sensitivity."""
    m = sim_matrix.shape[0]
    results = {}
    for bw in bandwidths:
        # Convert similarity to kernel via Gaussian: K(i,j) = exp(-(1-s)^2 / (2*bw^2))
        dist_sq = (1.0 - sim_matrix) ** 2
        kernel = np.exp(-dist_sq / (2 * bw ** 2))
        trace = np.trace(kernel)
        if trace < 1e-12:
            results[bw] = 0.0
            continue
        rho = kernel / trace
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = np.clip(eigenvalues, 1e-15, None)
        eigenvalues = eigenvalues / eigenvalues.sum()
        results[bw] = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))
    return results


def snne_entropy(sim_matrix, temperature=1.0):
    """SNNE: Soft Nearest-Neighbor Entropy via LogSumExp aggregation.

    Nguyen et al. (2025). Aggregates pairwise similarities without
    explicit clustering.

    Note: this returns the raw per-response softmax entropy. For AUROC
    prediction of INCORRECTNESS, negate the score: higher SNNE reflects
    a denser neighborhood (more consensus, lower uncertainty), so
    scoring should pass `-snne` (or equivalently, `1 - snne/log2(M-1)`).
    The raw positive value is retained here for downstream flexibility.
    """
    m = sim_matrix.shape[0]
    if m <= 1:
        return 0.0

    # For each response, compute softmax attention over others
    entropies = []
    for i in range(m):
        sims = np.array([sim_matrix[i, j] for j in range(m) if j != i])
        # Softmax
        sims_scaled = sims / temperature
        sims_scaled -= sims_scaled.max()  # numerical stability
        exp_sims = np.exp(sims_scaled)
        probs = exp_sims / exp_sims.sum()
        entropies.append(_entropy_bits(probs))

    return np.mean(entropies)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def normalize_answer(text):
    """Basic normalization for majority vote."""
    import re
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def check_answer(response, gold_answers):
    """Fuzzy match against gold answer set.

    Guards against gold answers that normalize to empty (e.g., punctuation-only
    gold), which would otherwise spuriously match every response since
    "" in any_string is True.
    """
    norm_resp = normalize_answer(response)
    for gold in gold_answers:
        norm_gold = normalize_answer(gold)
        if norm_gold and (norm_gold in norm_resp or norm_resp in norm_gold):
            return 1
    return 0


def compute_auroc(scores, labels):
    """AUROC for uncertainty scores predicting incorrectness.

    Higher uncertainty score should predict incorrect answers (label=0).
    We flip: score predicts P(incorrect), so AUROC measures discrimination.
    """
    if len(set(labels)) < 2:
        return float('nan')
    return roc_auc_score(labels, scores)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_dataset(dataset, model_keys, data_dir, checkpoint_dir):
    """Run full comparison for one dataset."""
    print(f"\n{'='*70}")
    print(f"  {dataset.upper()} COMPARISON: Admissibility vs KLE vs SNNE")
    print(f"{'='*70}")

    all_results = []

    for model_key in model_keys:
        gen_file = os.path.join(data_dir, f"checkpoint_gen_{dataset}_{model_key}.json")
        if not os.path.exists(gen_file):
            print(f"\n  SKIP {model_key}: no generation checkpoint")
            continue

        print(f"\n  Processing {model_key}...")
        with open(gen_file) as f:
            gen_data = json.load(f)
        print(f"    {len(gen_data)} questions loaded")

        # Load existing analysis for comparison
        analysis_file = os.path.join(data_dir, f"checkpoint_analysis_{dataset}_{model_key}.json")
        existing_analysis = {}
        if os.path.exists(analysis_file):
            with open(analysis_file) as f:
                existing_analysis = json.load(f)

        # Checkpoint file for similarity matrices
        sim_checkpoint = os.path.join(
            checkpoint_dir, f"sim_matrices_{dataset}_{model_key}.json"
        )
        cached_sims = {}
        if os.path.exists(sim_checkpoint):
            with open(sim_checkpoint) as f:
                cached_sims = json.load(f)
            print(f"    Loaded {len(cached_sims)} cached similarity matrices")

        n_computed = 0
        t0 = time.time()

        for idx, (qid, qd) in enumerate(gen_data.items()):
            gold_answers = qd["gold_answers"]
            all_responses = qd["responses"]
            top6 = [r["text"] for r in all_responses[:6]]
            accuracies = [check_answer(t, gold_answers) for t in top6]

            # Majority vote
            normalized = [normalize_answer(t) for t in top6]
            counts = Counter(normalized)
            mv_winner = counts.most_common(1)[0][0]
            mv_idx = next(i for i, n in enumerate(normalized) if n == mv_winner)
            mv_correct = accuracies[mv_idx]

            # Similarity matrix (cached or computed)
            if qid in cached_sims:
                sim_matrix = np.array(cached_sims[qid])
            else:
                sim_matrix = compute_similarity_matrix(top6)
                cached_sims[qid] = sim_matrix.tolist()
                n_computed += 1

                # Checkpoint every 50 questions
                if n_computed % 50 == 0:
                    elapsed = time.time() - t0
                    rate = n_computed / elapsed
                    remaining = (len(gen_data) - idx - 1) / rate if rate > 0 else 0
                    print(f"    [{idx+1}/{len(gen_data)}] {n_computed} sim matrices computed "
                          f"({rate:.1f}/s, ~{remaining/60:.0f}m remaining)")
                    with open(sim_checkpoint, 'w') as f:
                        json.dump(cached_sims, f)

            # --- Compute all methods ---
            # 1. Admissibility
            adm, rej, se_lo, se_hi = admissibility_result(sim_matrix)
            se_width = se_hi - se_lo

            # 2. Fixed-threshold SE at various thresholds
            se_085 = fixed_threshold_se(sim_matrix, 0.85)
            se_075 = fixed_threshold_se(sim_matrix, 0.75)
            se_090 = fixed_threshold_se(sim_matrix, 0.90)

            # 3. KLE (direct on similarity matrix)
            kle = kernel_language_entropy(sim_matrix)

            # 4. KLE bandwidth sweep
            kle_bw = kle_bandwidth_sweep(sim_matrix, [0.1, 0.2, 0.3, 0.5, 1.0])

            # 5. SNNE
            snne = snne_entropy(sim_matrix, temperature=1.0)
            snne_05 = snne_entropy(sim_matrix, temperature=0.5)

            # 6. Admissible-set majority vote
            adm_normalized = [normalized[i] for i in sorted(adm)]
            if adm_normalized:
                adm_counts = Counter(adm_normalized)
                adm_mv_winner = adm_counts.most_common(1)[0][0]
                adm_mv_idx = next(i for i in sorted(adm)
                                  if normalized[i] == adm_mv_winner)
                adm_mv_correct = accuracies[adm_mv_idx]
            else:
                adm_mv_correct = mv_correct

            all_results.append({
                "dataset": dataset,
                "model": model_key,
                "qid": qid,
                "mv_correct": mv_correct,
                "adm_mv_correct": adm_mv_correct,
                "n_admissible": len(adm),
                "n_rejected": len(rej),
                "se_width": se_width,
                "se_085": se_085,
                "se_075": se_075,
                "se_090": se_090,
                "kle": kle,
                "kle_bw": kle_bw,
                "snne": snne,
                "snne_05": snne_05,
                "has_rejection": len(rej) > 0,
                "accuracies": accuracies,
            })

        # Save similarity matrix checkpoint
        if n_computed > 0:
            with open(sim_checkpoint, 'w') as f:
                json.dump(cached_sims, f)
            print(f"    Saved {len(cached_sims)} similarity matrices to cache")

        elapsed = time.time() - t0
        print(f"    Done: {n_computed} new sim matrices in {elapsed:.0f}s")

    return all_results


def print_results(all_results):
    """Print comparison tables."""

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")

    # --- 1. AUROC comparison ---
    print(f"\n  AUROC for predicting majority-vote INCORRECTNESS")
    print(f"  (higher = better uncertainty estimator)")
    print(f"  {'-'*65}")

    datasets = sorted(set(r["dataset"] for r in all_results))
    models = sorted(set(r["model"] for r in all_results))

    for dataset in datasets:
        print(f"\n  Dataset: {dataset.upper()}")
        print(f"  {'Model':<10} {'SE(.85)':>8} {'SE(.75)':>8} {'SE(.90)':>8} "
              f"{'KLE':>8} {'SNNE':>8} {'SE width':>8}")
        print(f"  {'-'*60}")

        for model in models:
            subset = [r for r in all_results
                      if r["dataset"] == dataset and r["model"] == model]
            if not subset:
                continue

            labels = [1 - r["mv_correct"] for r in subset]  # 1 = incorrect
            if len(set(labels)) < 2:
                print(f"  {model:<10} (all correct or all incorrect — AUROC undefined)")
                continue

            se85 = compute_auroc([r["se_085"] for r in subset], labels)
            se75 = compute_auroc([r["se_075"] for r in subset], labels)
            se90 = compute_auroc([r["se_090"] for r in subset], labels)
            kle_auroc = compute_auroc([r["kle"] for r in subset], labels)
            # SNNE: negate so that higher "uncertainty" (lower neighborhood
            # density) predicts incorrectness. See snne_entropy() note.
            snne_auroc = compute_auroc([-r["snne"] for r in subset], labels)
            sew_auroc = compute_auroc([r["se_width"] for r in subset], labels)

            print(f"  {model:<10} {se85:>8.3f} {se75:>8.3f} {se90:>8.3f} "
                  f"{kle_auroc:>8.3f} {snne_auroc:>8.3f} {sew_auroc:>8.3f}")

        # Pooled
        subset = [r for r in all_results if r["dataset"] == dataset]
        labels = [1 - r["mv_correct"] for r in subset]
        if len(set(labels)) >= 2:
            se85 = compute_auroc([r["se_085"] for r in subset], labels)
            se75 = compute_auroc([r["se_075"] for r in subset], labels)
            se90 = compute_auroc([r["se_090"] for r in subset], labels)
            kle_auroc = compute_auroc([r["kle"] for r in subset], labels)
            # SNNE: negate so that higher "uncertainty" (lower neighborhood
            # density) predicts incorrectness. See snne_entropy() note.
            snne_auroc = compute_auroc([-r["snne"] for r in subset], labels)
            sew_auroc = compute_auroc([r["se_width"] for r in subset], labels)
            print(f"  {'-'*60}")
            print(f"  {'Pooled':<10} {se85:>8.3f} {se75:>8.3f} {se90:>8.3f} "
                  f"{kle_auroc:>8.3f} {snne_auroc:>8.3f} {sew_auroc:>8.3f}")

    # --- 2. KLE bandwidth sensitivity ---
    print(f"\n\n  KLE BANDWIDTH SENSITIVITY")
    print(f"  (mean KLE across all questions, by bandwidth parameter)")
    print(f"  {'-'*65}")

    bandwidths = [0.1, 0.2, 0.3, 0.5, 1.0]
    for dataset in datasets:
        subset = [r for r in all_results if r["dataset"] == dataset]
        if not subset:
            continue
        print(f"\n  Dataset: {dataset.upper()}")
        print(f"  {'BW':<8}", end="")
        for bw in bandwidths:
            print(f" {bw:>8}", end="")
        print(f" {'Range':>8}")
        print(f"  {'-'*55}")

        for model in models:
            model_subset = [r for r in subset if r["model"] == model]
            if not model_subset:
                continue
            means = []
            print(f"  {model:<8}", end="")
            for bw in bandwidths:
                vals = [r["kle_bw"].get(str(bw), r["kle_bw"].get(bw, 0)) for r in model_subset]
                m = np.mean(vals)
                means.append(m)
                print(f" {m:>8.3f}", end="")
            print(f" {max(means)-min(means):>8.3f}")

    # --- 3. Downstream: majority vote with/without filtering ---
    print(f"\n\n  DOWNSTREAM: MAJORITY VOTE ACCURACY")
    print(f"  {'-'*65}")

    for dataset in datasets:
        print(f"\n  Dataset: {dataset.upper()}")
        print(f"  {'Model':<10} {'Full MV':>8} {'Adm MV':>8} {'Delta':>8} "
              f"{'% w/ rej':>8}")
        print(f"  {'-'*45}")

        for model in models:
            subset = [r for r in all_results
                      if r["dataset"] == dataset and r["model"] == model]
            if not subset:
                continue

            full_mv = np.mean([r["mv_correct"] for r in subset])
            adm_mv = np.mean([r["adm_mv_correct"] for r in subset])
            pct_rej = np.mean([r["has_rejection"] for r in subset]) * 100

            print(f"  {model:<10} {full_mv:>7.1%} {adm_mv:>7.1%} "
                  f"{adm_mv - full_mv:>+7.1%} {pct_rej:>7.1f}%")

        subset = [r for r in all_results if r["dataset"] == dataset]
        full_mv = np.mean([r["mv_correct"] for r in subset])
        adm_mv = np.mean([r["adm_mv_correct"] for r in subset])
        pct_rej = np.mean([r["has_rejection"] for r in subset]) * 100
        print(f"  {'-'*45}")
        print(f"  {'Pooled':<10} {full_mv:>7.1%} {adm_mv:>7.1%} "
              f"{adm_mv - full_mv:>+7.1%} {pct_rej:>7.1f}%")

    # --- 4. Admissibility as complementary filter ---
    print(f"\n\n  ADMISSIBILITY + KLE COMPLEMENTARITY")
    print(f"  AUROC: KLE alone vs KLE + admissibility binary feature")
    print(f"  {'-'*50}")

    for dataset in datasets:
        subset = [r for r in all_results if r["dataset"] == dataset]
        labels = [1 - r["mv_correct"] for r in subset]
        if len(set(labels)) < 2:
            continue

        kle_scores = np.array([r["kle"] for r in subset])
        has_rej = np.array([float(r["has_rejection"]) for r in subset])

        kle_auroc = compute_auroc(kle_scores, labels)

        # Combined: simple average of normalized scores
        kle_norm = (kle_scores - kle_scores.mean()) / (kle_scores.std() + 1e-8)
        combined = kle_norm + has_rej * 0.5  # add rejection signal
        combined_auroc = compute_auroc(combined, labels)

        print(f"  {dataset.upper():<10} KLE alone: {kle_auroc:.3f}  "
              f"KLE+adm: {combined_auroc:.3f}  "
              f"delta: {combined_auroc - kle_auroc:+.3f}")


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / "modal_data"
    checkpoint_dir = script_dir / "sim_cache"
    checkpoint_dir.mkdir(exist_ok=True)

    results_file = script_dir / "kle_snne_results.json"

    model_keys = ["Qwen", "Mistral", "Llama", "Gemma"]

    all_results = []

    # ARC: all 4 models have generation checkpoints
    arc_results = analyze_dataset("arc", model_keys, data_dir, checkpoint_dir)
    all_results.extend(arc_results)

    # MMLU: only Gemma has generation checkpoint
    mmlu_results = analyze_dataset("mmlu", ["Gemma"], data_dir, checkpoint_dir)
    all_results.extend(mmlu_results)

    # Save raw results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to {results_file}")

    # Print tables
    print_results(all_results)


if __name__ == "__main__":
    main()
