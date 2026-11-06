"""
Interval-endpoint sweep (Reviewer Q1).

Question: How sensitive is the admissibility effect to the choice of threshold
interval [theta_min, theta_max] itself (not just grid density within a fixed
interval)?

Method: for a grid of (theta_min, theta_max) pairs, recompute admissibility
with 10 evenly-spaced thresholds in each interval, pool across 4 base models
on TriviaQA and NQ-Open, and report:
    - admissible-vs-rejected accuracy ratio (pooled)
    - AUROC of 1-maxCP (the parameter-free scalar derived from the interval)
    - rejection rate (% of responses admissible=False)

Plus AUROC.

Endpoints:
    theta_min in {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
    theta_max in {0.80, 0.85, 0.90, 0.95, 0.99}

We include the paper default (0.50, 0.95) and a narrower (0.70, 0.90) point,
plus some extrema for stress-testing.

Datasets:
    - TriviaQA (4 models, n=2000)
    - NQ-Open (4 models, n=2000 if available)

Usage:
    ~/ip_llm/bin/python experiments/interval_endpoint_sweep.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

MODELS = ["Qwen", "Mistral", "Llama", "Gemma"]
RNG_SEED = 42
N_GRID = 10  # number of thresholds in interval


def cluster_cc(sim, theta):
    m = len(sim)
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(m):
        for b in range(a + 1, m):
            if sim[a][b] >= theta:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for i in range(m):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def admissibility_and_scalar(sim, theta_min, theta_max, n_grid=N_GRID):
    """Admissibility booleans + 1-maxCP scalar for a given interval."""
    m = len(sim)
    admissible = [False] * m
    q_max_prob = 0.0
    thresholds = np.linspace(theta_min, theta_max, n_grid)
    for t in thresholds:
        clusters = cluster_cc(sim, float(t))
        sizes = [len(c) for c in clusters]
        max_size = max(sizes)
        q_max_prob = max(q_max_prob, max_size / m)
        for c in clusters:
            if len(c) == max_size:
                for r in c:
                    admissible[r] = True
    return admissible, 1.0 - q_max_prob


def bootstrap_ratio(adm_flags, accs, n_boot=5000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    adm_flags = np.asarray(adm_flags, dtype=bool)
    accs = np.asarray(accs, dtype=float)
    n = len(accs)
    ratios = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = adm_flags[idx]
        c = accs[idx]
        if not a.any() or not (~a).any():
            continue
        rej = c[~a].mean()
        if rej == 0:
            continue
        ratios.append(c[a].mean() / rej)
    ratios = np.array(ratios)
    if len(ratios) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(ratios)), float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def load_dataset(sim_cache_dir, kle_snne_path, dataset_name, prereg_glob=None):
    """Return sim_by_qm, acc_by_qm, q_correct_by_qm (question-level MV correct).

    kle_snne_path is tried first (matches TriviaQA / ARC). If the path is None or
    has no records for the dataset, fall back to prereg_glob (NQ-Open / TruthfulQA).
    """
    acc_by_qm = {}
    if kle_snne_path is not None and Path(kle_snne_path).exists():
        with open(kle_snne_path) as f:
            records = json.load(f)
        records = [r for r in records if r.get("dataset") == dataset_name]
        acc_by_qm = {(r["model"], r["qid"]): [int(x) for x in r["accuracies"]] for r in records}

    if not acc_by_qm and prereg_glob is not None:
        import glob
        for path in sorted(glob.glob(prereg_glob)):
            d = json.load(open(path))
            pq = d.get("per_question", {})
            # Infer model from filename (e.g., prereg_nq_open_Qwen_*.json)
            name = Path(path).stem
            model = None
            for m in MODELS:
                if f"_{m}_" in name or name.endswith(f"_{m}"):
                    model = m
                    break
            if model is None:
                continue
            for qid, entry in pq.items():
                acc_by_qm[(model, qid)] = [int(bool(x)) for x in entry["accuracies"]]

    sim_by_qm = {}
    for model in MODELS:
        sim_path = sim_cache_dir / f"sim_matrices_{dataset_name}_{model}.json"
        if not sim_path.exists():
            print(f"  (missing: {sim_path.name}; skipping model)")
            continue
        with open(sim_path) as f:
            sims = json.load(f)
        for qid, matrix in sims.items():
            if (model, qid) in acc_by_qm:
                sim_by_qm[(model, qid)] = matrix
    return sim_by_qm, acc_by_qm


def sweep(sim_by_qm, acc_by_qm, theta_mins, theta_maxs):
    """Run the interval sweep; return list of dicts."""
    results = []
    for tmin in theta_mins:
        for tmax in theta_maxs:
            if tmax <= tmin:
                continue
            adm_flags_all, accs_all, scalars_q, q_correct = [], [], [], []
            for key, sim in sim_by_qm.items():
                acc = acc_by_qm.get(key)
                if acc is None or len(sim) == 0:
                    continue
                adm, scalar = admissibility_and_scalar(sim, tmin, tmax)
                for i, a in enumerate(adm):
                    adm_flags_all.append(a)
                    accs_all.append(acc[i])
                # AUROC: question-level scalar (1-maxCP) vs MV-wrong indicator.
                mv_correct = int(np.mean(acc) >= 0.5)
                scalars_q.append(scalar)
                q_correct.append(mv_correct)

            adm_flags_all = np.array(adm_flags_all, dtype=bool)
            accs_all = np.array(accs_all, dtype=float)
            n = len(accs_all)
            n_adm = int(adm_flags_all.sum())
            rej_rate = (n - n_adm) / n if n else float("nan")

            if n_adm > 0 and (n - n_adm) > 0:
                adm_acc = float(accs_all[adm_flags_all].mean())
                rej_acc = float(accs_all[~adm_flags_all].mean())
                ratio_point = adm_acc / rej_acc if rej_acc > 0 else float("inf")
                ratio_mean, ratio_lo, ratio_hi = bootstrap_ratio(adm_flags_all, accs_all)
            else:
                adm_acc = float(accs_all.mean()) if n > 0 else float("nan")
                rej_acc = float("nan")
                ratio_point = float("nan")
                ratio_mean = ratio_lo = ratio_hi = float("nan")

            # AUROC of 1-maxCP for MV-wrong detection.
            q_correct = np.array(q_correct)
            scalars_q = np.array(scalars_q)
            # higher 1-maxCP -> more uncertain -> wrong. So label = (1 - mv_correct).
            y = 1 - q_correct
            try:
                if y.sum() > 0 and y.sum() < len(y):
                    auroc = roc_auc_score(y, scalars_q)
                else:
                    auroc = float("nan")
            except Exception:
                auroc = float("nan")

            results.append({
                "theta_min": float(tmin),
                "theta_max": float(tmax),
                "n_responses": int(n),
                "n_admissible": n_adm,
                "rejection_rate": rej_rate,
                "adm_acc": adm_acc,
                "rej_acc": rej_acc,
                "ratio": float(ratio_point),
                "ratio_bootstrap_mean": ratio_mean,
                "ratio_ci_lo": ratio_lo,
                "ratio_ci_hi": ratio_hi,
                "auroc_one_minus_maxcp": float(auroc),
            })
    return results


def main():
    script_dir = Path(__file__).parent
    sim_cache_dir = script_dir / "sim_cache"

    theta_mins = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    theta_maxs = [0.80, 0.85, 0.90, 0.95, 0.99]

    out = {"config": {"theta_mins": theta_mins, "theta_maxs": theta_maxs, "n_grid": N_GRID}, "datasets": {}}

    for dataset_name, kle_snne_file, prereg_glob in [
        ("triviaqa", "kle_snne_triviaqa_results.json", None),
        ("nq_open", None, str(script_dir / "modal_data" / "prereg_nq_open_*.json")),
    ]:
        print(f"\n=== Interval-endpoint sweep: {dataset_name} ===")
        ks_path = (script_dir / kle_snne_file) if kle_snne_file else None
        sim_by_qm, acc_by_qm = load_dataset(sim_cache_dir, ks_path, dataset_name, prereg_glob=prereg_glob)
        print(f"Loaded {len(sim_by_qm)} (model,qid) pairs.")
        if len(sim_by_qm) == 0:
            print("  (no data — skipping)")
            continue

        results = sweep(sim_by_qm, acc_by_qm, theta_mins, theta_maxs)

        print(f"{'theta_min':>9}  {'theta_max':>9}  {'ratio':>6}  {'95% CI':>14}  {'rej_rate':>8}  {'auroc':>6}  {'adm_acc':>7}  {'rej_acc':>7}")
        for r in results:
            ci = f"[{r['ratio_ci_lo']:.2f},{r['ratio_ci_hi']:.2f}]" if np.isfinite(r['ratio_ci_lo']) else "   (n/a)    "
            print(f"{r['theta_min']:>9.2f}  {r['theta_max']:>9.2f}  "
                  f"{r['ratio']:>6.2f}  {ci:>14}  "
                  f"{r['rejection_rate']*100:>7.2f}%  "
                  f"{r['auroc_one_minus_maxcp']:>6.3f}  "
                  f"{r['adm_acc']*100:>6.2f}%  "
                  f"{r['rej_acc']*100:>6.2f}%")

        out["datasets"][dataset_name] = results

    with open(script_dir / "interval_endpoint_sweep_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print()
    print(f"Saved to {script_dir/'interval_endpoint_sweep_results.json'}")


if __name__ == "__main__":
    main()
