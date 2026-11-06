"""
Stitch per-model pre-registration results into a single aggregate.

Use this after running ``run_preregistered.py`` in pieces (e.g., two models
at a time due to Modal timeouts). It reads each model's latest
``/results/prereg_{dataset}_{model}_{timestamp}.json`` off the Modal volume,
pools the NLI admissible / rejected contingency tables across models, and
writes ``experiments/results/prereg_{dataset}_aggregate.json`` in the same
format produced by ``run_preregistered.main()``.

Run:
    modal run stitch_aggregate.py --dataset truthfulqa
    modal run stitch_aggregate.py --dataset truthfulqa --models Qwen,Mistral,Llama,Gemma
"""

import modal

app = modal.App("choice-functions-stitch")
volume = modal.Volume.from_name("choice-fn-results")

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "scipy")

ALL_MODELS = ["Qwen", "Mistral", "Llama", "Gemma"]


@app.function(image=image, volumes={"/results": volume}, timeout=600)
def collect(dataset: str, model_keys: list) -> dict:
    """Read the latest per-model pre-registration JSON for each model."""
    import glob
    import json
    import os

    summaries = {}
    for model_key in model_keys:
        pattern = f"/results/prereg_{dataset}_{model_key}_*.json"
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No results found for {model_key} on {dataset}: {pattern}"
            )
        latest = matches[-1]
        print(f"  {model_key}: {os.path.basename(latest)}")
        with open(latest) as f:
            data = json.load(f)
        summaries[model_key] = data["summary"]
    return summaries


@app.local_entrypoint()
def main(dataset: str = "truthfulqa", models: str = ""):
    import json
    import os

    if dataset not in ("mmlu", "arc", "truthfulqa"):
        raise ValueError(f"Unknown dataset: {dataset}")

    model_keys = [m.strip() for m in models.split(",")] if models else ALL_MODELS
    print(f"Stitching {dataset}: {model_keys}")

    summaries = collect.remote(dataset, model_keys)

    total_nli_adm_correct = 0
    total_nli_adm = 0
    total_nli_rej_correct = 0
    total_nli_rej = 0

    print()
    print(f"{'Model':<10} {'NLI Adm':>8} {'NLI Rej':>8} {'Ratio':>8} "
          f"{'n_rej':>6} {'p-value':>10}")
    print("-" * 60)
    for model_key in model_keys:
        s = summaries[model_key]
        ratio = s["nli_ratio"]
        ratio_str = f"{ratio:.2f}x" if ratio < 100 else "inf"
        p = s.get("fisher_p")
        p_str = f"{p:.2e}" if p is not None else "N/A"
        print(f"{model_key:<10} {s['nli_admissible_acc']:>7.1%} "
              f"{s['nli_rejected_acc']:>7.1%} {ratio_str:>8} "
              f"{s['nli_n_rejected']:>6} {p_str:>10}")

        total_nli_adm_correct += int(round(
            s["nli_admissible_acc"] * s["nli_n_admissible"]
        ))
        total_nli_adm += s["nli_n_admissible"]
        total_nli_rej_correct += int(round(
            s["nli_rejected_acc"] * s["nli_n_rejected"]
        ))
        total_nli_rej += s["nli_n_rejected"]

    overall_adm_acc = total_nli_adm_correct / max(total_nli_adm, 1)
    overall_rej_acc = total_nli_rej_correct / max(total_nli_rej, 1)
    overall_ratio = overall_adm_acc / max(overall_rej_acc, 1e-6)

    from scipy import stats
    table = [
        [total_nli_adm_correct, total_nli_adm - total_nli_adm_correct],
        [total_nli_rej_correct, max(total_nli_rej - total_nli_rej_correct, 0)],
    ]
    _, pooled_p = stats.fisher_exact(table, alternative="two-sided") \
        if total_nli_rej > 0 else (None, None)

    print("-" * 60)
    pooled_p_str = f"{pooled_p:.2e}" if pooled_p is not None else "N/A"
    print(f"{'Pooled':<10} {overall_adm_acc:>7.1%} {overall_rej_acc:>7.1%} "
          f"{overall_ratio:>7.2f}x {total_nli_rej:>6} {pooled_p_str:>10}")

    agg = {
        "dataset": dataset,
        "models": summaries,
        "overall": {
            "nli_admissible_acc": overall_adm_acc,
            "nli_rejected_acc": overall_rej_acc,
            "nli_ratio": overall_ratio,
            "total_admissible": total_nli_adm,
            "total_rejected": total_nli_rej,
            "pooled_fisher_p": pooled_p,
        },
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"prereg_{dataset}_aggregate.json")
    with open(out_file, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregate saved to {out_file}")
