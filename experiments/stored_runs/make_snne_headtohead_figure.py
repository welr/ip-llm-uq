"""
Risk-coverage curves for the SNNE head-to-head comparison.

Two panels (TriviaQA, ARC), each showing accuracy vs coverage for four scalar
signals: SE(theta=0.75), KLE, -SNNE, 1-maxCP. Curves are tightly overlapping
on both datasets - this visual tie is the point of the figure.

Output: figures/snne_headtohead_curves.pdf

Usage:
    ~/ip_llm/bin/python experiments/make_snne_headtohead_figure.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

from snne_headtohead import (
    load_triviaqa, load_arc_or_mmlu, signal_values,
)
from selective_prediction import risk_coverage_curve


COLORS = {
    "SE(0.75)": "#4DA6FF",
    "KLE": "#7B5B9A",
    "-SNNE": "#D35400",
    "1-maxCP": "#0055A4",
}
LINESTYLES = {
    "SE(0.75)": "--",
    "KLE": ":",
    "-SNNE": "-.",
    "1-maxCP": "-",
}
SIGNAL_KEYS = {
    "SE(0.75)": "SE(0.75)",
    "KLE": "KLE",
    "-SNNE": "SNNE",   # signal_values already negates
    "1-maxCP": "1-maxCP",
}


def bootstrap_band(scores, correct, coverage_grid, n_boot=1000, seed=42):
    """Return pointwise 2.5%/97.5% percentiles of accuracy at each coverage in grid."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    accs_samples = np.zeros((n_boot, len(coverage_grid)))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        covs, _, accs = risk_coverage_curve(scores[idx], correct[idx])
        # Interpolate onto common coverage grid
        accs_samples[b] = np.interp(coverage_grid, covs, accs)
    lo = np.percentile(accs_samples, 2.5, axis=0)
    hi = np.percentile(accs_samples, 97.5, axis=0)
    return lo, hi


def plot_panel(ax, records, title, ylim=None, n_boot=1000):
    correct = np.array([r["mv_correct"] for r in records])
    min_acc, max_acc = 1.0, 0.0
    # Common coverage grid for bands
    coverage_grid = np.linspace(0.2, 1.0, 200)
    for label in ["SE(0.75)", "KLE", "-SNNE", "1-maxCP"]:
        scores = signal_values(records, SIGNAL_KEYS[label])
        covs, _, accs = risk_coverage_curve(scores, correct)
        mask = covs >= 0.2
        # Bootstrap band
        lo, hi = bootstrap_band(scores, correct, coverage_grid, n_boot=n_boot)
        ax.fill_between(coverage_grid, lo, hi,
                        color=COLORS[label], alpha=0.12, linewidth=0)
        ax.plot(covs[mask], accs[mask],
                label=label,
                color=COLORS[label],
                linestyle=LINESTYLES[label],
                linewidth=1.8)
        min_acc = min(min_acc, lo.min())
        max_acc = max(max_acc, hi.max())
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim([0.2, 1.0])
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        pad = 0.01
        ax.set_ylim([max(0.0, min_acc - pad), min(1.0, max_acc + pad)])


def main():
    script_dir = Path(__file__).parent
    tqa = load_triviaqa(script_dir)
    arc = load_arc_or_mmlu(script_dir, "arc")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    plot_panel(axes[0], tqa, f"TriviaQA  ($n_Q$={len(tqa)})")
    plot_panel(axes[1], arc, f"ARC  ($n_Q$={len(arc)})")

    # Single legend top-right
    axes[1].legend(loc="lower left", fontsize=9, framealpha=0.9)

    plt.tight_layout()

    out_dir = script_dir.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "snne_headtohead_curves.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
