"""
Pareto frontier figure: rejection rate vs discrimination ratio for single-theta
plurality partitions and admissibility on TriviaQA.

Each theta in {0.50, ..., 0.95} produces one (rej_rate, ratio) point; admissibility
is overlaid as a distinct point. The averaged cluster-fraction baseline (tuned
post-hoc to match rejection rate) is also shown.

Output: figures/pareto_frontier.pdf

Usage:
    ~/ip_llm/bin/python experiments/make_pareto_figure.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np


def main():
    script_dir = Path(__file__).parent
    with open(script_dir / "partition_baselines_results.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    # Single-theta frontier with colorbar mapping theta -> marker color
    frontier = data["single_theta_frontier"]
    rej = np.array([r["rej_rate"] * 100 for r in frontier])
    ratios = np.array([r["ratio"] for r in frontier])
    thetas = np.array([r["theta"] for r in frontier])

    # Connecting line (sorted by theta so the line traces the parameter path)
    order = np.argsort(thetas)
    ax.plot(rej[order], ratios[order], "-", color="#999999", alpha=0.5,
            linewidth=1.0, zorder=2)

    # Markers colored by theta
    sc = ax.scatter(rej, ratios, c=thetas, cmap="viridis",
                    s=65, edgecolor="black", linewidth=0.6,
                    vmin=0.50, vmax=0.95, zorder=3,
                    label=r"single-$\theta$ plurality")

    # Label only the endpoints of the theta sweep
    for t, x, y in zip(thetas, rej, ratios):
        if t in (0.50, 0.95):
            ax.annotate(f"$\\theta$={t:.2f}", (x, y),
                        xytext=(8, -3), textcoords="offset points",
                        fontsize=8, alpha=0.85)

    # Admissibility point
    adm = data["admissibility"]
    ax.plot([adm["rej_rate"] * 100], [adm["ratio"]], "s",
            color="#0055A4", markersize=13, markeredgecolor="black",
            markeredgewidth=1.2, zorder=5, label="admissibility (ours)")

    # Avg cluster-fraction point
    avg = data["averaged_cluster_fraction"]
    ax.plot([avg["rej_rate"] * 100], [avg["ratio"]], "^",
            color="#D35400", markersize=12, markeredgecolor="black",
            markeredgewidth=1.0, zorder=5,
            label="avg cluster-fraction (post-hoc, matched rate)")

    ax.set_xlabel("Rejection rate (%)")
    ax.set_ylabel("Discrimination ratio (adm / rej accuracy)")
    ax.set_title("TriviaQA partition frontier ($n_Q{=}2000$, pooled)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(r"single-$\theta$ value", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = script_dir.parent / "figures" / "pareto_frontier.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
