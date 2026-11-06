"""
Compound figure for the NeurIPS submission:
  Left panel  — headline discrimination ratios (5 benchmarks, 95% CIs)
  Right panel — pooled ARC risk-coverage curves (1-maxCP vs SE(0.75) vs KLE)

Output: figures/results_overview.pdf

Usage:
  ~/ip_llm/bin/python experiments/make_results_figure.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

# NeurIPS requires Type 1 or TrueType fonts in submitted PDFs.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Values taken verbatim from main.tex (single source of truth for the paper)
BARS = [
    # (label, ratio, ci_lo, ci_hi, regime)
    ("TriviaQA",              2.94, 2.50, 3.52, "pass"),
    ("TruthfulQA (substring)", 1.88, 1.70, 2.09, "pass"),
    ("NQ-Open",               1.62, 1.48, 1.77, "pass"),
    ("MMLU",                  1.18, 1.12, 1.26, "attenuated"),
    ("ARC-Challenge",         1.10, 1.06, 1.15, "attenuated"),
]

COLOR_PASS = "#2B5F87"        # calm blue
COLOR_ATT  = "#B8B8B8"        # muted grey
COLOR_1MCP = "#C44536"        # warm red — our method
COLOR_SE   = "#2B5F87"        # blue
COLOR_KLE  = "#7E6B8F"        # plum
COLOR_RAND = "#888888"        # grey

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "miscalibration_combination_results.json"
OUT_DIR = HERE.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def risk_coverage(scores, correct):
    """Return (coverages, accuracies) sorted by ascending uncertainty."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    order = np.argsort(scores)
    c = correct[order]
    n = len(c)
    cum = np.cumsum(c)
    coverages = np.arange(1, n + 1) / n
    accuracies = cum / np.arange(1, n + 1)
    return coverages, accuracies


def main():
    data = json.load(open(RESULTS))
    arc = [r for r in data if r["dataset"] == "arc"]
    print(f"ARC rows: {len(arc)}")

    one_mcp = [r["one_minus_maxcp"] for r in arc]
    se_075  = [r["se_075"] for r in arc]
    kle     = [r["kle"] for r in arc]
    correct = [r["mv_correct"] for r in arc]

    cov_1m, acc_1m = risk_coverage(one_mcp, correct)
    cov_se, acc_se = risk_coverage(se_075, correct)
    cov_kl, acc_kl = risk_coverage(kle, correct)

    # Random baseline: shuffle scores => curve tends to flat at overall accuracy
    rng = np.random.default_rng(0)
    rand_scores = rng.random(len(correct))
    cov_r, acc_r = risk_coverage(rand_scores, correct)

    # --- Figure ---
    fig = plt.figure(figsize=(10.5, 2.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.30)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_rc  = fig.add_subplot(gs[0, 1])

    # Left: bars
    labels = [b[0] for b in BARS]
    ratios = [b[1] for b in BARS]
    lo     = [b[2] for b in BARS]
    hi     = [b[3] for b in BARS]
    colors = [COLOR_PASS if b[4] == "pass" else COLOR_ATT for b in BARS]

    y = np.arange(len(BARS))[::-1]
    xerr = np.array([[r - l for r, l in zip(ratios, lo)],
                     [h - r for r, h in zip(ratios, hi)]])
    ax_bar.barh(y, ratios, xerr=xerr, color=colors,
                edgecolor="black", linewidth=0.6, height=0.6,
                error_kw=dict(ecolor="black", elinewidth=1.0, capsize=3))
    ax_bar.axvline(1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.set_xlabel(r"Discrimination ratio  (admissible acc / rejected acc)",
                       fontsize=9)
    ax_bar.set_xlim(0.5, 4.0)
    ax_bar.tick_params(axis="x", labelsize=8)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # Annotate each bar with the ratio (offset from the upper CI whisker)
    for yi, r, h in zip(y, ratios, hi):
        ax_bar.text(h + 0.08, yi, rf"{r:.2f}$\times$", va="center",
                    fontsize=8.5, color="black")

    # Legend
    legend_pass = mpatches.Patch(color=COLOR_PASS, label="free-form (works)")
    legend_att  = mpatches.Patch(color=COLOR_ATT,  label="multiple-choice (attenuated)")
    ax_bar.legend(handles=[legend_pass, legend_att], loc="lower right",
                   fontsize=8, frameon=False)
    ax_bar.set_title("(a) Admissibility predicts accuracy across 5 benchmarks",
                     fontsize=9.5, loc="left", pad=6)

    # Right: risk-coverage — plot order matters (1-maxCP last, on top).
    # Clip coverage > 0.02 to avoid the noisy left edge (sample starvation).
    def _trim(cov, acc, cov_min=0.02):
        mask = cov >= cov_min
        return cov[mask], acc[mask]

    cr, ar = _trim(cov_r, acc_r)
    ck, ak = _trim(cov_kl, acc_kl)
    cs, as_ = _trim(cov_se, acc_se)
    cm, am = _trim(cov_1m, acc_1m)

    ax_rc.plot(cr, ar, color=COLOR_RAND, linewidth=1.2, linestyle=":",
                label="random")
    ax_rc.plot(ck, ak, color=COLOR_KLE,  linewidth=1.5, linestyle="--",
                label="KLE (tuned)")
    ax_rc.plot(cs, as_, color=COLOR_SE,  linewidth=1.5, linestyle="-.",
                label=r"SE($\theta{=}0.75$, tuned)")
    ax_rc.plot(cm, am, color=COLOR_1MCP, linewidth=2.1, linestyle="-",
                label=r"$1{-}\text{maxCP}$ (parameter-free)")

    ax_rc.set_xlabel("Coverage (fraction of questions answered)", fontsize=9)
    ax_rc.set_ylabel("Majority-vote accuracy", fontsize=9)
    ax_rc.set_xlim(0.0, 1.0)
    # Expanded lower bound opens whitespace for an in-plot legend.
    ax_rc.set_ylim(0.775, 1.005)
    ax_rc.tick_params(axis="both", labelsize=8)
    ax_rc.spines[["top", "right"]].set_visible(False)
    ax_rc.grid(axis="y", alpha=0.25, linewidth=0.5)
    # Legend inside the plot, in the lower whitespace created by the expanded y-range.
    handles, labels_ = ax_rc.get_legend_handles_labels()
    ax_rc.legend(handles[::-1], labels_[::-1],
                  loc="lower center",
                  ncol=2, fontsize=7.5, frameon=False,
                  handlelength=2.0, columnspacing=1.2, handletextpad=0.5)
    ax_rc.set_title("(b) ARC risk-coverage (pooled, 4 base models, $n{=}2000$)",
                     fontsize=9.5, loc="left", pad=6)

    plt.tight_layout()
    out = OUT_DIR / "results_overview.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
