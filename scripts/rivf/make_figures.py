"""Figure 2: review-budget curves (IEEE two-column width, TrueType fonts)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = json.loads((ROOT / "experiments" / "rivf2026" / "results.json").read_text(encoding="utf-8"))
OUT = ROOT / "paper_rivf2026" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.6,
    "lines.linewidth": 1.2, "mathtext.fontset": "stix",
})

STYLE = {
    "full(family)": ("AutoCSV-LED (ours)", "#1f4e9c", "-", "o"),
    "profilers(family)": ("Profilers only (ours)", "#4f9fd8", "--", None),
    "llm-only": ("LLM judges only", "#d9822b", "-.", None),
    "RAHA-20 (seed0)": ("RAHA, 20 labeled tuples", "#6a3d9a", ":", "s"),
    "chat LLM scorer": ("Prompted LLM scorer", "#999999", (0, (3, 1, 1, 1)), None),
    "oracle": ("Oracle", "#000000", (0, (1, 1)), None),
}

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), sharey=True)
for ax, ds, title in zip(axes, ["hospital", "flights", "rayyan"], ["Hospital", "Flights", "Rayyan"]):
    c = RES[ds]["budget_curves"]
    beta = c["beta"]
    ax.plot(beta, c["random"], color="#bbbbbb", lw=0.8, ls="-", label="Random")
    for key, (lab, col, ls, mk) in STYLE.items():
        if key in c:
            ax.plot(beta, c[key], color=col, ls=ls, label=lab, marker=mk, markersize=2.2 if mk else 0,
                    markevery=2)
    prev = RES[ds]["row_prev"]
    ax.set_title(f"{title} (row prevalence {prev:.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel(r"Review budget $\beta$")
    ax.grid(True, lw=0.3, alpha=0.5)
axes[0].set_ylabel("Erroneous rows caught")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06),
           handlelength=2.4, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.0)
fig.savefig(OUT / "fig2_budget.pdf", bbox_inches="tight", pad_inches=0.01)
print("saved", OUT / "fig2_budget.pdf")
