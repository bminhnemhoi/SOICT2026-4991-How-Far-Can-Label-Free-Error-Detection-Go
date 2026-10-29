"""Figure 2 of the SOICT 2026 camera-ready: review-budget curves.

Drawn at its printed size (0.9 x LLNCS text width = 11 cm) in Computer Modern, so
the lettering prints at 7 pt like the body text, with line styles, filled and
hollow markers and grey levels that stay distinguishable in grayscale."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = json.loads((ROOT / "experiments" / "rivf2026" / "results.json").read_text(encoding="utf-8"))
OUT = ROOT / "SOICT2026" / "camera_ready" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
})

# label, colour (chosen for distinct grey levels), line style, marker, filled marker
STYLE = {
    "full(family)": ("AutoCSV-LED (ours)", "#08306b", "-", "o", True),
    "profilers(family)": ("Profilers only (ours)", "#6baed6", "--", "^", False),
    "llm-only": ("LLM judges only", "#e6550d", "-.", "v", False),
    "RAHA-20 (seed0)": ("RAHA, 20 labeled tuples", "#54278f", ":", "s", True),
    "chat LLM scorer": ("Prompted LLM scorer", "#969696", (0, (3, 1, 1, 1)), "x", True),
    "oracle": ("Oracle", "#000000", (0, (1, 1)), None, True),
}

fig, axes = plt.subplots(1, 3, figsize=(4.32, 1.75), sharey=True)
for ax, ds, title in zip(axes, ["hospital", "flights", "rayyan"], ["Hospital", "Flights", "Rayyan"]):
    c = RES[ds]["budget_curves"]
    beta = c["beta"]
    ax.plot(beta, c["random"], color="#bdbdbd", lw=0.8, ls="-", label="Random")
    for key, (lab, col, ls, mk, filled) in STYLE.items():
        if key in c:
            ax.plot(beta, c[key], color=col, ls=ls, label=lab, marker=mk, markersize=2.8 if mk else 0,
                    markevery=2, markerfacecolor=col if filled else "white", markeredgewidth=0.8)
    prev = RES[ds]["row_prev"]
    ax.set_title(f"{title} (prevalence {prev:.2f})", pad=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", ".5", "1"])
    ax.set_xlabel(r"Review budget $\beta$", labelpad=1)
    ax.grid(True, lw=0.4, alpha=0.6)
axes[0].set_ylabel("Fraction of erroneous\nrows caught")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0),
           handlelength=2.2, columnspacing=0.9, handletextpad=0.4)
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.0)
fig.savefig(OUT / "fig2_budget.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "fig2_preview.png", dpi=220, bbox_inches="tight")
print("saved", OUT / "fig2_budget.pdf")
