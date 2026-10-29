"""Figure 2 of the SOICT 2026 camera-ready: review-budget curves.

Drawn at its printed size (LLNCS text width = 12.2 cm = 4.80 in, included with
width=\\textwidth, i.e. at scale 1.0) in Computer Modern, so the lettering prints at
exactly 8 pt (\\footnotesize). Line styles, filled/hollow markers and grey levels stay
distinguishable in grayscale; the shaded band spans random review to the oracle."""
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

W, H = 4.80, 1.85  # inches = printed size; H = 1.65 is the low-height fallback
FS = 8
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "legend.fontsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "lines.linewidth": 1.1,
})

# key: label, colour, line style, line width, marker, filled, first marker index, zorder.
# beta = 0 is prepended, so index k is beta = 0.05 k; markers every 0.2, staggered so that
# coincident curves do not stack markers; ours sits on beta = 0.2, 0.4, 0.6, 0.8.
STYLE = {
    "full(family)":      ("AutoCSV-LED (ours)",  "#08306b", "-",                      1.6, "o", True,  4, 10),
    "profilers(family)": ("Profilers only",      "#4a98c9", (0, (3.5, 1.5)),          1.1, "^", False, 2, 8),
    "llm-only":          ("LLM judges only",     "#e6550d", (0, (5, 1.5, 1, 1.5)),    1.1, "v", False, 3, 6),
    "RAHA-20 (seed0)":   ("RAHA (20 labels)",    "#54278f", (0, (1, 1.2)),            1.2, "s", True,  1, 7),
    "chat LLM scorer":   ("Prompted LLM scorer", "#8c8c8c", (0, (4, 1, 1, 1, 1, 1)),  1.1, "x", True,  3, 5),
}
TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
TICKLABELS = ["0", "0.2", "0.4", "0.6", "0.8", "1"]

fig, axes = plt.subplots(1, 3, figsize=(W, H), sharey=True)
handles = {}
for ax, ds, title in zip(axes, ["hospital", "flights", "rayyan"], ["Hospital", "Flights", "Rayyan"]):
    c = RES[ds]["budget_curves"]
    beta = [0.0] + c["beta"]  # reviewing nothing catches nothing
    rnd, orc = [0.0] + c["random"], [0.0] + c["oracle"]
    ax.fill_between(beta, rnd, orc, color="0.92", lw=0, zorder=0)
    handles["Random"], = ax.plot(beta, rnd, color="0.55", lw=0.7, zorder=1)
    handles["Oracle"], = ax.plot(beta, orc, color="black", lw=0.7, zorder=2)
    for key, (lab, col, ls, lw, mk, filled, first, z) in STYLE.items():
        handles[lab], = ax.plot(beta, [0.0] + c[key], color=col, ls=ls, lw=lw, marker=mk,
                                markersize=3.8 if mk == "x" else 3.4, markevery=slice(first, 20, 4),
                                markerfacecolor=col if filled else "white", markeredgecolor=col,
                                markeredgewidth=0.8, zorder=z, clip_on=False)
    ax.set_title(f"{title} ({100 * RES[ds]['row_prev']:.0f}% erroneous)", pad=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks(TICKS); ax.set_xticklabels(TICKLABELS)
    ax.set_yticks(TICKS); ax.set_yticklabels(TICKLABELS)
    ax.grid(True, lw=0.4, color="0.85"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Erroneous rows caught" if H >= 1.8 else "Erroneous rows\ncaught", labelpad=2)
axes[1].set_xlabel(r"Review budget $\beta$ (fraction of rows reviewed)", labelpad=1.5)
# ncol=4 fills column-major ([2, 2, 2, 1]); reads row-wise as
#   AutoCSV-LED | Profilers only | LLM judges only | Oracle
#   Prompted LLM scorer | RAHA | Random
order = ["AutoCSV-LED (ours)", "Prompted LLM scorer", "Profilers only", "RAHA (20 labels)",
         "LLM judges only", "Random", "Oracle"]
fig.legend([handles[k] for k in order], order, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 0.0), handlelength=2.4, columnspacing=1.0, handletextpad=0.4,
           borderaxespad=0.0, borderpad=0.1, labelspacing=0.2)
fig.subplots_adjust(left=0.088 if H >= 1.8 else 0.105, right=0.985, top=1 - 0.18 / H,
                    bottom=0.62 / H, wspace=0.17)
fig.savefig(OUT / "fig2_budget.pdf")  # no bbox_inches="tight": the page is exactly W x H
fig.savefig(OUT / "fig2_preview.png", dpi=300)
print("saved", OUT / "fig2_budget.pdf", f"{W} x {H} in")
