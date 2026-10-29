"""LaTeX tables of the SOICT 2026 camera-ready (Springer LLNCS), generated from JSON.

Reuses the RIVF table builders (scripts/rivf/make_tables.py) for the unchanged
tables, adapts them to the single-column LLNCS layout, and adds the tables for
the reviewer requests from experiments/soict2026/soict_results.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "rivf"))
import make_tables as MT  # noqa: E402

S = json.loads((ROOT / "experiments" / "soict2026" / "soict_results.json").read_text(encoding="utf-8"))
OUT = ROOT / "SOICT2026" / "camera_ready" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
f3 = MT.f3
DSR = ["hospital", "flights", "rayyan"]
NAME = MT.NAME | {"movies_1": r"Movies$^{\dagger}$", "tax": r"Tax$^{\dagger}$"}
MODELS = [("gpt-5-nano", r"\texttt{gpt-5-nano}"), ("gpt-5-mini", r"\texttt{gpt-5-mini}"), ("claude-haiku-5-5", "Claude Haiku 5.5"), ("gemini-3.8-flash", "Gemini 3.8 Flash")]

# Detection F1 reported by the GIDCL authors (Table 3 of their full version); 20 labeled tuples,
# GPT-4-turbo plus fine-tuned RoBERTa/Mistral-7B; Hospital uses their own injected errors.
GIDCL = {"hospital": 0.98, "flights": 0.96, "beers": 0.99, "rayyan": 0.90}


def llncs(tex: str) -> str:
    """IEEE two-column float -> LLNCS single-column float."""
    return tex.replace(r"\begin{table*}[!t]", r"\begin{table}[t]").replace(r"\end{table*}", r"\end{table}") \
              .replace(r"\begin{table}[!t]", r"\begin{table}[t]")


def ci(lo_hi) -> str:
    lo, hi = lo_hi
    return r"\,{\scriptsize[" + f"{lo:+.3f},{hi:+.3f}".replace("0.", ".") + "]}"


def cell_table() -> str:
    t = llncs(MT.cell_table())
    t = t.replace("\\footnotesize\n\\setlength{\\tabcolsep}{4.2pt}", "\\scriptsize\n\\setlength{\\tabcolsep}{2.3pt}")
    gidcl = "GIDCL~\\cite{gidcl}$^{b}$ & & & " + " & & & ".join(f"{GIDCL[d]:.2f}"[1:] for d in MT.DS) + r"\\"
    t = t.replace(r"ZeroED~\cite{zeroed}", gidcl + "\n" + r"ZeroED~\cite{zeroed}")
    t = t.replace(r"\emph{Whole tables, F1 as reported by the authors (other LLMs and protocols)}",
                  r"\emph{Whole tables, F1 as reported by the authors (other models and protocols)}")
    t = t.replace(".}\n\\label{tab:cell}",
                  r". $^{b}$Uses 20 labeled tuples, GPT-4 Turbo, and fine-tuned RoBERTa/Mistral-7B; its Hospital "
                  r"errors are injected by the authors.}" "\n" r"\label{tab:cell}")
    return t


def models_table() -> str:
    rows = []
    for key, lab in MODELS:
        cells = []
        for d in DSR:
            M = S["models"][d].get(key)
            if not M:
                cells += ["--"] * 3
                continue
            star = "" if M["complete"] else "$^{*}$"
            lo = M["gain_vs_profilers"]["CI"][0]
            sig = "" if lo > 0 else "$^{\circ}$"
            cells += [f3(M["judges_only"]), f3(M["fused"]) + star,
                      f"{M['gain_vs_profilers']['diff']:+.3f}".replace("0.", ".").replace("-", "$-$") + sig]
        rows.append(lab + " & " + " & ".join(cells) + r"\\")
    prof = " & ".join(f" & {f3(S['models'][d]['gpt-5-nano']['profilers'])} & " for d in DSR)
    return (r"\begin{table}[t]" "\n" r"\centering" "\n"
            r"\caption{Row-level AUROC on the test rows with judges from different models: judges only (J), "
            r"fused with the profilers (Fused), and the gain $\Delta$ of the fusion over the profilers alone. "
            r"$^{\circ}$The 95\% paired bootstrap interval of $\Delta$ includes zero; all other intervals exclude it. "
            r"Prompts and fusion are unchanged; $\Delta$ is computed before rounding.}" "\n"
            r"\label{tab:models}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{2.2pt}" "\n"
            r"\begin{tabular}{@{}l" + "ccc" * 3 + r"@{}}" "\n" r"\toprule" "\n"
            + " & " + " & ".join(r"\multicolumn{3}{c}{" + MT.NAME[d] + "}" for d in DSR) + r"\\" "\n"
            + "".join(r"\cmidrule(lr){" + f"{2 + 3 * i}-{4 + 3 * i}" + "}" for i in range(3)) + "\n"
            + "Judge model & " + " & ".join(["J", "Fused", r"$\Delta$"] * 3) + r"\\" "\n" r"\midrule" "\n"
            + "Profilers only & " + prof + r"\\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")


def fusion_table() -> str:
    keys = [("DS fusion (ours)", r"Evidential fusion (ours)"), ("mean", "Mean of scores"),
            ("max", "Maximum"), ("noisy-OR", "Noisy-OR"), ("rank average", "Rank average"),
            ("stacking (5-fold, labels)", r"Logistic stacking$^{a}$")]
    cols = [(d, m) for m in ("gpt-5-nano", "claude-haiku-5-5") for d in DSR if m in S["aggregators"][d]]
    lines = []
    for k, lab in keys:
        cells = []
        for d, m in cols:
            A = S["aggregators"][d][m]
            free = [A[kk] for kk, _ in keys[:-1]]
            v = A[k]
            s = f3(v)
            if k != "stacking (5-fold, labels)" and abs(v - max(free)) < 5e-4:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(lab + " & " + " & ".join(cells) + r"\\")
        if k == "rank average":
            lines.append(r"\midrule")
    diff = []
    for d, m in cols:
        dd = S["aggregators"][d][m]["_ds_vs_best_simple"]
        v = round(dd["diff"], 3)
        diff.append("$-$" + f"{-v:.3f}"[1:] if v < 0 else "+" + f"{v:.3f}"[1:])
    lines.insert(1, r"\quad $\Delta$ vs.\ best alternative & " + " & ".join(diff) + r"\\")
    head = " & " + " & ".join(r"\multicolumn{3}{c}{" + lab + "}" for lab in (r"\texttt{gpt-5-nano} judges", "Claude Haiku 5.5 judges")) + r"\\"
    return (r"\begin{table}[t]" "\n" r"\centering" "\n"
            r"\caption{Row-level AUROC of different ways to combine the same inputs (profiler support and the two judges' "
            r"error probabilities) on the test rows. Best label-free value in bold. $^{a}$Supervised reference: 5-fold "
            r"cross-fitted logistic regression on the weights of evidence, using the test labels (mean of 5 fold seeds). $\Delta$ is computed before rounding.}" "\n"
            r"\label{tab:fusion}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
            r"\begin{tabular}{@{}l" + "c" * len(cols) + r"@{}}" "\n" r"\toprule" "\n" + head + "\n"
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
            + "Combination & " + " & ".join(MT.NAME[d][:4] + "." if d == "hospital" else MT.NAME[d] for d, _ in cols)
            + r"\\" "\n" r"\midrule" "\n" + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")


def data_table() -> str:
    t = llncs(MT.taxonomy_table())
    R = S["routing"]
    out = []
    for line in t.split("\n"):
        for key in ("hospital", "flights", "beers", "rayyan", "movies_1", "tax"):
            if line.startswith(NAME[key] + " &"):
                r = R[key]
                line = line[:-2] + f" & {f3(r['P_keep_profilers'])} & {f3(r['expected_F1_routed'])}" + r"\\"
        out.append(line)
    t = "\n".join(out)
    t = t.replace(r"\begin{tabular}{@{}lrcccccc@{}}", r"\begin{tabular}{@{}lrcccccccc@{}}")
    t = t.replace(r"Table & Rows & Cols & Err. & C/M/F & Cons. & Ours & RAHA\\",
                  r"Table & Rows & Cols & Err. & C/M/F & Cons. & Ours & RAHA & Keep & Routed\\")
    t = t.replace(r"\setlength{\tabcolsep}{2.4pt}", r"\setlength{\tabcolsep}{3pt}")
    t = t.replace(r"--: fewer than 1\% format errors.}",
                  r"--: too few format errors. Keep: probability that the 20-tuple check of "
                  r"Sect.~\ref{sec:taxonomy} keeps the label-free detector; Routed: expected F1 after the check.}")
    return t


def polish(t: str) -> str:
    """Springer style: bold marks best values only; SD; spaced intervals; one decimal style."""
    t = t.replace(r"\textbf{AutoCSV-LED} (cell detector)", "AutoCSV-LED (cell detector)")
    t = t.replace(r"\textbf{AutoCSV-LED}, whole table", "AutoCSV-LED, whole table")
    t = t.replace(r"\textbf{AutoCSV-LED} &", "AutoCSV-LED &")
    t = t.replace("Best F1 per dataset", "Best F1 per table")
    t = t.replace("F1 population standard deviations", "F1 SDs")
    t = t.replace(r"Mean$\pm$population sd over three seeds", r"Mean$\pm$SD over three seeds")
    t = re.sub(r"\[(\.\d{3}),(\.\d{3})\]", r"[\1, \2]", t)
    t = t.replace(r"Variant & Hosp. & Flights", r"Variant & Hospital & Flights")
    t = re.sub(r"& 0\.(\d{3}) & (\d+/\d+/\d+)", r"& .\1 & \2", t)   # Err. column of Table 1
    return t


if __name__ == "__main__":
    (OUT / "tab_cell.tex").write_text(cell_table(), encoding="utf-8")
    (OUT / "tab_row.tex").write_text(llncs(MT.row_table()).replace(r"\setlength{\tabcolsep}{2.5pt}", r"\setlength{\tabcolsep}{4pt}")
                                     .replace("(Beers is omitted: every row is erroneous).", "(Beers is omitted: every row is erroneous). Best value in bold."),
                                     encoding="utf-8")
    (OUT / "tab_ablation.tex").write_text(llncs(MT.ablation_table()), encoding="utf-8")
    (OUT / "tab_data.tex").write_text(data_table(), encoding="utf-8")
    (OUT / "tab_models.tex").write_text(models_table(), encoding="utf-8")
    (OUT / "tab_fusion.tex").write_text(fusion_table(), encoding="utf-8")
    for f in OUT.glob("tab_*.tex"):
        f.write_text(polish(f.read_text(encoding="utf-8")), encoding="utf-8")
    print("tables written to", OUT)
