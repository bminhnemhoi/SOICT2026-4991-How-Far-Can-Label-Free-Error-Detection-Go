"""Generate every LaTeX table and inline number of the RIVF paper from JSON results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = json.loads((ROOT / "experiments" / "rivf2026" / "results.json").read_text(encoding="utf-8"))
HO_PATH = ROOT / "experiments" / "rivf2026" / "heldout" / "heldout.json"
HO = json.loads(HO_PATH.read_text(encoding="utf-8")) if HO_PATH.exists() else {}
OUT = ROOT / "paper_rivf2026" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
DS = ["hospital", "flights", "beers", "rayyan"]
NAME = {"hospital": "Hospital", "flights": "Flights", "beers": "Beers", "rayyan": "Rayyan"}

# Numbers reported by the original authors on the full tables (cell-level F1).
ZEROED = {"hospital": 0.811, "flights": 0.722, "beers": 0.774, "rayyan": 0.732}
FORESTED = {"hospital": 0.756, "flights": 0.720, "beers": 0.947, "rayyan": 0.704}


def f3(x: float) -> str:
    return f"{x:.3f}".replace("0.", ".", 1) if x < 1 else "1.00"


def f2(x: float) -> str:
    return f"{x:.2f}"


def bold_best(vals: list[float | None]) -> list[str]:
    real = [v for v in vals if v is not None]
    best = max(real) if real else None
    out = []
    for v in vals:
        if v is None:
            out.append("--")
        elif best is not None and abs(v - best) < 5e-4:
            out.append(r"\textbf{" + f3(v) + "}")
        else:
            out.append(f3(v))
    return out


# ------------------------------------------------------------------ Table: cell level
def cell_table() -> str:
    rows = [
        (r"\multicolumn{13}{l}{\emph{Requires labels}}", None),
        ("RAHA, 20 labeled tuples$^{a}$", "RAHA-20"),
        (r"\multicolumn{13}{l}{\emph{Label-free}}", None),
        ("LLM judges, flagged fields", "llm-only"),
        ("Syntax profilers only", "prof-minus-value"),
        ("Value profilers only", "prof-minus-syntax"),
        (r"\textbf{AutoCSV-LED} (cell detector)", "profilers(family)"),
    ]
    lines = []
    f1col = {k: [RES[d]["cell"][k]["F1"] if k in RES[d]["cell"] else None for d in DS]
             for _, k in rows if k}
    best_f1 = {d: max(f1col[k][i] for k in f1col if f1col[k][i] is not None) for i, d in enumerate(DS)}
    for label, key in rows:
        if key is None:
            lines.append(label + r"\\")
            continue
        cells = []
        for d in DS:
            c = RES[d]["cell"].get(key)
            if not c:
                cells += ["--"] * 3
                continue
            F = f3(c["F1"])
            if abs(c["F1"] - best_f1[d]) < 5e-4:
                F = r"\textbf{" + F + "}"
            cells += [f3(c["P"]), f3(c["R"]), F]
        lines.append(label + " & " + " & ".join(cells) + r"\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{13}{l}{\emph{Whole tables, F1 as reported by the authors (other LLMs and protocols)}}\\")
    lines.append("ZeroED~\\cite{zeroed} & & & " + " & & & ".join(f3(ZEROED[d]) for d in DS) + r"\\")
    lines.append("ForestED~\\cite{forested} & & & " + " & & & ".join(f3(FORESTED[d]) for d in DS) + r"\\")
    lines.append(r"\textbf{AutoCSV-LED}, whole table & " + " & ".join(
        f"{f3(RES[d]['cell_full_table']['P'])} & {f3(RES[d]['cell_full_table']['R'])} & {f3(RES[d]['cell_full_table']['F1'])}"
        for d in DS) + r"\\")
    head = (r"\begin{table*}[!t]" "\n" r"\centering" "\n"
            r"\caption{Cell-level error detection on the test rows (precision, recall, F1). "
            r"Best F1 per dataset among methods evaluated on the same rows in bold. "
            r"$^{a}$Mean of three sampling seeds; F1 population standard deviations "
            + ", ".join(f"{RES[d]['cell']['RAHA-20']['F1_sd']:.3f}" for d in DS) + ".}"
            "\n" r"\label{tab:cell}" "\n" r"\footnotesize" "\n" r"\setlength{\tabcolsep}{4.2pt}" "\n"
            r"\begin{tabular}{@{}l" + "ccc" * 4 + r"@{}}" "\n" r"\toprule" "\n"
            + " & " + " & ".join(r"\multicolumn{3}{c}{" + NAME[d] + "}" for d in DS) + r"\\" "\n"
            + "".join(r"\cmidrule(lr){" + f"{2 + 3 * i}-{4 + 3 * i}" + "}" for i in range(4)) + "\n"
            + "Method & " + " & ".join(["P", "R", "F1"] * 4) + r"\\" "\n" r"\midrule" "\n")
    return head + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table*}" "\n"


# ------------------------------------------------------------------ Table: row level
def row_table() -> str:
    dsr = ["hospital", "flights", "rayyan"]
    rows = [
        ("Isolation Forest~\\cite{iforest}", "IsolationForest"),
        ("Prompted LLM scorer", "chat LLM scorer"),
        ("LLM judges only", "llm-only"),
        ("RAHA, 20 labeled tuples$^{a}$", "RAHA"),
        ("Profilers only", "profilers(family)"),
        (r"\textbf{AutoCSV-LED}", "full(family)"),
    ]
    vals = {}
    for _, k in rows:
        vals[k] = []
        for d in dsr:
            if k == "RAHA":
                vals[k].append(float(np.mean(RES[d]["raha_row_auroc_seeds"])))
            else:
                vals[k].append(RES[d]["row"][k]["AUROC"])
    lines = []
    for label, k in rows:
        cells = []
        for i, d in enumerate(dsr):
            v = vals[k][i]
            best = max(vals[kk][i] for kk in vals)
            s = f3(v)
            if abs(v - best) < 5e-4:
                s = r"\textbf{" + s + "}"
            if k == "RAHA":
                s += r"{\scriptsize$\pm$" + f"{np.std(RES[d]['raha_row_auroc_seeds']):.3f}".replace("0.", ".") + "}"
            else:
                lo, hi = RES[d]["row"][k]["AUROC_CI"]
                s += r"\,{\scriptsize[" + f3(lo) + "," + f3(hi) + "]}"
            cells.append(s)
        lines.append(label + " & " + " & ".join(cells) + r"\\")
    return (r"\begin{table}[!t]" "\n" r"\centering" "\n"
            r"\caption{Row-level AUROC on the test rows with 95\% bootstrap intervals (Beers is omitted: every row is erroneous). "
            r"$^{a}$Mean$\pm$population sd over three seeds.}" "\n" r"\label{tab:row}" "\n" r"\footnotesize" "\n"
            r"\setlength{\tabcolsep}{2.5pt}" "\n" r"\begin{tabular}{@{}lccc@{}}" "\n" r"\toprule" "\n"
            r"Method & Hospital & Flights & Rayyan\\" "\n" r"\midrule" "\n"
            + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")


# ------------------------------------------------------------------ Table: ablation
def ablation_table() -> str:
    rows = [
        (r"AutoCSV-LED cell detector", "profilers(family)"),
        (r"\quad $-$ pattern rarity", "prof-minus-pattern"),
        (r"\quad $-$ null tokens", "prof-minus-missing"),
        (r"\quad $-$ near-duplicate typos", "prof-minus-typo"),
        (r"\quad $-$ dependency violations", "prof-minus-fd"),
        (r"\quad $-$ unit inconsistency", "prof-minus-unit"),
        (r"\quad Dempster or cautious rule everywhere$^{a}$", "prof(dempster)"),
        (r"\quad $+$ LLM flagged fields", "full(family)"),
        (r"\quad $+$ LLM clean verdicts", "profilers+LLM-clean-veto"),
    ]
    for d in DS:  # the two alternative rules must give identical flags for the merged row
        assert abs(RES[d]["cell"]["prof(dempster)"]["F1"] - RES[d]["cell"]["prof(cautious)"]["F1"]) < 1e-12
    lines = []
    for label, k in rows:
        lines.append(label + " & " + " & ".join(f3(RES[d]["cell"][k]["F1"]) for d in DS) + r"\\")
        if k == "profilers(family)" or k == "prof-minus-unit" or k == "prof(dempster)":
            lines.append(r"\midrule")
    return (r"\begin{table}[!t]" "\n" r"\centering" "\n"
            r"\caption{Ablation of the cell detector (cell-level F1, test rows). $^{a}$Both rules give exactly the same flags as the family rule.}" "\n" r"\label{tab:ablation}" "\n"
            r"\footnotesize" "\n" r"\setlength{\tabcolsep}{3.2pt}" "\n" r"\begin{tabular}{@{}lcccc@{}}" "\n"
            r"\toprule" "\n" r"Variant & Hosp. & Flights & Beers & Rayyan\\" "\n" r"\midrule" "\n"
            + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")


# ------------------------------------------------------------------ Table: six tables + error taxonomy
def taxonomy_table() -> str:
    tax = json.loads((ROOT / "experiments" / "rivf2026" / "error_taxonomy.json").read_text(encoding="utf-8"))
    names = [("hospital", "Hospital"), ("flights", "Flights"), ("beers", "Beers"), ("rayyan", "Rayyan"),
             ("movies_1", r"Movies$^{\dagger}$"), ("tax", r"Tax$^{\dagger}$")]
    lines = []
    for key, lab in names:
        if key not in tax:
            continue
        t = tax[key]
        sh = t["share"]
        cons = t["format_with_modal_shape"]
        raha = t["RAHA20_F1_whole"]
        raha_s = "--" if raha is None else f3(raha[0])
        rows = f"{t['rows']:,}".replace(",", "{,}")
        lines.append(f"{lab} & {rows} & {t['cols']} & {t['cell_err_rate']:.3f} & "
                     f"{100 * sh['content']:.0f}/{100 * sh['missing']:.0f}/{100 * sh['format']:.0f} & "
                     f"{'--' if cons is None or sh['format'] < 0.01 else f'{100 * cons:.0f}'} & "
                     f"{f3(t['profiler_F1_whole'])} & {raha_s}" + r"\\")
        if key == "rayyan":
            lines.append(r"\midrule")
    return (r"\begin{table}[!t]" "\n" r"\centering" "\n"
            r"\caption{The six tables: size, cell error rate, error classes (content/missing/format, \%), share of format errors "
            r"written in the column's most frequent shape (Cons.), and whole-table cell F1 of the label-free detector and of RAHA "
            r"with 20 labeled tuples. $^{\dagger}$Held out; Tax: first 20{,}000 rows. --: fewer than 1\% format errors.}" "\n"
            r"\label{tab:data}" "\n" r"\footnotesize" "\n" r"\setlength{\tabcolsep}{2.4pt}" "\n"
            r"\begin{tabular}{@{}lrcccccc@{}}" "\n" r"\toprule" "\n"
            r"Table & Rows & Cols & Err. & C/M/F & Cons. & Ours & RAHA\\" "\n" r"\midrule" "\n"
            + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")


def dataset_table() -> str:
    types = {"hospital": "typos", "flights": "missing, conflicting sources",
             "beers": "units, N/A, merged fields", "rayyan": "date format, missing",
             "movies_1": "number formats, other", "tax": "formats, typos"}
    total_rows = {"hospital": 1000, "flights": 2376, "beers": 2410, "rayyan": 1000}
    lines = []
    for d in DS:
        r = RES[d]
        lines.append(f"{NAME[d]} & {total_rows[d]}/{r['n_test']} & {r['m']} & {r['cell_err_rate']:.3f} & {r['row_prev']:.2f} & {types[d]}" + r"\\")
    if HO:
        lines.append(r"\midrule")
        for d, lab in (("movies_1", "Movies$^{\dagger}$"), ("tax", "Tax$^{\dagger}$")):
            if d in HO:
                h = HO[d]
                lines.append(f"{lab} & {h['n']}/-- & {h['m']} & {h['cell_err_rate']:.3f} & {h['row_prev']:.2f} & {types[d]}" + r"\\")
    return (r"\begin{table}[!t]" "\n" r"\centering" "\n"
            r"\caption{Benchmarks (rows all/test, data columns, cell error rate, test row prevalence). "
            r"$^{\dagger}$Held out: never inspected during design; Tax uses its first 20{,}000 rows.}" "\n"
            r"\label{tab:data}" "\n" r"\footnotesize" "\n" r"\setlength{\tabcolsep}{2.6pt}" "\n"
            r"\begin{tabular}{@{}lccccl@{}}" "\n" r"\toprule" "\n"
            r"Dataset & Rows & Cols & Err. & Prev. & Dominant errors\\" "\n" r"\midrule" "\n"
            + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")


def summary() -> dict:
    s = {}
    for d in DS:
        r = RES[d]
        b = r["budget"]
        s[d] = {
            "cellF1": r["cell"]["profilers(family)"]["F1"], "cellF1_CI": r["cell"]["profilers(family)"]["F1_CI"],
            "raha": r["cell"]["RAHA-20"]["F1"], "full_table": r["cell_full_table"]["F1"],
            "llm_fields": r["cell"]["llm-only"]["F1"], "plus_llm": r["cell"]["full(family)"]["F1"],
            "budget_max_abs_gap": max(abs(v["load_mean"] - float(k)) for k, v in b.items()),
            "budget_max_sd": max(v["load_sd"] for v in b.values()),
            "recall_at_0.4": b["0.4"]["row_recall"], "recall_at_0.2": b["0.2"]["row_recall"],
            "llm_usd": r["llm_cost"]["est_usd_full"], "profiler_s": r["profiler_seconds"],
            "sens_range": [min(r["sensitivity"].values()), max(r["sensitivity"].values())],
            "thr_sweep": r["cell_threshold_sweep"], "n_fds": r["n_fds"],
        }
        if "row" in r:
            s[d]["row"] = {k: v["AUROC"] for k, v in r["row"].items() if not k.startswith("_")}
            s[d]["paired"] = {k: v for k, v in r["row"].items() if k.startswith("_")}
            s[d]["raha_row_mean"] = float(np.mean(r["raha_row_auroc_seeds"]))
        s[d]["conflict"] = r["conflict"]
    s["rayyan_without_date"] = RES["rayyan"].get("rayyan_without_date")
    s["heldout"] = {d: {"profilers": h["profilers"], "RAHA-20": {k: h["RAHA-20"][k] for k in ("P_mean", "R_mean", "F1_mean", "F1_sd")},
                        "secs": h["profiler_seconds"]} for d, h in HO.items()} if HO else None
    s["total_llm_usd"] = sum(RES[d]["llm_cost"]["est_usd_full"] for d in DS)
    return s


if __name__ == "__main__":
    (OUT / "tab_cell.tex").write_text(cell_table(), encoding="utf-8")
    (OUT / "tab_row.tex").write_text(row_table(), encoding="utf-8")
    (OUT / "tab_ablation.tex").write_text(ablation_table(), encoding="utf-8")
    (OUT / "tab_data.tex").write_text(taxonomy_table(), encoding="utf-8")
    S = summary()
    (ROOT / "experiments" / "rivf2026" / "paper_numbers.json").write_text(json.dumps(S, indent=2), encoding="utf-8")
    print(json.dumps(S, indent=1))
