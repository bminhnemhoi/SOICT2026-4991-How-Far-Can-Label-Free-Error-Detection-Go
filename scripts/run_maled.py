"""MA-LED pipeline runner.

Reads a config.yaml and executes:
    for each row in dataset:
        payload_auditor = serialize_row(row, stats)
        payload_expert  = serialize_row_with_constraints(row, neighbors, constraints)
        a_out = AuditorAgent.run(payload_auditor)
        e_out = ExpertAgent.run(payload_expert)
        m_A   = BCL(a_out)
        m_E   = BCL(e_out)
        ds    = DS.combine(m_A, m_E)
        dec   = CADP.decide(ds, thresholds)
        log per-row output to experiments/runs/<run_id>/per_sample.jsonl

At the end, compute detection metrics on rows where y_true is known and dump
experiments/runs/<run_id>/summary.json.

Usage:
    python scripts/run_maled.py --config configs/quicktest_hospital.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python scripts/run_maled.py` without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.agents import AuditorAgent, ExpertAgent
from src.agents.base import AgentOutput
from src.bcl import BCLWeights, build_mass
from src.cadp import CADPThresholds, decide
from src.data import (
    compute_column_stats,
    load_constraints,
    select_constraints,
    serialize_row,
    serialize_row_with_constraints,
)
from src.ds import ds_combine
from src.eval import metrics as M
from src.uq import PerturbSpec, apply_perturbation
from src.utils.cost import CostTracker, estimate_cost
from src.utils.io import read_yaml, write_json
from src.utils.llm_client import build_client
from src.utils.seeds import set_global_seed

logger = logging.getLogger("maled.runner")

# Columns protected from row-level perturbations (also used as PK-like guards).
_HOSPITAL_PROTECTED_COLS: set[str] = {"row_id", "ProviderNumber", "y_true", "err_type"}


# -------------------- dataset loading --------------------

def _dataset_path(cfg: dict[str, Any]) -> Path:
    """Resolve the parquet path for a dataset config block.

    Priority:
      1. cfg['dataset']['auto_input_path']  (auto_pipeline-generated configs)
      2. data/processed/<name>_<version>/<split>.parquet  (default convention)
    """
    auto_path = cfg["dataset"].get("auto_input_path")
    if auto_path:
        p = Path(auto_path)
        if p.exists():
            return p
    name = cfg["dataset"]["name"]
    version = cfg["dataset"].get("version", "v1")
    split = cfg["dataset"].get("split", "dev")
    base = Path("data/processed") / f"{name}_{version}"
    for candidate in (f"{split}.parquet", "fixture.parquet"):
        p = base / candidate
        if p.exists():
            return p
    raise FileNotFoundError(f"No parquet under {base}/ (tried {split}.parquet, fixture.parquet)")


def _constraints_path(cfg: dict[str, Any]) -> Path:
    """Resolve constraints YAML path.

    Priority order:
      1. cfg['dataset']['constraints_path']  (allows auto-pipeline overrides)
      2. cfg['method']['agents']['expert']['constraints_path']
      3. data/constraints/<name>.yaml (default convention)
    """
    override = cfg["dataset"].get("constraints_path")
    if override:
        return Path(override)
    expert_cfg = cfg.get("method", {}).get("agents", {}).get("expert", {}) or {}
    expert_override = expert_cfg.get("constraints_path")
    if expert_override:
        return Path(expert_override)
    name = cfg["dataset"]["name"]
    return Path("data/constraints") / f"{name}.yaml"


# -------------------- per-row pipeline --------------------

def _build_auditor_payload(
    row: pd.Series,
    column_stats: dict[str, Any],
) -> dict[str, Any]:
    return serialize_row(
        row,
        row_id=int(row["row_id"]),
        column_stats=column_stats,
        include_stats_for_target_only=False,
    )


def _find_neighbors(df: pd.DataFrame, row: pd.Series, k: int, fd_lhs_cols: list[str]) -> list[dict]:
    """Return up to k other rows sharing the FD-LHS values with `row`."""
    if not fd_lhs_cols:
        return df.iloc[:k].drop(columns=["y_true", "err_type"], errors="ignore").to_dict("records")
    mask = np.ones(len(df), dtype=bool)
    for col in fd_lhs_cols:
        if col in df.columns:
            mask &= df[col] == row[col]
    mask &= df["row_id"] != row["row_id"]
    hits = df.loc[mask].head(k)
    return hits.drop(columns=["y_true", "err_type"], errors="ignore").to_dict("records")


def _build_expert_payload(
    row: pd.Series,
    df: pd.DataFrame,
    constraints: list[dict],
    neighbor_k: int,
) -> dict[str, Any]:
    fd_lhs: list[str] = []
    for c in constraints:
        if c.get("type") == "functional_dependency":
            fd_lhs = c.get("formal", {}).get("lhs", [])
            break
    selected = select_constraints(row.index.tolist(), constraints, top_k=8)
    neighbors = _find_neighbors(df, row, neighbor_k, fd_lhs)
    payload = serialize_row_with_constraints(row, selected, row_id=int(row["row_id"]))
    payload["neighbor_rows"] = neighbors
    return payload


def _bcl_weights(cfg: dict[str, Any]) -> BCLWeights:
    w = cfg["method"]["bcl"].get("weights", {})
    return BCLWeights(
        w1=float(w.get("w1", 1.0)),
        w2=float(w.get("w2", 1.0)),
        w3=float(w.get("w3", 1.0)),
        w4=float(w.get("w4", 1.0)),
    )


def _thresholds(cfg: dict[str, Any]) -> CADPThresholds:
    c = cfg["method"]["cadp"]
    def _f(x: Any, default: float) -> float:
        return default if x is None else float(x)
    return CADPThresholds(
        tau_e=_f(c.get("tau_e"), 0.5),
        tau_c=_f(c.get("tau_c"), 0.5),
        tau_k=_f(c.get("tau_k"), 0.4),
        tau_omega=_f(c.get("tau_omega"), 0.4),
    )


# -------------------- runner --------------------

def run(config_path: str) -> dict[str, Any]:
    load_dotenv()  # populate OPENAI_API_KEY etc.
    cfg = read_yaml(config_path)
    run_id = cfg["run_id"]
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    # Prepare output directory.
    out_dir = Path("experiments/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    per_sample_path = out_dir / "per_sample.jsonl"
    per_sample_path.unlink(missing_ok=True)

    # Load data + constraints.
    df = pd.read_parquet(_dataset_path(cfg))
    subset_n = cfg["dataset"].get("subset_n")
    if subset_n:
        if cfg["dataset"].get("stratified", False) and "y_true" in df.columns:
            n = int(subset_n)
            err = df[df["y_true"] == "err"].sample(n=min(n // 2, (df["y_true"] == "err").sum()), random_state=seed)
            clean = df[df["y_true"] == "clean"].sample(n=min(n - len(err), (df["y_true"] == "clean").sum()), random_state=seed)
            df = pd.concat([err, clean]).sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            df = df.head(int(subset_n)).reset_index(drop=True)
    constraints = load_constraints(_constraints_path(cfg))

    # Pre-compute column stats (leak-free: on full df is OK for smoke test).
    stats_df = df.drop(columns=["y_true", "err_type"], errors="ignore")
    column_stats = compute_column_stats(stats_df)

    # Build clients + agents.
    cost = CostTracker()
    client = build_client(cost_tracker=cost)
    m_cfg = cfg["method"]["agents"]
    seed = int(cfg.get("seed", 42))

    def _resolve_prompt_path(agent_cfg: dict, default: str) -> str | None:
        """Return full path for prompt name from config, or None to use class default."""
        name = agent_cfg.get("prompt")
        if not name:
            return None
        # If caller already passed a full path (contains '/'), use as-is.
        if "/" in name or "\\" in name:
            return name
        return f"prompts/{name}.md"

    auditor = AuditorAgent(
        client=client, model=m_cfg["auditor"]["model"],
        temperature=m_cfg["auditor"].get("temperature", 0.0),
        max_tokens=m_cfg["auditor"].get("max_tokens", 400),
        seed=seed,
        self_consistency_m=cfg["method"]["uq"].get("self_consistency_M", 1),
        prompt_path=_resolve_prompt_path(m_cfg["auditor"], AuditorAgent.prompt_path),
    )
    expert = ExpertAgent(
        client=client, model=m_cfg["expert"]["model"],
        temperature=m_cfg["expert"].get("temperature", 0.0),
        max_tokens=m_cfg["expert"].get("max_tokens", 500),
        seed=seed,
        self_consistency_m=cfg["method"]["uq"].get("self_consistency_M", 1),
        prompt_path=_resolve_prompt_path(m_cfg["expert"], ExpertAgent.prompt_path),
    )
    neighbor_k = int(m_cfg["expert"].get("neighbor_k", 3))
    weights = _bcl_weights(cfg)
    thr = _thresholds(cfg)

    # Pre-flight cost estimate.
    est = estimate_cost(
        n_samples=len(df),
        tokens_in_per_call=500,
        tokens_out_per_call=500,  # gpt-5-nano: ~200 reasoning + ~150 JSON output
        n_agents=2,
        self_consistency_m=cfg["method"]["uq"].get("self_consistency_M", 1),
        n_perturb=cfg["method"]["uq"].get("n_perturb", 0),
    )
    run_cap = float(cfg.get("budget", {}).get("max_usd", 5.0))
    logger.warning(
        "Pre-flight cost: ~$%.4f for %d rows (cap=$%.2f)", est["estimated_usd"], len(df), run_cap
    )
    if est["estimated_usd"] > run_cap:
        raise SystemExit(
            f"ABORT: estimated ${est['estimated_usd']:.4f} > run cap ${run_cap:.2f}"
        )
    t0 = time.time()
    # Aleatoric uncertainty config (idea_main.md §4.5).
    n_perturb = int(cfg["method"]["uq"].get("n_perturb", 0))
    perturb_specs = [
        PerturbSpec(op=p["op"], params={k: v for k, v in p.items() if k != "op"})
        for p in cfg["method"]["uq"].get("perturbations", [])
    ]
    protected_cols = _HOSPITAL_PROTECTED_COLS  # extend per-dataset later

    results: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        row_no_label = row.drop(labels=["y_true", "err_type"], errors="ignore")
        auditor_payload = _build_auditor_payload(row_no_label, column_stats)
        expert_payload = _build_expert_payload(row, df, constraints, neighbor_k)

        a_out = auditor.run(auditor_payload)
        try:
            e_out = expert.run(expert_payload)
        except Exception as exc:
            logger.warning("expert.run failed row_id=%s: %s — emitting clean-fallback", row.get("row_id"), exc)
            e_out = AgentOutput(label="clean", p=0.5, c=0.5, a=0.5, raw={"_fallback": str(exc)[:200]}, tokens_in=0, tokens_out=0, cache_hit=False)

        # --- Aleatoric: re-run Auditor on n_perturb perturbed rows; track p variance.
        u_ale = 0.0
        if n_perturb > 0 and perturb_specs:
            p_err_samples: list[float] = [a_out.p if a_out.label == "err" else 1.0 - a_out.p]
            for k_p in range(n_perturb):
                spec = perturb_specs[k_p % len(perturb_specs)]
                try:
                    perturbed_row = apply_perturbation(
                        spec, row_no_label, df, protected_cols=protected_cols, seed=seed + i * 100 + k_p,
                    )
                except Exception as exc:  # perturbation may fail on edge cases (e.g., all-NaN col)
                    logger.warning("perturb failed row=%d k=%d: %s", i, k_p, exc)
                    continue
                p_payload = _build_auditor_payload(perturbed_row, column_stats)
                p_out = auditor.run(p_payload)
                p_err_samples.append(p_out.p if p_out.label == "err" else 1.0 - p_out.p)
            u_ale = float(np.std(p_err_samples)) if len(p_err_samples) > 1 else 0.0
            # Discount auditor self-agreement `a` proportionally to aleatoric noise.
            # High U_ale -> the row is sensitive to small perturbations -> less reliable signal.
            a_eff = max(0.0, a_out.a - 2.0 * u_ale)
            m_A = build_mass(a_out.label, a_out.p, a_out.c, a_eff, weights=weights)
        else:
            m_A = build_mass(a_out.label, a_out.p, a_out.c, a_out.a, weights=weights)

        m_E = build_mass(e_out.label, e_out.p, e_out.c, e_out.a, weights=weights)
        ds = ds_combine(m_A, m_E)
        dec = decide(ds, thr)

        rec = {
            "row_id": int(row["row_id"]),
            "y_true": row.get("y_true", None),
            "err_type": row.get("err_type", None),
            "auditor": {"label": a_out.label, "p": a_out.p, "c": a_out.c, "a": a_out.a, "suspicious_fields": list(a_out.suspicious_fields)},
            "expert":  {"label": e_out.label, "p": e_out.p, "c": e_out.c, "a": e_out.a, "suspicious_fields": list(e_out.suspicious_fields)},
            "u_ale": u_ale,
            "m_A": m_A.as_dict(),
            "m_E": m_E.as_dict(),
            "ds":  ds.as_dict(),
            "decision": dec.decision,
            "p_hat": None if np.isnan(dec.p_hat) else float(dec.p_hat),
            "reason": dec.reason,
        }
        results.append(rec)
        with per_sample_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dt = time.time() - t0

    # Metrics on labeled rows.
    y_true = np.array([1 if r["y_true"] == "err" else 0 for r in results if r["y_true"] in ("err", "clean")])
    scores = np.array([r["ds"]["bel_err"] for r in results if r["y_true"] in ("err", "clean")])
    preds = np.array([
        1 if r["decision"] == "error" else (0 if r["decision"] == "clean" else -1)
        for r in results if r["y_true"] in ("err", "clean")
    ])

    commit_mask = preds != -1
    n_commit = int(commit_mask.sum())
    det = M.detection_scores(y_true[commit_mask], preds[commit_mask]) if n_commit else {}
    ece = float(M.expected_calibration_error(y_true, scores)) if len(y_true) else None
    brier = float(M.brier(y_true, scores)) if len(y_true) else None
    coverage = float(n_commit / len(y_true)) if len(y_true) else None

    # Selective prediction: risk-coverage curve over ALL rows using bel_err as
    # confidence proxy and threshold 0.5 on score. AURC + risk@target_cov.
    selective = {}
    if len(y_true):
        # Force-commit predictions for risk computation (RC curve doesn't use abstain).
        force_pred = (scores >= 0.5).astype(int)
        # Confidence score for ranking: max(bel_err, 1-bel_err) — distance from 0.5.
        conf = np.abs(scores - 0.5) * 2.0
        rc = M.risk_coverage_curve(y_true, force_pred, conf, threshold=0.5)
        selective = {
            "aurc": M.aurc(rc),
            "risk_at_cov_70": M.risk_at_coverage(rc, 0.70),
            "risk_at_cov_80": M.risk_at_coverage(rc, 0.80),
            "risk_at_cov_90": M.risk_at_coverage(rc, 0.90),
            "risk_at_cov_full": M.risk_at_coverage(rc, 1.00),
        }
        # Persist full RC curve (101 points) for later plotting.
        write_json(rc, out_dir / "rc_curve.json")

    summary = {
        "run_id": run_id,
        "n_rows": len(results),
        "n_commit": n_commit,
        "coverage": coverage,
        "abstain_rate": float((preds == -1).mean()) if len(preds) else None,
        "detection": det,
        "ece": ece,
        "brier": brier,
        "selective": selective,
        "cost": cost.as_dict(),
        "wall_clock_s": round(dt, 2),
    }
    write_json(summary, out_dir / "summary.json")
    logger.warning("Done: %s", json.dumps(summary, indent=2))
    return summary


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Run MA-LED pipeline from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = run(args.config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
