"""Run the two automatically prompted LLM judges (Auditor, Expert) on one split
with any provider, in parallel, with resume.  Added for the SOICT 2026
camera-ready to (i) obtain judge outputs on the dev rows, so that judge-related
choices can be made without test rows, and (ii) repeat the judges with other
model families.

Prompts, payloads, constraints and the per-row JSON format are identical to the
gpt-5-nano runs in experiments/runs/auto_<ds>_test_full/ (scripts/run_maled.py).

Usage:
    python scripts/soict/run_judges.py --ds hospital --split dev --provider openai --model gpt-5-nano
    python scripts/soict/run_judges.py --ds flights --split test --provider anthropic --model claude-haiku-5-5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

import run_maled as RM  # noqa: E402
from src.agents import AuditorAgent, ExpertAgent  # noqa: E402
from src.agents.base import AgentOutput  # noqa: E402
from src.data import compute_column_stats, load_constraints  # noqa: E402
from src.utils.cost import CostTracker  # noqa: E402
from src.utils.io import read_yaml  # noqa: E402
from src.utils.llm_client import build_client  # noqa: E402

RUNS = ROOT / "experiments" / "runs"


def run_tag(model: str) -> str:
    return model.replace(".", "").replace("-", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--split", default="test", choices=("test", "dev"))
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    load_dotenv(ROOT / ".env")  # does not override variables already set

    cfg = read_yaml(RUNS / f"auto_{args.ds}_test_full" / "config_used.yaml")
    cfg["dataset"]["split"] = args.split
    # the cached configs store absolute paths of the original machine; resolve them inside this checkout
    for block, key in ((cfg["dataset"], "constraints_path"), (cfg["method"]["agents"]["expert"], "prompt")):
        p = str(block.get(key, ""))
        if "experiments/auto/" in p and not Path(p).exists():
            block[key] = str(ROOT / ("experiments/auto/" + p.split("experiments/auto/", 1)[1]))
    df = pd.read_parquet(RM._dataset_path(cfg)).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    constraints = load_constraints(RM._constraints_path(cfg))
    stats = compute_column_stats(df.drop(columns=["y_true", "err_type"], errors="ignore"))

    client = build_client(provider=args.provider, cost_tracker=CostTracker())
    a_cfg, e_cfg = cfg["method"]["agents"]["auditor"], cfg["method"]["agents"]["expert"]
    seed = int(cfg.get("seed", 42))
    auditor = AuditorAgent(client=client, model=args.model, temperature=0.0,
                           max_tokens=a_cfg.get("max_tokens", 1500), seed=seed, self_consistency_m=1,
                           prompt_path=f"prompts/{a_cfg['prompt']}.md")
    expert = ExpertAgent(client=client, model=args.model, temperature=0.0,
                         max_tokens=e_cfg.get("max_tokens", 1500), seed=seed, self_consistency_m=1,
                         prompt_path=e_cfg["prompt"])
    k = int(e_cfg.get("neighbor_k", 3))

    out_dir = RUNS / f"soict_{run_tag(args.model)}_{args.ds}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "per_sample.jsonl"
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            done.add(json.loads(line)["row_id"])
    todo = [i for i in range(len(df)) if int(df.loc[i, "row_id"]) not in done]
    print(f"[{args.ds}/{args.split}/{args.model}] {len(df)} rows, {len(done)} done, {len(todo)} to run", flush=True)

    lock = threading.Lock()
    stats_run = {"tin": 0, "tout": 0, "fallback": 0, "n": 0}

    def one(i: int) -> dict:
        row = df.loc[i]
        a_out = auditor.run(RM._build_auditor_payload(row.drop(labels=["y_true", "err_type"], errors="ignore"), stats))
        try:
            e_out = expert.run(RM._build_expert_payload(row, df, constraints, k))
        except Exception as exc:  # same fallback as scripts/run_maled.py
            e_out = AgentOutput(label="clean", p=0.5, c=0.5, a=0.5, raw={"_fallback": str(exc)[:200]},
                                tokens_in=0, tokens_out=0, cache_hit=False)
            with lock:
                stats_run["fallback"] += 1
        rec = {"row_id": int(row["row_id"]), "y_true": row.get("y_true"), "model": args.model}
        for name, o in (("auditor", a_out), ("expert", e_out)):
            rec[name] = {"label": o.label, "p": o.p, "suspicious_fields": list(o.suspicious_fields)}
        rec["tokens_in"] = a_out.tokens_in + e_out.tokens_in
        rec["tokens_out"] = a_out.tokens_out + e_out.tokens_out
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool, out.open("a", encoding="utf-8") as fh:
        futs = {pool.submit(one, i): i for i in todo}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except Exception as exc:  # auditor failure: leave the row for a resume pass
                print(f"  row {futs[f]} failed: {str(exc)[:160]}", flush=True)
                continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                stats_run["n"] += 1
                stats_run["tin"] += rec["tokens_in"]
                stats_run["tout"] += rec["tokens_out"]
                if stats_run["n"] % 100 == 0:
                    print(f"  {stats_run['n']}/{len(todo)} rows, {time.time() - t0:.0f}s", flush=True)
    summary = {"ds": args.ds, "split": args.split, "provider": args.provider, "model": args.model,
               "rows_this_pass": stats_run["n"], "expert_fallbacks": stats_run["fallback"],
               "tokens_in": stats_run["tin"], "tokens_out": stats_run["tout"],
               "seconds": round(time.time() - t0, 1)}
    with (out_dir / "passes.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
