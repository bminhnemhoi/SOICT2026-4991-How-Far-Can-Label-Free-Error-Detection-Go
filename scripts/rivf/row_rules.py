"""Row-level AUROC of the three fusion rules (family / Dempster everywhere / cautious everywhere)."""
import json
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evidence import profilers as P  # noqa: E402
from src.evidence.fusion import fuse, load_llm_evidence  # noqa: E402
from src.evidence.ground_truth import load_benchmark  # noqa: E402

out = {}
for ds in ("hospital", "flights", "rayyan"):
    b = load_benchmark(ds)
    t = b.splits["test"]
    y = b.errors[t].any(1)
    ev = P.run_profilers(b.dirty)
    llm = load_llm_evidence(ROOT / "experiments" / "runs" / f"auto_{ds}_test_full" / "per_sample.jsonl", len(b.dirty), b.columns)
    out[ds] = {}
    for rule in ("family", "dempster", "cautious"):
        fz = fuse(ev, P.FAMILY, llm, rule=rule, use_llm=True)
        out[ds][rule] = float(roc_auc_score(y, fz["row"].score[t]))
    print(ds, {k: round(v, 3) for k, v in out[ds].items()})
(ROOT / "experiments" / "rivf2026" / "row_rules.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
