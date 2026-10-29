# `src/` — Source code

> ⏳ Trạng thái: **CHƯA CODE**. File này mô tả thiết kế module để khi bắt đầu (21/04) viết đúng chỗ.

## Module map

| Module | File chính | Trách nhiệm |
|--------|-----------|-------------|
| `src/data/` | `serialize.py`, `bart_inject.py`, `splits.py`, `loaders.py` | Load REIN/KDD/Fraud/Retail; serialize row → JSON+NL; BART injection; stratified splits |
| `src/agents/` | `auditor.py`, `expert.py`, `judge.py`, `base.py` | Wrapper LLM call cho 3 vai trò; load prompt; validate JSON schema |
| `src/bcl/` | `bcl.py`, `calibrators.py` | Belief Construction Layer: $(p, c, a) \to (s^{err}, s^{clean}, s^{unk}) \to m$. Platt scaling cho $\phi, \psi, \rho$ |
| `src/ds/` | `combine.py` | Dempster–Shafer combine + conflict $K$ + Bel/Pl |
| `src/uq/` | `perturb.py`, `decomp.py`, `nli.py` | Họ perturbation $\Pi$, BALD epistemic, variance aleatoric, NLI score cho $c_i$ |
| `src/cadp/` | `policy.py`, `conformal.py` | Decision policy + conformal calibration cho $(\tau_k, \tau_\Omega)$ |
| `src/baselines/` | `iforest.py`, `dsvdd.py`, `ae.py`, `anollm.py`, `mad.py`, `gptcot.py`, `selfcons.py`, `zeroed_lite.py` | 5 lớp baseline |
| `src/eval/` | `metrics.py`, `calibration.py`, `risk_coverage.py`, `faithfulness.py` | Tất cả công thức metric trong [docs/06-metrics-spec.md](../docs/06-metrics-spec.md) |
| `src/utils/` | `io.py`, `llm_client.py`, `cache.py`, `manifest.py`, `seeds.py`, `logging.py` | I/O parquet, LLM client + retry + cache, manifest auto-gen, seeding global |

## Naming convention code

- snake_case cho file/function, PascalCase cho class.
- Mỗi module có docstring ở `__init__.py` mô tả 1 dòng.
- Mỗi function public có type hints + docstring 1 dòng.
- Test ngang hàng: `tests/test_<module>.py`.

## Dependency rules

- `src/agents/` chỉ import `src/utils/`.
- `src/bcl/`, `src/ds/`, `src/uq/`, `src/cadp/` độc lập với LLM client (chỉ nhận data).
- `src/eval/` không import bất cứ thứ gì khác trong `src/` (pure functions).
- Pipeline orchestration ở `src/pipelines/` (sẽ tạo khi cần) hoặc trong `scripts/run.sh`.

## Khi bắt đầu code

Theo thứ tự (giảm risk):
1. `utils/{io, seeds, llm_client + cache}` (1 buổi)
2. `eval/metrics.py` + tests (½ buổi) — để có thể đo ngay
3. `data/{loaders, serialize, splits}` (1 buổi)
4. `bcl/bcl.py` + `ds/combine.py` + tests (1 buổi)
5. `uq/perturb.py` (½ buổi)
6. `cadp/{policy, conformal}` (½ buổi)
7. `agents/{base, auditor, expert, judge}` (1 buổi)
8. `data/bart_inject.py` (½ buổi sau khi BART setup OK)
9. `baselines/*` (rolling, theo thứ tự ưu tiên ở plan)
