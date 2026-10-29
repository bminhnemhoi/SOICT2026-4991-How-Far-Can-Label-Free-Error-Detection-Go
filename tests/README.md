# `tests/` — Unit tests

> Pytest. Chạy: `pytest tests/ -v`

## Cấu trúc

```
tests/
├── README.md                   # file này
├── conftest.py                 # fixtures chung (sample row, sample evidence)
├── test_bcl.py                 # BCL: 5 case (full agree err/clean, full conflict, partial ignorance)
├── test_ds.py                  # DS combine: known textbook examples
├── test_uq_perturb.py          # Perturbations bảo toàn key, không touch target cell
├── test_cadp.py                # Decision policy + conformal threshold
├── test_metrics.py             # AUC-PR, ECE, AURC với expected values
├── test_eval_calibration.py    # Reliability bins
├── test_data_serialize.py      # Serialize → JSON + NL
├── test_data_splits.py         # Stratified 20/20/60 deterministic
├── test_data_bart_inject.py    # Số lỗi inject khớp config
├── test_agents_schema.py       # Output JSON validate against schema
└── test_utils_cache.py         # Cache key correctness
```

## Quy tắc

- Test nhanh (<1s mỗi test). Test chậm gắn `@pytest.mark.slow`.
- Không gọi API thật — mock LLM client.
- Mỗi function public trong `src/` cần ≥ 1 test.
- Coverage target: **≥ 70%** cho `src/{bcl, ds, uq, cadp, eval}` (core method).

## Bắt đầu

`tests/test_smoke.py` — test smoke kiểm tra import OK:

```python
def test_imports():
    import src
    assert src.__version__
```
