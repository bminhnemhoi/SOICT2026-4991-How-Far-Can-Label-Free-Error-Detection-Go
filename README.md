# SOICT 2026 · Paper 4991 · How Far Can Label-Free Error Detection Go?

Code, prompts, cached LLM outputs and per-row results of **AutoCSV-LED** for the paper

> Binh Minh Ngo. *How Far Can Label-Free Error Detection Go? Evidential Fusion of Data Profilers and LLM Judges for Tables.*
> The 15th International Symposium on Information and Communication Technology (SOICT 2026), Paper ID 4991.
> Communications in Computer and Information Science, Springer.

AutoCSV-LED detects erroneous cells in a table without labels. Five lightweight data profilers (pattern rarity, null tokens, near-duplicate typos, violations of mined functional dependencies, unit inconsistency) and two LLM judges are fused with Dempster–Shafer weights of evidence: the cautious rule inside a family of non-distinct sources, Dempster's rule across families. The system outputs cell flags, a row score with conflict and ignorance diagnostics, and a review queue whose expected size is bounded without labels.

## Layout

| Path | Content |
|---|---|
| `src/evidence/` | profilers, evidential fusion, review-budget threshold, positional ground truth |
| `src/agents/`, `prompts/`, `experiments/auto/*_llm/` | the two LLM judges, their prompt templates, and the per-table domain prompts and constraints synthesized offline |
| `src/utils/llm_client.py` | cached clients for OpenAI, Anthropic and Google models |
| `scripts/rivf/` | main experiments (cell level, row level, budget, ablations, held-out tables, error taxonomy) |
| `scripts/soict/` | additions for the camera-ready: judges on dev rows, other judge models, fusion vs. aggregators, 20-tuple routing check |
| `experiments/runs/` | cached per-row judge outputs (`per_sample.jsonl`) for every model and split |
| `experiments/rivf2026/`, `experiments/soict2026/` | result JSON files from which every table of the paper is generated |
| `data/processed/` | the fixed test/dev/calibration splits (seed 42) of Hospital, Flights, Beers, Rayyan |

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate        # or source .venv/bin/activate
pip install -e . rapidfuzz anthropic google-genai
```

The benchmark tables come from the RAHA repository (https://github.com/BigDaMa/raha). Copy its `datasets/` folder (hospital, flights, beers, rayyan, movies_1, tax) to `data/raw/raha/datasets/`. RAHA itself is needed only for the RAHA baseline (`scripts/rivf/run_raha.py`, `scripts/rivf/heldout.py`).

## Reproduce

No LLM call is needed: every judge output used in the paper is cached in `experiments/runs/`.

```bash
python -m pytest tests/test_evidence.py -q              # fusion and budget-bound tests
python scripts/rivf/run_raha.py --seeds 0 1 2            # RAHA-20 baseline
python scripts/rivf/run_experiments.py                   # cell/row/budget/ablation results
python scripts/rivf/heldout.py                           # Movies and Tax (first 20,000 rows)
python scripts/rivf/error_taxonomy.py                    # content/missing/format taxonomy
python scripts/rivf/extra_stats.py
python scripts/soict/soict_analysis.py                   # dev-row choices, other judge models, fusion vs. aggregators, routing
python scripts/soict/make_tables_soict.py                # LaTeX tables
python scripts/soict/make_figure_soict.py                # review-budget figure
```

To regenerate judge outputs (costs API credits), set the keys in `.env` and run, for example:

```bash
python scripts/soict/run_judges.py --ds hospital --split dev  --provider openai    --model gpt-5-nano
python scripts/soict/run_judges.py --ds flights  --split test --provider anthropic --model claude-haiku-5-5
```

## Cached judge outputs

| Directory | Model | Rows |
|---|---|---|
| `auto_<table>_test_full` | gpt-5-nano | test rows of the four development tables |
| `soict_gpt5nano_<table>_dev` | gpt-5-nano | dev rows (used to re-make the judge-related choices) |
| `soict_gpt5mini_<table>_test` | gpt-5-mini | test rows of Hospital, Flights, Rayyan |
| `soict_claudehaiku55_<table>_test` | Claude Haiku 5.5 | test rows of Hospital, Flights, Rayyan |
| `soict_claudehaiku55_<table>_test_s43` | Claude Haiku 5.5 | second query with a new request seed (run-to-run variation) |
| `soict_aggemini38flashlow_<table>_test` | Gemini 3.8 Flash (low thinking, via an OpenAI-compatible router) | test rows of Hospital, Flights, Rayyan |
| `anollm_<table>_full` | gpt-5-nano | prompted LLM scorer baseline |

## License

MIT (see `LICENSE`). The benchmark tables belong to their original authors.
