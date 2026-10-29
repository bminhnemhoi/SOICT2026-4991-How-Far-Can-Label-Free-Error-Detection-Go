# Hyper-parameter selection log (dev split only, 20% of rows; test rows never used)

Criterion: mean cell-level F1 of the profiler detector over the dev splits of
Hospital, Flights, Beers and Rayyan (threshold m_err >= 0.5).

| Decision | Candidates (mean dev F1) | Selected |
|---|---|---|
| FD minimum confidence (strength = conf x dominance) | 0.5: 0.598, 0.6: 0.598, 0.7: 0.600, 0.8: 0.600, 0.9: 0.633, 0.95: 0.632 | 0.9 |
| FD evidence strength | conf^0: 0.636, conf^0.5: 0.633, conf^1: 0.633 (at conf 0.9) | 1 - c(v)/c(mode) |
| Typo similarity (rapidfuzz ratio) | 80: 0.636, 85: 0.661, 90: 0.664, 95: 0.659, no typo profiler: 0.658 | 90 |

Fixed a priori (not tuned): PATTERN_RARE_SHARE=0.05, FD_MIN_GROUP=3, TYPO_DOMINANCE=3,
null-token list, unit-profiler constants, cell decision threshold 0.5, fusion rule
(cautious within family, Dempster across families), LLM strength max(0, 2p-1).
Profilers were designed while inspecting dev rows of the four benchmarks; movies_1 and
tax were never inspected and serve as held-out datasets.
