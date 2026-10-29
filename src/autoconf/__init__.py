"""AutoCSV-MA-LED: zero-configuration pipeline for tabular error detection.

Given an arbitrary CSV/parquet dataset, this module produces:
  L1: DatasetProfile   — schema + column stats + sample rows
  L2a: stats constraints — domain / pattern / range / missingness from stats
  L2b: FD constraints   — approximate functional-dependency mining
  L2c: LLM constraints  — (optional) LLM-synthesized domain rules
  L3:  Auto prompt      — (optional) domain-inferred expert prompt template

The outputs are consumed by the MA-LED pipeline (Auditor + Expert + BCL/DS/CADP)
without any hand-crafted YAML.
"""
