# Prompt Changelog

> Quy tắc xem [docs/05-prompt-registry.md](../docs/05-prompt-registry.md).

## 2026-04-24 — judge_v1
- **Initial.**
- LLM Judge chỉ summarize + đề xuất error_type + suspicious_fields.
- `decision_hint` chỉ dùng cho ablation A6/A7. Final decision do code BCL+CADP.

## 2026-04-23 — expert_v1
- **Initial.**
- Yêu cầu cite `constraint_id` khi flag violation.
- Schema: fd_violations, denial_violations, rule_violations, cross_row_issues.

## 2026-04-23 — auditor_v1
- **Initial.**
- Strict JSON output schema enforced.
- Few-shot 1 ví dụ.
- Tập trung row-level signals; không invoke business rules.
