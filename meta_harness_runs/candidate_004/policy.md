# Acceptance-aligned full charter policy

Aligns local verifiers, gates, traces, and paper claims to the final artifact acceptance object.

## Capabilities
- `bootstrap_detection`
- `stale_detection`
- `structured_gate`
- `blocked_prerequisite`
- `blocked_missing_tool`
- `json_schema_validation`
- `missing_deliverable_detection`
- `timeout_progress`
- `timeout_no_progress`
- `trace_store`
- `acceptance_alignment`
- `paper_claim_audit`
- `no_file_change_policy`

## Recommended Changes
- Adopt file-backed run packets for conductor tasks.
- Require acceptance-object declarations for local verifiers.
- Run paper-claim audits before public claim upgrades.
