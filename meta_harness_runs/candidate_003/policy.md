# Trace-rich restart policy

Adds full trace preservation and distinct timeout classifications.

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

## Recommended Changes
- Create run packets with stdout, stderr, gates, and artifacts.
