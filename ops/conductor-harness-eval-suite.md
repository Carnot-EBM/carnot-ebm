# Conductor Harness Eval Suite

**Experiment:** 1281 meta-harness conductor search
**Status:** Complete

This suite defines cheap deterministic cases for evaluating conductor-policy
candidates. The cases are not model benchmarks. They measure whether a harness
policy classifies conductor failure modes honestly and preserves enough trace
history for later credit assignment.

## Eval Cases

| Case | Required behavior | Failure taxonomy |
|------|-------------------|------------------|
| `bootstrap_only_artifact` | Do not count a pre-run skeleton as complete. | `bootstrap_only_artifact` |
| `stale_skeleton` | Reject artifacts not updated by the current run. | `stale_skeleton` |
| `missing_sota_model` | Write blocked artifact with exact missing GPU/model prerequisite. | `blocked_no_sota_gguf` |
| `gated_downstream_task` | Evaluate upstream path, field, operator, and value. | `gate_blocked` |
| `local_verifier_mismatch` | Final artifact or benchmark gate overrides local-only success. | `local_verifier_mismatch` |
| `paper_unsupported_claim` | Flag claim and cite missing evidence. | `artifact_schema_invalid` |
| `timeout_with_progress` | Preserve partial evidence and classify separately. | `timeout_with_progress` |
| `timeout_without_progress` | Avoid pretending success when no evidence exists. | `timeout_without_progress` |
| `no_file_changes_produced` | Retry, block, or retire with evidence. | `no_file_changes_produced` |
| `malformed_json_artifact` | Fail JSON validation and request repair. | `malformed_json_artifact` |
| `missing_deliverable` | Do not count missing result paths as complete. | `missing_deliverable` |
| `blocked_missing_tool` | Write blocked artifact with exact missing command or package. | `blocked_missing_tool` |

Two cases, `local_verifier_mismatch` and `missing_deliverable`, are marked
held-out in the deterministic script so the search can report whether candidate
policies generalize beyond the visible baseline cases.

## Scoring

The deterministic scalar score is intentionally simple:

```text
score = eval_cases_passed
        - 3 * false_complete_count
        - 2 * gate_error_count
        - 1 * false_block_count
        - 1 * missing_trace_count
        - 5 * hardcoded_leakage_detected
```

The result artifact also reports a Pareto frontier over score, trace
completeness, and runtime. The frontier is the preferred reporting object when
tradeoffs matter.

## Trace Requirements

Every candidate directory under `meta_harness_runs/` must include:

- `policy.md`
- `policy.py`
- `score.json`
- `traces/task_prompt.md`
- `traces/stdout.log`
- `traces/stderr.log`
- `traces/tool_calls.jsonl`
- `traces/gate_evaluation.json`
- `traces/artifact_timeline.jsonl`
- `traces/verifier_outputs.jsonl`
- `traces/diff.patch`
- `results/final_artifact.json`
