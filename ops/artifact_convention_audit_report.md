# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 3 |

## experiment_6760_prefix_backtracking_repair_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because both prerequisite gates failed when the upstream artifact `exp6759-oracle-distinct-diagnostic-energy-v2` could not be found.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6761_procedural_memory_stream.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims the six-order procedural-memory stream passed all readiness gates, including matched capacity, nonzero accepts and rejects, chronology, restart, rollback, and poison checks.

## WHAT IS MISSING
The artifact is truncated mid-JSON, so the promised top-level `"rows"` data cannot be found or confirmed; only aggregate fields such as `"eligible_accepts_by_order"`, `"eligible_rejects_by_order"`, and boolean `"gate_check_summary.checks"` are visible.

## THE CHECK A READER CANNOT DO
Did every order-and-event row actually support the reported readiness gates, rather than the aggregate results being driven by exceptional or degenerate units?

## experiment_6762_procedural_vs_trace_csl_ab.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be determined because the artifact ends mid-record.

## WHAT IS MISSING
The actual values of `"honest_verdict"`, `"verdict_class"`, `"status"`, `"prospective_csl_completed"`, `"gate_check_summary"`, and `"rows"`; these names appear only inside `"field_principles"` before the artifact truncates.

## THE CHECK A READER CANNOT DO
Was the experiment blocked, and if so, which gate failed with what observed value?

## experiment_6763_csl_hard_case_forgetting_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp6762-procedural-vs-trace-csl-ab.prospective_csl_completed` was `false` but required to equal `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6764_arc_exclusive_load_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Both specified models completed owned local CUDA loading, production self-parse dispatch, teardown, lease release, and VRAM recovery, making exclusive ARC loading ready.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6765_object_table_fetch_ab_v2.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the `arc_exclusive_load_ready` preflight check failed, so no comparative result was produced.

## WHAT IS MISSING
nothing; the present `"status": "blocked"`, row-level `"stop_reason": "preflight_blocked"`, `"failure_class": "preflight_blocked:arc_exclusive_load_ready"`, and `"live_model_invoked": false` identify the blocker.

## THE CHECK A READER CANNOT DO
none

## experiment_6766_thermalizer_independent_trajectory_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The independent evaluator reproduced the context-matching trajectory reduction, with context-matched and trajectory-refinement methods outperforming the independent-factor method.

## WHAT IS MISSING
The complete `"rows"` array containing per-unit `"trajectory_tv"` values for every pair summarized by `"paired_trajectory_deltas"`; the artifact is truncated during its fourth visible row despite reporting 192 source rows.

## THE CHECK A READER CANNOT DO
Do the 64 paired per-unit differences support the reported pooled improvement, or is it driven by a small number of outliers or degenerate units?

## experiment_6767_v589_branch_disposition.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The simulator audit claims context-matched and trajectory-refinement methods reduced trajectory TV versus the independent-factor method, while the overall milestone remained partial or blocked elsewhere.

## WHAT IS MISSING
Per-unit metric rows for the 64 paired cases: condition/cell identifiers and each method’s trajectory TV or paired delta. `"paired_all_depth_deltas"`, `"mean_trajectory_tv_by_method"`, and `"rows"` are present, but `"rows"` contains experiment-level receipts rather than the underlying paired measurements.

## THE CHECK A READER CANNOT DO
Were the reported mean TV improvements broad across the 64 pairs, or driven by a few extreme cases?
