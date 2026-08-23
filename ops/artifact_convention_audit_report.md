# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 4 |

## experiment_6530_external_constraint_corpus_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The external-constraint corpus audit was blocked because seven named prerequisites failed.

## WHAT IS MISSING
nothing; `gate_check_summary.failed_checks` records each failed `check` with its `expected` and `observed` values, corroborated by `honest_verdict` and the gate entries in `per_unit_rows`.

## THE CHECK A READER CANNOT DO
none

## experiment_6541_v566_direct_source_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V566 direct-source readiness gate passed because all required source, replay, cache, split, field, dependency, evidence, and file-integrity checks succeeded.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6542_drift_bench_external_intake_v2.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The external drift-bench intake completed successfully, with source hashes, chronological rows, local Z3 receipts, family-blind splits, shards, attacks, and fixture round-trip checks passing.

## WHAT IS MISSING
The artifact is truncated inside `"source_to_local_identity_rows"` and lacks the remainder and closing JSON; although `"honest_verdict"` states that all checks pass and `"upstream_gate_receipt"` records `"expected"`, `"observed"`, and `"passed"`, the visible text does not contain the claimed chronological-row, local-Z3, shard, attack, or fixture-round-trip evidence.

## THE CHECK A READER CANNOT DO
Do all recorded units individually pass the chronological, local-Z3, shard, attack, and fixture-round-trip checks claimed by `"honest_verdict"`?

## experiment_6543_external_corpus_independent_audit_v2.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The external-corpus audit passed its source, fixture, chronology, split, replay, transaction, aggregate, and attack checks.

## WHAT IS MISSING
The artifact is truncated inside `"source_identity_audit_rows"`, so the remainder—including any per-unit rows supporting the `"aggregate"` and `"attack"` pass claims—cannot be inspected; `"honest_verdict"` and source-identity rows are present.

## THE CHECK A READER CANNOT DO
Did the aggregate check pass across the individual units, rather than because of an outlier or degenerate units?

## experiment_6544_external_structural_headroom.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The analytical arm has positive charged-work value over both native and random controls on held units across logic-grid, scheduling, and seating families.

## WHAT IS MISSING
The artifact is truncated mid-row, so the complete `"per_unit_rows"`—especially rows with `"split_name": "held"` and their `"arm_id"`, `"seed"`, `"family"`, and `"total_charged_work_units"` values—cannot be found.

## THE CHECK A READER CANNOT DO
Does the analytical arm beat both controls broadly across all held units and families, rather than through a few outliers or no-headroom units?

## experiment_6545_external_safety_net_router.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The selected `linear_compact_router_abstention_exception_exact_fallback` arm beats the certified structural control on held charged cost while preserving exact equality, calibrated abstention, immutable train-only exceptions, and reachable native fallback.

## WHAT IS MISSING
The complete `"per_unit_rows"` array, specifically held-split rows containing `"arm_id"` and `"charged_total_cost_units"` for both compared arms; the artifact truncates during a train row before any held rows are shown.

## THE CHECK A READER CANNOT DO
Did the selected arm reduce charged cost broadly across the 18 held units, or was its claimed win driven by one outlier or units with no headroom?

## experiment_6546_smt_cost_guard_sota.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Guarded exact dispatch reduces held charged-token or time cost without reducing exact-completion performance across at least two mandated model families.

## WHAT IS MISSING
nothing; `"per_unit_rows"` records `"model_hf_id"`, `"logical_instance_id"`, `"surface_id"`, `"arm_id"`, `"charged_tokens"`, `"charged_time_s"`, and `"exact_valid"` for paired arm comparisons.

## THE CHECK A READER CANNOT DO
none

## experiment_6547_external_transfer_independent_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The analytical/learned router and guarded cost arm passed by reducing charged work, tokens, and time versus their controls while preserving correctness.

## WHAT IS MISSING
Per-unit outcome fields such as charged work, prompt/output tokens, charged time, and exact-completion result for every unit and arm; `"effect_rows"`, `"held_total_charged_work_by_arm"`, `"by_arm"`, and `"model_support_rows"` contain only aggregates, while `"per_unit_rows"` shows audit booleans such as `"cost_matches"` rather than the compared metric values.

## THE CHECK A READER CANNOT DO
Did the reported savings occur broadly across paired units, or were they driven by a few outliers while many units showed no improvement or had no headroom?
