# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 3 |

## experiment_7508_v657_static_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The static audit completed but was disqualified because required current validation failed and the scientific benefit claims were not qualified.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7509_v657_causal_online.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The causal measurement was valid, but the online-benefit gate failed because support, effect-size, and all-five-contrast requirements were not met.

## WHAT IS MISSING
Per-unit comparative metric rows—such as the actual `"per_source_results"` or seed-level arm deltas underlying `"measurement_reduction.primary_contrasts"`; only aggregate `"mean_delta"`, `"upper95_delta"`, p-values, and `"source_count"` are shown. The blocker itself is diagnosed in `"gate_check_summary"`.

## THE CHECK A READER CANNOT DO
Did the reported pooled Brier-score difference occur broadly across sources and schedule seeds, or was it driven by a small number of outliers or degenerate units?

## experiment_7510_v657_causal_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment yielded an honest null verdict (`complete_null_v657_causal_audit_benefit_gate_failed`) because online causal adaptation failed the primary support and primary contrast benefit gates.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7511_v657_arc_evidence_recovery.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Panel B qualified on validity and readiness, but showed no scientific benefit because 0 of 18 episodes progressed.

## WHAT IS MISSING
The actual `"per_episode_results"` and `"per_game_results"` rows containing each unit’s `"progressed"` value are missing; only aggregate gate observations and unrelated per-unit `"exclusive_cost_rows"` are present.

## THE CHECK A READER CANNOT DO
Did every scheduled episode genuinely record zero progress, or was the aggregate produced from degenerate, censored, or no-headroom units?

## experiment_7512_v657_arc_opportunity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The completed audit found zero eligible supervisor opportunities, no game-level progress, and insufficient denominator coverage for a numeric cost claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7513_v657_placement_continuity.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Integer placement is numerically ready because int8/int16 match float32 policy actions across 472 rows, with zero overflow and probability error below 0.01, while whole-service speedup remains unmeasured.

## WHAT IS MISSING
The actual per-unit `"quantization_rows"` containing each group’s arithmetic format, probability error, action comparisons, and overflow result; only aggregate fields such as `"action_disagreement_count"`, `"max_abs_probability_error"`, and `"overflow_count"` plus an `"evidence_sidecars"` path and hash are present. The GateMate block is diagnosed by `"gate_check_summary"` and `"accepted_receipt_count": 0`.

## THE CHECK A READER CANNOT DO
Were parity and acceptable error broad across all 472 units, or were some units degenerate, pinned, or hiding materially worse errors beneath the reported maximum and zero aggregate disagreement?

## experiment_7514_v657_service_trace.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The controlled service trace is complete and gate-ready, while making no efficacy, measured-100×, or SLA claim.

## WHAT IS MISSING
nothing; `acceptance_gate_results` records each check’s `expected`, `observed`, and `passed` values, `gate_check_summary` records no failures, and `native_call_rows` provides per-unit arm-level measurements.

## THE CHECK A READER CANNOT DO
none

## experiment_7515_v657_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V657 capstone is complete but disqualified because Exp7508 failed the required strict row-consistency validation and the qualified benefit gates were not met.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
