# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 1 |

## experiment_7550_v660_count_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims an honest null verdict where count claims are qualified but the exploratory effect benefit gate failed because local count-learning lost to the global count control, despite outperforming shuffled local (`holm_passed: true`).

## WHAT IS MISSING
Per-unit rows reporting evaluation metrics for each of the 159 independent source groups or 795 ordered event instances. The artifact contains `rows`, but it records only 4 arm-level aggregate summaries (`"unit_id": "arm-frozen"`, `"arm-global_count"`, `"arm-local_count"`, `"arm-shuffled_local"`), while `independent_reduction.primary_contrasts` provides only pooled bootstrap aggregates and `raw_evidence` merely points to external file paths.

## THE CHECK A READER CANNOT DO
A reader cannot verify whether the null effect against global count and the advantage over shuffled local reflect broad unit-level behavior or were driven by extreme outliers, degenerate control behavior, or floor/ceiling headroom constraints among the 159 source groups.

## experiment_7551_native_pilot.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the pre-gate layer because upstream dependency `exp7548-capture-runner` failed resource availability and verdict status checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7556_v660_arc_corrected_custody.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The corrected B2 evidence is authenticated and ready, but induction efficacy was not observed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_10008_b2_induction_gate_measurement_v2.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All 33 durable responses exhausted the 4,096-token budget in hidden reasoning and emitted empty content, so the run did not measure usable induction output.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_10009_b2_induction_gate_measurement_v3.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible claim is that completion lengths were non-uniform, although two responses ended at the 4,096-token limit.

## WHAT IS MISSING
The artifact is truncated inside `"episode_rows"` at `"elapsed"`, so any later headline `"verdict"` or `"gate_check_summary"` fields—and their diagnostics—cannot be inspected; the visible `"completion_cap_interpretation"` is supported by `"completion_tokens_distribution"`.

## THE CHECK A READER CANNOT DO
Did the complete artifact ultimately declare the experiment blocked or gated, and if so, which check failed at what observed value?

## experiment_7557_v660_arc_generalization.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims an honest null result with feasibility only and no comparative advantage (`positive_claim`: false, `verdict_class`: "null"), failing acceptance gates due to insufficient sample support and missing causal joins.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7558_v660_service_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports an honest null: durable service benchmarking and crash recovery succeeded, but hardware acceleration benefit remains unmeasured, failing the benefit acceptance gate and showing that new hardware purchase is not justified.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7559_v660_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact makes no positive comparative claim and reports that the run was blocked due to an external GPU-capacity precondition failure.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
