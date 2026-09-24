# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| AGGREGATE_ONLY | 3 |
| CANNOT_DETERMINE | 1 |

## experiment_7579_v662_decision_learning_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The audit completed valid static and learning measurements but found no supported static or learning benefit and found harmful retention recalibration.

## WHAT IS MISSING
The complete `"rows"` array: the artifact is truncated mid-row, and the visible `"rows"` contain only `"row_kind": "static_source_arm"` entries, so it is unknown whether per-unit learning and retention rows underlying `"learning_interval_reduction"`, `"online_reduction"`, and `"retention_reduction"` are recorded later.

## THE CHECK A READER CANNOT DO
Were the reported learning null and retention degradation broad across individual units, or driven by a few outliers or floor/ceiling-pinned units?

## experiment_7580_v662_arc_verifier_support.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The verifier integrity guard achieves support readiness on deterministic test fixtures while explicitly disclaiming live model benefit because no live panel or model call was run.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7581_v662_arc_bounded_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked before model invocation due to a failed `arc_e2e` acceptance gate check where three validation commands failed with non-zero exit codes.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7582_arc_panel_a.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked prior to execution because 3 of 7 evaluated upstream gate checks failed on exp7581.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7584_v662_arc_independent_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment is blocked because required upstream producer artifacts for live panel A and panel B are missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_10010_b2_think_on_pilot.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Think-on achieved mean change fidelity 0.574 across 10 windows versus 0.128 for code-only.

## WHAT IS MISSING
Per-window think-on metric rows corresponding to the aggregate in `"honest_verdict"`; `"windows"` contains only window metadata, while `"controls.per_window"` records control metrics but no matching think-on results.

## THE CHECK A READER CANNOT DO
Did think-on improve broadly across the ten windows, or was the reported mean advantage driven by only one or two outlier windows?

## experiment_7585_v662_portable_service.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The Rust service delivered a statistically supported whole-service speedup over Python, including a warm lower-95% speedup of 31.122×.

## WHAT IS MISSING
Per-repeat cold and warm service rows containing paired Python and Rust complete-service latency measurements; `"rows"` as shown contains only `"parity_stream"` rows, while `"service_latency_summary"`, `"whole_service_speedup"`, and `"raw_measurement_receipts"` provide aggregates or a path/hash to `service_rows.json`, not the service rows themselves.

## THE CHECK A READER CANNOT DO
Was the reported service speedup broad across the 30 paired repeats, or driven by a few extreme Python/Rust latency pairs?

## experiment_7586_v662_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Rust improved the equal-durability host service, with reported cold and warm speedups over Python.

## WHAT IS MISSING
Per-pair Python and Rust latency rows for all 30 cold and 30 warm measurements are missing; `"whole_service_speedup"` provides only `"estimate"`, `"lower95"`, `"upper95"`, and `"pair_count"`. The blocked branch is diagnosed by `"gate_check_summary"` with four failed checks, so no blocker diagnostic is missing.

## THE CHECK A READER CANNOT DO
Were the reported Rust speedups broad across the 30 pairs, or driven by a few extreme Python timings?
