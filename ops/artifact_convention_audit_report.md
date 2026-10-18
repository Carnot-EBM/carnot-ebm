# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 2 |

## experiment_8620_v740_triviaplus_extraction.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The extraction audit is ready and reports a complete null, with no selected winner.

## WHAT IS MISSING
The remainder of "extraction_rows": the artifact cuts off mid-"fixed_scores", while "completed_count" is 190 and "descriptive_auroc_intervals" reports 135 usable rows.

## THE CHECK A READER CANNOT DO
Do all 135 usable rows’ "fixed_scores" and "binary_target" values reproduce the reported AUROCs?

## experiment_8621_v740_lean_resident_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The listed validation commands passed; no comparative research claim or blocked verdict is recorded.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8622_v740_lean_resident_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
At length 16 and batch 1, `lean_native` outperforms `lean_python` in throughput and p95 request latency, with full parity.

## WHAT IS MISSING
Per-repeat paired throughput and latency measurements are absent from the visible text: `"arithmetic_contrast"` contains only `"paired_repeats"`, `"throughput_ratio_ci95"`, and `"p95_request_latency_ratio"`. The artifact cuts off inside `"cited_upstream_artifacts"`, so whether measurement rows appear later cannot be determined; paths and hashes alone supply no metric values.

## THE CHECK A READER CANNOT DO
Do the ten paired repeats show a consistent native advantage, or does an outlier drive the reported improvement?

## experiment_8624_v740_arc_outcome_observer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The outcome observer is mechanically ready, but this run establishes no causal benefit, solve credit, or promotion readiness.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8625_v740_hardware_operation_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Compatible physical execution and learning costs remain unqualified; the reported speedup ceilings are conditional calculations, not measured acceleration.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8626_v740_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Lean native provides a qualified benefit over legacy native, with a throughput ratio confidence interval of 2.602–2.714 across 10 paired repeats.

## WHAT IS MISSING
Per-repeat baseline and treatment throughput and request-latency measurements for all 10 pairs. “primary_contrast” records “throughput_ratio_ci95” and “p95_request_latency_ratio”; “rows” contains only R0 primitive timings in “partition_s” and “total_s”.

## THE CHECK A READER CANNOT DO
Did the throughput benefit occur consistently across the 10 paired repeats, or depend on a few outliers?

## experiment_8618_causal_brier_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Execution was blocked because upstream `learning_execution_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8623_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8623 was blocked because `runtime_repair_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
