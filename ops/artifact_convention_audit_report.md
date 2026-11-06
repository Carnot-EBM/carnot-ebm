# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| CANNOT_DETERMINE | 5 |

## experiment_8858_v758_installed_indexed_verifier.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The installed verifier matched all 3,614 constructed requests and met all acceptance gates.

## WHAT IS MISSING
The supplied artifact cuts off after 240 entries in "operation_parity_rows"; the remaining 3,374 request results supporting "completed_count": 3614 and "installed_index_ready_score": 1 are missing.

## THE CHECK A READER CANNOT DO
Did all 3,614 requests pass exact installed parity?

Not rerun; full-roster parity remains unverified.

## experiment_8859_v758_split_certificate_evaluation.json

**CANNOT_DETERMINE**

## VERDICT

CANNOT_DETERMINE

## WHAT THE CLAIM IS

Visible rows label learned_energy useful, with zero errors and 50.56% decisive coverage, but the headline verdict is not visible.

## WHAT IS MISSING

The final verdict and gate results: "acceptance_gates" and per-unit "census_risk_rows" are present, but the artifact cuts off mid-field at "metr".

## THE CHECK A READER CANNOT DO

Did the final verdict report that all acceptance gates were met, or that a particular check blocked acceptance?

The omitted remainder was not checked; it may contain the missing verdict and diagnostics.

## experiment_8860_v758_linear_benefit_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“addition_benefit_score”: 1 reports addition benefit, while “causal_benefit_score”: 0 reports no causal benefit.

## WHAT IS MISSING
The artifact cuts off inside “delay_subgroups”. Visible “causal_integrity_rows” provides integrity flags per “cluster_id”, but no performance metrics; visible “delay_subgroups” provides pooled “brier” and “nll”. Paired per-cluster performance rows and the final verdict are unavailable.

## THE CHECK A READER CANNOT DO
Is the reported addition benefit spread across stream clusters or driven by a few outliers?

The missing remainder was not checked; it could contain the required evidence.

## experiment_8861_v758_installed_index_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“benefit_gates” asserts improved mixture speed without coverage loss or increased false accepts, while reporting failure of the size-16 latency gate.

## WHAT IS MISSING
Paired per-unit measurements supporting those comparisons are unavailable in the supplied excerpt: “censoring_rows” shows censored size-1024 observations, and the artifact ends mid-record. The recorded “upper” of 0.3017865229990757 exceeds “threshold” 0.05, so that gate has a diagnostic.

## THE CHECK A READER CANNOT DO
Do paired timings across all sizes support “mixture_speed_lower_above_one”?

Only the supplied excerpt was assessed; the truncated remainder may contain the required rows.

## experiment_8862_v758_linear_durable_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The linear durable-cost experiment completed all 60 units and reports a null result, with per-unit movement measurements and per-block cost differences recorded.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Referenced files and execution were not checked; this verdict covers only the supplied text.

## experiment_8863_qwen_source_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp8853-branch-contract.runtime_repair_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8864_v758_hardware_budget.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Hardware acquisition should be deferred because no complete, supported physical workload benefit has been measured.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8865_v758_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports administrative and method readiness while withholding activation and natural-generalization approval.

## WHAT IS MISSING
The artifact cuts off inside experiment 8857’s "draw_ids". The overall conclusion and any subsequent evidence are unavailable; visible "stream_ids" and "scores" contain no per-stream outcome metrics.

## THE CHECK A READER CANNOT DO
Does the complete artifact make a comparative claim supported by per-unit metrics?

Unchecked: the truncated remainder; its contents could change this verdict.
