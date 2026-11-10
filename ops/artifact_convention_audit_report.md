# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 2 |
| CANNOT_DETERMINE | 6 |

## experiment_8962_v766_graph_corpus_b.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports 512 completed responses and passing readiness gates.

## WHAT IS MISSING
The artifact cuts off mid-"graph_identity_rows"; "honest_verdict", "status", and "response_rows" are named in "field_principles", but their actual values are not supplied.

## THE CHECK A READER CANNOT DO
Does the final verdict make a comparative claim supported by per-unit metrics?

Unchecked: the truncated remainder, which could resolve the classification.

## experiment_8963_v766_corpus_admission.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible “acceptance_gates” report that every corpus admission gate passed.

## WHAT IS MISSING
The artifact ends mid-string inside “corpus_manifest”; “acceptance_gates” and “class_support_rows” are visible, but the remaining claims and evidence are unavailable.

## THE CHECK A READER CANNOT DO
Does the omitted remainder contain a comparative claim without supporting per-unit metrics?

Only the supplied excerpt was checked; truncation could change the verdict.

## experiment_8964_v766_relational_decisions.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The completed 128-graph study failed to establish relational-energy benefit in all five registered comparisons.

## WHAT IS MISSING
The supplied artifact truncates inside "rows", showing only relational_energy records. Complete paired rows for all arms—including "unit_id", "value", "error", and "selected_50"—are unavailable; "descriptive_rows" and "paired_intervals" provide aggregate results.

## THE CHECK A READER CANNOT DO
Do the paired per-graph results reproduce all five failed comparisons?

The truncated remainder cannot be checked; the comparative null remains unverified.

## experiment_8965_v766_relational_controls.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
No headline claim is visible before the artifact cuts off mid-row.

## WHAT IS MISSING
The remainder of the artifact, including any headline conclusion or verdict. Present "ablation_rows" contain per-unit "unit_id", "brier", "baseline_probability", and "value", but stop during a "confidence_logistic" row.

## THE CHECK A READER CANNOT DO
Does the complete artifact provide per-unit evidence for its headline conclusion?

The missing remainder was not assessed; it could change the classification.

## experiment_8966_v766_online_response_streams.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible "acceptance_gates" reports that authenticated inputs, causal controls, and owned checks passed.

## WHAT IS MISSING
The complete artifact: the supplied text ends mid-"sha256" value inside "adapter_state_hashes", leaving any subsequent claims, metric rows, verdict, or diagnostics unavailable.

## THE CHECK A READER CANNOT DO
Does the omitted remainder contain a comparative claim without per-unit metrics or a blocked verdict without a diagnostic?

The remainder could not be checked; its contents could change this assessment.

## experiment_8967_v766_online_benefit_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V766 protocol failed the required online-benefit gate, so only this exact protocol is retired.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Not checked: experiment execution or referenced files.

## experiment_8968_v766_complete_service_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The adaptive arm did not meet the joint cost-and-quality criterion against confidence_logistic.

## WHAT IS MISSING
The text truncates inside "field_principles": "service_cost_rows", "paired_cost_intervals", and "quality_reference" are described, but their results are absent from the supplied fragment; "durability_rows" provides durability metrics, not per-unit timings or quality outcomes.

## THE CHECK A READER CANNOT DO
Which paired cost or quality result establishes "complete_cost_quality_comparison.favorable": false?

Only the supplied fragment was checked; the missing remainder may contain the required evidence.

## experiment_8969_arc_transfer_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The assessment was blocked because `exp8957-branch-contract.arc_new_outcomes_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
