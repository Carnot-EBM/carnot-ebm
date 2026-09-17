# Experiment claim-refutation audit

One question per artifact: what would REFUTE the headline claim, and was that
checked? Fabrication is out of scope (adversarial_verify covers it); this audit
targets claims that are true by construction, circular, in-sample, baseline-weak,
or contradicted by their own rows.

This audit never edits an artifact and never blocks anything. It surfaces; the
operator decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity
guard rest on evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CLAIM_SUPPORTED | 1 |
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 2 |

## experiment_7347_v645_plan_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the receipt-level transport assertion, fewer than four completed responses, any censored or failed call, an invalid parse, invalid assignment, or failed runtime provenance would refute readiness; no comparative model-value claim is made.

## WAS THAT CHECKED
Yes. All four call rows expose completion, censoring, parsing, assignments, allowed starts, and runtime identity, while parser controls demonstrate that invalid formatting could be rejected.

## EVIDENCE
`complete_positive_plan_transport_ready_4_of_4_usable`; `generation_calls_attempted`: `4`; `generation_calls_completed`: `4`; `generation_calls_failed`: `0`; `censored`: `false`; `parse_status`: `valid`; `terminal_state`: `response`; `cuda_provenance_ok`: `true`; `cpu_invalid_format`; `observed`: `invalid`; `promotion_ready_score`: `0`

## RECOMMENDATION
KEEP

## experiment_7348_v645_plan_capture.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7349_prospective_learning.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the intended learning claim, distinct future requests would need to show no structural improvement—or no advantage over a serious non-learning baseline—but this blocked artifact contains no experimental outcomes.

## WAS THAT CHECKED
No. Execution stopped at `conductor_pre_gate`; only prerequisite gates were evaluated.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"gate-unsat(final): 5 of 6 gate(s) failed"`, `"blocked_at_layer": "conductor_pre_gate"`, `"duration_s": 0.0`

## RECOMMENDATION
KEEP

## experiment_7351_v645_acquisition_prototype.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed experiment is disqualified because required validation or prototype-safety/value gates failed.

## WHAT WOULD REFUTE IT
All required checks passing—particularly affected validation and cost-value—with no disqualifying safety failure would falsify the headline.

## WAS THAT CHECKED
Yes. The acceptance gates explicitly tested affected validation, cost-value, adversarial clearance, resource bounds, infeasible outputs, and terminal validation; the first two failed. The repository-wide required test observation also failed and timed out. The exact-oracle circularity does not rescue a positive value claim because the artifact makes none and assigns zero value/readiness.

## EVIDENCE
`honest_verdict`: `complete_disqualified: current required validation or prototype safety failed`; `verdict_class`: `disqualified`; `required_checks_passed`: `false`; `affected_validation`: `observed`: `false`, `passed`: `false`; `cost_value`: `observed`: `false`, `passed`: `false`; `acquisition_value_score`: `0`; `acquisition_prototype_ready_score`: `0`; `finite_bias_utility`: `327.5`; `conservative_utility`: `327.5`; `timed_out`: `true`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_7352_acquisition_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. Any future claim of acquisition-strategy superiority would be refuted if a serious baseline tied or outperformed the tested method at the full execution boundary.

## WAS THAT CHECKED
No. Execution was blocked at the pre-gate, so no acquisition strategies or comparator results were evaluated.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"blocked_at_layer": "conductor_pre_gate"`; `"gate_check_summary": "gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7351-acquisition-prototype.acquisition_prototype_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_7354_v645_arc_transfer.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7355_v645_board_state.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a read-only state/receipt artifact, not a comparative value claim; its accounting record would be contradicted by an eligible post-Exp6559 operator-authored physical-change receipt, any current hardware operation, or any nonzero readiness, value, or promotion score.

## WAS THAT CHECKED
Yes. The artifact searched the named approved receipt sources, recorded zero accepted receipts, enumerated operation counts, and reported the relevant scores as zero. The oracle circularity does not affect a verifier-value claim because no such claim is made.

## EVIDENCE
`inference_substrate`: `aggregation_from_upstream_artifacts`; `current_evidence`: `read_only_artifact_aggregation`; `accepted_receipt_count`: `0`; `exists`: `false`; `hardware_operations_issued_count`: `0`; `hardware_readiness_score`: `0`; `hardware_value_score`: `0`; `hardware_promotion_score`: `0`; `scientific_value_score`: `0`; `model_invoked`: `false`; `status`: `blocked`; `Accounting completeness does not assert board availability.`

## RECOMMENDATION
KEEP

## experiment_7356_v645_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
The capstone completed fourteen-task disposition accounting while withholding scientific promotion because the required evidence remained unavailable.

## WHAT WOULD REFUTE IT
A disposition count other than fourteen, an unclassified task, available qualifying science, or a positive readiness/value/promotion score would refute the administrative headline; there is no comparative efficacy claim to falsify.

## WAS THAT CHECKED
Yes. The exact roster count, required-science availability, and promotion/readiness/value scores were explicitly checked, and the five claim rows record abstention or blocking rather than efficacy promotion. The oracle circularity therefore does not create a circular value claim here.

## EVIDENCE
`"fourteen_dispositions"`, `"observed": 14`, `"passed": true`, `"required_science_available"`, `"observed": false`, `"passed": false`, `"capstone_complete_score": 1`, `"capstone_promotion_score": 0`, `"capstone_readiness_score": 0`, `"capstone_value_score": 0`, `"promotes_scientific_efficacy": false`, `"metric_kind": "disposition_accounting_only"`, `"verdict_class": "blocked"`, `"inference_substrate": "aggregation_from_upstream_artifacts"`

## RECOMMENDATION
KEEP
