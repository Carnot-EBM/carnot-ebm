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
| CLAIM_SUPPORTED | 4 |
| NO_CLAIM | 4 |

## experiment_7238_v637_mention_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is made; the administrative blocked-no-run record would be contradicted by completed model calls or comparison rows.

## WAS THAT CHECKED
No scientific refutation was attempted because the experiment was blocked before inference; this is explicitly recorded.

## EVIDENCE
`"honest_verdict": "blocked_exp7238_structured_quarantine"`; `"status": "blocked"`; `"inference_mode": "not_run"`; `"inference_substrate": "blocked_no_run"`; `"model_invoked": false`; `"attempted_calls": 0`; `"completed_comparison_rows": 0`; `"paired_unit_rows": []`; `"rows": []`

## RECOMMENDATION
KEEP

## experiment_7239_v637_semantic_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked, no-run disposition rather than a comparative or value claim.

## WAS THAT CHECKED
No. No comparison was attempted: all 64 units were censored, the comparison rows are empty, and every acceptance criterion was unevaluated.

## EVIDENCE
`"honest_verdict": "blocked_exp7239_structured_quarantine"`; `"inference_substrate": "blocked_no_run"`; `"status": "blocked"`; `"attempted_independent_units": 0`; `"completed_independent_units": 0`; `"censored_independent_units": 64`; `"paired_comparison_rows": []`; `"rows": []`; `"evaluated": false`; `"eligible_for_promotion": false`

## RECOMMENDATION
KEEP

## experiment_7240_v637_recurrence_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
The recurrence fixture, controller, and positive controls are runnable.

## WHAT WOULD REFUTE IT
A failed positive control, incomplete execution, malformed arm rows, or failed acceptance gate would refute the narrow readiness claim; comparative efficacy would require a separate non-oracle evaluation against a serious baseline.

## WAS THAT CHECKED
Yes for runnability: completion, six-arm row count, acceptance gates, and positive controls were checked. No scientific efficacy or added-value claim was tested.

## EVIDENCE
`"complete_circular_positive: recurrence fixture, controller, and positive controls are runnable"`, `"science_efficacy"`, `"not_scored_by_fixture"`, `"pass": null`, `"failed_count": 0`, `"passed_count": 6`, `"status": "complete"`, `"verdict_class": "circular_positive"`, `"verifier_is_oracle": true`

## RECOMMENDATION
KEEP

## experiment_7241_v637_recurrence_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Validated archive reuse did not pass every frozen learning gate.

## WHAT WOULD REFUTE IT
All frozen learning gates passing, with the recurrence-learning value score equal to one.

## WAS THAT CHECKED
Yes, in `acceptance_gate_results` and `recurrence_learning_value_score`; several gates failed, including comparisons against the serious reset baseline and the shuffled control.

## EVIDENCE
`honest_verdict`: `complete_null: validated archive reuse did not pass every frozen learning gate`; `recurrence_learning_value_score`: `0`; `future_error_vs_reset_upper_ci95_lt_zero`: `pass`: `false`; `false_accept_vs_reset_upper_ci95_lte_zero`: `pass`: `false`; `recurrence_error_increase_vs_frozen_lte_0_02`: `pass`: `false`; `recurrence_error_vs_shuffled_upper_ci95_lt_zero`: `estimate`: `0.0`, `pass`: `false`; `verdict_class`: `null`.

## RECOMMENDATION
KEEP

## experiment_7242_v637_recurrence_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The recurrence audit completed, but the method did not satisfy every promotion criterion.

## WHAT WOULD REFUTE IT
A promotion score of 1 with every acceptance gate passing—including superiority to reset/relearn and shuffled controls and recurrence-error increase versus frozen at or below 0.02—would refute the null claim.

## WAS THAT CHECKED
Yes. The acceptance gates directly tested those conditions, including the serious reset/relearn baseline; multiple gates failed, so the potential refutation was given a real chance but did not occur. The oracle limitation does not invalidate this null claim because no positive added-value claim is made.

## EVIDENCE
`honest_verdict`: `complete_null: recurrence audit completed but promotion criteria did not all pass`; `recurrence_promotion_score`: `0`; `false_accept_vs_reset_upper_ci95_lte_zero`: `pass`: `false`; `future_error_vs_reset_upper_ci95_lt_zero`: `pass`: `false`; `recurrence_error_increase_vs_frozen_lte_0_02`: `actual`: `0.09228515625`, `pass`: `false`; `recurrence_error_vs_shuffled_upper_ci95_lt_zero`: `estimate`: `0.0`, `ci95`: `[0.0, 0.0]`, `pass`: `false`; `verdict_class`: `null`.

## RECOMMENDATION
KEEP

## experiment_7243_v637_native_memory.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The native implementation achieved exact oracle parity, but failed the batch-one full-event cost gate against the Python reference.

## WHAT WOULD REFUTE IT
Any scheduled Python/native parity mismatch would refute exact parity; a batch-one lower CI95 above 1 would refute the claim that the cost gate failed.

## WAS THAT CHECKED
Yes. Parity was checked across 32 completed independent streams with mismatch counts, and batch-one cost was compared against the Python reference using 30 paired blocks. The oracle limits parity to an execution-equivalence claim, but the artifact makes no positive verifier-value claim.

## EVIDENCE
`honest_verdict` = `complete_null: exact oracle parity passed but batch-one full-event cost gate failed`; `verifier_is_oracle` = `true`; `native_execution_and_exact_fresh_parity` has `actual` = `1`, `expected` = `1`, `pass` = `true`; `batch_one_total_event_lower_ci95` has `actual` = `0.3632253505817759`, `expected` = `>1`, `pass` = `false`; `arm` = `python_reference`; `arm` = `native_pyo3`; `paired_blocks` = `30`; `independent_stream_units_completed` = `32`; `independent_stream_units_censored` = `0`; `native_archive_cost_value_score` = `0`; `verdict_class` = `null`.

## RECOMMENDATION
KEEP

## experiment_7244_v637_board_disposition.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a read-only disposition receipt, not a comparative claim about a method’s value.

## WAS THAT CHECKED
No comparative refutation was applicable; the artifact checked only disposition completeness, source authentication, and exact next conditions.

## EVIDENCE
`inference_substrate`: `aggregation_from_upstream_artifacts`; `hardware_operations_issued_count`: `0`; `board_disposition_complete_score`: `1`; `model_invocation_count`: `0`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7245_v637_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V637 thirteen-task matrix is complete, while mention science is blocked, recurrence learning and native-cost value are null, and scored-policy evidence is disqualified.

## WHAT WOULD REFUTE IT
An incomplete task matrix, unquarantined evidence establishing mention value or authenticated scored-policy use, recurrence learning passing its reset, recurrence, and shuffled-control gates, or full batch-one native event cost meeting the 10× target would refute the corresponding headline finding.

## WAS THAT CHECKED
Yes. The artifact checks all thirteen task slots, explicitly tests and fails overall claim agreement, applies quarantine and promotion flags to scored-policy and mention evidence, and records failed value outcomes for recurrence learning and native cost. The oracle-linked recurrence and native checks support null findings, not claims of verifier-added value.

## EVIDENCE
`"honest_verdict": "blocked_external_source_value: V637 matrix is complete; mention science remains blocked, recurrence and native cost are null, and scored-policy evidence is disqualified"`; `"criterion": "thirteen_task_matrix_complete"`; `"actual_value": 13`; `"passed": true`; `"criterion": "all_recomputed_claims_match"`; `"actual_value": false`; `"passed": false`; `"source_quarantined": true`; `"value": null`; `"verdict_class": "disqualified"`; `"The complete learner failed reset, recurrence, and shuffled-control gates."`; `"Exact parity completed, but full batch-one event cost was slower than Python."`; `"accepted_for_promoted_evidence": false`; `"promoted": false`; `"error": "quarantined_upstream"`; `"complete": false`; `"full batch-one native event cost did not meet the 10x target"`

## RECOMMENDATION
KEEP
