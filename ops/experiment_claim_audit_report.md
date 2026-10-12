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
| NO_CLAIM | 8 |

## experiment_8444_source_control_panel.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental claim to falsify. A passing readiness gate would contradict only the blocking receipt.

## WAS THAT CHECKED
Yes, the readiness gate was checked: its observed value was 0 against a required 1, and it failed. This records a blocked prerequisite, not an experimental outcome.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `failed_field`: `methods_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8447_brier_online_learning.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact records a blocked attempt to run a comparison, without asserting a comparative outcome.

## WAS THAT CHECKED
No comparison was executed. The prerequisite checks detected a failed methods-readiness gate and blocked execution.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`, `"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_field": "methods_ready_score"`, `"blocked_reason": "actual=0 == expected=1"`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_8449_v727_persistence_localization.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked measurement receipt. It asserts neither comparative benefit nor an observed zero effect.

## WAS THAT CHECKED
No empirical comparison completed. All 20 intended observations are censored; the displayed baseline and instrumented timings and paired overheads are null. Authority checks failed. Passing software validation does not establish a performance result, and missing measurements do not establish a tie.

## EVIDENCE

- `honest_verdict`: `complete_blocked_persistence_localization`
- `verdict_class`: `blocked`
- `completed_count`: `0`
- `censored_count`: `20`
- `complete_pairs`: `false`
- `complete_wall_s`: `null`
- `missing_reason`: `required_authority_or_operand_absent`
- `methodology_note`: `Complete disk-backed service clocks preserve original delayed feedback. Nested exclusive spans isolate serialization and storage. Instrumentation overhead stays visible. No arithmetic acceleration or learning benefit is inferred.`

## RECOMMENDATION
KEEP

## experiment_8450_journal_equivalence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the title describes a qualification objective, while the artifact reports a blocked prerequisite gate. It does not assert journal equivalence or unchanged durable acknowledgments.

## WAS THAT CHECKED
No journal-equivalence test is reported. The prerequisite gate was checked and failed: expected 1, observed 0. The blocked status accurately reflects that failure.

## EVIDENCE

- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8453_v727_arc_seal_and_support.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to falsify. The limited receipt qualification would fail if intact original/restored receipts were rejected, or missing, forged, or rehashed receipts were accepted.

## WAS THAT CHECKED
Yes. Custody recovery receipts show original and restored replay exiting 0, and all three corruption controls exiting 1. Support gates record insufficient game overlap and zero resolved outcomes. The artifact reports blocked refinement and makes no generalization claim.

## EVIDENCE
- `honest_verdict`: `complete_blocked_v727_arc_seal_and_support`
- `arc_receipt_ready_score`: `1`; `arc_support_ready_score`: `0`; `arm_refinement_authorized`: `false`
- Within `receipts`, `original` and `restored_cold` have `actual_exit`: `0`; `missing`, `forged`, and `rehashed` have `actual_exit`: `1`.
- `scientific_generalization`: `Exposed engineering evidence supplies no hidden-game or causal learning claim.`
- `censoring_counts`: `resolved_redirects`: `0`; `pending_redirects`: `12`

## RECOMMENDATION
KEEP

## experiment_8454_v727_board_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records engineering continuity and blocked readiness, without asserting comparative acceleration or generalization.

## WAS THAT CHECKED
No comparative refutation test is shown. The acceptance gates and board operation rows document unsupported operations and incomplete clocks; these are readiness records, not evidence of added value.

## EVIDENCE
- `exposure_scope`: `historical_public_labels_retrospective_overlap_unknown; engineering_continuity_only`
- `current_contract_ready_score`: `0`
- `compatible_operations`: `false`; `complete_transfer_device_clocks`: `false`
- `full_service_acceleration`: `false`
- `Bind exact invocation and immutable producer evidence; missing clocks remain null and continuity grants no scientific or acceleration benefit.`

## RECOMMENDATION
KEEP

## experiment_8455_v727_gatemate_obligation.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. An obligation row documenting authenticated completion would contradict the receipt’s reported zero completions.

## WAS THAT CHECKED
Yes, at the receipt level: all seven obligation rows are censored and incomplete, and no evidence delta was supplied. This documents a blocked state. It does not assert verifier value, scientific benefit, or generalization, so oracle circularity does not invalidate a value claim here.

## EVIDENCE
- `honest_verdict`: `complete_blocked_gatemate_obligation`
- `verdict_class`: `blocked`
- `arm`: `read_only_obligation`
- `completed_count`: `0`; `censored_count`: `7`
- `censored`: `true`; `completed`: `false`
- `supplied_evidence_delta`: `[]`
- `scientific_benefit`: `false`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8456_v727_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the supplied artifact reports aggregation and readiness checks without asserting comparative scientific benefit.

## WAS THAT CHECKED
Not applicable to a comparative claim. The visible acceptance gates record failures, and independent generalization remains explicitly open. Oracle-defined verification does not create a circular value claim when no added value is asserted.

## EVIDENCE

- `scientific_claims_independently_generalize`: `false`
- `full_current_authority`: `false`
- `generalized_learning_benefit_score`: `0`
- `exposure_scope`: `public-label retrospective and historical engineering evidence; independent factuality and hidden-game generalization remain open`
- `Authority, complete slot accounting and owned validation qualify the reconciliation; none establishes independent scientific benefit.`

## RECOMMENDATION
KEEP
