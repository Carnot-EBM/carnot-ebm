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
| NO_CLAIM | 7 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8470_v728_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the supplied text reports evidence reconciliation without asserting comparative scientific benefit.

## WAS THAT CHECKED
Not applicable to a benefit claim. Producer checks visibly fail, and independent scientific acceptance is withheld. The oracle flag therefore does not establish a circular value claim here.

## EVIDENCE
- `acceptance_gates`: `scientific_claims_independently_generalize` = `false`
- `generalized_learning_benefit_score` = `0`
- `producer_terminal_replay`: `required_checks_passed` = `false`
- `Authority, complete slot accounting and owned validation qualify the reconciliation; none establishes independent scientific benefit.`

## RECOMMENDATION
KEEP

## experiment_8471_v729_contract_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit claim is made. Satisfying every required current-contract gate would overturn the administrative blocked status; it would not establish scientific benefit.

## WAS THAT CHECKED
Yes, administrative readiness was checked in `gate_check_summary`, where required authority and complete-contract checks failed. The supplied artifact reports methods scaffolding and historical replay. Oracle qualification and exposed labels are acknowledged without claiming verifier value or generalization. No comparative refutation test is applicable.

## EVIDENCE

- `current_contract`: `false`
- `scientific_benefit`: `false`
- `generalized_learning_benefit_score`: `0`
- `current_llm_invocations`: `0`
- `required_authority` — `observed`: `list index out of range`; `passed`: `false`
- `original V728 producer primitive replay; no new benchmark`
- `Bind to sealed invocation operands; administrative or input readiness never establishes scientific benefit.`

## RECOMMENDATION
KEEP

## experiment_8472_source_control_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No substantive claim is asserted. Successful qualification would be refuted by a failed required prerequisite gate.

## WAS THAT CHECKED
Yes—the prerequisite gates were checked. Methods readiness failed, and the artifact reports a blocked run without asserting successful qualification.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `methods_ready_score`; `blocked_reason`: `actual=0 == expected=1`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8475_objective_checkpoint_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-execution receipt, not an assertion that objective updates were successfully qualified.

## WAS THAT CHECKED
No qualification outcome was checked. The prerequisite gates were evaluated; the methods-readiness gate failed, blocking execution.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `blocked_at_layer`: `conductor_pre_gate`.

The `methods_ready_score` gate records `actual`: `0`, `expected`: `1`, and `passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_8478_publication_cost_model.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The proposed hypothesis would be refuted by measurements showing that synchronization remains a bottleneck when stable journal names are used. This artifact asserts no experimental result.

## WAS THAT CHECKED
No. Only upstream readiness gates were evaluated. The methods readiness gate failed, blocking the experiment before it could test the hypothesis. This is a blocking receipt, not a positive or null finding.

## EVIDENCE
`schema`: `blocked_gate_check_v1`  
`status`: `blocked`  
`honest_verdict`: `blocked_gate_check_failed`  
`failed_field`: `methods_ready_score`  
`blocked_reason`: `actual=0 == expected=1`  
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8482_v729_arc_panel_qualification.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8483_v729_board_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit claim is asserted. A qualified current board-operation measurement satisfying the acceptance gates would contradict the recorded blocked qualification status.

## WAS THAT CHECKED
No current device-performance comparison is shown. The displayed operation rows describe host execution, and the cost evidence is explicitly historical. This artifact records qualification boundaries without claiming acceleration, verifier value, or generalization; a comparative refutation opportunity is therefore not required.

## EVIDENCE
- `assigned_substrate`: `host_CPU`
- `deployment_ready`: `false`
- `full_service_acceleration`: `false`
- `current_contract_ready_score`: `0`
- `current_measurement`: `false`
- `generalized_learning_benefit_score`: `0`
- `scope`: `historical_only_zero_current_calls`

## RECOMMENDATION
KEEP

## experiment_8484_v729_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this capstone records blocked aggregation and does not assert comparative scientific benefit.

## WAS THAT CHECKED
No current benefit comparison is established. The artifact explicitly leaves source/Brier/append comparisons unmeasured and generalization open. Authority checks failed, and the reported verdict preserves that blockage. Oracle status and retrospective exposure therefore do not undermine an asserted value claim here.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v729_capstone`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `full_current_authority`: `false`
- `scientific_claims_independently_generalize`: `false`
- `exposure_scope`: `public-label retrospective and historical engineering evidence; independent factuality and hidden-game generalization remain open`

## RECOMMENDATION
KEEP
