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
| NO_CLAIM | 7 |

## experiment_8498_v730_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative headline is asserted. Qualified independent evidence of benefit could change the blocked assessment; the artifact does not claim that benefit is absent.

## WAS THAT CHECKED
No comparative falsification test is established in the supplied excerpt. The displayed gate checks record qualification failures, and the capstone withholds independent generalization. Upstream readiness receipts and historical publication gates are not presented as current scientific success.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v730_capstone`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `scientific_claims_independently_generalize`: `false`
- `exposure_scope`: `public-label retrospective and historical engineering evidence; independent factuality and hidden-game generalization remain open`
- `Authority, complete slot accounting and owned validation qualify the reconciliation; none establishes independent scientific benefit.`

## RECOMMENDATION
KEEP

## experiment_8499_v731_contract_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. An assertion that the current contract was ready would be refuted by failed independent-authority or task-completeness checks.

## WAS THAT CHECKED
Yes, for administrative readiness: gate_check_summary records those failures, and current_contract_ready_score is zero. This is a blocked administrative receipt. Oracle verification establishes no claimed added value; exposed historical observations receive no generalization or benefit credit. A serious comparative baseline is therefore unnecessary for its stated scope.

## EVIDENCE

- `exposure_scope`: `administrative contract; released human labels exposed; older overlap unknown`
- `current_contract_ready_score`: `0`
- `scientific_benefit`: `false`
- `required_authority`: `observed` is `list index out of range`; `passed` is `false`.
- `independent_visible_task_table`: `expected` is `13`; `observed` is `null`; `passed` is `false`.
- `scope`: `historical observation only; no new qualification or benefit`
- `raw_score_gains`: `exploratory exposed-panel only`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8500_strongest_baseline_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact reports a blocked prerequisite check, without claiming comparative performance or added value.

## WAS THAT CHECKED
No comparative experiment was evaluated. The prerequisite was checked in gates_evaluated and failed: methods_ready_score was 0 against an expected 1.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `methods_ready_score`
- `blocked_reason`: `actual=0 == expected=1`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8503_objective_publication_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked prerequisite-check receipt, with no claim of successful qualification or comparative value.

## WAS THAT CHECKED
No experimental claim was tested. The prerequisite was checked in `gates_evaluated`: methods readiness was 0 against required 1, and the gate failed.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `methods_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8506_append_import_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental success claim to falsify. Both required gates passing would contradict the receipt’s reported blocking condition.

## WAS THAT CHECKED
Yes, in the gate evaluation: methods readiness failed (0 versus required 1), while service-input readiness passed. Append-replay qualification did not run, so this artifact provides no evidence of its success.

## EVIDENCE
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `methods_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `gate_check_summary`: `gate-unsat(final): 1 of 2 gate(s) failed; first failure: exp8499-contract-methods.methods_ready_score (actual=0 == expected=1)`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8509_v731_arc_redirect_effect_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
All 32 live redirect selections remain right-censored, so treatment benefit cannot be estimated and arm refinement lacks outcome support.

## WHAT WOULD REFUTE IT
A valid, uncensored live outcome would contradict the all-censored finding. Resolved outcomes meeting the stated threshold—three overlapping games and five outcomes per shared arm cell, with modes separate—would contradict the reason for withholding refinement.

## WAS THAT CHECKED
Yes. `effect_support_by_arm` counts selected, resolved, and censored events separately; `per_game_results` reports resolution by game; `leave_one_game_out` checks support gates. Every reported cell has zero resolved outcomes, and the gates fail. Private fixtures remain separate from live evidence. This supports the evidence-insufficiency conclusion; causal benefit itself was not tested.

## EVIDENCE

- `honest_verdict`: `complete_null_v731_arc_redirect_effect_audit`
- `censored_count`: `32`
- `resolved_count`: `0`
- `support_gate_passed`: `false`
- `arm_refinement_authorized`: `false`
- `fixture_is_live_evidence`: `false`
- `All32 events remain right-censored. No failure rate or scientific benefit is estimable.`

## RECOMMENDATION
KEEP

## experiment_8510_v731_board_cost_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A qualified current board-service measurement with complete device and transport clocks would contradict the recorded blocked boundary. No comparative acceleration or generalization benefit is asserted.

## WAS THAT CHECKED
No current device comparison is shown. The displayed operation rows record host CPU execution with incomplete clocks. Historical acceleration ceilings are explicitly counterfactual, and missing current append evidence remains unqualified. This artifact records evidence boundaries without claiming comparative success.

## EVIDENCE

- `Bind V731 invocation and original producer hashes; recording gives no deployment or benefit.`
- `assigned_substrate`: `host_CPU`
- `deployment_ready`: `false`
- `device_clock_ns`: `null`; `transport_clock_ns`: `null`
- `counterfactual`: `true`; `deployed_speed`: `false`
- `fallback_is_current_append_measurement`: `false`

## RECOMMENDATION
KEEP

## experiment_8511_v731_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the supplied excerpt reports administrative reconciliation and leaves independent scientific benefit unestablished. It asserts no comparative improvement to falsify.

## WAS THAT CHECKED
Not applicable to a comparative headline. The acceptance gates explicitly record failed authority and scientific generalization. The oracle flag therefore does not undermine an asserted verifier-value claim here. Comparative performance and row-level validity cannot be assessed from the displayed excerpt.

## EVIDENCE

- `administrative_complete`: `true`
- `full_current_authority`: `false`
- `scientific_claims_independently_generalize`: `false`
- `generalized_learning_benefit_score`: `0`
- `Bind exact producer bytes and this invocation; administrative completion grants no scientific benefit.`
- `public-label retrospective and historical engineering evidence; independent factuality and hidden-game generalization remain open`

## RECOMMENDATION
KEEP
