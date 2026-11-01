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
| CLAIM_SUPPORTED | 2 |
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8667_v744_contract_methods.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8668_frozen_learning_core.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt with no claim of successful qualification or comparative value.

## WAS THAT CHECKED
Yes, the prerequisites were checked in `gates_evaluated`: the methods readiness gate failed with actual 0 against expected 1. The artifact explicitly reports blockage.

## EVIDENCE
`status`: `blocked`  
`honest_verdict`: `blocked_gate_check_failed`  
`failed_field`: `methods_ready_score`  
`blocked_reason`: `actual=0 == expected=1`  
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8670_relational_evidence_controls.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No method-performance claim is made. The blockage diagnosis would be contradicted if both prerequisite gates passed.

## WAS THAT CHECKED
Yes, for the blockage diagnosis: `gates_evaluated` shows methods readiness failed while extraction-input readiness passed. No comparative experiment is reported.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"artifact_field": "methods_ready_score"`; `"expected": 1`; `"actual": 0`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`.

## RECOMMENDATION
KEEP

## experiment_8674_installed_batch_prototype.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No experimental claim is asserted. The blocked-gate report would be contradicted if all required gate rows passed.

## WAS THAT CHECKED
Yes. The gate rows show methods_ready_score failing with actual 0 against expected 1, while deployment_inputs_ready_score passes. This is a prerequisite-check receipt; no batch qualification result is reported.

## EVIDENCE
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `methods_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`
- `artifact_field`: `deployment_inputs_ready_score`; `expected`: `1`; `actual`: `1`; `passed`: `true`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8677_v744_arc_observation_ownership.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Observation-delivery ownership passes diagnostic controls and preserves behavior on the tested pairs, without establishing solve, promotion, or generalization benefit.

## WHAT WOULD REFUTE IT
Any paired difference in actions, observations, final state, resets, RNG, termination, or legacy behavior; repeated callbacks advancing the observation horizon; or a conflicting duplicate being accepted without rejection.

## WAS THAT CHECKED
Yes. All eight action-parity rows report equality across those dimensions. Constructed controls exercise repeated callbacks, conflicting duplicates, full horizons, resets, and termination, and report passing results. These checks could have failed.

The recorder-off comparator ties, supporting behavioral preservation rather than added performance. Constructed controls establish diagnostic behavior only; the artifact expressly withholds broader benefit credit.

## EVIDENCE
- `honest_verdict`: `complete_null_arc_observation_ownership`
- `methodology_note`: `Single delivery ownership is diagnostic readiness only. Constructed horizon controls establish no natural supervisor benefit. Seeds and recorder twins are dependent; no solve, policy promotion or generalization credit.`
- `equality`: `actions`, `final_state`, `legacy`, `observations`, `reset_requests`, `rng`, `termination`; each `true`
- `case`: `repeated_is_done`; `observed_horizon`: `3`
- `reason`: `conflicting_duplicate`; `horizon_status`: `censored`
- `headline_solve_credit`: `0`; `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8678_v744_arc_observer_transfer.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Across four development games, recorder off/on execution matched exactly, while all 16 live observer windows were censored and established no transfer benefit.

## WHAT WOULD REFUTE IT
Any recorder off/on pair differing in actions, observations, RNG, or final state; or any completed live observation horizon contradicting the all-censored accounting.

## WAS THAT CHECKED
Yes. Eight action-parity rows check execution equality across four games and two seeds. Live horizon accounting reports 16 censored windows and zero complete horizons; constructed controls exercise completion, reset, and conflict handling.

The recorder-off comparator ties exactly, supporting the parity claim. Censoring leaves efficacy unresolved, and the artifact makes no positive efficacy or generalization claim.

## EVIDENCE

- `honest_verdict`: `complete_null_arc_observer_transfer`.
- All eight `action_parity_rows` report `passed`: `true`, with `actions`, `observations`, `final_state`, and `rng` each `true`.
- `censored_count`: `16`; `complete_horizon_count`: `0`.
- `diagnostic_disposition`: `closed_censored_under_unchanged_protocol`.
- `population_efficacy_claim`: `false`.
- `generalized_learning_benefit_score`: `0`; `independent_generalization_score`: `0`; `headline_solve_credit`: `0`.

## RECOMMENDATION
KEEP

## experiment_8679_v744_hardware_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. A qualified compatible hardware execution with complete cost measurements would contradict the narrower recorded blocked status.

## WAS THAT CHECKED
No physical performance test was performed. The gate summary records missing upstream cost artifacts, and board status rows report no physical execution. This is an evidence aggregation and readiness receipt; it asserts no hardware speedup, verifier benefit, or generalization result.

## EVIDENCE
- `honest_verdict`: `complete_blocked_v744_hardware_operation_boundary`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `physical_execution_this_invocation`: `false`
- `complete_new_cost_boundaries`: `false`
- `achieved_speedup`: `null`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8680_v744_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative efficacy claim to falsify. Qualified measurements establishing benefit within these rows would contradict the reported unmeasured status; unmeasured benefit does not imply ineffectiveness.

## WAS THAT CHECKED
No comparative efficacy test was completed here. This aggregation records failed or missing gate operands, censored ARC horizons, and unexecuted deployment comparisons. It does not promote those rows into efficacy or generalization evidence.

## EVIDENCE
- `exposure_scope`: `Exposed upstream evidence and deterministic constructed registration; no independent generalization`
- `status`: `diagnostic_completed_efficacy_unmeasured`
- `complete_horizon_count`: `0`
- `population_efficacy_claim`: `false`
- `supervisor_benefit`: `null`
- `missing_reason`: `Exp8674/8675 never executed`
- `operand_state`: `missing_field`
- `verdict_class`: `disqualified`

## RECOMMENDATION
KEEP
