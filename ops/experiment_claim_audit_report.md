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

## experiment_8336_continuous_local_learning.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact asserts no learning, retention, or comparative result.

## WAS THAT CHECKED
No learning or retention evaluation is reported. The prerequisite gates were checked; local-kernel readiness failed with observed 0 against required 1.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `local_kernel_ready_score`; `blocked_reason`: `actual=0 == expected=1`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8337_bounded_feedback_capacity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental outcome claim to falsify. The title describes an intended test; the artifact records that a prerequisite failure blocked it.

## WAS THAT CHECKED
No capacity or delayed-learning hypothesis was checked. Only the prerequisite gate was evaluated: it required 1, observed 0, and failed.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `blocked_at_layer`: `conductor_pre_gate`; `artifact_field`: `local_kernel_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_8340_v719_runtime_reader_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A successful-qualification claim would be refuted by a mandatory owned check failing. That failure appears here, and the artifact reports disqualification.

## WAS THAT CHECKED
Yes: the gate summary records failed required checks, and the false-zero control remains unresolved. This is a qualification receipt making no comparative benefit or generalization claim.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_owned_checks`
- `required_checks_passed`: `false`
- `acceptance_gates`: `authenticated_change`: `false`, `cuda_context_ready`: `false`, `reader`: `false`
- False-zero disposition: `recomputed`: `false`, `resolved`: `false`
- `Retain exact invocation, byte custody and measured execution; no scientific benefit follows.`

## RECOMMENDATION
KEEP

## experiment_8341_changed_runtime_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-run receipt, with no comparative or performance claim to falsify.

## WAS THAT CHECKED
No experimental claim was tested. Prerequisites were checked in `gates_evaluated`: both failed, and the artifact records blocking before the canary.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `runtime_changed_score` and `cuda_context_ready_score` each have `actual`: `0`, `expected`: `1`, and `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8342_v719_arc_supervisor_frontier.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. An authenticated new outcome omitted from the empty frontier would contradict its accounting, but there is no supervisor-benefit claim to falsify.

## WAS THAT CHECKED
No comparative test is reported: current executions and comparison rows are zero or empty. Authentication checks are reported; the artifact records disqualification.

## EVIDENCE
- `honest_verdict`: `complete_disqualified_supervisor_frontier`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `required_checks_passed`: `false`
- `current_game_execution_count`: `0`
- `current_model_invocation_count`: `0`
- `new_outcome_count`: `0`
- `rows`: `[]`
- `per_game_arm_rows`: `[]`
- `selection_recommendations`: `[]`

## RECOMMENDATION
KEEP

## experiment_8343_v719_kv260_workload_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact reports disqualification and unproved benefit, without asserting superiority, generalization, or acceleration.

## WAS THAT CHECKED
No comparative benefit claim was established. The visible timing rows compare constructed CPU arms; service speedup remains unavailable. The validation receipts record a failed owned check, reflected in the disqualified verdict.

## EVIDENCE
- `honest_verdict`: `complete_disqualified_owned_checks`
- `verdict_class`: `disqualified`
- `accelerator_benefit`: `unproved_no_compatible_operation`
- `arithmetic_scientific_class`: `unqualified`
- `full_service_speedup`: `null`
- `current_device_execution_count`: `0`
- `required_checks_passed`: `false`
- `name`: `private_E2E018_consumers`; `passed`: `false`

## RECOMMENDATION
KEEP

## experiment_8344_v719_gatemate_change_ledger.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. An authenticated new physical-change receipt would contradict the reported receipt absence; recorded current device execution would contradict the unexecuted status.

## WAS THAT CHECKED
Yes, at the documentation level: physical-change evidence is marked absent, receipt rows are empty, and current device executions total zero. The obligation remains excluded and incomplete. Physical preflight was not performed, so hardware capability remains untested.

## EVIDENCE

- `claim_scope`: `Evidence ledger; future physical preflight remains unexecuted`
- `honest_verdict`: `complete_blocked_gatemate_history`
- `physical_change_receipt_rows`: `[]`
- `current_device_execution_count`: `0`
- `excluded`: `true`; `completed`: `false`
- `scientific_benefit_score`: `0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8345_v719_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative headline is asserted. A benefit claim would be refuted by valid sealed evaluation rows showing the method ties or loses to a serious comparator.

## WAS THAT CHECKED
No scientific comparison was completed here: H1/H2 comparison metrics, statistics, and support are null; evaluator access is zero. The artifact records blocked science and failed qualification gates. These are not measured null results.

## EVIDENCE
- `H1` and `H2`: `status` = `blocked_unmeasured`; `statistics` = `null`; `support` = `null`.
- `paired_cost_gain` = `null`.
- `evaluator_access_count` = `0`.
- `independent_science` = `false`; `owned_validation` = `false`.
- `continuation_decision` = `not_earned_unmeasured; require qualified sealed science operands`.

## RECOMMENDATION
KEEP
