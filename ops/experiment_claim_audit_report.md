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
| NO_CLAIM | 6 |

## experiment_8778_delayed_memory_prototype.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked prerequisite-check receipt, with no experimental success or comparative claim to falsify.

## WAS THAT CHECKED
Only prerequisite gates were checked. The memory-protocol readiness gate failed with an observed score of 0 against the required 1; no experimental result is asserted.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `memory_protocol_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8779_v752_evidence_set_evaluation.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
On the constructed replication and inspected six-edge evaluation sets, evidence_set reduces three-class error against both comparators without added false acceptances, valid-alternative rejections, or loss of correct-control invariance.

## WHAT WOULD REFUTE IT
The cheapest serious comparator shown, joint_symbolic, tying or beating evidence_set would refute its claimed advantage. Added false acceptances, valid-alternative rejections, or broken correct-control invariance would also defeat the promotion criterion.

## WAS THAT CHECKED
Yes, within this constructed benchmark. Both strata report comparator errors, paired improvement intervals, rejection counts, and invariance checks. Both comparator improvement bounds are positive; evidence_set has zero reported error and decision harms.

Correctness uses independently enumerated Boolean-completion labels sealed before input-only measurement, giving disagreement a real opportunity to register. The verifier is not the oracle. Prior inspection is disclosed, and blinded or natural-data generalization is expressly unclaimed. The cited metric keys exist, and no failed, censored, or excluded rows are reported.

## EVIDENCE

- `methodology_note`: `Labels are independently enumerated Boolean completions, sealed before input-only measurement.`
- `verifier_is_oracle`: `false`; `input_only_child_passed`: `true`.
- In `new_family_metrics`, `macro_error` is `0.0` for `evidence_set`, `0.1111111111111111` for `joint_symbolic`, and `0.3361111111111111` for `local_record`.
- `lower_simultaneous_97_5`: `0.3333333333333333` and `0.1111111111111111`.
- For `evidence_set`: `false_acceptances`: `0`; `valid_alternative_rejections`: `0`. `correct_control_invariance`: `true`.
- `exposure_scope`: `V751 is exposed replication. Six-edge graphs were frozen by Exp8777 and are unseen by the unchanged V751 rule, but qualification and this implementation inspected them. No blinded or natural-data generalization claim.`
- `failed_count`: `0`; `censored_count`: `0`; `excluded_count`: `0`.

## RECOMMENDATION
KEEP

## experiment_8782_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No extraction-fidelity or comparative claim is asserted. The blocking report would be contradicted if both evaluated gates passed.

## WAS THAT CHECKED
Yes. The evaluated gates record one failure and one success, consistent with the blocked status. This is a gate-check receipt, not an extraction-fidelity result.

## EVIDENCE
`schema`: `blocked_gate_check_v1`  
`status`: `blocked`  
`honest_verdict`: `blocked_gate_check_failed`  
`gate_check_summary`: `gate-unsat(final): 1 of 2 gate(s) failed; first failure: exp8776-branch-methods.runtime_repair_ready_score (actual=0 == expected=1)`  
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8783_v752_durable_producer_identity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The operational qualification would fail if an alias were authenticated as the current producer, a measured state differed from its expected state, or restart recovery failed. The artifact makes no comparative claim about verifier value, learning, or acceleration.

## WAS THAT CHECKED
Yes, for operational qualification: consumer controls reject both aliases, displayed parity rows match expected hashes, and restart rows pass. Alias readiness checks succeed, but identity checks reject them; those false authentication flags are expected negative controls. These checks could expose implementation failures without establishing comparative benefit.

## EVIDENCE
- `exposure_scope`: `constructed_ordinary_trajectory_identity_qualification_only`
- `Bind exact current identity and actual sealed operands; qualification does not establish scientific or hardware benefit.`
- `historical_alias_admitted`: `false`
- `same_number_alias`, `historical_alias`, `descriptor_authenticated`: `false`
- `parity_rows`: `expected_state_hash`, `state_hash`, `passed`
- `restart_rows`: `process_lifetimes`, `restart_after`, `passed`
- `independent_generalization_score`: `0`
- `honest_verdict`: `complete_circular_positive_durable_producer_identity`

## RECOMMENDATION
KEEP

## experiment_8784_v752_durable_cost_components.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The run provides descriptive complete durable-cost measurements for one constructed workload, while component attribution remains unestablished.

## WHAT WOULD REFUTE IT
A failed restart, invalid state, or incomplete update window counted toward complete-cost readiness would contradict the measurement claim. A component decomposition meeting the frozen overhead and reconciliation gates across the required pairs would overturn the attribution null.

## WAS THAT CHECKED
Yes, at the reported scope. Windows record observed updates, completion and restart verification. Paired plain/instrumented runs provide observer-overhead and component-reconciliation measurements. Complete-cost readiness passes; component attribution fails, and the displayed component rows are unadmitted.

Python and JSON-native are serious comparators already present. The artifact makes no positive speedup, verifier-added-value or generalization claim. Elided rows limit independent auditing of the aggregate gates, but the displayed evidence does not contradict the scoped null.

## EVIDENCE

- `honest_verdict`: `complete_null_durable_cost_components`
- `complete_costs`: `true`; `component_attribution`: `false`
- `durable_cost_rows_ready_score`: `1`; `component_cost_ready_score`: `0`
- `admitted`: `false`
- `observed_updates`: `32`; `denominator_updates`: `32`; `restart_verified`: `true`; `passed`: `true`
- `maximum_observer_overhead`: `0.05`; `maximum_reconciliation_error`: `0.02`
- `arms`: `python`, `json_native`, `resident_native`
- `descriptive_only`: `true`; `independent_generalization_score`: `0`; `long_run_claim_allowed`: `false`

## RECOMMENDATION
KEEP

## experiment_8786_arc_new_outcome_decision.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite check and asserts no result about supervisor choices or comparative value.

## WAS THAT CHECKED
No substantive outcome claim was tested. The prerequisite was checked in `gates_evaluated`: new-outcome availability was 0 against required 1, the gate failed, and execution was blocked.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `arc_new_outcomes_available_score`
- `expected`: `1`
- `actual`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8787_v752_hardware_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. Qualified learning-cost and audit inputs, together with compatible physical execution and complete timing, would overturn the recorded prerequisite blockage.

## WAS THAT CHECKED
Admission checks: yes, in branch_admission_rows and gate_check_summary. Physical acceleration: no. Missing inputs remain censored, rejected diagnostic components are ineligible for bounds, and scientific promotion remains blocked. The artifact makes no positive verifier-value or generalization claim.

## EVIDENCE

- `honest_verdict`: `complete_blocked_hardware_operation_boundary`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `operand_state`: `missing_file`
- `eligible_for_bound`: `false`
- `physical_hardware_ready_score`: `0`
- `complete_100x_learning_ready_score`: `0`
- `reason`: `No admitted complete learning denominator with compatible fraction>0.99 and physical setup/transfer.`

## RECOMMENDATION
KEEP

## experiment_8788_v752_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is an administrative reconciliation artifact reporting blocked work, with no comparative scientific headline to falsify.

## WAS THAT CHECKED
No comparative hypothesis is tested here. The gate_check_summary records failed prerequisites and missing files, consistent with the blocked verdict. Those failures are not presented as scientific success.

## EVIDENCE
- `honest_verdict`: `complete_blocked_v752_capstone`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `methodology_note`: `Account for thirteen exact tasks without executing historical controllers. Preserve qualified constructed evidence and durable nulls. Missing learning, natural extraction, capacity and ARC outcomes remain blocked. No model, training, physical-device or independent generalization claim.`

## RECOMMENDATION
KEEP
