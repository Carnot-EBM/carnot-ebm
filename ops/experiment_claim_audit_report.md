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

## experiment_8318_v718_contract_replay.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is an administrative contract replay recording disqualification and unmeasured hypotheses, without a comparative scientific claim.

## WAS THAT CHECKED
No scientific benefit claim was tested: H1 and H2 remain unmeasured. Administrative checks had a real chance to fail and did, including a reduction mismatch and failed upstream qualification. Oracle agreement and exposed development receive no generalization credit.

## EVIDENCE
- `H1`, `H2`: `status` = `blocked_unmeasured`; `statistics` = `null`.
- `generalized_learning_benefit_score` = `0`.
- `exposure_scope` = `exposed_cached_development`.
- `first_reduction_mismatch`: `field` = `gate_check_summary[15].hash`.
- `qualified_current_evidence`: `required_checks_passed` = `false`; `verdict_class` = `disqualified`.
- `Bind this conclusion to byte-bound primitives; execution readiness never upgrades failed or unmeasured science.`
- `Constructed oracle agreement and exposed development give no independent generalization credit.`

## RECOMMENDATION
KEEP

## experiment_8319_local_evidence_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt, with no experimental success or comparative-value claim.

## WAS THAT CHECKED
Yes, the prerequisite gate was checked in gates_evaluated and failed: history_reader_ready_score was 0 against a required 1. The artifact reports that failure without claiming qualification succeeded.

## EVIDENCE
`blocked_gate_check_v1`; `blocked`; `blocked_gate_check_failed`; `failed_field`: `history_reader_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `passed`: `false`; `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8320_sentence_spline_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the title describes intended work; the artifact asserts no successful fit or advantage over static controls.

## WAS THAT CHECKED
No fitting or comparative outcome was checked. The artifact records a prerequisite gate failure: readiness was expected to equal 1 but was observed as 0, blocking execution.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `cached_support_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8326_runtime_reader_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific claim is asserted. This artifact records a blocked prerequisite check, not successful reader qualification or comparative value.

## WAS THAT CHECKED
No qualification experiment was performed. In gates_evaluated, the readiness prerequisite was checked and failed: expected 1, observed 0.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `history_reader_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8328_v718_arc_supervisor_frontier.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a disqualified aggregation receipt making no claim of supervisor benefit, arm superiority, or generalization.

## WAS THAT CHECKED
No comparative refutation was tested here: current execution counts are zero and outcome and arm rows are empty. The artifact reports disqualification and makes no positive scientific claim.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_supervisor_frontier`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `required_checks_passed`: `false`
- `current_game_execution_count`: `0`; `current_model_invocation_count`: `0`
- `rows`: `[]`; `per_game_arm_rows`: `[]`
- `selection_recommendations`: `[]`; `solve_claims`: `[]`
- `proposed_arm_change`: `null`; `proposed_generalization_change`: `null`

## RECOMMENDATION
KEEP

## experiment_8329_v718_kv260_workload_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to falsify. A completed eligible measurement or authenticated current device execution would contradict the receipt’s reported absence of those observations.

## WAS THAT CHECKED
No comparative test was completed: all six source-arm rows are censored, ineligible, and incomplete, with no timing results. This is a blocked receipt, not a measured null. Neither oracle checks nor exposed inputs are presented as establishing added value or generalization.

## EVIDENCE

- `verdict_class`: `blocked`
- `accelerator_benefit`: `unproved_no_compatible_operation`
- `completed_count`: `0`; `censored_count`: `6`
- `eligible`: `false`; `completed`: `false`; `evidence_status`: `unavailable_external_operand`
- `timing_rows`: `[]`; `current_device_execution_count`: `0`
- `independent_generalization_score`: `0`; `generalized_learning_benefit_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8330_v718_gatemate_change_ledger.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. A qualifying authenticated post-frontier physical-change receipt would contradict the ledger’s missing-evidence status; a current device-execution receipt would contradict its unexecuted status.

## WAS THAT CHECKED
Yes at the documentary level, through the change ledger and physical-change receipt audit. No current hardware reachability check occurred. The artifact reports a blocked obligation and makes no claim of recovery, generalization, or measured benefit.

## EVIDENCE

- `claim_scope`: `Evidence ledger; future physical preflight remains unexecuted`
- `honest_verdict`: `complete_blocked_gatemate_physical_change`
- `verdict_class`: `blocked`
- `physical_change_receipt_rows`: `[]`
- `current_reachability`: `not_probed`
- `current_device_execution_count`: `0`
- `scientific_benefit_score`: `0`
- `execution_ready_score`: `Readiness certifies owned evidence checks and grants no scientific benefit.`

## RECOMMENDATION
KEEP

## experiment_8331_v718_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact asserts no comparative benefit or generalization result. Its unmeasured comparisons establish neither improvement nor an empirical null.

## WAS THAT CHECKED
No comparative refutation was tested in the shown data. H1 and H2 remain unmeasured, and ARC contains no completed outcome rows or recommendations. The artifact reports disqualification.

## EVIDENCE

- `H1` and `H2`: `status` = `blocked_unmeasured`; `statistics` = `null`; `support` = `null`.
- `acceptance_gates`: `independent_science` = `false`; `owned_validation` = `false`.
- `arc_support`: `completed_count` = `0`; `rows` = `[]`; `selection_recommendations` = `[]`.
- `arc_support`: `honest_verdict` = `complete_disqualified_supervisor_frontier`.

## RECOMMENDATION
KEEP
