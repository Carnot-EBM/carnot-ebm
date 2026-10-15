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

## experiment_8542_canonical_update_accounting.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No event-accounting qualification result is asserted. The gate-failure diagnostic would be contradicted by its evaluated gate satisfying the required equality.

## WAS THAT CHECKED
Yes, for the gate diagnostic: the sole gate required 1, observed 0, and failed. Event-accounting qualification was blocked before evaluation.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `historical_learning_validation_ready_score`
- `expected`: `1`; `actual`: `0`; `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8545_v734_nine_input_native_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative headline to falsify. A native/reference residual discrepancy exceeding the specified tolerance, or a decision mismatch after the threshold guard, would refute the narrower finite-parity result.

## WAS THAT CHECKED
Yes, for finite parity: the artifact reports 1,180 comparisons and shows passing request and threshold examples. Qualification remains disqualified because mandatory validation gates are false. Python/native ties support parity; no superiority is claimed. The oracle defines compatibility here, and the exposed development data is not presented as independent generalization.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_v734_nine_input_native_qualification`
- `finite_parity`: `true`; `finite_parity_rows`: `1180`
- `owned_validation`: `false`; `all_required_contracts`: `false`
- `decision_mismatch`: `false`; `probability_error`: `0.0`
- `independent_generalization_score`: `0`; `generalized_learning_benefit_score`: `0`
- `One only after native calls, finite controls, complete request consumer and all required validation pass; no speed or learning claim.`
- `Invocation-bound qualification evidence; no new independent science or speed estimate.`

## RECOMMENDATION
KEEP

## experiment_8546_complete_native_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to refute. The blocked-gate report would be contradicted if its gate rows showed that all prerequisites passed.

## WAS THAT CHECKED
Yes. The gate rows show native readiness was 0 against a required 1, with the gate marked false. This is a blocked-run receipt; no native-versus-Python performance result is asserted.

## EVIDENCE
`blocked_gate_check_v1`, `blocked_gate_check_failed`, `nine_input_native_ready_score`, `actual=0 == expected=1`, `passed`, `false`, `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8547_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite check and makes no claim about the canary’s performance or comparative value.

## WAS THAT CHECKED
The prerequisite was checked and failed: runtime readiness was 0 against required 1. The canary therefore did not run; no performance claim was tested.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `runtime_repair_ready_score`
- `blocked_reason`: `actual=0 == expected=1`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8548_v734_arc_panel_replay_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to falsify. Infrastructure qualification would fail if terminal replay failed or required coverage were incomplete.

## WAS THAT CHECKED
Yes. Both infrastructure acceptance gates are false, readiness is zero, and the verdict records disqualification. The artifact claims no scientific benefit; the oracle verifier and stored replays therefore do not create a circular value or generalization claim.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_v734_arc_panel_replay_qualification`
- `acceptance_gates`: `terminal_replay`: `false`; `owned_coverage`: `false`
- `no_scientific_benefit_claim`: `true`
- `arc_panel_execution_ready_score`: `0`
- `headline_solve_credit`: `0`
- `independent_count`: `0`
- `generalized_learning_benefit_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8549_arc_capability_panel.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked prerequisite-check receipt, not an assertion of measured capability or comparative value.

## WAS THAT CHECKED
Only prerequisite gates were checked. Execution readiness required 1 and returned 0, blocking the experiment before capability measurement. There is no capability result to assess for refutation.

## EVIDENCE
- `blocked_gate_check_v1`
- `blocked_gate_check_failed`
- `arc_panel_execution_ready_score`
- `actual=0 == expected=1`
- `gate-unsat(final): 1 of 2 gate(s) failed; first failure: exp8548-arc-panel-replay-qualification.arc_panel_execution_ready_score (actual=0 == expected=1)`
- `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8550_v734_hardware_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative headline to falsify. An authenticated, compatible nine-input/109-coefficient device execution with parity and complete costs would contradict the recorded qualification blocker.

## WAS THAT CHECKED
No comparative test is shown. The shown operation rows record unsupported execution and missing costs. The readiness score explicitly covers scope accounting, so it does not assert hardware performance or added value.

## EVIDENCE

- `exposure_scope`: `historical_exposed_evidence_engineering_accounting_only`
- `hardware_boundary_ready_score`: `1`
- `One qualifies scope accounting only.`
- `missing_reason`: `no_qualified_cost_and_no_compatible_installed_kernel`
- `deployment_ready`: `false`
- `measured_acceleration`: `null`
- `generalized_learning_benefit_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8551_v734_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. Administrative completion would be contradicted by an unaccounted scheduling slot; it does not establish scientific benefit.

## WAS THAT CHECKED
Yes for administrative accounting: the acceptance gates report all slots accounted for, while individual rows record failed prerequisites and skipped execution. No comparative benefit is asserted, so the oracle verifier does not create a circular value claim here.

## EVIDENCE
- `administrative_complete`: `true`
- `all_slots_accounted`: `true`
- `scientific_claims_independently_generalize`: `false`
- `disposition`: `pre_gate_receipt`
- `disposition`: `cascade_skip`
- `producer_executed`: `false`
- `Stored ARC replay checks execution mechanics and grants no hidden-game capability benefit.`

## RECOMMENDATION
KEEP
