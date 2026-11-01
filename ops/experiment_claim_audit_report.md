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

## experiment_8681_v745_contract_methods.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8682_frozen_learning_core.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a blocked-gate receipt, not an experimental result. All evaluated gates passing would contradict its blocking diagnosis.

## WAS THAT CHECKED
Yes. The methods readiness gate records actual 0 against expected 1 and fails; the learning operands gate passes. The receipt reports the resulting block without claiming successful qualification.

## EVIDENCE
`blocked_gate_check_v1`, `blocked`, `blocked_gate_check_failed`, `gate-unsat(final): 1 of 2 gate(s) failed; first failure: exp8681-contract-methods.methods_ready_score (actual=0 == expected=1)`, `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8684_relational_evidence_controls.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-run receipt, with no comparative result or claim of added value.

## WAS THAT CHECKED
Only prerequisite gates were checked. Methods readiness was 0 against required 1, blocking the experiment before evaluation.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `methods_ready_score`; `blocked_reason`: `actual=0 == expected=1`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8688_installed_batch_prototype.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a prerequisite failure and makes no claim that batch qualification succeeded or that the evaluator adds value.

## WAS THAT CHECKED
No experimental claim was tested. Prerequisite gates were checked; methods readiness failed with an observed 0 against a required 1, blocking evaluation.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `methods_ready_score`; `blocked_reason`: `actual=0 == expected=1`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8691_v745_arc_censoring_decision.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8692_v745_spline_precision_feasibility.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The 32-bit format with 24 fractional bits passes CPU numerical-feasibility checks against float64 on the constructed controls.

## WHAT WOULD REFUTE IT
A primary-format row failing numerical acceptance, showing nonfinite arithmetic or overflow, or disagreeing with float64 outside fallback would refute the claim.

## WAS THAT CHECKED
Yes, at the reported-control level: numerical errors, arithmetic failures and acceptance outcomes are recorded, and the primary-format gate passes. The 16-bit failure shows that numerical acceptance could fail; it does not contradict the specified primary format.

Fallback decisions reuse float64, so their parity is guaranteed and supplies no independent evidence. The numerical-error measurements remain falsifiable. Float64 is an appropriate fidelity baseline: agreement supports this feasibility claim without establishing verifier advantage. No generalization benefit is claimed. Most detailed rows and the acceptance tolerances are not shown.

## EVIDENCE
- `primary_format`: `32`, `24`; `primary_32_24`: `true`.
- `bits`: `32`; `fractional`: `24`; `passed`: `true`; `maximum_probability_error`: `1.036318339919795e-06`; `overflow`: `0`.
- `bits`: `16`; `passed`: `false`; `decay_underflow`: `1536`.
- `fallback`: `original_float64`; `decision_parity`: `true`.
- `verifier_is_oracle`: `true`; `independent_generalization_score`: `0`.
- `methodology_note`: `CPU fixed-point numerical controls; no causal stream, retention labels, FPGA synthesis, deployment or device performance measured.`

## RECOMMENDATION
KEEP

## experiment_8693_v745_hardware_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted to falsify. This is an evidence-aggregation and readiness report; its blocked status does not assert that hardware acceleration is impossible.

## WAS THAT CHECKED
Not applicable to comparative value. Readiness checks record missing cost inputs and failed physical-execution gates, and the verdict remains blocked.

## EVIDENCE
- `honest_verdict`: `complete_blocked_v745_hardware_operation_boundary`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `compatible_physical_execution`: `false`
- `complete_new_cost_boundaries`: `false`
- `physical_hardware_ready_score`: `0`
- `achieved_speedup`: `null`
- `target_met`: `false`

## RECOMMENDATION
KEEP

## experiment_8694_v745_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this aggregation asserts no comparative performance advantage. Missing qualification is not an empirical finding that the method fails.

## WAS THAT CHECKED
No comparative refutation test is established here. The displayed gate rows record failed qualification and missing fields; quarantined producers remain ineligible, and censored ARC observations receive no efficacy estimate or solve credit. These limitations are reflected in the reported outcomes.

## EVIDENCE
- `eligible`: `false`; `producer_qualified`: `false`; `scientific_result_measured`: `false`
- `operand_state`: `missing_field`; `observed`: `null`; `passed`: `false`
- `complete_horizon_count`: `0`; `efficacy_estimate`: `null`
- `headline_solve_credit`: `0`; `hidden_generalization_established`: `false`
- `exposure_scope`: `Exposed cached observations and constructed integer controls; no unexposed natural generalization`

## RECOMMENDATION
KEEP
