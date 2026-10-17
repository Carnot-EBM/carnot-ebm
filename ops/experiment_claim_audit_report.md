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

## experiment_8605_causal_brier_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-execution receipt, with no assertion that training ran or produced a benefit.

## WAS THAT CHECKED
Yes, prerequisite readiness was checked: both gates required 1, observed 0, and failed. No training-performance claim was tested or asserted.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `gate_check_summary`: `gate-unsat(final): 2 of 2 gate(s) failed; first failure: exp8603-deadline-edge-qualification.learning_execution_ready_score (actual=0 == expected=1)`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8608_v739_lean_resident_prototype.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit is asserted. A lean–baseline response mismatch, failed fixture or owned validation gate, or training/sklearn imports remaining in a lean process would contradict the bounded prototype receipt.

## WAS THAT CHECKED
Yes, within the stated fixture scope. All eight response rows pass and corresponding baseline/lean outputs match exactly. Both lean processes report training/sklearn imports false, and required checks pass. Those ties support fixture parity; no superiority, speed, or generalization claim is made.

## EVIDENCE

- `honest_verdict`: `complete_null_v739_lean_resident_prototype`
- `accepted_benchmark_claim`: `false`
- `Always false: import diagnostics and process durations cannot establish speed.`
- `One means both lean arms executed bounded fixtures and owned validation passed; Exp8609/8610 remain future dependencies.`
- `required_checks_passed`: `true`
- `independent_count`: `0`

## RECOMMENDATION
KEEP

## experiment_8609_v739_lean_resident_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit is asserted in the supplied excerpt, so there is no headline advantage to falsify.

## WAS THAT CHECKED
Not applicable. The artifact records qualification and timing receipts, explicitly skips benchmarking, and withholds speed inference.

## EVIDENCE

- `accepted_benchmark_claim`: `false`
- `benchmark_skipped`: `true`
- `speed_inferred`: `false`
- `timing_benefit_inferred`: `false`

## RECOMMENDATION
KEEP

## experiment_8610_lean_resident_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative outcome is asserted. A passing prerequisite gate would contradict the reported gate failure.

## WAS THAT CHECKED
Yes, the prerequisite gate was checked: expected 1, observed 0. The artifact records a blocked run, with no comparative outcome to assess.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `lean_measurement_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8611_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked execution receipt, with no experimental success or comparative value claim.

## WAS THAT CHECKED
No experimental claim was tested. The readiness checks in `gates_evaluated` show the runtime repair gate failing and the canary input gate passing, explaining the block.

## EVIDENCE
`schema`: `blocked_gate_check_v1`  
`status`: `blocked`  
`honest_verdict`: `blocked_gate_check_failed`  
`failed_field`: `runtime_repair_ready_score`  
`blocked_reason`: `actual=0 == expected=1`  
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8612_arc_outcome_observer.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt, with no claim about experimental success or added value.

## WAS THAT CHECKED
No outcome claim was tested. The prerequisite was checked in gates_evaluated: methods_ready_score was 0 against a required 1, blocking execution.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8613_v739_hardware_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a hardware qualification and accounting receipt, with no comparative claim of acceleration, verifier value, or generalization.

## WAS THAT CHECKED
Not applicable to comparative efficacy. The displayed gates record blocked readiness; conditional ceilings retain unknown costs and report no measured acceleration. These support the receipt’s limited scope.

## EVIDENCE
- `exposure_scope`: `exposed_engineering_accounting_only`
- `Each unit is an operation-to-board mapping, not an independent question or device measurement.`
- `One requires authenticated mapping, qualified cost, current authority and passing owned checks. It makes no acceleration claim.`
- `hardware_boundary_ready_score`: `0`
- `measured_acceleration`: `null`
- `compatible_physical_execution`: `false`

## RECOMMENDATION
KEEP

## experiment_8614_v739_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a blocked aggregation receipt, not a new comparative result. An ineligible diagnostic credited as qualified learning or complete-session cost would contradict its stated accounting.

## WAS THAT CHECKED
Yes, in the displayed acceptance gates and deterministic samples: learning remains unqualified, complete lean-session cost remains unavailable, and the verdict remains blocked. The native-speed comparison is explicitly historical. Oracle-based checks are not presented as evidence of verifier added value.

## EVIDENCE
- `honest_verdict`: `complete_blocked_v739_capstone`
- `administrative_complete`: `true`
- `successful_studies_not_implied`: `true`
- `qualified_learning`: `false`
- `qualified_lean_cost`: `false`
- `independent_generalization_score`: `0`
- `missing_reason`: `Exp8610 pre-gated; no complete lean-session measurement`

## RECOMMENDATION
KEEP
