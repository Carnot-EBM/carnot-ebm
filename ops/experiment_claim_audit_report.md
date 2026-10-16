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

## experiment_8569_causal_brier_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked run and makes no claim of successful training or comparative value.

## WAS THAT CHECKED
No experimental claim was tested. Both prerequisite gates were checked and failed in `gates_evaluated`, blocking execution at `conductor_pre_gate`.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `gate_check_summary`: `gate-unsat(final): 2 of 2 gate(s) failed; first failure: exp8565-contract-methods.methods_ready_score (actual=0 == expected=1)`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8571_v736_resident_inference_prototype.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a prototype readiness receipt, without a comparative benefit claim. Its narrower readiness assertion could fail through a typed-decision mismatch, acceptance of an intentionally invalid request, a failed restart or shutdown, or a required validation failure.

## WAS THAT CHECKED
Yes, within the recorded protocol scope: six valid evaluations and two intentional rejections pass; Python and native outputs match on the repeated inputs; lifecycle receipts record restarts and clean shutdowns.

Oracle-defined parity supports this execution check without establishing verifier value. Exposed records are not presented as independent generalization. The Python comparator ties, consistent with the parity objective. The two error responses are expected rejection successes, not invalid evaluations counted as accuracy wins. Timing evidence is explicitly imported historical evidence.

## EVIDENCE

- `honest_verdict`: `complete_null_v736_resident_inference_prototype`
- `prototype_ready_score`: `1`
- `harmful_intervention_criterion`: `Any typed decision change from shipped complete evaluator fails this prototype; this is numerical parity, not correctness labeling.`
- `completed_count`: `Eight protocol attempts include six valid evaluations and two intentional rejections; repeated inputs add no independence.`
- `kind`: `malformed_json`, `changed_config_hash`; `status`: `error`; `passed`: `true`
- `clean_shutdown`: `true`; `required_checks_passed`: `true`
- `comparative_timing_trials`: `0`; `new_timing_comparison`: `false`
- `independent_generalization_score`: `0`; `generalized_learning_benefit_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8572_v736_resident_inference_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit is asserted. The limited operational qualification would fail if a resident/native path changed a reference decision, a failure/recovery control failed, or clock intervals overlapped.

## WAS THAT CHECKED
Yes, for operational qualification: displayed four-path parity and numerical-control rows report no decision mismatches; displayed failure controls and clock accounting pass. Ties support this equivalence check. Oracle-defined correctness and exposed inputs are not presented as verifier value or generalization. The timing null is explicitly historical.

## EVIDENCE

- `honest_verdict`: `complete_null_v736_resident_inference_qualification`
- `methodology_note`: `Qualification only. No timing benefit is inferred from fixtures or launch-through-exit clock accounting.`
- `new_timing_comparison`: `false`
- `decision_mismatch`: `false`
- `no_overlapping_intervals`: `true`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8573_resident_complete_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked experiment, with no comparative cost conclusion to falsify.

## WAS THAT CHECKED
No cost comparison is reported. Prerequisite gates were checked, and methods readiness failed, blocking execution.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `methods_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8574_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No experimental benefit is asserted. For the blocking report itself, gate rows showing all prerequisites passed would contradict the stated reason for blocking.

## WAS THAT CHECKED
Yes. The gates_evaluated entries record two failed prerequisites and one passed prerequisite, consistent with the blocked outcome. Canary effectiveness was not evaluated.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `gate_check_summary`: `gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp8565-contract-methods.methods_ready_score (actual=0 == expected=1)`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8575_arc_publication_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A passing prerequisite gate would contradict the recorded reason for blocking. There is no claim of successful qualification or comparative value to refute.

## WAS THAT CHECKED
Yes. The prerequisite was checked in gates_evaluated: methods_ready_score was 0 against a required value of 1. The artifact records the failure and stops at the prerequisite gate.

## EVIDENCE
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `methods_ready_score`
- `expected`: `1`; `actual`: `0`; `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8577_v736_resident_hardware_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit is asserted. A hardware acceleration claim would be refuted by a qualified resident CPU baseline tying or beating compatible hardware on complete session cost with matching correctness.

## WAS THAT CHECKED
No comparative benefit test is reported. This is an evidence-boundary artifact: resident costs remain unqualified, physical-execution gates fail, and operation-mapping readiness is explicitly distinguished from physical readiness. Those limitations are acknowledged rather than promoted into a positive result.

## EVIDENCE
- `hardware_boundary_ready_score`: `1`
- `One means qualified operation mapping; resident cost and physical readiness are separate.`
- `physical_execution`: `false`
- `resident_cost`: `false`
- `measured_acceleration`: `null`
- `Exp8573_complete_rows_and_parity_unqualified`
- `future_dependency_not_observation`
- `generalized_learning_benefit_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8578_v736_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. A blocked or ineligible row credited as qualified learning, ARC, or resident-speed benefit would contradict the artifact’s stated qualification boundary.

## WAS THAT CHECKED
Yes, at the qualification boundary: the corresponding acceptance gates are false, generalization credit is zero, and resident complete cost is explicitly unmeasured. Blocked slots count toward administrative completion. This provides no comparative benefit test, and the elided rows limit a complete accounting audit.

## EVIDENCE
- `administrative_complete`: `true`
- `qualified_learning_benefit`: `false`
- `qualified_arc_panel`: `false`
- `qualified_resident_cost`: `false`
- `generalization_credit`: `0`
- `no_qualified_learning_or_arc_benefit_no_resident_speed_measurement`
- `Exp8573 pre-gated; qualification clocks are not a serving benchmark`
- `Exposed original questions and public ARC diagnostic; no independent labels or learning credit`

## RECOMMENDATION
KEEP
