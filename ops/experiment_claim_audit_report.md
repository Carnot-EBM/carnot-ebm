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

## experiment_8531_causal_brier_learning.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact records a blocked prerequisite check and makes no claim that continuous delayed Brier learning succeeded.

## WAS THAT CHECKED
No learning result was tested. The prerequisite was checked in `gates_evaluated`: actual 0 failed the requirement of 1, blocking the experiment before execution.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `online_adapter_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8533_v733_nine_input_native_qualification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Native execution passes the reported finite numerical parity checks, but overall qualification remains disqualified because required owned validation is unsatisfied.

## WHAT WOULD REFUTE IT
A native residual exceeding the stated per-row tolerance, a typed decision disagreement after the threshold guard, or a failed complete-request comparison would contradict the reported finite success. Evidence that every required validation passed would contradict the disqualification.

## WAS THAT CHECKED
Yes. Independent dense SciPy and actual Rust computations permit numerical disagreement; residual errors, probability errors, threshold neighbors and paired request responses were checked. Reported errors are small and decision mismatches are zero.

The failed owned-validation gate propagates into zero readiness and a disqualified verdict. Oracle status and exposed development data do not invalidate implementation parity: no added verifier value or generalization is claimed. Python/SciPy is the relevant comparator, and agreement is the stated objective.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_v733_nine_input_native_qualification`
- `finite_parity`: `true`; `complete_request_consumer`: `true`; `owned_validation`: `false`
- `max_residual_error`: `5.551115123125783e-17`
- `max_probability_error`: `2.220446049250313e-16`
- `typed_decision_mismatch_count`: `0`
- `required_checks_passed`: `false`; `nine_input_native_ready_score`: `0`
- `numerical_scope`: `Finite binary64 panel only; no universal error certificate, speed or Rust/Python NFR-01 claim.`

## RECOMMENDATION
KEEP

## experiment_8534_nine_input_inference_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative inference-cost claim to falsify; the title states an intended measurement, and the artifact records a blocked attempt.

## WAS THAT CHECKED
No inference-cost comparison is reported. Only the prerequisite gate was checked: native readiness was 0 against a required 1, blocking execution before measurement.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `nine_input_native_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8535_bounded_qwen_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental claim to falsify; this artifact records a prerequisite failure before the experiment proceeded.

## WAS THAT CHECKED
No experimental comparison was checked. The prerequisite gates were checked: runtime readiness was 0 against a required 1, while comparator-input readiness passed.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `runtime_repair_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8536_v733_arc_capability_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to falsify. The limited engineering qualification would fail if masking selected an explicitly unavailable arm, receipt collection changed execution, or an owned validation check failed.

## WAS THAT CHECKED
Yes, within the engineering scope shown: the masked disabled-runtime fixture blocks reinduction, displayed receipt-on/off parity hashes match, and required checks report passing. Some rows are elided.

The oracle checks fixture conformance; no verifier-added-value claim follows. Generalization and solve credit are zero. The parity ties concern receipt instrumentation, not a claimed performance advantage. Fixture rows are explicitly marked as development evidence.

## EVIDENCE

- `scientific_benefit_claimed`: `false`
- `Require complete owned validation, original direct evidence, exact frozen panel and receipt parity; fixtures grant selection readiness only.`
- `capability_reason`: `runtime_explicitly_disabled`
- `baseline_eligible`: `true`; `capability_eligible`: `false`; `eligible`: `false`
- `action_parity_rows`; `passed`: `true`
- `required_checks_passed`: `true`
- `fixture_is_live_evidence`: `false`
- `solve_provenance`: `development_proxy`
- `independent_generalization_score`: `0`
- `headline_solve_credit`: `0`

## RECOMMENDATION
KEEP

## experiment_8537_v733_arc_capability_panel.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This panel is disqualified and does not demonstrate useful efficiency from capability masking.

## WHAT WOULD REFUTE IT
Rows satisfying the declared efficiency conditions—fewer actual disabled attempts, nondecreasing levels in every pair, and no complete wall-cost increase—together with passing qualification, owned-validation, and arm-support checks would contradict this outcome.

## WAS THAT CHECKED
Yes. All eight paired comparisons show one fewer actual disabled attempt and tied peak levels, but reported runtime increases on average across games. Thus the necessary cost condition fails. Arm-support cells also report zero resolved effects, and owned validation fails its gate. These failures are reflected in the disqualified verdict. This is a supported negative finding limited to this panel.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_v733_arc_capability_panel`
- `useful_efficiency`: `false`
- `capability_efficiency_score`: `0`
- `descriptive_uncertainty` → `actual_disabled_attempts` → `game_mean`: `-1.0`
- `descriptive_uncertainty` → `peak_level` → `game_mean`: `0.0`
- `descriptive_uncertainty` → `runtime_s` → `game_mean`: `0.08438168237499999`
- `effect_support_by_arm` → `resolved`: `0`
- `owned_validation`: `false`
- `collection_qualified`: `false`

## RECOMMENDATION
KEEP

## experiment_8538_v733_nine_input_board_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. An authenticated, installed operation supporting nine inputs with qualified cost evidence would contradict the recorded readiness blocker.

## WAS THAT CHECKED
Yes, as an engineering readiness check: the gate summary records cost readiness as null and deployment readiness as false; the displayed operation rows report no compatible installed operation. This supports a blocked accounting record, without claiming acceleration, verifier value, or generalization.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v733_nine_input_board_boundary`
- `verdict_class`: `blocked`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `exposure_scope`: `historical_exposed_evidence_engineering_accounting_only`
- `missing_reason`: `no_qualified_cost_and_no_compatible_installed_kernel`
- `methodology_note`: `Engineering accounting only; no new board operation or independent scientific observation.`

## RECOMMENDATION
KEEP

## experiment_8539_v733_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no new comparative scientific headline to falsify. An unaccounted slot or a disqualified producer credited as independent benefit would contradict the administrative reconciliation.

## WAS THAT CHECKED
Yes, at the receipt level: acceptance gates report slot accounting, failed producer checks remain visible, and independent evidence receives zero credit. The blocked verdict does not promote administrative completion, oracle checks, or exposed historical results into demonstrated scientific value.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v733_capstone`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `administrative_complete`: `true`
- `all_slots_accounted`: `true`
- `scientific_claims_independently_generalize`: `false`
- `independent_count`: `0`
- `generalized_learning_benefit_score`: `0`
- `verdict_class`: `disqualified`
- `required_checks_passed`: `false`

## RECOMMENDATION
KEEP
