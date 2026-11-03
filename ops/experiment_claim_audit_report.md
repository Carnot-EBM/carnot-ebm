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
| NO_CLAIM | 3 |
| SKIPPED_ALREADY_FLAGGED | 3 |

## experiment_8739_learning_union.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt asserting no causal or comparative result.

## WAS THAT CHECKED
The prerequisites were checked: both gate rows report 0 against required 1 and fail. The causal qualification was blocked before execution, so its refutation was not tested.

## EVIDENCE
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `learning_protocol_ready_score`; `deadline_controls_ready_score`
- Both gate rows: `expected`: `1`, `actual`: `0`, `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8742_verifier_family_transfer.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No experimental outcome is asserted. The title describes an intended transfer test; it does not claim successful transfer.

## WAS THAT CHECKED
No. Transfer testing was blocked before execution. Only the prerequisite readiness gate was evaluated, returning 0 against a required 1.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `verifier_transfer_methods_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8743_v749_single_extraction_prototype.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8744_single_extraction_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No outcome claim is asserted. The title describes a measurement objective; the artifact records a blocked prerequisite.

## WAS THAT CHECKED
The prerequisite was checked and failed: expected 1, observed 0. No installed-cost results or comparative conclusions are reported.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `feature_reuse_ready_score`
- `expected`: `1`
- `actual`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8745_v749_bounded_durable_cost.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8746_v749_arc_outcome_frontier.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This authenticated frontier update found no new supervisor outcomes and supplies no new support for policy changes or solve credit.

## WHAT WOULD REFUTE IT
An authenticated event outside the retained frontier, or an eligible new supervisor outcome omitted from the reported empty delta.

## WAS THAT CHECKED
Yes, within the ledger-update scope: the stated method compares exact event identities, the displayed before/after frontiers match, and the reported delta is empty. This supports an inventory null, not a conclusion that supervisor interventions cannot help. No comparative efficacy or positive generalization claim requires a rival baseline here.

## EVIDENCE
- `honest_verdict`: `complete_null_no_new_supervisor_outcomes`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `new_event_rows`: `[]`
- `frontier_before` and `frontier_after` share `closed_invocation_id`: `1793566223757576517`
- `policy_decision`: `no_supported_policy_change`
- `usable_cross_game_support`: `0`
- `new_level_credit`: `0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8747_v749_hardware_cost_boundary.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8748_v749_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The capstone remains disqualified despite a limited batching benefit, without establishing same-batch 10× acceleration or independent generalization.

## WHAT WOULD REFUTE IT
The batching benefit would fail if its paired lower confidence bound were ≤1.05 or native B16 p95 exceeded 50 ms. The disqualification would be contradicted by qualified evidence satisfying the acceptance gates and passing owned validation.

## WAS THAT CHECKED
Yes. The deployment rows report a batching lower bound of 1.12185 and B16 p95 of 18.42019 ms, clearing both stated thresholds. The serious same-batch native/Python B16 comparison includes parity; the artifact accordingly retains a zero 10× score. Generalization is explicitly unmeasured. Failed producer qualifications and owned validation remain visible, and the capstone is disqualified rather than promoted to success.

## EVIDENCE
- `honest_verdict`: `complete_disqualified_v749_capstone`
- `activation_authorized`: `false`
- `owned_validation_passed`: `false`
- `producer_qualified`: `false`
- `batching_threshold`: `paired lower95 > 1.05 and native B16 p95 <= 50ms`
- `contrast`: `batching`; `lower95`: `1.1218473605571395`
- `native_B16_steady_p95_ms`: `18.42018975`
- `contrast`: `native_python_B16`; `lower95`: `0.998091298984346`; `upper95`: `1.0138702202694783`
- `same_batch_10x_score`: `0`
- `generalization_measured`: `false`
- `operand_state`: `missing_file`, `observed_zero`

## RECOMMENDATION
KEEP
