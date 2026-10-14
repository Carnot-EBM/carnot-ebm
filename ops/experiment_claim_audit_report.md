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

## experiment_8518_objective_benefit_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no empirical benefit claim to falsify. This artifact records a blocked prerequisite check, not a result about loss or retention.

## WAS THAT CHECKED
No benefit evaluation occurred. The prerequisite was checked in `gates_evaluated`: readiness was 0 against an expected 1, so the gate failed.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `objective_stream_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8519_v732_append_import_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to refute. The limited import-readiness assertion would fail if historical replay diverged, required current checks failed, or owned coverage were incomplete.

## WAS THAT CHECKED
Yes, at the engineering-qualification level: displayed replay and test receipts pass, required checks are reported successful, and owned coverage is complete. The artifact claims no verifier advantage or generalization benefit; the oracle flag alone does not make this receipt an overstated value claim.

## EVIDENCE
- `Bind current qualification to sealed evidence; import readiness is not speed or scientific benefit.`
- `methodology_note`: `Cold replay authenticates original crash/equivalence and complete-clock operands. Explicit import repair only; zero new benchmark measurements and zero model calls. Historical Exp8494 remains disqualified.`
- `required_checks_passed`: `true`
- `percent_covered`: `100.0`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`
- `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_8520_v732_append_service_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Append storage speeds up complete durable service versus snapshot storage on this cached, same-host fixture.

## WHAT WOULD REFUTE IT
Matched complete-service measurements showing append no faster than snapshot, or showing that its speed advantage sacrifices acknowledged work or durable-state parity.

## WAS THAT CHECKED
Yes. The artifact reports five measured pairs per cell, separate warmups, complete clocks including startup and final drain, and accounting and durability checks. Snapshot storage—including batching at eight—is a serious baseline; append beats it in both cells, with both reported speedup intervals above one.

The numerical verifier defines correctness, but does not determine elapsed time: append could have lost this comparison. No verifier-added-value or generalization benefit is asserted. The displayed timing fields exist, and no invalid-row flag is shown.

## EVIDENCE

- `pooled_speedup`: `2.162681701494454`
- `primary_speedup_ci95`: `2.0387707933429033`, `2.2915668815786066`
- `secondary_speedup_ci95`: `1.3913805124176435`, `2.012610341607594`
- `measured_pairs_per_cell`: `5`; `warmup_pairs_per_cell`: `1`
- `accounting_equality`: `true`; `durability_checks_passed`: `true`
- `service_boundary`: `unchanged cached numerical oracle; cold reconstruction, startup, features, predictions, delayed feedback, durable response and final drain`
- `verifier_is_oracle`: `true`; `generalized_learning_benefit_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8521_bounded_qwen_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact reports a blocked prerequisite check, not a finding about the method’s effectiveness.

## WAS THAT CHECKED
No effectiveness test occurred. Both prerequisite gates were checked and failed in `gates_evaluated`.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `blocked_at_layer`: `conductor_pre_gate`. Both gate entries record `expected`: `1`, `actual`: `0`, and `passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_8522_v732_arc_effect_receipt_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit claim is made. The narrower instrumentation-parity assertion would fail if recording receipts changed actions, policy state, RNG state, or LLM-call counts relative to the control.

## WAS THAT CHECKED
Yes, for the three displayed action-parity cases: shadow, ineligible, and already_active have matching control and observed hashes. Six additional cases are elided. These ties support the intended parity check; they are not presented as performance gains. The oracle flag does not establish a circular value claim because scientific benefit is explicitly unclaimed.

## EVIDENCE
- `scientific_benefit_claimed`: `false`
- `headline_solve_credit`: `0`
- `independent_generalization_score`: `0`
- `live_episodes_measured`: `0`
- `fixture_is_live_evidence`: `false`
- `Real wrapper execution with scripted observations qualifies instrumentation only.`
- `action_parity_rows`, `compared_fields`, `actions`, `policy_state`, `rng_sha256`, `llm_calls`, `control_sha256`, `observed_sha256`, `passed`: `true`

## RECOMMENDATION
KEEP

## experiment_8523_v732_arc_effect_panel.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Within this panel’s 512-action window, applied redirects showed no observed level-progress advantage over matched shadow runs.

## WHAT WOULD REFUTE IT
An applied run reaching a higher peak level than its matched shadow run would contradict the observed null.

## WAS THAT CHECKED
Yes. All eight matched game/seed comparisons report zero peak-level difference. The cheapest serious baseline—the same runtime with redirects off—is present and ties on progress. Action divergence is recorded separately and receives no benefit credit.

All episodes remain action-limit censored, so this supports a bounded observational null, not a conclusion that redirects can never help. Scientific benefit and generalization credit are withheld.

## EVIDENCE

- `honest_verdict`: `complete_null_v732_arc_effect_panel`
- `matched_initial_state`: `true`
- `peak_level_delta`: `0`
- `peak_level`: `0`
- `mode`: `applied`, `shadow`
- `censoring`: `action_limit`
- `resolved_redirects`: `0`
- `scientific_benefit_claimed`: `false`
- `arm_refinement_authorized`: `false`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8524_v732_board_cost_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative headline is asserted. A qualified, installed compatible operation with complete device/transfer clocks and validated full-service execution would contradict the recorded deployment boundary.

## WAS THAT CHECKED
Yes, as an evidence/readiness check in the acceptance gates and board operation rows. No new device execution is reported. The artifact records blocked deployment and conditional cost ceilings without asserting demonstrated acceleration or generalization.

## EVIDENCE
- `compatible_operations`: `false`
- `complete_transfer_device_clocks`: `false`
- `deployment_ready`: `false`
- `device_commands_executed`: `[]`
- `generalized_learning_benefit_score`: `0`
- `limit`: `Counterfactual zero-device-time upper bound; retain host storage, transfer and orchestration. No deployed speed or five-to-nine-input scaling.`

## RECOMMENDATION
KEEP

## experiment_8525_v732_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit claim is asserted in the supplied excerpt. A verifier-defined pass or publication-readiness flag alone would not establish added value.

## WAS THAT CHECKED
No new comparative test is shown. The excerpt preserves upstream results, explicitly separates publication readiness from branch benefit, assigns zero independent-generalization credit, and leaves the unqualified service result unqualified. The text ends mid-record, so this assessment covers only the supplied excerpt.

## EVIDENCE
- `publication_readiness_grants_branch_benefit`: `false`
- `decision`: `preserve_missing_authority_and_unmeasured_science; no_hypothesis_retirement`
- `evidence_scope`: `retrospective exposed public labels or engineering feasibility only`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`
- `original_verdict`: `complete_disqualified_append_service_cost`
- `qualified_producer`: `false`
- `qualified_retrospective_benefit`: `null`

## RECOMMENDATION
KEEP
