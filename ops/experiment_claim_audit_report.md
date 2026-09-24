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
| CLAIM_SUPPORTED | 4 |
| NO_CLAIM | 4 |

## experiment_7588_v663_evidence_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact makes no empirical, comparative, or capability claim, serving strictly as a receipt recording a blocked run that halted prior to data collection or scoring.

## WAS THAT CHECKED
No; no claim was checked because execution halted at the `"selected_role_roster"` acceptance gate before any models were loaded or evaluations were performed.

## EVIDENCE
`"positive_claim": false`
`"verdict_class": "blocked"`
`"honest_verdict": "complete_blocked_selected_role_roster"`
`"passed": false`
`"observed": "source_group_incomplete"`
`"rows": []`
`"model_invoked": false`
`"no_model_load": true`
`"inference_substrate_class": "blocked_no_run"`

## RECOMMENDATION
KEEP

## experiment_7589_v663_arc_output_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The ARC output boundary and history observer meet readiness and validation requirements without claiming any solve or policy benefit.

## WHAT WOULD REFUTE IT
The claim would be refuted by:
- Any divergence in policy actions between observer-on and observer-off executions (`equal` false in parity rows).
- A failure of the protected path guard to block outputs targeted inside `results/` (`output_exists` true or exit code other than 2).
- Any failure or timeout across the five applicable E2E suites or the foreign working directory smoke test.
- Any failed check in the causal history observer fixtures (such as failure to clear on reset, key conflict detection failure, or memory unbounded beyond capacity).
- Any claim of solve or policy benefit, or authorization of default promotion, without executing a live comparison.

## WAS THAT CHECKED
Yes:
- Policy parity was checked across 10 actions in `policy_parity_rows` and `policy_parity_summary`.
- Output boundary protection was verified in `protected_path_reproduction` (`exit_code` 2, `output_exists` false).
- Regression suite parity and live smoke execution were checked across 6 commands in `e2e_receipts`.
- History causality, bounded memory, and conflict properties were verified across 6 unit fixtures in `observer_fixture_evidence` and `rows`.
- Benefit was explicitly checked and marked unperformed in `acceptance_gate_results.benefit` (`passed` false, `observed` "not_run"), keeping `default_promotion_authorized` false.

## EVIDENCE
- `honest_verdict`: `"complete_null_output_boundary_and_history_observer_ready_no_benefit_claim"`
- `acceptance_gate_results`: `benefit`: `"expected": "separate_authorized_live_comparison"`, `"observed": "not_run"`, `"passed": false`, `"principle": "Readiness and oracle-defined fixtures do not measure solve or policy benefit."`
- `acceptance_gate_results`: `readiness`: `"expected": { "arc_output_boundary_ready_score": 1, "history_observer_ready_score": 1 }`, `"observed": { "arc_output_boundary_ready_score": 1, "history_observer_ready_score": 1 }`, `"passed": true`
- `gate_check_summary`: `"benefit_gate_intentionally_closed": true`
- `default_promotion_authorized`: `false`
- `methodology_note`: `"Fixture success is circular positive control evidence only. The terminal result is a null because no solve, action-efficiency, or acceptance-threshold benefit was tested."`
- `protected_path_reproduction`: `"exit_code": 2`, `"guard_message": "e3 requires --output outside results/ (immutable evidence)"`, `"output_exists": false`, `"passed": true`
- `independent_reduction`: `"all_units_passed": true`, `"passed_unit_count": 8`, `"row_count": 8`

## RECOMMENDATION
KEEP

## experiment_7590_evidence_pilot.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a pre-execution gate receipt and makes no empirical or comparative claim.

## WAS THAT CHECKED
No; execution was blocked at the pre-gate stage before any experimental trials were run.

## EVIDENCE
`schema`
`blocked_gate_check_v1`
`status`
`blocked`
`honest_verdict`
`blocked_gate_check_failed`
`duration_s`
`0.0`
`blocked_at_layer`
`conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7596_v663_evidence_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this artifact is a gate check and audit receipt that disclaims any comparative or scientific finding, there is no substantive empirical claim to refute. The artifact's determination of a blocked run would only be falsified if valid upstream evidence producers were actually present and eligible on disk rather than missing.

## WAS THAT CHECKED
Yes; the preconditions for required scientific producers were explicitly checked under `preconditions_checked` and `gate_check_summary`, confirming that the upstream files failed `exists_and_eligible`, which halted execution and left `rows` empty.

## EVIDENCE
- `honest_verdict`: `complete_blocked_missing_v663_evidence_producers`
- `verdict_class`: `blocked`
- `rows`: `[]`
- `fresh_confirmatory_claim_allowed`: `false`
- `oracle_distinct_claim_allowed`: `false`
- `hidden_score_claimed`: `false`
- `production_activation_authorized`: `false`
- `gate_check_summary` -> `passed`: `false`
- `first_failure` -> `check`: `required_scientific_producer`, `observed`: `false`
- `acceptance_gate_results` -> `observed`: `missing_external`, `observed`: `not_started`, `observed`: `not_measured`
- `branch_conclusions` -> `conclusion`: `missing_external_evidence_not_scientific_failure`, `validity`: `blocked_missing_capture`
- `validation_receipts`
- `mutation_receipts`

## RECOMMENDATION
KEEP

## experiment_7597_v663_arc_history_generalization.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
History conditioning across ARC game trajectories provides insufficient matched repeated-key support across games to evaluate cross-game generalization or claim policy benefit (`complete_null_insufficient_history_support`).

## WHAT WOULD REFUTE IT
Observing at least 3 independent qualifying games that each met the floor of at least 20 matched repeated keys across evaluation seeds (`qualifying_game_count` >= 3, `history_support_floor_met` = true, and `history_support_ready_score` = 1), which would have triggered the cross-game bootstrap evaluation.

## WAS THAT CHECKED
Yes. Matched repeated-key support was checked across multi-seed trajectory pairs for multiple games (`su15`, `sp80`, `ft09`, `sb26`) in `per_game_comparisons` and aggregated in `cross_game_comparison`. Every evaluated game fell below the 20-key floor (yielding counts of 6, 10, 0, and 4), resulting in 0 qualifying games.

## EVIDENCE
- `honest_verdict`: `"complete_null_insufficient_history_support"`
- `beneficial_policy_change_claimed`: `false`
- `cross_game_comparison`:
  - `minimum_matched_repeated_keys_per_game`: `20`
  - `required_game_count`: `3`
  - `qualifying_game_count`: `0`
  - `ready`: `false`
  - `reason`: `"insufficient_support"`
- `history_support_ready_score`: `0`
- `gate_check_summary`:
  - `history_support_floor_met`: `false`
  - `benefit_gate_intentionally_closed`: `true`
- `new_solve_claimed`: `false`
- `default_promotion_authorized`: `false`

## RECOMMENDATION
KEEP

## experiment_7598_v663_rust_consumer.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The Rust consumer meets functional parity and consumer readiness requirements but fails the speed benefit gate because its lower 95% confidence bound ratio does not exceed 1.0 against the strongest comparator.

## WHAT WOULD REFUTE IT
A paired comparison result in which the lower 95% bootstrap confidence bound (`lower95`) against the fastest eligible comparator (`python_inprocess`) exceeded 1.0 across all evaluation modes (specifically in `cold:1`), causing the `strongest_comparator_lower95` acceptance gate check to pass and setting `consumer_speed_benefit_score` to 1.

## WAS THAT CHECKED
Yes; evaluated across 120 paired blocks covering four distinct conditions (`cold:1`, `cold:8`, `warm:1`, and `warm:8`) in `consumer_comparison_rows`, evaluated against `python_inprocess` and `python_service`, summarized in `comparison_summary`, and adjudicated under `acceptance_gate_results`.

## EVIDENCE
- `"honest_verdict": "complete_null_rust_consumer_ready_speed_gate_failed"`
- `"verdict_class": "null"`
- `"consumer_ready_score": 1`
- `"consumer_speed_benefit_score": 0`
- `"check": "strongest_comparator_lower95"`
- `"expected": 1.0`
- `"observed": 0.6797261332013704`
- `"op": "gt"`
- `"passed": false`
- `"primary_comparator": "python_inprocess"`
- `"negative_count": 27`
- `"positive_count": 3`
- `"lower95": 0.6797261332013704`
- `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7599_v663_board_continuity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Board continuity is complete across three authenticated historical board scopes with zero present hardware operations, while accelerator placement benefit remains unmeasured.

## WHAT WOULD REFUTE IT
The claim is an honest null audit. Any of the following observations in the artifact's own data would refute it:
1. Historical board dispositions failing authentication, or historical hashes being counted as evidence of current reachability (`historical_hash_is_current_reachability` observing `true`).
2. Present board hardware execution taking place (`hardware_command_count` greater than zero, non-empty `hardware_operations_issued`, or `zero_current_hardware_operations` observing non-zero).
3. Upstream client timing stages providing complete measured IPC and persistence breakdowns sufficient to compute placement speedup, contradicting `placement_unmeasured` (`new_hardware_benefit_measured` observing `1`).
4. Authorizing an accelerator purchase or asserting accelerator speedup without measured client placement (`new_accelerator_justified_by_small_host_update` or `purchase_authorized` observing `true`).

## WAS THAT CHECKED
Yes. Preconditions and acceptance gates verified SHA256 hashes for all three board artifacts (`KV260`, `PolarFire`, and `GateMate`); physical receipts were scanned (35 candidate entries evaluated in `physical_state_receipt`, zero accepted); hardware operations were audited and confirmed empty; and upstream Exp7598 stage timings were checked and confirmed to lack separate measured IPC and persistence fields.

## EVIDENCE
- `honest_verdict`: `complete_null_board_continuity_placement_unmeasured`
- `verdict_class`: `null`
- `board_continuity_complete_score`: `1`
- `placement_scope`: `placement_unmeasured`
- `new_hardware_benefit_measured`: `0`
- `zero_current_hardware_operations`: `0`
- `historical_hash_is_current_reachability`: `false`
- `hardware_command_count`: `0`
- `hardware_operations_issued`: `[]`
- `authenticated_board_count`: `3`
- `blocked_board_count`: `1`
- `accepted_receipt_count`: `0`
- `public_client_stage_timings`: `missing separate measured IPC and persistence fields`
- `new_accelerator_justified_by_small_host_update`: `false`
- `purchase_authorized`: `false`

## RECOMMENDATION
KEEP

## experiment_7600_v663_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or value claim is made. The recorded blocked disposition would be falsified if the required scientific producer artifacts existed, were eligible, and the aggregate gate passed.

## WAS THAT CHECKED
Yes, as a completeness check in `gate_check_summary`; it found eight failed checks, including five absent producer artifacts and an ineligible required scientific producer. No method-benefit claim was tested.

## EVIDENCE
`"honest_verdict": "complete_blocked_required_v663_external_evidence"`; `"claim_scope": "descriptive_reuse"`; `"positive": 0`; `"missing": 14`; `"passed": false`; `"failed_count": 8`; `"check": "producer_artifact_exists"`; `"observed": false`; `"benefit": "not_measured"`; `"fresh_confirmatory_claim_allowed": false`; `"model_compute_used": false`; `"external_contact_performed": false`

## RECOMMENDATION
KEEP
