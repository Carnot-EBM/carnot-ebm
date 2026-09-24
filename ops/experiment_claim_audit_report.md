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
| CLAIM_SUPPORTED | 3 |
| NO_CLAIM | 5 |

## experiment_7605_fit_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact makes no empirical or comparative claim, serving strictly as a pre-execution receipt documenting that upstream preconditions were unmet and execution was blocked.

## WAS THAT CHECKED
no; no experiment was executed and no hypothesis or claim was evaluated.

## EVIDENCE
`schema`: `blocked_gate_check_v1`
`status`: `blocked`
`honest_verdict`: `blocked_gate_check_failed`
`duration_s`: `0.0`
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7606_test_online_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A finding that upstream dependencies had in fact satisfied all gate requirements, or evidence of an empirical/comparative claim being asserted by the artifact. Because this artifact is purely an execution receipt recording a pre-gate block, there is no hypothesis or comparative claim under test to refute.

## WAS THAT CHECKED
No; the experiment was aborted at the pre-gate check prior to execution, so no experimental evaluation was conducted.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `duration_s`: `0.0`
- `honest_verdict`: `blocked_gate_check_failed`
- `blocked_at_layer`: `conductor_pre_gate`
- `gate_check_summary`: `gate-unsat(final): 2 of 7 gate(s) failed; first failure: exp7604-evidence-pilot.evidence_transport_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_7610_v664_evidence_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this artifact is an audit receipt that explicitly disclaims all benefit, calibration, decision value, and causal learning claims, there is no substantive comparative claim to refute. To refute its status as a non-claiming audit receipt, the artifact would need to report authorized promotion, assert incremental information or causal learning, or record unblocked downstream evaluation while lacking valid upstream scientific producers (`exp7607-evidence-energy`, `exp7608-decision-evaluation`, `exp7609-guarded-learning`) and lacking schema-valid pilot evidence rows.

## WAS THAT CHECKED
Yes. Preconditions and acceptance gates were checked across all branches in `preconditions_checked`, `gate_check_summary`, `acceptance_gate_results`, and `branch_conclusions`, confirming that required upstream scientific producers are missing, all eight pilot transport outputs are schema-invalid, readiness is blocked upstream, and no comparative, probability, cost, coverage, or causal learning claims are authorized.

## EVIDENCE
- `honest_verdict`: `complete_blocked_v664_evidence_chain_unavailable`
- `verdict_class`: `blocked`
- `default_promotion_authorized`: `false`
- `production_activation_authorized`: `false`
- `fresh_confirmatory_claim_allowed`: `false`
- `oracle_distinct_claim_allowed`: `false`
- `model_invoked`: `false`
- `conclusion`: `transport_failure_is_not_an_incremental_information_null`
- `conclusion`: `no_probability_claim`
- `conclusion`: `no_cost_or_coverage_claim`
- `conclusion`: `no_causal_learning_claim`
- `conclusion`: `no_retention_claim`
- `conclusion`: `fresh_confirmatory_claim_not_allowed`
- `conclusion`: `current_audit_cost_is_aggregation_only`
- `scientific_hypothesis_retired`: `false`
- `status`: `not_retired_pre_inference_block`

## RECOMMENDATION
KEEP

## experiment_7611_v664_arc_matched_support.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Matched-prefix protocol fixtures are verified ready, but empirical history-disambiguation benefit is not established due to zero matched keys observed in natural trajectories.

## WHAT WOULD REFUTE IT
The headline claim would be refuted by either:
1. Observing empirical history-disambiguation benefit in the natural trajectory data: specifically, finding candidate keys with divergent histories (`different_history_candidate_keys` > 0, `selected_matched_keys` > 0), producing a non-zero evaluation denominator (`stable_denominator` > 0), a positive disambiguation witness rate for h1 over h0 (`primary_h0_vs_h1.witness_numerator` > 0, `rate` > 0), and a passing benefit gate (`benefit.result` = true).
2. Protocol readiness failure: any calibration fixture failing to conform (`matched_support_ready_score` != 1, `readiness.result` = false, or any entry in `fixture_results` returning false).

## WAS THAT CHECKED
Yes. 
- Empirical benefit was checked during the `natural_episodes` and `natural_selection` phases across 12 episodes and 6 games (`su15`, `sp80`, `ft09`, `sb26`, `g50t`, `dc22`) covering 7,062 eligible occurrences; all 758 distinct keys exhibited identical histories (`repeated_same_history_keys` = 758, `different_history_candidate_keys` = 0), yielding 0 selectable matched keys and leaving the benefit gate unpassed.
- Protocol readiness was checked in `protocol_fixtures` across 6 distinct fixture tests (`coordinate_action`, `same_action_level_boundary`, `singleton`, `stateful_reset`, `two_hidden_histories`, `unstable_prefix`) and unit tests, all of which passed.

## EVIDENCE
- `honest_verdict`: `complete_null_matched_prefix_fixture_ready_empirical_benefit_not_established`
- `verdict_class`: `null`
- `acceptance_gate_results`:
  - `benefit`: `expected`: `separate_empirical_gate`, `observed`: `not_run`, `result`: `false`
  - `readiness`: `expected`: `1`, `observed`: `1`, `result`: `true`
  - `validity`: `expected`: `true`, `observed`: `true`, `result`: `true`
- `gate_check_summary`: `benefit_gate_intentionally_closed`: `true`, `empirical_matched_key_count`: `0`, `passed`: `true`
- `natural_trajectory_denominator`:
  - `eligible_target_occurrences`: `7062`
  - `distinct_pre_outcome_keys`: `758`
  - `repeated_same_history_keys`: `758`
  - `different_history_candidate_keys`: `0`
  - `selected_matched_keys`: `0`
- `primary_h0_vs_h1`:
  - `direction`: `history_disambiguation_witness_rate_h1_over_h0`
  - `rate`: `null`
  - `stable_denominator`: `0`
  - `witness_numerator`: `0`
- `rows`:
  - `unit`: `dc22`
  - `arm`: `h0`
  - `denominator`: `0`
  - `numerator`: `0`
  - `rate`: `null`
  - `provenance`: `fixture_only`
- `protocol_fixtures`:
  - `matched_support_ready_score`: `1`
  - `verifier_is_oracle`: `true`
  - `methodology_note`: `Exact fixture outcomes calibrate the protocol and are not empirical benefit.`

## RECOMMENDATION
KEEP

## experiment_7612_v664_arc_history_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because the artifact records a blocked run and explicitly asserts no comparative claim, there is no headline claim to refute. For any future claim asserting that history disambiguation provides empirical benefit, observing zero or negative disambiguation across independent games, failing upstream protocol verification, or failing validity and benefit acceptance gates would refute it.

## WAS THAT CHECKED
Yes. Precondition checks were evaluated during the preconditions phase. The upstream protocol check `exp7611_protocol` failed, resulting in failed validity and readiness gates and blocking the experiment before model execution or benefit measurement occurred.

## EVIDENCE
- `cross_game_history_claim`: `null`
- `honest_verdict`: `complete_blocked_exp7611_protocol`
- `verdict_class`: `blocked`
- `actual_inference_substrate_class`: `blocked_no_run`
- `new_game_level_solve_claimed`: `false`
- `external_publication_authorized`: `false`
- `passed`: `false`
- `observed`: `not_run`
- `support_disposition`: `insufficient_matched_support`
- `independent_validity`: `no_current_units`

## RECOMMENDATION
KEEP

## experiment_7613_v664_service_attribution.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Service-level latency attribution preserves the prior Exp7598 null verdict by demonstrating that update arithmetic accounts for under 9% of whole-service time across all tested strata, yielding no empirical accelerator benefit.

## WHAT WOULD REFUTE IT
Observing that update arithmetic accounts for a dominant fraction of end-to-end service latency (yielding an Amdahl upper bound meaningfully above 1.0x), observing `empirical_accelerator_benefit` as true, or failing the `exp7598_aggregate_null_preserved` acceptance check.

## WAS THAT CHECKED
Yes. Stage attribution was measured across 120 paired stage blocks spanning cold and warm modes at batch sizes 1 and 8 in `consumer_stage_rows`, reduced with explicit Amdahl bounds in `stage_reduction`, controlled for telemetry overhead across 40 blocks in `instrumentation_overhead_rows`, and evaluated in `acceptance_gate_results`.

## EVIDENCE
- `"honest_verdict": "complete_null_service_attribution_preserves_exp7598_null"`
- `"verdict_class": "null"`
- `"prior_exp7598_aggregate_verdict_preserved": true`
- `"prior_exp7598_honest_verdict": "complete_null_rust_consumer_ready_speed_gate_failed"`
- `"check": "empirical_accelerator_benefit"`, `"expected": true`, `"observed": false`, `"passed": false`
- `"check": "exp7598_aggregate_null_preserved"`, `"expected": "complete_null_rust_consumer_ready_speed_gate_failed"`, `"observed": "complete_null_rust_consumer_ready_speed_gate_failed"`, `"passed": true`
- `"kernel_only_timing_justifies_purchase": false`
- `"purchase_authorized": false`
- `"cold:1"` `"estimate": 0.00588140452732116`
- `"cold:1"` `"estimate": 1.005916200182142`
- `"warm:8"` `"estimate": 0.08910116474756832`
- `"warm:8"` `"estimate": 1.097819815869542`

## RECOMMENDATION
KEEP

## experiment_7614_v664_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
As a capstone disposition and governance receipt artifact, no comparative or empirical benefit claim is made. If evaluated as an operational claim that milestone execution is blocked by missing prerequisites, observing valid, schema-conforming producer artifacts and passing conductor pre-gates for all 14 task dispositions would refute the blocked status.

## WAS THAT CHECKED
Yes. The artifact audited upstream task receipts and recorded 7 failed prerequisites in `gate_check_summary`, confirming that required external evidence remains blocked.

## EVIDENCE
`positive_claim`
`false`
`oracle_distinct_positive_claimed`
`false`
`honest_verdict`
`complete_blocked_required_v664_external_evidence`
`status`
`complete_blocked_required_v664_external_evidence`
`inference_substrate`
`aggregation_from_upstream_artifacts`
`numbered_runtime_e2e`
`not_applicable_read_only_reporting`
`failed_count`
`7`
`claim_scope`
`descriptive_reuse`
`direction`
`exact_custody_is_required`

## RECOMMENDATION
KEEP

## experiment_10012_gate_usefulness.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Gate usefulness was measured across three informative windows with none unrecoverable; the artifact does not claim that any gate proved useful.

## WHAT WOULD REFUTE IT
The claim would fail if usefulness labels were derived from the gate metrics themselves, if the evaluated windows lacked independently labeled positive and negative outcomes, or if unrecoverable rows were counted as informative.

## WAS THAT CHECKED
Yes. Execution outcomes define the labels separately from metric-based gate decisions, and the contingency tables contain both gate errors and correct decisions. The real-candidate-only cohort even records rejected positives and accepted negatives, demonstrating that the gates were genuinely allowed to fail.

## EVIDENCE
The headline is `complete_gate_usefulness_measured_3_informative_windows_0_unrecoverable`. The substrate is `verifier_ensemble_against_cached_candidates`. Rows carry `window_status` equal to `label_informative`; labels include `FAITHFUL` and `USEFUL`, while execution separately reports `real_level_up`. The `without_experts` cohort is `Real candidates only: THINK and CODEONLY; both EXPERT and IDENTITY controls removed`. In the live real-candidate table, `cell_recall_0.8` has `n_pairs` `12`, `accepted_and_positive` `0`, `accepted_and_negative` `1`, and `rejected_positive` `1`. The artifact also states `No p-values were computed and no threshold was selected from these outcomes.` and `No live gate or agent default was changed.`

## RECOMMENDATION
KEEP
