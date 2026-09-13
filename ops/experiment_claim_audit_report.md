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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 3 |

## experiment_7266_semantic_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a failed prerequisite gate and makes no substantive comparative or value claim.

## WAS THAT CHECKED
No; the experiment was blocked before the semantic audit ran.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_field": "mention_capture_complete_score"`; `"failed_expected": 1`; `"failed_observed": 0`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7267_v639_recognition_prototype.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The active recognition fixture is ready, implying positive value despite held-out learning value not being scored.

## WHAT WOULD REFUTE IT
On sealed prospective streams, the active-recognition arm tying or producing more full-denominator error than the serious `previous_coverage` baseline under equal query and memory limits, as judged independently of the correctness-defining verifier.

## WAS THAT CHECKED
No. Prospective rows were generated, but held-out learning value was explicitly not scored, and the verifier defining success was itself the oracle. Operational readiness gates and changed query selections do not test comparative value.

## EVIDENCE
`honest_verdict`: `complete_circular_positive: active recognition fixture is ready; held-out learning value is not scored`; `held_out_learning_value_scored`: `false`; `verifier_is_oracle`: `true`; `verdict_class`: `circular_positive`; `arm`: `previous_coverage`; `model_invoked`: `false`; `attempted_generation_calls`: `0`.

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_7268_v639_recognition_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Autonomous recognition completed, but one or more frozen value gates failed.

## WHAT WOULD REFUTE IT
Either an incomplete recognition run or every frozen value gate passing would falsify the claim.

## WAS THAT CHECKED
Yes. Run completion was checked through the complete event-arm matrix, lifecycle and end-to-end receipts, and the completion score; value was separately tested by frozen acceptance gates, several of which failed. The serious full-memory comparator also beat active recognition on future error, consistent with the null rather than a positive value claim.

## EVIDENCE
`honest_verdict`: `complete_null: autonomous recognition completed but one or more frozen value gates failed`; `recognition_run_complete_score`: `1`; `recognition_value_score`: `0`; `complete_event_arm_matrix`; `expected`: `196608`; `observed`: `196608`; `pass`: `true`; `false_accept_vs_reset`; `pass`: `false`; `future_error_vs_reset`; `pass`: `false`; `recurrence_degradation_vs_frozen`; `pass`: `false`; `recurrence_error_vs_shuffle`; `pass`: `false`; `active_superiority_claimed`: `false`; `future_error_vs_full_memory`; `estimate`: `0.009672619047619048`; `ci95`: `[0.0017206101190476218, 0.017159598214285716]`; `verifier_is_oracle`: `true`; `verdict_class`: `null`

## RECOMMENDATION
KEEP

## experiment_7269_v639_recognition_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The recognition audit completed, but the promotion gates did not all pass.

## WHAT WOULD REFUTE IT
A promotion score of 1 with every promotion gate passing, or evidence that the planned audit units were incomplete or censored, would refute the headline.

## WAS THAT CHECKED
Yes. The artifact reports full planned-stream completion with zero censored streams, reconstructs the complete stream-arm matrix, and evaluates each promotion gate individually; multiple gates fail.

## EVIDENCE
`honest_verdict`: `complete_null: recognition audit completed but promotion gates did not all pass`; `recognition_audit_complete_score`: `1`; `recognition_promotion_score`: `0`; `attempted_stream_count`: `24`; `completed_stream_count`: `24`; `censored_stream_count`: `0`; `false_accept_vs_reset`: `pass`: `false`; `future_error_vs_reset`: `pass`: `false`; `recurrence_degradation_vs_frozen`: `pass`: `false`; `recurrence_error_vs_shuffle`: `pass`: `false`; `verifier_is_oracle`: `true`; `verdict_class`: `null`

## RECOMMENDATION
KEEP

## experiment_7270_v639_durable_profile.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Durable synchronization costs and insufficient replaceable snapshot work do not warrant a delta-log prototype.

## WHAT WOULD REFUTE IT
A measured replaceable fraction of at least 0.5 together with an estimated delta-log speedup of at least 1.5×—without parity or durability failures—would refute the claim.

## WAS THAT CHECKED
Yes. The artifact applied both thresholds in `journal_warrant` using a complete 12-block component profile; all blocks were uncensored and checked for component accounting, parity, and durability.

## EVIDENCE
`honest_verdict` = `complete_null: durable sync or insufficient replaceable snapshot work blocks a delta-log prototype`; `replaceable_fraction_expected` = `>=0.5`; `replaceable_fraction_observed` = `0.011790551645999563`; `replaceable_fraction_passed` = `false`; `delta_log_speedup_expected` = `>=1.5`; `delta_log_speedup_observed` = `0.7561440834982079`; `delta_log_speedup_passed` = `false`; `sync_dominates` = `true`; `sync_fraction` = `0.9355276068704991`; `completed_blocks` = `12`; `censored_blocks` = `0`; `component_sum_failure_count` = `0`; `durability_failure_count` = `0`; `parity_failure_count` = `0`; `journal_optimization_warranted_score` = `0`.

## RECOMMENDATION
KEEP

## experiment_7271_delta_log.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies because the artifact reports only a blocked pre-gate receipt and makes no comparative or performance claim.

## WAS THAT CHECKED
No; the experiment did not proceed beyond the upstream gate.

## EVIDENCE
`"status"` is `"blocked"`; `"honest_verdict"` is `"blocked_gate_check_failed"`; `"passed"` is `false`; `"blocked_at_layer"` is `"conductor_pre_gate"`.

## RECOMMENDATION
KEEP

## experiment_7272_v639_board_state.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
This receipt records three authenticated board dispositions and their next conditions while preserving prior KV260 and PolarFire states and leaving GateMate blocked.

## WHAT WOULD REFUTE IT
Fewer than three authenticated dispositions, a missing next condition, a hash or receipt authentication failure, a board state inconsistent with the stated disposition, any hardware operation during this host-only aggregation, or evidence that GateMate’s required operator-authored change receipt existed.

## WAS THAT CHECKED
Yes. The three per-board rows, acceptance gates, operator-receipt search, authentication fields, and hardware-operation counts directly check those receipt-level assertions. No comparative method-value claim was made that would require a rival baseline.

## EVIDENCE
`"inference_substrate": "aggregation_from_upstream_artifacts"`; `"independent_units_completed": 3`; `"three_dispositions_reduce"`; `"observed": 1`; `"passed": true`; `"latest_receipt_authenticated": true`; `"hardware_operations_issued": []`; `"disposition": "blocked_changed_physical_state"`; `"exists": false`; `"programmable_logic_sampling_observed": false`

## RECOMMENDATION
KEEP

## experiment_7273_v639_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V639 matrix is complete, but held-out fidelity and scientific value are null or blocked, so no positive closeout is warranted.

## WHAT WOULD REFUTE IT
Any of these observations would falsify the headline: fewer than 14 completed dispositions; exact held-out source fidelity; a completed semantic audit; positive ARC generalization or policy consumption; recognition beating its frozen efficacy and recurrence controls; or established delta-log semantic value.

## WAS THAT CHECKED
Yes. Completion was counted in `sample_size_budget`; fidelity and semantic-audit outcomes appear in `rows` and `gate_check_summary`; ARC, recognition, and delta-log outcomes appear in `prd_gap_matrix`. The artifact preserves failed and blocked outcomes rather than promoting them positively.

## EVIDENCE
`"completed": 14`; `"planned": 14`; `"source_fidelity_exact"`; `"metric_value": false`; `"source_semantic_value"`; `"claim_class": "blocked"`; `"required_semantic_audit_available"`; `"observed": "blocked"`; `"public_game_generalization": false`; `"policy_consumption": false`; `"recognition_learning_value": false`; `"durable_log_semantics": null`; `"positive_promoted": false`; `"status": "blocked"`

## RECOMMENDATION
KEEP
