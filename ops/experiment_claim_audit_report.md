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
| CLAIM_SUPPORTED | 7 |
| CLAIM_OVERSTATED | 1 |

## experiment_7294_v641_reuse_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The reuse audit completed but failed validation and is therefore disqualified from promotion.

## WHAT WOULD REFUTE IT
All required validation checks passing, with no failed receipt and the reuse promotion gate satisfied, would refute the disqualification.

## WAS THAT CHECKED
Yes. The validation receipts include the full Python suite, and the artifact records both the audit-completeness and promotion outcomes.

## EVIDENCE
`"honest_verdict"`: `"complete_disqualified_reuse_audit_validation_failed"`; `"status"`: `"complete"`; `"name"`: `"full_python_suite"`; `"exit_code"`: `2`; `"passed"`: `false`; `"reuse_audit_complete_score"`: `1`; `"reuse_promotion_score"`: `0`; `"verdict_class"`: `"disqualified"`; `"verifier_is_oracle"`: `true`

## RECOMMENDATION
KEEP

## experiment_7295_v641_mixture_prototype.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bounded fixed-share fixture passed its declared mechanical, causal-order, state-size, and prediction-change gates, without claiming prospective efficacy.

## WHAT WOULD REFUTE IT
A state exceeding 69,632 bytes, chronology or future-label-read violations, fewer than 24 changed predictions, a reducer mismatch, or any declared mechanical gate failing would refute the headline.

## WAS THAT CHECKED
Yes. The acceptance gates, chronology controls, per-arm rows, and cold reduction checked those failure modes and report no such failure. The oracle does not independently establish efficacy, but that value claim is explicitly excluded.

## EVIDENCE
`honest_verdict` is `complete_circular_positive: bounded fixed-share fixture mechanics pass; prospective efficacy is not claimed; repository-wide suite retained unrelated failures`. `prospective_efficacy_claimed` is `false`. `verifier_is_oracle` is `true`. `bounded_serialized_state` reports `expected` `<=69632`, `observed` `11594`, and `pass` `true`. `changed_weight_predictions` reports `expected` `>=24`, `observed` `324`, and `pass` `true`. The displayed rows report `chronology_violation_count` `0`, `future_label_read_count` `0`, and `censored` `false`. `evaluation_streams_scored_by_this_prototype` is `0`.

## RECOMMENDATION
KEEP

## experiment_7296_v641_mixture_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Prospective fixed-share learning completed, but its frozen value gates failed.

## WHAT WOULD REFUTE IT
All frozen value comparisons meeting their prespecified thresholds—or the named failed gates actually having passing results—would refute the null headline.

## WAS THAT CHECKED
Yes. The artifact evaluates fixed-share learning against reset, unconditional-recognition, and frozen-warmup controls across 24 completed, uncensored prospective streams; six named value gates fail. The oracle defines correctness, but the headline makes no positive claim about the verifier’s added value.

## EVIDENCE
`honest_verdict`: `complete_null: prospective fixed-share learning completed but frozen value gates failed: future_error_vs_reset,future_error_vs_unconditional_recognition,non_feedback_error_vs_reset,non_feedback_error_vs_unconditional_recognition,separated_recurrence_vs_frozen_warmup,overlapping_recurrence_vs_frozen_warmup; repository-wide suite retained pre-existing collection failures`

`mixture_value_score`: `0`

`future_error_vs_reset`: `ci95_upper<0`, `0.005533854166666665`, `false`

`future_error_vs_unconditional_recognition`: `ci95_upper<0`, `0.005394345238095237`, `false`

`non_feedback_error_vs_reset`: `ci95_upper<0`, `0.004394531250000004`, `false`

`non_feedback_error_vs_unconditional_recognition`: `ci95_upper<0`, `0.004503038194444447`, `false`

`separated_recurrence_vs_frozen_warmup`: `ci95_upper<=0.01`, `0.15071614583333334`, `false`

`overlapping_recurrence_vs_frozen_warmup`: `ci95_upper<=0.01`, `0.0361328125`, `false`

`completed_stream_count`: `24`

`censored_stream_count`: `0`

`outcome_based_extension`: `false`

## RECOMMENDATION
KEEP

## experiment_7297_v641_mixture_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Bounded fixed-share selection provides no qualifying efficacy benefit under delayed feedback and should be retired for the stated Exp7295/Exp7296 scope.

## WHAT WOULD REFUTE IT
A paired stream-level interval showing lower prospective future error than a serious baseline—reset or unconditional recognition—with its 95% upper bound below zero, including on non-feedback future outcomes and without material recurrence harm.

## WAS THAT CHECKED
Yes. The artifact reports paired comparisons against reset and unconditional recognition across all-future, non-feedback, recurrence, and feedback-selected subsets, split across overall and both recurrence strata. The qualifying efficacy comparisons failed; only the shuffled-label sanity control was beaten.

## EVIDENCE
`"mixture_promotion_score": 0`; `"future_error_vs_reset"`; `"ci95_upper": 0.005533854166666662`; `"future_error_vs_unconditional_recognition"`; `"observed": 0.005487351190476185`; `"non_feedback_error_vs_reset"`; `"observed": 0.004557291666666668`; `"non_feedback_error_vs_unconditional_recognition"`; `"observed": 0.004503038194444445`; `"true_feedback_future_error_vs_shuffled"`; `"pass": true`; `"independent_unit": "stream"`; `"independent_unit_count": 24`; `"count": 48`; `"censored_stream_count": 0`; `"retirement_triggered": true`; `"verdict_class": "null"`

## RECOMMENDATION
KEEP

## experiment_7298_v641_snapshot_journal.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The persistent full-snapshot acknowledgment protocol passed as a positive correctness/readiness result.

## WHAT WOULD REFUTE IT
An independently judged recovery showing any acknowledged event missing, duplicated, reordered, or restored to the wrong state would refute the claim; disagreement between an independent oracle and the protocol’s verifier would also refute its claimed value.

## WAS THAT CHECKED
No. Mechanical loss, duplication, parity, corruption, and capacity failures were exercised, but correctness was defined by the same oracle used to verify success; no independent correctness check was present.

## EVIDENCE
`honest_verdict` is `complete_circular_positive: persistent full-snapshot acknowledgment protocol passed`; `verifier_is_oracle` is `true`; `verdict_class` is `circular_positive`. The execution-grounded checks report `lost_acknowledged_event_count` of `0`, `duplicate_apply_count` of `0`, `valid_complete_state` of `true`, and `parity_failure_count` of `0`.

## RECOMMENDATION
NARROW_CLAIM

## experiment_7299_v641_snapshot_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Persistent SQLite full snapshots do not meet every frozen V640 storage deployment bound.

## WHAT WOULD REFUTE IT
All frozen deployment gates passing, including the paired group-16 throughput lower CI95 reaching 1.5 and the full-boundary cold lower CI95 reaching 10.0.

## WAS THAT CHECKED
Yes. The acceptance gates directly tested those thresholds using eight paired seeds against the `atomic_replace` baseline; both required throughput gates failed. The oracle relationship does not circularly establish this negative performance claim.

## EVIDENCE
`honest_verdict`: `complete_null: persistent SQLite full snapshots do not meet every frozen V640 storage deployment bound`; `fixed_group16_burst_throughput`; `lower_ci95`: `1.4558337555069685`; `passed`: `false`; `nfr01_full_boundary_10x`; `observed`: `1.5549724849050095`; `passed`: `false`; `snapshot_value_score`: `0`; `paired_seeds`: `8`; `storage_arms`: `atomic_replace`, `sqlite_persist`; `censored_trial_units`: `0`; `capture_complete`: `true`

## RECOMMENDATION
KEEP

## experiment_7300_v641_board_continuity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The artifact records authenticated dispositions and exact next conditions for all three boards, preserves prior KV260 fabric execution and PolarFire CPU dispatch without mislabeling it as FPGA sampling, and correctly blocks GateMate because no later physical-change receipt exists.

## WHAT WOULD REFUTE IT
An unauthenticated or missing board disposition; a missing next condition; KV260 evidence showing no fabric execution or host emulation; PolarFire showing failed dispatch/hash matching or FPGA sampling; a qualifying later GateMate receipt despite the blocked disposition; or any hardware operation issued by this invocation.

## WAS THAT CHECKED
Yes. The per-board rows check authentication, execution venue, processor class, execution/hash state, exact next conditions, abstention, and censoring. The receipt search checks whether a qualifying GateMate change exists, while the acceptance gates and operation records check fail-closed behavior and zero hardware operations.

## EVIDENCE
`"three_continuity_rows_reduce"` has `"observed": 1` and `"passed": true`. All three rows have `"latest_receipt_authenticated": true`, `"censored": false`, and an `"exact_next_condition"`. KV260 records `"fabric_execution_completed": true`, `"host_emulation": false`, and `"observed_venue": "kv260_fpga_fabric"`. PolarFire records `"dispatch_completed": true`, `"input_hash_matches": true`, `"output_hash_matches": true`, `"processor_class": "cpu"`, and `"programmable_logic_sampling_observed": false`. GateMate records `"disposition": "blocked_changed_physical_state"`, `"abstention": true`, `"terminal_criterion_met": false`, and `"exists": false`. `"hardware_operations_issued_count"` has `"observed": 0` and `"passed": true`. `"negative_receipt_fixtures_fail_closed"` has `"all_failed_closed": true`.

## RECOMMENDATION
KEEP

## experiment_7301_v641_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Required V641 science is blocked: ARC evidence is quarantined, the self-parse session stopped at its pre-gate, and the completed reuse, mixture, and storage branches yielded no promoted value.

## WHAT WOULD REFUTE IT
An authenticated, unquarantined ARC boundary followed by a completed self-parse session, or a positive value score for any completed reuse, mixture, or storage branch, would refute the claim.

## WAS THAT CHECKED
Yes. The ARC gate replay checked for an unquarantined readiness score, the self-parse disposition checked completion, and the independent-branch gate checked the reuse, mixture, and snapshot value scores against a positive target.

## EVIDENCE
`"honest_verdict": "blocked_required_v641_science_unavailable: all fourteen dispositions are represented; ARC boundary evidence is quarantined and Exp7290 stopped at its failed pre-gate; complete reuse, mixture, and storage branches have no promoted value"`; `"arc_boundary_ready_score"`; `"expected_value": 1`; `"observed_value": 0`; `"quarantined": true`; `"status": "blocked"`; `"arc_method_value_score": null`; `"reuse_promotion_score": 0`; `"mixture_promotion_score": 0`; `"snapshot_value_score": 0`; `"positive_promoted": false`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP
