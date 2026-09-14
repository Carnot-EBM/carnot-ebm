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
| NO_CLAIM | 2 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7280_v640_arc_live.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7281_v640_admission_prototype.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the limited fixture-mechanics statement, any failed acceptance gate, lifecycle stage, mutation rejection, reducer mismatch, censored stream, or missing arm would refute readiness; no learning or comparative-value claim is made.

## WAS THAT CHECKED
Yes. The acceptance gates, eight lifecycle stages, mutation tests, independent reduction, all 24 prospective streams, and all seven arms were checked; these establish fixture conformance only.

## EVIDENCE
`honest_verdict` `complete_circular_positive: admission fixture mechanics pass; fixed eight-label bounds are often infeasible and no learning value is claimed` `verdict_class` `circular_positive` `verifier_is_oracle` `true` `model_invoked` `false` `admission_fixture_ready_score` `1` `failed_checks` `[]` `censored_stream_count` `0` `completed_stream_arm_rows` `168`

## RECOMMENDATION
KEEP

## experiment_7282_v640_admission_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Admission learning completed, but the frozen value gates failed.

## WHAT WOULD REFUTE IT
A complete run in which every frozen value gate passed—particularly statistically favorable future-error and false-accept comparisons against reset and unconditional recognition, plus passing recurrence controls—would refute the null headline.

## WAS THAT CHECKED
Yes. The artifact reports explicit expected values, observed values, and pass statuses for run completeness and each frozen value gate, including serious reset, unconditional-recognition, and frozen-warmup controls. The gates failed. The oracle-defined verifier therefore supports the execution-grounded null, not a positive value claim.

## EVIDENCE
`"honest_verdict": "complete_null: admission learning completed but frozen value gates failed: future_error_vs_reset,future_error_vs_unconditional_recognition,false_accept_vs_reset,false_accept_vs_unconditional_recognition,recurrence_degradation_vs_frozen_warmup,recurrence_error_vs_label_shuffled_admission"`; `"admission_run_complete_score": 1`; `"admission_value_score": 0`; `"future_error_vs_reset"`; `"observed": 0.007952008928571433`; `"pass": false`; `"false_accept_vs_reset"`; `"observed": 0.025065104166666668`; `"pass": false`; `"recurrence_degradation_vs_frozen_warmup"`; `"observed": 0.07112630208333333`; `"pass": false`; `"verdict_class": "null"`; `"verifier_is_oracle": true`

## RECOMMENDATION
KEEP

## experiment_7283_v640_admission_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The cold admission audit completed, but the admission method failed the frozen upstream efficacy value gate.

## WHAT WOULD REFUTE IT
A passed upstream efficacy gate or an incomplete/censored audit would refute the headline.

## WAS THAT CHECKED
Yes. The upstream-efficacy gate explicitly failed, while planned and completed stream counts matched, all 168 stream-arm units completed, and zero streams were censored.

## EVIDENCE
`honest_verdict` `complete_null: cold admission audit completed; upstream efficacy value gate failed` `upstream_efficacy` `observed` `0` `pass` `false` `upstream_admission_value_score` `0` `planned_stream_count` `24` `completed_stream_count` `24` `planned_stream_arm_units` `168` `completed_stream_arm_units` `168` `censored_stream_count` `0`

## RECOMMENDATION
KEEP

## experiment_7284_v640_commit_prototype.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The host group-commit prototype satisfies the changed acknowledgment contract.

## WHAT WOULD REFUTE IT
A lost acknowledged event, duplicate application, partial recovered state, serial-transition mismatch, leaked queued state, or unbounded producer disposition would refute operational conformance; an independent verifier rejecting the result would refute the positive value claim.

## WAS THAT CHECKED
No, not independently. The operational failure modes were checked across six crash boundaries, 96 event rows, semantic-parity rows, and queue controls, but correctness was judged by the oracle that defines the contract itself.

## EVIDENCE
`honest_verdict` `complete_circular_positive: host group commit satisfies the changed acknowledgment contract` `verifier_is_oracle` `true` `verdict_class` `circular_positive` `lost_acknowledged_event_count` `0` `duplicate_apply_count` `0` `invalid_partial_state` `false` `parity_failure_count` `0` `all_controls_passed` `true` `completed_event_count` `96` `completed_crash_units` `6`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7285_v640_commit_frontier.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The measured host commit frontier does not satisfy every bounded deployment gate.

## WHAT WOULD REFUTE IT
Every declared deployment gate passing, including steady acknowledgment p95 being at or below 50,000,000 ns, would falsify the claim.

## WAS THAT CHECKED
Yes. The acceptance-gate results directly compare steady acknowledgment p95 with its declared threshold, and the independent reducer reports the overall value gate result.

## EVIDENCE
`honest_verdict`: `complete_null: measured host commit frontier does not meet every bounded deployment gate`; `steady_acknowledgment_p95`: `expected`: `<=50000000 ns`, `observed`: `220197149.6`, `passed`: `false`; `value_gate_passed`: `false`; `verdict_class`: `null`

## RECOMMENDATION
KEEP

## experiment_7286_v640_board_state.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
As a receipt, it would be internally contradicted if any board lacked an authenticated disposition or exact next condition, if KV260 lacked its preserved fabric evidence, if PolarFire were represented as FPGA sampling rather than CPU dispatch, if a qualifying GateMate change receipt existed, or if this invocation issued a hardware operation.

## WAS THAT CHECKED
Yes. The three board rows expose authentication, disposition, next-condition, terminal-state, processor-class, and hardware-operation fields; the acceptance gates reduce those records, and two negative receipt fixtures exercised fail-closed behavior. These checks establish receipt consistency, not comparative method value.

## EVIDENCE
`inference_substrate` `aggregation_from_upstream_artifacts` `model_invoked` `false` `hardware_operations_issued_count` `observed` `0` `three_dispositions_reduce` `passed` `true` `latest_receipt_authenticated` `exact_next_condition` `graduated_preserved` `blocked_changed_physical_state` `graduated_cpu_dispatch_preserved` `processor_class` `cpu` `programmable_logic_sampling_observed` `false` `negative_receipt_fixtures_fail_closed` `all_failed_closed` `true`

## RECOMMENDATION
KEEP

## experiment_7287_v640_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The capstone is blocked because Exp7280 carries critical inference-provenance flags, despite all fourteen dispositions being represented, and all four scientific value scores remain zero.

## WHAT WOULD REFUTE IT
An Exp7280 observation of `flagged_adversarial` equal to `false`, making the required science authentic and eligible rather than quarantined, would refute the stated reason for blocking promotion.

## WAS THAT CHECKED
Yes. `gate_check_summary` directly compares Exp7280’s `flagged_adversarial` field with the expected value and records the contrary observed value; the acceptance gates separately test required-science authenticity and the four value scores.

## EVIDENCE
`"upstream": "exp7280-arc-live"`; `"artifact_field": "flagged_adversarial"`; `"expected_value": false`; `"observed_value": true`; `"passed": false`; `"terminal_classification": "blocked"`; `"criterion": "required_science_authentic"`; `"observed": false`; `"source_promotion_score": 0`; `"arc_method_value_score": 0`; `"admission_promotion_score": 0`; `"commit_cost_value_score": 0`; `"criterion": "fourteen_task_dispositions"`; `"observed": 14`; `"passed": true`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP
