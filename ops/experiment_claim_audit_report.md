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
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 3 |

## experiment_8393_python_transaction_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the title states an intended measurement; the artifact reports a blocked run without asserting a transaction-cost result or comparative benefit.

## WAS THAT CHECKED
No transaction-cost result was tested. The prerequisite gate was checked and failed: readiness was 0, while 1 was required.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `direct_state_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8395_label_criterion_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-run receipt, with no factuality result or comparative claim to falsify.

## WAS THAT CHECKED
No empirical claim was tested. The prerequisite gate was checked in gates_evaluated and failed: actual 0 versus expected 1, with passed false.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `released_label_panel_ready_score`
- `failed_expected`: `1`; `failed_observed`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8396_v723_cuda_failure_cause.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8397_bounded_qwen_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No experimental claim is made. This artifact records a blocked attempt, so there is no canary-performance claim to falsify.

## WAS THAT CHECKED
No canary hypothesis was tested. The three prerequisite checks were evaluated, each observing 0 against an expected 1 and recording failure. The artifact reports that blockage without claiming experimental success.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `blocked_at_layer`: `conductor_pre_gate`. In `gates_evaluated`, all three entries record `actual`: `0`, `expected`: `1`, and `passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_8398_v723_arc_generalization_panel.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8399_v723_board_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative headline to falsify. A valid current board execution demonstrating a compatible workload, complete transfer-inclusive timing, and measured whole-service benefit would overturn the recorded blocked operational boundary.

## WAS THAT CHECKED
No. The artifact reports zero current device executions and unmeasured service benefit. It records historical evidence and operational prerequisites; it does not assert added verifier value, generalization, or superiority over a rival. The oracle flag therefore does not establish a circular value claim.

## EVIDENCE

- `actual_substrate` = `host_CPU_aggregation`
- `current_device_execution_count` = `0`
- `compatible_fraction_status` = `unmeasured`
- `full_service_speedup` = `null`
- `scientific_benefit` = `false`
- `generalized_learning_benefit_score` = `0`
- `scope` = `historical_only_zero_current_calls`

## RECOMMENDATION
KEEP

## experiment_8400_v723_gatemate_continuity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a continuity receipt recording unresolved obligations without asserting comparative or scientific benefit.

## WAS THAT CHECKED
No comparative refutation was tested or claimed. The rows record one completed continuity receipt and six censored prerequisites. The blocked verdict preserves those limitations; passing validation does not assert verifier value, generalization, or hardware success.

## EVIDENCE

- `honest_verdict`: `complete_blocked_gatemate_continuity`
- `verdict_class`: `blocked`
- `Qualified documentation records the unchanged obligation and grants no scientific benefit.`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`
- `completed_count`: `1`; `censored_count`: `6`
- `device_command_count`: `0`

## RECOMMENDATION
KEEP

## experiment_8401_v723_capstone.json

**SKIPPED_ALREADY_FLAGGED**
