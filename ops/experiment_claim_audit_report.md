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

## experiment_7331_learning_adapter.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifiable outcome claim is made; this is a blocked gate-check receipt.

## WAS THAT CHECKED
No. The experiment did not run because both prerequisite gates failed.

## EVIDENCE
`status` `blocked` `honest_verdict` `blocked_gate_check_failed` `gate-unsat(final): 2 of 2 gate(s) failed; first failure: exp7330-executor-isolation.executor_fixture_ready_score (actual=0 == expected=1)` `blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7332_plan_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies because the artifact reports a blocked pre-gate check, not an experimental result or comparative claim.

## WAS THAT CHECKED
No; the experiment was blocked at `conductor_pre_gate`, so the method and any substantive success criterion were never evaluated.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"gate_check_summary": "gate-unsat(final): 2 of 2 gate(s) failed; first failure: exp7330-executor-isolation.executor_fixture_ready_score (actual=0 == expected=1)"`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7336_v644_arc_resume.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a disqualified mechanism receipt rather than an efficacy, readiness, generalization, or verifier-value claim.

## WAS THAT CHECKED
Not applicable; no positive comparative claim was advanced for refutation.

## EVIDENCE
`honest_verdict` `complete_disqualified_affected_validation_failed` `verdict_class` `disqualified` `status` `disqualified` `promotion_value` `0` `model_invoked` `false` `solve_provenance` `no_game_solve_cpu_transport_fixture`

## RECOMMENDATION
KEEP

## experiment_7337_arc_transfer.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no substantive performance or transfer claim to falsify; this is only a blocked-run receipt.

## WAS THAT CHECKED
No; the experiment stopped at the pre-gate, so no transfer result or comparator was evaluated.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"blocked_at_layer": "conductor_pre_gate"`, `"passed": false`, `"actual": 0`, `"expected": 1`

## RECOMMENDATION
KEEP

## experiment_7339_v644_native_binding.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The imported native binding exactly matches the reference schedule evaluator and completes the fixed boundary protocol, without asserting speed or learning value.

## WHAT WOULD REFUTE IT
A valid parity case with different Python and Rust outputs, any nonzero parity mismatch, use of a fallback instead of the imported extension, or missing/censored protocol blocks would refute the claim.

## WAS THAT CHECKED
Yes. The artifact reports an actual imported-extension replay, exact comparison across captured and adverse cases, mutation checks, and all 270 planned timing rows across 90 paired blocks. Although the verifier is the oracle, the claim is narrowly about execution-grounded parity—not independent correctness or added value.

## EVIDENCE
`"verifier_is_oracle": true`; `"verdict_class": "circular_positive"`; `"mismatches": 0`; `"all_matched": true`; `"rows": 3308`; `"actual imported extension round trip with zero differences"`; `"python_fallback_used": false`; `"compiled_execution": true`; `"cost_rows_completed": 270`; `"cost_rows_censored": 0`; `"paired_blocks": 90`; `"speed_claimed": false`; `"learning_value_claimed": false`; `"claim_boundary": "prototype parity and readiness only; Exp7340 owns speed inference"`

## RECOMMENDATION
KEEP

## experiment_7340_v644_native_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The native evaluator completed with exact parity but failed the predeclared ten-times-throughput lower-bound gate at every tested batch size.

## WHAT WOULD REFUTE IT
CI95 lower bounds of at least 10 for native-over-Python throughput at batch sizes 1, 32, and 256 would refute the performance-null claim.

## WAS THAT CHECKED
Yes. The fixed native-versus-Python comparison was evaluated over 30 paired blocks at each size, with the protocol sealed before timing and no outcome-based extension. All observed CI95 lower bounds were below 10. The oracle defines correctness parity, but the artifact makes no positive claim about the verifier’s added value.

## EVIDENCE
`"expected": "CI95 lower bound >= 10 at sizes 1, 32, and 256"`; `"passed": false`; `"1": 1.9070958551116404`; `"32": 1.684761623050079`; `"256": 1.660640300423487`; `"randomized_paired_blocks_each": 30`; `"sealed_before_timing": true`; `"stopping_rule": "exactly 30 paired blocks for each fixed size; no outcome extension"`; `"parity_mismatches": 0`; `"native_ten_x_score": 0`; `"verdict_class": "null"`

## RECOMMENDATION
KEEP

## experiment_7341_v644_board_continuity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
GateMate remains blocked because no qualifying post-Exp6559 physical-state receipt exists, while three board dispositions are complete and no hardware readiness, promotion, or operations are claimed.

## WHAT WOULD REFUTE IT
A qualifying operator-authored receipt dated after 20260823 documenting a GateMate cable, port, power, board, JTAG, or DirtyJTAG change would refute the blocking claim; nonzero readiness, promotion, or issued operations would contradict the remaining headline assertions.

## WAS THAT CHECKED
Yes. The artifact applies an explicit receipt contract, reports the receipt search result and accepted count, includes a GateMate row that could have recorded qualifying evidence, and separately records readiness, promotion, and operation counts. The oracle warning does not create circularity here because the artifact makes a receipt-status claim, not a positive claim about the verifier’s added value.

## EVIDENCE
`"gatemate_changed_physical_state_receipt"`; `"accepted_receipt_count": 0`; `"exists": false`; `"newer_than_exp6559": false`; `"operator_changed_conditions": {}`; `"terminal_criterion_met": false`; `"board_disposition_complete_score": 1`; `"hardware_readiness_score": 0`; `"hardware_promotion_score": 0`; `"hardware_operations_issued_count": 0`; `"status": "blocked"`; `"verdict_class": "blocked"`

## RECOMMENDATION
KEEP

## experiment_7342_v644_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative efficacy claim to falsify. Treating the administrative headline as an assertion, it would be contradicted by fewer than fourteen dispositions, available current science, or a qualifying current board-state receipt.

## WAS THAT CHECKED
No comparative claim was checked. The administrative assertions were checked through the fourteen-disposition count, the required-science availability gate, and the board-receipt gate.

## EVIDENCE
`inference_substrate` `aggregation_from_upstream_artifacts` `model_invoked` `false` `capstone_value_score` `0` `promotes_scientific_efficacy` `false` `fourteen_dispositions` `14` `required_science_available` `false` `gatemate_changed_physical_state_receipt_missing` `verdict_class` `blocked`

## RECOMMENDATION
KEEP
