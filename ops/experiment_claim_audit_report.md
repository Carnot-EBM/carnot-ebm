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
| CLAIM_SUPPORTED | 1 |
| NO_CLAIM | 7 |

## experiment_8556_causal_brier_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-execution receipt, with no experimental success or comparative claim to falsify.

## WAS THAT CHECKED
No experimental claim was tested. Only prerequisite gates were evaluated; the adapter-readiness gate failed before the experiment ran.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `online_adapter_ready_score`
- `failed_expected`: `1`; `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8558_v735_nine_input_native_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is an engineering qualification receipt without a comparative benefit claim. Its limited qualification would fail if a non-fallback native residual exceeded the stated tolerance, a native decision disagreed with the reference, or required validation failed.

## WAS THAT CHECKED
Yes, for qualification: the artifact reports 1,180 finite parity comparisons, explicit boundary fallbacks, complete-request comparisons, and passing acceptance gates. These checks could expose implementation failures. Agreement with the reference oracle establishes fidelity; the artifact assigns zero independent generalization and learning benefit.

## EVIDENCE

- `nine_input_native_ready_score`: `One only after native calls, finite controls, complete request consumer and all required validation pass; no speed or learning claim.`
- `max_residual_error`: `Maximum absolute dense-SciPy versus native residual difference; every row must satisfy 1e-12*(1+abs(reference)).`
- `finite_parity_rows`: `1180`; `failed_count`: `0`.
- `generalized_learning_benefit_score`: `0`; `independent_generalization_score`: `0`.
- `exposure_scope`: `all 900 answers and 300 questions are exposed development history`.

## RECOMMENDATION
KEEP

## experiment_8559_v735_complete_native_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The primary batch-1 complete-cost comparison does not establish a native throughput advantage over Python.

## WHAT WOULD REFUTE IT
A primary batch-1 throughput confidence interval entirely above 1, based on complete measured costs and valid paired runs, would contradict this scoped null.

## WAS THAT CHECKED
Yes. Paired native/Python wall times and bootstrap intervals directly test throughput; neither arm was guaranteed to win. The primary interval crosses 1.

Python is a serious comparator. Batch-8 throughput and p95 latency show improvements, so the null must remain scoped to primary throughput. Oracle status and exposed inputs do not make this timing comparison circular; no independent generalization benefit is claimed. No exclusions or censoring are reported, and the cited timing fields exist.

## EVIDENCE
- `honest_verdict`: `complete_null_v735_complete_native_cost`
- `batch`: `1`; `complete_throughput_ratio`: `1.0101493212248978`; `throughput_ratio_ci95`: `0.9974283026343478`, `1.0234577501521551`
- `batch`: `8`; `throughput_ratio_ci95`: `1.0069912996713701`, `1.0202474770692178`
- `native_s`: `1.63813082`; `python_s`: `1.633614336`
- `independent_generalization_score`: `0`
- `excluded_count`: `0`; `censored_count`: `0`; `failed_count`: `0`

## RECOMMENDATION
KEEP

## experiment_8560_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental claim to falsify. This artifact records a failed prerequisite and blocked execution, without claiming canary success or comparative value.

## WAS THAT CHECKED
No experimental claim was tested. The prerequisite was checked in `gates_evaluated`: runtime readiness was 0 against a required 1, and the gate failed.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `runtime_repair_ready_score`
- `failed_expected`: `1`; `failed_observed`: `0`
- `duration_s`: `0.0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8561_v735_arc_panel_replay_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. Failed owned coverage or terminal replay would defeat infrastructure qualification; those failures would not refute an artifact that already reports disqualification.

## WAS THAT CHECKED
Yes. The acceptance gates report both failures, execution readiness is zero, and the verdict remains disqualified. Replay completion and fixture parity are not presented as scientific benefit or generalization.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_v735_arc_panel_replay_qualification`
- `no_scientific_benefit_claim`: `true`
- `owned_coverage`: `false`
- `terminal_replay`: `false`
- `arc_panel_execution_ready_score`: `0`
- `headline_solve_credit`: `0`
- `independent_count`: `0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8562_arc_capability_panel.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-run receipt, not a claim about capability or comparative value.

## WAS THAT CHECKED
No capability evaluation occurred. The prerequisite readiness gate was checked and failed: observed 0, required 1. The artifact reports the block without claiming success.

## EVIDENCE
`blocked_gate_check_v1`, `blocked`, `blocked_gate_check_failed`, `arc_panel_execution_ready_score`, `actual=0 == expected=1`, `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8563_v735_hardware_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit is asserted. A qualified compatible device execution with parity and complete host/device clocks would contradict the recorded blocked hardware boundary.

## WAS THAT CHECKED
No physical execution was tested in this invocation. The artifact records compatibility checks and host-only operation costs, while explicitly treating acceleration ceilings as conditional. This is an engineering accounting receipt, not a positive performance or generalization result.

## EVIDENCE

- `exposure_scope`: `historical_exposed_evidence_engineering_accounting_only`
- `One qualifies scope accounting only.`
- `cost_scope`: `Python batch1 ten complete repeats; no device execution`
- `deployment_ready`: `false`
- `device_commands_executed`: `[]`
- `conditional`: `true`
- `measured_acceleration`: `null`

## RECOMMENDATION
KEEP

## experiment_8564_v735_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. An unaccounted scheduling slot or failed required owned check would contradict administrative completeness; blocked scientific work does not.

## WAS THAT CHECKED
Yes, for administrative accounting: the acceptance gates report complete slot accounting and required checks. Failed scientific prerequisites remain recorded as blocked or skipped. Oracle verification is not presented as evidence of added value, and ineligible scheduling receipts are not presented as scientific successes.

## EVIDENCE

- `administrative_complete`: `true`
- `all_slots_accounted`: `true`
- `required_owned_checks`: `true`
- `scientific_claims_independently_generalize`: `false`
- `producer_executed`: `false`; `eligible`: `false`
- `disposition`: `pre_gate_receipt`
- `disposition`: `cascade_skip`
- `Stored ARC replay checks execution mechanics and grants no hidden-game capability benefit.`

## RECOMMENDATION
KEEP
