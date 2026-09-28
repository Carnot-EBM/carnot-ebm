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

## experiment_7798_view_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A comparative experimental result or model performance claim being asserted despite failing pre-execution gates, or runtime execution metrics appearing when the experiment was blocked.

## WAS THAT CHECKED
no, because the experiment was halted at conductor pre-gate prior to execution.

## EVIDENCE
`"schema"`
`"blocked_gate_check_v1"`
`"status"`
`"blocked"`
`"honest_verdict"`
`"blocked_gate_check_failed"`
`"duration_s"`
`0.0`
`"blocked_at_layer"`
`"conductor_pre_gate"`
`"gate_check_summary"`
`"gate-unsat(final): 5 of 7 gate(s) failed; first failure: exp7796-source-view-qualification.sentence_protocol_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_7800_v678_counter_evidence_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no positive scientific claim to refute. The artifact records a disqualified protocol and an unstarted evaluation.

## WAS THAT CHECKED
No. The evaluation rows were not run; only a fixture and validation checks were recorded.

## EVIDENCE
`honest_verdict`: `complete_disqualified_required_checks`; `verdict_class`: `disqualified`; `readiness`: `false`; `started`: `0`; `calls`: `0`; `unstarted_no_model_load`

## RECOMMENDATION
KEEP

## experiment_7801_qwen_counter_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None. The artifact is a gate-check receipt recording an execution block prior to test execution, and makes no comparative or empirical claim that could be refuted.

## WAS THAT CHECKED
No; the experiment was blocked prior to execution at `conductor_pre_gate`, so no experimental conditions were evaluated.

## EVIDENCE
`schema`: `blocked_gate_check_v1`
`status`: `blocked`
`honest_verdict`: `blocked_gate_check_failed`
`duration_s`: `0.0`
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7803_v678_arc_runner_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to refute. If this artifact asserted runner readiness, a failed required validation check would refute it.

## WAS THAT CHECKED
Yes. The required full Python suite was checked and failed. The planned comparison had no completed rows.

## EVIDENCE
`honest_verdict`: `complete_disqualified_required_runner_validation`; `full_python_suite`: `observed` `-15`, `passed` `false`; `readiness`: `false`; `decision_benefit`: `null`; `effective_independent_n`: `0`.

## RECOMMENDATION
KEEP

## experiment_7804_arc_organic_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no outcome claim to falsify. A future claim that the method improves organic selection would be refuted by scored live-path results showing no improvement over a serious baseline.

## WAS THAT CHECKED
No. The experiment stopped at the upstream gate before producing organic-selection results.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `organic_runner_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7806_v678_hardware_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this artifact asserts no comparative or performance advantage claim, there is no active hypothesis under test to falsify. If a claim of hardware acceleration or execution advantage were made, it would be refuted by observed end-to-end service latency or throughput showing no advantage over host CPU execution, or by non-zero transport and readout overhead eliminating candidate board speedup.

## WAS THAT CHECKED
No. The artifact did not execute workloads on any device or probe reachability, issued zero hardware operations, had no measured service timing data, and failed prerequisite validation checks before execution. It serves solely as a historical inventory and custody receipt.

## EVIDENCE
- `hardware_advantage_claimed`
- `false`
- `honest_verdict`
- `complete_disqualified_required_checks`
- `verdict_class`
- `disqualified`
- `terminal_scope`
- `historical_board_custody_only`
- `arm`
- `historical_accounting`
- `current_board_reachability`
- `not_probed`
- `service_measured`
- `required_checks_passed`

## RECOMMENDATION
KEEP

## experiment_7807_v678_independent_evidence_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The audit disqualifies independent evidence because required science sources are missing and required validation failed.

## WHAT WOULD REFUTE IT
Eligible current science sources and passing required validation alongside the disqualified verdict would contradict the audit.

## WAS THAT CHECKED
Yes. The artifact records the disposition of each required source and the results of the required validation commands. The refuting conditions did not appear.

## EVIDENCE
`Exp7799`, `Exp7801`, and `Exp7802` each have `state` `missing`; `independent_n` is `0`; `required_checks_passed` is `false`; `failed_required_commands` lists `focused_pytest`, `changed_module_coverage`, and `changed_module_coverage_report`; `honest_verdict` is `complete_disqualified_required_validation`.

## RECOMMENDATION
KEEP

## experiment_7808_v678_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The recorded blocked status would be contradicted by complete, eligible current evidence for the required V678 branches. The artifact makes no comparative claim about a method’s value.

## WAS THAT CHECKED
Yes, for the blocked status: the branch outcomes, gate checks, and task rows record missing or disqualified evidence. No method-value test is reported.

## EVIDENCE
`honest_verdict` = `complete_blocked_required_v678_evidence`; `actual_inference_substrate_class` = `aggregation`; `capstone_complete_score` = `0`; `decision_benefit` = `null`; `no eligible current scored organic SDK rows`; `no complete current service rows`; `current decision and learning science absent`.

## RECOMMENDATION
KEEP
