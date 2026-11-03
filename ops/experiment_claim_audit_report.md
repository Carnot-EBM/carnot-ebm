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
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8768_v751_evidence_set_methods.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8769_evidence_set_transfer.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is made. This artifact records a blocked run, so there is no claimed method advantage to falsify.

## WAS THAT CHECKED
No—the intended comparison was blocked. Only the prerequisite gate was checked; it failed with observed 0 versus required 1.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `actual`: `0`
- `expected`: `1`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8770_v751_optimized_cost_attribution.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The diagnostic completed but failed low-overhead qualification, leaving no supported next optimization target.

## WHAT WOULD REFUTE IT
Valid, reconciled attribution with observer overhead meeting the registered 5% limit on both backends would contradict the stated reason for withholding qualification.

## WAS THAT CHECKED
Yes. Three instrumented/uninstrumented process pairs per backend provide a direct control on the same workload. Python’s overhead estimate is 5.84%; observed maxima reach 8.02% for Python and 5.29% for native. Qualification could have passed, but the measured overhead prevented it, and the artifact reports a null.

This supports the scoped qualification failure, not a conclusion that optimization is impossible. No verifier-value, generalization, or new speed claim is promoted.

## EVIDENCE

- `honest_verdict`: `complete_null_optimized_cost_attribution`
- `overhead_limit`: `0.05`
- `observer_overhead_rows`: `python`, `estimate`: `0.05836651124330672`, `maximum`: `0.08017429961098888`
- `observer_overhead_rows`: `native`, `estimate`: `0.043936580067839515`, `maximum`: `0.05287655599140395`
- `paired_repeats`: `3`
- `name`: `low_observer_overhead`, `passed`: `false`
- `reason`: `No supported target without complete low-overhead attribution.`
- `selected_target`: `null`
- `same_batch_10x_claim_this_invocation`: `false`

## RECOMMENDATION
KEEP

## experiment_8771_v751_lifecycle_archive_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative value claim is made. The execution qualification would fail on a state or acknowledgment mismatch, an incorrect dense-write count, a changed restart checkpoint, or replay of an acknowledged update.

## WAS THAT CHECKED
Yes, for execution qualification: the displayed ordinary and threshold rows compare expected and observed results, and all three restart arms check checkpoint identity and acknowledged-update replay. These checks could fail; agreement with the oracle establishes conformance. The Python arm ties the other arms on the displayed checks, consistent with the stated parity scope. No verifier benefit or generalization claim follows.

The historical failure remains explicitly disqualified, and the full-suite timeout is accompanied by an explicit denial of a full-suite pass.

## EVIDENCE

- `scope`: `fixed-point replay of exposed historical labels; no learning benefit evaluated`
- `complete_state_probability_touched_coefficients_ack_identical`: `true`
- `dense_writes`: `109`; `expected_dense_writes`: `109`
- `checkpoint_identical`: `true`; `acknowledged_update_replayed`: `false`
- `old_primary_remains_disqualified`: `true`
- `full_suite_pass_claimed`: `false`

## RECOMMENDATION
KEEP

## experiment_8772_v751_ordinary_durable_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bounded ordinary durable study did not establish a resident-native performance benefit over the original native implementation.

## WHAT WOULD REFUTE IT
Valid paired measurements against json_native showing a complete-throughput ratio with a lower 95% bound above 1.05, together with no more than 5% p95 latency regression, would contradict the reported null.

## WAS THAT CHECKED
Yes. The paired repeat comparison against json_native reports a lower bound of 0.9441418639991097 and a p95 latency ratio of 2.355170021011086, failing both conditions. Measured execution times could have satisfied those conditions; the null was not predetermined.

The serious baseline is present. Oracle-defined correctness does not make the timing comparison circular, and no generalization benefit is claimed. The artifact reports 90 completed windows with zero failures, censoring, or exclusions. The separate repository-health timeout is explicitly accompanied by no full-suite pass claim.

## EVIDENCE
- `honest_verdict`: `complete_null_ordinary_durable_cost`
- `benefit_lower95_gt`: `1.05`; `maximum_p95_regression`: `0.05`
- `paired_repeat_intervals` → `json_native`: `lower95`: `0.9441418639991097`; `p95_ratio`: `2.355170021011086`
- `bounded_durable_benefit`: `false`
- `completed_window_count`: `90`; `failed_count`: `0`; `censored_count`: `0`; `excluded_count`: `0`
- `independent_generalization_score`: `0`; `full_suite_pass_claimed`: `false`

## RECOMMENDATION
KEEP

## experiment_8773_arc_outcome_frontier.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite, without claiming that supervisor refinement succeeds or adds value.

## WAS THAT CHECKED
No refinement comparison occurred. The sole prerequisite gate was checked and failed: its observed value was 0 against a required value of 1.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `arc_new_outcomes_available_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8774_v751_hardware_operation_decision.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit claim is asserted. Authenticated compatible execution with complete service costs could challenge the blocked readiness decision; unknown performance does not establish a negative performance result.

## WAS THAT CHECKED
Prerequisite qualification was checked: the artifact records missing input, zero readiness, and rejected producer identity. It reports no new physical execution. This is an operational aggregation receipt, not a verifier-value or acceleration result.

## EVIDENCE
- `honest_verdict`: `complete_blocked_hardware_operation_decision`
- `device_commands_executed`: `[]`
- `physical_execution_this_invocation`: `false`
- `achieved_speedup`: `null`
- `measured_acceleration`: `false`
- `operand_state`: `missing_file`, `observed_zero`, `rejected_producer`
- `admitted`: `false`
- `eligible_for_bound`: `false`

## RECOMMENDATION
KEEP

## experiment_8775_v751_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The capstone remains disqualified, while reporting a modest batching benefit without establishing tenfold same-batch acceleration or independent generalization.

## WHAT WOULD REFUTE IT
A batching lower confidence bound at or below 1.05, or native B16 p95 above 50 ms, would defeat the positive batching conclusion. Fully qualified evidence satisfying the required gates would contradict disqualification.

## WAS THAT CHECKED
Yes, at the reported aggregation level. The batching lower bound is 1.12185 and p95 is 18.42 ms. Qualification checks record actual failures, including failed owned validation.

The native/Python B16 comparison is effectively tied; the artifact assigns zero tenfold-speedup credit. It explicitly acknowledges exposure and assigns zero independent-generalization credit. Unqualified rows remain in the accounting with zero success numerators; their presence does not establish success. This supports disqualification, not a scientific conclusion that benefit is impossible.

## EVIDENCE
- `honest_verdict`: `complete_disqualified_v751_capstone`
- `owned_validation_passed`: `false`
- `activation_authorized`: `false`
- `batching_threshold`: `paired lower95 > 1.05 and native B16 p95 <= 50ms`
- `contrast`: `batching`; `lower95`: `1.1218473605571395`
- `native_B16_steady_p95_ms`: `18.42018975`
- `contrast`: `native_python_B16`; `estimate`: `1.0053949467902554`
- `same_batch_10x_score`: `0`
- `exposure_scope`: `Constructed and exposed upstream evidence; no natural independent generalization`
- `producer_qualified`: `false`; `raw_numerator`: `0`

## RECOMMENDATION
KEEP
