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

## experiment_8715_v747_verifier_completeness.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The extended verifier demonstrates added verification value, warranting a positive verifier-completeness verdict.

## WHAT WOULD REFUTE IT
Independent correctness judgments show that the ten additional detections are false alarms, or that the baseline detects the same genuine defects. Merely avoiding worse detection cannot establish value because the extended suite includes the baseline.

## WAS THAT CHECKED
No, for independent value. The artifact reports 130 versus 120 detections, so the reported baseline does not tie. However, the verifier itself defines correctness; labels independent of tested variants do not resolve that circularity. The count increase can stand as a constructed execution result, but cannot establish the verifier’s added value.

The suite costs eleven probes versus five and explicitly disclaims equal-budget superiority. Reserved worlds and no fitting provide no remedy for the oracle issue.

## EVIDENCE

- `honest_verdict`: `complete_circular_positive_v747_verifier_completeness`
- `verdict_class`: `circular_positive`
- `verifier_is_oracle`: `true`
- `baseline_detected_defect_count`: `120`
- `extended_detected_defect_count`: `130`
- `incremental_detected_defect_count`: `10`
- `Labels are independent of tested variants.`
- `Extended suite includes baseline probes and costs more; no equal-budget superiority claim.`

## RECOMMENDATION
NARROW_CLAIM

## experiment_8716_v747_installed_import_isolation.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Import isolation preserves the export assertions without increasing the measured parent peak memory, while maintaining installed-worker parity on the fixed deployment corpus.

## WHAT WOULD REFUTE IT
Failure of the unchanged assertions in the isolated child, an unreaped child, increased parent peak memory, or an installed-worker output mismatch against the reference implementation. These outcomes were possible; correctness of packaging and process isolation does not follow automatically from using the verifier as oracle.

## WAS THAT CHECKED
Yes. The original guard failure was reproduced, then identical assertions passed in a reaped child with unchanged parent peak memory. The artifact reports 2,048 completed parity evaluations and zero failures.

Python/native agreement tests implementation preservation; it does not demonstrate added verifier value. The exposed corpus is explicitly identified, and generalization scores are zero. The displayed rows show no invalid measurement being counted as valid; expected rejection errors and the reproduced guard failure are intentional controls.

## EVIDENCE

- `guard_reproduction`: `assertions_identical` = `true`; `observed_delta_mb` = `579`; `actual_exit` = `1`; `expected_exit` = `1`.
- `memory_lifecycle_rows`: `assertions_passed` = `true`; `actual_exit` = `0`; `reaped` = `true`; `parent_peak_before_kib` = `515744`; `parent_peak_after_kib` = `515744`.
- `intended_count` = `2048`; `completed_count` = `2048`; `failed_count` = `0`.
- `exposure_scope` = `original exposed fixed deployment corpus`.
- `verifier_is_oracle` = `true`; `generalized_learning_benefit_score` = `0`; `independent_generalization_score` = `0`.
- `batch_service_ready_score` = `1`; displayed `parity_rows` have `passed` = `true`, `excluded` = `false`, and `censored` = `false`.

## RECOMMENDATION
KEEP

## experiment_8717_v747_installed_batch_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Native B16 batching improves complete-process throughput over native B1 on the fixed exposed workload.

## WHAT WOULD REFUTE IT
B16 taking at least as long as B1 for the same completed work, or batching changing the computed outputs.

## WAS THAT CHECKED
Yes. Randomized paired process measurements include setup through shutdown, with output-parity checks. Across ten paired repeats, the batching speedup is 1.1382×, with a 95% interval entirely above 1.

The serious comparator, equal-batch Python, is statistically indistinguishable from native at B16. That limits attribution to native implementation advantages; it does not contradict the within-native batching benefit. The artifact explicitly records failure of the separate 10× target. Exposed requests support this timing comparison without establishing generalization.

## EVIDENCE

- `honest_verdict`: `complete_positive_v747_installed_batch_cost`
- `exposure_scope`: `fixed exposed original64 requests; timings only`
- `inference_boundary`: `launch through clean shutdown; setup included; offline environment installation reported separately`
- `contrast`: `batching`; `estimate`: `1.1382345321809237`; `lower95`: `1.1218473605571395`; `upper95`: `1.1600777873719108`; `paired_repeats`: `10`
- `parity_rows`: `passed`: `true`; `probability_error`: `0.0`
- `contrast`: `native_python_B16`; `estimate`: `1.0053949467902554`; `lower95`: `0.998091298984346`; `upper95`: `1.0138702202694783`
- `NFR-01_same_batch_B16`: `passed`: `false`; `same_batch_10x_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8718_qwen_symbolic_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no claimed comparative advantage to falsify; this artifact records a blocked prerequisite check.

## WAS THAT CHECKED
No comparison ran. In gates_evaluated, runtime readiness was 0 against the required 1, blocking execution. Symbolic-control readiness passed, but that establishes no comparative result.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `duration_s`: `0.0`; `failed_field`: `runtime_repair_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8719_v747_compiled_sparse_update.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The compiled sparse update matches its reference on the tested lifecycle inputs, supporting CPU port readiness within the stated domain.

## WHAT WOULD REFUTE IT
Any native/reference disagreement in integer state, historical outputs, counters, rejection behavior, or resumed checkpoint bytes on an included test case would refute exact parity.

## WAS THAT CHECKED
Yes—in the native-parity, boundary, rejection, and restart rows. Displayed checks pass, and the summary reports zero failures, exclusions, or censored cases.

Comparing separately executed native and Python implementations gives implementation errors a real chance to surface. Python is the relevant serious baseline; a tie establishes the claimed equivalence.

The verifier defines reference conformance here. That supports an execution claim without establishing verifier added value. The artifact explicitly confines its claim to exposed-input language-port parity and reports zero generalization benefit. No displayed validity flag or missing-key contradiction undermines that claim.

## EVIDENCE

- `acceptance_gates`: `exact_native_parity`: `true`; `loaded_extension`: `true`.
- `native_parity_rows` and `boundary_rows`: `integer_state_identical`: `true`; `historical_identical`: `true`; `counters_identical`: `true`.
- `rejection_rows`: `native` and `python` each report `value`: `-4` in the first displayed comparison.
- `restart_rows`: `state_acknowledgement_and_bytes_identical`: `true`.
- `failed_count`: `0`; `excluded_count`: `0`; `censored_count`: `0`.
- `exposure_scope`: `same_original_exposed_lifecycle_inputs_language_port_parity_only`.
- `verifier_is_oracle`: `true`; `independent_generalization_score`: `0`; `speedup_claim`: `null`.
- `Bind this field to sealed exact parity evidence; CPU port readiness grants no learning, hardware or throughput claim.`

## RECOMMENDATION
KEEP

## experiment_8720_v747_arc_supervisor_frontier.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A newly authenticated supervisor event outside the retained closed frontier would contradict the empty-delta receipt. The artifact makes no comparative claim about supervisor effectiveness.

## WAS THAT CHECKED
Yes, for the receipt: the methodology specifies exact event-identity comparison, the reported before/after frontiers match, and the event delta is empty. No new episodes were run; this records an absence of new evidence, rather than an experimental finding of no benefit.

## EVIDENCE

- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `honest_verdict`: `complete_null_no_new_supervisor_outcomes`
- `new_event_rows`: `[]`
- `rows`: `[]`
- `new_events`: `0`; `new_live_episodes`: `0`
- `decision`: `retain_unchanged`; `usable_cross_game_support`: `0`
- `no_policy_change`: `true`; `new_level_credit`: `0`

## RECOMMENDATION
KEEP

## experiment_8721_v747_hardware_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. Authenticated compatible physical execution with precision parity and complete learning-cost measurements would overturn the blocked readiness status; those observations alone would not establish acceleration.

## WAS THAT CHECKED
Yes, for evidence availability: source-admission checks explicitly identify the missing, unqualified learning-cost source, and the physical-execution gate observes zero. The missing source remains censored. This is a qualification report, with no claimed measured speedup, generalization benefit, or verifier added value.

## EVIDENCE

- `honest_verdict`: `complete_blocked_missing_complete_cost_or_physical_execution`
- `verdict_class`: `blocked`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `operand_state`: `missing_file`
- `qualified`: `false`
- `status`: `censored`
- `compatible_physical_execution`; `operand_state`: `observed_zero`
- `achieved_speedup`: `null`
- `interpretation`: `Conditional1/(1-f); neither observed speed nor a100x complete-learning result`

## RECOMMENDATION
KEEP

## experiment_8722_v747_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V747 capstone remains blocked because required evidence is insufficient to authorize activation.

## WHAT WOULD REFUTE IT
Qualified measurements satisfying all outstanding acceptance requirements—including independent semantics, future loss and retention, physical execution, and same-batch 10× performance—would contradict the blocked conclusion.

## WAS THAT CHECKED
Yes, as an evidence-sufficiency decision. The acceptance gates and gate-edge rows record failed requirements and distinguish missing evidence from observed zero. The deployment comparison also falls short of 10×. Disqualified production is explicitly marked scientifically unmeasured.

This supports withholding activation; it does not establish that unexecuted studies would find no benefit.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v747_capstone`; `activation_authorized`: `false`.
- `future_loss_and_retention`: `false`; `independent_semantics`: `false`; `physical_execution`: `false`; `same_batch_10x`: `false`.
- `field`: `learning_authority_ready_score`; `expected`: `1`; `observed`: `0`; `passed`: `false`.
- `operand_state`: `missing_file`; `observed`: `null`.
- `producer_qualified`: `false`; `scientific_result_measured`: `false`.
- `same_batch_threshold`: `paired lower95 >= 10`; `contrast`: `native_python_B16`; `upper95`: `1.0138702202694783`.

## RECOMMENDATION
KEEP
