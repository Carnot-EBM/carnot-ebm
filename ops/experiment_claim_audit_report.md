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
| CLAIM_SUPPORTED | 2 |
| NO_CLAIM | 6 |

## experiment_8362_v721_threshold_guard.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This run produced zero certified fast paths and therefore demonstrated no certified usefulness while the numerical-policy proof remained unresolved.

## WHAT WOULD REFUTE IT
A counted vector actually taking a certified fast path would contradict the zero-use claim; empirical eligibility alone would not.

## WAS THAT CHECKED
Yes, at aggregate level: all 4,218 completed vectors are accounted for by fallback reasons, and actual fast-path coverage is zero. Per-vector execution and validity rows are absent from the supplied excerpt.

The unresolved proof forces fallback, so zero action mismatches cannot establish added value over always evaluating the direct reference—the cheapest serious baseline. The artifact reports an honest operational null; its positive empirical candidate coverage does not become a certification or generalization claim.

## EVIDENCE

- `fast_path_fraction`: `0.0`
- `completed_count`: `4218`
- `fallback_counts`: `threshold_intersection`: `10`; `unproven_numerical_policy`: `4208`
- `numerical_policy_proof_passed`: `false`
- `empirical_candidate_fast_path_fraction`: `1.0`
- `guarded_action_mismatch_count`: `0`
- `useful`: `false`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8363_atomic_table_state.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the title describes intended work, and the artifact records a blocked prerequisite rather than claiming success.

## WAS THAT CHECKED
The prerequisite was checked in gates_evaluated: actual 0 versus expected 1, with passed false. Atomic publication and recovery were not evaluated.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `blocked_at_layer`: `conductor_pre_gate`; `artifact_field`: `guard_ready_score`; `actual`: `0`; `expected`: `1`; `passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_8368_v721_typed_runtime_closure.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An authenticated difference between current and previous causal environment operands would contradict the recorded unchanged disposition. The artifact makes no comparative benefit claim to falsify.

## WAS THAT CHECKED
Yes, for environment identity: seven change-evidence rows compare observed and previous operands and report no change. Current CUDA functionality was not tested, so this receipt does not establish continued CUDA failure.

## EVIDENCE
- `honest_verdict`: `complete_blocked_cuda_environment_unchanged`
- `change_evidence`, `available`: `true`, `changed`: `false`
- `current_probe_count`: `0`
- `methodology_note`: `Read full versioned authority, authenticate failed historical execution and compare current causal identity bytes. No unchanged CUDA matrix or scientific measurement runs.`
- `Retain exact invocation, byte custody and measured execution; no scientific benefit follows.`
- `root_cause_status`: `unproved`

## RECOMMENDATION
KEEP

## experiment_8369_changed_runtime_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental headline to falsify. All gate rows passing would contradict this receipt’s reported gate failure.

## WAS THAT CHECKED
Yes. The gate rows record one passing prerequisite and two failing prerequisites, consistent with the reported block. No canary outcome or comparative value is claimed.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `gate_check_summary`: `gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp8368-typed-runtime-closure.runtime_changed_score (actual=0 == expected=1)`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8370_v721_arc_outcome_delta.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. An authenticated new post-Exp8355 environment outcome would contradict the receipt’s reported no-change status.

## WAS THAT CHECKED
Authentication and inspection are recorded for the receipt. No comparative refutation was tested; the artifact contains no outcome or comparison rows.

## EVIDENCE
- `honest_verdict`: `complete_null_no_supervisor_outcomes`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `phase`: `authenticate_and_inspect`
- `new_outcome_count`: `0`
- `rows`: `[]`
- `per_game_arm_rows`: `[]`
- `selection_recommendations`: `[]`
- `solve_credit_claimed`: `false`

## RECOMMENDATION
KEEP

## experiment_8371_v721_hardware_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A valid current hardware execution with a compatible kernel, complete transport timing, and measured service benefit would contradict the recorded disqualification. No comparative scientific claim is asserted.

## WAS THAT CHECKED
No current hardware trial is shown. This artifact aggregates historical evidence and records unmet qualification gates; its unmeasured benefit is not an experimental finding of zero benefit.

## EVIDENCE
- `actual_substrate`: `host_CPU_aggregation`
- `current_device_execution_count`: `0`
- `compatible_fraction_status`: `unmeasured`
- `full_service_speedup`: `null`
- `acquisition_relevance`: `defer_no_measured_compatible_service_benefit_no_V721_acquisition`
- `scope`: `historical_only_zero_current_calls`

## RECOMMENDATION
KEEP

## experiment_8372_v721_gatemate_missing_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a receipt recording an obligation and unresolved evidence, without claiming comparative value, execution readiness, or generalization.

## WAS THAT CHECKED
No comparative refutation test is reported. The rows show one completed read-only obligation and three censored source/hardware items, consistent with the blocked verdict. The oracle verifier supports bookkeeping here; no added-value claim is made. Zero generalization scores make no generalization assertion, and the censored rows are counted separately from completion.

## EVIDENCE
- `honest_verdict`: `complete_blocked_gatemate_missing_evidence`
- `verdict_class`: `blocked`
- `obligation_recorded_score`: `1`
- `completed_count`: `1`; `censored_count`: `3`
- `history_authenticated`: `false`; `hardware_execution`: `false`
- `scientific_benefit`: `false`
- `independent_generalization_score`: `0`
- `generalized_learning_benefit_score`: `0`
- `no_model_load`: `true`

## RECOMMENDATION
KEEP

## experiment_8373_v721_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The exposed development comparisons demonstrate no decision-cost benefit from spline34 over RBF34 or from online updating over frozen baselines.

## WHAT WOULD REFUTE IT
A positive paired decision-cost gain for either method over its serious comparator, with a lower uncertainty bound above zero and the gain surviving accounting for unqualified rows.

## WAS THAT CHECKED
Yes, in H1’s paired-cost/bootstrap summaries and H2’s paired-cost/block-bootstrap summaries. H1 reports negative gain; H2 reports zero gain despite changed probabilities. The comparisons could register improvement.

The oracle defines the measured costs, but the headline claims no verifier added value. The artifact explicitly labels the data exposed and descriptive. Cheap serious controls are present: linear6 ties spline34’s decision cost, while frozen and calibration-only arms tie online updating. These ties support the reported null.

Unqualified rows are disclosed with explicit costs and bounds, and qualified denominators are reported separately. No wrong-key error is evident in the supplied text. H1 leaves representation-level conclusions unqualified.

## EVIDENCE

- `H1`: `science_disposition` = `null_optimization_limited`; `mean_gain` = `-0.00390625` under `all_intended` and `-0.005154639175257732` under `complete_case`; `procedure_null_informative` = `true`; `geometry_qualified` = `false`.
- `H2`: `science_disposition` = `null_delayed_decision_benefit`; `mean_gain` = `0.0`; `online_sparse` has `probability_changes` = `66` and `action_changes` = `0`.
- `frozen_spline`, `online_sparse`, and `calibration_only`: `cost_mean` = `0.4431818181818182`.
- `scope` = `descriptive_exposed_development_not_confirmatory`; `semantic_benefit` = `false`; `whole_service_benefit` = `false`.
- H2’s `missing_bounds`: `qualified` = `false`; `frozen_cost` = `0.5`; `online_cost` = `0.5`; `gain_lower` = `0.0`; `gain_upper` = `0.0`.

## RECOMMENDATION
KEEP
