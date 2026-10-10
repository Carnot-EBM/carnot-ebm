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

## experiment_8352_v720_spline_table_fidelity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The selected 65-point float64 linear table meets the frozen fidelity gates on the constructed controls and runs faster than the measured direct spline implementation.

## WHAT WOULD REFUTE IT
The selected table exceeding maximum probability error 0.00015, producing any action flip where the direct top-two margin exceeds 0.0002, or taking at least as long as direct evaluation would refute the claim.

## WAS THAT CHECKED
Yes. The selected configuration reports error 0.0001302382749995834, zero far flips, and median evaluation times of 10720 versus 108885 nanoseconds. Rejected configurations demonstrate that fidelity failure could occur: the displayed nearest-neighbor configurations exceed the error gate and fail.

The direct spline is the defining fidelity oracle and a substantive timing comparator. This supports the measured approximation claim; the artifact makes no verifier-added-value claim and assigns zero to generalization and learning benefit. Failed candidate gates are retained as configuration outcomes. The fields supporting the selected candidate exist in its displayed row.

## EVIDENCE
- `selected_candidate`: `65-float64-linear`; `table_candidate_score`: `1`.
- `Only a measured table meeting both frozen gates is a candidate.`
- `65-float64-linear`: `candidate_passed` is `true`; `probability_error_max` is `0.0001302382749995834`; `far_flip_count` is `0`.
- `direct_ns_per_vector_median`: `108885.0`; `table_ns_per_vector_median`: `10720.0`.
- `65-float64-nearest`: `candidate_passed` is `false`; `probability_error_max` is `0.0027465688056047544`.
- `verifier_is_oracle`: `true`.
- `exposure_scope`: `constructed_controls_from_exposed_development_coefficients`.
- `independent_generalization_score`: `0`; `generalized_learning_benefit_score`: `0`.

## RECOMMENDATION
KEEP

## experiment_8353_v720_runtime_reader_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. An authenticated difference between previous and observed runtime identities would contradict the narrower recorded no-change observation.

## WAS THAT CHECKED
Yes, for that descriptive observation: seven change-evidence rows compare previous and observed identities and report no change. Failed qualification checks remain failures in the final disposition. This is a disqualified execution receipt, with no asserted scientific benefit or generalization.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_owned_checks`
- `required_checks_passed`: `false`
- `acceptance_gates`: `authenticated_change`: `false`, `cuda_context_ready`: `false`, `reader`: `false`
- `change_evidence` rows: `available`: `true`, `changed`: `false`
- `methodology_note`: `Read full versioned authority, authenticate failed historical execution and compare current causal identity bytes. No unchanged CUDA matrix or scientific measurement runs.`
- `Retain exact invocation, byte custody and measured execution; no scientific benefit follows.`

## RECOMMENDATION
KEEP

## experiment_8354_changed_runtime_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental claim to falsify: this artifact records a blocked prerequisite check, not a canary outcome.

## WAS THAT CHECKED
No canary claim was tested. The prerequisite checks appear in `gates_evaluated`; all three failed, blocking execution before the experiment.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `gates_evaluated`: each gate has `expected`: `1`, `actual`: `0`, and `passed`: `false`.
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8355_v720_arc_supervisor_frontier.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is made. A qualifying authenticated new supervisor outcome in the inspected sources would contradict the reported no-outcome status.

## WAS THAT CHECKED
No comparative test was performed. The recorded phase authenticates and inspects sources; outcome and arm tables are empty. This is a frontier-status receipt, with no asserted supervisor advantage or generalization benefit.

## EVIDENCE
- `honest_verdict`: `complete_null_no_supervisor_outcomes`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `phase`: `authenticate_and_inspect`
- `new_outcome_count`: `0`
- `rows`: `[]`
- `per_game_arm_rows`: `[]`
- `selection_recommendations`: `[]`
- `arc_reader_ready_score`: `1`
- `Qualified mechanics do not require scientific success.`

## RECOMMENDATION
KEEP

## experiment_8356_v720_kv260_workload_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit is asserted. If arithmetic superiority were claimed, the reported dense/active tie would refute it; constructed readiness alone establishes no added value.

## WAS THAT CHECKED
Yes, arithmetic parity was checked and an exact tie reported. The artifact withholds accelerator and service-benefit conclusions and assigns zero generalization benefit. Oracle-defined correctness therefore is not promoted into a verifier-value claim.

## EVIDENCE

- `accelerator_benefit`: `unproved_no_compatible_operation`
- `cpu_cost_scope`: `constructed_spline_arithmetic`
- `arithmetic_scientific_class`: `circular_positive`
- `arithmetic_pair_parity`: `identical_actions`: `true`; `max_probability_delta`: `0.0`
- `full_service_speedup`: `null`
- `generalized_learning_benefit_score`: `0`
- `current_device_execution_count`: `0`

## RECOMMENDATION
KEEP

## experiment_8357_v720_gatemate_change_ledger.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked evidence ledger, making no comparative benefit, generalization, or current hardware-success claim.

## WAS THAT CHECKED
No comparative experiment was performed or claimed. Historical authentication was checked and its failure retained; the board obligation remains excluded, with zero current device executions. The future physical preflight remains unexecuted.

## EVIDENCE

- `claim_scope`: `Evidence ledger; future physical preflight remains unexecuted`
- `honest_verdict`: `complete_blocked_gatemate_history`
- `verdict_class`: `blocked`
- `history_authentication` → `passed`: `false`
- `current_device_execution_count`: `0`
- `scientific_benefit_score`: `0`
- `exclusion_reason`: `gatemate_history`

## RECOMMENDATION
KEEP

## experiment_8358_v720_terminal_replay_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. An operational contradiction would be granting replay readiness or scientific credit despite failed authentication or required checks.

## WAS THAT CHECKED
Yes, operationally. Row 8344 records failed replay and unavailable authenticated history; closure checks fail and the repository suite times out. The overall verdict remains blocked, with readiness and scientific-benefit scores zero. Successful worker execution does not erase these failures. The oracle supplies qualification checks without an added-value claim.

## EVIDENCE
- `Authenticate exact invocation and primitive bytes; mechanical replay preserves failed dispositions and grants no independent science credit.`
- `honest_verdict`: `complete_blocked_terminal_replay_qualification`
- `required_checks_passed`: `false`
- `branch_replay_ready_score`: `0`
- Row `8344`: `replay_passed`: `false`; `status`: `authenticated-history-unavailable`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8359_v720_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The shown exposed-development comparisons demonstrate no decision-cost advantage for spline34 or online updates, while H1 remains unqualified for representation-level conclusions.

## WHAT WOULD REFUTE IT
A positive paired cost improvement for spline34 over RBF34, or online updates over the frozen and calibration-only baselines, with a positive lower bootstrap bound under the stated row accounting.

## WAS THAT CHECKED
Yes, in H1’s paired cost comparisons and H2’s arm comparisons and bootstrap summary. Costs depend on actions and outcomes, so improvement was possible independently of predicted probabilities. H1’s gain is negative; H2’s gain is zero despite probability updates.

Serious inexpensive controls are present: linear6 ties spline34’s decision costs, RBF34 achieves lower costs, and calibration_only ties online updates. These support the reported null. Circular implementation controls are not promoted into a verifier-value claim. The measurements are explicitly scoped as exposed development, and qualification failures and unqualified-row cost bounds remain visible.

## EVIDENCE

- H1: `science_disposition`: `null_optimization_limited`; `procedure_null_informative`: `true`; `representation_null_informative`: `false`; `qualified`: `false`.
- H1 complete-case bootstrap: `mean_gain`: `-0.005154639175257732`; `lower_one_sided_975`: `-0.02577319587628866`; `scope`: `descriptive_exposed_development_not_confirmatory`.
- H1 complete-case `cost_mean`: spline34 `0.4381443298969072`; linear6 `0.4381443298969072`; RBF34 `0.4329896907216495`.
- H2: `mean_gain`: `0.0`; online_sparse `probability_changes`: `66`; `action_changes`: `0`. Frozen, online, and calibration-only arms share `cost_mean`: `0.4431818181818182`.
- H2 unqualified rows: `qualified`: `false`; `frozen_cost`: `0.5`; `online_cost`: `0.5`; `gain_lower`: `0.0`; `gain_upper`: `0.0`.

## RECOMMENDATION
KEEP
