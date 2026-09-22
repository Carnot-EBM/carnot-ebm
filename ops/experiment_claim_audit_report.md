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

## experiment_7518_source_pilot.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation that the upstream dependency was satisfied or that pilot intervention trials were executed and measured; however, the artifact makes no comparative or empirical claim and serves solely as a pre-flight receipt recording that execution was halted.

## WAS THAT CHECKED
No; execution never ran because the run was halted at the conductor pre-gate layer when upstream prerequisites failed validation.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`
`"status": "blocked"`
`"honest_verdict": "blocked_gate_check_failed"`
`"duration_s": 0.0`
`"blocked_at_layer": "conductor_pre_gate"`
`"blocked_reason": "actual=0 == expected=1"`
`"failed_upstream": "exp7517-source-protocol"`
`"failed_field": "source_protocol_ready_score"`
`"failed_observed": 0`
`"failed_expected": 1`
`"passed": false`

## RECOMMENDATION
KEEP

## experiment_7525_v658_decision_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Required upstream science is incomplete, so neither the static nor online branch qualifies a scientific claim.

## WHAT WOULD REFUTE IT
Every required upstream artifact appearing with its readiness field equal to 1, followed by non-blocked branch reductions with qualified claims, would refute the headline.

## WAS THAT CHECKED
Yes. `preconditions_checked`, `gate_check_summary`, and both branch reductions explicitly tested artifact existence and readiness; they found an existing source protocol with readiness 0 and multiple required artifacts absent.

## EVIDENCE
`honest_verdict`: `complete_blocked_required_science_incomplete`

`all_passed`: `false`

`failed_count`: `10`

`source_protocol_ready_score`

`expected`: `1`

`observed`: `0`

`eval_capture_ready_score`

`exists`: `false`

`claims_qualified_score`: `0`

`disposition`: `blocked`

`support`: `null`

`verdict_class`: `blocked`

## RECOMMENDATION
KEEP

## experiment_7526_v658_arc_eligibility.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Candidate intervention arms demonstrated decision eligibility during execution, but provided no applied mutation support or efficacy benefit.

## WHAT WOULD REFUTE IT
The claim would be refuted if `raw_reduction` recorded `selected_count` of 0 (or `selected_recommendation_support` showed `observed` 0), disproving that eligibility occurred; if `applied_count` was greater than 0 with verified positive treatment effect (`applied_mutation_support` passing with `observed` >= 1 and non-null `efficacy_estimate`), contradicting the null verdict; or if execution validity checks (`required_scoped_and_capability_checks` or `parity_rows`) failed, invalidating the eligibility observations.

## WAS THAT CHECKED
Yes. Checked in `acceptance_gate_results` (where `selected_recommendation_support` passed with `observed` 38, while `applied_mutation_support` and `applied_episode_uncertainty_support` both failed with `observed` 0), in `raw_reduction` (recording `selected_count` 38, `applied_count` 0, and `eligible_count` 5568 across 40,272 observations), and in `parity_rows` across all 5 verification surfaces.

## EVIDENCE
- `honest_verdict`: `"complete_null_eligibility_observed_without_applied_effect_support"`
- `verdict_class`: `"null"`
- `status`: `"complete_null_eligibility_observed_without_applied_effect_support"`
- `benefit_supported_score`: `0`
- `eligibility_receipt_ready_score`: `1`
- `efficacy_estimate`: `null`
- `solve_claimed`: `false`
- `verifier_is_oracle`: `false`
- `check`: `"selected_recommendation_support"`, `expected`: `1`, `observed`: `38`, `passed`: `true`
- `check`: `"applied_mutation_support"`, `expected`: `1`, `observed`: `0`, `passed`: `false`
- `check`: `"applied_episode_uncertainty_support"`, `expected`: `10`, `observed`: `0`, `passed`: `false`
- `raw_reduction`: `applied_count`: `0`, `eligible_count`: `5568`, `selected_count`: `38`, `observation_count`: `40272`
- `gate_check_summary`: `all_passed`: `false`, `failed_count`: `2`

## RECOMMENDATION
KEEP

## experiment_7527_v658_arc_opportunities.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An empirical performance comparison between arms, model generation results, or executed episode data appearing within the artifact's results.

## WAS THAT CHECKED
No. No experimental episodes or comparative evaluations were run because execution was halted at the prerequisite gate check before any episodes were started.

## EVIDENCE
`"honest_verdict": "complete_blocked_owned_gpu"`
`"verdict_class": "blocked"`
`"inference_substrate_class": "blocked_no_run"`
`"model_invoked": false`
`"all_passed": false`
`"check": "owned_gpu"`
`"observed": false`
`"benefit_supported": false`
`"opportunity_support_score": 0`
`"attempted": 0`
`"unstarted": 12`

## RECOMMENDATION
KEEP

## experiment_7528_v658_service_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact asserts neither service performance nor comparative value.

## WAS THAT CHECKED
No; the service benchmark never ran, so no falsifiable performance claim was tested.

## EVIDENCE
`honest_verdict` `complete_blocked_missing_count_memory_prototype` `service_branch_status` `blocked_external_prerequisite` `service_rows` `[]` `whole_service_speedup` `null` `target_100x_met` `false` `attempted` `0` `unstarted` `30`

## RECOMMENDATION
KEEP

## experiment_7530_b2_induction_gate_telemetry.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation in task execution showing behavioral divergence, altered policy actions, shifted random state, or execution regressions when induction telemetry is enabled versus disabled. The artifact lacks any comparative task benchmarks, model evaluations, or performance comparisons against an empirical baseline.

## WAS THAT CHECKED
no. The artifact lacks empirical model runs, baseline comparator arms, and task-level evaluations; it records only engineering build steps, static checks, and unit-level parity assertions.

## EVIDENCE
`"honest_verdict"`: `"complete_b2_induction_attempt_outcome_telemetry_built_default_off_and_parity_proven"`
`"status"`: `"complete_stage1_build_and_test"`
`"model_invoked"`: `false`
`"inference_substrate_class"`: `"no_model_load"`
`"inference_substrate"`: `"aggregation_from_upstream_artifacts"`
`"gate_ready_to_ship"`: `false`
`"submission_kernel_changed"`: `false`
`"default_off"`: `true`
`"induction_logic_changed"`: `false`

## RECOMMENDATION
KEEP

## experiment_7529_v658_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The required V658 scientific evidence is absent or externally gated, despite complete accounting of the milestone.

## WHAT WOULD REFUTE IT
Completed, valid, non-censored required-science rows with available producer artifacts and qualified scientific results—or fulfillment of the stated prerequisites, such as 480 fresh eligible source groups—would refute the blocked/absent status.

## WAS THAT CHECKED
Yes. The artifact checks the 14 planned task dispositions, producer-artifact existence, fresh-source inventory, and scientific claim eligibility. These checks could have recorded present artifacts and qualified results, but instead found missing prerequisites, censored or unstarted work, and zero qualification.

## EVIDENCE
`honest_verdict`: `complete_blocked_required_v658_science_absent_or_externally_gated`; `inference_substrate`: `aggregation_from_upstream_artifacts`; `failed_count`: `11`; `field`: `fresh_eligible_groups`; `expected`: `480`; `observed`: `0`; `check`: `producer_artifact_exists`; `observed`: `false`; `evidence_state`: `missing`; `unstarted`: `true`; `planned`: `14`; `censored`: `7`; `excluded`: `12`; `static_probability_value`; `causal_online_learning`; `positive_aggregate_eligible`: `false`; `qualified_value`: `0`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7531_b2_induction_gate_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no gate-quality, efficacy, or generalization claim to falsify; the artifact limits itself to a feasibility receipt. That receipt would be contradicted by failed or incomplete model execution.

## WAS THAT CHECKED
Yes. Execution completion, generation-call completion, GPU offload, and terminal status were recorded. The artifact also explicitly withholds stronger claims because the sample floor was unmet and the positive control had no headroom.

## EVIDENCE
`"publication_mode": "feasibility_only"`, `"numeric_gate_quality_claim": false`, `"hidden_game_efficacy_claim": false`, `"gate_ready_to_ship": false`, `"met": false`, `"positive_control_headroom_exists": false`, `"generation_calls_attempted": 112`, `"generation_calls_completed": 112`, `"generation_calls_failed": 0`, `"offload_real": true`, `"status": "complete_b2_measurement"`

## RECOMMENDATION
KEEP
