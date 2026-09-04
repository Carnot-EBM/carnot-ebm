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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 3 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6976_exact_candidate_certification.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The calibration-selected policy adds value by producing a positive held-out exact-success gain and capturing exact candidate headroom.

## WHAT WOULD REFUTE IT
An always-direct comparator tying the selected policy would refute added value from policy selection; oracle-independent correctness judgments would also be required to support any claim that exact certification itself adds value.

## WAS THAT CHECKED
Yes for the comparator: the selected policy is literally the `direct` schedule, so the held-out `direct` arm necessarily ties it. No for oracle independence: correctness is defined by the verifier itself.

## EVIDENCE
`"schedule_id": "direct"`; `"selected": true`; `"selection_frozen": true`; `"metric_role": "schedule"`; `"exact_success_count": 2`; `"selected_policy_positive_score": 1`; `"verdict_class": "circular_positive"`; `"verifier_is_oracle": true`; `"oracle_used_for_selection": false`; `"comparison": "direct_minus_trigger_switched"`; `"comparison": "direct_minus_draft_conditioned"`; `"wins": 2`; `"losses": 0`; `"ties": 16`

## RECOMMENDATION
NARROW_CLAIM

## experiment_6977_certified_pwa_kan_energy.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no asserted result to falsify; the intended positive claim would be refuted by KAN/PWA tying or losing to the constant or size-matched MLP on genuinely held-out rows, or by failed certification.

## WAS THAT CHECKED
No. The run stopped at a failed precondition before training, held-out comparison, or substantive certification rows were produced.

## EVIDENCE
`"honest_verdict": "blocked_certified_pwa_kan_energy"`, `"verdict_class": "blocked"`, `"certified_pwa_energy_ready_score": 0`, `"pwa_energy_heldout_positive_score": 0`, `"status": "not_trained_failed_precondition"`, `"failed_check": "calibration_labels_nonconstant"`, `"observed_value": false`, `"heldout_comparison_rows": []`, `"training_rows": []`, `"per_candidate_rows": []`

## RECOMMENDATION
KEEP

## experiment_6978_transactional_constraint_self_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed run found no transactional constraint self-learning benefit over read-only memory.

## WHAT WOULD REFUTE IT
Positive held-future paired gains for transactional writes over read-only memory—especially recovery on recurring errors without offsetting losses—would refute the null claim.

## WAS THAT CHECKED
Yes. The artifact compares matched `transactional_write` and `read_only` arms on chronological held-future events, records per-event paired outcomes, and aggregates learning gain, plasticity, forgetting, and the positive-learning criterion.

## EVIDENCE
`honest_verdict`: `complete_null_transactional_constraint_self_learning`; `verdict_class`: `null`; `chronological_gain_over_readonly`: `0`; `plasticity_score`: `0.0`; `transactional_learning_positive_score`: `0`; `max_forgetting`: `0`; shown `held_future_rows` have `paired_delta`: `0`; both compared arms begin with the same `initial_state_hash`; `memory_write` is `false` for `read_only` and `true` for `transactional_write`; `verifier_is_oracle`: `false`.

## RECOMMENDATION
KEEP

## experiment_6979_self_learning_cold_audit.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6980_spilled_energy_requalification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Spilled energy failed the preregistered held-out superiority gate, repeated the prior null, and should not be requalified.

## WHAT WOULD REFUTE IT
Held-out spilled-energy AUROC at least 0.65 with its interval excluding 0.5, plus strictly positive lower confidence bounds against both entropy and top probability in at least two model families.

## WAS THAT CHECKED
Yes. The held-out metrics, clustered bootstrap interval, and paired control comparisons directly test the preregistered gate. Spilled energy reached AUROC 0.7, but its interval included 0.5 and it tied the serious entropy baseline overall and in the two model families with paired evidence. Ineligible rows were explicitly abstained rather than silently scored.

## EVIDENCE
The gate requires `heldout_spilled_auroc_at_least` `0.65`, `spilled_auroc_interval_excludes` `0.5`, `paired_delta_lower_above` `0.0`, and `minimum_model_families_beating_both_controls` `2`. The held-out spilled-energy row reports `auroc` `0.7`, while its bootstrap interval reports `lower` `0.5` and `upper` `0.75`. Against `entropy`, the overall comparison reports `delta_auroc` `0.0`, `delta_lower` `0.0`, and `delta_upper` `0.0`; the displayed model-level entropy comparisons likewise report `delta_auroc` `0.0`. Coverage was `0.12962962962962962`, with `eligible_count` `7` and `abstention_count` `47`. The result records `spilled_energy_requalified_score` `0`, `verdict_class` `null`, and `honest_verdict` `complete_null_spilled_energy_requalification_retired`.

## RECOMMENDATION
KEEP

## experiment_6981_arc_live_engine_generalization_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no affirmative result to falsify. A future generalization claim would be refuted if eligible held-out rows showed the engine tying or losing to a matched no-op or copied-delta control, or failing to change a production-path action.

## WAS THAT CHECKED
No. There were zero eligible candidates, no selected run, and no execution, holdout, ranking, control, or live-influence rows.

## EVIDENCE
`honest_verdict`: `blocked_arc_live_engine_generalization_audit`; `verdict_class`: `blocked`; `eligible_count`: `0`; `selected`: `null`; `arc_generalization_positive_score`: `0`; `engine_execution_rows`: `[]`; `control_rows`: `[]`; `per_transition_rows`: `[]`; `live_influence_fixture_rows`: `[]`; `solve_claimed`: `false`; `level_claimed`: `false`

## RECOMMENDATION
KEEP

## experiment_6982_hard_feasible_hybrid_selection.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked gate-check receipt and reports no hybrid-selection outcome or comparative claim to falsify.

## WAS THAT CHECKED
No. The experiment stopped at `conductor_pre_gate` because the required readiness score was `0` instead of `1`; no method result was evaluated.

## EVIDENCE
`"status"`: `"blocked"`; `"honest_verdict"`: `"blocked_gate_check_failed"`; `"failed_field"`: `"certified_pwa_energy_ready_score"`; `"failed_expected"`: `1`; `"failed_observed"`: `0`; `"blocked_at_layer"`: `"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6983_v611_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V611 capstone completed without producing any unexcluded, row-backed, non-circular science-positive result.

## WHAT WOULD REFUTE IT
At least one valid, unexcluded science-task row classified as a row-backed non-circular positive, making the science-positive score at least one.

## WAS THAT CHECKED
Yes. The per-task and branch-status reconciliation feeds the capstone science-positive score, and the corresponding gate explicitly tested for at least one non-circular science positive. The oracle-defined classification would prevent a claim about the verifier’s added value, but the reported claim is an execution-grounded null, not such a value claim.

## EVIDENCE
`honest_verdict` `complete_null_v611_capstone_no_non_circular_science_positive` `v611_science_positive_score` `0` `failed_check` `non_circular_science_positive` `expected_value` `1` `observed_value` `0` `passed` `false` `verdict_class` `null` `verifier_is_oracle` `true`

## RECOMMENDATION
KEEP
