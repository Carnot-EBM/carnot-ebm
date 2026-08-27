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
| NO_CLAIM | 4 |

## experiment_6654_prospective_repair_memory_evolution.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Validation-gated repair memory adds value by improving future exact outcomes without forgetting or support collapse.

## WHAT WOULD REFUTE IT
An ungated accept-all repair-memory arm receiving the same proposals and tying or beating `verified_memory` would refute the validation gate’s added value.

## WAS THAT CHECKED
No. The artifact compares only `frozen`, `context_only`, and `verified_memory`; it lacks an ungated memory arm. Moreover, every proposed update was admitted, so the gate never changed the memory trajectory and an accept-all arm would necessarily tie under the stated method. The artifact did genuinely check memory against no-memory arms, future-label isolation, forgetting, and support collapse—but not the contribution implied by “validation-gated.”

## EVIDENCE
`honest_verdict`: `complete: validation-gated repair memory improved future exact outcomes with no forgetting or support collapse`

`verified_memory`: `the context-only ranking plus admitted earlier repair memory`

`acceptance_rate`: `1.0`

`accepted_count`: `108`

`proposed_count`: `108`

`rejected_count`: `0`

`verified_memory_minus_context_only`: `0.6666666666666667`

`count`: `0`

`collapse_count`: `0`

`verifier_is_oracle`: `false`

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_6655_repair_memory_safety_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Safety replay and point-estimate recomputation passed, but the order-level effect remains compatible with zero and should be limited to this fixture.

## WHAT WOULD REFUTE IT
An independently recomputed order-level 95% interval that excluded zero—or a failed safety or recomputation unit—would falsify the headline.

## WAS THAT CHECKED
Yes. The aggregate recomputation reports the order-level interval, and the audit includes event and metric recomputation, poison attacks, restart atomicity, rollback exactness, and unit-hash checks.

## EVIDENCE
`"status": "complete_null"`; `"claim_disposition": "narrow"`; `"all_audit_units_pass": true`; `"order_delta_mean_95_interval": [-0.013448269770851111, 0.12455938088196222]`; `"event_recomputation": true`; `"metric_recomputation": true`; `"poison_attacks": true`; `"restart_atomicity": true`; `"rollback_exactness": true`; `"verifier_is_oracle": false`; `"recommendation": "do_not_deploy_and_collect_independent_orders"`

## RECOMMENDATION
KEEP

## experiment_6656_arc_trace_automaton_live_loo.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative value claim to falsify. The stated blocker would be refuted by exact post-redirect outcomes for changed actions, allowing the on/off arms to be compared for live-policy benefit.

## WAS THAT CHECKED
No. The artifact checked whether such outcomes existed, but the archived replay transport could not generate them after actions diverged; therefore benefit was not genuinely tested.

## EVIDENCE
`honest_verdict` = `blocked: the canonical E3 action seam applied held-family redirects, but the archived transport has no exact outcomes after changed actions; no live-policy benefit, game solve, or level solve is claimed`; `transport_observed` = `archived_live_e3_action_receipt_replay`; `failed_check` = `exact_live_next_outcome_after_applied_redirect`; `missing_exact_on_outcome_count` = `2067`; `benefit_claim_allowed` = `false`; `verdict_class` = `blocked`

## RECOMMENDATION
KEEP

## experiment_6657_bounded_treewidth_ising_reference.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Bounded exact-reference readiness was not established because required test gates failed or were missing.

## WHAT WOULD REFUTE IT
All required readiness gates passing, with no missing test scopes and `ising_reference_ready` becoming true, would refute the claim.

## WAS THAT CHECKED
Yes. The readiness reducer checked the required test scopes and recorded a failing full Python suite plus missing full-suite/spec-coverage gates, making a positive readiness result genuinely possible but absent.

## EVIDENCE
`honest_verdict`: `blocked_tests_check_failed: bounded exact-reference readiness was not established`; `ising_reference_ready`: `false`; `tests_all_passed`: `false`; `ready`: `false`; `missing_test_scopes`: `full_python_suite`, `spec_coverage`; `exit_code`: `2`; `verdict_class`: `blocked`

## RECOMMENDATION
KEEP

## experiment_6658_thermodynamic_schedule_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific outcome is claimed; the blocking diagnosis would be falsified if the prerequisite gate had observed `true` or passed.

## WAS THAT CHECKED
Yes, in `gates_evaluated`; the sole prerequisite gate was evaluated and failed before the experiment ran.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_field": "ising_reference_ready"`; `"failed_expected": true`; `"failed_observed": false`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`; `"duration_s": 0.0`

## RECOMMENDATION
KEEP

## experiment_6659_v580_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V580 provides bounded admission, direct-corpus, and verifier-unit evidence, while memory benefit is null and the ARC, repair, and scheduling branches remain blocked or unsupported.

## WHAT WOULD REFUTE IT
Valid recomputed rows showing fewer than 3 of 3 model families admitted, a direct-exact result other than 8 of 48, fewer than 8 of 8 two-step catches or any two-step false rejects, a memory interval excluding zero, exact ARC outcomes supporting solve credit, or completed repair/schedule comparisons would refute the corresponding headline assertions.

## WAS THAT CHECKED
Yes. The artifact recomputes the admission, corpus, verifier, memory, ARC, suffix-regeneration, reference, and schedule results in `headline_recomputation`; it also reports gate validity and field presence. The missing audit artifact and failed full-suite receipt are explicitly disclosed and are not converted into positive claims.

## EVIDENCE
`complete_partial: V580 has authentic admission, direct-corpus, and bounded verifier-unit evidence; memory general benefit is null under independent uncertainty; ARC and Ising branches are blocked; one audit artifact is missing; no pooled success, ARC solve, repair win, or hardware speedup is claimed`; `mandated_model_families_admitted` `3`; `mandated_model_families_total` `3`; `direct_exact_success_count` `8`; `candidate_row_count` `48`; `unit_id` `two_steps`; `catch_count` `8`; `pair_count` `8`; `false_reject_count` `0`; `interval_includes_zero` `true`; `arc_solve_credit` `0`; `changed_action_exact_outcome_count` `0`; `status` `blocked_headroom_gate`; `status` `blocked_reference_gate`; `verifier_is_oracle` `false`; `stored_metric_mismatches` `[]`

## RECOMMENDATION
KEEP

## experiment_6661_triggered_tail_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact reports fixture readiness and test status, not model quality or comparative value.

## WAS THAT CHECKED
No; no model inference or comparative evaluation was attempted.

## EVIDENCE
`honest_verdict`: `blocked_triggered_tail_fixture: tests failed`; `inference_substrate`: `cpu_fixture_and_exact_checker_no_llm`; `verifier_is_oracle`: `true`; `tests`: `false`; `The verdict reports fixture readiness and makes no model-quality claim.`

## RECOMMENDATION
KEEP

## experiment_6662_triggered_structured_tail_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No result claim exists to falsify; a completed A/B run with comparative outcome rows would be required before refutation is possible.

## WAS THAT CHECKED
No. The experiment stopped at the pre-gate, so the method, success criterion, and comparator were never evaluated.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_observed`: `false`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP
