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

## experiment_6608_family_headroom_reducer.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No efficacy observation could refute it because no treatment benefit is asserted; a complete, replayable family marked eligible would only contradict the artifact’s blocked-readiness disposition.

## WAS THAT CHECKED
Yes, for readiness: the family replay, completeness, identity, and eligibility checks could have admitted a family, but none qualified. No comparative treatment effect was tested.

## EVIDENCE
`honest_verdict` states `blocked_family_baseline_evidence: no family has complete replayable baseline rows; eligibility is frozen empty and no treatment benefit is claimed`; `eligible_model_specs` is `[]`; `headroom_benchmark_ready_score` is `0.0`; `verdict_class` is `blocked`; `eligible_family_count` has `expected` `>=1`, `observed` `0`, and `passed` `false`.

## RECOMMENDATION
KEEP

## experiment_6609_two_level_constrained_decoding.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked gate rather than an experimental result.

## WAS THAT CHECKED
No; execution stopped at `conductor_pre_gate`, so the method and any comparative outcome were not tested.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6611_live_arc_invariant_projection.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Selected invariant projection had no effect on held exact-next-frame prediction error and makes no solve claim.

## WHAT WOULD REFUTE IT
A lower or higher charged exact-mismatch total or mean for selected invariant projection than for no projection on the held rows would refute the no-effect claim.

## WAS THAT CHECKED
Yes. The held-arm comparison includes the serious no-projection baseline, retains invalid rows, and charges failures; all three arms tie on exact mismatch despite nonzero projection distance and iterations, so an effect had a real opportunity to appear.

## EVIDENCE
`complete_held_world_model_prediction_projection_no_effect_no_solve_claim`; `verdict_class`; `null`; `game_disjoint`; `true`; `observation_opened_after_prediction`; `no_projection`; `selected_invariant_projection`; `norm_matched_random_projection`; `charged_exact_mismatch_total`; `9376`; `charged_exact_mismatch_mean`; `180.30769230769232`; `projection_distance_total`; `0.009485742262737467`; `iterations_total`; `16`; `invalid_rows_retained`; `true`; `strictly_lower_than_no_projection`; `false`; `candidate_held_win`; `false`

## RECOMMENDATION
KEEP

## experiment_6612_spectral_k_block_scale_rust_parity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the headline explicitly withholds stationary, transition, wall-time, and parity conclusions rather than asserting comparative value.

## WAS THAT CHECKED
No; there is no positive headline claim to falsify. The artifact instead records a blocking gate.

## EVIDENCE
`honest_verdict` `blocked_reference_parity_cost_or_protection: stationary, transition, wall, and parity conclusions are withheld; CPU software only` `verdict_class` `blocked` `tests_complete` `false` `toolchain_or_test_failure` `exit_code` `3`

## RECOMMENDATION
KEEP

## experiment_6613_invariant_memory_lifecycle.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No utility or comparative claim is advanced. The limited conformance receipt would fail if any required lifecycle, recovery, protection, or adversarial check failed, or if a poisoned record became active.

## WAS THAT CHECKED
Yes, for conformance: attack, lifecycle, restart, rollback, corruption, and poison/injection rows could report failures. The oracle is circular only for an added-value claim, which the artifact explicitly declines to make.

## EVIDENCE
`honest_verdict`: `complete_invariant_memory_lifecycle_conformance_no_utility_claim`; `no_utility_claim`: `true`; `utility_claimed`: `false`; `verifier_is_oracle`: `true`; `verdict_class`: `null`

## RECOMMENDATION
KEEP

## experiment_6614_prospective_invariant_self_learning.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6605_qwen36_direct_headroom.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The direct baseline’s integrity did not complete because GPU process receipts failed.

## WHAT WOULD REFUTE IT
A complete integrity recomputation with GPU process receipts passing and every GPU session authenticated would refute the blocked-integrity claim.

## WAS THAT CHECKED
Yes. The gate summary, GPU-session receipts, and integrity recomputation explicitly evaluated that condition and recorded its failure.

## EVIDENCE
`honest_verdict` = `blocked_gpu_process_receipts: direct baseline integrity did not complete`; `all_sessions_authentic` = `false`; `failed_condition` = `gpu_process_receipts`; `complete` = `false`; `failed_checks` = `gpu_process_receipts`.

## RECOMMENDATION
KEEP

## experiment_6615_v576_independent_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Independent replay found decoding blocked, live projection null, sampler evidence partial and software-only, lifecycle conformant without utility, and prospective self-learning with zero benefit.

## WHAT WOULD REFUTE IT
A nonzero held-data advantage for `selected_invariant_projection` over `no_projection`, or for `governed_online_memory` over the serious `static_projector` control, would refute the null/zero-utility headline; a failed lifecycle transition would refute the conformance claim.

## WAS THAT CHECKED
Yes. The projection comparison used 52 rows per arm and compared against both no projection and norm-matched random projection. The learning comparison used 88 rows per arm, 36 held-future pairs, and included static and no-learning controls. Twenty-seven lifecycle transitions were checked. All compared effects were zero and all lifecycle transitions passed. Although `verifier_is_oracle` is true, the artifact does not promote verifier-defined conformance into a positive value claim.

## EVIDENCE
`selected_minus_no_projection_mean_error`: `0.0`; `selected_minus_random_mean_error`: `0.0`; `held_future_benefit_over_static`: `0.0`; `held_future_benefit_over_shuffled`: `0.0`; `held_future_pairs`: `36`; `row_count_by_arm`: `governed_online_memory`, `no_learning`, `shuffled_admission_control`, `static_projector`, each `88`; `all_lifecycle_transitions_passed`: `true`; `lifecycle_transition_count`: `27`; `oracle_defined_win_promoted_to_positive`: `false`; `verdict_class`: `partial`; `scope`: `cpu_software_only`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP
