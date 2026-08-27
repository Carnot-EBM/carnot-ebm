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
| NO_CLAIM | 4 |

## experiment_6651_failure_localized_suffix_regeneration.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative result to falsify. Had the artifact claimed that failure-localized suffix regeneration outperformed its A/B rival, a tie or win by that rival would refute it.

## WAS THAT CHECKED
No. Execution stopped at the pre-gate before any A/B outcomes were produced.

## EVIDENCE
`status` `blocked` `honest_verdict` `blocked_gate_check_failed` `regeneration_headroom_count` `expected` `8` `actual` `2` `passed` `false` `blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6653_state_grounded_repair_memory_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or future-benefit claim is made; for the limited readiness receipt, a failed gate, undetected seeded attack, non-reversible transition, or malformed fixture row would refute readiness.

## WAS THAT CHECKED
Yes. Readiness was checked through aggregate gates, six adversarial attack rows, schema and partition checks, checksums, and rollback receipts. No future-benefit comparison or serious baseline was checked because none was claimed.

## EVIDENCE
`"honest_verdict"`: `"complete: state-grounded repair-memory fixture is ready; no future benefit was measured or claimed"`; `"memory_fixture_ready"`: `true`; `"all_checks_passed"`: `true`; `"attack_count"`: `6`; `"detected"`: `true`; `"failed_closed"`: `true`; `"restored_state_equal"`: `true`; `"inference_substrate"`: `"deterministic_exact_repair_memory_fixture_no_llm"`

## RECOMMENDATION
KEEP

## experiment_6654_prospective_repair_memory_evolution.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Validation-gated repair memory improved future exact outcomes over matched non-memory controls without observed forgetting or recoverable-support collapse.

## WHAT WOULD REFUTE IT
A matched `context_only` arm tying or beating `verified_memory` on future exact yield, any preregistered order showing a non-positive memory delta, an influential retrieval producing an incorrect outcome, a positive forgetting count, or a support collapse would falsify the headline.

## WAS THAT CHECKED
Yes. The artifact compares matched arms on prequentially committed actions, reports future-event results, reports positive deltas for all three preregistered orders, credits only action-changing retrievals, and separately measures forgetting and recoverable support. Exact outcomes were opened after action commitment, so failure remained possible.

## EVIDENCE
`"exact_outcome_opening": "after_live_action_commit"`; `"verified_memory_minus_context_only": 0.6666666666666667`; `"exact_success_count": 9`; `"exact_success_count": 3`; `"minimum_delta": 0.02777777777777779`; `"maximum_delta": 0.08333333333333333`; `"influential_count": 6`; `"credited_exact_count": 6`; `"count": 0`; `"collapse_count": 0`; `"matched_arms": true`; `"all_recomputations_match": true`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6655_repair_memory_safety_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The independent safety replay passed and reproduced the point estimates, but the order-level uncertainty includes zero, so the result is limited to this fixture.

## WHAT WOULD REFUTE IT
An order-level 95% interval wholly above or below zero would refute the null conclusion; any metric mismatch, unsafe poison acceptance, non-atomic restart, or inexact rollback would refute the audit-passed claim.

## WAS THAT CHECKED
Yes. The artifact reports independent event and metric recomputation, adversarial poison tests, restart tests, rollback checks, and an order-level interval spanning negative and positive values. These checks could have failed, and the verifier was not the correctness oracle.

## EVIDENCE
`honest_verdict`: `complete_null: safety replay passed and point estimates reproduced, but the order-level interval includes zero; narrow the result to this fixture`; `order_delta_mean_95_interval`: `-0.013448269770851111`, `0.12455938088196222`; `all_audit_units_pass`: `true`; `stored_matches_rebuilt`: `true`; `failed_closed`: `true`; `atomicity_result`: `old_or_new_complete`; `byte_exact_restoration`: `true`; `verifier_is_oracle`: `false`; `claim_disposition`: `narrow`; `verdict_class`: `null`

## RECOMMENDATION
KEEP

## experiment_6656_arc_trace_automaton_live_loo.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No value claim is made; a future benefit claim would be refuted by exact post-redirect outcomes showing the on arm ties or underperforms the off arm on progress or solves.

## WAS THAT CHECKED
No. Changed on-arm actions lack exact subsequent outcomes, so the artifact cannot compare their consequences with the off arm.

## EVIDENCE
`"verdict_class": "blocked"`; `"status": "blocked_archived_transport_lacks_redirect_outcomes"`; `"failed_check": "exact_live_next_outcome_after_applied_redirect"`; `"passed": false`; `"missing_exact_on_outcome_count": 2067`; `"benefit_claim_allowed": false`; `"prevented_violation_measurable": false`; `"claimed_game_or_level_solve": false`; `"no live-policy benefit, game solve, or level solve is claimed"`

## RECOMMENDATION
KEEP

## experiment_6657_bounded_treewidth_ising_reference.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Bounded exact-reference readiness was not established because required test checks failed.

## WHAT WOULD REFUTE IT
All required test scopes—including the full Python suite and spec coverage—passing, with the aggregate readiness gate reporting true and no failed checks.

## WAS THAT CHECKED
Yes. The aggregate readiness recomputation and gate summary checked the required test category, while the test receipts show the full Python suite failed. The oracle defines correctness, but the artifact makes no positive claim about the verifier’s added value.

## EVIDENCE
`honest_verdict`: `blocked_tests_check_failed: bounded exact-reference readiness was not established`; `ready`: `false`; `tests_all_passed`: `false`; `missing_test_scopes`: `full_python_suite`, `spec_coverage`; `failed_checks`; `check`: `tests`; `passed`: `false`; `exit_code`: `2`; `summary`: `interrupted after baseline failure was established: 710 failed, 20674 passed, 65 skipped, 118 warnings in 1041.10s`; `ising_reference_ready`: `false`; `verdict_class`: `blocked`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_6658_thermodynamic_schedule_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim was made to falsify; an A/B result showing the autocorrelation-aware schedule tying or losing to a serious baseline would refute a positive value claim if one existed.

## WAS THAT CHECKED
No. The experiment stopped at `conductor_pre_gate` because its sole upstream gate failed, before any A/B data was produced.

## EVIDENCE
`"status"`: `"blocked"`; `"honest_verdict"`: `"blocked_gate_check_failed"`; `"failed_field"`: `"ising_reference_ready"`; `"failed_expected"`: `true`; `"failed_observed"`: `false`; `"passed"`: `false`; `"blocked_at_layer"`: `"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6659_v580_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V580 provides authentic admission, direct-corpus, and bounded two-step verifier evidence, while memory benefit remains null and the ARC, repair, and Ising conclusions remain blocked or unclaimed.

## WHAT WOULD REFUTE IT
A valid frozen pair on which the two-step verifier missed the paired error or falsely rejected the clean member would refute the positive verifier component; admission failure, an independent memory interval excluding zero, or completed exact ARC/Ising outcomes would likewise contradict other headline components.

## WAS THAT CHECKED
Yes. The frozen verifier comparison reports catches and false rejects for one-step, two-step, and full-suffix arms; admission gates were recomputed; memory was evaluated with an independent interval; and the blocked branches record missing denominators and failed readiness gates. The missing audit and failed full suite are explicitly disclosed rather than counted as positive evidence.

## EVIDENCE
`complete_partial: V580 has authentic admission, direct-corpus, and bounded verifier-unit evidence; memory general benefit is null under independent uncertainty; ARC and Ising branches are blocked; one audit artifact is missing; no pooled success, ARC solve, repair win, or hardware speedup is claimed`

`catch_count` `8`

`false_reject_count` `0`

`unit_id` `two_steps`

`unit_id` `one_step`

`catch_count` `0`

`unit_id` `full_remaining_suffix`

`false_reject_count` `2`

`independent_order_delta_interval_95`

`-0.013448269773437542`

`0.12455938088454865`

`interval_includes_zero` `true`

`arc_solve_credit` `0`

`changed_action_exact_outcome_count` `0`

`status` `blocked_headroom_gate`

`status` `blocked_reference_gate`

`field_present` `true`

`contract_state` `valid`

## RECOMMENDATION
KEEP
