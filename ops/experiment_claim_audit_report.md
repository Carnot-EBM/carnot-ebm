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
| CLAIM_SUPPORTED | 5 |
| NO_CLAIM | 2 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7645_v667_arc_validation_requalification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The ARC goal guard CPU planner is validation-ready across constructed regression fixtures while demonstrating complete null hidden-game benefit.

## WHAT WOULD REFUTE IT
The claim would be refuted if:
1. Any regression fixture failed execution, threw an assertion, or triggered an invalid transition swallow (e.g., `passed` reporting `false`, or `hud_dedup_swallow` reporting `swallows` as `true`), or if any validation process exited non-zero, which would falsify the readiness claim.
2. The run credited game solve advantages (e.g., `new_game_level_solve_credit` reporting `true` or `probability_benefit` passing with non-zero `hidden_game_groups`), which would contradict the null benefit claim.

## WAS THAT CHECKED
Yes. Readiness refutation was checked across 6 regression fixtures in `goal_guard_rows` and 17 command runs in `validation_receipts`, where all checks completed and passed without swallow or failure. Game benefit refutation was checked at the gate level in `acceptance_gate_results.probability_benefit`, which verified zero hidden-game exposure and explicitly failed the benefit, utility, and retention gates, constraining downstream reporting to a null verdict.

## EVIDENCE
- `"honest_verdict": "complete_null_arc_goal_guard_ready_no_hidden_game_benefit"`
- `"verdict_class": "null"`
- `"planner_goal_guard_ready_score": 1`
- `"readiness_only": true`
- `"hidden_game_groups": 0`
- `"new_game_level_solve_credit": false`
- `"passed": true`
- `"passed": false`
- `"unit_kind": "exact_regression_fixture"`
- `"independent_groups": 6`
- `"passed_groups": 6`
- `"verifier_is_oracle": true`

## RECOMMENDATION
KEEP

## experiment_7646_v667_source_feature_corpus.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7647_witness_energy.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no training or comparative result to refute. The artifact records a gate check before training.

## WAS THAT CHECKED
Yes. The gate check ran and recorded two failed gates; the experiment stopped at the pre-gate layer.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7646-source-feature-corpus.flagged_adversarial (actual=True == expected=False)"`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7650_v667_independent_source_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The audit completed, but the available source evidence could not support a learning benefit.

## WHAT WOULD REFUTE IT
Eligible independent source rows, available downstream producer results, or failed audit validation would contradict the blocked or completed parts of the claim.

## WAS THAT CHECKED
Yes. The artifact checks producer eligibility and existence, benefit gates, and terminal validation. The benefit gates did not pass, while terminal validation passed.

## EVIDENCE
`honest_verdict`: `complete_blocked_source_producers_unavailable`; `benefit_eligible`: `false`; `eligible`: `0`; `disposition`: `disqualified`; `reader_result`: `unavailable`; `passed`: `false`; `passed`: `null`; `independent_audit_complete_score`: `1`; `verifier_is_oracle`: `true`; `oracle_distinct_benefit`: `false`.

## RECOMMENDATION
KEEP

## experiment_7651_v667_qwen_witness_challenge.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The Qwen3.8-27B bounded witness challenge yielded a complete null pilot result with zero conservative intersection supported across all paired comparison items.

## WHAT WOULD REFUTE IT
Any observation in the artifact's own paired rows showing a non-zero count of supported witness propositions (`conservative_intersection_supported > 0`), or any structural control witness evaluation returning a valid verified proposition (`status` resolving to `supported` rather than `unknown`). Refutation was given a genuine opportunity to occur by executing 16 generation calls across 8 independent paired groups on GPU hardware under two distinct generation arms (`schema_constrained_decoding` and `explicit_schema_prompt`).

## WAS THAT CHECKED
Yes. Evaluated across 8 paired pilot groups (16 generation calls) recorded in `paired_pilot_rows` and `rows`, where all 8 groups were evaluated against structural grammar rules and all produced `conservative_intersection_supported` equal to `0`.

## EVIDENCE
- `honest_verdict`: `complete_null_bounded_witness_pilot`
- `conservative_intersection_supported`: `0`
- `qwen_challenge_complete_score`: `1`
- `groups`: `8`
- `raw_rows`: `16`
- `generation_calls_completed`: `16`
- `arms`: `explicit_schema_prompt`, `schema_constrained_decoding`
- `status`: `unknown`
- `reason`: `claim_outside_structural_grammar`
- `principle`: `A complete bounded pilot establishes feasibility only.`

## RECOMMENDATION
KEEP

## experiment_7652_v667_arc_wrapper_measurement.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
HUD_DEDUP showed no development-proxy benefit over OFF, and the wrapper was disqualified by failed validation.

## WHAT WOULD REFUTE IT
A valid induced window where HUD_DEDUP succeeded and OFF failed, with no offsetting loss, would refute the no-benefit finding; a passing full Python suite would refute the stated validation failure.

## WAS THAT CHECKED
Yes. The artifact compares paired HUD_DEDUP and OFF results across induced windows and records the validation receipt. Neither refuting observation occurred.

## EVIDENCE
`"honest_verdict": "complete_disqualified_wrapper_validation_failed"`; `"induced_new_successes": 0`; `"induced_lost_successes": 0`; `"usable_induced_windows": 38`; `"HUD_DEDUP"` and `"OFF"` each have `"successes": 1`; `"failed_receipts": ["full_python_suite"]`.

## RECOMMENDATION
KEEP

## experiment_7653_v667_arc_live_generalization.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The run completed, but it did not demonstrate HUD dedup benefit or generalization and was disqualified by required validation.

## WHAT WOULD REFUTE IT
A validated HUD dedup arm achieving a higher level than its matched baseline, together with passing evidence for hidden-game benefit and retention, would contradict that conclusion.

## WAS THAT CHECKED
Yes for the matched outcomes and validation gates: all three paired level differences were zero, and the gates failed. Hidden-game benefit and retention were not measured. The HUD dedup planner lever was unreachable, so these rows do not establish that the method has no value when active.

## EVIDENCE
`honest_verdict`: `complete_disqualified_required_validation`; `paired_level_deltas`: `bp35` `0`, `ls20` `0`, `tn36` `0`; `planner_lever_reachable`: `false`; `hidden_games`: `0`; `groups`: `0`; `executed`: `0`.

## RECOMMENDATION
KEEP

## experiment_7656_v667_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A benefit claim would be refuted by eligible evidence showing no improvement over a serious baseline. This artifact makes no such claim. Its blocked disposition would be contradicted if the required gates passed on eligible evidence.

## WAS THAT CHECKED
Yes for the blocked disposition: the artifact reports failed producer and readiness checks and records the missing inputs. It does not present a comparative benefit claim to test.

## EVIDENCE
`complete_blocked_required_v667_external_evidence`; `blocked`; `A complete prefix records terminal work without implying benefit.`; `failed_upstream_checks`: `9`; `checked_predicates`: `0`; `accounting_completion`.

## RECOMMENDATION
KEEP
