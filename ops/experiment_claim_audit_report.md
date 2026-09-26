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
| CLAIM_SUPPORTED | 7 |
| CLAIM_OVERSTATED | 1 |

## experiment_7659_v668_atom_corpus.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The atom corpus meets the stated readiness gate, while scientific benefit remains unestablished.

## WHAT WOULD REFUTE IT
Fewer than 16 covered fit groups, constant fit features, invalid inputs, or failed required checks would refute readiness. Paired predictions with independent outcomes demonstrating benefit would refute the null benefit assessment.

## WAS THAT CHECKED
Yes for readiness: the gates report 22 covered fit groups, nonconstant features, authenticated inputs, and passing required checks. Benefit was not tested because the required paired predictions and outcomes were absent; the artifact makes no positive benefit claim.

## EVIDENCE
`"honest_verdict": "complete_null_atom_corpus_ready"`; `"fit_covered_groups": 22`; `"threshold": 16`; `"fit_nonconstant": true`; `"authenticated_inputs": true`; `"required_checks_passed": true`; `"paired_probabilities": 0`; `"typed_decisions": 0`; `"delayed_feedback_events": 0`; `"fresh_confirmatory_groups": 0`; `"fresh_confirmatory_claim_allowed": false`

## RECOMMENDATION
KEEP

## experiment_7660_v668_atom_energy.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The energy head architecture is execution-ready with valid frozen heads, but demonstrates null scientific benefit over baseline with no held-out benefit claimed.

## WHAT WOULD REFUTE IT
Any of the following observations in the artifact's own data would refute the claim:
1. Operational failure: non-zero exit codes in validation checks, non-empty `failed_required_commands`, `required_checks_passed` being false, `frozen_heads` being 0, or `flagged_adversarial` being true.
2. Verdict/row contradiction: claiming positive benefit despite `identity` achieving a lower Brier score than `atom` (0.030189 vs 0.049949) or despite zero held-out groups opened.
3. Failing consistency checks: `verdict_row_consistency` or `adversarial_verify` flagging circularity or discrepancies between the 720 paired rows and the null verdict.

## WAS THAT CHECKED
Yes. Operational readiness was checked via 9 affected-scope validation commands (`focused_pytest`, `changed_module_coverage`) and 3 terminal readers (`cold_reduce`, `adversarial_verify`, `verdict_row_consistency`), all passing with zero failures. The absence of scientific benefit was explicitly checked through acceptance gates: `validity` passed, `readiness` passed, while `freshness` was evaluated and failed (`passed`: false due to 0 fresh confirmatory groups) and `probability_benefit` was evaluated and nulled (`passed`: null due to 0 held-out groups opened), confirming an honest null.

## EVIDENCE
- `honest_verdict`: `complete_null_energy_head_ready`
- `verdict_class`: `null`
- `energy_ready_score`: `1`
- `claim_limits`: `Paired arms do not enlarge N; no held-out benefit is claimed.`
- `prior_exposure`: `All 120 inherited learning groups were previously exposed.`
- `fresh_confirmatory_groups`: `0`
- `held_out_groups_opened`: `0`
- `frozen_heads`: `5`
- `paired_rows`: `720`
- `required_checks_passed`: `true`
- `failed_required_commands`: `[]`
- `missing_required_commands`: `[]`
- `flagged_adversarial`: `false`
- `verifier_is_oracle`: `false`
- `target_is_witness_output`: `false`

## RECOMMENDATION
KEEP

## experiment_7661_v668_decision_evaluation.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Atom scoring provides no registered decision benefit over prespecified controls across the evaluated groups.

## WHAT WOULD REFUTE IT
Observing a statistically significant decision benefit for atom scoring over prespecified controls, specifically achieving a paired Brier score 95% confidence interval lower bound exceeding 0.01 against `cheap_atom` or meeting the registered utility cost and non-escalation coverage thresholds.

## WAS THAT CHECKED
Yes. Paired comparisons and 10,000-draw bootstrap confidence intervals were evaluated against prespecified controls (including `cheap_atom`, `scalar`, `identity`, `source_deranged`, and `source_erased`) in `acceptance_gate_results` under `probability_benefit` and `utility`, as well as in `independent_reduction.confidence_intervals` across 40 independent groups and 240 paired rows.

## EVIDENCE
`"honest_verdict"`: `"complete_null_no_registered_decision_benefit"`
`"verdict_class"`: `"null"`
`"probability_benefit_score"`: `0`
`"utility_benefit_score"`: `0`
`"source_dependence_score"`: `0`
`"best_prespecified_control"`: `"cheap_atom"`
`"gate"`: `"probability_benefit"`
`"passed"`: `false`
`"estimate"`: `-0.0010338017975771076`
`"ci95"`: `[-0.011074405974452384, 0.009060886357545343]`
`"threshold"`: `0.01`
`"gate"`: `"utility"`
`"passed"`: `false`
`"claim_limit"`: `"no fresh confirmatory claim"`
`"prior_exposure"`: `"all groups exposed; exploratory only"`

## RECOMMENDATION
KEEP

## experiment_7662_v668_delayed_update_protocol.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The delayed update protocol yields no fresh benefit over frozen and scalar controls, resulting in a complete null.

## WHAT WOULD REFUTE IT
The source arm achieving Brier score and decision cost improvements over both frozen and scalar controls exceeding the 0.01 margin on fresh confirmatory data, causing the `probability_benefit`, `utility`, and `freshness` acceptance gates to pass.

## WAS THAT CHECKED
Yes; evaluated in `acceptance_gate_results` (under `probability_benefit`, `utility`, and `freshness`) and in `independent_reduction`.

## EVIDENCE
`honest_verdict`: `"complete_null_delayed_update_no_fresh_benefit"`
`verdict_class`: `"null"`
`claim_limit`: `"no fresh confirmatory benefit"`
`gate`: `"probability_benefit"`
`passed`: `false`
`threshold`: `0.01`
`brier`: `frozen`: `0.14649937700872429`, `scalar`: `0.14645674990801422`, `source`: `0.1487339496444748`
`gate`: `"utility"`
`passed`: `false`
`threshold`: `0.01`
`decision_cost`: `frozen`: `0.17750000000000002`, `scalar`: `0.17750000000000002`, `source`: `0.175`
`gate`: `"freshness"`
`passed`: `false`
`fresh_confirmatory_groups`: `0`

## RECOMMENDATION
KEEP

## experiment_7663_v668_continuous_atom_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Continuous atom learning provides no registered benefit in predictive accuracy or decision utility over frozen or scalar baselines.

## WHAT WOULD REFUTE IT
Observing a statistically significant improvement in predictive accuracy or decision cost where the lower bound of the paired block 95% confidence interval exceeds 0.01 (`lower_ci95` > 0.01) against frozen or scalar baselines, or observing the `probability_benefit` or `utility` acceptance gates passing.

## WAS THAT CHECKED
Yes. Checked in `acceptance_gate_results` (under `probability_benefit` and `utility`) and in `independent_reduction.paired_contrasts` across 80 independent groups (400 paired rows) evaluated against `frozen` and `scalar` baseline arms.

## EVIDENCE
- `"honest_verdict": "complete_null_continuous_learning_no_registered_benefit"`
- `"verdict_class": "null"`
- `"continuous_benefit_score": 0`
- `"utility_benefit_score": 0`
- `"gate": "probability_benefit"`
- `"lower_bound_threshold": 0.01`
- `"lower_ci95": -0.0002996276115452211`
- `"lower_ci95": -6.780939558444666e-05`
- `"gate": "utility"`
- `"threshold": 0.01`
- `"passed": false`
- `"independent_groups": 80`
- `"paired_rows": 400`

## RECOMMENDATION
KEEP

## experiment_7664_v668_independent_evidence_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Atom evidence and continuous learning provide no independent benefit in probability calibration or decision utility over baseline controls.

## WHAT WOULD REFUTE IT
A statistically significant positive improvement in Brier score or decision cost (such as a 95% bootstrap confidence interval strictly above zero or meeting the 0.01 acceptance gate threshold) for the active evidence arm relative to baseline controls (such as `identity`, `scalar`, or `frozen`).

## WAS THAT CHECKED
Yes. It was evaluated across multiple contrast gates and slices (`probability_benefit`, `utility`, `retention`, and independent contrasts across `continuous`, `delayed`, and `evaluation` sets against controls like `cheap_atom`, `identity`, `scalar`, `frozen`, and `omission`), where all measured improvements failed to clear thresholds, point estimates were near-zero or negative, and bootstrap confidence intervals covered zero or negative values.

## EVIDENCE
- `honest_verdict`: `complete_null_no_independent_benefit`
- `verdict_class`: `null`
- `gate`: `probability_benefit`, `passed`: `false`
- `gate`: `utility`, `passed`: `false`
- `principle`: `Both block-eight lower Brier improvements must exceed 0.01.`
- `finding`: `No registered held-out Brier benefit`
- `finding`: `No calibrated decision-cost and coverage benefit`
- `finding`: `Delayed updates and replay exist; benefit is not established`
- `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7665_v668_qwen_grounded_claims.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bounded measurement completed and found zero exactly supported claims, without claiming a benefit for the typed index or for whole answers.

## WHAT WOULD REFUTE IT
An eligible row with an exactly supported claim would refute the reported zero-support result. An incomplete or excluded call counted toward completion would refute the completion claim.

## WAS THAT CHECKED
Yes. The artifact reports zero exactly supported claims across 24 sources, 48 completed calls, no exclusions, and passing independent reduction and row-consistency checks. The displayed rows show outcomes that could differ, including an invalid pointer and a generated response with no pointer. The remaining rows are elided, so their individual outcomes cannot be inspected here.

## EVIDENCE
`complete_null_bounded_grounded_claim_mechanism`; `verdict_class`: `null`; `exact_supported`: `0`; `sources`: `24`; `forward_calls_completed`: `48`; `excluded`: `0`; `independent_reduction`: `passed`: `true`; `verdict_row_consistency_strict`: `passed`: `true`; `whole_answer_benefit_claim`: `false`; `verifier_is_oracle`: `true`.

## RECOMMENDATION
KEEP

## experiment_7666_v668_arc_goal_confirmation.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The goal-confirmation guard has a positive, fixture-ready result.

## WHAT WOULD REFUTE IT
A direct check of the same SDK level and terminal state that ties the guard would refute any claim that the guard adds value. An independently labeled fixture on which the guard gives the wrong status would refute its fixture accuracy.

## WAS THAT CHECKED
No for added value. The verifier is the correctness oracle, and the shown OFF row has no correctness score, so the arms cannot establish a gain. Scripted fixtures did exercise the guard’s status handling.

## EVIDENCE
`complete_circular_positive_goal_confirmation_fixture_ready` · `circular_positive` · `verifier_is_oracle` · `true` · `oracle_matches` · `48` · `OFF` · `correct` · `null` · `scripted_sdk_fixture_oracle`

## RECOMMENDATION
NARROW_CLAIM
