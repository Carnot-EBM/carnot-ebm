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
| NO_CLAIM | 2 |
| CANNOT_DETERMINE | 1 |

## experiment_6913_relation_source_tuple_qualification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bounded claim is that source-tuple qualification completed with replayable evidence, qualifying `rule:anchored_lexical_v1` and disqualifying the other evaluated arms under the stated oracle.

## WHAT WOULD REFUTE IT
A missing, duplicate, or nonterminal cell; replayed outputs or recomputed aggregates disagreeing with reported results; or a rival arm tying or beating the rule arm under the same qualification criterion would refute the claim.

## WAS THAT CHECKED
Yes. Terminal-row completeness, uniqueness, replay, aggregate agreement, and arm qualification decisions were gated; row-level failures were retained, and every rival arm scored below the qualified rule arm. This supports only the bounded execution-grounded qualification, not independent verifier value or generalization.

## EVIDENCE
`honest_verdict` `complete_relation_source_tuple_qualification`; `expected_terminal_row_count` `1400`; `observed` `1400`; `reported_vs_recomputed_metrics` `agreement` `true`; `rule:anchored_lexical_v1` `source_grounded_correctness` `1.0` `qualification_decision` `qualified`; `enoki:pinned_openie_encoder` `0.48` `disqualified`; `unsloth/gemma-4-31B-it-GGUF` `0.14` `disqualified`; `unsloth/Qwen3.6-35B-A3B-GGUF` `0.0` `disqualified`; `source_grounded_correct` `false`; `source_grounding_failed_omission`; `verifier_is_oracle` `true`; `verdict_class` `circular_positive`

## RECOMMENDATION
KEEP

## experiment_6914_relation_asp_isomorphic_qualification.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence [', including ', '. Every arm’s ', ', and the shown '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The artifact claims a complete positive relation-ASP isomorphic qualification.

## WHAT WOULD REFUTE IT
A genuinely independent semantic oracle rejecting verifier-qualified outputs, or a serious baseline tying or beating the tested methods, would refute added value; the rule baseline does tie every method on isomorphic invariance.

## WAS THAT CHECKED
No—not non-circularly. Exact execution supplies both the correctness labels and verification, while the second solver merely checks parity on the same compiled programs. Moreover, invariance is calculated only for selected exact-valid pairs; the artifact’s baseline tie and universal arm disqualifications prevent a positive value conclusion.

## EVIDENCE
`honest_verdict` is `complete_circular_positive_relation_asp_isomorphic_qualification`. For `verifier_is_oracle`, the artifact states `True discloses that exact execution supplies the labels.` Every reported `isomorphic_invariance_rate` is `1.0`, including `rule:anchored_lexical_v1`. Every arm’s `qualification_decision` is `disqualified`. `model_inference_call_count` is `0`, and the shown `split` is `calibration`.

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_6915_qualified_relation_event_bank.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The qualified relation event bank is disqualified because its readiness thresholds were not met.

## WHAT WOULD REFUTE IT
A passing readiness gate: at least 90 qualified model relation events, at least 10 from every required model family, coverage of every required constraint family, positive source-group headroom, and a ready score of 1.

## WAS THAT CHECKED
Yes, in `gate_check_summary`, with supporting counts in `model_summary_rows` and `family_summary_rows`; the gate failed rather than the refuting thresholds being observed.

## EVIDENCE
`honest_verdict`: `complete_disqualified_qualified_relation_event_bank_thresholds_not_met`; `qualified_model_relation_event_count`: `8`; `minimum_events`: `90`; `qualified_relation_event_bank_ready_score`: `0`; `failed_check`: `qualified_model_relation_event_count`; `passed`: `false`

## RECOMMENDATION
KEEP

## experiment_6916_isomorphic_prospective_relation_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked pre-gate check and makes no outcome or comparative claim to falsify.

## WAS THAT CHECKED
No; execution stopped at `conductor_pre_gate`, so the method and any rival were not evaluated.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"qualified_relation_event_bank_ready_score"`, `"actual": 0`, `"expected": 1`, `"passed": false`, `"qualified_model_relation_event_count"`, `"actual": 8`, `"expected": 90`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6919_exact_prefix_viability_fixture.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The exact prefix-viability fixture is ready based on successful verifier agreement and gate checks.

## WHAT WOULD REFUTE IT
A prefix whose true viability differs from the engines’ shared decision would refute readiness; alternatively, any disagreement between the two engines would refute parity.

## WAS THAT CHECKED
No for independently established correctness; the engines themselves define correctness. Yes for mutual parity, through `exact_engine_disagreement_count`, but agreement between an oracle and the method scored against that oracle cannot establish added value.

## EVIDENCE
`honest_verdict` `complete_exact_prefix_viability_fixture_ready` `verifier_is_oracle` `true` `verdict_class` `circular_positive` `exact_engine_disagreement_count` `0` `prefix_viability_canary_ready_score` `1` `model_inference_call_count` `0`

## RECOMMENDATION
NARROW_CLAIM

## experiment_6920_sota_exact_guided_relation_generation.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed experiment did not demonstrate utility from exact guidance.

## WHAT WOULD REFUTE IT
Guided generation would need to satisfy the preregistered validity-and-Pareto gate—beating the serious direct-generation and matched unguided-best-of-k controls rather than tying them—so that the utility score became 1.

## WAS THAT CHECKED
Yes. The artifact includes both serious comparator arms, final exact outcomes, resource-cost rows, and an explicit utility gate; the observed utility score was 0. The disclosed oracle status would also prevent these data from supporting an independent positive value claim about the verifier itself.

## EVIDENCE
`"honest_verdict": "complete_null_exact_guidance_utility_not_shown"`; `"exact_guidance_utility_score": 0`; `"exact_guidance_utility_score": "One requires the full preregistered validity and Pareto gate."`; `"arm": "direct_generation"`; `"arm": "unguided_best_of_k"`; `"arm": "guided_frontier"`; `"verifier_is_oracle": "True discloses that an exact solver defines final correctness."`; `"failed_checks": []`

## RECOMMENDATION
KEEP

## experiment_6921_arc_dynamic_supervisor_banked_credit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed audit found insufficient banked-progress evidence to qualify the supervisor arms for policy refinement or an effect claim.

## WHAT WOULD REFUTE IT
A passed frozen evidence floor—such as every evaluated arm reaching the required 10 firings and consequently making the overall banked-credit eligibility score positive—would refute the insufficiency claim.

## WAS THAT CHECKED
Yes. The gate explicitly checked the frozen evidence floor, and the per-arm rows report firing counts and floor status for every arm; all four arms fell short.

## EVIDENCE
`"honest_verdict": "complete_insufficient_banked_progress_evidence"`; `"banked_credit_eligible_score": 0`; `"check": "frozen_evidence_floor_passed"`; `"observed": false`; `"passed": false`; `"min_fired_per_arm": 10`; `"fired": 6`; `"fired": 9`; `"fired": 6`; `"fired": 2`; `"meets_floor": false`; `"recommendations": []`; `"solve_claim": false`; `"verdict_class": "null"`

## RECOMMENDATION
KEEP

## experiment_6922_v605_independent_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
V605 evidence synthesis is complete.

## WHAT WOULD REFUTE IT
An expected task missing or unclassified, an unreadable required source, a present artifact that was not replayable, or any scientific result falsely promoted as positive.

## WAS THAT CHECKED
Yes. The synthesis gate checked all 12 contracts and task states, source readability, replayability, and false promotions; all passed. These checks validate a receipt/completeness claim, not comparative scientific value.

## EVIDENCE
`"claim": "V605 evidence synthesis is complete."`; `"scientific": false`; `"document_yaml_contract"`; `"expected": 12`; `"observed": 12`; `"task_states_classified"`; `"present_artifacts_replayable"`; `"false_promotion_count"`; `"observed": 0`; `"passed": true`; `"verdict_class": "partial"`

## RECOMMENDATION
KEEP
