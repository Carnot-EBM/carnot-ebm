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
| CLAIM_OVERSTATED | 2 |
| NO_CLAIM | 3 |

## experiment_6913_relation_source_tuple_qualification.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The relation-source tuple qualification is complete and positively establishes the anchored lexical rule as qualified.

## WHAT WOULD REFUTE IT
Independent correctness labels disagreeing with the verifier—especially showing the anchored lexical rule below the qualification threshold or tied/beaten by a serious extractor on the same rows—would refute the value claim.

## WAS THAT CHECKED
No. The artifact replayed saved proposals and recomputed verifier-defined metrics, but the verifier itself is the oracle; no independent correctness adjudication was consulted.

## EVIDENCE
`"verifier_is_oracle": true`; `"verdict_class": "circular_positive"`; `"honest_verdict": "complete_relation_source_tuple_qualification"`; `"source_tuple_shard_ready_score": 1`; `"value": "rule:anchored_lexical_v1"`; `"source_grounded_correctness": 1.0`; `"qualification_decision": "qualified"`; `"held_sidecar_access_count": 0`; `"model_inference_call_count": 0`; `"agreement": true`

## RECOMMENDATION
NARROW_CLAIM

## experiment_6914_relation_asp_isomorphic_qualification.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The artifact claims a complete positive qualification of the relation method under ASP-isomorphic verification.

## WHAT WOULD REFUTE IT
An applicable valid pair with failed solver parity or isomorphic invariance would refute execution equivalence; an independent oracle disagreement or a cheap deterministic lexical baseline tying or winning would refute added value.

## WAS THAT CHECKED
No—not with a genuinely independent correctness oracle. Correctness is defined by the verifier itself, and the present `rule:anchored_lexical_v1` comparator ties every tested arm on invariance and solver parity while winning on exact-atom validity and coverage. Every arm is also marked disqualified. The checks could expose implementation disagreement, but could not establish the claimed positive value.

## EVIDENCE
`honest_verdict`: `complete_circular_positive_relation_asp_isomorphic_qualification`; `verifier_is_oracle`: `true`; `inference_substrate`: `deterministic_cpu_asp_isomorphic_qualification_no_llm`; `model_inference_call_count`: `0`; `rule:anchored_lexical_v1`; `isomorphic_invariance_rate`: `1.0`; `solver_parity_rate`: `1.0`; `exact_atom_validity`: `1.0`; `proposal_coverage`: `1.0`; `qualification_decision`: `disqualified`

## RECOMMENDATION
NARROW_CLAIM

## experiment_6915_qualified_relation_event_bank.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The qualified relation event bank is disqualified because its readiness thresholds were not met.

## WHAT WOULD REFUTE IT
All readiness gates passing—including at least 90 qualified model relation events and the required minimum for every model family—would refute the disqualification claim.

## WAS THAT CHECKED
Yes. The explicit gate summary compared expected and observed values, identified all failed checks, and returned a failed overall gate; ineligible controls were marked separately and excluded from admitted model events.

## EVIDENCE
`honest_verdict`: `complete_disqualified_qualified_relation_event_bank_thresholds_not_met`; `qualified_model_relation_event_count`: `8`; `minimum_events`: `90`; `qualified_relation_event_bank_ready_score`: `0`; `failed_check`: `qualified_model_relation_event_count`; `expected`: `>=90`; `observed`: `8`; `passed`: `false`; `control_substitution_count`: `0`; `decision`: `control_only`; `eligible`: `false`; `model_produced`: `false`

## RECOMMENDATION
KEEP

## experiment_6916_isomorphic_prospective_relation_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt and reports no experimental result or comparative claim.

## WAS THAT CHECKED
No; the method was never evaluated because both prerequisite gates failed at `conductor_pre_gate`.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`, `"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"qualified_relation_event_bank_ready_score"`, `"actual": 0`, `"expected": 1`, `"passed": false`, `"qualified_model_relation_event_count"`, `"actual": 8`, `"expected": 90`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6919_exact_prefix_viability_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the operational readiness receipt, any failed gate or disagreement between the two exact engines would refute readiness; no comparative method-value claim is made.

## WAS THAT CHECKED
Yes, through the gate checks and exact-engine parity rows; both engines could have disagreed, and failed checks were recordable.

## EVIDENCE
`honest_verdict`: `complete_exact_prefix_viability_fixture_ready`; `exact_engine_disagreement_count`: `0`; `failed_checks`: `[]`; `observed`: `all checks pass`; `verifier_is_oracle`: `true`; `verdict_class`: `circular_positive`; `model_inference_call_count`: `0`

## RECOMMENDATION
KEEP

## experiment_6920_sota_exact_guided_relation_generation.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Exact guidance utility was not demonstrated.

## WHAT WOULD REFUTE IT
A value of 1 for the preregistered utility score—showing guided generation beat the matched best-of-k comparator on validity while satisfying the Pareto gate—would refute the null claim.

## WAS THAT CHECKED
Yes. The artifact evaluates the guided arm against matched unguided and direct-generation arms, records final exact outcomes and resource costs, and applies the preregistered utility gate.

## EVIDENCE
`honest_verdict` is `complete_null_exact_guidance_utility_not_shown`; `exact_guidance_utility_score` is `0`; the compared arms include `guided_frontier`, `unguided_best_of_k`, and `direct_generation`; `guided_generation_run_complete_score` is `1`; `verifier_is_oracle` is `true`.

## RECOMMENDATION
KEEP

## experiment_6921_arc_dynamic_supervisor_banked_credit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed audit found insufficient banked-progress evidence to support an effect or policy-refinement claim.

## WHAT WOULD REFUTE IT
A passed frozen evidence floor—supported by adequate per-arm firing counts and banked progress—would refute the claim of insufficient evidence.

## WAS THAT CHECKED
Yes. `gate_check_summary` tests `frozen_evidence_floor_passed`, and `per_arm_rows` reports each arm’s floor status and shortfall.

## EVIDENCE
`honest_verdict`: `complete_insufficient_banked_progress_evidence`; `banked_credit_eligible_score`: `0`; `failed_check`: `frozen_evidence_floor_passed`; `observed`: `false`; `passed`: `false`; `meets_floor`: `false`; `floor_shortfall`: `4`; `floor_shortfall`: `1`; `floor_shortfall`: `8`; `verdict_class`: `null`; `solve_claim`: `false`

## RECOMMENDATION
KEEP

## experiment_6922_v605_independent_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A missing task classification, failed synthesis check, unreadable required source, unreplayable present artifact, or false promotion of a scientific result would refute the administrative assertion that the evidence synthesis is complete.

## WAS THAT CHECKED
Yes. The global preconditions and synthesis gate summary checked source readability, all 12 task classifications, replayability of present artifacts, and false-promotion count.

## EVIDENCE
`V605 evidence synthesis is complete.`; `scientific`: `false`; `all synthesis checks pass`; `task_states_classified`; `present_artifacts_replayable`; `false_promotion_count`: `0`; `complete_partial_v605_evidence_synthesized_without_science_promotion`

## RECOMMENDATION
KEEP
