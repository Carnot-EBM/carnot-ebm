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
| CLAIM_SUPPORTED | 1 |
| CLAIM_OVERSTATED | 2 |
| NO_CLAIM | 5 |

## experiment_7672_v669_bound_relations.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The bound relation protocol has achieved complete readiness based on passing validation across 72 fixture groups.

## WHAT WOULD REFUTE IT
A non-zero count of false support or wrong unknown classifications when evaluated against an independent ground-truth oracle on natural, unexposed data.

## WAS THAT CHECKED
no. All 72 evaluated fixture groups used constructed truth where the verifier was its own oracle (`verifier_is_oracle: true`), while evaluation on natural instances was unhandled and unpassed (`fresh_natural_groups: 0` and the freshness gate passed value is `null`).

## EVIDENCE
- `verifier_is_oracle`: `true`
- `verdict_class`: `"circular_positive"`
- `honest_verdict`: `"complete_circular_positive_bound_relation_protocol_ready"`
- `relation_protocol_ready_score`: `1`
- `provenance`: `"exact_fixture_oracle"`
- `oracle_scope`: `"exact_fixture_only"`
- `prior_exposure`: `"Eight V664 pilots exposed before this run; fixture oracle truth is constructed."`
- `limits`: `"No natural-answer accuracy or learned-verifier advantage."`
- `fresh_natural_groups`: `0`
- `passed`: `null`
- `principle`: `"Fixtures and exposed pilots cannot establish natural accuracy."`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7673_v669_fresh_relation_cohort.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation claiming a learned-verifier benefit or successful validation despite failing acceptance gates (including validity, readiness, and freshness) and failing required coverage checks.

## WAS THAT CHECKED
Yes; acceptance gates and required checks were executed. The artifact recorded failures for validity (`gate: "validity"`, `passed: false`), readiness (`gate: "readiness"`, `passed: false`), freshness (`gate: "freshness"`, `observed: 0`), and coverage (`changed_module_coverage_report`, `passed: false`), resulting in a disqualified outcome.

## EVIDENCE
`claim_scope`
`"cohort infrastructure only; no learned-verifier improvement"`
`honest_verdict`
`"complete_disqualified_required_validation"`
`verdict_class`
`"disqualified"`
`model_invoked`
`false`
`required_checks_passed`
`false`
`"Completion describes the task, not a verifier benefit."`

## RECOMMENDATION
KEEP

## experiment_7674_relation_energy.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact asserts no empirical or comparative claim to refute, serving solely as an execution receipt recording that upstream gate checks failed before the experiment could execute.

## WAS THAT CHECKED
No; no experimental evaluation or comparative run took place because execution was halted at `conductor_pre_gate`.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `blocked_at_layer`: `conductor_pre_gate`
- `duration_s`: `0.0`
- `gate_check_summary`: `gate-unsat(final): 3 of 4 gate(s) failed; first failure: exp7673-fresh-relation-cohort.relation_features_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_7676_v669_qwen_quote_relations.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The exact quote prompt mechanism fails required acceptance checks and is disqualified from production promotion and retired.

## WHAT WOULD REFUTE IT
Observing non-zero supported relations (`supported_relations > 0` or `full_proposition_supported > 0`) or passing acceptance gates (`passed: true` across required gates) in the artifact's own rows would refute the disqualification claim.

## WAS THAT CHECKED
Yes; 48 generation calls were executed across 24 units in paired arms (`exact_quote` and `numeric_offset`), evaluated against 8 acceptance gates, and recorded in `acceptance_gate_results`, `proposal_metrics`, and `rows`.

## EVIDENCE
- `honest_verdict`: `complete_disqualified_required_checks`
- `verdict_class`: `disqualified`
- `activation`: `false`
- `production_promotion`: `false`
- `quote_pilot_measurement_complete_score`: `0`
- `acceptance_gate_results`:
  - `coverage`: `measured_operands`: `supported_relations`: `0`, `passed`: `false`
  - `readiness`: `measured_operands`: `supported_relations`: `0`, `passed`: `false`
  - `validity`: `passed`: `false`
- `proposal_metrics`:
  - `full_proposition_supported`: `0`
  - `schema_valid`: `27`
- `retirement_decision`:
  - `applied`: `true`
  - `scope`: `exact_quote_prompt_mechanism`
- `sample_size_budget`:
  - `independent_unit`: `source_group`
  - `observed`: `24`
- `invocation_counts`:
  - `forward_calls_completed`: `48`
  - `generation_calls_completed`: `48`

## RECOMMENDATION
KEEP

## experiment_7679_v669_independent_evidence_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Authenticated fresh static and online evidence passing the stated gates would contradict the audit’s blocked disposition. The artifact makes no comparative benefit claim.

## WAS THAT CHECKED
Yes for the blocked disposition: the source inventory and acceptance gates record missing evidence and failed gates. No comparative benefit test is reported.

## EVIDENCE
`honest_verdict`: `complete_blocked_required_static_online_evidence`; `independent_quality_confirmed_gates`: `[]`; `independent_quality_confirmed_score`: `0`; `activation`: `false`; `production_promotion`: `false`; `missing_evidence`; `passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_7680_v669_arc_probe_protocol.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The ARC probe protocol achieves a complete positive verdict for probe readiness and generalization with zero false goal confirmations.

## WHAT WOULD REFUTE IT
An independent external oracle or live game environment demonstrating non-zero false goal confirmations, probe rejection failures, or probe-guided policies failing to achieve decision utility over baselines on unobserved ARC tasks.

## WAS THAT CHECKED
No. The evaluation was run exclusively on scripted CPU proxy fixtures where the verifier served as its own oracle; no model was invoked, and live hidden game validation was not performed.

## EVIDENCE
`honest_verdict`
`complete_circular_positive_probe_fixture`
`verdict_class`
`circular_positive`
`verifier_is_oracle`
`true`
`arc_probe_protocol_ready_score`
`1`
`arc_generalization_task`
`true`
`model_invoked`
`false`
`inference_substrate`
`cpu_scripted_scored_wrapper_no_model_load`
`decision_utility`
`passed`
`false`
`new_hidden_game_wins`
`0`
`independent_hidden_games`
`prior_exposure`
`scripted development proxy; no hidden-game inference`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7681_v669_arc_live_probes.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An assertion of empirical task performance, benchmark solving, or comparative capability would be refuted by the absence of model execution, executed actions, or novel target games; however, no such claim is advanced by this receipt.

## WAS THAT CHECKED
Yes. Preflight gate checks were executed (`preconditions_checked`, `gate_check_summary`), which determined that no eligible novel SDK targets remained and blocked execution before any model was loaded or evaluated.

## EVIDENCE
`"honest_verdict"`
`"complete_blocked_eligible_novel_target"`
`"verdict_class"`
`"blocked"`
`"counterfactual_solve_rate_claim"`
`false`
`"solve_provenance"`
`"no_live_attempt"`
`"model_invoked"`
`false`
`"inference_substrate"`
`"host_preflight_no_model_call"`
`"registry_credit_granted"`
`false`
`"rows"`
`[]`
`"check"`
`"eligible_novel_target"`
`"observed"`
`false`
`"passed"`
`false`

## RECOMMENDATION
KEEP

## experiment_7684_v669_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is made. The artifact records accounting completion and blocked scientific gates, so there is no benefit claim for its own data to refute.

## WAS THAT CHECKED
No comparative refutation applies. The artifact checks the gates and records that the required fresh comparisons and intervals are unavailable.

## EVIDENCE
`verdict_class`: `blocked`; `honest_verdict`: `complete_blocked_required_v669_scientific_evidence`; `paired_brier_improvement_ci`: `null`; `paired_cost_improvement_ci`: `null`; `inference_substrate`: `aggregation_from_upstream_artifacts_no_llm`.

## RECOMMENDATION
KEEP
