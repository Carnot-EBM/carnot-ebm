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

## experiment_6919_exact_prefix_viability_fixture.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The deterministic dual-engine exact-prefix fixture is complete and ready as a viability canary.

## WHAT WOULD REFUTE IT
Any disagreement between the independent exact engines, a failed readiness gate, missing required case coverage, or fixtures that cannot distinguish extendable from impossible prefixes would refute readiness.

## WAS THAT CHECKED
Yes. Engine parity was checked across prefix cases, readiness gates were evaluated, required case types and counts were recorded, and the rows include both extendable and rejected prefixes. The oracle-defined labels support fixture readiness only—not any claim that verification adds downstream value.

## EVIDENCE
`"honest_verdict": "complete_exact_prefix_viability_fixture_ready"`; `"verifier_is_oracle": true`; `"verdict_class": "circular_positive"`; `"exact_engine_disagreement_count": 0`; `"implementation_independent": true`; `"prefix_case_count": 330`; `"expected": "all checks pass"`; `"observed": "all checks pass"`; `"passed": true`; `"final_engine_extendable": true`; `"final_engine_extendable": false`; `"model_inference_call_count": 0`; `"train_free": true`

## RECOMMENDATION
KEEP

## experiment_6920_sota_exact_guided_relation_generation.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Exact guidance utility was not demonstrated.

## WHAT WOULD REFUTE IT
Guided frontier generation satisfying the preregistered utility and Pareto gate—most importantly, reliably outperforming direct generation and matched unguided best-of-k on final exact validity without losing on required cost criteria—would refute the null claim.

## WAS THAT CHECKED
Yes. The artifact computes the utility gate, compares guided frontier against both direct generation and unguided best-of-k, and evaluates selected outputs with a separate final exact engine. The displayed cell instead shows all three arms producing the same valid program, while guidance incurs additional feasibility computation. The oracle circularity prevents a positive verifier-value interpretation, but does not undermine this observed null.

## EVIDENCE
`honest_verdict`: `complete_null_exact_guidance_utility_not_shown`; `exact_guidance_utility_score`: `0`; `guided_generation_run_complete_score`: `1`; `direct_generation`; `unguided_best_of_k`; `guided_frontier`; `exact_final_valid`: `true`; `clingo_stable_model_final_engine_v1`; `python_bounded_relation_enumerator_v1`; `set_19_alpha selects option_a`; `set_19_beta selects option_b`; `energy_proxy`: `0`; `energy_proxy`: `4`

## RECOMMENDATION
KEEP

## experiment_6921_arc_dynamic_supervisor_banked_credit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed audit found insufficient banked-progress evidence to support an effect claim.

## WHAT WOULD REFUTE IT
The frozen evidence floor passing—such as relevant arms meeting the required receipt count—and consequently establishing eligibility for banked-credit evidence would refute the headline.

## WAS THAT CHECKED
Yes. The gate summary explicitly tested the frozen evidence floor, and the per-arm rows report whether each arm met it; the check failed and every reported arm fell short.

## EVIDENCE
`honest_verdict` `complete_insufficient_banked_progress_evidence` `frozen_evidence_floor_passed` `observed` `false` `passed` `false` `min_fired_per_arm` `10` `meets_floor` `false` `banked_credit_eligible_score` `0` `verdict_class` `null`

## RECOMMENDATION
KEEP

## experiment_6922_v605_independent_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific or comparative claim is promoted; treating the receipt assertion alone, it would fail if required synthesis checks failed, task states were unclassified, present artifacts were not replayable, or scientific results were falsely promoted.

## WAS THAT CHECKED
Yes. The synthesis gate checks document/YAML coverage, task classification, replayability, and false promotions; all passed. The artifact separately records blocked, null, circular, skipped, and disqualified scientific outcomes without promoting them.

## EVIDENCE
`V605 evidence synthesis is complete.`; `scientific`: `false`; `false_promotion_count`: `0`; `all synthesis checks pass`; `verdict_class`: `partial`; `complete_partial_v605_evidence_synthesized_without_science_promotion`

## RECOMMENDATION
KEEP

## experiment_6923_v606_lifecycle_evidence_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific or comparative claim is asserted. Operationally, a fully matching 14-task YAML contract with every required check passing would contradict the blocked status.

## WAS THAT CHECKED
Yes, operational readiness was checked in `gate_check_summary` and `rows`; the artifact found only 4 of 14 expected YAML tasks and multiple failed checks. No method-value claim required a rival or generalization test.

## EVIDENCE
`science_claim_approved`: `false`; `model_bearing`: `false`; `status`: `complete_blocked`; `verdict_class`: `blocked`; `v606_execution_contract_ready_score`: `0`; `yaml_executable_contract`; `task_count`: `14`; `task_count`: `4`; `passed`: `false`

## RECOMMENDATION
KEEP

## experiment_6924_task_runtime_receipt_adoption.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is made; operational readiness would fail if fresh-process validation rejected the receipt, ownership or phase order were invalid, teardown were incomplete, or forged receipts were accepted.

## WAS THAT CHECKED
Yes. Fresh-process validation checked ownership, phase order, and teardown, while four forged-receipt mutations tested rejection paths.

## EVIDENCE
`honest_verdict` `complete_null_task_runtime_receipt_adoption_ready` `verdict_class` `null` `inference_substrate` `deterministic_runtime_receipt_fixture_no_llm` `accepted` `true` `ownership_valid` `true` `phase_order_valid` `true` `teardown_complete` `true` `rejected` `true` `verifier_is_oracle` `false`

## RECOMMENDATION
KEEP

## experiment_6925_v606_sota_ingestion.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the receipt’s completion statement: fewer than 15 terminal source families or candidates, any failed gate, or other than exactly one newly appended compatibility finding.

## WAS THAT CHECKED
Yes. `gate_check_summary` compares the required and observed terminal counts, and `ledger_append_rows` records the single new finding. These checks establish process completion, not comparative method value.

## EVIDENCE
`honest_verdict`: `complete_v606_sota_ingestion_with_one_new_compatibility_finding`; `expected`: `15 terminal source families and 15 terminal named candidates`; `observed`: `15 source families and 15 candidates`; `all_gates_passed`: `true`; `failed_check`: `null`; `status`: `complete`; `ledger_append_rows`; `ISM reference implementation compatibility update`; `inference_substrate`: `bounded_primary_source_web_research_no_model_inference`

## RECOMMENDATION
KEEP

## experiment_6926_span_first_relation_fixture.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The span-first relation fixture is fully qualified and ready based on complete, exact verification.

## WHAT WOULD REFUTE IT
A valid row with an incorrect canonical tuple, an unmet expected ASP effect, failed solver parity, or a failed gate would refute execution correctness; an independent non-oracle adjudicator or a serious direct rule-based extractor tying or beating the span-first method would refute added value.

## WAS THAT CHECKED
No. Internal consistency and solver parity were checked, but the correctness oracle defines the expected result, no independent semantic adjudicator or serious baseline is reported, and the deterministic substrate makes the held-out partition irrelevant to generalization. Rows with `valid` set to `false` are expected rejection cases and do not refute the headline.

## EVIDENCE
`honest_verdict` is `complete_circular_positive_span_first_relation_fixture`; `span_relation_fixture_ready_score` is `1`; `verdict_class` is `circular_positive`; `verifier_is_oracle` is `true`; `inference_substrate` is `deterministic_cpu_span_first_fixture_no_llm`; the displayed ASP rows report `expected_effect_met` as `true` and `solver_parity` as `true`; `gate_check_summary` reports `passed` as `true`.

## RECOMMENDATION
NARROW_CLAIM
