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

## experiment_6968_arc_post_refit_induction_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A positive generalization claim would be refuted by held-out accuracy that ties or loses to a serious matched control, or by a positive generalization gap showing prefix fit does not transfer.

## WAS THAT CHECKED
No. The audit stopped after the immutable-transition-source precondition failed, before producing execution, transition, purity, or control comparisons.

## EVIDENCE
`"honest_verdict": "blocked_arc_post_refit_induction_audit"`; `"verdict_class": "blocked"`; `"arc_induction_audit_complete_score": 0`; `"arc_induction_generalization_positive_score": 0`; `"failed_check": "immutable_transition_source"`; `"observed_value": false`; `"heldout_exact_accuracy": null`; `"generalization_gap": null`; `"control_rows": []`; `"engine_execution_rows": []`; `"per_transition_rows": []`

## RECOMMENDATION
KEEP

## experiment_7009_v614_source_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V614 preflight is disqualified because the Markdown and YAML task contracts do not conform.

## WHAT WOULD REFUTE IT
A successfully parsed YAML contract containing all 14 expected tasks in IDs 7009–7022, producing `v614_task_contract_conforms_score` equal to 1, would refute the disqualification.

## WAS THAT CHECKED
Yes. The task-contract gate, expected-versus-observed counts and ID order, and direct YAML parsing assertion tested this possibility; conformity failed.

## EVIDENCE
`honest_verdict`: `disqualified_v614_markdown_yaml_contract_mismatch`; `verdict_class`: `disqualified`; `expected_task_count`: `14`; `observed_task_count`: `7`; `v614_task_contract_conforms_score`: `0`; `failed_check`: `task_contract`; `expected_value`: `1`; `observed_value`: `0`; `yaml_parsing`; `exit_code`: `1`; `AssertionError`

## RECOMMENDATION
KEEP

## experiment_7010_arc_eval_provenance_contract.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The ARC evaluation provenance contract is ready: producer, consumer, rejection, round-trip, and row-eligibility gates pass.

## WHAT WOULD REFUTE IT
Acceptance of a missing or aliased required field, rejection of a valid fixture, a producer/consumer schema mismatch, broken wiring, an unstable round-trip hash, or an ineligible provenance row entering the headline would refute the claim.

## WAS THAT CHECKED
Yes. The artifact includes accepted fixtures, missing-field and alias rejection rows, producer and consumer wiring checks, gate summaries, row eligibility, and a fresh-process hash receipt. These checks gave the contract multiple concrete ways to fail.

## EVIDENCE
`arc_eval_provenance_contract_ready_score`: `1`; `honest_verdict`: `positive_arc_eval_provenance_contract_ready`; `verifier_is_oracle`: `false`; `producer_wired`: `true`; `wired`: `true`; `round_trip`: `true`; `fresh_process_stable_hash`; `passed`: `true`; `accepted`: `false`; `missing required field: gpu_uuid`; `unknown or aliased field: gpu_id`; `headline_eligible`: `true`; `historical_artifacts_modified`: `false`

## RECOMMENDATION
KEEP

## experiment_7011_v614_sota_ingestion.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative efficacy claim is made. If method value were inferred from the positive receipt, a serious matched baseline tying or winning on held-out valid rows would refute it.

## WAS THAT CHECKED
No. The artifact names control arms and prospective failure conditions but reports no comparative outcome rows; it only documents source ingestion and mechanism mapping.

## EVIDENCE
`command_receipt_rows`; `[]`; `deterministic_primary_source_ingestion_no_llm`; `This task verifies sources and maps mechanisms; it runs no paper training or evaluation.`; `No belief-state policy experiment runs in this ingestion task.`; `Only public software and model metadata were inspected.`

## RECOMMENDATION
KEEP

## experiment_7012_exact_intervention_pair_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
If treated solely as a fixture-readiness receipt, a failed gate, missing expected pair or family, nonterminal authority result, authority disagreement, nonminimal intervention, or invalid balance row would refute readiness; no comparative method-value claim is made to falsify.

## WAS THAT CHECKED
Yes for fixture readiness: the gate summary, expected-versus-observed counts, authority witnesses, and per-block validity checks cover those failure conditions. No model, rival method, or added-value comparison was tested.

## EVIDENCE
`honest_verdict` is `circular_positive: exact_intervention_pair_fixture_ready`; `verdict_class` is `circular_positive`; `inference_substrate` is `deterministic_exact_pair_fixture_no_llm`; `verifier_is_oracle` is `true`; `expected_pair_count` is `48`; `observed_pair_count` is `48`; `observed_value` is `all checks pass`; `rejected_block_rows` is `[]`.

## RECOMMENDATION
KEEP

## experiment_7013_three_family_intervention_surface.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed experiment found no consistent signed intervention response across all three model families.

## WHAT WOULD REFUTE IT
A consistently directed primary effect across all three families—most concretely, three `primary_signed_effect` confidence intervals excluding zero in the predicted positive direction—would refute the cross-family null claim.

## WAS THAT CHECKED
Yes. `family_effect_rows` reports separate 48-pair primary-effect estimates and bootstrap confidence intervals for each model family, allowing positive, null, or reversed results. Only one family’s interval excludes zero positively; the other two include zero, and one has a negative mean.

## EVIDENCE
`honest_verdict` `complete_null_signed_intervention_response` `verdict_class` `null` `family_effect_rows` `pair_count` `48` `primary_direction` `positive` `mean` `0.00882860856814365` `ci_low` `-0.10367785134917545` `ci_high` `0.12351360803439296` `primary_direction` `reversal` `mean` `-0.036006200690215207` `ci_low` `-0.33197466958897237` `ci_high` `0.2702579490081689` `primary_direction` `positive` `mean` `0.12453014122100646` `ci_low` `0.016900694855811915` `ci_high` `0.23567172757030663` `pooled_families` `false` `observed_family_count` `3` `expected_family_count` `3` `failed_cell_rows` `[]` `verifier_is_oracle` `false`

## RECOMMENDATION
KEEP

## experiment_7014_causal_feature_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The causal feature bank fails its release rules and is not ready.

## WHAT WOULD REFUTE IT
A completed audit in which the release score is positive—particularly with the required identifiable-family floor met, nuisance-predictability bounds acceptable, and invariance checks passing—would refute the disqualification.

## WAS THAT CHECKED
Yes. The artifact reports the release score and its underlying family-identifiability, nuisance-probe, leakage, and invariance results; these checks could have passed but instead produced multiple release failures.

## EVIDENCE
`causal_bank_audit_complete_score`: `1`; `causal_feature_bank_ready_score`: `0`; `honest_verdict`: `disqualified: causal_feature_bank_release_rules_failed`; `identifiable_family_count`: `0`; `identifiable`: `false`; `identifiable_positive_direction`: `false`; `prohibited_auroc_upper_bound_max`: `1.0`; `direct_leakage_count`: `0`; `passed`: `false`; `verifier_is_oracle`: `False states that the audit does not define exact correctness.`

## RECOMMENDATION
KEEP

## experiment_7015_pair_centered_pwa_kan.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no scientific headline to falsify. The gate-failure receipt would be refuted by an observed value of 1 and a passing gate.

## WAS THAT CHECKED
Yes, in `gates_evaluated`; the sole prerequisite gate was evaluated and failed.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_expected": 1`, `"failed_observed": 0`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP
