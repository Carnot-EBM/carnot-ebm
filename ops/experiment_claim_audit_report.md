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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 2 |

## experiment_6885_v603_executable_manifest_branch_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or value claim is made; treating the operational blocked status as a claim, all manifest-contract checks passing and the readiness score being 1 would refute it.

## WAS THAT CHECKED
Yes. The contract checks could pass or fail, and multiple checks failed, including task parity, executable completeness, gates, branch isolation, and terminal-tail checks.

## EVIDENCE
`inference_substrate` `deterministic_manifest_contract_no_llm` `verifier_is_oracle` `false` `status` `complete_blocked` `verdict_class` `blocked` `v603_manifest_contract_ready_score` `0` `document_yaml_task_contract` `passed` `false` `document_task_count` `13` `yaml_task_count` `4`

## RECOMMENDATION
KEEP

## experiment_6886_enoki_exact_relation_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or accuracy claim is made. The narrower readiness receipt would fail if any required gate failed, solver disagreement occurred, asset revisions mismatched, or calibration and held groups overlapped.

## WAS THAT CHECKED
Yes. The artifact reports gate outcomes, solver disagreements, immutable revision checks, and split overlap; these checks could have produced failures.

## EVIDENCE
`honest_verdict` `complete_enoki_exact_relation_fixture_ready_no_accuracy_claim` `enoki_accuracy_claimed` `false` `encoder_loaded` `false` `llm_inference_count` `0` `disagreement_count` `0` `split_overlap_count` `0` `verifier_is_oracle` `true`

## RECOMMENDATION
KEEP

## experiment_6887_three_family_relation_proposal_corpus.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6888_independent_relation_qualification.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The anchored lexical method achieved positive independent relation qualification on held data.

## WHAT WOULD REFUTE IT
Independent, non-verifier ground truth disagreeing with enough accepted relations to make the anchored lexical arm fail a frozen threshold—or a serious independently implemented baseline tying or beating it—would refute the claim of added value.

## WAS THAT CHECKED
No. Threshold failure was possible and checked, but correctness was defined by the verifier itself; no independent oracle or competitive anchored baseline tested the value claim.

## EVIDENCE
`honest_verdict`: `complete_circular_positive_independent_relation_qualification`; `verdict_class`: `circular_positive`; `verifier_is_oracle`: `true`; `reference_arm`: `rule:anchored_lexical_v1`; `passed`: `true`; `failed_thresholds`: `[]`; `qualified_event_count`: `90`; `relation_qualification_ready_score`: `1`; `held_leakage_count`: `0`.

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_6898_v604_evidence_admissibility_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific headline claim to falsify; this is a blocked structural audit receipt.

## WAS THAT CHECKED
No substantive claim was tested. The artifact checked manifest and evidence-admissibility preconditions and recorded their failure.

## EVIDENCE
`inference_substrate`: `deterministic_manifest_and_admissibility_audit_no_llm`; `status`: `complete_blocked`; `verdict_class`: `blocked`; `v604_manifest_contract_ready_score`: `0`; `v603_admissible_science_source_count`: `0`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6899_live_relation_acquisition_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative, accuracy, or added-value claim to falsify. The narrow operational readiness assertion would be refuted by missing live cells, unauthentic execution, empty outputs, absent parse attempts, or failed readiness gates.

## WAS THAT CHECKED
Yes. The artifact checks the live-cell matrix, execution authenticity, empty outputs, parser attempts, and readiness gates. It also exposes imperfect parse coverage rather than claiming semantic accuracy. The failed Enoki arm is not used to claim superiority.

## EVIDENCE
`complete_positive_live_relation_acquisition_canary_ready`; `relation_canary_ready_score`; `1`; `exact_live_cell_matrix`; `count`; `60`; `live_cell_authenticity_errors`; `observed`; `{}`; `empty_cell_count`; `0`; `parser_attempted`; `true`; `parse_coverage`; `0.65`; `0.95`; `1.0`; `enoki_accuracy_claimed`; `false`; `complete_enoki_exact_relation_fixture_ready_no_accuracy_claim`; `verifier_is_oracle`; `false`

## RECOMMENDATION
KEEP

## experiment_6900_authentic_anchored_relation_corpus.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6901_independent_model_relation_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked qualification run rather than a positive, comparative, or generalization claim.

## WAS THAT CHECKED
No comparative claim was checked: admission failed before held scoring, no model inference occurred, and no evaluation rows were produced.

## EVIDENCE
`status` `blocked` `verdict_class` `blocked` `no_model_inference` `true` `held_sidecar_open_count` `0` `model_eligible_arm_rows` `[]` `rows` `[]` `qualified_model_relation_event_count` `0` `model_relation_qualification_ready_score` `0`

## RECOMMENDATION
KEEP
