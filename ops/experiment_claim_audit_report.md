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
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 3 |

## experiment_6872_bounded_reliability_controller_quarantine.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6873_prospective_sealed_self_learning_audit.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6874_v602_evidence_substrate_manifest_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific claim to falsify. If the operational blocked status were treated as a claim, it would be refuted by full document/YAML parity, a passing gate contract, and a readiness score of 1.

## WAS THAT CHECKED
Yes. The artifact checked document/YAML parity, the V602 gate contract, and the evidence-contract readiness score; all three failed.

## EVIDENCE
`honest_verdict`: `complete_blocked_v602_evidence_substrate_manifest_contract`; `inference_substrate`: `aggregation_from_upstream_artifacts_no_llm`; `verdict_class`: `blocked`; `failed_check`: `v602_document_yaml_parity`; `all_fields_equal`: `false`; `yaml_task_count`: `4`; `document_task_count`: `11`; `v602_evidence_contract_ready_score`: `0`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6875_text_anchored_relation_asp_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a blocked-gate receipt and makes no outcome or comparative claim.

## WAS THAT CHECKED
No; the experiment did not run past the upstream gate. It only recorded the gate failure.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6885_v603_executable_manifest_branch_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or scientific value claim is made. Treating the operational blocked status as an assertion, it would be refuted by all 13 tasks appearing in executable YAML with every contract check passing and the readiness score equal to 1.

## WAS THAT CHECKED
Yes, in `gate_check_summary`, `hard_failures`, and `v603_manifest_contract_ready_score`; the observed contract failed rather than passed.

## EVIDENCE
`inference_substrate`: `deterministic_manifest_contract_no_llm`; `status`: `complete_blocked`; `verdict_class`: `blocked`; `document_task_count`: `13`; `yaml_task_count`: `4`; `all_task_fields_equal`: `false`; `v603_manifest_contract_ready_score`: `0`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6886_enoki_exact_relation_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no accuracy, generalization, or comparative-value claim to refute. If fixture readiness is treated as an operational assertion, any failed gate, solver disagreement, revision mismatch, atom collision, split overlap, or prompt-nonexposure failure would falsify it.

## WAS THAT CHECKED
Yes. The artifact reports gate checks, revision checks, solver parity across 150 fixture calls, collision rows, split overlap, and prompt-nonexposure results. These checks could have recorded failures, but they do not test Enoki accuracy or added value.

## EVIDENCE
`honest_verdict` `complete_enoki_exact_relation_fixture_ready_no_accuracy_claim` `enoki_accuracy_claimed` `false` `encoder_loaded` `false` `llm_inference_count` `0` `model_loaded` `false` `no_llm_or_encoder_inference` `true` `verifier_is_oracle` `true` `disagreement_count` `0` `fixture_calls` `150` `split_overlap_count` `0` `failed_checks` `[]`

## RECOMMENDATION
KEEP

## experiment_6887_three_family_relation_proposal_corpus.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6888_independent_relation_qualification.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The artifact claims a positive independent relation qualification, with one arm producing 90 qualified events and satisfying the readiness gate.

## WHAT WOULD REFUTE IT
An independent correctness oracle showing that the qualifying arm fails the frozen thresholds—or that the same anchored-lexical pipeline without the verifier ties its results—would refute the claimed independent value.

## WAS THAT CHECKED
No. Threshold failures were possible and occurred for other arms, but the decisive correctness check used the verifier that defines semantic validity; no independently scored no-verifier comparator tested its added value.

## EVIDENCE
`honest_verdict` `complete_circular_positive_independent_relation_qualification` `verdict_class` `circular_positive` `verifier_is_oracle` `true` `relation_qualification_ready_score` `1` `qualified_relation_event_count` `90` `rule:anchored_lexical_v1` `passed` `true` `exact_semantic_parity` `1.0` `held_leakage_count` `0` `sealed_reduction_of_frozen_relation_outputs_no_llm`

## RECOMMENDATION
NARROW_CLAIM
