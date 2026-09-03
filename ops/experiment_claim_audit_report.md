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
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6921_arc_dynamic_supervisor_banked_credit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The available live receipts contain insufficient banked-progress evidence to qualify any supervisor arm under the frozen evidence floor.

## WHAT WOULD REFUTE IT
A passed frozen evidence floor—such as qualifying support for an arm, yielding banked-credit eligibility—would falsify the insufficiency claim.

## WAS THAT CHECKED
Yes. The gate explicitly tested the frozen evidence floor, and the per-arm rows report floor attainment and shortfall for every arm.

## EVIDENCE
`honest_verdict` is `complete_insufficient_banked_progress_evidence`; `frozen_evidence_floor_passed` has `observed` `false` and `passed` `false`; `failed_check` is `frozen_evidence_floor_passed`; `banked_credit_eligible_score` is `0`; every shown `meets_floor` value is `false`; `min_fired_per_arm` is `10`, while the arm-level `fired` values are `6`, `9`, `6`, and `2`.

## RECOMMENDATION
KEEP

## experiment_6922_v605_independent_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V605 evidence synthesis is complete, without promoting unsupported scientific conclusions.

## WHAT WOULD REFUTE IT
Missing or unclassified planned tasks, unreadable source artifacts, failed document/YAML contract checks, unreplayable present artifacts, or any unsupported scientific result marked as promoted would refute the claim.

## WAS THAT CHECKED
Yes. The global preconditions and synthesis gate checks covered source readability, all 12 document/YAML contracts, classification of all 12 task states, replayability of present artifacts, and false scientific promotions.

## EVIDENCE
`"claim": "V605 evidence synthesis is complete."`; `"scientific": false`; `"promoted": true`; `"check": "document_yaml_contract"`; `"expected": 12`; `"observed": 12`; `"check": "task_states_classified"`; `"check": "present_artifacts_replayable"`; `"check": "false_promotion_count"`; `"observed": 0`; `"passed": true`; `"honest_verdict": "complete_partial_v605_evidence_synthesized_without_science_promotion"`

## RECOMMENDATION
KEEP

## experiment_6923_v606_lifecycle_evidence_contract.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V606 lifecycle-aware execution and evidence contract is blocked and not execution-ready.

## WHAT WOULD REFUTE IT
All required contract checks passing, including 14 executable YAML tasks matching the 14 documented tasks, with no failed contract rows and a nonzero readiness score.

## WAS THAT CHECKED
Yes. The gate summary directly compared the documented contract with the executable YAML contract and recorded the failing checks; the refuting all-pass outcome was possible but did not occur.

## EVIDENCE
`honest_verdict`: `complete_blocked_v606_lifecycle_evidence_contract`; `status`: `complete_blocked`; `verdict_class`: `blocked`; `v606_execution_contract_ready_score`: `0`; `failed_check`: `yaml_executable_contract`; expected `task_count`: `14`; observed `task_count`: `4`; `document_yaml_task_contract` observed `false`; `prompt_contract` observed `false`; `science_claim_approved`: `false`.

## RECOMMENDATION
KEEP

## experiment_6924_task_runtime_receipt_adoption.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is an operational runtime-receipt validation artifact, not a comparative or scientific value claim.

## WAS THAT CHECKED
No—there is no headline hypothesis or comparator to refute; only receipt validity and adoption readiness were checked.

## EVIDENCE
`honest_verdict`: `complete_null_task_runtime_receipt_adoption_ready`; `verdict_class`: `null`; `inference_substrate`: `deterministic_runtime_receipt_fixture_no_llm`; `verifier_is_oracle`: `false`; `The class separates advisory readiness from a positive science result.`

## RECOMMENDATION
KEEP

## experiment_6925_v606_sota_ingestion.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative method-value claim is made; this is a research-ingestion receipt, so there is no method-performance observation to falsify.

## WAS THAT CHECKED
No; not applicable. The artifact records source-review completion and compatibility findings, not a method-versus-rival test.

## EVIDENCE
`honest_verdict`: `complete_v606_sota_ingestion_with_one_new_compatibility_finding`; `inference_substrate`: `bounded_primary_source_web_research_no_model_inference`; `invoked_models`: `[]`; `status`: `complete`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6926_span_first_relation_fixture.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The span-first relation fixture is positively qualified and ready based on complete, balanced, exact held-out evidence.

## WHAT WOULD REFUTE IT
An independently produced held-out relation tuple failing the oracle, or a cheap direct parser/solver baseline tying the tested pipeline, would refute added-value qualification.

## WAS THAT CHECKED
No. The artifact checks deterministic fixture replay and solver parity against oracle-defined expected effects; it does not score independently generated held-out candidates. The direct `clingo_direct` comparator ties the primary enumerator, so the available comparison shows no added value.

## EVIDENCE
`honest_verdict`: `complete_circular_positive_span_first_relation_fixture`; `verifier_is_oracle`: `true`; `verdict_class`: `circular_positive`; `inference_substrate`: `deterministic_cpu_span_first_fixture_no_llm`; `span_relation_fixture_ready_score`: `1`; `expected_effect_met`: `true`; `solver_parity`: `true`; `engine`: `clingo_direct`; `engine`: `bounded_asp_energy_enumerator`; `expected_outputs_exposed_in_live_prompts`: `false`

## RECOMMENDATION
NARROW_CLAIM

## experiment_6927_v607_literature_delta.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bounded V607 literature-delta review completed across all required source families and named candidates and produced one verified correction.

## WHAT WOULD REFUTE IT
Any required source family or named candidate being missing or nonterminal, a failed gate, or no qualifying correction row would falsify the completion claim.

## WAS THAT CHECKED
Yes. The gate summary compares the expected and observed terminal counts; source and candidate rows record terminal status; and the ledger contains one correction. This supports only bounded workflow completion, not comprehensive literature coverage or method superiority.

## EVIDENCE
`"honest_verdict": "complete_v607_literature_delta_with_one_verified_correction"`; `"expected": "15 terminal source families and 10 terminal named candidates"`; `"observed": "15 source families and 10 candidates"`; `"all_gates_passed": true`; `"failed_check": null`; `"status": "complete"`; `"v607_literature_delta_complete_score": 1`; `"terminal": true`; `"ledger_append_rows"`; `"candidate_id": "kan_verification_metadata_code_state"`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6928_sota_runtime_receipt_qualification.json

**SKIPPED_ALREADY_FLAGGED**
