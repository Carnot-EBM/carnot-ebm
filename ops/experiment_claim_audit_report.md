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
| CLAIM_SUPPORTED | 2 |
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 2 |

## experiment_7153_v629_grounding_runtime.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7154_v629_qwen_dual_side_grounding.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7156_v630_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V630 contract is disqualified because the independently parsed Markdown and YAML task contracts do not match the required 13-task contract.

## WHAT WOULD REFUTE IT
Both parses yielding exactly 13 tasks with matching IDs, order, and task metadata, with every contract row and the aggregate gate passing.

## WAS THAT CHECKED
Yes. Independent Markdown and YAML parses were compared through task counts, ordered IDs, per-task contract rows, and the aggregate gate; mismatches occurred.

## EVIDENCE
`honest_verdict` `complete_disqualified_v630_markdown_yaml_contract_mismatch`  
`inference_substrate` `aggregation_from_active_contract: independent Markdown and active YAML parses`  
`expected_task_count` `13`  
`observed_task_count` `3`  
`failed_check` `markdown_task_count`  
`expected_value` `13`  
`observed_value` `14`  
`passed` `false`  
`v630_task_contract_conforms_score` `0`  
`verdict_class` `disqualified`

## RECOMMENDATION
KEEP

## experiment_7157_v630_qwen38_runtime.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An available idle RTX 3090 or evidence that model inference actually ran would contradict the artifact’s blocked-no-run status, but there is no performance or value claim to falsify.

## WAS THAT CHECKED
Yes. The idle-GPU precondition was checked and failed; the empty generation and model-load receipts confirm no inference occurred.

## EVIDENCE
`honest_verdict` `blocked_idle_rtx_3090`; `inference_substrate` `no_inference`; `inference_substrate_class` `blocked_no_run`; `status` `blocked`; `verdict_class` `blocked`; `idle_rtx_3090`; `count` `0`; `passed` `false`; `generation_receipts` `[]`; `model_load_receipts` `[]`

## RECOMMENDATION
KEEP

## experiment_7158_v630_entity_evidence_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A failed fixture-integrity check, incomplete condition coverage, an unexpected mutation-term change, or a failed sealing receipt would refute fixture readiness; no verifier-performance or added-value claim is made.

## WAS THAT CHECKED
Yes. Fixture readiness was checked in `gate_check_summary`, with supporting condition counts in `source_family_rows`, term-level checks in `mutation_test_rows`, and sealing checks in `sealed_field_rows`.

## EVIDENCE
`"honest_verdict": "complete_positive_counterfactual_fixture_ready_no_verifier_value_claim"`; `"counterfactual_fixture_ready_score": 1`; `"observed_value": "all_fixture_integrity_checks_pass"`; `"passed": true`; `"status": "complete"`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7159_v631_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The active V631 YAML contract is disqualified because it contains 7 tasks while the independently parsed Markdown contract requires 14.

## WHAT WOULD REFUTE IT
An active-YAML parse yielding 14 tasks in the expected order, with all contract comparisons passing and a conformity score of 1.

## WAS THAT CHECKED
Yes. Independent Markdown and active-YAML parses were compared through the task counts, ordered IDs, contract rows, gate summary, and conformity score.

## EVIDENCE
`"inference_substrate": "aggregation_from_active_contract: independent Markdown and active YAML parses"`; `"expected_task_count": 14`; `"observed_task_count": 7`; `"failed_check": "yaml_task_count"`; `"passed": false`; `"v631_task_contract_conforms_score": 0`; `"honest_verdict": "complete_disqualified_v631_markdown_yaml_contract_mismatch"`; `"verdict_class": "disqualified"`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7160_v631_qwen38_lease_diagnosis.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The diagnostic conclusion would be refuted by at least one RTX 3090 appearing available with no conflicting process.

## WAS THAT CHECKED
Yes. The GPU gate explicitly checked for at least one available GPU and no conflicting process; it found none available and two conflicts.

## EVIDENCE
`honest_verdict`: `blocked_idle_rtx_3090`; `status`: `blocked`; `available_gpu_uuids`: `[]`; `passed`: `false`; `ownership_classification`: `conflicting`; `inference_substrate_class`: `no_model_load`; `weights_opened`: `false`.

## RECOMMENDATION
KEEP

## experiment_7161_qwen38_bounded_structured_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No substantive performance or value claim exists to falsify; the procedural blocking record would be contradicted by the gate having passed.

## WAS THAT CHECKED
Yes, the sole gate was evaluated and recorded as failed, but no experiment was run and no comparative claim was tested.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"passed": false`; `"actual": 0`; `"expected": 1`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP
