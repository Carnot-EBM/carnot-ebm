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
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7156_v630_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The active V630 Markdown and YAML task contracts do not conform, so the preflight is disqualified.

## WHAT WOULD REFUTE IT
Matching independently parsed Markdown and YAML task counts, ordered IDs, and per-task contract fields, with all contract rows passing and a nonzero conformance score.

## WAS THAT CHECKED
Yes. Independent parses appear in `markdown_task_rows` and `yaml_task_rows`, with comparisons in `task_contract_rows`, `gate_check_summary`, and the conformance score. The checks could have passed, but instead recorded multiple mismatches.

## EVIDENCE
`inference_substrate`: `aggregation_from_active_contract: independent Markdown and active YAML parses`; `expected_task_count`: `13`; `observed_task_count`: `3`; `failed_check`: `markdown_task_count`; `expected_value`: `13`; `observed_value`: `14`; `passed`: `false`; `id_parity`: `false`; `milestone_parity`: `false`; `title_parity`: `false`; `v630_task_contract_conforms_score`: `0`; `verdict_class`: `disqualified`; `honest_verdict`: `complete_disqualified_v630_markdown_yaml_contract_mismatch`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7157_v630_qwen38_runtime.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked precondition rather than a positive or comparative runtime claim.

## WAS THAT CHECKED
No; no inference or comparison occurred because the idle-GPU gate failed.

## EVIDENCE
`"honest_verdict": "blocked_idle_rtx_3090"`, `"inference_substrate": "no_inference"`, `"inference_substrate_class": "blocked_no_run"`, `"generation_receipts": []`, `"qwen38_runtime_ready_score": 0`, `"status": "blocked"`, `"verdict_class": "blocked"`, `"check": "idle_rtx_3090"`, `"count": 0`, `"passed": false`

## RECOMMENDATION
KEEP

## experiment_7158_v630_entity_evidence_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A failed fixture-integrity gate, incorrect mutation-term response, truth leakage into the model view, or incomplete split/condition coverage would refute the limited fixture-readiness assertion; no verifier-performance or added-value claim is made.

## WAS THAT CHECKED
Yes. The artifact checks fixture integrity in `gate_check_summary`, term behavior in `mutation_test_rows`, truth separation in `sealed_field_rows`, and corpus coverage in `source_family_rows` and `split_rows`.

## EVIDENCE
`honest_verdict` = `complete_positive_counterfactual_fixture_ready_no_verifier_value_claim`; `counterfactual_fixture_ready_score` = `1`; `observed_value` = `all_fixture_integrity_checks_pass`; `passed` = `true`; `candidate_verifier_only` = `true`; `evaluation_truth_accessed` = `false`; `verifier_is_oracle` = `false`; `inference_substrate` = `exact_source_fixture_construction`

## RECOMMENDATION
KEEP

## experiment_7159_v631_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The active V631 YAML contract is disqualified because it contains 7 tasks instead of the required 14 and therefore does not conform to the Markdown contract.

## WHAT WOULD REFUTE IT
An independently parsed active YAML contract containing all 14 expected tasks in the required order, with matching task fields, would refute the mismatch claim.

## WAS THAT CHECKED
Yes. The artifact independently parsed the Markdown and active YAML, compared expected and observed task counts and ID order, and permitted a conforming score of 1; instead, the YAML count check failed.

## EVIDENCE
`inference_substrate`: `aggregation_from_active_contract: independent Markdown and active YAML parses`; `expected_task_count`: `14`; `observed_task_count`: `7`; `failed_check`: `yaml_task_count`; `passed`: `false`; `v631_task_contract_conforms_score`: `0`; `honest_verdict`: `complete_disqualified_v631_markdown_yaml_contract_mismatch`; `verdict_class`: `disqualified`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7160_v631_qwen38_lease_diagnosis.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or added-value claim to falsify; the operational diagnosis would be refuted by at least one available RTX 3090 with no conflicting process.

## WAS THAT CHECKED
Yes, in `gate_check_summary`, which checks GPU availability and conflicting processes.

## EVIDENCE
`"honest_verdict": "blocked_idle_rtx_3090"`, `"status": "blocked"`, `"verdict_class": "blocked"`, `"inference_substrate_class": "no_model_load"`, `"available_gpu_uuids": []`, `"ownership_classification": "conflicting"`, `"passed": false`, `"qwen38_runtime_preflight_ready_score": 0`

## RECOMMENDATION
KEEP

## experiment_7161_qwen38_bounded_structured_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no method-performance or comparative claim to falsify; this artifact only records a failed prerequisite gate.

## WAS THAT CHECKED
No substantive claim was tested. The sole upstream readiness gate was checked and failed in `gates_evaluated`.

## EVIDENCE
`status` `blocked` `honest_verdict` `blocked_gate_check_failed` `actual` `0` `expected` `1` `passed` `false` `blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7166_v632_contract_preflight.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7167_v632_claim_evidence_trace_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or value claim exists to falsify. If the blocked-run status were treated as an operational claim, a passed GPU gate or completed generation traces would refute it.

## WAS THAT CHECKED
Yes. The GPU precondition was checked and failed before invocation; no scored generation occurred.

## EVIDENCE
`"status": "blocked"`; `"verdict_class": "blocked"`; `"inference_substrate_class": "blocked_no_run"`; `"passed": false`; `"claim_evidence_trace_ready_score": 0`; `"claim_evidence_trace_rows": []`; `"generation_receipts": []`; `"rows": []`; `"it makes no verifier-value claim."`

## RECOMMENDATION
KEEP
