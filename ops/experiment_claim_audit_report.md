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
The V630 Markdown and YAML task contracts mismatch, so the preflight is disqualified.

## WHAT WOULD REFUTE IT
Both independent parses yielding the same 13 tasks in the same order, with all identity and contract parity checks passing.

## WAS THAT CHECKED
Yes. Independent Markdown and active-YAML parses were compared through task counts, ordered IDs, per-task parity rows, a gate summary, and a conformance score; the refuting observation did not occur.

## EVIDENCE
`inference_substrate`: `aggregation_from_active_contract: independent Markdown and active YAML parses`; `expected_task_count`: `13`; `observed_task_count`: `3`; `failed_check`: `markdown_task_count`; `observed_value`: `14`; `passed`: `false`; `id_parity`: `false`; `milestone_parity`: `false`; `title_parity`: `false`; `v630_task_contract_conforms_score`: `0`; `verdict_class`: `disqualified`; `honest_verdict`: `complete_disqualified_v630_markdown_yaml_contract_mismatch`

## RECOMMENDATION
KEEP

## experiment_7157_v630_qwen38_runtime.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific or comparative claim is made; operationally, finding at least one idle RTX 3090 would refute the recorded blocked status.

## WAS THAT CHECKED
Yes, in the `idle_rtx_3090` gate, which checked for a minimum count of `1` and observed `0`; no inference or comparative model evaluation was attempted.

## EVIDENCE
`honest_verdict`: `blocked_idle_rtx_3090`; `status`: `blocked`; `verdict_class`: `blocked`; `inference_substrate`: `no_inference`; `generation_receipts`: `[]`; `model_load_receipts`: `[]`; `qwen38_runtime_ready_score`: `0`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7158_v630_entity_evidence_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No verifier-value claim is made; fixture readiness would be falsified by a failed integrity, mutation, sealing, or split check.

## WAS THAT CHECKED
Yes. The gate summary reports all fixture-integrity checks passing, mutation rows report passed term changes, and sealing and split receipts are present.

## EVIDENCE
`honest_verdict` `complete_positive_counterfactual_fixture_ready_no_verifier_value_claim` `counterfactual_fixture_ready_score` `1` `all_fixture_integrity_checks_pass` `passed` `true` `verifier_is_oracle` `false` `candidate_verifier_only` `true`

## RECOMMENDATION
KEEP

## experiment_7159_v631_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V631 contract is disqualified because the active YAML contains 7 tasks instead of the required 14-task Markdown contract.

## WHAT WOULD REFUTE IT
An independent active-YAML parse finding all 14 expected tasks in the correct order, with matching per-task contracts and a conformance score of 1, would refute the mismatch claim.

## WAS THAT CHECKED
Yes. The artifact independently parsed Markdown and active YAML, compared their task counts and ordering, and recorded a failed YAML task-count gate.

## EVIDENCE
`"honest_verdict": "complete_disqualified_v631_markdown_yaml_contract_mismatch"`; `"inference_substrate": "aggregation_from_active_contract: independent Markdown and active YAML parses"`; `"expected_task_count": 14`; `"observed_task_count": 7`; `"failed_check": "yaml_task_count"`; `"passed": false`; `"v631_task_contract_conforms_score": 0`; `"verdict_class": "disqualified"`

## RECOMMENDATION
KEEP

## experiment_7160_v631_qwen38_lease_diagnosis.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; treating the blocking status as an operational assertion, an available RTX 3090 with no conflicting process would refute it.

## WAS THAT CHECKED
Yes. The GPU gate checked for at least one available GPU with no conflicting process and recorded the observed conflicts.

## EVIDENCE
`honest_verdict` `blocked_idle_rtx_3090` `status` `blocked` `verdict_class` `blocked` `qwen38_runtime_preflight_ready_score` `0` `check` `idle_rtx_3090` `passed` `false` `available_gpu_uuids` `[]` `ownership_classification` `conflicting` `inference_substrate_class` `no_model_load` `weights_opened` `false`

## RECOMMENDATION
KEEP

## experiment_7161_qwen38_bounded_structured_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A gate row showing the runtime preflight succeeded—such as an observed value of 1 matching the expected value of 1—would refute the artifact’s blocked-status receipt, but there is no substantive method-performance claim to falsify.

## WAS THAT CHECKED
Yes. The sole row in `gates_evaluated` records the observed and expected values, comparison operator, and failed result.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_field": "qwen38_runtime_preflight_ready_score"`; `"failed_expected": 1`; `"failed_observed": 0`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

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
No comparative or performance claim is made to falsify; this is a blocked-run receipt.

## WAS THAT CHECKED
No; generation never ran, so there are no result rows on which a substantive claim could succeed or fail.

## EVIDENCE
`"status": "blocked"`; `"verdict_class": "blocked"`; `"inference_substrate_class": "blocked_no_run"`; `"claim_evidence_trace_ready_score": 0`; `"claim_evidence_trace_rows": []`; `"generation_receipts": []`; `"rows": []`; `"passed": false`

## RECOMMENDATION
KEEP
