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
The active V630 Markdown and YAML task contracts do not conform.

## WHAT WOULD REFUTE IT
Independent parses showing the same 13 tasks in the same order with matching IDs and contract metadata would refute the mismatch claim.

## WAS THAT CHECKED
Yes. The artifact independently parsed the Markdown and active YAML, compared task counts and per-task identities, and recorded mismatches.

## EVIDENCE
`honest_verdict`: `complete_disqualified_v630_markdown_yaml_contract_mismatch`; `inference_substrate`: `aggregation_from_active_contract: independent Markdown and active YAML parses`; `expected_task_count`: `13`; `observed_task_count`: `3`; `failed_check`: `markdown_task_count`; `expected_value`: `13`; `observed_value`: `14`; `passed`: `false`; `v630_task_contract_conforms_score`: `0`; `verdict_class`: `disqualified`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_7157_v630_qwen38_runtime.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific claim exists to refute; operationally, finding at least one idle RTX 3090 would contradict the stated reason for blocking.

## WAS THAT CHECKED
Yes—the `idle_rtx_3090` gate checked for at least one available device and found none; no model-value claim was tested because inference did not run.

## EVIDENCE
`honest_verdict`: `blocked_idle_rtx_3090`; `status`: `blocked`; `verdict_class`: `blocked`; `inference_substrate`: `no_inference`; `inference_substrate_class`: `blocked_no_run`; `generation_receipts`: `[]`; `model_load_receipts`: `[]`; `count`: `0`; `indices`: `[]`; `passed`: `false`; `qwen38_runtime_ready_score`: `0`

## RECOMMENDATION
KEEP

## experiment_7158_v630_entity_evidence_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The artifact’s limited fixture-readiness assertion would be falsified by a failed integrity gate, a mutation whose changed terms differed from the expected terms, a failed sealing check, or an incomplete fixture schedule.

## WAS THAT CHECKED
Yes. The artifact reports the aggregate gate result and exposes mutation-test, sealing, split, condition-count, and completion receipts where such failures could appear. It does not test verifier value, but explicitly makes no such comparative claim.

## EVIDENCE
`honest_verdict`: `complete_positive_counterfactual_fixture_ready_no_verifier_value_claim`; `counterfactual_fixture_ready_score`: `1`; `expected_value`: `all_fixture_integrity_checks_pass`; `observed_value`: `all_fixture_integrity_checks_pass`; `failed_check`: `null`; `passed`: `true`; `status`: `complete`; `verifier_is_oracle`: `false`; `inference_substrate`: `exact_source_fixture_construction`; `candidate_verifier_only`: `true`; `evaluation_truth_accessed`: `false`

## RECOMMENDATION
KEEP

## experiment_7159_v631_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V631 Markdown/YAML task contract is disqualified because active YAML contains 7 tasks rather than the required 14.

## WHAT WOULD REFUTE IT
An independent active-YAML parse finding all 14 expected tasks in the required order, with the contract-conformance score equal to 1, would refute the claimed mismatch.

## WAS THAT CHECKED
Yes. The artifact independently parses the Markdown and active YAML, compares the expected and observed task counts and ID order, and records the failed gate in `gate_check_summary`.

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
There is no comparative or value claim to falsify. Treating the operational status as an assertion, an available RTX 3090 with no conflicting process would refute the reported block.

## WAS THAT CHECKED
Yes, in `gate_check_summary`: availability and conflicting processes were explicitly observed.

## EVIDENCE
`honest_verdict` `blocked_idle_rtx_3090` `status` `blocked` `inference_substrate_class` `no_model_load` `available_gpu_uuids` `[]` `ownership_classification` `conflicting` `passed` `false`

## RECOMMENDATION
KEEP

## experiment_7161_qwen38_bounded_structured_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no substantive performance or value claim to falsify; this is a blocked-run receipt.

## WAS THAT CHECKED
No; the canary did not run because its prerequisite gate failed at `conductor_pre_gate`.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_observed": 0`; `"failed_expected": 1`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

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
No falsifying observation applies because the artifact reports a blocked run and makes no comparative or value claim.

## WAS THAT CHECKED
No; no generation or scored rows exist, so no method claim was tested.

## EVIDENCE
`status` `blocked` `honest_verdict` `blocked_idle_task_ownable_rtx_3090` `inference_substrate_class` `blocked_no_run` `claim_evidence_trace_ready_score` `0` `rows` `[]` `generation_receipts` `[]`

## RECOMMENDATION
KEEP
