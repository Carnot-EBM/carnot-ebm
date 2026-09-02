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
| NO_CLAIM | 2 |
| SKIPPED_ALREADY_FLAGGED | 5 |

## experiment_6868_three_family_semantic_scoring_stream_v2.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6869_calibration_only_paired_semantic_rule.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6870_sealed_independent_semantic_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact is a blocked-gate receipt and makes no semantic compatibility or comparative value claim.

## WAS THAT CHECKED
No; the semantic audit never ran because the upstream readiness gate failed.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_field": "semantic_contrast_rule_ready_score"`; `"failed_observed": 0`; `"failed_expected": 1`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6871_observable_reliability_opportunity_stream.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6872_bounded_reliability_controller_quarantine.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6873_prospective_sealed_self_learning_audit.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6874_v602_evidence_substrate_manifest_contract.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V602 evidence, substrate, and manifest-parity contract is blocked because document/YAML parity and dependent gate checks failed.

## WHAT WOULD REFUTE IT
Complete document/YAML parity—11 tasks in both sources with all compared fields equal—plus a passing gate contract and readiness score of 1 would refute the blocked verdict.

## WAS THAT CHECKED
Yes. `gate_check_summary` directly compares expected and observed parity, gate-contract status, and readiness; the artifact also supplies per-task `v602_document_yaml_parity_rows`.

## EVIDENCE
`honest_verdict`: `complete_blocked_v602_evidence_substrate_manifest_contract`; `status`: `complete_blocked`; `verdict_class`: `blocked`; `failed_check`: `v602_document_yaml_parity`; `all_fields_equal`: `false`; `document_task_count`: `11`; `yaml_task_count`: `4`; `v602_gate_contract`: `false`; `v602_evidence_contract_ready_score`: `0`; `verifier_is_oracle`: `false`.

## RECOMMENDATION
KEEP

## experiment_6875_text_anchored_relation_asp_fixture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific claim is made. Treating the blocked status as a procedural claim, an observed gate value of 1 with a passing result would refute it.

## WAS THAT CHECKED
No scientific claim was checked; execution stopped at the upstream gate. The gate itself was checked and failed.

## EVIDENCE
`status`: `blocked`; `blocked_at_layer`: `conductor_pre_gate`; `failed_expected`: `1`; `failed_observed`: `0`; `passed`: `false`

## RECOMMENDATION
KEEP
