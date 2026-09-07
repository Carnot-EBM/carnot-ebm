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
| CLAIM_SUPPORTED | 6 |
| NO_CLAIM | 2 |

## experiment_7107_v623_continual_memory_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The continual-memory cold audit is complete and ready, while the upstream value conclusion remains independently classified.

## WHAT WOULD REFUTE IT
Any failed replay/parity check, current-feedback leakage, invalid reconstruction or signature, unrecoverable transaction attack, or failed protected retention for the delayed-procedural method would falsify audit readiness.

## WAS THAT CHECKED
Yes. The artifact includes event reconstruction, producer–auditor parity, future-label isolation, signatures, protected retention, and mutation, crash, partial-write, poison, reorder, stale-parent, and rollback probes. These checks could record failures; the visible delayed-procedural retention row passes, and the aggregate gate reports no failed check.

## EVIDENCE
`honest_verdict` `complete: continual memory cold audit ready; upstream value remains independently classified` `continual_memory_cold_audit_ready_score` `1` `failed_check` `null` `observed_value` `all checks pass` `passed` `true` `verifier_is_oracle` `false` `fresh_process` `true` `current_event_feedback_used` `false` `future_label_accessed` `false` `arm` `delayed_procedural` `retention_passed` `true`

## RECOMMENDATION
KEEP

## experiment_7108_v623_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The twelve-slot V623 evidence matrix is complete as a record of terminal outcomes, including blocked and disqualified outcomes, without claiming that every branch demonstrated value.

## WHAT WOULD REFUTE IT
A missing required task, mismatched task count or ordering, absent artifact or disposition record, unrecorded completion, or promotion of value from invalid or absent evidence would refute the completeness claim.

## WAS THAT CHECKED
Yes. Expected and observed task counts were compared; contract, artifact, per-unit, provenance, gate, and disposition rows were audited. Checks demonstrably could fail: identity, checksum, gate, and headline-recomputation failures appear and lead to blocked or disqualified dispositions rather than value promotion.

## EVIDENCE
`expected_task_count`: `12`; `observed_task_count`: `12`; `gate_check_summary`: `expected_value`: `12`, `observed_value`: `12`, `passed`: `true`; `completion_separate_from_value`: `true`; `disposition`: `disqualified`; `reason`: `artifact_check_failed`; `value_promoted`: `false`; `v623_sota_ingestion_complete_score`; `declared_value`: `1`; `recomputed_value`: `0`; `passed`: `false`; `disposition`: `blocked`; `do not infer energy value from absent rows`; `verifier_is_oracle`: `false`.

## RECOMMENDATION
KEEP

## experiment_7109_v624_contract_preflight.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The active V624 YAML contract is incomplete relative to the 12-task Markdown contract, disqualifying contract conformance.

## WHAT WOULD REFUTE IT
An independently parsed active YAML containing all 12 expected tasks in matching order, with every contract clause passing and a conformance score of 1.

## WAS THAT CHECKED
Yes. The artifact independently replayed the Markdown and active YAML contracts, compared expected and observed task counts and IDs, and applied the contract gate.

## EVIDENCE
`inference_substrate`: `aggregation_from_upstream_artifacts: independent Markdown and YAML contract replay`; `expected_task_count`: `12`; `observed_task_count`: `6`; `failed_check`: `yaml_task_count`; `passed`: `false`; `v624_task_contract_conforms_score`: `0`; `honest_verdict`: `complete_disqualified_v624_markdown_yaml_contract_mismatch`

## RECOMMENDATION
KEEP

## experiment_7110_v624_evidence_ingress_quarantine.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The evidence-ingress quarantine is ready because its acceptance, rejection, historical-census, and regression gates all passed.

## WHAT WOULD REFUTE IT
A clean fixture being rejected, a flagged or invalid-date fixture being accepted, a historical-census mismatch, historical evidence being rewritten, or the flagged experiment 7099 entering experiment 7108 would falsify the readiness claim.

## WAS THAT CHECKED
Yes. Positive and negative fixtures exercised both acceptance and rejection; malformed and missing dates were rejected; the historical census matched 55 rows; non-rewriting was checked; and the experiment 7099 regression remained excluded. These checks could have produced mismatched observed values or a failed gate.

## EVIDENCE
`"fixture": "accepted_clean_input"`, `"expected_accepted_count": 1`, `"observed_accepted_count": 1`, `"fixture": "artifact_level_flag"`, `"expected_accepted_count": 0`, `"observed_accepted_count": 0`, `"fixture": "verifier_critical_flag"`, `"passed": true`, `"reason_codes": ["run_date_malformed"]`, `"reason_codes": ["run_date_missing"]`, `"check": "historical_capstone_census_count"`, `"expected_value": 55`, `"observed_value": 55`, `"check": "historical_artifacts_rewritten"`, `"observed_value": false`, `"check": "exp7108_rejects_flagged_exp7099"`, `"observed_value": true`, `"failed_check": null`, `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7111_v624_arc_provenance_canary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V624 ARC forward-provenance canaries passed.

## WHAT WOULD REFUTE IT
A canary failure: invalid or missing provenance being accepted, an unreceipted/non-live row becoming headline-eligible, incorrect dashboard grouping, or mutation of historical bytes or the ARC registry.

## WAS THAT CHECKED
Yes. Positive and negative writer cases, four eligibility cases, dashboard output, byte preservation, and registry stability were checked; every recorded check passed.

## EVIDENCE
`"honest_verdict": "complete: positive V624 ARC forward provenance canaries passed"`; `"writer_rejects_missing_or_invalid_provenance"` with `"observed_value": true`; `"only_receipted_live_row_is_headline_eligible"` with `"observed_value": [true, false, false, false]`; `"dashboard_preserves_provenance_groups"` with `"observed_value": true`; `"historical_row_bytes_unchanged"` with `"passed": true`; `"arc_registry_hash_unchanged"` with `"passed": true`; `"verifier_is_oracle": false`; `"offline_reproduced": false`.

## RECOMMENDATION
KEEP

## experiment_7112_v624_sota_ingestion.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bounded V624 literature and repository audit completed and adopted one decision-relevant control into an existing experiment.

## WHAT WOULD REFUTE IT
No verified, decision-relevant candidate being promoted; an adopted candidate lacking a source receipt, claim boundary, or experiment hook; or a failed completion gate would refute the claim.

## WAS THAT CHECKED
Yes. The candidate, adoption, source-receipt, claim-boundary, precondition, and gate rows expose those failure conditions; none occurred. The claim is limited to the bounded audit, not method quality or Carnot performance.

## EVIDENCE
`honest_verdict` is `complete_positive_v624_sota_ingestion_adopted_delta`; `candidate_id` is `arxiv-2609-02750`; `classification` is `control`; `decision_relevant` is `true`; `experiment_hook` is `exp7118-principle-step-memory-csl`; `core_claim_verified` is `true`; `identity_verified` is `true`; `content_verified` is `true`; `claim_boundary_present` is `true`; `passed` is `true`; `failed_check` is `null`; `v624_sota_ingestion_complete_score` is `1`; `inference_substrate` is `bounded network literature and repository audit`; `verifier_is_oracle` is `false`.

## RECOMMENDATION
KEEP

## experiment_7113_v624_arc_generation_liveness.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to refute. The terminal blocked-status statement would be contradicted by successful preconditions and completed generation-request rows.

## WAS THAT CHECKED
Yes. The precondition gate recorded terminal failures, and all generation-request and result rows are empty.

## EVIDENCE
`honest_verdict`: `complete_blocked_arc_generation_liveness_precondition_failed`; `verdict_class`: `blocked`; `healthy_idle_gpus`; `expected_value`: `2`; `observed_value`: `0`; `passed`: `false`; `rows`: `[]`; `per_model_request_rows`: `[]`; `arc_generation_liveness_ready_score`: `0`; `offline_reproduced`: `false`

## RECOMMENDATION
KEEP

## experiment_7114_adapter_withheld_arc_loo_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked gate rather than a measurement or comparative conclusion.

## WAS THAT CHECKED
No; execution stopped at `conductor_pre_gate`, so no claim-testing data was produced.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_observed": 0`; `"failed_expected": 1`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP
