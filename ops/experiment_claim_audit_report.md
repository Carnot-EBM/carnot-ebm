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
| NO_CLAIM | 7 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_9013_v770_branch_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim. The administrative custody assertion would fail if task contracts disagreed, frozen bytes changed, corrupted authority passed validation, or historical disqualifications were erased.

## WAS THAT CHECKED
Yes, administrative checks are reported through task comparisons, expected-versus-observed hashes, authority mutation controls, and preserved historical verdicts. Shown checks pass. Scientific bounds remain explicitly unmeasured, and exposed development results are not presented as independent generalization.

The excerpt elides some checks and underlying control outcomes; those details remain unverified.

## EVIDENCE

- `claim_scope`: `Administrative custody only; no scientific benefit or independent generalization claim.`
- `honest_verdict`: `complete_null_v770_branch_contract`
- `metric`: `ordered_task_contract_match`; `matched`: `true`
- `authority_mutation_rows`; `mutation`: `rehash`; `passed`: `true`
- `scope`: `exposed_development_only`
- `status`: `frozen_unmeasured_until_fresh_calibration_and_test`
- `independent_generalization_score`: `0`
- `repository_health_passed`: `false`
- `original_verdict_class`: `disqualified`

## RECOMMENDATION
KEEP

## experiment_9014_v770_scoped_provenance_qualification.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_9015_v770_human_disagreement_source.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. A claim of source readiness would be refuted by failed required source access or unresolved source identities.

## WAS THAT CHECKED
Yes. The source-access check failed with HTTP 429, exposure integrity failed, and both readiness gates failed. The artifact records a blocked qualification rather than a positive result.

## EVIDENCE
- `claim_scope`: `Original human disagreement source qualification only; no detector result, factual oracle or span leaderboard claim.`
- `honest_verdict`: `complete_blocked_source_resolution_access`
- `verdict_class`: `blocked`
- `access_error`: `HTTP Error 429: Too Many Requests`
- `completed_count`: `0`
- `independent_label_ready_score`: `0`
- `natural_pilot_ready_score`: `0`

## RECOMMENDATION
KEEP

Unchecked: external files and elided rows. This receipt establishes no detector benefit.

## experiment_9016_retained_shard_a.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a blocked-gate receipt, not an authentication result. Both recorded gates passing would contradict its reported blockage.

## WAS THAT CHECKED
Yes. In `gates_evaluated`, the provenance gate fails with actual 0 versus expected 1; the authority gate passes. The receipt consistently reports blockage. Authentication itself remains untested.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `provenance_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_9017_graph_corpus_b.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental claim to falsify. The title describes intended work; the artifact records a blocked prerequisite.

## WAS THAT CHECKED
Yes—the provenance gate was evaluated and failed: actual 0 versus expected 1. This supports the blocking diagnosis.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `provenance_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

Upstream artifacts were not inspected; this receipt provides no evidence of experimental performance.

## experiment_9019_human_pilot_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked prerequisite receipt, with no comparative or performance claim.

## WAS THAT CHECKED
The readiness gate was checked: observed 0 versus required 1, so execution was blocked. No method comparison was evaluated.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `natural_pilot_ready_score`; `blocked_reason`: `actual=0 == expected=1`; `passed`: `false`.

## RECOMMENDATION
KEEP

Upstream data was not checked; this receipt supports no inference about probe performance.

## experiment_9025_arc_transfer_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact reports a blocked prerequisite gate, without asserting a transfer result or comparative benefit.

## WAS THAT CHECKED
The readiness gate was checked and failed: observed 0, required 1. Transfer performance was not assessed here; the receipt supports no conclusion about transfer.

## EVIDENCE
`blocked_gate_check_v1`; `blocked`; `blocked_gate_check_failed`; `arc_new_outcomes_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`; `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_9026_v770_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to falsify. An omitted or duplicated task, or a failed qualification presented as scientific success, would contradict the administrative accounting.

## WAS THAT CHECKED
Partly. Visible contract rows check accounting, and a failed provenance gate produces quarantine and a blocked verdict. The unqualified slot counts as accounted, not as scientific success. Oracle status creates no circular value claim here; no generalization or superiority is asserted, so a rival baseline is unnecessary.

## EVIDENCE
- `claim_scope`: `Fourteen administrative outcomes; graph, independent human annotation, causal learning and deployment claims are separate and unmeasured.`
- `honest_verdict`: `complete_blocked_declared_gate`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `artifact_field`: `provenance_ready_score`; `expected`: `1`; `observed`: `0`; `passed`: `false`
- `disposition`: `quarantined_primary`; `qualified`: `false`; `accounted`: `true`
- `activation_authorized`: `false`
- `publication_scope`: `Historical FoVer G1-G4 only; no V770 science certification or external publication.`

## RECOMMENDATION
KEEP

Elided rows and upstream artifacts were not checked; full accounting remains unverified from this excerpt.
