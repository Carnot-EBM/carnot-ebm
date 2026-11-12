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
| NO_CLAIM | 8 |

## experiment_9028_v771_historical_provenance_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim. The scoped custody assertion would fail if missing or stale receipts, altered authority inputs, or contradictory invocation records were accepted as valid.

## WAS THAT CHECKED
Yes, for the displayed controls: valid inputs exit successfully; missing/stale receipts, altered authority, and contradictory invocation records are rejected with the expected exit codes. Nested receipt failures are expected negative-control outcomes. These checks establish receipt behavior without claiming verifier benefit or generalization.

## EVIDENCE

- `claim_scope`: `Current scoped provenance custody with immutable historical test inputs; no repository health or verifier benefit claim.`
- `current_receipt_mutation_rows`: `valid` has `expected_exit` = `0`, `observed_exit` = `0`; `missing` and `stale` each have `expected_exit` = `1`, `observed_exit` = `1`.
- `historical_authority_rows`: `authority_rehash` has `expected_exit` = `1`, `observed_exit` = `1`.
- `mutation_rows`: `contradictory_current` has `expected_exit` = `1`, `observed_exit` = `1`.
- `honest_verdict`: `complete_positive_historical_provenance`.

## RECOMMENDATION
KEEP

Not checked: elided rows or external files; omitted evidence remains unassessed.

## experiment_9029_v771_source_identity_resolution.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. The reported identity-closure blockage would be contradicted by complete release and historical identity resolution with all readiness gates passing.

## WAS THAT CHECKED
Yes, for readiness. The closure checks report 2 unresolved release identities and 213 historical identities against requirements of zero. Readiness fails, all 795 rows are censored, and no positive result is claimed. Comparator and generalization controls are unnecessary for this metadata receipt.

## EVIDENCE
- `claim_scope`: `Public source identities and prior exposure joins only. No label, training, factual verification or detector claim.`
- `honest_verdict`: `complete_blocked_release_identity_closure`
- `release_identity_closure`: `expected` = `0`, `observed` = `2`
- `historical_identity_closure`: `expected` = `0`, `observed` = `213`
- `source_identity_ready_score`: `0`
- `censored_count`: `795`; `completed_count`: `0`

## RECOMMENDATION
KEEP

Elided rows and external artifacts were not inspected; this assessment covers only the supplied text.

## experiment_9030_v771_retained_shard_a.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. The custody audit would fail if retained bytes disagreed with recorded hashes, historical receipts failed authentication, or recomputed parser/BFS totals disagreed with retained rows.

## WAS THAT CHECKED
Yes, for custody: the displayed authority-hash comparison matches, the receipt audit passes, and summary recomputation is reported. Non-exposure certification actually fails, and the artifact remains blocked. Oracle labels are explicitly denied evidentiary value for detector benefit or generalization; no rival comparison is needed for this custody-only scope.

## EVIDENCE
- `title`: `Retained shard A custody audit`
- `claim_scope`: `Historical source custody only. BFS labels are circular oracle evidence; no detector benefit or independent generalization.`
- `honest_verdict`: `complete_blocked_nonexposure_audit`
- `nonexposure_audit`: `passed`: `false`; `published_test_outcomes`: `64`
- `independent_generalization_score`: `0`
- `shard_ready_score`: `0`
- `verdict_class`: `blocked`

## RECOMMENDATION
KEEP

Not rerun; elided rows and external receipts were not independently checked. Custody evidence establishes no detector benefit.

## experiment_9031_v771_graph_corpus_b.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative efficacy claim to falsify. A completed current-run response or model invocation would contradict the narrower blocked-execution receipt.

## WAS THAT CHECKED
Yes, execution accounting reports no model invocation, zero completed responses, and 512 censored responses. The changed-input gate failed after finding 512 previously completed identities against an expected zero. No performance test occurred; oracle circularity supports no value claim here.

## EVIDENCE
- `claim_scope`: `First-response B source custody only; BFS oracle admission is circular. No detector benefit or independent generalization.`
- `honest_verdict`: `complete_blocked_changed_input`
- `model_invoked`: `false`
- `completed_count`: `0`
- `censored_count`: `512`
- `prior_B_inventory`: `previous_completed_B_identities`, `expected`: `0`, `observed`: `512`, `passed`: `false`

## RECOMMENDATION
KEEP

Unchecked: elided content and external artifacts. This receipt establishes no model performance.

## experiment_9032_corpus_admission.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked admission receipt, with no experimental success or comparative value claim.

## WAS THAT CHECKED
The prerequisite gates were checked: both required a score of 1, observed 0, and failed. The artifact records blocking before the experiment, so it provides no experimental claim to test.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`, `"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"expected": 1`, `"actual": 0`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`.

## RECOMMENDATION
KEEP

## experiment_9033_human_label_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite gate, without claiming successful label qualification or comparative method value.

## WAS THAT CHECKED
The prerequisite gate was checked and failed: readiness was 0 against a required 1. Execution stopped before the experiment; method performance was not tested.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `source_identity_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_9039_arc_transfer_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite check, without claiming successful transfer or comparative value.

## WAS THAT CHECKED
The readiness gate was checked and failed: observed 0 versus required 1. Transfer performance was not evaluated; this receipt provides no evidence for or against transfer.

## EVIDENCE
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `arc_new_outcomes_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9040_v771_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is made. An intended task missing from the accounting would refute the administrative completeness statement.

## WAS THAT CHECKED
Yes, for the visible administrative rows: contract matches and slot accounting are recorded, and failed identity checks remain blocked. Oracle correctness is not presented as added scientific value.

## EVIDENCE
- `claim_scope`: `Fourteen administrative outcomes; graph, independent human annotation, causal learning and deployment claims are separate and unmeasured.`
- `honest_verdict`: `complete_blocked_declared_gate`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `graph_claim`: `BFS is a cheap oracle. A graph benefit would be circular_positive and cannot close the no-cheap-oracle target.`

## RECOMMENDATION
KEEP

Unchecked: elided rows; complete administrative accounting cannot be independently assessed from this excerpt.
