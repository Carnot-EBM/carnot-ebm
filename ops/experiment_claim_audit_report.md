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
| NO_CLAIM | 7 |

## experiment_9041_v772_branch_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to refute. A task-contract mismatch, changed frozen input bytes, or historical exposed rows credited as fresh TEST evidence would contradict the limited administrative custody assertions.

## WAS THAT CHECKED
Yes, within the visible custody checks: contract and input-byte comparisons match, seven authority mutation controls report passing, and historical exposure rows remain ineligible as new TEST evidence. Scientific comparisons are explicitly deferred; development results are labeled exposed.

## EVIDENCE
- `claim_scope`: `Administrative custody only; no scientific benefit or independent generalization claim.`
- `honest_verdict`: `complete_null_v772_branch_contract`
- `metric`: `ordered_task_contract_match`; `matched`: `true`
- `check`: `input_bytes`; `passed`: `true`
- `eligible_as_new_test`: `false`
- `scope`: `exposed_development_only`
- `status`: `frozen_unmeasured_until_fresh_calibration_and_test`
- `evaluation_status`: `frozen_unmeasured; later real streams required`

## RECOMMENDATION
KEEP

Elided rows were not inspected and nothing was rerun; scientific performance remains unmeasured here.

## experiment_9042_v772_fresh_seal_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative efficacy claim to falsify. Within the administrative custody scope, accepting an altered roster or failing exact continuation after interruption would contradict the custody assurances.

## WAS THAT CHECKED
Partially: interruption, continuation, and altered-roster controls report passing, explicitly as private fixtures. Their internal assertions are not shown. Incomplete roster construction produces a blocked verdict, so the artifact does not promote administrative checks into a positive performance result.

## EVIDENCE

- `claim_scope`: `Prospective graph custody only; no verifier benefit or oracle-distinct generalization claim.`
- `honest_verdict`: `complete_blocked_fresh_roster_complete`
- `status`: `blocked`
- `completed_count`: `414`; `intended_count`: `1024`; `censored_count`: `610`
- `real_interruption`, `exact_continuation`, `altered_roster`; `fixture_only`: `true`; `passed`: `true`
- `outcome_status`: `withheld`; `model_invoked`: `false`

## RECOMMENDATION
KEEP

Unchecked: elided entries and external files. Custody receipts establish no predictive benefit.

## experiment_9043_v772_identity_obstruction_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
All 215 retained identity obstructions were classified as missing pages or existing pages without QIDs; source-identity closure remains blocked.

## WHAT WOULD REFUTE IT
An omitted retained identity, or a retained response containing an alias-resolvable valid QID for an identity classified as missing or without a QID.

## WAS THAT CHECKED
Yes, within the retained-metadata scope: the methodology specifies original-response inspection and cold replay. Reported counts cover all 215 identities, and shown classifications expose supporting markers and absent identifiers. Unresolved release identities remain blocked. This is a descriptive diagnosis without a comparative-benefit claim.

## EVIDENCE

- `intended_count`: `215`; `completed_count`: `215`; `excluded_count`: `0`.
- `category_counts`: `missing_page`: `206`; `existing_page_without_qid`: `9`; `unresolved_other`: `0`.
- `missing_marker`: `true`; `pageid`: `null`; `wikibase_item`: `null`.
- `proven_omissions`: `[]`.
- `source_identity_ready_score`: `0`.
- `honest_verdict`: `complete_blocked_release_identity_closure`.
- `Cold replay recomputes classifications from immutable source bytes.`

## RECOMMENDATION
KEEP

Not independently checked: elided rows and referenced files; exhaustive classification relies on the reported inspection and checks.

## experiment_9044_v772_annotation_feasibility.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative effectiveness claim to falsify. The limited structural-readiness finding would fail if an inventoried record lacked three distinct nonempty annotator IDs or the original span schema.

## WAS THAT CHECKED
Yes, at the structural level: access receipts cover both splits, and annotation-schema rows report distinct annotator counts and schema presence. The artifact reports an inventory of all 2,401 records, although most rows are elided here. Pilot readiness explicitly fails and the verdict remains blocked.

Annotation values and span validity were not checked; structural availability cannot establish usable outcomes or model effectiveness.

## EVIDENCE

- `claim_scope`: `Original individual annotation availability only; no fitted detector or factual-correctness claim.`
- `distinct_nonempty_annotator_count`: `3`
- `original_span_schema_present`: `true`
- `annotation_structure_ready_score`: `1`
- `natural_pilot_ready_score`: `0`
- `honest_verdict`: `complete_blocked_source_identity_ready_score`
- `verdict_class`: `blocked`
- `label_values_read`: `false`

## RECOMMENDATION
KEEP

## experiment_9045_fresh_corpus_a.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt with no comparative or performance claim.

## WAS THAT CHECKED
No experimental claim was tested. Only the prerequisite gate was checked in `gates_evaluated`: the readiness score was 0, the required value was 1, and the gate failed.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `fresh_source_ready_score`
- `expected`: `1`
- `actual`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9046_fresh_corpus_b.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact reports a blocked collection attempt and makes no comparative or performance claim.

## WAS THAT CHECKED
Yes—the prerequisite gate was checked and failed: readiness was 0 against required 1. This records a blocked attempt, not experimental success. Review is limited to the supplied artifact; no rerun was performed.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `fresh_source_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_9052_arc_transfer_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked prerequisite receipt, with no asserted transfer result to falsify.

## WAS THAT CHECKED
The prerequisite was checked and failed: readiness was 0 against a required 1. No transfer evaluation or comparative claim was checked; the artifact supports only the blocked status.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `arc_new_outcomes_ready_score`
- `expected`: `1`; `actual`: `0`; `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9053_v772_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to refute. The administrative accounting would fail if a required task were unaccounted or a failed readiness gate were reported as scientific success.

## WAS THAT CHECKED
Yes, the visible contract and slot rows check administrative accounting. Gate rows explicitly record fresh-source readiness failing, and the final verdict remains blocked. Comparative bounds, including those against converged linear, remain unmeasured. Oracle status therefore does not establish a circular value claim here: no verifier benefit is asserted.

## EVIDENCE
- `claim_scope`: `Thirteen administrative outcomes; no measured detector, learning, independent-label, cost or deployment benefit.`
- `honest_verdict`: `complete_blocked_declared_gate`
- `independent_verifier_benefit`: `false`
- `artifact_field`: `fresh_source_ready_score`; `operand_status`: `observed_zero`; `passed`: `false`
- `control`: `converged_linear`; `status`: `unmeasured`
- `paper_ready`: `false`

## RECOMMENDATION
KEEP

Not checked: elided rows or upstream artifacts; administrative completion establishes no comparative benefit.
