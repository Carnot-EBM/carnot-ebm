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

## experiment_9054_v773_branch_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A task-contract mismatch, altered retained bytes, or a historical failure promoted to scientific success would contradict the administrative custody assertions. There is no scientific benefit or generalization headline to falsify.

## WAS THAT CHECKED
Yes, administratively: contract comparisons, input-byte checks, authority mutations, and preserved-outcome checks report passes. Scientific comparisons remain explicitly unmeasured, and exposed development results are labeled accordingly.

No rerun or inspection of elided rows was performed; conclusions are limited to the supplied excerpt.

## EVIDENCE
- `claim_scope`: `Administrative custody only; no scientific benefit or independent generalization claim.`
- `honest_verdict`: `complete_null_v773_branch_contract`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `scope`: `exposed_development_only`
- `evaluation_status`: `frozen_unmeasured; later real streams required`
- `check`: `input_bytes`; `passed`: `true`
- `check`: `preserved_v772`; `passed`: `true`

## RECOMMENDATION
KEEP

## experiment_9055_v773_prospective_roster.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. A hash mismatch, incomplete roster, or acceptance of an altered roster would contradict the limited custody assertions.

## WAS THAT CHECKED
Yes, at the custody level: input comparisons appear in gate_check_summary, completeness in acceptance_gates and roster counts, and alteration controls in custody_mutation_rows. These checks support a receipt, not a performance conclusion. Elided records were not reviewed.

## EVIDENCE

- `claim_scope`: `Prospective source identities and custody only. Executable graph truth is an oracle; no independent generalization or detector benefit is measured.`
- `honest_verdict`: `complete_null_prospective_roster`
- `verdict_class`: `null`
- `intended_count`: `1024`; `completed_count`: `1024`
- `name`: `altered_roster`; `passed`: `true`
- `metric`: `graph_role_frozen`; `outcome_status`: `withheld`
- `model_invoked`: `false`

## RECOMMENDATION
KEEP

## experiment_9056_v773_human_release_authority.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative performance claim to falsify. Authenticated original individual judgments attributable to annotators and linked to responses would contradict the reported missing-judgment obstruction.

## WAS THAT CHECKED
Yes, at the reported release-audit level: the displayed annotation rows and workbook schemas assess judgment attribution and authentication and report them unresolved. No model performance or generalization conclusion is asserted.

## EVIDENCE
- `claim_scope`: `Original MDH release authority audit only; no verifier, detector, factual oracle or generalization result.`
- `honest_verdict`: `complete_blocked_original_independent_human_judgments_per_response`
- `verdict_class`: `blocked`
- `original_annotator_ids`: `missing_field`
- `completed_individual_judgments`: `unknown`
- `original_judgment_fields_authenticated`: `false`
- `inference_substrate_class`: `no_model_load`

## RECOMMENDATION
KEEP

Elided rows and external files were not checked; this assessment covers only the supplied excerpt.

## experiment_9057_human_probe_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite gate and makes no claim about probe performance or comparative value.

## WAS THAT CHECKED
Yes—the prerequisite gate was evaluated and failed. No method-performance claim was tested; execution was blocked before the experiment.

## EVIDENCE
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `human_source_ready_score`
- `failed_expected`: `1`; `failed_observed`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9058_v773_fresh_corpus_a.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A missing or failed first response, or a custody hash mismatch, would refute the operational assertion of complete collection. No comparative verifier benefit or generalization claim is made.

## WAS THAT CHECKED
Yes, for collection and custody: intended, attempted and completed counts are all 512; failures, exclusions and censoring are zero; authentication and authority checks pass. Transport completion could have failed independently of graph correctness. The oracle therefore creates no circular value claim here, and no rival is needed for this receipt.

## EVIDENCE
- `claim_scope`: `First-response custody only. Executable graph truth is an oracle; no detector benefit or independent generalization is measured.`
- `metric`: `transport_completed`
- `intended_count`: `512`; `completed_count`: `512`; `failed_count`: `0`
- `generation_calls_attempted`: `512`; `generation_calls_completed`: `512`
- `authenticated`: `true`; `required_checks_passed`: `true`
- `claim_scope`: `Existing producer serialization and scoped custody only; no repository health or verifier benefit claim.`

## RECOMMENDATION
KEEP

Review limited to supplied text; elided rows were not inspected. This receipt establishes no comparative benefit.

## experiment_9060_sealed_corpus_admission.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked admission receipt, with no claim of successful admission, generalization, or comparative value.

## WAS THAT CHECKED
Yes—the prerequisite gates were checked, and one failed because an upstream artifact was missing. The artifact reports that block. No experimental performance claim was evaluated.

## EVIDENCE
`schema`: `blocked_gate_check_v1`  
`status`: `blocked`  
`honest_verdict`: `blocked_gate_check_failed`  
`blocked_reason`: `upstream artifact not found for task id 'exp9059-fresh-corpus-b'`  
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9066_arc_transfer_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite and asserts no transfer result.

## WAS THAT CHECKED
The prerequisite was checked and failed: readiness was 0 against a required 1. Transfer performance was not tested, so this artifact cannot establish or refute transfer.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `arc_new_outcomes_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_9067_v773_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to falsify. An omitted intended task or an incorrectly passed contract check would contradict the administrative accounting statement.

## WAS THAT CHECKED
Yes, administrative checks are reported in full_contract_rows and gate_edge_rows, including failed readiness gates. Most accounting rows are elided.

No comparative benefit was tested: even the serious converged-linear controls remain unmeasured. Oracle status is not presented as verifier added value, and no generalization success is asserted. Unmeasured rows retain null values and failed flags; they are not reported as scientific successes.

## EVIDENCE

- `claim_scope`: `Fourteen administrative outcomes; seven graph, ten learning and four human bounds unmeasured. No new oracle-distinct, cost or deployment benefit.`
- `honest_verdict`: `complete_blocked_declared_gate`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `seven_bound_rows`: `converged_linear`, `unmeasured`, `missing_file`, `value`, `null`, `passed`, `false`

## RECOMMENDATION
KEEP

Not checked: upstream artifacts or elided rows. Risk: administrative completion establishes no comparative scientific benefit.
