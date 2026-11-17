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

## experiment_9141_v779_human_contrast_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative efficacy claim to falsify. The descriptive agreement summaries would be contradicted by paired annotation rows yielding different totals, including missing second annotations incorrectly counted as agreements.

## WAS THAT CHECKED
Partly: the agreement intervals distinguish observed from missing qids, but the elided rows prevent full reconciliation. No model benefit comparison was performed or claimed; this is a descriptive annotation audit and protocol receipt, not an empirical finding of no benefit.

## EVIDENCE

- `honest_verdict`: `complete_null_v779_human_contrast_audit`
- `scope`: `Human disagreement and diagnostic protocol only; no truth ceiling, model benefit, ARC solve, GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208 closure`
- `independent_generalization_score`: `0`
- `diagnostic_outcomes`: `unavailable: no model predictions consumed`
- `estimand`: `equal_weight_observed_qid_agreement`
- `observed_qids`: `100`; `missing_qids`: `200`

## RECOMMENDATION
KEEP

Elided rows and referenced files were not inspected; complete agreement aggregation remains unverified from this excerpt.

## experiment_9142_logit_runtime.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite check, without claiming successful qualification, completed shard budgets, or comparative value.

## WAS THAT CHECKED
Yes—the readiness gate required `forward_validation_ready_score` to equal `1`, observed `0`, and failed. The artifact reports the resulting block. Experimental performance was not evaluated.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `blocked_reason`: `actual=0 == expected=1`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9150_v779_complete_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is made. Completed scientific work or measured service clocks in the artifact’s own rows would contradict its administrative-only, no-run account.

## WAS THAT CHECKED
The visible receipts and scientific request counts consistently report no execution. No performance comparison occurred, and the artifact does not claim one.

## EVIDENCE
- `description`: `Administrative availability inventory only; no measured service or adaptation cost, independent verifier benefit or ARC level solve. Does not close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208.`
- `honest_verdict`: `complete_blocked_upstream_readiness`
- `inference_substrate_class`: `blocked_no_run`
- `model_invoked`: `false`
- `scientific_request_counts`: `attempted`: `0`, `completed`: `0`, `unstarted`: `3963`

## RECOMMENDATION
KEEP

Not checked: elided rows or external files. This inventory supports no performance conclusion.

## experiment_9151_v779_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is an administrative aggregation reporting blocked scientific work, without asserting comparative benefit or generalization.

## WAS THAT CHECKED
Administrative failure was possible and recorded: the schedule projection exceeded its limit, and forward qualification failed. Scientific requests remained unstarted; their absence was not counted as scientific success.

## EVIDENCE

- `Fourteen administrative outcomes. Scientific static and online predictions and complete costs remain unmeasured. No ARC solve, hardware benefit or general theorem.`
- `scientific_readiness`: `0`
- `current_science_paper_ready`: `false`
- `scientific_request_counts`: `attempted`: `0`, `completed`: `0`, `unstarted`: `3963`
- `complete_schedule_projection`: `expected`: `3300`, `observed`: `27880.142389201996`, `passed`: `false`
- `forward_validation_ready_score`: `expected`: `1`, `observed`: `0`, `passed`: `false`

## RECOMMENDATION
KEEP

Not checked: elided and truncated content; this verdict covers only the supplied excerpt.

## experiment_9152_v780_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A task-object mismatch, altered retained methods, or invalid receipt would refute administrative readiness. There is no comparative scientific claim here to falsify.

## WAS THAT CHECKED
Yes, administrative checks are reported: the visible design-byte, milestone, and task-count comparisons pass. Most checks are elided, so the excerpt cannot establish complete object equality independently. Scientific benefit is explicitly unmeasured; exposed development is acknowledged rather than claimed as generalization. Historical unqualified outcomes are preserved as failures.

## EVIDENCE
- `Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208.`
- `unmeasured; fixtures and historical receipts grant no independent benefit`
- `Original LPHB900 answers/300 qids remain exposed development`
- `independent_generalization_score`: `0`
- `current_science_paper_ready`: `false`

## RECOMMENDATION
KEEP

Review limited to the supplied excerpt; elided checks and external receipts were not independently inspected.

## experiment_9153_v780_validation_schedule.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or scientific-benefit claim is asserted. Failed required checks or censored obligations would defeat a claim of full qualification, but the artifact withholds that claim.

## WAS THAT CHECKED
Only partial administrative validation is reported: 32 completed checks, 642 censored, and both full-qualification scores zero. Oracle-based fixture checks establish no verifier advantage; none is claimed. Review is limited to the supplied excerpt; no rerun was performed.

## EVIDENCE

- `honest_verdict`: `complete_blocked_validation_prerequisite`
- `exposure_scope`: `Exposed human development; administrative schedule only`
- `scientific_benefit_claimed`: `false`
- `independent_generalization_score`: `0`
- `intended_count`: `674`; `completed_count`: `32`; `censored_count`: `642`
- `full_621_node_qualification_score`: `0`
- `full_674_node_qualification_score`: `0`
- `computational_root_cause`: `unknown; retained stack shows a subprocess wait only`

## RECOMMENDATION
KEEP

## experiment_9154_forward_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific claim is asserted. This is a receipt recording a failed prerequisite, not evidence that forward validation succeeded.

## WAS THAT CHECKED
Yes: the prerequisite required a readiness score of 1, observed 0, and failed. Execution was blocked before validation; no comparative result or generalization claim was tested.

## EVIDENCE
`schema`: `blocked_gate_check_v1`
`status`: `blocked`
`honest_verdict`: `blocked_gate_check_failed`
`artifact_field`: `schedule_ready_score`
`expected`: `1`
`actual`: `0`
`passed`: `false`
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9165_v780_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. For the administrative accounting, a missing or failed prerequisite accepted as passing would contradict the reported gating behavior.

## WAS THAT CHECKED
Yes, in the visible declared-gate rows: the zero schedule-readiness value and missing forward-validation field both fail. Scientific requests remain unstarted, and administrative completion is explicitly scoped separately from scientific readiness. No scientific performance comparison was executed.

## EVIDENCE
- `scope`: `Fourteen administrative outcomes. Scientific static and online predictions and complete costs remain unmeasured; measured validation remains blocked. No ARC solve, hardware benefit or general theorem.`
- `scientific_request_counts`: `attempted` = `0`, `completed` = `0`, `unstarted` = `3963`.
- `operand_status`: `missing_field`; `passed`: `false`.
- `scientific_readiness`: `0`.

## RECOMMENDATION
KEEP

Not checked: elided rows or upstream artifacts. This receipt establishes no scientific benefit.
