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

## experiment_9137_v778_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is made. Counting blocked or unstarted work as measured verifier benefit, generalization, or service performance would contradict the stated administrative scope.

## WAS THAT CHECKED
The visible accounting preserves failed gates, reports zero scientific readiness, and records all 720 service requests as unstarted. Scientific benefit was not tested here. Elided rows and upstream artifacts were not inspected.

## EVIDENCE
- `claim_scope`: `Fourteen administrative outcomes; two static and seven online bounds remain unmeasured. No independent verifier benefit or ARC solve is claimed.`
- `acceptance_gates`: `independent_generalization_score`: `0`, `scientific_readiness`: `0`.
- `scientific_request_counts`: `attempted`: `0`, `completed`: `0`, `intended`: `720`, `unstarted`: `720`.
- `artifact_field`: `receipt_ready_score`, `expected`: `1`, `observed`: `0`, `passed`: `false`.

## RECOMMENDATION
KEEP

Review limited to the supplied excerpt; administrative completion establishes no scientific benefit.

## experiment_9138_v779_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to refute. A mismatched task object or failed required gate would contradict the administrative readiness assertion.

## WAS THAT CHECKED
Yes, for the visible administrative checks: design bytes, milestone and task count match. Remaining checks are partly elided. Scientific benefit was explicitly left unmeasured; readiness is not presented as a scientific win.

## EVIDENCE

- `honest_verdict`: `complete_null_v779_contract`
- `scientific_benefit`: `unmeasured; fixtures and historical receipts grant no independent benefit`
- `contract_ready_score`: `Exact task equality is independent of scientific availability.`
- `learning_signal_score`: `all seven bounds plus exact recovery; methods readiness requires no scientific win`

## RECOMMENDATION
KEEP

Not checked: elided material. Risk: administrative readiness must not be interpreted as measured scientific benefit.

## experiment_9139_v779_validation_schedule.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative value claim is made. The administrative account would be contradicted by an expected/observed mismatch marked as passing, or censored nodes counted toward full qualification.

## WAS THAT CHECKED
Yes, within the displayed administrative scope: semantic controls match their expectations, rejecting controls retain diagnostic flags, and execution receipts report success. The 589 censored nodes remain separate from the 32 completed nodes; full qualification scores zero. Oracle-defined fixture success is not presented as verifier benefit or generalization.

## EVIDENCE
- `exposure_scope`: `Exposed human development; administrative schedule only`
- `scientific_benefit_claimed`: `false`
- `honest_verdict`: `complete_null_schedule_qualified`
- `independent_generalization_score`: `0`
- `completed_count`: `32`; `censored_count`: `589`; `intended_count`: `621`
- `full_621_node_qualification_score`: `0`
- `Exact administrative controls confer circular validation, not an empirical benefit.`

## RECOMMENDATION
KEEP

Not checked: elided material; no rerun. Full qualification remains unestablished.

## experiment_9140_v779_forward_qualification.json

**NO_CLAIM**

## VERDICT

NO_CLAIM

## THE HEADLINE CLAIM

no claim

## WHAT WOULD REFUTE IT

There is no comparative claim to falsify. Complete successful execution, coverage, and satisfied qualification gates would contradict the recorded blocked administrative status.

## WAS THAT CHECKED

Yes, administrative qualification was assessed: execution and coverage gates remain false, with 23 completed obligations and 651 censored. Scripted fixture passes are not promoted into verifier benefit or generalization claims. A rival baseline is therefore unnecessary.

## EVIDENCE

- `honest_verdict`: `complete_blocked_infeasible_forward_schedule`
- `exposure_scope`: `Exposed human development; administrative schedule only`
- `scientific_benefit_claimed`: `false`
- `independent_generalization_score`: `0`
- `complete_original_execution`: `false`
- `complete_original_and_new_coverage`: `false`
- `completed_count`: `23`
- `censored_count`: `651`
- `qualified`: `false`

## RECOMMENDATION

KEEP

Review limited to supplied text; elided rows were not checked.

## experiment_9141_v779_human_contrast_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. The descriptive agreement report would be contradicted if missing second annotations were counted as agreements or observed paired labels failed to reproduce the reported totals.

## WAS THAT CHECKED
No comparative test applies: this records annotation summaries and protocol readiness, without model outcomes or an empirical claim of benefit. Observed and missing annotations are explicitly distinguished. The elided rows prevent independently verifying the complete aggregation.

## EVIDENCE
- `honest_verdict`: `complete_null_v779_human_contrast_audit`
- `scope`: `Human disagreement and diagnostic protocol only; no truth ceiling, model benefit, ARC solve, GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208 closure`
- `independent_generalization_score`: `0`
- `diagnostic_outcomes`: `unavailable: no model predictions consumed`
- `estimand`: `equal_weight_observed_qid_agreement`
- `observed_qids`: `100`; `missing_qids`: `200`

## RECOMMENDATION
KEEP

## experiment_9142_logit_runtime.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a blocked-run receipt. Both prerequisite gates passing would contradict its gate-failure diagnosis; it asserts no experimental success.

## WAS THAT CHECKED
Yes, the prerequisite gates were checked: forward qualification failed, while static-method readiness passed. No experimental performance was checked, so the receipt establishes no performance result.

## EVIDENCE
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `forward_validation_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

Not checked: experimental performance; the receipt supports only the blocking diagnosis.

## experiment_9150_v779_complete_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is made. Actual executed model work or qualified service measurements would contradict the administrative description that this stopped before execution.

## WAS THAT CHECKED
Yes, the displayed receipts and scientific counts record no model invocation and zero attempted or completed scientific requests. Service and adaptation acceptance gates remain false. No scientific benefit was tested or claimed.

## EVIDENCE
- `description`: `Administrative availability inventory only; no measured service or adaptation cost, independent verifier benefit or ARC level solve. Does not close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208.`
- `honest_verdict`: `complete_blocked_upstream_readiness`
- `inference_substrate_class`: `blocked_no_run`
- `model_invoked`: `false`
- `scientific_request_counts`: `attempted`: `0`, `completed`: `0`, `unstarted`: `3963`
- `complete_service`: `false`; `complete_adaptation`: `false`

## RECOMMENDATION
KEEP

Review limited to the supplied excerpt; elided rows were not inspected. No performance conclusion follows.

## experiment_9151_v779_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is an administrative accounting artifact that explicitly leaves scientific benefit, generalization, and complete costs unmeasured.

## WAS THAT CHECKED
No scientific comparison is reported or claimed. Administrative failure was possible and occurred: the schedule projection exceeded its limit, forward qualification failed, and scientific readiness remained zero. Administrative completion is not presented as scientific success.

## EVIDENCE
- `Fourteen administrative outcomes. Scientific static and online predictions and complete costs remain unmeasured. No ARC solve, hardware benefit or general theorem.`
- `Hashed primitive denominators bind accounting; administrative readiness grants no new scientific claim.`
- `complete_schedule_projection`: `expected` = `3300`, `observed` = `27880.142389201996`, `passed` = `false`.
- `scientific_readiness`: `0`.
- `current_science_paper_ready`: `false`.
- `exposure_scope`: `exposed_development`.

## RECOMMENDATION
KEEP

Not checked: elided rows or upstream artifacts. This verdict covers only the supplied excerpt.
