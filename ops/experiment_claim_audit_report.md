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

## experiment_8789_v753_branch_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. A positive qualification claim would be refuted by failed mandatory validation or incomplete required coverage.

## WAS THAT CHECKED
Yes, administrative qualification could fail and did: owned validation failed and original statement coverage remained incomplete. The artifact reports disqualification and zero readiness rather than promoting these results as scientific success.

## EVIDENCE

- `exposure_scope`: `administrative custody only; planned constructed protocols are not executed science`
- `honest_verdict`: `complete_disqualified_v753_branch_qualification`
- `acceptance_gates`: `owned_validation`: `false`; `original_failure_publication_and_coverage`: `false`
- `original_owned_statement_coverage`: `complete`: `false`
- `memory_protocol_ready_score`: `0`
- `new_scientific_measurements`: `0`
- `learning_execution_authorized`: `false`

## RECOMMENDATION
KEEP

## experiment_8790_delayed_memory_prototype.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-execution receipt, with no claim of performance, generalization, or added value.

## WAS THAT CHECKED
The prerequisite gate was checked and failed: the readiness score was 0 against a required 1. The artifact reports blocking at that gate, not a positive experimental result.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `memory_protocol_ready_score`
- `expected`: `1`; `actual`: `0`; `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8791_v753_evidence_intervention_methods.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8794_evidence_intervention_evaluation.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a failed prerequisite gate and makes no claim about evidence invariance or semantic sensitivity.

## WAS THAT CHECKED
Only prerequisite readiness was checked: the observed score was 0 against a required 1. The artifact reports the evaluation as blocked.

## EVIDENCE
`schema` = `blocked_gate_check_v1`; `status` = `blocked`; `honest_verdict` = `blocked_gate_check_failed`; `failed_field` = `evidence_interventions_ready_score`; `failed_expected` = `1`; `failed_observed` = `0`; `blocked_at_layer` = `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8795_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked attempt, not an extraction result or comparative claim.

## WAS THAT CHECKED
Only prerequisite readiness was checked. Both gates observed 0 against required 1 and failed; the artifact reports no extraction outcomes.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `gate_check_summary`: `gate-unsat(final): 2 of 2 gate(s) failed; first failure: exp8789-branch-qualification.runtime_repair_ready_score (actual=0 == expected=1)`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8796_v753_observer_intervention.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative hypothesis is asserted. Completed windows with measured timings would contradict the narrower factual report that execution stopped at admission.

## WAS THAT CHECKED
Admission was checked and rejected. Performance refutation had no opportunity: controls and measurement windows never executed. The artifact explicitly reports this blockage and makes no positive claim about verifier value, generalization, or performance.

## EVIDENCE

- `honest_verdict`: `complete_blocked_upstream_contract`
- `observed`: `V752 JSON contract missing`
- `measurement_executed`: `false`
- `completed_count`: `0`
- `censored_count`: `36`
- `methodology_note`: `Admission only: original upstream contract rejected; no observer controls or durable windows executed. All costs remain missing. Historical negative overhead is instability, not native acceleration. No population throughput, stage-local gain or complete-service benefit is established.`

## RECOMMENDATION
KEEP

## experiment_8798_arc_new_outcome_decision.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-execution receipt with no claim about supervisor performance or added value.

## WAS THAT CHECKED
Only the prerequisite gate was evaluated. It failed with actual 0 versus expected 1. No supervisor-choice assessment is reported.

## EVIDENCE
`blocked_gate_check_v1`, `blocked_gate_check_failed`, `actual=0 == expected=1`, `gates_evaluated`, `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8800_v753_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this administrative reconciliation asserts no comparative method benefit, learning result, or generalization finding.

## WAS THAT CHECKED
No scientific comparison was executed. The displayed source-admission rows exclude disqualified, blocked, and quarantined sources from headline evidence; the capstone reports zero scientific measurements. This records unexecuted work, not an empirical null result.

## EVIDENCE
- `exposure_scope`: `administrative twelve-slot reconciliation; no scientific measurement or independent reproduction`
- `honest_verdict`: `complete_blocked_v753_capstone`
- `scientific_measurement_count`: `0`
- `independent_generalization_score`: `0`
- `headline_admitted`: `false`
- `scientific_measurement_admitted`: `false`

## RECOMMENDATION
KEEP
