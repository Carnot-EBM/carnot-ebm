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
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 3 |

## experiment_8999_v769_branch_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to falsify. The administrative readiness assertion would fail if required authority bytes or task fields mismatched, corrupted authority was accepted, or owned validation failed despite readiness being reported.

## WAS THAT CHECKED
Yes, at administrative scope: expected/observed comparisons appear in gate_check_summary, seven mutation controls report passing in authority_mutation_rows, and validation_receipts records successful normal exits. These support custody checks, not scientific added value.

## EVIDENCE
- `claim_scope`: `Administrative custody only; no scientific benefit or independent generalization claim.`
- `honest_verdict`: `complete_null_v769_branch_contract`
- `metric`: `ordered_task_contract_match`
- `independent_count`: `0`
- `independent_generalization_score`: `0`
- `status`: `frozen_unmeasured_until_fresh_calibration_and_test`

## RECOMMENDATION
KEEP

Not checked: elided rows or referenced files. Custody readiness establishes no scientific benefit.

## experiment_9000_v769_invocation_provenance.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_9001_v769_human_source_authority.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_9002_retained_shard_a.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental claim to falsify. The recorded blocking decision would be contradicted by a provenance score satisfying the required gate while that gate was reported as failed.

## WAS THAT CHECKED
Yes. In `gates_evaluated`, the provenance gate records actual 0 against expected 1 and fails; the branch-contract gate passes. This supports the reported block. Authentication itself was not evaluated.

## EVIDENCE
`blocked_gate_check_v1`, `blocked`, `blocked_gate_check_failed`, `provenance_ready_score`, `actual=0 == expected=1`, `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9003_graph_corpus_b.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No experimental claim is made. The blockage diagnosis would be contradicted if both recorded prerequisite gates passed.

## WAS THAT CHECKED
Yes. The provenance gate failed with actual 0 against expected 1; the graph-authority gate passed. Only prerequisite checks are recorded.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `gate_check_summary`: `gate-unsat(final): 1 of 2 gate(s) failed; first failure: exp9000-invocation-provenance.provenance_ready_score (actual=0 == expected=1)`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

Not checked: experimental performance. This receipt establishes only the gate blockage.

## experiment_9005_natural_pilot_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt, with no claim of method performance or added value.

## WAS THAT CHECKED
The readiness gate was checked and failed: observed 0 versus required 1. The artifact reports no pilot comparison to assess.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `natural_pilot_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9011_arc_transfer_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked prerequisite receipt, not an empirical claim about supervisor transfer.

## WAS THAT CHECKED
The readiness gate was checked and failed: observed 0, required 1. Transfer performance was not evaluated and remains unknown.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `arc_new_outcomes_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_9012_v769_capstone.json

**SKIPPED_ALREADY_FLAGGED**
