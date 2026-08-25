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
| CLAIM_REFUTED_BY_OWN_DATA | 1 |
| NO_CLAIM | 1 |
| SKIPPED_ALREADY_FLAGGED | 6 |

## experiment_3377_archive_v310_activate_v311.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_3392_archive_v311_activate_v312.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_833_constraint_delta_root_cause.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6582_gemma4_31b_flagship_source_shard.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6583_gemma4_26b_a4b_flagship_source_shard.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6585_v573_terminal_recovery_and_execution_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For the limited operational assertion, a terminal state or hard-limit attempt failing replay, fewer than three distinct Exp6584 attempts, or an unbounded V573 task would refute it; there is no comparative scientific or verifier-value claim to falsify.

## WAS THAT CHECKED
Yes, for operational integrity: terminal and attempt rows, execution-budget contracts, gate checks, and fail-closed attack rows were checked. No scientific comparison was applicable.

## EVIDENCE
`honest_verdict` `complete: all six V572 terminal states and three Exp6584 hard-limit attempts replay; V573 model tasks are bounded; no science verdict was created` `verdict_class` `null` `inference_substrate` `v572_terminal_receipt_replay_no_llm` `verifier_is_oracle` `true` `infrastructure_contract_no_science` `method_contract_no_model_outcome` `runtime_evidence_no_quality_verdict`

## RECOMMENDATION
KEEP

## experiment_6586_isolated_full_suite_truth_baseline.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6587_v573_constraint_first_method_contract.json

**CLAIM_REFUTED_BY_OWN_DATA**

## VERDICT
CLAIM_REFUTED_BY_OWN_DATA

## THE HEADLINE CLAIM
The source, fixture, router, metric, and exact-authority contracts are all ready before any V573 model outcome.

## WHAT WOULD REFUTE IT
Any required contract-readiness check remaining open or failed would falsify the claim of complete readiness.

## WAS THAT CHECKED
Yes, in the gate-check summary; it reports that checks were not closed and identifies one failed check.

## EVIDENCE
`honest_verdict`: `complete: source, fixture, router, metric, and exact-authority contracts are ready before any V573 model outcome`; `checks_closed`: `false`; `failed_check_count`: `1`; `check`: `arms_authority_metrics`; `v573_constraint_first_method_ready_score`: `1.0`

## RECOMMENDATION
CORRECT_THE_RECORD
