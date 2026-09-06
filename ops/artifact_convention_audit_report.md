# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| CANNOT_DETERMINE | 1 |

## experiment_7071_bcit_drift_rollback_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `bcit_comparison_complete_score` was observed as 0 but required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7072_v619_live_arc_compaction_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live ARC compaction A/B experiment was blocked because zero eligible hidden or rotation units were available, below the required minimum of 30.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7075_v619_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
V619 ended with null terminal dispositions because all four branches were resource-blocked.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7076_v620_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V620 contract preflight was disqualified because the Markdown task contract could not be parsed.

## WHAT IS MISSING
nothing; `gate_check_summary` records `failed_check` as `"contract_parse"`, `expected_value` as `"two_parseable_independent_contracts"`, and the specific malformed row in `observed_value`.

## THE CHECK A READER CANNOT DO
none

## experiment_7077_v620_sota_ingestion.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V620 SOTA ingestion completed successfully and met its gate.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7078_v620_gpu_lease_migration.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
GPU lease migration compatibility passed all recorded real and synthetic safety gates, with no scientific comparative claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7079_v620_gpu_lease_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The GPU lease cold audit passed every gate and is ready, while making no model-quality claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7080_v620_three_family_entrance_bank.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim ascertainable because the artifact is truncated mid-row

## WHAT IS MISSING
The complete artifact, including any headline/verdict field and any `gate_check_summary`; present fields include `"entrance_proposal_bank_complete_score"`, `"cleanup_rows"`, and per-unit `"exact_label_rows"`.

## THE CHECK A READER CANNOT DO
Did the final verdict claim a comparative result or declare the task blocked, and if blocked, did it record the failed check and observed value?
