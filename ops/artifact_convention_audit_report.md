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

## experiment_7825_v680_training_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The training and online runtime checks passed on six public fixture families; the artifact claims no natural or held-out benefit.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7826_view_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked prior to execution because upstream experiment `exp7824-source-feature-isolation` failed two pre-execution gate requirements (`source_isolation_ready_score` was 0 instead of 1, and `verdict_class` was `disqualified`).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7828_v680_counter_evidence_protocol.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The fixture completed but was disqualified by required validation, with no claim of generalization or benefit.

## WHAT IS MISSING
The supplied JSON cuts off inside `preconditions_checked`. The visible `gate_check_summary` records only a passing `worktree_imports.exit_code`; any later `validation_receipts` or failed check and observed value are unavailable.

## THE CHECK A READER CANNOT DO
Which required validation failed, and what value did it report?

## experiment_7829_qwen_counter_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7831_v680_arc_supervisor_refinement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7834_v680_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact disqualifies the run and defers acquisition because service evidence is missing and a required coverage check failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7835_v680_independent_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The independent evidence audit is disqualified due to required validation failures and missing or disqualified upstream experiment artifacts.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7836_v680_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone is disqualified because required validation and upstream evidence checks failed.

## WHAT IS MISSING
nothing — `gate_check_summary` and `preconditions_checked.failed_operands` name the failed checks and give their expected and observed values.

## THE CHECK A READER CANNOT DO
none
