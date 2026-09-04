# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 1 |

## experiment_6976_exact_candidate_certification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Candidate certification was complete; all three schedules had 0/18 exact successes, while direct achieved 9/18 parse successes and the other schedules achieved none.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6977_certified_pwa_kan_energy.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6978_transactional_constraint_self_learning.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact is truncated mid-object inside `"checkpoint_rows"` and lacks the remaining fields, including any headline/verdict or blocked-status fields; only `"MODEL_SPECS"`, `"arm_config_rows"`, `"budget_rows"`, and a partial `"checkpoint_rows"` are present.

## THE CHECK A READER CANNOT DO
Did the complete artifact report a comparative result or blocked verdict, and did it include the required per-unit metrics or blocker diagnostic?

## experiment_6979_self_learning_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The completed audit found no self-learning benefit: transactional write produced zero gain over read-only and failed the positive gate.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6980_spilled_energy_requalification.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Spilled energy improves AUROC over top probability by 0.30 overall and ties entropy.

## WHAT IS MISSING
Per-`pair_id` metric rows containing each unit’s label and spilled-energy, entropy, and top-probability scores; only aggregate `"control_comparison_rows"` and `"bootstrap_interval_rows"` are present, while `"abstention_rows"` contain no metric values.

## THE CHECK A READER CANNOT DO
Did spilled energy outperform top probability broadly across the paired units, or was the reported AUROC gain driven by one outlier pair?

## experiment_6981_arc_live_engine_generalization_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The ARC live-engine generalization audit was blocked because none of the three post-cutoff candidates was eligible and required hashes and completed live-run engine binding were absent or invalid.

## WHAT IS MISSING
nothing; `gate_check_summary`, `preconditions_checked`, `rows[].rejection_reasons`, and `selected_run_provenance` record the failed checks and their observed values.

## THE CHECK A READER CANNOT DO
none

## experiment_6982_hard_feasible_hybrid_selection.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp6977-certified-pwa-kan-energy.certified_pwa_energy_ready_score` was 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6983_v611_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V611 capstone completed but found no non-circular science-positive result.

## WHAT IS MISSING
nothing; `"per_task_results"`, `"branch_status_rows"`, `"blocked_cause_rows"`, and `"gate_check_summary"` record the task-level evidence and identify the failed check as `"non_circular_science_positive"` with observed value `0` versus expected value `1`.

## THE CHECK A READER CANNOT DO
none
