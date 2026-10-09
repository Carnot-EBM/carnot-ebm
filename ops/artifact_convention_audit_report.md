# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| CANNOT_DETERMINE | 3 |

## experiment_8336_continuous_local_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8336 was blocked because upstream `local_kernel_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8337_bounded_feedback_capacity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream "local_kernel_ready_score" was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8340_v719_runtime_reader_qualification.json

**CANNOT_DETERMINE**

## VERDICT

CANNOT_DETERMINE

## WHAT THE CLAIM IS

The headline claim cannot be identified from the truncated artifact.

## WHAT IS MISSING

The complete top-level verdict and any associated blocker diagnostic. "acceptance_gates" records three false gates, but the supplied text ends mid-field inside "execution_authority.contract_rows".

## THE CHECK A READER CANNOT DO

Does the final verdict declare the task blocked, and identify which failed check caused that disposition?

## experiment_8341_changed_runtime_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The bounded Qwen canary was blocked because both qualification gates, "runtime_changed_score" and "cuda_context_ready_score", recorded 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8342_v719_arc_supervisor_frontier.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact truncates inside "coverage_statement_counts", leaving its headline and final verdict unavailable. "acceptance_gates" records "owned_checks": false, but the visible "verdict_class" describes an upstream artifact in "authority_locator_rows". Any later "gate_check_summary" is unavailable.

## THE CHECK A READER CANNOT DO
Does this artifact’s final verdict assert a comparative result, report a diagnosed blocker, or make no claim?

## experiment_8343_v719_kv260_workload_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible artifact reports accelerator benefit as unproved because no compatible operation was available.

## WHAT IS MISSING
The remainder of "arithmetic_rows" and subsequent fields: the artifact ends mid-key at "source_i". "accelerator_benefit" already records a diagnostic, "unproved_no_compatible_operation".

## THE CHECK A READER CANNOT DO
Does the missing remainder make a comparative claim and supply the per-unit metrics needed to check it?

## experiment_8344_v719_gatemate_change_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
GateMate physical preflight remains unexecuted; "gate_check_summary" diagnoses "gatemate_history" as "identity_or_configuration_drift" and records "physical_change_evidence.exists" as false.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8345_v719_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone has execution-ready score 0, with H1 and H2 blocked and unmeasured; no arm superiority is claimed.

## WHAT IS MISSING
nothing — “paired_cost_gain” is null; “sealed_predictions.purpose” identifies missing evaluator rows and scientific controls, and “branch_replay_receipts” records failed checks with “actual_exit”: 1 versus “expected_exit”: 0.

## THE CHECK A READER CANNOT DO
none
