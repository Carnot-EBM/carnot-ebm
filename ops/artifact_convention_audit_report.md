# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| CANNOT_DETERMINE | 2 |

## experiment_8922_v763_interpolated_causal_streams.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact is truncated inside "cluster_manifest"; any subsequent claim, verdict, outcome rows, or blocker diagnostic is unavailable. The visible "acceptance_gates", "arm_work_ledger", and "attempt_rows" do not establish a comparative outcome or blocked verdict.

## THE CHECK A READER CANNOT DO
Does the missing remainder assert a comparative result or blocked outcome with the required supporting evidence?

Only the supplied fragment was checked; the missing remainder could change the classification.

## experiment_8923_label_demand_evaluation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `label_demand_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8924_v763_qwen_natural_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The 24-case canary completed, recording operational extraction and dependent annotation sensitivity without claiming independent generalization.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

External receipts were not checked; independent ground truth is explicitly absent.

## experiment_8925_v763_interpolated_benefit_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The fixed interpolated optimizer failed all three online benefit gates while passing all six retention gates.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Execution and external files were not checked; this verdict covers the supplied record.

## experiment_8926_v763_durable_cost_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Historical adaptation costs qualify, while the benefit gate fails with “observed”: 0 versus “expected”: 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Upstream files and execution were not checked; this assessment covers the supplied excerpt, which ends mid-row.

## experiment_8927_inline_request_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8927 was blocked because upstream `inline_clock_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8928_v763_hardware_operation_budget.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The hardware mapping audit passed, but hardware-operation budget readiness is blocked by documented upstream failures, and physical acceleration has not been demonstrated.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8929_v763_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible pilot branch claims all 50 cases completed and the gate requiring at least 10 errors passed.

## WHAT IS MISSING
The artifact’s remainder: it cuts off inside `"frozen_evaluation_manifest"`, before an overall verdict or measured outcome rows are visible. The visible `"unit_id"`, `"world_seed"`, and `"error_type"` fields describe scheduled conditions, not measured results.

## THE CHECK A READER CANNOT DO
Do per-case completion and error outcomes substantiate `"all_fifty_completed": true` and `"at_least_ten_errors": true`?

Only the supplied excerpt was assessed; truncation prevents establishing whether supporting rows appear later.
