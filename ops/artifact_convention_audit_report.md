# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 2 |
| CANNOT_DETERMINE | 6 |

## experiment_8099_v701_fit_source_capture.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8100_radial_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked at the conductor pre-gate because upstream dependency exp8099-fit-source-capture failed the fit_capture_ready_score check with an observed value of 0 versus the expected 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8102_v701_learning_stream_capture.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8105_v701_native_radial_kernel.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8106_v701_radial_service_cost.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8107_v701_arc_supervisor_evidence.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8108_v701_radial_hardware_boundary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8109_v701_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The v701 capstone run is blocked due to failed upstream acceptance gates and missing prerequisites.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
