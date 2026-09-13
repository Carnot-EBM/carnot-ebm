# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| AGGREGATE_ONLY | 1 |

## experiment_7252_v638_semantic_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the upstream `experiment_7251_v638_mention_heldout` artifact was missing, so no audit or comparative evaluation ran.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7253_v638_coverage_memory.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The bounded coverage fixture and its controls are ready and passed the stated acceptance gates.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7254_v638_coverage_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The completed experiment produced a null verdict because bounded coverage memory failed several frozen acceptance gates.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7255_v638_coverage_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The coverage audit completed, but promotion criteria did not all pass.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7256_v638_native_controller.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
All three arms achieved exact oracle parity, and the persistent native controller eliminated active reconstruction and hot-path JSON parsing.

## WHAT IS MISSING
Complete per-event `"parity_rows"` for all three arms; the supplied `"parity_rows"` shows only `"python_reference"` before truncating, while `"acceptance_gate_results.exact_three_arm_parity"` and `"independent_reducer_receipt"` provide only aggregate zero-mismatch summaries.

## THE CHECK A READER CANNOT DO
Did every old-native-wrapper and persistent-native-controller event match the Python reference, or were their claimed zero mismatches produced only by the aggregate reducer?

## experiment_7257_v638_native_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims the 10× and 100× performance gates were unmet, capacity-one passed, capacity-four failed, and parity and durability checks passed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7258_v638_board_state.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Three authenticated board dispositions were recorded: KV260 and PolarFire graduations remain preserved, while GateMate is blocked because the required operator-authored physical-state-change receipt is missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7259_v638_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V638 capstone is blocked because required external scientific evidence is unavailable and multiple acceptance gates failed despite completion of the fourteen-task matrix.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
