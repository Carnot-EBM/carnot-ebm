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

## experiment_1644_cerce_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The CERCE ledger was added, implemented, and is ready.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1767_e2e_qwen.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1736_kanele_synth.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 1736 reports `"honest_verdict": "vivado_simulated_success"` and is excluded from headline aggregation as adversarially flagged.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6469_unique_event_csl_corruption_restart.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Clean unique-event CSL improved exact yield over the frozen baseline while all corrupt events were blocked, quarantined, rolled back, and remained non-resurrecting after restart.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6470_independent_unique_event_csl_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All seven critical attacks fail closed, and independently recomputed event aggregates match the reported upstream fields.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6471_arc_generic_safety_shield_objective_ab.checkpoints.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing; `"cells"` contains per-unit records identified by `"row_id"` with `"arm"`, `"game"`, `"seed"`, `"prefix_id"`, and `"recorded_next_state_reachability"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6471_arc_generic_safety_shield_objective_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The generic safety shield passed all readiness gates without making a level-solve claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6472_v556_adversarial_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The V556 capstone claims continuous-learning and ARC eligibility based on improved future exact yield and preserved-or-improved held reachability.

## WHAT IS MISSING
Per-unit arm/control metric rows for exp6468–exp6471; the artifact provides only aggregate assertions such as `"verifier_beats_frozen_future_yield": true`, `"clean_future_exact_effect": true`, `"effects_recompute": true`, and `"shield_preserves_or_improves_held_reachability": true`, while its detailed `"rows"` contain inventory, gate, or file-identity data rather than comparative outcomes.

## THE CHECK A READER CANNOT DO
Did the claimed improvement occur broadly across held units, or was the aggregate result driven by one outlier while other units tied, regressed, or had no headroom?
