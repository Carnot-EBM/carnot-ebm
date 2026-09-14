# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| AGGREGATE_ONLY | 2 |
| CANNOT_DETERMINE | 2 |

## experiment_7280_v640_arc_live.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The four-episode pilot completed evidence capture but produced a null result because the useful consumed-plan and scoped-validation gates were not met.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7281_v640_admission_prototype.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims the admission-fixture mechanics passed all acceptance gates, while fixed eight-label bounds are often infeasible and no learning benefit is claimed.

## WHAT IS MISSING
The artifact is truncated inside `"nomination_overlap_rows"`, so the complete `"rows"` field referenced by `"field_principles"`—including per-unit metrics for the claimed 168-row seven-arm panel—cannot be found; `"acceptance_gate_results"` records only its aggregate count.

## THE CHECK A READER CANNOT DO
Did all 168 stream-arm units produce valid, nondegenerate results supporting the passed gate, or does the aggregate hide missing, unchanged, or boundary-pinned units?

## experiment_7282_v640_admission_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Admission learning completed, but all six frozen comparative value gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7283_v640_admission_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The cold admission audit completed, but the admission-learning efficacy gate failed against the reset, unconditional-recognition, frozen-warmup, and label-shuffled controls.

## WHAT IS MISSING
The per-unit efficacy `rows` containing each stream/seed/arm’s future-error, false-accept, recurrence-degradation, recurrence-error, headroom, and censoring values; only summaries such as `efficacy_verdict`, `acceptance_gate_results`, `opportunity_reduction`, and `causal_intervention_rows` are present.

## THE CHECK A READER CANNOT DO
Did the efficacy comparisons fail broadly across independent streams/seeds, or was the pooled null driven by outliers or units with no headroom?

## experiment_7284_v640_commit_prototype.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Host group commit satisfies the changed acknowledgment contract and all acceptance gates.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7285_v640_commit_frontier.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Host-qualified batch acknowledgment achieved a best burst-throughput speedup of 15.324×, making the 10× target feasible.

## WHAT IS MISSING
A complete `"component_rows"` population: `"paired_seeds"` declares 20 seeds, but the artifact is truncated during seed 7285007, so rows for seeds 7285008–7285019 and the remainder of the artifact cannot be found.

## THE CHECK A READER CANNOT DO
Do the complete per-seed burst comparisons reproduce `"measured_best_burst_throughput_speedup": 15.324046170149051` and `"burst_throughput_lower_ci95": 11.076732511327313`?

## experiment_7286_v640_board_state.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Three authenticated board dispositions are complete: KV260 fabric graduation and PolarFire CPU dispatch remain preserved, while GateMate is blocked because the required post-Exp6559 operator-authored physical-change receipt is missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7287_v640_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The capstone is blocked because required ARC evidence is quarantined and all four scientific-value families scored zero, including several claims that one arm did not outperform its comparator.

## WHAT IS MISSING
Per-unit metric rows for the comparative source, admission, ARC, and acknowledgment arms are missing; `"current_honest_verdict"`, `"recomputed_claims"`, `"raw_evidence.row_counts"`, and aggregate value scores are present, but only counts and conclusions—not each game/seed/cell/condition’s arm values. The blocker diagnostic itself is present in `"acceptance_gate_results"` and `"current_honest_verdict"`.

## THE CHECK A READER CANNOT DO
Did the claimed failure of the mention-pointer arm versus equal-budget direct hold broadly across units, or was the aggregate null caused by outliers, degenerate controls, or units pinned at floors or ceilings?
