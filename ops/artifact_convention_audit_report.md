# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| AGGREGATE_ONLY | 3 |
| CANNOT_DETERMINE | 2 |

## experiment_6608_family_headroom_reducer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The benchmark is blocked because no model family has complete replayable baseline rows, leaving zero eligible families and no treatment-benefit claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6609_two_level_constrained_decoding.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `headroom_benchmark_ready_score` was 0.0 instead of the required 1.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6611_live_arc_invariant_projection.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The selected invariant projection had no effect on held exact-next-frame prediction error compared with no projection and random projection.

## WHAT IS MISSING
The complete `"per_unit_rows"` array: `"held_arm_summary"` reports 52 rows per arm, but the artifact ends partway through the first recorded row, before the remaining games, transitions, seeds, and arms can be inspected.

## THE CHECK A READER CANNOT DO
Did every held unit have identical `"charged_exact_mismatch"` across all three arms, or do the equal pooled means conceal offsetting wins, losses, outliers, or no-headroom cases?

## experiment_6612_spectral_k_block_scale_rust_parity.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
`spectral_k_block_rust` beat `sequential_gibbs`, qualifying as a software win with positive transition-efficiency and wall-time gains.

## WHAT IS MISSING
The actual `per_unit_rows` containing each size, fixture, seed, arm, and metric value; only aggregate `efficiency_summary.arm_means`, `efficiency_summary.rows`, `matched_row_count`, and `sample_size` values are shown, while `field_provenance.per_unit_rows` merely asserts that the rows exist.

## THE CHECK A READER CANNOT DO
Were the reported gains broad across the 60 matched units, or driven by a few outliers or degenerate control rows?

## experiment_6613_invariant_memory_lifecycle.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The invariant-memory lifecycle passed its conformance gate and is ready, while making no utility claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6614_prospective_invariant_self_learning.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The governed online-memory arm failed to outperform the static and shuffled controls, so continuous self-learning was not ready.

## WHAT IS MISSING
Actual `"per_unit_rows"` containing each event/seed/arm’s held-future metric are missing; only aggregate booleans appear in `"acceptance_gate_rows"` for `"held_future_over_static"` and `"held_future_over_shuffled"`, while `"field_provenance"` merely references `"per_unit_rows"`.

## THE CHECK A READER CANNOT DO
Were both failed benefit comparisons broad across units, or driven by a few outliers, degenerate controls, or units with no headroom?

## experiment_6605_qwen36_direct_headroom.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no identifiable headline claim in the visible fragment

## WHAT IS MISSING
The completed artifact, including any top-level verdict or gate result; the JSON truncates inside `"failure_rows"`, although `"attack_rows"`, `"failed_checks"`, and `"failed_closed"` are present.

## THE CHECK A READER CANNOT DO
Did the experiment’s final gate pass, fail, or become blocked, and on what recorded value?

## experiment_6615_v576_independent_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Live projection had zero effect versus both controls, and prospective self-learning had zero held-future benefit versus static and shuffled controls.

## WHAT IS MISSING
Actual per-game and per-pair metric rows; `"arm_summaries"`, `"row_count_by_arm"`, and `"row_store_counts"` provide only aggregates or row counts, while the top-level `"per_unit_rows"` contains task/comparative-group receipts rather than unit-level measurements.

## THE CHECK A READER CANNOT DO
Were the reported zero effects consistent across games and held-future pairs, or produced by degenerate, pinned, or offsetting unit-level outcomes?
