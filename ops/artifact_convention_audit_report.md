# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| AGGREGATE_ONLY | 4 |
| CANNOT_DETERMINE | 1 |

## experiment_6429_constraint_saturation_verification_cost_ab.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Selective verification matched always-refine accuracy with lower median and tail cost.

## WHAT IS MISSING
Actual per-unit metric rows are missing: `per_unit_rows` is named in `field_principles` and `field_provenance`, but no `per_unit_rows` array is present in the written artifact; only aggregates such as `arms`, `by_constraint_count`, `by_constraint_count_interaction_model`, and `selective_vs_always_median_and_tail_cost_deltas` are referenced or summarized.

## THE CHECK A READER CANNOT DO
Can each underlying row show that `selective_refine` matched `always_refine` accuracy while reducing cost, rather than the aggregate being driven by a few rows or degenerate cases?

## experiment_6430_prospective_write_once_memory_capacity_frontier.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Capacity 16 is the best nonzero memory capacity/frontier point, with higher utility than smaller capacities and better utility than capacity 32.

## WHAT IS MISSING
Per-unit metric rows by capacity, e.g. rows containing `event_id`, `capacity`, `selection_success` or `future_exact_yield` contribution, `coverage`, `retention`, `write_precision`, and `utility`; present fields include aggregate `capacity_utility_frontier.rows`, `recomputed_capacity_results.by_capacity`, and only a `per_unit_row_hash`.

## THE CHECK A READER CANNOT DO
A reader cannot tell whether capacity 16’s reported `future_exact_yield: 0.75` came from broad per-event improvement or a small subset/outlier pattern hidden behind the aggregate.

## experiment_6431_controlled_memory_interference_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The authority-aware retrieval/write-control arm outperformed the capacity-matched baseline on future exact yield and failure counts.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6432_held_shift_process_restart_csl_replication.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims `selected_capacity_memory` achieved a positive held future exact yield delta over `frozen_memory` and readiness was met.

## WHAT IS MISSING
`per_unit_rows` with one row per held event/unit and its metrics; present fields include `aggregate_recomputation_receipts.recomputed_results.by_arm`, `aggregate_recomputation_receipts.recomputed_results.cells`, `effective_sample_sizes_and_uncertainty.rows`, and `per_unit_row_hash`, but those are aggregates or receipts, not the actual rows.

## THE CHECK A READER CANNOT DO
Can the 59/72 selected-capacity successes be traced across individual held events to rule out concentration in a few sessions or degenerate frozen-control rows?

## experiment_6433_csl_row_recomputation_safety_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The row recomputation audit completed, but the prospective CSL claim remains ineligible.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6434_arc_state_key_reachability_ab.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
the artifact content itself; no fields are present, including any per-unit rows or `gate_check_summary`

## THE CHECK A READER CANNOT DO
A reader cannot tell whether there was a comparative claim, a blocked verdict, or any recorded diagnostic from this artifact.

## experiment_6435_v553_adversarial_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims the V553 batch is complete but blocked: 10 of 11 upstream tasks completed, exp6434 is missing/malformed, public factor eligibility is allowed, and other claim classes are ineligible with listed blockers.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6440_inducer_h2h_qwen38_vs_gemma31b.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
No valid head-to-head comparison was made because the comparison arms were incomplete/truncated.

## WHAT IS MISSING
nothing; the blocker is recorded in `"both_arms_complete": false`, `"honest_verdict"`, `"completeness"`, and `"per_game_detail_side_by_side"` showing `"gemma31b"` and `"qwen38_27b"` with no completed cells.

## THE CHECK A READER CANNOT DO
none

## experiment_1644_cerce_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims the task completed because the CERCE ledger was added/implemented.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1736_kanele_synth.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
experiment 1736 succeeded with a generated bitfile and simulated Vivado success

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1767_e2e_qwen.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports aggregate metrics for experiment 1767: latency_ms 150.5, parse_rate 0.95, energy_score 0.88 over 100 prompts.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6436_v554_terminal_handoff_and_queue_preflight.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims V553 narrow factor evidence is eligible while verification-cost, prospective CSL, ARC reachability, public ARC, and hardware claims remain blocked by flagged, underpowered, missing, or unauthenticated evidence.

## WHAT IS MISSING
Per-unit evidence rows are missing: the artifact has `"v553_terminal_rows"`, `"v553_flagged_artifacts"`, `"v553_underpowered_artifacts"`, `"row_availability"`, `"row_count"`, and `"rows_embedded_in_primary_artifact"`, but no actual per-row metrics for the claimed improvements or gates.

## THE CHECK A READER CANNOT DO
A reader cannot check whether the claimed “improved future yield” or “lower median and tail cost” effects were broad across units or driven by one/few outliers.
