# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 2 |
| AGGREGATE_ONLY | 4 |
| CANNOT_DETERMINE | 2 |

## experiment_6428_clean_write_time_factor_admission_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The `exact_admission` arm improved future exact yield over `frozen` by 0.083333333 without increasing contamination or reducing protected retention.

## WHAT IS MISSING
nothing; `"aggregate_recomputation_receipts.by_cell"` records arm-level metrics for each cell, and `"atomic_disposition_records.rows"` provides row-level dispositions and reasons.

## THE CHECK A READER CANNOT DO
none

## experiment_6429_constraint_saturation_verification_cost_ab.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Selective verification matched always-refine accuracy while achieving lower median and tail cost.

## WHAT IS MISSING
Actual per-unit records under `"per_unit_rows"` containing each row’s arm, verdict accuracy, checker use, and elapsed cost; `"field_principles"` and `"field_provenance"` mention `"per_unit_rows"`, but the written artifact only supplies aggregates such as `"arms"`, `"by_constraint_count"`, and `"by_constraint_count_interaction_model"`.

## THE CHECK A READER CANNOT DO
Did selective refinement match always-refine accuracy and reduce cost broadly across matched rows, or was the aggregate advantage driven by a few units or degenerate controls?

## experiment_6430_prospective_write_once_memory_capacity_frontier.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Capacity 16 is the best nonzero memory capacity, achieving utility 0.59 and future exact yield 0.75, outperforming the other capacities.

## WHAT IS MISSING
Per-event, per-capacity metric rows showing each capacity arm’s outcome or utility contribution; `"capacity_utility_frontier.rows"` contains only capacity-level aggregates, while `"chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals.events"` records event attributes but not outcomes for each capacity, and `"all_recomputed_from_per_unit_rows": true` plus `"per_unit_row_hash"` merely assert and hash unshown rows.

## THE CHECK A READER CANNOT DO
Did capacity 16 outperform capacity 8 broadly across the 40 future events, or was its higher aggregate utility driven by only a few events?

## experiment_6431_controlled_memory_interference_ab.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The authority-aware arm outperformed the capacity-matched baseline, with higher `"future_exact_yield"` (0.75 versus 0.15) and fewer exposure and downstream-use failures.

## WHAT IS MISSING
The complete `"cells"` array: `"cell_count"` says 400, but the artifact ends mid-row at `"n"` and does not provide all per-unit records or the remainder containing any final verdict or gate fields.

## THE CHECK A READER CANNOT DO
Can all reported `"by_arm"` values be recomputed from the full 400 per-unit cells, rather than being driven by omitted rows or outliers?

## experiment_6432_held_shift_process_restart_csl_replication.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Selected-capacity memory beat frozen memory on held future exact yield by 0.819444444 while satisfying readiness and safety gates.

## WHAT IS MISSING
The actual `"per_unit_rows"` array containing each event’s arm, identifiers, and metrics is missing; only `"aggregate_recomputation_receipts"`, `"recomputed_results.by_arm"`, aggregated `"cells"` with `n: 4`, `"effective_sample_sizes_and_uncertainty.rows"`, and a `"per_unit_row_hash"` are present.

## THE CHECK A READER CANNOT DO
Were the selected arm’s 59 successes broadly paired across individual held events, or concentrated in particular units while other units showed no improvement or no headroom?

## experiment_6433_csl_row_recomputation_safety_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The prospective CSL claim remains ineligible because exp6432 triggered the critical `DURATION_TOO_SHORT` adversarial check.

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
the artifact contents; no fields are present after `---ARTIFACT---`

## THE CHECK A READER CANNOT DO
Does the artifact make a comparative claim or report a blocked verdict with a diagnostic?

## experiment_6435_v553_adversarial_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Clean exact write-time factor admission improved future yield without contamination or retention harm and is eligible for a public factor claim.

## WHAT IS MISSING
The actual per-unit Exp6428 A/B metric rows: unit identifiers, arm assignments, baseline and treatment future-yield values, contamination outcomes, and retention outcomes. The artifact only records `"row_availability": {"per_unit_rows_present": true, "row_count": 144}` for a separate upstream artifact; its own `"per_unit_rows"` contains task and claim-decision summaries, not experimental measurements.

## THE CHECK A READER CANNOT DO
Was the reported future-yield improvement broad across the 144 units, or driven by a few outliers or units with unequal headroom?
