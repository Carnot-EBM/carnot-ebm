# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 1 |
| AGGREGATE_ONLY | 3 |
| CANNOT_DETERMINE | 4 |

## experiment_7198_v634_feedback_capacity_stream.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The bounded feedback stream is ready, but no learning arm demonstrated a benefit.

## WHAT IS MISSING
The artifact is truncated mid-`"information_budget_rows"` and does not expose a complete top-level `"rows"` field containing each unit’s arm, seed, outcome metric, error, and abstention; `"headroom_rows"` records control headroom, while the visible `"information_budget_rows"` records resource and feedback counts rather than comparative outcomes.

## THE CHECK A READER CANNOT DO
Did the learning arms consistently fail to improve over the static control across seeds and capacity-delay cells, or is that null conclusion driven by only a few units?

## experiment_7194_v634_arc_gap_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit completed with no missing tool identified, while observed banked progress was explicitly classified as noncausal.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7199_v634_bounded_acquisition.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims comparative admission-policy effects and reports that the acquisition run completed but failed to demonstrate acquisition value (`"acquisition_value_score": 0`).

## WHAT IS MISSING
Per-`"stream_seed"` metric rows for every arm, control, capacity, delay schedule, window, and metric; `"comparison_rows"` contains only aggregate `"difference"`, `"ci95_low"`, `"ci95_high"`, and `"independent_unit_count"` values.

## THE CHECK A READER CANNOT DO
Were the reported arm-versus-control differences broad across the 10 stream seeds, or driven by one outlier or degenerate seeds with no headroom?

## experiment_7200_v634_acquisition_cold_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The `"acquisition_audit_complete_score": 1` claims the acquisition audit completed successfully.

## WHAT IS MISSING
The complete `"cold_reload_rows"` array is missing: the artifact truncates mid-`"unit_id"` despite `"checkpoint_receipt"` reporting `"unit_count": 40`.

## THE CHECK A READER CANNOT DO
Did every one of the 40 cold-reload units pass its recorded checks?

## experiment_7201_v634_slice_pyo3.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The bridge-overhead hypothesis is supported because persistent PyO3 had lower latency on all 40 matched units.

## WHAT IS MISSING
The artifact is truncated inside `"distribution_rows"`, so it is impossible to determine whether later per-unit latency rows exist; the visible portion has only aggregates such as `"matched_units"`, `"pyo3_lower_latency_units"`, `"median_phase_duration_s"`, and `"median_subprocess_to_pyo3_kernel_ratio"`, while the visible rows report distribution metrics rather than matched latency measurements.

## THE CHECK A READER CANNOT DO
Did each of the 40 matched units actually show lower persistent-PyO3 latency than subprocess-bridge latency?

## experiment_7202_v634_slice_cost_quality.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The primary local boundary gate failed, sample quality was insufficient, and the 10× NFR-01 speedup target was not met.

## WHAT IS MISSING
The artifact is visibly truncated mid-`quality_rows`, so the per-seed paired latency rows supporting `primary_gate.python_speedup_ci95` and `primary_gate.latency_speedup_over_subprocess_ci95` cannot be found or confirmed absent; only aggregate fields such as `"paired_units": 10` are visible.

## THE CHECK A READER CANNOT DO
Did the 10× speedup fail broadly across the ten paired seeds, or only because of one or two extreme latency observations?

## experiment_7203_v634_hardware_correction.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
A hypothetical delayed-acceptance device would beat exact-host latency and break even under the condition rows where `"hypothesis_breaks_even": true`.

## WHAT IS MISSING
Per-seed rows for the 10 seeds underlying `"measured_exact_host_latency_s_per_proposal_mean"`, `"measured_correction_host_latency_s_per_proposal_mean"`, and `"measured_stage_one_acceptance_rate_mean"`; only `"source_seed_count": 10` and aggregate means are present.

## THE CHECK A READER CANNOT DO
Did the predicted break-even advantage occur broadly across the 10 seeds, or was the pooled mean driven by one outlier seed?

## experiment_7204_v634_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The V634 capstone matrix is complete, but several value and performance gates were not met, warranting retirement or changed prerequisites.

## WHAT IS MISSING
Per-game, seed, cell, or condition metric rows underlying the failed comparative gates; `"evidence_matrix"` provides task-level `"metric_value"` and `"raw_row_count"` summaries, while `"branch_decisions"` gives verdicts and reasons but not the underlying unit measurements.

## THE CHECK A READER CANNOT DO
Were the reported gate failures broad across units, or caused by a few outliers, degenerate controls, or floor/ceiling-pinned cases?
