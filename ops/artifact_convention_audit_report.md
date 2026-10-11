# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 3 |

## experiment_8408_native_transaction_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The comparison was blocked because upstream `native_parity_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8409_v724_retrospective_label_criterion.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
All listed acceptance gates are met, including a positive primary lower bound.

## WHAT IS MISSING
Per-unit primary metric rows identifying each question/cluster and arm or condition. "acceptance_gates.positive_primary_lower_bound" records only `true`; "authority.contract_rows" contains contract checks, not primary comparison outcomes.

## THE CHECK A READER CANNOT DO
Is the claimed positive comparison supported broadly across question clusters, or driven by a few outliers?

## experiment_8410_v724_cuda_reader_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt marks the cached-reader gate accepted, but ends before revealing the current terminal verdict.

## WHAT IS MISSING
The remainder of the artifact: current `"honest_verdict"`, `"verdict_class"`, and reader-qualification outcome rows. `"acceptance_gates"` contains booleans; `"execution_authority.contract_rows"` documents task matching; the excerpt ends inside `"execution_authority.tasks[1].prompt"`.

## THE CHECK A READER CANNOT DO
Does `"cached_reader": true` follow from recorded per-unit qualification results?

## experiment_8411_bounded_qwen_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The canary was blocked because `runtime_changed_score` and `cuda_context_ready_score` each equaled 0 while their gates required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8412_v724_arc_panel_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The eight-episode panel qualified without establishing scientific benefit.

## WHAT IS MISSING
The eighth episode’s “per_game_results” row, for “lf52-271a04aa:22:on”, including “absolute_metric”, “progress_count”, and “status”; the artifact cuts off inside row seven, while “intended_count” and “completed_count” both report 8.

## THE CHECK A READER CANNOT DO
Did supervisor-on produce different progress from supervisor-off for lf52-271a04aa at seed 22?

## experiment_8413_v724_board_operation_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
KV260 offload remains unqualified because compatible kernels and complete transport/cost evidence are missing, as documented in "acceptance_gates" and "board_operation_rows".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8414_v724_gatemate_obligation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The GateMate obligation is recorded, but historical authentication and hardware execution remain blocked by missing source bytes and physical/device evidence.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8415_v724_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible H1 summary reports no cost improvement for spline34 over RBF34, with “mean_gain” = −0.00390625 across 128 intended units.

## WHAT IS MISSING
“paired_cost_rows” is truncated in its 100th row at “gain_l”; remaining fields, including “gain_lower”, “gain_upper”, “spline_cost”, and “qualified”, and rows 101–128 are missing.

## THE CHECK A READER CANNOT DO
Do the complete 128 paired cost rows reproduce the reported “mean_gain” of −0.00390625?
