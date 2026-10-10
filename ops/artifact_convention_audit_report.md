# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| CANNOT_DETERMINE | 5 |

## experiment_8352_v720_spline_table_fidelity.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible “configurations” report that ten of twelve table configurations passed candidate criteria, but the headline verdict is not shown.

## WHAT IS MISSING
The artifact ends mid-“historical_model_provenance”; actual “honest_verdict” and “verdict_class” values are absent, appearing only as names in “field_principles”.

## THE CHECK A READER CANNOT DO
Does the final verdict report a blocker whose cause is recorded?

## experiment_8353_v720_runtime_reader_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt records failed acceptance gates, but the overall headline claim is not shown.

## WHAT IS MISSING
The overall verdict and remaining artifact text: the JSON cuts off inside "execution_authority.contract_rows". "acceptance_gates" identifies three false checks, so blocker diagnostics are not wholly absent.

## THE CHECK A READER CANNOT DO
Does the final verdict make a comparative claim whose supporting per-unit results are recorded?

## experiment_8354_changed_runtime_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The bounded Qwen evidence canary was blocked because all three upstream qualification gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8355_v720_arc_supervisor_frontier.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports reader readiness and zero outcome support, but the headline verdict is not shown.

## WHAT IS MISSING
The artifact truncates inside "field_principles.per_game_arm_rows"; actual values for "honest_verdict", "gate_check_summary", and "per_game_arm_rows" are not visible.

## THE CHECK A READER CANNOT DO
Does the final verdict report comparative success, a diagnosed blocker, or no outcome claim?

## experiment_8356_v720_kv260_workload_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Accelerator benefit is unproved because no compatible operation exists, while arithmetic-cost and table-cost acceptance gates are marked passed.

## WHAT IS MISSING
Per-unit cost measurements supporting "acceptance_gates.arithmetic_cost" and "acceptance_gates.table_cost" are not visible; "arithmetic_rows" contains bookkeeping counts, but the artifact cuts off mid-row, so additional evidence may be omitted.

## THE CHECK A READER CANNOT DO
What were the dense and active costs for each paired "source_id" underlying the passed arithmetic-cost gate?

## experiment_8357_v720_gatemate_change_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
GateMate preflight remains unexecuted, with "board_rows" recording "blocker": "0xffffffff" and "missing_status": "physical_receipt_absent".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8358_v720_terminal_replay_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Terminal replay qualification is blocked; "gate_check_summary" records source-closure hash mismatches and a repository-suite exit code of -9 instead of 0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8359_v720_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Spline34’s reported mean cost gain over RBF34 is −0.00390625 across 128 intended units.

## WHAT IS MISSING
The remainder of "paired_cost_rows": only 100 complete rows are supplied, versus "intended_count": 128, before the artifact cuts off mid-row; "qualification_reason" does record an optimizer diagnostic.

## THE CHECK A READER CANNOT DO
Do the paired costs across all 128 units reproduce the reported "mean_gain" of −0.00390625?
