# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 4 |

## experiment_7184_v633_revocable_template_csl.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be determined because the artifact is truncated mid-`cost_rows` entry.

## WHAT IS MISSING
The remainder of the artifact, including any verdict, headline claim, gate result, diagnostic, and outcome-metric rows; only `"arms"` and partial `"cost_rows"` are present.

## THE CHECK A READER CANNOT DO
Does the unseen verdict make a comparative or blocked claim, and—if so—are the necessary per-unit outcome rows or failure diagnostics recorded?

## experiment_7185_v633_memory_cold_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim identifiable because the artifact is truncated mid-row

## WHAT IS MISSING
The complete artifact, including any headline/verdict or gate-status field; the fragment contains `"actual_controller_addition_count"`, `"addition_audit_rows"`, `"cold_reload_rows"`, and an incomplete `"credit_control_rows"` array.

## THE CHECK A READER CANNOT DO
Does the experiment ultimately make a comparative or blocked claim, and does it record the per-unit evidence or failed check supporting that verdict?

## experiment_7186_v633_arc_withheld_transfer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked during preflight because the required source `python/carnot/agentic/arc_eval_runner.py` was empty or missing.

## WHAT IS MISSING
nothing; `gate_check_summary` identifies `failed_check` as `required_source_bytes`, names the affected `field` as `REQUIRED_SOURCE_PATHS`, and records the file’s `observed_value` as `0` versus expected `"nonempty"`.

## THE CHECK A READER CANNOT DO
none

## experiment_7187_v633_slice_sampler.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7188_v633_quantized_transition_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The `"honest_verdict"` claims naive quantized energy changed the target in 162/162 precision-conditioned exact laws, while the delayed-acceptance kernel preserved the full target.

## WHAT IS MISSING
The artifact is truncated inside `"law_comparison_rows"` at `"first_moment_bias_m"`, so the complete set of 162 per-condition rows is missing; the visible rows do include `"condition_id"`, `"arm"`, `"precision_bits"`, `"exact_target_tv_from_full"`, and `"full_target_detailed_balance_error_max"`.

## THE CHECK A READER CANNOT DO
Do all 162 claimed conditions individually show nonzero target error for naive quantization and zero target error for delayed acceptance?

## experiment_7189_v633_rust_slice_parity.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion suggests that compiled Rust execution matches Python on cross-language replay checks, but the artifact is truncated before any definitive headline verdict.

## WHAT IS MISSING
The remainder of the artifact, including the completion of `"cross_language_rows"` and any final summary or verdict fields; visible rows do include `"unit_id"`, `"passed"`, `"python_delta_energy"`, and `"rust_delta_energy"`.

## THE CHECK A READER CANNOT DO
Did every replay unit pass, or do omitted rows contain mismatches that invalidate the apparent cross-language agreement?

## experiment_7190_v633_board_placement_receipt.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims complete visibility across the three boards and per-instance host compatibility with degree limits, while explicitly leaving physical topology fit unknown and claiming no new hardware performance.

## WHAT IS MISSING
nothing; `"board_rows"`, `"placement_rows"`, `"rows"`, `"gate_check_summary"`, `"disposition"`, `"last_observed_value"`, and `"exact_next_prerequisite"` provide per-unit evidence and blocker diagnostics.

## THE CHECK A READER CANNOT DO
none

## experiment_7191_v633_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The V633 capstone is complete and several tested methods failed to beat their comparison baselines or value gates.

## WHAT IS MISSING
The per-unit metric rows supporting the comparative `honest_verdict` claims are missing; only `producer_row_count`, `benefit_established`, `row_consistency`, and external `selected_evidence_path` references are present.

## THE CHECK A READER CANNOT DO
Were the reported null comparisons broad across paired blocks or driven by outliers, degenerate controls, or units pinned at metric floors and ceilings?
