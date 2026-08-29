# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 2 |
| CANNOT_DETERMINE | 1 |

## experiment_1736_ebt_gradient_refinement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required `prior_failures` entry was missing or incomplete.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6275_flagship_asp_constraint_verification_benchmark.formal_sidecar.json

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
The experiment reported simulated Vivado success, but was flagged as adversarial because it was not a live hardware measurement.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6275_flagship_asp_constraint_verification_benchmark.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Energy-guided repair improved format and/or semantic validity over one-shot in several model-family cells, according to positive repair margins and paired mean deltas.

## WHAT IS MISSING
Per-unit rows containing each model, task, seed, arm, parse-success result, and semantic-validity result; only aggregates such as `"format_repair_margin_by_model_family"` and `"paired_intervals_and_sample_sizes"` are present, while `"flagship_asp_event_corpus_path_and_hash"` merely references an external corpus.

## THE CHECK A READER CANNOT DO
Were the positive mean deltas broad across paired units, or caused by one outlier while the remaining units were unchanged or lacked headroom?

## experiment_6751_thermalizer_factor_trajectory_fidelity.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Context matching reduced exact mean trajectory total variation relative to the independent-factor arm, satisfying the positive-result gate.

## WHAT IS MISSING
The complete `"rows"` array: `"frozen_config.expected_row_count"` says 192, but the artifact truncates mid-row, so all 192 per-unit `"trajectory_tv"` values cannot be found; only aggregate values appear in `"positive_result_gate"`.

## THE CHECK A READER CANNOT DO
Do the 192 per-unit trajectory-TV rows actually reproduce the reported arm means and show that the reduction is broad rather than driven by a few units?

## experiment_6752_arc_code_carrying_tool_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Both specified models passed the 32K CUDA admission and code-carrying tool transport preflight, without making a solve claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6753_object_table_fetch_on_demand_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the `cuda_device_available` preflight check failed, so no comparative result was produced.

## WHAT IS MISSING
nothing; `"status": "blocked"`, `"stop_reason": "preflight_blocked"`, `"failure_class": "preflight_blocked:cuda_device_available"`, `"live_model_invoked": false`, and empty `"gpu_receipts"` identify the blocker.

## THE CHECK A READER CANNOT DO
none

## experiment_6754_v588_branch_disposition.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Context matching reduced mean trajectory total variation versus the independent-factor arm, supporting positive simulator-only stochastic-portability evidence.

## WHAT IS MISSING
Per-unit trajectory-TV values for each seed, cell, or condition in each arm; `"trajectory_tv_by_arm"` provides only aggregate `"value"` and `"denominator"` fields, while `"rows"` is described as task-level and branch-level rows rather than experimental-unit rows.

## THE CHECK A READER CANNOT DO
Was the lower context-matched mean a broad effect across the 64 units, or was it driven by a few outliers or floor/ceiling-pinned units?
