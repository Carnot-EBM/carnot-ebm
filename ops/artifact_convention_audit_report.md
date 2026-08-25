# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| AGGREGATE_ONLY | 3 |
| CANNOT_DETERMINE | 1 |

## experiment_6590_qwen36_constraint_first_stream.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no identifiable headline claim because the artifact is truncated mid-record

## WHAT IS MISSING
The remainder of `"exact_checker_receipts"`, the closing structure, and any final verdict/status or aggregate claim fields; present fields include `"attack_rows"`, `"checkpoint_receipts"`, and per-unit `"exact_checker_receipts"`.

## THE CHECK A READER CANNOT DO
Did the completed artifact ultimately declare a comparative gate met or the task blocked, and—if blocked—record which check failed and its observed value?

## experiment_6591_gemma4_31b_constraint_first_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All recorded attacks passed because each `candidate_ready_score` matched its `expected_ready_score` at 0.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6592_v575_terminal_intake_and_method_lock.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All recorded adversarial attacks passed by failing closed with `candidate_acceptance_score` equal to `expected_acceptance_score` at 0.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6593_cfr_independent_row_reducer.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The CFR candidates showed no exact-success improvement over direct control, while reporting aggregate latency and token differences, and the reducer was declared ready.

## WHAT IS MISSING
The actual `"per_unit_rows"` containing each unit’s control and candidate outcomes, headroom, tokens, latency, and failures; only aggregate `"family_effect_rows"`, `"acceptance_gate_rows"`, `"constraint_quality_summary"`, and references to `"Exp6590.per_unit_rows"` and `"Exp6591.per_unit_rows"` in `"field_provenance"` are present.

## THE CHECK A READER CANNOT DO
Were the reported mean latency and token differences broad across individual units or driven by a few outliers?

## experiment_6594_cfr_counterfactual_authority_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The tested attacks passed by triggering their expected detectors, with exact-checker removal or substitution causing release to be blocked for each recorded unit.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6595_invariant_projection_world_model_canary.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The learned invariant projection met all acceptance gates by improving conservative held rollout error over no projection, separating from the norm-matched random control, remaining stable, and showing no false damped benefit.

## WHAT IS MISSING
The actual `"per_unit_rows"` containing each fixture/seed/arm’s rollout-error and convergence outcomes are missing; only `"acceptance_gate_rows"` and `"arm_summary_rows"` aggregates are present, despite `"source": "per_unit_rows"` and provenance assertions that such rows exist.

## THE CHECK A READER CANNOT DO
Did the conservative improvement occur broadly across the 30 paired units, or was the positive interval and pooled mean driven by a few outliers or units with unusual headroom?

## experiment_6596_convergeflow_feasible_token_canary.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The headline claim is that the comparative three-arm experiment is canary-ready, reported as `"convergeflow_canary_ready_score": 1.0`.

## WHAT IS MISSING
Per-unit outcome rows keyed by geometry, error, seed, and arm, including `endpoint_distortion`, `path_length`, convergence/failure status, and charged work; `"distortion_and_cost_summary"` contains only pooled means, totals, and counts, while `"perturbation_rows"` and `"start_rows"` record inputs rather than outcomes.

## THE CHECK A READER CANNOT DO
Did the apparent arm-level performance difference occur broadly across geometries and seeds, or was it driven by a few outliers or degenerate units?

## experiment_6597_spectral_k_block_ising_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The spectral block arms passed all acceptance gates, showing stationary-distribution noninferiority and positive ESS-per-transition effects across fixtures, block sizes, and matched seeds.

## WHAT IS MISSING
nothing; per-unit metrics are recorded in `"effective_sample_size_summary.rows"` and `"exact_distribution_comparison"`, keyed by `"row_id"`, `"seed"`, `"arm"`, `"fixture_id"`, and `"block_size"`.

## THE CHECK A READER CANNOT DO
none
