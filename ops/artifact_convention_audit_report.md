# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 11 |
| AGGREGATE_ONLY | 1 |

## experiment_6458_arc_representation_objective_generalization_ab.checkpoints.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing; the artifact includes per-cell rows under `"cells"` with fields such as `"row_id"`, `"arm"`, `"game"`, `"seed"`, `"prefix_id"`, `"chosen_action"`, `"recorded_next_state_reachability"`, `"policy_influence"`, and `"timeout"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6458_arc_representation_objective_generalization_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit completed but readiness gates were unmet because `frozen_safety_roster_not_regressed` failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6459_v555_adversarial_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
the artifact is complete-blocked because the V555 queue/prior-failure validation gate failed.

## WHAT IS MISSING
nothing; `"status"`, `"artifact_state"`, `"honest_verdict"`, `"readiness_fields"`, and `"gate_check_summary"` are present, including `"failed_check"`, `"expected_condition"`, and `"observed_value"`.

## THE CHECK A READER CANNOT DO
none

## experiment_1644_cerce_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims the CERCE ledger was added/implemented and the task completed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1767_e2e_qwen.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The model achieved latency_ms 150.5, parse_rate 0.95, and energy_score 0.88 over 100 prompts.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1736_kanele_synth.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 1736 reports success: `bitfile_generated: true`, utilization, `wns: 1.25`, and `honest_verdict: "vivado_simulated_success"`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_2031.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims a successful run with best_candidate "Thus, we can see it." and min_energy 0.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6021_inducer_head_to_head_qwen27b_vs_gemma31b.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Gemma31B beat Qwen27B on the decisive coverage criterion, with 12/13 nonzero games versus 4/13 and coverage p=0.007812.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6011_world_model_change_gate_four_arm.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims a four-arm world-model verification matrix was scored across per-seed rows, with gate-enabled arms failing for `degenerate_engine_no_correct_changed_cells` and gate-disabled arms passing because `gate_disabled`.

## WHAT IS MISSING
nothing; per-unit `rows` are present with `game`, `seed`, `cell`, `arms`, `passed`, `reason`, `change_fidelity`, `correct_changed_cells`, `min_correct_changed_cells`, `noop_hallucination_rate`, and related diagnostic values.

## THE CHECK A READER CANNOT DO
none

## experiment_6012_hidden_state_trust_gate_hole.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The hidden-state trust gate admits spurious writer arms that REQ-6011 rejects.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6013_hidden_state_change_gate_closure.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims the change gate closes the hidden-state coverage hole by passing clean/base arms and rejecting degenerate or hallucinating control arms across game/seed rows.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_10003_goal_defect_dedup_ab.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Control found nondegenerate plans more often than treatment: 7 of 12 vs 4 of 11, directionally adverse.

## WHAT IS MISSING
The artifact has `"per_cell"` rows for 11 paired cells, but the headline denominator uses 12 control cells; the missing per-unit row is the control-only `"ar25 t2"` condition with fields like `"plan_found"`, `"plan_len"`, and `"plan_is_degenerate"`.

## THE CHECK A READER CANNOT DO
Can the reported 7/12 control nondegenerate-plan count be reproduced from the listed per-cell rows, including the unpaired control-only ar25 trial 2 cell?
