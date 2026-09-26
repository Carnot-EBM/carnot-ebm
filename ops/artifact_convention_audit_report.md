# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 2 |

## experiment_7659_v668_atom_corpus.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The atom corpus is ready, while probability benefit and utility remain untested and fresh confirmation failed.

## WHAT IS MISSING
nothing; `rows` records per-unit arm metrics, and `acceptance_gate_results` records the failed freshness check and its observed value.

## THE CHECK A READER CANNOT DO
none

## experiment_7660_v668_atom_energy.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports an honest null result with the energy head ready (`honest_verdict`: "complete_null_energy_head_ready") and makes no comparative superiority claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7661_v668_decision_evaluation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The completed evaluation found no registered decision benefit.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7662_v668_delayed_update_protocol.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible `acceptance_gate_results` say the source arm failed the probability-benefit and utility gates, and there were no fresh confirmatory groups.

## WHAT IS MISSING
The artifact cuts off at `event_rows[32].origin_ordinal`, so the remaining fields, including any final verdict or per-group Brier and decision-cost rows, are unavailable; the visible `acceptance_gate_results` give pooled arm values.

## THE CHECK A READER CANNOT DO
Do per-group results support the reported pooled gate results, or are those results driven by a few groups?

## experiment_7663_v668_continuous_atom_learning.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The candidate failed the probability-benefit and utility gates, so the artifact does not establish a benefit over the controls.

## WHAT IS MISSING
Per-group paired Brier and cost results for each arm. `acceptance_gate_results` gives `paired_block_ci95` summaries, while `admission_decisions` gives batch-level `candidate_admission_brier` and `prior_admission_brier`, not the individual results behind those gates. The failed gates do record their checks and measured values.

## THE CHECK A READER CANNOT DO
Were the reported differences spread across the groups, or driven by a few outliers?

## experiment_7664_v668_independent_evidence_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The audit found no independent probability or decision-cost benefit.

## WHAT IS MISSING
The supplied JSON cuts off inside `rows`. The visible `rows[].raw_metrics` contain `checked_propositions` and `lexical_membership`, but no per-`unit_id`, per-`arm` `brier` or `decision_cost` values; the missing remainder may contain them.

## THE CHECK A READER CANNOT DO
Were the reported Brier differences spread across groups or driven by a few outliers?

## experiment_7665_v668_qwen_grounded_claims.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The 48 calls established bounded mechanism feasibility, while the artifact claims no whole-answer benefit.

## WHAT IS MISSING
nothing; `rows` records per-unit, per-arm metrics, and `acceptance_gate_results.freshness.measured_operands` records the failed gate’s value.

## THE CHECK A READER CANNOT DO
none

## experiment_7666_v668_arc_goal_confirmation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The goal-confirmation check passed its scripted-fixture readiness gate, while hidden-game benefit remains unestablished.

## WHAT IS MISSING
nothing; `rows` and `goal_rows` record per-fixture outcomes, and `acceptance_gate_results` records each gate’s result and operands.

## THE CHECK A READER CANNOT DO
none
