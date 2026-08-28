# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 1 |

## experiment_6685_autocorrelation_schedule_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the `torx_factor_parity_ready` gate observed `false` when `true` was required.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6687_v582_branch_synthesis.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The supervisor reduced live utility, caused false interventions, and produced no solve, while the other branches were blocked or null.

## WHAT IS MISSING
Per-game or per-condition live A/B rows containing each unit’s supervisor-on and supervisor-off utility, utility delta, intervention result, and headroom; `"live_arc_outcome_branch.utility"` records only aggregate `"delta"`, `"denominator"`, `"losses"`, `"ties"`, and `"wins"`, while `"per_unit_rows"` contains task-level synthesis rows rather than live outcome units.

## THE CHECK A READER CANNOT DO
Which nine live units were losses or ties, and what paired on/off measurements produced the reported utility delta of -0.3333?

## experiment_6690_exact_planning_fixture_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required upstream artifact `exp6689-exact-planning-fixture` was not found, so the `planning_fixture_ready` gate failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6692_structural_plan_energy.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because the upstream artifact `exp6691-sota-planning-proposal-corpus` was not found, so the `proposal_corpus_ready` gate failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6694_energy_backtracking_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the upstream artifact `exp6693-energy-backtracking-ab` was not found, so its `backtracking_ab_ready` field could not be verified as `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6696_prequential_online_energy_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required upstream artifact `exp6695-online-energy-update-fixture` was not found.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6702_exact_planning_fixture_recovery.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The exact finite-horizon planning fixture passed all readiness checks and is ready.

## WHAT IS MISSING
The artifact is truncated mid-`instance_rows`, so the top-level `"per_unit_rows"` referenced by `"field_provenance.per_unit_rows"` and any remaining gate-driving row fields cannot be confirmed; `"aggregate_row_recomputation.checks"`, `"exact_solver_rows"`, `"instance_rows"`, and `"gate_check_summary"` are present.

## THE CHECK A READER CANNOT DO
Do the unseen per-unit results substantiate every `true` readiness check, or are some checks supported only by aggregate booleans?

## experiment_6705_structural_plan_energy.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required upstream artifact `exp6704-sota-planning-proposal-bank` was missing, so `proposal_bank_ready` could not be observed as `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
