# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| AGGREGATE_ONLY | 1 |

## experiment_6683_ising_reference_scope_receipt.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims the validation suite passed and is ready, with expected and observed results agreeing across the recorded fixtures, rows, and attacks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6684_torx_typed_factor_parity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims readiness was not achieved because the `applicable_e2e` test returned `exit_code: 1` instead of the required `exit_code: 0`.

## WHAT IS MISSING
nothing; `"ready": false`, `"failed_check_count": 1`, and `"failures"` identify the failed check and record its expected and observed values.

## THE CHECK A READER CANNOT DO
none

## experiment_6685_autocorrelation_schedule_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the upstream `torx_factor_parity_ready` gate observed `false` instead of the required `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6687_v582_branch_synthesis.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
V582 achieved no pooled success, and the live ARC supervisor performed worse on utility, caused false interventions, and made no solve.

## WHAT IS MISSING
Per-ARC-unit A/B rows containing the game or condition ID, control and supervisor utility, intervention outcome, and headroom; the present `"per_unit_rows"` are task-level synthesis rows, while `"utility"`, `"false_intervention"`, and `"action"` contain only aggregates. Blocker diagnostics are present in `"gate_check_summary"`, `"cause"`, and validation `"finding"` fields.

## THE CHECK A READER CANNOT DO
Which individual ARC units produced the three utility losses and nine false interventions, and whether their controls had meaningful headroom rather than being pinned or degenerate?

## experiment_6690_exact_planning_fixture_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because the required upstream artifact `exp6689-exact-planning-fixture` was not found, so the `planning_fixture_ready` gate failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6692_structural_plan_energy.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because the upstream artifact `exp6691-sota-planning-proposal-corpus` was not found, so `proposal_corpus_ready` was observed as null instead of true.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6694_energy_backtracking_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required upstream artifact `exp6693-energy-backtracking-ab` was not found, so its `backtracking_ab_ready` field could not be verified as `true`.

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
