# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 2 |

## experiment_6678_constraint_family_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The constraint-family stream is not ready because the aggregate recomputation’s `"tests"` check is `false`.

## WHAT IS MISSING
nothing; `"constraint_family_stream_ready": false`, `"aggregate_row_recomputation.ready": false`, and `"aggregate_row_recomputation.checks.tests": false` identify the failed check and its value.

## THE CHECK A READER CANNOT DO
none

## experiment_6679_prequential_cross_family_csl_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp6678-constraint-family-stream.constraint_family_stream_ready` was `false`, while the gate expected a value in `[true]`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6681_arc_post_redirect_outcomes.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims the gate passed because all 30 applied held-family redirects had exactly one joined live next-outcome row.

## WHAT IS MISSING
The actual per-redirect `"redirect_outcome_rows"` or `"per_unit_rows"` are missing; only their `"field_provenance"` entries and aggregate fields such as `"eligible_redirect_outcome_rows"`, `"aggregate_row_recomputation"`, and `"gate_check_summary"` are present, while the shown `"non_redirect_control_rows"` do not establish outcomes for the redirects.

## THE CHECK A READER CANNOT DO
Did each of the 30 distinct redirects actually have one unique, fully joined live outcome, rather than the aggregate count hiding missing or duplicated redirect outcomes?

## experiment_6682_arc_held_family_supervisor_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The held-family supervisor A/B is blocked by a verification failure, and no transition benefit or solve is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6683_ising_reference_scope_receipt.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The observed marginals, correlations, decompositions, and attack outcomes match their expected values across the recorded fixtures, making `"ready": true`.

## WHAT IS MISSING
nothing; per-unit evidence is recorded in `"_correlation_rows"`, `"_marginal_rows"`, `"attack_rows"`, and `"decomposition_rows"` with expected and observed values, errors, and `"passed"` results.

## THE CHECK A READER CANNOT DO
none

## experiment_6684_torx_typed_factor_parity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact is not ready because the `applicable_e2e` test returned exit code 1 instead of the expected 0.

## WHAT IS MISSING
nothing; `"ready": false`, `"failed_check_count": 1`, and `"failures"` record the failed check, expected value, and observed value.

## THE CHECK A READER CANNOT DO
none

## experiment_6685_autocorrelation_schedule_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the `torx_factor_parity_ready` gate observed `false` instead of the required `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6687_v582_branch_synthesis.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
V582 achieved no pooled success: three branches were blocked, while the live ARC supervisor lost utility, caused false interventions, and solved nothing.

## WHAT IS MISSING
Actual per-game or per-condition ARC rows containing arm-level utility and intervention outcomes; `"per_unit_rows"` is present but contains task-level summaries, while `"utility"` and `"false_intervention"` provide only aggregate counts and deltas. `"gate_check_summary"` does record diagnostics for the blocked branches.

## THE CHECK A READER CANNOT DO
Which three of the nine ARC units lost utility under the supervisor, and what were each unit’s paired on/off outcomes?
