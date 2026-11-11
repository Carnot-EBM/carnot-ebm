# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| CANNOT_DETERMINE | 4 |

## experiment_8986_v768_spline_basis_methods.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Fixed-basis numerical mechanics and owned checks pass, with no claim of new scientific benefit.

## WHAT IS MISSING
The supplied artifact truncates inside the second `"head_rows"` entry’s `"gradient"`, leaving the remaining arm/seed convergence evidence unavailable; `"acceptance_gates"` reports `"numerical_mechanics": true.

## THE CHECK A READER CANNOT DO
Does every arm/seed fit satisfy `"convergence_rule"`?

Only the supplied text was inspected; truncation prevents a complete audit.

## experiment_8987_v768_spline_development_admission.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims exposed development feasibility, without independent generalization or an oracle-distinct verifier win.

## WHAT IS MISSING
The artifact cuts off inside "construction.graph_manifest". "acceptance_gates.development_signal" is true, but supporting per-unit development metrics and a final verdict are not visible; "basis_fixture_rows" and "clipping_rows" report mechanics and clipping.

## THE CHECK A READER CANNOT DO
Do individual development results support the asserted "development_signal" gate?

The unsupplied remainder was not checked; it may contain the needed evidence.

## experiment_8988_v768_online_spline_methods.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible claim concerns delayed fixed-basis coefficient mechanics only, with no scientific benefit or oracle-distinct verifier win claimed.

## WHAT IS MISSING
The artifact truncates inside `"fixture_request.guard_design.linear"`; `"honest_verdict"`, `"gate_check_summary"`, and `"rows"` are named in `"field_principles"`, but their actual contents are unavailable.

## THE CHECK A READER CANNOT DO
Does the unavailable remainder contain a comparative verdict with per-unit metrics supporting it?

## experiment_8989_v768_natural_annotation_authority.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Independent-label readiness is gated: "acceptance_gates" records "independent_label_ready_score" with "observed": 0, "expected": 1, and "passed": false.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

No rerun or upstream files checked; assessment covers only the supplied text.

## experiment_8990_v768_graph_corpus_a.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims custody of 512 completed first responses, with no detector-benefit claim.

## WHAT IS MISSING
The artifact cuts off inside `"first_response_rows"`, which does contain per-unit metrics. Actual `"status"`, `"honest_verdict"`, `"verdict_class"`, and `"gate_check_summary"` values are not visible; their names appear in `"field_principles"`. `"acceptance_gates.owned_checks.passed"` is false, but the final disposition and diagnostic are unseen.

## THE CHECK A READER CANNOT DO
Does the final verdict declare a blocker and identify the failed check and its observed result?

Unchecked: truncated remainder; it may contain the necessary diagnostic.

## experiment_8992_corpus_admission.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8992 was blocked because corpus A’s `shard_ready_score` was 0 instead of 1 and corpus B’s upstream artifact was missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8997_arc_transfer_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The assessment was blocked because `arc_new_outcomes_ready_score` was 0 and the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8998_v768_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Administrative reconciliation is complete, while fresh science remains blocked; "gate_check_summary" identifies failed checks and their expected and observed values.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Referenced files were not inspected and experiments were not rerun; this verdict covers only the supplied text.
