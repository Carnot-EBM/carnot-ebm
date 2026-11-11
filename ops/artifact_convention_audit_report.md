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

## experiment_8971_v767_branch_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports administrative contract readiness while ARC support remains unavailable.

## WHAT IS MISSING
The artifact truncates inside "exclusion_scope_rows"; the final verdict and remaining evidence are unavailable. "honest_verdict", "gate_check_summary", and "rows" appear in upstream "fields_imported", but their current values are not shown.

## THE CHECK A READER CANNOT DO
Does the final verdict make a comparative claim supported by per-unit rows?

The unseen remainder could not be checked; the final claim and its support remain unknown.

## experiment_8972_v767_encoder_head_diagnosis.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The development-only encoder/head diagnosis is ready, but fresh scientific benefit is unestablished.

## WHAT IS MISSING
The supplied artifact truncates inside "diagnostic_prediction_rows", showing only part of one arm’s fit rows; "diagnostic_metric_rows" reports aggregates for four arms across five roles, whose complete per-unit evidence is unavailable here.

## THE CHECK A READER CANNOT DO
Do paired per-graph results support the reported differences across all four arms?

The unseen remainder was not checked; truncation prevents judging whether the complete artifact supplies that evidence.

## experiment_8973_v767_shared_feature_methods.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports failed “development_admission” and “scientific_benefit” gates, but the headline verdict is unavailable.

## WHAT IS MISSING
The artifact ends mid-“graph_hash”. Current “honest_verdict”, “gate_check_summary”, and per-unit metric rows are unavailable; their names appear only in “fields_imported”.

## THE CHECK A READER CANNOT DO
Does the omitted remainder diagnose the failed “development_admission” gate with a specific check and observed value?

## experiment_8974_feature_online_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8974 was blocked because upstream `shared_methods_ready_score` was 0, failing the requirement that it equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8975_v767_natural_label_eligibility.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The natural-evidence readiness gate failed: "acceptance_gates" records "natural_evidence_ready_score" as 0 against expected 1, and "dialogue_join_rows" documents missing authorization and source matches.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Not checked: omitted content or external files; this verdict covers only the supplied excerpt.

## experiment_8976_graph_corpus_a.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Collection was blocked because upstream `shared_methods_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8983_arc_transfer_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8983 was blocked because upstream `arc_new_outcomes_ready_score` was 0, failing the required `== 1` gate.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8984_v767_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
V767 completed accounting but remained blocked, and common features did not rescue the fixed energy head.

## WHAT IS MISSING
The artifact ends mid-`publication_gate.gates.G4`. Visible `admission_rows` and `acceptance_gate_rows` summarize metrics and bounds, without per-unit arm results; these might appear in the missing remainder. `gate_check_summary` explicitly records blocker diagnostics.

## THE CHECK A READER CANNOT DO
Do paired per-graph/per-seed results support the claim that common features failed to rescue the energy head?

Upstream files were not checked; truncation prevents establishing whether per-unit evidence is absent.
