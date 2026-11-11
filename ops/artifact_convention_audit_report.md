# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_9013_v770_branch_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
“claim_scope” limits this artifact to administrative custody, with no scientific benefit or independent generalization claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9014_v770_scoped_provenance_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Scoped qualification fails acceptance; "current_receipt_mutation_rows" documents the valid control’s "observed_exit" as 1 versus "expected_exit" of 0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

No rerun or external checks performed; assessment covers only the supplied, truncated excerpt.

## experiment_9015_v770_human_disagreement_source.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Source qualification was blocked: "gate_check_summary" records "source_resolution_access" failing with HTTP 429 against expected 200, consistent with "honest_verdict".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Not checked: referenced files; no experiment rerun. Assessment is limited to the supplied text.

## experiment_9016_retained_shard_a.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Authentication of retained shard A was blocked because "provenance_ready_score" was 0 while the gate required 1, as recorded in "gates_evaluated" and "gate_check_summary".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9017_graph_corpus_b.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9017 was blocked because upstream experiment 9014’s `provenance_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9019_human_pilot_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9019 was blocked because upstream `natural_pilot_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9025_arc_transfer_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The assessment was blocked because `arc_new_outcomes_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9026_v770_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Fourteen administrative outcomes are accounted for, while graph, independent human annotation, causal learning, and deployment benefits remain unmeasured.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
