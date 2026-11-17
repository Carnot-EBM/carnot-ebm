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

## experiment_9181_v782_causal_validation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
On the recorded fixed input, the changed arm reduced full scans from 91 to 1 while preserving the reported flags.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

No rerun or external-file verification performed; assessment is limited to the supplied, truncated text.

## experiment_9182_forward_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The validation revision was blocked because upstream `causal_fix_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9192_v782_arc_mechanic_routing.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible artifact specifies routing-compatibility gates but does not show a final claim that they passed.

## WHAT IS MISSING
The artifact’s remainder: it cuts off mid-string inside "changed_behavior_rows". Per-condition "before", "after", and "reference" arrays are present, but the complete results and any final verdict are unavailable.

## THE CHECK A READER CANNOT DO
Did all 52 recognized rows match their references and all 72 legacy controls remain unchanged?

The unseen remainder was not checked; truncation prevents a definitive classification.

## experiment_9193_v782_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“G2” claims FoVer headline reproduction passed, including a positive learning contribution within the published interval.

## WHAT IS MISSING
The artifact’s continuation: it cuts off inside `"arc_summary.rows"` at `"arm": "explicit_mecha`. Visible rows record `"metric": "operator_list_equality"`; whether FoVer per-unit comparison metrics appear later is unknown.

## THE CHECK A READER CANNOT DO
With the truncated continuation unavailable and unchecked, do per-unit FoVer results support the positive learning contribution asserted in `"G2"`?

## experiment_9194_v783_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion describes administrative task binding and explicitly disclaims measured scientific benefit.

## WHAT IS MISSING
The excerpt cuts off inside "full_task_objects"; current "honest_verdict", "verdict_class" and top-level "gate_check_summary" values are not shown. Their mentions in "field_principles" are descriptions, not results.

## THE CHECK A READER CANNOT DO
Does "contract_ready_score": 0 represent a blocked outcome, and which check caused it?

Only the supplied excerpt was checked; truncation leaves the current outcome unknown.

## experiment_9195_corpus_prefilter.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp9194-contract.contract_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9197_label_ambiguity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9197 was blocked because upstream `exp9194-contract.source_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9207_v783_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Routing mismatches fell from 22 to zero, achieving operator-list equality on 124/124 units.

## WHAT IS MISSING
The complete `"arc_summary.rows"`: the supplied artifact cuts off mid-entry after 21 complete rows, while `"completed"`, `"denominator"`, and `"equality_numerator"` report 124.

## THE CHECK A READER CANNOT DO
Do all 124 per-unit comparisons support `"before_mismatches": 22` and `"after_mismatches": 0`?
