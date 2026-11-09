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

## experiment_8933_v764_compatible_replay_methods.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact cuts off inside "acceptance_gates.frozen_evaluation_manifest". "evaluation_plan" records planned comparisons, and "evaluation_opened" is false; the remainder containing any verdict, observed results, or blocker diagnostic is unavailable.

## THE CHECK A READER CANNOT DO
Does the omitted remainder make a comparative or blocked claim and provide supporting per-unit results or a blocker reason?

Unseen content was not checked; truncation could conceal either failure.

## experiment_8934_v764_depth_control.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The paired depth control reports a null result: errors of 0.84 at k=2 and 0.78 at k=3, with a paired 95% interval spanning zero.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Execution and external artifacts were not checked; this assessment covers the supplied record only.

## experiment_8935_graph_corpus_a.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Collection was blocked because `depth_source_ready_score` was 0, while the gate required 1, as recorded in `gate_check_summary` and `gates_evaluated`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8937_v764_calibration_methods.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fixture diagnostics report that the benefit gate failed because energy performed worse than logistic on Brier score.

## WHAT IS MISSING
The artifact ends mid-row in "development_evidence.rows"; complete paired rows for the comparison arms are unavailable. "fixture_diagnostics.contrasts.brier_vs_logistic" records the failure diagnostic: estimate −0.0573125, bound −0.108932.

## THE CHECK A READER CANNOT DO
Do paired per-unit results support the reported Brier difference? The truncated remainder cannot be checked and may contain the required rows.

## experiment_8940_v764_compatible_replay_streams.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
No headline claim is visible in the supplied excerpt.

## WHAT IS MISSING
The artifact’s ending and headline conclusion/verdict: it cuts off mid-row in "feedback_rows". "acceptance_gates" records "benefit_required": false, and "attempt_rows" records completion, but neither establishes the final claim.

## THE CHECK A READER CANNOT DO
Does the missing ending assert a comparative result, report a blocker, or simply record completed work?

Only the supplied excerpt was checked; its truncation prevents reliable classification.

## experiment_8941_v764_compatible_replay_benefit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The fixed compatible-replay configuration failed the required online benefit gates despite passing all retention bounds.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Not checked: experiment execution or truncated content; this verdict covers the supplied text.

## experiment_8942_v764_replay_complete_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The finite-workload cost comparison reports a null result, and deployment is withheld because the quality gate observed 0 against an expected 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8943_v764_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion asserts complete fourteen-slot accounting and owned checks, with no scientific benefit, but the overall verdict is not shown.

## WHAT IS MISSING
The complete "branch_conclusions" and overall verdict: the supplied artifact stops mid-entry inside "frozen_evaluation_manifest", which records planned units rather than measured outcomes.

## THE CHECK A READER CANNOT DO
Does the completed artifact make a comparative or blocked claim with supporting measurements or diagnostics?

The omitted remainder cannot be checked; assuming what it contains would require trust.
