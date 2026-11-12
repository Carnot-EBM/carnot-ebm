# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| CANNOT_DETERMINE | 1 |

## experiment_9054_v773_branch_contract.json

**CANNOT_DETERMINE**

## VERDICT

CANNOT_DETERMINE

## WHAT THE CLAIM IS

The visible excerpt records passed acceptance gates and no new ARC outcomes, but the headline verdict is not visible.

## WHAT IS MISSING

The artifact’s ending: it cuts off inside "authority_manifest", before any final verdict or "gate_check_summary" is visible. "arc_frontier.ready": false alone does not establish an unexplained blocker.

## THE CHECK A READER CANNOT DO

Does the omitted portion report a blocked verdict and record its reason?

Only the supplied excerpt was checked; the missing ending could change the classification.

## experiment_9055_v773_prospective_roster.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The prospective roster is complete and custody-ready; no comparative performance or detector benefit is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

External files and hashes were not checked; this assessment covers only the supplied excerpt.

## experiment_9056_v773_human_release_authority.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Annotation authority is blocked because attributable individual judgments and original annotator IDs are missing, as recorded in "missingness" and "original_annotator_ids".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Source files were not checked; this verdict covers the supplied excerpt.

## experiment_9057_human_probe_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because upstream `human_source_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9058_v773_fresh_corpus_a.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact records first-response custody only; it claims no detector benefit or independent generalization.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9060_sealed_corpus_admission.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Admission of the full graph corpus was blocked because upstream artifact `exp9059-fresh-corpus-b` was missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9066_arc_transfer_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9066 was blocked because upstream "arc_new_outcomes_ready_score" was 0 and the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9067_v773_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone reports fourteen administrative outcomes, with scientific bounds unmeasured and no new scientific or deployment benefit claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Referenced files and the truncated remainder were not checked; omitted claims remain unaudited.
