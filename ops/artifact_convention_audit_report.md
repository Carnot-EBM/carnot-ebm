# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| CANNOT_DETERMINE | 3 |

## experiment_9028_v771_historical_provenance_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Current scoped provenance custody qualifies while historical failed verdicts remain failed; no repository health or verifier benefit is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Not checked: referenced files or experiment execution.

## experiment_9029_v771_source_identity_resolution.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports that source-identity readiness failed, scoring 0 against an expected 1.

## WHAT IS MISSING
The artifact cuts off inside "exposure_inventory.exposed_keys", leaving any later claims and supporting rows unavailable; "acceptance_gates" already records the failed check and its observed value.

## THE CHECK A READER CANNOT DO
Does the omitted remainder make a comparative claim without per-unit evidence?

## experiment_9030_v771_retained_shard_a.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims historical source custody only, with no detector benefit or independent generalization.

## WHAT IS MISSING
The supplied artifact ends mid-`first_response_rows`; the actual top-level `honest_verdict` and `gate_check_summary` are unavailable—their names appear only in `field_principles`.

## THE CHECK A READER CANNOT DO
Does experiment 9030’s truncated remainder record a blocked verdict with the failed check and observed value?

## experiment_9031_v771_graph_corpus_b.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
B first-response collection is blocked by external inputs, with 0 completed and 512 censored responses; no comparative performance claim is made.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

External files were not checked; assessment is limited to the supplied excerpt, which ends mid-row.

## experiment_9032_corpus_admission.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The join was blocked because both upstream checks recorded "shard_ready_score" = 0 against a required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9033_human_label_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `source_identity_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9039_arc_transfer_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `arc_new_outcomes_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9040_v771_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Fourteen administrative outcomes were completed, while scientific benefit claims remain unmeasured.

## WHAT IS MISSING
The supplied text truncates inside “full_contract_rows” at “raw_nume”: only ten rows are complete, while “completed_count” is 14. The final verdict is unavailable. “continuation_evidence” does record failed checks and observed values.

## THE CHECK A READER CANNOT DO
Do complete per-outcome records support all 14 reported completions?

Unseen records were not checked; truncation limits this audit.
