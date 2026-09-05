# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 2 |

## experiment_6995_v612_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The roadmap gate audit passed, with blocked tasks carrying explicit diagnostics and no model-quality or solve claim being made.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6996_v613_source_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V613 source delta is complete and its 13-task contract conforms.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6997_authority_sidecar_rebuild.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The recorded integrity checks passed and the blinded learner view is ready.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6998_three_family_commitment_controls.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion claims pooled mean effects of −0.02778 for `true_provenance_hint_minus_clean` and +0.01736 for `permuted_decoy_hint_minus_clean`.

## WHAT IS MISSING
The artifact is truncated mid-`choice_flip_rows`, so its verdict and any later per-unit rows underlying `bootstrap_interval_rows.mean` are unavailable; the visible `choice_flip_rows` records only `choice_flip_count` and does not link that metric to the reported means.

## THE CHECK A READER CANNOT DO
A reader cannot recompute the two pooled means from all 36 unit-level observations or determine whether omitted rows make the comparison checkable.

## experiment_6999_blinded_feature_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact records a `blinded_feature_bank_ready_score` of 0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7000_certified_blinded_pwa_kan.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp6999-blinded-feature-cold-audit.blinded_feature_bank_ready_score` was 0 but was required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7005_arc_live_envelope_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
`bootstrap_interval_rows` claims the engine improved mean paired error versus two controls.

## WHAT IS MISSING
The artifact is truncated inside `engine_score_rows`; the complete per-transition control scores or paired-error rows, plus any final verdict or `gate_check_summary`, cannot be found.

## THE CHECK A READER CANNOT DO
Were the reported mean improvements broad across transitions, or driven by one outlier?

## experiment_7008_v613_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The V613 capstone is a complete null with no oracle-distinct science-positive result, including the claim that the ARC engine did not beat both controls.

## WHAT IS MISSING
Per-transition or per-unit metric rows underlying the aggregate `"arc_quality_rows"` and `"commitment_control_rows"`; only summary fields such as `"exact_next_frame_accuracy"`, `"calibrated_frame_error"`, `"mean"`, and `"sample_count"` are present.

## THE CHECK A READER CANNOT DO
Were the comparative null results broad across held-out transitions and samples, or caused by a few outliers, degenerate controls, or units with no headroom?
