# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 1 |

## experiment_6960_certified_selection_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
convex_factor_energy did not beat syntax_validity: paired mean delta was -0.037037037037037035 with CI95 [-0.09259259259259259, 0.0] and `strictly_above_zero` false.

## WHAT IS MISSING
nothing; the artifact includes aggregate fields in `aggregate_consistency_rows` and per-unit/per-arm rows in `arm_recompute_rows` with `arm`, `pair_id`, `group_id`, `attempt_key`, `exact_mapping_correct`, `score`, and `upstream_score`.

## THE CHECK A READER CANNOT DO
none
