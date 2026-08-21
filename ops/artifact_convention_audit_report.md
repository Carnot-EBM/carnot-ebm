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

## experiment_3377_archive_v310_activate_v311.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The archive of milestone 2026.05.310 and activation of milestone 2026.05.311 completed successfully and is ready.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_3392_archive_v311_activate_v312.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The archive is complete and milestone `2026.05.312` is ready after archiving `2026.05.311`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_833_constraint_delta_root_cause.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The write path is missing: verification performed 10 store retrievals but zero store writes.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6479_verify_repair_factor_cache_shadow_adapter.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The default-off verify-repair factor-cache shadow adapter shipped successfully and met all readiness gates without changing public outputs or exact-verifier authority.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1644_cerce_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The CERCE ledger was added and is ready, with no policy certificates, violations, FR11 events, or policy updates evaluated or recorded.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6480_v557_terminal_evidence_and_v558_preflight.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
All seven V557 tasks completed and the exact-energy, ARC-shield, and factor-cache readiness gates were met.

## WHAT IS MISSING
The artifact is truncated mid-field; actual `"per_unit_rows"`, `"honest_verdict"`, and `"gate_check_summary"` values cannot be found—only their names in `"field_principles"` and `"field_provenance"` are visible.

## THE CHECK A READER CANNOT DO
Do the 24 exact-energy units individually support `"harmful_flip_count_vs_first": 0`, or is that merely an unrecoverable aggregate assertion?

## experiment_6481_monotonic_phase_concurrency_receipt_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims the phase-concurrency receipt gate achieved a row-derived readiness score of 1.0, with every required check passing and all nine attacks failing closed.

## WHAT IS MISSING
The artifact is truncated mid-`field_provenance.status.source_paths`; although `attack_matrix.rows`, `concurrency_decision_rows`, and `dependency_hash_rows` are present, the actual `phase_rows`, `process_identity_rows`, `resource_ownership_rows`, and `output_rows` referenced by `aggregate_row_recomputation.row_type_counts` cannot be found in the supplied text.

## THE CHECK A READER CANNOT DO
Do all 59 underlying rows actually satisfy every conjunct needed to recompute `phase_concurrency_receipt_ready_score_from_rows` as 1.0?

## experiment_6482_immutable_prospective_constraint_stream_commitment.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims full backend parity across 48 units, all nine attacks failing closed, and a prospective contract-readiness score of 1.0.

## WHAT IS MISSING
The artifact is truncated inside `"backend_parity_rows"` at `"unit_id": "exp6482-qu`, so the complete per-unit rows underlying `"backend_parity_mismatch_count": 0` and `"prospective_contract_ready_score_from_rows": 1.0` cannot be found.

## THE CHECK A READER CANNOT DO
Do all 48 units actually have matching Z3 and exhaustive-backend results with no mismatches?
