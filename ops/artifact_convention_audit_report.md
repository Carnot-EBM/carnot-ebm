# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 3 |

## experiment_6480_v557_terminal_evidence_and_v558_preflight.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
All seven V557 tasks completed, the ARC-shield and factor-cache readiness gates scored 1.0, and exact-energy selection had zero harmful flips across 24 unit-seed cases.

## WHAT IS MISSING
Actual per-unit metric rows for the 24 unit-seed cases and gate inputs; only aggregates such as `"finite_no_llm_unit_seed_count"`, `"harmful_flip_count_vs_first"`, and readiness `"score"` values appear, while `"field_principles.per_unit_rows"` and `"field_provenance.per_unit_rows"` merely describe/provenance rows without providing them.

## THE CHECK A READER CANNOT DO
Were the zero harmful flips and passed gates supported uniformly across all units, or concealed by degenerate, pinned, or offsetting unit-level results?

## experiment_6481_monotonic_phase_concurrency_receipt_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion claims the phase-concurrency receipt readiness gate scored 1.0, with all checks passing and all nine attacks failing closed.

## WHAT IS MISSING
The artifact is truncated before the actual `status`, `honest_verdict`, and `gate_check_summary` values and before the claimed `phase_rows`, `process_identity_rows`, `resource_ownership_rows`, and `output_rows`; only metadata such as `aggregate_row_recomputation.row_count`, `row_type_counts`, and `field_provenance` is visible for them.

## THE CHECK A READER CANNOT DO
Can the reported `phase_concurrency_receipt_ready_score_from_rows` of 1.0 actually be recomputed from all 59 claimed rows?

## experiment_6482_immutable_prospective_constraint_stream_commitment.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims a perfect prospective-contract readiness score of 1.0, including complete backend parity with zero mismatches across 48 units.

## WHAT IS MISSING
The artifact is truncated mid-entry in `"backend_parity_rows"` and does not provide all 96 claimed parity rows or the remaining row types counted in `"row_type_counts"`, including all `"unit"`, `"headroom"`, `"protected_clause"`, `"commitment"`, and `"split_membership"` rows.

## THE CHECK A READER CANNOT DO
Do all 48 units actually have paired matching backend results, headroom, protected clauses, and pre-inference commitments sufficient to recompute `"prospective_contract_ready_score_from_rows": 1.0`?

## experiment_6483_v559_latent_energy_sota_ingestion.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V559 source-ingestion and method-mapping task completed, met its source, mapping, and no-execution gates, and makes no experimental performance claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6484_non_generation_representation_receipt_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The non-generation representation receipt contract is ready with a score of 1.0, and all 10 tested attacks fail closed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6485_online_cache_transition_eprocess_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The online transition contract is ready, with a row-recomputed readiness score of 1.0 and all mutation attacks failing closed.

## WHAT IS MISSING
nothing; the artifact includes `"action_rows"`, `"event_rows"`, `"evidence_process_rows"`, `"attack_matrix.rows"`, and `"aggregate_row_recomputation"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6486_three_family_forced_candidate_representations.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion claims candidate commitments and preconditions are complete, but the artifact is truncated before any final headline claim or verdict.

## WHAT IS MISSING
The remainder of the JSON, including any final verdict, gate status, `gate_check_summary`, and per-unit outcome metrics; only fields such as `"aggregate_row_recomputation"` and `"candidate_commitment_manifest"` are visible.

## THE CHECK A READER CANNOT DO
Does the complete artifact make a comparative or blocked headline claim, and if so, does it include the per-unit metrics or failed-check diagnostic needed to verify it?

## experiment_6487_representation_integrity_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The representations are disqualified because three shortcut controls survived and token, candidate, and prompt lengths were unavailable.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
