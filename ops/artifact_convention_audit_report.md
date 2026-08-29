# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 2 |

## experiment_6484_non_generation_representation_receipt_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The non-generation representation receipt contract is ready with a row-recomputed score of 1.0, including successful positive validation and fail-closed handling of all attacks.

## WHAT IS MISSING
nothing; supporting unit-level evidence appears in `"attack_matrix.rows"`, `"candidate_commitment_rows"`, and `"family_separation_receipts"`, while `"aggregate_row_recomputation"` records the reducers, row counts, checks, and `"failed_checks"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6487_representation_integrity_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The representation audit is disqualified because three shortcut controls survived and three required length checks were unavailable.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6491_sota_factor_proposal_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The factor-proposal stream met its readiness gate with a score of 1.0 and all boundary attacks failed closed.

## WHAT IS MISSING
nothing; `"exact_compile_rows"` records every event/model outcome and reason, while `"boundary_attack_matrix.rows"` records every attack result and blocker.

## THE CHECK A READER CANNOT DO
none

## experiment_6520_safety_net_branch_router_ab.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The learned linear router seed 6520002 passed the gate and delivered 695 held benefit units, beating the best structural arm’s 667 units by 28.

## WHAT IS MISSING
Per-held-unit rows containing each arm’s charged work or benefit, keyed by problem family and seed; only aggregate `"arm_held_summaries"`, `"held_total_charged_work_units"`, `"held_charged_benefit_units"`, and win/loss counts are present, while `"candidate_preservation_rows"` contain no outcome metric.

## THE CHECK A READER CANNOT DO
Was the claimed 28-unit advantage broad across held units, or concentrated in one or two outliers while other units tied or lost?

## experiment_6561_v568_evidence_gate_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V567 inputs are content-addressed, Exp6549–Exp6551 are eligible production evidence, and the V568 gate, prior-failure, model, hardware, and protected-file contracts are ready.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6565_v569_evidence_and_retirement_contract.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The V569 evidence contract is ready, including the claims that Exp6563 and Exp6564 are clean nulls and Exp6564 achieved only 0.764013447× speedup against a 10× requirement.

## WHAT IS MISSING
Per-benchmark or per-repetition timing rows for the Python and Rust arms, and per-workload enabled/control metric rows for Exp6563; the present `"per_unit_rows"` contains only one eligibility summary per upstream artifact, while `"measured_speedup_vs_requirement"` and `"honest_verdict"` report aggregate conclusions.

## THE CHECK A READER CANNOT DO
Was Exp6564’s 0.764013447× median speedup miss broad across benchmark units, or produced by a few extreme measurements?

## experiment_6571_v570_evidence_gate_and_retirement_root.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V570 evidence and retirement contract is ready, with blocked and missing V569 scopes explicitly accounted for and no claim that source extraction or graph-Potts utility ran.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6716_object_table_fetch_on_demand_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The paired A/B experiment was blocked before starting because `CARNOT_ARC_INDUCE_N_CTX` was unset.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
