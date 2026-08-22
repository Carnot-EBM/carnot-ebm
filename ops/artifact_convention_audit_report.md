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

## experiment_6012_hidden_state_trust_gate_hole.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live hidden-state trust gate admits a spurious writer that the REQ-6011 change gate rejects.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6013_hidden_state_change_gate_closure.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The change gate’s hidden-state coverage hole was closed, as tested across base and adversarial engine arms with and without HUD masking.

## WHAT IS MISSING
nothing; `"rows"` provides per-`"game"`/`"seed"` results, including `"change_gate_pass"`, `"change_gate_reason"`, `"change_fidelity"`, and diagnostic counts.

## THE CHECK A READER CANNOT DO
none

## experiment_1644_cerce_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The CERCE ledger was added and is ready, with no policy certificates evaluated or policy updates attempted.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1767_e2e_qwen.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6510_v563_independent_exact_root.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The V563 independent exact root is ready with `"v563_independent_root_ready_score": 1.0` because all 480 replay rows and all seven dependency attacks passed.

## WHAT IS MISSING
The complete `"per_unit_rows"` array covering all 480 claimed rows; the supplied artifact ends mid-row, although `"raw_row_count": 480` and `"overall_independent_row_checks_passed": true` are present.

## THE CHECK A READER CANNOT DO
Do all 480 individual replay rows actually pass and support the reported ready score, rather than only the displayed subset?

## experiment_1736_kanele_synth.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment reports `vivado_simulated_success` with a generated bitfile, while being flagged as adversarial and excluded from headline aggregation.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_2031.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run succeeded and reported `"Thus, we can see it."` as `"best_candidate"` with `"min_energy": 0.0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6512_branch_dataset_independent_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit was blocked because the upstream artifact was missing, leaving zero rows for independent recomputation and no complete shard manifest.

## WHAT IS MISSING
nothing; `gate_check_summary`, `upstream_artifact_receipt.exists`, `independent_row_recomputation.row_count`, and `shard_and_censoring_audit.manifest_complete` record the failed checks and observed values.

## THE CHECK A READER CANNOT DO
none
