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

## experiment_9083_v775_prospective_custody.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Supervised fixture custody qualifies, but proposals and collection are blocked by missing historical attempt topology coverage.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

External references were not checked. The supplied `"gate_check_summary"` explicitly records `"historical_attempt_topology_coverage"` failing with `"observed": "missing_field"` and `"expected": true.

## experiment_9084_v775_human_methods.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible “claim_scope” covers the exposed LPHB inventory and frozen comparison protocol, with no detector-benefit claim.

## WHAT IS MISSING
The artifact’s remainder: it ends mid-key inside “exposure_ledger.” “blocker”: “historical_producer_hash” and populated “gate_check_summary” entries provide diagnostics; those are not missing.

## THE CHECK A READER CANNOT DO
Does any headline comparison have supporting per-unit metric rows?

The truncated remainder was not checked; its claims and evidence remain unknown.

## experiment_9085_collect_a.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Collection was blocked because `custody_ready_score` and `roster_ready_score` were both 0, while each gate required 1.

## WHAT IS MISSING
nothing — `gates_evaluated` records each failed check, operator, expected value, observed value, and reason.

## THE CHECK A READER CANNOT DO
none

## experiment_9086_collect_b.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Collection was blocked because “custody_ready_score” and “roster_ready_score” each equaled 0 while both gates required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9088_human_embeddings.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9088 was blocked because upstream `human_methods_ready_score` was 0, failing the requirement that it equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9093_v775_service_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The "scientific_service_complete" gate failed, with its "reason" identifying unavailable scientific request clocks, model/device attribution, and denominators; no comparative performance claim is recorded.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9094_v775_susceptibility.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible "acceptance_gates" mark equal binary energies and several other checks as passing.

## WHAT IS MISSING
The artifact is truncated inside "binary_energy_rows"; the headline verdict and remaining evidence are unavailable. Visible rows contain per-unit "energies", "alphas", "seed", and "condition", so they are not aggregate-only.

## THE CHECK A READER CANNOT DO
Do all registered conditions support the reported gate outcomes?

The omitted remainder cannot be checked; overall checkability remains unknown.

## experiment_9095_v775_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible “claim_scope” reports fourteen administrative outcomes and no new detector, online, or general verifier benefit.

## WHAT IS MISSING
The artifact ends mid-field at “cost_summary” → “bottleneck_opera”. Any subsequent verdict, diagnostic fields, or actual per-unit measurements are unavailable; “task.per_unit_rows”: true records a requirement, not measurements.

## THE CHECK A READER CANNOT DO
Does the retained susceptibility result have seed-paired measurements supporting its “complete_circular_positive_susceptibility_mechanism” verdict?

No reruns or referenced files checked; truncation prevents a conclusive classification.
