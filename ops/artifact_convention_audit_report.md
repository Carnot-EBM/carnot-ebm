# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7110_v624_evidence_ingress_quarantine.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact records per-capstone reference classifications and asserts that each listed capstone’s byte hash was unchanged.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7111_v624_arc_provenance_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The ARC forward-provenance canaries passed: writer and consumer behavior matched expectations, while the registry remained unchanged.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7112_v624_sota_ingestion.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The bounded V624 source audit completed successfully, passed its ingestion gate with an observed score of 1, and adopted one candidate as a controlled experiment hook.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7113_v624_arc_generation_liveness.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The ARC generation-liveness run was blocked because required preconditions failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7114_adapter_withheld_arc_loo_measurement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `arc_generation_liveness_ready_score` was `0` instead of the required `1`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7121_v625_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V625 contract is disqualified because the Markdown specifies 13 tasks while the active YAML contains only 3.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7122_v625_sota_ingestion.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V625 SOTA ingestion completed successfully and its gate passed with an observed score of 1 against an expected value of 1.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records `"passed": true`, `"expected_value": 1`, `"observed_value": 1`, and `"failed_check": null`, with supporting source, model, mapping, date, and precondition rows.

## THE CHECK A READER CANNOT DO
none

## experiment_7123_v625_arc_loo_shard_a.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked without running because neither required arm completed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
