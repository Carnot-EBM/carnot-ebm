# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| AGGREGATE_ONLY | 1 |

## experiment_3582_capstone_v329.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
A second pair of eyes provides real verifier lift within a math-only, domain-bound scope, despite not generalizing to code or facts.

## WHAT IS MISSING
Per-unit verifier and control-arm metric rows for each game, seed, cell, or condition; only aggregate conclusions such as `"second_pair_of_eyes_lift_real": true`, `"code_generalizes": false`, and `"facts_generalize": false` are present.

## THE CHECK A READER CANNOT DO
Did verifier lift occur broadly across units, or was the claimed improvement driven by one outlier or degenerate controls?

## experiment_1736_kanele_synth.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 1736 reports a simulated Vivado success with a generated bitfile, while being flagged adversarial and excluded from headline aggregation.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_3361_archive_v309_activate_v310.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_3377_archive_v310_activate_v311.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The archive of milestone 2026.05.310 is complete and milestone 2026.05.311 is ready for activation.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_3392_archive_v311_activate_v312.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The v311 archive is complete and v312 activation is ready.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_833_constraint_delta_root_cause.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment claims the embedding constraint store’s write path is missing, evidenced by `"n_store_write_calls": 0` despite 10 retrieval calls.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6479_verify_repair_factor_cache_shadow_adapter.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The default-off verify-repair factor-cache shadow adapter shipped successfully and met all readiness gates.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_1644_cerce_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The CERCE ledger scaffolding was added and is ready, with no policy certificates, violations, FR11 events, or policy updates evaluated.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
