# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 2 |
| CANNOT_DETERMINE | 1 |

## experiment_3403_archive_v313_activate_v314.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted field(s) ['gate_check_summary'] do not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
The v313 archive and v314 activation are complete and ready, despite `exp3392-gatemate-n16-bootstrap-fix` being blocked.

## WHAT IS MISSING
A per-artifact blocker reason or `gate_check_summary` for the entry in `"blocked_artifacts"`; only its identifier is recorded.

## THE CHECK A READER CANNOT DO
Which check blocked `exp3392-gatemate-n16-bootstrap-fix`, and what value did that check observe?

## experiment_2514_kv260_pynq_flash.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The KV260 HWH file was successfully generated, while physical SD card flashing was not attempted because it requires manual operator preparation.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_3361_archive_v309_activate_v310.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The archive of milestone 2026.05.309 is complete and milestone 2026.05.310 is ready for activation.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_3377_archive_v310_activate_v311.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_3392_archive_v311_activate_v312.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The archive of milestone 2026.05.311 is complete and activation of 2026.05.312 is ready.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_833_constraint_delta_root_cause.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The verification pipeline’s constraint-store write path is missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6582_gemma4_31b_flagship_source_shard.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The Gemma4-31B source shard completed all readiness gates with `"gemma4_31b_family_source_shard_ready_score": 1.0`.

## WHAT IS MISSING
The actual top-level `"rows"` containing each source unit’s readiness checks and metrics; only aggregate fields such as `"claim_bearing_row_count": 4`, `"failure_row_count": 0`, and `"ready_score": 1.0`, plus per-unit checkpoint and parser receipts, are present.

## THE CHECK A READER CANNOT DO
Did every one of the four source units independently satisfy all readiness conditions, or did the aggregate conceal a degenerate or anomalous unit?

## experiment_6583_gemma4_26b_a4b_flagship_source_shard.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The Gemma4-26B-A4B runtime and immutable four-unit source shard completed with readiness score 1.0.

## WHAT IS MISSING
The actual per-unit terminal `rows` containing each unit’s readiness inputs and results are missing; only aggregate fields such as `"gemma4_26b_a4b_family_source_shard_ready_score"`, `"aggregate_row_recomputation"`, and auxiliary `"checkpoint_receipts"` and `"parser_diagnostic_rows"` are present, despite `"field_provenance"` referring to `"rows"`.

## THE CHECK A READER CANNOT DO
Did each of the four source units independently satisfy every readiness condition, or does the reported aggregate readiness conceal a deficient or degenerate unit?
