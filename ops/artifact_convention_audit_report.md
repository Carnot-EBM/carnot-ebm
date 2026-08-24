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

## experiment_3361_archive_v309_activate_v310.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Milestone 2026.05.309 was archived and milestone 2026.05.310 was activated successfully.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6572_content_derived_gguf_metadata_resolver.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The GGUF metadata resolver achieved readiness score 1.0 because all three flagship blobs passed and all 13 negative fixtures failed closed.

## WHAT IS MISSING
The per-fixture `"negative_fixture_rows"` with each `"unit_id"`, observed result, and rejection reason; only `"negative_fixture_pass_count"`, `"passed_negative_fixture_ids"`, and read-only `"bounded_read_receipts"` are present.

## THE CHECK A READER CANNOT DO
Did every negative fixture actually fail closed for its intended reason, rather than merely being counted as passing by the aggregate reducer?

## experiment_1644_cerce_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The CERCE ledger scaffolding was added and completed, with no policy certificates, violations, events, or policy updates evaluated.

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

## experiment_1736_kanele_synth.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 1736 reports simulated Vivado success with a generated bitfile, while explicitly flagged as adversarial and excluded from headline aggregation.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_2031.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment succeeded, producing `"Thus, we can see it."` as `"best_candidate"` with `"min_energy": 0.0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6573_sequential_flagship_gguf_admission_v2.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All three mandated model families passed the runtime-admission gate, so downstream model science is open.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6574_joint_sufficiency_method_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The joint-sufficiency method is complete, executable, and frozen, with its schemas, fixtures, attacks, gates, splits, arms, and retirement rules ready.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
