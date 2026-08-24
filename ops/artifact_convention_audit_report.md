# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| CANNOT_DETERMINE | 2 |

## experiment_6562_constraint_saturation_independent_audit_v2.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact’s headline claim is that the result was disqualified because four named audit checks failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6563_production_safety_net_workload_canary.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Disabled identity, exact equality, fallback, restart, and rollback passed, while enabled routing produced no measured work or latency benefit.

## WHAT IS MISSING
The artifact is truncated inside `"per_unit_rows"` at `"row_hash"`; complete per-unit metrics are missing for the declared `"unsupported"`, `"fallback_heavy"`, `"exception"`, `"restart"`, and `"rollback"` workloads and their conditions.

## THE CHECK A READER CANNOT DO
Did enabled routing fail the work and latency thresholds on every workload—including restart and rollback—rather than only on the visible rows?

## experiment_6564_rust_pyo3_safety_net_nfr01.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Exact Python/Rust parity passed, but the Rust PyO3 batch achieved only 0.764013447× median speedup—below the 10.0× gate—with p99 latency of 6.312575e-05 seconds.

## WHAT IS MISSING
The artifact is truncated inside `"per_unit_rows"` and shows only `"implementation": "python_scalar"` rows; the corresponding per-unit `"rust_pyo3_batch"` measurements needed to check `"honest_verdict"` are not present in the supplied text.

## THE CHECK A READER CANNOT DO
Do paired Python and Rust per-unit timings actually reproduce the claimed 0.764013447× median batched speedup and exact parity result?

## experiment_6565_v569_evidence_and_retirement_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V569 evidence and retirement contract is ready: V568 artifacts are individually classified, required contracts close, and the Rust-fusion boundary may reopen.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6566_proof_obligation_and_graph_potts_method_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The source method contract is complete and ready, with its proof-obligation schema, splits, graph features, Potts equations, matched-dose arms, gates, attacks, and retirement rules frozen.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6567_sequential_flagship_gguf_admission.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked because all three flagship model families failed runtime admission after the `model_identity_and_file_shape` precondition failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6568_immutable_source_span_claim_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp6567-sequential-flagship-gguf-admission.all_mandated_models_loaded_score` was 0.0 instead of the required 1.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6570_proof_obligation_independent_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The independent audit was blocked because required evidence was missing or unusable, so audit readiness and promotion were not confirmed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
