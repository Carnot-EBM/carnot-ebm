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

## experiment_6565_v569_evidence_and_retirement_contract.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims the V569 evidence contract is ready, including that Exp6564 is a clean null because its median Rust speedup was 0.764013447×, below the 10× gate.

## WHAT IS MISSING
Benchmark-level rows containing each workload/trial’s Python time, Rust time, and resulting speedup are missing; the present `"per_unit_rows"` contains only one eligibility row per experiment, while `"measured_speedup_vs_requirement"` and `"exp6564_speedup_vs_python_scalar"` record only the aggregate 0.764013447× value.

## THE CHECK A READER CANNOT DO
Can the reported 0.764013447× median be recomputed from individual benchmark measurements, and was the failed 10× gate broad across workloads rather than caused by a degenerate or anomalous subset?

## experiment_6571_v570_evidence_gate_and_retirement_root.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V570 evidence and retirement contract is ready, with prior blocked and missing scopes documented, while no extraction, model-generation, or graph-Potts utility run is claimed.

## WHAT IS MISSING
nothing; `"per_unit_rows"`, `"v569_artifact_eligibility_rows"`, `"gguf_admission_root_cause.per_model_rows"`, `"failed_scope"`, `"reason"`, and detailed `"honest_verdict"` diagnostics are present.

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

## experiment_6729_v587_activation_evidence_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V587 activation contract is blocked because one or more source, manifest, gate, prior-failure, or model-policy checks failed.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records the failed checks with `"expected_value"`, `"observed_value"`, `"reason"`, and `"unit"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6730_arc_context_tool_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp6729-v587-activation-evidence-contract.v587_contract_ready` was `false` but required to equal `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6732_object_table_ab_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required upstream artifact `exp6731-object-table-fetch-on-demand-ab` was not found.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6733_hardness_controlled_certificate_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because `exp6729-v587-activation-evidence-contract.v587_contract_ready` was `false` instead of the required `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6735_oracle_distinct_diagnostic_energy.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required upstream artifact for task `exp6734-sota-dual-encoding-proposal-corpus` was not found, causing the `dual_encoding_corpus_ready == true` gate to fail.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
