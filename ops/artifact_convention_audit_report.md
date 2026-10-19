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

## experiment_8645_complete_source_comparison.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `streaming_extractor_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8646_v742_installed_compatibility.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The recorded acceptance checks pass except for “exact_node_union” and “installed_compatibility.”

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8647_installed_complete_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8647 was blocked because the upstream `installed_measurement_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8648_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked because `runtime_repair_ready_score` was 0 while the gate required 1, as recorded in `gates_evaluated` and `gate_check_summary`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8649_v742_arc_reset_provenance.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Reset-provenance recording passed qualification on three public fixtures with action parity preserved, without claiming learning benefit or solve credit.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8650_v742_arc_reset_transfer.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment reports passing acceptance gates and action parity, but the overall headline verdict is unavailable.

## WHAT IS MISSING
The artifact’s remainder: it cuts off mid-string inside "censored_reason_rows". "action_parity_rows" and "cause_support_rows" are present, but the complete verdict and any "gate_check_summary" cannot be inspected.

## THE CHECK A READER CANNOT DO
Does the complete artifact report a blocked verdict, and if so, does it identify the failed check and observed value?

## experiment_8651_v742_hardware_operation_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Hardware-operation evidence is blocked by missing artifacts and failed readiness checks, documented in "gate_check_summary" with expected and observed values; no achieved speedup is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8652_v742_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion claims administrative completion and capstone readiness, and says the FoVer headline independently reproduced.

## WHAT IS MISSING
The artifact cuts off inside "cited_upstream_artifacts"; its final verdict, any blocker diagnostics, and any per-unit metric rows are unavailable. "administrative_complete", "capstone_ready_score", and "G2" do not resolve what the omitted remainder records.

## THE CHECK A READER CANNOT DO
Does the complete artifact supply per-unit evidence for its claimed learning contribution, or only aggregate assertions?
