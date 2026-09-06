# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7030_arc_gguf_model_identity_bridge.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The ARC model identity bridge is ready because all positive, negative, legacy-compatibility, wiring, hash-join, and hub/revision checks passed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7031_arc_model_identity_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The ARC model-identity cold audit completed positively, with all readiness gates passing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7032_repaired_belief_shadow_live_trace.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The belief-shadow live trace was blocked because `live_trace_execution` rejected `observed_server_model_path` as an alias rather than the required canonical path.

## WHAT IS MISSING
nothing; `gate_check_summary.failed_check`, `gate_check_summary.observed_value`, `gate_check_summary.expected_value`, and the failed entry in `gate_check_summary.checks` record the exact blocker and observed error.

## THE CHECK A READER CANNOT DO
none

## experiment_7038_v617_active_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V617 active-contract preflight was blocked because the required `openspec/change-proposals/research-roadmap-vNEXT.md` file was missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7039_v617_model_report_forensics.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims that model-report-channel forensic evidence was successfully captured for the selected Qwen model, including one live one-token probe and safe cleanup, without claiming an ARC solve.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7040_v617_typed_identity_bridge.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked because the experiment 7039 artifact failed validation.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7041_identity_report_channel_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because `arc_typed_identity_bridge_ready_score` was 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7049_v617_capstone_disposition.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V617 capstone was blocked because the required markdown contract was missing.

## WHAT IS MISSING
nothing; `"gate_check_summary"` identifies `"failed_check": "v617_markdown_readable"` and `"observed_value": "missing"`.

## THE CHECK A READER CANNOT DO
none
