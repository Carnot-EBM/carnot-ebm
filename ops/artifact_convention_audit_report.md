# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7038_v617_active_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V617 active-contract preflight was blocked because the required roadmap Markdown file was missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7039_v617_model_report_forensics.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims the Qwen3.6 model-report forensics completed successfully with CUDA-backed live inference and all recorded gates passing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7040_v617_typed_identity_bridge.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the exp7039 artifact failed validation.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7041_identity_report_channel_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `arc_typed_identity_bridge_ready_score` was 0 but was required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7049_v617_capstone_disposition.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V617 capstone is blocked because the required markdown contract is missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7050_v618_active_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V618 Markdown/YAML task contract is disqualified because the active YAML contains only 3 of the expected 13 tasks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7051_v618_model_report_requalification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims that live-model report evidence was successfully requalified, with every recorded gate passing, while making no ARC game-level solve claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7052_v618_typed_identity_attack_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The typed identity attack audit passed: all positive fixtures were accepted, all one-factor attacks failed closed, fresh-process results agreed, legacy handling was explicit, and all listed producers and consumers used shared code.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
