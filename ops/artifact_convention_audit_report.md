# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| CANNOT_DETERMINE | 4 |

## experiment_8659_v743_complete_source_comparison.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact reports a complete positive source comparison, with all acceptance gates passed and a coverage-benefit score of 1.

## WHAT IS MISSING
Per-source paired coverage outcomes supporting "coverage_benefit_score" are not visible. "complete_extraction_cost_rows" contains per-unit costs, but the supplied artifact ends mid-row, so whether coverage rows appear later is unknown.

## THE CHECK A READER CANNOT DO
Is the claimed coverage benefit spread across sources or driven by a single source?

## experiment_8660_v743_installed_compatibility.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports passed compatibility and measurement-readiness checks, with no comparative learning-benefit claim or blocked verdict.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8661_v743_installed_complete_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact reports qualified measurement and packaging noninferiority passed, while NFR-01 failed.

## WHAT IS MISSING
The remainder of the artifact: it ends mid-"cell_id" inside "complete_operation_spans". Per-cell timing rows are present, and "acceptance_gates" plus "branch_gate_values.nfr01_score" identify the failed check and score, but the complete comparison and its decision criteria are unavailable.

## THE CHECK A READER CANNOT DO
Do the complete per-cell results satisfy the criterion used to mark "packaging_noninferiority" as passed?

## experiment_8662_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked because `runtime_repair_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8663_v743_arc_reset_provenance.json

**CHECKABLE**

## VERDICT

CHECKABLE

## WHAT THE CLAIM IS

The bounded reset-provenance engineering witness completed with readiness score 1, claiming no solve, policy benefit, or generalization.

## WHAT IS MISSING

nothing

## THE CHECK A READER CANNOT DO

none

## experiment_8664_v743_arc_reset_transfer.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports passing acceptance gates and action parity for eight game–seed pairs, but the overall headline claim is unavailable.

## WHAT IS MISSING
The remainder of the artifact: it cuts off mid-string inside "callback_sequence_rows". "acceptance_gates" and per-unit "action_parity_rows" are present, but the overall verdict and any subsequent diagnostics are unavailable.

## THE CHECK A READER CANNOT DO
Does the overall verdict report a blocker, and if so, which check failed and what value did it observe?

## experiment_8665_v743_hardware_operation_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Acquisition priority remains unchanged because no compatible physical kernel or complete new cost boundary has been established.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8666_v743_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports unmet learning and physical-execution gates and a successful FoVer headline reproduction.

## WHAT IS MISSING
The artifact ends mid-value inside "cold_reductions.hardware.complete_boundary_rows", leaving the remaining evidence unavailable. "G2" reports aggregate AUROC and learning-contribution results without corresponding unit-level metrics in the visible portion. "branch_evidence_limit" supplies blocker diagnostics.

## THE CHECK A READER CANNOT DO
Does the omitted remainder provide per-condition metrics supporting the FoVer learning-contribution PASS reported in "G2"?
