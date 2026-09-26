# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| BLOCKED_WITHOUT_DIAGNOSTIC | 1 |

## experiment_7672_v669_bound_relations.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The bound-relation protocol passed validity, readiness, and coverage checks on fixture groups, while the remaining benefit gates were not established.

## WHAT IS MISSING
nothing; `acceptance_gate_results` records the gate values, and `fixture_results` provides per-unit `truth`, `observed`, and `raw_metrics`.

## THE CHECK A READER CANNOT DO
none

## experiment_7673_v669_fresh_relation_cohort.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports cohort infrastructure only and makes no learned-verifier improvement claim.

## WHAT IS MISSING
nothing; `acceptance_gate_results` records each failed gate with its `expected`, `observed`, and `passed` values.

## THE CHECK A READER CANNOT DO
none

## experiment_7674_relation_energy.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked at conductor pre-gate because three of four upstream gate checks on exp7673-fresh-relation-cohort failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7676_v669_qwen_quote_relations.json

**BLOCKED_WITHOUT_DIAGNOSTIC**

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
The run was blocked and disqualified by required checks (`honest_verdict`: `"complete_disqualified_required_checks"`), failing all acceptance gates and denying activation and production promotion.

## WHAT IS MISSING
A diagnostic identifying which check failed and what value it observed. While `honest_verdict` reports `"complete_disqualified_required_checks"` and all gates in `acceptance_gate_results` report `"passed": false`, `gate_check_summary` reports `"failed_checks": []`, `"failed_count": 0`, and `"first_failure": null`, all 7 entries in `preconditions_checked` report `"passed": true`, and `acceptance_gate_results.validity.measured_operands` reports `"required_checks_passed": true`.

## THE CHECK A READER CANNOT DO
Which required check failed and what value did it observe to trigger the disqualification verdict `complete_disqualified_required_checks`?

## experiment_7679_v669_independent_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit is blocked because required static and online evidence was missing or failed producer checks, so no independent benefit was confirmed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7680_v669_arc_probe_protocol.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7681_v669_arc_live_probes.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7684_v669_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V669 capstone is accounting-ready but blocked from claiming scientific benefit because required evidence and upstream gates are missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
