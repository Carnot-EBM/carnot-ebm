# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| BLOCKED_WITHOUT_DIAGNOSTIC | 2 |

## experiment_7716_v672_qwen_semantic_pilot.json

**BLOCKED_WITHOUT_DIAGNOSTIC**

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
The pilot run completed but was disqualified on required acceptance gates (`"honest_verdict": "complete_disqualified_required_checks"`).

## WHAT IS MISSING
The diagnostic failure reasons or threshold criteria explaining why the acceptance gates failed. While `"gate_check_summary"` is present, it is empty (`[]`), and within `"acceptance_gate_results"`, `"coverage"`, `"readiness"`, and `"validity"` are marked `"passed": false` despite benign operands (e.g., `"validity"` reports `"failed_checks": 0` and `"coverage"` reports `"observed_families": 24` of `"intended_families": 24`) with no explanatory failure messages or thresholds attached.

## THE CHECK A READER CANNOT DO
Which specific criterion or threshold failed to cause `"passed": false` across `"coverage"`, `"readiness"`, and `"validity"` when `"failed_checks": 0` and `"gate_check_summary"` contains no diagnostic entries?

## experiment_7717_latent_evidence_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because three upstream gate checks failed, led by exp7715-natural-source-cohort natural_cohort_ready_score observing 0 instead of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7719_v672_acquisition_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was disqualified because a required validation check failed with exit code 2 instead of 0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7721_v672_independent_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The independent evidence audit is blocked because required upstream v672 evidence artifacts from Exp7718 and Exp7720 are missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7722_v672_arc_evidence_recovery.json

**BLOCKED_WITHOUT_DIAGNOSTIC**

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
The recovery is complete but disqualified by the terminal-reader gate, with no new solve credit.

## WHAT IS MISSING
The terminal-reader check’s observed result is missing. `honest_verdict` names the gate, but `gate_check_summary.failed_checks` is `[]`, `failed_count` is `0`, and `terminal_reader_receipts_path` provides only a path.

## THE CHECK A READER CANNOT DO
What did the terminal-reader check observe that caused the disqualification?

## experiment_7723_v672_native_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The qualification run is disqualified because administrative readiness checks failed (`focused_pytest`, `changed_module_coverage`, `changed_module_coverage_report`), despite passing measured validity parity checks between Python and Rust.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7724_complete_service_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims the experiment was blocked before execution at the conductor pre-gate because three upstream qualification gates on `exp7723-native-qualification` failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7725_v672_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The v672 capstone is blocked on required scientific evidence due to 18 failed gate and contract checks across upstream tasks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
