# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 2 |
| CANNOT_DETERMINE | 6 |

## experiment_8701_v746_relational_feature_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
No headline claim is visible in the supplied fragment.

## WHAT IS MISSING
The artifact cuts off inside "code_config_hashes"; its headline claim and final verdict are unavailable. "acceptance_gates" and "branch_gate_values" are present but do not establish the reported outcome.

## THE CHECK A READER CANNOT DO
What headline outcome does the complete artifact report?

## experiment_8702_v746_installed_batch_prototype.json

**CANNOT_DETERMINE**

reviewer call failed: codex launch error: TimeoutExpired(['/home/ianblenke/.local/bin/codex', 'exec', '--model', 'gpt-6.1-sol', '-'], 300)

## experiment_8703_installed_batch_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `batch_service_ready_score` was 0, failing the requirement that it equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8704_qwen_symbolic_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8704 was blocked because both readiness gates observed 0 where 1 was required.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8705_v746_arc_decision_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
ARC decision qualification is ready (`arc_decision_ready_score: 1`), with all four arms retained unchanged.

## WHAT IS MISSING
The artifact truncates inside `authority.tasks[1].prompt`. `arm_support_rows` gives arm totals, and `authority.comparison.rows` gives contract checks; game/episode-level outcome rows and the terminal verdict are not visible, so their absence from the complete artifact cannot be established.

## THE CHECK A READER CANNOT DO
Which individual game/episode outcomes account for the 16 censored outcomes reported for `force_exploration_diversity`?

## experiment_8706_v746_spline_lifecycle_stress.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible "acceptance_gates" claim complete lifecycle evidence and passing numerical criteria.

## WHAT IS MISSING
The supplied artifact cuts off mid-"state_hash" in "boundary_rows"; the top-level "honest_verdict", "verdict_class", complete "update_rows", "restart_rows", and remaining "boundary_rows" are not visible.

## THE CHECK A READER CANNOT DO
Does the complete artifact contain per-unit numerical results supporting "acceptance_gates.numerical_criteria": true?

## experiment_8707_v746_hardware_operation_boundary.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion claims all 14 task contracts match, but the experiment’s headline verdict is not visible.

## WHAT IS MISSING
The artifact truncates inside "authority.tasks[2].prompt"; top-level "honest_verdict" and "verdict_class" are not visible. "authority.comparison.rows" does contain per-task checks.

## THE CHECK A READER CANNOT DO
Does the final verdict assert experimental success, report a blocker, or merely record contract validation?

## experiment_8708_v746_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports successful FoVer headline reproduction while withholding activation authorization.

## WHAT IS MISSING
The supplied JSON cuts off inside "code_config_hashes", so per-unit results supporting the summary in "G2.detail" may be in the missing remainder. Blocker diagnostics are present in "acceptance_gates" and "branch_gate_values".

## THE CHECK A READER CANNOT DO
Does the positive FoVer learning contribution hold broadly across evaluation units, or depend on a few outliers?
