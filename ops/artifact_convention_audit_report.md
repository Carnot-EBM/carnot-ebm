# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 1 |

## experiment_6557_constraint_saturation_independent_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required `rerun_discipline` entry naming a prior failure was absent.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6558_arc_live_redirect_ledger_reachability.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live receipt path was reachable and seven arm firings were inspected, but the per-firing outcomes provided no supported reason to change the curated policy order.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6559_gatemate_changed_state_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because no dated, operator-authored GateMate physical-state receipt newer than Exp6525 was found, so zero hardware commands ran.

## WHAT IS MISSING
nothing; `"gate_check_summary.failed_check"`, `"gate_check_summary.observed_latest_receipt_date"`, `"operator_physical_state_receipt.candidate_rows"`, and `"per_unit_rows"` record the failed check, observed value, and candidate-level evidence.

## THE CHECK A READER CANNOT DO
none

## experiment_6560_v567_independent_capstone.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted field(s) ['gate_check_summary'] do not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
The artifact claims 12 tasks closed as three positive, five null, and four blocked, including a positive constraint-saturation result whose independent audit was blocked.

## WHAT IS MISSING
For `exp6557`, a `gate_check_summary` or equivalent failed-check identifier and observed value is missing; the artifact records only `"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, and `"blocked_gate_artifact": true`.

## THE CHECK A READER CANNOT DO
Which constraint-saturation audit gate failed, and what value caused it to fail?

## experiment_6561_v568_evidence_gate_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V567 artifacts are content-addressed, Exp6549–Exp6551 are eligible production inputs, and the V568 gate, prior-failure, model, hardware, and protected-file contracts are complete.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6562_constraint_saturation_independent_audit_v2.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims a `"disqualified"` verdict because four explicitly named checks failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6563_production_safety_net_workload_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The safety-net conditions passed, while enabled routing showed no preregistered work or latency benefit.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6564_rust_pyo3_safety_net_nfr01.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Exact parity passed, but Rust/PyO3 batching achieved only 0.764013447× median speedup—below the 10.0× threshold—while p99 latency met the frozen bound.

## WHAT IS MISSING
Although `"per_unit_rows"` is present, every written row has `"implementation": "python_scalar"` and `"batch_size": 1`; per-unit `"rust_pyo3_batch"` timing/parity rows needed to reproduce the comparative speedup and exact-parity claims are missing.

## THE CHECK A READER CANNOT DO
Did Rust/PyO3 batching underperform Python broadly across units and batch sizes, or did a small number of slow Rust measurements drive the reported 0.764013447× median?
