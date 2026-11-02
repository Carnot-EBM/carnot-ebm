# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 1 |
| CANNOT_DETERMINE | 7 |

## experiment_8715_v747_verifier_completeness.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fields report all four acceptance gates passed and a verifier audit readiness score of 1.

## WHAT IS MISSING
The artifact ends mid-hash in "code_config_hashes". "acceptance_gates" and "branch_gate_values" are visible, but the remainder is unavailable, so supporting per-unit results or blocker diagnostics cannot be assessed.

## THE CHECK A READER CANNOT DO
Which individual check outcomes support the four reported gate passes?

## experiment_8716_v747_installed_import_isolation.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible "acceptance_gates" report success, including "installed_parity", but the artifact is truncated before a headline verdict is visible.

## WHAT IS MISSING
The remainder of "batch_parity_rows", which ends inside a "response", and any subsequent verdict or comparison evidence; all visible rows have "arm": "python".

## THE CHECK A READER CANNOT DO
Does the complete artifact record matching per-unit outputs from both implementations supporting the "installed_parity" pass?

## experiment_8717_v747_installed_batch_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The qualified complete-cost and batching-benefit gates passed, while the B16 same-batch 10× gate failed.

## WHAT IS MISSING
The complete artifact: it ends mid-"process_id" within "complete_cost_rows". Per-process "duration_s" measurements are present, but the remaining measurements and calculations supporting "benefit_gates" are not visible.

## THE CHECK A READER CANNOT DO
Does the complete measurement set support marking "batching_benefit" as passed?

## experiment_8718_qwen_symbolic_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `runtime_repair_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing — `"gate_check_summary"` and `"gates_evaluated"` identify the failed check, observed value, operator, and required value.

## THE CHECK A READER CANNOT DO
none

## experiment_8719_v747_compiled_sparse_update.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
All listed “acceptance_gates” passed, including “complete_rows” and “exact_native_parity”.

## WHAT IS MISSING
The remainder of the artifact: “boundary_rows” cuts off inside an “ack” value, so the complete set of per-unit results is unavailable.

## THE CHECK A READER CANNOT DO
Do all required units support “exact_native_parity”, or only the rows visible before the cutoff?

## experiment_8720_v747_arc_supervisor_frontier.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment claims readiness through `"arc_frontier_ready_score": 1` and `"authority.ready": true`.

## WHAT IS MISSING
The complete artifact: it ends mid-string inside `"authority.rows[3].task_object.prompt"`, leaving the remaining rows and terminal verdict unavailable.

## THE CHECK A READER CANNOT DO
Are all fourteen task contracts represented by completed per-task check rows?

## experiment_8721_v747_hardware_operation_boundary.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible artifact declares board authority and owned validation satisfied.

## WHAT IS MISSING
The artifact ends mid-string in the fourth `"authority.rows"` entry; the remaining rows and actual terminal `"honest_verdict"` / `"verdict_class"` values are unavailable. Visible `"authority.ready"` is true and `"authority.gate_check_summary"` is empty.

## THE CHECK A READER CANNOT DO
Did all fourteen task contracts pass the checks supporting the readiness claim?

## experiment_8722_v747_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The excerpt reports successful FoVer headline reproduction and `"activation_authorized": false`, but the complete headline verdict is unavailable.

## WHAT IS MISSING
The artifact is truncated inside `"code_config_hashes"`, preventing inspection of any later verdict or per-unit results supporting the intervals in `"G2.detail"`. `"acceptance_gates"` and `"branch_gate_values"` already identify failed gates, so a blocker without diagnostics is not established.

## THE CHECK A READER CANNOT DO
Was the positive FoVer learning contribution broadly distributed across units or driven by a few outliers?
