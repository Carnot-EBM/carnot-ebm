# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| CANNOT_DETERMINE | 5 |

## experiment_8379_v722_native_direct_parity.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Native and Python computations pass finite parity, while the run remains disqualified from readiness.

## WHAT IS MISSING
The complete "rows" array: the artifact ends mid-record despite "completed_count" being 4533. The remaining artifact is also unavailable, so empty "gate_check_summary" and null "owned_failure" cannot establish that diagnostics are absent everywhere.

## THE CHECK A READER CANNOT DO
Do all 4533 completed units support the reported zero action mismatches and maximum probability error of 1.1102230246251565e-16?

## experiment_8381_v722_logit_policy_certificate.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment reports certificate readiness and passing acceptance gates.

## WHAT IS MISSING
The remainder of the artifact: it cuts off inside "cited_upstream_artifacts", preventing inspection of any final verdict, comparative results, per-unit rows, or blocker diagnostics.

## THE CHECK A READER CANNOT DO
Does the complete artifact report a comparative or blocked verdict with the evidence needed to check it?

## experiment_8382_v722_runtime_evidence_delta.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim is unavailable in the truncated artifact.

## WHAT IS MISSING
The artifact’s remainder: it cuts off inside "cited_upstream_artifacts" at "snapshot_path", leaving the final verdict and any comparative evidence unavailable. "acceptance_gates" does identify two failed checks: "authenticated_change": false and "current_context_copy": false.

## THE CHECK A READER CANNOT DO
Does the missing portion make a comparative claim and provide per-unit rows supporting it?

## experiment_8383_changed_runtime_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The canary was blocked because `runtime_changed_score` and `cuda_context_ready_score` were both 0 while their gates required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8384_v722_arc_supervisor_live_panel.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All eight bounded panel episodes completed without established scientific benefit, while readiness remained blocked by “independent_design_contract_available” observing null instead of true in “gate_check_summary”.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8385_v722_board_operation_evidence.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment reports failed qualification gates, but the headline verdict is unavailable.

## WHAT IS MISSING
The artifact truncates mid-"original_path" inside "cited_upstream_artifacts", leaving subsequent verdict and result fields unavailable. "acceptance_gates" and "board_obligations.kv260.missing" do record failure diagnostics.

## THE CHECK A READER CANNOT DO
Does the headline verdict make a comparative claim supported by per-unit metrics?

## experiment_8386_v722_gatemate_obligation_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The continuity obligation was recorded, while historical authentication and hardware execution remain blocked by missing source bytes and physical/device evidence.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8387_v722_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
H1 reports a negative mean cost gain for spline34 versus RBF34: −0.00390625 across 128 intended units.

## WHAT IS MISSING
The artifact ends mid-field inside "paired_cost_rows" at slot 100; that row’s remainder and slots 101–128 are unavailable, while "intended_count" and "bootstrap_summary.all_intended.source_count" both report 128.

## THE CHECK A READER CANNOT DO
Do all 128 paired cost differences reproduce "bootstrap_summary.all_intended.mean_gain" of −0.00390625?
