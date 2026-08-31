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

## experiment_6796_agent_model_dispatch_requalification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The dispatch requalification was blocked because `research-roadmap-next.yaml` and `scripts/agent_model_compatibility.py` did not exist.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6797_canonical_transaction_byte_replay.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims that a complete transaction-byte snapshot fixture is ready, with 3,189 commits retaining verified canonical parent and new-state bytes.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6798_csl_causal_safety_byte_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The compositional-online arm beat the frozen controller on held-future utility across all five orders, with an `online_minus_frozen_lcb` of 0.34375.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6799_model_output_formal_constraint_probes.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim identifiable because the artifact is truncated mid-record

## WHAT IS MISSING
The remainder of the artifact, including any final verdict, gate status, or headline-claim fields; `"adversarial_attack_receipts"`, `"dual_encoding_diagnostics.source_case_receipts"`, and `"exact_replay_receipts"` are present, but the JSON ends inside `"observed_valid_set_hash"`.

## THE CHECK A READER CANNOT DO
Does the complete artifact ultimately claim a comparative gate result or a blocked verdict, and if so, are the required per-unit rows or blocking diagnostic recorded?

## experiment_6800_real_output_fixed_point_transfer_ab.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim is visible in the provided fragment

## WHAT IS MISSING
The artifact is truncated mid-value inside `"checkpoint_receipt.payload_hashes"` and lacks the remainder of the JSON, so any verdict, comparative metrics, per-unit rows, or blocker diagnostic that may follow cannot be inspected; visible fields include `"candidate_budget_by_arm"`, `"checkpoint_receipt"`, and `"completed_row_count"`.

## THE CHECK A READER CANNOT DO
Does the complete artifact make a comparative or blocked claim, and if so, does it include the required per-unit metrics or blocker diagnostic?

## experiment_6801_real_output_fixed_point_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Grouped fixed point produced only negligible exact-valid-rate differences versus flat recurrent control, with all clustered confidence intervals spanning zero.

## WHAT IS MISSING
nothing; `"metrics_by_transformation_model_family.by_case"` records case-level metrics for both arms, supplemented by `"by_seed"` and `"clustered_confidence_intervals"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6802_operational_obligation_automaton_v2.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because the required `openspec/capabilities/agentic-verification/spec.md` did not exist.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6803_sota_operational_handoff_corpus.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the upstream `operational_automaton_fixture_ready` gate observed `false` but required `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
