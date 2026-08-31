# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_6796_agent_model_dispatch_requalification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Dispatch requalification was blocked because `research-roadmap-next.yaml` and `scripts/agent_model_compatibility.py` did not exist.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6797_canonical_transaction_byte_replay.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims that 3,189 commits retain verified canonical parent and new-state bytes, making the transaction-byte snapshot fixture ready.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6798_csl_causal_safety_byte_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The compositional-online arm beat the frozen controller across all five orders, with an `online_minus_frozen_lcb` of 0.34375, while replay and receipt checks passed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6799_model_output_formal_constraint_probes.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The adversarial checks passed, dual-encoding diagnostics found 87 reasoning errors and 10 valid cases with no translation disagreements, and exact replays matched expected outputs.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6800_real_output_fixed_point_transfer_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6801_real_output_fixed_point_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Grouped fixed-point performance differs only marginally from the flat recurrent control across transformations, with all reported 95% confidence intervals spanning zero.

## WHAT IS MISSING
nothing; `"clustered_confidence_intervals"` provides comparative estimates, while `"metrics_by_transformation_model_family.by_case"` and `"by_seed"` provide unit-level arm metrics.

## THE CHECK A READER CANNOT DO
none

## experiment_6802_operational_obligation_automaton_v2.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because `openspec/capabilities/agentic-verification/spec.md` did not exist.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6803_sota_operational_handoff_corpus.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `operational_automaton_fixture_ready` was observed as `false` but required to equal `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
