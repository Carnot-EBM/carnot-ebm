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

## experiment_8999_v769_branch_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Administrative custody readiness is asserted, with no scientific benefit or independent generalization claim.

## WHAT IS MISSING
The artifact is truncated inside "frozen_science.checks" at "observed.test". Actual "honest_verdict", "verdict_class", and "gate_check_summary" values are unseen; those names appear only in "field_principles".

## THE CHECK A READER CANNOT DO
Does the final verdict confirm readiness or report a blocker with its failed check and observed value?

No rerun or external files checked; the missing tail may contain the answer.

## experiment_9000_v769_invocation_provenance.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact assesses producer invocation custody only, with no verifier benefit or new inference claimed.

## WHAT IS MISSING
The supplied artifact ends mid-entry in "gate_check_summary". "acceptance_gates.owned_checks.passed" is false, but all visible diagnostic entries pass; the final verdict and any later failure diagnostic are unavailable.

## THE CHECK A READER CANNOT DO
Does the omitted remainder identify which owned check failed and what value it observed?

## experiment_9001_v769_human_source_authority.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The source-authority audit fails readiness: “acceptance_gates” records both readiness scores as 0 against expected 1, and “required_checks_passed” as false against expected true.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9002_retained_shard_a.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9002 was blocked because upstream `provenance_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9003_graph_corpus_b.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9003 was blocked because upstream `exp9000-invocation-provenance.provenance_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9005_natural_pilot_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because upstream `natural_pilot_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9011_arc_transfer_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The assessment was blocked because `arc_new_outcomes_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9012_v769_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports fourteen administrative outcomes, supported by "full_contract_rows", with failed prerequisites documented in "gate_check_summary" and scientific benefits left unmeasured.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Unchecked: referenced artifacts and reruns; assessment covers the supplied, truncated text.
