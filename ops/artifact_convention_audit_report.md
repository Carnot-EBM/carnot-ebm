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

## experiment_9096_v776_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment describes registry/frontier context only; the full headline claim is unavailable.

## WHAT IS MISSING
The artifact cuts off inside "authority_snapshots". The remainder containing any final verdict, comparative claim, or blocker diagnostic is unavailable. "activation_authorized": false alone does not establish a blocked verdict.

## THE CHECK A READER CANNOT DO
Does the omitted remainder assert a comparative result or report a blocked verdict with a reason?

Unchecked: the truncated remainder; classifying the complete artifact from this fragment risks error.

## experiment_9097_v776_human_authority.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Producer authentication, exposed label custody, retained roles, and sealed inputs are administratively qualified; no scientific benefit is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Referenced files were not inspected; this verdict covers only the supplied artifact and the two specified failures.

## experiment_9098_v776_embedding_runtime.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Administrative runtime capability on twelve fixed texts, with logits accepted and embeddings unqualified, without scientific benefit or an ARC solve claimed.

## WHAT IS MISSING
The supplied artifact ends inside `"fixture_rows"[0]."vector"`. Despite `"completed_count": 12`, only `"unit_id": "fixture-00"` is visible. Remaining rows and terminal verdict are unavailable; `"field_principles"` describes `"gate_check_summary"` but supplies no diagnostic values.

## THE CHECK A READER CANNOT DO
Do all twelve fixtures support `"acceptance_gates"` → `"actual_logits": true`?

The omitted rows and terminal diagnostics remain unchecked; their contents could change the classification.

## experiment_9099_v776_online_methods.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The coefficient methods pass authentication, causality, recovery, and validation gates, without establishing human learning benefit.

## WHAT IS MISSING
The artifact truncates inside `"gate_check_summary"`. `"fixture_metrics"` contains arm aggregates, but actual per-unit `"rows"` and `"honest_verdict"` values are not visible; `"field_principles"` only describes those fields.

## THE CHECK A READER CANNOT DO
Do per-unit fixture results support the passed `"positive_learning"` check?

The unseen remainder was not checked; it could contain the required evidence.

## experiment_9100_embeddings_a.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9100 was blocked because `exp9098-embedding-runtime.embeddings_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9101_embeddings_b.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Extraction was blocked because upstream `embeddings_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9107_v776_qwen_judge.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The run completed 450 frozen Qwen judgments but reports disqualification by artifact checks, with no comparative benefit claimed.

## WHAT IS MISSING
The failed check’s identifier and observed value explaining "honest_verdict": "complete_disqualified_artifact_checks" are not visible; "gate_check_summary" shows only passes.

## THE CHECK A READER CANNOT DO
Which artifact check caused disqualification, and what value did it observe?

The artifact cuts off inside "qid_rows"; the unreviewed remainder may contain that diagnostic.

## experiment_9109_v776_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Fourteen administrative outcomes were completed, with retained descriptive evidence and no new scientific benefit claim.

## WHAT IS MISSING
The supplied JSON cuts off inside the first entry of "confirmed_branch_results"; the remaining outcomes, their verdicts, and any "gate_check_summary" diagnostics are not shown, although "completed_count" is 14.

## THE CHECK A READER CANNOT DO
Can the fourteen reported completions be reconciled with fourteen documented branch outcomes?

The omitted remainder was not checked; it may contain the necessary evidence.
