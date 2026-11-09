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

## experiment_8949_v765_matched_corpus_a.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment reports 160 completed responses with all listed acceptance gates passing.

## WHAT IS MISSING
The artifact ends mid-entry in "graph_identity_rows". Actual "honest_verdict", "status", "response_rows", and "rows" values are unavailable; these names appear only in "field_principles". The visible "gate_check_summary" is empty, but no blocked verdict is visible.

## THE CHECK A READER CANNOT DO
Does the complete artifact make a comparative or blocked claim with supporting rows or diagnostics?

The truncated remainder could change the classification; it was unavailable for review.

## experiment_8950_v765_matched_corpus_b.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports 160 completed responses and all listed acceptance gates passing.

## WHAT IS MISSING
The artifact cuts off inside "graph_identity_rows". Actual "honest_verdict", "status", and "verdict_class" values are absent; their descriptions in "field_principles" do not supply them. "gate_check_summary" is present but empty.

## THE CHECK A READER CANNOT DO
Does the missing remainder report a comparative result or a blocked outcome, and supply the corresponding per-unit rows or blocker diagnostic?

## experiment_8951_v765_matched_corpus_admission.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment marks all "acceptance_gates" true, but truncation leaves the headline claim unknown.

## WHAT IS MISSING
The artifact cuts off inside "corpus_manifest" at "unit_id": "matched-graph-077-"; its final verdict and any subsequent supporting evidence are unavailable. "class_support_rows" does contain counts and thresholds for the class-support check.

## THE CHECK A READER CANNOT DO
Does the omitted remainder make a comparative claim supported by per-unit outcomes or report a blocker with a diagnostic?

## experiment_8952_v765_matched_calibrated_decisions.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The completed calibration experiment failed to establish benefit under its three comparative acceptance gates.

## WHAT IS MISSING
The supplied artifact ends mid-string inside "preconditions_checked.references". Visible "coverage_rows", "descriptive_rows", and "paired_intervals" contain aggregates; "test_prediction_rows" appears only in "field_principles", without actual per-graph data shown.

## THE CHECK A READER CANNOT DO
Do the 80 graphs’ labels and paired arm predictions support the reported comparative results?

Unchecked: truncated contents; declaring per-unit rows absent would require an assumption.

## experiment_8953_v765_gradient_causal_streams.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The supplied JSON ends mid-entry in "candidate_gradient_rows". "acceptance_gates" records counts and scores, and "attempt_rows" records completion statuses, but no headline result or blocking verdict is visible.

## THE CHECK A READER CANNOT DO
Does the complete artifact assert a comparative result or blocked outcome and provide the corresponding evidence?

The omitted remainder was not checked; it could change the classification.

## experiment_8954_v765_gradient_benefit_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The tested gradient-replay configuration failed the required acceptance gates and should be retired.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Experiment not rerun; assessment limited to the supplied excerpt.

## experiment_8955_v765_gradient_complete_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt withholds deployment recommendation, recording “gradient_benefit_score” as 0 against a required 1.

## WHAT IS MISSING
The remainder of the artifact: it ends mid-“durability_rows”. “deployment_baseline_rows” provides per-seed baseline timings; corresponding trainable-arm timings and the final verdict are unavailable in this excerpt. “deployment_quality_gate” does record a diagnostic: “observed”: 0, “expected”: 1.

## THE CHECK A READER CANNOT DO
For each seed, how does gradient replay’s complete-process time compare with ordinary deployment?

The omitted remainder could not be checked; truncation prevents distinguishing missing evidence from evidence outside the excerpt.

## experiment_8956_v765_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The arc is suspended, with scientific benefit and upstream evidence qualification marked false.

## WHAT IS MISSING
The artifact cuts off inside "frozen_evaluation_manifest"; any subsequent verdict, diagnostics, or measured outcomes are unavailable. The visible "gate_check_summary" arrays are empty.

## THE CHECK A READER CANNOT DO
Does the omitted remainder explain the suspension? Omitted content was not checked; truncation prevents a definitive classification.
