# Experiment claim-refutation audit

One question per artifact: what would REFUTE the headline claim, and was that
checked? Fabrication is out of scope (adversarial_verify covers it); this audit
targets claims that are true by construction, circular, in-sample, baseline-weak,
or contradicted by their own rows.

This audit never edits an artifact and never blocks anything. It surfaces; the
operator decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity
guard rest on evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CLAIM_SUPPORTED | 4 |
| NO_CLAIM | 4 |

## experiment_7307_v642_batch_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. The artifact’s blocked/no-run characterization would be refuted by any model invocation, generation, replay, or completed comparative row in its own data.

## WAS THAT CHECKED
Yes. Invocation counts, comparative rows, replay status, model invocation, and inference status are explicitly recorded; all show that no qualifying computation occurred.

## EVIDENCE
`"methodology_note"`: `"This is a bounded transport canary with no source-verification value claim. The disqualified fixture stopped work before model resolution, loading, generation, or replay."`; `"status"`: `"blocked"`; `"verdict_class"`: `"blocked"`; `"model_invoked"`: `false`; `"inference_mode"`: `"not_invoked"`; `"generation_calls_attempted"`: `0`; `"complete_comparative_rows"`: `0`; `"rows"`: `[]`; `"per_call_rows"`: `[]`; `"status"`: `"not_attempted_external_block"`

## RECOMMENDATION
KEEP

## experiment_7308_batch_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked prerequisite gate and makes no measurement or comparative claim.

## WAS THAT CHECKED
No; the experiment stopped at the prerequisite gate before the titled measurement ran.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_observed": 0`, `"failed_expected": 1`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7310_v642_factor_prototype.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A prospective evaluation in which factor-local revision ties or loses to global or local reset on future and non-feedback error would refute an efficacy claim, but this artifact explicitly makes no such claim.

## WAS THAT CHECKED
No. The evaluation panel was sealed but not scored; only fixture-readiness checks and development rows were completed.

## EVIDENCE
`honest_verdict`: `complete_circular_positive: bounded factor-local revision fixture is ready; prospective efficacy remains unmeasured`; `verdict_class`: `circular_positive`; `prototype_readiness_depends_on_efficacy`: `false`; `evaluation_stream_arm_units_measured`: `0`; `evaluation_panel_sealed_not_scored`; `observed`: `24`; `model_invoked`: `false`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_7311_v642_factor_learning.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Prospective factor-local learning completed, but it failed the frozen recurrence-error and false-accept value gates.

## WHAT WOULD REFUTE IT
The claim would be refuted if the method showed no prospective learning, or if both named gates passed—recurrence-error confidence bounds at or below 0.02 in every drift stratum and the false-accept confidence bound at or below zero versus the serious local-reset baseline.

## WAS THAT CHECKED
Yes. Prospective capture, pre-label prediction, state changes, and changed later predictions were checked. The prespecified value gates directly compared the method with frozen and local-reset controls, and both named gates failed. Thus the null result had a real chance not to occur.

## EVIDENCE
`honest_verdict`: `complete_null: prospective factor-local learning completed but frozen value gates failed: recurrence_error_vs_frozen,false_accept_vs_local_reset`

`factor_capture_complete_score`: `1`

`factor_value_score`: `0`

`legitimate_later_changed_predictions`: `989`; `pass`: `true`

`recurrence_error_vs_frozen`: `each drift-stratum ci95_upper<=0.02`; `isolated_factor_changes`: `0.05419921875`; `overlapping_factor_changes`: `0.142578125`; `pass`: `false`

`false_accept_vs_local_reset`: `ci95_upper<=0`; `observed`: `0.003952752976190476`; `pass`: `false`

`later_prediction_was_pre_label`: `true`

`chronology_violations`: `count`: `0`

`verdict_class`: `Exactly positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial; unchanged external failure is blocked.`

## RECOMMENDATION
KEEP

## experiment_7312_v642_factor_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Retained factor witnesses improved some future errors but failed the complete promotion contract, warranting retirement of the tested revision mechanism.

## WHAT WOULD REFUTE IT
All preregistered promotion gates passing—especially no recurrence harm versus frozen and no excess false accepts versus the serious local-reset baseline—would refute the failure-and-retirement claim.

## WAS THAT CHECKED
Yes. The acceptance gates directly compared the method with global reset, local reset without retained witnesses, and frozen warmup across 24 completed, uncensored streams; two required gates failed.

## EVIDENCE
`"verdict_class": "null"`; `"factor_promotion_score": 0`; `"retirement_triggered": true`; `"false_accept_vs_local_reset"` with `"pass": false`; `"recurrence_error_vs_frozen"` with `"pass": false`; `"completed_stream_count": 24`; `"censored_stream_count": 0`; `"outcome_based_extension": false`; `"headline_llm_accuracy_gain_established": false`

## RECOMMENDATION
KEEP

## experiment_7313_v642_cost_envelope.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
No point-identified replaceable group-16 cost is large enough to justify another same-acknowledgment implementation.

## WHAT WOULD REFUTE IT
A valid same-acknowledgment group-16 result whose measured exclusive replaceable lower CI95 met or exceeded the required warm-savings fraction of 0.02944416299535435, thereby making a specific future technique warranted.

## WAS THAT CHECKED
Yes. The explicit warrant gate compared the measured identifiable replaceable lower CI95 of 0.024195805711275806 against the required warm-savings fraction of 0.02944416299535435 and failed. All 24 planned units completed without censoring. The criterion and measured cost come from distinct quantities, so the refuting outcome was possible. The oracle flag does not circularly establish a positive verifier-value claim here; this is an execution-grounded null.

## EVIDENCE
`honest_verdict` = `complete_null: no point-identified replaceable group-16 cost is large enough to warrant another same-acknowledgment implementation`; `specific_future_technique_warrant`; `expected` = `measured exclusive replaceable lower bound closes warm gap`; `observed` = `false`; `passed` = `false`; `measured_identifiable_replaceable_lower_ci95` = `0.024195805711275806`; `required_warm_savings_fraction` = `0.02944416299535435`; `warranted` = `false`; `serialization_identifiability` = `bounded_not_point_identified`; `completed_paired_seed_arrival_units` = `24`; `censored_paired_seed_arrival_units` = `0`; `verdict_class` = `null`

## RECOMMENDATION
KEEP

## experiment_7314_v642_board_continuity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact records authenticated board dispositions and prerequisites but makes no comparative, performance, generalization, or added-value claim.

## WAS THAT CHECKED
No; there is no headline value claim requiring a rival or falsification arm. The artifact only checks receipt integrity and disposition completeness.

## EVIDENCE
`inference_substrate`: `aggregation_from_upstream_artifacts`; `model_invoked`: `false`; `hardware_operations_issued`: `[]`; `availability_is_scientific_result`: `false`; `local_performance_claim`: `false`; `status`: `complete`.

## RECOMMENDATION
KEEP

## experiment_7315_v642_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Required V642 science is unavailable because the batch audit is absent, ARC and factor branches are null, and durability and board receipts do not establish efficacy, despite closure of all fourteen dispositions.

## WHAT WOULD REFUTE IT
Authenticated batch-audit evidence completing the blocked branch, a positive ARC or factor value score, durability or board evidence establishing scientific efficacy, or fewer than fourteen represented dispositions would falsify the claim.

## WAS THAT CHECKED
Yes. The acceptance gates, authenticated audit-score rows, five branch reductions, and capstone dimensions separately checked disposition closure, required evidence availability, and scientific value; the batch branch was allowed to be absent or positive rather than silently counted as zero.

## EVIDENCE
`"fourteen_task_dispositions"`; `"observed": 14`; `"passed": true`; `"batch_audit_complete_score"`; `"observed_value": null`; `"source_authenticated": false`; `"source_disposition_class": "absent"`; `"arc_tool_use_score"`; `"observed_value": 0`; `"factor_promotion_score"`; `"observed_value": 0`; `"positive_branch_count": 0`; `"required_science_complete": false`; `"positive_promoted": false`; `"verdict_class": "blocked"`; `"verdict_class": "null"`; `"verdict_class": "circular_positive"`; `"scientific_efficacy_positive": false`

## RECOMMENDATION
KEEP
