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
| CLAIM_SUPPORTED | 3 |
| NO_CLAIM | 5 |

## experiment_7741_training_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental result to refute. This artifact records a blocked qualification gate.

## WAS THAT CHECKED
No experiment was run past the gate; the artifact reports two failed gate checks.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"gate_check_summary": "gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7740-sentence-label-protocol.sentence_protocol_ready_score (actual=0 == expected=1)"`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7742_v674_bank_qualification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bank completed its lifecycle protocol under the exact-verifier oracle.

## WHAT WOULD REFUTE IT
An eligible lifecycle case that failed, duplicate feedback that changed the bank twice, or a restart that failed to preserve exact state.

## WAS THAT CHECKED
Yes. The artifact reports lifecycle pass flags, exactly-once behavior, and restart parity. It does not claim decision benefit or fresh generalization.

## EVIDENCE
`complete_circular_positive_bank_lifecycle`; `beneficial_commit`; `harmful_rejection`; `passed`; `true`; `exactly_once`; `restart_exact_parity`; `decision_benefit`; `null`; `fresh_generalization_eligible`; `false`; `verifier_is_oracle`.

## RECOMMENDATION
KEEP

## experiment_7745_v674_qwen_localization.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Localized prompting yields no statistically significant decision benefit over direct prompting for Qwen3.8-27B-GGUF, representing a null pilot result.

## WHAT WOULD REFUTE IT
A statistically significant decision benefit favoring localized prompting over direct prompting, such as a paired bootstrap 95% interval for binary accuracy delta strictly excluding 0.0, or widespread non-zero accuracy improvements across families.

## WAS THAT CHECKED
Yes. It was checked across 24 paired families under `paired_family_results` and `acceptance_gate_results.decision_benefit`, where 23 of 24 families tied with a delta of 0 and the paired bootstrap 95% interval [0.0, 0.125] included 0.0.

## EVIDENCE
`"honest_verdict"`: `"complete_null_exposed_localization_pilot"`
`"verdict_class"`: `"null"`
`"activation"`: `false`
`"production_promotion"`: `false`
`"claim_scope"`: `"development_only"`
`"verifier_is_oracle"`: `false`
`"binary_accuracy_delta"`: `0.041666666666666664`
`"paired_bootstrap_95_interval"`
`0.0`
`0.125`
`"paired_n"`: `24`
`"sha256:b48c248419d6cd53a35fd49b7e4b09068437b12ca4f0d50b5adb1fa48880f974": 1`

## RECOMMENDATION
KEEP

## experiment_7747_v674_independent_evidence_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because the artifact is an audit receipt that asserts no comparative, model, or performance claim, there is no headline empirical claim to refute. To test and potentially refute a substantive claim, the artifact would require an experimental intervention, comparative arms, baseline rivals, or performance metrics, which are not present.

## WAS THAT CHECKED
No. The artifact evaluated no comparative hypotheses or rival baselines. It performed only administrative prerequisite checks on the presence and status of upstream files in `gate_check_summary` and `preconditions_checked`.

## EVIDENCE
- `schema`: `"carnot.exp7747.v674.independent_evidence_audit.v1"`
- `honest_verdict`: `"complete_blocked_required_v674_evidence"`
- `verdict_class`: `"blocked"`
- `inference_substrate`: `"aggregation_from_upstream_artifacts"`
- `model_invoked`: `false`
- `MODEL_SPECS`: `[]`
- `arm`: `"custody"`
- `fresh_generalization_eligible`: `false`
- `decision_benefit`: `null`
- `efficiency`: `null`
- `probability_quality`: `null`
- `readiness`: `null`
- `retention`: `null`

## RECOMMENDATION
KEEP

## experiment_7748_v674_arc_runner_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A claim that the total or organic arm adds value would be refuted by matched arms tying on actions and progress. The artifact makes no such claim.

## WAS THAT CHECKED
Yes. The matched fixture and SDK rows show ties, and the artifact reports failed readiness and unmeasured decision benefit.

## EVIDENCE
`honest_verdict`: `complete_disqualified_required_runner_validation`; `verdict_class`: `disqualified`; `decision_benefit`: `null`; `readiness`: `false`; `model_invoked`: `false`. The `off`, `total`, and `organic` rows have matching `actions_charged` and `peak_level` within each game.

## RECOMMENDATION
KEEP

## experiment_7749_arc_organic_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact is an operational gate-check receipt recording that the experiment was blocked prior to execution, so no empirical or comparative claim is made.

## WAS THAT CHECKED
No; execution was halted at the pre-gate layer, so no experimental hypothesis or measurement was evaluated.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`
`"status": "blocked"`
`"honest_verdict": "blocked_gate_check_failed"`
`"duration_s": 0.0`
`"blocked_at_layer": "conductor_pre_gate"`
`"gate_check_summary": "gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7748-arc-runner-qualification.organic_runner_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_7751_v674_hardware_continuity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The hardware continuity inventory is complete, while hardware advantage and scientific benefit remain unclaimed.

## WHAT WOULD REFUTE IT
An eligible row showing a qualified, measured hardware advantage that the null verdict ignored, or an inventory row lacking the evidence needed for its stated status.

## WAS THAT CHECKED
Yes, within the inventory’s scope. The rows identify eligible and excluded entries, preserve the missing service-cost input as a failed check, and show no measured service fraction. This was an aggregation, so it did not test a new hardware workload.

## EVIDENCE
`hardware_continuity_complete_score`: `1`; `honest_verdict`: `complete_null_hardware_continuity_only`; `hardware_advantage_claimed`: `false`; `eligible`: `2`; `excluded`: `4`; `measured_service_fraction`: `null`; `service_fractions.quadratic_ising_kernel`; `observed`: `missing`; `passed`: `false`; `hardware_operations_issued`: `[]`.

## RECOMMENDATION
KEEP

## experiment_7752_v674_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative result to refute. The reported blocked status would be contradicted if the required sources were eligible and the capstone findings were complete.

## WAS THAT CHECKED
Yes, for the blocked status: the required source checks failed and the scientific findings are null. No comparative value claim was tested here.

## EVIDENCE
`honest_verdict` = `complete_blocked_required_v674_evidence`; `verdict_class` = `blocked`; `capstone_complete_score` = `0`; `decision_value` = `null`; `required_source_eligible` shows `observed` = `false` for two sources and `observed` = `blocked` for another.

## RECOMMENDATION
KEEP
