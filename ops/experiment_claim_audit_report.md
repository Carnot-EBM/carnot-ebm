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
| NO_CLAIM | 3 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7385_v648_decision_training.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The calibrated decision head demonstrated no registered value on the bounded evaluation.

## WHAT WOULD REFUTE IT
The primary arm would need to pass the complete registered value conjunction, especially a Brier-score improvement whose paired 95% confidence interval is below zero against both the logistic and training-prevalence controls.

## WAS THAT CHECKED
Yes. The artifact compares the preregistered primary arm against both controls using paired confidence intervals. It beat the logistic control on Brier score, but its interval against the training-prevalence control crossed zero, so the registered value gate failed. The oracle-defined labels do not create circularity for this null claim because the artifact makes no positive claim about verifier value.

## EVIDENCE
`honest_verdict`: `complete_null_calibrated_decision_head_no_registered_value`; `primary_value_arm`: `natural_prevalence_bernoulli_gibbs`; `choices_sealed_before_final_test`: `true`; `brier_ci_below_both_controls`: `false`; `calibration_value_score`: `0`; `passed`: `false`; `training_prevalence`; `brier_delta`; `ci95`: `[-0.0003270994119388504, 2.327706365149327e-06]`; `l2_logistic_calibration`; `ci95`: `[-7.488123656868094e-05, -1.5393199902023303e-05]`; `verdict_class`: `null`; `verifier_is_oracle`: `true`.

## RECOMMENDATION
KEEP

## experiment_7386_v648_online_decisions.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7387_decision_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a blocked-gate receipt and reports no experimental result or comparative claim.

## WAS THAT CHECKED
No; the experiment was blocked at the pre-gate, so the titled audit was not conducted.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"gate_check_summary": "gate-unsat(final): 3 of 6 gate(s) failed; first failure: exp7386-online-decisions.online_capture_complete_score (actual=0 == expected=1)"`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7388_proposal_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or value claim to falsify; this is a blocked pre-gate receipt, not an experiment result.

## WAS THAT CHECKED
No. The method was never evaluated; only prerequisite gates were checked.

## EVIDENCE
`"status"`: `"blocked"`; `"honest_verdict"`: `"blocked_gate_check_failed"`; `"blocked_at_layer"`: `"conductor_pre_gate"`; `"gate_check_summary"`: `"gate-unsat(final): 4 of 6 gate(s) failed; first failure: exp7383-canary-reducer.assignment_reducer_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_7391_arc_generalization.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked prerequisite check and makes no result or comparative claim about ARC generalization.

## WAS THAT CHECKED
No. The experiment stopped at the conductor pre-gate, so no ARC measurement rows were produced or evaluated.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"gate_check_summary": "gate-unsat(final): 3 of 3 gate(s) failed; first failure: exp7384-arc-invocation-boundary.arc_invocation_ready_score (actual=0 == expected=1)"`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7392_v648_ising_reduction.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed Ising reduction is disqualified because a required validation or safety gate failed.

## WHAT WOULD REFUTE IT
All terminal-blocking required-validation and safety checks passing—specifically, `capability_e2e_passed` being true—would refute the disqualification claim.

## WAS THAT CHECKED
Yes. The acceptance-gate summary and independent reduction explicitly checked the capability end-to-end gate; it failed, while safety passed, so the disjunctive headline remains true.

## EVIDENCE
`"honest_verdict": "complete_disqualified_ising_reduction_validation_or_safety_failure"`; `"verdict_class": "disqualified"`; `"check": "capability_e2e_passed"`; `"observed": false`; `"passed": false`; `"terminal_blocking": true`; `"required_validation_passed": false`; `"current_safety_passed": true`; `"promotion_score": 0`; `"prospective_pass_claimed": false`

## RECOMMENDATION
KEEP

## experiment_7393_v648_hardware_placement.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Hardware placement remains blocked because no qualifying GateMate physical-state change was recorded after Exp6559, and incomplete service-cost measurements support no acceleration claim.

## WHAT WOULD REFUTE IT
A qualifying operator-authored, dated GateMate cable, port, board, power, JTAG, or DirtyJTAG change after Exp6559—or complete measured service costs demonstrating current hardware acceleration—would refute the corresponding headline conclusions.

## WAS THAT CHECKED
Yes. The artifact searched the specified provenance for a qualifying physical-state receipt, checked all three placement inputs for a complete service boundary, and explicitly withheld hardware-readiness, hardware-value, and speed claims. The oracle relationship is disclosed, but the claim concerns recorded inputs and blocking status rather than the verifier’s added value.

## EVIDENCE
`"changed_state_receipt": null`; `"accepted_receipt_count": 0`; `"selected_source_path": null`; `"passed": false`; `"complete_service_rows": 0`; `"hardware_value_score": 0`; `"hardware_ready_score": 0`; `"new_hardware_runs_attempted": 0`; `"assumed_device_rate_is_measured": false`; `"complete measured service denominator is unavailable"`; `"three board dispositions are complete and incomplete stage costs create no speed claim"`; `"verifier_is_oracle": true`

## RECOMMENDATION
KEEP

## experiment_7394_v648_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
All fourteen V648 tasks were accounted for, but required science failed because online learning was disqualified on validation and safety while the decision audit and proof-memory measurement and audit were pre-gated.

## WHAT WOULD REFUTE IT
Fewer than fourteen accounted dispositions, a valid online-learning result that passed required validation and safety, or evidence that the decision audit or proof-memory measurement and audit actually ran and produced eligible results.

## WAS THAT CHECKED
Yes. The artifact checks the fourteen-disposition count, reports the online arm’s expected and observed gate states, and identifies the audit and proof-memory sources as blocked pre-gate records. These checks could have recorded passing or completed outcomes instead.

## EVIDENCE
`"fourteen_dispositions"`, `"expected": 14`, `"observed": 14`, `"passed": true`, `"required_science_complete"`, `"expected": 1`, `"observed": 0`, `"passed": false`, `"required_online_measurement"`, `"class": "disqualified"`, `"flag": true`, `"score": 0`, `"decision_audit_complete_score"`, `"proof_learning_capture_complete_score"`, `"proof_audit_complete_score"`, `"verdict_class": "blocked"`, `"source_kind": "conductor_pre_gate_artifact"`, `"source_kind": "conductor_log_record"`, `"scientific_value_score": 0`, `"promotion_score": 0`

## RECOMMENDATION
KEEP
