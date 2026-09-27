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
| CLAIM_SUPPORTED | 2 |
| NO_CLAIM | 6 |

## experiment_7731_development_decisions.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None. The artifact is a pre-execution gate failure receipt indicating that upstream prerequisites were not met, so it makes no empirical, comparative, or performance claim that could be refuted.

## WAS THAT CHECKED
No. Execution was blocked at the pre-gate check before the experiment could execute or evaluate any hypothesis.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`
`"status": "blocked"`
`"honest_verdict": "blocked_gate_check_failed"`
`"duration_s": 0.0`
`"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7732_v673_causal_admission.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The experiment completed but was disqualified because a required validation check failed.

## WHAT WOULD REFUTE IT
All required validation checks passing would refute the stated reason for disqualification.

## WAS THAT CHECKED
Yes. The required coverage report ran and failed: it required 100% coverage and reported 99%.

## EVIDENCE
`honest_verdict`: `complete_disqualified_required_checks`; `verdict_class`: `disqualified`; `check`: `required_validation`; `name`: `changed_module_coverage_report`; `exit_code`: `2`; `passed`: `false`; `99%`.

## RECOMMENDATION
KEEP

## experiment_7733_continuous_set_learning.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a gate receipt, not a result about continuous set learning. Its blocked status would be contradicted if every evaluated gate had passed.

## WAS THAT CHECKED
Yes. The gate rows include failures, so the blocked status is consistent with the artifact’s own data. No learning outcome was measured here.

## EVIDENCE
`"status": "blocked"`; `"blocked_at_layer": "conductor_pre_gate"`; `"gate_check_summary": "gate-unsat(final): 4 of 6 gate(s) failed; first failure: exp7730-set-energy-fit.set_heads_ready_score (actual=0 == expected=1)"`; `"passed": false`

## RECOMMENDATION
KEEP

## experiment_7734_v673_independent_evidence_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The audit completed, but required v673 evidence remains blocked and does not support activation or fresh generalization.

## WHAT WOULD REFUTE IT
Both required sources passing the eligibility gate, with independent evidence eligible for the claimed use, would refute the blocked disposition.

## WAS THAT CHECKED
Yes. The required-source gate checked both sources and recorded failures; the eligibility and activation fields reflect the blocked result.

## EVIDENCE
`honest_verdict`: `complete_blocked_required_v673_evidence`; `Exp7731` and `Exp7733`: `eligible`: `false`, `state`: `pre_gate`; `required_source_eligible`: `observed`: `blocked_gate_check_failed`; `independent_online_eligible`: `false`; `independent_static_eligible`: `false`; `fresh_generalization_eligible`: `false`; `activation`: `false`.

## RECOMMENDATION
KEEP

## experiment_7735_v673_arc_organic_visits.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies. If the artifact claimed organic visits improved outcomes, a control arm that tied or beat it would refute that claim.

## WAS THAT CHECKED
The fixture arms tied, but the public game rows ended in policy exceptions. The artifact makes no comparative outcome claim from them.

## EVIDENCE
`honest_verdict`: `complete_disqualified_required_validation_or_runner`; `verdict_class`: `disqualified`; `solve_provenance`: `development_proxy_for_fixtures_no_game_level_solve_claim`; `censoring`: `policy_exception`.

## RECOMMENDATION
KEEP

## experiment_7736_arc_organic_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this artifact makes no empirical or comparative claim, as pipeline execution was halted prior to any measurement.

## WAS THAT CHECKED
No; the experiment was never run because upstream gate checks failed before execution could begin.

## EVIDENCE
`schema`
`blocked_gate_check_v1`
`status`
`blocked`
`honest_verdict`
`blocked_gate_check_failed`
`duration_s`
`0.0`
`blocked_at_layer`
`conductor_pre_gate`
`gate_check_summary`
`gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7735-arc-organic-visits.organic_runner_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_7737_set_service_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; this artifact is a gate-check execution receipt recording an unexecuted, blocked experiment rather than asserting an empirical or comparative claim. A refutation would only be possible if the experiment had run and asserted a substantive measurement or performance advantage.

## WAS THAT CHECKED
No; the experiment was never executed because upstream preconditions failed at the pre-gate layer, blocking execution before any measurement occurred.

## EVIDENCE
`schema`
`blocked_gate_check_v1`
`status`
`blocked`
`duration_s`
`0.0`
`honest_verdict`
`blocked_gate_check_failed`
`blocked_at_layer`
`conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7738_v673_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative method claim to refute. Three eligible required tasks and a ready acceptance gate would contradict the report’s blocked accounting conclusion.

## WAS THAT CHECKED
Yes, through the required-source gate checks and coverage count. The artifact reports zero eligible required tasks out of three.

## EVIDENCE
`honest_verdict` `complete_blocked_required_v673_evidence` `required` `3` `required_eligible` `0` `readiness` `false` `model_invoked` `false` `verdict_class` `blocked`

## RECOMMENDATION
KEEP
