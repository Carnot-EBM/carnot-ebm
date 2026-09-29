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
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7839_v681_intervention_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to refute. If intervention benefit were claimed, completed rows showing no selective difference between the intervention arms would refute it.

## WAS THAT CHECKED
No. No intervention rows were started or completed.

## EVIDENCE
`claim_scope`: `scripted fixture conformance; exposed development labels only; no fresh generalization`; `readiness`: `false`; `started`: `0`; `completed`: `0`; `verdict_class`: `disqualified`.

## RECOMMENDATION
KEEP

## experiment_7845_v681_arc_supervisor_delta.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7847_v681_hardware_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative headline claim to refute. A qualified current board run showing whole-service benefit would challenge the recorded decision to defer.

## WAS THAT CHECKED
No. The artifact reports no current hardware execution, no completed runs, and no qualified service measurement.

## EVIDENCE
`hardware_advantage`: `unmeasured`; `acquisition_relevance`: `defer: no measured board whole-service benefit`; `current_hardware_execution`: `false`; `completed`: `false`; `whole_service_ms`: `null`; `qualified`: `false`; `status`: `missing`; `verdict_class`: `disqualified`.

## RECOMMENDATION
KEEP

## experiment_7848_v681_length_shortcut.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The length shortcut did not qualify on the exposed development data, and the run was disqualified by required validation failures.

## WHAT WOULD REFUTE IT
A cost gain meeting the stated threshold, a positive lower confidence bound for both cost and Brier gain, and passing required validation would refute that qualification result.

## WAS THAT CHECKED
Yes. The 64 evaluation families were compared with the prevalence baseline, bootstrap bounds were reported, and required validation was run. The artifact makes no fresh generalization claim.

## EVIDENCE
`claim_scope`: `exposed_human_labeled_development_only; no fresh generalization`; `cost_gain_over_prevalence`: `mean` `0.0`, `lower95` `0.0`; `brier_gain_over_prevalence`: `lower95` `-0.0155699037924997`; `required_validation_failures`: `affected_pytest`, `changed_coverage`; `validity`: `false`.

## RECOMMENDATION
KEEP

## experiment_7849_v681_independent_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
No comparative benefit claim is made; the audit reports that the required V681 science is blocked.

## WHAT WOULD REFUTE IT
A completed, eligible upstream science row with qualified evidence would refute the blocked status.

## WAS THAT CHECKED
Yes, for upstream eligibility: all eight branch rows were excluded. No qualified benefit comparison was available to check.

## EVIDENCE
`"honest_verdict": "complete_blocked_required_v681_science"`; `"independent_evidence_ready_score": 0`; `"qualified_science": null`; `"eligible": 0`; `"excluded": 8`; `"verdict_class": "blocked"`

## RECOMMENDATION
KEEP

## experiment_7850_v681_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The v681 milestone is blocked because required current evidence is not qualified.

## WHAT WOULD REFUTE IT
Required v681 evidence passing the current eligibility and validation checks, with a positive readiness result, while the artifact still declared the milestone blocked.

## WAS THAT CHECKED
Yes. The upstream gate checks and task accounting report failed prerequisites and no eligible independent evidence. The separate publication result is scoped to legacy G1–G4.

## EVIDENCE
`honest_verdict`: `complete_blocked_required_v681_evidence`; `milestone_evidence_ready_score`: `0`; `validity`: `false`; `eligible`: `0`; `independent_n`: `0`; `publication`: `legacy_G1_G4_only`; `activation`: `false`.

## RECOMMENDATION
KEEP

## experiment_7840_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact is a pre-execution gate receipt recording that the run was aborted before start, asserting no scientific, comparative, or performance claims.

## WAS THAT CHECKED
No; the experiment halted at the conductor pre-gate phase and never executed, so no candidate method or hypothesis was evaluated.

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

## RECOMMENDATION
KEEP

## experiment_7842_qwen_counter_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no result claim to refute. A future positive claim about source sensitivity could be refuted by matched-deletion measurements showing no sensitivity.

## WAS THAT CHECKED
No. The run was blocked before measurement.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP
