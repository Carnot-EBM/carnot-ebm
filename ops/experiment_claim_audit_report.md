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

## experiment_7716_v672_qwen_semantic_pilot.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The pilot completed but was disqualified by required checks and was not activated.

## WHAT WOULD REFUTE IT
All required checks passing, with the pilot still labeled disqualified for failed checks.

## WAS THAT CHECKED
Yes. The artifact records the required command results; the changed module coverage report failed. It also records failed acceptance gates.

## EVIDENCE
`"honest_verdict": "complete_disqualified_required_checks"`; `"status": "complete"`; `"name": "changed_module_coverage_report"`; `"exit_code": 2`; `"passed": false`; `"activation": false`; `"production_promotion": false`

## RECOMMENDATION
KEEP

## experiment_7717_latent_evidence_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The gate receipt would be contradicted if its recorded gate results showed that every required gate passed.

## WAS THAT CHECKED
Yes. The evaluated gates include three failures, so the recorded blocked status is consistent with the rows.

## EVIDENCE
`"status": "blocked"`; `"gate_check_summary": "gate-unsat(final): 3 of 7 gate(s) failed; first failure: exp7715-natural-source-cohort.natural_cohort_ready_score (actual=0 == expected=1)"`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7719_v672_acquisition_qualification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The acquisition qualification is disqualified because required checks did not pass.

## WHAT WOULD REFUTE IT
A successful required-validation run, with the mandatory qualification gates passing, would refute the disqualification.

## WAS THAT CHECKED
Yes. The required-validation check ran and failed; the validity and readiness gates also failed.

## EVIDENCE
`honest_verdict`: `complete_disqualified_required_checks`; `required_validation`: `expected` `0`, `observed` `2`, `passed` `false`; `validity`: `passed` `false`; `readiness`: `passed` `false`; `acquisition_protocol_ready_score`: `0`.

## RECOMMENDATION
KEEP

## experiment_7721_v672_independent_evidence_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
If the required upstream artifacts were present and valid while the audit still reported them missing, that would refute its blocked disposition.

## WAS THAT CHECKED
Yes. The audit checked both required artifacts and recorded each as absent.

## EVIDENCE
`claim_eligibility_disposition` `blocked` `required_science_exists` `observed` `false` `independent_audit_complete_score` `0` `per_game_results` `[]`

## RECOMMENDATION
KEEP

## experiment_7722_v672_arc_evidence_recovery.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The recovered historical evidence remains disqualified and earns no new ARC solve credit.

## WHAT WOULD REFUTE IT
A historical game row showing an accepted engine and a credited solve, together with passing evidence gates, would contradict that claim.

## WAS THAT CHECKED
Yes. The artifact reports per-game results and acceptance gates; both games show zero accepted engines, no credited solve, and failed readiness. It makes no positive claim about verifier value or hidden-game generalization.

## EVIDENCE
`honest_verdict`: `complete_disqualified_terminal_reader`; `arc_evidence_ready_score`: `0`; `accepted_engines`: `0`; `new_solve_credit`: `false`; `readiness` `passed`: `false`; `hidden_games`: `0`; `paired_arms`: `0`

## RECOMMENDATION
KEEP

## experiment_7723_v672_native_qualification.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7724_complete_service_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this artifact is a pre-execution gate receipt recording a blocked run and makes no empirical, comparative, or performance claim.

## WAS THAT CHECKED
No; the experiment did not execute because prerequisite qualification gates failed, blocking execution before any service cost measurement could take place.

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
`gate-unsat(final): 3 of 3 gate(s) failed; first failure: exp7723-native-qualification.native_service_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_7725_v672_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The required V672 scientific evidence is blocked.

## WHAT WOULD REFUTE IT
Eligible results from all three required science producers, with the required gates passing and the authority contract matching, would refute the blocked verdict.

## WAS THAT CHECKED
Yes. The artifact checks producer eligibility, registered gates, and contract matching; each shows a barrier to V672 acceptance.

## EVIDENCE
`"honest_verdict": "complete_blocked_required_v672_evidence"`; `"eligible_required_producers": 0`; `"required_producers": 3`; `"required_science_eligible": false`; `"contract_match": false`; `"failed_count": 18`; `"passed": false`; `"verdict_class": "blocked"`.

## RECOMMENDATION
KEEP
