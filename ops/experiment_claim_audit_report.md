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

## experiment_7632_fit_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this artifact is a scaffolding/receipt artifact recording that the experiment was blocked prior to execution, making no comparative, empirical, or performance claim.

## WAS THAT CHECKED
No; the experiment was never executed, having been halted before launch at the pre-gate validation layer.

## EVIDENCE
`"schema"`
`"blocked_gate_check_v1"`
`"status"`
`"blocked"`
`"honest_verdict"`
`"blocked_gate_check_failed"`
`"duration_s"`
`0.0`
`"blocked_at_layer"`
`"conductor_pre_gate"`
`"gate_check_summary"`
`"gate-unsat(final): 4 of 9 gate(s) failed; first failure: exp7631-schema-pilot.evidence_transport_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_7633_online_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this artifact is a pre-execution gate-check receipt recording an unexecuted run, there is no substantive or comparative claim to refute. A demonstration that upstream dependencies had in fact met their gating criteria (e.g., `evidence_transport_ready_score` evaluating to `1`) would contradict the diagnostic trigger, but no empirical hypothesis was formulated or tested.

## WAS THAT CHECKED
No. The execution was halted at `conductor_pre_gate` before any experimental condition or comparator arm was run.

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
`gate-unsat(final): 4 of 9 gate(s) failed; first failure: exp7631-schema-pilot.evidence_transport_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_7634_evaluation_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no evaluation claim to falsify; this is a record of a blocked gate.

## WAS THAT CHECKED
No evaluation was run. The artifact records upstream gate checks and a block at the conductor pre-gate.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_field": "evidence_transport_ready_score"`; `"failed_expected": 1`; `"failed_observed": 0`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7638_v666_evidence_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no scientific benefit claim to refute. Eligible outputs from the six scientific producers, with raw evaluation rows, would contradict the artifact’s blocked status.

## WAS THAT CHECKED
Yes, for the blocked status: the producer eligibility gate checked all six and found none eligible. The artifact contains no raw rows, so it gave no scientific benefit claim a chance to fail.

## EVIDENCE
`positive_claim` `false`; `honest_verdict` `complete_blocked_v666_scientific_producers_unavailable`; `failed_count` `6`; `rows` `[]`; `observed` `0`; `audited_evidence_benefit_score` `null`.

## RECOMMENDATION
KEEP

## experiment_7639_v666_arc_goal_dedup.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7640_arc_wrapper_generalization.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or empirical claim is asserted; the artifact is a gate-check receipt recording that the experiment was prevented from executing due to upstream gate failures.

## WAS THAT CHECKED
No. The experiment was blocked at the pre-gate stage before any experimental trials, measurements, or evaluations were executed.

## EVIDENCE
`schema`: `"blocked_gate_check_v1"`
`status`: `"blocked"`
`honest_verdict`: `"blocked_gate_check_failed"`
`duration_s`: `0.0`
`blocked_at_layer`: `"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7641_v666_native_consumer.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The direct PyO3 native consumer package integration is functional and contract-ready with verified integration units and no claimed model execution or downstream benefit (an honest null).

## WHAT WOULD REFUTE IT
Any failure among the integration units (`failed_units > 0`, `native_consumer_ready_score` not equaling 1, or `passed` being false on contract conditions), a runtime failure/crash during PyO3 integration, or an unearned positive claim of probability benefit, cost benefit, or speedup while no model was invoked or benchmarked.

## WAS THAT CHECKED
Yes; contract readiness was tested across 12 independent units in `integration_rows` (all passing with 0 failures), and acceptance gates explicitly evaluated and reported `false` for `new_probability_benefit_measured` and `new_total_cost_benefit_measured`, properly constraining `honest_verdict` to `complete_null_native_consumer_ready`.

## EVIDENCE
- `honest_verdict`: `complete_null_native_consumer_ready`
- `native_consumer_ready_score`: `1`
- `verdict_class`: `null`
- `integration_rows_complete`: `true`
- `failed_units`: `0`
- `passed_units`: `12`
- `independent_units`: `12`
- `new_probability_benefit_measured`: `false`
- `new_total_cost_benefit_measured`: `false`
- `new_speed_claim`: `false`
- `model_invoked`: `false`
- `production_defaults_changed`: `false`
- `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_7642_v666_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Milestone V666 is blocked from publication and roadmap activation due to unready upstream conductor pre-gates and missing required external evidence.

## WHAT WOULD REFUTE IT
An observation in the artifact's own upstream checks showing that conductor pre-gates passed (`evidence_transport_ready_score` of 1 across upstream producers rather than 0) and that external scientific source groups were observed rather than missing (`observed` > 0 and `missing_external` == 0).

## WAS THAT CHECKED
Yes; checked in `gate_check_summary` (recording 9 failed conductor pre-gate checks where `evidence_transport_ready_score` was 0 instead of 1), `sample_size_budget` (recording 0 observed and 280 missing external units across scientific branches), and `acceptance_gate_results` (where freshness, readiness, probability_benefit, retention, and utility all failed).

## EVIDENCE
`"honest_verdict"`: `"complete_blocked_required_v666_external_evidence"`
`"verdict_class"`: `"blocked"`
`"status"`: `"complete"`
`"gate_check_summary"`: `"passed"`: `false`, `"failed_count"`: `9`
`"first_failure"`: `"check"`: `"conductor_pre_gate"`, `"field"`: `"evidence_transport_ready_score"`, `"expected"`: `1`, `"observed"`: `0`, `"passed"`: `false`
`"acceptance_gate_results"`:
`"freshness"`: `"observed"`: `false`, `"passed"`: `false`
`"readiness"`: `"observed"`: `false`, `"passed"`: `false`
`"probability_benefit"`: `"observed"`: `null`, `"passed"`: `false`
`"retention"`: `"observed"`: `null`, `"passed"`: `false`
`"utility"`: `"observed"`: `null`, `"passed"`: `false`
`"validity"`: `"observed"`: `true`, `"passed"`: `true`
`"scientific_source_groups"`: `"missing_external"`: `120`, `"observed"`: `0`
`"authorizes_submission"`: `false`
`"publication_performed"`: `false`
`"roadmap_activation_performed"`: `false`
`"model_invoked"`: `false`
`"verifier_is_oracle"`: `false`

## RECOMMENDATION
KEEP
