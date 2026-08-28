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
| NO_CLAIM | 7 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6685_autocorrelation_schedule_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies because the artifact reports no comparative outcome for the schedule A/B experiment.

## WAS THAT CHECKED
No. The experiment was blocked during the pre-gate check, so no method, rival, success criterion, or result rows were evaluated.

## EVIDENCE
`status` `blocked` `honest_verdict` `blocked_gate_check_failed` `failed_field` `torx_factor_parity_ready` `failed_observed` `false` `blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6687_v582_branch_synthesis.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6690_exact_planning_fixture_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No substantive result claim exists to falsify; this is only a blocked-gate receipt.

## WAS THAT CHECKED
No; the experiment did not run because its sole prerequisite gate failed.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6692_structural_plan_energy.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no substantive claim to falsify; the artifact is only a blocked-run receipt and contains no method results or comparison.

## WAS THAT CHECKED
No. Execution stopped at `conductor_pre_gate` because the required upstream artifact was unavailable, so no candidate, oracle, rival baseline, or scored rows were evaluated.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"blocked_reason": "upstream artifact not found for task id 'exp6691-sota-planning-proposal-corpus'"`; `"actual": null`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6694_energy_backtracking_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No substantive claim is made to falsify; the artifact is only a blocked-gate receipt.

## WAS THAT CHECKED
No; the experiment never reached execution because its upstream gate failed.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_observed": null`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6696_prequential_online_energy_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports only a blocked prerequisite gate and contains no A/B outcome to falsify.

## WAS THAT CHECKED
No; execution stopped at the pre-gate, before comparative data could be produced.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6702_exact_planning_fixture_recovery.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or capability claim is made; for the narrower fixture-readiness statement, any failed required check, undetected seeded mutation, invalid unit row, or inconsistent aggregate would refute readiness.

## WAS THAT CHECKED
Yes for fixture readiness: aggregate checks, row recomputation, metamorphic tests, and six mutation-detection rows could fail. No method-value comparison was checked because none was claimed.

## EVIDENCE
`honest_verdict` `complete: exact finite-horizon planning fixture ready` `verdict_class` `null` `planning_fixture_ready` `true` `failed_checks` `[]` `expected_detection` `true` `observed_detection` `true` `pass_state` `true` `inference_substrate` `cpu_exact_dynamic_programming_no_llm`

## RECOMMENDATION
KEEP

## experiment_6705_structural_plan_energy.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifying observation applies because the artifact reports only a blocked gate check, not an experimental result or comparative claim.

## WAS THAT CHECKED
No; the experiment stopped at `conductor_pre_gate` because its upstream artifact was unavailable.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"duration_s": 0.0`, `"failed_observed": null`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP
