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

## experiment_6678_constraint_family_stream.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The exact constraint-family stream fixture is not ready because its required test gate failed.

## WHAT WOULD REFUTE IT
A recomputed readiness value of true, with every aggregate check—including the test gate—true and all required test commands successful.

## WAS THAT CHECKED
Yes, in `aggregate_row_recomputation`, `gate_check_summary`, and the individual `tests_run` receipts; the test gate failed, so the refutation did not occur.

## EVIDENCE
`honest_verdict` = `blocked_fixture_gate: exact constraint-family stream is not ready`; `constraint_family_stream_ready` = `false`; `ready` = `false`; `tests` = `false`; `failed_check` = `tests`; `observed_value` = `false`; `status` = `blocked_fixture_gate`; `exit_code` = `3`; `summary` = `failure`; `inference_substrate` = `cpu_stream_fixture_and_exact_checkers_no_llm`; `verifier_is_oracle` = `true`.

## RECOMMENDATION
KEEP

## experiment_6679_prequential_cross_family_csl_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact makes no comparative A/B claim. A completed run with outcome rows would be required before a substantive claim could be falsified.

## WAS THAT CHECKED
No. Execution stopped at the pre-gate because the required upstream readiness condition failed; no method, rival, or success criterion was evaluated.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_field": "constraint_family_stream_ready"`, `"failed_observed": false`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6681_arc_post_redirect_outcomes.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no solve or comparative-value claim to falsify; treating the transport receipt as a limited operational assertion, it would fail if any applied redirect lacked exactly one live, identity-joined next-outcome row.

## WAS THAT CHECKED
Yes. The aggregate recomputation checks redirect count, eligible outcome count, and exact joins; the gate checks one outcome per redirect; lineage attacks test dropped, duplicated, and reordered records.

## EVIDENCE
`honest_verdict`: `complete: every applied held-family redirect has one exact live next-outcome row (30 eligible); this is transport evidence with no solve claim`; `solve_claim_scope`: `none`; `verdict_class`: `null`; `eligible_redirect_outcome_rows`: `30`; `all_redirects_exactly_joined`: `true`; `one_outcome_per_redirect`: `true`; `all_attacks_passed`: `true`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_6682_arc_held_family_supervisor_ab.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The held-family supervisor A/B is partial because verification failed and does not establish transition benefit or a solve.

## WHAT WOULD REFUTE IT
A strict positive supervisor-on versus supervisor-off transition-utility interval on valid matched episodes, or a passed verification gate supporting a completed positive result.

## WAS THAT CHECKED
Yes for transition benefit: the matched off/on arm could differ, and the off arm actually won three episodes while the supervisor-on arm won none. The verification gate was also checked and failed. Forbidden-action benefit had no headroom, but the artifact makes no positive claim from it. A solve was not targeted or claimed.

## EVIDENCE
`honest_verdict`: `blocked: held-family supervisor A/B is partial because verification_failure; no transition benefit or solve is claimed`; `arc_supervisor_ab_ready`: `false`; `failed_check`: `verification_failure`; `transition_utility_summary`; `delta`: `-0.3333333333333333`; `lower`: `-0.6666666666666666`; `upper`: `0.0`; `losses`: `3`; `wins`: `0`; `off_total`: `3.0`; `on_total`: `0.0`; `false_intervention_count`: `9`; `solve_rate_claimed`: `false`; `solve_claim_scope`: `none`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_6683_ising_reference_scope_receipt.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or added-value claim to falsify. The narrower operational readiness receipt would be refuted by a supported fixture whose exact outputs exceeded the stated tolerances, an invalid decomposition, an incorrect rejection, or a failed task-owned gate.

## WAS THAT CHECKED
Yes, for operational readiness: retained probability, marginal, correlation, decomposition, attack, rejection, and owned-test rows could fail and were reduced into readiness. No independent value comparison was checked or claimed.

## EVIDENCE
`"verdict_class": null`, `"verifier_is_oracle": true`, `"inference_substrate": "cpu_bounded_treewidth_exact_inference_no_llm"`, `"honest_verdict": "complete: bounded-treewidth exact Ising reference is ready under task-owned evidence"`, `"ready": true`, `"global_suite_in_reducer": false`, `"gating": false`, `"readiness_influence": false`, `"passed": true`

## RECOMMENDATION
KEEP

## experiment_6684_torx_typed_factor_parity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Installed Torx CPU parity did not pass every required check.

## WHAT WOULD REFUTE IT
Every required parity and gate check passing, including the applicable end-to-end check returning exit code 0, with parity readiness true.

## WAS THAT CHECKED
Yes. The applicable end-to-end gate expected exit code 0 but returned 1; aggregate readiness and Torx factor parity readiness were consequently false.

## EVIDENCE
`honest_verdict` = `blocked_parity_check_failed: installed Torx CPU parity did not pass every row`; `check_id` = `applicable_e2e`; `exit_code` = `1`; `passed` = `false`; `torx_factor_parity_ready` = `false`; `failed_check_count` = `1`

## RECOMMENDATION
KEEP

## experiment_6685_autocorrelation_schedule_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no asserted result to falsify; an implied claim that the autocorrelation-aware schedule improves outcomes would be refuted by a serious baseline tying or outperforming it on valid held-out rows.

## WAS THAT CHECKED
No. The experiment stopped at the upstream gate and contains no A/B observations or comparative metrics.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"blocked_reason": "actual=False in expected=[True]"`, `"failed_upstream": "exp6684-torx-typed-factor-parity"`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6687_v582_branch_synthesis.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V582 produced no pooled success: three branches were blocked, execution integrity was null, and the only outcome branch showed adverse supervisor utility.

## WHAT WOULD REFUTE IT
A completed branch showing scientific success, or live ARC rows showing positive utility, wins without false interventions, or a solve would refute the headline.

## WAS THAT CHECKED
Yes. All five branch rows were recomputed, and the live ARC arm directly measured utility, wins, losses, false interventions, and solve status. Missing branches remained explicitly null rather than being read as zero. Oracle-defined checks were not promoted into an added-value claim.

## EVIDENCE
`"all_branch_rows_recomputed": true`; `"pooled_success_claim": false`; `"positive": 0`; `"blocked": 3`; `"null": 1`; `"partial": 1`; `"delta": -0.3333333333333333`; `"wins": 0`; `"losses": 3`; `"false_intervention"`; `"count": 9`; `"solve_claim": false`; `"verification_gate_passed": false`; `"branch metric remains null; absence is not a zero"`; `"there is no pooled success claim"`

## RECOMMENDATION
KEEP
