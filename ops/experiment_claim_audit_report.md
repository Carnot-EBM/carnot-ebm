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
| NO_CLAIM | 8 |

## experiment_8318_v718_contract_replay.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is an administrative replay and qualification receipt, not a claim of comparative benefit. Oracle agreement cannot establish added value, but this artifact grants no scientific benefit or generalization credit.

## WAS THAT CHECKED
No comparative scientific claim was tested; H1 and H2 remain unmeasured. Administrative failure had a real chance to occur and did: the artifact records a reduction mismatch and an unresolved false-zero finding. These failures are not presented as scientific success.

## EVIDENCE
- `acceptance_gates`: `scientific_benefit` is `false`.
- `generalized_learning_benefit_score` is `0`.
- Historical `H1` and `H2`: `status` is `blocked_unmeasured`; `statistics` is `null`.
- `first_reduction_mismatch` identifies `gate_check_summary[15].hash`.
- The second adversarial disposition has `recomputed` = `false`, `resolved` = `false`, and `passed` = `false`.

## RECOMMENDATION
KEEP

## experiment_8319_local_evidence_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation showing substantive experimental results or demonstrating that the upstream gate check dependency was satisfied.

## WAS THAT CHECKED
No. The artifact is a pre-execution gate failure receipt; the experiment never executed.

## EVIDENCE
`schema`: `blocked_gate_check_v1`
`status`: `blocked`
`duration_s`: `0.0`
`honest_verdict`: `blocked_gate_check_failed`
`blocked_at_layer`: `conductor_pre_gate`
`gate_check_summary`: `gate-unsat(final): 1 of 1 gate(s) failed; first failure: exp8318-contract-replay.history_reader_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_8320_sentence_spline_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Any empirical claim regarding sentence spline fitting would require the experiment to execute; refuting the recorded receipt of gate failure would require observing that the upstream gate condition was actually satisfied (`"cached_support_ready_score"` equal to `1`).

## WAS THAT CHECKED
No; the experiment was aborted prior to execution at `"conductor_pre_gate"`.

## EVIDENCE
`"schema"`: `"blocked_gate_check_v1"`
`"status"`: `"blocked"`
`"honest_verdict"`: `"blocked_gate_check_failed"`
`"duration_s"`: `0.0`
`"blocked_at_layer"`: `"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_8326_runtime_reader_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or scientific claim is made to refute; the artifact is a gate-check receipt recording that upstream qualification failed and blocked experiment execution before any trial took place. Within the scope of a gate receipt, observing that the upstream gate check actually satisfied the condition (`history_reader_ready_score` equal to 1 or `passed` being true) while being logged as blocked would refute the receipt's failure record.

## WAS THAT CHECKED
No comparative hypothesis or experimental condition was checked because the run halted at the conductor pre-gate layer; the artifact evaluated only the upstream readiness condition `exp8318-contract-replay.history_reader_ready_score == 1`, which failed.

## EVIDENCE
`schema`: `blocked_gate_check_v1`
`status`: `blocked`
`honest_verdict`: `blocked_gate_check_failed`
`blocked_at_layer`: `conductor_pre_gate`
`duration_s`: `0.0`
`failed_upstream`: `exp8318-contract-replay`
`failed_field`: `history_reader_ready_score`
`failed_expected`: `1`
`failed_observed`: `0`

## RECOMMENDATION
KEEP

## experiment_8328_v718_arc_supervisor_frontier.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no supervisor-benefit claim to falsify. An authenticated new supervisor outcome within the stated frontier would contradict the receipt’s zero-outcome record, but would not itself establish comparative benefit.

## WAS THAT CHECKED
No comparative benefit test was conducted. The recorded phase authenticates and inspects upstream artifacts; current executions and outcome rows are empty. The artifact reports disqualification rather than a positive result or an evaluated null.

## EVIDENCE
`honest_verdict`: `complete_disqualified_supervisor_frontier`; `phase`: `authenticate_and_inspect`; `current_game_execution_count`: `0`; `current_model_invocation_count`: `0`; `new_outcome_count`: `0`; `rows`: `[]`; `selection_recommendations`: `[]`; `required_checks_passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_8329_v718_kv260_workload_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit is asserted. An eligible, completed source-arm measurement would contradict the reported absence of qualified measurements, but there is no performance headline to falsify.

## WAS THAT CHECKED
No comparative performance test occurred. All six full/indexed source-arm rows are censored and ineligible, with no completed measurements or timings. This is a blocked qualification receipt; it claims neither verifier added value nor independent generalization.

## EVIDENCE

- `verdict_class`: `blocked`
- `accelerator_benefit`: `unproved_no_compatible_operation`
- `completed_count`: `0`
- `censored_count`: `6`
- `eligible`: `false`; `censored`: `true`
- `timing_rows`: `[]`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8330_v718_gatemate_change_ledger.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact is an administrative evidence ledger and documentation audit recording that GateMate hardware remains blocked and unchanged, asserting no comparative or empirical performance claim.

## WAS THAT CHECKED
No; refutation checks do not apply to an administrative change ledger and receipt artifact with no comparative hypothesis.

## EVIDENCE
- `claim_scope`: `"Evidence ledger; future physical preflight remains unexecuted"`
- `arm`: `"documentation_audit"`
- `honest_verdict`: `"complete_blocked_gatemate_physical_change"`
- `condition`: `"unchanged_or_missing_receipt"`
- `scientific_benefit_score`: `0`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`
- `model_invoked`: `false`
- `current_model_calls`: `0`
- `current_device_execution_count`: `0`

## RECOMMENDATION
KEEP

## experiment_8331_v718_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation refuting a claim would require an empirical claim of comparative superiority or performance effect to be asserted (such as non-null Brier degradation or cost gains for active arms over comparators); here, no substantive claim is asserted.

## WAS THAT CHECKED
No; execution was blocked and unmeasured, with zero game executions, zero model invocations, and empty result rows.

## EVIDENCE
`"status": "blocked_unmeasured"`
`"support": null`
`"statistics": null`
`"completed_count": null`
`"independent_science": false`
`"owned_validation": false`
`"required_checks_passed": false`
`"honest_verdict": "complete_disqualified_supervisor_frontier"`
`"solve_claims": []`
`"rows": []`

## RECOMMENDATION
KEEP
