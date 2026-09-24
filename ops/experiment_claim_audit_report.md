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
| CLAIM_SUPPORTED | 5 |
| NO_CLAIM | 3 |

## experiment_7620_evaluation_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Evidence that upstream gate dependencies were satisfied and that experimental evaluation arms were executed. Because this artifact is an operational pre-gate diagnostic record rather than a test of an empirical hypothesis, there is no comparative or methodological claim to falsify.

## WAS THAT CHECKED
no; execution halted at pre-gate verification prior to running any experimental evaluation.

## EVIDENCE
- `schema`: `"blocked_gate_check_v1"`
- `status`: `"blocked"`
- `honest_verdict`: `"blocked_gate_check_failed"`
- `duration_s`: `0.0`
- `blocked_at_layer`: `"conductor_pre_gate"`
- `gate_check_summary`: `"gate-unsat(final): 4 of 9 gate(s) failed; first failure: exp7617-schema-pilot.evidence_transport_ready_score (actual=0 == expected=1)"`

## RECOMMENDATION
KEEP

## experiment_7624_v665_evidence_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this artifact is an audit receipt that records blocked upstream dependencies rather than asserting a comparative hypothesis, there is no headline claim of benefit or capability to refute. A substantive scientific claim would be refuted if a method under test failed to beat an appropriate rival baseline on held-out evaluation rows.

## WAS THAT CHECKED
No. No comparative evaluation was checked because required scientific producers were unavailable, leaving all evaluation rows absent (`missing_external`).

## EVIDENCE
`positive_claim`: `false`
`honest_verdict`: `complete_blocked_v665_scientific_producers_unavailable`
`verdict_class`: `blocked`
`rows`: `[]`
`audited_evidence_benefit_score`: `null`
`audited_learning_benefit_score`: `null`
`fresh_confirmatory_claim_allowed`: `false`
`default_promotion_authorized`: `false`

## RECOMMENDATION
KEEP

## experiment_10012_gate_usefulness.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Gate usefulness was measured across ten informative windows with none unrecoverable; the artifact does not claim that any tested gate proved useful.

## WHAT WOULD REFUTE IT
The claim would fail if the usefulness labels were derived from the gate metrics, if fewer than ten eligible windows were informative, or if an unrecoverable window were counted as informative. Any positive gate-value claim would also be refuted by the strict gate tying reject-all or by every gate rejecting a useful candidate.

## WAS THAT CHECKED
Yes. Execution outcomes define labels separately from gate decisions, the two control categories total ten informative windows, and all windows are marked recoverable. The candidate-only tables give failure a real chance: every gate rejects the useful candidate, relaxed gates additionally accept negatives, and no separating gate is reported.

## EVIDENCE
`honest_verdict`: `complete_gate_usefulness_measured_10_informative_windows_0_unrecoverable`; `inference_substrate`: `verifier_ensemble_against_cached_candidates`; `informative_window_count_by_control_category`; `expert_live_planner`: `6`; `informative_by_registry_solver`: `4`; `state_recoverable`: `true`; `gate_decisions`; `labels_by_arm`; `real_level_up`; `gate_tables_useful_without_controls`; `n_pairs`: `20`; `accepted_and_positive`: `0`; `rejected_positive`: `1`; `live_exact_1.0`; `accepted_and_negative`: `0`; `masked_exact_0.75`; `accepted_and_negative`: `6`; `exploratory_separating_gates_live_scored`; `main_without_controls`: `[]`; `no-separation survives only as a tiny-cohort observation`

## RECOMMENDATION
KEEP

## experiment_7625_v665_arc_supervisor_transfer.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
No causal treatment benefit is established and no live policy defaults should be changed because zero actual supervisor firings occurred across the evaluated games.

## WHAT WOULD REFUTE IT
The headline claim would be refuted if upstream episode receipts recorded non-zero actual supervisor firings (`actual_firing_count > 0` or `applied_redirect_count > 0`) that produced a measurable positive treatment benefit (`helped_count > 0`), justifying policy refinement.

## WAS THAT CHECKED
Yes. The aggregation inspected 24 rows across 6 deduplicated game episodes (`dc22`, `ft09`, `g50t`, `sb26`, `sp80`, `su15`), verifying that all candidate redirects operated exclusively under shadow mode (`receipt_mode`: `"shadow"`) and confirming zero actual firings, zero applied redirects, and zero helped outcomes across all evaluated arms.

## EVIDENCE
- `"honest_verdict"`: `"complete_null_no_firings_nothing_to_refine"`
- `"verdict_class"`: `"null"`
- `"acceptance_gate_results"` -> `"benefit"`: `"condition"`: `"causal treatment benefit established"`, `"observed"`: `false`, `"result"`: `false`
- `"selection_recommendation"`: `"action"`: `"no_change"`, `"causal_claim"`: `false`, `"live_defaults_changed"`: `false`, `"reason"`: `"no_firings_nothing_to_refine"`
- `"redirect_counts"`: `"actual_firing_count"`: `0`, `"applied_redirect_count"`: `0`, `"censored_firing_count"`: `0`, `"helped_count"`: `0`, `"proposed_redirect_count"`: `20`, `"would_have_redirect_count"`: `20`
- `"receipt_mode"`: `"shadow"`
- `"solve_claim"`: `false`

## RECOMMENDATION
KEEP

## experiment_7626_v665_native_service.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The native service protocol is ready with verified three-arm parity and crash durability, while claiming no speed or scientific benefit.

## WHAT WOULD REFUTE IT
Any non-zero numerical or typed mismatch across arms (`numeric_mismatch_count` > 0, `typed_mismatch_count` > 0, or `three_arm_parity` observing `false`), loss of state or acknowledgment of corrupted state on crash (`prior_state_survived` observing `false`, `durable_reload_match` observing `false`, or `interrupted_write_acknowledged` observing `true`), or asserting a positive advantage while `speed_or_scientific_benefit` observed `false`.

## WAS THAT CHECKED
Yes; parity was evaluated across 30 requests in `parity_rows` and summarized in `parity_reduction`, crash and interrupted-write durability was evaluated in `durability_rows` and `durability_reduction`, and protocol readiness versus null benefit was evaluated across the gates in `acceptance_gate_results`.

## EVIDENCE
`"honest_verdict": "complete_null_native_service_protocol_ready"`
`"verdict_class": "null"`
`"speed_benefit_claimed": false`
`"scientific_benefit_claimed": false`
`"check": "three_arm_parity"`
`"check": "native_service_ready_score"`
`"check": "speed_or_scientific_benefit"`
`"numeric_mismatch_count": 0`
`"typed_mismatch_count": 0`
`"durable_reload_match": true`
`"prior_state_survived": true`
`"interrupted_write_acknowledged": false`
`"max_abs_delta": 1.1102230246251565e-16`

## RECOMMENDATION
KEEP

## experiment_7627_v665_native_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Direct native execution satisfies the fixed total-cost performance gate (achieving >= 1.10x geometric mean speedup over comparator arms with stratum lower 95% confidence bounds >= 0.95x, decision and state parity, and zero additional errors) under host CPU durable consumer timing.

## WHAT WOULD REFUTE IT
Any of the following observations in the paired benchmark rows or reduction: an equal-stratum geometric mean speedup ratio of comparator over direct native falling below 1.10 (or lower 95% bootstrap bound < 1.10), any individual stratum lower 95% bound falling below 0.95, any decision or state parity mismatch between direct native and comparators, or any extra native execution errors.

## WAS THAT CHECKED
Yes. Evaluated in `timing_reduction` across 120 three-arm paired blocks (360 rows across 4 cold/warm strata) measuring `direct_native` against both `python_inprocess` and `rust_jsonl`. That refutation was genuinely possible is confirmed by the separate ten-times cost target (`nfr_10x_total_cost`), which evaluated to 7.827x and failed.

## EVIDENCE
`"honest_verdict": "complete_positive_native_total_cost_gate_met"`
`"verdict_class": "positive"`
`"check": "fixed_total_cost_gate"`
`"condition": "geomean estimate/lower95 >=1.10, each lower95 >=0.95, parity, no errors"`
`"observed": 1`
`"passed": true`
`"estimate": 7.827002746229495`
`"lower95": 7.284307759345477`
`"estimate": 2.6492877005903535`
`"lower95": 2.4659437842692813`
`"minimum_primary_stratum_lower95": 2.6682914199518204`
`"parity_complete": true`
`"extra_native_errors": 0`
`"nfr_10x_met": false`
`"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7628_v665_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V665 capstone milestone is complete and blocked by missing required external scientific evidence and unfulfilled GPU readiness prerequisites, with no treatment benefit observed.

## WHAT WOULD REFUTE IT
The claim that the milestone is blocked would be refuted if the artifact's own data showed that all required milestone gate checks passed, that the three missing external scientific producer artifacts were present and verified on disk, and that exclusive CUDA readiness conditions were satisfied.

## WAS THAT CHECKED
Yes. The artifact explicitly evaluated the milestone prerequisites in `gate_check_summary` and `syntax_readiness.block`. It checked whether `results/experiment_7621_v665_evidence_energy.json`, `results/experiment_7622_v665_decision_evaluation.json`, and `results/experiment_7623_v665_guarded_learning.json` existed (all evaluated to `false`), whether `evidence_transport_ready_score` was 1 (observed `0`), and whether exclusive GPU memory met the required threshold (observed busy GPU processes). Rather than asserting an unearned positive or assuming blocked status without testing, refutation was given a genuine opportunity to occur and the blockers were confirmed by row-level checks.

## EVIDENCE
- `honest_verdict`: `"complete_blocked_required_v665_external_evidence"`
- `verdict_class`: `"blocked"`
- `status`: `"complete"`
- `gate_check_summary`: `passed`: `false`, `failed_count`: `4`
- `check`: `"evidence_transport_ready"`, `field`: `"evidence_transport_ready_score"`, `expected`: `1`, `observed`: `0`, `passed`: `false`
- `check`: `"required_scientific_producer"`, `path`: `"results/experiment_7621_v665_evidence_energy.json"`, `observed`: `false`, `passed`: `false`
- `check`: `"required_scientific_producer"`, `path`: `"results/experiment_7622_v665_decision_evaluation.json"`, `observed`: `false`, `passed`: `false`
- `check`: `"required_scientific_producer"`, `path`: `"results/experiment_7623_v665_guarded_learning.json"`, `observed`: `false`, `passed`: `false`
- `check`: `"exclusive_cuda_capacity"`, `expected`: `20000`, `passed`: `false`
- `architecture_status`: `"arc_supervisor"`: `"valid_observational_null_no_firings"`, `"evidence_path"`: `"blocked_before_measurement"`, `"native_service"`: `"measured_benefit_below_10x_nfr"`
- `acceptance_gate_results`: `benefit`: `passed`: `false`, `freshness`: `passed`: `false`, `readiness`: `passed`: `false`, `retention`: `passed`: `false`
- `observational_arc_support`: `actual_firings`: `0`, `causal_benefit`: `false`
- `deployment_speed`: `nfr_10x_met`: `false`, `python_over_direct_native_estimate`: `7.827002746229495`
- `publication_gates`: `authorizes_submission`: `false`, `publication_performed`: `false`

## RECOMMENDATION
KEEP

## experiment_10013_planner_dedup_tiebreak.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative efficacy claim is made. If the completion receipt were treated as a claim, missing arm measurements or failed preconditions would refute it.

## WAS THAT CHECKED
Yes. All three planner arms have reported rows and summaries, and the precondition list reports no failures.

## EVIDENCE
`honest_verdict`: `complete_planner_dedup_tiebreak_measured`; `failed_preconditions`: `[]`; `OFF`; `HUD_DEDUP`; `HUD_DEDUP+TIEBREAK`

## RECOMMENDATION
KEEP
