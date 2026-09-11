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
| CLAIM_SUPPORTED | 6 |
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 1 |

## experiment_7198_v634_feedback_capacity_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The operational-readiness receipt would fail if required capacity/delay cells or comparison units were incomplete, exclusions occurred, or queue, memory, chronology, or replay contracts failed; there is no learning-benefit claim to falsify.

## WAS THAT CHECKED
Yes for operational readiness: all planned cells and units completed, with no exclusions and passed contract checks. No learning-benefit test is claimed. Oracle circularity therefore does not establish—or refute—added learning value.

## EVIDENCE
`honest_verdict` is `complete: bounded feedback stream ready; no learning benefit measured`. `stream_capacity_ready_score` is `1`. `capacity_delay_cells_completed` and `capacity_delay_cells_planned` are both `12`. `comparison_units_completed` and `comparison_units_planned` are both `360`. `exclusions` is `[]`. `model_invoked` is `false`. `verifier_is_oracle` is `true`.

## RECOMMENDATION
KEEP

## experiment_7194_v634_arc_gap_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed audit found no missing-tool request, while observed banked progress was explicitly not attributed causally to the tools or supervisor.

## WHAT WOULD REFUTE IT
A capture-complete induction containing at least one gap event or eligible tool candidate would refute the no-missing-tool finding; causal tool credit or promotion of supervisor help to banked progress would contradict the noncausal qualification.

## WAS THAT CHECKED
Yes. One capture-complete induction exposed 18 parsed tool calls with zero parser failures and zero gap events. The two capture-not-recorded inductions were marked as abstentions for the gap metric. Causal attribution was explicitly withheld rather than inferred from the unpaired session. The oracle status does not create circularity because no verifier-added-value claim is made.

## EVIDENCE
`honest_verdict`: `complete_null_no_missing_tool_requested_banked_progress_noncausal`; `gap_capture_state`: `capture_complete`; `parsed_tool_calls`: `18`; `parser_failures`: `0`; `gap_event_count`: `0`; `capture_complete_induction_count`: `1`; `eligible_candidate_count`: `0`; `kind`: `honest_no_gap`; `banked_level_transitions`: `2`; `causal_tool_credit`: `false`; `supervisor_help_promoted_to_banked_progress`: `false`; `paired_causal_efficacy_estimate_reported`: `false`; `verdict_class`: `null`

## RECOMMENDATION
KEEP

## experiment_7199_v634_bounded_acquisition.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Bounded acquisition did not pass the frozen primary-cell gate.

## WHAT WOULD REFUTE IT
The frozen capacity-4, burst-schedule primary cell passing every prospective-error, false-accept, recurrence-error, capacity, and priority-versus-random requirement, producing an acquisition value score of 1, would refute the claim.

## WAS THAT CHECKED
Yes. The artifact specifies the primary cell and thresholds before outcomes, reports no post-outcome cell selection, includes FIFO and random comparator arms, completes every planned cell and stream unit without exclusions, and records a failed value outcome.

## EVIDENCE
`honest_verdict` is `complete_null: bounded acquisition did not pass the frozen primary-cell gate`; `acquisition_value_score` is `0`; `priority_specific_benefit_score` is `0`; `version_space_acquisition_benefit` is `-0.1501953125`; `cell_selection_after_outcomes` is `false`; `capacity` is `4`; `delay_schedule` is `burst`; `capacity_delay_cells_completed` is `12`; `capacity_delay_cells_planned` is `12`; `independent_stream_units_completed` is `10`; `independent_stream_units_planned` is `10`; `exclusions` is `[]`; `verdict_class` is `null`.

## RECOMMENDATION
KEEP

## experiment_7200_v634_acquisition_cold_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The cold causal audit completed, but Exp7199 showed no acquisition value.

## WHAT WOULD REFUTE IT
A valid prospective comparison showing the acquisition method consistently outperforming the serious FIFO-admission baseline, producing a positive acquisition-value or memory-promotion score, would refute the null claim.

## WAS THAT CHECKED
Yes. The artifact includes prospective metric rows, a FIFO comparator, random and frozen controls, 600 completed terminal units with no exclusions, and separate completion and value gates. The oracle-based verifier would make a positive verifier-value claim circular, but the headline instead reports the observed null.

## EVIDENCE
`"honest_verdict": "complete_null: the cold causal audit completed, but Exp7199 acquisition value was null"`; `"verdict_class": "null"`; `"acquisition_run_complete_score": 1`; `"acquisition_value_score": 0`; `"known_failed_value_promoted": false`; `"memory_promotion_score": 0`; `"arm": "fifo_admission"`; `"window": "prospective"`; `"completed_terminal_units": 600`; `"exclusions": []`; `"verifier_is_oracle": true`

## RECOMMENDATION
KEEP

## experiment_7201_v634_slice_pyo3.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The compiled persistent PyO3 path matched the Python transition behavior, preserved required state, passed finite-law checks, and had lower measured latency than the subprocess path on every tested unit, without claiming a 10x or scientific benefit.

## WHAT WOULD REFUTE IT
A valid, non-abstained transition with differing Rust and Python results, a finite-law row exceeding its stated error limit, a failed restart-state round trip, or a matched timing unit where persistent PyO3 was not lower latency would refute the corresponding headline assertion.

## WAS THAT CHECKED
Yes. The artifact reports 96 completed transition comparisons, 20 completed distribution rows, serialization and replay E2E receipts, and 40 matched timing units. These checks could have failed independently. Although the verifier is the oracle, the headline makes execution-parity and measured-overhead claims—not a claim that the verifier itself adds scientific value.

## EVIDENCE
`compiled`: `true`; `python_fallback_used`: `false`; `planned_transition_rows`: `96`; `completed_transition_rows`: `96`; `planned_distribution_rows`: `20`; `completed_distribution_rows`: `20`; `exclusions`: `[]`; `delta_energy_error`: `0.0`; `magnetization_preserved`: `true`; `total_variation`: `0.12360871466692779`; `total_variation_limit`: `0.15`; `matched_units`: `40`; `pyo3_lower_latency_units`: `40`; `supported`: `true`; `new_10x_gate_created`: `false`; `speed_claim_authorized`: `false`; `verdict_class`: `null`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP

## experiment_7202_v634_slice_cost_quality.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The experiment completed, but sample quality was insufficient, the primary boundary gate failed, and the required 10× speedup was not achieved.

## WHAT WOULD REFUTE IT
Sufficient quality evidence, a passing primary boundary gate, or a Python-control speedup meeting the 10× target would falsify the corresponding null findings.

## WAS THAT CHECKED
Yes. The artifact reports planned-versus-completed rows, explicit quality gates, a fixed primary cell, ten paired seeds, and direct comparisons against both Python and subprocess Rust; those checks failed or fell below the declared thresholds.

## EVIDENCE
`verdict_class`: `null`; `sample_quality_sufficient`: `false`; `boundary_value_score`: `0`; `nfr_01_10x_met`: `false`; `python_speedup_ci95`: `{"estimate": 7.2520510513246155, "lower": 5.625209952972525, "paired_units": 10, "upper": 9.104745221069075}`; `nfr_01_speedup_target`: `10.0`; `parity_passed`: `false`; `all_rows_meet_draw_and_ess_minimum`: `false`; `passed`: `false`; `exclusions`: `[]`; `completed_quality_rows`: `180`; `planned_quality_rows`: `180`; `completed_throughput_rows`: `1080`; `planned_throughput_rows`: `1080`

## RECOMMENDATION
KEEP

## experiment_7203_v634_hardware_correction.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The full-target correction law passed the n=8 authority check and the bounded CPU cost envelope completed.

## WHAT WOULD REFUTE IT
An independently evaluated valid n=8 case where the corrected chain fails the prespecified full-target stationary-law or detailed-balance tolerance; any performance-value reading would also be refuted if full-precision sampling tied or won on effective-sample throughput.

## WAS THAT CHECKED
No. The law was checked only by the correctness authority itself, and effective-sample throughput and mixing speed were not established. The artifact supports an execution-grounded residual measurement, not an independent correctness or added-value claim.

## EVIDENCE
`verifier_is_oracle`: `true`; `verdict_class`: `circular_positive`; `inference_substrate`: `cpu_exact_solver_or_simulator`; `passed`: `true`; `corrected_stationary_residual_max`: `5.551115123125783e-17`; `equal_effective_sample_throughput_established`: `false`; `mixing_speed_established`: `false`; `device_timing_available`: `false`; `hardware_execution_claimed`: `false`.

## RECOMMENDATION
NARROW_CLAIM

## experiment_7204_v634_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V634 evidence matrix is complete, while source semantics, useful continual learning, live generalization efficacy, and measured production deployment remain incomplete.

## WHAT WOULD REFUTE IT
Any contracted task being missing, nonterminal, quarantined, or mismatched would refute matrix completeness; evidence establishing any of the four named scientific capabilities would refute the corresponding incompleteness claim.

## WAS THAT CHECKED
Yes. The artifact checks all 13 contracted tasks for terminal, unquarantined coverage and separately records all four capability outcomes as false, with failed value receipts retained rather than promoted. The oracle verifier does not invalidate this result because the headline makes no claim about the verifier’s added value.

## EVIDENCE
`honest_verdict`: `complete_null: V634 evidence matrix is complete; source semantics, useful continual learning, live generalization efficacy, and measured production deployment remain incomplete`; `planned_task_rows`: `13`; `completed_task_rows`: `13`; `independent_units`: `13`; `exclusions`: `[]`; `expected_value`: `all_declared_v634_artifacts_terminal_and_unquarantined`; `observed_value`: `all_declared_v634_artifacts_terminal_and_unquarantined`; `passed`: `true`; `source_semantics`: `false`; `useful_continual_learning`: `false`; `real_live_generalization`: `false`; `measured_production_deployment`: `false`; `promoted_as_positive`: `false`; `verdict_class`: `null`; `verifier_is_oracle`: `true`

## RECOMMENDATION
KEEP
