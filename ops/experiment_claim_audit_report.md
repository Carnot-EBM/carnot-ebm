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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8491_objective_benefit_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. The readiness prerequisite would fail if objective_stream_ready_score differed from 1.

## WAS THAT CHECKED
Yes—the sole gate observed 0 against required 1 and failed. This is a blocked receipt; it reports no benefit or retention evaluation.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `objective_stream_ready_score`
- `expected`: `1`
- `actual`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8492_v730_publication_cost_model.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The positive verdict treats publication-cost accounting and append-protocol qualification as evidence of engineering value.

## WHAT WOULD REFUTE IT
Direct accounting of the same upstream clocks, using the original validators, produces identical qualification and cost bounds: the cheapest serious baseline adds no less value than this method.

## WAS THAT CHECKED
No. The artifact reports reconstruction and fixture qualification, without an added-value comparison against that baseline. The verifier defines correctness, so passing its gates cannot independently establish verifier value.

The speed ceiling subtracts assumed removable costs from existing clocks under zero-new-overhead assumptions; it does not expose the proposed append implementation to a measured loss. The artifact correctly disclaims measured speedup and generalization, but its positive verdict exceeds those accounting and qualification results.

## EVIDENCE

- `honest_verdict`: `complete_circular_positive_publication_cost_model`
- `verdict_class`: `circular_positive`
- `verifier_is_oracle`: `true`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `append_protocol_ready_score`: `1`
- `conditional engineering cost bound; requires Exp8494 measurement`
- `retained work unchanged`
- `new append overhead zero`
- `snapshot pointer marginal cost equals journal certificate marginal cost`
- `measured_append_speedup`: `null`
- `scientific_speed_benefit_score`: `0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
NARROW_CLAIM

## experiment_8493_v730_append_store_qualification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The append store preserves snapshot transition behavior and recovers correctly after process exits in the tested engineering fixtures.

## WHAT WOULD REFUTE IT
A recovered append state differing from the snapshot reference, a returned stable ID lost during recovery, or an unacknowledged frame applied twice at any tested crash boundary.

## WAS THAT CHECKED
Yes: the equivalence rows, crash rows, and acknowledgment ledger address these failures across two equivalence cells and fourteen process exits, with sixteen completions and zero reported failures. Recovery could have failed; passing is not guaranteed by the oracle defining correctness. Snapshot equality supports the stated equivalence claim. No verifier advantage, generalization, or speed benefit is asserted.

## EVIDENCE
- `append_store_ready_score`: `1`
- `completed_count`: `16`; `failed_count`: `0`
- `equivalence_rows`; `crash_rows`; `acknowledgment_ledger`
- `actual_exit`: `73`; `passed`: `true`
- `methodology_note`: `Stable-file framed append using independently sealed original five-input service tapes and unchanged snapshot transitions. Fourteen actual process exits and bitwise recovery qualify engineering fixtures only. No model calls, scientific benefit, speed measurement or physical power-loss certification.`

## RECOMMENDATION
KEEP

## experiment_8494_v730_append_service_cost.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8495_bounded_qwen_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental claim to falsify. The reported gate failure would be contradicted if both evaluated gates passed.

## WAS THAT CHECKED
Yes, for gate status: `gates_evaluated` records runtime readiness failing and source-control qualification passing. The artifact is a blocked-run receipt, not an experimental result.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `runtime_repair_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8496_v730_arc_panel_qualification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The bounded public-game panel completed execution qualification but observed no redirect-resolved level-ups and established no scientific benefit.

## WHAT WOULD REFUTE IT
A valid applied-redirect row recording an actual SDK level-up would contradict the zero-resolution result; progress exceeding the matched same-game, same-seed off arm would provide evidence against the benefit null.

## WAS THAT CHECKED
Yes, within the 512-action horizon. Applied redirects have outcome rows, and the unchanged policy with redirects off supplies the cheapest serious comparator. The visible paired comparisons tie at zero level progress; the aggregate reports zero resolved redirects.

The intervention actually changed policy state, including exploration diversity and induction attempts, so the observed tie was not imposed by identical treatment. Runtime level progress supplies the outcome rather than receipt emission. No held-out generalization or verifier-value claim is asserted.

Censored outcomes remain pending, so this supports a bounded observational null, not a conclusion that redirects can never help. The supplied excerpt shows detailed comparisons for two games, rather than every panel pair.

## EVIDENCE
- `honest_verdict` = `complete_null_v730_arc_panel_qualification`
- `completed_count` = `16`; `intended_count` = `16`; `excluded_count` = `0`
- `applied_redirects` = `16`; `resolved_redirects` = `0`; `pending_redirects` = `32`
- Visible pairs: `peak_level_delta_on_minus_off` = `0`; `progress_count` = `0`
- `outcome_window_status` = `pending_censored`; `censoring` = `action_limit`; `max_actions` = `512`
- `scientific_benefit` = `false`; `independent_generalization_score` = `0`

## RECOMMENDATION
KEEP

## experiment_8497_v730_board_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is made. A qualified current device-operation row with complete transfer/device clocks and deployment readiness would contradict the recorded blocked boundary.

## WAS THAT CHECKED
No current device trial is reported. This artifact records historical evidence and unresolved qualification gates; it does not claim performance gains, verifier value, or generalization.

## EVIDENCE
- `Bind complete V730 authority and original producer hashes; no device or model work.`
- `exposure_scope`: `historical_public_labels_retrospective_overlap_unknown; engineering_continuity_only`
- `deployment_ready`: `false`
- `device_commands_executed`: `[]`
- `compatible_operations`: `false`
- `complete_transfer_device_clocks`: `false`
- `generalized_learning_benefit_score`: `0`
- `fallback_is_current_append_measurement`: `false`
- `missing_field`: `service_cost_ready_score`

## RECOMMENDATION
KEEP

## experiment_8498_v730_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this capstone reports blocked evidence aggregation, without asserting comparative scientific benefit. Its zero generalization scores should not be interpreted as measured evidence that generalization is impossible.

## WAS THAT CHECKED
No comparative scientific claim was tested here. Administrative checks could fail and did fail in gate_check_summary; the capstone retains a blocked verdict.

## EVIDENCE
- `honest_verdict`: `complete_blocked_v730_capstone`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `scientific_claims_independently_generalize`: `false`
- `Authority, complete slot accounting and owned validation qualify the reconciliation; none establishes independent scientific benefit.`
- `exposure_scope`: `public-label retrospective and historical engineering evidence; independent factuality and hidden-game generalization remain open`

## RECOMMENDATION
KEEP
