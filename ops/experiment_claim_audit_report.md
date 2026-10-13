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
| NO_CLAIM | 4 |
| CANNOT_DETERMINE | 1 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8462_brier_online_learning.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact is a blocked-gate receipt, with no comparative result to falsify.

## WAS THAT CHECKED
No comparative refutation was tested. The prerequisite gate was checked and failed: objective readiness was 0 against a required 1, blocking the experiment.

## EVIDENCE

- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `objective_kernel_ready_score`
- `blocked_reason`: `actual=0 == expected=1`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8464_v728_persistence_localization.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## THE HEADLINE CLAIM
The persistence-localization experiment completed with a null result, as reported by the mechanical pre-pass.

## WHAT WOULD REFUTE IT
Valid paired timing rows meeting the experiment’s criterion for a positive persistence-localization result would contradict the null. The supplied excerpt does not specify that criterion, so a numerical falsifier cannot be identified.

## WAS THAT CHECKED
No—not demonstrably in the supplied excerpt. It ends inside historical model metadata, before the measured paired timing rows, stage summaries, and null-versus-positive decision rule. Passing qualification gates does not establish that the localization null survived a possible refutation. This does not demonstrate a design flaw; it limits what can be reviewed.

## EVIDENCE
- `complete_clocks`: `true`; `exact_stage_nesting`: `true`.
- `persistence_profile_ready_score`: `Authentic complete paired clocks and unchanged causal semantics with exact stage nesting and all required checks; favorable bottleneck is unnecessary.`
- `generalized_learning_benefit_score`: `0`.
- `exposure_scope`: `retrospective_cached_five_input_engineering_service`.

## RECOMMENDATION
KEEP

## experiment_8465_v728_journal_equivalence.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The journal preserves reference behavior and durable acknowledgments on the cached engineering fixture under the tested process exits.

## WHAT WOULD REFUTE IT
A journal response or state differing from the reference for the same transaction, or recovery losing or altering an acknowledged transaction after a tested exit.

## WAS THAT CHECKED
Yes: equivalence_rows report passing checks at batch sizes 1 and 8; crash_rows report passing recovery checks after actual process exits at named write barriers. These storage operations could have lost acknowledgments or produced divergent states.

The reference behavior is the relevant baseline; matching it supports equivalence. The verifier’s oracle status does not establish added value, but the artifact reports zero learning benefit and explicitly limits its durability scope. The visible rows show no invalid-row inclusion or missing metric key.

## EVIDENCE
- `equivalence_rows`: `barrier` values `1` and `8`; `passed`: `true`.
- `crash_rows`: `before_payload`, `after_payload`, `before_fsync`; `actual_exit`: `73`; `passed`: `true`.
- `completed_count`: `18`; `failed_count`: `0`.
- `generalized_learning_benefit_score`: `0`.
- `actual file/directory fsync and private process-exit recovery; physical power-loss behavior is not certified`
- `Bind immutable invocation operands; engineering equivalence grants no scientific benefit.`

## RECOMMENDATION
KEEP

## experiment_8466_v728_journal_service_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The cached same-host journal-service comparison did not establish a primary complete-service speed benefit.

## WHAT WOULD REFUTE IT
A qualifying paired comparison against the ordinary non-journal service performing identical durable work, with a complete-service speedup whose 95% confidence interval lies entirely above 1.

## WAS THAT CHECKED
Yes, at the reported cell level: both cells completed all five intended pairs, and both speedup intervals include 1. The timing comparison could have demonstrated a benefit; the reported intervals did not.

The verifier’s oracle role does not make this timing null circular. No positive generalization claim is made. The cited speedup fields exist in both cell rows. Comparator implementation details and repeat-level validity flags are not visible in the supplied excerpt, limiting further audit; they do not contradict the reported null.

## EVIDENCE

- `primary_speed_benefit`: `false`; `tenfold_host_service`: `false`.
- First cell: `pooled_speedup`: `1.0447833367347368`; `speedup_ci95` endpoints: `0.9942346767648512`, `1.1095895447657544`.
- Second cell: `pooled_speedup`: `0.9111690109390167`; `speedup_ci95` endpoints: `0.8024861484958734`, `1.0188553601484447`.
- Both cells: `completed_pairs`: `5`; `intended_pairs`: `5`.
- `excluded_count`: `0`; `failed_count`: `0`.
- `exposure_scope`: `retrospective_cached_same_host_engineering_fixture`; `generalized_learning_benefit_score`: `0`.

## RECOMMENDATION
KEEP

## experiment_8467_bounded_qwen_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked gate receipt, with no claim about Qwen performance or added value to falsify.

## WAS THAT CHECKED
The prerequisite checks were recorded in gates_evaluated: both observed 0 against required 1 and failed. No Qwen evaluation is reported.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `runtime_repair_ready_score` and `source_controls_ready_score`: each has `expected` = `1`, `actual` = `0`, and `passed` = `false`.
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8468_v728_arc_generalization_panel.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8469_v728_board_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records an operational boundary and evidence receipts without asserting comparative benefit.

## WAS THAT CHECKED
Not applicable to a comparative claim. The visible operation rows record host-only execution, and the acceptance gates explicitly withhold operation compatibility and complete device/transfer timing.

## EVIDENCE
- `Bind original hashes and actual invocation; recording grants no deployment, learning or purchase claim.`
- `board_boundary_recorded_score`: `1`
- `compatible_operations`: `false`
- `complete_transfer_device_clocks`: `false`
- `compatibility`: `host_only_no_installed_compatible_operation`
- `full_service_acceleration`: `false`
- `generalized_learning_benefit_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8470_v728_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the supplied excerpt is an evidence-aggregation and qualification receipt that withholds independent scientific benefit. It asserts no comparative effect to falsify.

## WAS THAT CHECKED
No comparative hypothesis is tested here. The gate_check_summary records producer qualification failures. Those failures and the oracle status do not contradict a positive scientific headline, because none is asserted in the supplied text.

## EVIDENCE

- `scientific_claims_independently_generalize`: `false`
- `generalized_learning_benefit_score`: `0`
- `Authority, complete slot accounting and owned validation qualify the reconciliation; none establishes independent scientific benefit.`
- `public-label retrospective and historical engineering evidence; independent factuality and hidden-game generalization remain open`

## RECOMMENDATION
KEEP
