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
| CLAIM_SUPPORTED | 3 |
| NO_CLAIM | 4 |
| CANNOT_DETERMINE | 1 |

## experiment_8883_v760_evidence_loss_boundary.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence [', and '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The positive evidence-loss-boundary verdict implies that exact-view verification demonstrates added value beyond the frozen learned heads.

## WHAT WOULD REFUTE IT
Verifier errors against independently defined correctness would challenge its claimed benefit. The cheap exact_view-only solver matching learned_exact_view on errors and decisive coverage across paired views would defeat any claimed benefit from adding the learned component.

## WAS THAT CHECKED
No independent test of verifier value is established: the trusted pre-pass identifies the verifier as the correctness oracle. Agreement supports execution-grounded behavior, but makes the value claim circular.

Omission controls report passing. The exact_view baseline is present, and all three arms tie on the displayed complete-view case; the elided comparisons prevent concluding a global tie or advantage. Generalization and source-fidelity scores are explicitly zero, so neither warrants an additional accusation.

## EVIDENCE
- `honest_verdict`: `complete_circular_positive_evidence_loss_boundary`
- `visible_view_truth`: `Independent entailment from the exact visible record set.`
- `omission_controls`: `missing_hop_exact_abstention`: `true`; `irrelevant_removal_invariance`: `true`
- The displayed `complete` rows for `learned`, `learned_exact_view`, and `exact_view` each have `view_semantic_error`: `0`, `wrong_source_acceptance`: `0`, and `decisive_coverage`: `1`.
- `independent_generalization_score`: `0`; `source_fidelity_established_score`: `0`

## RECOMMENDATION
NARROW_CLAIM

Not checked: elided rows or reruns. Risk: the positive verdict promotes circular oracle agreement into evidence of added value.

## experiment_8884_v760_topology_causal_streams.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit claim is asserted. Its operational readiness assertion would fail if required rows were missing, predictions used feedback before sealing, or required controls failed.

## WAS THAT CHECKED
Operational checks are reported through completion counts, prediction seals, release ordinals and passing controls. Comparative effects are explicitly descriptive; independent benefit testing belongs to Exp8886. Oracle status therefore does not create a circular value claim here.

## EVIDENCE
`honest_verdict`: `complete_null_topology_causal_streams`

`verdict_class`: `null`

`independent_benefit_score`: `0`

`independent_generalization_score`: `0`

`Complete causal evidence and original validation release independent audit; no scientific benefit claim.`

`Frozen Exp8881 paired bootstrap estimates are descriptive; Exp8886 independently tests benefit.`

## RECOMMENDATION
KEEP

Elided rows and Exp8886 were not checked; this artifact does not establish independent benefit.

## experiment_8885_v760_sequential_label_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This run did not establish sequential label-cost savings under the prespecified coverage and safety gates.

## WHAT WOULD REFUTE IT
A learned-family savings lower bound above zero, with coverage-difference lower bounds at least −0.01 in every original mixture and served-group census risks at most 0.05, would contradict the reported gated null.

## WAS THAT CHECKED
Yes: paired costs, coverage contrasts, simultaneous intervals and census risks address those gates. The serious comparator—fixed-budget certification of the same policy—is present.

The displayed shared-union costs tie. Learned standalone savings are recorded, but positive point savings alone do not establish the joint acceptance criteria. Oracle-defined correctness supports this scoped measurement; the artifact claims neither verifier value nor independent generalization. Certificate refusals remain paid observations rather than being excluded as invalid.

## EVIDENCE

- `honest_verdict`: `complete_null_sequential_label_cost`; `verdict_class`: `null`; `label_savings_score`: `0`.
- `shared_union_costs`: `fixed_total_labels`: `4800`; `sequential_total_labels`: `4800`.
- `standalone_costs`: `arm`: `learned_energy`; `fixed`: `4800`; `sequential`: `2160`.
- `agreement_only/balanced/coverage_difference`: `estimate`: `0.0`; `lower_99`: `-0.6493770700631948`.
- `verifier_is_oracle`: `true`; `independent_generalization_score`: `0`.
- `methodology_note`: `Constructed oracle truth establishes no independent generalization.`

## RECOMMENDATION
KEEP

Not checked: elided rows or execution. This supports a run-specific failure to establish savings, not universal absence of savings.

## experiment_8886_v760_label_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The audited matched-label mechanism did not meet the prespecified joint requirements for label benefit and retention on constructed signed-record streams.

## WHAT WOULD REFUTE IT
All eight joint gates passing: the simultaneous upper bounds meet the required Brier improvements against frozen and permuted comparators, and retention Brier and error increases stay within 0.01 at every checkpoint.

## WAS THAT CHECKED
Yes. The joint gate rows report all eight checks across 24 paired streams. Two fail: the online Brier upper bound against the frozen comparator is −0.006665 versus a required −0.01, and the checkpoint-24 retention error upper bound is 0.024692 versus an allowed 0.01.

The frozen comparator is a serious baseline; the conclusion does not rely solely on beating permuted labels. Oracle-defined correctness supports this scoped measurement without establishing verifier value. The artifact claims no independent generalization. Visible rows contain the metric fields and complete statuses; no exclusions are reported.

## EVIDENCE

- `honest_verdict`: `complete_null_label_benefit_audit`; `verdict_class`: `null`; `label_benefit_score`: `0`.
- `matched_true_minus_frozen_verified`, `online`, `brier`: `upper`: `-0.006664814992460993`; `threshold`: `-0.01`; `passed`: `false`.
- `matched_true_minus_frozen_verified`, `retention`, `checkpoint`: `24`, `error`: `upper`: `0.02469184027777752`; `threshold`: `0.01`; `passed`: `false`.
- `independent_count`: `24`; `independent_generalization_score`: `0`; `verifier_is_oracle`: `true`.
- `exposure_scope`: `Constructed signed-record truth only; no natural-language generalization or generator training.`
- `excluded_count`: `0`; `status`: `complete`; `missing_reason`: `null`.

## RECOMMENDATION
KEEP

Elided rows and source code were not checked. Failure to pass the joint gates does not prove zero benefit.

## experiment_8887_qwen_source_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite check and asserts no extraction-performance or comparative result.

## WAS THAT CHECKED
The prerequisite gates were checked; runtime readiness failed. The intended extraction comparison did not run, so its possible refutation remains untested.

## EVIDENCE
`honest_verdict`: `blocked_gate_check_failed`; `status`: `blocked`; `failed_field`: `runtime_repair_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

No rerun performed; extraction performance remains untested.

## experiment_8888_v760_matched_adaptation_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The completed four-arm CPU cost comparison detects no statistically resolved cost difference, without claiming quality improvement or speed benefit.

## WHAT WOULD REFUTE IT
Any reported familywise-adjusted paired interval excluding zero would contradict the cost null. Incomplete paid work or a restart-state mismatch would contradict the separate readiness assertion.

## WAS THAT CHECKED
Yes. All six paired cost intervals across 12 streams cross zero, including comparisons against the serious frozen baselines. Completion counts and restart receipts also report success.

The oracle defines correctness, but the claim concerns measured cost rather than verifier value. No generalization benefit is asserted. The null therefore survives these checks; it does not establish equal costs.

## EVIDENCE
- `honest_verdict`: `complete_null_matched_adaptation_cost`
- `adaptation_cost_ready_score`: `Complete paid work and exact restart parity; never scientific benefit.`
- `scope`: `complete CPU cost only; no admitted scientific quality evidence`
- `speed_claim`: `false`
- `control_arm`: `frozen_verified`; `treatment_arm`: `matched_true`; `lower_ns`: `-1217350045.0124445`; `upper_ns`: `386200776.3457779`
- `control_arm`: `matched_true`; `treatment_arm`: `matched_permuted`; `lower_ns`: `-698042565.9393728`; `upper_ns`: `1189377194.272706`
- `paired_streams`: `12`; `missing_pairs`: `0`
- `intended_count`: `48`; `completed_count`: `48`; `excluded_count`: `0`; `failed_count`: `0`
- `exact_state`: `true`; `process_ids_paid`: `true`

## RECOMMENDATION
KEEP

Elided rows were not independently checked; non-detection must not be presented as cost equivalence.

## experiment_8889_v760_hardware_operation_map.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Equal or slower complete board runtime than existing CPU/native batching, at matched correctness, would refute an acceleration claim. This artifact makes no such claim; it records an operation map and withholds acceleration qualification.

## WAS THAT CHECKED
No acceleration comparison was performed. The artifact reports zero physical probes and null board timings and gain bounds. Its audit completion does not assert a measured performance benefit.

## EVIDENCE
- `honest_verdict`: `complete_null_hardware_operation_map`
- `principle`: `Valid mapping evidence does not establish acceleration`
- `physical_probes_performed`: `0`
- `measured_board_time_s`: `null`
- `end_to_end_gain_bound`: `null`
- `inference_substrate`: `aggregation_from_upstream_artifacts`

## RECOMMENDATION
KEEP

Review limited to the supplied artifact; hardware performance remains untested.

## experiment_8890_v760_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to falsify. Its administrative accounting would be contradicted by an unaccounted scheduled slot or rejected evidence contributing to admitted scientific results.

## WAS THAT CHECKED
Yes, partially: the displayed gate checks preserve runtime and producer-admission failures, and the rejected canary has scientific admission disabled. Elided rows prevent independently checking all thirteen slots. This artifact does not establish comparative benefit or generalization.

## EVIDENCE

- `honest_verdict`: `complete_blocked_v760_capstone`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `exposure_scope`: `Administrative custody and method ingestion only; no new scientific benefit, natural-text or hidden-game generalization.`
- `science_admitted`: `false`
- `natural_generalization_count`: `0`
- `historical_publication_scope`: `historical FoVer paper only`
- `publication_authorized`: `false`

## RECOMMENDATION
KEEP
