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

## experiment_8845_v757_indexed_witness.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The opt-in indexed witness implementation preserves the reference evaluator’s decisions on the exposed encoded cases and passes scoped semantic and lifetime checks.

## WHAT WOULD REFUTE IT
A completed same-input pair with different decisions, an accepted invalid witness, or a failed index lifetime check would refute semantic readiness. Agreement between separately executed implementations is not guaranteed merely because they use the same facts.

## WAS THAT CHECKED
Yes, as reported in the parity, adversarial, and lifetime checks; the displayed comparisons agree and the aggregate reports no failures or censoring. The unchanged joint evaluator is a serious comparator. Its tie supports the stated equivalence claim; it establishes no added value.

The oracle defines encoded correctness, but the headline claims implementation parity, not verifier benefit. Exposed inputs are acknowledged, and no generalization or cost improvement is asserted.

## EVIDENCE
- `methodology`: `Opt-in exact predicate index versus unchanged always-joint; all original V756 inputs; exhaustive small-domain and adversarial semantic controls. Readiness alone, no cost benefit claim.`
- `indexed_decision`: `support`; `reference_decision`: `support`; `passed`: `true`.
- `completed_count`: `7200`; `failed_count`: `0`; `censored_count`: `0`; `excluded_count`: `0`.
- `source_mutation_isolated`, `immutable_table`, `stale_reuse_rejected`, `cross_query_rejected`: each reports `passed`: `true`.
- `verifier_is_oracle`: `true`; `independent_generalization_score`: `0`.

## RECOMMENDATION
KEEP

No rerun or independent inspection of elided rows; support remains limited to reported semantic readiness.

## experiment_8846_v757_future_benefit_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The categorical combined method fails the audit’s overall future-benefit gates, and the simpler linear head is preferred.

## WHAT WOULD REFUTE IT
Categorical combined passing all benefit gates: sufficient improvement over frozen and calibration-only arms, superiority to shuffled feedback, noninferiority to linear within the 0.005 Brier margin, and acceptable retention.

## WAS THAT CHECKED
Yes. The paired intervals include those comparators across 24 streams, alongside retention checks. Linear wins: categorical-minus-linear Brier is positive throughout its interval. The shuffled comparison crosses zero.

Categorical does improve over frozen and calibration-only arms, so the supported null concerns the full acceptance criteria—not absence of every benefit. Oracle-defined correctness does not invalidate this null; no positive verifier-value or generalization claim is asserted.

## EVIDENCE
- `honest_verdict`: `complete_null_future_benefit_audit`; `verdict_class`: `null`.
- `categorical_combined_minus_linear_combined`: `lower`: `0.04901471759739543`; `upper`: `0.13321966450168252`.
- `categorical_noninferiority_margin`: `0.005`.
- `categorical_combined_minus_categorical_shuffled_combined`: `lower`: `-0.0742778821541686`; `upper`: `0.008794601228550662`.
- `categorical_combined_minus_categorical_calibration_only`: `upper`: `-0.03341303625977893`.
- `retention_pass_score`: `0`; `causal_benefit_score`: `0`.

## RECOMMENDATION
KEEP

Not independently checked: elided rows or upstream execution. The conclusion is limited to the reported audit gates.

## experiment_8847_v757_indexed_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
On the exposed encoded evaluation worlds, indexed_witness reduces counted paid operations by at least 20% versus always_joint while preserving completed decisions and measured quality.

## WHAT WOULD REFUTE IT
An evaluation operation-reduction lower bound below 0.2, disagreement between completed decisions, or an upper bound above 0.01 for coverage loss or false-accept increase would defeat acceptance. Stress quality regression would also defeat it.

Oracle-defined correctness cannot establish independent semantic value, but operation savings against another evaluator can fail independently of correctness.

## WAS THAT CHECKED
Yes. The evaluation gates report a reduction lower bound of approximately 0.830, decision parity, and zero bounds for both quality losses. The frozen always_joint exact evaluator supplies a substantive comparator. Index construction is charged.

Failure was possible: indexing increases paid operations at every reported stress size, and those cases receive zero benefit scores. They do not contradict the specified evaluation benefit because stress acceptance requires quality preservation.

The artifact explicitly disclaims generalization. No visible row-validity or wrong-key issue contradicts the headline. Review is limited to the supplied text; elided rows were not independently inspected.

## EVIDENCE
- `indexed_cost_benefit_score`: `1`
- `paid_operation_reduction`: `lower`: `0.8298495977674142`; `threshold`: `0.2`
- `completed_decision_parity`: `passed`: `true`
- `coverage_loss` and `semantic_false_accept_increase`: `upper`: `0.0`
- `always_joint`; `indexed_witness`
- `index_build_rows`: `Charge actual index construction for every indexed request, including exhausted requests.`
- `stress_results`: reduction estimates `-0.003558718861210064`, `-0.016973741476860482`, `-0.02430269946907959`; `indexed_cost_benefit_score`: `0`
- `verifier_is_oracle`: `true`
- `exposure_scope`: `exposed V756 encoded worlds and constructed stress; no language or hidden-game generalization`

## RECOMMENDATION
KEEP

Not rerun. Support covers execution cost on this exposed corpus; independent semantic value remains unestablished.

## experiment_8848_v757_memory_size_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This is a complete cost measurement across memory occupancies on fixed constructed workloads, without a claim of demonstrated benefit.

## WHAT WOULD REFUTE IT
A scheduled window omitted from the denominator, an invalid window counted as ready, or reported costs inconsistent with the stated timing boundary would contradict the cost-accounting claim.

## WAS THAT CHECKED
Yes, at the reported accounting level: all 60 scheduled windows completed, none were failed, excluded, or censored, and the paired comparisons report no missing pairs. Elapsed time could vary independently of occupancy; the cost result was not predetermined by correctness.

The oracle and workload overlap do not support verifier value or generalization, but neither is claimed. Elided rows and calculation code were not independently checked; these costs cannot establish utility.

## EVIDENCE

- `honest_verdict`: `complete_null_memory_size_cost`
- `methodology_note`: `Fixed constructed five-predicate workloads on the unchanged V754 store. Ten paired blocks; no benefit claim. Preparation costs remain separate.`
- `intended_count`: `60`; `completed_count`: `60`
- `failed_count`: `0`; `excluded_count`: `0`; `censored_count`: `0`
- `paired_blocks`: `10`; `missing_pairs`: `0`
- `complete_windows`: `passed`: `true`; `principle`: `Every scheduled window remains in the denominator.`

## RECOMMENDATION
KEEP

## experiment_8849_qwen_witness_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked prerequisite-check receipt, with no claim about extraction performance or added value.

## WAS THAT CHECKED
No extraction evaluation occurred. Prerequisite gates were checked; runtime readiness failed, blocking execution before the experiment.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `runtime_repair_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8850_arc_outcome_transfer.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-gate receipt making no comparative or transfer-performance claim.

## WAS THAT CHECKED
The prerequisite gate was checked: availability scored 0 against the required 1, blocking execution. Transfer performance was not evaluated; no transfer conclusion follows.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `arc_new_outcomes_available_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_8851_v757_hardware_operation_budget.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative performance claim to falsify. A compatible board workload completing faster than the CPU baseline, including host, transfer, programming, readout and durable costs, would overturn the stated reason for deferring acquisition.

## WAS THAT CHECKED
No physical comparison was performed. This artifact inventories upstream CPU evidence and explicitly withholds acceleration qualification. It reports an evidence gap, rather than an experimentally demonstrated absence of hardware benefit.

## EVIDENCE
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `honest_verdict`: `complete_null_hardware_operation_budget`
- `reason`: `no measured complete supported physical workload benefit`
- `fabric_supported`: `false`
- `measured_board_time_s`: `null`
- `physical_probes_performed`: `0`
- `research_100x`: `operation-specific future target; no measured gain`

## RECOMMENDATION
KEEP

Physical performance was not tested; whether hardware could provide a benefit remains unresolved.

## experiment_8852_v757_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no positive scientific headline to falsify. The narrower accounting assertion would fail if a scheduled slot lacked primary or skip custody, or rejected evidence contributed to admitted scientific totals.

## WAS THAT CHECKED
Yes, as reported: the accounting gate passes, rejected producers are marked scientifically inadmissible, and administrative completion preserves disqualified and null outcomes. The overall verdict remains blocked. The linear continuation recommendation and historical paper readiness do not assert current scientific success.

Elided rows and upstream calculations were not independently checked; they limit confirmation of the accounting totals.

## EVIDENCE
- `honest_verdict`: `complete_blocked_incomplete_v757_design_authority`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `complete_accounting`: `passed`: `true`
- `cluster_id`: `administrative_task_slots`
- `disposition`: `rejected_producer`; `science_admitted`: `false`
- `independent_generalization_score`: `0`
- `historical_publication_scope`: `historical FoVer paper only`
- `publication_authorized`: `false`

## RECOMMENDATION
KEEP
