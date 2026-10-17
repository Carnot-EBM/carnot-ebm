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

## experiment_8583_learning_benefit_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No benefit result is asserted. The readiness prerequisite would fail if its observed score were 0 instead of the required 1.

## WAS THAT CHECKED
Yes. The sole entry in gates_evaluated records exactly that failure and blocks the experiment. Later Brier benefit and simultaneous retention were not evaluated here; the title describes intended work.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `causal_learning_rows_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8584_v737_resident_complete_cost.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8585_v737_hallutruth_release_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The 4,000-row release inspection is complete, while independent generalization and unseen-question or unseen-generator eligibility remain unestablished.

## WHAT WOULD REFUTE IT
Unaccounted-for release rows would contradict completion; documented independent evaluation on genuinely unexposed questions or withheld generator families would contradict the stated evaluation limits.

## WAS THAT CHECKED
Yes, within the audit’s scope. Split completion counts reconcile to 4,000. Exposure and overlap checks establish that the official splits provide no unexposed evaluation here; generator inventories show one family, and independent evaluation units are zero. This is a bounded null finding, with no comparative detector-value claim requiring a rival arm.

## EVIDENCE
- `honest_verdict`: `complete_null_hallutruth_release_audit`
- `completed_count`: `4000`; `intended_count`: `4000`; `failed_count`: `0`
- `exposure_scope`: `Complete public Arabic release inspected; all official splits exposed to this audit.`
- `exact_question_overlap_count`: `5`, `1`, `2`
- `generator_counts`: `QCRI/Fanar-1-9B-Instruct`
- `independent_count`: `0`; `independent_generalization_score`: `0`
- `unseen_question_eligible_score`: `0`; `unseen_generator_eligible_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8586_qwen_evidence_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no empirical headline to falsify. All three prerequisite gates passing would contradict the receipt’s reported gate failure.

## WAS THAT CHECKED
Yes, in the three gate evaluations: one passed and two failed. The artifact records a blocked run and makes no comparative claim about model or verifier value.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `gate_check_summary`: `gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp8579-contract-methods.canary_inputs_ready_score (actual=0 == expected=1)`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8587_v737_arc_publication_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A replay mismatch, incomplete test union or owned coverage, or failed publication closure would refute the infrastructure qualification. No comparative benefit or generalization claim is made.

## WAS THAT CHECKED
Yes for the visible test-union and coverage receipts; terminal replay and publication closure are reported through acceptance gates, without their detailed receipts visible here. Oracle-defined correctness establishes no verifier advantage, but none is claimed. The action-parity ties are preservation checks. The positive verdict token accompanies infrastructure qualification whose checks could fail.

## EVIDENCE

- `honest_verdict`: `complete_circular_positive_v737_arc_publication_qualification`
- `no_scientific_benefit_claim`: `true`
- `headline_solve_credit`: `0`; `generalized_learning_benefit_score`: `0`; `independent_generalization_score`: `0`
- `independent_count`: `0`; `inference_substrate`: `aggregation_from_upstream_artifacts`
- `exact_union_receipt`: `disjoint`: `true`, `passed`: `true`
- `child_coverage_receipts`: `missing_lines`: `[]`, `passed`: `true`
- `Bind this invocation to immutable operands; infrastructure qualification grants no scientific benefit or solve credit.`

## RECOMMENDATION
KEEP

## experiment_8588_v737_arc_capability_panel.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
In this model-disabled public-game panel, capability masking reduces disabled attempts but does not meet the stated useful-efficiency criterion.

## WHAT WOULD REFUTE IT
Fewer actual disabled attempts, nondecreasing levels in every pair, and no increase in complete wall cost would contradict the reported efficiency null.

## WAS THAT CHECKED
Yes. All eight matched pairs show one fewer actual disabled attempt, with tied levels and action counts. Runtime differences have both signs, but the game-mean increase is approximately 0.1533 seconds, failing the wall-cost requirement.

The unmasked baseline supplies the direct comparison. Mechanical exclusion of disabled operations alone does not determine the efficiency verdict: measured wall cost can defeat it, and did here. Action-limit censoring is disclosed; the null does not assert generalization or reasoning gains.

## EVIDENCE

- `honest_verdict`: `complete_null_v737_arc_capability_panel`
- `useful_efficiency`: `false`; `capability_efficiency_score`: `0`
- `paired_results`, `deltas`: `actual_disabled_attempts`: `-1`; `peak_level`: `0`; `actions_used`: `0`
- `descriptive_uncertainty`, `runtime_s`, `game_mean`: `0.15330201387499998`
- `censoring`: `action_limit`; `censored_count`: `16`
- `hidden_game_performance`: `unmeasured`; `scientific_reasoning_benefit_claimed`: `false`

## RECOMMENDATION
KEEP

## experiment_8589_v737_resident_hardware_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative performance claim to falsify. Qualified complete resident costs and authenticated compatible device execution would contradict the reported qualification blockers.

## WAS THAT CHECKED
Yes, for the blockers: gate_check_summary records resident-cost readiness as 0 against an expected 1, and compatible execution as null for all five devices. These support the blocked status. The artifact makes no claim that hardware acceleration was measured or ruled out.

## EVIDENCE

- `exposure_scope`: `exposed_engineering_accounting_only`
- `measured_resident_acceleration`: `null`
- `missing_reason`: `resident_cost_unqualified_and_no_compatible_installed_kernel`
- `qualified_resident_cost`: `false`
- `compatible_physical_execution`: `false`
- `Each unit is one board/operation mapping; completion records scope accounting, not a measured device sample.`

## RECOMMENDATION
KEEP

## experiment_8590_v737_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the supplied excerpt records administrative completion and blocked qualification without asserting comparative benefit.

## WAS THAT CHECKED
The visible gates allow qualification to fail and record failures, including unexecuted producers. No positive verifier-value or generalization claim is made here, so the oracle flag alone does not establish overstatement.

## EVIDENCE

- `administrative_complete`: `true`
- `qualified_learning_benefit`: `false`
- `qualified_resident_cost`: `false`
- `producer_executed`: `false`
- `disposition`: `pre_gate_receipt`
- `Stored ARC replay checks execution mechanics and grants no hidden-game capability benefit.`

## RECOMMENDATION
KEEP
