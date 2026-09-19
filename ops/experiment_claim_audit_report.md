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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 3 |
| CANNOT_DETERMINE | 2 |
| SKIPPED_ALREADY_FLAGGED | 2 |

## experiment_7401_v649_online_audit.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7402_v649_proposal_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no scientific or comparative claim to falsify; the procedural blockage would be refuted by a passing runtime prerequisite followed by actual model calls and candidate results.

## WAS THAT CHECKED
Yes for the prerequisite: the runtime gate was checked and failed. No experiment was run, so no efficacy claim was tested.

## EVIDENCE
`"learning_claim": false`, `"verdict_class": "blocked"`, `"status": "blocked_precondition_failed"`, `"honest_verdict": "blocked_one_owned_rtx3090_slot"`, `"attempted_call_count": 0`, `"completed_call_count": 0`, `"candidate_rows": []`, `"rows": []`, `"model_invoked": false`

## RECOMMENDATION
KEEP

## experiment_7403_v649_synthetic_memory.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Source-certified synthetic memory provides positive value under the frozen efficacy and safety gates.

## WHAT WOULD REFUTE IT
A non-oracular correctness evaluation showing unsafe or incorrect reused decisions, or a serious persistent baseline tying or beating the memory method on complete-service cost and paid exact-query use, would refute the value claim.

## WAS THAT CHECKED
No. Efficacy comparisons and adversarial attacks were checked, so ordinary metric failure was possible, but correctness was defined by the same exact-formula authority used by the verifier. The artifact therefore gave the execution metrics—but not the claimed verifier-added value—an independent chance to fail.

## EVIDENCE
`"honest_verdict": "complete_circular_positive_source_certified_synthetic_memory_value"`; `"verdict_class": "circular_positive"`; `"verifier_is_oracle": true`; `"synthetic_memory_value_score": 1`; `"exact_decision_coverage": 1.0`; `"unsafe_decisions": 0`; `"comparators": ["persistent_incremental_exact_solver", "persistent_source_graph_reachability_cache"]`; `"live_model_benefit_established": false`; `"promotion_score": 0`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7404_live_memory.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked prerequisite gate and no memory-performance result.

## WAS THAT CHECKED
No. The experiment stopped at `conductor_pre_gate` because an upstream capture requirement failed.

## EVIDENCE
`status` `blocked` `honest_verdict` `blocked_gate_check_failed` `blocked_reason` `actual=0 == expected=1` `failed_field` `candidate_capture_complete_score` `blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7405_v649_proof_audit.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence [', while simultaneously classifying the result as ', ' and setting ', ' has '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Persistent methods provide confirmed synthetic efficacy/value, although combined confirmation is blocked because the live cohort is unavailable.

## WHAT WOULD REFUTE IT
Valid independently adjudicated rows showing incorrect or unsafe decisions, or cost/query ratios tying or exceeding the reset exact solver or an equally persistent non-feedback baseline, would refute the value claim; eligible live rows showing no benefit would refute general applicability.

## WAS THAT CHECKED
No. Synthetic cost reduction could have failed numerically, but the broader value claim lacked a genuine independent test: the correctness verifier was itself the oracle, and the live cohort produced no rows. The reset arm tested savings against restarting, but not against an equally persistent baseline without the claimed feedback value.

## EVIDENCE
The synthetic cohort reports `confirmed_value` as `1` and `efficacy` as `passed` `true`, while simultaneously classifying the result as `circular_positive` and setting `verifier_is_oracle` to `true`. The live cohort has `eligible` `false`, `efficacy` `null`, and `confirmed_value` `0`; `live_rows_attempted` and `live_rows_completed` are both `0`. The combined check `combined_live_plus_synthetic_value` has `observed` `0` and `passed` `false`, and `proof_value_confirmed_score` is `0`. The terminal `verdict_class` is `blocked`.

## RECOMMENDATION
NARROW_CLAIM

## experiment_7406_v649_arc_generalization.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7407_v649_service_cost.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence ['[0.9614007200306944, 1.0074236343567287]'] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_REFUTED_BY_OWN_DATA

## THE HEADLINE CLAIM
Vectorized NumPy provides a full-service performance benefit without limiting that claim to larger batches.

## WHAT WOULD REFUTE IT
A measured batch where the scalar comparator ties or outperforms vectorization, or where the paired confidence interval includes no benefit.

## WAS THAT CHECKED
Yes. Paired scalar-versus-vectorized timings were checked at batch sizes 1, 32, and 128; batch size 1 produced the refuting result.

## EVIDENCE
The headline verdict is `complete_positive_vectorized_full_service_benefit`. For `batch_size` `1`, `full_service_benefit` is `false`; `complete_ratio_scalar_over_vector` has `estimate` `0.9844497695795345` and `interval_95` `[0.9614007200306944, 1.0074236343567287]`. The larger-batch benefit is specifically anchored by `primary_batch_size` `128`.

## RECOMMENDATION
CORRECT_THE_RECORD

## experiment_7408_v649_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable to a comparative scientific claim; the administrative reconciliation would be falsified by fewer than fourteen accounted dispositions or by presenting unavailable, disqualified, or circular evidence as scientific success.

## WAS THAT CHECKED
Yes, for the administrative reconciliation: the artifact checks fourteen roster entries and dispositions, preserves invalid rows as unavailable or disqualified, and assigns zero scientific value. No comparative efficacy claim is made that requires a rival baseline.

## EVIDENCE
`"capstone_complete_score": 1`, `"scientific_value_score": 0`, `"verdict_class": "disqualified"`, `"model_invoked": false`, `"generation_calls_attempted": 0`, `"promotion_score": 0`, `"live_model_benefit_established": false`, `"verdict_class": "circular_positive"`, `"accepted_for_science": false`, `"observed": 14`, `"expected": 14`

## RECOMMENDATION
KEEP
