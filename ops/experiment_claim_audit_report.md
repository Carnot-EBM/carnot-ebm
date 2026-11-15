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

## experiment_9068_v774_dispatch_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim. An authority mismatch, duplicate dispatch key, or failure to select B in the new replay would contradict administrative readiness.

## WAS THAT CHECKED
Yes, for the visible administrative checks: task objects match, the shown dispatch key occurs once, and the new fixture selects B while the old fixture does not. These checks support a receipt, not scientific benefit.

## EVIDENCE
- `claim_scope`: `Administrative dispatch custody and historical fitted-method qualification only; no new scientific benefit.`
- `honest_verdict`: `complete_null_v774_dispatch_contract`
- `metric`: `ordered_full_object_match`; `matched`: `true`
- `queue_occurrences`: `1`
- `fixture_only`: `true`; `selected_id`: `exp9071-collect-b`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

Elided rows and referenced files were not checked; fixture replay alone does not establish production behavior.

## experiment_9069_v774_source_custody.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. The stated custody policy would be contradicted by granting readiness while complete access or invocation authority remained unproven.

## WAS THAT CHECKED
Yes, for custody readiness: the inventories record incomplete authority, readiness scores are zero, and the verdict is blocked. Oracle use supports no asserted verifier-benefit claim. Elided entries were not inspected; custody readiness remains unproven.

## EVIDENCE
- `claim_scope`: `Transport and source custody only; graph truth is an executable oracle. No detector, human correctness or generalization result.`
- `honest_verdict`: `complete_blocked_a_access_coverage`
- `complete_access_authority`: `false`
- `complete_no_attempt_authority`: `false`
- `a_custody_ready_score`: `0`
- `b_collection_ready_score`: `0`
- `exclusion_reason`: `a_access_coverage_unproven`

## RECOMMENDATION
KEEP

## experiment_9070_v774_adjudication_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative result is asserted. Under the frozen protocol, a future benefit claim would fail if any improvement lower bound against prevalence, projected linear, or full linear were zero or negative.

## WAS THAT CHECKED
No comparative measurement occurred. The authority audit blocked execution; all 4,377 intended targets were excluded, with zero completed units. The comparator arms and benefit criterion describe planned methods.

## EVIDENCE
- `honest_verdict`: `complete_blocked_base_question_family_authority`
- `verdict_class`: `blocked`
- `completed_count`: `0`
- `intended_count`: `4377`; `excluded_count`: `4377`
- `targets_opened`: `false`
- `claim_scope`: `Pinned MDHJudgments authority audit and frozen human-target methods; no fitted head, medical correctness or general verifier moat.`

## RECOMMENDATION
KEEP

## experiment_9071_collect_b.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is made. This artifact records a blocked collection attempt, so there is no experimental success claim to falsify.

## WAS THAT CHECKED
The prerequisite was checked and failed: collection readiness was 0 against a required 1. The artifact reports blocking before execution; no comparative experiment was tested.

## EVIDENCE
`blocked_gate_check_v1`, `blocked`, `blocked_gate_check_failed`, `b_collection_ready_score`, `actual=0 == expected=1`, `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9073_human_features.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked prerequisite-check receipt, with no claim of successful feature capture or comparative value.

## WAS THAT CHECKED
No experimental claim was tested. Both prerequisite gates were checked and failed: each required 1 and observed 0.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`. Both `gates_evaluated` entries report `expected`: `1`, `actual`: `0`, `passed`: `false`. `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_9079_v774_hardware_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Qualified, correctness-matched current-device whole-service timings in its own rows would contradict the recorded evidence-blocked status. There is no comparative performance or verifier-value claim to falsify.

## WAS THAT CHECKED
No empirical performance comparison is reported. The displayed rows contain unmeasured, null timings; the artifact records zero completed and 35 censored observations. This supports treating it as an evidence-inventory receipt. Elided rows and upstream contents were not inspected.

## EVIDENCE

- `claim_scope`: `Workload feasibility only. Graph truth is executable; graph wins are circular_positive. No verifier benefit, purchase, 100x or actual-device speedup claim.`
- `honest_verdict`: `complete_blocked_current_cost`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `completed_count`: `0`
- `censored_count`: `35`
- `current_device_implementation_measured`: `false`
- `value`: `null`
- `physical_acceleration_demonstrated`: `false`

## RECOMMENDATION
KEEP

## experiment_9080_arc_transfer.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked-run receipt, with no claim about transfer performance.

## WAS THAT CHECKED
The readiness gate was checked and failed: observed 0 versus required 1. Transfer performance was not evaluated, so the artifact supports no conclusion about transfer.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `arc_new_outcomes_ready_score`; `failed_expected`: `1`; `failed_observed`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_9081_v774_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit is asserted. A tie or loss against projected or full linear would defeat a benefit claim, but this artifact reports administrative completion and blocked scientific evaluation.

## WAS THAT CHECKED
No. The human comparator bounds are unmeasured, and no new ARC gameplay occurred. These limitations are preserved as blocked outcomes rather than converted into a positive verdict.

## EVIDENCE
- `claim_scope`: `Fourteen administrative outcomes. No measured graph, selected human-panel, broad no-cheap-oracle or complete-cost benefit. Graph wins remain circular_positive; human agreement is neither medical correctness nor general verifier benefit.`
- `honest_verdict`: `complete_blocked_declared_gate`
- `status`: `unmeasured`
- `new_gameplay_executed`: `false`
- `publication_scope`: `historical_FoVer`
- `current_science_paper_ready`: `false`

## RECOMMENDATION
KEEP

Upstream experiments were not checked; this receipt establishes no comparative scientific benefit.
