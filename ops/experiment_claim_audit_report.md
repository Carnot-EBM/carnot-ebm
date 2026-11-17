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
| CLAIM_SUPPORTED | 1 |
| NO_CLAIM | 7 |

## experiment_9168_forward_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A passing sole prerequisite gate would contradict the reported blocking diagnosis. This receipt asserts no method-performance or comparative-value claim.

## WAS THAT CHECKED
Yes—the sole gate records actual 0 against expected 1 and reports failure. Execution was blocked at the prerequisite gate; validation performance was not tested.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `causal_fix_ready_score`
- `expected`: `1`; `actual`: `0`; `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9178_v781_arc_mechanic_routing.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The development controls show routing invariance and preserved legacy fallbacks, while overall readiness remains disqualified.

## WHAT WOULD REFUTE IT
A recognized case whose resulting operator list differs from its empty-game reference, or an unknown/absent-mechanic control whose routing changes from the frozen baseline.

## WAS THAT CHECKED
Yes: the artifact reports 52 recognized cases and 72 legacy controls, with zero post-change mismatches versus 22 beforehand. The prior implementation’s mismatches demonstrate that this comparison could fail.

The failed consumer validation is acknowledged through disqualification, rather than presented as readiness. No verifier-value or generalization improvement is claimed.

## EVIDENCE
- `allowed_claims`: `deterministic routing invariance`, `legacy fallback preservation`
- `recognized_total`: `52`; `legacy_controls`: `72`
- `before_mismatches`: `22`; `after_mismatches`: `0`
- `excluded_count`: `0`
- `verifier_is_oracle`: `false`
- `exposure_scope`: `exposed_development`
- `direct_consumers`: `actual_exit` = `1`, `passed` = `false`
- `required_checks_passed`: `false`
- `honest_verdict`: `complete_disqualified_arc_mechanic_routing`

## RECOMMENDATION
KEEP

Only the supplied artifact was reviewed; failed consumer validation leaves readiness disqualified.

## experiment_9179_v781_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. Crediting failed controls or unmeasured predictions as scientific success would contradict the artifact’s administrative accounting.

## WAS THAT CHECKED
Yes, in the visible accounting: declared gates fail on zero or missing operands, ARC controls are disqualified, and scientific measurements remain unmeasured. These outcomes receive no claimed scientific benefit.

## EVIDENCE

- `allowed_claims`: `fourteen terminal administrative outcomes`
- `unmeasured_claims`: `detector benefit`, `continuous learning benefit`, `live ARC generalization`
- `attempted_scientific_answers`: `0`
- `forward_validation_ready_score`: `observed` is `missing_field`, and `passed` is `false`
- `scope`: `offline deterministic method controls; no game or independent solve evidence`
- `current_science_paper_ready`: `false`

## RECOMMENDATION
KEEP

Not checked: elided rows or upstream artifacts; this assessment is limited to the supplied excerpt.

## experiment_9180_v782_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific or comparative claim is made. A mismatch between the required ordered task objects and the recorded contract would contradict administrative readiness.

## WAS THAT CHECKED
Yes, partially: visible gate_check_summary entries compare expected and observed task counts, hashes, and milestones. These support administrative consistency; scientific benefit is explicitly unmeasured.

## EVIDENCE
- `complete_null_v782_contract`
- `administrative task binding`
- `14 exact ordered objects including prompts, gates, deliverables and model metadata; independent of scientific availability`
- `unmeasured; fixtures and historical receipts grant no independent benefit`
- `task_count`, `expected`, `observed`, `passed`, `true`

## RECOMMENDATION
KEEP

Only the supplied excerpt was reviewed; elided entries leave full administrative coverage unverified.

## experiment_9181_v782_causal_validation.json

**NO_CLAIM**

## VERDICT

NO_CLAIM

## THE HEADLINE CLAIM

no claim

## WHAT WOULD REFUTE IT

There is no comparative headline to falsify. A claim that the correction was ready would be refuted by failed affected checks or incomplete required obligations; the artifact explicitly withholds that claim.

## WAS THAT CHECKED

Yes, qualification checks had a real chance to fail and did: the repository-health command timed out, required and owned checks are marked unsuccessful, and obligations remain pending. The artifact reports disqualification rather than promoting diagnostic measurements into readiness or generalization.

## EVIDENCE

- `allowed_claims`: `Executed diagnostic operations and exact remaining obligations only`
- `honest_verdict`: `complete_disqualified_causal_validation`
- `causal_fix_ready_score`: `0`
- `owned_checks_passed`: `false`
- `required_checks_passed`: `false`
- `repository_health`: `passed`: `false`, `timed_out`: `true`
- `remaining_obligations`: `current_qualification_credit`: `0`, `status`: `pending`
- `independent_generalization_score`: `0`

## RECOMMENDATION

KEEP

Unchecked: elided rows and external receipts. This review establishes no broader qualification.

## experiment_9182_forward_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No scientific claim is asserted. A passing prerequisite gate would contradict the reported reason for blocking execution.

## WAS THAT CHECKED
Yes—the prerequisite gate was evaluated and failed: observed 0 versus required 1. This is a blocked-execution receipt, not a validation result; experimental performance was not assessed.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `causal_fix_ready_score`
- `expected`: `1`; `actual`: `0`; `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9192_v782_arc_mechanic_routing.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A recognized output differing from its frozen reference, or a changed legacy fallback, would falsify the limited routing controls. There is no comparative solver-value headline: the artifact records disqualification and grants no readiness or generalization credit.

## WAS THAT CHECKED
Yes for the routing controls: the summary reports 22 mismatches before and zero after, with 52 recognized cases and 72 legacy controls. Required validation failed, and the artifact records that failure rather than declaring readiness.

The dispatcher matches its baseline and explicitly reports no added value. No held-out or live efficacy comparison is claimed. Elided rows prevent a complete row-level audit.

## EVIDENCE

- `honest_verdict`: `complete_disqualified_arc_ordering_compatibility`
- `verdict_class`: `disqualified`
- `mechanic_routing_ready_score`: `0`; `required_checks_passed`: `false`
- `scope`: `Development controls only. No model, SDK episode, level solve, leaderboard score or generalization improvement measured.`
- `before_mismatches`: `22`; `after_mismatches`: `0`; `recognized_total`: `52`; `legacy_controls`: `72`
- `baseline_approach`: `default_graph_explore`; `executed_approach`: `default_graph_explore`; `value_added`: `false`

## RECOMMENDATION
KEEP

Not rerun; elided rows were not individually checked. The routing receipt establishes no solver benefit.

## experiment_9193_v782_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to falsify. Its administrative accounting would be contradicted by crediting blocked, disqualified, or unmeasured work as scientific success.

## WAS THAT CHECKED
Yes, in the visible gate accounting: observed-zero and missing-field prerequisites both fail. ARC observations remain disqualified, CES comparisons remain unmeasured, and scientific collection records zero attempts. These are administrative outcomes, not evidence of efficacy or generalization.

## EVIDENCE
- `allowed_claims`: `fourteen terminal administrative outcomes`; `unmeasured primary science and secondary CES recorded separately`
- `scope`: `Offline deterministic method controls; no independent solve evidence`
- `verdict_class`: `disqualified`
- `status`: `unmeasured`
- `attempted_scientific_answers`: `0`
- `operand_status`: `observed_zero`; `missing_field`
- `current_science_paper_ready`: `false`

## RECOMMENDATION
KEEP

Unchecked: elided rows and upstream artifacts; full accounting consistency remains unverified.
