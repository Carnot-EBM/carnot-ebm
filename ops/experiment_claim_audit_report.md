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

## experiment_9153_v780_validation_schedule.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to falsify. Completed full qualification or an established subprocess cause would contradict the reported blocked status.

## WAS THAT CHECKED
The visible receipts document partial execution, while qualification remains deferred: 32 completed, 642 censored, and 674 intended. Oracle-defined fixture checks are administrative evidence; no verifier benefit or independent generalization is claimed.

## EVIDENCE
- `honest_verdict`: `complete_blocked_validation_prerequisite`
- `scientific_benefit_claimed`: `false`
- `exposure_scope`: `Exposed human development; administrative schedule only`
- `full_674_node_qualification_score`: `0`
- `computational_root_cause`: `unknown; retained stack shows a subprocess wait only`
- `Exact fixture expectations are circular administrative evidence.`

## RECOMMENDATION
KEEP

Not checked: elided content or external files; this review covers only the supplied artifact.

## experiment_9154_forward_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No experimental claim is asserted. A claim of completed validation would conflict with the failed prerequisite, but this artifact reports execution as blocked.

## WAS THAT CHECKED
Yes—the sole prerequisite gate was evaluated and failed. The artifact records that failure without asserting a validation result.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`.

`artifact_field`: `schedule_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`.

`blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

## experiment_9165_v780_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. Completed scientific measurements contradicting the stated unmeasured status would challenge its administrative accounting.

## WAS THAT CHECKED
The visible accounting reports 3,963 intended scientific requests, all unstarted, with zero attempted or completed. Administrative gates visibly fail; a missing operand is explicitly identified rather than silently treated as zero. These outcomes support the blocked status, without establishing scientific performance.

## EVIDENCE

- `Fourteen administrative outcomes. Scientific static and online predictions and complete costs remain unmeasured; measured validation remains blocked. No ARC solve, hardware benefit or general theorem.`
- `scientific_request_counts`: `attempted`: `0`, `completed`: `0`, `intended`: `3963`, `unstarted`: `3963`.
- `operand_status`: `missing_field`; `passed`: `false`.
- `scientific_readiness`: `0`.

## RECOMMENDATION
KEEP

Unchecked: elided rows and upstream artifacts; this verdict covers only the supplied excerpt.

## experiment_9166_v781_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A mismatch between the frozen and scheduled task objects would refute administrative contract readiness. The artifact makes no comparative scientific claim to falsify.

## WAS THAT CHECKED
Yes for the displayed design-byte and task-count comparisons in `gate_check_summary`. Most checks and task objects are elided, so complete object equality cannot be assessed here. Scientific benefit is explicitly unmeasured.

## EVIDENCE
- `scope`: `Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208.`
- `condition`: `unmeasured; fixtures and historical receipts grant no independent benefit`
- `contract_ready_score`: `1`
- `current_science_paper_ready`: `false`

## RECOMMENDATION
KEEP

Not checked: elided records; full contract equality remains unverified here.

## experiment_9167_v781_causal_validation.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative efficacy claim to falsify. A claim of validated correction would be defeated by an unfinished corrected execution or failed tamper refusals.

## WAS THAT CHECKED
Partly: the corrected execution timed out; unchanged tamper controls were unattempted. The artifact records these limitations as censoring, zero readiness, and disqualification. It claims only diagnostic observations, so the incomplete comparison does not contradict its stated scope.

## EVIDENCE
- `allowed_claims`: `Executed diagnostic operations and exact remaining obligations only`
- `honest_verdict`: `complete_disqualified_causal_validation`
- `causal_fix_ready_score`: `0`
- `completed_count`: `0`
- Corrected receipt: `normal_exit`: `false`, `timed_out`: `true`
- Controls row: `condition`: `corrected_controls`, `attempted`: `false`, `status`: `censored`

## RECOMMENDATION
KEEP

Review limited to the supplied artifact; no rerun performed. Corrective efficacy remains unestablished.

## experiment_9168_forward_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The blocking diagnosis would be contradicted by the recorded prerequisite meeting its required value while being marked failed. This receipt makes no experimental efficacy claim.

## WAS THAT CHECKED
Yes—the single gate records an observed 0 against required 1 and correctly reports failure. Experimental outcomes were not checked because execution was blocked.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"artifact_field": "causal_fix_ready_score"`, `"expected": 1`, `"actual": 0`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`.

## RECOMMENDATION
KEEP

## experiment_9178_v781_arc_mechanic_routing.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A recognized case whose output differs from its empty-game reference, or an unknown/absent-mechanic control whose output changes, would refute the local regression assertions. The artifact makes no comparative solve or generalization claim and explicitly disqualifies readiness.

## WAS THAT CHECKED
Yes, for the local assertions: the artifact reports 52 recognized cases, 72 legacy controls, and zero after-mismatches. The previous implementation’s 22 mismatches demonstrate that this check could fail.

Required validation also had a real chance to fail: direct-consumer checks failed, and the artifact retained its disqualified verdict. These are development controls, with no claimed verifier advantage or held-out performance.

Review is limited to the supplied text; elided rows and referenced logs were not inspected.

## EVIDENCE
- `scope`: `Development controls only. No model, SDK episode, level solve, leaderboard score or generalization improvement measured.`
- `before_mismatches`: `22`; `after_mismatches`: `0`; `recognized_total`: `52`; `legacy_controls`: `72`.
- `name`: `direct_consumers`; `passed`: `false`; `actual_exit`: `1`.
- `required_checks_passed`: `false`.
- `honest_verdict`: `complete_disqualified_arc_mechanic_routing`; `verdict_class`: `disqualified`.
- `mechanic_routing_ready_score`: `0`; `independent_generalization_score`: `0`.

## RECOMMENDATION
KEEP

## experiment_9179_v781_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. The administrative account would be contradicted by crediting failed gates, unavailable measurements, or deterministic fixture equality as demonstrated scientific benefit.

## WAS THAT CHECKED
Yes, in the displayed accounting: zero and missing gate operands fail, ARC controls are disqualified, and scientific benefits remain explicitly unmeasured. This is a blocked administrative receipt, not a positive result requiring a rival baseline.

Review covers only the supplied excerpt; elided rows and upstream artifacts were not inspected.

## EVIDENCE

- `allowed_claims`: `fourteen terminal administrative outcomes`
- `unmeasured_claims`: `detector benefit`, `continuous learning benefit`
- `attempted_scientific_answers`: `0`
- `operand_status`: `missing_field`; `passed`: `false`
- `arc_summary`: `verdict_class`: `disqualified`
- `scope`: `offline deterministic method controls; no game or independent solve evidence`
- `current_science_paper_ready`: `false`

## RECOMMENDATION
KEEP
