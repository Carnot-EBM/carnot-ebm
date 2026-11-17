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

## experiment_9181_v782_causal_validation.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit is claimed. A readiness claim would be refuted by an affected validation command failing or timing out; readiness is already withheld here.

## WAS THAT CHECKED
Yes, qualification had a real chance to fail: repository_health timed out, required checks did not pass, and readiness received zero credit. The diagnostic comparisons are not promoted into a general performance claim.

## EVIDENCE
- `allowed_claims`: `Executed diagnostic operations and exact remaining obligations only`
- `honest_verdict`: `complete_disqualified_causal_validation`
- `causal_fix_ready_score`: `0`
- `required_checks_passed`: `false`
- `repository_health`: `passed`: `false`, `timed_out`: `true`
- `scope`: `One paired operation observation; callback counters retained in immutable traces; no general speedup claim`

## RECOMMENDATION
KEEP

Not checked: external receipts or reruns; this assessment covers the supplied, truncated artifact.

## experiment_9182_forward_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No experimental value claim is made. A passing prerequisite gate would contradict the reported reason for blocking execution.

## WAS THAT CHECKED
Yes. The gate required 1, observed 0, and recorded failure. This is a blocked-execution receipt, not a completed validation result.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `expected`: `1`; `actual`: `0`; `passed`: `false`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

Upstream contents were not inspected; this receipt establishes no experimental outcome.

## experiment_9192_v782_arc_mechanic_routing.json

**CLAIM_SUPPORTED**

## VERDICT

CLAIM_SUPPORTED

## THE HEADLINE CLAIM

The development controls show deterministic routing invariance and legacy fallback preservation, while failed required validation disqualifies readiness.

## WHAT WOULD REFUTE IT

A recognized mechanic producing a different operator order from its empty-game reference, or an unknown/absent mechanic changing its legacy output.

## WAS THAT CHECKED

Yes. The artifact reports 52 recognized permutations and 72 unchanged legacy controls. The frozen predecessor produced 22 mismatches; the revised selector produced zero, demonstrating that the comparison could fail.

Required validation also had a real chance to reject readiness: both owned tests and direct-consumer tests failed, and the verdict records disqualification. The dispatcher baseline ties the selected candidate, with added value explicitly false. No generalization or verifier-value improvement is claimed.

## EVIDENCE

- `allowed_claims`: `deterministic routing invariance`, `legacy fallback preservation`
- `comparison`: `52 recognized permutations;72 unchanged legacy controls;8 mechanic groups`
- `before_mismatches`: `22`; `after_mismatches`: `0`
- `value_added`: `false`
- `required_checks_passed`: `false`
- `honest_verdict`: `complete_disqualified_arc_ordering_compatibility`
- `scope`: `Development controls only. No model, SDK episode, level solve, leaderboard score or generalization improvement measured.`

## RECOMMENDATION

KEEP

Elided rows and implementation were not independently checked; this assessment covers only the supplied development-control evidence.

## experiment_9193_v782_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. The administrative account would be contradicted by current rows showing completed scientific execution despite the reported execution deficit, or disqualified observations counted as independent scientific support.

## WAS THAT CHECKED
Yes, within the displayed accounting: failed prerequisites block downstream work, missing operands fail gates, CES comparisons remain unmeasured, and ARC observations remain disqualified. Zero scientific execution is reported as missing evidence, not a measured null effect. No superiority or generalization claim is advanced.

## EVIDENCE

- `fourteen terminal administrative outcomes`
- `unmeasured primary science and secondary CES recorded separately`
- `attempted_scientific_answers`: `0`
- `independent_generalization_score`: `0`
- `Offline deterministic method controls; no independent solve evidence`
- `verdict_class`: `disqualified`
- `operand_status`: `missing_field`; `passed`: `false`
- `censor_reason`: `missing_reference_and_scientific_predictions`; `status`: `unmeasured`

## RECOMMENDATION
KEEP

Elided rows and upstream files were not checked; this assessment covers the supplied excerpt.

## experiment_9194_v783_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A mismatch between the required task contracts or a failed mandatory check would defeat administrative readiness. There is no asserted scientific benefit or comparative win to falsify.

## WAS THAT CHECKED
Administrative readiness is explicitly withheld: contract readiness is zero, activation is unauthorized, and paper readiness is false. The scientific comparisons are protocols, not reported results.

Only the supplied excerpt was reviewed; elided checks cannot be assessed.

## EVIDENCE

- `scope`: `Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208.`
- `scientific_benefit`: `condition`: `unmeasured; fixtures and historical receipts grant no independent benefit`
- `contract_ready_score`: `0`
- `activation_authorized`: `false`
- `current_science_paper_ready`: `false`

## RECOMMENDATION
KEEP

## experiment_9195_corpus_prefilter.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A corpus match discarded by the prefilter would refute successful lossless qualification. This artifact reports no qualification result.

## WAS THAT CHECKED
No. Execution stopped at the prerequisite gate because contract_ready_score was 0 when 1 was required. Prefilter losslessness remains unchecked.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`  
`"status": "blocked"`  
`"honest_verdict": "blocked_gate_check_failed"`  
`"failed_expected": 1`  
`"failed_observed": 0`  
`"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_9197_label_ambiguity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental success or comparative claim to refute. A passing prerequisite gate would contradict the reported reason for blockage.

## WAS THAT CHECKED
Yes. The sole evaluated gate required 1, observed 0, and failed. The artifact reports stopping before experiment execution; it provides no result about label ambiguity or method value.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `source_ready_score`
- `expected`: `1`; `actual`: `0`; `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9207_v783_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. An executed, scientifically successful task classified as blocked or unmeasured would contradict the administrative account.

## WAS THAT CHECKED
The visible accounting distinguishes two executed tasks from twelve skips and explicitly denies inferred scientific success. Scientific efficacy remains untested; the artifact does not present administrative completion as efficacy.

## EVIDENCE
- `allowed_claims`: `fourteen terminal administrative outcomes`
- `executed`: `2`; `skipped`: `12`
- `scientific_success_inferred`: `false`
- `attempted_scientific_answers`: `0`
- `scope`: `Offline deterministic method controls; no independent solve evidence`
- `verdict_class`: `disqualified`

## RECOMMENDATION
KEEP

Not checked: elided rows or upstream artifacts; this assessment covers only the supplied excerpt.
