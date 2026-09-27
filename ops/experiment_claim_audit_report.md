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

## experiment_7770_v676_qwen_runner_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
The artifact records CPU fixture conformance on an exposed development panel and makes no comparative value claim.

## WHAT WOULD REFUTE IT
A failed required protocol check or invalid fixture rows would refute the limited conformance claim.

## WAS THAT CHECKED
Yes, through the validity gate, row validity fields, and validation receipts. Semantic benefit was left unclaimed.

## EVIDENCE
`"claim_scope": "CPU fixture conformance; exposed development panel; no semantic benefit"`; `"validity"` `"passed": true`; `"readiness"` `"passed": false`; `"qwen_runner_ready_score": 0`; `"verdict_class": "disqualified"`

## RECOMMENDATION
KEEP

## experiment_7771_view_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; as a pre-flight execution receipt recording a blocked gate check, the artifact makes no comparative or empirical claim to refute.

## WAS THAT CHECKED
No; the run was blocked prior to execution at the conductor pre-gate layer and never ran.

## EVIDENCE
`blocked_gate_check_v1`
`blocked`
`blocked_gate_check_failed`
`0.0`
`conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7773_qwen_event_confidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no result claim to refute. A later claim of added value would be refuted if explicit source-support probability tied or lost to generic confidence.

## WAS THAT CHECKED
No. The experiment was blocked before the comparison ran.

## EVIDENCE
`"status"` `"blocked"` `"honest_verdict"` `"blocked_gate_check_failed"` `"blocked_at_layer"` `"conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_7775_v676_independent_evidence_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative claim to falsify. The blocking status would be contradicted if the artifact’s own producer rows contained present, eligible evidence while it still reported the producers as missing.

## WAS THAT CHECKED
Yes, for producer custody: the gate summary, source records, and rows record both required producers as missing. No scientific comparison was possible.

## EVIDENCE
`honest_verdict`: `complete_blocked_required_v676_evidence`; `producer_states`: `missing`; `arm`: `custody`; `metrics`: `null`; `effective_independent_n`: `0`; `decision_benefit`: `null`.

## RECOMMENDATION
KEEP

## experiment_7776_v676_arc_runner_qualification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The runner is disqualified from readiness because required validation failed.

## WHAT WOULD REFUTE IT
Passing results for the required current validation checks, with a valid and ready acceptance gate.

## WAS THAT CHECKED
Yes. The current validation results report failures for both required checks, and the acceptance gate reports no readiness.

## EVIDENCE
`honest_verdict`: `complete_disqualified_required_runner_validation`; `verdict_class`: `disqualified`; `ruff_format`: `observed` `1`, `expected` `0`, `passed` `false`; `scoped_spec_coverage`: `observed` `1`, `expected` `0`, `passed` `false`; `validity`: `false`; `readiness`: `false`.

## RECOMMENDATION
KEEP

## experiment_7777_arc_organic_measurement.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A completed organic selection measurement would contradict the artifact’s report that the run stopped at the gate. There is no selection result here to falsify.

## WAS THAT CHECKED
Yes. The artifact records three gate evaluations, two of which failed, and reports no measurement.

## EVIDENCE
`status`: `blocked`; `blocked_at_layer`: `conductor_pre_gate`; `gate_check_summary`: `gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp7776-arc-runner-qualification.organic_runner_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_7779_v676_hardware_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or hardware advantage claim to refute.

## WAS THAT CHECKED
No comparative test was attempted. The artifact records historical custody and disqualifies readiness because required checks failed.

## EVIDENCE
`hardware_advantage_claimed`: `false`; `hardware_operations_issued`: `[]`; `service_opportunities`: `null`; `required_checks_passed`: `false`; `honest_verdict`: `complete_disqualified_required_checks`.

## RECOMMENDATION
KEEP

## experiment_7780_v676_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Evidence of a comparative experimental claim or an assertion of capability/readiness being made without valid empirical support; for this bookkeeping artifact, refutation would require demonstrating that the artifact claims substantive scientific progress or model superiority over a baseline despite only aggregating task accounting.

## WAS THAT CHECKED
No. The artifact checks administrative prerequisites, pipeline receipts, and upstream validation gates—which failed and resulted in disqualification—without executing or testing any comparative experimental claims.

## EVIDENCE
`honest_verdict`
`complete_disqualified_v676_capstone_validation`
`verdict_class`
`disqualified`
`inference_substrate`
`aggregation_from_upstream_artifacts`
`inference_substrate_class`
`aggregation`
`model_invoked`
`false`
`capstone_complete_score`
`0`
`arm`
`task_accounting`
`validation_receipts`
`Completed bookkeeping cannot substitute for completed science.`
`host aggregation; no model or board invocation`

## RECOMMENDATION
KEEP
