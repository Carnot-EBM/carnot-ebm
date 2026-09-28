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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7812_view_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no performance claim to refute. The gate receipt’s blocked finding would be contradicted if every evaluated gate passed.

## WAS THAT CHECKED
Yes. The receipt evaluated seven gates and reports two failures; it contains no energy-head training result.

## EVIDENCE
`status` `blocked` `gate-unsat(final): 2 of 7 gate(s) failed; first failure: exp7811-training-runtime.training_runtime_ready_score (actual=0 == expected=1)` `blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7814_v679_counter_evidence_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no positive headline claim to falsify. A claim that this run demonstrated verifier benefit would be contradicted by the absence of completed panel results.

## WAS THAT CHECKED
No benefit comparison was completed. The artifact records zero started and completed cases and failed required validation.

## EVIDENCE
`"claim_scope": "Exposed development fixture only; no hidden generalization or oracle-distinct benefit."`  
`"started": 0`  
`"completed": 0`  
`"honest_verdict": "complete_disqualified_required_validation"`  
`"verdict_class": "disqualified"`

## RECOMMENDATION
KEEP

## experiment_7815_qwen_counter_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Evidence that the experiment ran or that the pre-execution gates were satisfied, but as an execution receipt recording a blocked run, the artifact asserts no comparative or empirical claim.

## WAS THAT CHECKED
No; the experiment was halted prior to execution due to upstream gate failures.

## EVIDENCE
`schema`
`blocked_gate_check_v1`
`status`
`blocked`
`honest_verdict`
`blocked_gate_check_failed`
`duration_s`
`0.0`
`blocked_at_layer`
`conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7817_v679_arc_runner_qualification.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The runner earns a positive readiness verdict from its qualification fixture, without claiming a measured gameplay benefit.

## WHAT WOULD REFUTE IT
An independently scored, completed panel in which the runner fails the required execution checks would refute readiness beyond the fixture. A gameplay value claim would also need to beat the cheap serious comparator: the off arm.

## WAS THAT CHECKED
No for independent readiness or value. Six qualification probes ran, but none of the 48 planned panel episodes completed. The verifier is the scoring oracle, and the observed arms tie on charged actions and peak level.

## EVIDENCE
`readiness`: `true`; `organic_runner_ready_score`: `1`; `verdict_class`: `circular_positive`; `verifier_is_oracle`: `true`; `selector_fixture`: `deterministic`; `qualification_probes_completed`: `6`; `effective_independent_n`: `0`; `completed`: `0`; `intended`: `48`; `decision_benefit`: `null`; `honest_verdict`: `complete_circular_positive_runner_qualified_no_benefit`.

## RECOMMENDATION
NARROW_CLAIM

## experiment_7818_v679_arc_organic_measurement.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7820_v679_hardware_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because the artifact disclaims any comparative performance or acceleration claim, there is no substantive empirical finding to refute. If construed as asserting hardware readiness or operational speedup, verified local execution logs demonstrating either an acceleration advantage over a CPU baseline or an execution failure under a matched workload would test and potentially refute the finding.

## WAS THAT CHECKED
No. The artifact lacks fresh hardware runs, benchmark executions, and local service timing measurements. It evaluated neither active hardware execution nor rival baselines; all candidate rows lack measured metrics and were never started or completed.

## EVIDENCE
- `hardware_advantage_claimed`: `false`
- `honest_verdict`: `complete_blocked_missing_service_evidence`
- `verdict_class`: `blocked`
- `terminal_scope`: `historical_board_custody_only`
- `actual_inference_substrate_class`: `aggregation`
- `arm`: `historical_accounting`
- `hardware_advantage`: `unmeasured`
- `decision`: `defer`
- `basis`: `No qualified local whole-service board benefit`
- `gate_check_summary`: `A missing producer differs from a scientific null.`
- `hardware_continuity_accounted`: `Inventory completeness is not fresh execution.`
- `acceptance_gate_results`: `Working custody does not establish benefit.`

## RECOMMENDATION
KEEP

## experiment_7821_v679_independent_evidence_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Independent V679 evidence is blocked because the required science sources are missing, leaving no eligible independent observations.

## WHAT WOULD REFUTE IT
The declared science sources being present and eligible, with independent observations available for the audit, would refute the stated reason for the block.

## WAS THAT CHECKED
Yes. The artifact checks each declared source and records all three as missing and excluded.

## EVIDENCE
`honest_verdict`: `complete_blocked_required_v679_evidence`; `Exp7813`: `missing`; `Exp7815`: `missing`; `Exp7816`: `missing`; `independent_n`: `0`; `excluded`: `3`; `independent_evidence_ready_score`: `0`.

## RECOMMENDATION
KEEP

## experiment_7822_v679_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The V679 capstone is blocked by missing or disqualified required evidence, despite passing its scoped publication gates.

## WHAT WOULD REFUTE IT
Eligible current science producers with passing required validation and evidence sufficient to open the capstone gates would refute the blocked verdict.

## WAS THAT CHECKED
Yes. The artifact records failed gate operands, producer eligibility, missing current science inputs, and branch outcomes. Its passing publication gates are scoped to the stable G1–G4 claims.

## EVIDENCE
`honest_verdict`: `complete_blocked_required_v679_evidence`; `capstone_complete_score`: `0`; `readiness`: `0`; `external_inputs_present`: `false`; `current_science_missing`; `producer_eligible`: `false`; `verdict_class`: `disqualified`; `publication`: `stable_G1_to_G4_only`.

## RECOMMENDATION
KEEP
