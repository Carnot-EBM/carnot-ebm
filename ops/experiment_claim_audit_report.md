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
| NO_CLAIM | 3 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_7853_v682_natural_runtime.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A passing source gate or completed natural measurement rows would contradict the reported block. There is no method-benefit claim to refute.

## WAS THAT CHECKED
Yes. The source gate failed, and the sample budget records zero started or completed rows.

## EVIDENCE
`claim_scope`: `blocked_before_natural_measurement`; `verdict_class`: `blocked`; `source_boundary_ready_score`: `observed`: `0`; `sample_size_budget`: `started`: `0`, `completed`: `0`; `decision_benefit`: `null`.

## RECOMMENDATION
KEEP

## experiment_7854_v682_intervention_protocol.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7855_energy_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental claim to refute. The gate receipt itself would be contradicted if its evaluated gates had passed while it reported a block.

## WAS THAT CHECKED
Yes. The artifact lists six evaluated gates, four of which failed, supporting the reported block.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `gate_check_summary`: `gate-unsat(final): 4 of 6 gate(s) failed; first failure: exp7852-source-boundary.source_boundary_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_7857_qwen_sufficiency.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no Qwen result to falsify. The gate receipt would be contradicted if it reported a blocked run while every prerequisite gate passed.

## WAS THAT CHECKED
Yes, for the gate receipt: five of six gates failed. No Qwen decisions were measured.

## EVIDENCE
`status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `gate_check_summary`: `gate-unsat(final): 5 of 6 gate(s) failed; first failure: exp7852-source-boundary.source_boundary_ready_score (actual=0 == expected=1)`; `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_7860_v682_arc_supervisor_delta.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The ARC supervisor ledger observed no new supervisor outcomes or level solves across the evaluated upstream artifacts.

## WHAT WOULD REFUTE IT
Any observation of supervisor firings (`firings` > 0), new level solves (`new_level_solves` > 0), or an active supervisor outcome (`no_new_outcomes` set to false) arising from an eligible upstream row.

## WAS THAT CHECKED
Yes; candidate rows were checked in `outcome_rows` during the `preconditions_and_reduce` phase, where all 3 candidate upstream sources were evaluated, found to be from a disqualified producer, and excluded, confirming zero firings and zero new level solves.

## EVIDENCE
`"honest_verdict": "complete_null_no_new_supervisor_outcomes"`
`"verdict_class": "null"`
`"claim_scope": "observational live supervisor ledger; no causal action saving or new solve"`
`"no_new_outcomes": true`
`"firings": 0`
`"new_level_solves": 0`
`"status": "excluded"`
`"reason": "disqualified_producer"`
`"eligible": 0`
`"excluded": 3`
`"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_7862_v682_hardware_evidence.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
No new hardware execution or measured whole-service benefit exists across the evaluated boards, maintaining an honest null status of dated read-only capability custody.

## WHAT WOULD REFUTE IT
The observation of any new qualified device execution, a resolved physical blocker on GateMate yielding a valid IDCODE, a qualified current service cost artifact, or any evaluated board showing changed evidence or a measured whole-service hardware advantage.

## WAS THAT CHECKED
Yes; checked across `change_trigger_rows` (all evaluated targets confirmed `changed_evidence_present` false), `board_rows` (all boards confirmed `current_hardware_execution` false, GateMate blocked by `0xffffffff`), `service_check` (confirmed `qualified` false and status `missing`), and `new_device_execution_count` (0).

## EVIDENCE
- `"honest_verdict"`: `"complete_null_historical_board_scope"`
- `"verdict_class"`: `"null"`
- `"new_device_execution_count"`: `0`
- `"current_hardware_execution"`: `false`
- `"acquisition_relevance"`: `"defer: no measured board whole-service benefit"`
- `"status"`: `"historical_read_only"`
- `"hardware_advantage"`: `"unmeasured"`
- `"new_execution"`: `"none"`
- `"current_service"`: `"missing"`
- `"board_evidence"`: `"dated read-only capability custody"`
- `"changed_evidence_present"`: `false`
- `"blocker"`: `"0xffffffff"`
- `"qualified"`: `false`

## RECOMMENDATION
KEEP

## experiment_7863_v682_independent_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The independent audit completed, but the required V682 scientific evidence remains blocked.

## WHAT WOULD REFUTE IT
Eligible upstream science producers with completed independent, labeled results satisfying the required evidence gates would refute the blocked verdict.

## WAS THAT CHECKED
Yes. The audit checked upstream preconditions and reduced producer rows; the reported science producers were disqualified or blocked, with no labeled predictions in the shown reductions.

## EVIDENCE
`honest_verdict` `complete_blocked_required_v682_science`  
`verdict_class` `blocked`  
`milestone_evidence_complete_score` `0`  
`labeled_prediction_count` `0`  
`status` `disqualified` `blocked`  
`validity` `false`

## RECOMMENDATION
KEEP

## experiment_7864_v682_capstone.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The capstone is complete, but the required v682 science remains blocked.

## WHAT WOULD REFUTE IT
Qualified required science producers, completed dependent runs, and measured scientific gates in the capstone’s own rows would contradict the blocked verdict.

## WAS THAT CHECKED
Yes. The capstone checks upstream dispositions and gate results; it records a disqualified source producer, a blocked dependent run, and unmeasured scientific gates.

## EVIDENCE
`"honest_verdict": "complete_blocked_required_v682_science"`; `"status": "disqualified"` for `"exp7852-source-boundary"`; `"status": "blocked"` and `"completed": 0` for `"exp7853-natural-runtime"`; `"decision_benefit": null`; `"milestone_benefit_score": 0`.

## RECOMMENDATION
KEEP
