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
| NO_CLAIM | 1 |
| SKIPPED_ALREADY_FLAGGED | 4 |

## experiment_7371_v647_proof_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The synthetic proof-boundary protocol is complete and ready, while learning value and live-cohort efficacy remain unmeasured.

## WHAT WOULD REFUTE IT
A required synthetic boundary gate failing—such as an accepted authority attack, invalid SAT output, false-proof rejection, stale-version effect, incomplete rows, insufficient erasure witnesses, lost exact-decision coverage, or failure against the persistent exact-solver or reachability-cache cost controls—would refute boundary readiness.

## WAS THAT CHECKED
Yes. The artifact checks synthetic safety, protocol and row completeness, erasure witnesses, exact-decision coverage, adversarial authority attacks, and cost ratios against two serious persistent comparators. The oracle defines formal correctness, but the artifact does not claim that the verifier or learning system adds value; it explicitly reports zero learning value and says the live cohort was not collected.

## EVIDENCE
`"honest_verdict": "complete_null_proof_boundary_ready_learning_value_not_measured"`; `"proof_boundary_ready_score": 1`; `"learning_value_score": 0`; `"live_cohort_complete": false`; `"live_proposal_cohort"`; `"observed": false`; `"status": "frozen_not_collected_by_exp7371"`; `"model_invoked": false`; `"verifier_is_oracle": true`; `"rows_complete": true`; `"safety_passed": true`; `"protocol_complete": true`; `"authority_controls_passed": true`; `"exact_decision_coverage": 1.0`; `"false_proof_rejections": 0`; `"invalid_sat_outputs": 0`; `"stale_version_effects": 0`; `"erasure_witness_count": 608`; `"persistent_incremental_exact_solver"`; `"persistent_source_graph_reachability_cache"`; `"synthetic_rows_censored": 0`; `"synthetic_rows_completed": 3840`

## RECOMMENDATION
KEEP

## experiment_7372_v647_qwen_canary.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7373_proposal_capture.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no outcome claim to falsify; if the title were treated as claiming successful proposal capture, a blocked pre-gate with no captured proposals would refute it.

## WAS THAT CHECKED
No; execution stopped at the pre-gate, so no method outcome or comparative test occurred.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"blocked_at_layer": "conductor_pre_gate"`, `"duration_s": 0.0`

## RECOMMENDATION
KEEP

## experiment_7376_v647_arc_outcomes.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7377_v647_ising_law.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Exact enumeration establishes that the fixture is ready and that proof assistance preserves the source law.

## WHAT WOULD REFUTE IT
An independently computed energy or probability mismatch, an undetected intentional law mutation, or proof assistance performing no better than the source-only baseline would refute the claimed added value.

## WAS THAT CHECKED
No. Exhaustive rows and negative controls were checked, but correctness was defined by the same oracle used to certify it; no independent verifier was used. Moreover, the serious source-only baseline was present and tied the proof-assisted arm exactly.

## EVIDENCE
`"verdict_class": "circular_positive"`; `"verifier_is_oracle": true`; `"inference_substrate": "host_cpu_exact_2cnf_enumeration_no_model"`; `"condition": "source_only"`; `"condition": "proof_assisted_source_only"`; `"compiled_normalizer": 36.17717151469089`; `"source_normalizer": 36.17717151469089`; `"source_law_total_variation": 0.0`; `"law_fixture_ready_score": 1`; `"promotion_score": 0`

## RECOMMENDATION
NARROW_CLAIM

## experiment_7378_v647_ising_audit.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_7379_v647_hardware_envelope.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The run is blocked because no qualifying post-Exp6559 GateMate physical-state receipt exists, all three board dispositions were accounted for, and unavailable placement inputs support no speed claim.

## WHAT WOULD REFUTE IT
A qualifying operator-authored GateMate cable, port, board, power, JTAG, or DirtyJTAG change after Exp6559; an unaccounted required board; or an eligible measured placement row showing hardware gain would refute the headline.

## WAS THAT CHECKED
Yes. The artifact checked the specified physical-state provenance, explicitly accounted for KV260, GateMate, and PolarFire, and inspected both planned placement sources. Those checks could have found a qualifying receipt or eligible placement measurement, but instead found no accepted receipt and two unavailable placement inputs. The oracle defines record truth, but the artifact makes no claim that this verifier adds hardware value.

## EVIDENCE
`"changed_state_receipt": null`; `"accepted_receipt_count": 0`; `"passed": false`; `"board_accounting"`; `"observed": ["KV260", "GateMate", "PolarFire"]`; `"placement_sources_attempted": 2`; `"placement_sources_eligible": 0`; `"placement_sources_unavailable": 2`; `"performance_evidence_eligible": false`; `"hardware_value_score": 0`; `"new_hardware_runs_attempted": 0`; `"new_hardware_execution_claimed": false`; `"verifier_is_oracle": true`; `"verdict_class": "blocked"`

## RECOMMENDATION
KEEP

## experiment_7380_v647_capstone.json

**SKIPPED_ALREADY_FLAGGED**
