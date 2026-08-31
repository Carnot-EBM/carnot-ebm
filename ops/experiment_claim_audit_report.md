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
| NO_CLAIM | 2 |
| CANNOT_DETERMINE | 1 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6794_temporal_exchange_cold_hardware_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Fresh-process CPU replay supports the source experiment’s null verdict for temporal exchange, while exposing overhead and stationary-law sensitivity and making no hardware-performance claim.

## WHAT WOULD REFUTE IT
A valid cold replay in which temporal exchange beat ordinary Gibbs across every required stratum under a serious work denominator and simultaneously passed the target-law preservation gate would refute the null verdict.

## WAS THAT CHECKED
Yes. The artifact compares temporal exchange with ordinary Gibbs across six graph/temperature strata, recomputes efficiency and target-law gates, tests alternative work denominators, verifies row and trajectory integrity, and reports no invalid rows. Both headline gates could have passed but did not.

## EVIDENCE
`cold_verdict_class`: `null`; `efficiency_gate_passed`: `false`; `target_law_gate_passed`: `false`; `source_verdict_supported`: `true`; `source_all_strata_positive_under_attempted_updates`: `false`; `endpoint_depends_on_favorable_denominator`: `true`; `ordinary_gibbs`; `temporal_exchange`; `invalid_row_ids`: `[]`; `source_row_hash_invalid_count`: `0`; `trajectory_hash_mismatch_count`: `0`; `verifier_is_oracle`: `false`; `physical_hardware_invoked`: `false`; `status`: `complete_null`.

## RECOMMENDATION
KEEP

## experiment_6795_v592_branch_disposition.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V592 provides durable-checkpoint and narrowly bounded synthetic fixed-point evidence, while dispatch and cold CSL causality remain incomplete and temporal exchange is null.

## WHAT WOULD REFUTE IT
Checkpoint interruption/resume failure; the matched flat recurrent control tying or beating grouped fixed-point on exact validity; a held-topology confidence interval reaching zero; successful cold CSL causal replay; or temporal exchange passing both efficiency and target-law gates would refute one or more parts of the headline.

## WAS THAT CHECKED
Yes. Checkpoint preservation and resume rows were checked; grouped fixed-point faced a matched-parameter, matched-work flat recurrent control and a held-topology analysis; CSL faced an independent cold causal audit; and temporal exchange faced matched-update efficiency and target-law gates. These checks produced both positive and adverse outcomes, so the mixed result was not forced.

## EVIDENCE
`"honest_verdict": "complete_partial: V592 has checkpoint and bounded fixed-point evidence, but dispatch and cold CSL causality remain incomplete and temporal exchange is null; no branch metrics were pooled."`; `"fresh_resume_rows": 15`; `"prefix_rows_preserved": 9`; `"matched_candidate_work": true`; `"matched_parameter_counts": true`; `"flat_recurrent_control": 0.565625`; `"grouped_fixed_point": 0.6677083333333333`; `"lower": 0.0510251308751085`; `"nearest_valid_distance_mean_by_arm"`; `"flat_recurrent_control": 0.0`; `"grouped_fixed_point": 0.05625`; `"runtime_s_by_arm"`; `"flat_recurrent_control": 0.18932899999999991`; `"grouped_fixed_point": 0.598373`; `"oracle_leakage_free": true`; `"cold_causal_audit_passed": false`; `"promotion_gate": false`; `"efficiency_gate_passed": false`; `"target_law_gate_passed": false`; `"pooled_score_computed": false`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6796_agent_model_dispatch_requalification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the artifact records a blocked precondition audit and makes no positive or comparative claim about dispatch requalification performance.

## WAS THAT CHECKED
No; no requalification rows were evaluated because required inputs were missing.

## EVIDENCE
`"status": "blocked"`, `"rows": []`, `"v593_pairs_audited": 0`, `"dispatch_contract_ready": false`, `"state": "not_requalified_because_precondition_failed"`, `"failed_check": "required_dispatch_requalification_inputs_exist_and_parse"`, `"verdict_class": "blocked"`, `"honest_verdict": "complete_blocked_dispatch_requalification"`

## RECOMMENDATION
KEEP

## experiment_6797_canonical_transaction_byte_replay.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6798_csl_causal_safety_byte_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Independent byte replay verified causal routing and utility effects while finding zero admitted or influential poison.

## WHAT WOULD REFUTE IT
Any replay/hash mismatch, a credited factor lacking an action or utility change under identical parent bytes, any admitted or influential poison, or a frozen/retrieval-disabled serious baseline tying or beating the online arm on future utility would refute the claim.

## WAS THAT CHECKED
Yes. Replay, receipt, action, utility, counterfactual-witness, poison, and comparator checks were included across five orders. Negative outcomes were representable: the phase-level harm check actually recorded capacity-pressure harm. The excerpt omits the numeric `influenced_poison_count`, but the trusted mechanical pre-pass reports it as zero.

## EVIDENCE
`all_actions_replayed`: `true`; `all_receipt_hashes_matched`: `true`; `all_utilities_replayed`: `true`; `all_passed`: `true`; `failed_checks`: `[]`; `credited_factor_count`: `116`; `same_parent_bytes`: `true`; `action_changed`: `true`; `utility_difference`: `1.25`; `online_minus_frozen_bootstrap_ci`: `[ 0.34375, 0.35625 ]`; `online_minus_placebo_order_effects`: `0.390625`, `0.3625`, `0.346875`, `0.334375`, `0.33125`; `admitted_poison_count`: `0`; `capacity_pressure`: `true`.

## RECOMMENDATION
KEEP

## experiment_6799_model_output_formal_constraint_probes.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
As a receipt assertion, it would fail if frozen source provenance did not replay exactly, paired formal probes failed their gates, or new LLM inference replaced the source outputs.

## WAS THAT CHECKED
Yes. The artifact records source hashes, fresh-process replay matches, matching receipts, operation proofs, adversarial mutation checks, and whether live LLM inference occurred. These establish artifact readiness but make no comparative or value claim.

## EVIDENCE
`honest_verdict`: `complete: frozen authentic outputs produced exact paired formal probes`; `fresh_process`: `true`; `matches`: `true`; `all_passed`: `true`; `failed_checks`: `[]`; `live_llm_invoked`: `false`; `verifier_is_oracle`: `false`; `model_output_constraint_probe_ready`: `true`

## RECOMMENDATION
KEEP

## experiment_6800_real_output_fixed_point_transfer_ab.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The frozen real-output comparison completed without the preregistered positive effect.

## WHAT WOULD REFUTE IT
A positive preregistered decision—specifically, the grouped-minus-flat restructuring exact-valid effect having a clustered lower confidence bound above zero while all required precondition and no-harm gates passed—would refute the headline.

## WAS THAT CHECKED
Yes. The completed paired comparison reports the decision gates, transformation-level clustered confidence intervals, equal candidate budgets, and a substantive flat recurrent comparator. The restructuring lower bound was negative, so the positive criterion did not occur.

## EVIDENCE
`honest_verdict`: `complete: frozen real-output comparison finished without the preregistered positive effect`; `completed_row_count`: `2910`; `planned_row_count`: `2910`; `pending_row_ids`: `[]`; `positive`: `false`; `restructuring_exact_valid_lower_bound_above_zero`: `false`; `restructuring_exact_valid_lower_bound`: `-0.0037037037`; `point`: `0.0015432099`; `lower`: `-0.0037037037`; `upper`: `0.0074074074`; `flat_recurrent_control`: `4365`; `grouped_fixed_point`: `4365`.

## RECOMMENDATION
KEEP

## experiment_6801_real_output_fixed_point_cold_audit.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence [' paired estimate is ', ' with ', ', while their reported ', ' values are ', ' and '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_REFUTED_BY_OWN_DATA

## THE HEADLINE CLAIM
Grouped fixed-point inference improves exact validity over the flat recurrent control.

## WHAT WOULD REFUTE IT
The flat recurrent control tying or outperforming grouped fixed-point inference on exact-validity rates, or paired confidence intervals including zero.

## WAS THAT CHECKED
Yes. The artifact directly compares both arms overall and reports source-case-clustered paired intervals for every transformation; the control wins overall for base, while every interval includes zero.

## EVIDENCE
Under `overall`, `flat_recurrent_control` has `exact_valid_rate` `0.0323024055`, versus `0.0302405498` for `grouped_fixed_point`. The `base` paired estimate is `-0.0018518519` with `lower` `-0.0074074074` and `upper` `0.0037037037`. The `refinement` interval runs from `-0.0037037037` to `0.0058641975`, and the `restructuring` interval from `-0.0043209877` to `0.0077160494`. Both arms also have `convergence_rate` `0.0`, while their reported `runtime_s` values are `1.968041` and `11.365803`.

## RECOMMENDATION
CORRECT_THE_RECORD
