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
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8701_v746_relational_feature_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative performance or verifier-value claim to refute. The reported numerical blockage would be contradicted by authenticated probabilities and completed numerical-head evaluations for the intended rows.

## WAS THAT CHECKED
Yes, for the blockage: the operand gate reports all 480 required probabilities missing, and the displayed row heads contain null probabilities and decisions. The target model was not evaluated. Complete-record oracle agreement is explicitly limited to constructed capacity; it is not presented as added value or generalization.

## EVIDENCE
- `honest_verdict`: `complete_blocked_missing_numerical_head_operand`
- `verdict_class`: `blocked`
- `expected`: `480 authenticated candidate probabilities`; `missing_count`: `480`
- `operand_state`: `missing_field`; `probability`: `null`
- `evaluated_this_invocation`: `false`
- `numerical_completed_count`: `0`
- `verifier_is_oracle`: `true`
- `exposure_scope`: `Fully exposed constructed oracle records; no natural-language extractor is evaluated.`

## RECOMMENDATION
KEEP

## experiment_8702_v746_installed_batch_prototype.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8703_installed_batch_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked run and makes no comparative throughput or speed claim.

## WAS THAT CHECKED
No throughput or speed comparison was performed. The prerequisite was checked in gates_evaluated and failed: actual 0 versus expected 1.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `artifact_field`: `batch_service_ready_score`; `expected`: `1`; `actual`: `0`; `passed`: `false`.

## RECOMMENDATION
KEEP

## experiment_8704_qwen_symbolic_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: the title describes an intended comparison, and the artifact makes no claim about its outcome.

## WAS THAT CHECKED
No comparative evaluation is reported. Both prerequisite gates failed; this artifact records the resulting block, so the intended comparison had no opportunity to succeed or fail.

## EVIDENCE
`schema`: `blocked_gate_check_v1`  
`status`: `blocked`  
`honest_verdict`: `blocked_gate_check_failed`  
`gate_check_summary`: `gate-unsat(final): 2 of 2 gate(s) failed; first failure: exp8698-relational-evidence-controls.runtime_repair_ready_score (actual=0 == expected=1)`  
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8705_v746_arc_decision_qualification.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The qualified ledger supports no policy change because intervention efficacy remains unresolved.

## WHAT WOULD REFUTE IT
Valid, uncensored outcomes providing independent cross-game support sufficient to retire or reprioritize an intervention arm, contradicting the reported absence of support.

## WAS THAT CHECKED
Yes, for this narrow evidence-status claim: `arm_support_rows` reports zero independent games with uncensored support for every arm, and `per_game_results` reports all 16 episodes censored with zero complete outcomes. Censored observations receive no efficacy estimate or promotion credit. This supports withholding a policy change; it does not establish that interventions are ineffective. No comparative verifier-value or generalization benefit is claimed.

## EVIDENCE

- `honest_verdict`: `complete_null_arc_decision_qualification`
- `policy_decision`: `no_supported_policy_change`
- `censored_count`: `16`
- `complete_horizon_count`: `0`
- `independent_games_with_uncensored_support`: `0`
- `efficacy_estimate`: `null`
- `promotion_ready_score`: `0`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8706_v746_spline_lifecycle_stress.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The primary32/24 fixed-point head passed numerical accuracy, renormalization continuity, supported-domain accumulator, and restart checks on the constructed CPU lifecycle runs.

## WHAT WOULD REFUTE IT
A probability error above 0.001, boundary continuity error above 2.9802322387695312e-08, decision mismatch, supported-domain accumulator violation, or restarted state differing from uninterrupted execution would refute the claim.

## WAS THAT CHECKED
Yes. The update and boundary rows compare fixed-point results against float64; accumulator rows check the declared envelope; restart rows compare state, serialization, and acknowledgements after an actual process exit. These comparisons could fail; shared inputs do not guarantee numerical agreement.

The summary reports 25 completed runs and zero failures or exclusions, although most rows are elided. Visible rows contain the relevant measurement fields and show no invalidity flags.

The oracle defines numerical correctness here; the artifact makes no claim that the verifier adds predictive value. Float64 execution and uninterrupted execution are appropriate comparators for accuracy and restart fidelity. Generalized learning and acceleration are explicitly outside the claim.

## EVIDENCE
- `primary_format`: `32`, `24`
- `probability_error_max`: `0.001`
- `continuity_error_max`: `2.9802322387695312e-08`
- `boundary_continuity_error`: `2.918843478028066e-08`
- `decision_parity`: `true`; `observed_inside_envelope`: `true`
- `state_identical`: `true`; `serialized_state_identical`: `true`; `acknowledgement_identical`: `true`
- `real_exit`: `73`; `resume_exit`: `0`
- `completed_count`: `25`; `failed_count`: `0`; `excluded_count`: `0`
- `verifier_is_oracle`: `true`
- `methodology_note`: `CPU emulation with original Brier equation, scaling and temperature; marked boundary states are constructed, not learned. No generalized learning, FPGA execution or acceleration measured.`

## RECOMMENDATION
KEEP

## experiment_8707_v746_hardware_operation_boundary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative benefit claim to falsify. Admitted complete-cost evidence and authenticated compatible physical execution would contradict the reported blocked readiness status.

## WAS THAT CHECKED
Yes, readiness was checked in `source_admission_rows` and `gate_check_summary`: both cost inputs are unqualified and censored, and physical readiness is zero. This is an evidence aggregation receipt. It makes no claim of speedup, verifier advantage, or generalization, so the oracle flag does not establish circular overclaiming.

## EVIDENCE

- `honest_verdict`: `complete_blocked_missing_complete_cost_or_physical_execution`
- `methodology_note`: `Read-only reduction of authenticated original board and CPU lifecycle artifacts. No model, numerical benchmark, probe, flash or physical acceleration.`
- `operand_state`: `missing_file`, `observed_zero`
- `interpretation`: `Conditional1/(1-f); neither observed speed nor a100x complete-learning result`
- `g1_g4_separation`: `Original G1–G4 publication conditions remain separate; this map grants none.`

## RECOMMENDATION
KEEP

## experiment_8708_v746_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative efficacy claim is asserted. This is a readiness and qualification aggregation; its blocked status does not establish a measured null effect.

## WAS THAT CHECKED
Administrative qualification was checked in the gate rows, including failed prerequisites and disqualified producers. Comparative benefit testing is not demonstrated: deployment measurements remain unmeasured, and independent generalization is explicitly disclaimed.

## EVIDENCE
- `policy_decision`: `no_supported_policy_change`
- `activation_authorized`: `false`
- `scientific_result_measured`: `false`
- `producer_qualified`: `false`
- `deployment_decision_rows`: `measured`: `false`
- `exposure_scope`: `Cached exposed and constructed upstream evidence; no independent natural generalization`

## RECOMMENDATION
KEEP
