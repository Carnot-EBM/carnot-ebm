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
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 2 |

## experiment_8379_v722_native_direct_parity.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8381_v722_logit_policy_certificate.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The dyadic_logit_v1 policy passed numerical qualification with zero interval escapes or action disagreements and 100% certified fast-path coverage on the constructed random panel.

## WHAT WOULD REFUTE IT
A reference logit outside its reported interval, an action differing from the exact reference action, or certified random-panel coverage below the stated 0.5 usefulness threshold.

## WAS THAT CHECKED
Yes. Rows report intervals, reference logits, actions, and reference actions; aggregate counts report zero escapes and mismatches. Coverage is reported separately, and the panel includes knot and threshold neighbors.

The approximate interval calculation and independent rational reference can disagree, so this numerical qualification could fail. The oracle defines numerical correctness; the artifact claims no added semantic benefit. Its generalization scores are zero and its development exposure is disclosed.

Direct exact-policy evaluation is the serious correctness comparator already represented by the reference fields. Comparative speed or verifier value would require further controls, but neither is the stated claim. No exclusions are reported, and the relevant comparison fields exist on the displayed rows. The recorded old-policy disagreement does not contradict correctness under the explicitly new policy.

## EVIDENCE
- `methodology_note`: `Independent rational polynomial checks an exact new action policy. Oracle-checked numerical correctness grants no semantic benefit or production migration. Missing inputs are absent, not measured zero.`
- `independent_remainder_audit` = `true`; `verifier_is_oracle` = `true`.
- `interval_escape_count` = `0`; `action_mismatch_count` = `0`; `fast_path_fraction` = `1.0`.
- `useful_score`: `Safety-qualified random fast coverage >=0.5; never changes the safety threshold.`
- `completed_count` = `4166`; `excluded_count` = `0`.
- `exposure_scope`: `constructed_label_free_panel_from_exposed_development_head`.
- `independent_generalization_score` = `0`; `generalized_learning_benefit_score` = `0`.

## RECOMMENDATION
KEEP

## experiment_8382_v722_runtime_evidence_delta.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An authenticated receipt showing an environment change, or a successful current CUDA context/copy probe, would overturn the recorded blocked disposition. The artifact makes no comparative method-value claim.

## WAS THAT CHECKED
No current runtime refutation was tested: all seven environment operands were censored, and zero current probes ran. The receipt gates were checked and failed. This establishes missing evidence; physical environmental equality remains untested.

## EVIDENCE
- `verdict_class`: `blocked`
- `honest_verdict`: `complete_blocked_cuda_environment_unchanged`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `censored_count`: `7`; `completed_count`: `0`
- `current_probe_count`: `0`
- `missing_reason`: `new_authenticated_operand_absent`
- `methodology_note`: `Authenticate the qualified typed reader and immutable aliases, compare only explicitly supplied new environment receipts, then admit one direct driver context/copy probe. No model or semantic-benefit measurement.`

## RECOMMENDATION
KEEP

## experiment_8383_changed_runtime_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records blocked prerequisites and makes no claim of canary success or comparative value.

## WAS THAT CHECKED
No canary outcome was tested. The prerequisite checks in `gates_evaluated` recorded two failures, blocking execution before the canary.

## EVIDENCE
`schema`: `blocked_gate_check_v1`  
`status`: `blocked`  
`honest_verdict`: `blocked_gate_check_failed`  
`gate_check_summary`: `gate-unsat(final): 2 of 3 gate(s) failed; first failure: exp8382-runtime-evidence-delta.runtime_changed_score (actual=0 == expected=1)`  
`blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8384_v722_arc_supervisor_live_panel.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific headline to falsify. The artifact reports completed bounded execution with blocked readiness, without asserting supervisor benefit or generalization.

## WAS THAT CHECKED
Yes—the paired supervisor-on/off comparison was recorded in per-game results. Both arms show zero progress in every pair. All supervisor outcome windows remain censored, so this supports neither superiority nor a conclusion that later benefit is impossible. The artifact withholds positive credit.

## EVIDENCE
- `honest_verdict`: `complete_blocked_fresh_public_supervisor_panel`
- `scientific_benefit`: `false`
- `generalized_learning_benefit_score`: `0`
- `independent_generalization_score`: `0`
- Paired `arm` values: `off`, `on`; every row has `progress_count`: `0` and `peak_level`: `0`.
- `outcome_window_status`: `pending_censored`
- `No causal semantic benefit is established by numerical or wrapper parity.`

## RECOMMENDATION
KEEP

## experiment_8385_v722_board_operation_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. An authenticated current board execution satisfying compatibility, authority, and complete transport-inclusive timing requirements would contradict the reported blocked readiness.

## WAS THAT CHECKED
No current execution establishing those conditions is shown. This artifact aggregates historical evidence and records unmet prerequisites; it does not assert verifier value, generalization, or superiority over a comparator. Unmeasured speedup is not a measured null result.

## EVIDENCE
- `actual_substrate`: `host_CPU_aggregation`
- `current_device_execution_count`: `0`
- `current_contract_ready_score`: `0`
- `acceptance_gates`: `authority`: `false`, `compatible_kernel`: `false`, `complete_transport`: `false`, `scientific_benefit`: `false`
- `full_service_speedup`: `null`
- `compatible_fraction_status`: `unmeasured`
- `claim_class`: `historical`
- `terminal_criterion_met`: `false`

## RECOMMENDATION
KEEP

## experiment_8386_v722_gatemate_obligation_delta.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no scientific or comparative headline to falsify. An authenticated completion receipt for an obligation still reported as missing would contradict the bookkeeping.

## WAS THAT CHECKED
No comparative refutation test is shown or required for this receipt. The supplied-evidence list is empty; the rows consistently show one completed continuity record and six censored obligations. Oracle-based validation is not presented as added scientific value.

## EVIDENCE
- `honest_verdict`: `complete_blocked_gatemate_obligation_delta`
- `verdict_class`: `blocked`
- `scientific_benefit`: `false`
- `generalized_learning_benefit_score`: `0`
- `supplied_evidence_delta`: `[]`
- `completed_count`: `1`
- `censored_count`: `6`
- `arm`: `read_only_obligation`

## RECOMMENDATION
KEEP

## experiment_8387_v722_capstone.json

**SKIPPED_ALREADY_FLAGGED**
