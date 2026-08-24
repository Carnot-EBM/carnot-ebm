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
| CLAIM_SUPPORTED | 5 |
| NO_CLAIM | 2 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6562_constraint_saturation_independent_audit_v2.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The constraint-saturation evidence is disqualified because required provenance, exact replay, release/harm, and cost evidence is not independently recomputable.

## WHAT WOULD REFUTE IT
All required gates passing—especially recomputable live provenance and raw responses, executable exact clause-and-joint replay, a clean harm/release audit, and fully recomputed costs including retries—would refute the disqualification.

## WAS THAT CHECKED
Yes. The aggregate recomputation and gate-check summary explicitly evaluated those gates; some checks passed, showing success was possible, while four required checks failed. The oracle defines audit eligibility, but the artifact makes no positive claim about the oracle’s added value.

## EVIDENCE
`verdict_class`: `disqualified`; `constraint_saturation_independent_audit_ready_score`: `0.0`; `fixture_replay_passed`: `true`; `live_provenance_recomputable`: `false`; `exact_clause_and_joint_replay_passed`: `false`; `harm_and_release_audit_passed`: `false`; `charged_cost_recomputed`: `false`; `raw_response_path_count`: `0`; `raw_response_rows_present`: `false`; `process_command_present`: `false`; `exact_clause_checker_replayed`: `null`; `exact_joint_checker_replayed`: `null`; `invalid_release_count`: `1`.

## RECOMMENDATION
KEEP

## experiment_6563_production_safety_net_workload_canary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The safety-net canary preserved disabled identity, exact outputs, candidates, fallback, restart, and rollback behavior, while enabled routing delivered no preregistered work or latency benefit.

## WHAT WOULD REFUTE IT
Any failed identity, equality, preservation, fallback, restart, or rollback check would refute the safety claim; alternatively, enabled routing reducing checker calls by at least one or meeting both latency-savings thresholds would refute the reported null benefit.

## WAS THAT CHECKED
Yes. The 48 complete paired rows compare enabled and disabled conditions against the native baseline, while dedicated identity, fallback, restart, rollback, and aggregate recomputation sections expose the relevant failure outcomes.

## EVIDENCE
`expected_per_unit_row_count`: `48`; `observed_per_unit_row_count`: `48`; `complete_rows`: `true`; `disabled_identity_exact`: `true`; `all_exact_outputs_equal`: `true`; `all_candidates_preserved`: `true`; `fallback_passed`: `true`; `restart_passed`: `true`; `rollback_passed`: `true`; `native_checker_calls`: `14.0`; `enabled_checker_calls`: `14.0`; `enabled_checker_call_delta`: `0.0`; `enabled_wall_time_saved_s`: `-0.001244641`; `tail_latency_regression`: `true`; `measured_enabled_benefit`: `false`; `verifier_is_oracle`: `false`; `verdict_class_from_rows`: `null`

## RECOMMENDATION
KEEP

## experiment_6564_rust_pyo3_safety_net_nfr01.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The Rust PyO3 implementation preserved exact Python behavior but achieved only 0.764013447× median batched throughput speedup, failing the frozen 10× NFR01 threshold while remaining within the p99 latency bound.

## WHAT WOULD REFUTE IT
A recomputed median batched speedup of at least 10.0× versus Python scalar would refute the headline null result.

## WAS THAT CHECKED
Yes. The artifact directly benchmarks `rust_pyo3_batch` against the serious baseline `python_scalar`, records complete throughput rows, and recomputes their median throughputs and speedup against the frozen threshold.

## EVIDENCE
`python_scalar_median_throughput_ops_s`: `21532.190854`; `rust_pyo3_batch_median_throughput_ops_s`: `16450.883352`; `steady_state_median_batched_speedup_vs_python_scalar`: `0.764013447`; `nfr01_threshold_speedup`: `10.0`; `nfr01_passed`: `false`; `parity_passed`: `true`; `rust_pyo3_batch_p99_latency_s`: `6.312575e-05`; `p99_latency_bound_s`: `0.05`; `p99_latency_within_bound`: `true`; `verdict_class_from_rows`: `null`.

## RECOMMENDATION
KEEP

## experiment_6565_v569_evidence_and_retirement_contract.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6566_proof_obligation_and_graph_potts_method_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific headline to falsify. For the limited readiness receipt, a missing prerequisite, failed conformance check, nonterminal row, or unfrozen contract component would refute completeness.

## WAS THAT CHECKED
Yes, as contract conformance: readiness was reduced from conformance rows, prerequisite and gate summaries were recorded, and negative, counterexample, and abstention fixtures were included. This does not test scientific value, and the artifact expressly disclaims such a claim.

## EVIDENCE
`"honest_verdict"`: `"complete_source_method_contract_ready: proof-obligation schema, immutable splits, graph features, Potts equations, matched-dose arms, gates, attacks, and retirement rules are frozen"`; `"verdict_class"`: `"Method readiness is infrastructure evidence, not positive science."`; `"inference_substrate"`: `"primary_source_method_preregistration_and_local_conformance_no_llm"`; `"checks_closed"`: `true`; `"failed_checks"`: `[]`; `"missing_prerequisite"`: `false`; `"exact_result"`: `"counterexample"`; `"exact_result"`: `"unsupported_relation"`; `"release_action"`: `"abstain"`; `"verifier_is_oracle"`: `true`; `"Contract conformance is audit authority and cannot create a scientific positive."`

## RECOMMENDATION
KEEP

## experiment_6567_sequential_flagship_gguf_admission.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
No required flagship family completed authentic runtime admission, so the run was blocked.

## WHAT WOULD REFUTE IT
At least one required family appearing as `attempted` and `admitted` with passing process, token, GPU-telemetry, and unload/recovery receipts—and inclusion in `admitted_hf_ids`—would refute the claim.

## WAS THAT CHECKED
Yes, through one admission row for each required family and the aggregate admission reducer. All three failed preconditions before runtime execution, so this supports only the narrow blocked-admission claim, not model incapability or verifier value.

## EVIDENCE
`honest_verdict` = `blocked: no flagship family completed authentic runtime admission; admitted=[]; blocked=[unsloth/Qwen3.6-35B-A3B-GGUF,unsloth/gemma-4-31B-it-GGUF,unsloth/gemma-4-26B-A4B-it-GGUF]`; `exactly_one_required_row_per_family` = `true`; `required_family_row_count` = `3`; `admitted_hf_ids` = `[]`; `ready_score_from_rows` = `0.0`; `all_gates_passed` = `false`; `attempted` = `false`; `admitted` = `false`; `failed_preconditions` = `model_identity_and_file_shape`; `live_process_and_token_rows` = `[]`; `gpu_telemetry_rows` = `[]`; `verdict_class` = `blocked`; `This is infrastructure admission, not verifier science.`

## RECOMMENDATION
KEEP

## experiment_6568_immutable_source_span_claim_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked gate receipt and reports no result about an immutable live source-span claim stream.

## WAS THAT CHECKED
No. The experiment stopped at `conductor_pre_gate`; the method and its claimed outcome were not evaluated.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_field": "all_mandated_models_loaded_score"`, `"failed_expected": 1.0`, `"failed_observed": 0.0`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6570_proof_obligation_independent_audit.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The independent audit is blocked by missing and non-recomputable evidence, so extractor promotion is not confirmed.

## WHAT WOULD REFUTE IT
Usable required inputs plus populated replay rows showing reproducible provenance, spans, compilation, exact releases, costs, and all audit and promotion checks passing.

## WAS THAT CHECKED
Yes, for the blocked-audit claim: input receipts, gate-check rows, and row-based aggregate recomputation explicitly tested those conditions and failed them. The underlying promotion claim was therefore not tested, but the artifact does not assert promotion.

## EVIDENCE
`"required_inputs_exist": false`, `"inputs_usable": false`, `"upstream_terminal_evidence": false`, `"independent_paired_metric_rows": []`, `"promotion_from_rows": false`, `"proof_carrying_extractor_promotion_score": 0.0`, `"status": "blocked_proof_obligation_independent_audit_missing_inputs"`, `"verdict_class": "blocked"`, `"verifier_is_oracle": true`

## RECOMMENDATION
KEEP
