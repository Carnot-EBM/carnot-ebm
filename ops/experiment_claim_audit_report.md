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
| CANNOT_DETERMINE | 2 |

## experiment_bonsai2_n32_collapse_followup.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The standard model's N=32 throughput collapse bisects on the fork to an N-dependent upstream auto-probe disabling fused Gated Delta Net between N=16 and N=20, while stock master also collapses to the identical per-stream rate of 0.27 tok/s at N=32 via a different disabled fused-op fallback (Flash Attention).

## WHAT WOULD REFUTE IT
1. In the bisection: observing `gdn_chunked_disabled_at_startup` evaluate to `true` at N=16 (or N=8), or evaluating to `false` at N=20 or N=24; or finding that throughput does not degrade between N=16 and N=24.
2. In the fork-versus-stock isolation: observing stock `ggml-org/llama.cpp` master sustain high, uncollapsed per-stream throughput at N=32 (e.g., ~10–13 tok/s), proving the collapse was an isolated regression introduced by the PrismML fork; or finding stock disables fused GDN via the same code path; or observing that the collapse was caused by OS/driver faults (Xid errors, OOM kills) or deadlocks.

## WAS THAT CHECKED
Yes. Bisection runs at N=16, N=20, and N=24 (`n_bisection_standard_model_only`) verified that `gdn_chunked_disabled_at_startup` switches from `false` to `true` exactly between N=16 and N=20 where per-stream throughput collapses from 4.318 to 1.514 tok/s. Stock upstream `ggml-org/llama.cpp` master was built with CUDA and tested at N=32 (`fork_vs_stock_isolation`), reproducing the identical 0.27 tok/s collapse while confirming `auto_fgdn = false` and capturing the fallback to disabled Flash Attention instead. Potential alternative causes were evaluated and ruled out via `dmesg`, `gdb`, and `nvidia-smi dmon` (`gpu_diagnostics_during_collapse`). The artifact also explicitly bounded its claim by acknowledging it could not resolve why the fallback penalizes the standard model far more than the ternary model.

## EVIDENCE
- `honest_verdict`: `"complete: bisected the standard-model N=32 collapse to a real, N-dependent, upstream llama.cpp code path (an auto-probe that disables the fused Gated Delta Net kernel once the worst-case batch shape crosses a threshold between N=16 and N=20), confirmed the fork and current stock master both collapse to the same converged per-stream rate at N=32 via two different disabled-fused-op fallbacks, and could not fully explain why the fallback costs the standard (K-quant) model far more than the ternary model at the same N"`
- `n16` -> `per_stream_tok_s_mean`: `4.318`
- `n16` -> `gdn_chunked_disabled_at_startup`: `false`
- `n20` -> `per_stream_tok_s_mean`: `1.514`
- `n20` -> `gdn_chunked_disabled_at_startup`: `true`
- `n24` -> `per_stream_tok_s_mean`: `0.4136`
- `n24` -> `gdn_chunked_disabled_at_startup`: `true`
- `n32` -> `per_stream_tok_s_mean`: `0.27`
- `stock_n32_result` -> `converged_per_stream_tok_s`: `0.27`
- `stock_n32_result` -> `gdn_chunked_disabled_at_startup`: `false`
- `stock_n32_result` -> `flash_attention_disabled_at_startup`: `true`
- `fork_src_llama_context_cpp_line_324`: `"cparams.auto_fgdn = true;"`
- `stock_src_llama_context_cpp_line_235`: `"cparams.auto_fgdn = false;"`
- `log_line`: `"resolve_fused_ops: layer 3 is assigned to device CPU but Flash Attention is assigned to device CUDA0 (usually due to missing support) / Flash Attention not supported, set to disabled"`
- `dmesg_finding`: `"Zero Xid errors and zero OOM-killer / 'Out of memory: Kill' entries anywhere in the kernel ring buffer, covering the entire session back to the last boot (2026-09-25). Whatever is causing the collapse, it is a userspace performance problem -- not a kernel-level GPU fault, driver crash, or memory-pressure kill."`

## RECOMMENDATION
KEEP

## experiment_bonsai2_quality_eval.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
There is no detectable quality difference between Ternary-Bonsai-2-27B and the mandated standard model Qwen3.8-27B-Q4_K_M across N=60 execution-graded HumanEval tasks.

## WHAT WOULD REFUTE IT
A statistically significant difference in unit-test pass rates between Ternary-Bonsai-2-27B and the standard model (e.g., McNemar p < 0.05 driven by an asymmetric win/loss margin on discordant tasks indicating quantization degradation), or a 95% bootstrap confidence interval on the delta that excludes zero.

## WAS THAT CHECKED
Yes. Both models generated solutions for 60 HumanEval coding tasks that were executed against real test suites under identical parameters. Performance avoided floor and ceiling collapse (51/60 vs 52/60 passed), and discordant cases were actively observed and balanced (3 wins for Bonsai vs 4 wins for the standard model), yielding McNemar p = 1.0000 and a 95% CI spanning [-0.10, +0.067].

## EVIDENCE
`"honest_verdict": "complete: no detectable quality difference between bonsai-2 ternary and the mandated standard model at N=60 execution-graded HumanEval tasks (McNemar p=1.0000)"`
`"positive_control_passed": true`
`"n_main_sampled": 60`
`"n_passed": 51`
`"pass_rate": 0.85`
`"n_passed": 52`
`"pass_rate": 0.8666666666666667`
`"energy_descent_wins": 3.0`
`"ar_wins": 4.0`
`"p_value": 1.0`
`"point_estimate": -0.016666666666666666`
`"ci_lower": -0.1`
`"ci_upper": 0.06666666666666667`
`"verifier_is_oracle": true`

## RECOMMENDATION
KEEP

## experiment_bonsai2_ternary_eval.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_semif_readout_ebm_eval_a1.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The A1 readout product-of-experts failed the pre-registered acceptance gate against the verifier-only baseline.

## WHAT WOULD REFUTE IT
The claim that the readout product-of-experts failed its acceptance gate would be refuted if the product-of-experts demonstrated a statistically significant Brier score improvement over the verifier-only expert on held-out evaluation folds (specifically, a paired group bootstrap 95% confidence interval with an upper bound strictly below zero), or if the evaluation framework itself was shown to be broken (such as a failure on the synthetic positive control).

## WAS THAT CHECKED
Yes. Checked in `acceptance_gates.poe_brier_delta_vs_verifier_only_ci95_upper_below_zero`, `paired_group_bootstrap.brier_delta_vs_verifier_only`, and `oof_cross_validation.per_fold`, while testing harness validity was checked in `positive_control.passed`.

## EVIDENCE
- `honest_verdict`: `"complete: a1_readout_poe_failed_pre_registered_gate"`
- `gate_passed`: `false`
- `poe_brier_delta_vs_verifier_only_ci95_upper_below_zero`
- `value`: `0.0`
- `passed`: `false`
- `brier_delta_vs_verifier_only`
- `point`: `0.0`
- `ci95`: `[0.0, 0.0]`
- `alpha`: `0.0`
- `beta`: `4.0`
- `positive_control`
- `passed`: `true`

## RECOMMENDATION
KEEP

## experiment_semif_readout_ebm_eval_a2.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_semif_readout_ebm_eval_a3.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The A3 calibrated policy failed its pre-registered acceptance gate due to an action balance failure where the reject action is never selected at the primary operating cost.

## WHAT WOULD REFUTE IT
The headline claim would be refuted if non-zero decision counts appeared for all three actions (`accept`, `reject`, and `escalate`) at the primary cost of 0.2 across the 4,603 held-out rows, causing `all_three_actions_occur_at_primary_cost` to pass and resulting in `gate_passed: true`.

## WAS THAT CHECKED
Yes. The action distribution was evaluated across all 4,603 held-out rows for 5 independent random seeds and across the entire cost grid, recorded in `acceptance_gates.all_three_actions_occur_at_primary_cost`, `headline_metrics.primary_cost_confusion_matrix`, and each entry of `per_seed_results[*].cost_grid_results`.

## EVIDENCE
- `"honest_verdict": "complete: a3_calibrated_policy_failed_pre_registered_gate"`
- `"gate_passed": false`
- `all_three_actions_occur_at_primary_cost`
- `"passed": false`
- `"reject": 0`
- `"degenerate": true`
- `"zero_actions": [ "reject" ]`
- `"reason": "not mathematically forced by the cost grid -- a real balance failure"`
- `"escalation_saves_cost": false`

## RECOMMENDATION
KEEP

## experiment_7938_v688_hardware_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An affirmative comparative performance or hardware advantage claim evaluated against an experimental baseline. Because this artifact is strictly a historical custody receipt and gate-validation audit recording zero device executions, no empirical hypothesis is under test to falsify.

## WAS THAT CHECKED
No. The artifact explicitly disclaims execution and measurement, noting zero current device executions, and conducts only upstream receipt verification and validation checking.

## EVIDENCE
- `hardware_speedup_claimed`: `false`
- `hardware_advantage`: `unmeasured`
- `workload`: `unmeasured`
- `board_evidence`: `historical custody`
- `current_measurement`: `host receipt analysis only`
- `current_device_execution_count`: `0`
- `arm`: `historical_accounting`
- `status`: `historical_read_only`
- `honest_verdict`: `complete_disqualified_required_checks`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `Receipt custody does not measure independent scientific benefit.`
- `Amdahl and placement comparisons are estimates; no hardware speedup was measured.`

## RECOMMENDATION
KEEP

## experiment_7939_v688_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this is an administrative aggregation and receipt artifact that asserts no comparative model superiority or scientific advantage, there is no comparative empirical claim to falsify. If the artifact had claimed positive scientific readiness or modeled capability gain, observing `honest_verdict` as `"complete_blocked_missing_science"`, `independent_benefit` as `false`, or upstream prerequisite gates resolving to `"disqualified"` would refute it.

## WAS THAT CHECKED
No comparative hypothesis was evaluated. The artifact checked upstream artifacts, execution environments, and pipeline gate conditions (recorded under `gate_check_summary` and `preconditions_checked`), finding unmet prerequisites that blocked downstream claims.

## EVIDENCE
`honest_verdict`: `"complete_blocked_missing_science"`
`inference_substrate`: `"aggregation_from_upstream_artifacts"`
`inference_substrate_class`: `"no_model_load"`
`model_invoked`: `false`
`MODEL_SPECS`: `[]`
`claim_scope`
`independent_benefit`: `false`
`DiffusionGemma`: `"pending_actual_distinct_oracle_accuracy_and_efficiency_evidence"`
`fixture_agreement`: `"circular_positive"`
`gap_decisions`
`decision`: `"blocked"`
`outcome_rows`
`arm`: `"administrative_disposition"`
`field_principles`: `"Bind actual producer bytes and unit counts; audit completion alone proves no scientific benefit."`

## RECOMMENDATION
KEEP
