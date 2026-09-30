# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| AGGREGATE_ONLY | 2 |
| CANNOT_DETERMINE | 3 |

## experiment_bonsai2_n32_collapse_followup.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims that standard-model throughput collapses super-linearly between N=16 and N=24 due to an N-dependent disabled fused kernel path, and that upstream stock llama.cpp and the fork both collapse to the identical per-stream generation rate (0.27 tok/s) at N=32.

## WHAT IS MISSING
Per-stream rows recording individual throughput or token counts across each of the concurrent streams (slots 0 to N-1), as well as the per-sample rows behind the 841 samples. The artifact records only aggregate and pooled fields: `per_stream_tok_s_mean`, `aggregate_tok_s`, `converged_per_stream_tok_s`, `converged_per_stream_tok_s_3s_window`, and `aggregate_tok_s_estimate_from_converged_rate`, along with a scalar count in `n_print_timing_samples_captured`.

## THE CHECK A READER CANNOT DO
A reader cannot determine whether the 0.27 tok/s per-stream rate (or the steep degradation from N=16 to N=24) was uniform across all concurrent streams, or if a few stalled or outlier streams dragged down the pooled mean while others ran at higher throughput.

## experiment_bonsai2_quality_eval.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
There is no detectable quality difference between Bonsai-2 ternary and the mandated standard model across N=60 execution-graded HumanEval tasks (McNemar p=1.0000).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_bonsai2_ternary_eval.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Ternary Bonsai 2 27B is faster (1.44x single-stream decode) and smaller (2.88x) than the mandated Qwen3.8-27B-Q4_K_M baseline with identical concurrent capacity limits (N=152) and a 62.5% agreement rate on a 160-row quality proxy.

## WHAT IS MISSING
Per-unit row records. In `bounded_quality_retention_proxy`, summary fields `"n_rows_compared"`, `"argmax_agreement_rate"`, `"standard_model_accuracy_vs_label"`, `"ternary_model_accuracy_vs_label"`, and `"standard_model_escalate_rate"` are present, but per-row items (such as corpus row IDs, ground-truth labels, and individual model letter outputs/logits) are absent. In `throughput_single_stream`, summary fields `"mean_tok_s"`, `"var_tok_s"`, and `"n_runs"` are present, but the individual latency numbers for the 5 runs are absent. In `aggregate_throughput_concurrent`, `"aggregate_tok_s"` and `"per_stream_tok_s_mean"` are present, but individual stream timings are absent.

## THE CHECK A READER CANNOT DO
A reader cannot determine whether the ternary model's 50% accuracy on the 160-row proxy represents genuine mathematical evaluation on specific items or a degenerate collapse to guessing between choices A and B (given its near-zero `"ternary_model_escalate_rate"` of 0.00625 vs the standard model's 0.23125).

## experiment_semif_readout_ebm_eval_a1.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live LLM single-token logit readout Product of Experts (PoE) failed its pre-registered acceptance gate because it achieved zero Brier score improvement over the verifier-only baseline across all cross-validation folds.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_semif_readout_ebm_eval_a2.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_semif_readout_ebm_eval_a3.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_7938_v688_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7939_v688_capstone.json

**CANNOT_DETERMINE**

reviewer call failed
