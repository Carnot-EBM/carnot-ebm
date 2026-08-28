# Pipeline Capability Specification

**Capability:** pipeline
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-08

## Overview

Defines the multi-tier cascade pipeline that routes LLM-generated reasoning steps through
progressively more expensive verifiers (Tier 1: fast energy gate, Tier 2: JEPA ranking,
Tier 3: full formal verification).  The JEPA predictor tier must be loaded from a
version-tagged checkpoint to prevent silent rollbacks to sub-threshold models.

## Requirements

### REQ-CASAL-001: CASALTier continuous attributes support
The VerifyRepairPipeline MUST support a CASALTier for continuous attributes, returning an artifact with schema, integration_successful, latency_ms, and acceptance_gate_passed fields.

### REQ-INFRA-043: JEPA Cascade Version-Pinned Checkpoint Loading

The Tier 2 JEPA predictor in the cascade MUST load its model via a version-tagged
checkpoint path.  The active version is recorded in the conductor exclusion manifest
under the `jepa_v18_active` flag.  Loading an older version (v15/v16/v17) is
prohibited while the manifest marks them excluded; attempting to load an excluded
version MUST raise `ValueError`.

**Rationale:** JEPA v15/v16/v17 all produced OOD AUC below random chance (0.47–0.48),
meaning the cascade would actively harm verification quality.  The manifest is the
authoritative gate that prevents accidental rollback to a below-threshold model.

**Acceptance criteria:**
- `tier2_jepa.load_v18_from_manifest()` returns a `JEPALambdaRankV18` instance.
- Passing `version="v17"` raises `ValueError` with a message naming the blocked version.
- The loaded model's `predict_score()` returns a float for any string input.

### REQ-INFRA-044: Cascade AUC Gating

The cascade integration smoke test (Exp 718) MUST achieve cascade_auc >= 0.70 on a
50-question held-out GSM8K validation set before the JEPA v18 deployment is considered
successful.  If cascade_auc < 0.70, the honest_verdict MUST be "cascade_deploy_auc_fail"
and the gate file for Exp 719 MUST record `gate: "fail"`.

### REQ-INFRA-045: Cascade Latency Budget

The per-question latency overhead added by the JEPA v18 Tier 2 scorer MUST be less than
5 ms (latency_delta_ms < 5).  Exceeding this budget causes honest_verdict
"cascade_deploy_latency_fail" regardless of AUC.

### REQ-FST-2240: Fast-Slow Verify-Repair Context Update

The VerifyRepairPipeline MUST provide an optional `use_fst=False` mode that keeps the
base LLM and verifier ensemble as frozen slow weights while treating verifier-output
summaries as fast weights.  When enabled for verify-repair, the verifier-output summary
from the current failed verification step MUST be prepended at the terminal start of the
next repair prompt.

**Acceptance criteria:**
- `SlowWeights` freezes any `requires_grad` parameters exposed by the base LLM and
  verifier ensemble.
- `FastWeights` converts failed `VerificationResult` feedback into a deterministic
  verifier-output context prefix.
- `verify_and_repair(..., use_fst=True)` sends the next repair prompt with that prefix
  before the normal `Question:` repair prompt body.

### REQ-ODAR-2243: ODAR Free-Energy Routing Gate

The VerifyRepairPipeline MUST provide an optional `use_odar=False` routing gate that
fuses Tier 0 probe outputs into an expected free-energy (EFE) score before Tier 1
constraint extraction and downstream deliberative verification.  The gate MUST use
`FreeEnergyRouter(risk_threshold=...)` from `python/carnot/pipeline/odar_router.py`.
When EFE is below the configured risk threshold, `verify()` MUST return a fast-path
`VerificationResult` without running Tier 1 extraction.  When EFE is at or above the
threshold, or when no Tier 0 probe evidence is present, verification MUST continue down
the normal deliberative path.

**Acceptance criteria:**
- `FreeEnergyRouter.route(probe_outputs)` returns `RoutingDecision.FAST_PATH` for
  low-EFE Tier 0 outputs.
- `FreeEnergyRouter.route(probe_outputs)` returns `RoutingDecision.DELIBERATIVE` for
  high-EFE Tier 0 outputs.
- Changing `risk_threshold` changes the route for the same bounded EFE input.
- `VerifyRepairPipeline.verify(..., use_odar=True)` records the EFE and decision in the
  certificate and skips Tier 1 extraction only for the ODAR fast path.

### REQ-ODAR-2244: ODAR Routing Benchmark Gate

The pipeline MUST provide a deterministic Exp 2244 benchmark that compares ODAR
routing with a uniform Tier 0 through Tier 3 cascade on a 30-example reasoning
corpus.  The corpus MUST contain 15 high-confidence low-EFE examples that should
fast-path and 15 ambiguous high-EFE examples that should route to deliberative
verification.

**Acceptance criteria:**
- The benchmark writes `results/experiment_2244_odar_benchmark.json`.
- The artifact includes `honest_verdict`, `odar_benchmark_passed`,
  `compute_reduction_pct`, `accuracy_delta`, `n_corpus`, and
  `preconditions_checked`.
- `compute_reduction_pct` is computed as
  `(tier_calls_A - tier_calls_B) / tier_calls_A * 100`.
- `accuracy_delta` is reported in percentage points as ODAR accuracy minus
  uniform-cascade accuracy.
- `odar_benchmark_passed` is true only when `compute_reduction_pct >= 30` and
  `accuracy_delta >= -2.0`.
- If `python/carnot/pipeline/odar_router.py` cannot be imported, the benchmark
  writes a blocked artifact with `blocked_router_missing`.

### REQ-ODAR-2257: ODAR Real Tier 0 Probe Benchmark

The pipeline MUST provide a deterministic Exp 2257 benchmark that measures ODAR
routing on 100 synthetic reasoning examples using Tier 0 probe outputs produced
through the verify-repair pipeline's existing probe interfaces rather than
preconstructed EFE labels.  The benchmark MUST compare a uniform full verification
cascade against ODAR threshold routing and MUST record the median
`FreeEnergyRouter.evaluate(...)` routing overhead per decision.

**Acceptance criteria:**
- The benchmark writes `results/experiment_2257_odar_real_benchmark.json`.
- `python/carnot/pipeline/odar_router.py` is import-checked before the run; import
  failure writes a blocked artifact with `honest_verdict` prefixed by
  `blocked_odar_missing`.
- The artifact includes `honest_verdict`, `odar_real_validated`,
  `compute_reduction_pct`, `routing_overhead_ms`, `fast_path_fraction`,
  `accuracy_delta`, `n_corpus`, and `preconditions_checked`.
- `n_corpus` is exactly 100.
- `compute_reduction_pct` is computed as
  `(tier_calls_A - tier_calls_B) / tier_calls_A * 100`.
- `accuracy_delta` is reported in percentage points as ODAR accuracy minus
  uniform-cascade accuracy and MUST be at least `-2.0`.
- `odar_real_validated` is true only when `compute_reduction_pct >= 25.0`,
  `routing_overhead_ms <= 5.0`, `accuracy_delta >= -2.0`, and the corpus size is 100.
- The artifact records that external LLM calls were not used.

### REQ-FST-2399: FST Live Path A/B/C Reporting

The pipeline MUST provide an Exp 2399 live-path runner that attempts FST
generation in this order: PATH A local llama.cpp GGUF inference using one of
the mandated SOTA GGUF model repositories, PATH B local transformers
AutoModel/AutoTokenizer inference, and PATH C cached telemetry responses from
`results/live_sota_balanced_telemetry_manifest_1480.jsonl`.  The runner MUST
write `results/experiment_2399_fst_live_path_ab.json` even when PATH A and
PATH B are unavailable, and it MUST record which path succeeded.

**Acceptance criteria:**
- The artifact includes `honest_verdict`, `fst_live_validated`,
  `live_path_used`, `first_live_generation_text`, `path_a_attempted`,
  `path_a_blocked_reason`, `path_b_attempted`, `model_used`,
  `n_test_prompts`, `duration_s`, and `preconditions_checked`.
- `fst_live_validated` is true when the FST verify/terminal-prefix flow runs
  end-to-end on PATH A, PATH B, or PATH C.
- PATH A is attempted first when a mandated GGUF cache entry and `llama_cpp`
  import are available; PATH B is attempted if PATH A fails or is unavailable;
  PATH C is used if both live paths fail and telemetry exists.
- If PATH C succeeds, the artifact records `live_path_used="C_cached"` and
  `model_used=null`.
- The honest verdict uses a terminal prefix such as `complete:` or `blocked:`.

### REQ-SPOE-3657: Deployable Calibrated Second-Pair Detector

The pipeline MUST provide a deployable calibrated fused detector API that
combines an ensemble energy score and a model-confidence error score into one
calibrated probability of error.  Calibration MUST be fit only on a train split,
then evaluated on held-out examples.  The API MUST also evaluate confidence-only
and ensemble-only baselines, Brier score, expected calibration error (ECE), and
per-domain operating points at fixed FPR budgets.

**Acceptance criteria:**
- The detector exposes `fit`, `predict_proba`, and `evaluate_domains` behavior
  from `python/carnot/pipeline/`.
- Calibration uses both ensemble energy and confidence features when fitted.
- Held-out evaluation reports fused AUROC, confidence-alone AUROC,
  ensemble-alone AUROC, Brier, ECE, recall at fixed FPR in `{0.05, 0.10, 0.20}`,
  and the selected deployer operating point.
- If no labeled corpus with both required scores is present, the evaluation
  returns the blocked verdict without fabricating metrics.

### REQ-SPOE-3657-ARTIFACT: Exp 3657 Detector Artifact Contract

The pipeline MUST provide an Exp 3657 runner that re-derives labels, ensemble
energy, and confidence scores from cached corpora, fits the deployable fused
detector, and writes
`results/experiment_3657_deployable_second_pair_of_eyes_detector.json`.

**Acceptance criteria:**
- The artifact includes `honest_verdict`, `inference_substrate`,
  `detector_module_path`, `fused_detector_auroc`,
  `confidence_alone_auroc`, `recall_at_fixed_fpr_table`,
  `calibration_brier_ece`, `fusion_beats_confidence_alone`,
  `n_examples_per_domain`, `random_seed`, `reproducibility_checksum`, and
  `duration_s`.
- `fusion_beats_confidence_alone` is a bare top-level bool.
- The honest verdict is one of:
  `complete: deployable_second_pair_of_eyes_detector_built_fusion_wins_calibrated`,
  `complete: deployable_detector_built_fusion_redundant_with_confidence_product_value_weak`,
  or `complete: blocked_no_labeled_corpus_for_fusion`.
- The runner does not modify `scripts/research_conductor.py`.

### REQ-SPOE-3671: Phase-1 Shipped Second-Pair Detector Surface

The pipeline MUST expose the calibrated second-pair detector through a shipped
`score_candidates` product surface callable from MCP and the CLI.  The shipped
surface MUST fit calibration from labeled cached corpora using ensemble energy
and confidence-error features, score caller-provided candidates with the same
feature contract, and return a calibrated error probability plus the applicable
per-domain operating point.  Code-domain calibration MUST use the balanced Exp
3658 corpus when present, not the older imbalanced code corpus.

**Acceptance criteria:**
- The deployable detector module lives under `python/carnot/pipeline/` and
  exposes the fitted detector API and `score_candidates` surface.
- MCP exposes a `score_candidates` tool and the CLI exposes a matching
  `score-candidates` command.
- The surface returns `calibrated_error_score`, `ensemble_energy`,
  `confidence_error`, `domain`, and `operating_point` for each candidate.
- The Exp 3671 artifact reports fused, confidence-alone, and ensemble-alone
  AUROC per domain; Brier and ECE per domain; fixed-FPR recall table; wired
  surface name; E2E surface result; and a bare top-level `detector_shipped`
  bool.
- `detector_shipped` is true only when the detector is wired to a caller
  surface, the E2E surface call returns a calibrated score, and the fused
  detector beats confidence alone on at least one headroom-bearing domain.
- If no labeled FoVer math or balanced Exp 3658 code corpus can be loaded, the
  artifact returns `complete: blocked_no_labeled_corpus_for_detector` without
  fabricating metrics.

### REQ-SPOE-3671-ARTIFACT: Exp 3671 Detector Ship Artifact Contract

The pipeline MUST provide an Exp 3671 runner that builds the shipped detector
artifact from the FoVer math corpus and the balanced Exp 3658 code corpus, calls
the shipped `score_candidates` surface end-to-end on a held-out example, and
writes `results/experiment_3671_ship_second_pair_of_eyes_detector.json`.

**Acceptance criteria:**
- The artifact includes `honest_verdict`, `inference_substrate`,
  `detector_module_path`, `wired_surface`,
  `fused_detector_auroc_per_domain`,
  `confidence_alone_auroc_per_domain`, `ensemble_alone_auroc_per_domain`,
  `recall_at_fixed_fpr_table`, `calibration_brier_ece_per_domain`,
  `e2e_test_passed`, `detector_shipped`, `n_examples_per_domain`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- `detector_shipped` is a bare top-level bool.
- The honest verdict is one of:
  `complete: second_pair_of_eyes_detector_shipped_math_strong_code_honest_e2e_green`,
  `complete: second_pair_of_eyes_detector_shipped_math_only_code_weak_documented_e2e_green`,
  or `complete: blocked_no_labeled_corpus_for_detector`.
- The runner does not modify `scripts/research_conductor.py`.

### REQ-SPOE-3769: Phase-1 Package CLI MCP Software E2E Smoke

The pipeline MUST provide an Exp 3769 software E2E smoke runner that verifies
the Phase-1 integrator path without publishing or making a headline accuracy
claim.  The runner MUST use the repository venv interpreter, confirm the
`carnot` package import and version, run `VerifyRepairPipeline` on a tiny
arithmetic-slip example with a small CPU smoke model when cached, exercise the
`score_candidates` MCP tool through a real stdio JSON-RPC protocol exchange,
invoke the packaged CLI `score-candidates` command, and write
`results/experiment_3769_package_cli_mcp_e2e_smoke.json`.

**Acceptance criteria:**
- Preconditions are checked before E2E work: `.venv/bin/python`, package import,
  MCP server module importability, MCP protocol runtime, and CLI module
  resolution.  If any required resource is missing, the artifact returns a
  `blocked_<resource>` terminal verdict without fabricating surface passes.
- The package step records whether `import carnot` succeeds, the resolved
  version, and whether an optional local `python -m build` package build
  succeeds.
- The pipeline step records a structured verify/repair result over a hardcoded
  arithmetic-slip example and labels the result as a wiring smoke, not an
  accuracy claim.
- The MCP step starts `python -m carnot.mcp` as a subprocess and calls
  `score_candidates` over the MCP stdio protocol; in-process mocked handler
  calls do not satisfy this requirement.
- The CLI step invokes `python -m carnot.cli score-candidates` on the same tiny
  candidate payload and verifies the JSON output shape.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `package_importable`, `pipeline_e2e_passed`,
  `mcp_protocol_exchange_passed`, `cli_passed`, `surfaces_passed`,
  `is_wiring_smoke_not_accuracy_claim`, `preconditions_checked`,
  `model_specs`, `random_seed`, `reproducibility_checksum`, and `duration_s`.

### REQ-SPOE-3683: Code Operating Point Hardening Verdict

The pipeline MUST provide an Exp 3683 runner that re-measures the shipped
second-pair detector's code operating point on the balanced Exp 3658 corpus,
tests the Exp 3667 dependency-aware verifier weighting on the same code rows,
and evaluates code-specific recalibration on a deterministic code train/holdout
split.  A recovered code operating point MUST beat the 0.5 AUROC chance floor
with a CI excluding 0.5 and improve held-out calibration; otherwise the runner
MUST honestly scope the shipped detector as math-only for code discrimination.

**Acceptance criteria:**
- The runner checks that the balanced Exp 3658 code corpus is loadable and that
  `python/carnot/pipeline/second_pair_detector.py` is importable before scoring.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `code_auroc_baseline`, `code_auroc_dependency_aware`,
  `code_auroc_recalibrated`, `code_calibration_brier_ece_after`,
  `code_recall_at_fixed_fpr`, `module_code_path_updated`, `e2e_test_passed`,
  `code_operating_point_recovered`, `n_examples_code`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- `code_operating_point_recovered` is a bare top-level bool.
- Baseline reporting includes fused, ensemble-alone, and confidence-alone code
  AUROC with CI plus calibration on the held-out balanced corpus rows.
- The dependency-aware row uses the Exp 3667 dependency-aware crossfit weighting
  against row-aligned code verifier scores rather than replaying Exp 3671
  metrics or returning a constant score.
- Code-specific recalibration is fit on a code train split and evaluated on
  held-out code rows, with Brier, ECE, and recall at fixed FPR reported.
- The honest verdict is one of:
  `complete: code_operating_point_recovered_detector_now_math_and_code`,
  `complete: code_remains_math_only_detector_scoped_honestly`, or
  `complete: blocked_no_balanced_code_corpus_or_detector_module`.
- Tests cover recovered, math-only, and blocked outcomes using synthetic
  fixtures rather than hard-coding a real-corpus success string.

### REQ-SPOE-3684: Product Value Rebaseline Against Self-Certainty

The pipeline MUST provide an Exp 3684 runner that re-measures the shipped
second-pair detector against the stronger free self-certainty comparator on the
FoVer math corpus and balanced Exp 3658 code corpus.  The runner MUST report
self-certainty-alone AUROC, plain-confidence-alone AUROC, ensemble-alone AUROC,
and fused ensemble+self-certainty AUROC per domain.  It MUST compute paired
bootstrap delta CIs for fused ensemble+self-certainty minus self-certainty-alone
and recall-at-fixed-FPR tables.  If token-level logits or probabilities are not
available in the cached corpora, the runner MUST use a disclosed proxy and state
the verifier-authenticity gap rather than silently relabeling plain confidence.

**Acceptance criteria:**
- The artifact includes `honest_verdict`, `inference_substrate`,
  `self_certainty_auroc_per_domain`,
  `plain_confidence_auroc_per_domain`,
  `fused_ensemble_self_certainty_auroc_per_domain`,
  `ensemble_minus_self_certainty_delta_ci_per_domain`,
  `self_certainty_implementation`,
  `ensemble_adds_value_over_self_certainty`, `n_examples_per_domain`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- `ensemble_adds_value_over_self_certainty` is a bare top-level bool and is
  true only when at least one domain has fused-minus-self-certainty paired delta
  CI excluding zero on the positive side.
- The honest verdict is one of:
  `complete: ensemble_adds_value_over_self_certainty_product_value_robust`,
  `complete: product_value_collapses_vs_self_certainty_claim_narrowed`, or
  `complete: blocked_no_labeled_corpus_for_rebaseline`.
- The runner keeps per-domain null discipline: a collapsed edge against
  self-certainty narrows only the detector product-value claim and does not
  affect the separate FoVer discrimination headline.
- Tests cover `ensemble_adds_value_over_self_certainty`,
  `value_collapses_vs_stronger_baseline`, and `blocked` with synthetic fixtures
  rather than hard-coding a real-corpus success string.

### REQ-SPOE-3695: Code-Native Second-Pair Verifier Verdict

The pipeline MUST provide an Exp 3695 runner that evaluates a genuinely
code-native verifier on the balanced Exp 3658 code corpus without live LLM
generation when the cached corpus is available:
- The runner checks that the balanced Exp 3658 corpus is loadable and that the
  CodeExtractor / AST tooling can be imported before scoring.
- The verifier parses candidate Python code and produces AST/structural
  features such as parse failures, undefined names, missing value returns,
  early unconditional returns, and branch/loop structure.  When runnable code
  has an entry point, it also executes deterministic runtime probes through the
  existing execution path and records the execution-trace signal.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `code_auroc_baseline`, `code_native_auroc`, `code_native_auroc_ci`,
  `code_native_calibration_brier_ece`, `code_native_recall_at_fixed_fpr`,
  `code_native_verifier_implementation`, `code_signal_recovered`,
  `n_examples_code`, `adversarial_verify_clean`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- `code_signal_recovered` is a bare top-level boolean and may be true only when
  code-native AUROC beats 0.5 with a CI excluding 0.5 and calibration improves
  against the reconfirmed code baseline.
- The runner records `inference_substrate` as
  `verifier_ensemble_against_cached_candidates` for cached-only scoring and
  uses `live_llm_inference` only if a real GGUF generation step runs.
- The honest verdict is one of:
  `complete: code_native_signal_recovered_beats_chance_floor`,
  `complete: code_remains_math_only_code_native_signal_also_fails_earned`, or
  `complete: blocked_no_code_corpus_or_ast_tooling`.
- Tests cover recovered, math-only, and blocked outcomes using synthetic
  fixtures rather than hard-coding a real-corpus success string.

### REQ-SPOE-3696: Re-ship Detector With Math+Code Operating Point

The pipeline MUST re-ship the `second_pair_detector` surface when Exp 3695 has
already recovered a code-native signal.  The shipped detector MUST use the Exp
3695 AST/runtime verifier score for code-domain ensemble energy while preserving
the math-domain operating point from Exp 3671.  The re-ship artifact MUST
measure the updated math and code AUROC/calibration on cached corpora, call the
shipped `score_candidates` surface end-to-end, and write
`results/experiment_3696_reship_detector_math_plus_code.json`.

**Acceptance criteria:**
- The runner checks `results/experiment_3695_code_native_verifier.json` and
  blocks unless `code_signal_recovered == true` and the detector module is
  importable.
- The shipped module's code loading and runtime code candidate scoring paths use
  the Exp 3695 code-native verifier score, not the older math-transfer code
  score.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `module_code_path_updated`, `math_operating_point_unchanged`,
  `code_operating_point_auroc`, `code_operating_point_calibration`,
  `e2e_test_passed`, `adversarial_verify_clean`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- `module_code_path_updated`, `math_operating_point_unchanged`,
  `e2e_test_passed`, and `adversarial_verify_clean` are bare top-level
  booleans.
- `math_operating_point_unchanged` is true only when the remeasured math AUROC
  still rounds to the Exp 3671 0.98 operating point and math calibration does
  not regress.
- The honest verdict is one of:
  `complete: detector_reshipped_math_plus_code_operating_point_e2e_green` or
  `complete: blocked_code_signal_not_recovered_or_module_unavailable`.
- Tests cover `detector_math_plus_code_shipped` and `blocked` with synthetic
  fixtures rather than hard-coding a real-corpus success string.

### REQ-SPOE-3706: Held-Out Reconciliation for Shipped Detector Code Claim

The pipeline MUST reconcile the shipped `second_pair_detector` code operating
point against the Exp 3705 held-out leak audit unconditionally.  If Exp 3705
reports `code_signal_survives_heldout == true`, the shipped code operating
point MUST be recalibrated to the Exp 3705 held-out AUROC and calibration
rather than the inflated in-corpus 1.0.  If Exp 3705 reports
`leak_detected == true` or `code_signal_survives_heldout == false`, the shipped
surface MUST narrow to math-only and return an explicit no-code-verdict
abstention for code-domain candidates.

**Acceptance criteria:**
- The runner checks that `results/experiment_3705_code_native_leak_audit_heldout.json`
  exists and that `python/carnot/pipeline/second_pair_detector.py` is importable
  before reconciling.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `reconciliation_action`, `shipped_code_operating_point_auroc`,
  `math_operating_point_unchanged`, `overclaim_removed`, `e2e_test_passed`,
  `operating_envelope_docstring_updated`, `adversarial_verify_clean`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- `overclaim_removed`, `math_operating_point_unchanged`, `e2e_test_passed`,
  `operating_envelope_docstring_updated`, and `adversarial_verify_clean` are
  bare top-level booleans.
- `shipped_code_operating_point_auroc` is the Exp 3705 held-out AUROC when the
  code signal survives held-out audit; otherwise it is `null` because the
  shipped code surface abstains instead of emitting a code verdict.
- `math_operating_point_unchanged` is true only when the strong math operating
  point still rounds to AUROC 0.98 and ECE 0.009.
- The honest verdict is one of:
  `complete: shipped_detector_code_recalibrated_to_heldout_e2e_green`,
  `complete: shipped_detector_narrowed_to_math_only_abstain_on_code_e2e_green`,
  or `complete: blocked_heldout_audit_unavailable`.
- Tests cover `code_operating_point_recalibrated_to_heldout`,
  `narrowed_to_math_only_abstain_on_code`, and `blocked` with synthetic
  fixtures rather than hard-coding a real-corpus success string.

### REQ-SPOE-3718: FoVer Risk-Coverage Abstention Characterization

The pipeline MUST provide an Exp 3718 runner that characterizes the frozen
FoVer step-error discriminator as a selective-prediction abstention signal,
not as a best-of-N selector.  The runner MUST load the cached FoVer per-step
labels, verifier ensemble energy scores, and a same-corpus entropy /
self-certainty baseline, compute risk-coverage curves for each signal, and
report AURC, risk at fixed coverage, coverage at a 5% selective-risk target,
calibration, CIs over at least five deterministic seeds, and an honest terminal
verdict.  If the per-step scores are unavailable, the runner MUST block without
fabricating metrics.

**Acceptance criteria:**
- The runner checks that the FoVer per-step score corpus is cached and contains
  at least 1000 scored math rows before reporting runnable metrics.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `energy_aurc`, `baseline_aurc`, `energy_beats_baseline_abstention`,
  `coverage_at_5pct_risk`, `risk_at_fixed_coverage`, `energy_aurc_ci`,
  `n_seeds`, `n_examples`, `calibration_brier_ece`,
  `adversarial_verify_clean`, `random_seed`, `reproducibility_checksum`, and
  `duration_s`.
- `energy_beats_baseline_abstention` is a bare top-level bool and is true only
  when energy has lower AURC than the baseline and the paired CI separates in
  favor of energy at the AURC or at least one fixed-coverage operating point.
- AURC, AUROC, and fixed-coverage risk are reported as distinct metrics; the
  runner MUST reject bit-identical values across these metric classes.
- If energy AURC implies AUROC >= 0.99 on at least 1000 rows, the runner MUST
  treat the result as a leak guard failure and return the honest negative or
  blocked verdict with diagnostic evidence rather than publishing a positive
  abstention signal.
- The honest verdict is one of:
  `complete: energy_is_a_better_selective_prediction_signal_than_entropy_deployable_abstention_gate`,
  `complete: energy_ties_or_loses_to_entropy_as_abstention_signal_honest_negative`,
  or `complete: blocked_fover_perstep_scores_unavailable`.
- Tests cover `energy_better_abstention_signal`,
  `energy_ties_or_loses_to_entropy`, and `blocked` with synthetic fixtures
  rather than hard-coding a real-corpus success string.

### REQ-SPOE-3836: Formal-Core Certified Abstention Operating Point

The pipeline MUST provide an Exp 3836 runner that certifies a deployable
abstention threshold for the contamination-free FoVer formal core alone.  The
runner MUST gate on Exp 3835 reporting `formal_only_auroc_mean >= 0.85`, MUST
check that `carnot.verify` imports and `data/fover_test_v4.json` is present,
MUST score the Exp 3771-aligned cached FoVer candidate rows using only
`0.9*tier0r_curry_howard + 0.1*tier0u_logical_consistency`, and MUST NOT
include `fr11_session_memory` or any trained-weight scorer in the formal-core
scalar.

The runner MUST build a risk-coverage curve over the formal-core score, reuse
the Exp 3771 split-conformal calibration/test protocol at selective-risk target
0.05 and delta 0.05, compare certified coverage against the Exp 3771 full
ensemble operating point, and write
`results/experiment_3836_formal_core_certified_abstention_operating_point.json`.
If any precondition fails, the runner MUST write a blocked artifact with an
`honest_verdict` beginning with `blocked_` and MUST NOT fabricate threshold,
coverage, or AURC metrics.

**Acceptance criteria:**
- The artifact includes `formal_core_certified_threshold`,
  `formal_core_certified_coverage_at_risk_0_05`,
  `coverage_delta_vs_full_ensemble`, `conformal_delta`, `n_calibration`,
  `n_test`, `cited_upstream_artifacts`, `preconditions_checked`,
  `honest_verdict`, `random_seed`, `reproducibility_checksum`, `duration_s`,
  `inference_substrate`, and a `field_provenance` block that states the
  principle for every required field.
- `cited_upstream_artifacts` records the Exp 3835 and Exp 3771 artifact paths
  and SHA256 checksums, plus the Exp 3835 formal-only AUROC gate value and the
  Exp 3771 full-ensemble certified coverage reference.
- The honest verdict is
  `complete: formal_core_certified_abstention_threshold<t>_coverage<c>_at_risk_0.05_contamination_free`
  when a certified threshold exists at coverage greater than 0.90; otherwise it
  is
  `complete: formal_core_certified_abstention_WEAK_coverage<c>_clean_core_low_coverage_full_ensemble_remains_product`.
- Tests cover `formal_core_certified_threshold_shipped`,
  `formal_core_certified_threshold_weak`, `blocked_precondition`, and
  `no_fr11_session_memory_in_formal_score` with synthetic fixtures rather than
  hard-coding a real-corpus success string.

### REQ-SPOE-3779: Certified Abstention Operating Point Product Surface

The shipped `score_candidates` verifier-scoring surface MUST expose an opt-in
abstention mode that is disabled by default.  When disabled, callers MUST
receive the same score rows as the existing Phase-1 surface.  When enabled,
each score row MUST include an abstention decision derived from a
configurable certified operating point whose default values are loaded from
`results/experiment_3771_certified_abstention_operating_point.json` rather
than embedded as unexplained literals.

The abstention mode MUST orient its product score so larger values mean a more
confident verifier judgment.  Rows at or above the configured certified
threshold MUST return a confident verdict.  Rows below the threshold MUST
return an abstain verdict that routes the row to review and carries the
certified coverage, certified risk bound, delta, calibration sample size, and
threshold source metadata.  Operator overrides MAY supply the threshold through
the API, MCP tool, or CLI without changing the default certified source.

The Exp 3779 runner MUST check the `.venv/bin/python` interpreter can import
`carnot`, MUST read the certified threshold from the absolute Exp 3771 artifact
before claiming the product surface is wired, MUST run a no-live-LLM wiring
smoke over cached FoVer scoring data, SHOULD confirm the MCP `score_candidates`
tool through a real stdio protocol exchange when the MCP runtime is available,
MUST write a separate documentation-update proposal instead of editing the
operator-curated MCP or CLI docs, and MUST write
`results/experiment_3779_abstention_operating_point_product_wiring.json`.

The terminal artifact MUST include `honest_verdict`, `inference_substrate`,
`abstention_mode_wired`, `default_off_preserves_prior_behavior`,
`certified_threshold_used`, `e2e_abstention_passed`,
`mcp_surface_confirmed`, `doc_proposal_emitted_not_curated_edit`,
`tests_assert_real_behavior`, `model_specs`, `random_seed`,
`reproducibility_checksum`, and `duration_s`.

### REQ-SPOE-3789: Abstention CLI Batch Surface

The shipped CLI MUST expose a batch verification surface that reads more than
one candidate from a file and can opt into the certified Exp 3771 abstention
operating point without changing default behavior.  The batch input MUST accept
either a JSON array of candidate objects or one candidate per non-empty line.
Line-delimited inputs MAY be JSON objects or raw candidate text rows.

When abstention mode is disabled, the CLI batch surface MUST preserve the
existing calibrated `score_candidates` row shape and omit abstention verdict
fields.  When abstention mode is enabled, every scoreable row MUST include the
confident-or-abstain verdict, review route, certified threshold, coverage,
certified risk bound, delta, calibration sample size, and threshold-source
metadata loaded through the same certified-abstention config used by the Python
and MCP surfaces.  The batch surface MUST process at least two candidates in a
single invocation and emit structured JSON.

The Exp 3789 runner MUST check the `.venv/bin/python` interpreter can import
`carnot`, MUST load the certified threshold from the absolute Exp 3771 artifact
before claiming the CLI surface is wired, MUST run a no-live-LLM CLI wiring
smoke over cached FoVer-style verifier-scoring candidates, MUST write a
documentation-update proposal instead of editing operator-curated CLI/MCP docs,
and MUST write
`results/experiment_3789_abstention_cli_batch_surface.json`.

The terminal artifact MUST include `honest_verdict`, `inference_substrate`,
`cli_abstention_surface_added`, `batch_path_works`,
`default_off_preserves_prior_behavior`, `certified_threshold_used`,
`e2e_cli_abstention_passed`, `doc_proposal_emitted_not_curated_edit`,
`tests_assert_real_behavior`, `model_specs`, `random_seed`,
`reproducibility_checksum`, and `duration_s`.

### REQ-SPOE-3801: Abstention HTTP REST Surface

The shipped verifier-scoring product MUST expose a minimal HTTP/REST POST
surface for non-Python integrators to score one or more candidates and opt into
the certified Exp 3771 abstention operating point.  The endpoint MUST use only
the existing lightweight dependency set; if no web framework is already used
for this product path, it MUST use the Python standard-library HTTP server
rather than adding a heavy framework dependency.

The HTTP payload MUST accept either a single candidate object or a batch of
candidate objects.  Abstention mode MUST remain disabled by default; when
disabled, response rows MUST preserve the existing calibrated
`score_candidates` row shape and omit confident-or-abstain verdict fields.
When a request enables abstention mode, each scoreable row MUST return a
network-facing verdict of `confident` or `abstain`, the confidence-oriented
abstention score, and certified Exp 3771 coverage, risk, delta, threshold, and
threshold-source metadata loaded through the same certified-abstention config
used by the Python, MCP, and CLI surfaces.

The Exp 3801 runner MUST check the `.venv/bin/python` interpreter can import
`carnot` and the abstention-mode parameter path, MUST read the certified
threshold from the absolute Exp 3771 artifact before claiming the HTTP surface
is wired, MUST start the HTTP endpoint on a local test port and POST a
no-live-LLM batch of cached FoVer-style verifier-scoring candidates, MUST
confirm above-threshold-to-confident, below-threshold-to-abstain, default-off,
and batch processing behavior over the network, MUST write a documentation
proposal instead of editing operator-curated CLI/MCP docs, and MUST write
`results/experiment_3801_abstention_http_rest_surface.json`.

The terminal artifact MUST include `honest_verdict`, `inference_substrate`,
`http_rest_surface_added`, `batch_post_works`,
`default_off_preserves_prior_behavior`, `certified_threshold_used`,
`e2e_http_abstention_passed`, `no_heavy_new_dependency`,
`doc_proposal_emitted_not_curated_edit`, `tests_assert_real_behavior`,
`model_specs`, `random_seed`, `reproducibility_checksum`, and `duration_s`.

### REQ-SPOE-3810: Abstention HTTP REST Surface Repair

The Exp 3801 HTTP/REST abstention surface repair MUST diagnose the blocked
E2E assertion before claiming completion and MUST preserve the minimal
standard-library HTTP implementation.  The repair MUST show that the endpoint
loads the certified abstention operating point through the same configuration
path used by the wiring smoke, with the default threshold sourced from the
absolute Exp 3771 artifact rather than a hard-coded value.

The repaired endpoint MUST accept either one candidate or a batch, keep
abstention disabled by default, and when abstention is explicitly enabled,
return per-candidate `verdict`, `score`, `coverage`, `risk`, and `delta`
metadata over HTTP.  The repair E2E MUST exercise a real above-threshold
candidate that returns `confident`, a real below-threshold candidate that
returns `abstain`, default-off prior behavior, and a batch POST containing
more than one candidate.  The smoke MUST use cached verifier-scoring examples
only and MUST not make an accuracy claim.

The Exp 3810 runner MUST write
`results/experiment_3810_abstention_http_rest_surface_v2.json` and include
`honest_verdict`, `inference_substrate`, `e2e_failure_root_cause`,
`http_rest_surface_added`, `batch_post_works`,
`default_off_preserves_prior_behavior`, `certified_threshold_used`,
`e2e_http_abstention_passed`, `no_heavy_new_dependency`,
`doc_proposal_emitted_not_curated_edit`, `tests_assert_real_behavior`,
`model_specs`, `random_seed`, `reproducibility_checksum`, and `duration_s`.

### REQ-SPOE-3811: Abstention Cross-Surface Parity Smoke

The pipeline MUST provide an Exp 3811 no-live-LLM parity smoke that compares
the certified Exp 3771 abstention operating point across the verify API,
CLI/batch, and HTTP/REST surfaces.  The smoke MUST first confirm that the
`.venv/bin/python` interpreter imports `carnot`, the Exp 3810 artifact reports
`http_rest_surface_added=true`, all three surfaces are invokable, and cached
FoVer verifier-scoring examples are reachable by absolute path.  If any
precondition is missing, it MUST write an honest `blocked_<resource>` artifact
without fabricating parity evidence.

When runnable, the smoke MUST choose a deterministic fixed sample of at least
10 cached FoVer-style verifier-scoring candidates spanning both scores at or
above the certified threshold and scores below it.  It MUST call the verify
API, the CLI batch path, and the HTTP/REST endpoint with the same certified
threshold loaded from Exp 3771 and compare each candidate's
confident-or-abstain verdict plus coverage, risk, delta, and threshold
metadata within float tolerance.  A mismatch MUST be reported as drift in the
artifact rather than hidden.

The Exp 3811 runner MUST write
`results/experiment_3811_abstention_cross_surface_parity_smoke.json` and
include `honest_verdict`, `inference_substrate`, `surfaces_compared`,
`all_surfaces_agree`, `n_candidates_compared`, `mismatches`,
`certified_threshold_used`, `tests_assert_real_behavior`,
`cited_upstream_artifacts`, `model_specs`, `random_seed`,
`reproducibility_checksum`, and `duration_s`.

## Scenarios

### SCENARIO-INFRA-052: Version-Blocked Model Raises Error

**Given** the exclusion manifest marks `jepa_v17_blocked: true`
**When** code calls `tier2_jepa.load_v18_from_manifest(version="v17")`
**Then** a `ValueError` is raised with message containing "blocked"

**Spec traces:** REQ-INFRA-043

### SCENARIO-INFRA-053: v18 Loads and Scores Successfully

**Given** a trained `JEPALambdaRankV18` instance
**When** `predict_score("Step 1: 3 + 5 = 8.")` is called
**Then** a scalar float is returned without raising any exception

**Spec traces:** REQ-INFRA-043

### SCENARIO-INFRA-054: Cascade AUC on Held-Out Groups

**Given** the cascade is loaded with JEPA v18 as Tier 2
**When** `evaluate_auc(eval_groups)` is called on 50 held-out GSM8K groups
**Then** the returned float is in [0, 1]

### SCENARIO-FST-2240: Failed Verification Feeds Next Repair Prompt

**Given** a verify-repair loop running with `use_fst=True`
**When** a verification iteration finds one or more violations
**Then** the next LLM repair call receives a terminal-prefix verifier-output summary
before the normal repair prompt.

**Spec traces:** REQ-FST-2240

### SCENARIO-ODAR-2243: Low EFE Skips Deliberative Verification

**Given** Tier 0 probe outputs with low risk and high confidence
**When** `FreeEnergyRouter(risk_threshold=0.5).route(...)` evaluates the outputs
**Then** the decision is `FAST_PATH`, and the verify-repair pipeline can return a
fast-path result before Tier 1 extraction.

**Spec traces:** REQ-ODAR-2243

### SCENARIO-ODAR-2244: ODAR Benchmark Clears Compute Gate

**Given** the balanced 30-example Exp 2244 reasoning corpus
**When** the benchmark runs the uniform cascade and ODAR-threshold regimes
**Then** the artifact reports at least 30% fewer tier calls for ODAR while keeping
accuracy within two percentage points of the uniform cascade.

**Spec traces:** REQ-ODAR-2244

### SCENARIO-ODAR-2257: Real Probe EFE Clears Routing Overhead Gate

**Given** 100 synthetic reasoning examples and the verify-repair Tier 0 semantic
energy probe
**When** the benchmark routes each example through `FreeEnergyRouter` using the
actual Tier 0 probe records
**Then** the artifact reports at least 25% fewer cascade tier calls, median
routing overhead no greater than 5 ms, and accuracy within two percentage points
of the uniform cascade.

**Spec traces:** REQ-ODAR-2257

### SCENARIO-FST-2399: Cached Telemetry Completes When Live Paths Fail

**Given** no usable live llama.cpp or transformers runner and a readable
`results/live_sota_balanced_telemetry_manifest_1480.jsonl`
**When** Exp 2399 runs
**Then** it MUST run the FST verify/terminal-prefix flow over cached telemetry,
write `results/experiment_2399_fst_live_path_ab.json`, set
`fst_live_validated=true`, and record `live_path_used="C_cached"`.

**Spec traces:** REQ-FST-2399

### SCENARIO-SPOE-3657: Fusion Outcomes Are Honest

**Given** synthetic labeled domains covering `fusion_wins`,
`fusion_redundant`, and `blocked` outcomes
**When** the deployable second-pair detector evaluation runs
**Then** it returns the corresponding terminal verdict without hard-coding a
real-corpus success string.

**Spec traces:** REQ-SPOE-3657, REQ-SPOE-3657-ARTIFACT

### SCENARIO-SPOE-3658: Fixed-FPR Operating Points Are Per-Domain

**Given** a held-out domain with both correct and error examples
**When** the detector computes operating points
**Then** recall is reported at FPR budgets `0.05`, `0.10`, and `0.20`, and the
recommended operating point is chosen from the same table.

**Spec traces:** REQ-SPOE-3657

### SCENARIO-SPOE-3671: Shipped Detector Verdicts Stay Honest

**Given** synthetic fixture domains covering
`ships_math_and_code`, `ships_math_only_code_weak`, and `blocked`
**When** the Exp 3671 detector ship artifact is built
**Then** it returns the corresponding terminal verdict and bare
`detector_shipped` bool without hard-coding a real-corpus success string.

**Spec traces:** REQ-SPOE-3671, REQ-SPOE-3671-ARTIFACT

### SCENARIO-SPOE-3672: Score Candidates Surface Returns Calibrated Scores

**Given** a fitted detector context with held-out operating points
**When** the shipped `score_candidates` surface is called through the package
surface
**Then** each candidate receives a calibrated error score in `[0, 1]` and a
per-domain operating point from the held-out evaluation table.

**Spec traces:** REQ-SPOE-3671

### SCENARIO-SPOE-3673: Balanced Code Corpus Is Preferred

**Given** both the old code corpus and the Exp 3658 balanced code corpus are
present
**When** labeled detector examples are loaded for Exp 3671
**Then** the code-domain rows come from `data/code_verification_corpus_v2.jsonl`.

**Spec traces:** REQ-SPOE-3671

### SCENARIO-SPOE-3769: Package CLI MCP E2E Smoke Is Real Protocol Evidence

**Given** the venv interpreter, importable package, importable MCP server, MCP
runtime, cached small CPU smoke model, and resolvable CLI module
**When** Exp 3769 runs
**Then** it writes the package import, pipeline, MCP protocol, and CLI surface
outcomes as bare booleans, calls MCP through stdio JSON-RPC rather than an
in-process function, labels the run as a wiring smoke, and returns the terminal
verdict
`complete: phase1_e2e_smoke_package_import_pipeline_mcp_protocol_cli_passed_wiring_smoke_not_accuracy_claim`
only when all four surfaces pass.

**Spec traces:** REQ-SPOE-3769

### SCENARIO-SPOE-3683: Code Operating Point Outcomes Stay Honest

**Given** synthetic code-operating-point fixtures covering
`code_operating_point_recovered`, `code_remains_math_only`, and `blocked`
**When** the Exp 3683 verdict artifact is assembled
**Then** it returns the corresponding terminal verdict, preserves
`code_operating_point_recovered` as a bare bool, and only counts AUROC signal
when the CI excludes the 0.5 chance floor.

**Spec traces:** REQ-SPOE-3683

### SCENARIO-SPOE-3684: Product Value Outcomes Stay Honest Against Self-Certainty

**Given** synthetic rebaseline fixtures covering
`ensemble_adds_value_over_self_certainty`,
`value_collapses_vs_stronger_baseline`, and `blocked`
**When** the Exp 3684 product-value artifact is assembled
**Then** it returns the corresponding terminal verdict, preserves
`ensemble_adds_value_over_self_certainty` as a bare bool, and only counts
additive value when the paired delta CI excludes zero.

**Spec traces:** REQ-SPOE-3684

### SCENARIO-SPOE-3695: Code-Native Verdicts Stay Honest

**Given** synthetic code-native verifier fixtures covering
`code_signal_recovered`, `code_remains_math_only`, and `blocked`
**When** the Exp 3695 artifact is assembled
**Then** it returns the corresponding terminal verdict, preserves
`code_signal_recovered` as a bare bool, reports the AUROC CI separately from the
point estimate, and only counts signal when the CI excludes the 0.5 chance
floor and calibration improves.

**Spec traces:** REQ-SPOE-3695

### SCENARIO-SPOE-3696: Math+Code Re-ship Outcomes Stay Honest

**Given** synthetic Exp 3696 fixtures for `detector_math_plus_code_shipped` and
`blocked`
**When** the Exp 3696 re-ship artifact is assembled
**Then** terminal verdicts, bare booleans, code AUROC, and calibration fields are
derived from the fixture measurements instead of hard-coded to a real-corpus
success string.

**Spec traces:** REQ-SPOE-3696

### SCENARIO-SPOE-3706: Held-Out Reconciliation Outcomes Stay Honest

**Given** synthetic Exp 3705 audit fixtures covering
`code_operating_point_recalibrated_to_heldout`,
`narrowed_to_math_only_abstain_on_code`, and `blocked`
**When** the Exp 3706 reconciliation artifact is assembled
**Then** it derives the terminal verdict, shipped code AUROC/null, bare
overclaim-removal bool, math-preservation bool, and E2E result from the
fixture measurements rather than hard-coding a real-corpus success string.

**Spec traces:** REQ-SPOE-3706

### SCENARIO-SPOE-3718: Selective-Prediction Outcomes Stay Honest

**Given** synthetic FoVer step fixtures covering
`energy_better_abstention_signal`, `energy_ties_or_loses_to_entropy`, and
`blocked`
**When** the Exp 3718 risk-coverage artifact is assembled
**Then** it derives the terminal verdict, bare
`energy_beats_baseline_abstention` bool, AURC values, fixed-coverage risk
points, and coverage-at-5%-risk operating point from the fixture measurements
rather than hard-coding a real-corpus success string.

**Spec traces:** REQ-SPOE-3718

### SCENARIO-SPOE-3836: Formal-Core Certification Stays Contamination-Free

**Given** synthetic FoVer step fixtures covering
`formal_core_certified_threshold_shipped`,
`formal_core_certified_threshold_weak`, `blocked_precondition`, and
`no_fr11_session_memory_in_formal_score`
**When** the Exp 3836 formal-core certified abstention artifact is assembled
**Then** it derives the formal-only score from tier0r and tier0u only, reports
the split-conformal threshold and certified coverage when the risk bound holds,
reports the honest weak verdict when it does not, blocks on missing or weak
upstream prerequisites, and records the Exp 3835 / Exp 3771 SHA256 provenance
without editing operator-curated docs.

**Spec traces:** REQ-SPOE-3836

### SCENARIO-SPOE-3779: Abstention Mode Is Opt-In And Protocol-Reachable

**Given** the Exp 3771 certified operating point artifact is readable and the
Phase-1 `score_candidates` surface can score math candidates from cached FoVer
data
**When** abstention mode is disabled
**Then** the returned score rows do not contain abstention-mode verdict fields
and preserve the prior calibrated score and operating-point behavior.

**When** abstention mode is enabled
**Then** a candidate whose abstention score is at or above the certified
threshold returns a confident verdict, a candidate below the threshold returns
`uncertain / route to review`, and every abstain row carries the Exp 3771
coverage, certified risk, delta, calibration sample size, and threshold-source
metadata.

**When** the Exp 3779 runner performs the MCP confirmation
**Then** it calls the packaged `score_candidates` MCP tool through stdio
JSON-RPC, not an in-process handler, and records whether the external surface
accepted the opt-in abstention-mode parameter.

**Spec traces:** REQ-SPOE-3779

### SCENARIO-SPOE-3789: CLI Batch Abstention Is Default-Off

**Given** a readable batch candidate file containing at least two cached
verifier-scoring candidates and the Exp 3771 certified operating point
**When** the CLI batch surface runs without `--abstention-mode`
**Then** each row preserves the existing calibrated score shape without
abstention verdict fields.

**When** the same CLI batch surface runs with `--abstention-mode`
**Then** the above-threshold candidate returns a confident verdict, the
below-threshold candidate returns `uncertain / route to review`, both rows carry
certified Exp 3771 metadata, and the single CLI invocation reports more than one
processed candidate.

**Spec traces:** REQ-SPOE-3789

### SCENARIO-SPOE-3801: HTTP REST Abstention Is Default-Off And Batch-Capable

**Given** a local HTTP endpoint is serving the packaged verifier-scoring
surface, a request body contains at least two cached FoVer-style
verifier-scoring candidates, and the Exp 3771 certified operating point is
readable
**When** the caller POSTs without enabling abstention mode
**Then** each response row preserves the existing calibrated score shape
without network-facing `confident` or `abstain` verdict fields.

**When** the caller POSTs the same batch with abstention mode enabled
**Then** the above-threshold candidate returns `confident`, the
below-threshold candidate returns `abstain`, each row carries the certified
coverage, risk, delta, and threshold metadata, and the single HTTP request
reports more than one processed candidate.

**Spec traces:** REQ-SPOE-3801

### SCENARIO-SPOE-3810: HTTP REST Repair Confirms Real Abstention Branch

**Given** the Exp 3771 certified abstention artifact is readable, a local
HTTP endpoint is serving the packaged verifier-scoring surface, and the repair
runner has reproduced the Exp 3801 failed assertion
**When** the caller POSTs a cached verifier-scoring batch with abstention mode
enabled
**Then** one row with score at or above the certified threshold returns
`confident`, one row with score below the certified threshold returns
`abstain`, both rows carry certified metadata from the configured threshold
artifact, and the same endpoint still preserves default-off score row shape
when the flag is omitted.

**Spec traces:** REQ-SPOE-3810

### SCENARIO-SPOE-3811: Verify API CLI And HTTP Abstention Metadata Match

**Given** Exp 3771, Exp 3779, Exp 3789, and Exp 3810 artifacts are readable,
Exp 3810 reports `http_rest_surface_added=true`, and cached FoVer-style
verifier-scoring candidates are reachable
**When** the Exp 3811 smoke scores the fixed candidate sample through verify
API, CLI/batch, and HTTP/REST with abstention mode enabled
**Then** every candidate has the same `confident` or `abstain` verdict across
all three surfaces, the certified coverage, risk, delta, and threshold metadata
match within float tolerance, at least one confident and one abstain candidate
are exercised, and any mismatch is recorded in the artifact as drift.

**Spec traces:** REQ-SPOE-3811

### REQ-SAMPLE-020: SparseIsingEBM K-Regular Graph

`SparseIsingEBM` MUST implement a K-regular sparse connectivity graph where each spin
has exactly `n_neighbors` neighbors, no spin is its own neighbor, and all neighbor
indices are valid spin indices in `[0, n_vars)`. The sparse energy computation MUST
use only K-neighbor sums (O(N*K) instead of O(N^2)) and produce a scalar matching
the manual sparse sum formula:
    E(s) = -0.5 * sum_i sum_{j in nbrs(i)} J_sparse[i,k] * s_i * s_j - b^T s

**Acceptance criteria:**
- `SparseIsingEBM(n_vars=64, n_neighbors=16)` constructs without error.
- `neighbor_idx.shape == (n_vars, n_neighbors)` with integer dtype.
- `J_sparse.shape == (n_vars, n_neighbors)` with float dtype.
- No self-loops: `i not in neighbor_idx[i]` for all i.
- `energy(spins)` matches manual computation within 1e-4.
- `ValueError` raised when `n_neighbors >= n_vars` or `n_neighbors < 2` or odd.

### SCENARIO-SAMPLE-035: Sparse vs Dense Convergence Comparison

**Given** a `SparseIsingEBM` with `n_vars=64` and `n_neighbors=16`
**When** `compare_with_dense(n_trials=10)` is called
**Then** the result dict contains keys `steps_dense_mean`, `steps_sparse_gibbs_mean`,
`steps_emvl_mean`, `speedup_ratio_emvl_vs_dense`, `speedup_ratio_gibbs_vs_dense`,
all values are finite, and all step counts are non-negative.

**Given** `energy_trajectory(n_steps, sampler="emvl")` is called
**Then** the returned list has length `n_steps + 1` and all values are finite.

**Spec traces:** REQ-SAMPLE-020

**Spec traces:** REQ-INFRA-044

### REQ-INFRA-046: EORM Confidence Gate for Tier 3 Ising Skip

The cascade router MUST support an EORM confidence gate that skips Tier 3 Ising
sampling when the EORM confidence score exceeds a configurable threshold (default 0.92).
When EORM confidence > eorm_ising_skip_threshold, the result is marked "verified_fast"
and Tier 3 Ising is not run.  The threshold is configurable at CascadeRouter
construction time.  Each query MUST log ising_skip (bool) and eorm_confidence (float).

**Acceptance criteria:**
- `CascadeRouter(eorm_ising_skip_threshold=0.92)` skips Ising when EORM confidence > 0.92.
- `CascadeRouter(eorm_ising_skip_threshold=0.92)` runs Ising when EORM confidence <= 0.92.
- Per-query logs contain ising_skip and eorm_confidence fields.

### REQ-INFRA-047: EORM Gate False-Negative Delta

The EORM confidence gate MUST NOT increase the false-negative rate by more than 5
percentage points versus the full cascade (no gate) on a representative test set.
Formally: fn_delta = false_negative_rate_gated - false_negative_rate_baseline < 0.05.

**Acceptance criteria:**
- Exp 727 measures fn_delta < 0.05 on 200-question test set at threshold=0.92.

## Scenarios

### SCENARIO-INFRA-055: EORM Gate Skips Ising Above Threshold

**Given** a CascadeRouter with eorm_ising_skip_threshold=0.92
**When** EORM returns confidence=0.95 for a query
**Then** Tier 3 Ising is NOT invoked and result is marked "verified_fast"

**Spec traces:** REQ-INFRA-046

### SCENARIO-INFRA-056: EORM Gate Does Not Skip Ising Below Threshold

**Given** a CascadeRouter with eorm_ising_skip_threshold=0.92
**When** EORM returns confidence=0.80 for a query
**Then** Tier 3 Ising IS invoked as normal

**Spec traces:** REQ-INFRA-046

### REQ-INFRA-046b: Conductor Dispatch Manifest Enforcement

The conductor MUST call `validate_manifest_at_dequeue(task_id)` before dispatching
any experiment to execution.  If the function returns False (task_id is in the exclusion
manifest), the task MUST be silently skipped — no agent spawned, no GPU allocated.

**Acceptance criteria:**
- `validate_manifest_at_dequeue("exp308-legacy")` returns False when exp 308 is in manifest.
- `validate_manifest_at_dequeue("exp999-new")` returns True when exp 999 is not in manifest.
- Retired tasks are never dispatched to an agent subprocess.

**Spec traces:** REQ-INFRA-046b (replaces text-level-only exclusion, closes .55 787-min gap)

### REQ-INFRA-047b: GPU VRAM Clean at Milestone Start

All GPU devices MUST have < 100 MB VRAM allocated at the start of each conductor
milestone.  If any device exceeds 100 MB, the conductor MUST kill the holding process
before dispatching the first experiment.

**Acceptance criteria:**
- `gpu1_vram_mb < 100` measured after zombie kill at milestone start.
- Conductor pre-flight logs the before/after VRAM delta.

### SCENARIO-INFRA-055b: Manifest Validator Blocks Excluded Task

**Given** `conductor_exclusion_manifest.json` lists experiment_id=308
**When** `validate_manifest_at_dequeue("exp308-legacy")` is called
**Then** the function returns False and logs "task_id=exp308-legacy allowed=False"

**Spec traces:** REQ-INFRA-046b

### SCENARIO-INFRA-056b: Manifest Validator Passes Unknown Task

**Given** `conductor_exclusion_manifest.json` does not list experiment_id=999
**When** `validate_manifest_at_dequeue("exp999-new")` is called
**Then** the function returns True and logs "task_id=exp999-new allowed=True"

**Spec traces:** REQ-INFRA-046b

### REQ-INFRA-048: Exp 527 Class Mandatory Retirement

Exp 527 (live 100-question precision inference) MUST be present in the conductor
exclusion manifest before milestone 2026.04.57 dequeue.  This retirement is mandated by
governance rule "3-consecutive-mandatory": an experiment that appears in the slowest-5
for three consecutive milestones is automatically retired regardless of research value.

**Acceptance criteria:**
- `ExclusionManifest.is_excluded(527)` returns True after Exp 740 runs.
- The manifest entry includes `governance_rule: "3-consecutive-mandatory"`.
- The entry includes `retired_in_milestone: "2026.04.57"`.

**Spec traces:** REQ-INFRA-048 (governance: Exp 308/309 precedent, RETRO-033)

### REQ-INFRA-049: EORM+JEPA Retrain MUST Use DualGPU ThreadPoolExecutor

EORM+JEPA retrain MUST use a `ThreadPoolExecutor(max_workers=2)` with EORM on
`cuda:0` and JEPA on `cuda:1` when both GPUs are available.  Sequential GPU
training for this class is retired as of milestone 2026.04.57.  The validated
speedup from Exp 685 (2.0175x) is the baseline; any new parallel implementation
MUST achieve >= 1.5x speedup vs sequential.

**Acceptance criteria:**
- `DualGPURetrain.retrain_parallel()` submits both tasks concurrently to a `ThreadPoolExecutor`.
- When only 1 GPU is available, the implementation falls back to sequential execution without error.
- Speedup measurement >= 1.5x on a 2-GPU host.

**Spec traces:** REQ-INFRA-049 (Exp 685 validated 2.0175x, 11 milestones idle GPU 1)

### SCENARIO-INFRA-057: Exp 527 Appears in Exclusion Manifest After Exp 740

**Given** Exp 527 has appeared in the slowest-5 for three consecutive milestones
**When** Exp 740 runs and adds Exp 527 to the exclusion manifest
**Then** `ExclusionManifest.is_excluded(527)` returns True and the entry contains
  `governance_rule: "3-consecutive-mandatory"` and `retired_in_milestone: "2026.04.57"`

**Spec traces:** REQ-INFRA-048

### SCENARIO-INFRA-058: DualGPURetrain Falls Back to Sequential on Single GPU

**Given** only 1 CUDA GPU is available
**When** `DualGPURetrain.retrain_parallel(eorm_model, jepa_model, data)` is called
**Then** both models train sequentially on `cuda:0` without raising any exception,
  and the result dict contains `fallback_reason: "single_gpu"`.

**Spec traces:** REQ-INFRA-049

### REQ-INFRA-050: EORM+JEPA Joint Retrain MUST Use DualGPU When 2 GPUs Available

EORM+JEPA joint retrain calls MUST use `DualGPURetrain.retrain_parallel()` via
`ThreadPoolExecutor` when 2 or more CUDA GPUs are detected.  Sequential single-GPU
retrain of the combined EORM+JEPA pair is deprecated for all Exp 383-class runs.

**Rationale:** Exp 383 appeared in the slowest-5 for 11 consecutive milestones.  Exp 685
validated 2.0175x speedup with GPU 1 idle the entire time.  Exp 746 cements this as a
permanent infrastructure default: `retrain_parallel()` replaces any call site that ran
EORM and JEPA sequentially on a host with >= 2 GPUs.

**Spec traces:** REQ-INFRA-050 (Exp 685 validated 2.0175x; Exp 746 production rollout)

### SCENARIO-INFRA-059: DualGPU EORM+JEPA Retrain Achieves >= 1.8x Speedup

**Given** two CUDA GPUs are available (cuda:0, cuda:1)
**When** `DualGPURetrain.retrain_parallel(eorm_fn, jepa_fn)` runs on FoVer v2 data
**Then** the measured `speedup = wall_time_sequential / wall_time_parallel >= 1.8`
  and both `eorm_loss_after` and `jepa_loss_after` are finite positive floats.

**Spec traces:** REQ-INFRA-050

### REQ-INFRA-051: Manifest Patch MUST Be Applied at Dispatch Site

The guard clause calling `validate_manifest_at_dequeue(task_id)` MUST be present in
`scripts/research_conductor.py` inside `research_step()`, immediately after the three
`logger.info("RESEARCH STEP: ...")` lines and before the `if dry_run:` check.
Enforcement via code change at the dispatch site, not retro text or manifest-only update.

**Rationale:** Four consecutive milestones (.54-.57) closed with the patch unnapplied,
wasting 1,264 minutes (21.1 hours) cumulative.  String IDs like "jepa_v15_cascade" bypass
`_task_is_excluded`'s integer regex; only a dispatch-site guard closes this gap.

**Acceptance criteria:**
- `grep validate_manifest_at_dequeue scripts/research_conductor.py` returns at least one match.
- The match appears inside the `research_step()` function body.
- The patch from `results/manifest_fix_patch.txt` is fully applied (no diff).

**Spec traces:** REQ-INFRA-051 (closes 4-milestone enforcement gap, Exp 754)

### REQ-INFRA-052: Pre-flight v10 MUST Confirm Patch Application

The pre-flight v10 artifact (`results/experiment_754_preflight_v10.json`) MUST include a
`patch_applied` boolean field set by searching `scripts/research_conductor.py` for the
guard clause pattern `validate_manifest_at_dequeue`.  Only a code-level search counts;
inspecting the patch file alone is insufficient.

**Acceptance criteria:**
- `artifact["patch_applied"] == True` when guard clause is present in the file.
- `artifact["patch_applied"] == False` when guard clause is absent.
- `honest_verdict` is one of: "preflight_v10_patch_applied_gpu_clean",
  "preflight_v10_patch_applied_gpu_dirty", "preflight_v10_patch_failed",
  "preflight_v10_exp527_leak".

**Spec traces:** REQ-INFRA-052 (Exp 754 pre-flight v10)

### SCENARIO-INFRA-060: Dispatch Guard Blocks Excluded Task at Dequeue

**Given** `conductor_exclusion_manifest.json` lists experiment_id=527
**When** `research_step()` is called with task `{"id": "exp527-legacy", ...}`
**Then** `validate_manifest_at_dequeue("exp527-legacy")` returns False, the task is
  skipped without spawning an agent, and `research_step()` returns True.

**Spec traces:** REQ-INFRA-051

### SCENARIO-INFRA-061: Pre-flight v10 Records patch_applied=True After Patch Application

**Given** `scripts/research_conductor.py` contains the guard clause calling
  `validate_manifest_at_dequeue`
**When** the pre-flight v10 check reads the file and searches for the guard clause
**Then** `artifact["patch_applied"]` is True and `honest_verdict` is
  "preflight_v10_patch_applied_gpu_clean" (assuming GPUs are clean and 527 is excluded).

**Spec traces:** REQ-INFRA-052

### REQ-INFRA-053: Exclusion Manifest Check MUST Be Applied at ALL Dequeue Sites

Every site in `scripts/research_conductor.py` where a task/experiment is fetched from any
source (YAML, history, queue) MUST call `_task_is_excluded(task)` before dispatching an agent.
No dequeue may bypass the manifest check.  A single unguarded dequeue is sufficient to re-admit
a retired experiment.

**Rationale:** Exp 425 appeared for the 22nd consecutive milestone (.37 through .58, 1,672 min
cumulative = 27.9 hours of zero-value compute) because the Exp 754 manifest patch covered only
the conductor's managed cycle.  Other dequeue sites existed without the guard.  Full coverage
means EVERY site is guarded.

**Acceptance criteria:**
- `coverage_pct = guarded_sites / total_dequeue_sites * 100 == 100.0`
- `full_coverage == True` in the pre-flight v11 artifact.
- `honest_verdict == "full_manifest_coverage_achieved"` when all sites are guarded and
  `n_excluded_total >= 27`.

**Spec traces:** REQ-INFRA-053 (Exp 767 pre-flight v11)

### REQ-INFRA-054: Exps 425, 491, 603, 627 MUST Be in conductor_exclusion_manifest.json

`scripts/conductor_exclusion_manifest.json` MUST contain entries for experiment IDs 425, 491,
603, and 627 with `completed_milestone` set to at least "2026.04.58" (the milestone where they
last appeared in the slowest-5 full-milestone timing).

**Rationale:**
- Exp 425: 22nd consecutive slowest-5 appearance, 1,672 min cumulative overhead.
- Exp 491: JEPA curriculum diagnostic, 12th appearance, unbounded training loop.
- Exp 603: CoACEExtractorV4 repeated carry-over from unguarded historical queue source.
- Exp 627: interwhen mid-generation monitor, repeated carry-over from unguarded source.

**Acceptance criteria:**
- `manifest["n_excluded_total"] >= 27` (23 before this patch + 4 new entries).
- `new_exclusions_added` list in pre-flight v11 artifact contains all four IDs.

**Spec traces:** REQ-INFRA-054 (Exp 767 pre-flight v11)

### SCENARIO-INFRA-062: Full Dequeue Coverage Confirmed at 100% After v11 Patch

**Given** all dequeue sites in `scripts/research_conductor.py` have been audited
**When** the pre-flight v11 script counts guarded vs unguarded dequeue sites
**Then** `coverage_pct == 100.0`, `full_coverage == True`, and `guarded_sites_after_patch` equals
  `total_dequeue_sites`.

**Spec traces:** REQ-INFRA-053

### SCENARIO-INFRA-063: New Exclusions 425/491/603/627 Present in Manifest After v11

**Given** `conductor_exclusion_manifest.json` previously had 23 entries
**When** the pre-flight v11 script adds Exps 425, 491, 603, 627 for milestone "2026.04.58"
**Then** the manifest has at least 27 entries and all four IDs are excluded when queried
  via `ExclusionManifest.is_excluded()`.

**Spec traces:** REQ-INFRA-054

### REQ-INFRA-055: kill_gpu_zombies() MUST Be Called Before Model Load in setup_gpu()

`kill_gpu_zombies()` from `carnot.pipeline.gpu_zombie_killer` MUST be called inside
`ExperimentTemplate.setup_gpu()` before any model load attempt when `CARNOT_FORCE_LIVE=1`.
The function MUST use `subprocess` to run `nvidia-smi --query-compute-apps=pid
--format=csv,noheader,nounits` to enumerate PIDs holding GPU memory, then send `SIGKILL`
to each PID that is NOT the current process and NOT in the caller-supplied exclude list.
The result MUST be recorded in setup_gpu()'s return dict under `zombie_kill_result`.

**Rationale:** RETRO-028 (Gemma4 14.89 GiB allocation fails with 15 GiB already in use)
and RETRO-SOTA-GGUF-TIMEOUT (Exp 769 timeout) share a common root cause: zombie processes
holding GPU VRAM before model load.  Fixing only at setup() (session start) is insufficient
because mid-session failures can accumulate zombies between experiments.

**Acceptance criteria:**
- `setup_gpu()` return dict contains `zombie_kill_result` key.
- When CARNOT_FORCE_LIVE=1 and zombie PIDs exist, they are sent SIGKILL.
- The calling process PID is never in the kill list.

**Spec traces:** REQ-INFRA-055

### REQ-INFRA-056: kill_gpu_zombies() MUST Be a No-Op When No Zombies Exist

When `nvidia-smi` reports no compute processes on the target GPU, `kill_gpu_zombies()`
MUST return a `GPUZombieResult` with `pids_killed=[]`, `vram_freed_mb=0.0`, and
`honest_verdict="no_zombies_found"`.  The function MUST NOT kill the calling process
itself under any circumstances.  When `nvidia-smi` is unavailable, `honest_verdict`
MUST be `"nvidia_smi_unavailable"`.

**Acceptance criteria:**
- Empty nvidia-smi output → `honest_verdict="no_zombies_found"`, `pids_killed=[]`.
- Calling PID is always in `exclude_pids`; never sent SIGKILL.
- Missing nvidia-smi → `honest_verdict="nvidia_smi_unavailable"`.

**Spec traces:** REQ-INFRA-056

### SCENARIO-INFRA-064: kill_gpu_zombies() No-Op on Clean GPU

**Given** `nvidia-smi --query-compute-apps=pid` returns no output (empty GPU)
**When** `kill_gpu_zombies(gpu_index=0)` is called
**Then** `honest_verdict="no_zombies_found"`, `pids_killed=[]`, `vram_freed_mb=0.0`

**Spec traces:** REQ-INFRA-056

### SCENARIO-INFRA-065: kill_gpu_zombies() Excludes Calling Process

**Given** the calling process PID appears in `nvidia-smi --query-compute-apps=pid` output
**When** `kill_gpu_zombies(gpu_index=0)` is called without explicit exclude_pids
**Then** `os.getpid()` is never sent SIGKILL

**Spec traces:** REQ-INFRA-055, REQ-INFRA-056

### REQ-INFRA-079: GPU Zombie Sweeps MUST Spare Inference Servers and MUST Gate Utilization Per-GPU

**Statement:** Every GPU zombie sweep in this project MUST (1) skip any process whose
cmdline identifies it as an inference server (`llama-server`,
`vllm.entrypoints.openai.api_server`, or `vllm serve`), and (2) when a sweep gates its
kill decision on GPU utilization or idleness, it MUST judge the candidate on the GPU(s)
that process actually runs on — never an aggregate (minimum, mean) across all GPUs, and,
for a process holding memory on several GPUs, on the MAXIMUM utilization across its own
GPUs (busy anywhere means not a zombie). The known sweep set as of 2026-08-23 — kept
in sync by the same-day adversarial review, which found half of it missing from the
first enumeration:

1. `ExperimentTemplate.kill_gpu_zombies()` in `scripts/experiment_template.py` — both
   the pynvml path and the nvidia-smi fallback (SIGTERM; runs in every `setup()`).
2. `kill_gpu_zombies()` in `python/carnot/pipeline/gpu_zombie_killer.py` (SIGKILL).
3. `ExpandedGPUReaper.reap()` in `python/carnot/pipeline/expanded_gpu_reaper.py`
   (SIGKILL; the conductor's `preflight_gpu_reap`).
4. `detect_zombies()`/`kill_zombies()` in `scripts/gpu_monitor.py` (SIGTERM; runs
   LIVE with `dry_run=False` in every conductor task pre-check — its cumulative
   cpu_time/wall_time idle proxy matches a mostly-idle server by construction).
5. `evict_gpu_vram()`'s step-3 residual pkill sweep in
   `python/carnot/pipeline/gemma_isolation.py` (SIGKILL — must not defeat step 2's
   exemption ten lines below it).
6. `evict_vram_with_loop()`'s retry loop in
   `python/carnot/pipeline/vram_loop_eviction.py` (SIGKILL — same defeat shape).

A NEW sweep (any code that discovers GPU-holding processes and signals them) joins
this list with the same two protections, or names itself in an acknowledged-exempt
note with a written reason.

**Why this matters (origin, 2026-08-23):** the standing unsolved "llama-server reaper"
(ops/known-issues.md, five entries dated 2026-08-09) was this code. The nvidia-smi fallback
took the MINIMUM utilization across all GPUs as its idle signal. An idle GPU 0 dragged the
gate to 0%, and the sweep then SIGTERMed a llama-server on GPU 1 that was decoding at
34 tok/s with GPU 1 at 97% utilization (live-reproduced 2026-08-23; the sweep logged
`gpu_util=0.0%` for that process). The sweep runs unconditionally in
`ExperimentTemplate.setup()`, including at pytest import time, so every conductor
experiment start re-fired it. This voided every scored-path live measurement in the
2026-08-22/23 window (six of six A/B rows `llm_on_row_valid: false`). A serving process
idles between requests by design, so "big VRAM + idle GPU" is its normal healthy state —
the zombie heuristic is category-invalid for servers. A genuinely dead or orphaned server
is `run_stop_authority.py`'s job, which verifies ownership before acting; a blunt sweep is
the wrong tool. The same reasoning produced the training-process exemption on 2026-06-13
after the same sweep repeatedly killed an outer-loop training run; this requirement
generalizes that fix to the server class that was left out.

**Acceptance criteria:**
- A process whose cmdline contains `llama-server` or `vllm.entrypoints.openai.api_server`
  is never signalled by any zombie sweep, and each skip is logged with the reason.
- The nvidia-smi fallback joins each compute process to its own GPU via `gpu_uuid` and
  gates on that GPU's utilization. When per-GPU attribution is unavailable, the sweep
  MUST skip the kill (fail toward not killing), never fall back to an aggregate gate.
- The training-process exemption (REQ-INFRA-074 lineage, `_TRAINING_ENTRYPOINT_MARKERS`)
  is unchanged.

**Spec traces:** REQ-INFRA-079

### SCENARIO-INFRA-6560: Busy GPU-1 Server Survives a Sweep While GPU 0 Idles

**Given** nvidia-smi reports a llama-server process holding 8 GB on GPU 1 at 97% utilization
**And** GPU 0 reports 0% utilization
**When** `ExperimentTemplate.kill_gpu_zombies()` runs via the nvidia-smi fallback
**Then** the llama-server PID receives no signal
**And** the result's `killed_pids` is empty

**Spec traces:** REQ-INFRA-079

### SCENARIO-INFRA-6561: Idle Inference Server Is Exempt by Name

**Given** a llama-server process holds more than 1 GB VRAM on a GPU at 0% utilization
**When** any project zombie sweep runs (`experiment_template` either path, or
`gpu_zombie_killer.kill_gpu_zombies`)
**Then** the server PID is skipped with a logged reason
**And** a non-server `python3` process meeting the same thresholds on the same GPU is
still killed

**Spec traces:** REQ-INFRA-079

### SCENARIO-INFRA-6562: Missing Per-GPU Attribution Fails Toward Not Killing

**Given** nvidia-smi output lacks a parseable `gpu_uuid` for a candidate process
**When** the nvidia-smi fallback evaluates that candidate
**Then** the candidate is skipped, not killed under an aggregate utilization gate

**Spec traces:** REQ-INFRA-079

### REQ-INFRA-057: NPU Unblock Automated Install Strategy Limit

**Statement:** NPU unblock experiments MUST attempt Option A (GitHub Releases wheel) first,
MUST attempt Option B (Ryzen AI SDK installer) if and only if Option A fails, and MUST NOT
attempt more than 2 automated install strategies per experiment run.

**Why this matters:**
    Eight consecutive milestones (Exps 292, 303, 314, 335, 435, 714) were blocked by the
    same root cause — mlir-aie not on PyPI and VitisAI requiring a compiled-in onnxruntime.
    Without a hard cap on strategy attempts, experiments can spiral into open-ended install
    loops that consume 45+ minutes without producing a binary verdict.  The two-strategy
    cap forces a clean "exhausted" verdict so the conductor can escalate rather than retry
    forever.  Option A (GitHub Releases wheel) is tried first because it requires no auth
    and targets the exact missing package; Option B (Ryzen AI SDK installer) requires AMD
    account credentials and is therefore a fallback.

**Spec traces:** Exp 790, RETRO-NPU-v9

### SCENARIO-INFRA-066: NPU Unblock Option B Tried Only After Option A Failure

**Given** Option A (GitHub Releases wheel) install fails (option_a_success=False)
**When** the NPU unblock script proceeds to Option B
**Then** option_b_attempted=True in the result artifact

**Spec traces:** REQ-INFRA-057

### REQ-HW-010: Ising Sampler v4 HLS C++ Kernel

The Ising sampler v4 MUST be expressed in Vitis HLS C++ with loop-pipelining
pragmas embedded as comments (so the same file compiles as plain C++ for CPU
validation).  The top-level function `update_spin_kernel` MUST include:

- Sequential Gibbs updates with xorshift32 RNG (HLS-compatible, no stdlib rand)
- EMA inertia field `h_ema[i]` per spin, blending instantaneous and historical fields
- All HLS PIPELINE / UNROLL / ARRAY_PARTITION pragmas as `// #pragma HLS ...` comments
- A CPU-compilable `main()` guarded by `#ifndef __SYNTHESIS__` for validation

The same `ising_sampler_hls.cpp` MUST:
1. Compile under `g++ -O2 -std=c++17` without errors.
2. Produce a final energy within 20% of the ground-state energy for a 4-spin test case.
3. Be synthesisable by Vitis HLS 2024.2 when `synth_ising_hls.tcl` is executed.

**Rationale:** KV260 bitfile synthesis is blocked locally due to missing Vivado
installation.  HLS C++ can be synthesised on any cloud instance with AMD Vitis 2024.2
without requiring a full Vivado install.  The dual-compile approach (same C++ for CPU
and FPGA) allows local validation before remote synthesis.

**Acceptance criteria:**
- `hardware/kv260/ising_sampler_hls.cpp` exists and compiles with g++.
- Compiled binary returns exit code 0 (energy within tolerance).
- `hardware/kv260/synth_ising_hls.tcl` references the correct KV260 part number.
- `results/experiment_750_vitis_hls_ising_v4.json` records `cpp_compiles: true`.

### SCENARIO-HW-010: HLS Kernel CPU Validation

**Given** `hardware/kv260/ising_sampler_hls.cpp` is compiled with
  `g++ -O2 -std=c++17 hardware/kv260/ising_sampler_hls.cpp -o /tmp/ising_hls_test`
**When** the resulting binary is executed
**Then** it prints "PASS" and exits with code 0, meaning the final energy of a
  4-spin antiferromagnetic chain is within 20% of the -3.0 ground-state energy.

**Spec traces:** REQ-HW-010

### REQ-SAMPLE-017: DWaveNealBackend Protocol Implementation

``DWaveNealBackend`` MUST implement the ``SamplerBackend`` protocol by providing
a ``sample()`` method.  It MUST convert an ``IsingEBM`` (``IsingModel``) coupling
matrix and bias vector to a ``dimod.BinaryQuadraticModel`` via a ``to_bqm()``
method before submitting to ``neal.SimulatedAnnealingSampler``.

**Acceptance criteria:**
- ``DWaveNealBackend().available`` is True when dwave-ocean-sdk is installed.
- ``to_bqm(ising_ebm)`` returns a BQM with ``num_variables == ising_ebm.config.input_dim``.
- Quadratic interactions in the BQM match non-zero entries of ``ising_ebm.coupling``.
- Linear biases in the BQM match ``ising_ebm.bias``.

**Spec:** REQ-SAMPLE-017

---

### REQ-SAMPLE-018: DWaveNealBackend Reports Energy and Wall Time

``DWaveNealBackend.sample()`` MUST return a ``SampleResult`` with:
- ``spins``: boolean array of shape ``(n_spins,)`` (the lowest-energy configuration
  found across all ``num_reads`` SA runs).
- ``energy``: float energy of ``spins`` under the IsingEBM Hamiltonian, computed in
  the ``{0,1}`` convention (compatible with ``IsingModel.energy``).
- ``wall_time_s``: float wall-clock seconds for the full call.

**Acceptance criteria:**
- ``result.energy`` is a float.
- ``result.wall_time_s > 0``.
- ``result.spins.shape == (n_spins,)`` and ``result.spins.dtype == bool``.

**Spec:** REQ-SAMPLE-018

---

### SCENARIO-SAMPLE-030: Neal vs Gibbs Energy Comparison on Random Problems

**Given** 20 synthetic IsingModel instances with n=50 spins and coupling sparsity=0.3
**When** both DWaveNealBackend and CpuBackend (Gibbs) are run on each instance
**Then** ``energy_improvement_pct`` is computed as
  ``(mean_energy_gibbs - mean_energy_neal) / |mean_energy_gibbs| * 100``
  and ``honest_verdict`` is one of
  ``{"neal_better_energy", "neal_comparable_energy", "neal_worse_energy"}``.

**Spec traces:** REQ-SAMPLE-017, REQ-SAMPLE-018

### SCENARIO-SAMPLE-031: DWaveNealBackend Blocked on Dependency

**Given** dwave-ocean-sdk is not installed (``neal`` import fails)
**When** ``DWaveNealBackend().available`` is False
**Then** ``sample()`` returns a ``SampleResult`` with ``energy == float('inf')``
  and the experiment artifact records ``honest_verdict == "blocked_on_dependency"``.

**Spec traces:** REQ-SAMPLE-017

---

## REQ-PUBLISH-001: HuggingFace Model Card Requirements

Every model published to the Carnot-EBM HuggingFace organisation MUST include a model card with:
- Architecture description (what the model does and WHY the design choices were made)
- Training data citation (dataset name, size, and collection methodology)
- Evaluation metrics (AUC, AUROC, FP rate, latency as appropriate)
- Usage example showing `pip install carnot` and inference code
- Apache 2.0 license declaration
- Explicit labeling of any simulated or synthetic evaluation results

This requirement exists because novel model artifacts without model cards are invisible to the
community. A discoverable, well-documented model card is the primary mechanism for directing
users to `pip install carnot` and establishing the Carnot-EBM HuggingFace presence.

Where REQ-SAFE-011 (teacher-duration invariant) applies, the model card MUST cite it.

**Spec traces:** SCENARIO-PUBLISH-001

---

### SCENARIO-PUBLISH-001: HuggingFace Artifact Preparation

**Given** two production-quality models exist (StepLevelJEPAProbe from Exp 738,
  KAN Tier 0b from Exp 735) with validated weights and evaluation metrics
**When** the operator runs `models/hf_upload_commands.sh` after `huggingface-cli login`
**Then** both models are published to HuggingFace with complete model cards,
  safetensors weights, and config JSON — all satisfying REQ-PUBLISH-001.

**Acceptance criteria:**
- Model cards have no emojis (professional presentation standard).
- Config JSON contains all required fields (model_type, metrics, architecture, training_data).
- Upload script references valid local file paths.
- `honest_verdict` is one of `{"hf_artifacts_ready", "hf_artifacts_partial", "hf_jepa_weights_missing"}`.

**Spec traces:** REQ-PUBLISH-001

### REQ-LOADER-010: Gemma4 Models MUST Use GemmaTransformersLoader

All model loading for `google/gemma-4-*` HuggingFace model IDs MUST use
`GemmaTransformersLoader`.  The llama.cpp backend MUST NOT be used for any
`google/gemma-4-*` model until the tokenizer bug (llama.cpp issue #21516) is
confirmed fixed upstream.  This requirement covers non-GGUF (FP16) model loading;
GGUF-quantized variants loaded via `Gemma4QuantizedLoader` are excluded because
the Q4_K_M GGUF format bypasses the problematic tokenizer path.

**Rationale:** RETRO-028: llama.cpp's Gemma4 tokenizer emits infinite `<unused8>`
tokens (token_id=14), causing 0% accuracy on all benchmarks.  This blocked Gemma4
experiments in milestones .55, .56, .57, and .58.

**Acceptance criteria:**
- All call sites loading `google/gemma-4-E4B-it` (non-GGUF) use `GemmaTransformersLoader`.
- Exp 768 loader_test_passed=True: `GemmaTransformersLoader.generate("Hello", max_new_tokens=5)` returns text with no `<unused>` tokens.
- `GemmaTransformersLoader.is_valid_output(result)` returns True for the 5-token smoke test.

**Implementation Status:** Planned (Exp 768)

### SCENARIO-LOADER-010: GemmaTransformersLoader Smoke Test Passes

**Given** `GemmaTransformersLoader("google/gemma-4-E4B-it", device="cuda:0")`
**When** `.load()` then `.generate("Hello", max_new_tokens=5)` is called
**Then** the returned string contains no `<unused8>` / `<unusedN>` tokens and `is_valid_output()` returns True

**Spec traces:** REQ-LOADER-010
**Implementation Status:** Planned (Exp 768)

### REQ-LOADER-011: kill_gpu_zombies() MUST Be Called Before Any Gemma4 Load Attempt

A Gemma4 model load attempt MUST call `kill_gpu_zombies(gpu_index=0)` before any
`GemmaTransformersLoader.load()` call.  The result's `vram_after_mb` MUST be recorded
as `free_vram_mb_after_kill` in the artifact.  If `free_vram_mb_after_kill < 12000`,
the experiment MUST NOT attempt the load and MUST write `honest_verdict="blocked_insufficient_vram"`.

**Rationale:** RETRO-028 Exp 768 failed with CUDA OOM: 14.89 GiB allocation on a 24 GiB
card with ~15 GiB occupied by zombie processes.  With the GPU cleared, 24 GiB - 14.89 GiB
= 9.11 GiB free overhead — sufficient for the loader.

**Acceptance criteria:**
- `kill_gpu_zombies(gpu_index=0)` is called before any `GemmaTransformersLoader.load()`.
- `free_vram_mb_after_kill` is recorded in every artifact, regardless of outcome.
- If `free_vram_mb_after_kill < 12000`, artifact contains `honest_verdict="blocked_insufficient_vram"`.
- If load succeeds, `loader_test_passed=True` is recorded.

**Implementation Status:** Planned (Exp 786)

### SCENARIO-LOADER-011: Insufficient VRAM After Zombie Kill Blocks Load

**Given** `kill_gpu_zombies(gpu_index=0)` returns a result where `vram_after_mb` corresponds
to less than 12000 MB free (i.e., total VRAM minus vram_after_mb < 12000 MB)
**When** the experiment checks the VRAM threshold
**Then** the artifact records `honest_verdict="blocked_insufficient_vram"` and `GemmaTransformersLoader.load()` is NOT called

**Spec traces:** REQ-LOADER-011
**Implementation Status:** Planned (Exp 786)

---

### REQ-PROBE-020: SemanticEnergyProbe Logit-Space Energy Computation

`SemanticEnergyProbe` MUST compute energy as `E = -sum_i log p(t_i)` where `p(t_i)` is
the token probability from logits (TF-IDF as proxy when logits unavailable).  It MUST
group responses into semantic clusters via TF-IDF cosine similarity with threshold=0.9.
`SemanticCluster.compute_cluster_energy(responses)` MUST return the mean of per-response
energies computed as `-sum(log(tfidf_score(token) + eps))`.

**Rationale:** arXiv 2508.14496 ("Semantic Energy", August 2025) shows logit-space energy
outperforms entropy-based detection by retaining intensity information lost during softmax.
The TF-IDF proxy is used for offline/text-only evaluation; real logits provide the full signal.

**Acceptance criteria:**
- `SemanticEnergyProbe().score(text)` returns a non-negative float for any string.
- `SemanticCluster().compute_cluster_energy(responses)` returns the mean of negative log-score sums.
- `SemanticCluster().group_by_semantics(responses)` partitions all responses into clusters.
- Every test in `tests/python/test_experiment_772_semantic_energy_probe.py` passes.

### REQ-PROBE-010: DRIFTProbe v3 Depth-Recurrent Attention Pooling

`DRIFTProbeV3` MUST accept a list of per-layer hidden-state activations
(shape [seq_len, hidden_dim] per layer), compute per-layer cosine drift scalars,
and pass the resulting N-dimensional layer-drift profile through a learned 2-layer MLP
(hidden_dim=32) that outputs a single scalar P(incorrect).  The probe MUST expose
`fit(X_layers, y_labels)` and `predict_proba(X_layers)` with scikit-learn-compatible
semantics, plus `layer_attention_weights()` returning normalised per-layer importances.

**Acceptance criteria:**
- `fit()` trains MLP weights on labeled (hidden_states, is_incorrect) pairs.
- `predict_proba()` raises RuntimeError if called before `fit()`.
- `layer_attention_weights()` returns a non-negative array summing to 1.
- Probe AUROC on synthetic data with heterogeneous per-layer noise MUST exceed 0.50.

**Motivation:** arXiv 2604.17121 proves transformer state is non-local; arXiv 2604.13386
validates that learned per-layer weights improve AUROC 3-8% vs uniform weights.

**Implementation Status:** Implemented — Exp 947 (drift_probe_v3.py, DRIFTProbeV3)

### SCENARIO-PROBE-015: Attention Pooling Learns Drift-Relevant Layer Weights

**Given** 160 synthetic (correct, incorrect) pairs where incorrect responses have large
drift injected at alternating layers
**When** `DRIFTProbeV3().fit(X_train, y_train)` is called and
`predict_proba(X_test)` is evaluated on 40 held-out pairs
**Then** `roc_auc_score(y_test, proba)` > 0.50 (above-random discrimination)
AND `layer_attention_weights()` shows non-uniform weights (some layers weighted higher)

**Spec traces:** REQ-PROBE-010

### REQ-PROBE-021: Tier 0g Advisory Flag

`SemanticEnergyProbe` MUST report `semantic_energy_score` and `is_high_energy=True` when
the score exceeds `energy_threshold`.  The probe is ADVISORY — it does NOT short-circuit
the pipeline; Tiers 1-3 still run.  Wiring as Tier 0g requires `auc >= NUP v4 AUC - 0.05`.

**Acceptance criteria:**
- `is_high_energy(text)` returns True when `score(text) > energy_threshold`.
- `is_high_energy(text)` returns False when `score(text) <= energy_threshold`.
- Exp 772 records `tier0g_deployed` (bool) and `honest_verdict` in its artifact.

### SCENARIO-PROBE-030: SemanticCluster Groups High-Similarity Responses

**Given** two responses with cosine similarity >= 0.9
**When** `SemanticCluster(threshold=0.9).group_by_semantics([r1, r2])` is called
**Then** both responses appear in the same cluster (one cluster total)

**Spec traces:** REQ-PROBE-020

### SCENARIO-PROBE-031: is_high_energy Flags Above-Threshold Response

**Given** a `SemanticEnergyProbe(energy_threshold=0.0)`
**When** `is_high_energy("some non-empty response")` is called
**Then** the method returns True (any non-empty text has energy > 0.0)

**Spec traces:** REQ-PROBE-021

### REQ-PUBLISH-010: HuggingFace Upload Requires Authentication

The HuggingFace upload MUST be executed via `huggingface-cli upload`.  Upload MUST NOT
be attempted if `huggingface-cli whoami` returns a non-zero exit code (HF_TOKEN absent or
login session expired).  The honest_verdict MUST be `blocked_hf_not_authenticated` when
the authentication check fails.

### REQ-PUBLISH-011: All Existing Carnot-EBM Model READMEs Must Include pip install carnot

All 16 existing Carnot-EBM model READMEs MUST include a "## Production Use" section
pointing users at `pip install carnot` and clarifying that the per-token activation EBMs
are Phase 1 research artifacts (confidence detection, not correctness).  The update MUST
be idempotent: re-running when the section already exists MUST succeed without re-uploading.

### SCENARIO-PUBLISH-010: Blocked When HF_TOKEN Not Set

**Given** `huggingface-cli whoami` returns exit code 1
**When** `run_experiment(tmpl)` is called
**Then** `honest_verdict == "blocked_hf_not_authenticated"` and no upload is attempted

**Spec traces:** REQ-PUBLISH-010

### SCENARIO-PUBLISH-011: README Updated With pip install carnot

**Given** an existing Carnot-EBM model README without a "## Production Use" section
**When** `update_readme_with_production_section(repo_id)` is called
**Then** the README gains a section containing "pip install carnot" and the GitHub URL

**Spec traces:** REQ-PUBLISH-011

### REQ-PUBLISH-005: HuggingFace Authentication Token MUST Be Stored via SOPS Encryption

**Statement:** The HuggingFace authentication token (HF_TOKEN) MUST be stored at rest using
SOPS encryption (age or PGP key).  Plaintext HF_TOKEN values MUST NOT appear in any
committed file.  The token MUST be decrypted at runtime via
`sops -d secrets/hf_token.enc.yaml` and injected into the conductor environment with
`eval $(sops -d ... | grep HF_TOKEN)`.  The decrypted value MUST reach the process only
as an environment variable — never written back to disk.

Corrected 2026-07-31 (security audit): this requirement previously named
`secrets/hf_token.yaml`.  No such file exists — the real artifact is
`secrets/hf_token.enc.yaml` — and that plaintext path was, at the time, not covered by
`.gitignore`, so the spec named a committable plaintext location.  Both the ignore rules
and the path are fixed; see `docs/sops-hf-token-setup.md`.

**Why this matters:**
    Exp 777 (.59) revealed that HF_TOKEN was absent from the conductor environment,
    blocking all model publishing.  The root cause was no standardised secret-injection
    workflow.  SOPS with age keys provides at-rest encryption (keys never committed),
    per-repo access control via .sops.yaml, and a single decryption command that works
    in both interactive and automated (conductor) sessions without requiring a secrets
    manager service.

**Spec traces:** CLAUDE.md security requirements, RETRO-HF-AUTH, Exp 803

### REQ-PUBLISH-006: models/hf_upload_commands.sh MUST Provide Authenticated Push Commands

**Statement:** `models/hf_upload_commands.sh` MUST contain `huggingface-cli upload` commands
for all three model tiers: Ising (carnot-ising-sampler-v1), KAN (carnot-kan-energy-tier),
and EORM (carnot-eorm-55m).  The script MUST source HF_TOKEN from SOPS before calling
huggingface-cli login.  The script MUST be executable and idempotent.

**Why this matters:**
    Without a single authoritative upload script, each publish attempt re-discovers the
    correct repo IDs and file lists from scratch.  A versioned script with SOPS wiring
    ensures the conductor can re-run publishes deterministically without manual token
    injection each time.

**Spec traces:** REQ-PUBLISH-005, Exp 803

### SCENARIO-PUBLISH-009: HF_TOKEN Present; huggingface-cli Login Succeeds; README Updated

**Given** HF_TOKEN is present in the environment (from SOPS decryption or env var)
**And** `huggingface-cli whoami` returns exit code 0
**When** `run_experiment(tmpl)` is called
**Then** at least one model README is updated via `huggingface-cli upload`
**And** `honest_verdict == "hf_models_published"`

**Spec traces:** REQ-PUBLISH-005, REQ-PUBLISH-006

### REQ-DCCD-1606: DCCD Multi-Hop Logical Evaluation

**Statement:** The DCCD multi-hop evaluator MUST accept a list of multi-hop logical
reasoning questions, produce a draft response via `DraftConditionedVerifier`, extract
structural constraints, and score each hop for structural consistency using energy-based
constraint signals.  The evaluator MUST work purely on synthetic deterministic data (no
live model calls required) so tests run in CI without GPU.

**Why multi-hop:** Single-hop questions (e.g., "3 + 5 = ?") have shallow constraint
graphs.  Multi-hop logical problems ("If A implies B, and B implies C, is A sufficient
for C?") require chaining constraint checks across N inference steps.  DCCD's
draft-conditioning is more valuable here: the draft structure reveals which hops the
model believes are sequential vs. parallel, allowing the Ising tier to penalize
structurally inconsistent reasoning chains.

**Acceptance criteria:**
- `DCCDMultiHopEvaluator.evaluate(questions)` returns a list of `MultiHopResult`.
- Each result contains `structural_constraints`, `hop_count`, `chain_valid` bool.
- `evaluate_multihop_dataset()` returns aggregate metrics with `accuracy_rate`,
  `mean_hop_count`, `mean_constraint_count`, `dccd_applied` bool.
- Runner function writes artifact to `results/experiment_1606_dccd_multihop.json`.
- `honest_verdict` starts with `complete:` prefix.

**Spec traces:** REQ-DCCD-1606 (Exp 1606)

### SCENARIO-DCCD-1606: Multi-Hop Chain Correctly Detected

**Given** a multi-hop question with 3 logical inference steps
**When** `DCCDMultiHopEvaluator.evaluate([question])` is called
**Then** the result has `hop_count >= 3` and `structural_constraints` is non-empty
**And** `chain_valid` is True for well-formed transitive chains

**Spec traces:** REQ-DCCD-1606

## Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-SAMPLE-017 | Implemented | Exp 751 — dwave_neal_backend.py + to_bqm() |
| REQ-SAMPLE-018 | Implemented | Exp 751 — SampleResult with energy + wall_time_s |
| REQ-HW-010 | Implemented | Exp 750 — ising_sampler_hls.cpp + synth_ising_hls.tcl |
| REQ-INFRA-043 | Implemented | Exp 718 — tier2_jepa.py |
| REQ-INFRA-044 | Implemented | Exp 718 smoke test |
| REQ-INFRA-045 | Implemented | Exp 718 latency measurement |
| REQ-INFRA-046 | Implemented | Exp 727 — cascade_router.py |
| REQ-INFRA-047 | Implemented | Exp 727 fn_delta measurement |
| REQ-INFRA-046b | Implemented | Exp 731 — conductor_manifest_validator.py; patch in results/manifest_fix_patch.txt |
| REQ-INFRA-047b | Implemented | Exp 731 — GPU 1 zombie cleared, vram_after=4 MiB |
| REQ-INFRA-048 | Implemented | Exp 740 — Exp 527 added to exclusion manifest |
| REQ-INFRA-049 | Implemented | Exp 740 — DualGPURetrain in python/carnot/pipeline/dualgpu_retrain.py |
| REQ-PUBLISH-001 | Implemented | Exp 752 — model cards, safetensors exports, hf_upload_commands.sh |
| REQ-INFRA-050 | Implemented | Exp 746 — DualGPU retrain made default; sequential deprecated |
| REQ-INFRA-051 | Implemented | Exp 754 — manifest patch applied to research_conductor.py dispatch site |
| REQ-INFRA-052 | Implemented | Exp 754 — pre-flight v10 confirms patch application via guard clause search |
| REQ-INFRA-053 | Implemented | Exp 767 — pre-flight v11 confirms 100% dequeue-site manifest coverage |
| REQ-INFRA-054 | Implemented | Exp 767 — Exps 425, 491, 603, 627 added to exclusion manifest (.58) |
| REQ-INFRA-055 | Implemented | Exp 780 — kill_gpu_zombies() in gpu_zombie_killer.py; wired into ExperimentTemplate.setup_gpu() |
| REQ-INFRA-056 | Implemented | Exp 780 — kill_gpu_zombies() is a no-op when no GPU zombies exist |
| REQ-INFRA-057 | Implemented | Exp 790 — NPU unblock: Option A (GitHub wheel) first, Option B fallback, max 2 strategies |
| REQ-INFRA-058 | Scaffolding | Exp 793 — manifest full-scope audit; patch spec written to results/experiment_793_manifest_full_scope_audit.json |
| REQ-INFRA-059 | Scaffolding | Exp 793 — WARNING-level logging requirement documented; patch required in pick_next_task |
| REQ-VERIFY-1427 | Implemented | Exp 1427 — repair_rejection_ledger.py + rejection ledger artifact and repair v2 contract |
| REQ-LOADER-010 | Planned | Exp 768 — Gemma4 call site audit + GemmaTransformersLoader enforcement |
| REQ-LOADER-011 | Planned | Exp 786 — kill_gpu_zombies() mandatory before Gemma4 load; VRAM threshold guard |
| REQ-PROBE-020 | Implemented | Exp 772 — SemanticEnergyProbe + SemanticCluster in python/carnot/pipeline/semantic_energy_probe.py |
| REQ-PROBE-021 | Implemented | Exp 772 — is_high_energy advisory flag; tier0g_deployed=False (AUC=0.46, below NUP v4 baseline) |
| REQ-PUBLISH-010 | Implemented | Exp 777 — huggingface-cli upload executed; blocked cleanly when HF_TOKEN absent |
| REQ-PUBLISH-011 | Implemented | Exp 777 — all existing Carnot-EBM model READMEs updated with pip install carnot pointer |
| REQ-PUBLISH-005 | Implemented | Exp 803 — SOPS HF_TOKEN spec in docs/sops-hf-token-setup.md |
| REQ-PUBLISH-006 | Implemented | Exp 803 — models/hf_upload_commands.sh with SOPS wiring for all 3 tiers |

## EBM Calibration Alignment (Exp 789)

### REQ-CALIB-001

**Statement:** EBMCalibrator MUST compute Expected Calibration Error (ECE) from energy-binned
accuracy using 10 equal-frequency bins, and MUST apply isotonic regression to learn an
energy -> P(correct) mapping.

**Why this matters:**
    arXiv 2603.06604 "Know When You're Wrong" shows SFT models have well-calibrated
    confidence but RL-trained models are overconfident by 15-25pp.  Carnot energy is
    currently a discriminative signal (violated/not-violated).  This requirement makes
    energy a calibrated probabilistic signal: low energy = high P(correct).

**Rationale:** ECE (Expected Calibration Error) measures the gap between predicted
confidence and observed accuracy.  Equal-frequency binning ensures each bin has
enough samples to estimate accuracy reliably.  Isotonic regression is the standard
non-parametric post-hoc calibration method (Zadrozny & Elkan 2002).

**Spec traces:** Exp 789, arXiv 2603.06604, arXiv 2602.11364

### REQ-CALIB-002

**Statement:** The calibration curve MUST be saved to results/ebm_calibration_curve.json.
ECE_before and ECE_after MUST be reported in the experiment artifact.

**Why this matters:**
    Without persisting the calibration curve, downstream experiments cannot use the
    fitted isotonic regression to convert raw energy scores to calibrated probabilities.

**Spec traces:** Exp 789

### SCENARIO-CALIB-001: Perfectly Calibrated Energies Yield ECE=0.0

**Given** a set of energies where sigmoid(-energy) exactly equals label accuracy in each bin
**When** compute_ece(energies, labels) is called
**Then** ECE == 0.0

**Spec traces:** REQ-CALIB-001

### SCENARIO-CALIB-002: Isotonic Regression Reduces ECE

**Given** a set of uncalibrated energies with ECE_before > 0
**When** fit_isotonic(energies, labels) is applied and ECE_after is computed
**Then** ECE_after <= ECE_before (isotonic regression never worsens calibration on training data)

**Spec traces:** REQ-CALIB-001, REQ-CALIB-002

### REQ-INFRA-058: ExclusionManifest.check() MUST Be Called at ALL Dequeue Sites

**Statement:** ExclusionManifest.check() (via _task_is_excluded()) MUST be called at EVERY
location in the research conductor where a task_id is selected for execution from any queue
or list data structure. A "dequeue site" is any line where a task moves from a data structure
into the dispatch pipeline — including for-loops over RESEARCH_TASKS, .pop() calls,
.popleft() calls, next(iter(...)), random.choice(), and queue.get() patterns. Placing the
manifest check only in the primary dispatch path (pick_next_task) is insufficient if
secondary code paths bypass pick_next_task and touch RESEARCH_TASKS directly.

**Why this matters:**
    Exp 527 appeared in the slowest-5 for 7+ consecutive milestones after being added to
    the exclusion manifest, because the manifest check in pick_next_task was not adjacent
    to the for-loop that iterates RESEARCH_TASKS. The five-line window heuristic used by
    the audit scanner (Exp 793) confirmed the check is present but logically distant —
    making it easy for future refactors to accidentally bypass. Placing the check immediately
    at the point of dequeue (within 5 lines of the loop/pop/choice statement) is the
    enforcement pattern that prevents recurrence. This requirement documents the FULL-SCOPE
    enforcement goal — every dequeue site must independently guard against excluded tasks.

**Spec traces:** Exp 793, RETRO-MANIFEST-FULL-SCOPE

### REQ-INFRA-059: Excluded Tasks MUST Be Logged at WARNING Level Before Skip

**Statement:** When _task_is_excluded(task) returns True (excluded), the conductor MUST
emit a log.warning() that includes the task title, experiment ID, and the exclusion reason
string before skipping the task. The warning MUST use logger.warning() not logger.info()
so that exclusion events appear in stderr-level log aggregators even when INFO logging is
suppressed. Silently skipping an excluded task without a WARNING-level log makes it
impossible to audit whether the manifest check actually fired for a given run.

**Why this matters:**
    The RETRO-MANIFEST-FULL-SCOPE investigation required manually correlating seven
    milestones of conductor logs to confirm Exp 527 ran despite being manifested.
    If every exclusion emitted a WARNING with the experiment ID, any single conductor log
    would have shown the absence of that WARNING — proving immediately that the guard
    did not fire. This requirement transforms exclusion enforcement from implicit
    (absence of evidence) to explicit (presence of WARNING).

**Spec traces:** Exp 793, RETRO-MANIFEST-FULL-SCOPE, REQ-INFRA-058

### SCENARIO-INFRA-067: Conductor Dequeues Exp 527 From Unmanaged Path; Manifest Guard Fires

**Given** Exp 527 is listed in conductor_exclusion_manifest.json
**And** a dequeue site calls _task_is_excluded() on any task with exp_id=527
**When** _task_is_excluded() is evaluated
**Then** is_excluded=True is returned
**And** the conductor emits logger.warning with "EXCLUDED" in the message
**And** the task is skipped without calling run_agent()

**Spec traces:** REQ-INFRA-058, REQ-INFRA-059

### SCENARIO-INFRA-068: Conductor Dequeues Exp 793 (Not in Manifest); Task Runs Normally

**Given** Exp 793 is NOT listed in conductor_exclusion_manifest.json
**When** the conductor evaluates _task_is_excluded() for a task with exp_id=793
**Then** is_excluded=False is returned
**And** the conductor proceeds to call run_agent() with the task prompt

**Spec traces:** REQ-INFRA-058

### REQ-INFRA-060: MILESTONE_PREREQS.md MUST Exist and Gate Experiment Execution

**Statement:** A MILESTONE_PREREQS.md file MUST exist at the project root listing all
IMMEDIATE-class actions from the prior milestone retro. Each action MUST be marked as
either verified_complete or escalated_retro before any milestone experiment runs.
The file MUST contain a checklist that the conductor or operator verifies manually.
Without this gate, the retro process generates documentation overhead with zero
operational improvement, as observed across three consecutive milestones (.59, .60, .61).

**Why this matters:**
    The .61 retro identified that IMMEDIATE-class improvements were documented but never
    applied, because there was no structural enforcement mechanism. The prereqs gate
    converts the retro from a record-keeping exercise into an actionable pre-flight check.

**Spec traces:** Exp 806, RETRO-.61-PREREQS-GATE

### REQ-INFRA-061: JEPA Retrain Scripts MUST Assert augmentation_ratio > 1.0 at Startup

**Statement:** All JEPA retrain experiment scripts MUST assert augmentation_ratio > 1.0
before any model training begins. Failure raises AssertionError with message:
"CPMI corpus not wired in — check training data loader merges all sources."
This invariant catches the Exp 798→799 disconnect where JEPA trained without CPMI triples,
producing the all-time low ood_auc=0.2444 due to missing data augmentation.

**Why this matters:**
    Exp 799 trained for 5+ minutes before the missing wiring was detected manually.
    An assertion at startup would have caught this in under 1 second and preserved
    the experiment slot for a corrected run. The ood_auc=0.2444 result was an
    implementation error, not an algorithmic failure — this requirement prevents recurrence.

**Spec traces:** Exp 806, RETRO-.61-JEPA-ASSERT

### SCENARIO-INFRA-069: Prereqs Gate Reads MILESTONE_PREREQS.md; All IMMEDIATE Items Verified; Gate Passes

**Given** MILESTONE_PREREQS.md exists at project root
**And** all IMMEDIATE-class items are marked verified_complete or escalated_retro
**When** the prereqs gate check runs
**Then** prereqs_gate_ready is returned
**And** experiment execution proceeds normally

**Spec traces:** REQ-INFRA-060

### SCENARIO-INFRA-070: JEPA Retrain Script Startup; augmentation_ratio=1.0 Detected; AssertionError Raised

**Given** a JEPA retrain script is invoked
**And** augmentation_ratio is computed as 1.0 (no CPMI triples augmenting input pairs)
**When** check_cpmi_wiring() is called at startup
**Then** AssertionError is raised with message "CPMI corpus not wired in — check training data loader merges all sources."
**And** training does NOT begin
**And** the experiment writes a blocked artifact

**Spec traces:** REQ-INFRA-061

### REQ-REPAIR-056: GGUF Loader Import Self-Diagnostic

The GGUF model loader MUST succeed `from llama_cpp import Llama` before any inference
experiment proceeds.  If `ImportError` is raised at load time, the experiment MUST
diagnose the error, log the full error message, and attempt auto-repair via
`pip install --upgrade llama-cpp-python`.  If the import still fails after auto-repair,
the experiment writes a blocked artifact with `honest_verdict="still_blocked_import"`.

**Rationale:** Exp 811 produced `honest_verdict="blocked_model_load_failed"` due to a
Python `ImportError` on `carnot.pipeline.gguf_cache`, blocking every live code repair
experiment since milestone .58.  RETRO-028 resolution shifted the gate from OOM to import
error.  The auto-repair loop prevents the same single-package absence from blocking
multiple consecutive milestones.

**Acceptance criteria:**
- When `from llama_cpp import Llama` raises `ImportError`, the error message is logged
  and `import_repair_attempted` is set to `True` in the artifact.
- When auto-repair via `pip install --upgrade llama-cpp-python` succeeds and the subsequent
  import succeeds, `import_repair_succeeded` is set to `True`.
- When auto-repair fails and the import still raises, the artifact sets
  `honest_verdict="still_blocked_import"` and `import_repair_succeeded=False`.

**Spec traces:** REQ-REPAIR-056 (RETRO-GGUF-CACHE-IMPORT, closes milestone .58 blocker)

### SCENARIO-REPAIR-089: GGUF Loader Import Failure Triggers Auto-Repair

**Given** `from llama_cpp import Llama` raises `ImportError` at experiment startup
**When** the experiment's import diagnostic runs
**Then**
  1. The error message is logged with the full exception text.
  2. `subprocess.run(["pip", "install", "--upgrade", "llama-cpp-python"])` is called.
  3. The import is retried.
  4. If the retry succeeds: the experiment proceeds to GPU setup and inference.
  5. If the retry fails: the experiment writes a blocked artifact with
     `honest_verdict="still_blocked_import"`, `import_repair_attempted=True`,
     `import_repair_succeeded=False`.

**Spec traces:** REQ-REPAIR-056

### REQ-VERIFY-143: MultiAgentArbiter Must Use External Field Energy

**Statement:** MultiAgentArbiter MUST use IsingConstraintInjector.compute_energy_with_external_field
when scoring agent responses, not the legacy IsingEBM.energy() method.

**Rationale:** The legacy method adds a constant diagonal energy shift that is identical for all
spin configurations (because s_i^2 = 1 for ±1 spins), making it impossible to discriminate between
correct and incorrect agent responses.  The external field method changes sign based on spin
orientation: violation spins (s_i=+1) receive +h[i] (energy increases) and correct spins
(s_i=-1) receive -h[i] (energy decreases), producing discriminating per-response scores.

**Spec traces:** REQ-VERIFY-143

---

### REQ-VERIFY-144: MultiAgentArbiter Must Z-Score Normalize Per-Query Energies

**Statement:** MultiAgentArbiter MUST z-score normalize agent energies within each query before
ranking.  For N agent responses to the same query: mu = mean(energies), sigma = std(energies).
If sigma > 1e-6: normalized_energies = (energies - mu) / sigma.  If sigma <= 1e-6: use raw
energies (all equal → random tie-break).  The arbiter selects the agent with the LOWEST
normalized energy.

**Rationale:** Raw energy magnitudes vary significantly across queries (due to different constraint
embeddings and spin configurations).  Without per-query normalization, a query with large energy
variance can dominate consensus detection thresholds calibrated for small-variance queries.
Z-scoring puts all queries on a common scale (mean=0, std=1) so the consensus threshold of
0.01 standard deviations is meaningful across all queries.

**Spec traces:** REQ-VERIFY-144

---

### SCENARIO-VERIFY-172: Standard Arbiter Picks Correct Agent

**Statement:** Given 3 agents where 2 are wrong (higher energy) and 1 is correct (lower energy),
the arbiter MUST return the correct agent in >= 4/6 standard scenarios after z-score normalization
and optional consensus penalty.

**Given** a MultiAgentArbiter with external field scoring and z-score normalization
**And** 6 standard scenarios each with 3 agents: 1 correct (lower energy), 2 wrong (higher energy)
**When** arbitrate() is called on each scenario
**Then** the arbiter selects the correct agent (lowest normalized energy) in at least 4 of 6 cases

**Spec traces:** REQ-VERIFY-143, REQ-VERIFY-144

---

### REQ-VERIFY-145: Cross-Domain PRM Degradation Reporting

**Statement:** For cross-domain PRM evaluation, Carnot MUST compute and report
`cross_domain_degradation = auc_in_dist - auc_ood` for each OOD domain (HumanEval,
ARC-Challenge).  If the maximum degradation across domains exceeds 0.08 (the 8% baseline
published in arXiv 2506.00027), the experiment MUST identify which domain shows the
largest gap.  The artifact MUST include `beats_baseline` (bool), `published_baseline=0.08`,
and `honest_verdict` drawn from {"above_baseline", "at_baseline", "below_baseline",
"data_unavailable"}.

**Rationale:** arXiv 2506.00027 reports that PRMs trained on math reasoning degrade ~8%
AUC when applied to code verification.  Without a concrete cross-domain metric, Carnot
cannot claim its JEPA-based verifier generalises better than the published baseline.
This requirement creates a traceable, reproducible benchmark comparison.

**Acceptance criteria:**
- `cross_domain_degradation_humaneval` and `cross_domain_degradation_arc` are computed
  as `in_dist_auc - auc_domain` and recorded in the artifact.
- `beats_baseline` is True iff `cross_domain_degradation_max <= 0.08`.
- `honest_verdict` is "above_baseline" when beats_baseline is True, "at_baseline" when
  abs(degradation_max - 0.08) <= 0.01, "below_baseline" when degradation_max > 0.09.
- `corroboration_rate` = fraction of 20 VerificationCertificates where z3_verdict
  direction agrees with jepa_energy_delta direction (unsat ↔ high energy).

**Spec traces:** Exp 826, arXiv 2506.00027, arXiv 2601.17223

### SCENARIO-VERIFY-174: Load Exp 825 AUC; Compute Degradation; Emit Certificates for Failed OOD Steps

**Given** Exp 825 results file exists with `auc_gsm8k`, `auc_humaneval`, `auc_arc`,
  `overall_ood_auc`, and 20 `verification_certificates`
**And** Exp 824 results file exists with `in_dist_auc`
**When** Exp 826 runs cross-domain PRM benchmark
**Then**
  1. `cross_domain_degradation_humaneval = in_dist_auc - auc_humaneval` is computed.
  2. `cross_domain_degradation_arc = in_dist_auc - auc_arc` is computed.
  3. `cross_domain_degradation_max = max(degradation_humaneval, degradation_arc)`.
  4. `beats_baseline = (cross_domain_degradation_max <= 0.08)`.
  5. `corroboration_rate` is computed from Exp 825 certificates (unsat ↔ energy_delta > 0).
  6. If degradation_max > 0.08: `worst_domain` is identified as the higher-degradation domain.
  7. Artifact is written with all required fields per REQ-VERIFY-145.
  8. `honest_verdict` reflects the degradation comparison against 0.08 baseline.

**Spec traces:** REQ-VERIFY-145

### REQ-VERIFY-146: ActivationJailbreakProbe Layer Activation Extraction

**Statement:** ActivationJailbreakProbe MUST extract intermediate layer activations
from a small transformer model (Qwen3.5-0.8B or fallback hash projection) at layers
[4, 8, 12, 16] and train a LogisticRegression probe on labeled jailbreak/benign examples.
CPU inference latency for the probe forward pass (activation extraction + LR predict)
MUST be < 1 ms per query.

**Rationale:** arXiv 2602.11495 shows that jailbreak prompts produce a linear signal
in intermediate transformer layers detectable by logistic regression trained on 100
examples with AUC >= 0.90 at < 1 ms CPU latency.  This is orthogonal to the TF-IDF
KAN signal in Tier 0h: the KAN detects surface n-gram patterns, the activation probe
detects where the prompt sits in the model's internal representation space.

**Acceptance criteria:**
- `extract_activations(prompt)` returns np.ndarray of shape (n_layers * hidden_dim,).
- `train(prompts_labeled)` returns a fitted sklearn.linear_model.LogisticRegression.
- `evaluate(probe, test_labeled)` returns (auc: float, latency_ms: float).
- latency_ms < 1.0 for the LR forward pass alone (activation extraction excluded from
  latency budget since it is amortised across all probes in the pipeline).

**Spec traces:** Exp 828, arXiv 2602.11495

### REQ-VERIFY-147: ActivationJailbreakProbe Viability Threshold

**Statement:** ActivationJailbreakProbe probe_auc MUST be >= 0.85 on a 50/50 balanced
holdout (25 jailbreak + 25 benign, after 60/40 train/test split from 100 total) to be
considered viable for production wiring alongside Tier 0h KAN.  If probe_auc >= 0.85
AND latency_ms < 1.0 then probe_viable MUST be True; otherwise probe_viable MUST be False.

**Rationale:** The 0.85 AUC threshold is the minimum for a useful complementary signal.
Below this level, the probe adds false positives without sufficient jailbreak detection
gain to justify the additional inference cost.  The 0.85 threshold is 5 percentage
points below the published 0.90 baseline to account for the smaller training set (60
examples vs. 100 in the paper) and the synthetic vs. real JailbreakBench distribution gap.

**Acceptance criteria:**
- On 40-example holdout (20 jailbreak + 20 benign): probe_auc is computed and recorded.
- probe_viable = (probe_auc >= 0.85 and latency_ms < 1.0).
- honest_verdict in {"probe_viable", "probe_partial", "probe_not_viable"}.

**Spec traces:** Exp 828, arXiv 2602.11495

### SCENARIO-VERIFY-175: Activation Probe Train/Eval on Synthetic JailbreakBench

**Given** 50 synthetic jailbreak prompts (seed=42) + 50 synthetic benign prompts (seed=42)
**And** 60/40 train/test split: 30 jailbreak + 30 benign train, 20 jailbreak + 20 benign test
**When** ActivationJailbreakProbe.train() is called on 60 labeled prompts
**And** ActivationJailbreakProbe.evaluate() is called on 40 labeled holdout prompts
**Then**
  1. extract_activations returns shape (n_layers * hidden_dim,) for every prompt.
  2. LogisticRegression fits without error on 60 examples.
  3. probe_auc is computed from ROC AUC on the 40-example holdout.
  4. latency_ms is measured as mean of 20 predict_proba calls on one prompt.
  5. probe_viable = (probe_auc >= 0.85 AND latency_ms < 1.0).
  6. If probe_viable=True: honest_verdict = "probe_viable".
  7. If probe_auc >= 0.85 but latency_ms >= 1.0: honest_verdict = "probe_partial".
  8. If probe_auc < 0.85: honest_verdict = "probe_not_viable".
  9. Artifact written to results/experiment_828_activation_jailbreak_probe.json.

**Spec traces:** REQ-VERIFY-146, REQ-VERIFY-147

### REQ-INFRA-062: HuggingFace Model Cards MUST Include Phase 1 Disclaimer

**Statement:** All HuggingFace model cards published under the Carnot-EBM organisation
MUST include a disclaimer section with the exact text: "Phase 1 research artifact.
Trained on simulated data unless explicitly stated as live-GPU-validated. Do not use
in production without independent validation."

**Rationale:** Carnot-EBM's first 16 published models were trained on simulated data
and have not been validated on live GPU runs.  Without an explicit disclaimer, downstream
users may mistake these research artifacts for production-ready models, eroding trust
in the project and violating the project's own honesty principle ("all headline results
must have live GPU provenance").  This requirement ensures every model card is honest
about its provenance.

**Acceptance criteria:**
- huggingface_hub.list_models(author="Carnot-EBM") returns >= 1 model after this update.
- Each returned model's README contains the substring "Phase 1 research artifact".
- The disclaimer appears before any usage section.

**Spec traces:** Exp 829, CLAUDE.md honesty principle

### SCENARIO-INFRA-070: Carnot-EBM Model Count >= 17 After Exp 829 Publish

**Given** 16 existing Carnot-EBM models on HuggingFace (trained on simulated data)
**And** at least one new model artifact (JEPA v23 or IsingConstraintInjector) is eligible for publish
**When** experiment_829_huggingface_v3_publish.py runs with a valid HF_TOKEN
**Then**
  1. huggingface_hub.list_models(author="Carnot-EBM") returns >= 17 models.
  2. Every model card in the list contains "Phase 1 research artifact".
  3. n_cards_updated >= 1 (at least one existing README was updated).
  4. honest_verdict in {"hf_publish_success", "hf_publish_partial", "hf_auth_blocked"}.
  5. Artifact written to results/experiment_829_huggingface_v3_publish.json.

**Spec traces:** REQ-INFRA-062, Exp 829

### REQ-INFRA-063: Governance Pre-flight MUST Audit RETRO Closure Against Experiment Result JSONs

**Statement:** Before any new milestone experiments begin, a governance pre-flight check
MUST read the authoritative experiment result JSONs (not the operational retrospective
narrative) to determine which RETROs are genuinely still open.  If a RETRO is listed as
open in the retrospective but the referenced experiment result JSON shows a closure field
set to True (or an honest_verdict that confirms resolution), the pre-flight MUST mark
that RETRO as CLOSED in MILESTONE_PREREQS.md and remove it from the corrected_open_retros
list fed to the conductor gate.

**Why this matters:**
    The Exp 830 operational retrospective was written before Exps 819 and 820 completed,
    creating a reporting-lag error where two already-closed RETROs appeared as still-open.
    If MILESTONE_PREREQS.md carries these stale statuses, the .64 experiment gate will
    block legitimate work on a factually incorrect basis.  The experiment result JSON is
    the authoritative source of truth; the retrospective narrative is a summary that can
    fall out of sync.

**Acceptance criteria:**
- Given Exp N result JSON contains retro_injection_closed=True or honest_verdict that
  confirms closure, the governance pre-flight produces corrected_open_retros excluding
  that RETRO ID.
- MILESTONE_PREREQS.md updated section shows the RETRO as CLOSED with explicit label.
- Pre-existing content in MILESTONE_PREREQS.md is never removed.

**Spec traces:** Exp 831, RETRO-ISING-INJECTION-NO-DISCRIMINATION, RETRO-GGUF-CACHE-IMPORT

### SCENARIO-INFRA-071: Reporting-Lag RETRO Corrected From CLOSED in Exp Result JSON

**Given** the Exp 830 operational retrospective lists RETRO-ISING-INJECTION-NO-DISCRIMINATION
and RETRO-GGUF-CACHE-IMPORT as still-open (reporting-lag error)
**And** results/experiment_819_injection_field_fix.json contains retro_injection_closed=True
**And** results/experiment_820_gguf_import_fix_code_repair_v5.json contains
honest_verdict="import_fixed_repair_positive"
**When** the governance pre-flight (Exp 831) runs
**Then**
  1. audit_retro_closures() returns retros_confirmed_closed containing both RETRO IDs.
  2. corrected_open_retros does NOT contain either RETRO ID.
  3. MILESTONE_PREREQS.md updated section marks both RETROs as CLOSED.
  4. honest_verdict = "governance_ready".
  5. Artifact written to results/experiment_831_governance_preflight.json.

**Spec traces:** REQ-INFRA-063, Exp 831

---

### REQ-VERIFY-148: SymCodeVerifier.batch_verify() Single exec() Batching

`SymCodeVerifier.batch_verify(paragraphs)` MUST process N paragraphs in a single
`exec()` call, avoiding N separate `exec()` invocations.  Latency for 10 paragraphs
MUST be < 2× single paragraph latency (not N× single paragraph latency).

**Rationale (RETRO-SYMCODE-SERIAL):** verify_response() processes multi-paragraph
responses one paragraph at a time (~50ms each).  For Exp 627-style responses with
10+ paragraphs this is 500ms+ total.  Batching collects all arithmetic expressions
in one regex pass and evaluates them in a single shared exec() namespace, reducing
overhead from O(N) to O(1).

**Acceptance criteria:**
- `batch_verify(paragraphs)` returns `SymCodeBatchResult` with `per_paragraph_results`,
  `total_violations`, `batch_latency_ms`, and `n_paragraphs`.
- `n_paragraphs == len(paragraphs)`.
- Violations detected by `batch_verify()` match violations from N serial `verify_step()` calls.
- Latency for 10 paragraphs < 2× single paragraph latency.

**Spec traces:** RETRO-SYMCODE-SERIAL, Exp 841

### SCENARIO-VERIFY-173: 10-Paragraph Batch Verification Speed and Correctness

**Given** 10 synthetic paragraphs each containing 1-2 arithmetic expressions
**When** `batch_verify(paragraphs)` is called once
**And** 10 serial `verify_step()` calls are made for comparison
**Then**
  1. `batch_latency_ms` < 2× the latency of a single `verify_step()` call.
  2. `total_violations` equals the count of violations from serial calls.
  3. Each `per_paragraph_results[i].violation_detected` matches `verify_step(paragraphs[i])`.
  4. `n_paragraphs == 10`.

**Spec traces:** REQ-VERIFY-148, RETRO-SYMCODE-SERIAL, Exp 841

### REQ-VERIFY-150: EmbeddingConstraintStore MUST L2-Normalize Embeddings

`EmbeddingConstraintStore` MUST L2-normalize every embedding before storage and every
query vector before similarity computation.

**Rationale (Exp 847):** Sentence-transformer embeddings have L2 norm ~0.9-1.1, not
exactly 1.0.  Prior code applied Gram-Schmidt orthogonalization which deflected stored
embeddings away from their original semantic directions, causing cosine similarity between
query and stored constraint to be near-zero even for matching constraint types.  This made
`retrieve()` return empty lists, so IsingEBM received zero-magnitude external field input
and `delta_overall` remained 0.0 despite 15 constraints being written to the store.

**Acceptance criteria:**
- `store()` MUST normalize each embedding to unit L2 norm before appending to `_store`.
- `retrieve()` MUST normalize the query embedding to unit L2 norm before similarity computation.
- The class attribute `retrieval_l2_normalized = True` is always set.
- An assertion in `store()` and `retrieve()` verifies the invariant at runtime.
- Default `cosine_threshold` in `retrieve()` MUST be <= 0.5 (prior default 0.7 was too high
  for constraint-type variations that typically score 0.5-0.7 in sentence-transformer space).
- The constructor MUST support an explicit deterministic `ci_hash` embedding mode that does
  not load MiniLM weights, so CI and memory-constrained test runs can exercise the store
  without tripping the pytest RSS watchdog.

**Spec traces:** Exp 847, RETRO-RETRIEVAL-NEAR-ZERO-COSINE

### SCENARIO-VERIFY-230: L2-Normalized Store Produces High Cosine Similarity for Matching Constraints

**Given** an `EmbeddingConstraintStore` containing 5 stored constraints (one per violation
type: carry, sign, unit, comparison, causal)
**When** `retrieve(query)` is called with a query semantically similar to one of the stored
constraint types
**Then**
  1. `cosine_similarity(normalize(query_embedding), stored_embedding) >= 0.5`
     (not ~0.1 as produced by orthogonalized embeddings).
  2. The correct violation type is ranked first in the results.
  3. `retrieval_auroc > 0.80` over 25 (query, correct_type) pairs across 5 types × 5 variants.
  4. `retrieval_l2_normalized == True` on the store instance.

**Spec traces:** REQ-VERIFY-150, Exp 847


### REQ-VERIFY-155: SemanticEnergyProbe Tier 0f Pairwise Boltzmann Energy

`SemanticEnergyProbe` MUST compute pairwise Boltzmann-inspired semantic energy over sentence
clusters extracted from the response text.  High energy (> threshold) MUST set
`is_unstable=True` in the returned `SemanticEnergyResult` and MUST be recorded in the
`VerificationCertificate` under key `tier_0f_semantic_energy`.  The probe MUST be advisory
only (no short-circuit of downstream tiers).

**Rationale (Exp 852):** Hallucinated responses tend to contain semantically incoherent
sentences — one or more sentences that contradict or are semantically distant from the rest.
Pairwise Boltzmann energy (E = -mean k_ij where k_ij = exp(-||e_i-e_j||^2/sigma^2))
captures this incoherence without requiring logits or GPU access.  The probe is orthogonal
to all existing tiers (logit-based: 0b; latent-space: 0c, 0d; thermodynamic: 0e;
symbolic: 2.5, 2.7) and adds a diverse advisory signal.

**Acceptance criteria:**
- `SemanticEnergyProbe(sigma, threshold, embedding_dim).score(response)` returns a
  `SemanticEnergyResult` with fields: energy, is_unstable, sentence_count, cluster_entropy, threshold.
- Energy is near zero for incoherent (hallucinated) responses and negative for coherent ones.
- `is_unstable = (energy > threshold)` where default threshold is -0.5.
- When `semantic_energy_probe` is passed to `verify()`, `result.certificate["tier_0f_semantic_energy"]`
  is populated.  No tier short-circuit occurs.
- AUC on 50 synthetic pairs (25 correct, 25 hallucinated) MUST be reported in results JSON.

**Spec traces:** Exp 852


### SCENARIO-VERIFY-180: Coherent Response Has Low Semantic Energy

**Given** a factually correct, internally consistent response with 4+ sentences on one topic
**When** `SemanticEnergyProbe().score(response)` is called
**Then**
  1. `result.energy < result.threshold` (coherent cluster → low / negative energy)
  2. `result.is_unstable == False`
  3. `result.sentence_count >= 4`

**Spec traces:** REQ-VERIFY-155, Exp 852


### SCENARIO-VERIFY-181: Hallucinated Response Has High Semantic Energy

**Given** a response that inserts one sentence contradicting the others (rogue-sentence pattern)
**When** `SemanticEnergyProbe().score(response)` is called
**Then**
  1. `result.energy > result.threshold` (incoherent sentences → energy near zero)
  2. `result.is_unstable == True`
  3. `result.cluster_entropy > 0` (non-trivial entropy due to spread embeddings)

**Spec traces:** REQ-VERIFY-155, Exp 852


### REQ-PIPELINE-030: GGUFCacheResolver Export

`carnot.pipeline` MUST export `GGUFCacheResolver` for resolving GGUF model file paths
from HuggingFace model IDs without requiring downloads.  The module MUST also export
`GGUFCacheConfig`, `GGUFModelNotFoundError`, and the `resolve_gguf_path` convenience
function.

**Rationale (RETRO-GGUF-CACHE-IMPORT):** Eight consecutive milestones of SOTA code-repair
experiments failed with ImportError because no authoritative resolver existed.  Ad-hoc
path-guessing logic was scattered across experiment scripts with no shared contract.

**Acceptance criteria:**
- `from carnot.pipeline import GGUFCacheResolver` MUST NOT raise ImportError.
- `GGUFCacheResolver.resolve(model_id)` MUST raise `GGUFModelNotFoundError` (not FileNotFoundError)
  with `details["expected_path"]` populated when the file is absent.
- `GGUFCacheResolver.is_cached(model_id)` MUST return bool without raising.
- `resolve_gguf_path(model_id, cache_dir=...)` MUST return the same path as `resolver.resolve()`.

**Spec traces:** Exp 849, RETRO-GGUF-CACHE-IMPORT

### SCENARIO-PIPELINE-040: GGUFModelNotFoundError on Missing File

**Given** `model_id = "unsloth/Qwen3.6-35B-A3B-GGUF"`, `cache_dir = "models/"`,
and the file `models/unsloth_Qwen3.6-35B-A3B-GGUF-Q4_K_M.gguf` is not present on disk
**When** `GGUFCacheResolver(GGUFCacheConfig(cache_dir="models/")).resolve(model_id)` is called
**Then**
  1. `GGUFModelNotFoundError` is raised (not FileNotFoundError or KeyError).
  2. `exc.details["expected_path"]` contains the expected path string.
  3. `exc.details["model_id"]` equals `"unsloth/Qwen3.6-35B-A3B-GGUF"`.
  4. The error message mentions the expected path so a user can act on it.

**Spec traces:** REQ-PIPELINE-030, Exp 849, RETRO-GGUF-CACHE-IMPORT


### REQ-INFRA-070: ExperimentTemplate MUST Load Session Env at Init

``ExperimentTemplate.__init__`` MUST call ``EnvPropagationGuard.load_session_env()``
as its FIRST action to propagate ``CARNOT_FORCE_LIVE`` across ``claude -p`` subprocess
boundaries.

**Rationale (RETRO-LIVE-ENV-NOT-PROPAGATED, 6th consecutive recurrence):** Setting
``os.environ["CARNOT_FORCE_LIVE"]`` in one process does not propagate to ``claude -p``
subprocesses spawned by the conductor.  Writing to ``~/.carnot_session_env`` and loading
it in every ``__init__`` is the only cross-process propagation path that survives fresh
interpreter invocations.

**Acceptance criteria:**
- ``EnvPropagationGuard.write_session_env({"CARNOT_FORCE_LIVE": "1"})`` creates or
  updates ``~/.carnot_session_env`` with the entry.
- ``EnvPropagationGuard.load_session_env()`` reads ``~/.carnot_session_env`` and sets
  each key in ``os.environ`` when not already present.
- ``ExperimentTemplate.__init__`` calls ``load_session_env()`` before any other logic.
- ``apply_env_autofix()`` calls both ``os.environ[...]`` AND ``write_session_env()``.

**Spec traces:** Exp 855, RETRO-LIVE-ENV-NOT-PROPAGATED


### SCENARIO-INFRA-080: GPU Experiment Sources CARNOT_FORCE_LIVE via Session File

**Given** a prior invocation wrote ``CARNOT_FORCE_LIVE=1`` to ``~/.carnot_session_env``
via ``apply_env_autofix()``
**When** a GPU experiment is launched via ``claude -p`` (fresh process, bare env)
  and ``ExperimentTemplate(exp_id, ..., requires_gpu=True)`` is constructed
**Then**
  1. ``EnvPropagationGuard.load_session_env()`` is called in ``__init__``.
  2. ``os.environ["CARNOT_FORCE_LIVE"]`` equals ``"1"`` after construction.
  3. ``assert_live_env_if_gpu()`` does NOT raise ``RuntimeError``.

**Spec traces:** REQ-INFRA-070, Exp 855, RETRO-LIVE-ENV-NOT-PROPAGATED


### REQ-INFRA-073: GGUFCacheResolver MUST Support pre_download_and_verify()

**Requirement:** ``GGUFCacheResolver`` MUST expose a ``pre_download_and_verify(hf_repo,
filename, dest_dir)`` method that attempts to download a single GGUF file from
HuggingFace Hub and returns a result dict ``{"success": bool, "path": str|None,
"size_mb": float|None, "error": str|None}`` without raising.

**Rationale (RETRO-SOTA-MODEL-DOWNLOAD):** Exp 857's ``download()`` call failed at
runtime with an unknown error — the experiment artifact showed ``blocked_by`` with no
diagnostic.  The new method makes failure explicit and diagnosable: callers receive
the exact error string and can write an honest ``download_verified=False`` artifact
instead of hitting an unhandled exception.  Exp 869 uses a small model (Qwen3.5-0.8B
GGUF, ~500MB) to prove the mechanism end-to-end before Exp 870 trusts it for 20GB+ files.

**Acceptance criteria:**
- ``pre_download_and_verify()`` on a valid HF repo returns ``success=True`` and ``size_mb > 0``.
- ``pre_download_and_verify()`` when ``huggingface_hub`` is absent returns ``success=False``
  with a descriptive ``error`` string.
- ``pre_download_and_verify()`` when ``hf_hub_download`` raises returns ``success=False``
  with the exception message in ``error``.
- After a successful call, ``resolver.download_tested`` is ``True``.
- ``resolve_or_download()`` falls back to ``pre_download_and_verify()`` when the file
  is not in the configured ``cache_dir``.

**Spec traces:** REQ-INFRA-073, Exp 869, RETRO-SOTA-MODEL-DOWNLOAD


### SCENARIO-INFRA-082: pre_download_and_verify() on Valid HF Repo Returns Success

**Given** a ``GGUFCacheResolver`` with a writable ``dest_dir``
**When** ``pre_download_and_verify("Qwen/Qwen3.5-0.8B-GGUF", "<filename>", dest_dir)``
  is called with ``huggingface_hub.hf_hub_download`` returning a valid non-empty file
**Then**
  1. The return dict has ``success=True``.
  2. ``size_mb`` is greater than 0.
  3. ``path`` points to an existing file.
  4. ``error`` is ``None``.
  5. ``resolver.download_tested`` is ``True``.

**Spec traces:** REQ-INFRA-073, Exp 869, RETRO-SOTA-MODEL-DOWNLOAD


### REQ-VERIFY-140: StreamingCoTHalluDetector Tier 0g Advisory Wiring

**Status:** Implemented (Exp 874)

The pipeline MUST expose a `STREAMING_COT_ENABLED` class attribute on
`VerifyRepairPipeline`, set from the `CARNOT_STREAMING_COT` environment variable
(default `"0"`).  When `STREAMING_COT_ENABLED` is True, `verify()` MUST:

1. Call `extract_cot_steps(response)` to split the response into CoT steps.
2. Instantiate `StreamingCoTHalluDetector(alpha=0.3, threshold=0.35)` and call
   `detect(steps)`.
3. Set `result.streaming_cot_unstable = streaming_result.is_streaming_unstable`.
4. Set `result.streaming_cot_phas = streaming_result.final_phas`.
5. Record `result.certificate["tier_0g_streaming_cot"]` with `is_streaming_unstable`,
   `final_phas`, and `n_steps`.
6. NOT short-circuit the Ising cascade based on this signal (advisory only).

When `STREAMING_COT_ENABLED` is False (default), `verify()` MUST NOT import or
instantiate `StreamingCoTHalluDetector` — the flag must be opt-in to preserve
full backward compatibility.

**Acceptance criteria:**
- `VerifyRepairPipeline.STREAMING_COT_ENABLED` reflects the env var at import time.
- When enabled, `result.streaming_cot_unstable` and `result.streaming_cot_phas` are
  populated after a `verify()` call on any non-empty response.
- The Ising/constraint path still runs to completion (no early return from streaming signal).
- When disabled (default), `result.streaming_cot_unstable` is `False` and
  `result.streaming_cot_phas` is `0.0`.

**Spec traces:** REQ-VERIFY-140, Exp 861, Exp 874


### SCENARIO-VERIFY-165: STREAMING_COT_ENABLED Populates Certificate on Unstable CoT

**Given** `CARNOT_STREAMING_COT=1` is set and `VerifyRepairPipeline.STREAMING_COT_ENABLED` is True
**When** `verify()` is called with a response containing compounding-error CoT steps
**Then**
  1. `result.streaming_cot_unstable` is `True`.
  2. `result.streaming_cot_phas` is greater than `0.35`.
  3. `result.certificate["tier_0g_streaming_cot"]["n_steps"]` equals the number of steps extracted.
  4. `result.verified` reflects the Ising verdict, NOT the streaming signal.

**Spec traces:** REQ-VERIFY-140, SCENARIO-VERIFY-165, Exp 874


### SCENARIO-VERIFY-166: STREAMING_COT_ENABLED Disabled by Default

**Given** `CARNOT_STREAMING_COT` is not set (default `"0"`)
**When** `verify()` is called on any response
**Then**
  1. `result.streaming_cot_unstable` is `False`.
  2. `result.streaming_cot_phas` is `0.0`.
  3. `"tier_0g_streaming_cot"` is NOT a key in `result.certificate`.
  4. `StreamingCoTHalluDetector` is never imported during the call.

**Spec traces:** REQ-VERIFY-140, SCENARIO-VERIFY-166, Exp 874


### REQ-VERIFY-160: VJEPA v2 Expanded Corpus Training

**Status:** Implemented (Exp 883)

The VJEPA predictor MUST be trainable on an expanded corpus of 200+ step-label
pairs combining real FoVer pairs with synthetic GSM8K/ARC/SVAMP-style pairs,
using DomainReweightedLoss to balance signal across domain sizes.

**Acceptance criteria:**
- Synthetic pair generator produces exactly one incorrect step per problem.
- DomainReweightedLoss weights sum to 1.0 across all domains present.
- Train/eval split by question_id is reproducible (same seed → same split).
- OOD AUC after 200 epochs with 207+ training pairs exceeds Exp 877 baseline (0.5833).
- KL magnitude remains > 0.01 (no posterior collapse).

**Spec traces:** REQ-VERIFY-160, Exp 883


### SCENARIO-VERIFY-231: Synthetic Pair Generator Produces Correct Step Labels

**Given** `generate_gsm8k_synthetic(n_steps=100, seed=42)` is called
**When** the returned pairs are grouped by question_id
**Then**
  1. Each question_id group has exactly one step labelled "incorrect".
  2. All other steps in each group are labelled "correct".
  3. All steps have domain "gsm8k_synthetic".
  4. Calling twice with the same seed produces identical output.

**Spec traces:** REQ-VERIFY-160, SCENARIO-VERIFY-231, Exp 883


### SCENARIO-VERIFY-232: DomainReweightedLoss Balances 4-Domain Corpus

**Given** a corpus with 4 domains of sizes [10, 30, 20, 40]
**When** `DomainReweightedLoss.compute_domain_weights()` is called
**Then**
  1. All four domain keys appear in the returned weight dict.
  2. Weights sum to 1.0 (within 1e-5 tolerance).
  3. The smallest domain (10 samples) has a strictly higher weight than the largest (40 samples).
  4. `weighted_loss()` returns a positive scalar for non-trivial logits/labels.

**Spec traces:** REQ-VERIFY-160, SCENARIO-VERIFY-232, Exp 883


### REQ-TIER0-005: DRIFTProbe Multi-Layer Hallucination Detection (Tier 0i)

**Status:** Implemented (Exp 911)

`python/carnot/verify/drift_probe.py` implements `DRIFTProbe`, a Tier 0i advisory
probe that detects hallucination by measuring cosine distance drift between consecutive
transformer layer hidden-state representations.  Inspired by arXiv 2604.13386
(Multi-Layer Probe Ensembling): probing layer N+1 vs N captures drift signal invisible
to single-layer probes.

**Acceptance criteria:**
- REQ-TIER0-005-1: `extract_drift_signature(hidden_states)` returns a float32 array of
  shape `(n_drift_pairs,)` with values clamped to `[0, 2]`.
- REQ-TIER0-005-2: `fit(correct_examples, hallucinated_examples)` trains a
  `LogisticRegression` probe on `(drift_signature, label)` pairs with label 0=correct,
  1=hallucinated.
- REQ-TIER0-005-3: `predict_violation_prob(hidden_states)` returns a float in `[0, 1]`;
  returns 0.5 when probe has not been fitted yet.
- REQ-TIER0-005-4: Default `layers` resolves to last `n_drift_pairs+1` layer indices
  `[-(n_drift_pairs+1), ..., -1]`.
- REQ-TIER0-005-5: Missing or absent layer keys in `hidden_states` produce zero drift
  for that pair (no crash, no inflation).

**Spec traces:** REQ-TIER0-005, Exp 911


### SCENARIO-TIER0-005: DRIFTProbe AUC > 0.65 on GSM8K Hallucination Pairs

**Given** 100 GSM8K (question, correct_response, hallucinated_response) triples
where hallucinated responses inject a wrong numerical answer while preserving the
reasoning style,
**When** hidden states are extracted at the last 4 transformer layers and
`DRIFTProbe.fit()` is called on 80 training examples followed by
`roc_auc_score` on 20 held-out examples,
**Then**
  1. `ood_auc_drift` > 0.65 → honest_verdict = "tier0i_viable"
  2. `ood_auc_drift` > 0.55 → honest_verdict = "tier0i_marginal"
  3. Otherwise → honest_verdict = "tier0i_not_viable"

**Spec traces:** REQ-TIER0-005, SCENARIO-TIER0-005, Exp 911



### REQ-TIER0-006: DRIFTProbeEnsemble Per-Layer Ensemble Hallucination Detection (Tier 0i)

**Status:** Implemented (Exp 923)

`python/carnot/verify/drift_probe_ensemble.py` implements `DRIFTProbeEnsemble`, a Tier 0i
upgrade over DRIFTProbe (REQ-TIER0-005) that trains one LogisticRegression probe per
adjacent layer pair and combines predictions via learned alpha weights on a held-out
validation set.  Inspired by arXiv 2604.13386 which shows per-layer ensemble beats
single-probe concatenation by 3-8% AUROC.

**Acceptance criteria:**
- REQ-TIER0-006-1: `fit(correct_examples, hallucinated_examples)` trains N separate
  LogisticRegression probes (N = len(layers)-1), one per adjacent layer pair, each
  using only that pair's cosine distance scalar as the feature.
- REQ-TIER0-006-2: Ensemble weights alpha are learned via grid search over a 20-point
  simplex (alpha >= 0, sum(alpha) = 1) that maximises accuracy on a 20% held-out split.
- REQ-TIER0-006-3: `predict_violation_prob(hidden_states)` returns float in [0, 1];
  returns 0.5 when ensemble has not been fitted.
- REQ-TIER0-006-4: Default `layers` is `[-4, -3, -2, -1]` (last 4 layer indices, model-
  size-agnostic).
- REQ-TIER0-006-5: Missing layer keys produce zero drift for that pair (no crash).

**Spec traces:** REQ-TIER0-006, Exp 923


### SCENARIO-TIER0-006: DRIFTProbeEnsemble AUC > 0.65 on GSM8K Hallucination Pairs

**Given** 100 GSM8K (question, correct_response, hallucinated_response) triples
where hallucinated responses inject a wrong numerical answer while preserving reasoning style,
**When** hidden states are extracted at the last 4 transformer layers and
`DRIFTProbeEnsemble.fit()` is called on 80 training examples followed by
`roc_auc_score` on 20 held-out examples,
**Then**
  1. `ood_auc_drift_ensemble` > 0.65 → honest_verdict = "tier0i_viable"
  2. `ood_auc_drift_ensemble` > baseline from Exp 911 (0.565) → honest_verdict = "tier0i_improved_marginal"
  3. Otherwise → honest_verdict = "tier0i_no_improvement"

**Spec traces:** REQ-TIER0-006, SCENARIO-TIER0-006, Exp 923


### REQ-TIER0-007: SemanticEnergyDetector Synthetic Logit Prototype (Tier 0g)

**Status:** Prototype (Exp 2338)

`python/carnot/verify/semantic_energy.py` shall implement `SemanticEnergyDetector`,
a CPU-only Tier 0g prototype for hallucination detection using Boltzmann free energy
computed directly from pre-softmax logit arrays.  The detector is a prototype gate
for synthetic logits only; full validation on real LLM penultimate logits is deferred.

**Acceptance criteria:**
- REQ-TIER0-007-1: `compute_energy(logits, temperature)` returns a scalar float using
  `E = -temperature * log(sum(exp(logits / temperature)))` with a numerically stable
  log-sum-exp implementation.
- REQ-TIER0-007-2: `cluster_semantics(responses)` deterministically groups normalized
  response strings by a simple string hash and returns cluster keys mapped to response
  indices.
- REQ-TIER0-007-3: `detect(logits_per_response, responses)` returns `energy_mean`,
  `energy_std`, `semantic_entropy_estimate`, `semantic_energy_score`, and
  `is_hallucination_predicted`, with the prediction controlled by a configurable
  threshold on `semantic_energy_score`.
- REQ-TIER0-007-4: `results/experiment_2338_semantic_energy.json` records AUROC,
  FPR at TPR=0.80, energy ordering, focused test count, module path, 100 synthetic
  examples, and random seed 42.
- REQ-TIER0-007-5: `results/experiment_2351_semantic_energy_real.json` records
  a 100-example real-distribution validation using cached or live LLM logit/logprob
  vectors, including AUROC, logit source, validation boolean, and random seed 42.

**Spec traces:** REQ-TIER0-007, Exp 2338


### SCENARIO-TIER0-007: High-Variance Synthetic Logits Trigger Semantic Energy

**Given** multiple synthetic response logit arrays drawn from a high-variance normal
distribution,
**When** `SemanticEnergyDetector.detect()` is called,
**Then** `is_hallucination_predicted` is true.

**Given** multiple synthetic response logit arrays drawn from a low-variance normal
distribution,
**When** `SemanticEnergyDetector.detect()` is called,
**Then** `is_hallucination_predicted` is false.

**Spec traces:** REQ-TIER0-007, SCENARIO-TIER0-007, Exp 2338


### REQ-TIER0-008: HaltProbeDetector Cached-Logprob Probe (Tier 0j)

**Status:** Prototype (Exp 2394)

`python/carnot/verify/halt_probe.py` shall implement `HaltProbeDetector`, a
CPU-only Tier 0j hallucination-risk probe for cached logprob telemetry when
direct intermediate hidden states are unavailable.

**Acceptance criteria:**
- REQ-TIER0-008-1: `compute_halt_score(entry)` returns a finite float risk score
  using available cached logprob fields.
- REQ-TIER0-008-2: Training-free proxy A computes top-k logprob variance and
  proxy B computes softmax(top-k logprob) kurtosis for entries with
  `top_logprobs`.
- REQ-TIER0-008-3: Lightweight proxy C fits a deterministic scikit-learn
  `LogisticRegression` probe on logprob-derived features when labels are
  supplied.
- REQ-TIER0-008-4: `verify(entry)` returns `halt_risk_score`, `is_high_risk`,
  and `proxy_used`.
- REQ-TIER0-008-5: `results/experiment_2394_halt_tier0j.json` records AUROC,
  mean risk score, proxy used, sample count, random seed 42, duration, checked
  preconditions, and the Semantic Energy AUROC delta against the 0.685 baseline.

**Spec traces:** REQ-TIER0-008, Exp 2394


### REQ-TIER0-010: FregeLogic Hybrid Neural Prefilter With Z3 Tiebreaker

**Status:** Prototype (Exp 2395)

`python/carnot/verify/fregelogic_hybrid.py` shall implement `FregeLogicHybrid`, a
CPU-only hybrid hallucination-risk verifier that uses Semantic Energy and LaaB-style
logical alignment scores as neural pre-filters, invoking Z3 only when those two
pre-filter verdicts disagree.

**Acceptance criteria:**
- REQ-TIER0-010-1: `verify(entry)` returns `fregelogic_verdict`,
  `tiebreaker_invoked`, and `z3_verdict` when Z3 is invoked.
- REQ-TIER0-010-2: When Semantic Energy and LaaB agree on high-risk or low-risk,
  `verify(entry)` returns the consensus without invoking Z3.
- REQ-TIER0-010-3: When Semantic Energy and LaaB disagree, `verify(entry)` encodes
  the prompt/response answer constraint as SMT-LIB and uses Z3 as the tiebreaker.
- REQ-TIER0-010-4: `results/experiment_2395_fregelogic.json` records FregeLogic
  AUROC on 36 cached telemetry examples, Z3 tiebreaker invocation rate, delta
  against the Semantic Energy 0.685 AUROC baseline, random seed 42, duration,
  checked preconditions, and an honest terminal-prefix verdict.

**Spec traces:** REQ-TIER0-010, Exp 2395


### SCENARIO-TIER0-010: FregeLogic Invokes Z3 Only On Neural Disagreement

**Given** a telemetry entry whose Semantic Energy and LaaB pre-filter verdicts agree,
**When** `FregeLogicHybrid.verify()` is called,
**Then** `tiebreaker_invoked` is false and the returned verdict is the neural
consensus.

**Given** a telemetry entry whose Semantic Energy and LaaB pre-filter verdicts
disagree,
**When** `FregeLogicHybrid.verify()` is called,
**Then** `tiebreaker_invoked` is true and `z3_verdict` records the symbolic
tiebreaker result.

**Spec traces:** REQ-TIER0-010, SCENARIO-TIER0-010, Exp 2395


### REQ-TIER0-011: Frequency-Aware Attention Top-K Proxy (Tier 0f)

**Status:** Prototype (Exp 2397)

`python/carnot/verify/freq_aware_attention.py` shall implement
`FreqAwareAttentionDetector`, a CPU-only Tier 0f proxy for Frequency-Aware
Attention when raw attention matrices are unavailable in cached telemetry. The
proxy shall score high-frequency stopword and punctuation-fragment mass in the
recorded top-k token distribution.

**Acceptance criteria:**
- REQ-TIER0-011-1: `verify(entry)` returns `freq_attn_score`,
  `is_high_freq_pattern`, and `tier="0f"` for a telemetry entry with
  `top_logprobs`.
- REQ-TIER0-011-2: The default proxy strategy uses the fraction of recorded top-k
  token probability mass assigned to high-frequency stopwords or fragmented
  punctuation tokens.
- REQ-TIER0-011-3: `results/experiment_2397_freq_aware_attn.json` records
  `freq_attn_validated`, `freq_attn_auroc`, `freq_attn_mean_score`,
  `freq_attn_vs_semantic_energy_delta`, `proxy_strategy`, `n_eval_examples`,
  `random_seed`, `duration_s`, checked preconditions, and an honest
  terminal-prefix verdict.
- REQ-TIER0-011-4: If
  `results/live_sota_balanced_telemetry_manifest_1480.jsonl` is missing, the
  artifact uses `honest_verdict="blocked_telemetry_manifest_missing"` and does
  not fabricate metric values.

**Spec traces:** REQ-TIER0-011, Exp 2397


### SCENARIO-TIER0-011: Top-K Stopword Mass Flags High-Frequency Pattern

**Given** a cached telemetry entry whose top-k alternatives contain mostly
high-frequency stopwords,
**When** `FreqAwareAttentionDetector.verify()` is called,
**Then** `freq_attn_score` is high and `is_high_freq_pattern` is true.

**Given** a cached telemetry entry whose top-k alternatives contain mostly
content-bearing tokens,
**When** `FreqAwareAttentionDetector.verify()` is called,
**Then** `freq_attn_score` stays low and `is_high_freq_pattern` is false.

**Spec traces:** REQ-TIER0-011, SCENARIO-TIER0-011, Exp 2397


### REQ-TIER0-012: HIVE Soft-Vote Tier 0 Ensemble

**Status:** Prototype (Exp 2398)

`python/carnot/verify/hive_ensemble.py` shall implement `HiveEnsembleDetector`,
a CPU-only HIVE-style soft-vote ensemble over the importable Tier 0f
Frequency-Aware Attention, Tier 0g Semantic Energy, Tier 0h LaaB logical
consistency, and Tier 0j HALT cached-logprob verifier modules.

**Acceptance criteria:**
- REQ-TIER0-012-1: The detector discovers the four Tier 0 verifier module names
  and skips missing modules without failing module import.
- REQ-TIER0-012-2: Evaluation requires at least two importable verifier modules,
  a present `results/live_sota_balanced_telemetry_manifest_1480.jsonl`, and
  importable scikit-learn.
- REQ-TIER0-012-3: Soft-vote weights are learned from verifier score columns
  using deterministic scikit-learn `LogisticRegression` inside a stratified
  5-fold evaluation loop, and held-out weighted scores are computed as
  `sum(weight_i * score_i) / sum(weights)`.
- REQ-TIER0-012-4: `results/experiment_2398_hive_ensemble.json` records
  `honest_verdict`, `hive_ensemble_auroc`,
  `hive_gap_closed_vs_hallscan`, `ensemble_auroc_improved`,
  `n_verifiers_fused`, `verifier_weights`, `n_eval_examples`,
  `random_seed`, `duration_s`, and checked preconditions.
- REQ-TIER0-012-5: If fewer than two Tier 0 verifier modules are importable, the
  artifact reports `honest_verdict="blocked_insufficient_verifiers"` and does
  not fabricate AUROC values.

**Spec traces:** REQ-TIER0-012, Exp 2398


### SCENARIO-TIER0-012: HIVE Ensemble Fuses Available Tier 0 Scores

**Given** at least two importable Tier 0 verifier modules and the 36-row cached
telemetry manifest,
**When** the HIVE ensemble evaluates the split,
**Then** it reports 36 held-out weighted scores, learned soft-vote weights per
fused verifier, and an AUROC that is compared honestly against the 0.685
Semantic Energy baseline and the 0.88 HalluScan reference.

**Spec traces:** REQ-TIER0-012, SCENARIO-TIER0-012, Exp 2398


### REQ-TIER28-002: Typed CoT Verifier Between VERGE And Ising

**Status:** Prototype (Exp 2396)

`python/carnot/verify/typed_cot.py` shall implement `TypedCoTVerifier`, a
CPU-only Tier 2.8 verifier stage positioned after Tier 2 VERGE repair and before
Tier 3 Ising.  The verifier assigns lightweight Curry-Howard-inspired types to
chain-of-thought steps without invoking a solver.

**Acceptance criteria:**
- REQ-TIER28-002-1: `TypedCoTVerifier.classify_step(text, index, total)` returns
  Proposition, Inference, Conclusion, or Unknown using deterministic text
  patterns.
- REQ-TIER28-002-2: `TypedCoTVerifier.verify_text(text)` splits reasoning text
  into steps and returns a `typed_cot_score` equal to the fraction of steps whose
  local type dependency checks pass.
- REQ-TIER28-002-3: Inference steps type-check only after at least one
  Proposition, and Conclusion steps type-check only after at least one
  Inference.
- REQ-TIER28-002-4: `results/experiment_2396_typed_cot.json` records Typed CoT
  AUROC on 36 cached telemetry examples, mean score, delta against the Semantic
  Energy 0.685 AUROC baseline, detected CoT fields, random seed 42, duration,
  checked preconditions, and an honest terminal-prefix verdict.

**Spec traces:** REQ-TIER28-002, Exp 2396


### SCENARIO-TIER28-002: Typed CoT Checks Local Proof Dependencies

**Given** reasoning text with a proposition, an inference, and a conclusion,
**When** `TypedCoTVerifier.verify_text()` is called,
**Then** every step type-checks and `typed_cot_score=1.0`.

**Given** reasoning text whose conclusion appears before any inference,
**When** `TypedCoTVerifier.verify_text()` is called,
**Then** the conclusion fails its dependency check and lowers the
`typed_cot_score`.

**Spec traces:** REQ-TIER28-002, SCENARIO-TIER28-002, Exp 2396


### REQ-TIER28-001: DraftConditionedVerifier Tier 2.8 — Structural Constraint Injection

**Status:** Implemented (Exp 912)

`python/carnot/pipeline/draft_conditioned_verifier.py` implements `DraftConditionedVerifier`,
a Tier 2.8 stage positioned between Tier 2 (EORM/JEPA) and Tier 3 (Ising).  Inspired by
arXiv 2603.03305 (Draft-Conditioned Constrained Decoding).

Mechanism: generates a cheap 50-token draft from a small model, extracts four structural
markers (has_equals_sign, has_numeric_answer, has_reasoning_steps, final_number) using
deterministic regex — NOT ArithmeticExtractor — then injects those as soft constraints
into the Ising energy scoring.

**Acceptance criteria:**
- REQ-TIER28-001-1: `extract_structural_constraints(draft_text)` returns a list of exactly
  four dicts, each with keys "type" (str) and "value" (bool | int | None).
- REQ-TIER28-001-2: `verify_with_draft(question, full_response)` returns a `VerificationResult`
  dataclass with fields energy (float), draft_used (bool), n_constraints (int),
  draft_text (str), constraints (list).
- REQ-TIER28-001-3: When draft_runner raises an exception, draft_used=False and n_constraints=0.
- REQ-TIER28-001-4: When ising_sampler is None, score_with_constraints() returns a synthetic
  energy in range [0.0, 1.5] computed from structural signals.
- REQ-TIER28-001-5: `condition_and_verify(question, response)` returns a plain dict with
  the same fields (interface for ThreeTierPipeline.wire_tier_28()).

**Spec traces:** REQ-TIER28-001, Exp 912


### SCENARIO-TIER28-001: Draft-Conditioned Constraints Improve Ising Solve Quality on GSM8K

**Given** 25 GSM8K-style (question, correct_response, hallucinated_response) pairs,
**When** DraftConditionedVerifier is run with a Qwen3.5-0.8B draft runner and constraints
are injected before the Ising energy scoring,
**Then**
  1. auc_with_draft > auc_baseline → honest_verdict = "tier28_viable"
  2. auc_with_draft <= auc_baseline → honest_verdict = "tier28_no_improvement"
  3. mean_constraints_injected is recorded per question.

**Spec traces:** REQ-TIER28-001, SCENARIO-TIER28-001, Exp 912


### REQ-PERF-004: DualGPURunner Wired to ThreeTierPipeline Batch Dispatch

When `CARNOT_DUAL_GPU=1` and a `DualGPURunner`-compatible runner is attached via
`ThreeTierPipeline.wire_dual_gpu_runner()`, the pipeline's `benchmark()` method
MUST dispatch verification tasks across two concurrent worker threads, one per GPU
partition.  When `CARNOT_DUAL_GPU=0` (or the runner is None), the pipeline MUST
fall back to sequential single-GPU processing with no performance regression.

**Rationale:** DualGPURunner was validated at 1.979x throughput improvement in Exp 856
but was never connected to ThreeTierPipeline.  Wiring it closes the gap between the
validated component and the production pipeline.

**Acceptance criteria:**
- `pipeline.wire_dual_gpu_runner(runner)` stores the runner and does not raise.
- With `CARNOT_DUAL_GPU=1` and runner wired, `benchmark()` uses two threads.
- With `CARNOT_DUAL_GPU=0`, `benchmark()` runs sequentially (no regression).
- Observed throughput with CARNOT_DUAL_GPU=1 is >= 1.0x baseline on any hardware.

**Spec traces:** REQ-PERF-004, Exp 913


### SCENARIO-PERF-004: CARNOT_DUAL_GPU=1 Enables Parallel Batch Verification

**Given** 20 synthetic GSM8K-style (question, response) pairs and a
ThreeTierPipeline with stub EORM and Ising,
**When** `CARNOT_DUAL_GPU=0` (baseline) and `CARNOT_DUAL_GPU=1` (dual-GPU) are
each used to run `benchmark()`,
**Then**
  1. observed_speedup = baseline_wall_time / dualgpu_wall_time is measured.
  2. honest_verdict = "dualgpu_wired_speedup_confirmed" if observed_speedup > 1.7
  3. honest_verdict = "dualgpu_wired_partial_speedup" if 1.0 < observed_speedup <= 1.7
  4. honest_verdict = "dualgpu_wired_no_speedup" if observed_speedup <= 1.0
  5. Falling back to CARNOT_DUAL_GPU=0 does NOT raise and does NOT regress
     sequential throughput by more than 5%.

**Spec traces:** REQ-PERF-004, SCENARIO-PERF-004, Exp 913


### REQ-PIPE-025: DraftConditionedVerifier (Tier 2.8) Wired into ThreeTierPipeline

`ThreeTierPipeline.wire_tier_28(verifier)` MUST attach a `DraftConditionedVerifier`
instance so that `verify()` calls `verifier.condition_and_verify(question, response)`
for every response that reaches Tier 3 (Ising).  The advisory result MUST be stored
in `self._last_tier28_advisory`.  When `draft_conditioned_verifier` is None, the
behaviour MUST be identical to the pre-Tier-2.8 pipeline (ADDITIVE, no regression).

**Rationale:** Exp 912 confirmed DraftConditionedVerifier is viable standalone
(AUC 0.42 → 0.48, signed_energy_improvement=0.011).  Exp 938 wires it into the
production pipeline so the improvement is captured end-to-end.

**Acceptance criteria:**
- `pipeline.wire_tier_28(verifier)` MUST NOT raise.
- After wiring, calling `pipeline.verify(response, question=q)` MUST invoke
  `verifier.condition_and_verify(q, response)`.
- `pipeline._last_tier28_advisory` MUST be populated after each `verify()` call
  that reaches Tier 3.
- When `wire_tier_28` is not called, `pipeline._last_tier28_advisory` MUST remain None
  (or be unset) after each `verify()` call.
- tier28_activation_count >= 3 in a 20-question run is the acceptance gate (Exp 938).

**Spec traces:** REQ-PIPE-025, REQ-TIER2-010, Exp 912, Exp 938


### SCENARIO-PIPE-010: DraftConditioned Tier 2.8 Activates on Causal Uncertainty

**Given** a ThreeTierPipeline with stub EORM (energy=0.9, above eorm_threshold=0.5)
and a DraftConditionedVerifier wired via wire_tier_28(),
**When** 20 arithmetic questions are run through the full pipeline end-to-end,
**Then**
  1. `tier28_activation_count >= 3` — Tier 2.8 fires for at least 3 questions.
  2. `pipeline._last_tier28_advisory` is a dict with keys energy, draft_used,
     n_constraints, draft_text, constraints after each verify() call.
  3. `honest_verdict == "tier28_wired"` if both activation and energy delta conditions hold.
  4. `honest_verdict == "tier28_wired_no_activation"` if Tier 2.8 is wired but never fires.
  5. `honest_verdict == "tier28_wiring_failed"` if wire_tier_28() raises.

**Spec traces:** REQ-PIPE-025, SCENARIO-PIPE-010, Exp 938


### REQ-VERIFY-098: ThinkPRM Generative Step Verifier

The pipeline MUST provide a ThinkPRMVerifier component that accepts a reasoning step
string and returns a step-level verdict (correct/incorrect/uncertain) derived from
a model-generated chain-of-thought explanation, NOT a heuristic rule.

The verifier MUST:
1. Build a 3-step CoT verification prompt (extract claim, check arithmetic/logic, state verdict).
2. Call an LLM to generate the CoT before emitting VERDICT: CORRECT or VERDICT: INCORRECT.
3. Parse the LAST occurrence of VERDICT: CORRECT/INCORRECT from the LLM output.
4. Return verdict='uncertain' (confidence=0.5) when no VERDICT line is found.
5. Operate in CI stub mode (llm_caller=None) without loading any model.
6. Support batch_verify(steps) returning results in input order.

Motivation: Exp 924 showed AUC delta=0 using heuristic rule-based explanations.
arXiv 2504.16828 (ThinkPRM) proves model-generated CoT achieves +8% on GPQA-Diamond
vs discriminative PRM using only 1% of labels. Exp 945 validates this on synthetic
GSM8K step corpus (AUROC 0.99 vs heuristic baseline 0.85, delta=+0.14).

**Acceptance criteria:**
- `ThinkPRMVerifier().verify_step("3+4=7")` returns ThinkPRMResult with verdict='uncertain' (CI stub).
- With arithmetic-checking llm_caller, verify_step("3+4=7") returns verdict='correct'.
- With arithmetic-checking llm_caller, verify_step("3+4=8") returns verdict='incorrect'.
- AUROC on 100-step correct/incorrect corpus > 0.70.

**Spec traces:** Exp 924 (baseline), Exp 945 (ThinkPRM), arXiv 2504.16828


### SCENARIO-VERIFY-130: ThinkPRM Verify Step with CoT

**Given** a ThinkPRMVerifier with a stub LLM caller that returns "VERDICT: CORRECT"
**When** verify_step("10 + 5 = 15") is called
**Then**
  1. result.verdict == 'correct'
  2. result.confidence == 0.95
  3. result.step_text == "10 + 5 = 15"
  4. result.reasoning_steps contains the LLM output
  5. result.latency_ms >= 0.0

**Given** a ThinkPRMVerifier with a stub LLM caller that returns "VERDICT: INCORRECT"
**When** verify_step("10 + 5 = 16") is called
**Then** result.verdict == 'incorrect' and result.confidence == 0.95

**Given** a ThinkPRMVerifier with llm_caller=None (CI stub)
**When** verify_step("any step") is called
**Then** result.verdict == 'uncertain' and result.confidence == 0.5

**Spec traces:** REQ-VERIFY-098, Exp 945

### REQ-PROBE-022: SpilledEnergyDetector Training-Free Tier 0 Pre-filter

SpilledEnergyDetector MUST compute per-response "spilled energy" from LLM token
log-probabilities with zero additional inference cost (no secondary model call, no
training required).  It MUST expose `compute_spill()`, `flag_response()`, and
`benchmark()` methods.  The `benchmark()` method MUST return a `SpilledEnergyResult`
with `auroc`, `optimal_threshold`, `skip_rate`, `fn_rate`, and `honest_verdict` fields.

**Rationale:** arXiv 2602.18671 shows that hallucinations produce tokens whose
log-probability exceeds the contextual expectation (high entropy context + overconfident
token = "spilled energy").  Using only the existing generation logits makes this
pre-filter zero-cost compared to ThinkProbe (~50–200 ms secondary LLM call).

**Acceptance criteria:**
- `compute_spill(log_probs, context_entropy)` returns 0.0 when all tokens are at
  or below the expected log_p (no spill).
- `compute_spill([-0.5, -0.5, -0.5], 2.0)` returns ≈ 1.5 (each token 1.5 nats above
  expectation).
- `flag_response(score, threshold)` returns True iff score >= threshold.
- `benchmark(corpus, labels)` returns auroc > 0.60 on a synthetic corpus where
  hallucinated responses have higher mean log_p than correct responses
  (Exp 949 achieves auroc=1.0 with full separation).

**Spec traces:** Exp 949

## Scenarios

### SCENARIO-PROBE-022: Spill Computation on Correct vs Hallucinated Mock Responses

**Given** a SpilledEnergyDetector with context_entropy=2.0
**When** compute_spill is called with:
  - correct tokens: log_probs=[-2.0, -2.0, -2.0] (at expectation)
  - hallucinated tokens: log_probs=[-0.5, -0.5, -0.5] (above expectation)
**Then**
  - correct spill = 0.0
  - hallucinated spill = 1.5
  - hallucinated spill >> correct spill (clear separation)

**Spec traces:** REQ-PROBE-022

### REQ-VERIFY-1143: HalluGuard Cascade Router v3 Features

The Lagrangian cascade router experiment MUST extend the Exp 1131 verifier-score
router inputs from three features to five features by adding:

- `entropy_proxy`: unique token count divided by total token count for the response.
- `embedding_distance`: cosine distance from the query embedding to the FoVer
  training-set centroid, using a local sentence-transformers
  `all-MiniLM-L6-v2` encoder when available.

The Exp 1143 artifact MUST record the feature-count change, FoVer train/holdout
sizes, adaptive TP rate, fixed TP rate, accuracy delta, cost savings, whether the
HalluGuard features were measured, and whether the features explain ThinkPRM
misses on the Goodfire exemplar failure set.

**Acceptance criteria:**
- The router feature vector has five columns in the order
  `sem_energy_score`, `response_length`, `step_count`, `entropy_proxy`,
  `embedding_distance`.
- The MLP keeps the Exp 1131 two-hidden-layer 128-unit architecture with a
  five-feature input layer.
- `results/experiment_1143_halluguard_cascade_router_v3.json` includes
  `halluguard_features_added`, `n_router_features_before`,
  `n_router_features_after`, `training_set_size`, `holdout_set_size`,
  `adaptive_tp_rate`, `fixed_tp_rate`, `accuracy_delta`, `cost_savings_pct`,
  `halluguard_features_explain_goodfire_failures`,
  `halluguard_routing_feature_measured`, and `honest_verdict`.

**Spec traces:** Exp 1143, HalluGuard data/reasoning decomposition

### SCENARIO-VERIFY-1143: HalluGuard Features Route ThinkPRM Misses to k=5

**Given** the FoVer training corpus and the Goodfire exemplar cascade artifact,
**When** Exp 1143 trains the five-feature Lagrangian cascade router and scores
Goodfire exemplars missed by ThinkPRM,
**Then** the artifact reports whether high `entropy_proxy` or high
`embedding_distance` predicts routing those misses to the k=5 full cascade.

**Spec traces:** REQ-VERIFY-1143

### REQ-VERIFY-1145: Goodfire Cheap-Tier Distillation Calibration

The Goodfire cheap-tier distillation experiment MUST read the Exp 1143
HalluGuard diagnostic and the Exp 1132 Goodfire cascade artifact, then score the
Goodfire exemplar corpus with the same cheap-tier score conventions used by Exp
1132:

- `ThinkPRM`: the Tier 0a SpilledEnergy proxy score with the default threshold
  `0.372`.
- `SemEnergy`: `SemEnergyProbe.score_response_proxy` with the default threshold
  `-0.5`.

The experiment SHALL treat the k=5 label as positive ground truth for every
Goodfire exemplar, find the true-positive-maximizing threshold for each cheap
tier, and apply only the HalluGuard-feature-consistent threshold adjustment:
lower the ThinkPRM threshold when `entropy_proxy` is the dominant missed-failure
feature, or lower the SemEnergy threshold when `embedding_distance` is dominant.
The adjusted cheap tier SHALL use OR-logic across ThinkPRM and SemEnergy.

The artifact SHALL be written to
`results/experiment_1145_goodfire_cheap_tier_distillation.json` and include
`n_exemplars`, `thinkprm_tp_before`, `semenergy_tp_before`,
`combined_cheap_tp_before`, `thinkprm_threshold_adjusted`,
`semenergy_threshold_adjusted`, `combined_cheap_tp_after`,
`false_positive_rate_after`, `cheap_tier_tp_rate_improved`, and
`honest_verdict`.

**Acceptance criteria:**
- The baseline rates match Exp 1132 for the 36-exemplar Goodfire corpus:
  `thinkprm_tp_before=0.138889` and `semenergy_tp_before=0.222222`.
- `combined_cheap_tp_after` is computed with OR-logic over the adjusted
  ThinkPRM and SemEnergy flags.
- `false_positive_rate_after` is measured on 100 correct FoVer examples.
- `honest_verdict` is one of
  `cheap_tier_calibrated_tp_improved`, `calibration_no_improvement`,
  `threshold_trade_off_fp_increase`, or `honest_negative`.

**Spec traces:** Exp 1145, REQ-VERIFY-1143, Exp 1132

### SCENARIO-VERIFY-1145: Entropy-Driven Cheap-Tier Misses Lower ThinkPRM Threshold

**Given** the Exp 1143 artifact reports HalluGuard features explain the Goodfire
cheap-tier misses,
**And** `entropy_proxy` flags misses more often than `embedding_distance`,
**When** Exp 1145 calibrates cheap-tier thresholds and evaluates Goodfire
exemplars plus 100 correct FoVer examples,
**Then** the artifact reports a ThinkPRM threshold adjustment, leaves the
SemEnergy threshold unadjusted, improves combined cheap-tier TP under OR-logic,
and records whether that improvement persists by category.

**Spec traces:** REQ-VERIFY-1145

### REQ-VERIFY-1160: MARCH Blinded Multi-Agent Claim Check

The MARCH experiment MUST implement a lightweight three-role self-check loop
over the Goodfire exemplar corpus:

- Solver output is the candidate response already present in each corpus row.
- Proposer extracts 2-4 atomic verifiable claims from the response with local
  deterministic rules rather than a live LLM call.
- Checker evaluates each extracted claim while blinded to the original response;
  the checker input contract is only the original question plus the extracted
  claim. Numeric claims SHALL be routed through `Z3MathVerifier` or equivalent
  exact arithmetic checks, while non-numeric claims SHALL use deterministic
  rule-based checks scoped to the question and claim.
- A response SHALL be marked hallucinated when any blinded claim check fails.

The experiment SHALL evaluate all 36 Goodfire exemplars and 100 correct FoVer
examples, write `results/experiment_1160_march_multiagent_claim_check.json`,
and include at least these artifact fields:
`n_exemplars`, `n_correct_examples`, `thinkprm_baseline_tp`,
`semenergy_baseline_tp`, `march_tp_rate`, `march_fpr`,
`march_tp_above_baseline`, `claims_per_response_mean`,
`blinded_checker_used`, `march_multiagent_honest_result`, and
`honest_verdict`.

**Acceptance criteria:**
- `thinkprm_baseline_tp` is reported as `0.139` and
  `semenergy_baseline_tp` is reported as `0.222`, matching Exp 1132 to the
  precision requested by Exp 1160.
- `blinded_checker_used` and `march_multiagent_honest_result` are `true`.
- `march_tp_above_baseline` is true iff `march_tp_rate > 0.222`.
- `honest_verdict` is one of `march_tp_above_semenergy_baseline`,
  `march_tp_between_baselines`, `march_below_all_baselines`, or
  `extractor_failed`.

**Spec traces:** Exp 1160, REQ-VERIFY-1145, Exp 1132

### SCENARIO-VERIFY-1160: Blinded Checker Flags Goodfire Claims Without Original Response

**Given** a Goodfire exemplar with a buggy response and an original question,
**When** the Proposer extracts atomic claims and the Checker validates each
claim using only `(question, claim)`,
**Then** the per-claim result records that the original response was not visible
to the checker,
**And** the response-level MARCH verdict is hallucinated if any claim fails.

**Spec traces:** REQ-VERIFY-1160

### SCENARIO-PROBE-023: Benchmark AUROC and Honest Verdict on Synthetic Corpus

**Given** a SpilledEnergyDetector and 200 synthetic responses (100 correct, 100 hallucinated)
  where hallucinated responses have log_probs drawn from N(-1.5, 2.0) and correct from N(-2.0, 0.5)
  with context_entropy=2.0 for all responses
**When** benchmark(responses, labels) is called
**Then**
  - auroc > 0.60
  - honest_verdict == 'spilled_energy_viable'
  - skip_rate in [0.0, 1.0]
  - fn_rate in [0.0, 1.0]

**Spec traces:** REQ-PROBE-022

### REQ-VERIFY-1352: Certificate Completion-Budget Preflight Before SOTA Spend

The certificate pipeline MUST perform a CPU-only completion-budget and dynamic
dispatch preflight before any Exp 1353-class SOTA certificate run spends GPU
time.  The preflight SHALL reuse the Exp 1339 SAT, UNSAT, UNKNOWN, and repair
dispatch surface, build tiny structurally valid synthetic certificates for each
state, estimate the minimum completion tokens required after an emitted
structural tag, and compare those estimates against the active certificate
`max_tokens` setting.

The artifact SHALL be written to
`results/experiment_1352_truncproof_xgrammar_certificate_completion_preflight.json`
and include at least `status`, `grammar_states`,
`min_completion_tokens_by_state`, `max_token_budget_sufficient`,
`structural_tag_supported`, `xgrammar_backend_available`,
`dynamic_dispatch_preserved`, `sota_run_allowed`, `blocker_if_not_allowed`, and
`honest_verdict`.  `sota_run_allowed` SHALL be true only when
`max_token_budget_sufficient` and `dynamic_dispatch_preserved` are both true.
If XGrammar is unavailable locally, the artifact SHALL record that honestly and
test the pure-Python TagDispatch fallback instead of calling a SOTA model.

**Acceptance criteria:**
- The preflight writes an `in_progress` artifact before the local probes run,
  then overwrites it with a terminal `complete` artifact.
- SAT, UNSAT, UNKNOWN, and repair states all dispatch dynamically from
  structural-tagged synthetic completions.
- `max_token_budget_sufficient` is computed from the active `max_tokens`
  setting and the largest minimum completion-token estimate.
- When either budget sufficiency or dynamic dispatch fails,
  `sota_run_allowed=false` and `blocker_if_not_allowed` names the exact blocker.

### SCENARIO-VERIFY-1352: CPU Preflight Gates Exp 1353 GPU Spend

**Given** Exp 1339 proved local dynamic TagDispatch for SAT, UNSAT, UNKNOWN, and
repair states
**When** Exp 1352 runs with the active certificate runtime settings
**Then** it records structural-tag support, XGrammar availability, dynamic
dispatch preservation, and the completion-token budget decision
**And** it either allows Exp 1353 GPU spend with `sota_run_allowed=true` or
writes a terminal blocker explaining why the SOTA run remains closed.

**Spec traces:** REQ-VERIFY-1352

### REQ-VERIFY-1371: Margin-Aware Cactus/BEAVER Scheduler Replay

The Cactus/BEAVER scheduler evaluation MUST run as a CPU-only replay over Exp
1369 semantic-validator rows and Exp 1370 MCS repair-localization rows.  It
MUST assign conservative semantic margins, directly accept only high-margin SAT
rows that the full verifier also accepts, and escalate every low-margin,
UNSAT, UNKNOWN, or REPAIR_HINT row to the full verifier.  UNKNOWN rows MUST
never be silently accepted.

The artifact SHALL be written to
`results/experiment_1371_margin_aware_cactus_beaver_scheduler_v3.json` and
include at least `status`, `proxy_accept_rate`, `low_margin_escalation_rate`,
`full_verifier_call_reduction`, `false_acceptance_rate`,
`repair_hint_reuse_rate`, `verifier_cost_reduction_proxy`,
`triage_claim_allowed`, and `honest_verdict`.
`triage_claim_allowed` SHALL be true only when false acceptance is exactly zero
and UNKNOWN rows are never silently accepted.

**Acceptance criteria:**
- The runner writes an `in_progress` artifact before replaying policy rows, then
  overwrites it with a terminal artifact.
- If Exp 1370 reports `repair_hint_precision < 0.5`, the scheduler artifact is
  blocked and no call-reduction claim is allowed.
- `full_verifier_call_reduction` is positive only when the scheduler avoids at
  least one full verifier call, and the claim is allowed only at zero false
  acceptance.

### SCENARIO-VERIFY-1371: Conservative Scheduler Escalates Unknowns

**Given** Exp 1369 contains SAT, UNSAT, UNKNOWN, and REPAIR_HINT semantic rows
**And** Exp 1370 supplies precise MCS repair hints for the non-SAT rows
**When** Exp 1371 replays the margin-aware scheduler
**Then** only the high-margin SAT row is accepted by the proxy
**And** UNSAT, UNKNOWN, and REPAIR_HINT rows are escalated with repair-hint
reuse where available
**And** `triage_claim_allowed=true` only when `false_acceptance_rate=0.0`.

**Spec traces:** REQ-VERIFY-1371

### REQ-VERIFY-1382: Full-Scale Certificate Semantic Repair Pipeline

The repository shall provide a full-scale Exp 1382 runner that executes the
certificate extraction, semantic validation, MCS repair-localization, and
margin-aware scheduler chain over at least 50 local FoVer corpus cases, targeting
100 cases when available.

The runner MUST write
`results/experiment_1382_fullscale_certificate_semantic_repair_100cases.json`
with `status="in_progress"` before loading model specs, corpus rows, or the DVI
checkpoint. It MUST load the Exp 1381 artifact, require `dvi_deployed=true`, and
use that checkpoint path as the DVI verifier for semantic validation. It MUST
resolve headline model specs through `cached_sota_pair(gpu_indices=(0, 1))` and
headline LLM generation rows MUST use those specs.

The runner MUST save `results/exp1382_ckpt.json` every 25 processed cases. The
terminal artifact SHALL include at least `status`, `total_fover_cases`,
`certificate_extract_count`, `certificate_parse_rate`,
`semantic_validation_pass_rate`, `mcs_repair_localization_rate`,
`repair_hint_precision`, `scheduler_accept_rate`,
`scheduler_false_acceptance_rate`, `full_pipeline_pass_rate`,
`dvi_checkpoint_used`, `headline_result_allowed`, and `honest_verdict`.
`headline_result_allowed` SHALL be true only when at least 50 FoVer cases ran
and a mandated cached SOTA GGUF generation source produced
`certificate_parse_rate >= 0.75`.

**Acceptance criteria:**
- The runner writes an in-progress artifact first and a terminal artifact last.
- A checkpoint artifact is written at 25-case intervals.
- If the DVI checkpoint is absent or not deployed, the terminal artifact is
  blocked and the headline gate is false.
- Full-pipeline pass rate counts only cases that parse, pass semantic
  validation, and are scheduler-accepted without requiring repair.

### SCENARIO-VERIFY-1382: Full Pipeline Produces Auditable Paper Statistics

**Given** Exp 1381 deployed a DVI checkpoint
**And** the local FoVer corpus contains at least 50 labeled rows
**And** `cached_sota_pair(gpu_indices=(0, 1))` resolves a mandated GGUF pair
**When** Exp 1382 runs the full pipeline
**Then** the terminal artifact reports certificate, semantic validation, MCS
repair, scheduler, full-pipeline, DVI-checkpoint, and headline-gate statistics
for the processed FoVer cases.

**Spec traces:** REQ-VERIFY-1382, SCENARIO-VERIFY-1382

### REQ-VERIFY-1391: Exp 1382 Semantic-Failure Diagnosis

The repository shall provide a deterministic diagnostic that reads
`results/experiment_1382_fullscale_certificate_semantic_repair_100cases.json`
and writes
`results/experiment_1391_fullscale_pipeline_failure_diagnosis.json` with
`status="in_progress"` before loading the source artifact. The terminal artifact
MUST classify every Exp 1382 semantic-validation failure into one of
`Z3_CONSTRAINT_MISMATCH`, `MISSING_CERTIFICATE_FIELD`,
`SEMANTIC_CONTRADICTION`, `CORPUS_SPECIFIC`, `VALIDATOR_BUG`, or `OTHER`, and it
MUST report category counts, top category, fixable-failure fraction, estimated
semantic-validation pass rate after tractable fixes, recommended fixes, and an
honest verdict.

**Acceptance criteria:**
- The diagnostic confirms the Exp 1382 parse rate and semantic failure count.
- Every failed semantic row receives exactly one taxonomy category.
- Category counts include zero-count categories so downstream Exp 1396 gates can
  distinguish absent parser/Z3 failures from missing analysis.
- The artifact records whether the diagnosis is complete.

### SCENARIO-VERIFY-1391: DVI Disagreements Are Ranked For Exp 1396

**Given** Exp 1382 contains parseable certificates but failed semantic rows
whose DVI state disagrees with the FoVer label
**When** Exp 1391 diagnoses the artifact
**Then** corpus-specific false-SAT failures and validator false-repair failures
are counted separately, ranked, and mapped to concrete Exp 1396 fixes.

**Spec traces:** REQ-VERIFY-1391, SCENARIO-VERIFY-1391

### REQ-VERIFY-1397: Full-Scale Pipeline V2 200-Case Headline Run

The repository shall provide an Exp 1397 runner that replays the full
certificate extraction, calibrated semantic validation, VERGE/MCS
repair-localization, and scheduler chain over 200 local FoVer corpus cases after
the Exp 1396 semantic validation fixes are confirmed.

The runner MUST write
`results/experiment_1397_fullscale_pipeline_v2_200cases.json` with
`status="in_progress"` before loading model specs, source artifacts, corpus
rows, or the DVI checkpoint. It MUST read
`results/experiment_1396_semantic_validation_pass_rate_fix_v1.json` and proceed
only when `semantic_validation_improvement_measured=true`. It MUST resolve
`MODEL_SPECS` through `cached_sota_pair(gpu_indices=(0, 1),
preferred_quant="Q4_K_M")`, and headline LLM generation rows MUST come from
those cached SOTA GGUF specs.

The terminal artifact SHALL include at least `status`, `cases_evaluated`,
`models_used`, `certificate_extract_count`, `certificate_parse_rate`,
`semantic_validation_pass_rate`, `full_pipeline_pass_rate`,
`semantic_validation_improvement_vs_exp1382`,
`full_pipeline_improvement_vs_exp1382`, `headline_result_allowed`, and
`honest_verdict`. Improvements SHALL be measured against the Exp 1382 baselines
`semantic_validation_pass_rate=0.59` and `full_pipeline_pass_rate=0.29`.
`headline_result_allowed` SHALL be true only when the semantic validation pass
rate is at least 0.70, the full pipeline pass rate is at least 0.40, and the
certificate generation rows have mandated cached SOTA GGUF provenance.

**Acceptance criteria:**
- The runner writes an in-progress artifact first and a terminal artifact last.
- The Exp 1396 prerequisite gate blocks the run when semantic improvement was
  not measured.
- Exactly 200 FoVer cases are evaluated when enough labeled rows exist.
- The terminal artifact records improvement deltas versus Exp 1382 and applies
  the metric and SOTA-provenance headline gate.

### SCENARIO-VERIFY-1397: Publication-Quality 200-Case Pipeline Result

**Given** Exp 1396 measured semantic validation improvement
**And** the local FoVer corpus contains at least 200 labeled rows
**And** `cached_sota_pair(gpu_indices=(0, 1), preferred_quant="Q4_K_M")`
resolves a mandated SOTA GGUF pair
**When** Exp 1397 runs the full pipeline
**Then** the terminal artifact reports 200-case certificate parse, semantic
validation, repair, full-pipeline, baseline-improvement, model-provenance, and
headline-gate statistics for the processed FoVer cases.

**Spec traces:** REQ-VERIFY-1397, SCENARIO-VERIFY-1397

### REQ-VERIFY-1413: Certificate Repair Execution Diagnosis

The repository shall provide a deterministic Exp 1413 diagnostic that reads
`results/experiment_1397_fullscale_pipeline_v2_200cases.json`, classifies the
repair-hint rows emitted by VERGE MCS, and writes
`results/experiment_1413_certificate_repair_execution_diagnosis.json` with
`status="in_progress"` before loading the source artifact. The terminal
artifact SHALL include at least `status`, `total_cases_analyzed`,
`repair_hint_cases_total`, `no_repair_cases_total`,
`repair_execution_diagnosis_complete`, `hint_category_counts`,
`executable_hint_pct`, `recommended_executor_contract`,
`expected_full_pipeline_pass_rate_if_50pct_repaired`, and `honest_verdict`.

Repair hints SHALL be categorized into `FIELD_REWRITE`, `STEP_REWRITE`,
`CONSTRAINT_REWRITE`, `CERTIFICATE_REGENERATE`, and `UNKNOWN`. The diagnostic
SHALL estimate `executable_hint_pct` as the fraction of repair-hint rows that
can be handled by a bounded local LLM rewrite prompt. When Exp 1397 provides a
repair-specific denominator, the expected full-pipeline pass-rate estimate for
50 percent repaired SHALL add half of repair-hint cases divided by total cases
to the measured full-pipeline pass rate; otherwise it SHALL fall back to
`full_pipeline_pass_rate + 0.5 * (1 - full_pipeline_pass_rate)`.

**Acceptance criteria:**
- The runner writes an in-progress artifact first and a terminal artifact last.
- Category counts include all five repair-hint categories, including zero-count
  categories.
- The recommended executor contract names inputs, outputs, the validation call,
  timeout, and fallback behavior needed by Exp 1414.
- The honest verdict explicitly states whether Exp 1397's blocker is missing
  repair execution rather than certificate parsing or semantic validation.

### SCENARIO-VERIFY-1413: Exp 1397 Repair Hints Are Classified For Executor Design

**Given** Exp 1397 reports parse and semantic validation pass rates of 1.0 but
a sub-threshold full-pipeline pass rate
**And** the repair-localization rows include VERGE MCS repair hints
**When** Exp 1413 diagnoses the source artifact
**Then** every repair-hint row contributes to exactly one taxonomy category,
the executable hint percentage is computed from those rows, and the executor
contract describes a bounded local LLM rewrite followed by semantic and
scheduler validation before accepting a repaired certificate.

**Spec traces:** REQ-VERIFY-1413, SCENARIO-VERIFY-1413

### REQ-VERIFY-1414: Bounded Local LLM Certificate Repair Executor

The repository shall provide an opt-in certificate repair executor for Exp
1414 that accepts the original prompt or question, current certificate,
`REPAIR_HINT`, validator feedback, and an allowed output schema. The executor
MUST build a bounded JSON-only prompt for a local open-weight GGUF model, parse
only the allowed schema from the model output, and return a corrected
certificate plus metadata.

The core pipeline MUST depend on a generator protocol or callable rather than a
vendor SDK. Closed-weight dependencies are prohibited in `python/carnot/pipeline/`.
The executor MUST accept a repair candidate only after replaying the existing
semantic validation contract: the validation result must show
`constraint_passed=true`, `semantic_result="SAT"`, `repair_required=false`,
and `false_acceptance=false`. Invalid JSON, schema violations, timeout, local
model unavailability, or validation failure MUST preserve the original repair
hint and return a structured fallback result instead of silently accepting.

New LLM-bearing Exp 1414 artifacts MUST include the mandated SOTA GGUF
`model_specs` for `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`; model resolution MUST use
`cached_sota_pair()` or an equivalent local cache resolver. If no mandated
local SOTA GGUF is available, the experiment MUST write a blocked artifact
with cache details rather than simulating headline repair results.

**Acceptance criteria:**
- The executor prompt contains the input contract and the allowed output schema
  while bounding field lengths before model invocation.
- Model output is rejected unless it is JSON with a string
  `corrected_certificate` and JSON-compatible metadata.
- The pipeline exposes a disabled-by-default hook that invokes the executor
  only when explicitly enabled.
- The terminal Exp 1414 artifact includes `status`, `model_specs`,
  `repair_executor_deployed`, `repair_hint_cases_tested`,
  `repaired_cases_successful`, `repaired_case_success_rate`,
  `semantic_equivalence_pass_rate_after_repair`, `local_sota_model_used`,
  `tests_run`, and `honest_verdict`.

### SCENARIO-VERIFY-1414: Exp 1397 Repair Hints Are Executed Or Honestly Blocked

**Given** Exp 1397 contains at least twenty repair-hint cases
**And** a mandated local SOTA GGUF resolves from the cache
**When** Exp 1414 runs with the opt-in certificate repair executor
**Then** at least twenty repair-hint cases are sent through the bounded local
LLM prompt, accepted repairs are counted only after semantic validation, and
the artifact records the actual model used and post-repair pass rates.

**Given** no mandated local SOTA GGUF resolves from the cache
**When** Exp 1414 runs
**Then** the artifact is written with `status="blocked"`, zero headline repair
successes, cache diagnostic details, and an honest verdict naming local model
cache unavailability.

**Spec traces:** REQ-VERIFY-1414, SCENARIO-VERIFY-1414

### REQ-VERIFY-1419: Full-Scale Pipeline V3 Repair Executor Rerun

The repository shall provide an Exp 1419 runner that reads the 200-case Exp
1397 full-scale pipeline artifact, confirms
`results/experiment_1414_certificate_llm_repair_executor_v1.json` has
`repair_executor_deployed=true`, resolves the mandated SOTA GGUF repair model
through the local cache, and reruns the final pipeline accounting with the Exp
1414 opt-in repair executor enabled for every Exp 1397 repair-hint case.

The runner MUST write
`results/experiment_1419_fullscale_pipeline_v3_repair_executor.json` with
`status="in_progress"` before loading source artifacts or models. If Exp 1414
is not deployed, fewer than 200 Exp 1397 cases are available, no mandated SOTA
GGUF resolves, or the local model/GPU runtime cannot load, the terminal artifact
MUST use `status="blocked"` and include an exact blocker. Otherwise the
terminal artifact SHALL include at least `status`, `model_specs`,
`cases_evaluated`, `certificate_parse_rate`,
`semantic_validation_pass_rate`, `repair_hint_cases_total`,
`repaired_cases_successful`, `repair_success_rate`,
`full_pipeline_pass_rate`, `full_pipeline_headline_gate_met`, and
`honest_verdict`.

`full_pipeline_pass_rate` SHALL count original Exp 1397 scheduler passes plus
repair-hint cases accepted by the Exp 1414 executor after semantic validation,
without double-counting any case. The artifact SHALL compare the final rate
against the Exp 1397 baseline of `0.305` and the headline target of `0.40`.

**Acceptance criteria:**
- The runner writes an in-progress artifact first and a terminal artifact last.
- The Exp 1414 deployment gate blocks the run when
  `repair_executor_deployed` is not true.
- At least 200 source cases are required for a non-blocked terminal artifact.
- The final pass rate, repair success rate, Exp 1397 delta, and headline gate
  are computed from audited per-case scheduler and repair-executor rows.

### SCENARIO-VERIFY-1419: 200-Case Repair Executor Rerun Applies Headline Gate

**Given** Exp 1397 contains 200 evaluated cases and repair-hint rows
**And** Exp 1414 deployed the bounded local LLM repair executor
**And** a mandated local SOTA GGUF resolves from the cache
**When** Exp 1419 runs with the repair executor enabled
**Then** the terminal artifact records parse, semantic validation, repair, and
final full-pipeline pass rates separately, records the actual model used, and
sets `full_pipeline_headline_gate_met=true` exactly when
`full_pipeline_pass_rate >= 0.40`.

**Spec traces:** REQ-VERIFY-1419, SCENARIO-VERIFY-1419

### REQ-VERIFY-1427: Repair Executor Rejection Ledger And V2 Acceptance Contract

The repository shall provide a deterministic rejection-ledger builder for Exp
1427 that reads the Exp 1414 and Exp 1419 repair-executor artifacts, records one
ledger entry for every rejected repair candidate, and classifies each rejection
into an auditable reason class. The reason taxonomy MUST distinguish schema
failures, semantic failures, validator mismatches, prompt noncompliance, missing
outputs, and timeout classes, even when a class has zero observed examples.

When per-case raw model outputs, prompts, logs, or validator transcripts are
missing from the source artifacts, the ledger MUST reconstruct the
highest-confidence rejection reason from available aggregate and per-case fields
and record the missing evidence explicitly instead of inventing unsupported
detail.

The Exp 1427 artifact MUST include `status`, `rejection_ledger_path`,
`rejection_ledger_complete`, `cases_analyzed`, `top_rejection_reason`,
`rejection_reason_counts`, `repair_v2_contract_ready`,
`nonzero_repair_gate_required`, and `honest_verdict`. The repair v2 contract
MUST require schema validity before semantic validation, preserve rejection
reasons for every candidate, and gate any downstream scale run on a measured
nonzero validated repair success rate.

**Acceptance criteria:**
- Exp 1427 writes an in-progress artifact before source artifact analysis and a
  complete terminal artifact after the ledger and contract are written.
- The terminal artifact reports complete counts for observed rejection reasons
  and keeps zero-count taxonomy classes visible for repair v2 design.
- `repair_v2_contract_ready=true` only when the contract requires schema
  validation before semantic validation and per-candidate rejection logging.
- `nonzero_repair_gate_required=true` prevents another Exp 1419-style scale run
  without a positive repair-success gate.

### SCENARIO-VERIFY-1427: Zero-Repair Failures Produce A Complete Rejection Ledger

**Given** Exp 1414 and Exp 1419 both reached the local repair execution path
**And** every attempted repair candidate was rejected
**When** Exp 1427 builds the rejection ledger
**Then** every rejected candidate contributes exactly one rejection reason, the
top rejection reason is quantified, missing raw-output evidence is recorded, and
the repair v2 acceptance contract is marked ready only with schema-first
validation and nonzero-repair scale gating.

**Spec traces:** REQ-VERIFY-1427, SCENARIO-VERIFY-1427

### REQ-VERIFY-1428: DCCD Schema-Constrained Repair Executor V2

The repository shall provide a draft-conditioned, schema-constrained repair
executor v2 for Exp 1428. The executor MUST build a bounded local-model prompt
from the original repair request and the current draft certificate, and its
allowed output schema MUST separate `draft_certificate`, `repair_action`,
`final_certificate`, and `validator_metadata`.

The executor MUST validate the bounded repair schema before semantic
validation. Candidates that are not valid JSON, omit required schema sections,
include disallowed fields, exceed configured field limits, time out, or fail
the semantic validator MUST be rejected with exactly one recorded rejection
reason. The semantic validator MUST NOT be called for schema-invalid
candidates. Accepted repairs MUST pass both schema validation and the existing
semantic repair acceptance contract: `constraint_passed=true`,
`semantic_result="SAT"`, `repair_required=false`, and
`false_acceptance=false`.

The Exp 1428 terminal artifact at
`results/experiment_1428_dccd_schema_constrained_repair_v2.json` MUST include
`status`, `model_specs`, `local_sota_model_used`,
`repair_executor_v2_deployed`, `repair_hint_cases_tested`,
`repaired_cases_successful`, `repaired_case_success_rate`,
`schema_valid_rate`, `semantic_acceptance_rate`, `rejection_reason_counts`,
`tests_run`, and `honest_verdict`. The runner MUST resolve the mandated local
SOTA GGUF model specs through `cached_sota_pair()` or an equivalent local cache
resolver and record cache/runtime blockers instead of simulating headline local
model results. When repair-hint cases are available, the bounded validation
sample SHOULD test at least twenty cases before any non-blocked terminal
artifact is written.

**Decentralization implications:** The core executor depends on generator and
validator callables only; it does not import a closed-weight vendor SDK and
keeps local open-weight GGUF resolution in the reporting layer.

**Acceptance criteria:**
- The v2 parser accepts only the bounded DCCD repair schema and rejects
  malformed or nonconforming candidates before validator handoff.
- Rejection classification records one reason for every rejected candidate,
  including schema and semantic rejection classes.
- The validator handoff receives the schema-valid final certificate and is
  skipped for schema-invalid candidates.
- The terminal Exp 1428 artifact records model-cache diagnostics and all
  required artifact fields with `status="complete"` or `status="blocked"`.

### SCENARIO-VERIFY-1428: Schema-First DCCD Repair Validates A Bounded Micro Sample

**Given** Exp 1397 exposes repair-hint cases and a mandated SOTA GGUF cache
resolver returns the local model metadata
**When** Exp 1428 runs a bounded DCCD schema-constrained repair sample
**Then** every candidate is schema-validated before semantic validation, every
rejected candidate contributes exactly one rejection reason, accepted repairs
are counted only after semantic validation passes, and the terminal artifact
records schema-valid and semantic-acceptance rates for the repair-hint sample.

**Spec traces:** REQ-VERIFY-1428, SCENARIO-VERIFY-1428

### REQ-VERIFY-1464: Repair Retry Validation Error Context A/B

The repository shall provide a bounded Exp 1464 A/B evaluator that compares
DCCD repair retries without validator-error retry context against retries that
include the exact concrete validation error from the failed candidate. The
baseline retry MUST preserve the current path by including the failed model
output without adding the concrete validation error message. The context retry
MUST include both the failed model output and the exact schema or semantic
validation error message produced for that failed candidate.

Both variants MUST evaluate the same FoVer repair-hint cases drawn from the
same Exp 1397-style subset used by the repair-executor lineage when available.
Headline evidence MUST resolve mandated local SOTA GGUF model metadata through
`cached_sota_pair()` or an equivalent helper following
`scripts/experiment_template.py`, and MUST record model path, GPU assignment,
and whether live local SOTA inference actually ran. Legacy small-model or
injected generators may be used only for tests and CPU smoke paths, and must not
be labelled as headline evidence.

The terminal Exp 1464 artifact at
`results/experiment_1464_repair_validation_error_context_ab.json` MUST include
`status`, `model_specs`, `live_sota_model_inference_used`,
`validation_error_context_enabled`, `cases_evaluated`,
`baseline_acceptance_rate`, `context_acceptance_rate`,
`acceptance_delta_pp`, `schema_validity_delta_pp`,
`semantic_correctness_delta_pp`, `spilled_energy_diagnostic_available`,
`repair_executor_lineage_preserved`, `repair_executor_lineage_retired`,
`commands_run`, and `honest_verdict`. If no acceptance, schema-validity,
semantic-correctness, or false-acceptance metric improves for the context
variant, the artifact MUST set `repair_executor_lineage_retired=true`.

**Decentralization implications:** The evaluator depends on local open-weight
GGUF model metadata and generator callables; it does not add closed-weight SDK
dependencies to `python/carnot/pipeline/`.

**Acceptance criteria:**
- The retry prompt helper can build both the baseline retry prompt and the
  validation-error-context retry prompt without changing the original DCCD
  prompt path.
- The A/B evaluator uses identical case IDs for baseline and context retries
  and records per-case baseline/context outcomes.
- The artifact computes acceptance, schema-validity, semantic-correctness, and
  false-acceptance deltas from the audited per-case rows.
- The lineage-preserved/retired decision follows the metric-improvement rule
  and records live SOTA evidence honestly.

### SCENARIO-VERIFY-1464: Validation Error Context Either Salvages Or Retires Repair Executor

**Given** Exp 1397 repair-hint rows are available
**And** Exp 1463 has shown that a mandated local SOTA GGUF runtime can complete
live inference
**When** Exp 1464 runs baseline and validation-error-context retries on the same
bounded FoVer repair subset
**Then** the terminal artifact records the exact model path and GPU used,
reports the three required deltas in percentage points, and preserves the
repair-executor lineage only when at least one acceptance, schema-validity,
semantic-correctness, or false-acceptance metric improves.

**Spec traces:** REQ-VERIFY-1464, SCENARIO-VERIFY-1464

### REQ-VERIFY-1429: MCMC Constrained Repair Candidate Search

The repository shall provide a bounded constrained candidate-search layer around
the Exp 1428 repair executor v2 for Exp 1429. The search layer MUST propose more
than one DCCD schema-shaped repair candidate per repair-hint case, validate each
candidate with the existing schema-first repair v2 parser, skip semantic
validation for schema-invalid candidates, and accept candidates only when the
semantic repair acceptance contract passes.

The search layer MUST score schema-valid candidates with verifier or energy
signals and select the lowest-energy accepted candidate as the best-of-N repair.
It MUST report one-candidate success from the first proposal separately from
best-of-N success across the bounded proposal set, and it MUST compute an MCMC
acceptance rate as accepted schema-valid semantic repairs divided by total
candidate proposals evaluated.

The Exp 1429 terminal artifact at
`results/experiment_1429_mcmc_constrained_repair_candidate_search.json` MUST
include `status`, `model_specs`, `candidate_search_complete`,
`cases_evaluated`, `candidates_per_case`, `mcmc_acceptance_rate`,
`repair_success_rate_one_candidate`, `repair_success_rate_best_of_n`,
`energy_rerank_improved`, `local_sota_model_used`, and `honest_verdict`.
The runner MUST first confirm that
`results/experiment_1428_dccd_schema_constrained_repair_v2.json` has
`repair_executor_v2_deployed=true`; if not, it MUST write a blocked artifact
with `candidate_search_complete=false`. It MUST resolve mandated local SOTA GGUF
model metadata through `cached_sota_pair()` or an equivalent local cache
resolver, and any prototype or smoke-test execution mode MUST be labelled so it
cannot be mistaken for headline local-SOTA candidate-search evidence.

**Decentralization implications:** The core search layer depends only on local
generator, validator, and energy-scorer callables; it does not import
closed-weight vendor SDKs and keeps GGUF cache resolution in the reporting
layer.

**Acceptance criteria:**
- Candidate search evaluates a bounded number of proposals per case and records
  one audit row per proposal.
- Schema-invalid candidates do not reach semantic validation.
- Best-of-N success can improve over one-candidate success only through a later
  schema-valid semantic-valid candidate selected by verifier/energy scoring.
- The terminal Exp 1429 artifact records model-cache/runtime blockers honestly
  and includes all required fields with `status="complete"` or
  `status="blocked"`.

### SCENARIO-VERIFY-1429: Best-Of-N Constrained Search Beats A Failed First Candidate

**Given** Exp 1428 has deployed repair executor v2
**And** a repair-hint case has a bounded proposal budget greater than one
**When** the first DCCD candidate fails semantic validation but a later
schema-valid candidate passes the semantic acceptance contract
**Then** one-candidate success is false for that case, best-of-N success is true,
the accepted proposal contributes to MCMC acceptance rate, and the selected
candidate is the lowest-energy accepted repair.

**Spec traces:** REQ-VERIFY-1429, SCENARIO-VERIFY-1429

### REQ-VERIFY-1430: PRM-Guided Repair Candidate Selector

The repository shall provide a deterministic PRM-guided selector for Exp 1430
that consumes the bounded repair candidate pool produced by Exp 1429, ranks
each case's repair candidates before semantic acceptance labels are consulted,
and then measures whether the selected top-ranked candidate is semantically
accepted. The selector MUST prefer a trained PRM v1 artifact and checkpoint
from Exp 1423 when available. If the trained checkpoint is unavailable, the
selector MAY use a deterministic proxy scorer only as a non-headline fallback
and MUST record that fallback in the terminal artifact and honest verdict.

The Exp 1430 runner MUST first write
`results/experiment_1430_prm_guided_repair_selector.json` with
`status="in_progress"`. It MUST then confirm that Exp 1429 contains a non-empty
candidate pool. If no candidate pool exists, it MUST write a blocked artifact
with `prm_guided_selection_ready=false`. Complete artifacts MUST report
`status`, `prm_guided_selection_ready`, `cases_evaluated`, `selector_auroc`,
`raw_best_of_n_repair_success_rate`, `selected_repair_success_rate`,
`selection_improvement_pp`, `prmv1_artifact_used`, and `honest_verdict`.

**Decentralization implications:** The selector consumes local artifacts and a
CPU checkpoint only; it performs no closed-weight model calls and does not
import vendor SDKs in the core pipeline.

**Acceptance criteria:**
- The selector ranks each case's candidates before inspecting acceptance labels.
- The selector AUROC is computed from candidate acceptance labels only after
  scores are frozen.
- Raw best-of-N repair success from Exp 1429 is reported separately from the
  selected-candidate repair success rate.
- Missing Exp 1429 candidate pools produce a blocked artifact with all required
  fields and `prm_guided_selection_ready=false`.
- Complete artifacts distinguish trained PRM v1 scoring from deterministic
  proxy fallback scoring.

### SCENARIO-VERIFY-1430: PRM Ranking Selects A Later Accepted Repair Candidate

**Given** Exp 1429 provides a bounded candidate pool where the first candidate
fails semantic validation and a later candidate succeeds
**And** Exp 1423 provides a trained PRM v1 checkpoint
**When** Exp 1430 scores candidates before semantic validation
**Then** the selected candidate is the highest PRM-scored candidate, selected
repair success is computed from the frozen selection, selector AUROC is in
`[0, 1]`, and the artifact reports the selection improvement in percentage
points relative to Exp 1429 raw best-of-N success.

**Spec traces:** REQ-VERIFY-1430, SCENARIO-VERIFY-1430

### REQ-VERIFY-1448: PRM V3 Online Process-Reward Repair Agent

The repository shall provide a deterministic PRM v3 online process-reward repair
agent for Exp 1448 that consumes the Exp 1429 bounded repair candidate pool,
the Exp 1430 PRM v1 selector artifact, and the Exp 1434 PRM v2 label-completion
artifact. The agent MUST score intermediate repair or reasoning steps within
each candidate before consulting final semantic acceptance labels, aggregate
those step scores into a candidate score, and compare the frozen PRM v3
selection behavior against raw best-of-N and the Exp 1430 PRM v1 selection.

The Exp 1448 runner MUST first write
`results/experiment_1448_prm_v3_online_process_reward_agent.json` with
`status="in_progress"`. Complete artifacts MUST report `status`,
`pra_selector_ready`, `prm_v2_labels_used`, `traces_evaluated`,
`step_scores_generated`, `selection_improvement_pp`,
`false_acceptance_rate_delta`, `regression_against_prm_v1`, `commands_run`, and
`honest_verdict`. The honest verdict MUST avoid an improvement claim when PRM
v3 only ties raw best-of-N, regresses against PRM v1, or worsens semantic false
acceptance. Missing or incomplete PRM v2 labels MUST produce a blocked artifact
with `pra_selector_ready=false`.

**Decentralization implications:** The PRM v3 agent consumes local artifacts and
CPU checkpoints only; it performs no closed-weight model calls and does not
import vendor SDKs in the core pipeline.

**Acceptance criteria:**
- Candidate scoring emits one or more step-level score rows per candidate when
  candidate process text is available.
- Step-score aggregation is frozen before final candidate acceptance labels are
  used for selection metrics.
- Complete artifacts compare raw best-of-N, PRM v1, and PRM v3 selected repair
  success rates.
- Complete artifacts report false-acceptance deltas and an explicit
  `regression_against_prm_v1` boolean.
- Missing or incomplete PRM v2 label artifacts produce a blocked artifact with
  all required fields and `pra_selector_ready=false`.

### SCENARIO-VERIFY-1448: Online Step Scores Prefer A Better Repair Trace

**Given** Exp 1429 provides a bounded candidate pool with multiple repair
candidates and Exp 1434 provides a trained PRM v2 checkpoint
**When** Exp 1448 scores candidate repair steps before consulting final semantic
acceptance labels
**Then** the PRM v3 selector chooses the candidate with the highest aggregated
step score, reports the number of traces and step scores evaluated, computes
selection improvement relative to raw best-of-N, and records whether the result
regresses against Exp 1430 PRM v1 selection.

**Spec traces:** REQ-VERIFY-1448, SCENARIO-VERIFY-1448

### REQ-VERIFY-1431: Full-Scale Pipeline V4 Micro-Gated Validation

The repository shall provide a bounded Exp 1431 full-pipeline v4 validation
runner that writes
`results/experiment_1431_fullscale_pipeline_v4_micro_gated.json` with
`status="in_progress"` before loading source artifacts. The runner MUST confirm
the structured gates from Exp 1428 and Exp 1430 before evaluating cases:
Exp 1428 must be complete, deploy repair executor v2, and report a nonzero
validated repair success rate; Exp 1430 must be complete and set
`prm_guided_selection_ready=true`.

When gates are satisfied, the runner SHALL evaluate exactly 50 cases from the
Exp 1397/1419 200-case source using a deterministic sample that is not the
original Exp 1419 order, unless no source rows are available. The artifact MUST
record the sample source and local invocation flags showing that repair v2 and
PRM-guided selection were enabled. Repair credit SHALL be counted only for
sampled repair-hint cases selected and accepted by the frozen Exp 1430
PRM-guided selector, without double-counting original full-pipeline passes.

The terminal artifact MUST include at least `status`, `model_specs`,
`local_sota_model_used`, `cases_evaluated`, `certificate_parse_rate`,
`semantic_validation_pass_rate`, `repair_hint_cases_total`,
`repair_success_rate`, `full_pipeline_pass_rate`,
`beats_exp1419_baseline`, `eligible_for_200_case_scaleup`, and
`honest_verdict`. `beats_exp1419_baseline` SHALL compare the measured
50-case rate against the Exp 1419 baseline of `0.305`.
`eligible_for_200_case_scaleup` MAY be true only when the structured gates are
satisfied, `repair_success_rate > 0`, `full_pipeline_pass_rate > 0.305`, and
the runtime evidence is not labeled as prototype or smoke-test-only.

**Acceptance criteria:**
- The runner writes an in-progress artifact first and a terminal complete or
  blocked artifact last.
- Missing or failed Exp 1428/1430 gates produce a blocked artifact with
  `eligible_for_200_case_scaleup=false`.
- The sampled 50-case order differs from the Exp 1419 source order and the
  sample source is recorded.
- Full-pipeline pass rate, repair success rate, Exp 1419 baseline comparison,
  and scale-up eligibility are computed from audited source and selector rows.

### SCENARIO-VERIFY-1431: Micro Gate Applies Repair V2 And PRM Selection

**Given** Exp 1397 exposes 200 source scheduler rows with repair-hint cases
**And** Exp 1428 reports nonzero validated repair executor v2 acceptance
**And** Exp 1430 reports PRM-guided selection readiness with accepted selected
repair candidates
**When** Exp 1431 runs the 50-case micro-gated validation
**Then** the terminal artifact records 50 evaluated cases, counts only selected
accepted repairs for repair-hint rows in the sample, compares against the
Exp 1419 baseline `0.305`, and marks 200-case scale-up eligibility only when
the positive repair and baseline gates are met without prototype-only runtime
evidence.

**Spec traces:** REQ-VERIFY-1431, SCENARIO-VERIFY-1431

### REQ-VERIFY-1408: Structured Verdict Record Schema

The pipeline MUST expose a documented `VerdictRecord` dataclass for downstream
verification consumers that need more than a boolean pass/fail result.  The
record SHALL include at least `verdict`, `energy`, `calibrated_confidence`,
`producing_tier`, `tier_reached`, `rationale`, `budget_ms_consumed`,
`repairs_applied`, and `extras`.

`verdict` SHALL be one of `"pass"`, `"fail"`, or `"abstain"`.  Energy SHALL be
stored as a float.  `calibrated_confidence` SHALL be clamped to `[0.0, 1.0]`.
Tier fields SHALL be integer tier identifiers, where higher numbers indicate a
deeper verifier tier.  The record SHALL serialize to a JSON-compatible dict
without requiring callers to inspect pipeline internals.

**Acceptance criteria:**
- `VerdictRecord.to_dict()` returns every required field with JSON-compatible
  values.
- Invalid verdict strings are rejected.
- Confidence values outside `[0.0, 1.0]` are clamped rather than leaked.
- `repairs_applied` and `extras` default to empty containers.

### REQ-VERIFY-1409: Energy-To-Confidence Calibration Helper

The pipeline MUST provide a deterministic post-hoc calibration helper for
structured verdicts.  The helper SHALL map raw energy to a probability-like
confidence in `[0.0, 1.0]`, monotonic in negative energy: lower energy MUST
produce confidence greater than or equal to higher energy under the same
threshold and temperature.

The helper SHALL be documented as a fallback calibration surface that can be
replaced by held-out isotonic or Platt calibration parameters.  It SHALL handle
NaN and infinities deterministically so verdict records remain serializable.

**Acceptance criteria:**
- Confidence monotonically decreases as energy increases.
- NaN energy returns `0.0` confidence.
- Negative infinity maps to `1.0`; positive infinity maps to `0.0`.
- Non-positive temperatures are rejected.

### REQ-VERIFY-1410: Structured Verdict API With Legacy Compatibility

`VerifyRepairPipeline` and `ThreeTierPipeline` MUST expose a structured verdict
API that returns `VerdictRecord` while preserving existing legacy callers.
`verify_record(...)` SHALL return the structured record.  Existing `verify(...)`
callers SHALL keep their current return shape, and `verify_legacy(...)` SHALL be
available as an explicit compatibility alias.

For `VerifyRepairPipeline`, the record SHALL be derived from the existing
`VerificationResult` and include the raw energy, pass/fail verdict, a rationale
derived from violations or certificate errors, and certificate details in
`extras`.  For `ThreeTierPipeline`, the record SHALL derive from the deciding
tier returned by the legacy cascade and SHALL include `tier_used` in `extras`.

**Acceptance criteria:**
- `VerifyRepairPipeline.verify_record()` returns a `VerdictRecord` without
  changing `VerifyRepairPipeline.verify()` callers.
- `ThreeTierPipeline.verify_record()` returns a `VerdictRecord` without changing
  the legacy `(verified, tier_used, energy)` tuple from `verify()`.
- `verify_legacy()` delegates to the unchanged legacy return path.
- Structured records include elapsed budget milliseconds.

### SCENARIO-VERIFY-1408: Structured Verdicts Preserve Legacy Semantics

**Given** a correct arithmetic response and an incorrect arithmetic response,
**When** `VerifyRepairPipeline.verify_record()` is called,
**Then** the correct response returns `verdict="pass"` and the incorrect
response returns `verdict="fail"`,
**And** both records include energy, calibrated confidence, tier identifiers,
elapsed budget milliseconds, a rationale, and JSON-compatible extras.

**Given** a `ThreeTierPipeline` configured to fall through to an Ising stub,
**When** `verify_record()` and legacy `verify()` are called on the same response,
**Then** the structured record preserves the same verified outcome, tier name,
and energy while the legacy tuple shape remains unchanged.

**Spec traces:** REQ-VERIFY-1408, REQ-VERIFY-1409, REQ-VERIFY-1410

### REQ-VERIFY-1411: Streaming Verification Async Iterator

The pipeline MUST expose a `verify_stream(...)` Python API that accepts a pool of
candidate responses and returns an async iterator of `VerdictRecord` objects as
candidate verification completes.  Each emitted record SHALL include the
candidate identifier in `extras["candidate_id"]`, a zero-based completion index
in `extras["stream_index"]`, and a provisional energy rank in
`extras["stream_rank"]`.

The API MUST accept candidates with `id`, `question`, `answer` or `response`, and
optional per-candidate `domain`.  The API SHALL support bounded concurrency via
`max_concurrency`, optional total and per-candidate millisecond budgets, and a
default `VerifyRepairPipeline(model=None)` when the caller does not provide a
pipeline instance.

**Acceptance criteria:**
- The async iterator emits records in completion order, not input order.
- Each record serializes through `VerdictRecord.to_dict()` with the stream
  metadata intact.
- Invalid `top_k`, budget, or concurrency parameters are rejected.
- Synchronous pipeline implementations are offloaded without blocking the event
  loop, while async pipeline implementations are awaited directly.

### REQ-VERIFY-1412: Streaming Top-K Early Stop And Cancellation

`verify_stream(...)` MUST support `top_k` and `early_stop_margin` controls over
candidate pools.  When at least `top_k + 1` candidates have completed and the
energy margin between the current kth and kth-plus-one candidate is greater than
or equal to `early_stop_margin`, the stream SHALL stop scheduling new work,
cancel pending workers, and annotate the final emitted record with
`extras["stream_end"]`.

`extras["stream_end"]` SHALL include at least `event="stream_end"`,
`stopped_early`, `stop_reason`, `total_candidates`, `emitted_count`,
`scored_count`, `residual_candidates_unscored`, `top_k`, and
`early_stop_margin`.  When the consumer closes the async iterator before the
pool is exhausted, the implementation MUST cancel outstanding worker tasks
rather than leaking background verification work.

**Acceptance criteria:**
- A synthetic candidate pool with a decisive top-1 margin stops before scoring
  every candidate.
- Pending async workers observe cancellation when the consumer closes the
  iterator early.
- The stream-end annotation reports the correct residual unscored candidate
  count.

### REQ-VERIFY-1413: MCP Streaming Verification Event Surface

The MCP server MUST expose a `verify_stream` tool over the same candidate schema
as the Python API.  Because the current stdio MCP server handlers return a single
JSON value, the tool SHALL return a streaming-compatible event payload with
`events` containing ordered verdict events and `stream_end` containing the final
summary.  Each verdict event SHALL carry the serialized `VerdictRecord`.

The MCP tool SHALL preserve existing MCP safeguards: string input size checks,
structured error responses through `_guarded_call`, and no model loading beyond
`VerifyRepairPipeline(model=None)`.

**Acceptance criteria:**
- `health_check()` includes `verify_stream` in the tool list.
- The MCP `verify_stream` handler returns ordered verdict events plus one
  `stream_end` summary.
- `top_k` and `early_stop_margin` parameters are passed through to the Python
  streaming primitive.

### SCENARIO-VERIFY-1411: Streaming Verification Emits Early Top Candidate

**Given** three candidate responses with deterministic synthetic verifier
energies,
**When** `verify_stream(..., max_concurrency=2)` runs,
**Then** verdict records are emitted as each worker completes,
**And** the emitted records include candidate IDs and stream indexes.

**Given** the same pool with `top_k=1`, `early_stop_margin=1.0`, and
`max_concurrency=1`,
**When** the first two completed energies have a decisive margin,
**Then** the stream stops before scoring the third candidate,
**And** the final record includes a `stream_end` annotation with
`stopped_early=true`.

**Spec traces:** REQ-VERIFY-1411, REQ-VERIFY-1412, REQ-VERIFY-1413

### REQ-VERIFY-1414: Probability Calibration Verifier

The pipeline MUST provide a `ProbabilityCalibrationVerifier` side-car for
responses that make explicit probability claims.  The verifier SHALL expose
`score(chain, probability_claim) -> VerdictRecord`, where `probability_claim`
may be a parsed claim object or a string such as `P(event)=0.62`.

The verifier MUST extract simple reference-class evidence atoms from the
reasoning chain, including `n out of N`, `n/N`, percentages, and explicit base
rates.  It SHALL derive an implied posterior point estimate and tolerance range,
then score energy as the distance between the claimed probability and the
nearest edge of that implied range.  Claims inside the range SHALL pass with
zero energy.  Claims outside the range SHALL fail with positive energy.  Claims
without parseable probability or evidence SHALL abstain rather than guess.

**Acceptance criteria:**
- `score(...)` returns a `VerdictRecord` with claim probability, implied
  probability, implied range, and evidence count in `extras`.
- Synthetic in-range probability claims pass with zero energy.
- Synthetic overconfident or underconfident claims fail with positive energy.
- Underdetermined chains abstain without raising.

### REQ-VERIFY-1415: Opt-In Pipeline Probability Calibration Wire-Up

`VerifyRepairPipeline` MUST accept an optional probability-calibration verifier
without changing default verification behavior.  When supplied, the verifier
SHALL scan the response for explicit probability claims and add metadata-backed
constraints for each scored claim.  Failing probability-calibration records MUST
make `verify(...)` return `verified=false`; abstentions SHALL be recorded as
informational and MUST NOT create a violation.

Probability-calibration constraints SHALL include the underlying
`VerdictRecord.to_dict()` payload in metadata and contribute their positive
energy to the returned `VerificationResult.energy` certificate.

**Acceptance criteria:**
- Default `VerifyRepairPipeline(...)` behavior is unchanged when no probability
  verifier is supplied.
- Supplying `ProbabilityCalibrationVerifier(...)` makes miscalibrated
  probability claims appear as `probability_calibration` violations.
- Pipeline result energy includes the probability-calibration gap energy.

### SCENARIO-VERIFY-1414: Probability Claim Calibration Detects Reference-Class Gap

**Given** a reasoning chain that says 30 out of 100 comparable cases had an
event,
**When** the response claims `P(event)=0.30`,
**Then** `ProbabilityCalibrationVerifier.score(...)` returns
`verdict="pass"` and `energy=0.0`.

**Given** the same evidence but a response claims `P(event)=0.80`,
**When** `VerifyRepairPipeline` is constructed with the probability-calibration
verifier enabled,
**Then** verification fails with a `probability_calibration` violation and a
positive energy gap.

**Spec traces:** REQ-VERIFY-1414, REQ-VERIFY-1415

### REQ-PIPELINE-1615: ETS Decoder (Energy-Guided Decoding with Monte Carlo Estimation)

The pipeline MUST provide an ETSDecoder that implements Energy-Guided Decoding using
online Monte Carlo estimation of transition probabilities to dynamically scale compute at test-time.
It MUST weight the base LLM policy probabilities with an EBM score using Monte Carlo samples.

**Acceptance criteria:**
- `python/carnot/pipeline/ets_decoder.py` exposes `ETSDecoder`.
- Implements Monte Carlo transition probability formulation.
- `decode()` method uses candidate generation, Monte Carlo evaluation of the energy, and
  returns the selected sequence or next token.
- Returns a JSON artifact to `results/experiment_1615_ets_decoding.json` containing metrics
  about the decode step (e.g., number of MC samples used, energy values).

### SCENARIO-PIPELINE-1615: ETS Decoder Reduces Energy
**Given** a set of candidate tokens with base LLM probabilities
**When** `ETSDecoder.decode()` is run with a mock energy function and Monte Carlo sampling
**Then** the selected token maximizes the combined probability `p_llm * exp(-beta * E_mc)`.

### REQ-PIPELINE-1625: Entropy-Based Task Router

The pipeline MUST provide a task router that uses prompt entropy as a heuristic to classify and route queries. Logic/math questions MUST be routed to the EBM verifier, and general QA MUST be routed to the base LLM.

**Rationale:** Logic and math prompts often have different entropy profiles compared to general open-ended QA. Using prompt entropy provides a fast, zero-shot heuristic to decide whether the expensive constraint-based EBM verifier is needed.

**Acceptance criteria:**
- `python/carnot/pipeline/task_router.py` exposes `EntropyTaskRouter` with a `route(prompt)` method.
- The router correctly routes low-entropy (or high-entropy depending on the heuristic) math/logic questions to `"ebm_verifier"` and general QA to `"base_llm"`.
- A test runs on a mixed dataset of GSM8K (math) and OpenAssistant (QA) examples.
- The experiment artifact is saved to `results/experiment_1625_task_router.json` with metrics like accuracy, threshold, etc.

### SCENARIO-PIPELINE-1625: Entropy Router Routes GSM8K to EBM Verifier

**Given** a math question from GSM8K and a general QA question from OpenAssistant
**When** `EntropyTaskRouter.route(prompt)` is called
**Then** the math question is routed to `"ebm_verifier"` and the QA question is routed to `"base_llm"`.

### REQ-PIPELINE-1670: Energy-Guided Decoding (EGD) for SOTA Models

The pipeline MUST provide an Energy-Guided Decoding wrapper for SOTA models (e.g. `unsloth/gemma-4-31B-it-GGUF`).
It MUST apply EGD selection across inference calls.
It MUST be tested on a bounded dataset to evaluate the hallucination "Yes-ratio" bias.

**Acceptance criteria:**
- `python/carnot/pipeline/energy_guided_decoding.py` exposes `EGDWrapper` or similar.
- Wraps inference calls applying EGD selection.
- Tested on a bounded dataset evaluating the hallucination "Yes-ratio" bias.
- Writes an experiment artifact to `results/experiment_1670_egd.json`.

### SCENARIO-PIPELINE-1670: EGD Wrapper Evaluates Hallucination Bias

**Given** inference calls to a SOTA model
**When** wrapped with EGD selection
**Then** it evaluates hallucination Yes-ratio bias and produces `experiment_1670_egd.json`.

### REQ-PIPELINE-5138: Exact-Validator Energy-Guided Decoding Gate

The pipeline MUST provide the `exp5138-ets-ebd-guided-decoding-v471` runner that
consumes the clean Exp 5136 structured pool, selects exact-validator task families, and writes
`results/experiment_5138_ets_ebd_guided_decoding_v471.json`. The runner MUST
hard-block unless `results/experiment_5136_receipt_structured_pool_v2_v471.json`
reports `structured_pool_v2_clean=true`, MUST preserve the three mandated local
GGUF `MODEL_SPECS`, and MUST distinguish true token-level guided decoding from
best-of-N or fixed-token reranking under matched token and validator-call
budgets.

The artifact MUST set `inference_substrate` to
`local_sota_gguf_energy_guided_decoding_or_blocked` and MUST include
`experiment_id`, `milestone`, `honest_verdict`, `inference_substrate`,
`duration_s`, `MODEL_SPECS`, `upstream_pool_artifact`,
`exact_validator_authority`, `controls_differentiated`,
`rerank_only_control`, `token_nfe_accounting`, `guided_decoding_delta`,
`delta_ci95`, `violation_rate_delta`, `logprob_or_blocker_evidence`,
`guided_decoding_ready`, `conductor_modified`, and `tests_run`. If local
stepwise logprob/top-token telemetry is unavailable, the runner MUST write a
blocked artifact rather than relabeling reranked complete candidates as guided
decoding evidence.

### SCENARIO-PIPELINE-5138: Missing Stepwise Telemetry Blocks Guided Claim

**Given** a clean Exp 5136 structured pool with exact-validator candidates
**And** the local runtime cannot expose stepwise logprob or top-token telemetry
needed to alter decoding during generation,
**When** the Exp 5138 runner evaluates matched rerank-only controls,
**Then** the terminal artifact records the strongest matched control,
`controls_differentiated=false`, `guided_decoding_ready=false`, and an
`honest_verdict` prefixed with `blocked_` instead of claiming true guided
decoding.


### REQ-PIPELINE-1677: Energy-Driven Steering (EDS) Prototype

The pipeline MUST provide an Energy-Driven Steering (EDS) prototype in `python/carnot/pipeline/energy_driven_steering.py`.
It MUST include an `ExternalEBMAdapter` that maps internal model activations to an energy landscape,
and an `EnergyDrivenSteerer` that computes gradients of this energy with respect to hidden states to steer generation.
The steerer MUST support local SOTA GGUF models (`unsloth/gemma-4-31B-it-GGUF` and `unsloth/Qwen3.6-35B-A3B-GGUF`).
The evaluation MUST write results to `results/experiment_1677_eds.json` including fields: `models_tested`, `steered_generation_success`, `energy_landscape_mapped`, and `honest_verdict`.

**Acceptance criteria:**
- `ExternalEBMAdapter(hidden_dim).compute_energy(hidden_states)` returns a scalar energy.
- `EnergyDrivenSteerer(ebm_adapter).steer(hidden_states)` returns steered hidden states by subtracting the energy gradient.
- `run_eds_evaluation()` runs a logical task evaluation and produces the required JSON artifact.

**Spec traces:** REQ-PIPELINE-1677, Exp 1677

### REQ-PIPELINE-1678: CRANE Interleaved Decoding State Machine

The pipeline MUST provide CRANE decoding in `python/carnot/pipeline/crane_decoding.py`.
The decoder MUST alternate between an unconstrained free-text reasoning phase and a
grammar-enforced structured generation phase.  Strict grammar enforcement MUST apply only
inside the structured phase, while the free-text phase remains unconstrained so the model
can preserve semantic reasoning before emitting parseable output.

The Exp 1678 evaluation MUST target the mandated SOTA model identifier
`unsloth/gemma-4-26B-A4B-it-GGUF` and write
`results/experiment_1678_crane.json` with at least `reasoning_quality_delta` and
`parse_rate`.  The CRANE run MUST be compared against a strict grammar-only baseline and
report positive semantic-coherence delta while preserving a parse rate of at least 0.9.

**Acceptance criteria:**
- `CRANEDecoder.decode()` emits a trace whose phases alternate free-text then constrained.
- Constrained phases reject malformed structured outputs and preserve parseable records.
- `evaluate_crane_decoding()` compares CRANE against a strict grammar-only baseline for
  `unsloth/gemma-4-26B-A4B-it-GGUF`.
- `run_experiment()` writes `results/experiment_1678_crane.json` containing
  `reasoning_quality_delta` and `parse_rate`.

### SCENARIO-PIPELINE-1678: CRANE Improves Coherence Without Sacrificing Parse Rate

**Given** a Gemma-4-26B-A4B-shaped backend whose unconstrained reasoning contains task
semantics that strict grammar-only decoding omits,
**When** CRANE decoding runs a free reasoning phase followed by a constrained structured
phase,
**Then** the constrained output is parseable and its semantic coherence score exceeds the
strict grammar-only baseline.

**Spec traces:** REQ-PIPELINE-1678, SCENARIO-PIPELINE-1678, Exp 1678

### REQ-PIPELINE-1694: Phase 7 Full Pipeline Stack Evaluation

The pipeline MUST execute a combined integration test of the Phase 7 full pipeline stack:
Certified KArAt, NablaETS, and FR-11 Self-Play discoverer. The evaluation MUST target the
mandated SOTA model identifier `unsloth/gemma-4-26B-A4B-it-GGUF`. It MUST evaluate 5 multi-hop
reasoning questions, utilizing all three components.

The experiment MUST write an artifact to `results/experiment_1694_full_pipeline.json`
containing at least `status`, `experiment_id`, `experiment`, `model_used`, `questions_run`,
`components_active`, and `timestamp`.

**Acceptance criteria:**
- A run function `run_experiment_1694(output_path)` executes the 5 multi-hop reasoning questions.
- The pipeline configuration combines Certified KArAt, NablaETS, and FR-11 Self-Play.
- The JSON artifact is written successfully with `questions_run == 5`.

### SCENARIO-PIPELINE-1694: Phase 7 Pipeline Runs Multi-Hop Questions

**Given** the Phase 7 pipeline configured with KArAt, NablaETS, and FR-11 Self-Play
**When** 5 multi-hop reasoning questions are evaluated against `unsloth/gemma-4-26B-A4B-it-GGUF`
**Then** the combined stack executes successfully and produces the valid artifact.

**Spec traces:** REQ-PIPELINE-1694, SCENARIO-PIPELINE-1694, Exp 1694

### REQ-PIPELINE-1767: Full E2E Pipeline with Qwen3.6-35B-A3B

The pipeline MUST provide a deterministic script `scripts/experiment_1767_e2e_qwen.py` to evaluate the flagship MoE Qwen model (`unsloth/Qwen3.6-35B-A3B-GGUF`). It MUST collect `latency`, `parse_rate`, and `energy_scores` and output to `results/experiment_1767_e2e_qwen.json`.

### SCENARIO-PIPELINE-1767: Qwen E2E Pipeline Execution

**Given** the Qwen MoE model specification
**When** `scripts/experiment_1767_e2e_qwen.py` is executed
**Then** it outputs the evaluation results to `results/experiment_1767_e2e_qwen.json`.


## REQ-LEARN-1774: Differentiable Constraint Memory Bank

**Given** a need for multi-session continual learning without forgetting
**When** the DifferentiableMemoryBank is instantiated
**Then** it MUST provide differentiable read, write, and update operations using attention mechanisms.
**And** it MUST support gradients flowing through the memory read and write operations.

### REQ-LEARN-1774 Sub-requirements

- REQ-LEARN-1774-1: `DifferentiableMemoryBank` SHALL be initialised with `memory_size` and `vector_dim`.
- REQ-LEARN-1774-2: `read(query)` SHALL use softmax attention to return a weighted sum of memory vectors.
- REQ-LEARN-1774-3: `write(key, value)` SHALL write new information into the memory bank differentially.
- REQ-LEARN-1774-4: `update(query, value)` SHALL update existing memory slots based on attention weights.

### SCENARIO-LEARN-1774: Differentiable Memory Operations
**Given** an initialized memory bank
**When** a key-value pair is written, and a similar query is read
**Then** the retrieved value MUST be close to the written value
**And** backpropagation MUST successfully compute gradients for the query.

### REQ-PIPELINE-1787: Formal Verification Orchestrator

The pipeline MUST provide a Formal Verification Orchestrator that bounds EBM exploration
with external formal solvers iteratively.

**Acceptance criteria:**
- `python/carnot/pipeline/formal_orchestrator.py` exposes `FormalOrchestrator`.
- Iteratively queries solvers (e.g., Z3) within a generation loop.
- Writes an experiment artifact to `results/experiment_1787_formal_orchestrator.json` containing metrics.

### SCENARIO-PIPELINE-1787: Orchestrator Queries Solver

**Given** a set of constraints
**When** `FormalOrchestrator.run_generation_loop()` is called
**Then** it iteratively queries the solver and outputs `results/experiment_1787_formal_orchestrator.json`.

### REQ-PIPELINE-1788: NRGPT Explorer

The pipeline MUST provide an NRGPT-style explorer that uses energy-guided test-time compute scaling to improve logic generation.

**Acceptance criteria:**
- `python/carnot/inference/nrgpt_explorer.py` exposes `NRGPTExplorer`.
- Implements energy-guided test-time compute scaling.
- Writes an experiment artifact to `results/experiment_1788_nrgpt_exploration.json` containing metrics.

### SCENARIO-PIPELINE-1788: NRGPT Exploration

**Given** an energy function
**When** `NRGPTExplorer.explore()` is called
**Then** it scales compute based on energy guidance and outputs `results/experiment_1788_nrgpt_exploration.json`.

### REQ-PIPELINE-1797: Formal Orchestrator Adversarial Audit

The pipeline MUST ensure the Formal Orchestrator rejects mathematically invalid or contradictory proofs and achieves zero false accepts.

**Acceptance criteria:**
- `scripts/experiment_1797_orchestrator_audit.py` injects known contradictory proofs.
- The experiment confirms the orchestrator never accepts invalid proofs.
- Writes an experiment artifact to `results/experiment_1797_orchestrator_audit.json` containing metrics.

### SCENARIO-PIPELINE-1797: Adversarial Proof Injection

**Given** the `FormalOrchestrator`
**When** contradictory or unsatisfiable constraints are injected
**Then** it MUST NOT return success.
**And** it outputs `results/experiment_1797_orchestrator_audit.json`.

### REQ-EVAL-1823: Phase 18 Final Evaluation Run

The pipeline MUST combine Phase 18 advances into a final evaluation run. It MUST execute the MoE-distilled model with the KAN verifier active on 100 GSM8K problems.

**Acceptance criteria:**
- Uses models `["unsloth/Qwen3.6-35B-A3B-GGUF", "unsloth/gemma-4-31B-it-GGUF"]`.
- The evaluation script `python/carnot/eval/phase18_final_eval.py` exposes a function to run the evaluation.
- Records final accuracy, latency, and self-learning delta.
- Writes to `results/experiment_1823_final_eval.json`.

### SCENARIO-EVAL-1823: MoE and KAN integration on GSM8K

**Given** the complete system with MoE-distilled model and KAN verifier
**When** 100 GSM8K problems are evaluated
**Then** the script completes successfully and outputs `results/experiment_1823_final_eval.json`.


### REQ-PIPELINE-1826: Fail-Fast Doomed Reruns
The pipeline API MUST provide a fail-fast check for doomed reruns at activation time.
It MUST write a terminal artifact with `status="blocked"` and `honest_verdict="blocked_doomed_rerun"` when a task is doomed.

### SCENARIO-PIPELINE-1826: Fail-Fast Artifact Generation
**Given** a doomed task definition
**When** the pipeline fail-fast check is invoked
**Then** it writes a blocked artifact and returns True.

### REQ-PIPELINE-1831: Constrained Online Convex Optimization with Memory (COCOM)
The pipeline MUST implement Constrained Online Convex Optimization with Memory (COCOM) based on arXiv:2603.21375.
It MUST provide a `COCOMPipeline` class in `python/carnot/pipeline/cocom.py`.
The class MUST track memory-based constraints across online learning steps and optimize a defined objective function subject to these tracked memory constraints.

**Acceptance criteria:**
- `COCOMPipeline` is defined in `python/carnot/pipeline/cocom.py`.
- It implements `update(objective_grad, constraint_grad)` to perform online learning steps while tracking constraints in memory.
- It writes the results artifact to `results/experiment_1831_cocom.json`.

### SCENARIO-PIPELINE-1831: COCOM Online Learning Tracks Memory Constraints
**Given** a COCOM pipeline initialized with a learning rate and memory budget
**When** multiple online steps are processed with objective and constraint gradients
**Then** the parameters are updated such that memory constraints are respected.
**And** the outcome is recorded in `results/experiment_1831_cocom.json`.

### REQ-PIPELINE-1833: Unknown Constraints Estimation
The online learner MUST be able to estimate hidden safety constraints via an online regression oracle.
The `COCOMPipeline` class MUST implement an `estimate_hidden_constraint(features, true_constraint_value)` method that trains a regression oracle online to predict constraints from features.
It MUST provide a `predict_hidden_constraint(features)` method.
It MUST write the results artifact to `results/experiment_1833_unknown_constraints.json`.

### SCENARIO-PIPELINE-1833: Online Regression Oracle Estimates Hidden Constraints
**Given** a COCOM pipeline initialized with an online regression oracle
**When** hidden constraint values are provided sequentially with their corresponding features
**Then** the oracle updates its weights to estimate the hidden constraint
**And** the outcome is recorded in `results/experiment_1833_unknown_constraints.json`.

### REQ-PIPELINE-1843: Gradient-Guided Epsilon Constraint Tracking
The continuous learning loop MUST be extended with gradient-guided epsilon constraint tracking for the FR-11 non-forgetting loop.
The `COCOMPipeline` class MUST implement an `update_with_epsilon(objective_grad, constraint_grad, epsilon)` method that updates parameters with hard epsilon updates.
It MUST write the results artifact to `results/experiment_1843_epsilon_ocl.json`.

### SCENARIO-PIPELINE-1843: Hard Epsilon Updates in Continuous Learning
**Given** a COCOM pipeline configured for continuous learning
**When** multiple online steps are processed with objective gradients, constraint gradients, and an epsilon parameter
**Then** the parameters are updated using gradient-guided epsilon tracking to satisfy the FR-11 non-forgetting constraint
**And** the outcome is recorded in `results/experiment_1843_epsilon_ocl.json`.

### REQ-ROCE-1846: ROCE Abstraction for Dynamic Constraint Extraction
The system shall benchmark ROCE latency and validity limits on the MoE model (unsloth/Qwen3.6-35B-A3B-GGUF).

### SCENARIO-ROCE-1846: Validate ROCE Latency limits
**Given** the Qwen3.6 flagship MoE
**When** the pipeline processes dynamic constraints using ROCE limits
**Then** the extraction latency and validity are benchmarked successfully.

### REQ-ROCE-1847: ROCE Abstraction for Dynamic Constraint Extraction (Dense)
The system shall benchmark ROCE latency and validity limits on the flagship Dense model (unsloth/gemma-4-31B-it-GGUF).

### SCENARIO-ROCE-1847: Validate ROCE Latency limits on Dense Model
**Given** the gemma-4 flagship Dense model
**When** the pipeline processes dynamic constraints using ROCE limits
**Then** the extraction latency and validity are benchmarked successfully and written to `results/experiment_1847_gemma31_roce.json`.

### REQ-PIPELINE-1848: Zero-Forgetting FR-11 Constraint Learning via Epsilon
The pipeline MUST validate zero-forgetting FR-11 constraint learning loops via Epsilon constraint on `unsloth/gemma-4-26B-A4B-it-GGUF`.
It MUST write a terminal artifact with `experiment_id` 1848 to `results/experiment_1848_gemma26_epsilon.json`.
The artifact MUST include the status, model_specs, objective gradients applied, epsilon applied, and honest_verdict.

### SCENARIO-PIPELINE-1848: FR-11 Zero-Forgetting Epsilon Learning
**Given** the Gemma-4 26B model and the COCOM pipeline
**When** the pipeline processes continuous learning steps with epsilon constraints and strict utility/non-forgetting checks
**Then** the parameters are updated, enforcing zero-forgetting FR-11 checks, and written to `results/experiment_1848_gemma26_epsilon.json`.


### REQ-ROCE-1864: ROCE Open Constraint Elicitation Prototype
The system shall implement an open constraint elicitation prototype that extracts structured logic from unconstrained SOTA generation, specifically targeting `unsloth/Qwen3.6-35B-A3B-GGUF` natural language output. It MUST evaluate the extraction success rate on a 20-prompt dataset and save output to `results/experiment_1864_roce.json`.

### SCENARIO-ROCE-1864: ROCE Dynamic Logic Extraction
**Given** the Qwen3.6 MoE model output
**When** ROCE processes the natural language output for 20 prompts
**Then** dynamic verifiable logic is extracted successfully, success rate is evaluated, and results are written to `results/experiment_1864_roce.json`.

### REQ-PIPELINE-2053: Mouth/Brain Separation Audit
The system MUST perform an audit of the Python and Rust layers to identify tight coupling between the language generator ("mouth") and the energy verifier ("brain").
The findings MUST be formatted as a JSON artifact at `results/experiment_2053_mouth_brain_audit.json` with keys `experiment_id`, `title`, `findings`, and `recommendation`.
The Python module `carnot.verify.mouth_brain_audit` MUST expose a `run_audit()` function returning the dict, and it MUST be tested with 100% coverage.

### SCENARIO-PIPELINE-2053: Audit JSON Generation
**Given** the Carnot codebase with VerifyRepairPipeline
**When** `run_audit()` is called
**Then** it returns the JSON artifact dict with identified coupling points (e.g. `_model`, `_generate`) and a clean separation recommendation.

### REQ-PIPELINE-EMPIRICAL-DELTA: Empirical Delta Calculation

The pipeline MUST provide a function `compute_empirical_delta(results_dir: Path) -> float` to compute the single-step absorption probability (delta) from recent verify-repair runs by reading JSON logs containing iteration counts and success markers.

**Acceptance criteria:**
- `compute_empirical_delta` is implemented in `carnot.pipeline.empirical_delta`.
- Returns the ratio of successful repairs to total repair iterations.
- If no logs exist, returns 0.0.

### SCENARIO-PIPELINE-EMPIRICAL-DELTA: Computes delta

**Given** a directory containing repair JSON logs
**When** `compute_empirical_delta` is called
**Then** it returns the correct float delta.

**Spec traces:** REQ-PIPELINE-EMPIRICAL-DELTA

### REQ-PIPELINE-6479: Default-Off Factor Cache Shadow Adapter

`VerifyRepairPipeline` MUST expose a default-off FR-11 factor-cache shadow
adapter option. When the option is absent or false, public verification
decisions and certificates SHALL match the baseline path. Environment
variables SHALL NOT enable this factor-cache adapter.

When enabled, the adapter MAY observe exact verification receipts, propose
cache writes, and record rank advice. It SHALL NOT release an answer or admit a
cache write unless the existing exact checker supplied a prior exact receipt.

The adapter interface SHALL be versioned and SHALL provide `observe`,
`exact_admit`, `propose_rank`, `tombstone`, `rollback`, `save`, `load`, and
`close`.

### SCENARIO-PIPELINE-6479-SHADOW: Shadow Advice Preserves Release Behavior

**Given** a baseline `VerifyRepairPipeline.verify()` call
**When** the factor-cache shadow adapter is enabled
**Then** the returned `verified`, `energy`, `violations`, `mode`, and
non-shadow certificate fields SHALL match the baseline result
**And** the adapter SHALL record its proposed rank or abstention only as shadow
receipt data.

### REQ-PIPELINE-6549: Default-Off Production Safety-Net Adapter

`VerifyRepairPipeline` MUST expose a typed production Safety-Net adapter that
is disabled unless a caller explicitly passes an enabled configuration. The
disabled state SHALL preserve native serialized request bytes, candidate order,
checker calls, return values, exception types, side effects, and persistence
behavior.

When enabled, the adapter MAY reorder candidates or abstain. It SHALL preserve
the complete candidate set and SHALL keep the native exact fallback reachable.
The adapter SHALL record route, abstention, exception lookup, fallback reason,
exact result, and charged overhead after exact verification. Unsupported or
malformed inputs SHALL fall back without changing exact accepted outputs.

The adapter SHALL use the frozen V566 compact-router feature contract. It SHALL
not put held rows, held outcomes, source IDs, entity names, or row order into
policy state. The train-only exception table SHALL be immutable after
configuration freeze. Rollback SHALL disable the adapter and restore native
fallback routing.

### SCENARIO-PIPELINE-6549-DEFAULT-OFF: Disabled Adapter Is Byte-Identical

**Given** a native `VerifyRepairPipeline.verify()` request
**When** the production Safety-Net adapter is absent or configured disabled
**Then** the serialized request, candidate order, checker call count, result
fields, error type, side effects, and persistence behavior SHALL match the
native path byte-for-byte.

**Spec traces:** REQ-PIPELINE-6549

### SCENARIO-PIPELINE-6549-ENABLED-FALLBACK: Enabled Routing Preserves Candidates

**Given** an enabled adapter with the frozen V566 compact-router contract
**When** it routes, abstains, or hits a train-only exception
**Then** every candidate remains present exactly once, native exact fallback is
reachable, exact accepted outputs stay equal to native evaluation, and the
certificate records route, abstention, exception, fallback, exact result, and
charged overhead.

**Spec traces:** REQ-PIPELINE-6549

### SCENARIO-PIPELINE-6549-ATTACKS: Shortcuts Fail Closed

**Given** malformed inputs, stale configuration, candidate deletion, row-order
dependence, source or entity identity, held table writes, fallback recursion,
serialization drift, or disabled-path side effects
**When** the adapter evaluates the request
**Then** it SHALL fall back or roll back without accepting a changed exact
output, mutating the exception table, or deleting a candidate.

**Spec traces:** REQ-PIPELINE-6549

### REQ-PIPELINE-6563: Measured Production Safety-Net Workload Canary

Carnot MUST provide an Exp6563 canary that runs the production
`VerifyRepairPipeline` Safety-Net adapter on a frozen family-blind workload
matrix. The canary SHALL run native, disabled-adapter, enabled-adapter,
forced-abstain, forced-fallback, and rollback conditions over identical
requests, candidate order, warm-up count, seed, and process placement.

The workload matrix SHALL be frozen before execution. It SHALL use only
checked-in fixtures and SHALL cover normal, empty, malformed, unsupported,
fallback-heavy, exception, restart, and rollback strata. The route request
SHALL not contain model or family identity, source IDs, entity names, row
order, hidden outcomes, or future rows.

The canary SHALL record one per-unit row for every workload, seed, and adapter
condition. Each row SHALL include request bytes, route, abstention, candidate
set and order, exact result, checker calls, serialization bytes, persistence
bytes, process time, monotonic wall time, fallback reason, and rollback state.
Synthetic adapter cost units SHALL be excluded from headline work and latency
claims. They may appear only as diagnostic data.

The canary SHALL write a terminal artifact at
`results/experiment_6563_production_safety_net_workload_canary.json` with
`inference_substrate=production_verify_repair_workload_canary_exact_verifier_no_llm`
and `verifier_is_oracle=false`. The artifact SHALL include the required fields
listed by the V568 roadmap for Exp6563, field provenance, resource and fixture
preconditions, protected-file hashes, aggregate row recomputation, and a
reproducibility checksum.

`production_workload_canary_ready_score` SHALL equal 1.0 only when disabled
identity, exact equality, candidate preservation, fallback, restart, rollback,
complete rows, protected files, and checksum all pass. It SHALL equal 0.0 for
blocked, partial, disqualified, or incomplete evidence.
`production_workload_promotion_candidate_score` SHALL equal 1.0 only when the
enabled route improves preregistered measured checker work or measured latency
without safety, tail-latency, exact-output, fallback, rollback, or invalid
release regression. A safe canary with no measured enabled-path benefit SHALL
use `verdict_class=null`.

### SCENARIO-PIPELINE-6563-IDENTITY: Disabled Path Matches Native

**Given** the frozen workload matrix and the default-off adapter
**When** native and disabled-adapter conditions run
**Then** serialized request bytes, candidate order, exact result bytes,
checker calls, error type, side effects, and persistence SHALL match.

**Spec traces:** REQ-PIPELINE-6563

### SCENARIO-PIPELINE-6563-MEASURED-WORK: Enabled Rows Use Direct Receipts

**Given** enabled, forced-abstain, forced-fallback, malformed, unsupported, and
exception workload rows
**When** the canary computes benefit and latency
**Then** headline fields SHALL derive from checker calls, serialization bytes,
persistence bytes, process time, and monotonic wall time in emitted rows, not
from synthetic adapter cost units.

**Spec traces:** REQ-PIPELINE-6563

### SCENARIO-PIPELINE-6563-FALLBACK-ROLLBACK: Escape Paths Recover Exactly

**Given** forced fallback, abstention, exception, restart, and rollback
conditions
**When** the canary routes each workload through the production adapter
**Then** fallback remains reachable, ledger persistence is visible, restart
preserves exact equality, rollback disables the adapter, and exact accepted
outputs stay equal to native verification.

**Spec traces:** REQ-PIPELINE-6563

### SCENARIO-PIPELINE-6563-ATOMIC: Terminal Artifact Is Recomputable

**Given** complete per-unit rows, field provenance, protected hashes, and test
receipts
**When** Exp6563 writes its artifact
**Then** required fields match the roadmap contract, scores recompute from raw
rows, blocked checks name expected and observed values, and the checksum
detects mutation after the verdict.

**Spec traces:** REQ-PIPELINE-6563

### REQ-INFRA-6800: A Declared Scope MUST Bound What a Commit May Stage

**Statement:** A session MAY declare, before doing the work, the set of path globs it
intends to touch. While that declaration is active, any staged path outside it MUST refuse
the commit. The declaration MUST NOT be staged by the commit it governs. With no
declaration the check MUST be inert.

**Rationale:** Nothing in this repository recorded what an agent said it was going to
touch, so nothing could tell an intended edit from a stray one. On 2026-08-27 `git add -A`
swept another agent's staged file into an unrelated commit twice, and two agents
independently invented "commit with an explicit pathspec" as the fix while under pressure.
A pathspec typed by hand is forgotten exactly when it matters; a declaration cannot be.

The declaration is made BEFORE the work so it cannot be widened afterwards to cover
something inconvenient. That is why a scope file staged in its own commit is refused: an
agent that can edit its own scope in the commit it governs has no scope at all.

Inert-without-a-declaration is deliberate and is what makes this shippable. The conductor
commits with `git add -A` every few minutes and declares nothing; a check that refused by
default would wedge the research loop on its first day. The honest limit is stated in the
implementation: this cannot stop an agent that never declares a scope. It stops one that
declared a scope from quietly growing it, which is the failure that actually occurred.

**Implementation:** `scripts/harness_integrity_lint.py` (`check`, `_matches`,
`_scope_self_staged`), pre-commit hook `harness-integrity-lint`.

**Spec traces:** REQ-INFRA-6800

### REQ-PIPELINE-6703: Cold Audit Rows Own The Readiness Gate

Exp6703 SHALL reduce `planning_fixture_audit_passed` only from raw coverage,
solver, comparison, leakage, split, seal, metamorphic, mutation, row, test,
coverage, specification, and end-to-end checks. A missing or malformed row
SHALL remain a named failure. The reducer SHALL not replace it with zero.

### SCENARIO-PIPELINE-6703-ROW-REDUCTION: One Failed Unit Closes The Gate

**Given** the complete audit row set
**When** any required unit is absent, duplicated, malformed, or failing
**Then** readiness is false and `gate_check_summary` records the expected and
observed values for the first localized failure.

**Spec traces:** REQ-PIPELINE-6703

### SCENARIO-PIPELINE-6703-PER-UNIT-CONSERVATION: Every Audit Row Is Recheckable

**Given** coverage, solver, comparison, leakage, transform, and mutation rows
**When** `per_unit_rows` is built
**Then** it equals the typed union of those rows with no missing or extra unit.

**Spec traces:** REQ-PIPELINE-6703

### REQ-PIPELINE-6715: Exact Replay Rows Own The Audit Gate

Exp6715 SHALL reduce `exact_replay_audit_passed` only from raw precondition,
manifest, cap, edge, enumeration, comparison, test, artifact-validation, row-
consistency, adversarial-verification, and protected-file checks. Every
expected instance and state-action row SHALL exist exactly once. Every exact
comparison SHALL match. The method contract SHALL show no substitution, sample
widening, or cap reduction. A missing or malformed value SHALL remain a named
failure and SHALL not become zero.

### SCENARIO-PIPELINE-6715-ROW-REDUCTION: One Raw Failure Closes The Gate

**Given** the complete bounded audit rows
**When** one required row is missing, duplicated, mismatched, capped, or failing
**Then** `exact_replay_audit_passed` is false
**And** `gate_check_summary` records expected and observed values.

**Spec traces:** REQ-PIPELINE-6715

### SCENARIO-PIPELINE-6715-PER-UNIT: Every Unit Is Conserved

**Given** check, cap, enumeration, state-action, and comparison rows
**When** `per_unit_rows` is rebuilt
**Then** it equals their typed union without a missing or extra row
**And** aggregate counts and the gate recompute from that union.

**Spec traces:** REQ-PIPELINE-6715

## Implementation Status (REQ-PIPELINE-6715)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-PIPELINE-6715 and SCENARIO-PIPELINE-6715-* | Implemented (`python/carnot/experiment_6715_bounded_exact_replay_audit.py`) | Implemented (`tests/python/test_experiment_6715_bounded_exact_replay_audit.py`; row conservation, cap, and fail-closed reduction checks) |

### REQ-INFRA-6801: Harness Files MUST Be Sealed Against Unnamed Modification

**Statement:** A scope declaration MUST record a SHA-256 of every path in `SEALED_PATHS`
at declaration time, and the check MUST refuse the commit if any of those files differs in
the WORKING TREE at commit time. A sealed path MUST be exempted only by naming it exactly
in the declaration; a glob MUST NEVER unseal a path. Appearance and deletion of a sealed
path both count as drift.

**Rationale:** The sealed set is not a list of important files. It is the list of files
whose modification changes the VERDICT on other files — the fabrication gate, the
record-preservation lints, the gate config, the test infrastructure. Editing one while
working on something else is how a change makes itself pass.

The check reads the working tree rather than the index, and that is the whole point: an
agent that weakens a test file so its own change passes does not need to stage it, because
the suite reads the working tree. A staged-diff check cannot see that case at all.

A glob never unseals, because widening a glob is cheap and is what an agent under pressure
does, whereas typing the name of the fabrication gate is a deliberate act that leaves the
intent in the record.

**Implementation:** `scripts/harness_integrity_lint.py` (`SEALED_PATHS`, `declare`,
`check`).

**Spec traces:** REQ-INFRA-6801

### SCENARIO-INFRA-6802: No Declaration Leaves the Conductor Untouched

**Given** no scope declaration exists
**When** the pre-commit gate runs on any staged set
**Then** it exits 0 without inspecting the index.

**Spec traces:** REQ-INFRA-6800

### SCENARIO-INFRA-6803: A Swept-In File Outside the Scope Refuses the Commit

**Given** a session declared `scripts/foo.py`
**When** the index also contains `ops/known-issues.md`
**Then** the commit is refused and both the offending path and the declared scope are named.

**Spec traces:** REQ-INFRA-6800

### SCENARIO-INFRA-6804: A Sealed File Edited but Never Staged Refuses the Commit

**Given** a session declared a scope that does not name the fabrication gate
**And** the gate file is modified in the working tree and never staged
**When** the pre-commit gate runs
**Then** the commit is refused and the drifted path is named.

**Spec traces:** REQ-INFRA-6801

### SCENARIO-INFRA-6805: An Explicitly Named Sealed Path May Be Edited

**Given** a session declared the scope and named the sealed path with `--unseal`
**When** that sealed file is modified
**Then** the commit is permitted.

**Spec traces:** REQ-INFRA-6801

### SCENARIO-INFRA-6806: A Scope Staged in Its Own Commit Refuses

**Given** a session's declaration file is itself in the index
**When** the pre-commit gate runs
**Then** the commit is refused, because a scope edited in the commit it governs proves
nothing.

**Spec traces:** REQ-INFRA-6800

### SCENARIO-INFRA-6807: An Unreadable Declaration Refuses Rather Than Passes

**Given** a scope file that cannot be parsed
**When** the pre-commit gate runs
**Then** the commit is refused, because unreadable is not the same as absent.

**Spec traces:** REQ-INFRA-6800
