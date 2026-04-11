# Carnot — Session Metrics

## Session: 2026-04-11 Exp 161 Full GSM8K Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T14:05:56Z | 2026-04-11T14:11:24Z | Exp 161: Full GSM8K (1319 questions) × 2 models × 3 modes + 95% bootstrap CIs; real GSM8K dataset loaded; simulation fallback (CARNOT_SKIP_LLM=1); Qwen3.5: +13.8% [+12.0%,+15.7%], Gemma4: +10.7% [+9.1%,+12.4%]; Goal #6 PARTIAL (real data, simulated inference); saved results/experiment_161_results.json | TBD |

---

## Session: 2026-04-11 Exp 158 FactualExtractor

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T13:20:24Z | 2026-04-11T13:34:57Z | Exp 158: FactualExtractor (Wikidata SPARQL) — factual_extractor.py + 69 tests (100% cov) + AutoExtractor enable_factual_extractor= param; benchmark: coverage=96.0% (target >30% ✓), accuracy=83.3%; results/experiment_158_results.json | TBD |

---

## Session: 2026-04-11 Exp 157 Spilled Energy Pre-Filter

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T12:56:05Z | 2026-04-11T13:07:17Z | Exp 157: SpilledEnergyExtractor (arxiv 2602.18671 ICLR 2026) — spilled_energy.py + 33 tests (100% cov) + AutoExtractor logits= param; benchmark AUROC=1.000 (target >0.60 ✓); coverage 100% vs NLExtractor 60%; results/experiment_157_results.json | TBD |

---

## Session: 2026-04-11 Exp 156 JEPA Fast-Path v2

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T11:46:44Z | 2026-04-11T11:49:59Z | Exp 156: JEPA fast-path v2 validation — v2 predictor vs v1 at thresholds 0.3/0.5/0.7; target NOT MET (no threshold achieved <2% degradation); best: t=0.5 → 52.8% fast-path, 10.2% degradation (v1: 95.4%/19.8%); code domain still dominates errors (42/51 at t=0.5); root cause: code pipeline fast-paths entire domain; saved results/experiment_156_results.json | TBD |

---

## Session: 2026-04-11 Planning Agent — Milestone 2026.04.11

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T10:12:16Z | 2026-04-11T10:19:32Z | Plan milestone 2026.04.11: read 14 source files + arxiv research; wrote research-roadmap-vNEXT.md (v16) + research-roadmap-next.yaml (12 experiments, 4 phases); updated research-references.md with 5 new arxiv papers; 3 biggest gaps: JEPA multi-domain fix, factual extractor, live eGPU benchmarks | TBD |

---

## Session: 2026-04-11 Exp 152 ContinualGibbs

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T09:28:39Z | 2026-04-11T09:37:39Z | Exp 152: ContinualGibbsModel (orthogonal gradient continual learning); 29 tests, 100% module coverage; benchmark vs Ising/LNN (Exp 116) — ContinualGibbs 100% step-5 accuracy (target >80% met); results/experiment_152_results.json | TBD |
| 2 | 2026-04-11T09:52:26Z | 2026-04-11T10:02:54Z | Exp 153: KAN adaptive mesh refinement — compute_edge_curvature() + refine(threshold=1.5); AUROC 0.875→0.875 (delta=0.000, ✓), params 2310→2281 (-1.3%, ✓); 36 knots added, 65 removed; high-curv=domain×numeric edges, low-curv=within-group edges | TBD |

---

## Session: 2026-04-11 Constraint Propagation Model Export

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T08:58:24Z | 2026-04-11T09:11:35Z | Create exports/constraint-propagation-models/ (arithmetic AUROC=0.997, logic 1.0, code 0.867); python/carnot/inference/constraint_models.py (IsingConstraintModel + ConstraintPropagationModel with from_pretrained/save_pretrained); scripts/export_constraint_models.py (training + export); 3 model cards + collection README; 52 tests passing, constraint_models.py 100% coverage | TBD |

---

## Session: 2026-04-11 Exp 149 TruthfulQA Factual Coverage Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T07:45:33Z | 2026-04-11T07:49:58Z | Create + run scripts/experiment_149_truthfulqa.py; TruthfulQA 196q balanced 7 cats; overall coverage 43.4%; KB adds 0%; covered q: accept 100%, reject 0% (shallow extraction); top-1 missing: world_knowledge (8.1% gain); recommend FactualWorldKnowledgeExtractor; results/experiment_149_results.json | TBD |

---

## Session: 2026-04-11 Pre-research test suite check

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T07:32:37Z | 2026-04-11T07:43:14Z | Run full test suite to verify pre-research baseline — 2041 passed, 1 skipped, 0 failures, 99.26% coverage (≥99% threshold met) | TBD |

---

## Session: 2026-04-11 Exp 147 Apple GSM8K Adversarial

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T04:52:33Z | 2026-04-11T05:07:44Z | Create scripts/experiment_147_apple_gsm8k.py; 3-mode eval (baseline/verify/verify-repair) × 4 adversarial variants × 2 models; Qwen number-swapped: baseline 46% → VR 73% (+27pp); Gemma number-swapped: 53% → 77.5% (+24.5pp); control deltas: +10pp/+13pp; hypothesis direction supported (num-swap delta >> control delta); permutation test p=0.463 (N too small); results/experiment_147_results.json | TBD |

---

## Session: 2026-04-11 Exp 146 AMD XDNA NPU Latency Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T04:41:15Z | 2026-04-11T04:44:27Z | Create scripts/experiment_146_npu.py + python/carnot/samplers/npu_backend.py; detect NPU HW (present: /dev/accel0, amdxdna loaded) vs SW (AMDXDNAExecutionProvider absent from std onnxruntime); export JEPA MLP to ONNX; CPU benchmark p50=0.005ms p99=0.009ms; NPU blocked — needs conda install -c amd onnxruntime-vitisai; results/experiment_146_npu_results.json | TBD |

---

## Session: 2026-04-11 Exp 145 JEPA Fast-Path Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T04:17:02Z | 2026-04-11T04:23:55Z | Add JEPA fast-path gate to VerifyRepairPipeline.verify() (jepa_predictor=, jepa_threshold=, mode/skipped fields on VerificationResult); 8 new tests 100% coverage; create + run scripts/experiment_145_jepa_fastpath.py; threshold=0.3: 38% fast-path (miss), 11.6% degradation (miss); threshold=0.5: 95.4% fast-path (pass), 19.8% degradation (miss); targets not met — predictor AUROC 0.57 insufficient; results/experiment_145_results.json | TBD |

---

## Session: 2026-04-11 Exp 142 Combined Learning Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T04:03:52Z | 2026-04-11T04:09:25Z | Create scripts/experiment_142_combined_learning.py — 4-way benchmark (Baseline/Tier1/Tier2/Combined) on 500 questions; Tier2 beats Tier1: YES (+7%, 71.75%→78.75%); Combined≈Tier2 (no Tier1 additive gain); 100% of Tier2 gains from new constraints; results/experiment_142_results.json | TBD |

---

## Session: 2026-04-11 Exp 141 Constraint Generation from Memory

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T03:45:37Z | 2026-04-11T03:55:16Z | Create generation.py (ConstraintGenerator, CarryChainConstraint, BoundConstraint, NegationConstraint); extend AutoExtractor.extract(memory=); 62 tests 100% coverage; Exp 141 benchmark: static=0.85→memory=0.96 (+0.11 delta, hypothesis MET); adversarial review found+fixed dedup bug | TBD |

---

## Session: 2026-04-11 Exp 144 JEPA Predictor Training

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T03:32:26Z | 2026-04-11T03:37:53Z | Create python/carnot/pipeline/jepa_predictor.py (JEPAViolationPredictor MLP 256→64→32→3), tests/python/test_jepa_predictor.py (36 tests, 100% coverage), scripts/experiment_144_train_jepa.py; train on Exp 143 pairs: arithmetic AUROC=0.7126, macro AUROC=0.5709 (code/logic AUROC=0.5 — no positives in data); model saved to results/jepa_predictor.safetensors (73.1 KB) | TBD |

---

## Session: 2026-04-11 Exp 143 JEPA Training Pair Collection

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T03:09:05Z | 2026-04-11T03:12:06Z | Create scripts/experiment_143_collect_pairs.py — mines result logs (0 pairs found), generates 200 synthetic arithmetic pairs via AutoExtractor+VerifyRepairPipeline, embeds 4 prefix ratios with RandomProjection(256-dim, seed=42); 800 total pairs, 33.5% positive rate; saved to results/jepa_training_pairs.json | TBD |

---

## Session: 2026-04-11 Exp 140 Constraint-Projection Guided Latency

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T02:50:09Z | 2026-04-11T03:00:29Z | Create scripts/experiment_140_guided_latency.py — adds project_logits() to EnergyGuidedSampler; benchmarks constraint-projection at batch sizes 1/8/32 (p50=0.405/1.284/4.056 ms); GSM8K accuracy baseline=56% penalty=64% projection=60%; success criterion PASS (0.405ms < 5ms); results in results/experiment_140_results.json | TBD |

---

## Session: 2026-04-11 Exp 139 ArXiv Scan

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T02:30:45Z | 2026-04-11T02:34:00Z | Create scripts/experiment_139_arxiv_scan.py — scans arXiv for 8 query topics, selects top 10 papers, annotates with Carnot relevance, proposes EXP-140/141/142; appends 10 new papers to research-references.md; results in results/experiment_139_results.json | TBD |

---

## Session: 2026-04-11 Exp 138 Guided Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T02:20:12Z | 2026-04-11T02:24:15Z | Create scripts/experiment_138_guided_benchmark.py — 3-task guided decoding benchmark (GSM8K 200, HumanEval 50, TruthfulQA 100); 4 modes; latency profiling; results saved to results/experiment_138_results.json | TBD |

---

## Session: 2026-04-11 Doc update Exp 137

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T02:16:21Z | 2026-04-11T02:17:11Z | Update ops/status.md + ops/changelog.md for Exp 137 (HF guided decoding adapter export); changelog was already written by conductor; added status.md header update + new HuggingFace Guided Decoding Adapter Export section | TBD |

---

## Session: 2026-04-11 guided-decoding-adapter export

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T02:09:32Z | 2026-04-11T02:14:00Z | Create exports/guided-decoding-adapter/ — HuggingFace-publishable artifact for GuidedDecoder; added GuidedDecoder.from_pretrained() API to guided_decoding.py; 7 new tests all pass; example.py verified | TBD |

---

## Session: 2026-04-11 Exp 136 Cross-Session Memory

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T01:57:14Z | 2026-04-11T02:02:12Z | Create scripts/experiment_136_cross_session.py — 3-session cross-session memory test (200 arith S1, 200 arith S2 no-mem vs mem, 200 mixed S3); all 4 hypotheses pass: memory accumulates (115 patterns), S2 hint delta +1.0/q, repair speedup 1.43x, domain specificity (logic/code get 0 hints); 0.5s wall-clock | TBD |

---

## Session: 2026-04-11 Exp 134 Online Learning

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T01:21:28Z | 2026-04-11T01:36:32Z | Create scripts/experiment_134_online_learning.py — online learning demo with soft weighted-score verifier + NoisyHeuristicExtractor + ground-truth tracker recording; fixed=67.6%, adaptive=97.0%, delta=+29.4% overall; at q200 delta=+42.0% (target met) | TBD |

---

## Session: 2026-04-11 AdaptiveWeighter (Tier 1 Self-Learning)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T01:07:05Z | 2026-04-11T01:12:37Z | Create python/carnot/pipeline/adaptive.py (AdaptiveWeighter: from_tracker/apply_to_pipeline, run_comparison, ComparisonResult) + modify verify_repair.py to use _adaptive_weights + tests/python/test_adaptive.py (23 tests, 100% coverage, 1895 total pass) | TBD |

---

## Session: 2026-04-11 ConstraintTracker (Tier 1 Self-Learning)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T00:51:06Z | 2026-04-11T00:59:08Z | Create python/carnot/pipeline/tracker.py (ConstraintTracker: record/precision/recall/stats/save/load/merge) + integrate into VerifyRepairPipeline.verify(tracker=) + tests/python/test_tracker.py (28 tests, 100% coverage, 1872 total pass) | TBD |

---

## Session: 2026-04-11 Exp 121 Adversarial Verify-Repair

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T00:28:47Z | 2026-04-11T00:32:25Z | Run experiment_121_adversarial_verify_repair.py (CARNOT_SKIP_LLM=1 simulation); results/experiment_121_results.json created (17KB); Qwen3.5-0.8B hypothesis p=0.005 (supported), Gemma4-E4B-it p=0.290 (not significant) | TBD |
| 2 | 2026-04-11T00:35:58Z | 2026-04-11T00:36:29Z | Update docs for Exp 130 adversarial verify-repair: add Exp 121 entry to _bmad/traceability.md | TBD |
| 3 | 2026-04-11T00:40:11Z | 2026-04-11T00:43:33Z | Exp 131: Create adversarial writeup script; generates comparison tables (per-variant/mode/model), bootstrap CIs, appends Section 18 to docs/technical-report.md, saves experiment_131_results.json | TBD |

---

## Session: 2026-04-11 LiquidConstraintModel (lnn.py)

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-11T00:12:49Z | 2026-04-11T00:20:55Z | Create python/carnot/models/lnn.py (LiquidConstraintModel: MLP-driven dJ/dt ODE, step(), energy(), reset(), train() BPTT) + tests/python/test_lnn.py (31 tests, 100% coverage, 1844 total pass) | TBD |

---

## Session: 2026-04-10 Exp 126 Agent Rollback

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T23:41:35Z | 2026-04-10T23:48:24Z | Create scripts/experiment_126_agent_rollback.py: 20 4-step math problems with error propagation, CSM rollback on violation detection; step-3 errors 100% detected, step-2 errors 0% detected; overall 50% improvement (0%→50% accuracy) | TBD |

---

## Session: 2026-04-10 ConstraintStateMachine

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T23:31:09Z | 2026-04-10T23:34:18Z | Create python/carnot/pipeline/state_machine.py (ConstraintStateMachine, StepResult, rollback, history, verified_facts, pending_facts) + 26 tests, state_machine.py 100% coverage | TBD |

---

## Session: 2026-04-10 Exp 122 Adversarial Error Analysis

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T23:14:56Z | 2026-04-10T23:23:33Z | Create Exp 122 adversarial error analysis: error taxonomy, Carnot detection rates per type, energy-prediction ROC (n_violations AUC=0.677), irrelevant-number extraction robustness; results saved | TBD |

---

## Session: 2026-04-10 Exp 120 Adversarial Baseline

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T21:59:18Z | 2026-04-10T22:13:37Z | Create Exp 120 adversarial baseline: LLM accuracy on 4 adversarial GSM8K variants, simulation mode (models too slow on CPU), results saved | TBD |

---

## Session: 2026-04-10 Robust Model Loader

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T21:41:01Z | 2026-04-10T21:51:47Z | Create carnot.inference.model_loader — robust HF model loading with memory check, OOM retry, CARNOT_FORCE_LIVE; 35 tests, 100% coverage, 1787 full suite pass | TBD |

---

## Session: 2026-04-10 Exp 119 Adversarial GSM8K

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T21:00:56Z | 2026-04-10T21:04:56Z | Create Exp 119 adversarial GSM8K (Apple 2410.05229 repro): 4 datasets × 200q, all 40 spot-checks pass | TBD |

---

## Session: 2026-04-10 Exp 118 HuggingFace Publish v12 Artifacts

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T20:55:35Z | 2026-04-10T20:56:17Z | Update changelog/status docs for Exp 118 HuggingFace v12 artifact publish | TBD |

---

## Session: 2026-04-10 Exp 117 Full v12 Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T20:29:53Z | 2026-04-10T20:37:00Z | Create Exp 117 full 4-mode v12 benchmark (2000 evaluations), run comparison vs v10, guided gen wins 10/10 cells | TBD |

---

## Session: 2026-04-10 Exp 116 LNN Adaptive Constraint Model

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T20:15:29Z | 2026-04-10T20:21:53Z | Create LNNConstraintModel (LTCN-based adaptive EBM), 22 tests (100% module cov), Exp 116 synthetic chain comparison vs Ising | TBD |

---

## Session: 2026-04-10 Exp 113 FactualKBExtractor

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T18:15:15Z | 2026-04-10T18:27:05Z | Create FactualKBExtractor with 5000-fact KB, 78 tests (100% cov), register in AutoExtractor | TBD |

---

## Session: 2026-04-10 Exp 112 Embedding Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T17:53:16Z | 2026-04-10T17:59:11Z | Create fast_embedding.py (5 strategies + protocol), experiment_112 script, run benchmark, update ops | TBD |

---

## Session: 2026-04-10 Exp 110 Guided Decoding

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T17:10:32Z | 2026-04-10T17:19:14Z | Create EnergyGuidedSampler, 22 tests (100% cov), Exp 110 on 50 GSM8K problems, alpha sweep [0.1–2.0] | TBD |

---

## Session: 2026-04-10 Exp 102 Latency Benchmark

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T04:54:17Z | 2026-04-10T05:01:01Z | Create Exp 102 latency benchmark, run on CPU, save results + summary | TBD |

---

## Session: 2026-04-10 Exp 93 Multi-Model Comparison

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-10T01:59:33Z | 2026-04-10T02:07:00Z | Create Exp 93 multi-model comparison script, run benchmark, update ops | TBD |

---

## Session: 2026-04-09 Exp 57 Verify-Repair Loop

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-09T14:08:06Z | 2026-04-09T14:12:40Z | Create Exp 57 verify-repair loop script, run E2E with live LLM | TBD |

---

## Session: 2026-04-07 Research Roadmap v5 + Nemotron Analysis

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-07T18:59:33Z | 2026-04-07T19:03:43Z | Analyze Nemotron 3 Super paper, fold findings into roadmap v5 | TBD |
| 2 | 2026-04-07T19:08:49Z | 2026-04-07T19:15:52Z | Restructure roadmap v5 as weight-first (label-free) research program | TBD |
| 3 | 2026-04-07T19:21:22Z | 2026-04-07T19:32:00Z | Download Mixtral-8x7B, write Exp 32+33 scripts, update ops docs | TBD |

---

## Session: 2026-04-06 Documentation UI Modernization

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-06T18:00:00Z | 2026-04-06T18:05:00Z | Elevate docs/index.html to a premium aesthetic (glassmorphism, animations) | TBD |

---

## Session: 2026-04-06 GEMINI.md Initialization

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-06T16:47:49Z | TBD | Initialize GEMINI.md based on CLAUDE.md; adapt project mandates | TBD |

---

## Session: 2026-04-05 Hallucination Direction

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-05T05:18:58Z | TBD | Implement hallucination_direction.py with tests, exports, specs | TBD |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*

---

## Session: 2026-04-03 Bootstrap

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-04-03T14:20:41Z | TBD | Initial project bootstrap: BMAD, specs, Rust workspace, Python package, pre-commit | TBD |
| 2 | 2026-04-10T20:45:20Z | 2026-04-10T20:52:49Z | Publish KAN + guided decoding adapter as HF-ready artifacts in models/constraint-verifier-v2 | 7m29s |
| 3 | 2026-04-11T11:23:30Z | 2026-04-11T11:31:54Z | Exp 155: Retrain JEPA v2 on multi-domain data; generate v2 pairs, train with weighted BCE + early stopping, evaluate vs v1 | 8m24s |

### Session Summary

*To be filled by `scripts/session-metrics.py` at session end.*
