# Carnot — Changelog

## 2026-04-09

- Exp 82: Pipeline error handling and edge cases — structured error hierarchy (`CarnotError`, `ExtractionError`, `VerificationError`, `RepairError`, `ModelLoadError`, `PipelineTimeoutError`) in `python/carnot/pipeline/errors.py`; wall-clock timeout support in `VerifyRepairPipeline` via `timeout_seconds` parameter; graceful degradation for extraction, verification, repair, and model-loading failures; 737-line test suite covering all error paths (REQ-VERIFY-001, REQ-VERIFY-003, SCENARIO-VERIFY-004)
- MCP server hardening: migrated from `tools/verify-mcp/server.py` to `python/carnot/mcp/` package; added 4 new tools (verify_llm_output, verify_and_repair, list_domains, health_check); production safeguards: 30s execution timeout via ThreadPoolExecutor, 10K char input validation, structured error responses with machine-readable error_code; runnable as `python -m carnot.mcp`; 30 tests (REQ-CODE-001, REQ-CODE-006, REQ-VERIFY-001, REQ-VERIFY-003, SCENARIO-VERIFY-004)
- Exp 75: VerifyRepairPipeline class — user-facing API consolidating Exp 56 (live LLM verification) and Exp 57 (verify-repair loop) into `python/carnot/pipeline/verify_repair.py`; key classes: VerificationResult (per-call result with verified flag, constraint details, energy, violations, decomposition), RepairResult (full iteration history), VerifyRepairPipeline (main class with verify(), verify_and_repair(), extract_constraints()); verify-only mode (no model) and verify-and-repair mode (with LLM); exported from `carnot.pipeline`; 737-line test suite (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-004)
- Exp 74: Unified ConstraintExtractor API — consolidates constraint extraction from Exp 47 (arithmetic/logic), Exp 48 (code AST), and Exp 49 (NL claims) into a pluggable Protocol-based library at `python/carnot/pipeline/extract.py`; key classes: ConstraintResult (dataclass with optional energy term), ConstraintExtractor (Protocol), ArithmeticExtractor, CodeExtractor, LogicExtractor, NLExtractor, AutoExtractor (auto-detects domains and merges results); exported from `carnot.pipeline`; 678-line test suite (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-002)
- Exp 72: Autoresearch self-verification via Ising — dog-foods the Carnot constraint pipeline on the autoresearch loop's OWN hypothesis outputs; extracts verifiable claims from hypothesis code (Exp 48 AST extraction) and output text (Exp 49 NL extraction + numeric-claim patterns), then verifies with ComposedEnergy + Ising sampling; tests whether an Ising constraint-satisfaction "fourth gate" catches bogus hypotheses that the existing three gates (energy, time, memory) miss; simulates 20 mock hypotheses (10 correct, 10 bogus) with confusion matrix (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-002)
- Exp 70: Rust constraint extraction + verification — new `carnot-constraints` crate providing reusable built-in constraint types (`BoundConstraint`, `EqualityConstraint`, `IsingConstraint`) that implement `ConstraintTerm` from `carnot-core`, plus `VerificationCertificate` for serializable JSON proof of constraint satisfaction; re-exports core verification types for convenience; 243-line integration test suite covering composition, repair, Ising integration, certificate serialization, and deterministic reproducibility (REQ-VERIFY-001/002/003/004/005, SCENARIO-VERIFY-001/002/003/004/006)
- Exp 65: Embedding-space constraint verification — trains a Gibbs EBM on joint feature vectors concatenating semantic embeddings (all-MiniLM-L6-v2, 384-dim) with structural constraint vectors (per-constraint pass/fail from Ising verifier, N-dim); evaluates whether joint model discriminates correct/wrong answers better than either space alone; gradient-based repair in joint space with nearest-neighbor decoding; bridges semantic embedding space with structural constraint space (REQ-EBT-001, REQ-VERIFY-001)
- Exp 68: HumanEval subset verification + fuzzing — evaluates full Carnot code verification pipeline on 50 HumanEval-style coding problems; combines constraint extraction (Exp 48), runtime instrumentation (Exp 53), and Ising-guided fuzzing (Exp 54) into unified pipeline; measures pass@1 and pass@1+repair rates across generate → extract → instrument → test → fuzz → repair stages; bug detection breakdown by source (test-only, instrumentation-only, fuzzing-only); falls back to 50 manually-crafted problems if HumanEval dataset unavailable (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-006)
- Exp 67: GSM8K subset verification — first external benchmark of the verify-repair pipeline; 200 GSM8K test-split questions through 3 modes (baseline, verify-only, verify-repair with max 3 iterations); arithmetic chain-of-thought parsing with deterministic carry-chain verification (Exp 42c); error categorization (arithmetic/logic/reading); repair success rate per error type; uses Qwen3.5-0.8B with HuggingFace datasets fallback to synthetic GSM8K-style problems (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-006)
- Exp 63: Hierarchical Ising (1000+ vars) — block-structured coupling decomposition for large SAT instances; groups variables into blocks of size B (e.g., 50), with dense intra-block couplings and sparse inter-block couplings; two-phase training (intra-block CD then L1-regularized inter-block CD); two-level Gibbs sampler (inner parallel within blocks, outer inter-block messages) with simulated annealing; benchmarks hierarchical vs flat-sparse (Exp 61) vs flat-dense (Exp 60) vs random at 200/500/1000 variables; ~10x parameter reduction vs dense at 1000 vars
- Exp 62: Domain-specific constraint learning (10K triples) — trains discriminative Ising models on 10,000 (question, correct_answer, wrong_answer) triples across three domains (arithmetic, logic, code); 200+ binary features per answer; per-domain and combined models evaluated via AUROC on held-out test split; extends Exp 51 (discriminative CD) and Exp 60 (scaled CD) to multi-domain answer verification without an LLM
- Exp 73: Constraint coverage metric — quantifies "verification dark matter" by measuring what fraction of an LLM's verifiable claims are captured by the constraint extraction pipeline; defines 5-type claim taxonomy (arithmetic, logical, factual, structural, semantic); annotates 50 LLM answers (10 per domain) with total verifiable claims via heuristic counting (regex + AST); computes coverage = extracted_constraints / total_claims per domain and claim type; correlates coverage with post-repair accuracy to find the threshold below which repair stops helping (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-005)
- Exp 71: Extropic TSU sampler abstraction layer — adds `SamplerBackend` protocol in `python/carnot/samplers/backend.py` so experiments can swap between CPU-based parallel Gibbs sampling (`ParallelIsingSampler`) and Extropic's Thermodynamic Sampling Unit (TSU) hardware via a single config string or `CARNOT_BACKEND` env var; includes `CpuBackend` (wraps ParallelIsingSampler), `TsuBackend` (stub for future hardware), `get_backend()` factory; 183 tests added (REQ-SAMPLE-003)
- Exp 69: Multi-model constraint transfer validation (Qwen3.5+Gemma4) — tests whether Carnot constraint pipeline (arithmetic, logic, code AST, factual KB) transfers across model families WITHOUT retraining; runs same 20 Exp 56 questions through Exp 57 verify-repair loop on both Qwen3.5-0.8B and Gemma4-E4B-it; compares per-model accuracy, cross-model constraint transfer, model-specific hallucination patterns, constraint satisfaction rates (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-003)
- Exp 58: Multi-domain live benchmark (5 domains) — first comprehensive evaluation of the full verify-repair pipeline; 500 questions (100 per domain) across arithmetic, code, logic, factual, scheduling; three modes: LLM alone (baseline), LLM + Ising verification (detection), LLM + verify-repair loop (full pipeline); metrics: accuracy, hallucination rate, repair success rate, Ising energy, constraint count, wall-clock time; uses Qwen3.5-0.8B with fallback to simulated outputs (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-005)
- Exp 55: Learn constraints from execution traces — combines Exp 53's runtime instrumentation with Exp 51's discriminative CD training to LEARN bug-detection constraints from execution traces; collects correct and buggy execution traces (variable types, branch decisions, return values, loop iterations) as 200+ dim binary feature vectors; trains discriminative Ising model to assign low energy to correct traces, high energy to buggy traces; catches semantic bugs (wrong formulas, off-by-one accumulation) invisible to both static and dynamic analysis (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-002)
- Exp 54: Ising-guided fuzzing — uses Ising energy landscape to GENERATE adversarial test inputs (edge cases, boundary values, sign flips) for differential testing of LLM-generated code; encodes function parameters as Ising spins with edge-case-attracting biases; compares bug-finding rate against uniform random fuzzing across 8 common LLM code-gen bug types (off-by-one, null check, overflow, wrong operator, missing base case, type coercion, boundary error, sign error); uses ParallelIsingSampler with simulated annealing (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-003)
- Exp 64: Continuous Ising Relaxation — relaxes binary Ising spins s ∈ {0,1}^n to continuous s ∈ [0,1]^n and uses JAX gradient descent to minimize Ising energy; compares three rounding strategies (sigmoid annealing, penalty term, straight-through estimator) against ParallelIsingSampler (discrete Gibbs + simulated annealing) and random baseline; bridges discrete EBM sampling with continuous latent-space reasoning (toward Kona)
- Exp 61: Sparse Ising at 500+ Variables — exploits clause-graph sparsity to mask CD gradients, reducing effective parameters by ~20x vs dense CD (Exp 60); compares dense CD vs sparse CD vs hand-coded Ising at 200/500/1000 variables; hard sparsity eliminates "hallucinated" correlations between unrelated variables; tests generalization to unseen SAT instances of the same structure
- Exp 60: Scale CD Training to 100+ Variables — extends Exp 50 (10-var CD) to 50/100/200 variables (up to 40K parameters); bootstraps training data from hand-coded Ising + parallel annealing sampler; compares CD-trained vs hand-coded vs random couplings on both training and held-out SAT instances; tests whether learned couplings smooth the energy landscape better than hand-coded penalty mappings at scale; L1 regularization to prevent overfitting with 10K+ params from 5K samples
- Exp 59: Constraint-Aware Prompting — tests PREVENTIVE constraint injection (embed domain rules into prompt) vs POST-HOC verification (Exp 56-57); three modes on 15 questions (arithmetic, logic, factual): Mode A (baseline), Mode B (constraint-aware prompt), Mode C (combined: constraint prompt + verify-repair loop); measures accuracy, hallucination rate, constraint satisfaction, first-try accuracy; key question: does telling the LLM about constraints upfront reduce hallucination at generation time? (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-005)
- Exp 57: Verify-Repair Loop — closes the loop from Exp 56: constraint violations → NL feedback → LLM regeneration → re-verify, up to 3 iterations; 15 tricky questions (multi-step arithmetic, misleading logic, tricky factual); live LLM run: 9/15 initial accuracy, repair loop architecture works but constraint coverage limits effectiveness (only 1/6 wrong answers triggered violations); key finding: expanding constraint extractors to cover word problems and deeper factual KB is the bottleneck, not the repair mechanism (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-004)
- Exp 56: Live LLM → constraint → Ising verification — full end-to-end pipeline connecting Qwen3.5-0.8B to constraint extraction (Exp 47-49) and verification; 20 questions across 4 domains (arithmetic, logic, code, factual); live LLM generates answers + constraints, Carnot pipeline verifies (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-003)
- Exp 53: Runtime constraint instrumentation — dynamic AST rewriting with isinstance guards, bound checks, return-type assertions; complements Exp 48's static analysis (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-002)
- Exp 42c: Deterministic arithmetic verification via carry propagation (16/16 perfect)
- Research conductor: YAML extraction (research-roadmap.yaml + research-complete.yaml)
- Research conductor: CalVer milestones (2026.03.1, 2026.04.1, 2026.04.2)
- Research conductor: Self-healing for pre-flight test failures
- Conductor overnight run completed: Exp 48, 49, 51, 52, 44
- Roadmap v7: Toward Kona — live LLM + Ising end-to-end (phases 5-8)
- Documentation reconciliation audit and fixes

## 2026-04-08

- Parallel Ising Gibbs sampler: 183x faster than thrml (572x at 500 vars)
- thrml-compatible wrapper: parallel_sample_states() accepts IsingEBM
- Exp 42b: Arithmetic QUBO encoding (8/12, carry chains fail)
- Exp 46b: Scale SAT to 5000 vars (0.7s, +5.5% vs random)
- Exp 47: LLM self-constraint extraction (10/10 perfect)
- Exp 50: Learn Ising couplings via CD (89/100 perfect, generalizes)
- ROCm GPU: jax-rocm7-pjrt installed, validated on gfx1150 (iGPU slower than CPU)
- thrml ROCm bug filed: extropic-ai/thrml#41 (AQL packet crash)
- Research conductor updated for v6 roadmap experiments
- Test suite: 1130 passed, 100% coverage (added test_parallel_ising.py)
- docs/index.html: added fadeInUp animation (REQ-DOCUI-002)

---

- **2026-04-09 00:20 UTC** [orchestrator] Sprint complete for 'Epic: UI-001 - Modernize Documentation Aesthetic' (completed=2, failed=0)

- **2026-04-09 00:20 UTC** [orchestrator] Story DOCUI-002 completed and passed evaluation

- **2026-04-09 00:20 UTC** [orchestrator] Evaluator invoked for DOCUI-002 (initial attempt)

- **2026-04-09 00:20 UTC** [orchestrator] Generator invoked for DOCUI-002 (initial attempt)

- **2026-04-09 00:20 UTC** [orchestrator] Contract built for DOCUI-002

- **2026-04-09 00:20 UTC** [orchestrator] Story DOCUI-001 completed and passed evaluation

- **2026-04-09 00:20 UTC** [orchestrator] Evaluator invoked for DOCUI-001 (initial attempt)

- **2026-04-09 00:20 UTC** [orchestrator] Generator invoked for DOCUI-001 (initial attempt)

- **2026-04-09 00:20 UTC** [orchestrator] Contract built for DOCUI-001

- **2026-04-09 00:20 UTC** [orchestrator] Sprint started for epic 'Epic: UI-001 - Modernize Documentation Aesthetic' (run_id=b6ec974e, stories=2)

## 2026-04-08: Extropic thrml integration, LLM→Ising→repair pipeline (experiments 36-41)

### The Pivot
Proved activation-based hallucination detection doesn't work (confidence ≠ correctness).
Pivoted to structural constraint verification via Extropic-compatible Ising models.

### Key Experiments
- **Exp 36**: Logit lens divergence → 50.6% (chance). Dynamics identical for correct/wrong.
- **Exp 37**: EBT in sentence embeddings → 57.5%. Sentence encoders embed topic, not truth.
- **Exp 38**: NLI-based EBM → 70.8% test, 50% practical. NLI detects consistency, not facts.
- **Exp 39**: thrml Ising SAT solver → beats random at 50+ variables. First Extropic-compatible experiment.
- **Exp 40**: Graph coloring → Ising → thrml finds perfect solutions on 3/6 problems.
- **Exp 41**: **LLM → Ising verify → repair: 2/6 hallucinations caught and fixed (0%→100%).**

### Infrastructure
- Integrated Extropic's thrml library (IsingEBM, Block Gibbs sampling)
- SAT and graph coloring → Ising encoding pipelines
- Full LLM → constraint extraction → Ising → thrml → decoded solution pipeline
- Updated all 16 HuggingFace model cards with honest "research artifact" disclaimer
- Fixed all GitHub URLs (ianblenke/carnot → Carnot-EBM/carnot-ebm)

### The Definitive Finding
You cannot detect factual hallucination from model internals. You need external verification.
The "LLM proposes, Ising repairs" architecture works and maps to Extropic TSU hardware.

---

## 2026-04-07: Research Roadmap v5 — Weight-First EBM

### Paradigm Shift
Restructured the entire research program around a weight-first philosophy: derive hallucination signal from frozen weight structure and unlabeled forward passes. Labeled hallucination data becomes a validation tool, not a training dependency. 10 of 11 new experiments need zero training labels.

### Added
- `openspec/change-proposals/research-roadmap-v5.md`: Weight-First EBM roadmap
  - **Phase 1 (Weight Anatomy):** Exp 32-35 — pure weight analysis + unlabeled forward passes
  - **Phase 2 (Self-Supervised Energy):** Exp 36-39 — composite label-free energy functions
  - **Phase 3 (Consensus Landscape):** Exp 40-42 — multi-model weight geometry as energy
  - **Phase 4 (Standalone EBM):** 4a-4d — universal encoder → consensus landscape → LLM as I/O → hardware
  - Organized by label dependency, not tier difficulty
  - New introspection tools: weight profiler, channel profiler, routing extractor, logit lens, knowledge map

### Key Insights from Nemotron 3 Super Paper (NVIDIA, 2026-04-03)
- LatentMoE latent projection validates Carnot's universal encoder concept
- Expert routing patterns are a novel self-supervised feature source for hallucination detection
- Channel magnitude patterns in trained weights reveal knowledge structure without inference
- Multi-token prediction confidence is a temporal reasoning signal (no labels needed)
- Architectural diversity (Mamba + MoE + dense) makes cross-model consensus more meaningful
- The ARM↔EBM bijection means the weights already define the energy landscape — we don't need to train a second one

### Strategic Insight
The "everything" domain problem is solved by NOT requiring domain-specific labels. When features come from weight structure and model consensus rather than labeled examples, domain generalization is free — the features are inherently domain-agnostic.

### Model Acquisition
- Started download of `mistralai/Mixtral-8x7B-v0.1` (~93GB BF16 base model)
- Priority 1 model: unlocks 4 experiments (32 MoE weight profiling, 33 channel magnitude, 34 routing entropy, 38 consensus)

### Experiment Scripts
- `scripts/experiment_32_weight_profiling.py`: Pure weight analysis — effective rank, condition number, neuron norms, spectral gap, MoE expert specialization/overlap, router analysis. Zero inference needed.
- `scripts/experiment_33_channel_magnitude.py`: Nemotron-inspired FC1↔FC2 channel alignment analysis, dead channel detection, expert channel diversity. Zero inference needed.

## 2026-04-07: Multi-model EBM training, cross-model transfer (experiment 26)

### Added
- `scripts/train_ebm_multi_model.py`: Generalized pipeline for training EBMs across any HuggingFace model (15 models registered, auto-upload)
- `scripts/experiment_26_cross_model_transfer.py`: Cross-model transfer experiment
- `python/carnot/inference/ebm_loader.py`: Updated with all new model entries
- `exports/space-hallucination-detector/`: Gradio Space for interactive EBM scoring
- `exports/org-card/README.md`: HuggingFace organization card

### HuggingFace Published
- **8 EBM models** uploaded to Carnot-EBM: LFM2.5-350M, LFM2.5-1.2B, Bonsai-1.7B, Qwen3.5-2B, Qwen3.5-4B, Gemma4-E2B, Gemma4-E2B-it (+ 5 more training)
- **Activation datasets** uploaded to Carnot-EBM/token-activations
- **Interactive Space** at Carnot-EBM/hallucination-detector

### Key Results
- Experiment 26: Cross-model transfer at chance (~50%) — hallucination representations are model-specific
- **Principle 11**: No universal hallucination detector via activation analysis. Each model needs its own EBM.
- Gemma4-E2B base achieves highest EBM accuracy at 86.8% (confirms base models detect best)

Triggered by: user instruction to train EBMs for multiple models and investigate cross-model transfer.

---

## 2026-04-06: Ship MCP+CLI, thinking mode experiment (experiment 25)

### Added
- `python/carnot/cli.py`: Installable CLI module (`carnot verify` command)
- `examples/math_funcs.py`: Example functions for CLI testing
- `scripts/experiment_25_no_thinking.py`: Thinking vs no-thinking comparison
- `tests/python/test_cli.py`: 19 tests for CLI (parsing, type resolution, E2E verify)
- `pyproject.toml`: Added `[project.scripts]` entry point for `carnot` CLI command

### Key Results
- **Experiment 25**: Disabling thinking improves EBM detection from 61.3% → 75.5% (+14.2%)
- Energy gap 5.8x larger without thinking (2.4248 vs 0.4206)
- **Principle 10**: Chain-of-thought compresses hallucination signal. For detection, disable thinking.

### MCP Server Shipped
- 3 tools: `verify_code`, `verify_with_properties`, `score_candidates`
- `.mcp.json` registered, stdio transport, tested E2E with JSON-RPC
- CLI tested with correct and buggy functions, property-based testing

Triggered by: user instruction to ship MCP+CLI and investigate thinking mode.

---

## 2026-04-06: EBM rejection sampling, multi-layer probing, MCP server (experiments 23-24)

### Added
- `python/carnot/inference/ebm_rejection.py`: EBM-guided rejection sampling (REQ-INFER-015)
  - `EBMRejectionConfig`, `EBMCandidateScore`, `EBMRejectionResult`
  - `score_activations_with_ebm()`: scores per-token activations through trained EBM
  - `ebm_rejection_sample()`: generates N candidates, combines EBM + logprob, selects best
- `python/carnot/embeddings/layer_probing.py`: Multi-layer hallucination probing (REQ-INFER-016)
  - `train_layer_probe()`: trains a small Gibbs EBM probe at a single layer
  - `probe_all_layers()`: probes all layers and finds best
  - `extract_all_layer_activations()`: captures hidden states from all layers
- `tools/verify-mcp/server.py`: Added `score_candidates` tool for MCP-based candidate selection
- `scripts/experiment_23_ebm_rejection.py`: Experiment 23 (EBM rejection on TruthfulQA)
- `scripts/experiment_24_layer_probing.py`: Experiment 24 (multi-layer probing)
- 24 new tests in `test_ebm_rejection.py` and `test_layer_probing.py`
- REQ-INFER-015 and REQ-INFER-016 in llm-ebm-inference spec

### Key Results
- Experiment 23: EBM rejection sampling shows no improvement on adversarial QA (-3% to -6%)
- Experiment 24: Final layer IS the best probe layer (64%). U-curve: signal at layers 4 (60%) and 24 (64%), compressed mid-network.
- **Principle 9 discovered**: Adversarial questions defeat post-hoc detection. Detection must move upstream.

### Significance
Closes the loop on activation-based hallucination detection: we've proven it works on base models (84.5%), confirmed it's weaker on instruction-tuned models (67.2%), and shown it fails completely as a candidate filter on adversarial questions. The next frontier is upstream detection (analyzing questions, not answers).

Triggered by: user instruction to implement EBM rejection sampling, multi-layer probing, and ship MCP server.

---

## 2026-04-06: Documentation UI Modernization

### Added
- `openspec/capabilities/documentation-ui/spec.md`: Spec for Documentation UI
- `epics/stories/UI-001.md`: Epic for modernizing the documentation aesthetic
- `tests/python/test_docs.py`: Test asserting REQ-DOCUI-001 and REQ-DOCUI-002
- `scripts/update_index.py`: Script to apply CSS and HTML updates to `docs/index.html`

### Changed
- `docs/index.html`: Upgraded to a premium aesthetic (glassmorphism, depth, soft borders, refined typography, and fade-in animations).
- `_bmad/traceability.md`: Added FR-17 mapping to documentation UI capabilities.

### Significance
Elevates the open-source documentation page to reflect the sophisticated nature of Carnot's EBM tech, matching top-tier AI projects with fluid micro-interactions and depth.

Triggered by: user instruction to improve the design aesthetic of the documentation website.

---

## 2026-04-06: TruthfulQA + Qwen3.5-0.8B activation experiments (experiments 21-22)

### Added
- `scripts/collect_truthfulqa_activations.py`: Collects per-token activations from Qwen3.5-0.8B on 817 TruthfulQA adversarial questions (53% accuracy, 29,058 tokens)
- `scripts/collect_qa_activations_qwen35.py`: Re-collects QA dataset activations using Qwen3.5-0.8B (57% accuracy, 23,238 tokens)
- `scripts/merge_activations_qwen35.py`: Merges QA + TruthfulQA from same model (52,296 tokens total)
- `scripts/train_per_token_ebm_combined.py`: Training script with `--source` flag (qa/tqa/both/merged)
- `data/token_activations_qwen35_merged.safetensors`: 52,296 tokens from Qwen3.5-0.8B

### Key Results
- Experiment 21: Qwen3-0.6B QA (26,800 tokens) → 84.5% test (confirmed)
- Experiment 22: Qwen3.5-0.8B merged (52,296 tokens) → 67.2% test
- **Principle 8 discovered**: Instruction tuning compresses the hallucination signal. Base models (84.5%) have larger activation gaps than instruction-tuned models (67.2%). RLHF makes models sound confident even when wrong.

### Significance
Demonstrates that the models most in need of hallucination detection are the hardest to detect on via activation analysis alone. Future work should combine activation features with logprobs, attention patterns, and logit lens approaches.

Triggered by: user instruction to add TruthfulQA and use Qwen3.5-0.8B with thinking.

---

## 2026-04-05: Hallucination direction detection via activation-space analysis

### Added
- `python/carnot/embeddings/hallucination_direction.py`: `find_hallucination_direction()` (mean-difference + SVD), `hallucination_energy()` (projection-based scalar energy), `HallucinationDirectionConstraint` (BaseConstraint for ComposedEnergy), `HallucinationDirectionConfig`
- 35 tests in `tests/python/test_hallucination_direction.py` covering config validation, direction discovery, energy computation, constraint integration, and package exports
- REQ-INFER-014 and SCENARIO-INFER-014-001 in llm-ebm-inference spec
- Exported all new symbols from `carnot.embeddings`

### Significance
Given per-layer activations from correct vs hallucinated LLM outputs, discovers the principal direction separating them and turns it into a differentiable energy constraint. This direction becomes a real-time hallucination detector composable with other Carnot constraints.

Triggered by: user instruction to implement hallucination direction detection.

---

## 2026-04-04: Self-improving Python code verifier (capstone)

### Added
- **Code verification** (`verify/python_types.py`): `ReturnTypeConstraint`, `NoExceptionConstraint`, `TestPassConstraint`, `code_to_embedding()`, `safe_exec_function()`, `build_code_energy()`
- **Learned code verifier** (`inference/code_verifier.py`): `train_code_verifier()` via NCE on code embeddings, `verify_python_function()` full pipeline, `generate_code_training_data()` with template mutations
- **Self-improving loop** (`autoresearch/code_improvement.py`): `run_code_verification_autoresearch()` — autoresearch improving code verification accuracy via hypothesis generation
- REQ-CODE-001 through REQ-CODE-005 in new code-verification spec
- 53 new tests across 3 test files

### Significance
This is the capstone: EBM verifies Python code, and autoresearch improves the verifier. Proves the full thesis — energy-based verification + directed self-learning as the antidote to LLM hallucination.

---

## 2026-04-04: Learned energy functions — train EBMs to verify from examples

### Added
- `python/carnot/inference/learned_verifier.py`: `generate_sat_training_data()` (rejection sampling), `train_sat_verifier()` (NCE training loop), `LearnedEnergyWrapper` (BaseConstraint adapter), `build_learned_sat_energy()`, `compare_learned_vs_handcoded()`
- REQ-INFER-007 + SCENARIO-INFER-008 in spec
- 18 tests: data generation, training, wrapping, comparison, edge cases

### Significance
This is the strategic leap: instead of hand-coding constraints (SAT clauses), the EBM LEARNS what "correct" looks like from examples. Same pattern scales to code verification — replace SAT pairs with (correct_code, buggy_code) → learned code verifier.

---

## 2026-04-04: LLM solver integration for SAT/coloring pipeline

### Added
- `python/carnot/inference/llm_solver.py`: `LLMSolverConfig`, `solve_sat_with_llm()`, `solve_coloring_with_llm()`, `run_llm_sat_experiment()`, `run_llm_coloring_experiment()`
- SAT/coloring prompt construction for LLM (`_build_sat_prompt`, `_build_coloring_prompt`)
- Full end-to-end pipeline: LLM call → parse → verify → repair → certify
- Graceful degradation (missing openai, API failure, parse failure)
- REQ-INFER-006 + SCENARIO-INFER-007 in spec
- 16 new tests with mocked LLM calls

---

## 2026-04-04: Gradient clipping for samplers (fixes Rosenbrock NaN blocker)

### Added
- `clip_norm: float | None = None` on `LangevinSampler` and `HMCSampler`
- `_clip_gradient()` — rescales gradient L2 norm to <= clip_norm, preserving direction
- Clipping in Langevin `sample()`, `sample_chain()`, and HMC `_leapfrog()`
- REQ-SAMPLE-004 + SCENARIO-SAMPLE-004/005 in training-inference spec
- 8 new tests: activation, no-op, backward compat, Rosenbrock NaN prevention

### Fixed
- **Rosenbrock divergence**: `clip_norm=10.0` produces finite samples (energy 4.09 Langevin, 1.28 HMC) where unclipped diverged to NaN (grad norm ~4950)

---

## 2026-04-04: LLM-EBM inference — SAT/CSP verify-and-repair pipeline (user instruction: easiest domain for LLM+EBM anti-hallucination)

### Added
- **SAT constraints** (`python/carnot/verify/sat.py`): `SATClauseConstraint` using product relaxation, `SATBinaryConstraint`, `build_sat_energy()`, DIMACS CNF parser. REQ-INFER-001.
- **Graph coloring constraints** (`python/carnot/verify/graph_coloring.py`): `ColorDifferenceConstraint` (pairwise repulsion), `ColorRangeConstraint`, `build_coloring_energy()`. REQ-INFER-002.
- **Inference bridge** (`python/carnot/inference/verify_and_repair.py`): LLM output parsers (SAT + coloring, multiple formats), `verify_and_repair()` pipeline (parse → verify → repair → round → certify). REQ-INFER-003, REQ-INFER-004.
- **Benchmark harness** (`python/carnot/inference/benchmark.py`): Random SAT/graph instance generators, `run_sat_benchmark()`, `run_coloring_benchmark()`. REQ-INFER-005.
- **New capability spec**: `openspec/capabilities/llm-ebm-inference/` with 5 requirements and 6 scenarios.
- **3 new test files** (64 tests): Full coverage of all new modules.

### Quality
- 462 tests passing, 100% code coverage, 100% spec coverage
- All ruff, mypy, ruff format checks pass

---

## 2026-04-04: Trace2Skill integration — deep trajectory analysis for autoresearch (user instruction: incorporate ideas from arxiv 2603.25158)

### Added
- **Trajectory analyst** (`python/carnot/autoresearch/trajectory_analyst.py`): Parallel error/success analyst sub-agents that extract structured `Lesson` objects from experiment trajectories via LLM reasoning. REQ-AUTO-011.
- **Skill directory** (`python/carnot/autoresearch/skill_directory.py`): Persistent optimization playbook (SKILL.md + lessons.json + scripts/ + references/) that replaces shallow `recent_failures` list. Cross-tier transfer (Ising→Gibbs→Boltzmann). REQ-AUTO-012, REQ-AUTO-014.
- **Consolidator** (`python/carnot/autoresearch/consolidator.py`): Hierarchical tree-reduction merge of lessons via LLM. Deduplicates, resolves conflicts, filters low-confidence. REQ-AUTO-013.
- **`run_loop_with_skills()`** in orchestrator: New loop variant that dispatches analysts, consolidates periodically, and injects skill context into generator prompts.
- **4 new test files** (85+ tests total): Full coverage of all new modules.
- **4 new requirements** (REQ-AUTO-011–014) and **4 new scenarios** (SCENARIO-AUTO-008–011) in spec.
- **Design doc** updated with Stage 1.5: ANALYZE architecture diagram and Trace2Skill section.

### Changed
- `ExperimentEntry` gains `lessons` field for storing extracted lessons per experiment
- `DEFAULT_SYSTEM_PROMPT` in hypothesis_generator.py now includes Skill Playbook guidance
- `AutoresearchConfig` gains skill directory, analyst, and consolidation settings
- `__init__.py` exports all new types and functions

### Quality
- 398 tests passing, 100% code coverage, 100% spec coverage
- All ruff, mypy, ruff format checks pass

---

## 2026-04-04: Session handoff — autoresearch proven, all E2E debts cleared

### Summary
Full session: Gibbs JAX, PyO3 tests, Claude API bridge, LLM hypothesis generator, 5 benchmark energy functions, adversarial reviewer agent, E2E training+sampling tests, E2E serialization tests, JIT timing fix, 10-iteration autoresearch run with Sonnet. DoubleWell energy reduced 83% (0.95→0.16) via LLM-proposed improvements. Rosenbrock NaN identified as gradient clipping gap — next session priority.

### Commits
- `77e63d6` — Gibbs JAX, PyO3 tests, Claude API bridge, LLM autoresearch, benchmarks
- `41b3123` — Adversarial reviewer agent + close all review gaps
- `b8a0481` — E2E tests: training+sampling pipeline and serialization round-trip
- `7b5ab9f` — JIT grace period + 10-iteration Sonnet autoresearch run

---

## 2026-04-03: Gibbs JAX + PyO3 Tests + Claude API Bridge + LLM Autoresearch (user instruction: implement Gibbs JAX, PyO3 tests, real autoresearch with LLM)

### Added
- **Gibbs Python/JAX model** (`python/carnot/models/gibbs.py`): Full `GibbsConfig` + `GibbsModel` with SiLU/ReLU/Tanh activations, multi-layer dense energy network, AutoGradMixin for auto-differentiation. 20 tests in `test_models_gibbs.py`.
- **PyO3 integration tests** (`tests/python/test_pyo3_integration.py`): 24 tests covering all 3 Rust model tiers + both samplers via `carnot._rust`. Validates end-to-end Rust↔Python bridge.
- **Claude Code API bridge** (`tools/claude-api-bridge/`): FastAPI server + Dockerfile wrapping `claude -p` as OpenAI-compatible API. Supports streaming SSE, non-streaming JSON, `--mcp-config` for tool use, session management. Tested with Docker + OpenAI Python SDK.
- **LLM hypothesis generator** (`python/carnot/autoresearch/hypothesis_generator.py`): `GeneratorConfig`, `generate_hypotheses()`, `generate_hypotheses_batch()` using OpenAI SDK against any compatible endpoint.
- **Generator-based orchestrator** (`run_loop_with_generator()` in orchestrator.py): Lazy hypothesis generation with failure feedback loop. Backwards-compatible with existing `run_loop()`.
- **LLM autoresearch demo** (`scripts/run_autoresearch_llm.py`): End-to-end script connecting LLM → sandbox → evaluator. Verified working with Claude Haiku and Sonnet via API bridge.
- 27 new tests for hypothesis generator and generator-based loop.

### Added (continued)
- **Benchmark energy functions** (`python/carnot/benchmarks/`): All 5 analytical benchmarks (DoubleWell, Rosenbrock, Ackley, Rastrigin, GaussianMixture) as JAX EnergyFunction classes with AutoGradMixin. Known global minima for quantitative evaluation. 33 tests. Wired into autoresearch pipeline — baselines now computed from real mathematical landscapes.

### Fixed
- **PyO3 module name mismatch**: Renamed `#[pymodule] fn carnot_python` → `fn _rust` in `crates/carnot-python/src/lib.rs` to match `pyproject.toml`'s `module-name = "carnot._rust"`.
- **Ackley gradient NaN at origin**: Added epsilon in sqrt to prevent jax.grad NaN from d/dx sqrt(0).

### Updated
- `python/carnot/models/__init__.py`: exports `GibbsConfig, GibbsModel`
- `python/carnot/autoresearch/__init__.py`: exports `run_loop_with_generator`

### Test Results
- Python: 237 tests + 24 PyO3 integration tests, 100% code coverage
- Rust: 100 tests, all pass
- Real autoresearch run: 3 iterations with Sonnet, all 3 accepted, real Carnot sampler code executed in sandbox

---

## 2026-04-03: Spec Reconciliation (user instruction: reconcile specs with reality)

### Updated
- **All 5 OpenSpec Implementation Status tables** reconciled with actual code/test state
- **Traceability matrix** (`_bmad/traceability.md`): FR-08 Not Started → Partial, FR-11 Spec'd → Partial, FR-12 Spec'd → Implemented, test counts updated, NFR statuses updated
- **ops/status.md**: comprehensive update reflecting all implemented features and remaining gaps
- Added **spec-reconciler agent** (`.claude/agents/spec-reconciler.md`) and `/reconcile-specs` command to prevent future spec drift

### Key discrepancies found and fixed
- 24 requirements were implemented but specs still claimed "Not Started"
- FR-08 (PyO3 interoperability) had full bindings but traceability said "Not Started"
- FR-11 (autoresearch) had sandbox, evaluator, orchestrator, Docker sandbox but traceability said "Spec'd"
- FR-12 (verifiable reasoning) had 12 of 14 requirements implemented but traceability said "Spec'd"

---

## 2026-04-03: Docker+gVisor Sandbox (user instruction: use Docker+gVisor for sandbox)

### Added
- `Dockerfile.sandbox`: minimal Python+JAX+carnot image for isolated hypothesis execution
- `scripts/sandbox_runner.py`: in-container harness for hypothesis execution
- `python/carnot/autoresearch/sandbox_docker.py`: Docker+gVisor sandbox backend with 5 defense layers (gVisor, no network, read-only FS, memory/CPU limits, timeout)
- 21 new Python tests for Docker sandbox

---

## 2026-04-03: Autoresearch Orchestrator (user instruction: implement autoresearch orchestrator)

### Added
- `python/carnot/autoresearch/orchestrator.py`: `run_loop()` — full propose → sandbox → evaluate → log → update pipeline
- `python/carnot/autoresearch/experiment_log.py`: append-only experiment log with rejected registry and circuit breaker
- `scripts/demo_autoresearch.py`: end-to-end demo showing 90% DoubleWell and 80% Rosenbrock improvement
- 20 new Python tests

---

## 2026-04-03: Comprehensive Documentation (user instruction: add verbose layman docs)

### Added
- 4,475 lines of inline documentation across 18 files (Rust + Python)
- Two-tier format: terse researcher summary + detailed engineer explanation
- Every public type, trait, function documented with examples and analogies

---

## 2026-04-03: CI Fixes + Security Agent (user instruction: fix CI failures, add security agent)

### Fixed
- rustfmt: 10 files reformatted
- clippy: 7 warnings fixed (unused imports, derives, assign patterns)
- Flaky Langevin statistics test: increased samples and tolerance

### Added
- Security auditor agent + `/security-audit` command
- SOPS configuration for encrypted secrets at rest
- Gitea CI workflow (5 parallel jobs)

---

## 2026-04-03: Autoresearch Sandbox + Score Matching (user instruction: implement #2 and #4 in parallel)

### Added
- Process-level sandbox: import blocking, SIGALRM timeout, I/O capture
- Three-gate evaluator: energy, time, memory gates
- Baseline registry with JSON persistence
- Denoising score matching training (Rust + Python/JAX)
- 37 new Python tests

---

## 2026-04-03: PyO3 Bindings (user instruction: implement PyO3 bindings)

### Added
- RustIsingModel, RustGibbsModel, RustBoltzmannModel exposed via PyO3
- RustLangevinSampler, RustHMCSampler with per-model sample methods
- Zero-copy numpy array transfer via PyReadonlyArray

---

## 2026-04-03: Analytical Backprop (user instruction: implement analytical backprop)

### Fixed
- Gibbs tier: replaced finite-difference gradients with analytical backprop (SiLU, ReLU, Tanh)
- Boltzmann tier: replaced finite-difference with backprop through residual blocks

---

## 2026-04-03: Python Tests + Benchmarks + Agent Team

### Added
- 48 Python tests achieving 100% coverage (from 0)
- Benchmark suite: DoubleWell, Rosenbrock, Ackley, Rastrigin, GaussianMixture
- Benchmark runner with baseline recording
- 5 E2E integration tests (sampler + benchmark)
- Agent team: test-runner, lint-checker, spec-validator, evaluator, docs-keeper

---

## 2026-04-03: Verifiable Reasoning + Specs (user instruction: spec and implement autoresearch/verify)

### Added
- OpenSpec specs: autoresearch (10 REQs), verifiable-reasoning (7 REQs)
- ConstraintTerm trait, ComposedEnergy, VerificationResult, gradient-based repair
- Sudoku constraint satisfaction example (Rust + Python)
- 17 Rust + 12 Python verification tests

---

## 2026-04-03: Project Bootstrap (user instruction: initial project setup)

### Added
- BMAD strategic documents: PRD, architecture, traceability
- OpenSpec capability specs: core-ebm, model-tiers, training-inference
- Rust workspace with 7 crates
- Python/JAX package with core abstractions, Ising model, samplers
- Pre-commit hooks, spec coverage script
- README with anti-hallucination framing and self-learning vision
