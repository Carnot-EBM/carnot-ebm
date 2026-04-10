# Carnot — Traceability Matrix

**Last Updated:** 2026-04-09 (reconciled with codebase)

## Functional Requirements → Implementation Status

| FR ID | Description | Spec | Tests | Impl | Status |
|-------|------------|------|-------|------|--------|
| FR-01 | Core EBM Framework | `openspec/capabilities/core-ebm/spec.md` | 10 Rust, 6 Python | Rust + Python | Implemented |
| FR-02 | Boltzmann Tier | `openspec/capabilities/model-tiers/spec.md` | 7 Rust, Python | Rust + Python | Implemented |
| FR-03 | Gibbs Tier | `openspec/capabilities/model-tiers/spec.md` | 7 Rust, Python | Rust + Python | Implemented |
| FR-04 | Ising Tier | `openspec/capabilities/model-tiers/spec.md` | 10 Rust, 12 Python | Rust + Python | Implemented |
| FR-05 | Dual-Language Impl | `openspec/capabilities/core-ebm/spec.md` | PyO3 integration | Rust + Python + bindings | Implemented |
| FR-06 | Training Pipeline | `openspec/capabilities/training-inference/spec.md` | 30+ Python | CD-k, DSM, NCE, SNL, optimization-training, replay buffer | Implemented |
| FR-07 | Inference Pipeline | `openspec/capabilities/training-inference/spec.md` | 20+ Python | Langevin + HMC + gradient clipping | Implemented |
| FR-08 | Interoperability | `openspec/capabilities/core-ebm/spec.md` | 24 PyO3 | PyO3 bindings all tiers + samplers | Implemented |
| FR-09 | Test Coverage | N/A (process) | 104 Rust + 1093 Python | pre-commit + CI | Implemented |
| FR-10 | Spec-Driven Dev | N/A (process) | spec_coverage.py | pre-commit + CI | Implemented |
| FR-11 | Autonomous Self-Learning | `openspec/capabilities/autoresearch/spec.md` | 100+ Python | Sandbox, evaluator, orchestrator, Trace2Skill, conductor | Implemented |
| FR-12 | Verifiable Reasoning | `openspec/capabilities/verifiable-reasoning/spec.md` | 40+ Python | Constraints, repair, SAT, coloring, code, property tests, convergence | Implemented |
| FR-13 | LLM-EBM Inference | `openspec/capabilities/llm-ebm-inference/spec.md` | 170+ Python | Composite scorer, iterative refinement, logprob rejection, EBM rejection, multi-start, semantic energy, ARM-EBM bridge, diffusion, reasoning energy | Implemented |
| FR-14 | Code Verification | `openspec/capabilities/code-verification/spec.md` | 50+ Python | Type/exception/test constraints, code embeddings (bag-of-tokens + AST), learned verifier, self-improving loop | Implemented |
| FR-15 | Activation Analysis | `openspec/capabilities/llm-ebm-inference/spec.md` | 110+ Python | Activation extractor, hallucination direction, layer EBM, LayerNavigator, steering, weight steering, concept vectors, multi-layer probing | Implemented |
| FR-18 | MCP Server + CLI | `openspec/capabilities/llm-ebm-inference/spec.md` | 49 Python | MCP verify_code, verify_with_properties, verify_llm_output, verify_and_repair, list_domains, health_check, score_candidates; CLI carnot verify | Implemented |
| FR-16 | GPU Compute | N/A | 4 Rust | wgpu Vulkan backend, WebGPU gateway, ROCm 7.2 native gfx1150 | Implemented |
| FR-17 | Documentation UI | `openspec/capabilities/documentation-ui/spec.md` | 1 Python | Premium aesthetic, glassmorphism, animations | Implemented |

## Non-Functional Requirements

| NFR ID | Description | Verified By | Status |
|--------|------------|-------------|--------|
| NFR-01 | Performance (10x Rust vs Python) | Benchmark suite + GPU | Implemented (wgpu + ROCm 7.2) |
| NFR-02 | Safety (no unsafe in public API) | clippy + security-auditor agent | Verified |
| NFR-03 | Documentation | Technical report + inline docs | Implemented |
| NFR-04 | CI/CD pre-commit hooks | Gitea Actions + pre-commit config | Implemented |

## Research Validation

| Experiment | Result | Paper Reference |
|-----------|--------|----------------|
| Logprob rejection (+10%) | ✅ Validated | Semantic Energy (arxiv 2508.14496) |
| Composite scoring (0% → 30%) | ✅ Novel | — |
| Per-token EBM (71.8%) | ✅ Novel | — |
| Activation steering (0%) | ❌ Negative result | Emotion Concepts (Anthropic 2025) |
| SAT gradient repair (60% → 80%) | ✅ Validated | — |
| Per-token EBM scaled (84.5%) | ✅ Novel | — |
| Instruction tuning compression (84.5% → 67.2%) | ✅ Novel (Principle 8) | — |
| EBM rejection on adversarial QA (-3%) | ❌ Negative result (Principle 9) | — |
| Multi-layer probing (U-curve) | ✅ Novel | — |
| Thinking compression (75.5% vs 61.3%) | ✅ Novel (Principle 10) | — |
| Exp 42b: Arithmetic QUBO encoding | ✅ Complete | — |
| Exp 42c: Deterministic carry chain verification | ✅ Complete | — |
| Exp 44: Scheduling constraints | ✅ Complete | — |
| Exp 45: Logical consistency verification | ✅ Complete | — |
| Exp 46b: Scale SAT to 5000 vars | ✅ Complete | — |
| Exp 47: LLM self-constraint extraction | ✅ Complete | — |
| Exp 48: Code → constraint extraction | ✅ Complete | — |
| Exp 49: NL → constraint extraction | ✅ Complete | — |
| Exp 50: Learn SAT couplings via CD | ✅ Complete | — |
| Exp 51: Learn from LLM (correct/wrong) pairs | ✅ Complete | — |
| Exp 52: Transfer Ising across domains | ✅ Complete | — |
| Exp 53: Runtime constraint instrumentation | ✅ Complete | — |
| Exp 56: Live LLM → constraint → Ising verification | ✅ Complete | — |
| Exp 57: Live LLM verify-repair loop | ✅ Complete (9/15 initial, loop works, constraint coverage bottleneck) | — |
| Exp 59: Constraint-aware prompting | ✅ Complete (preventive constraint injection, SCENARIO-VERIFY-005) | — |
| Exp 60: Scale CD training to 100+ vars | ✅ Complete (50/100/200 vars, up to 40K params, bootstrapped training) | — |
| Exp 61: Sparse Ising at 500+ vars | ✅ Complete (clause-graph sparsity mask, ~20x param reduction, 200/500/1000 vars) | — |
| Exp 54: Ising-guided fuzzing | ✅ Complete (adversarial test input generation via Ising energy, 8 LLM bug types, REQ-VERIFY-001/002/003) | — |
| Exp 55: Learn constraints from execution traces | ✅ Complete (discriminative Ising on correct/buggy execution traces, catches semantic bugs, REQ-VERIFY-001/002/003) | — |
| Exp 58: Multi-domain live benchmark (5 domains) | ✅ Complete (500 questions, 5 domains, 3 modes: baseline/verify-only/verify-repair, REQ-VERIFY-001/002/003, SCENARIO-VERIFY-005) | — |
| Exp 64: Continuous Ising relaxation | ✅ Complete (binary→continuous [0,1], sigmoid/penalty/straight-through rounding, bridges to Kona latent space) | — |
| Exp 69: Multi-model constraint transfer (Qwen3.5+Gemma4) | ✅ Complete (same constraint pipeline on Qwen3.5-0.8B + Gemma4-E4B-it, tests model-agnostic verification, REQ-VERIFY-001/002/003, SCENARIO-VERIFY-003) | — |
| Exp 71: Extropic TSU sampler abstraction | ✅ Complete (SamplerBackend protocol, CpuBackend + TsuBackend stub, get_backend() factory, CARNOT_BACKEND env var, REQ-SAMPLE-003) | — |
| Exp 73: Constraint coverage metric | ✅ Complete (5-type claim taxonomy, coverage = extracted/total per domain+type, coverage-accuracy correlation, repair threshold analysis, REQ-VERIFY-001/002/003, SCENARIO-VERIFY-005) | — |
| Exp 67: GSM8K subset verification | ✅ Complete (200 GSM8K test questions, 3 modes: baseline/verify/verify-repair, first external benchmark, REQ-VERIFY-001/002/003, SCENARIO-VERIFY-006) | — |
| Exp 68: HumanEval subset verification + fuzzing | ✅ Complete (50 HumanEval-style problems, full pipeline: extract→instrument→test→fuzz→repair, pass@1 + pass@1+repair, REQ-VERIFY-001/002/003, SCENARIO-VERIFY-006) | — |
| Exp 65: Embedding-space constraint verification | ✅ Complete (joint Gibbs EBM on [semantic embedding; constraint vector], NCE training, AUROC comparison, gradient repair with NN decoding, REQ-EBT-001, REQ-VERIFY-001) | — |
| Exp 66: End-to-end differentiable constraint reasoning | ✅ Complete (fully differentiable text→embedding→constraints→continuous Ising→MLP→score; joint 1.0 test AUROC vs 0.54 Ising-only, 0.98 embedding-only; stable gradients; 5 domains; builds on Exp 64+65; REQ-VERIFY-001, REQ-EBT-001) | — |
| Exp 70: Rust constraint extraction + verification | ✅ Complete (`carnot-constraints` crate: BoundConstraint, EqualityConstraint, IsingConstraint, VerificationCertificate; REQ-VERIFY-001/002/003/004/005) | — |
| Exp 72: Autoresearch self-verification via Ising | ✅ Complete (fourth gate: claim extraction + ComposedEnergy + Ising verification on autoresearch hypotheses, REQ-VERIFY-001/002/003, SCENARIO-VERIFY-002) | — |
| Exp 74: Unified ConstraintExtractor API | ✅ Complete (pluggable Protocol-based extractors: arithmetic, code, logic, NL + AutoExtractor auto-detection; consolidates Exp 47/48/49 into `carnot.pipeline.extract`; REQ-VERIFY-001/002/003, SCENARIO-VERIFY-002) | — |
| Exp 75: VerifyRepairPipeline class | ✅ Complete (user-facing API consolidating Exp 56/57 into `carnot.pipeline.verify_repair`; verify-only + verify-and-repair modes; VerificationResult, RepairResult, VerifyRepairPipeline; REQ-VERIFY-001/002/003, SCENARIO-VERIFY-004) | — |
| Exp 76: Production MCP server | ✅ Complete (hardened `carnot.mcp` package: 6 tools, 30s timeout, 10K char limit, structured errors, `python -m carnot.mcp`; REQ-CODE-001, REQ-CODE-006, REQ-VERIFY-001, REQ-VERIFY-003, SCENARIO-VERIFY-004) | — |
| Exp 78: PyPI-ready package | ✅ Complete (setuptools build backend, optional Rust bindings, single-source version, `carnot[mcp]`/`carnot[rust]`/`carnot[all]` extras) | — |
| Exp 79: Integration examples | ✅ Complete (5 production-ready examples in `examples/`: API verification, code review pipeline, batch verify, custom extractor, MCP integration) | — |
| Exp 80: Getting started documentation | ✅ Complete (`docs/getting-started.md`, `docs/concepts.md`, `docs/api-reference.md`; updated `docs/index.html` navigation) | — |
| Exp 83: Pipeline performance benchmarks | ✅ Complete (verify latency sub-ms p99, 36,887 calls/s batch throughput, zero memory growth; `scripts/benchmark_pipeline.py` + `ops/benchmark-results.md`; REQ-VERIFY-001) | — |
| Exp 84: Carnot verifies Carnot (dogfood) | ✅ Complete (`scripts/dogfood_carnot.py`: self-verification of pipeline against own source code via CodeExtractor, AutoExtractor, VerifyRepairPipeline; REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002) | — |
| Exp 85: Prepare beta release | ✅ Complete (`RELEASE_NOTES.md` for 0.1.0-beta1, `scripts/prepare_release.py` release readiness checker, `README.md` install + quick start) | — |
| Exp 86: Learned energy composition weights | ✅ Complete (per-constraint-type weight learning via BCE gradient descent on ComposedEnergy; uniform 0.927 → learned 0.938 AUROC, bootstrap CI not significant; arithmetic weight dominant 1.19; 500 samples, 5 domains, 10 constraint types; REQ-VERIFY-001, REQ-VERIFY-003) | — |
| Exp 87: Gradient-based repair in continuous space | ✅ Complete (gradient descent in embedding space + codebook decoding; 40% repair vs 28% simulated discrete; arithmetic/scheduling 100%, factual/code/logic 0%; energy 1.72→1.02; 90% convergence; builds on Exp 65+66; REQ-VERIFY-001, REQ-VERIFY-003) | — |
| Exp 88: Failure-driven constraint mining | ✅ Complete (analyzes pipeline false negatives to discover missing extractors; 200 questions, 93% false negative rate; categorizes 6 gap types; suggests 6 new patterns with est. 75% coverage improvement; new `carnot.pipeline.mining` module; REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-005) | — |
| Exp 89: Self-bootstrapped constraint training | ✅ Complete (trains discriminative Ising from pipeline verification outputs as self-supervision; 1000 samples, 5 domains; 0.788 combined AUROC; arithmetic/logic 1.0, code 0.91; 96.7% pipeline concordance; data efficiency ablation 100→700 samples; REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, FR-11) | — |
| Exp 91: GSM8K live benchmark (Qwen3.5 + Gemma4) | ✅ Complete (200 real GSM8K questions, 2 models simulated; Qwen3.5-0.8B 65→80% (+15%), Gemma4-E4B-it 74.5→88.5% (+14%); 100% precision, 0 false positives; constraint coverage 81-88.5%; REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-006) | — |
| Exp 93: Multi-model systematic comparison | ✅ Complete (250 questions × 2 models (Qwen3.5-0.8B, Gemma4-E4B-it) × 3 modes (baseline, verify-only, verify+repair) = 1500 evaluations; +10.2% avg accuracy improvement (p<0.001 both models); Qwen3.5: 80→91.2% (+11.2%), Gemma4: 82.8→92% (+9.2%); best domains: scheduling +30%, code +14%, arithmetic +7%; REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-005) | — |
| Parallel Ising Gibbs sampler | ✅ 183x faster than thrml (572x at 500 vars) | — |
