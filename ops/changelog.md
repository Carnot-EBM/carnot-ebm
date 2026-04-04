# Carnot — Changelog

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
