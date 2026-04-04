# Carnot — Test Results

**Last Updated:** 2026-04-03

## Latest Run (2026-04-03, post-adversarial-review)

| Suite | Status | Count | Coverage | Notes |
|-------|--------|-------|----------|-------|
| Rust unit tests | PASS | 100 (96 unit + 4 doc) | N/A | `cargo test --workspace --exclude carnot-python` |
| Rust clippy | PASS | 0 warnings | N/A | `cargo clippy --workspace --exclude carnot-python -- -D warnings` |
| Rust fmt | PASS | 0 issues | N/A | `cargo fmt --all -- --check` |
| Python unit tests | PASS | 270 | 100% | `pytest tests/python/ --cov-fail-under=100` (excludes PyO3) |
| PyO3 integration | PASS | 24 | N/A | `pytest tests/python/test_pyo3_integration.py` |
| Python ruff | PASS | 0 issues | N/A | `ruff check python/ tests/` |
| Python mypy | PASS | 0 errors | N/A | `mypy python/carnot` |
| Spec coverage | PASS | 100% | N/A | `python scripts/check_spec_coverage.py` — all tests trace to REQ-*/SCENARIO-* |
| Security audit | CLEAN | N/A | N/A | No secrets, no unsafe, SOPS compliant |

**Total: 408 tests (100 Rust + 284 Python + 24 PyO3), 100% code coverage, 100% spec coverage**

## E2E Test Evidence

### E2E-003: PyO3 Binding Round-Trip (PASS)
- `tests/python/test_pyo3_integration.py` — 24 tests
- All 3 Rust model tiers (Ising, Gibbs, Boltzmann) created from Python
- Energy, energy_batch, grad_energy called from Python on Rust models
- Both samplers (Langevin, HMC) run from Python on all 3 Rust tiers
- Error handling verified (invalid activation raises Python ValueError)

### E2E: Claude API Bridge (PASS — manual verification)
- Docker image built and run: `docker build -t claude-api-bridge .`
- Health check: `GET /health` returned `{"status":"ok"}`
- Non-streaming: `POST /v1/chat/completions` returned correct OpenAI-format JSON
- Streaming: SSE chunks with correct `data: {...}` format, no duplication
- OpenAI Python SDK: both `create()` and `create(stream=True)` worked
- OAuth credentials mounted via `-v ~/.claude:/root/.claude:ro`

### E2E: Autoresearch with LLM (PASS — manual verification)
- `scripts/run_autoresearch_llm.py` executed against Claude API bridge
- 3 iterations with Sonnet model
- LLM generated real Carnot sampler code (HMC with step_size=0.05)
- Sandbox executed code against real benchmark energy functions (DoubleWell, Rosenbrock)
- Evaluator correctly identified improvements and regressions
- Mixed results → REVIEW verdict for Hypothesis 3

### E2E-002: Training + Sampling Pipeline (PASS — automated)
- `tests/python/test_e2e_training_sampling.py` — 5 tests
- Langevin finds DoubleWell minimum (energy decreases, x[0] near +/-1)
- Langevin chain explores (non-degenerate trajectory over 2000 steps)
- Rosenbrock convergence (energy decreases from origin toward minimum)
- DSM training reduces loss (gradient descent on parameterized model center)
- Full pipeline: train model center → sample → verify samples cluster near target

### E2E-004: Serialization Cross-Language (PASS — automated)
- `tests/python/test_e2e_serialization.py` — 9 tests
- Python round-trip: Ising, Gibbs, Boltzmann params survive save/load via safetensors
- safetensors format: preserves shapes (1D, 2D), preserves f32 dtype
- JAX ↔ NumPy interop through safetensors verified
- PyO3 cross-language: Rust and Python Ising/Gibbs produce finite energy on same input

### E2E-001: Training + Sampling (Rust) — NOT YET AUTOMATED
- Rust training pipeline E2E not yet in automated test suite
- Covered partially by Rust unit tests in carnot-training crate

## Known Gaps
- E2E-001 (Rust training pipeline) not yet automated as integration test
- Docker API bridge was tested manually, not in CI
- Autoresearch E2E was run interactively, not as a repeatable test
