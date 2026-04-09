# Carnot Research Roadmap v9: Ship It — Productionize the Constraint Pipeline

**Created:** 2026-04-09
**Milestone:** 2026.04.4
**Status:** Planned (activates when milestone 2026.04.3 completes)
**Supersedes:** research-roadmap-v8.md (milestone 2026.04.3)
**Informed by:** Production readiness assessment, experiments 47-64

## What v7+v8 Proved

The research validated the core thesis:
- **LLM proposes, Ising verifies, repair loop fixes** — 19/20 accuracy, 100% hallucination detection
- **Verify-repair improves answers** — ~27% accuracy improvement on tricky questions
- **Constraint extraction works** across code (AST), natural language (patterns), and LLM self-reports
- **Parallel Ising sampler** — 183x faster than thrml, scales to 5000 vars
- **Learned Ising models generalize** — CD training at 100+ vars, sparse at 500+

## The Problem

The pipeline works but lives in experiment scripts, not importable library code.
A developer cannot:
- `pip install carnot` and use it
- Call a single function to verify an LLM output
- Integrate constraint verification into their application
- Use the MCP server safely in production

## v9 Goal: Ship a Usable Product

Turn proven research into a library that developers can install and use in 5 minutes.

```python
# What a developer should be able to do after v9:
from carnot import VerifyRepairPipeline

pipeline = VerifyRepairPipeline(
    model="Qwen/Qwen3.5-0.8B",  # or any HF model, or None for API
    domains=["arithmetic", "logic", "code"],
)

result = pipeline.verify("What is 47 + 28?", response="The answer is 75.")
# result.verified = True, result.constraints = [...], result.energy = 0.0

result = pipeline.verify_and_repair(
    "What is 97 + 86?",
    response="The answer is 173.",
    max_repairs=3,
)
# result.final_answer = "The answer is 183."
# result.repaired = True, result.iterations = 1
```

## Phase 13: Extract Pipeline Library (experiments 74-77)

### Exp 74: Unified ConstraintExtractor API
Extract constraint extraction from experiment scripts into a pluggable library.
- Create `carnot.pipeline.extract` module
- `ConstraintExtractor` protocol with `extract(text, domain) -> list[Constraint]`
- Implementations: `ArithmeticExtractor`, `LogicExtractor`, `CodeExtractor`, `NLExtractor`
- Move logic from exp 47 (self-constraints), 48 (code AST), 49 (NL patterns)
- Each extractor is independently importable and testable

### Exp 75: VerifyRepairPipeline class
The main user-facing API — one class that wires everything together.
- Create `carnot.pipeline.verify_repair` module
- `VerifyRepairPipeline(model, domains, max_repairs, backend)`
- Methods: `verify()`, `verify_and_repair()`, `extract_constraints()`
- Model loading: HuggingFace local, or None (user provides responses)
- Backend: CPU Ising sampler (default), TSU stub (future)
- Returns structured `VerificationResult` with per-constraint breakdown

### Exp 76: Production MCP server
Harden the MCP server for real-world use.
- Move from `tools/verify-mcp/` to `carnot.mcp` (proper package location)
- Add: execution timeouts, memory limits, input validation
- Add: `verify_llm_output` tool (uses VerifyRepairPipeline)
- Add: `list_domains` tool (introspect available extractors)
- Add: health check endpoint
- Docker image with proper entrypoint

### Exp 77: CLI overhaul
Make `carnot` CLI the primary user interface.
- Fix entrypoint so `carnot` works after `pip install`
- Add `carnot pipeline` subcommand for verify-repair
- Add `carnot serve` subcommand to start MCP server
- Add `carnot check` subcommand for quick health check
- Interactive mode: `carnot pipeline --interactive` (reads from stdin)

## Phase 14: Packaging and Distribution (experiments 78-80)

### Exp 78: PyPI-ready package
Ship `carnot` to PyPI.
- Clean up `pyproject.toml`: proper versioning (CalVer), classifiers, URLs
- Remove Rust build dependency for pure-Python install (make Rust optional)
- Create `carnot[rust]` extra for Rust bindings
- Create `carnot[llm]` extra for transformers/torch
- Create `carnot[mcp]` extra for MCP server dependencies
- Verify clean install in fresh venv: `pip install carnot && carnot check`

### Exp 79: Integration examples
Real-world examples that developers can copy-paste.
- Example 1: Verify a FastAPI endpoint's responses
- Example 2: Add constraint checking to a LangChain pipeline
- Example 3: Use MCP server with Claude Code for code review
- Example 4: Batch-verify a dataset of LLM outputs
- Example 5: Train a domain-specific constraint extractor
- Each example is a standalone script in `examples/` with README

### Exp 80: Getting started documentation
Write the docs that turn researchers into users.
- `docs/getting-started.md` — install + first verification in 5 minutes
- `docs/concepts.md` — constraints, energy, verification, repair (no math)
- `docs/api-reference.md` — auto-generated from docstrings
- `docs/mcp-integration.md` — using with Claude Code
- Update project README.md with clear "what this does" section

## Phase 15: Robustness and Testing (experiments 81-83)

### Exp 81: Integration test suite
End-to-end tests that exercise the full pipeline as a user would.
- Test: `pip install -e . && carnot check` passes
- Test: `carnot pipeline verify` produces correct output
- Test: MCP server starts, accepts requests, returns results
- Test: VerifyRepairPipeline works with and without LLM loaded
- Test: Clean install in Docker container works

### Exp 82: Error handling and edge cases
Make the pipeline robust to bad input.
- Malformed LLM responses (no structured output)
- Constraint extraction returns empty (no constraints found)
- Ising sampler timeout (unsatisfiable constraints)
- Model loading failure (graceful degradation)
- Unicode, empty strings, very long inputs
- Concurrent MCP requests

### Exp 83: Performance benchmarks
Establish baseline performance for the production pipeline.
- Latency: verify() call on CPU (target: <100ms for simple constraints)
- Latency: verify_and_repair() with 3 iterations (target: <1s without LLM)
- Throughput: batch verification of 1000 responses
- Memory: peak RSS during pipeline operation
- Compare: Python-only vs Rust-accelerated (if Rust bindings are wired up)

## Phase 16: Dogfooding (experiments 84-85)

### Exp 84: Carnot verifies Carnot
Use the constraint pipeline to verify Carnot's own test outputs.
- Run existing test suite through VerifyRepairPipeline
- Check: do constraint violations correlate with test failures?
- Integrate into pre-commit hook: `carnot pipeline verify` on changed files

### Exp 85: Community feedback integration
Ship beta, collect feedback, iterate.
- Publish 0.1.0-beta1 to PyPI
- Create GitHub issue template for feedback
- Monitor: what breaks on other people's machines?
- Fix top 3 issues reported

## Dependencies

```
Phase 13 (library extraction):
  Exp 74 ← Exp 47-49 (constraint extraction experiments)
  Exp 75 ← Exp 74 + Exp 57 (verify-repair loop)
  Exp 76 ← Exp 75 + existing MCP server
  Exp 77 ← Exp 75 + existing CLI

Phase 14 (packaging):
  Exp 78 ← Exp 75 (pipeline must exist before packaging)
  Exp 79 ← Exp 75 + Exp 76 (pipeline + MCP must work)
  Exp 80 ← Exp 79 (examples inform docs)

Phase 15 (robustness):
  Exp 81 ← Exp 78 (package must be installable)
  Exp 82 ← Exp 75 (pipeline must exist)
  Exp 83 ← Exp 75 (pipeline must exist)

Phase 16 (dogfooding):
  Exp 84 ← Exp 75 + Exp 81 (pipeline + integration tests)
  Exp 85 ← Exp 78 (must be on PyPI)
```

## Execution Order

```
1. exp74  -- ConstraintExtractor API (foundation)
2. exp75  -- VerifyRepairPipeline (the product)
3. exp77  -- CLI overhaul (first user interface)
4. exp76  -- MCP server hardening (second user interface)
5. exp82  -- Error handling (before shipping)
6. exp78  -- PyPI packaging (ship it)
7. exp79  -- Integration examples (teach users)
8. exp80  -- Getting started docs (onboard users)
9. exp81  -- Integration test suite (verify shipping)
10. exp83  -- Performance benchmarks (know your limits)
11. exp84  -- Dogfood on Carnot itself
12. exp85  -- Beta release and feedback
```

## Success Criteria

- `pip install carnot && carnot check` works on a fresh machine
- `VerifyRepairPipeline` usable in 5 lines of Python
- MCP server runs safely with timeout/resource limits
- 5 real-world examples that developers can copy
- Getting started guide that works in 5 minutes
- 0.1.0-beta1 on PyPI
- Latency under 100ms for simple constraint verification

## What's Explicitly NOT in Scope

- New research experiments (that's what v8/2026.04.3 is for)
- Rust constraint crate (deferred until Python API is stable)
- Extropic TSU integration (hardware not available)
- Activation-based EBMs (proven insufficient for production)
- Multi-model training (Qwen3.5 + Gemma4 support is enough)
- Exp 66 (end-to-end differentiable) — research, not production
