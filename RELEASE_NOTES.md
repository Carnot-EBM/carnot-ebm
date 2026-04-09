# Carnot 0.1.0-beta1 Release Notes

**Date:** 2026-04-09

The first public beta of Carnot, an open-source Energy-Based Model framework for verifying LLM outputs.

## Highlights

- **Constraint verification pipeline** — extract arithmetic, code, logic, and natural-language constraints from LLM responses, then verify them against energy-based models. Catches wrong math, buggy code, and contradictory claims.
- **Verify-and-repair loop** — when verification finds a problem, Carnot generates structured feedback so the LLM can fix it. Up to 3 repair iterations, with energy scores tracking improvement.
- **CLI and MCP server** — `carnot verify` for command-line use; MCP server for Claude Code integration via `verify_code`, `verify_with_properties`, `score_candidates`, and more.
- **15 pre-trained EBMs on HuggingFace** — per-token activation EBMs trained on TruthfulQA across Qwen3.5, Gemma 4, Phi-4, and Mixtral model families. Published at [huggingface.co/Carnot-EBM](https://huggingface.co/Carnot-EBM).
- **Dual Rust + Python/JAX implementation** — core EBM types and samplers in Rust (ndarray, rayon) with equivalent JAX implementations. Safetensors for cross-language model exchange.
- **Three model tiers** — Ising (pairwise), Gibbs (multi-layer), Boltzmann (deep residual + attention) — all implementing the same `EnergyFunction` trait/protocol.

## What's included

### Python package (`pip install -e .`)
- `carnot.pipeline` — `VerifyRepairPipeline` for end-to-end verification
- `carnot.pipeline.extract` — `AutoExtractor` with domain-specific constraint extractors
- `carnot.pipeline.errors` — structured error hierarchy with timeouts
- `carnot.models` — Ising, Gibbs, Boltzmann in JAX
- `carnot.samplers` — Langevin dynamics, HMC, parallel Ising Gibbs
- `carnot.training` — contrastive divergence, noise-contrastive estimation, score matching
- `carnot.cli` — `carnot verify` and `carnot score` commands
- `carnot.mcp` — MCP server for Claude Code integration
- `carnot.inference` — pre-trained EBM loader and rejection sampling

### Rust crates
- `carnot-core` — `EnergyFunction` trait, types, serialization
- `carnot-ising`, `carnot-gibbs`, `carnot-boltzmann` — model tiers
- `carnot-samplers` — Langevin + HMC
- `carnot-training` — CD-k, score matching
- `carnot-constraints` — reusable constraint types and verification certificates
- `carnot-python` — PyO3 bindings (optional)

### Research artifacts
- 34 experiments across 15 model architectures (350M to 27B parameters)
- 14 principles on what works and what doesn't for activation-based hallucination detection
- Benchmark results, scaling data, and experiment scripts

## Known limitations

- **Activation EBMs detect confidence, not correctness.** The 75-88% test-set accuracy does not transfer to real-world hallucination detection. The pre-trained models are research artifacts, not production detectors. See the [technical writeup](docs/technical-writeup.md#8-limitations).
- **Constraint coverage varies by domain.** Arithmetic verification is strong; factual and semantic constraints are limited. The `carnot.pipeline.extract` module reports its coverage.
- **Rust bindings are optional.** The Python package works standalone; install `carnot[rust]` with a Rust toolchain for native performance.
- **ROCm/AMD GPU:** JAX's ROCm plugin may crash (`thrml` issue #41). Use `JAX_PLATFORMS=cpu` on AMD systems.

## Install

```bash
pip install -e ".[dev]"

# Verify installation
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4" --test "(7,13):1"
```

## Requirements

- Python 3.11+
- JAX 0.4.30+ (CPU by default)
- Rust stable (optional, for native extensions)
