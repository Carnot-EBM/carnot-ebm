"""LLM-powered hypothesis generator for the autoresearch loop.

**Researcher summary:**
    Uses an LLM (via OpenAI-compatible API) to propose EBM improvement
    hypotheses. Each hypothesis is a Python code snippet defining
    ``run(benchmark_data) -> dict`` that the sandbox evaluates.

**Detailed explanation for engineers:**
    This module connects the autoresearch pipeline to a language model.
    Instead of hand-writing hypotheses (as in the demo), the LLM proposes
    new sampler configurations, training modifications, or architectural
    changes based on:

    - Current baseline performance metrics
    - Recent failures (what didn't work and why)
    - Benchmark descriptions (what we're optimizing)

    The generator uses the OpenAI Python SDK, which means it can talk to:
    - The Claude API bridge (wrapping claude -p as an OpenAI API)
    - Any OpenAI-compatible endpoint (vLLM, Ollama, etc.)
    - OpenAI itself

    **How it works:**
    1. Build a context prompt with baselines, failures, and benchmark info
    2. Ask the LLM to propose a hypothesis as a Python code block
    3. Extract the code from the LLM's response
    4. Validate that the code defines ``def run(benchmark_data)``
    5. Return (description, code) pairs for the orchestrator

    **Safety:** The generated code runs in the sandbox — even malicious
    code is harmless. The energy function is the objective judge, not the
    LLM. The LLM is the creative engine; the math is the filter.

Spec: FR-11, REQ-AUTO-003
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carnot.autoresearch.baselines import BaselineRecord

logger = logging.getLogger(__name__)

# The system prompt that frames the LLM as an EBM researcher.
# It explains the hypothesis format, available tools, and constraints.
DEFAULT_SYSTEM_PROMPT = """\
You are an autonomous Energy-Based Model (EBM) researcher working on the \
Carnot framework. Your job is to propose hypotheses that improve EBM \
performance on benchmark problems.

## What you're optimizing

Energy-Based Models assign a scalar energy E(x) to each input configuration x. \
Lower energy = more likely/natural configuration. The benchmarks measure how \
well a sampler finds low-energy states.

## Available benchmarks

- **double_well**: A 2D potential with two minima at approximately x=[-1,0] and x=[1,0]. \
  Global minimum energy ~ -1.0. Tests ability to find and settle into energy minima.
- **rosenbrock**: A narrow curved valley in 2D. Global minimum at x=[1,1] with energy 0. \
  Tests optimization in ill-conditioned landscapes.

## EXACT Carnot API (use these signatures precisely)

```python
# Models — each takes a Config dataclass + optional JAX PRNG key
from carnot.models.ising import IsingModel, IsingConfig
# IsingConfig(input_dim=784, hidden_dim=None, coupling_init="xavier_uniform")
# IsingModel(config: IsingConfig, key: jax.Array | None = None)

from carnot.models.gibbs import GibbsModel, GibbsConfig
# GibbsConfig(input_dim=784, hidden_dims=[512,256], activation="silu", dropout=0.0)
# GibbsModel(config: GibbsConfig, key: jax.Array | None = None)

from carnot.models.boltzmann import BoltzmannModel, BoltzmannConfig
# BoltzmannConfig(input_dim=784, hidden_dims=[1024,512,256,128], num_heads=4, residual=True)
# BoltzmannModel(config: BoltzmannConfig, key: jax.Array | None = None)

# Samplers — dataclass with step_size, then call .sample()
from carnot.samplers.langevin import LangevinSampler
# LangevinSampler(step_size=0.01)
# sampler.sample(energy_fn, init, n_steps, key=None) -> jax.Array (final state)
# sampler.sample_chain(energy_fn, init, n_steps, key=None) -> jax.Array (all states)

from carnot.samplers.hmc import HMCSampler
# HMCSampler(step_size=0.1, num_leapfrog_steps=10)
# sampler.sample(energy_fn, init, n_steps, key=None) -> jax.Array
# sampler.sample_chain(energy_fn, init, n_steps, key=None) -> jax.Array

# Model methods:
# model.energy(x: jax.Array) -> scalar  (single input)
# model.energy_batch(xs: jax.Array) -> jax.Array  (batch)
# model.grad_energy(x: jax.Array) -> jax.Array  (gradient)
# model.input_dim -> int
```

## Benchmark energy functions (use these instead of model tiers)

```python
from carnot.benchmarks import DoubleWell, Rosenbrock, Ackley, Rastrigin, GaussianMixture
# DoubleWell(dim=2) — E(x) = (x[0]^2 - 1)^2 + sum(x[1:]^2), min at [+/-1, 0], E=0
# Rosenbrock(dim=2) — E(x) = sum[100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2], min at [1,1], E=0
# Ackley(dim=2) — multimodal, min at origin, E=0
# Rastrigin(dim=2) — many local minima, min at origin, E=0
# GaussianMixture.two_modes(4.0) — 1D, modes at -2 and +2
```

## Working example hypothesis

This is a CORRECT, WORKING hypothesis. Use it as a template:

```python
import time
import jax.numpy as jnp
import jax.random as jrandom
from carnot.benchmarks import DoubleWell, Rosenbrock
from carnot.samplers.langevin import LangevinSampler

def run(benchmark_data):
    start = time.time()
    key = jrandom.PRNGKey(42)
    results = {}

    # Benchmark energy functions — these define the landscape to minimize
    benchmarks = {
        "double_well": DoubleWell(dim=2),
        "rosenbrock": Rosenbrock(dim=2),
    }

    # Hypothesis: try a different step size
    sampler = LangevinSampler(step_size=0.005)

    for bench_name, energy_fn in benchmarks.items():
        k1, key = jrandom.split(key)
        x0 = jrandom.normal(k1, (energy_fn.input_dim,))
        x_final = sampler.sample(energy_fn, x0, n_steps=5000, key=key)
        energy = float(energy_fn.energy(x_final))
        results[bench_name] = {
            "final_energy": energy,
            "convergence_steps": 5000,
            "wall_clock_seconds": time.time() - start,
        }
    return results
```

## IMPORTANT RULES

1. Your code MUST actually call the Carnot API — do NOT return hardcoded energy values
2. Use the EXACT constructor signatures shown above — no extra kwargs
3. For 2D benchmarks (double_well, rosenbrock), use input_dim=2
4. The sandbox blocks: os, subprocess, socket, shutil, etc.
5. Allowed imports: jax, jax.numpy, jax.random, numpy, math, time, carnot.*

## Evaluation criteria

Your hypothesis PASSES if:
1. **Primary gate**: final_energy <= baseline energy (must improve or match)
2. **Secondary gate**: wall_clock_seconds <= 2x baseline time
3. **Tertiary gate**: peak_memory_mb <= 2x baseline memory

## Guidelines

- Propose ONE hypothesis at a time
- Include a brief description of your rationale
- Vary: step_size, n_steps, model tier (Ising/Gibbs/Boltzmann), sampler (Langevin/HMC)
- Start with simple parameter sweeps, then try model changes

## Skill Playbook

If an "Additional Context" section is provided below, it contains a curated \
optimization playbook learned from previous experiments. Treat it as high-value \
guidance — these are patterns that have been validated across multiple iterations. \
Prioritize approaches consistent with these lessons. Avoid approaches that \
contradict known failure patterns listed in the playbook.
"""


@dataclass
class GeneratorConfig:
    """Configuration for the LLM hypothesis generator.

    **For engineers:**
        - ``api_base``: URL of the OpenAI-compatible API. Point this at
          the Claude API bridge (http://localhost:8080/v1) or any other
          OpenAI-compatible endpoint.
        - ``model``: Model identifier to pass in the API request.
        - ``system_prompt``: The framing prompt that tells the LLM how
          to propose hypotheses. Override for custom domains.
        - ``max_hypotheses_per_call``: How many hypotheses to request
          per LLM invocation. More = fewer API calls but longer responses.
        - ``temperature``: Sampling temperature. Higher = more creative
          but potentially more broken hypotheses.
        - ``api_key``: API key for the endpoint. For the Claude bridge,
          this is unused (OAuth handles auth). Default "not-needed".
    """

    api_base: str = "http://localhost:8080/v1"
    model: str = "sonnet"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_hypotheses_per_call: int = 1
    temperature: float = 0.7
    api_key: str = "not-needed"
    extra_context: str = ""


@dataclass
class GenerationResult:
    """Result of a hypothesis generation attempt.

    **For engineers:**
        Contains the raw LLM response, extracted hypotheses, and any
        errors that occurred during generation or parsing.
    """

    hypotheses: list[tuple[str, str]] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None


def _build_user_prompt(
    baselines: BaselineRecord,
    recent_failures: list[dict[str, Any]] | None = None,
    iteration: int = 0,
    extra_context: str = "",
) -> str:
    """Build the user-turn prompt with current context.

    Includes baseline metrics, recent failure summaries, and the
    iteration number so the LLM can adjust its strategy over time.
    """
    parts: list[str] = []

    # Current baselines
    parts.append("## Current Baseline Performance\n")
    if baselines.benchmarks:
        for name, metrics in sorted(baselines.benchmarks.items()):
            parts.append(
                f"- **{name}**: energy={metrics.final_energy:.6f}, "
                f"steps={metrics.convergence_steps}, "
                f"time={metrics.wall_clock_seconds:.2f}s, "
                f"memory={metrics.peak_memory_mb:.1f}MB"
            )
    else:
        parts.append("No baselines established yet.")
    parts.append("")

    # Recent failures
    if recent_failures:
        parts.append("## Recent Failed Hypotheses (avoid these approaches)\n")
        for failure in recent_failures[-5:]:  # Last 5 failures
            desc = failure.get("description", "unknown")
            reason = failure.get("reason", "unknown")
            parts.append(f"- **{desc}**: {reason}")
        parts.append("")

    # Iteration info
    parts.append(f"## Iteration: {iteration}\n")
    if iteration == 0:
        parts.append(
            "This is the first iteration. Start with a simple, high-confidence improvement."
        )
    elif iteration < 5:
        parts.append(
            "Early iterations. Try straightforward hyperparameter tuning or known-good techniques."
        )
    else:
        parts.append(
            "Later iteration. Consider more creative approaches: "
            "annealing schedules, adaptive step sizes, or model "
            "architecture changes."
        )
    parts.append("")

    # Extra context
    if extra_context:
        parts.append(f"## Additional Context\n{extra_context}\n")

    parts.append(
        "Propose a hypothesis. Include a brief description, then a "
        "Python code block with the `run(benchmark_data)` function."
    )

    return "\n".join(parts)


def _extract_hypotheses(response: str) -> list[tuple[str, str]]:
    """Extract (description, code) pairs from an LLM response.

    Looks for Python code blocks (```python ... ```) and uses the text
    before each code block as the description. Validates that each code
    block defines a ``run`` function.
    """
    hypotheses: list[tuple[str, str]] = []

    # Find all Python code blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = list(re.finditer(pattern, response, re.DOTALL))

    if not matches:
        # Try without language specifier
        pattern = r"```\s*\n(.*?)```"
        matches = list(re.finditer(pattern, response, re.DOTALL))

    for match in matches:
        code = match.group(1).strip()

        # Validate: must define run(benchmark_data)
        if "def run(" not in code:
            logger.warning("Skipping code block without 'def run(': %s...", code[:80])
            continue

        # Extract description: text between previous code block end (or start)
        # and this code block start
        start = match.start()
        # Find the preceding text (after last ``` or start of string)
        preceding = response[:start].strip()
        # Take the last paragraph as description
        paragraphs = preceding.split("\n\n")
        description = paragraphs[-1].strip() if paragraphs else "LLM hypothesis"

        # Clean up description (remove markdown headers, etc.)
        description = re.sub(r"^#+\s*", "", description)
        description = description.strip("*_ \n")
        if not description:
            description = "LLM-generated hypothesis"

        hypotheses.append((description, code))

    return hypotheses


def generate_hypotheses(
    config: GeneratorConfig,
    baselines: BaselineRecord,
    recent_failures: list[dict[str, Any]] | None = None,
    iteration: int = 0,
) -> GenerationResult:
    """Generate hypotheses using an LLM via the OpenAI-compatible API.

    **For engineers:**
        This is the main entry point. It:
        1. Builds a prompt with current baselines and failure context
        2. Calls the LLM API (synchronous)
        3. Extracts code blocks from the response
        4. Validates each code block defines ``run(benchmark_data)``
        5. Returns a GenerationResult with hypotheses

        If the API call fails, returns a GenerationResult with the error
        field set. The caller (orchestrator) decides how to handle it.

    Args:
        config: Generator configuration (API endpoint, model, etc.)
        baselines: Current baseline performance to beat.
        recent_failures: List of dicts with "description" and "reason"
            for recently failed hypotheses. Helps the LLM avoid
            repeating mistakes.
        iteration: Current iteration number (affects prompt strategy).

    Returns:
        GenerationResult with extracted hypotheses and/or error info.
    """
    try:
        # Import openai lazily — not everyone will have it installed
        from openai import OpenAI
    except ImportError:
        return GenerationResult(error="openai package not installed. Run: pip install openai")

    user_prompt = _build_user_prompt(baselines, recent_failures, iteration, config.extra_context)

    try:
        client = OpenAI(
            base_url=config.api_base,
            api_key=config.api_key,
        )

        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
        )

        raw = response.choices[0].message.content or ""
        hypotheses = _extract_hypotheses(raw)

        if not hypotheses:
            return GenerationResult(
                raw_response=raw,
                error="LLM response contained no valid Python code blocks with run()",
            )

        return GenerationResult(
            hypotheses=hypotheses,
            raw_response=raw,
        )

    except Exception as e:
        logger.exception("Error calling LLM API")
        return GenerationResult(error=str(e))


def generate_hypotheses_batch(
    config: GeneratorConfig,
    baselines: BaselineRecord,
    recent_failures: list[dict[str, Any]] | None = None,
    iteration: int = 0,
    count: int = 3,
) -> list[tuple[str, str]]:
    """Generate multiple hypotheses, retrying on failure.

    **For engineers:**
        Convenience wrapper that calls ``generate_hypotheses`` up to
        ``count`` times, collecting all valid hypotheses. If any call
        fails, it logs the error and continues.

    Returns:
        List of (description, code) tuples. May be shorter than
        ``count`` if some calls failed.
    """
    all_hypotheses: list[tuple[str, str]] = []
    failures: list[dict[str, Any]] = list(recent_failures or [])

    for i in range(count):
        result = generate_hypotheses(config, baselines, failures, iteration + i)

        if result.error:
            logger.warning("Generation attempt %d failed: %s", i, result.error)
            failures.append(
                {
                    "description": f"generation_attempt_{i}",
                    "reason": result.error,
                }
            )
            continue

        all_hypotheses.extend(result.hypotheses)

    return all_hypotheses
