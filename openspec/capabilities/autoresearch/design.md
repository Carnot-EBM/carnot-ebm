# Autoresearch — Design Document

**Capability:** autoresearch
**Version:** 0.1.0

## Architecture Overview

The autoresearch loop has four stages, running as a continuous pipeline:

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  PROPOSE     │────>│  EVALUATE    │────>│  TRANSPILE     │────>│  DEPLOY      │
│  (Agent)     │     │  (JAX + EBM) │     │  (JAX → Rust)  │     │  (Rust prod) │
│              │     │              │     │                │     │              │
│  Generates   │     │  Runs in     │     │  Agent-assisted│     │  Merge +     │
│  hypothesis  │     │  sandbox     │     │  translation   │     │  monitor     │
│  as JAX code │     │  against     │     │  with spec     │     │  with auto-  │
│              │     │  benchmarks  │     │  conformance   │     │  rollback    │
└─────────────┘     └──────────────┘     └────────────────┘     └──────────────┘
       ↑                   │ fail                                       │
       │                   ↓                                           │
       │            ┌──────────────┐                                   │
       │            │  REJECTED    │                                   │
       │            │  (logged)    │                                   │
       │            └──────────────┘                                   │
       │                                                               │
       └───────────────── feedback (new baselines) ────────────────────┘
```

## Stage 1: PROPOSE

### Hypothesis Format

```yaml
# hypothesis.yaml
id: "auto-2026-04-03-001"
target: sampler
component: carnot_samplers::LangevinSampler
change_type: hyperparameter  # architecture | algorithm | hyperparameter
description: "Anneal step_size from 0.1 to 0.001 over sampling chain"
code_path: "experiments/hypothesis_001.py"  # self-contained JAX script
expected_impact:
  doublewell_energy: -5%
  convergence_steps: -20%
risk_assessment:
  wall_clock_time: +10%  # annealing adds overhead
```

### Hypothesis Sources

The system supports multiple hypothesis generators:

1. **Agent-generated** (primary): An LLM agent (Qwen3, Claude, etc.) proposes changes based on:
   - Current performance metrics and trends
   - Recent research papers (fed as context)
   - Failure analysis from rejected hypotheses
   - Known algorithmic improvements from the EBM literature

2. **Grid search** (systematic): Automated hyperparameter sweeps
3. **Ablation-driven** (diagnostic): Remove components to identify what matters

### Hypothesis Code Contract

Every hypothesis must be a self-contained Python/JAX script that:
- Imports only from `carnot` and standard JAX/numpy
- Defines a `run(benchmark_data: dict) -> dict` function
- Returns a dict with keys matching benchmark metric names
- Does not access the filesystem except for reading benchmark data passed to it
- Does not import `os`, `subprocess`, `socket`, or `shutil`

```python
# experiments/hypothesis_001.py
"""Annealed Langevin dynamics for improved convergence."""
import jax
import jax.numpy as jnp
from carnot.samplers.langevin import LangevinSampler
from carnot.models.ising import IsingModel, IsingConfig

def run(benchmark_data: dict) -> dict:
    model = IsingModel(IsingConfig(input_dim=benchmark_data["dim"]))
    # ... the proposed modification
    return {
        "final_energy": float(final_energy),
        "convergence_steps": n_steps,
        "wall_clock_seconds": elapsed,
    }
```

## Stage 2: EVALUATE

### Sandbox Architecture

```
┌─────────────────────────────────────────┐
│  Container / Subprocess                  │
│                                         │
│  ┌─────────────────┐  ┌──────────────┐ │
│  │ hypothesis.py    │  │ benchmark    │ │
│  │ (read-only mount)│  │ data (r/o)   │ │
│  └────────┬────────┘  └──────┬───────┘ │
│           │                   │         │
│           v                   v         │
│  ┌──────────────────────────────────┐   │
│  │  Python interpreter + JAX        │   │
│  │  (no network, no write access)   │   │
│  └──────────────┬───────────────────┘   │
│                 │                        │
│                 v                        │
│  ┌──────────────────────────────────┐   │
│  │  metrics.json (write to stdout)  │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Evaluation Protocol

```python
def evaluate_hypothesis(hypothesis_path: str, baselines: dict) -> EvalResult:
    # 1. Execute in sandbox
    metrics = sandbox_run(hypothesis_path, timeout=1800, memory_limit="16G")

    # 2. Check primary gate: energy improvement
    for benchmark, baseline in baselines.items():
        if metrics[benchmark]["final_energy"] > baseline["final_energy"] * 1.001:
            return EvalResult(verdict="FAIL", reason=f"Energy regression on {benchmark}")

    # 3. Check secondary gate: time budget
    for benchmark, baseline in baselines.items():
        if metrics[benchmark]["wall_clock_seconds"] > baseline["wall_clock_seconds"] * 2.0:
            return EvalResult(verdict="FAIL", reason=f"Time budget exceeded on {benchmark}")

    # 4. Check tertiary gate: memory
    if metrics["peak_memory_mb"] > baselines["peak_memory_mb"] * 2.0:
        return EvalResult(verdict="FAIL", reason="Memory budget exceeded")

    # 5. Check for mixed results (improvement on some, regression on others)
    improvements = [b for b in baselines if metrics[b]["final_energy"] < baseline["final_energy"] * 0.99]
    regressions = [b for b in baselines if metrics[b]["final_energy"] > baseline["final_energy"] * 1.001]
    if regressions:
        return EvalResult(verdict="REVIEW", reason=f"Mixed: improved {improvements}, regressed {regressions}")

    return EvalResult(verdict="PASS", metrics=metrics)
```

## Stage 3: TRANSPILE (JAX → Rust)

### Translation Strategy

The spec-driven architecture makes transpilation tractable:

1. Both implementations share the same REQ-*/SCENARIO-* contracts
2. The JAX code defines the mathematical behavior
3. The Rust implementation must produce identical results (within fp tolerance)

### Conformance Testing

```rust
/// Test that Rust implementation matches JAX output.
/// Generated automatically for each transpiled hypothesis.
#[test]
fn test_rust_matches_jax() {
    // REQ-AUTO-006: Cross-language validation
    let test_vectors = load_jax_test_vectors("hypothesis_001_vectors.json");
    let model = /* Rust implementation of the hypothesis */;

    for (input, expected_energy) in test_vectors {
        let actual = model.energy(&input.view());
        assert!(
            (actual - expected_energy).abs() < 1e-4,
            "Rust/JAX mismatch: rust={actual}, jax={expected_energy}"
        );
    }
}
```

### Performance Gate

```rust
/// Rust must be at least as fast as JAX.
#[test]
fn test_rust_perf_gate() {
    // REQ-AUTO-006: Performance requirement
    let jax_time_ms = load_jax_benchmark_time("hypothesis_001");
    let start = Instant::now();
    // ... run same benchmark in Rust ...
    let rust_time_ms = start.elapsed().as_millis();
    assert!(rust_time_ms <= jax_time_ms, "Rust slower than JAX: {rust_time_ms}ms vs {jax_time_ms}ms");
}
```

## Stage 4: DEPLOY

### Deployment Sequence

1. Create git branch `autoresearch/hypothesis-{id}`
2. Commit Rust implementation + conformance tests
3. Run full CI (all existing tests + new conformance tests)
4. If CI passes: merge to main, update baseline registry
5. Start monitoring window (default 1 hour)
6. If monitoring passes: mark as stable

### Rollback Implementation

```rust
/// Monitor production metrics and rollback if degraded.
fn monitor_and_rollback(
    config: &MonitorConfig,
    baseline_energy: Float,
) -> MonitorResult {
    let threshold = baseline_energy * (1.0 + config.regression_threshold);
    let window = config.monitoring_window;

    loop {
        let current_energy = measure_production_energy();
        if current_energy > threshold {
            // Rollback
            git_revert_to_last_stable();
            return MonitorResult::RolledBack {
                reason: format!("Energy {current_energy} > threshold {threshold}"),
            };
        }
        if elapsed > window {
            return MonitorResult::Stable;
        }
        sleep(config.poll_interval);
    }
}
```

## Baseline Registry Format

```json
{
  "version": "0.1.0",
  "commit": "abc123",
  "timestamp": "2026-04-03T14:00:00Z",
  "benchmarks": {
    "doublewell_2d": {
      "model": "IsingModel",
      "config": {"input_dim": 2},
      "sampler": "LangevinSampler",
      "sampler_config": {"step_size": 0.01},
      "metrics": {
        "final_energy": -5.2,
        "convergence_steps": 5000,
        "wall_clock_seconds": 1.3,
        "peak_memory_mb": 45
      }
    }
  }
}
```

## Experiment Log Format

```json
{
  "id": "auto-2026-04-03-001",
  "timestamp": "2026-04-03T14:30:00Z",
  "hypothesis": { /* full hypothesis.yaml content */ },
  "sandbox_result": {
    "exit_code": 0,
    "metrics": { /* benchmark results */ },
    "stdout": "...",
    "stderr": "...",
    "wall_clock_seconds": 45.2,
    "peak_memory_mb": 120
  },
  "evaluation": {
    "verdict": "PASS",
    "primary_gate": true,
    "secondary_gate": true,
    "tertiary_gate": true,
    "details": "Energy improved 11% on doublewell, 3% on rosenbrock"
  },
  "transpilation": {
    "status": "completed",
    "conformance_tests": 12,
    "conformance_passed": 12,
    "perf_gate": true
  },
  "deployment": {
    "status": "stable",
    "branch": "autoresearch/hypothesis-001",
    "merged_at": "2026-04-03T15:00:00Z",
    "monitoring_result": "stable"
  }
}
```

## Safety Architecture

### Defense in Depth

1. **Sandbox isolation**: No filesystem writes, no network, hard timeout + memory limit
2. **Import whitelist**: Hypothesis code can only import approved modules
3. **Evaluation immutability**: Benchmark definitions and evaluation protocol are not writable by hypotheses
4. **Baseline integrity**: Baselines are append-only; old baselines are never deleted
5. **Rollback readiness**: Previous known-good commit is always tagged and restorable
6. **Circuit breaker**: N consecutive failures halt the loop

### What Hypotheses Cannot Do

- Modify validation data
- Modify benchmark definitions or evaluation criteria
- Access the network
- Write to the filesystem (except stdout metrics)
- Import system modules (os, subprocess, socket, etc.)
- Run indefinitely (hard timeout enforced by sandbox)
- Consume unlimited memory (hard limit enforced by sandbox)
- Modify the autoresearch pipeline itself
