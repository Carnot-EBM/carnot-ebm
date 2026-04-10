# Experiment 102: Latency Benchmark Results

**Date**: 2026-04-10 05:00:47 UTC
**Platform**: CPU (JAX_PLATFORMS=cpu)
**Warmup**: 100 iterations | **Measured**: 1000 iterations

## 1. Component-Level Latency (Differentiable Pipeline)

| Component | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Std (ms) |
|-----------|----------|----------|----------|-----------|----------|
| simulated_embedding | 0.0323 | 0.0349 | 0.0377 | 0.0329 | 0.0088 |
| real_embedding_MiniLM | 7.4683 | 8.6456 | 9.0965 | 7.6022 | 0.5104 |
| constraint_extraction | 0.2752 | 0.2854 | 0.3365 | 0.2761 | 0.0114 |
| ising_energy_jit | 0.0131 | 0.0137 | 0.0151 | 0.0134 | 0.0052 |
| mlp_scoring_jit | 0.0045 | 0.0059 | 0.0122 | 0.0050 | 0.0037 |
| full_differentiable_forward_jit | 0.0063 | 0.0191 | 0.0340 | 0.0080 | 0.0085 |

### Classification: **VIABLE for per-token guided decoding**
- Full differentiable forward pass mean: **0.0080 ms**
- Verdict: **GREEN** — per-token tier

### Bottleneck: **real_embedding_MiniLM** (7.6022 ms)

## 2. Scale Sweep

### Extraction vs Input Length

| Tokens | Chars | Constraints | Mean (ms) | p95 (ms) |
|--------|-------|-------------|-----------|----------|
| 10 | 50 | 3 | 0.0434 | 0.0446 |
| 50 | 250 | 11 | 0.1427 | 0.1461 |
| 100 | 500 | 9 | 0.2727 | 0.2719 |
| 500 | 2500 | 11 | 1.2885 | 1.3056 |
| 1000 | 5000 | 10 | 2.6340 | 3.0515 |

### Forward Pass vs Constraint Count

| Constraints | Mean (ms) | p95 (ms) |
|-------------|-----------|----------|
| 1 | 0.0063 | 0.0089 |
| 5 | 0.0073 | 0.0096 |
| 10 | 0.0072 | 0.0100 |
| 50 | 0.0101 | 0.0232 |

### Combined (Extraction + Forward) Matrix

| Tokens \ Constraints | 1 | 5 | 10 | 50 |
|---|---|---|---|---|
| 10 | 0.062ms | 0.064ms | 0.066ms | 0.065ms |
| 50 | 0.174ms | 0.174ms | 0.171ms | 0.177ms |
| 100 | 0.301ms | 0.295ms | 0.294ms | 0.297ms |
| 500 | 1.637ms | 1.428ms | 1.398ms | 1.347ms |
| 1000 | 3.022ms | 2.797ms | 2.786ms | 3.146ms |

## 3. Backend Comparison (100 inputs each)

| Backend | Total (ms) | Per-call (ms) |
|---------|------------|---------------|
| python_verify_100inputs | 41.47 | 0.4147 |
| jax_jit_forward_100inputs | 0.76 | 0.0076 |
| rust_verify_100inputs | 162.23 | 1.6223 |

**Fastest**: jax_jit_forward_100inputs at 0.0076 ms/call

## 4. Generation Budget Analysis

- **Target generation speed**: 50 tokens/second
- **Budget per token**: 20.0 ms
- **Constraint check cost**: 0.0080 ms
- **Budget fraction**: 0.0%
- **Fits in budget**: Yes

**Recommendation**: Guided decoding viable — constraint check uses 0.0% of per-token budget

## 5. Extraction Scaling

- Input range: 50-5000 chars
- Latency range: 0.043-2.634 ms
- Scaling factor: 60.69x (vs 100.00x input growth)
- Roughly linear: Yes
