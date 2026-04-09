# Pipeline Benchmark Results

**Date**: 2026-04-09 22:33:27 UTC
**Platform**: CPU (JAX_PLATFORMS=cpu)

## 1. verify() Latency per Domain (100 calls each)

| Domain | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) |
|--------|----------|----------|----------|-----------|
| arithmetic | 0.01 | 0.01 | 0.01 | 0.01 |
| code | 0.06 | 0.07 | 0.08 | 0.06 |
| logic | 0.04 | 0.04 | 0.04 | 0.04 |
| nl | 0.03 | 0.03 | 0.03 | 0.03 |

## 2. extract_constraints() vs Input Length

| Input Length (chars) | Mean (ms) | p95 (ms) | Constraints Found |
|---------------------|-----------|----------|-------------------|
| 50 | 0.05 | 0.06 | 3 |
| 100 | 0.08 | 0.08 | 3 |
| 250 | 0.19 | 0.20 | 3 |
| 500 | 0.35 | 0.35 | 5 |
| 1000 | 0.69 | 0.70 | 5 |
| 2000 | 1.32 | 1.34 | 3 |
| 3000 | 1.88 | 1.98 | 3 |
| 5000 | 2.41 | 2.74 | 4 |

## 3. Batch Throughput (1000 sequential calls)

- **Total time**: 0.03 s
- **Throughput**: 36886.9 calls/s
- **Mean latency**: 0.03 ms/call

## 4. Memory Usage (Peak RSS)

- **Before init**: 425.5 MB
- **After init + JAX warmup**: 425.5 MB
- **After 500-call batch**: 425.5 MB
- **Growth during batch**: 0.0 MB
