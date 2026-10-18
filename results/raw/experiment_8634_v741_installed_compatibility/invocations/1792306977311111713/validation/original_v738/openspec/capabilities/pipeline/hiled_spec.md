# HILED Decoding Latency

**Capability:** pipeline
**Version:** 0.1.0
**Status:** Draft

## Requirements

### REQ-HILED-1719: HILED Latency Measurement
The system MUST measure the latency of Hardware-In-The-Loop Energy Decoding (HILED) per token. The artifact `results/experiment_1719_latency.json` MUST contain the keys `latency_per_token_hiled_ms`, `latency_per_token_baseline_ms`, `projection_tax_ms`, and `honest_verdict`.

## Scenarios

### SCENARIO-HILED-1719: Measure Latency
**Given** a HiledDecoder instance
**When** benchmarking with and without HILED
**Then** the projection tax is measured and recorded in the result JSON.
