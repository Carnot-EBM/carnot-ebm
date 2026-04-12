# Epic: SAMPLE-009 - KV260 FPGA Ising Sampler Design And Software Model

**Status:** Complete
**Goal:** Add a spec-backed FPGA Ising sampler design for the Kria KV260,
implement `python/carnot/samplers/fpga_ising.py` as a `SamplerBackend`
adapter with PYNQ-overlay hooks plus a software model, define the AXI-Lite
register contract for dynamic coupling updates, and benchmark the simulated
control path against the existing CPU sampler.
**Rationale:** Carnot already has a CPU Ising sampler and a backend
abstraction. The KV260-class FPGA path needs to be specified and exercised in
software now so the repository can validate dynamic coupling uploads, fallback
behavior, and the hardware-control contract before the actual overlay bitfile
is available.

## Stories
- [x] Add `REQ-SAMPLE-005`, `REQ-SAMPLE-006`, and
  `SCENARIO-SAMPLE-009` through `SCENARIO-SAMPLE-011` to the
  `training-inference` spec before implementation changes
- [x] Write tests first for the FPGA register map, software-model upload and
  readback, backend fallback behavior, `get_backend("fpga")`, and benchmark
  output
- [x] Implement `python/carnot/samplers/fpga_ising.py`
- [x] Integrate the new backend into the sampler factory and package exports
- [x] Document the 4K-spin p-bit array architecture and AXI-Lite register map
  in `docs/fpga-ising-design.md`
- [x] Run targeted 100% coverage for the new module plus the required Python
  suite, spec-coverage, lint, type-check, and applicable E2E/integration
  checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
