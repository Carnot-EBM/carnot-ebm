# Rust/Python Boundary Capability Specification

**Capability:** rust-python-boundary
**Status:** Draft

## Requirements

### REQ-RUSTPY-6194: Mode-Jump PyO3 Boundary Contract

Carnot MUST expose the Exp6194 fixed mode-jump sampler through PyO3 without
silent fallback. The boundary SHALL preserve Rust sampler semantics exactly:
typed construction, energy/proposal queries, one-step, multi-step, state
snapshot, state restore, serialization, deserialization, deterministic seed
replay, and explicit invalid-input errors.

Sub-requirements:
- REQ-RUSTPY-6194-TYPED-CONSTRUCTION: PyO3 SHALL expose typed configuration,
  state, and core classes for the mode-jump sampler.
- REQ-RUSTPY-6194-STEP-TRACE: A one-step PyO3 call SHALL return the same trace
  fields as the Rust kernel: proposal uniform, proposed label, acceptance
  uniform, current/proposed energies, proposal log probabilities, log
  acceptance, acceptance probability, accept decision, counters, RNG state, and
  after-state snapshot.
- REQ-RUSTPY-6194-MULTI-STEP: A multi-step PyO3 call SHALL return deterministic
  frequencies, acceptance counts, final state, and sample count for a bounded
  positive step budget.
- REQ-RUSTPY-6194-SERIALIZATION: Snapshot/restore and serialization/
  deserialization SHALL round-trip exactly and reject corrupt labels, counters,
  schemas, and strings with explicit `ValueError` exceptions.
- REQ-RUSTPY-6194-NO-FALLBACK: The binding SHALL not silently route to a
  Python implementation when Rust construction, stepping, or serialization
  fails.
- REQ-RUSTPY-6194-NO-HARDWARE: The boundary SHALL not create or imply any FPGA,
  TSU, CUDA, THRML scaling, latency, power, energy, or speedup claim.

### SCENARIO-RUSTPY-6194-BOUNDARY-PARITY: Python Calls Rust And Replays The Fixture

**Given** the frozen Exp6194 Python transition fixture and Rust/PyO3 classes
**When** Python constructs the Rust configuration and state, runs exact steps,
serializes/restores state, and runs a long chain
**Then** the returned traces and diagnostics match the Python fixture and
invalid inputs raise explicit errors instead of falling back.

## Implementation Status (REQ-RUSTPY-6194)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-RUSTPY-6194 | Planned (`crates/carnot-python/src/mode_jump.rs`, `crates/carnot-python/src/lib.rs`, `python/carnot/_rust_compat.py`) | Planned (`tests/python/test_experiment_6194_mode_jump_rust_pyo3_parity.py`) |
