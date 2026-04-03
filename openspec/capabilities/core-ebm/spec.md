# Core EBM Capability Specification

**Capability:** core-ebm
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-01, FR-05, FR-08

## Overview

Defines the core abstractions for Energy Based Models: energy functions, model state, and the trait/protocol interfaces that all tiers implement.

## Requirements

### REQ-CORE-001: Energy Function Trait (Rust)

The system shall define a Rust trait `EnergyFunction` with the following interface:
- `energy(&self, x: &ArrayView) -> f32` — compute scalar energy for input x
- `energy_batch(&self, xs: &ArrayView2) -> Array1<f32>` — compute energy for a batch
- `grad_energy(&self, x: &ArrayView) -> Array1<f32>` — compute gradient of energy w.r.t. x

### REQ-CORE-002: Energy Function Protocol (Python/JAX)

The system shall define a Python protocol `EnergyFunction` with:
- `energy(self, x: jax.Array) -> jax.Array` — scalar energy
- `energy_batch(self, xs: jax.Array) -> jax.Array` — batched energy
- `grad_energy(self, x: jax.Array) -> jax.Array` — auto-derived via `jax.grad`

### REQ-CORE-003: Model State

The system shall define a `ModelState` type (Rust struct / Python dataclass) containing:
- `parameters` — model parameters (weights, biases)
- `config` — model hyperparameters
- `metadata` — training metadata (step count, loss history)

### REQ-CORE-004: Serialization

Model state shall be serializable to/from safetensors format in both Rust and Python.

### REQ-CORE-005: PyO3 Binding Interface

The system shall expose Rust `EnergyFunction` implementations to Python via PyO3, allowing Python code to call Rust energy computations with zero-copy numpy array transfer where possible.

### REQ-CORE-006: Numeric Precision

The system shall default to f32 precision with configurable f64 support via a compile-time feature flag (Rust) or runtime configuration (Python).

## Scenarios

### SCENARIO-CORE-001: Compute Energy for Single Input

**Given** a model implementing `EnergyFunction`
**When** `energy(x)` is called with a 1-D input vector
**Then** a scalar energy value is returned
**And** the value is finite (not NaN or Inf)

### SCENARIO-CORE-002: Compute Batch Energy

**Given** a model implementing `EnergyFunction`
**When** `energy_batch(xs)` is called with a 2-D batch of inputs
**Then** a 1-D array of energy values is returned with length equal to batch size
**And** all values are finite

### SCENARIO-CORE-003: Compute Energy Gradient

**Given** a model implementing `EnergyFunction`
**When** `grad_energy(x)` is called with a 1-D input
**Then** a 1-D gradient vector is returned with same shape as input
**And** gradient is numerically consistent with finite-difference approximation (rtol=1e-4)

### SCENARIO-CORE-004: Serialize and Deserialize Model

**Given** a model with trained parameters
**When** the model is serialized to safetensors and deserialized
**Then** the deserialized model produces identical energy values for the same inputs

### SCENARIO-CORE-005: PyO3 Round-Trip

**Given** a Rust EnergyFunction exposed via PyO3
**When** called from Python with a numpy array
**Then** the returned energy matches the Rust-native computation
**And** no data copy occurs for contiguous arrays

## Implementation Status

| Requirement | Rust | Python | Tests |
|------------|------|--------|-------|
| REQ-CORE-001 | Not Started | N/A | Not Started |
| REQ-CORE-002 | N/A | Not Started | Not Started |
| REQ-CORE-003 | Not Started | Not Started | Not Started |
| REQ-CORE-004 | Not Started | Not Started | Not Started |
| REQ-CORE-005 | Not Started | Not Started | Not Started |
| REQ-CORE-006 | Not Started | Not Started | Not Started |
