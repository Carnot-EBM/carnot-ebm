# Model Tiers Capability Specification

**Capability:** model-tiers
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-02, FR-03, FR-04

## Overview

Defines the three model tiers — Boltzmann (large), Gibbs (medium), Ising (small) — each implementing the core EnergyFunction interface with tier-appropriate architectures and constraints.

## Requirements

### REQ-TIER-001: Ising Model (Small Tier)

The system shall provide an Ising-class EBM with:
- Single hidden layer or no hidden layers (direct pairwise interactions)
- Configurable input dimension and optional hidden dimension
- Suitable for binary/discrete data and teaching purposes
- Memory footprint < 10MB for default configuration

### REQ-TIER-002: Gibbs Model (Medium Tier)

The system shall provide a Gibbs-class EBM with:
- Multi-layer architecture (2-4 hidden layers)
- Configurable width and depth
- Support for continuous and mixed data types
- Suitable for applied ML tasks and domain adaptation

### REQ-TIER-003: Boltzmann Model (Large Tier)

The system shall provide a Boltzmann-class EBM with:
- Deep architecture with residual connections
- Configurable depth (4+ layers), width, and attention mechanisms
- Support for high-dimensional continuous data
- Suitable for research-scale generation tasks

### REQ-TIER-004: Tier Interface Conformance

All three tiers shall implement the `EnergyFunction` trait/protocol defined in core-ebm spec. A function written against the `EnergyFunction` interface shall work with any tier without modification.

### REQ-TIER-005: Tier-Specific Configuration

Each tier shall have a configuration struct/dataclass with tier-appropriate defaults:
- Ising: `IsingConfig { input_dim, hidden_dim: Option, coupling_init }`
- Gibbs: `GibbsConfig { input_dim, hidden_dims, activation, dropout }`
- Boltzmann: `BoltzmannConfig { input_dim, hidden_dims, num_heads, residual, layer_norm }`

### REQ-TIER-006: Parameter Initialization

Each tier shall support configurable parameter initialization strategies:
- Xavier/Glorot uniform (default)
- He/Kaiming normal
- Custom initializer function

## Scenarios

### SCENARIO-TIER-001: Ising Forward Pass

**Given** an Ising model with input_dim=784 (MNIST)
**When** energy is computed for a random binary input
**Then** a finite scalar energy is returned
**And** computation completes in < 1ms on CPU

### SCENARIO-TIER-002: Gibbs Forward Pass

**Given** a Gibbs model with input_dim=784, hidden_dims=[512, 256]
**When** energy is computed for a random continuous input
**Then** a finite scalar energy is returned
**And** computation completes in < 10ms on CPU

### SCENARIO-TIER-003: Boltzmann Forward Pass

**Given** a Boltzmann model with input_dim=784, hidden_dims=[1024, 512, 256, 128]
**When** energy is computed for a random continuous input
**Then** a finite scalar energy is returned

### SCENARIO-TIER-004: Tier Interchangeability

**Given** a generic function `fn compute<E: EnergyFunction>(model: &E, x: &Array)`
**When** called with Ising, Gibbs, and Boltzmann models respectively
**Then** all three calls succeed and return finite energies
**And** the function source contains no tier-specific code

### SCENARIO-TIER-005: Ising Memory Footprint

**Given** an Ising model with default configuration
**When** the model is instantiated
**Then** total parameter memory is < 10MB

### SCENARIO-TIER-006: Configuration Validation

**Given** a tier configuration with invalid values (e.g., input_dim=0)
**When** model construction is attempted
**Then** a descriptive error is returned (not a panic/exception)

## Implementation Status

| Requirement | Rust | Python | Tests |
|------------|------|--------|-------|
| REQ-TIER-001 | Implemented | Implemented | 10 Rust + 8 Python |
| REQ-TIER-002 | Implemented | Implemented | 7 Rust + 20 Python |
| REQ-TIER-003 | Implemented | Implemented | 7 Rust + 10 Python |
| REQ-TIER-004 | Implemented | Implemented | 3 Rust + 1 Python |
| REQ-TIER-005 | Implemented | Implemented | 3 Rust + 3 Python |
| REQ-TIER-006 | Implemented | Implemented | 4 Rust + 2 Python |
