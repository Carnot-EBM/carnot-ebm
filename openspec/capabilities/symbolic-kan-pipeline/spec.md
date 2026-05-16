# Symbolic-KAN Pipeline Integration Capability Specification

**Capability:** symbolic-kan-pipeline
**Version:** 0.1.0
**Status:** Draft

## Overview

Integrates the Symbolic-KAN prototype as a verifiable tier in the main reasoning pipeline.
Includes Neuro-Symbolic Energy-Based Models (NeSy-EBMs) symbolic encoders.

## Requirements

### REQ-SYMKAN-1751: Symbolic-KAN Pipeline Integration

The `ThreeTierPipeline` MUST support routing through the `SymbolicKANTier3` when applicable.
`scripts/experiment_1751_e2e.py` MUST instantiate `SymbolicKANTier3` using `load_symbolic_kan` and wire it into the `ThreeTierPipeline` as the `ising_pipeline`. The experiment script MUST output a result artifact to `results/experiment_1751_e2e.json` containing `honest_verdict`.

**Rationale:** To provide a verifiable tier that leverages the Symbolic-KAN model, it needs to be integrated into the E2E verification tier, allowing evaluation of its end-to-end performance and throughput.

**Acceptance criteria:**
- `scripts/experiment_1751_e2e.py` exists and can be executed.
- `scripts/experiment_1751_e2e.py` produces `results/experiment_1751_e2e.json`.
- The output JSON contains all required schema fields including `honest_verdict`.

### REQ-SYMKAN-2074: NeSy-EBM Symbolic Encoder Module

The system shall support a Neuro-Symbolic Energy-Based Model (NeSy-EBM) symbolic encoder module that maps deterministic constraints (like SMT rules) into differentiable energy penalty tensors.
- The module `python/carnot/embeddings/nesy_encoder.py` MUST implement a system to parse basic logic predicates and compile them to JAX tensors.
- It MUST provide full unit test coverage.

**Rationale:** To encode symbolic logic directly into the neural architecture, deterministic constraints must be mapped to energy penalties.

## Scenarios

### SCENARIO-SYMKAN-1751: End-to-End Evaluation of Symbolic-KAN Tier

**Given** a trained SymbolicKAN model directory (or synthetic stub)
**When** `scripts/experiment_1751_e2e.py` is executed
**Then** it successfully integrates `SymbolicKANTier3` as the Tier 3 verifier in `ThreeTierPipeline` and outputs the evaluation results to `results/experiment_1751_e2e.json`.

### SCENARIO-SYMKAN-2074: Compiling Predicates into Differentiable Energy Functions

**Given** an instance of `NeSyEncoder`
**When** a logic predicate (e.g. `VAR_0 == VAR_1`) is compiled
**Then** it returns a JAX-compatible energy function
**And** the energy function produces low energy when constraints are met and high energy when violated.
