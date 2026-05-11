# Symbolic-KAN Pipeline Integration Capability Specification

**Capability:** symbolic-kan-pipeline
**Version:** 0.1.0
**Status:** Draft

## Overview

Integrates the Symbolic-KAN prototype as a verifiable tier in the main reasoning pipeline.

## Requirements

### REQ-SYMKAN-1751: Symbolic-KAN Pipeline Integration

The `ThreeTierPipeline` MUST support routing through the `SymbolicKANTier3` when applicable.
`scripts/experiment_1751_e2e.py` MUST instantiate `SymbolicKANTier3` using `load_symbolic_kan` and wire it into the `ThreeTierPipeline` as the `ising_pipeline`. The experiment script MUST output a result artifact to `results/experiment_1751_e2e.json` containing `honest_verdict`.

**Rationale:** To provide a verifiable tier that leverages the Symbolic-KAN model, it needs to be integrated into the E2E verification tier, allowing evaluation of its end-to-end performance and throughput.

**Acceptance criteria:**
- `scripts/experiment_1751_e2e.py` exists and can be executed.
- `scripts/experiment_1751_e2e.py` produces `results/experiment_1751_e2e.json`.
- The output JSON contains all required schema fields including `honest_verdict`.

## Scenarios

### SCENARIO-SYMKAN-1751: End-to-End Evaluation of Symbolic-KAN Tier

**Given** a trained SymbolicKAN model directory (or synthetic stub)
**When** `scripts/experiment_1751_e2e.py` is executed
**Then** it successfully integrates `SymbolicKANTier3` as the Tier 3 verifier in `ThreeTierPipeline` and outputs the evaluation results to `results/experiment_1751_e2e.json`.
