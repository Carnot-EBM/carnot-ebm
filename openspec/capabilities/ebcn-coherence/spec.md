# EBCN Coherence Capability Specification

**Capability:** ebcn-coherence
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-12

## Requirements

### REQ-EBCN-001: Dual-Head Attention State-Space Model

The system shall implement an EBCN prototype (`python/carnot/models/ebcn_coherence.py`) for scoring the coherence of reasoning traces.
The model must be a dual-head attention state-space model that produces a scalar energy score for contradiction detection.

## Scenarios

### SCENARIO-EBCN-001: Contradiction Detection

**Given** a set of logical reasoning traces (some coherent, some containing contradictions)
**When** the EBCN coherence model processes these traces
**Then** the model produces a scalar energy score for each trace
**And** the energy score is higher for traces containing logical contradictions compared to coherent traces.
