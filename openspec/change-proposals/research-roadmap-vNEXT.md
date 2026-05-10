# Milestone 2026.05.128: NSVIF Constraint Extraction, EBCN Coherence, and Continuous Non-Forgetting

**Status:** Proposed
**Author:** Research Planning Agent
**Target Date:** 2026-05-12

## Context and Completed Work (.127)
Milestone `.127` demonstrated Energy-Guided Decoding, SMGI Continuous Learning, and EBRM Trace Scoring. However, significant gaps remain. The main product gap is extracting actionable constraints from natural language (NSVIF). Continuous self-learning exhibits intermittently positive utility but lacks a certified non-forgetting guarantee (CerCE). Hardware/simulation paths require independent-RNG audits for simulator parity.

## Vision and Primary Objectives
This milestone addresses the three largest gaps to fulfilling the PRD vision:
1. **Constraint Extraction & Structural Coherence:** Implement Neuro-Symbolic Verification on Instruction Following (NSVIF) parsing and Energy-Based Constraint Networks (EBCN).
2. **Continuous Self-Learning & Certification:** Add CerCE-style certified non-forgetting ledgers to FR-11 to guarantee zero regression during policy promotion.
3. **Structured EGD and Hardware Readiness:** Port Energy-Guided Decoding for constraint hallucination mitigation, conduct the mandatory THRML independent-RNG parity audit, and prototype Exact-Rational KANs (RKANs) for verifiable KAN tiers.

## Phase Descriptions

### Phase 1: NSVIF Extraction and EBCN Structural Coherence
* **Objective:** Enable Carnot to turn prompts into executable constraints and score reasoning structurally.
* **Key Deliverables:** NSVIF DSL Parser, EBCN State-Space Coherence Scorer.

### Phase 2: CerCE Certified Non-Forgetting for FR-11
* **Objective:** Guarantee that continuous self-learning does not suffer from mode collapse or forgetting.
* **Key Deliverables:** CerCE bounds checking ledger for FR-11 policy updates, LTLZinc continual learning benchmark adapter.

### Phase 3: Energy-Guided Decoding and Formal RKANs
* **Objective:** Validate hyperparameter-free energy-guided decoding on SOTA models and formally verify KAN properties.
* **Key Deliverables:** EGD prototype, RKAN (Exact-Rational KAN) verification audit in Lean 4 (or simulation), Interleaved Gibbs Diffusion (IGD) smoke test.

### Phase 4: Hardware Simulation Fidelity
* **Objective:** Ensure simulation artifacts are distinct and empirically sound before hardware claims.
* **Key Deliverables:** THRML Independent-RNG Parity Audit, Inertial Ising Machine (PIPIM) simulator updates, LagONN prototype.

## Dependency Graph
- **Phase 1** must complete before Phase 2 uses the new structured coherence for memory evaluation.
- **Phase 3** depends on Phase 1 for NSVIF schemas.
- **Phase 4** is standalone and can proceed concurrently.

## Hardware Requirements
- **Local SOTA GGUFs:** `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, `unsloth/gemma-4-26B-A4B-it-GGUF`
- **Simulation:** THRML package, CPU execution, No TSU/FPGA claims allowed without transcript.
