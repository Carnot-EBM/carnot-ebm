# Milestone 2026.05.156: Robust Constraint Extraction, Advanced Neural Solvers, and Constraint Memory

**Status:** Proposed
**Author:** Carnot Research Conductor
**Date:** 2026-05-12

## Overview
This milestone addresses the critical finding that previous positive GSM8K results were simulation artifacts due to crude regex-based constraint extraction. The primary focus is to rebuild constraint extraction using formal logic (NSVIF/Z3) and LLM-as-extractor patterns on live, instruction-tuned SOTA models. Additionally, we integrate advanced unsupervised neural solvers (RUN-CSP, DeepSaDe) and implement Tier 2 Constraint Memory for continuous self-learning.

## Architecture Context
Carnot's constraint verification infrastructure (Ising, KAN) is fast and solid. The weak link is turning model outputs into constraints that the verifier can check. This milestone replaces the `ArithmeticExtractor` with robust formal extractions.

## Phase Descriptions

### Phase 1: Constraint Extraction & Verification Baselines
Rebuild the constraint extraction layer to handle real instruction-tuned models. Use NSVIF/Z3 SMT approaches and LLM-as-extractor techniques. Establish live GPU baselines on GSM8K and HumanEval using these new extractors on the mandated SOTA GGUF models.

### Phase 2: Advanced Constraint Solvers
Implement neural structures that guarantee constraint satisfaction: DeepSaDe for domain constraints and RUN-CSP for message passing networks on binary CSPs. Also integrate COLD Decoding for energy-based constrained text generation.

### Phase 3: Continuous Self-Learning
Fulfill FR-11 by implementing Tier 2 Constraint Memory. Track constraints across sessions, consolidating patterns into reusable templates that can be hardware-accelerated for pattern matching.

### Phase 4: KAN Refinement & Audit
Deploy adaptive energy landscapes using KAN splines (Tier 4 learning) and perform comprehensive pre-retro audits and the final retrospective.

## Hardware Requirements
- **Primary:** Local GPU (2x RTX 3090) for live SOTA inference (unsloth/Qwen3.6-35B-A3B-GGUF, unsloth/gemma-4-31B-it-GGUF).
- **Secondary:** CPU for constraint memory tracking and Ising sampling.