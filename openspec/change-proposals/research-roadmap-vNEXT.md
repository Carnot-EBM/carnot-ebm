# Milestone 2026.05.145: Phase 5 Continuous Verification Learning and Symbolic KANs

## Overview
This milestone addresses three major gaps between Carnot's current state and the PRD vision:
1. **Continuous Self-Learning (FR-11):** We must move beyond simple replay and implement unsupervised Verification Learning (VL), where constraint optimization replaces labeled data.
2. **Formal Verification of KANs:** Integration of Softly Symbolified KANs (S2KAN) and GloroKAN-style robustness to ensure our energy tiers are formally verifiable.
3. **Energy Matching:** Bringing continuous latent constrained generation to our verification loops.

## What Previous Milestone Proved
Milestone 2026.05.144 proved our Active Inference models and THRML scaling, but highlighted that our constraint extraction requires more robust structural guarantees, and our continuous learning pipelines need unsupervised learning paths to scale efficiently.

## Phases

### Phase 1: Continuous Verification Learning
Implementing Verification Learning (arXiv:2503.12917) to allow Carnot to learn directly from constraint satisfaction without labeled data. This directly addresses the FR-11 continuous self-learning requirements.

### Phase 2: Formal KAN Verification
Integrating S2KAN (symbolic primitives) and GloroKAN (Lipschitz bounding) into our energy tiers to enable formal algebraic and MILP verification of the KAN layers.

### Phase 3: Hardware Readiness and E2E Integration
Running these newly verified tiers and continuous loops against our SOTA GGUF models (`unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF`) and ensuring E2E verification across Rust and Python.

## Hardware Requirements
- Local Dual RTX 3090 CUDA runtime for SOTA GGUF inference.
- CPU/RAM for Rust verification compilation and testing.

## Dependency Graph
Phase 1 (Verification Learning) -> Phase 2 (Symbolic KAN) -> Phase 3 (E2E SOTA Tests)
