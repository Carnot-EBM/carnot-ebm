# Milestone 2026.05.208: System-2 Compositional Energy Minimization, HardNet Integrations, and Constrained Augmented Generation

**Status:** Proposed
**Date:** 2026-05-16

## 1. Context and Motivation

Milestone `2026.05.207` achieved Kona-style continuous latent reasoning and NeSy-EBM symbolic encoding. However, to scale these latent energy spaces to highly complex problems, we must break down global constraints into compositional sub-landscapes. Recent 2025 literature highlights **Compositional Energy Minimization (PEM)** (arXiv:2510.20607) and **HardNet** (NeurIPS 2025) as necessary breakthroughs to enforce complex and strict logical bounds efficiently without sacrificing generation quality. Additionally, **CRANE** (ICML 2025) demonstrates how to interleave structural constraints with free-form chain-of-thought to prevent rigid grammar enforcement from crippling LLM reasoning capabilities.

This milestone addresses the following major gaps:
1. **Compositional Sub-Problems:** Energy minimization on a monolithic global Lagrangian can suffer from local minima. Parallel Energy Minimization (PEM) over composed sub-problem landscapes is needed.
2. **Hard Guarantees in Neural Forward Pass:** While NeSy MAP inference alternates updates, HardNet provides a closed-form differentiable enforcement layer ensuring hard constraints are met natively.
3. **Reasoning-Constraint Trade-offs in Generation:** Rigid grammar decoding harms CoT. We must implement CRANE-style augmented grammars to support reasoning blocks prior to strict syntax constraints.

## 2. Architecture Impact

- **Phase 3 (Continuous EBM) & Verification Pipeline:** Will incorporate Parallel Energy Minimization (PEM) to compose smaller EBM blocks (e.g., Sudoku subgrids or SMT clauses) dynamically.
- **Neural Constraint Embeddings:** A new HardNet differentiable layer module will be added alongside KAN and NeSy modules, offering guaranteed forward-pass bound satisfaction.
- **Inference Adapters:** The EBM-CoT sequence decoding adapters will be upgraded with CRANE-based augmented grammar logic.
- **Continuous Self-Learning (CSL):** The CSL loop will track energy prediction gradients on composed landscapes, persisting constraint violation frequencies (Tier 2 memory) across sessions.

## 3. Phase Descriptions

### Phase 0: Activation
- **Exp 2082:** Archive .207 and activate .208.

### Phase 1: Compositional Energy Minimization (PEM)
- **Exp 2083:** Implement an EBM Composition module that sums independent energy functions (e.g., row, col, subgrid for Sudoku).
- **Exp 2084:** Implement the Parallel Energy Minimization (PEM) optimizer based on arXiv:2510.20607.
- **Exp 2085:** Evaluate PEM vs. monolithic Lagrangian (from Phase 2071) on Hard Sudoku, proving escape from local minima.

### Phase 2: HardNet Differentiable Enforcement Layers
- **Exp 2086:** Develop the HardNet closed-form differentiable enforcement layer for multi-inequality logic bounds.
- **Exp 2087:** Integrate the HardNet layer into the Carnot verification pipeline as an alternative to KAN4CBC.
- **Exp 2088:** Benchmark zero-false-accepts on synthetic graph coloring constraints using the HardNet layer.

### Phase 3: CRANE Constrained Augmented Generation
- **Exp 2089:** Develop an Augmented Grammar Decoder for local GGUF supporting unconstrained CoT followed by strict BNF schemas.
- **Exp 2090:** Execute a 50-problem HumanEval pass using the CRANE decoder with structural output constraints.
- **Exp 2091:** Apply Tier-1 Continuous Self-Learning updates to the grammar probabilities based on constraint violation counts.

### Phase 4: Integration and Retrospective
- **Exp 2092:** Run an end-to-end integration test combining PEM, HardNet, and CRANE.
- **Exp 2093:** Full Milestone 2026.05.208 Retrospective.

## 4. Hardware Requirements
- **Local SOTA GGUFs:** `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF` required for Phase 3 generation tasks. 
- **GPU Cluster:** Standard 2x RTX 3090 configuration.

## 5. Dependency Graph
```text
Exp 2082 (Activation)
  |-- Phase 1 (PEM)
  |      |-- Exp 2083 --> Exp 2084 --> Exp 2085
  |-- Phase 2 (HardNet)
  |      |-- Exp 2086 --> Exp 2087 --> Exp 2088
  |-- Phase 3 (CRANE)
  |      |-- Exp 2089 --> Exp 2090 --> Exp 2091
  |-- Phase 4 (Integration)
         |-- Exp 2092 --> Exp 2093
```