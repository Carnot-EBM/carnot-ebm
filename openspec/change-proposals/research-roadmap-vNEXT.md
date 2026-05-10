# Milestone 2026.05.129: Energy-Driven Steering, KArAt Attention, and CRANE Decoding

## Context and Vision

This milestone tackles the fundamental gap between strict zero-false-accept parsing and high-quality semantic reasoning. As observed in prior milestones (including `.128` NSVIF extraction and EBCN scoring), forcing LLMs into early structured formatting creates a "reasoning tax" where the model's performance degrades. Furthermore, while we have established scalar energy scores for tracing contradictions, we still rely heavily on standard autoregressive decoding.

To bridge Carnot toward Kona-style continuous latent trace generation and true Energy-Based Models (EBMs), we will introduce three new capabilities in this milestone:
1.  **Energy-Driven Steering (EDS):** Direct guidance of LLM hidden states using external energy gradients.
2.  **CRANE Decoding:** An interleaved mechanism that separates free-form natural language "thinking" from structured "constraining".
3.  **Kolmogorov-Arnold Attention (KArAt):** An architectural exploration replacing traditional attention with explicitly learnable, verifiable spline/rational bases.

## What Previous Milestones Proved

Milestone `.128` proved that we can achieve zero false-accept parsing using the NSVIF bounded DSL and that EBCN can successfully score the structural coherence of logical traces. However, it also confirmed that strict grammar parsing early in the generation process lowers the semantic validity of the resulting trace.

## Phase Descriptions

### Phase 1: Decoding and Steering Integration (EDS & CRANE)
The first phase integrates Energy-Driven Steering (EDS) to guide internal hidden states away from high-energy (invalid) paths. In parallel, we implement CRANE decoding to mitigate the reasoning tax. By interleaving free-form unconstrained steps and strictly constrained generation, the model can reason safely before being forced to emit a structured certificate.

### Phase 2: KArAt Architecture Exploration
The second phase explores Kolmogorov-Arnold Attention (KArAt). We will substitute standard Softmax MLPs in a miniature model with rational function KANs. This tests the hypothesis that explicitly learnable bases in the attention mechanism yield better localized energy bounds than traditional attention maps.

### Phase 3: Continual Self-Learning Validation
Following the PRD mandate for continuous non-forgetting learning, we will deploy a CerCE-style ledger test against the EDS/CRANE stack to ensure the newly added constraints do not introduce recursive drift.

## Dependency Graph
`exp1677-eds-prototype` (Independent)
`exp1678-crane-decoding` (Independent)
`exp1679-karat-attention` (Independent)
`exp1680-continual-learning-eds` (Depends on exp1677 and exp1678)
`exp1681-milestone-retro` (Depends on all)

## Hardware Requirements
- Local Dual RTX 3090 (for local SOTA GGUFs inference: Qwen3.6-35B-A3B-GGUF and gemma-4-31B-it-GGUF)
- CPU Fallback for architecture testing and simulator parity tasks.