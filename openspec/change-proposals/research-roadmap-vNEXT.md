# Research Roadmap vNEXT (Milestone 2026.05.163)

## Objective
Rebuild constraint extraction for instruction-tuned models, establish live GPU baselines, and formalize KAN verification using SMT solvers.

## What Previous Milestone Proved
Milestone 2026.05.162 established the continuous self-learning framework, scaffolded the base KAN architecture (Symbolic-KAN, GloroKAN robustness), and validated the Energy-Guided Decoding Scaling. Crucially, prior research (2026-04-11) revealed that early improvements were simulation artifacts. The constraint verification infrastructure works (0.006ms per check), but regex-based extraction failed on instruction-tuned models.

## Architecture Update

```mermaid
graph TD
    A[Instruction-Tuned LLM (Live GPU)] -->|Partial Generation| B(JEPA Predictive Verifier - Tier 3)
    B -->|Fast Path Skip| C[Output]
    B -->|Likely Violation| D[Constraint Extractor]
    D -->|SMT/Z3 Translation| E[KAN Formal Verification via MILP/PWA]
    D -->|LLM-as-Extractor| F[Ising/KAN Energy Landscape]
    E --> G[Energy Guided Decoding / Repair]
    F --> G
    G -->|Trace Pairs| H[Continuous Self-Learning Tier 1/2]
    H -.->|Constraint Addition| D
```

## Phase Descriptions

### Phase 1: Constraint Extraction Rebuild
The `ArithmeticExtractor` regex approach is obsolete. We will implement dual extraction pathways:
1. **SMT/Z3 Approach** (Inspired by arxiv 2601.17789): Formalizing constraints as first-order logic.
2. **LLM-as-Extractor**: Using `gemma-4-31B-it-GGUF` to parse constraints from CoT responses.
Testing will be strictly on live GPUs using instruction-tuned models to establish real baselines (GSM8K).

### Phase 2: Formal Verification of KANs
Building on the KAN architecture scaffolded in Milestone 162, we integrate formal verification techniques from recent literature (arxiv 2602.06737, KAN4CBC). We will implement Piecewise Affine (PWA) abstractions for `CarnotKAN` and translate constraint satisfaction into Mixed Integer Linear Programs (MILP) solved via SMT.

### Phase 3: Code Verification (HumanEval) on Live GPU
Code verification operates via structural tests (execution) rather than regex matching, making it the most resilient path. We will instrument runtime evaluation with Ising-guided fuzzing and evaluate against HumanEval using live GPU inference.

### Phase 4: Continuous Self-Learning Advancement
Implementing Tier 1 (Online Constraint Learning) by enabling constraint *addition* directly from Tier 2 memory patterns. Furthermore, we scaffold Tier 3 (JEPA-Style Predictive Verification) to predict violations from partial LLM responses, enabling fast-path generation skips.

## Dependency Graph
- Phase 1 (Extraction) must precede Phase 3 (Evaluation).
- Phase 2 (KAN Verification) runs in parallel to Phase 1.
- Phase 4 (Learning) requires functional extraction from Phase 1.

## Hardware Requirements
- **Primary:** Dual RTX 3090 CUDA local SOTA runtime for live inference.
- **Secondary (Pending):** Kria KV260 FPGA (requires Vivado installation for v3 RTL validation).
- **Secondary (Pending):** AMD XDNA NPU (blocked on ninja/openblas/VitisAI).
