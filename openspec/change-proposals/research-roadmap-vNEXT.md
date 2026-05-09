# Carnot Research Roadmap vNEXT (Milestone 2026.05.123)

## 1. What the Previous Milestone Proved (.122)
Milestone .122 proved that we can build a bounded instruction-to-constraint DSL (NSVIF/RvLLM hook) that compiles down to Carnot constraints with zero false-accepts on mandated SOTA models. It also proved that DCCD and STATIC-style CSR-masks accelerate structured generation efficiently. Continuous self-learning was safely guarded with a CerCE-style certificate ledger ensuring no forgetting.

## 2. The 3 Biggest Gaps to PRD Vision
1. **Continuous Latent Repair Integration**: While VerifyRepairPipeline exists and DCCD handles text-level structural drafting, continuous latent space reasoning with gradient editing (Energy-Based Constraint Networks / Kona-style) is required for Phase 3/4 to fix deep logical contradictions without sequence regeneration.
2. **Formal Verification of the Energy Tiers**: The existing KAN energy tier lacks bit-identical formal correctness guarantees. Exact-Rational KANs (RKANs) provide a path for formal equivalence checking in Lean 4.
3. **Adaptive Energy Landscape Structure**: We have Tier 1/2 learning (weights/memory) and CerCE bounds, but Tier 4 requires the structure of the energy function itself to adapt (e.g., via Sparse KAN clustering with spectral constraints).

## 3. Architecture Diagram
```
[LLM (SOTA GGUF)] --> [Continuous Latent State]
                            |
                            v
[EBCN Scorer] <------ [Latent Gradient Editing]
      |
      v
[RKAN Energy Tier (Exact Rational)] <--> [Sparse KAN Clustering]
      |
      v
[Z3/PySAT Validator] --> [DSL Constraints]
```

## 4. Phase Descriptions
- **Phase 0: Infrastructure**: Archive .122 and set up .123.
- **Phase 1: Formal & Spectral KANs**: Build RKAN prototype with exact rational arithmetic, bridge to Z3, and prototype Sparse KAN clustering.
- **Phase 2: Continuous Latent Reasoning & EBCNs**: Prototype an EBCN dual-head attention scorer for structural coherence and test latent gradient editing.
- **Phase 3: Continuous Self-Learning**: Scale CerCE-ledger FR-11 to 1000 cases to prove Tier 4 structural adaptation without forgetting.
- **Phase 4: Production Integration**: Scale DCCD and DSL extraction to HumanEval and multi-hop reasoning, using mandated SOTA models, followed by milestone retro.

## 5. Hardware Requirements
- **Local GPUs**: 2x RTX 3090 (48GB total) for running mandated SOTA GGUF models.
- **CPU**: Standard x86 for RKAN rational arithmetic validation.

## 6. Dependency Graph
- exp1601 -> exp1602 -> exp1610
- exp1601 -> exp1603 -> exp1605 -> exp1611
- exp1601 -> exp1604 -> exp1608
- exp1601 -> exp1606
- exp1601 -> exp1607
- exp1601 -> exp1609
- exp1601 -> exp1612
- exp1601 -> exp1613
