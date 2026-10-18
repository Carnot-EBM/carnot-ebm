# Milestone 2026.05.161: Mouth/Brain EBT Architecture, ARM-EBM Bijection, and Thermodynamic Hardware Sampling

**Status:** Proposed
**Date:** 2026-05-13

## Context & Vision Gap

The current PRD vision drives towards autonomous, continuous learning energy-based verification. Milestone .160 completed continuous latent equilibrium matching and AIA hardware simulation, taking us closer to Fast Autoregressive continuous generation. 

However, recent findings present massive opportunities that directly map to our PRD gaps:
1. **The "Mouth vs Brain" Separation (System 2):** Energy-Based Transformers (EBTs) and Logical Intelligence's Kona 1.0 architecture have proven the viability of separating the language generator ("mouth") from the energy-based verifier ("brain"), reducing hallucinations significantly on structured tasks like Sudoku. 
2. **ARM-EBM Mathematical Bijection:** The finding that ARMs are secretly EBMs (arXiv:2512.15605) via the soft Bellman equation provides a theoretical mechanism to map LLM logprobs directly to energy values, solving our long-standing lookahead integration bottlenecks.
3. **Hardware-Accelerated Sampling:** Extropic AI's TSU (Thermodynamic Sampling Units) and the upcoming Z1 chip offer a path away from simulated AIA to actual native thermodynamic sampling using the `thrml` library.

This milestone updates Carnot's codebase to leverage these exact paradigms.

## Phase 1: Foundations & Architecture Upgrades (Mouth vs. Brain)
We establish a strict architectural boundary between the LLM and the Carnot verifier, implementing EBT computation layers and compositional energy landscapes to tackle multi-step reasoning.

## Phase 2: ARM-EBM Soft Bellman Distillation
We implement the mathematical bijection to map ARM logits to EBM energy bounds. This enables Energy-Based Fine-Tuning (EBFT) pipelines directly against Carnot's verifiers.

## Phase 3: Thermodynamic Hardware Readiness
We transition from simulated AIA to native Extropic TSU integrations, utilizing the `thrml` SDK stubs and scaffolding the Denoising Thermodynamic Model (DTM).

## Phase 4: Continuous Self-Learning & E2E Verification
We close the PRD loop with an unsupervised continuous self-learning system that refines its own energy landscape, testing it on a Kona-style system reasoning benchmark.

## Task Graph
- Phase 1: Exp 2053 -> 2054 -> 2055
- Phase 2: Exp 2053 -> 2056 -> 2057 -> 2058
- Phase 3: Exp 2053 -> 2059 -> 2060 -> 2061
- Phase 4: [2055, 2058, 2061] -> 2062 -> 2063
- Wrap: 2063 -> 2064 -> 2065
