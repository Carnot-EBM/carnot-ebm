# Phase-18: KAN-Guided EBM Generation, Dual-GPU Scaling, and MoE Distillation
Milestone: 2026.05.141

## Overview
Phase-18 addresses the critical operational gaps identified in Milestone 2026.05.140 while integrating state-of-the-art continuous learning architectures from 2025-2026 literature. Our goal is to achieve reliable Dual RTX 3090 evaluation for scaling verify-repair to 3B models, embedding a continuous self-learning loop for MoE routers, and extending hardware acceleration for continuous constraint landscapes.

## Context and Gaps Addressed
The previous milestone retrospective (`experiment_1813_retro.json`) highlighted three critical gaps:
1. **Dual RTX 3090 GPU Baseline:** Essential for throughput and latency benchmarking of larger models.
2. **3B Model Scaling:** We need empirical evidence that our Verify-Repair pipelines remain efficacious when scaling to the 3B parameter regime.
3. **DEFINITIVE GSM8K Benchmark:** Establishment of real GPU inference baselines on RTX 3090.

## ArXiv Findings Integration
Recent literature has provided architectural foundations for this phase:
- **arXiv:2509.11234 (Thermodynamic Gradients):** Methods to map discrete logic constraints onto continuous energy landscapes, which we will apply to the Bounce-Bind Ising Machine (BBIM) for KV260 deployment.
- **arXiv:2602.04567 (Continuous KAN Verifiers):** Using Kolmogorov-Arnold Networks as high-efficiency, real-time constraint satisfaction layers for intermediate LLM decoding.
- **arXiv:2604.08912 (Online Distillation into MoE):** Continuous online distillation of constraint successes into MoE routers, which directly satisfies our PRD requirement for continuous self-learning.

## Phase 1: Dual-GPU Baseline & Benchmark Infrastructure
We first establish the physical baseline, verifying dual-GPU execution and instantiating the SOTA local GGUF pipelines.
- **Exp 1814:** Dual RTX 3090 GPU Setup and VRAM Profiling
- **Exp 1815:** 3B SOTA Model Inference Pipeline (Integration of Qwen3.6-35B-A3B-GGUF, gemma-4-31B-it-GGUF, gemma-4-26B-A4B-it-GGUF)
- **Exp 1816:** Baseline GSM8K on SOTA Models without Verification

## Phase 2: Verifier Scaling & Continuous Self-Learning
We introduce the KAN verifier and test verify-repair at the 3B scale, followed by implementing the continuous self-learning loop.
- **Exp 1817:** Implement Continuous KAN Verifier (arXiv:2602.04567)
- **Exp 1818:** Verify-Repair Scaling on GSM8K using 3B Models
- **Exp 1819:** Evaluate KAN Decoding Latency vs Accuracy
- **Exp 1820:** Continuous Online Distillation of EBM Constraints into MoE Routers (arXiv:2604.08912)

## Phase 3: Hardware Acceleration & Final Evaluation
We push the continuous mapping down to RTL synthesis and combine all systems for the capstone evaluation.
- **Exp 1821:** Map Thermodynamic Gradients to BBIM (arXiv:2509.11234)
- **Exp 1822:** FPGA Bitstream Synthesis for Continuous EBM Constraints
- **Exp 1823:** EBM-CoT GSM8K Final Evaluation with Continuous Self-Learning
- **Exp 1824:** Milestone 2026.05.141 Retrospective

## Hardware Requirements
- Dual RTX 3090 GPUs (Local Node)
- Kria KV260 Vision AI Starter Kit
- OSS-CAD-Suite (Yosys, NextPNR) for Bitstream Synthesis

## Dependency Graph
Phase 1 -> Phase 2 -> Phase 3. 
Exp 1815 requires Exp 1814 success.
Exp 1816 requires Exp 1815 success.
Exp 1823 depends on Exp 1817, 1820, and 1815.