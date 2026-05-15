# Carnot Research Roadmap — Milestone 2026.05.185

**Title:** Continuous Self-Learning Integration, Fast-Slow Scaling, and KAN Verification
**Author:** outer-loop-claude
**Status:** Active

## Overview

Milestone .184 landed the Fast-Slow Variant prototype (Exp 1761), attempted the stranded PyPI push (Exp 1762), audited .183 findings (Exp 1763), and produced a decision artifact comparing the thermodynamic metric to the Fast-Slow variant (Exp 1764). 

Milestone .185 transitions from prototyping to scaling the Fast-Slow paradigm to the mandated local SOTA GGUF models (`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`). It bridges the gap in the Continuous Self-Learning (FR-11) requirement by embedding Token-Level Energy (arXiv:2605.14558) and Fast-Slow training (arXiv:2605.12484) to prevent catastrophic forgetting. Additionally, it addresses Phase 2 hardware-efficient verification with a KAN-to-PWA (Piecewise Affine) compiler for MILP-based formal property bounds.

## Top 3 Priority Gaps

1. **Continuous Self-Learning Stability:** Previous FR-11 iterations suffered from mode collapse or zero utility delta. We must scale the Fast-Slow Variant (arXiv:2605.12484) and Token-Level Energy metrics (arXiv:2605.14558) to stabilize learning without forgetting.
2. **Hardware-Efficient Formal Bounds:** KAN verification remains ad-hoc. The PWA abstraction (arXiv:2602.06737) and E-MVL sparse RTL (arXiv:2604.04606) provide a concrete path for formal MILP verification and KV260 hardware synthesis without blowing out LUT budgets.
3. **Structured Constraint Extraction:** We need an automated compiler that translates ROCE-extracted constraints (arXiv:2605.01124) into KAN representations and verifies them asynchronously during text generation (interwhen, arXiv:2602.11202).

## Phase Plan

### Phase 1: SOTA Fast-Slow & Telemetry
Scale the fast-weight context buffers to mandated GGUF models. Implement token-level energy metrics to provide fine-grained verification signals during generation.

### Phase 2: Formal Bounds & Extraction
Compile extracted ROCE constraints into formal Piecewise Affine (PWA) abstractions. Validate these abstractions against existing Z3/PySAT benchmarks.

### Phase 3: Hardware Translation & Continuous Loop
Synthesize the sparse E-MVL v4 RTL for KV260 accounting. Integrate the Fast-Slow continuous learning mechanism into the main autoresearch loop.

### Phase 4: Integration & Audit
Run the E2E verification cascade, audit the newly integrated verifiers, and generate the end-of-milestone operational retrospective.

## Hardware Requirements
- **Local:** Dual RTX 3090 (for SOTA GGUF inference and token-level telemetry computation)
- **KV260:** CPU-only no-synthesis accounting for the E-MVL RTL constraints.

## Dependency Graph
- Exp 1766 (Token Energy) -> Exp 1768 (SOTA Fast-Slow) -> Exp 1772 (Continual EBM)
- Exp 1769 (ROCE-to-KAN) -> Exp 1770 (PWA KAN) -> Exp 1771 (T-SKM Projection)
- Exp 1774 (E-MVL RTL) -> Exp 1776 (interwhen Test-Time)