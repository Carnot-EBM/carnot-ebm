# Carnot Research Roadmap: v124 (Phase-2 Latent Space Navigation, KANELÉ RTL Synthesis, and Energy-Guided Decoding)

## State of the Project (Milestone 123 Retrospective)
Milestone .123 successfully formalized our Exact-Rational KAN (RKAN) forward pass and bridged it to Z3. It proved the Energy-Based Constraint Network (EBCN) coherence scorer could operate without autoregressive generation and demonstrated Sparse KAN clustering for memory compression. Furthermore, the FR-11 continuous self-learning loop was scaled using the CerCE non-forgetting bounds, marking a major success in safe autonomous learning.

## The 3 Biggest Gaps (v124 Focus)
1. **Continuous Test-Time Scaling (Latent Navigation)**: While we have EBCN, we have not closed the loop on full gradient-based continuous latent space decoding (like Kona 1.0 or $\nabla$-Reasoner). We need Energy-Guided Test-Time Scaling (ETS) during inference.
2. **Formal MILP Verification for KANs**: RKAN translation to Z3 is limited. We need Piecewise Affine (PWA) abstractions to formally verify KAN logical bounds via MILP (Mixed Integer Linear Programming) per arXiv:2602.06737.
3. **Hardware KAN Mapping (KANELÉ)**: FPGA synthesis has focused on Ising and Potts. We need to implement KANELÉ-style direct LUT mapping (FPGA '26) to translate KAN edge splines into Verilog for our KV260 track.

## Phase 1: Continuous Latent Space Navigation & Energy-Guided Decoding
* **Exp 1614:** Archive .123 and initialize .124
* **Exp 1615:** Energy-Guided Test-Time Scaling (ETS). Implement online Monte Carlo energy estimation during inference.
* **Exp 1616:** $\nabla$-Reasoner Continuous Latent Optimization. Differentiable Textual Optimization over continuous logits via Langevin dynamics.
* **Exp 1617:** Live SOTA Validation of $\nabla$-Reasoner. Benchmark continuous reasoning on `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF`.

## Phase 2: Formal Verification of KANs (MILP Abstractions)
* **Exp 1618:** PWA Abstraction Layer for KANs. Implement piecewise affine boundaries for KAN splines (arXiv:2602.06737).
* **Exp 1619:** MILP Compilation of PWA KANs. Connect abstract KANs to PySAT/Z3 to establish formal correctness proofs.
* **Exp 1620:** Certify FR-11 Ledger via MILP. Use formal MILP bounds to enforce 100% safe continuous learning updates.

## Phase 3: Hardware Mapping (KANELÉ RTL)
* **Exp 1621:** KANELÉ LUT-Mapping Logic Synthesis. Construct Python-to-Verilog logic for KAN splines directly to LUTs.
* **Exp 1622:** KANELÉ RTL Linting and Simulation. Source-level Verilator/Icarus linting of KAN logic.
* **Exp 1623:** Latency/Resource Accounting for KANELÉ vs Ising. Theoretical logic depth modeling for KV260 deployment.

## Phase 4: Self-Learning & Consolidation
* **Exp 1624:** Adaptive Energy Landscape Reconfiguration. Implement Tier 4 self-learning via spectral constraint pruning.
* **Exp 1625:** EBM vs LLM Task Allocation Router. Heuristic routing of queries based on entropy.
* **Exp 1626:** Milestone .124 Retrospective.

## Hardware Dependencies
* **Local Dual RTX 3090**: Required for continuous gradient steps in Exp 1617.
* **KV260 Discrete RTL**: Simulator-only scope for Exp 1622. No board claim.
