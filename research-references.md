# Research References & Future Considerations

Items filed here are technologies, papers, repos, and ideas to consider
in future research milestones. The research conductor and planning agent
should read this file when designing new milestones.

## Inference Optimization

### RotorQuant — KV Cache Compression
- **Repo:** github.com/scrya-com/rotorquant
- **What:** Clifford algebra rotor-based KV cache compression for LLMs. 10.3x compression, better perplexity than Google TurboQuant, 28% faster decode, O(d) complexity via block-diagonal Givens/quaternion rotations.
- **Relevance:**
  1. Could enable running 1-3B+ models within same memory budget for the verify-repair pipeline (currently limited to 0.6-0.8B on CPU)
  2. Clifford algebra rotors (block-diagonal rotation math) could improve continuous Ising relaxation convergence
  3. llama.cpp integration path for lighter inference runtime in production
- **Tech:** Python + CUDA + Triton + Metal, targets llama.cpp
- **When to revisit:** When scaling local LLM size becomes a bottleneck, or exploring efficient inference for production

### HISA — Hierarchical Indexed Sparse Attention
- **Paper:** arxiv.org/abs/2603.28458
- **What:** Two-stage hierarchical sparse attention — block-level filtering then token-level refinement. Drop-in replacement for DeepSeek Sparse Attention, no retraining. Significant speedups at 64K+ context.
- **Relevance:** Not useful for current short-context verify-repair calls (<500 tokens). Becomes relevant when Carnot handles long-context verification (full codebases, multi-turn repair with conversation history, batch-packed examples).
- **When to revisit:** When adding long-document or codebase-level verification to the pipeline

## EBM Ecosystem

### Extropic / thrml
- **Repo:** github.com/extropic-ai/thrml
- **What:** Thermodynamic sampling framework targeting Extropic TSU hardware
- **Relevance:** Carnot's parallel Ising sampler is 183x faster than thrml on CPU. TSU abstraction layer (Exp 71) is ready for when hardware ships.
- **Known issue:** thrml crashes on ROCm (extropic-ai/thrml#41)

### TorchEBM
- **What:** PyTorch-based EBM framework
- **Relevance:** Alternative approach to Carnot's JAX-based pipeline. Worth monitoring for ideas.

### EB-JEPA
- **What:** Energy-Based Joint Embedding Predictive Architecture
- **Relevance:** JEPA-style context prediction was explored early (Exp 1-38 era) but activation-based approaches proved insufficient. May become relevant again with continuous Ising (Exp 64+).

## Hardware

### FPGA Ising Machine — TSU Simulation Before Hardware
- **What:** Implement parallel Ising sampling in RTL on FPGA as a hardware
  stand-in for Extropic TSU. Each p-bit is a flip-flop with stochastic update
  based on neighbors and couplings. LFSR-based random number generation.
- **Why:** Test Carnot's hardware path before Z1 ships. FPGA gives true
  parallelism, custom bit-width, low latency, 10-100x better power than GPU.
- **Scale:** Small FPGA (Kria KV260, DE10-Nano) = 1k-10k p-bits (matches
  current experiment scale). Large FPGA (VU13P, Agilex) = up to 256k p-bits.
- **Integration:** Create `FpgaBackend` for SamplerBackend (Exp 71) that sends
  couplings over PCIe/AXI/USB and reads back sampled spins. Rest of pipeline
  stays in Python.
- **Prior art:** Tohoku University FPGA Ising machines, Microsoft Azure Quantum
  FPGA solver, Fujitsu Digital Annealer (commercial FPGA Ising machine).
- **Benchmark:** Compare FPGA vs CPU ParallelIsingSampler on Exp 46b (5000-var SAT).
- **What FPGA lacks vs TSU:** True thermal noise (FPGA uses pseudo-random LFSRs),
  analog-speed sampling (TSU = nanoseconds, FPGA = microseconds).
- **When to pursue:** When FPGA hardware is available for testing. Quickest path:
  1k-4k spin Verilog sampler with AXI-Lite interface + Python FpgaBackend wrapper.

### AMD ROCm on gfx1150 (Radeon 890M iGPU)
- JAX GPU backend crashes (HIP runtime assertion failure)
- gfx1100 emulation works but is 96x slower than CPU for matmul
- `JAX_PLATFORMS=cpu` is mandatory on this machine
- PyTorch ROCm works (3.3x speedup on Qwen3 inference)

## Autonomous Agent Frameworks

### AutoAgent — Declarative Agent Engineering
- **Repo:** github.com/kevinrgu/autoagent
- **What:** Meta-agent reads `program.md` (human-written goals), autonomously modifies `agent.py`, benchmarks via Harbor, hill-climbs on score. Overnight autonomous refinement.
- **Relevance:** Carnot's autoresearch is more sophisticated (three-gate + Ising, milestone planning, self-heal) but AutoAgent's `program.md` pattern is cleaner for expressing human intent. Borrowed this idea as `research-program.md`.
- **Borrowed:** Declarative intent document pattern → `research-program.md`

## Papers & Concepts

### Exp 66: End-to-End Differentiable Constraint Reasoning (PRIORITY)
- **Source:** research-roadmap-v7.md Phase 8
- **What:** Full Kona-like pipeline, differentiable end-to-end:
  LLM generates logits → soft token probabilities → embedding →
  continuous Ising constraints (Exp 64) → energy →
  backpropagate energy gradient through constraints to logits →
  adjust LLM sampling distribution toward constraint-satisfying tokens.
- **Prerequisites:** Exp 64 (continuous relaxation ✅) + Exp 65 (embedding-space ✅) + live LLM (Exp 56 ✅)
- **Why it matters:** Moves from post-hoc verification to real-time energy-guided decoding. Constraints steer generation, not just verify after the fact. This is the path to Kona parity.
- **When:** Next research milestone after production shipping (2026.04.4 ✅)

(Add more papers, arxiv links, and theoretical ideas here as they come up)
