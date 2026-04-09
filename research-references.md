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

### AMD ROCm on gfx1150 (Radeon 890M iGPU)
- JAX GPU backend crashes (HIP runtime assertion failure)
- gfx1100 emulation works but is 96x slower than CPU for matmul
- `JAX_PLATFORMS=cpu` is mandatory on this machine
- PyTorch ROCm works (3.3x speedup on Qwen3 inference)

## Papers & Concepts

(Add papers, arxiv links, and theoretical ideas here as they come up)
