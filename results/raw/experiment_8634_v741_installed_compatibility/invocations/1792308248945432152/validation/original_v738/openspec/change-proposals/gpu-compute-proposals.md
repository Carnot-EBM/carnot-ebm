# GPU Compute Proposals: Vulkan Backend + WebGPU Distributed Compute

**Created:** 2026-04-04
**Status:** Proposed
**Triggered by:** User observation that an AMD GPU is available but JAX only sees CPU

---

## Proposal: Vulkan Compute Backend

### Problem

JAX defaults to CPU, and its GPU support requires CUDA (NVIDIA only). The development machine has an AMD GPU that goes unused. The Carnot framework should be GPU-vendor-agnostic.

### Solution: Vulkan Compute via jax-vulkan or custom backend

**Option A: JAX with XLA Vulkan backend**
- The XLA compiler (which JAX uses) has experimental Vulkan support
- This would make JAX GPU-accelerated on AMD, Intel, and NVIDIA without code changes
- Status: experimental, may not cover all XLA ops

**Option B: Custom Vulkan compute kernels for hot paths**
- Write Vulkan compute shaders for the critical operations:
  - `energy_batch()` — vectorized energy evaluation
  - `grad_energy()` — gradient computation  
  - `repair()` loop — the main optimization inner loop
  - NCE/SNL loss computation
- Call from Python via pyvulkan or wgpu-py
- More work but vendor-agnostic and production-ready

**Option C: Rust + wgpu**
- Use the Rust `wgpu` crate (WebGPU standard, works on Vulkan/Metal/DX12)
- The Carnot Rust crates already exist — add GPU kernels alongside CPU code
- Expose via PyO3 bindings
- Best of both worlds: production Rust + GPU acceleration + Python research

### Impact on Carnot

| Operation | CPU Time | Expected GPU Speedup |
|-----------|----------|---------------------|
| repair() 200 steps, dim=81 (Sudoku) | ~50ms | 5-10x |
| NCE training 100 epochs, batch=500 | ~10s | 20-50x |
| Diffusion generation 100 steps | ~1s | 10-20x |
| Autoresearch hypothesis evaluation | ~0.5s | 5x |

The biggest win is training (P5 optimization training needs Hessian-vector products — O(d²) per step).

---

## Proposal: WebGPU Distributed Compute Gateway

### Concept

A WebGPU gateway where **browsers can register as compute nodes** and contribute GPU cycles for Carnot workloads.

```
Browser (WebGPU)  ─┐
Browser (WebGPU)  ─┤
Browser (WebGPU)  ─┼──> Gateway Server ──> Carnot Orchestrator
Native Vulkan     ─┤
Cloud GPU         ─┘
```

### Architecture

**Gateway Server** (Rust + wgpu):
- Accepts WebSocket connections from compute nodes
- Distributes work units (energy evaluation, gradient computation, repair steps)
- Aggregates results
- Handles node discovery, heartbeat, and fault tolerance

**Browser Worker** (JavaScript/WASM + WebGPU):
- Loads Carnot energy functions as WGSL shaders
- Receives work units via WebSocket
- Executes on local GPU via WebGPU API
- Returns results

**Work Unit Types**:
1. **Batch energy evaluation**: Given N configurations, compute E(x_i) for each
2. **Gradient computation**: Given x, compute ∇E(x)
3. **Multi-start repair**: Each browser runs repair from a different starting point, gateway selects best
4. **Distributed training**: Parallel NCE gradient computation across browsers

### Why WebGPU?

- **Vendor-agnostic**: Works on Chrome/Firefox/Edge with any GPU (NVIDIA, AMD, Intel, Apple Silicon)
- **Zero install**: Users contribute compute by visiting a webpage
- **Safety**: WebGPU runs in browser sandbox — no filesystem/network access from compute
- **Scale**: Many browsers = many GPUs. Each contributes a small amount, aggregate is large.

### Implementation Phases

**Phase 1: WGSL shader compilation**
- Translate Carnot energy functions (SAT, coloring, Ising, Gibbs) to WGSL compute shaders
- Test locally via wgpu-native

**Phase 2: Gateway server**
- Rust server accepting WebSocket connections
- Work distribution protocol (protobuf or msgpack)
- Result aggregation

**Phase 3: Browser worker**
- JavaScript client connecting to gateway
- WebGPU shader execution
- Heartbeat and graceful disconnect

**Phase 4: Integration**
- Carnot Python client that submits work to the gateway
- Transparent fallback: if gateway unavailable, run on local CPU/GPU
- Multi-start repair across N browser workers simultaneously

### Security Considerations

- Compute nodes see energy function parameters but not training data
- Results are verified server-side (can't trust arbitrary browser output)
- Rate limiting and authentication for the gateway
- WGSL shaders are compiled by the browser, not executed as arbitrary code

### Estimated Effort

| Phase | Effort | Deliverable |
|-------|--------|-------------|
| Phase 1 | 2-3 sessions | WGSL shaders for core energy functions |
| Phase 2 | 3-4 sessions | Gateway server with WebSocket protocol |
| Phase 3 | 2-3 sessions | Browser worker with WebGPU execution |
| Phase 4 | 2-3 sessions | Python client integration |

Total: ~10-13 sessions for the complete distributed compute stack.

---

## Recommendation

Start with **Option C (Rust + wgpu)** for the local GPU backend — it's the natural fit for Carnot's dual-language architecture and works with the AMD GPU immediately. Then build the WebGPU gateway on top of the same wgpu crate, since wgpu implements both native Vulkan AND WebGPU standards.

The progression:
1. Add wgpu GPU kernels to Rust crates (local AMD GPU acceleration)
2. Expose via PyO3 (JAX-like API but GPU-accelerated on any vendor)
3. Build WebGPU gateway for distributed browser compute
4. Autoresearch runs on distributed GPU fleet
