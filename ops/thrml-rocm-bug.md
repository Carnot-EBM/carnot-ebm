# thrml + ROCm PJRT Plugin Crash

**Filed against:** https://github.com/extropic-ai/thrml
**Date:** 2026-04-08
**Severity:** Blocking for GPU usage

## Summary

thrml's block Gibbs sampling crashes with a HIP runtime assertion failure when the JAX ROCm PJRT plugin (`jax-rocm7-pjrt`) is installed and loaded, even when thrml code targets CPU.

## Environment

- OS: CachyOS (Arch Linux), kernel 6.19.11-1-cachyos
- GPU: AMD Radeon 890M (gfx1150, integrated)
- ROCm: 7.2.1
- Python: 3.14
- JAX: 0.9.2
- jaxlib: 0.9.2
- jax-rocm7-pjrt: 0.9.1.post3
- jax-rocm7-plugin: 0.9.1.post3
- thrml: (latest pip version)
- equinox: (dependency of thrml)

## Reproduction

```python
import jax
import jax.numpy as jnp
import jax.random as jrandom
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init

# Simple 10-spin Ising model
n = 10
nodes = [SpinNode() for _ in range(n)]
edges = [(nodes[i], nodes[j]) for i in range(n) for j in range(i+1, n)]
model = IsingEBM(
    nodes=nodes, edges=edges,
    biases=jnp.zeros(n), weights=jnp.zeros(n*(n-1)//2),
    beta=jnp.array(1.0),
)
free_blocks = [Block([nodes[i]]) for i in range(n)]
program = IsingSamplingProgram(model, free_blocks, [])
init_state = hinton_init(jrandom.PRNGKey(0), model, free_blocks, ())
schedule = SamplingSchedule(100, 10, 5)

# This crashes:
samples = sample_states(jrandom.PRNGKey(1), program, schedule, init_state, [], free_blocks)
```

## Error

Two different crashes observed:

### Crash 1: HIP code object assertion
```
python: /usr/src/debug/hip-runtime/rocm-systems-rocm-7.2.1/projects/clr/hipamd/src/hip_code_object.cpp:400:
hipError_t hip::StatCO::getStatFunc(ihipModuleSymbol_t**, const void*, int):
Assertion `err == hipSuccess' failed.
Aborted (core dumped)
```

### Crash 2: AQL packet queue assertion
```
/usr/include/c++/15.2.1/bits/stl_vector.h:1263:
std::vector<rocr::core::AqlPacket>::operator[](size_type):
Assertion '__n < this->size()' failed.
Aborted (core dumped)
```

## Key observations

1. **Plain JAX ops work fine on the same GPU.** `jnp.dot`, `jax.lax.scan`, `jax.random.bernoulli`, matrix multiply — all work correctly on `RocmDevice(id=0)`.

2. **The crash is triggered by thrml's equinox-based tracing**, specifically `BlockSamplingProgram` construction and the `jax.lax.scan` in `_run_blocks`. Something in equinox's tree manipulation or the complex nested `jax.lax.scan` with Python control flow triggers a bad code path in the HIP runtime.

3. **The crash happens regardless of execution order** — even if thrml runs before any explicit GPU ops, just having the ROCm PJRT plugin loaded is sufficient to cause the crash.

4. **Workaround:** Use `JAX_PLATFORMS=cpu` to disable GPU entirely, or use a custom parallel sampler that bypasses thrml's block sampling infrastructure (which is what we did).

## Impact

This blocks thrml from running on AMD ROCm GPUs, which is relevant for:
- Pre-TSU development on AMD hardware
- GPU-accelerated Ising sampling for large problem instances
- Testing hardware compatibility before TSU availability

## Suggested investigation

The crash appears to be in the HIP runtime's kernel dispatch path, suggesting that equinox's JAX tracing generates a computation graph that triggers a HIP compiler/dispatch bug on gfx1150. Testing on a different GPU architecture (gfx90a, gfx942) would help isolate whether this is gfx1150-specific.
