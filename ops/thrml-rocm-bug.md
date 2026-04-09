# thrml + ROCm PJRT Plugin Crash

**Filed against:** https://github.com/extropic-ai/thrml
**Date:** 2026-04-08
**Severity:** Blocking for GPU usage

## Summary

thrml's block Gibbs sampling crashes with a HIP runtime assertion failure when running on the JAX ROCm GPU backend via `jax-rocm7-pjrt`. The crash occurs during execution of thrml's nested `jax.lax.scan` with complex body functions involving multiple kernel dispatches per iteration.

## Environment

- OS: CachyOS (Arch Linux), kernel 6.19.11-1-cachyos
- GPU: AMD Radeon 890M (gfx1150, integrated)
- ROCm: 7.2.1
- Python: 3.12 (JAX 0.9.2 does not officially support Python 3.14)
- JAX: 0.9.2
- jaxlib: 0.9.2
- jax-rocm7-pjrt: 0.9.2b1
- jax-rocm7-plugin: 0.9.2b1
- thrml: 0.1.3
- equinox: (dependency of thrml)

**Note on gfx1150:** This GPU requires `HSA_OVERRIDE_GFX_VERSION=11.0.0` to load XLA kernels at all, since the JAX ROCm plugin does not ship with native gfx1150 targets. Without this override, even simple operations like `jnp.sum` fail during XLA compilation. All testing below was done with this override set.

## Reproduction

The confirmed crash path is through thrml's `SpinEBMFactor` sampling (e.g., `test_discrete_ebm.py::TestSampling::test_binary`). The following Ising API script is a plausible reproduction but has not been independently verified:

```python
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import jax
import jax.numpy as jnp
import jax.random as jrandom
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init

# Verify GPU is active
assert jax.default_backend() == "gpu", f"Expected GPU, got {jax.default_backend()}"

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

Two different crashes observed at different points in execution:

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

1. **Simple JAX ops work fine on the same GPU** (with `HSA_OVERRIDE_GFX_VERSION=11.0.0`). `jnp.dot`, simple `jax.lax.scan`, `jax.random.bernoulli`, matrix multiply — all work correctly on `RocmDevice(id=0)`.

2. **The crash is in the ROCm runtime's AQL packet dispatch**, triggered during execution of thrml's nested `jax.lax.scan` with a complex body function that dispatches multiple kernels per iteration (the `_run_blocks` → `sample_blocks` → `sample_single_block` call chain). Simple `jax.lax.scan` works; thrml's nested scan with many kernel dispatches per iteration triggers the bug. Whether equinox's tracing contributes to the computation graph complexity is a reasonable hypothesis but unproven.

3. **The crash happens when JAX's default backend is GPU.** We did not separately test whether explicitly placing thrml arrays on CPU while the ROCm plugin is loaded would avoid the crash. The `JAX_PLATFORMS=cpu` workaround (which disables GPU entirely) does avoid it.

4. **Workaround:** Use `JAX_PLATFORMS=cpu` to disable GPU entirely, or use a custom parallel sampler that bypasses thrml's block sampling infrastructure (which is what we did — see `carnot.samplers.parallel_ising`).

## Impact

This blocks thrml from running on AMD ROCm GPUs, which is relevant for:
- Pre-TSU development on AMD hardware
- GPU-accelerated Ising sampling for large problem instances
- Testing hardware compatibility before TSU availability

## Suggested investigation

The crash is in the HIP runtime's kernel dispatch path during execution of a complex nested scan. Testing on a different GPU architecture (gfx90a, gfx942) that doesn't require `HSA_OVERRIDE_GFX_VERSION` would help isolate whether this is gfx1150-specific or a general ROCm issue with thrml's computation graph structure.
