import os

spec_path = "openspec/capabilities/benchmarks/spec.md"
with open(spec_path, "a") as f:
    f.write("""

### REQ-HARDWARE-1991: ROCm eGPU Detection and JAX Initialization

Carnot MUST provide a hardware verification script to detect the presence of the RX 7900 XTX eGPU via ROCm and verify its initialization in JAX over Thunderbolt.
The implementation MUST use `rocminfo` and `jax.devices()`.
Results MUST be written to `results/experiment_1991_egpu_rocm.json`.
It MUST record the hardware state and an honest verdict.

**Acceptance criteria:**
- Script `scripts/experiment_1991_egpu_rocm.py` exists.
- Records detected hardware state.
- Artifact is written to `results/experiment_1991_egpu_rocm.json`.

### SCENARIO-HARDWARE-1991: eGPU ROCm Detection Execution

**Given** the hardware verification script
**When** the benchmark script runs
**Then** it performs the hardware detection check using `rocminfo` and JAX
**And** produces a valid `results/experiment_1991_egpu_rocm.json` artifact containing the schema, experiment ID, and an honest verdict.

**Spec traces:** REQ-HARDWARE-1991
""")
print("Appended REQ-HARDWARE-1991 to spec")