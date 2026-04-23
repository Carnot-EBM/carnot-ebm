# FPGA Capability Specification

**Capability:** FPGA Hardware Acceleration for Ising Sampler
**Status:** In Progress
**Owner:** Ian Blenke

---

## Requirements

### REQ-HW-037

**Title:** KV260 Ising v3 synthesis timing closure at >= 50 MHz

**Description:**
Synthesis of `ising_sampler_v3.v` targeting the KV260 (xck26-sfvc784-2LV-c) FPGA
must achieve timing closure at a clock frequency of at least 50 MHz (Worst Negative
Slack >= 0 ns).  This ensures the Ising sampler can sustain real-time operation
alongside the KV260 ARM Cortex-A53 at typical embedded inference workloads.

**Why 50 MHz:** The KV260 PL fabric runs at up to 300 MHz; 50 MHz is a conservative
floor that leaves ample margin for routing, temperature variation, and future
pipeline elaboration.  Any result below 50 MHz would indicate logic structure that
prevents timing closure even after basic constraint tuning.

**Rationale:** The EMA inertia dynamics in v3 add feedback registers.  Timing must
be verified to confirm those registers do not lengthen the critical path past the
target.

**Acceptance criteria:**
- Vivado synthesis produces WNS >= 0 ns at 50 MHz, OR
- Yosys proxy synthesis shows cell counts consistent with <= 50 MHz achievability
  (< 20% LUT utilization on KV260 fabric).

**Implementation status:** Pending (Exp 701)

---

### REQ-HW-038

**Title:** LUT utilization < 20% of KV260 fabric for Ising v3 N=64

**Description:**
Synthesis of `ising_sampler_v3.v` with N=64 spins must consume fewer than 20% of
the KV260 PL fabric's LUT resources (xck26 has ~117,120 LUTs).  This leaves room
for the control/DMA interface, AXI interconnect, and future capability expansion
without requiring a larger FPGA.

**Why 20%:** Industry practice reserves 50–80% of fabric for routing congestion and
incremental feature growth.  20% on the Ising sampler alone is generous enough for
N=64 and leaves capacity for a second sampler instance or the Gibbs/Boltzmann tiers.

**Acceptance criteria:**
- Vivado utilization report shows LUT count < 23,424 (20% of 117,120), OR
- Yosys `stat` output shows cells consistent with this bound (proxy estimate).

**Implementation status:** Pending (Exp 701)

---

## Scenarios

### SCENARIO-HW-037 — Vivado synthesis achieves timing closure

**Given:** `rtl/ising_sampler_v3.v` and Vivado 2024.2 available on the host.
**When:** `tcl/synth_ising_v3.tcl` is run with `-part xck26-sfvc784-2LV-c`.
**Then:** The timing summary reports WNS >= 0 ns, confirming REQ-HW-037.

---

### SCENARIO-HW-038 — Yosys proxy synthesis shows LUT bound

**Given:** Vivado is unavailable but yosys is present.
**When:** `yosys_script.ys` is run on `ising_sampler_v3.v` with `synth -top`.
**Then:** The `stat` output cell count is parsed and confirmed < 23,424 LUT
equivalent, validating REQ-HW-038 as a proxy estimate.

---

### SCENARIO-HW-039 — No synthesis tool available

**Given:** Neither Vivado nor yosys is installed.
**When:** Experiment 701 runs.
**Then:** `honest_verdict` = `"synthesis_blocked_no_tool"` and `ops/known-issues.md`
is updated with the RETRO-072 note.

---

---

### REQ-HW-039

**Title:** XDNA NPU GEMM kernel via IRON toolchain achieves >2x CPU speedup

**Description:**
Using the IRON toolchain (mlir-aie, pip-installable, no system ninja/openblas
dependency), a minimal 16x16 GEMM kernel must be compiled and executed on the
AMD XDNA NPU (Ryzen AI APU).  When benchmarked over 100 iterations, the NPU
GEMM latency must be less than 50% of the equivalent CPU GEMM latency — i.e.
gemm_speedup = cpu_time / npu_time >= 2.0.

**Why IRON instead of VitisAI:**
VitisAI requires ninja + openblas system packages.  These have been missing on
every experiment host across 7 consecutive milestones (Exps 292, 303, 314, 335,
435, and two others).  IRON (mlir-aie) is a Python-only install that targets
the AMD AI Engine (AIE) array directly via MLIR dialects, eliminating the
system-dependency bottleneck.  arXiv 2504.03083 demonstrates 2.8x GEMM speedup
over CPU on the same XDNA NPU hardware using the IRON approach.

**Acceptance criteria:**
- `import mlir_aie` succeeds after `pip install mlir-aie`, AND
- GEMM kernel compiles via `mlir_aie.compile()` without error, AND
- gemm_speedup >= 2.0 over 100 CPU-vs-NPU timing iterations.

**Fallback criteria (honest_verdict escalation):**
- If IRON install or import fails: `honest_verdict = "npu_iron_install_failed"`.
- If IRON works but speedup < 2.0: `honest_verdict = "npu_iron_installed_no_speedup"`.
- If neither IRON nor VitisAI works: `honest_verdict = "npu_still_blocked"`.

**Implementation status:** Pending (Exp 714)

---

### SCENARIO-HW-039 — IRON GEMM kernel achieves 2x speedup on XDNA NPU

**Given:** An AMD XDNA NPU is present and `pip install mlir-aie` succeeds.
**When:** Experiment 714 runs the 16x16 GEMM benchmark over 100 iterations.
**Then:** gemm_speedup >= 2.0 and honest_verdict = "npu_iron_working".

---

### REQ-HW-040

**Title:** HLS Ising kernel energy computation MUST use Hamiltonian E = -sum(J_ij * s_i * s_j), negative sign required

**Description:**
The HLS C++ kernel function `compute_ising_energy` and any CPU/FPGA validation harness
MUST implement the standard Ising Hamiltonian with a negative sign:

    E = -sum_{i<j} J_ij * s_i * s_j - sum_i h_i * s_i

The negative sign is critical: for a ferromagnetic system (J_ij > 0) with all spins
aligned (+1), the ground state energy MUST be negative. A positive energy for the
all-aligned ferromagnetic ground state is a sign-convention bug.

CPU validation MUST confirm:
1. For a fully connected N=4 ferromagnetic system (J[i][j]=1.0 for all i != j),
   all spins = +1, the energy equals -N*(N-1)/2 = -6.0.
2. `validate_ground_state()` returns True (energy of ground state is negative).
3. Any deviation > 10% from the expected energy fails the check.

This requirement was triggered by RETRO-HLS-ENERGY (Exp 750 CPU validation FAILED:
got +3.0 instead of -3.0 for the ferromagnetic ground state test case).

**Acceptance criteria:**
- `compute_ising_energy` uses `energy -= J[i][j] * s[i] * s[j]` (negative accumulation), AND
- Static energy test for N=4 ferromagnetic all-ones gives |E - (-6.0)| < 0.6, AND
- `HLSEnergyValidator.validate_ground_state()` returns True for ferromagnetic system.

**Implementation status:** Implemented (Exp 757)

---

### SCENARIO-HW-040 — HLS energy sign validated for ferromagnetic ground state

**Given:** A fully connected N=4 ferromagnetic Ising system with J[i][j]=1.0 for all
i != j and all spins = +1 (ground state).

**When:** `HLSEnergyValidator.compute_energy(spins=[1,1,1,1])` is called.

**Then:** The returned energy equals -6.0 (within tolerance 0.6), confirming that
the negative sign in the Hamiltonian is correctly implemented and the ground state
has negative energy.

---

## Implementation Status

| REQ | Status | Experiment |
|-----|--------|-----------|
| REQ-HW-037 | Pending | Exp 701 |
| REQ-HW-038 | Pending | Exp 701 |
| REQ-HW-039 | Pending | Exp 714 |
| REQ-HW-040 | Implemented | Exp 757 |
