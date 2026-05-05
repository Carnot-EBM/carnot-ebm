# FPGA Capability Specification

**Capability:** FPGA Hardware Acceleration for Ising Sampler
**Status:** In Progress
**Owner:** Ian Blenke

---

## Requirements

### REQ-HW-035

**Title:** Sparse adjacency Ising synthesis on iCE40 within 200 LUTs

**Description:**
Synthesis of `ising_inertia_n8_sparse_v2.v` on iCE40 HX8K must use fewer than 200
SB_LUT4 cells.  The design stores coupling pairs as a sparse adjacency list (K=12
non-zero pairs for N=8 ring+chord graph) rather than a dense N×N coupling matrix.
Sparse storage reduces register and logic costs: K=12 entries × 14 bits = 168 bits
versus N²=64 entries × 8 bits = 512 bits for a dense encoding — a 3× reduction in
coupling storage.  This enables N=16 synthesis without the 12258 LC expansion seen
in the dense approach (RETRO-ICE40-N16).

**Acceptance criteria:**
- yosys `synth_ice40` reports SB_LUT4 count <= 200
- nextpnr-ice40 completes without routing errors
- Python simulation shows sweeps_to_converge reduction >= 3× vs baseline (no inertia)

**Implementation status:** Implemented (Exp 876)

---

### SCENARIO-HW-035

**Title:** N=8 sparse ring+chord graph synthesizes within budget

**Given:** `ising_inertia_n8_sparse_v2.v` with N=8 spins, K=12 adjacency pairs, 4-bit EMA
**When:** `yosys synth_ice40 -top ising_inertia_n8_sparse_v2` is run
**Then:** The synthesis report shows SB_LUT4 count <= 200 and nextpnr reports 0 errors.

**Implementation status:** Implemented (Exp 876)

---

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

---

### REQ-FPGA-030

**Title:** FpgaBackend energy oracle MUST be pure combinational (no sequential spin state)

**Description:**
The iCE40 HX8K energy oracle must be implemented as a pure combinational circuit.
Spin configuration is an external input; energy is a combinational output.  No clock
pin, no flip-flops, and no Gibbs sweep inside the FPGA are permitted.

**Why combinational only:**
    Exp 851 demonstrated that adding even minimal sequential logic (a clocked spin_state
    register) to N=16 Verilog caused nextpnr-ice40 to infer 2077 DFFs + 9918 LUT4 cells
    = 12258 LCs total — 159% of the iCE40 HX8K 7680-LC budget.  The fundamental reason
    is that a Gibbs sweep requires N^2 MAC operations per clock; at N=16 this saturates
    the device.  Moving the Gibbs sampling to Python and using the FPGA only for
    combinational energy evaluation keeps LUT count within budget.

**Target:** N=8 spins, iCE40 HX8K (7680 LUTs), OSS-CAD-Suite tool flow.

**Acceptance criteria:**
- yosys synth_ice40 produces zero SB_DFF instances, AND
- nextpnr-ice40 reports ICESTORM_LC count < 500 (well below 7680 budget), AND
- icepack generates a valid .bin bitstream file.

**Implementation status:** Implemented (Exp 859, lut_count=134, bitstream_generated=True)

---

### SCENARIO-FPGA-040 — iCE40 N=8 pure combinational synthesis

**Given:** ising_energy_n8_comb.v is available and OSS-CAD-Suite is installed.
**When:** `yosys synth_ice40` + `nextpnr-ice40 --hx8k` + `icepack` are run.
**Then:**
- SB_DFF count = 0 in yosys output (no flip-flops inferred),
- ICESTORM_LC count < 500 in nextpnr utilisation report,
- .bin bitstream file is generated without error,
- honest_verdict = "fpga_oracle_ready".

**Result (Exp 859):** SB_LUT4=132, ICESTORM_LC=134 (1%), bitstream=135100 bytes.

---

### REQ-HW-036

**Synchronous PIMI sampler with truly parallel spin updates on iCE40.**

All spins must update simultaneously in a single clock cycle using `h_ema` from
the **previous** cycle.  No spin may see an updated neighbor within the same
cycle (which is the checkerboard mistake from Exp 860/876).

**Acceptance criteria:**
- `sweeps_reduction >= 5.0` vs no-inertia baseline (15-25x per arXiv 2604.17109
  requires harder/larger problems; 5x is the minimum for this N=8 graph)
- `lut_count <= 250` on iCE40 HX8K ct256
- All spins read `s_current[j]` from the SAME snapshot for `h_local` computation

### SCENARIO-HW-036

**Scenario:** Synchronous PIMI convergence benchmark.

**Given:** SynchronousPIMISampler on N=8 ring+chord ferromagnetic graph.
**When:** 100 independent trials from random init, target_energy=-3.0.
**Then:**
- `parallel_sweeps < checkerboard_sweeps` (synchronous is no worse than checkerboard)
- `sweeps_reduction = baseline_sweeps / parallel_sweeps` is reported honestly
- `honest_verdict` is one of the defined outcome strings
- Verilog synthesis: `lut_count <= 250`, `synthesis_clean = true`

**Result (Exp 889):** parallel_sweeps=3, checkerboard_sweeps=13, sweeps_reduction=4.33x,
lut_count=126, synthesis_clean=True, honest_verdict="pimi_improved_below_5x".

---

### REQ-HW-041

**Title:** SparsePIMISampler copy-node sparsification reduces convergence sweeps vs dense PIMI

**Description:**
SparsePIMISampler (python/carnot/samplers/sparse_pimi.py) implements PIMI sampling with
copy-node sparsification from arXiv 2503.01177: for each spin i, only the top-k couplings
by |J[i,j]| magnitude are retained; all others are zeroed out.  This converts O(N^2) dense
edges to O(N*k) sparse edges.

Hypothesis tested in Exp 901 (FINAL PIMI attempt): dense coupling creates spurious local
minima independent of update strategy.  If true, sparse adjacency should beat synchronous
PIMI (Exp 889) by further reducing convergence sweeps.

**Finding (Exp 901):** For the N=8 ring+chord ferromagnetic graph, every spin already has
degree exactly 3.  With k=3,4,5, J_sparse == J_dense (no edges removed).  SparsePIMISampler
is structurally equivalent to SynchronousPIMISampler for this graph.  Sweeps reduction =
4.33x (same as Exp 889), below the 5x target.  iCE40 PIMI retired after this final attempt.

**Acceptance criteria (Exp 901 gate):**
- k-sparsification keeps exactly top-k couplings per spin (verified by unit test)
- Synthesis on iCE40 HX8K: SB_LUT4 <= 250
- sweeps_reduction >= 5.0 for verdict "pimi_5x_retro_closed" (NOT achieved)

**Implementation status:** Implemented and retired (Exp 901)

---

### SCENARIO-HW-041

**Scenario:** SparsePIMISampler k-sparsification benchmark on N=8 ring+chord graph.

**Given:** SparsePIMISampler with J_dense = make_n8_coupling_matrix(), k ∈ {3,4,5}.
**When:** 100 independent trials, target_energy=-3.0, max_sweeps=400.
**Then:**
- J_sparse has exactly min(k, degree_i) nonzero entries per row
- sweeps_reduction is reported honestly
- honest_verdict is one of the defined outcome strings

**Result (Exp 901):** k=3,4,5 all give sparse_sweeps=3 (same as dense, since ring+chord
degree=3 <= k for all tested k).  sweeps_reduction=4.33x, lut_count=126, synthesis_clean=True,
honest_verdict="pimi_improved_below_5x".  iCE40 PIMI retired to exclusion_manifest.yaml.

---

### REQ-HW-042

**Title:** KV260 v5 DC-continuous Ising diagnostic MUST compare against v4 Gibbs KL

**Description:**
Experiment 1149 MUST provide a Python software diagnostic for a candidate KV260 v5
Ising sampler that relaxes binary spins to continuous values in `[-1, +1]`, decomposes
the dense Ising coupling matrix into positive and negative eigenspectrum parts, and
uses a clipped DC/proximal-gradient update before thresholding back to `{-1, +1}`.
The diagnostic MUST run on three deterministic J matrices, compare empirical
`KL(v5_dc || CPU_Gibbs)` against the prior v4 KL result from Exp 1134, and emit a JSON
artifact with the required v5 viability fields.

**Acceptance criteria:**
- The implementation reports `algorithm = "dc_continuous_relaxation"` and tests exactly
  three deterministic J matrices.
- The implementation computes and records `kl_v5_best`, `kl_v5_below_threshold`,
  `kl_improvement_over_v4`, `kv260_v5_diagnostic_complete`, `rtl_recommendation`, and
  an honest verdict from the approved Exp 1149 vocabulary.
- The implementation records energy-time-accuracy fields including energy at convergence,
  wall-clock seconds, and final constraint-satisfaction accuracy for each matrix.
- Unit tests reference `REQ-HW-042` or `SCENARIO-HW-042` and validate the DC split,
  thresholded output, verdict mapping, and artifact schema before the diagnostic is run.

**Implementation status:** Implemented (Exp 1149)

---

### SCENARIO-HW-042

**Scenario:** DC-continuous diagnostic writes a complete v5 artifact.

**Given:** Exp 1134 reported `kl_v4_best=0.1128` and catastrophic self-adaptive lambda
KL, and Exp 1122 provides the v4 Python-simulation structure.
**When:** `scripts/experiment_1149_kv260_v5_dc_continuous_diagnostic.py` runs.
**Then:** `results/experiment_1149_kv260_v5_dc_continuous_diagnostic.json` is written
with the required schema fields, three J-matrix measurements, energy-time-accuracy
metrics, and an honest verdict documenting whether v5 is threshold-viable or only an
improvement over v4.

**Implementation status:** Implemented (Exp 1149)

---

### REQ-HW-045

**Title:** KV260 v6 sequential Gibbs sampler MUST preserve detailed balance by single-site updates

**Description:**
Experiment 1161 MUST provide a Python reference for the KV260 v6 Ising sampler
that updates exactly one spin per Gibbs step in strict round-robin order
`i_t = t mod N`.  For the selected spin, the sampler MUST hold all other spins
fixed, compute `h_i = sum_j J_ij * s_j + b_i`, and draw `s_i = +1` with
probability `sigmoid(2 * beta * h_i)`.

The diagnostic MUST reuse the three deterministic Exp 1149 N=8, K=2 signed-ring
J matrices generated from seeds 1134, 1135, and 1136, compare the v6 sampler
against a CPU sequential-Gibbs reference for 10,000 steps per matrix, and report
whether mean KL is below the 0.05 threshold.  It MUST also exercise the KV260 v4
target topology at N=128, K=16 using sparse ring connectivity and report the
same v6-vs-CPU KL gate for that topology without enumerating `2**128` states.

**Acceptance criteria:**
- `SequentialGibbsSampler.sample(J, b, n_spins, n_steps, beta, seed)` returns
  `n_steps` strict single-site Gibbs samples with update index `t mod n_spins`.
- The Exp 1161 artifact reports `algorithm = "sequential_gibbs"`, the three N=8
  KL measurements, the N=128 K=16 KL measurement, v4/v5 improvement fields, the
  required threshold booleans, and an honest verdict from the approved Exp 1161
  vocabulary.
- `hardware/kv260/ising_sampler_v6_spec.md` documents one-spin-per-clock RTL
  pseudocode with `s[N]` state registers, `h[N]` field-cache registers, and the
  per-cycle update semantics.
- Unit tests reference `REQ-HW-045` or `SCENARIO-HW-045` and validate the
  single-site update order, J-matrix reuse, artifact schema, N=128 sparse graph
  support, and RTL pseudocode deliverable.

**Implementation status:** Implemented (Exp 1161)

---

### SCENARIO-HW-045

**Scenario:** Sequential Gibbs v6 writes a KL-correct KV260 artifact and RTL spec.

**Given:** Exp 1149 reported `kl_v5_best ~= 0.447` and recommended sequential
Gibbs for KL-correct RTL, and Exp 1134 reported `kl_v4_best ~= 0.1128`.
**When:** `scripts/experiment_1161_kv260_v6_sequential_gibbs.py` runs.
**Then:** `results/experiment_1161_kv260_v6_sequential_gibbs.json` is written
with the required schema fields, `kl_v6_below_threshold_n8 = true`, the N=128
K=16 topology result, positive KL improvements over v4 and v5, and
`rtl_spec_written = true` pointing to `hardware/kv260/ising_sampler_v6_spec.md`.

**Implementation status:** Implemented (Exp 1161)

---

### REQ-HW-046

**Title:** p-bit sampler portability packet MUST be CPU-only and hardware-claim gated

**Description:**
Experiment 1320 MUST produce a portability packet for future FPGA or accelerator
p-bit sampler work without claiming hardware execution.  The packet MUST define a
tiny Ising case, compute a CPU sequential-Gibbs baseline, simulate a p-bit update
loop with configurable p-bit reuse factor and DAC-bit quantization, and compare
the p-bit empirical distribution against the CPU Gibbs distribution using KL
divergence.  It MUST also include a dual-BRAM mapping sketch suitable for later
RTL work and record whether Vivado, KV260 bitfiles, or equivalent FPGA synthesis
tools were unavailable on the local host.

**Acceptance criteria:**
- `results/experiment_1320_pbit_sampler_portability_packet.json` includes
  `status`, `dual_bram_mapping_ready`, `reuse_factor_sweep`, `dac_bits_sweep`,
  `kl_to_cpu_gibbs`, `vivado_required_for_next_step`, `hardware_claim_allowed`,
  and `honest_verdict`.
- The reuse-factor sweep covers at least three reuse factors and records KL to
  the CPU Gibbs baseline for each factor.
- The DAC-bit sweep covers at least three bit widths and records KL to the CPU
  Gibbs baseline for each width.
- `dual_bram_mapping_ready=true` only when the mapping sketch identifies two
  BRAM banks, a spin-serial/read snapshot path, and a write/update path.
- `hardware_claim_allowed=false` and `vivado_required_for_next_step=true` unless
  real synthesis or board execution is performed in the current run.

**Implementation status:** Implemented (Exp 1320)

---

### SCENARIO-HW-046

**Scenario:** CPU p-bit portability packet writes honest sweep artifact.

**Given:** Vivado/KV260 synthesis is unavailable or not executed locally.
**When:** the Exp 1320 packet builder runs on the tiny Ising case.
**Then:** the artifact records the dual-BRAM mapping sketch, reuse-factor sweep,
DAC-bit sweep, KL divergence to the CPU Gibbs baseline, and an honest verdict
that disallows hardware claims until a Vivado or board run is performed.

**Implementation status:** Implemented (Exp 1320)

---

### REQ-HW-047

**Title:** p-bit update-dynamics and dual-BRAM packet MUST gate KV260/hardware claims

**Description:**
Experiment 1348 MUST extend the CPU-only p-bit portability packet with explicit
update-dynamics, memory-reuse, DAC precision, and finite-delay assumptions before
any KV260 or ASIC-style RTL milestone can claim hardware portability.  The packet
MUST translate the tiny Carnot Ising case and a tiny KAN spline/LUT case into
hardware-facing assumptions, using local CPU evidence only unless an actual
synthesis or board run is performed in the current run.

**Acceptance criteria:**
- `results/experiment_1348_pbit_update_dynamics_dual_bram_packet_v2.json`
  includes `status`, `sync_async_regime`, `reuse_factor_grid`, `bram_layout`,
  `dac_precision_assumption`, `finite_delay_assumption`,
  `kv260_claim_allowed`, `hardware_claim_allowed`, `next_rtl_requirements`, and
  `honest_verdict`.
- `sync_async_regime` covers synchronous snapshot updates, asynchronous
  single-site Gibbs-like updates, and at least one delayed or phase-serialized
  regime.
- `reuse_factor_grid` covers at least three reuse factors and records logical
  spins, physical p-bits, parallel update width, BRAM phase semantics, and CPU
  KL evidence inherited or recomputed from the tiny Ising case.
- `bram_layout` identifies two BRAM banks, snapshot-read and delayed-write
  roles, Ising coupling storage, KAN spline/LUT storage, field-cache storage,
  RNG threshold storage, and bank-swap semantics.
- DAC and finite-delay assumptions are explicit enough for RTL design review:
  bit widths, clipping, quantization rule, latency model, and acceptance gates
  must be recorded without claiming analog or board validation.
- `kv260_claim_allowed=false` and `hardware_claim_allowed=false` unless the
  packet records actual local synthesis or board execution in metadata.
- `next_rtl_requirements` names concrete future files and interfaces needed for
  a later hardware milestone.

**Implementation status:** Implemented (Exp 1348)

---

### SCENARIO-HW-047

**Scenario:** CPU-only p-bit update-dynamics packet writes honest RTL handoff.

**Given:** no Vivado synthesis, bitfile generation, or KV260 board execution is
performed in the current run.
**When:** the Exp 1348 packet builder consumes the tiny Ising/KAN CPU cases and
the Exp 1320 p-bit portability evidence.
**Then:** the artifact records sync/async regimes, reuse-factor grid,
dual-BRAM layout, DAC and finite-delay assumptions, concrete future RTL
requirements, and an honest verdict that disallows KV260 and hardware claims.

**Implementation status:** Implemented (Exp 1348)

---

## Implementation Status

| REQ | Status | Experiment |
|-----|--------|-----------|
| REQ-HW-036 | Implemented | Exp 889 |
| REQ-HW-037 | Pending | Exp 701 |
| REQ-HW-038 | Pending | Exp 701 |
| REQ-HW-039 | Pending | Exp 714 |
| REQ-HW-040 | Implemented | Exp 757 |
| REQ-HW-041 | Implemented (retired) | Exp 901 |
| REQ-HW-042 | Implemented | Exp 1149 |
| REQ-HW-045 | Implemented | Exp 1161 |
| REQ-HW-046 | Implemented | Exp 1320 |
| REQ-HW-047 | Implemented | Exp 1348 |
| REQ-FPGA-030 | Implemented | Exp 859 |
