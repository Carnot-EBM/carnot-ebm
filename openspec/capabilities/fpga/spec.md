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

### REQ-HW-048

**Title:** p-dit certificate-state mapping study MUST be CPU-only and hardware-claim gated

**Description:**
Experiment 1361 MUST produce a CPU-only mapping study for Carnot certificate
states using the p-dit/p-int literature as a conceptual mapping target.  The
study MUST define the SAT, UNSAT, UNKNOWN, and repair certificate-state
alphabet, compare binary one-hot p-bit expansion against one q=4 p-dit/p-int
variable, and run a tiny mapping-consistency or energy-equivalence proxy without
running Vivado, FPGA, KV260, TSU, analog, or external hardware.

**Acceptance criteria:**
- `results/experiment_1361_pdit_certificate_state_hardware_mapping.json`
  includes `status`, `certificate_states_mapped`, `binary_spin_count`,
  `pdit_variable_count`, `state_expansion_ratio`,
  `energy_equivalence_error`, `pbit_packet_delta`,
  `hardware_claim_allowed`, `kv260_claim_allowed`,
  `next_hardware_requirements`, and `honest_verdict`.
- `certificate_states_mapped` covers SAT, UNSAT, UNKNOWN, and repair states.
- The binary mapping uses one-hot p-bit spins while the p-dit/p-int mapping
  uses one multi-valued q=4 variable, so the artifact records both variable
  counts and their expansion ratio.
- The energy-equivalence proxy reports zero error for valid one-hot states when
  compared with the p-dit/p-int energy table.
- `pbit_packet_delta` explicitly records the delta from Exp 1348's CPU-only
  p-bit update-dynamics packet.
- `hardware_claim_allowed=false` and `kv260_claim_allowed=false` unless the
  artifact records actual local synthesis or board execution in metadata.

**Implementation status:** Implemented (Exp 1361)

---

### SCENARIO-HW-048

**Scenario:** CPU-only p-dit mapping artifact writes honest certificate-state packet.

**Given:** no Vivado synthesis, bitfile generation, KV260 board execution, TSU,
analog, or external hardware execution is performed in the current run.
**When:** the Exp 1361 p-dit certificate-state mapping builder runs.
**Then:** the artifact maps SAT, UNSAT, UNKNOWN, and repair states to binary
p-bit one-hot and q=4 p-dit/p-int representations, records the state-expansion
ratio and zero valid-state energy-equivalence error, lists next hardware
requirements, and disallows KV260 and hardware claims.

**Implementation status:** Implemented (Exp 1361)

---

### REQ-HW-049

**Title:** Hardware portfolio narrowing MUST cap active tracks and document deferred reopen gates

**Description:**
Experiment 1460 MUST narrow Carnot's active hardware portfolio to two or three
tracks based on immediate research value and current local readiness. The
decision MUST distinguish active work from deferred or out-of-scope tracks in
`_bmad/architecture.md`, `research-hardware-wishlist.md`, and
`docs/research-notes/hardware_portfolio_narrowing.md`, and the experiment
artifact MUST avoid claiming KV260 board execution, Extropic hardware access,
NPU acceleration, photonic execution, or live model inference unless those
commands actually ran in the current evidence chain.

**Acceptance criteria:**
- `results/experiment_1460_hardware_portfolio_narrowing.json` includes
  `status`, `active_hardware_tracks`, `active_hardware_track_count`,
  `deferred_hardware_tracks`, `architecture_updated`,
  `hardware_wishlist_updated`, `decision_note_path`, and `honest_verdict`.
- `active_hardware_track_count` is between two and three inclusive and equals
  the number of active track records.
- Each active track records its scope, evidence, immediate research value,
  readiness, and explicit claim boundary.
- Each deferred track records a concrete reopen condition.
- The architecture, wishlist, and decision note all show the active/deferred
  split and preserve the no-hardware-claim boundary for KV260, Extropic/THRML,
  NPU, and photonic paths.

**Implementation status:** Implemented (Exp 1460)

---

### SCENARIO-HW-049

**Scenario:** Exp 1460 writes an honest hardware portfolio decision artifact.

**Given:** local evidence from Exp 1442, Exp 1451, the hardware wishlist, and
the Extropic/THRML reference notes.
**When:** the Exp 1460 portfolio builder runs on 20260507.
**Then:** the artifact reports two or three active tracks, deferred tracks with
reopen conditions, updated architecture and wishlist flags, a decision note
path, and an honest verdict that does not claim hardware execution beyond the
local evidence.

**Implementation status:** Implemented (Exp 1460)

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
| REQ-HW-048 | Implemented | Exp 1361 |
| REQ-HW-049 | Implemented | Exp 1460 |
| REQ-FPGA-030 | Implemented | Exp 859 |
| REQ-HW-050 | Implemented | Exp 1585 |

---

### REQ-HW-050

**Title:** PolarFire SoC adaptive K-PCD prototype preflight MUST define toolchain and board boundaries

**Description:**
Experiment 1585 MUST produce a preflight artifact for the PolarFire SoC adaptive K-PCD prototype.
The artifact MUST define the toolchain availability, board availability, and simulator boundaries.
It MUST NOT overclaim board work if no board or vendor toolchain (Libero) is available.

**Acceptance criteria:**
- `results/experiment_1585_polarfire_soc_adaptive_kpcd_prototype_preflight.json` is produced.
- The artifact contains `status`, `polarfire_board_available`, `yosys_polarfire_available`, `libero_available`, `reusable_rtl_components_count`, `preflight_note_path`, `polarfire_preflight_ready`, `blocked_reason`, `no_board_execution_claim`, and `honest_verdict`.

---

### REQ-HW-051

**Title:** CIKAN FPGA implementation MUST benchmark inference latency against baselines

**Description:**
Experiment 1731 MUST provide an audit measuring the end-to-end inference latency for a batch
of 1000 CIKAN evaluations on the FPGA implementation. The audit must record the timing and
write the results to `results/experiment_1731_fpga_audit.json`.

**Acceptance criteria:**
- `scripts/experiment_1731_fpga_audit.py` successfully executes.
- `results/experiment_1731_fpga_audit.json` contains `experiment`, `batch_size`, and `latency_ms` metrics.
- Output JSON contains valid measurements for a batch size of 1000.

**Implementation status:** Pending (Exp 1731)

---

### SCENARIO-HW-051

**Scenario:** Audit measures inference latency for 1000 CIKAN evaluations.

**Given:** CIKAN FPGA implementation.
**When:** The Exp 1731 audit script runs.
**Then:** The latency for 1000 evaluations is measured and saved to `results/experiment_1731_fpga_audit.json`.

**Implementation status:** Pending (Exp 1731)

### SCENARIO-HW-050

**Scenario:** PolarFire preflight writes honest boundary artifact.

**Given:** No PolarFire SoC board is physically present and Libero toolchain is missing.
**When:** The Exp 1585 preflight runs.
**Then:** The artifact records toolchain and board boundaries, limits execution to simulator first, and writes an honest verdict.

**Implementation status:** Implemented (Exp 1585)

---

### REQ-HW-LEARN-101

**Title:** Unified HW Verification Loop with Online Updater

**Description:**
Wire the PYNQ-based CIKAN verifier into the VerificationLoop, routing violations to the Python-side online updater, then re-uploading weights to the FPGA.

**Acceptance criteria:**
- `UnifiedHWVerificationLoop` extends or wraps the hardware loop to route violations to the Python-side `OnlineUpdater` and re-upload weights to the FPGA.
- Results written to `results/experiment_1732_unified.json`.

**Implementation status:** Pending (Exp 1732)

---

### SCENARIO-HW-LEARN-101

**Scenario:** Wire PYNQ-based CIKAN verifier into the VerificationLoop.

**Given:** A PYNQ-based CIKAN verifier and OnlineUpdater.
**When:** The Exp 1732 unified script runs.
**Then:** Violations are routed to the OnlineUpdater and weights are re-uploaded to the FPGA.

**Implementation status:** Pending (Exp 1732)

---

### REQ-HW-052

**Title:** Synthesize KANELÉ RTL into a KV260 bitfile

**Description:**
Experiment 1736 MUST run Vivado (or simulate it if unavailable) to synthesize the KANELÉ RTL (`kanele_top.v` and `kanele_lut.v`) into a physical bitfile for the KV260 target. It MUST produce an artifact recording the status, utilization, and WNS.

**Acceptance criteria:**
- `results/experiment_1736_kanele_synth.json` is generated with required fields.
- Vivado batch script `synth_kanele.tcl` is executed or simulated.

**Implementation status:** Implemented (Exp 1736)

---

### SCENARIO-HW-052

**Scenario:** KANELÉ RTL is synthesized to bitfile.

**Given:** `kanele_top.v` and `synth_kanele.tcl`.
**When:** `experiment_1736_kanele_synth.py` runs.
**Then:** The bitfile (or simulated results) is generated and saved to `results/experiment_1736_kanele_synth.json`.

**Implementation status:** Implemented (Exp 1736)

---

### REQ-HW-053

**Title:** Execute KANELÉ bitfile on KV260 board

**Description:**
Experiment 1737 MUST run a latency benchmark of the KANELÉ bitfile on the KV260 using PYNQ AXI interfaces. It must load the bitfile, pass 1000 input states, and measure the hardware latency in microseconds.

**Acceptance criteria:**
- `results/experiment_1737_kanele_board.json` is generated.
- The artifact records `hardware_latency_us`, `throughput_fps`, and `batch_size`.
- The benchmark evaluates a batch size of 1000.

**Implementation status:** Implemented (Exp 1737)

---

### SCENARIO-HW-053

**Scenario:** Latency benchmark of KANELÉ bitfile.

**Given:** Synthesized KANELÉ bitfile and KV260 PYNQ environment (or a mocked fallback).
**When:** `scripts/experiment_1737_kanele_board.py` runs.
**Then:** It performs the 1000 state latency test and saves the results to `results/experiment_1737_kanele_board.json`.

**Implementation status:** Implemented (Exp 1737)

---

### REQ-HW-054

**Title:** HILED Hardware Integration Prototype for Energy Scoring and Decoding

**Description:**
Experiment 1766 MUST implement the HILED decoder prototype for offloading energy scoring and decoding to an FPGA simulator over AXI. The prototype must support asynchronous hardware polling for energy minimization steps.

**Acceptance criteria:**
- `python/carnot/inference/hiled_decoder.py` is implemented.
- The decoder implements asynchronous hardware polling for energy minimization.
- `results/experiment_1766_hiled.json` is generated.

**Implementation status:** Implemented (Exp 1766)

---

### SCENARIO-HW-054

**Scenario:** HILED decoder hardware prototype asynchronously polls FPGA simulator.

**Given:** An AXI interface to an FPGA simulator.
**When:** The HILED decoder runs energy minimization steps.
**Then:** Polling occurs asynchronously and results are written to `results/experiment_1766_hiled.json`.

**Implementation status:** Implemented (Exp 1766)

---

### REQ-HW-055

**Title:** HILED Hardware Benchmark vs Software Fallback

**Description:**
Experiment 1781 MUST benchmark the HILED hardware setup against a software fallback. The benchmark must compare inference speed (latency) and energy metrics.

**Acceptance criteria:**
- A software fallback is implemented and tested.
- `scripts/experiment_1781_hiled_benchmark.py` is implemented.
- `results/experiment_1781_hiled_benchmark.json` is generated with speed and energy comparisons.

**Implementation status:** Implemented (Exp 1781)

---

### SCENARIO-HW-055

**Scenario:** Compare HILED hardware decoder against software fallback.

**Given:** An initialized HILED decoder and a software fallback.
**When:** We run energy minimization on both setups.
**Then:** The inference speed and energy metrics are compared and written to `results/experiment_1781_hiled_benchmark.json`.

**Implementation status:** Implemented (Exp 1781)

---

### REQ-HW-056

**Title:** BBIM constraints module RTL synthesis

**Description:**
Experiment 1822 MUST synthesize the updated BBIM constraints module (`rtl/potts_machine_v2.v`) using Yosys for the KV260 target.

**Acceptance criteria:**
- `Makefile` has a `synth-constraints` target.
- Yosys synthesis succeeds and records utilization.
- `results/experiment_1822_rtl_synth.json` is generated.

**Implementation status:** Implemented (Exp 1822)

---

### SCENARIO-HW-056

**Scenario:** Yosys synthesizes the BBIM constraints module.

**Given:** `rtl/potts_machine_v2.v`
**When:** `make synth-constraints` is run.
**Then:** The synthesis finishes successfully and outputs utilization metrics.

**Implementation status:** Implemented (Exp 1822)

---

### REQ-HW-057

**Title:** p-dit hardware preflight resource accounting and preconditioning limits

**Description:**
Experiment 1989 MUST run sampler preconditioning and resource accounting over valid graphs. It MUST build a resource accounting mapping abstract Carnot nodes to p-dits, explicitly emit `hardware_execution_claim=false`, and detail preconditioning limits for Kona-style architecture comparisons.

**Acceptance criteria:**
- `results/experiment_1989_p_dit_hardware_preflight.json` is generated.
- The artifact records `hardware_execution_claim=false`.
- Resource accounting maps abstract nodes to p-dits.
- Preconditioning limits for Kona-style architecture comparisons are detailed.

**Implementation status:** Implemented (Exp 1989)

---

### SCENARIO-HW-057

**Scenario:** p-dit hardware preflight executes and writes artifact.

**Given:** A valid graph of abstract Carnot nodes.
**When:** The Exp 1989 preflight runs.
**Then:** It maps nodes to p-dits, outputs Kona-style limits, emits `hardware_execution_claim=false`, and writes `results/experiment_1989_p_dit_hardware_preflight.json`.

**Implementation status:** Implemented (Exp 1989)

---

### REQ-HW-058

**Title:** GateMate A1-EVB-2M toolchain preflight and n=16 Ising tile synthesis

**Description:**
Experiment 2105 MUST produce a hardware smoke test artifact for the Cologne Chip GateMate A1-EVB-2M board using the open-source toolchain (yosys, nextpnr-himbaechel, openFPGALoader). The artifact MUST check toolchain availability, Verilog synthesis of a minimal n=16 quadratic Ising tile, PnR LUT utilization, and bitstream generation. It MUST emit `honest_verdict="complete: gatemate_smoke_blocked_toolchain_missing"` if tools are missing, preserving the toolchain boundary.

**Acceptance criteria:**
- `results/experiment_2105_gatemate_smoke.json` is generated.
- The artifact records toolchain versions, USB enumeration, and PnR status.
- `honest_verdict` reflects whether the toolchain is available and/or bitstream generated.

**Implementation status:** Implemented (Exp 2105)

---

### SCENARIO-HW-058

**Scenario:** GateMate toolchain preflight and synthesis.

**Given:** An n=16 Ising Verilog tile and the open-source GateMate toolchain environment.
**When:** The Exp 2105 smoke test runs.
**Then:** It checks tool availability, attempts synthesis, and writes `results/experiment_2105_gatemate_smoke.json` with an honest verdict.

**Implementation status:** Implemented (Exp 2105)

---

### REQ-HW-059

**Title:** E-MVL sparse K=16 RTL arithmetic accounting for KV260 Ising v4

**Description:**
Experiment 1774 MUST perform a no-synthesis hardware accounting pass for the
E-MVL Ising sampler v4 RTL described in `hardware/kv260/ising_sampler_v4_spec.md`.
The pass computes RM (Real Multiplications), NABS (Additions/Shifts), and BOP
(Bit Operations) per sweep for both the dense baseline (N=128 full coupling) and
the sparse K=16 E-MVL architecture. It derives LUT fabric pressure using the
analytic constants in the v4 spec and verifies that the sparse estimate lies within
the 117,120-LUT XCK26 budget.

**Acceptance criteria:**
- `results/experiment_1774_kv260_emvl_rtl.json` is generated.
- `kv260_no_synthesis_claim: true` is set in the artifact.
- `estimated_lut_count` reflects the sparse K=16 v4 total LUT estimate.
- `within_budget: true` (sparse total < 117,120 LUTs).
- `honest_verdict` starts with `complete:`.

**Implementation status:** Implemented (Exp 1774)

---

### REQ-HW-060

**Title:** KV260 Ising sampler latency benchmark MUST be precondition-gated and transcript-backed

**Description:**
Experiment 2898 MUST measure the live KV260 Ising sampler through the board's
SSH-accessible UIO register window without modifying the bitstream or touching
host-side SD-card devices. The experiment MUST first verify SSH reachability,
overlay availability, and UIO device exposure; if any precondition fails it MUST
write a blocked artifact and stop before attempting register access. When the
board is available, the experiment MUST generate a deterministic n=64 Ising
problem for seeds 42, 137, and 271, load the `carnot_ising_v2_n64`/`carnot_ising_v4`
overlay, write the problem through the documented AXI-Lite memory windows, trigger
the sampler, poll the DONE bit, read spins and energy, and record per-sample
wall-clock latency for n_samples in {100, 1000, 10000}. The artifact MUST preserve
the literal board command transcript and MUST NOT claim a CPU speedup.

**Acceptance criteria:**
- `results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json`
  is generated with `inference_substrate="hardware_smoke"`.
- The artifact records `preconditions_checked`, `kv260_overlay_loaded`,
  `kv260_uio_devices_present`, `bitstream_sha256`, `random_seeds_used`, and
  `board_transcript_path`.
- Successful runs include three `per_seed_results` entries with positive
  median and p95 per-sample wall-clock microseconds and a host task
  `duration_s >= 30`.
- Blocked runs start `honest_verdict` with the specific `blocked_kv260_*`
  resource prefix and do not attempt measurement.
- No artifact field claims speedup over CPU.

**Implementation status:** Implemented (Exp 2898)

---

### SCENARIO-HW-060

**Scenario:** KV260 board latency transcript anchors first honest hardware timing.

**Given:** The KV260 responds to `ssh kria`, exposes the `carnot_ising_v4` or
`carnot_ising_v2_n64` overlay through `xmutil listapps`, and creates `/dev/uio*`
nodes after `xmutil loadapp`.
**When:** Experiment 2898 runs the board harness for seeds 42, 137, and 271.
**Then:** The artifact records successful hardware-smoke provenance, a transcript
at `results/experiment_2898_kv260_transcript.log`, bitstream SHA256 from the board,
and per-seed median and p95 per-sample wall-clock latency without a CPU-speedup
claim.

**Implementation status:** Implemented (Exp 2898)

---

### REQ-HW-061

**Title:** GateMate A1-EVB-2M n=16 Discrete SB bitstream build MUST capture LUT mapping blockers honestly

**Description:**
Experiment 2899 MUST adapt the KV260 row-serial Discrete SB RTL into a fixed
n=16 GateMate top module at `hardware/gatemate/ising_n16_gatemate.v`, then
attempt the yosys `synth_gatemate` and `nextpnr-gatemate --device CCGM1A1`
bitstream flow without flashing the board. The experiment MUST first check
`yosys`, `nextpnr-gatemate`, and DirtyJTAG USB enumeration for the on-board
GateMate A1-EVB-2M MCU (`1209:0xc0ca`). If any precondition is missing, the
artifact MUST stop with the specific blocked verdict. If synthesis or
place-and-route fails, the artifact MUST preserve the full stderr in
`lut_mapping_error_log` so the next run can diagnose the GateMate LUT mapping
root cause. If a bitstream is generated, the artifact MUST record its SHA256 and
stage flashing for operator review rather than invoking openFPGALoader.

**Acceptance criteria:**
- `hardware/gatemate/ising_n16_gatemate.v` exists and declares top module
  `ising_n16_gatemate` for exactly 16 spins.
- `results/experiment_2899_gatemate_a1_n16_ising_tile_bitstream_build_v1.json`
  is generated with `inference_substrate="hardware_smoke"` and all task-required
  schema fields.
- Missing `yosys` or `nextpnr-gatemate` yields
  `honest_verdict="blocked_gatemate_toolchain_missing"` before synthesis.
- Missing DirtyJTAG USB enumeration yields
  `honest_verdict="blocked_gatemate_usb_not_attached"` before synthesis.
- Any yosys or nextpnr failure records the full stderr in
  `lut_mapping_error_log`; successful PnR records `bitstream_sha256` and does
  not flash the bitstream.

**Implementation status:** Implemented (Exp 2899; current host blocked by missing `nextpnr-gatemate`)

---

### SCENARIO-HW-061

**Scenario:** GateMate n=16 Discrete SB build records bitstream or actionable LUT mapping failure.

**Given:** The GateMate A1-EVB-2M is attached through the on-board DirtyJTAG MCU
and the yosys plus nextpnr-gatemate tools are available.
**When:** Experiment 2899 runs the n=16 GateMate Discrete SB build flow.
**Then:** The artifact records preconditions, tool versions, synthesis and PnR
success booleans, either the bitstream path plus SHA256 or the full synthesis/PnR
stderr, and never flashes the board.

**Implementation status:** Implemented (Exp 2899)

---

### SCENARIO-HW-059

**Scenario:** E-MVL K=16 RTL accounting verifies budget compliance without synthesis.

**Given:** The v4 spec's analytic LUT constants and N=128, K=16 architecture parameters.
**When:** Exp 1774 runs `emvl_rtl_accounting.run_experiment()`.
**Then:** The artifact records `within_budget=True`, `estimated_lut_count` < 117,120,
and `honest_verdict` prefixed with `complete:`.

**Implementation status:** Implemented (Exp 1774)

---

### REQ-HW-062

**Title:** PolarFire SoC CPU constraint dispatch smoke MUST be precondition-gated and hash-verified

**Description:**
Experiment 2900 MUST run a CPU-only Carnot constraint-evaluation workload on the
Microchip PolarFire SoC Discovery Kit through SSH. The experiment MUST NOT claim
FPGA fabric access. Before dispatch it MUST verify SSH reachability to
`polarfire`, `uname -m == riscv64`, and `python3 >= 3.10`; if any precondition
fails it MUST write a blocked artifact with the specific PolarFire blocker and
MUST NOT run the workload. When preconditions pass, it MUST SCP a deterministic
50-clause SAT instance plus scorer to the board, run the scorer on the riscv64
CPU cluster, pull the transcript back, and verify the transcript's deterministic
scorer output SHA256 against the expected value computed locally.

**Acceptance criteria:**
- `results/experiment_2900_polarfire_carnot_dispatch_smoke_v1.json` is generated
  with `inference_substrate="hardware_smoke"`.
- The artifact records `preconditions_checked`, `polarfire_ssh_uptime_at_run`,
  `polarfire_kernel`, `polarfire_arch="riscv64"`, `sat_instance_sha256`,
  `scorer_output_sha256`, `scorer_output_hash_verified`,
  `per_clause_wall_clock_us`, `total_wall_clock_s`, and `duration_s`.
- Successful artifacts set `scorer_output_hash_verified=true` and
  `duration_s >= 10`.
- The workload uses only the PolarFire Linux CPU substrate for this milestone and
  records no FPGA fabric execution claim.

**Implementation status:** Pending (Exp 2900)

---

### SCENARIO-HW-062

**Scenario:** PolarFire riscv64 CPU evaluates a transcript-backed SAT scorer.

**Given:** `ssh polarfire` is reachable, reports `riscv64`, and provides
Python 3.10 or newer.
**When:** Experiment 2900 dispatches the deterministic 50-clause SAT scorer to
the board and pulls back the transcript.
**Then:** The local expected scorer output hash matches the remote transcript
hash, all 50 per-clause wall-clock timings are positive floats, the artifact
records `inference_substrate="hardware_smoke"`, and the run duration is at
least 10 seconds without claiming FPGA fabric execution.

**Implementation status:** Pending (Exp 2900)

---

### REQ-HW-063

**Title:** Operator hardware portfolio status card MUST aggregate KV260, GateMate, PolarFire, and THRML artifacts

**Description:**
Experiment 2907 MUST write one concise operator-facing portfolio artifact at
`results/experiment_2907_operator_hardware_portfolio_status_v1.json`. The
artifact MUST read the upstream hardware-port artifacts from Exp 2898 (KV260),
Exp 2899 (GateMate), Exp 2900 (PolarFire), and Exp 2901 (THRML), then summarize
each board into exactly `{state, last_artifact, next_step}`. The aggregation
MUST use `inference_substrate="aggregation_from_upstream_artifacts"` and MUST
NOT run a new board command, flash a bitstream, install a dependency, or convert
software-only THRML parity into a TSU hardware claim.

**Acceptance criteria:**
- The artifact includes `honest_verdict`, `inference_substrate`,
  `per_board_status`, `cited_upstream_artifacts`, and `duration_s`.
- `per_board_status` has exactly `kv260`, `gatemate`, `polarfire`, and `thrml`
  keys, and every board entry has exactly `state`, `last_artifact`, and
  `next_step`.
- `cited_upstream_artifacts` cites the four upstream result JSON files with
  SHA256 digests and imported field names.
- The artifact's honest verdict starts with `complete:` only when all four
  upstream artifacts are present, valid JSON objects, and summarized.

**Implementation status:** Implemented (Exp 2907)

---

### SCENARIO-HW-063

**Scenario:** Operator reads one concise hardware portfolio status card.

**Given:** Exp 2898, Exp 2899, Exp 2900, and Exp 2901 result artifacts exist in
`results/`.
**When:** Exp 2907 builds the portfolio status card.
**Then:** The artifact records a one-entry status for KV260, GateMate,
PolarFire, and THRML, cites each upstream artifact, sets
`inference_substrate="aggregation_from_upstream_artifacts"`, and records a real
non-negative aggregation duration.

**Implementation status:** Implemented (Exp 2907)

---

### REQ-HW-064

**Title:** Exp 2912 KV260 same-basis CPU Gibbs baseline MUST reconstruct the exact Exp 2898 problem basis

**Description:**
Experiment 2912 MUST read
`results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json`
and reconstruct the exact n=64 Ising basis that was uploaded to the KV260 in
Exp 2898. The run MUST extract the uploaded sparse topology, q8.8 coupling
tensor, q8.8 field tensor, coupling and field checksums, random seeds, and
sample-count sweep. If the upstream artifact is absent, the run MUST write
`honest_verdict="blocked_kv260_latency_artifact_missing"`. If any exact tensor
needed for the same-basis comparison is missing or malformed, the run MUST write
`honest_verdict="blocked_kv260_problem_basis_unrecoverable"` and identify the
missing field. When the basis is recoverable, the experiment MUST run a CPU
sequential Gibbs sampler on the uploaded sparse basis for sample counts 100,
1000, and 10000 for every Exp 2898 seed, recording median and p95 per-sample
wall-clock latency, final energy, energy trace checksum, and reproducibility
checksum. The artifact MUST set `speedup_claim_made=false` and MUST NOT claim
hardware speedup.

**Acceptance criteria:**
- `results/experiment_2912_kv260_same_basis_cpu_gibbs_baseline_v1.json` is
  generated with `inference_substrate="cpu_sampler"` and
  `same_basis_cpu_baseline_ready=true` only when all Exp 2898 tensors are
  recovered and all CPU runs complete.
- The artifact records `upstream_kv260_artifact`, `n_spins=64`,
  `matched_sparse_topology=true`, `matched_coupling_tensor=true`,
  `matched_field_tensor=true`, `random_seeds_used`, `sample_count_sweep`,
  `cpu_per_seed_results`, median and p95 latency dictionaries by sample count,
  `reproducibility_checksum`, `duration_s`, and `run_date="20260523"`.
- Missing Exp 2898 or unrecoverable tensor fields produce terminal blocked
  artifacts rather than silently synthesizing a substitute problem.
- `speedup_claim_made` remains false in every terminal artifact.

**Implementation status:** Pending (Exp 2912)

---

### SCENARIO-HW-064

**Scenario:** Same-basis CPU Gibbs baseline anchors the KV260 latency transcript.

**Given:** Exp 2898 completed with a recoverable n=64 sparse AXI upload for
seeds 42, 137, and 271.
**When:** Exp 2912 runs the CPU sequential Gibbs baseline for sample counts
100, 1000, and 10000 on those exact uploaded tensors.
**Then:** The artifact reports per-seed CPU latency and energy provenance,
marks all basis-match booleans true, records a reproducibility checksum, and
does not make a hardware speedup claim.

**Implementation status:** Pending (Exp 2912)

---

### REQ-HW-065

**Title:** Exp 2913 KV260 hardware/CPU claim boundary MUST gate speedup claims on matched evidence

**Description:**
Experiment 2913 MUST read the Exp 2898 KV260 hardware-smoke latency artifact and
the Exp 2912 same-basis CPU Gibbs baseline artifact, compare only matched
per-sample latency measurements, and write
`results/experiment_2913_kv260_hardware_cpu_claim_boundary_v1.json`. If the Exp
2912 artifact is absent or `same_basis_cpu_baseline_ready` is false, the
experiment MUST write `honest_verdict="blocked_cpu_baseline_not_ready"` and stop
without making a speedup claim. When both upstream artifacts are available, the
experiment MUST verify matching `n_spins`, sparse topology, random seeds, sample
counts, and microsecond per-sample timing units before computing any CPU/KV260
latency ratio. Numeric hardware speedup is eligible only when all match gates pass.

**Acceptance criteria:**
- The artifact includes `honest_verdict`, `kv260_claim_boundary_ready`,
  `same_basis_verified`, `hardware_speedup_claim_eligible`,
  `speedup_ratio_median_by_sample_count`, `speedup_ratio_p95_by_sample_count`,
  `comparison_notes`, `matrix_row_candidate`, `paper_claim_boundary`,
  `speedup_claim_made`, `inference_substrate`, `duration_s`, and
  `run_date="20260523"`.
- `inference_substrate` is
  `aggregation_from_upstream_artifacts`, and no board command, sampler run, or
  new timing measurement is performed.
- `hardware_speedup_claim_eligible` and `speedup_claim_made` are true only when
  problem basis, seeds, sample counts, and timing units all match.
- If any match gate fails, the artifact names the failed condition in
  `comparison_notes`, leaves all speedup-ratio dictionaries empty, and writes a
  paper boundary that forbids numeric speedup claims.

**Implementation status:** Implemented (Exp 2913)

---

### SCENARIO-HW-065

**Scenario:** Matched KV260 and CPU artifacts produce a bounded speedup claim.

**Given:** Exp 2898 has complete hardware-smoke per-sample microsecond latency
rows, and Exp 2912 has a ready same-basis CPU baseline for the same n=64 sparse
topology, seeds, and sample counts.
**When:** Exp 2913 compares the two artifacts.
**Then:** The artifact reports per-sample CPU/KV260 speedup ratios by sample
count, marks the hardware speedup claim eligible, emits a matrix-row candidate
for downstream aggregation, and writes paper text that scopes the claim to the
matched n=64 sparse Ising workload only.

**Implementation status:** Implemented (Exp 2913)

---

### REQ-HW-066

**Title:** Exp 2914 GateMate toolchain preflight MUST record exact tool paths and stop before synthesis or flash

**Description:**
Experiment 2914 MUST run a diagnostic-only preflight for the GateMate A1-EVB-2M
n=16 Ising tile. The preflight MUST check `command -v yosys`,
`command -v nextpnr-gatemate`, and `command -v openFPGALoader`, search common
OSS-CAD-Suite locations under the repository and the operator home directory,
run version commands for detected executables, confirm whether the n=16 GateMate
RTL source and explicit constraint files are present, and write
`results/experiment_2914_gatemate_toolchain_preflight_v2.json`. The preflight
MUST NOT run synthesis, place-and-route, gmpack packing, openFPGALoader board
programming, or any flash command. If `nextpnr-gatemate` is absent, the artifact
MUST emit `honest_verdict="blocked_gatemate_toolchain_missing"` with
`missing_toolchain` naming the absent executable.

**Acceptance criteria:**
- The artifact includes `honest_verdict`, `gatemate_toolchain_ready`,
  `yosys_path`, `yosys_version`, `nextpnr_gatemate_path`,
  `nextpnr_gatemate_version`, `openfpgaloader_path`,
  `openfpgaloader_version`, `missing_toolchain`, `rtl_sources_present`,
  `constraints_present`, `no_flash_attempted=true`, `inference_substrate`,
  `duration_s`, and `run_date="20260523"`.
- `gatemate_toolchain_ready` is true only when `yosys`, `nextpnr-gatemate`,
  and `openFPGALoader` are all detected.
- Missing `nextpnr-gatemate` yields
  `honest_verdict="blocked_gatemate_toolchain_missing"` and an otherwise
  successful diagnostic exit.
- The artifact records any detected modern GateMate alternatives such as
  `nextpnr-himbaechel` or `gmpack` without treating them as the requested
  `nextpnr-gatemate` executable.

**Implementation status:** Implemented (Exp 2914)

---

### SCENARIO-HW-066

**Scenario:** GateMate preflight emits an honest blocked artifact when the legacy nextpnr binary is absent.

**Given:** The n=16 GateMate RTL exists and `yosys` plus `openFPGALoader` are
available, but `nextpnr-gatemate` is absent from PATH and common
OSS-CAD-Suite locations.
**When:** Exp 2914 runs the diagnostic preflight.
**Then:** It writes the v2 preflight JSON with exact detected paths and
versions, `missing_toolchain=["nextpnr-gatemate"]`,
`honest_verdict="blocked_gatemate_toolchain_missing"`, and
`no_flash_attempted=true`.

**Implementation status:** Implemented (Exp 2914)

---

### REQ-HW-067

**Title:** Exp 2916 THRML/KV260 sampler parity MUST compare simulator samples against same-basis evidence without hardware claims

**Description:**
Experiment 2916 MUST read the Exp 2912 same-basis CPU Gibbs baseline before
doing simulator work. If `same_basis_cpu_baseline_ready` is not true, it MUST
write `honest_verdict="blocked_cpu_baseline_not_ready"` and stop. When the CPU
baseline is ready, the experiment MUST import the installed THRML package,
record version/provenance, reconstruct the Exp 2898 n=64 sparse Ising basis
where the THRML software API can represent it, run bounded THRML simulator
sampling with the same random seeds where possible, and compare energy mean,
variance, minimum energy, and histogram distance against the CPU Gibbs and
KV260 final-energy evidence. If the installed THRML API cannot represent the
full n=64 sparse upload, the experiment MUST write the largest explicitly
justified bounded fallback subset artifact, capped at n=16, and state the
limitation.

**Acceptance criteria:**
- The artifact is written to
  `results/experiment_2916_thrml_kv260_sampler_parity_v1.json` with
  `inference_substrate="simulator_parity"` and `run_date="20260523"`.
- Required fields include `honest_verdict`, `thrml_kv260_parity_ready`,
  `thrml_import_ok`, `thrml_version`, `matched_full_n64_basis`,
  `fallback_subset_used`, `random_seeds_used`,
  `energy_distribution_summary`, `cpu_vs_thrml_distance`,
  `kv260_vs_thrml_summary`, `no_tsu_hardware_claim`, `inference_substrate`,
  `duration_s`, and `run_date`.
- The ready path records THRML sample energy mean, variance, minimum, and
  histogram counts, plus CPU-vs-THRML and KV260-vs-THRML histogram distances.
- `no_tsu_hardware_claim` is true in every terminal artifact, and no TSU,
  Z1, Extropic hardware, KV260 board, synthesis, bitstream, or new hardware
  latency claim is made.

**Implementation status:** Planned (Exp 2916)

---

### SCENARIO-HW-067

**Scenario:** THRML simulator samples are compared against matched CPU/KV260 energy evidence.

**Given:** Exp 2912 reports a ready same-basis CPU Gibbs baseline for the Exp
2898 n=64 sparse KV260 upload, and the installed THRML package is importable.
**When:** Exp 2916 runs bounded THRML simulator sampling on the matched basis
or a justified n=16 fallback subset.
**Then:** The artifact records the THRML import provenance, the basis match or
fallback decision, distributional energy summaries, CPU/KV260 histogram
distances, `inference_substrate="simulator_parity"`, and
`no_tsu_hardware_claim=true`.

**Implementation status:** Planned (Exp 2916)

---

### REQ-HW-068

**Title:** Exp 2927 GateMate himbaechel/gmpack constraints preflight MUST replace the obsolete nextpnr-gatemate gate

**Description:**
Experiment 2927 MUST run a diagnostic-only GateMate A1-EVB-2M preflight for the
current open toolchain path: `yosys` -> `nextpnr-himbaechel --device CCGM1A1` ->
`gmpack`. The run MUST inspect PATH and known OSS-CAD-Suite locations for
`nextpnr-himbaechel`, `gmpack`, `yosys`, and `openFPGALoader`, record absolute
paths and versions, and verify that `nextpnr-himbaechel` accepts
`--device CCGM1A1` without invoking synthesis, place-and-route, packing, board
programming, or flashing. The run MUST locate the n=16 GateMate Ising RTL top,
locate an explicit GateMate constraints file, and emit a dry-run command plan.
If no current project constraints file exists and the repository has no
established GateMate generated-constraints pattern, the artifact MUST report
`honest_verdict="blocked_constraints_missing"` instead of inventing pin claims.

**Acceptance criteria:**
- `results/experiment_2927_gatemate_himbaechel_constraints_preflight_v3.json`
  is generated with `inference_substrate="hardware_toolchain_preflight"` and
  `run_date="20260523"`.
- The artifact includes `honest_verdict`, `gatemate_himbaechel_ready`,
  `constraints_ready`, `tool_paths`, `tool_versions`, `device="CCGM1A1"`,
  `nextpnr_command_template`, `gmpack_command_template`, `rtl_top`,
  `constraints_path`, `no_flash_attempted=true`, `inference_substrate`,
  `duration_s`, and `run_date`.
- `gatemate_himbaechel_ready` is true only when `yosys`,
  `nextpnr-himbaechel`, and `gmpack` are detected and the CCGM1A1 device probe
  succeeds.
- Missing constraints yield `honest_verdict="blocked_constraints_missing"`,
  `constraints_ready=false`, and an empty `constraints_path`; no candidate pin
  file is created unless the repository already provides a generated GateMate
  constraints pattern.
- The preflight records `openFPGALoader` path/version for operator context but
  MUST NOT call it in a board-programming or flash mode.

**Implementation status:** Planned (Exp 2927)

---

### SCENARIO-HW-068

**Scenario:** Current GateMate himbaechel/gmpack tools are ready but constraints remain blocked.

**Given:** `yosys`, `nextpnr-himbaechel`, and `gmpack` are present, the
`nextpnr-himbaechel --device CCGM1A1` probe succeeds, and the n=16 GateMate
Ising RTL exists, but no explicit GateMate constraints file exists.
**When:** Exp 2927 runs the corrected diagnostic preflight.
**Then:** It writes the v3 preflight JSON with exact tool paths and versions,
`gatemate_himbaechel_ready=true`, `constraints_ready=false`,
`honest_verdict="blocked_constraints_missing"`, a dry-run yosys/nextpnr/gmpack
command plan, and `no_flash_attempted=true`.

**Implementation status:** Planned (Exp 2927)

---

### REQ-HW-069

**Title:** Exp 2929 GateMate flash smoke MUST stop at the hashed-bitstream gate and preserve physical-board evidence

**Description:**
Experiment 2929 MUST perform a bounded GateMate A1-EVB-2M physical-board smoke
only after `results/experiment_2928_gatemate_n16_himbaechel_bitstream_build_v3.json`
exists, reports `gatemate_bitstream_built=true`, and identifies a bitstream whose
SHA256 matches the artifact. If that precondition is absent or false, Exp 2929
MUST write `honest_verdict="blocked_gatemate_bitstream_missing"` and MUST NOT
probe, reset, flash, or otherwise contact the board. When the gate is satisfied,
the run MUST capture exact command transcripts for initial GateMate/DirtyJTAG
detection, the flash command, and post-flash board contact. It MAY use
`openFPGALoader -c dirtyJtag -b olimex_gatemateevb <bitstream>` only because the
adapter/board path is documented in `research-hardware-wishlist.md`; otherwise
it MUST write `honest_verdict="blocked_flash_path_undocumented"`. The artifact
MUST never claim speedup, and it MUST report `timing_boundary_ready=false` unless
an existing timing counter, UART, or status-register path is already available.

**Acceptance criteria:**
- `results/experiment_2929_gatemate_flash_timing_boundary_v1.json` is generated
  with `inference_substrate="physical_board_smoke"` and `run_date="20260523"`.
- The artifact includes `honest_verdict`, `gatemate_flash_smoke_ready`,
  `board_detected`, `bitstream_sha256_verified`, `flash_attempted`,
  `flash_transcript_path`, `board_contact_transcript_path`,
  `timing_boundary_ready`, `timing_claim_allowed=false`,
  `speedup_claim_allowed=false`, `blocker`, `inference_substrate`,
  `duration_s`, and `run_date`.
- Missing or false Exp 2928 bitstream evidence yields
  `honest_verdict="blocked_gatemate_bitstream_missing"`,
  `gatemate_flash_smoke_ready=false`, `board_detected=false`,
  `bitstream_sha256_verified=false`, `flash_attempted=false`, and empty
  transcript paths.
- When physical-board commands run, their stdout/stderr are written verbatim to
  transcript files; board contact evidence is not summarized in place of the raw
  transcript.
- `timing_claim_allowed=false` and `speedup_claim_allowed=false` are invariant
  for every verdict.

**Implementation status:** Planned (Exp 2929)

---

### SCENARIO-HW-069

**Scenario:** Missing hashed Exp 2928 bitstream blocks the GateMate flash smoke before board contact.

**Given:** The expected Exp 2928 artifact is absent, or it exists with
`gatemate_bitstream_built=false`.
**When:** Exp 2929 runs.
**Then:** It writes the v1 flash-timing-boundary JSON with
`honest_verdict="blocked_gatemate_bitstream_missing"`, records the blocker,
sets both claim-allowed fields to false, leaves both transcript-path fields as
strings, and performs no board-detection or flash command.

**Implementation status:** Planned (Exp 2929)

---

### REQ-HW-070

**Title:** Exp 2930 KV260 p-bit/SSQA scaling projection MUST use real n=64 evidence and remain projection-only

**Description:**
Experiment 2930 MUST read the clean Exp 2913 KV260 hardware/CPU claim boundary
before projecting any n=128 or n=256 Ising-style sampler resource pressure. If
Exp 2913 is absent or `hardware_speedup_claim_eligible` is not true, it MUST
write `honest_verdict="blocked_clean_kv260_basis_missing"` and stop without
inventing scaling estimates. When the clean n=64 basis exists, Exp 2930 MUST
extract the real n=64 latency, sample count, random-seed, sparse topology, and
available resource-evidence fields from Exp 2913 and upstream artifacts, then
produce deterministic spreadsheet-style projections for dense, sparse, and
dual-BRAM-inspired memory layouts at n=128 and n=256. The artifact MUST state
assumptions explicitly, mark unavailable LUT/FF/BRAM evidence as unknown rather
than fabricating synthesis numbers, and MUST NOT perform a new hardware run or
make a new hardware speedup claim.

**Acceptance criteria:**
- `results/experiment_2930_kv260_pbit_ssqa_scaling_projection_v1.json` is
  generated with `projection_only=true`, `no_new_hardware_run=true`,
  `not_a_speedup_claim=true`,
  `inference_substrate="aggregation_plus_simulation"`, and
  `run_date="20260523"`.
- The artifact includes `honest_verdict`,
  `kv260_scaling_projection_ready`, `source_artifacts`,
  `n64_real_evidence_summary`, `projection_models`, `n128_projection`,
  `n256_projection`, `assumptions`, `duration_s`, and all invariant fields
  listed above.
- The ready path records dense, sparse, and dual-BRAM-inspired projection
  models with formulas for memory bits, BRAM36 pressure, and LUT pressure only
  where local evidence supports the estimate.
- The blocked path writes the same schema with empty projection dictionaries and
  preserves the no-new-hardware/no-speedup invariants.

**Implementation status:** Implemented (Exp 2930)

---

### SCENARIO-HW-070

**Scenario:** Clean n=64 KV260 evidence produces projection-only n=128/n=256 resource accounting.

**Given:** Exp 2913 reports `hardware_speedup_claim_eligible=true` for matched
n=64 sparse Ising KV260 and CPU evidence.
**When:** Exp 2930 runs the deterministic projection calculation.
**Then:** It writes the projection-only JSON artifact with real n=64 evidence,
dense/sparse/dual-BRAM projection models, n=128 and n=256 resource-pressure
summaries, explicit assumptions, no board execution, and no new speedup claim.

**Implementation status:** Implemented (Exp 2930)

---

### REQ-HW-071

**Title:** Exp 2938 KV260 MMD comparison MUST test synchronous Glauber energies against exact CPU sequential Gibbs

**Description:**
Experiment 2938 MUST read the completed Exp 2898 KV260 latency artifact,
regenerate the same n=64 dense Ising problems for seeds 42, 137, and 271, and
verify the regenerated dense `J` and `h` checksums against the recorded Exp 2898
metadata before any comparison. It MUST run an exact CPU sequential Gibbs chain
with random spin order and single-spin updates for 10,000 post-burn-in energy
samples per seed. It MUST also exercise the KV260 fabric through SSH and the
existing UIO register map, using the Exp 2898 fixed-sweep synchronous parallel
Glauber schedule without modifying the bitstream, and collect 10,000 per-sample
energies per seed. For every seed it MUST compute RBF-kernel MMD² between the CPU
and KV260 energy distributions using the median pairwise distance bandwidth, a
1000-permutation MMD significance p-value, and the Kolmogorov-Smirnov statistic
and p-value. The artifact MUST turn those three seed-level tests into a single
`distributions_distinguishable` bit at p<0.01 and a concrete paper-v6
recommendation for the "exact sampling on FPGA" claim.

**Acceptance criteria:**
- `results/experiment_2938_kv260_mmd_vs_cpu_sequential_gibbs_v1.json` is
  generated with `inference_substrate="hardware_smoke"` for successful hardware
  runs.
- The artifact includes `honest_verdict`, `preconditions_checked`,
  `cpu_sequential_gibbs_energies_sha256`,
  `kv260_synchronous_glauber_energies_sha256`, `bitstream_sha256_cited`,
  `per_seed_mmd_squared`, `per_seed_mmd_pvalue`, `per_seed_ks_statistic`,
  `per_seed_ks_pvalue`, `distributions_distinguishable`,
  `paper_v6_recommendation`, `random_seeds_used`, `reproducibility_checksum`,
  `methodology_note`, and `duration_s`.
- Successful artifacts record exactly three MMD² values and three MMD p-values,
  one for each Exp 2898 seed, and `duration_s >= 60`.
- If any seed has MMD p-value < 0.01 or KS p-value < 0.01, the artifact MUST
  set `distributions_distinguishable=true` and recommend retracting the exact
  FPGA sampling claim in favor of fixed-schedule heuristic samples.
- If SSH, Exp 2898 provenance, checksum reproduction, or active bitstream SHA
  verification fails, the artifact MUST start `honest_verdict` with the
  specific `blocked_<resource>` prefix and stop without producing a success
  recommendation.

**Implementation status:** Implemented (Exp 2938)

---

### SCENARIO-HW-071

**Scenario:** KV260 synchronous Glauber energies are statistically compared to exact CPU sequential Gibbs energies.

**Given:** `ssh kria` is reachable, Exp 2898 is present, its three dense Ising
problem checksums reproduce from seeds 42, 137, and 271, and the active KV260
bitstream SHA256 matches the Exp 2898 citation.
**When:** Exp 2938 collects 10,000 CPU sequential Gibbs energies and 10,000 KV260
synchronous Glauber energies per seed.
**Then:** It writes the MMD/KS comparison artifact, records the two energy-trace
SHA256 values, and emits a paper-v6 recommendation that retains the narrow
approximately-Boltzmann claim only when all three seed-level MMD and KS p-values
are at least 0.01.

**Implementation status:** Implemented (Exp 2938)

---

### REQ-HW-072

**Title:** Exp 2939 CPU synchronous-parallel same-schedule baseline MUST replace the invalid sequential-Gibbs speedup comparison

**Description:**
Experiment 2939 MUST read the completed Exp 2898 KV260 latency artifact and the
Exp 2912 CPU sequential-Gibbs baseline artifact before computing any speedup
verdict. It MUST regenerate and checksum-verify the same n=64 Exp 2898 Ising
problems, run a CPU synchronous-parallel Glauber updater with the same
even/odd checkerboard schedule used by the KV260 path, and collect 10,000
fixed-budget energy samples for each Exp 2898 seed without early termination.
The run MUST record high-precision per-sample CPU wall-clock timing, compare
the CPU timing against the cited KV260 24.0 us/sample timing, and emit a
paper-v6 recommendation that retracts the prior speedup claim whenever the
same-schedule CPU is faster at n=64.

**Acceptance criteria:**
- `results/experiment_2939_cpu_synchronous_parallel_same_schedule_baseline_v1.json`
  is generated with `inference_substrate="live_llm_inference"`.
- The artifact includes `honest_verdict`, `preconditions_checked`,
  `cpu_synchronous_parallel_per_sample_us_median`,
  `cpu_synchronous_parallel_per_sample_us_p95`,
  `kv260_per_sample_us_cited`, `kv260_speedup_vs_same_schedule_cpu`,
  `energy_distribution_equivalence_test`, `random_seed`,
  `random_seeds_used`, `reproducibility_checksum`,
  `paper_v6_recommendation`, `methodology_note`, and `duration_s`.
- `kv260_speedup_vs_same_schedule_cpu` records the numeric CPU/KV260 ratio and
  the principle that values below 1.0 mean KV260 is slower at this n.
- The energy-equivalence test records a KS p-value and MMD² value for the
  synchronous-parallel same-schedule energy distributions and requires
  `ks_pvalue >= 0.01`.
- Successful artifacts require positive CPU timing and `duration_s >= 20`.

**Implementation status:** Implemented (Exp 2939)

---

### SCENARIO-HW-072

**Scenario:** Same-schedule CPU Glauber baseline produces the apples-to-apples KV260 speedup verdict.

**Given:** Exp 2898 has recoverable n=64 KV260 problem provenance and Exp 2912
exists so the invalid sequential-Gibbs comparison remains auditable.
**When:** Exp 2939 runs 10,000 CPU synchronous checkerboard samples for seeds
42, 137, and 271 and computes the same-schedule timing ratio against
24.0 us/sample.
**Then:** It writes the Exp 2939 result JSON, records the CPU median and p95
per-sample timings, records the KS/MMD same-schedule energy equivalence
cross-check, and recommends retracting the paper-v6 speedup claim when the
ratio is below 1.0.

**Implementation status:** Implemented (Exp 2939)

---

### REQ-HW-073

**Title:** Exp 2941 PolarFire SAT dispatch continuation MUST scale the hash-verified scorer to 500 clauses

**Description:**
Experiment 2941 MUST continue the Exp 2900 PolarFire riscv64 CPU dispatch path
with a deterministic 500-clause SAT instance. Before dispatch it MUST verify
that `ssh polarfire` is reachable and that `ssh polarfire 'uname -m'` reports
`riscv64`; if either precondition fails it MUST write a blocked artifact and
MUST NOT claim a successful hardware smoke. When preconditions pass, it MUST
SCP the SAT instance and scorer to the board, run the scorer on the PolarFire
Linux CPU substrate, pull the transcript back, and verify the remote scorer
output SHA256 against the locally computed expected scorer output. The artifact
MUST compare the 500-clause median per-clause wall-clock time against the
Exp 2900 50-clause median and record that scaling ratio without claiming FPGA
fabric execution.

**Acceptance criteria:**
- `results/experiment_2941_polarfire_continuation_v1.json` is generated with
  `inference_substrate="hardware_smoke"` for successful runs.
- The artifact includes `honest_verdict`, `preconditions_checked`,
  `polarfire_ssh_uptime_at_run`, `n_clauses=500`,
  `per_clause_wall_clock_us_median`, `per_clause_wall_clock_us_p95`,
  `scaling_ratio_vs_exp2900`, `scorer_output_sha256`,
  `scorer_output_hash_verified`, `random_seed`, `reproducibility_checksum`,
  and `duration_s`.
- Successful artifacts require `scorer_output_hash_verified=true` and
  `duration_s >= 15`.
- The reproducibility checksum covers the 500-clause instance hash, scorer
  output hash, random seed, Exp 2900 median citation, and timing summary.

**Implementation status:** Pending (Exp 2941)

---

### REQ-HW-074

**Title:** Exp 2942 KV260 n-scaling latency profile MUST replace crossover extrapolation with hardware measurement or an explicit fixed-n limitation

**Description:**
Experiment 2942 MUST attempt a live KV260 latency profile for
`n in {64, 128, 256, 512, 1024}` using the active `carnot_ising_v4`
bitstream without modifying or reflashing the bitstream. Before measurement it
MUST verify `ssh kria`, the Carnot overlay, UIO access, and the active bitstream
SHA256. The experiment MUST generate deterministic Ising problems for the target
spin counts, run 1000 board samples for each supported `n`, and record median
and p95 per-sample latency in microseconds. If the active bitstream only
supports a fixed `n`, the artifact MUST set
`bitstream_supports_variable_n=false`, run the largest supported `n` that the
bitstream honestly exposes, and document that the requested crossover plot still
lacks direct measurements for the unsupported spin counts.

**Acceptance criteria:**
- `results/experiment_2942_kv260_continuation_n_scaling_v1.json` is generated
  with `inference_substrate="hardware_smoke"`.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `bitstream_supports_variable_n`, `per_n_results`,
  `bitstream_sha256`, `random_seeds_used`, `reproducibility_checksum`, and
  `duration_s`.
- `per_n_results` is a list of dictionaries shaped exactly as
  `{n: int, per_sample_us_median: float, per_sample_us_p95: float}` for every
  measured spin count, and every timing is positive.
- The run MUST NOT modify the active bitstream; unsupported target spin counts
  are represented by the fixed-n limitation instead of fabricated
  extrapolations.
- Successful hardware-smoke artifacts require a non-empty bitstream SHA256 and
  at least one measured `per_n_results` row.

**Implementation status:** Planned (Exp 2942)

---

### SCENARIO-HW-074

**Scenario:** KV260 n-scaling continuation records direct latency rows or the active bitstream's fixed-n boundary.

**Given:** `ssh kria` is reachable, the Carnot Ising overlay is available, UIO
registers are exposed, and the active bitstream SHA256 can be read from the
KV260 filesystem.
**When:** Exp 2942 runs the 1000-sample latency harness for the requested spin
counts `{64, 128, 256, 512, 1024}`.
**Then:** It writes the v1 n-scaling JSON artifact with hardware-smoke
provenance, positive median and p95 per-sample timings for every measured `n`,
the bitstream SHA256, deterministic seeds, a reproducibility checksum, and an
explicit `bitstream_supports_variable_n=false` limitation if the loaded
`carnot_ising_v4` image is fixed to a smaller maximum `n`.

**Implementation status:** Planned (Exp 2942)

---

### SCENARIO-HW-073

**Scenario:** PolarFire riscv64 CPU evaluates the 500-clause continuation scorer without a per-clause timing cliff.

**Given:** `ssh polarfire` is reachable, reports `riscv64`, and Exp 2900 has a
hash-verified 50-clause timing artifact.
**When:** Exp 2941 dispatches the deterministic 500-clause SAT scorer to the
board and pulls back the transcript.
**Then:** The local expected scorer output hash matches the remote transcript
hash, the artifact records positive median and p95 per-clause wall-clock
timings, `scaling_ratio_vs_exp2900` is computed from the Exp 2900 median, and
the success duration is at least 15 seconds.

**Implementation status:** Pending (Exp 2941)

---

### REQ-HW-076

**Title:** Exp 2958 PolarFire scorer continuation MUST extend the hash evidence to 1000 clauses without performance claims

**Description:**
Experiment 2958 MUST continue the narrow Exp 2941 PolarFire SAT scorer path with
a deterministic 1000-clause SAT instance. Before dispatch it MUST verify the
Exp 2941 500-clause baseline artifact and transcript are present and that the
baseline scorer hash was verified. It MUST then check whether `ssh polarfire`
is reachable through the documented path. If the board is reachable, the
experiment MUST SCP the 1000-clause instance and scorer to the PolarFire Linux
CPU substrate, pull back the transcript, and verify the remote scorer output
SHA256 against the locally computed expected scorer output. If the board is not
reachable, it MUST write a blocked artifact using the existing Exp 2941
baseline/transcript context rather than fabricating a successful hardware
smoke. The artifact MUST report only hash, transcript, reachability, and
elapsed-time evidence; it MUST NOT claim speedup or general acceleration.

**Acceptance criteria:**
- `results/experiment_2958_polarfire_1000_clause_scorer_v2.json` is generated.
- The artifact includes `honest_verdict`, `preconditions_checked`,
  `polarfire_1000_clause_hash_verified`, `baseline_500_clause_artifact`,
  `clause_count=1000`, `input_sha256`, `scorer_output_sha256`,
  `transcript_paths`, `elapsed_ms`, `board_reachable`,
  `no_speedup_claim=true`, `no_general_acceleration_claim=true`,
  `inference_substrate="hardware_smoke"`, and `duration_s`.
- Successful artifacts require `board_reachable=true`,
  `polarfire_1000_clause_hash_verified=true`, a 1000-clause input hash, a
  scorer output hash, and a transcript path for the 1000-clause run.
- Blocked artifacts require `board_reachable=false`,
  `polarfire_1000_clause_hash_verified=false`, and the Exp 2941 transcript path
  preserved as evidence for the baseline context.

**Implementation status:** Planned (Exp 2958)

---

### SCENARIO-HW-076

**Scenario:** PolarFire 1000-clause scorer records hash evidence or a precise board block.

**Given:** The Exp 2941 500-clause artifact and transcript exist and the
Exp 2941 artifact records a verified scorer output hash.
**When:** Exp 2958 checks `ssh polarfire` and either dispatches the deterministic
1000-clause SAT scorer or records the unreachable-board block.
**Then:** The v2 result JSON records the baseline artifact path, all required
hash/transcript fields, the board reachability result, elapsed time if measured,
and explicit `no_speedup_claim=true` plus
`no_general_acceleration_claim=true` boundaries.

**Implementation status:** Planned (Exp 2958)

---

### REQ-HW-075

**Title:** Exp 2955 GateMate constraints materialization MUST create a deterministic n=16 build package without flashing

**Description:**
Experiment 2955 MUST locate or create the minimum GateMate A1-EVB-2M package
needed for a later n=16 Ising bitstream build. The package MUST target
`hardware/gatemate/ising_n16_gatemate.v` with top module
`ising_n16_gatemate`, provide an explicit GateMate constraint file, provide a
deterministic test vector for the n=16 coupling/load interface, and record the
hash of that test vector. Before declaring readiness the experiment MUST check
`yosys`, `nextpnr-himbaechel`, `gmpack`, and `openFPGALoader`, MAY run the
read-only `openFPGALoader -c dirtyJtag --detect` probe, and MUST NOT flash or
program the board. If the package relies on unconstrained non-clock IO because
the repository has no authoritative board pinout for every RTL port, the
artifact MUST state that pin assumption explicitly so follow-up bitstream work
does not turn a build-only gate into a physical IO claim.

**Acceptance criteria:**
- `results/experiment_2955_gatemate_constraints_materialization_v4.json` is
  generated with `inference_substrate="deterministic_wiring"`.
- The artifact includes `honest_verdict`, `preconditions_checked`,
  `gatemate_constraints_ready`, `constraints_file_paths`,
  `test_vector_paths`, `top_module`, `clock_assumption`,
  `dirtyjtag_detected`, `toolchain_versions`, `files_changed`,
  `reproducibility_checksum`, `inference_substrate`, and `duration_s`.
- The explicit constraints package includes a GateMate `.ccf` path and a
  deterministic test-vector JSON path whose SHA256 contributes to the
  reproducibility checksum.
- A dry-run lint or synthesis-only check is attempted when yosys is present,
  and any skipped dry-run condition is recorded instead of fabricated.
- The experiment never invokes `openFPGALoader` with a board/programming flag
  such as `-b olimex_gatemateevb`.

**Implementation status:** Planned (Exp 2955)

---

### SCENARIO-HW-075

**Scenario:** GateMate n=16 constraints and test vector are materialized for the next build gate.

**Given:** The current GateMate himbaechel/gmpack toolchain is present and the
n=16 Ising RTL top exists, but the repository has no explicit
`hardware/gatemate/ising_n16_gatemate.ccf` constraints package.
**When:** Exp 2955 runs the materialization step.
**Then:** It writes the minimal constraints/test-vector files, records the
clock and pin assumptions, records the test-vector SHA256, attempts a
synthesis-only dry run when yosys is available, writes the v4 JSON artifact,
sets `gatemate_constraints_ready=true`, and performs no flash/programming
command.

**Implementation status:** Planned (Exp 2955)

---

### REQ-HW-076

**Title:** Exp 2956 GateMate n=16 bitstream build MUST use the exp2955 constraints package without flashing

**Description:**
Experiment 2956 MUST build the GateMate n=16 Ising top from the Exp 2955
constraints package using the current OSS CAD Suite flow:
`yosys synth_gatemate`, `nextpnr-himbaechel --device CCGM1A1`, and `gmpack`.
The experiment MUST first verify that Exp 2955 reports
`gatemate_constraints_ready=true`, the referenced RTL and constraints files
exist, and `yosys`, `nextpnr-himbaechel`, `gmpack`, and `openFPGALoader` are
available. The experiment MUST NOT flash or program the GateMate board. If any
tool or build step fails, the artifact MUST preserve the exact failing command
and the first actionable error excerpt.

**Acceptance criteria:**
- `results/experiment_2956_gatemate_n16_bitstream_build_v4.json` is generated
  with `inference_substrate="hardware_build"`.
- The artifact includes `honest_verdict`, `preconditions_checked`,
  `gatemate_bitstream_built`, `synthesis_command`, `pnr_command`,
  `pack_command`, `bitstream_path`, `bitstream_sha256`, `timing_summary`,
  `utilization_summary`, `build_log_paths`, `failure_command`,
  `failure_excerpt`, `inference_substrate`, and `duration_s`.
- A successful build records a non-empty bitstream path whose SHA256 matches the
  generated file, plus synthesis, place-and-route, and pack log paths.
- The synthesis command uses `synth_gatemate` with luttree mapping so the JSON
  handed to nextpnr avoids unsupported legacy `CC_LUT*` cells.
- The PnR command uses `nextpnr-himbaechel --device CCGM1A1`, the Exp 2955 CCF,
  `--freq 12.0`, and `--vopt allow-unconstrained`.
- No `openFPGALoader` command with a board/programming flag such as
  `-b olimex_gatemateevb` is invoked.

**Implementation status:** Planned (Exp 2956)

---

### SCENARIO-HW-076

**Scenario:** GateMate n=16 bitstream build records complete evidence or an actionable blocker.

**Given:** Exp 2955 has produced a ready constraints package, the current OSS
CAD Suite GateMate tools are present, and the n=16 GateMate RTL exists.
**When:** Exp 2956 runs the yosys, nextpnr-himbaechel, and gmpack build flow.
**Then:** It writes the v4 hardware-build artifact with command strings, log
paths, timing and utilization summaries, and either
`gatemate_bitstream_built=true` plus a bitstream SHA256 or a blocked verdict
with the exact failing command and first actionable error.

**Implementation status:** Planned (Exp 2956)

---

### REQ-HW-077

**Title:** Exp 2957 GateMate n=16 flash/timing smoke MUST preserve board evidence without sampler claims

**Description:**
Experiment 2957 MUST perform the first bounded GateMate n=16 flash/timing smoke
after Exp 2956 has produced a real `.bit` artifact. The experiment MUST first
verify `results/experiment_2956_gatemate_n16_bitstream_build_v4.json` reports
`gatemate_bitstream_built=true`, verify the local bitstream SHA256 equals the
Exp 2956 SHA256, locate `openFPGALoader`, and run
`openFPGALoader -c dirtyJtag --detect` to prove the DirtyJTAG/GateMate board is
present. It MAY flash only with the locally documented command shape
`openFPGALoader -c dirtyJtag -b olimex_gatemateevb <bitstream>`. Every hardware
command MUST be captured to a raw transcript file. If the board, bitstream, or
flash path is unavailable, the artifact MUST use a `blocked_*` verdict and
record the exact failing command. Because the current Exp 2955/2956 CCF is
build-only and intentionally leaves user IO unconstrained, Exp 2957 MUST NOT
claim sample readback, speedup, Boltzmann sampling, or thermodynamic behavior
unless a real board-output capture path is present.

**Acceptance criteria:**
- `results/experiment_2957_gatemate_flash_timing_smoke_v2.json` is generated
  with `inference_substrate="hardware_smoke"`.
- The artifact includes `honest_verdict`, `preconditions_checked`,
  `board_detected`, `bitstream_sha256_verified`, `flash_attempted`,
  `flash_succeeded`, `smoke_vector_passed`, `observed_output_sha256`,
  `timing_observation`, `transcript_paths`, `no_speedup_claim=true`,
  `no_boltzmann_claim=true`, `inference_substrate`, and `duration_s`.
- A successful flash records the initial detection transcript, flash transcript,
  post-flash detection transcript, transcript SHA256 values, and measured command
  durations, but keeps `smoke_vector_passed=false` when no readback/output
  capture path exists.
- Missing Exp 2956 evidence, SHA mismatch, absent `openFPGALoader`, failed board
  detection, or failed flash yields a blocked artifact with `flash_succeeded=false`
  and the exact failing command.
- `no_speedup_claim=true`, `no_boltzmann_claim=true`, and
  `inference_substrate="hardware_smoke"` are invariant for every verdict.

**Implementation status:** Planned (Exp 2957)

---

### SCENARIO-HW-077

**Scenario:** GateMate n=16 flash smoke records hardware contact or an exact blocker.

**Given:** Exp 2956 reports a built n=16 GateMate bitstream and the local bitstream
hash matches its recorded SHA256.
**When:** Exp 2957 runs the DirtyJTAG detection and documented openFPGALoader flash
flow.
**Then:** It writes the v2 hardware-smoke artifact with required schema fields,
raw transcript paths, transcript hashes, command timing observations, and either
a successful flash/contact smoke without sampler claims or a blocked verdict with
the exact failing command.

**Implementation status:** Planned (Exp 2957)

---

### REQ-HW-078

**Title:** Exp 2971 GateMate board detection preflight MUST verify contact and prepare flash command without flashing

**Description:**
Experiment 2971 MUST diagnose the GateMate A1-EVB-2M DirtyJTAG contact state
after Exp 2957 blocked at board detection. The experiment MUST check
`openFPGALoader` availability, DirtyJTAG/GateMate USB visibility, likely
permission/device-node evidence, Exp 2956 bitstream presence, and the local
bitstream SHA256 before deciding whether flash preconditions are ready. It MUST
run `openFPGALoader -c dirtyJtag --detect` and capture the raw transcript. If
the transcript contains the GateMate IDCODE/metadata and the Exp 2956 hash
matches, the artifact MUST prepare the exact
`openFPGALoader -c dirtyJtag -b olimex_gatemateevb <bitstream>` command and set
`gatemate_flash_preconditions_ready=true`, but MUST NOT execute that flash
command or claim a successful board run. If detection or hash verification
fails, the artifact MUST preserve the first actionable failing command and
transcript path.

**Acceptance criteria:**
- `results/experiment_2971_gatemate_board_detection_flash_harness_v3.json` is
  generated with `inference_substrate="hardware_preflight"`.
- The artifact includes `honest_verdict`, `preconditions_checked`,
  `gatemate_board_detected`, `bitstream_sha256_verified`,
  `gatemate_flash_preconditions_ready`, `detection_commands`,
  `detection_transcript_paths`, `bitstream_path`, `bitstream_sha256`,
  `flash_command`, `failure_command`, `failure_excerpt`, `files_changed`,
  `inference_substrate`, and `duration_s`.
- Successful preflight requires a matching Exp 2956 bitstream SHA256, a
  successful DirtyJTAG detect transcript that identifies `colognechip` /
  `GateMate Series` / `GM1Ax`, and a flash command string only; the flash
  command is never executed.
- Blocked artifacts preserve the earliest actionable blocker among missing
  `openFPGALoader`, missing USB visibility, missing or mismatched Exp 2956
  bitstream, and failed board detection.

**Implementation status:** Planned (Exp 2971)

---

### SCENARIO-HW-078

**Scenario:** GateMate detection preflight emits a ready flash command or a precise block.

**Given:** Exp 2956 reports a built n=16 GateMate bitstream and the host can run
non-destructive DirtyJTAG discovery commands.
**When:** Exp 2971 checks tool/USB/permission preconditions, verifies the local
bitstream SHA256, and runs `openFPGALoader -c dirtyJtag --detect`.
**Then:** It writes the v3 hardware-preflight JSON with raw detection
transcripts, the exact flash command to run later when ready, and either
`gatemate_flash_preconditions_ready=true` without flashing or a blocked verdict
with the first failing command and transcript path.

**Implementation status:** Planned (Exp 2971)

---

### REQ-HW-079

**Title:** Exp 2972 GateMate post-flash output-hash smoke MUST be gated on Exp 2971 and avoid sampler claims

**Description:**
Experiment 2972 MUST flash the GateMate n=16 bitstream only after
`results/experiment_2971_gatemate_board_detection_flash_harness_v3.json`
reports both `gatemate_board_detected=true` and
`bitstream_sha256_verified=true`. Immediately before flashing, the experiment
MUST re-run the DirtyJTAG/GateMate detection command and re-verify the local
bitstream SHA256 against the Exp 2971 command payload. The flash command MUST be
the exact command emitted by Exp 2971. Every detection, flash, and post-flash
smoke/readback command MUST be captured to a raw transcript file and hashed.
Because the current GateMate n=16 bitstream has no host-visible sampler output
or readback path, a successful run MAY report only flash/contact transcript
hashes and directly measured command timing. It MUST NOT claim speedup,
Boltzmann sampling, thermalization, or thermodynamic sampling behavior.

**Acceptance criteria:**
- `results/experiment_2972_gatemate_post_flash_output_hash_v3.json` is
  generated with `inference_substrate="hardware_smoke"`.
- The artifact includes `honest_verdict`, `preconditions_checked`,
  `board_detected`, `bitstream_sha256_verified`, `flash_attempted`,
  `flash_succeeded`, `smoke_vector_passed`, `observed_output_sha256`,
  `timing_observation`, `transcript_paths`, `failure_command`,
  `failure_excerpt`, `no_speedup_claim=true`, `no_boltzmann_claim=true`,
  `no_thermalization_claim=true`, `inference_substrate`, and `duration_s`.
- Preconditions fail closed unless Exp 2971 reports board detection and
  SHA256 verification, the parsed Exp 2971 flash command names an existing
  bitstream, the local bitstream SHA256 still matches the Exp 2971 hash, and
  the immediate pre-flash detection transcript identifies GateMate hardware.
- A flash failure or post-flash contact/readback failure writes a `blocked_*`
  artifact with `flash_attempted` and `flash_succeeded` reflecting what really
  happened, plus the exact failing command and transcript-backed excerpt.
- A successful no-readback smoke records the pre-flash detection transcript,
  flash transcript, post-flash detection transcript, transcript SHA256 values,
  measured command durations, `smoke_vector_passed=false`, and the post-flash
  transcript hash as `observed_output_sha256`.

**Implementation status:** Planned (Exp 2972)

---

### SCENARIO-HW-079

**Scenario:** GateMate post-flash smoke records output hash evidence or an exact blocker.

**Given:** Exp 2971 reports GateMate board detection, bitstream SHA256
verification, and an exact openFPGALoader flash command for the Exp 2956 n=16
GateMate bitstream.
**When:** Exp 2972 rechecks detection and hash evidence, executes the Exp 2971
flash command, and runs the smallest available post-flash contact/readback path.
**Then:** It writes the v3 hardware-smoke JSON with transcript paths,
transcript hashes, command timing observations, no forbidden sampler claims, and
either a successful flash/contact hash smoke or a blocked verdict with the exact
failing command.

**Implementation status:** Planned (Exp 2972)

---

### REQ-HW-080

**Title:** Exp 2984 GateMate readback/smoke-vector artifact MUST distinguish board contact from sampler IO

**Description:**
Experiment 2984 MUST attempt to convert the Exp 2971/2972 GateMate contact
evidence into readback-backed or smoke-vector-backed evidence without overstating
what the board can prove. The experiment MUST read the Exp 2971 and Exp 2972
artifacts, recover the board ID, exact flash command, bitstream path, bitstream
SHA256, and post-flash transcript hashes, then run only non-destructive local
tool and board probes unless a readback command is explicitly supported by the
installed tools. If the installed GateMate tool flow exposes no SRAM readback,
JTAG register, UART, GPIO, or physically constrained IO path for the n=16 tile,
the artifact MUST set `readback_supported=false`, `readback_attempted=false`,
`smoke_vector_attempted=false`, and `smoke_vector_passed=false` while explaining
the blocker from the CCF/RTL/test-vector evidence. `smoke_vector_passed=true`
is permitted only when an observed board output exactly matches the expected
smoke-vector output.

**Acceptance criteria:**
- `results/experiment_2984_gatemate_readback_smoke_vector_v4.json` is generated
  with `inference_substrate="physical_gatemate_board"`.
- The artifact includes `honest_verdict`, `board_detected`, `board_id`,
  `tool_versions`, `bitstream_path`, `bitstream_sha256`, `flash_succeeded`,
  `readback_supported`, `readback_attempted`, `readback_hash`,
  `smoke_vector_attempted`, `smoke_vector_passed`, `observed_smoke_output`,
  `expected_smoke_output`, `timing_observation`, `sampler_claim_allowed=false`,
  `speedup_claim_allowed=false`, `thermodynamic_claim_allowed=false`,
  `inference_substrate`, and `duration_s`.
- Board detection uses `openFPGALoader -c dirtyJtag --detect` and records a
  GateMate board ID only when the transcript identifies the Cologne Chip
  GateMate/GM1Ax IDCODE.
- The tool-version record covers the installed GateMate-relevant tools that are
  available locally and notes missing tools without failing the artifact.
- Readback is attempted only when the installed tool help exposes a non-flash,
  GateMate-compatible readback or verify path for the programmed n=16 bitstream.
- Smoke-vector IO is attempted only when the n=16 GateMate package exposes a
  host-visible physical IO protocol. Build-only unconstrained CCF evidence MUST
  block sampler-output claims.

**Implementation status:** Planned (Exp 2984)

---

### SCENARIO-HW-080

**Scenario:** GateMate readback/smoke-vector artifact records evidence or a precise IO blocker.

**Given:** Exp 2971 and Exp 2972 record GateMate detection, a flash command, a
matching n=16 bitstream SHA256, and post-flash contact evidence.
**When:** Exp 2984 probes the board and installed GateMate tooling for readback
and host-visible smoke-vector IO.
**Then:** It writes the v4 JSON with required schema fields, command timing
observations, claim-allowance fields fixed false unless real sample-level
evidence exists, and either readback/smoke-vector evidence or an exact blocker
explaining why the current bitstream cannot expose sampler output.

**Implementation status:** Planned (Exp 2984)

---

### REQ-HW-081

**Title:** Exp 2985 SSQA dual-BRAM register-map plan MUST remain projection-only

**Description:**
Experiment 2985 MUST produce a design-only register-map and memory-banking plan
for the Carnot n=16 GateMate tile and a future SSQA-style dual-BRAM sampling
state. The artifact MUST read the current GateMate bitstream/contact evidence
from Exp 2972, the readback/smoke-vector blocker from Exp 2984, and the
GateMate constraints/build package evidence from Exp 2955/2956. It MUST turn
that evidence into explicit host-visible register offsets, banked memory
layout assumptions, IO smoke vectors, readback checks, and resource-accounting
fields without flashing hardware, running a sampler, claiming physical
performance, or touching `scripts/research_conductor.py`.

**Acceptance criteria:**
- `results/experiment_2985_ssqa_dual_bram_register_map_plan_v1.json` is
  generated with `inference_substrate="architecture_projection_only"`.
- The artifact includes `honest_verdict`, `register_map_plan_ready`,
  `projection_only=true`, `target_boards`, `register_map`, `memory_layout`,
  `smoke_vectors`, `readback_checks`, `resource_accounting`, `risks`,
  `sampler_claim_allowed=false`, `speedup_claim_allowed=false`,
  `inference_substrate`, and `duration_s`.
- The register map separates input/control fields, seed/state fields,
  energy/verifier fields, output fields, and status/error fields, with stable
  32-bit aligned offsets and explicit access directions.
- The memory layout documents a two-bank plan: a read-snapshot bank for
  couplings/bias/current spins and a delayed-write bank for next spins, field
  cache, RNG threshold state, and phase flags. Resource accounting MUST include
  formulas, bit counts, per-bank BRAM rounding, and unknown fields where no
  synthesis utilization evidence exists.
- Smoke vectors and readback checks are executable only by a later milestone
  with a host-visible IO path; the current artifact MUST preserve
  `sampler_claim_allowed=false` and `speedup_claim_allowed=false`.

**Implementation status:** Planned (Exp 2985)

---

### SCENARIO-HW-081

**Scenario:** Projection-only register map prepares later GateMate/KV260 readback.

**Given:** Exp 2972 records a flashed GateMate n=16 bitstream but no sampler
readback path, Exp 2984 records that GateMate readback and host smoke-vector IO
are unavailable, and Exp 2955/2956 record the current constraints/build package.
**When:** Exp 2985 builds the SSQA dual-BRAM register-map plan.
**Then:** It writes the projection-only JSON with explicit registers, memory
banks, smoke vectors, readback checks, resource-accounting formulas, and claim
flags fixed false.

**Implementation status:** Planned (Exp 2985)

---

### REQ-HW-082

**Title:** Exp 2996 GateMate host-visible readback smoke MUST terminate at observable IO or a precise blocker

**Description:**
Experiment 2996 MUST perform a bounded GateMate A1-EVB-2M hardware smoke boundary
for the currently built n=16 bitstream without making sampler, thermodynamic, or
speedup claims. The experiment MUST identify the board connection, installed
GateMate tool versions, target bitstream/RTL, programmer command, and intended
host-visible IO path before programming. It MUST then record whether board
contact, flashing, readback support, readback attempt, smoke-vector attempt, and
host-visible smoke-vector pass occurred as separate booleans. If the current RTL,
CCF, test vector, or installed tools do not expose a host-visible path for
`spin_out`/`done`, the artifact MUST record the exact missing interface and
return a blocked verdict rather than upgrading flash/contact evidence into
sampler evidence.

**Acceptance criteria:**
- `results/experiment_2996_gatemate_host_visible_readback_smoke_v1.json` is
  generated with a terminal claim boundary.
- The artifact includes `hardware_smoke_boundary_recorded`,
  `preconditions_checked`, `board_detected`, `flash_attempted`,
  `flash_succeeded`, `readback_attempted`, `readback_supported`,
  `smoke_vector_attempted`, `smoke_vector_passed`, `host_visible_output_path`,
  `transcript_paths`, `sampler_claim_made=false`, `speedup_claim_made=false`,
  and `honest_verdict`.
- Preconditions distinguish setup failures from design/interface failures by
  recording board connection evidence, tool versions, target bitstream/RTL,
  programmer command, and the intended host-visible IO path before flash.
- Flash success MUST be recorded separately from readback and smoke-vector
  success.
- `smoke_vector_passed=true` is allowed only when a host-observable readback or
  IO transcript contains the expected deterministic smoke-vector output.
- When no UART, GPIO, JTAG register, status register, AXI/CSR, or physically
  constrained pin path exposes `spin_out`/`done`, the artifact MUST set
  `readback_supported=false`, `smoke_vector_passed=false`, and an honest blocked
  verdict with the exact missing interface.

**Implementation status:** Planned (Exp 2996)

---

### SCENARIO-HW-082

**Scenario:** GateMate host-visible smoke terminates at blocked IO without sampler claims.

**Given:** Prior GateMate artifacts record board contact, a matching n=16
bitstream, and a flash command for the GateMate A1-EVB-2M.
**When:** Exp 2996 checks preconditions, flashes the target bitstream, probes
readback support, and inspects the current GateMate RTL/CCF/test-vector package
for a host-visible output path.
**Then:** It writes the v1 JSON with required schema fields, replayable command
transcripts, no sampler or speedup claims, and either observable smoke/readback
evidence or a precise blocked diagnosis naming the missing host-visible
interface.

**Implementation status:** Planned (Exp 2996)

---

### REQ-HW-083

**Title:** Exp 3008 GateMate host-visible IO transport gate MUST expose real output or block downstream SSQA

**Description:**
Experiment 3008 MUST produce the terminal GateMate host-visible IO transport
artifact for milestone .282. The experiment MUST inspect the current GateMate
RTL, CCF, test vector, prior readback artifacts, board connection evidence,
toolchain versions, programmer command, target bitstream, and USB permission
evidence before any programming attempt. It MUST then record board detection,
flash, readback, smoke-vector, and IO-transport readiness as separate booleans.
`host_visible_io_ready=true` is permitted only when a host-observable transport
captures deterministic smoke output from the programmed GateMate design. If the
current board, toolchain, RTL, constraints, or missing reader cannot expose a
status byte/word, the artifact MUST set `host_visible_io_ready=false` and name
the missing interface without making sampler, speedup, Boltzmann, or
thermodynamic claims.

**Acceptance criteria:**
- `results/experiment_3008_gatemate_host_visible_io_transport_v2.json` is
  generated as a terminal hardware-smoke artifact.
- The artifact includes `host_visible_io_ready`,
  `hardware_smoke_boundary_recorded`, `preconditions_checked`,
  `board_detected`, `flash_attempted`, `flash_succeeded`,
  `readback_attempted`, `readback_supported`, `smoke_vector_attempted`,
  `smoke_vector_passed`, `io_transport_path`, `transcript_paths`,
  `sampler_claim_made=false`, `speedup_claim_made=false`, and
  `honest_verdict`.
- Preconditions name the board connection, GateMate tool versions, programmer
  command, target bitstream/RTL, USB permission evidence when available, and
  the observed or intended IO path before downstream consumers can proceed.
- Flash success, readback support, readback attempt, smoke-vector attempt, and
  host-visible smoke-vector pass MUST remain independently inspectable.
- Downstream SSQA MUST gate on `host_visible_io_ready=true`, not on board
  detection, successful flash, or post-flash JTAG contact alone.

**Implementation status:** Implemented (Exp 3008)

---

### SCENARIO-HW-083

**Scenario:** GateMate IO transport artifact blocks when no host-visible smoke output exists.

**Given:** Prior GateMate artifacts show board contact and the current n=16
GateMate bitstream, while Exp 2984/2996 show no supported readback or
host-visible smoke vector.
**When:** Exp 3008 checks the current RTL/CCF/test-vector package and runs the
bounded GateMate hardware smoke boundary.
**Then:** It writes the v2 JSON with required schema fields, replayable
transcripts, claim flags fixed false, and either `host_visible_io_ready=true`
from observed deterministic output or a blocked verdict naming the exact
missing transport, reader, flash, or board-contact interface.

**Implementation status:** Implemented (Exp 3008)

---

### REQ-HW-084

**Title:** Exp 3021 GateMate RTL/CCF transport shim diagnosis MUST gate board smoke on observable output

**Description:**
Experiment 3021 MUST add or precisely diagnose the minimum GateMate
host-visible transport shim for deterministic `done`/`spin_out` status. The
experiment MUST inspect the current GateMate RTL, CCF constraints, generated
GateMate build outputs, installed toolchain commands, USB/DirtyJTAG board
connection evidence, programmer command, and intended host-visible IO path before
any synthesis, PnR, or flashing. It MUST set `gatemate_transport_rtl_ready=true`
only when the RTL exposes a deterministic status bit or byte and the CCF binds
that signal to a physical output or supported host transport. It MUST set
`host_visible_io_plan_ready=true` only when a concrete reader path can observe
that bound status output from the host. If the repository lacks an authoritative
board pinout, physical `Pin_out` assignment, or supported UART/GPIO/JTAG/status
register/CSR/AXI/logic-analyzer reader, the experiment MUST emit a terminal
blocked diagnosis and MUST NOT make sampler, speedup, Boltzmann, or
thermalization claims.

**Acceptance criteria:**
- `results/experiment_3021_gatemate_rtl_ccf_host_visible_transport_shim_v1.json`
  is generated with strict claim boundaries.
- The artifact includes `gatemate_transport_rtl_ready`,
  `host_visible_io_plan_ready`, `preconditions_checked`, `board_detected`,
  `rtl_paths`, `ccf_paths`, `io_transport_path`, `simulation_or_lint_passed`,
  `pnr_or_synthesis_attempted`, `transcript_paths`, `sampler_claim_made`,
  `speedup_claim_made`, and `honest_verdict`.
- Preconditions distinguish setup failures from design/interface failures by
  recording board connection, toolchain versions, programmer availability,
  target RTL/CCF paths, USB permission evidence, and the intended IO path before
  any flash or synthesis action.
- `gatemate_transport_rtl_ready=true` requires a deterministic status output in
  RTL and a physical CCF binding or supported host transport for that signal.
- `host_visible_io_plan_ready=true` requires a concrete host reader path; build
  only unconstrained CCF output, internal Verilog ports, JTAG contact, or
  successful bitstream generation are insufficient.
- If no physical `Pin_out` or reader path exists, the artifact MUST set both
  readiness booleans false, record exact blockers, and leave
  `sampler_claim_made=false` and `speedup_claim_made=false`.

**Implementation status:** Implemented (Exp 3021)

---

### SCENARIO-HW-084

**Scenario:** GateMate transport shim diagnosis blocks on missing physical output binding.

**Given:** The current GateMate n=16 RTL exposes internal `done` and `spin_out`
signals, the GateMate tools and DirtyJTAG connection may be present, and the CCF
states that no physical `Pin_in` or `Pin_out` locations are assigned.
**When:** Exp 3021 checks preconditions, inspects RTL/CCF/toolchain transport
surfaces, and runs only available non-flashing design checks.
**Then:** It writes the v1 JSON with replayable transcripts, no sampler or
speedup claims, `gatemate_transport_rtl_ready=false`,
`host_visible_io_plan_ready=false`, and a terminal verdict naming the missing
pinout or reader path.

**Implementation status:** Implemented (Exp 3021)

---

### REQ-HW-085

**Title:** Exp 3023 SSQA gate artifact MUST be written even when upstream GateMate host-visible IO is closed

**Description:**
Experiment 3023 MUST produce
`results/experiment_3023_ssqa_explicit_gate_artifact_and_rtl_report_v1.json`
on every run so SSQA status cannot disappear from milestone matrix or capstone
evidence. The experiment MUST read Exp 3022 as the immediate upstream
GateMate host-visible IO gate. If Exp 3022 is missing, blocked, or does not
report `host_visible_io_ready=true`, Exp 3023 MUST set
`ssqa_gate_status="gate_skipped"`, skip synthesis/PnR/flash work, name the
upstream blocker precisely, and keep all sampler, speedup, Boltzmann,
thermodynamic, and FPGA-acceleration claims false. Only when Exp 3022 reports
`host_visible_io_ready=true` may Exp 3023 collect bounded RTL/PnR/resource
evidence for the SSQA dual-BRAM register-map path.

**Acceptance criteria:**
- `results/experiment_3023_ssqa_explicit_gate_artifact_and_rtl_report_v1.json`
  is generated on every run.
- The artifact includes `ssqa_artifact_written`, `ssqa_gate_status`,
  `ssqa_rtl_pnr_report_ready`, `preconditions_checked`,
  `upstream_host_visible_io_ready`, `rtl_path`, `pnr_report_path`,
  `resource_report_path`, `smoke_hook_paths`, `projection_only`,
  `sampler_claim_made`, `speedup_claim_made`, and `honest_verdict`.
- When Exp 3022 is absent, blocked, or has `host_visible_io_ready=false`, the
  artifact records `ssqa_gate_status="gate_skipped"`,
  `ssqa_rtl_pnr_report_ready=false`, `projection_only=true`, and a terminal
  verdict beginning with `complete:` that names the upstream gate closure.
- When Exp 3022 reports `host_visible_io_ready=true`, the artifact may mark
  RTL/PnR/resource evidence ready only when source RTL and inspectable PnR and
  resource report paths are present.
- The artifact MUST NOT claim sampling speedup, Boltzmann correctness,
  thermodynamic behavior, or FPGA acceleration.

**Implementation status:** Implemented (Exp 3023)

---

### SCENARIO-HW-085

**Scenario:** SSQA status is explicit when Exp 3022 blocks host-visible IO.

**Given:** Exp 3022 exists but is blocked before host-visible IO is ready, and
Exp 3021 identifies the missing GateMate physical output or reader boundary.
**When:** Exp 3023 builds the SSQA explicit gate artifact.
**Then:** It writes the v1 JSON with `ssqa_gate_status="gate_skipped"`, no
synthesis/PnR/resource claims, concrete upstream status fields, and claim
flags fixed false.

**Implementation status:** Implemented (Exp 3023)

---

### REQ-HW-086

**Title:** Exp 3034 GateMate output contract decision MUST either name an observable host path or a precise operator blocker

**Description:**
Experiment 3034 MUST produce
`results/experiment_3034_gatemate_output_contract_pinout_decision_v1.json`
before downstream GateMate RTL shim or flash-smoke work proceeds. The
experiment MUST inspect the current GateMate RTL, CCF, build evidence, Exp 3021
transport diagnosis, and non-flashing toolchain/DirtyJTAG preconditions. It MUST
choose one output path from UART, LED/GPIO pattern, JTAG-readable status, or
explicit no-ready-contract. `gatemate_output_contract_ready=true` and
`host_visible_io_plan_ready=true` are permitted only when a deterministic RTL
status output has a physical CCF binding and a concrete host reader command with
an expected pass/fail transcript. If the repository lacks an authoritative
physical pinout, CCF `Pin_out`, or supported reader command, the artifact MUST
set both readiness booleans false and record the exact operator action required
to unblock the contract.

**Acceptance criteria:**
- `results/experiment_3034_gatemate_output_contract_pinout_decision_v1.json`
  is generated with `gatemate_output_contract_ready`,
  `host_visible_io_plan_ready`, `selected_output_path`, `pinout_table`,
  `host_reader_command`, `exact_operator_action_required`,
  `board_detect_command`, `toolchain_preconditions`, `inference_substrate`, and
  `honest_verdict`.
- The non-flashing preconditions record availability/version evidence for
  `openFPGALoader`, `yosys`, `nextpnr-himbaechel`, and the GateMate packer
  command, the DirtyJTAG detection command shape
  `openFPGALoader -c dirtyJtag --detect`, target board
  `olimex_gatemateevb`, and USB permission evidence when available.
- The pinout table maps each candidate pass/fail signal to RTL source, CCF
  binding, host read command, expected transcript, and blocker status.
- If no physical `Pin_out` or supported reader exists, the selected output path
  is `explicit_no_ready_contract`, both readiness booleans are false, and the
  operator action list names the missing board pinout or reader decision.
- The artifact MUST NOT claim hardware execution, timing, speedup, sampling
  quality, Boltzmann correctness, or thermodynamic behavior.

**Implementation status:** Implemented (Exp 3034)

---

### SCENARIO-HW-086

**Scenario:** GateMate output contract blocks on missing pinout and reader.

**Given:** The current GateMate n=16 RTL exposes `done` and `spin_out`, the CCF
contains only build-only unconstrained comments, and Exp 3021 reports
`host_visible_io_plan_ready=false` because no physical output binding exists.
**When:** Exp 3034 audits the candidate UART, LED/GPIO, and JTAG/status paths
without flashing the board.
**Then:** It writes the v1 JSON with
`selected_output_path="explicit_no_ready_contract"`,
`gatemate_output_contract_ready=false`,
`host_visible_io_plan_ready=false`, a blocked pinout table, explicit operator
actions, and a terminal verdict beginning with `complete:`.

**Implementation status:** Implemented (Exp 3034)

---

### REQ-HW-087

**Title:** Exp 3037 SSQA bounded RTL/PnR gate artifact MUST distinguish run, blocked, and gate-skipped states

**Description:**
Experiment 3037 MUST produce
`results/experiment_3037_ssqa_bounded_rtl_pnr_gate_artifact_v2.json` for
milestone .284 so the capstone has an explicit SSQA hardware boundary row. The
artifact MUST load Exp 3034, Exp 3035, and Exp 3036 GateMate artifacts when they
exist, inspect the current `hardware/gatemate/` package, and set
`ssqa_gate_status="gate_skipped"` when Exp 3036 is missing, blocked, or lacks
observed host-visible output. Only when Exp 3036 reports
`gatemate_flash_smoke_ready=true` may Exp 3037 run bounded RTL/PnR/resource
commands, and those commands MUST be recorded as reproducible command lines and
report paths without implying latency, energy, annealing quality, or speedup.

**Acceptance criteria:**
- `results/experiment_3037_ssqa_bounded_rtl_pnr_gate_artifact_v2.json` is
  generated with `ssqa_boundary_ready`, `ssqa_gate_status`,
  `upstream_gatemate_status`, `rtl_or_pnr_commands_run`,
  `resource_report_paths`, `ssqa_performance_claim_allowed`,
  `exact_blocker_or_next_action`, `inference_substrate`, and
  `honest_verdict`.
- `ssqa_gate_status` uses distinct values for successful bounded evidence
  collection (`"run"`), bounded-command precondition failure (`"blocked"`),
  and upstream hardware gate closure (`"gate_skipped"`).
- If Exp 3036 is missing, blocked, or has no host-visible output transcript,
  the artifact records `ssqa_gate_status="gate_skipped"`, runs no RTL/PnR
  commands, cites the exact upstream blocker, and keeps
  `ssqa_performance_claim_allowed=false`.
- If Exp 3036 reports `gatemate_flash_smoke_ready=true`, the artifact runs only
  bounded RTL/PnR/resource commands for the current GateMate RTL package and
  records the command lines, return codes, and output/report paths.
- The artifact MUST NOT invent latency, energy, annealing, speedup,
  thermodynamic, or board-performance results.

**Implementation status:** Implemented (Exp 3037)

---

### SCENARIO-HW-087

**Scenario:** SSQA .284 boundary emits an explicit gate skip when Exp 3036 is absent or lacks host-visible output.

**Given:** Exp 3034 blocks the output contract, Exp 3035 is blocked by that
contract gate, and Exp 3036 is missing or does not report a host-visible flash
smoke transcript.
**When:** Exp 3037 builds the SSQA bounded RTL/PnR gate artifact.
**Then:** It writes the v2 JSON with `ssqa_boundary_ready=true`,
`ssqa_gate_status="gate_skipped"`, no RTL/PnR commands run, no performance
claim allowed, a concrete blocker or next action, and a terminal verdict
beginning with `complete:`.

**Implementation status:** Implemented (Exp 3037)

---

### REQ-HW-088

**Title:** Exp 3048 GateMate output contract operator package MUST gate downstream flash work on concrete pinout and reader authority

**Description:**
Experiment 3048 MUST produce
`results/experiment_3048_gatemate_output_contract_operator_package_v1.json`
as an operator-ready package for the GateMate output contract. The package MUST
load Exp 3034, extract its candidate output signals, blocked pinout table, and
operator actions, then inspect only local repository documentation, committed
`hardware/gatemate/` files, and tool availability. It MUST NOT invent a
physical pinout, flash hardware, or claim board execution. If a deterministic
status output has both a concrete CCF `Pin_out` binding and a concrete host
reader command, the package may set
`gatemate_output_contract_ready=true` and
`host_visible_io_plan_ready=true`; otherwise both readiness fields MUST remain
false and the package MUST name the exact missing operator actions so
downstream Exp 3049/3050 gates can skip cleanly.

**Acceptance criteria:**
- `results/experiment_3048_gatemate_output_contract_operator_package_v1.json`
  is generated with `gatemate_output_contract_ready`,
  `host_visible_io_plan_ready`, `selected_output_signal`, `ccf_binding`,
  `host_reader_command`, `expected_transcript`,
  `missing_operator_actions`, `hardware_execution_claim_made`,
  `speedup_claim_made`, `inference_substrate`, and `honest_verdict`.
- `selected_output_signal` is chosen from the Exp 3034 candidate output
  signals, preferring `done` when present, so later RTL/CCF work has a named
  status output even when the package is blocked.
- `ccf_binding` is populated only from committed GateMate CCF `Pin_out`
  evidence for the selected signal; otherwise it is an empty dict.
- `host_visible_io_plan_ready=true` is forbidden unless both the selected
  signal has a concrete physical CCF binding and the package has a concrete
  host reader command plus expected pass/fail transcript.
- When authority is missing, the package records the missing authoritative
  GateMate A1-EVB-2M output pinout, CCF binding, host reader command, or
  expected transcript without hardware execution, speedup, sampler, Boltzmann,
  or thermodynamic claims.

**Implementation status:** Implemented (Exp 3048)

---

### REQ-HW-089

**Title:** Exp 3063 GateMate no-rerun operator-action ledger MUST keep downstream hardware branches blocked until host-visible output evidence exists

**Description:**
Experiment 3063 MUST produce
`results/experiment_3063_gatemate_no_rerun_operator_action_ledger_v1.json`
as a no-rerun ledger for GateMate hardware work. The ledger MUST load the
checked-in Exp 3048 output-contract operator package, Exp 3049 blocked gate
artifact, Exp 3051 SSQA readback eligibility artifact when present under either
the requested or checked-in bounded filename, the conductor log, and the
hardware wishlist. It MUST inspect only local repository documents and
artifacts. It MUST NOT invent a physical pinout, CCF binding, host reader
command, expected transcript, safety limit, hardware execution result, or
speedup claim.

`gatemate_rerun_allowed=true` is permitted only when the repository contains
all evidence needed to make a selected GateMate output host-visible: an
authoritative CCF `Pin_out` binding for the selected signal, a concrete host
reader command, an expected transcript, and safety limits that allow the
downstream branch to run. Otherwise the ledger MUST set
`gatemate_rerun_allowed=false`, mark every downstream RTL, flash, and SSQA
branch `allowed_to_rerun=false`, and name the exact operator actions still
missing before any rerun can proceed.

**Acceptance criteria:**
- `results/experiment_3063_gatemate_no_rerun_operator_action_ledger_v1.json`
  is generated with `gatemate_no_rerun_ledger_ready`,
  `gatemate_rerun_allowed`, `missing_operator_actions`,
  `required_evidence_before_rerun`, `downstream_tasks_blocked`,
  `hardware_execution_claim_made`, `speedup_claim_made`,
  `source_artifacts`, `inference_substrate`, and `honest_verdict`.
- Missing evidence is machine-readable and includes the authoritative pinout or
  CCF binding, selected output signal, host reader command, expected
  transcript, and safety limits when any are absent.
- Each downstream row records the task id, branch type, upstream blocker,
  `allowed_to_rerun=false` unless every required evidence item is concrete, and
  a matrix-ready status such as `blocked` or `gate_skipped`.
- The artifact cites concrete local source files and declares no model
  inference, no board execution, no flash attempt, and no timing or speedup
  claim.

**Implementation status:** Implemented (Exp 3063)

---

### SCENARIO-HW-089

**Scenario:** GateMate no-rerun ledger blocks RTL, flash, and SSQA rows when the output contract remains missing.

**Given:** Exp 3048 reports `gatemate_output_contract_ready=false`, Exp 3049
is gate-blocked on Exp 3048, and Exp 3051 is gate-blocked because the
GateMate host-visible smoke artifact is absent.
**When:** Exp 3063 builds the operator-action ledger from local repo artifacts
without flashing hardware or running RTL as if the contract exists.
**Then:** It writes the v1 JSON with
`gatemate_no_rerun_ledger_ready=true`,
`gatemate_rerun_allowed=false`, exact missing operator actions, downstream
rows marked `allowed_to_rerun=false`, no hardware or speedup claims, and a
terminal verdict beginning with `complete:`.

**Implementation status:** Implemented (Exp 3063)

---

### REQ-HW-090

**Title:** Exp 3064 SSQA host-visible readback boundary ledger MUST gate readback on committed GateMate smoke transcript evidence

**Description:**
Experiment 3064 MUST produce
`results/experiment_3064_ssqa_host_visible_readback_boundary_ledger_v1.json`
as a matrix-ready SSQA boundary ledger. The ledger MUST load the checked-in
Exp 3051 SSQA readback eligibility artifact when present under either the
requested or bounded filename, the Exp 3053 capstone, the Exp 3063 GateMate
no-rerun ledger, the hardware wishlist, and the changelog. It MUST inspect
only local repository artifacts and MUST NOT attempt hardware readback, flash a
GateMate board, run RTL/PnR, execute live model inference, or infer sampler
behavior from board contact alone.

`ssqa_readback_allowed=true` is permitted only when a committed GateMate
host-visible smoke artifact contains a concrete host-visible transcript and the
minimum predeclared fields needed for SSQA readback are present. Until that
evidence exists, the ledger MUST keep `ssqa_readback_allowed=false`, set
`ssqa_status` and `current_allowed_claim` to the bounded gate-skipped wording
`gated_skipped_host_visible_smoke_missing`, and declare no hardware execution,
speedup, acceleration, Boltzmann sampling, thermodynamic, or sampler-behavior
claim.

**Acceptance criteria:**
- `results/experiment_3064_ssqa_host_visible_readback_boundary_ledger_v1.json`
  is generated with `ssqa_boundary_ledger_ready`,
  `ssqa_readback_allowed`, `ssqa_status`, `required_host_visible_fields`,
  `current_allowed_claim`, `hardware_execution_claim_made`,
  `speedup_claim_made`, `source_artifacts`, `inference_substrate`, and
  `honest_verdict`.
- `required_host_visible_fields` predeclares the transcript, hash, host-reader,
  expected-output, observed-output, selected-output, flash, readback, timing,
  and sampler-configuration evidence required before SSQA can claim readback,
  acceleration, or sampler behavior.
- Missing or non-passing GateMate smoke evidence keeps the SSQA status
  matrix-ready but `gated_skipped_host_visible_smoke_missing`.
- The artifact cites concrete local source files and declares no live model
  inference, no board execution, no hardware readback attempt, no flash
  attempt, and no timing or speedup claim by the boundary ledger itself.

**Implementation status:** Implemented (Exp 3064)

---

### SCENARIO-HW-090

**Scenario:** SSQA readback boundary remains gate-skipped until host-visible smoke evidence exists.

**Given:** Exp 3051 is gate-blocked because Exp 3050 host-visible GateMate
smoke is missing, Exp 3053 records
`ssqa_status=gated_skipped_host_visible_smoke_missing`, and Exp 3063 keeps the
SSQA branch stopped by the GateMate output-contract blocker.
**When:** Exp 3064 builds the SSQA host-visible readback boundary ledger from
local repo artifacts without attempting hardware readback.
**Then:** It writes the v1 JSON with
`ssqa_boundary_ledger_ready=true`, `ssqa_readback_allowed=false`,
`ssqa_status=gated_skipped_host_visible_smoke_missing`,
`current_allowed_claim=gated_skipped_host_visible_smoke_missing`, no hardware
or speedup claims, required transcript fields predeclared, and a terminal
verdict beginning with `complete:`.

**Implementation status:** Implemented (Exp 3064)

---

### REQ-HW-091

**Title:** Exp 3078 GateMate/SSQA no-rerun refresh MUST provide a matrix v21 hardware boundary without executing hardware

**Description:**
Experiment 3078 MUST produce
`results/experiment_3078_gatemate_ssqa_no_rerun_operator_refresh_v1.json`
as the current GateMate/SSQA hardware boundary for matrix v21. The refresh
MUST load the hardware wishlist, the Exp 3048 output-contract operator
package, the Exp 3063 GateMate no-rerun ledger, the Exp 3064 SSQA readback
ledger, and local GateMate/SSQA evidence candidates. It MUST inspect only
checked-in artifacts and already-present operator evidence. It MUST NOT flash a
GateMate board, rerun RTL/PnR as if output evidence exists, execute SSQA
readback, claim board execution, claim sampler behavior, or claim speedup.

`gatemate_rerun_allowed=true` is permitted only when committed evidence
contains an authoritative CCF `Pin_out` binding for the selected GateMate
output, a concrete host reader command, an expected transcript, and safety
limits that open the downstream rerun. `ssqa_readback_allowed=true` is
permitted only when committed host-visible smoke evidence contains the required
transcript, checksum, command, expected/observed transcript, CCF binding,
flash/readback fields, timing fields, and sampler configuration. If either
evidence set is missing, the refresh MUST fail closed, keep the corresponding
allowed field false, cite missing operator actions, and define the exact next
operator-owned hardware task instead of executing it.

**Acceptance criteria:**
- `results/experiment_3078_gatemate_ssqa_no_rerun_operator_refresh_v1.json`
  is generated with `gatemate_ssqa_refresh_ready`,
  `gatemate_rerun_allowed`, `ssqa_readback_allowed`,
  `missing_operator_actions`, `operator_ready_artifacts`,
  `next_allowed_hardware_task`, `hardware_execution_claim_made`,
  `speedup_claim_made`, `source_artifacts`, `inference_substrate`, and
  `honest_verdict`.
- Missing GateMate output-contract evidence keeps
  `gatemate_rerun_allowed=false` and records the missing pinout/CCF binding,
  host reader command, expected transcript, or safety-limit action.
- Missing or non-passing host-visible smoke evidence keeps
  `ssqa_readback_allowed=false` and records the missing smoke transcript fields
  needed before SSQA readback can run.
- When all committed evidence exists, the artifact records the concrete source
  file paths in `operator_ready_artifacts` and sets
  `next_allowed_hardware_task` to the precise next gated hardware action, while
  still declaring no hardware execution or speedup claim by Exp 3078 itself.

**Implementation status:** Pending (Exp 3078)

---

### REQ-HW-092

**Title:** Exp 3092 GateMate/SSQA operator evidence ingestion MUST refresh the no-rerun ledger from checked-in evidence only

**Description:**
Experiment 3092 MUST produce
`results/experiment_3092_gatemate_ssqa_operator_evidence_ingestion_v2.json`
as a refreshed GateMate/SSQA operator-evidence ledger for milestone .288. The
ingestion MUST compare the currently checked-in evidence against the missing
operator actions recorded by Exp 3078. It MUST inspect only existing local
artifacts, command transcripts, safety-limit records, pinout/CCF bindings, and
host-visible smoke logs. It MUST NOT flash a GateMate board, run RTL/PnR,
rerun timing, execute SSQA readback, execute live model inference, claim board
performance, or claim speedup.

`gatemate_rerun_allowed=true` is permitted only when the checked-in evidence
contains an authoritative selected-output CCF binding, a concrete host-reader
command, an expected transcript, and opened safety limits. `ssqa_readback_allowed=true`
is permitted only when the checked-in evidence also contains a passing
host-visible GateMate smoke transcript with checksum, command, expected and
observed transcript, matched transcript status, CCF binding, flash/readback
fields, timing fields, and sampler configuration. If either evidence set is
missing, Exp 3092 MUST fail closed, keep the corresponding allowed field false,
record every checked path, list the exact missing operator evidence, and state
the next allowed operator-owned scope without making a timing or speedup claim.

**Acceptance criteria:**
- `results/experiment_3092_gatemate_ssqa_operator_evidence_ingestion_v2.json`
  is generated with `operator_evidence_ingestion_ready`,
  `gatemate_rerun_allowed`, `ssqa_readback_allowed`,
  `operator_ready_artifacts`, `missing_operator_actions`,
  `speedup_claim_made`, `hardware_commands_run`, `source_artifacts`,
  `inference_substrate`, and `honest_verdict`.
- Missing GateMate output-contract evidence keeps
  `gatemate_rerun_allowed=false` and records the missing pinout/CCF binding,
  host reader command, expected transcript, or safety-limit action from the
  Exp 3078 missing-action list.
- Missing or non-passing host-visible smoke evidence keeps
  `ssqa_readback_allowed=false` and records the missing host-visible smoke
  fields required before SSQA readback can run.
- When all required evidence exists, Exp 3092 records the concrete source paths
  in `operator_ready_artifacts` and records the allowed next experiment scope,
  while still declaring no hardware command, no live model inference, no timing
  claim, and no speedup claim by Exp 3092 itself.

**Implementation status:** Implemented (Exp 3092)

---

### SCENARIO-HW-092

**Scenario:** GateMate/SSQA operator evidence ingestion keeps hardware blocked when the Exp 3078 missing actions are still unresolved.

**Given:** Exp 3078 records missing GateMate pinout/CCF binding, host reader
command, expected transcript, safety-limit opening, and host-visible smoke
evidence.
**When:** Exp 3092 inspects only checked-in artifacts and evidence candidates.
**Then:** It writes the v2 JSON with
`operator_evidence_ingestion_ready=true`, `gatemate_rerun_allowed=false`,
`ssqa_readback_allowed=false`, `hardware_commands_run=[]`,
`speedup_claim_made=false`, explicit checked paths and missing operator
actions, and a terminal verdict beginning with `complete:`.

**Implementation status:** Implemented (Exp 3092)

---

### REQ-HW-093

**Title:** Exp 3106 GateMate/SSQA operator evidence ingestion v3 MUST update the no-rerun/readback boundary from .288 evidence only

**Description:**
Experiment 3106 MUST produce
`results/experiment_3106_gatemate_ssqa_operator_evidence_ingestion_v3.json`
as the current GateMate/SSQA no-rerun/readback boundary for milestone .289.
The ingestion MUST load the .288 capstone hardware ledger, the Exp 3092 v2
operator-evidence ledger, and every checked-in operator-evidence path named or
implied by those ledgers. It MUST inspect only local repository evidence and
MUST NOT flash a GateMate board, run synthesis or PnR, execute readback,
execute live model inference, claim timing, or claim speedup.

`gatemate_rerun_allowed=true` is permitted only when all operator-owned
GateMate rerun evidence is present: authoritative selected-output CCF binding,
concrete host-reader command, expected transcript, and opened safety limits.
`ssqa_readback_allowed=true` is permitted only when the required
host-visible smoke transcript evidence is also present with checksum, command,
expected and observed transcript, matched transcript status, CCF binding,
flash/readback fields, timing fields, and sampler configuration. If either
evidence set is missing, Exp 3106 MUST fail closed, keep the corresponding
allowed field false, record every checked path, list the exact missing evidence,
and state the allowed next operator-owned scope without timing or speedup
claims.

**Acceptance criteria:**
- `results/experiment_3106_gatemate_ssqa_operator_evidence_ingestion_v3.json`
  is generated with `operator_evidence_ingestion_v3_ready`,
  `gatemate_rerun_allowed`, `ssqa_readback_allowed`,
  `operator_ready_artifacts`, `missing_operator_actions`,
  `speedup_claim_made`, `hardware_commands_run`, `source_artifacts`,
  `inference_substrate`, and `honest_verdict`.
- The artifact records checked paths named by the .288 hardware ledger and the
  Exp 3092 source/check-path ledger, including missing paths.
- Missing GateMate output-contract evidence keeps
  `gatemate_rerun_allowed=false` and records the missing pinout/CCF binding,
  host reader command, expected transcript, or safety-limit action.
- Missing or non-passing host-visible smoke evidence keeps
  `ssqa_readback_allowed=false` and records the missing host-visible smoke
  fields required before SSQA readback can run.
- When all required evidence exists, Exp 3106 records concrete evidence paths
  in `operator_ready_artifacts` and records the allowed next experiment scope,
  while still declaring no hardware command, no live model inference, no timing
  claim, and no speedup claim by Exp 3106 itself.

**Implementation status:** Pending (Exp 3106)

---

### SCENARIO-HW-093

**Scenario:** GateMate/SSQA operator evidence ingestion v3 keeps hardware blocked when .288 evidence still lacks operator-visible proof.

**Given:** The .288 capstone and Exp 3092 v2 ledger record missing GateMate
pinout/CCF binding, host reader command, expected transcript, safety-limit
opening, and host-visible smoke evidence.
**When:** Exp 3106 inspects only checked-in local evidence named or implied by
those ledgers.
**Then:** It writes the v3 JSON with
`operator_evidence_ingestion_v3_ready=true`, `gatemate_rerun_allowed=false`,
`ssqa_readback_allowed=false`, `hardware_commands_run=[]`,
`speedup_claim_made=false`, explicit checked paths and missing operator
actions, and a terminal verdict beginning with `complete:`.

**Implementation status:** Pending (Exp 3106)

---

### REQ-HW-094

**Title:** Exp 3119 GateMate/SSQA operator evidence ingestion v4 MUST preserve no-rerun, no-readback, and no-speedup boundaries

**Description:**
Experiment 3119 MUST produce
`results/experiment_3119_gatemate_ssqa_operator_evidence_ingestion_v4.json`
as the current GateMate/SSQA operator-evidence ingestion ledger. The ingestion
MUST read the Exp 3106 v3 evidence artifact and its required operator-action
list, inspect only documented local evidence paths named by that artifact, and
write an explicit v4 status artifact. It MUST NOT flash a GateMate board, run
synthesis or PnR, execute readback, execute live model inference, run hardware
commands, claim timing, or claim speedup.

`gatemate_rerun_allowed=true` is permitted only as an operator-owned future
task recommendation when documented local evidence includes an authoritative
GateMate CCF Pin_out binding, concrete host-reader command, expected
transcript, and opened safety limits. `ssqa_readback_allowed=true` is
permitted only as an operator-owned future task recommendation when the same
documented evidence also includes a host-visible smoke transcript with checksum,
command, expected and observed transcript, matched transcript status, CCF
binding, flash/readback fields, timing fields, sampler configuration, and local
transcript-file traceability. If any evidence is missing or incomplete, Exp
3119 MUST fail closed, keep the corresponding allowed field false, list concrete
missing fields, record evidence files seen, and preserve no-speedup/no-hardware
execution declarations.

**Acceptance criteria:**
- `results/experiment_3119_gatemate_ssqa_operator_evidence_ingestion_v4.json`
  is generated with `operator_evidence_ingestion_v4_ready`,
  `gatemate_rerun_allowed`, `ssqa_readback_allowed`,
  `missing_operator_actions`, `evidence_files_seen`,
  `hardware_commands_run`, `speedup_claim_made`, `source_artifacts`,
  `inference_substrate`, and `honest_verdict`.
- The artifact reads the Exp 3106 v3 ledger and inspects only paths documented
  by v3 `source_artifacts`, `checked_paths`, and `missing_operator_actions`.
- Missing GateMate output-contract evidence keeps
  `gatemate_rerun_allowed=false` and records concrete missing fields for the
  pinout/CCF binding, host reader command, expected transcript, or safety
  limits.
- Missing, incomplete, non-matching, or untraceable host-visible smoke evidence
  keeps `ssqa_readback_allowed=false` and records the missing transcript fields
  required before SSQA readback can be recommended.
- When all required evidence exists, Exp 3119 may set rerun/readback allowed
  only for an operator-owned future task while still declaring no hardware
  command, no live model inference, no readback execution, no timing claim, and
  no speedup claim by Exp 3119 itself.

**Implementation status:** Pending (Exp 3119)

---

### SCENARIO-HW-094

**Scenario:** GateMate/SSQA operator evidence ingestion v4 keeps hardware blocked when v3 evidence still lacks operator-visible proof.

**Given:** Exp 3106 v3 records missing GateMate pinout/CCF binding, host reader
command, expected transcript, safety-limit opening, and host-visible smoke
evidence.
**When:** Exp 3119 inspects only documented local evidence paths from the v3
ledger.
**Then:** It writes the v4 JSON with
`operator_evidence_ingestion_v4_ready=true`, `gatemate_rerun_allowed=false`,
`ssqa_readback_allowed=false`, `hardware_commands_run=[]`,
`speedup_claim_made=false`, explicit evidence files seen and missing operator
actions, and a terminal verdict beginning with `complete:`.

**Implementation status:** Pending (Exp 3119)

---

### REQ-HW-095

**Title:** Exp 3132 hardware evidence and sampler boundary ledger MUST keep all hardware claims bounded to authenticated local artifacts

**Description:**
Experiment 3132 MUST produce
`results/experiment_3132_hardware_evidence_sampler_boundary_v5.json` as the
`.291` hardware evidence and sampler boundary ledger. The ledger MUST read the
current checked-in cLUT sampler artifact, GateMate/SSQA operator-evidence
artifact, hardware status documentation, and available local result/transcript
artifacts. It MUST inspect only existing local files and MUST NOT flash boards,
run synthesis or PnR, execute board commands, perform hardware readback, run
live model inference, or promote speedup claims from unauthenticated evidence.

The ledger MUST record the current evidence state for GateMate, SSQA, KV260,
PolarFire, THRML/TSU, and cLUT separately. GateMate evidence is complete only
when the operator-visible output contract, host-reader command, expected
transcript, safety limits, and host-visible smoke evidence are all complete in
checked-in artifacts. SSQA readback is ready only when host-visible readback
evidence exists. KV260 and PolarFire rows MAY report authenticated historical
hardware evidence when matching local artifact/transcript files are present,
but they MUST keep those rows scoped to their recorded workload and MUST NOT
turn historical timing or dispatch evidence into a fresh sampler speedup claim.
THRML/TSU rows MUST disallow hardware claims unless authenticated TSU hardware
evidence exists locally. cLUT rows MUST remain CPU-only unless authenticated
hardware execution evidence exists.

**Acceptance criteria:**
- `results/experiment_3132_hardware_evidence_sampler_boundary_v5.json` is
  generated with `hardware_evidence_sampler_boundary_v5_ready`,
  `hardware_commands_run`, `gatemate_evidence_complete`,
  `ssqa_readback_ready`, `kv260_evidence_status`,
  `polarfire_evidence_status`, `thrml_tsu_claim_allowed`,
  `clut_sampler_boundary`, `missing_operator_evidence`,
  `speedup_claim_allowed`, `source_artifacts`, `inference_substrate`, and
  `honest_verdict`.
- `hardware_commands_run` is an empty list for Exp 3132 itself unless an
  existing checked-in command transcript is explicitly summarized as historical
  evidence.
- Missing GateMate/SSQA evidence from Exp 3119 remains actionable in
  `missing_operator_evidence`; no GateMate, SSQA, sampler, readback, timing, or
  speedup claim is allowed while those fields remain incomplete.
- KV260 and PolarFire statuses cite local artifact/transcript paths when
  authenticated historical evidence exists, and otherwise report a blocked or
  missing status without probing devices.
- THRML/TSU status disallows TSU hardware claims when only software simulator or
  import evidence exists.
- cLUT status records CPU-only sampler integration from Exp 3118 and preserves
  `hardware_claim_made=false`, empty hardware commands, and no hardware speedup
  promotion.

**Implementation status:** Planned (Exp 3132)

---

### SCENARIO-HW-095

**Scenario:** Hardware evidence and sampler boundary v5 refreshes .291 status without running hardware.

**Given:** Exp 3118 records CPU-only cLUT backend integration and Exp 3119
records incomplete GateMate/SSQA operator evidence.
**When:** Exp 3132 scans checked-in local hardware/status/result artifacts for
transcripts, hashes, pinouts, readbacks, and timing evidence.
**Then:** It writes the v5 JSON with
`hardware_evidence_sampler_boundary_v5_ready=true`,
`hardware_commands_run=[]`, `gatemate_evidence_complete=false`,
`ssqa_readback_ready=false`, per-substrate hardware statuses, CPU-only cLUT
boundary metadata, actionable missing operator evidence, no speedup claim
allowed, local source-artifact provenance, evidence-ingestion
inference-substrate metadata, and a terminal verdict beginning with
`complete:`.

**Implementation status:** Planned (Exp 3132)

---

### REQ-HW-096

**Title:** Exp 3146 hardware sampler evidence boundary v6 MUST refresh .292 hardware status from checked-in evidence only

**Description:**
Experiment 3146 MUST produce
`results/experiment_3146_hardware_sampler_evidence_boundary_v6.json` as the
`.292` hardware and sampler evidence boundary ledger. The ledger MUST ingest
only existing operator-provided or checked-in artifacts, including the Exp 3132
v5 boundary, current cLUT sampler evidence, GateMate/SSQA operator-evidence
ledgers, hardware status documentation, and available local result/transcript
artifacts. It MUST search checked-in artifacts created after Exp 3132 for new
operator-provided transcripts, hashes, pinouts, readbacks, timing artifacts, or
host-visible hardware evidence. It MUST NOT flash boards, synthesize or place
and route designs, execute board commands, perform hardware readback, run live
model inference, or promote speedup claims from unauthenticated evidence.

The ledger MUST record the current evidence state for GateMate, SSQA, KV260,
PolarFire, THRML/TSU, Kona, and cLUT separately. GateMate evidence is complete
only when operator-visible output contract evidence, host-reader command,
expected transcript, safety limits, and host-visible smoke evidence are all
complete in checked-in artifacts. SSQA readback is ready only when host-visible
readback evidence exists. KV260 and PolarFire rows MAY report authenticated
historical hardware evidence when matching local artifact/transcript files are
present, but they MUST keep those rows scoped to their recorded workload and
MUST NOT turn historical timing or dispatch evidence into a fresh sampler
speedup claim. THRML/TSU and Kona rows MUST disallow hardware or access claims
unless authenticated local hardware evidence exists; public architecture pages
or simulator parity artifacts are context only. cLUT rows MUST remain CPU-only
unless authenticated hardware execution evidence exists.

**Acceptance criteria:**
- `results/experiment_3146_hardware_sampler_evidence_boundary_v6.json` is
  generated with `hardware_sampler_evidence_boundary_v6_ready`,
  `hardware_commands_run`, `gatemate_evidence_complete`,
  `ssqa_readback_ready`, `kv260_evidence_status`,
  `polarfire_evidence_status`, `thrml_tsu_claim_allowed`,
  `kona_claim_allowed`, `clut_sampler_boundary`,
  `missing_operator_evidence`, `speedup_claim_allowed`, `source_artifacts`,
  `inference_substrate`, and `honest_verdict`.
- `hardware_commands_run` is an empty list for Exp 3146 itself unless an
  existing checked-in command transcript is explicitly summarized as historical
  evidence.
- The post-Exp 3132 scan records which checked-in artifacts were searched and
  distinguishes direct operator-provided hardware evidence from matrix,
  capstone, or carry-forward references.
- Missing GateMate/SSQA evidence remains actionable in
  `missing_operator_evidence`; no GateMate, SSQA, sampler, readback, timing, or
  speedup claim is allowed while those fields remain incomplete.
- KV260 and PolarFire statuses cite local artifact/transcript paths when
  authenticated historical evidence exists, and otherwise report a blocked or
  missing status without probing devices.
- THRML/TSU and Kona statuses disallow hardware claims when only software
  simulator evidence, public pages, or architecture signals exist.
- cLUT status records CPU-only sampler integration from Exp 3118 and preserves
  `hardware_claim_made=false`, empty hardware commands, and no hardware speedup
  promotion.

**Implementation status:** Planned (Exp 3146)

---

### SCENARIO-HW-096

**Scenario:** Hardware sampler evidence boundary v6 refreshes .292 status without running hardware.

**Given:** Exp 3132 records the v5 hardware boundary, Exp 3118 records CPU-only
cLUT backend integration, and Exp 3119 records incomplete GateMate/SSQA
operator evidence.
**When:** Exp 3146 scans checked-in local hardware/status/result artifacts for
new transcripts, hashes, pinouts, readbacks, timing evidence, and hardware
commands since Exp 3132.
**Then:** It writes the v6 JSON with
`hardware_sampler_evidence_boundary_v6_ready=true`,
`hardware_commands_run=[]`, `gatemate_evidence_complete=false`,
`ssqa_readback_ready=false`, per-substrate hardware statuses, CPU-only cLUT
boundary metadata, THRML/TSU and Kona claims disallowed, actionable missing
operator evidence, no speedup claim allowed, local source-artifact provenance,
evidence-ingestion inference-substrate metadata that declares no live
inference, and a terminal verdict beginning with `complete:`.

**Implementation status:** Planned (Exp 3146)

---

### REQ-HW-097

**Title:** Exp 3160 hardware sampler evidence boundary v7 MUST separate local evidence from public architecture context

**Description:**
Experiment 3160 MUST produce
`results/experiment_3160_hardware_sampler_evidence_boundary_v7.json` as an
evidence-ingestion-only update to the hardware and sampler claim boundary. The
ledger MUST ingest checked-in repository artifacts and documentation for CUDA,
KV260, GateMate, PolarFire, THRML/Extropic, and Kona/Aleph. It MUST separate
operator-supplied local evidence from public vendor/project pages, prior matrix
or capstone carry-forward rows, and hardware wishlist intent. It MUST NOT run
board commands, GPU probes, synthesis, place-and-route, hardware readback, or
live model inference while building the boundary.

Any hardware, GPU, TSU, Kona/Aleph, or sampler speedup claim MUST remain
blocked unless checked-in local evidence records at least a command transcript,
board or device identity, baseline comparator, source artifact checksum,
workload definition, and reproducibility notes. Public vendor/project pages,
architecture blog posts, roadmap/wishlist text, simulator parity, local runtime
readiness, and historical board smoke artifacts MAY be cited as bounded context
but MUST NOT promote a fresh speedup claim. THRML/Extropic and Kona/Aleph MUST
be treated as architecture references only unless authenticated local execution
evidence exists.

**Acceptance criteria:**
- `results/experiment_3160_hardware_sampler_evidence_boundary_v7.json` is
  generated with `hardware_sampler_evidence_boundary_v7_ready`,
  `authenticated_speedup_claim_allowed`, `no_hardware_commands_run`,
  `evidence_sources`, `missing_operator_evidence`, `cuda_status`,
  `kv260_status`, `gatemate_status`, `polarfire_status`,
  `extropic_thrml_status`, `kona_status`, `source_artifacts`,
  `inference_substrate`, and `honest_verdict`.
- `no_hardware_commands_run=true`, `authenticated_speedup_claim_allowed=false`,
  and `inference_substrate` declares no live inference, no hardware execution,
  no board flash, no readback, no synthesis, and no live model execution for Exp
  3160 itself.
- `evidence_sources` classifies source rows as local operator evidence, checked
  in local artifacts, public architecture references, wishlist intent, or ops
  documentation rather than mixing those classes.
- `missing_operator_evidence` records the missing command transcript, device or
  board identity, baseline, artifact checksum, workload, and reproducibility
  notes required before any speedup claim can be promoted.
- CUDA status is explicit: local runtime or GPU-offload readiness is not a
  hardware sampler speedup claim, and adversarially flagged runtime evidence
  cannot become headline evidence.
- KV260 and PolarFire statuses may cite authenticated historical local
  artifacts/transcripts, but must stay scoped to the recorded workloads and keep
  speedup promotion blocked.
- GateMate remains blocked until host-visible smoke/readback evidence and
  operator output-contract fields are present.
- THRML/Extropic and Kona/Aleph remain architecture-reference-only rows unless
  authenticated local TSU, XTR/Z1, Kona, or Aleph execution evidence exists.

**Implementation status:** Planned (Exp 3160)

---

### SCENARIO-HW-097

**Scenario:** Hardware sampler evidence boundary v7 writes a blocked/no-claim ledger without touching hardware.

**Given:** The repository contains the Exp 3146 v6 boundary, CUDA runtime
artifacts, local hardware artifacts for KV260 and PolarFire, GateMate/SSQA
operator-evidence ledgers, THRML simulator parity artifacts, Kona/Aleph public
architecture references, and hardware wishlist/status documents.
**When:** Exp 3160 builds the v7 ledger from checked-in files only.
**Then:** It writes the v7 JSON with
`hardware_sampler_evidence_boundary_v7_ready=true`,
`authenticated_speedup_claim_allowed=false`,
`no_hardware_commands_run=true`, separate status strings for CUDA, KV260,
GateMate, PolarFire, THRML/Extropic, and Kona/Aleph, evidence-source
classification, actionable missing speedup-evidence fields, local
source-artifact provenance, no-live-inference substrate metadata, and a
terminal verdict beginning with `complete:`.

**Implementation status:** Planned (Exp 3160)

---

### REQ-HW-098

**Title:** Exp 3174 hardware/tooling boundary v8 MUST separate public ecosystem context, local import availability, and authenticated performance evidence

**Description:**
Experiment 3174 MUST produce
`results/experiment_3174_hardware_tooling_boundary_v8.json` as the `.294`
hardware and tooling boundary ledger. The ledger MUST refresh CUDA/local GGUF,
KV260, GateMate, PolarFire, THRML/Extropic TSU, XGrammar, llguidance, and
Kona/Aleph status without running board commands, flashing bitstreams, calling
remote hardware, running synthesis or place-and-route, performing hardware
readback, or running live model inference.

The ledger MUST separate public ecosystem references from local Python
import/tool availability and from authenticated local performance evidence.
Local import availability for THRML, XGrammar, or llguidance MAY be checked with
package metadata/import-spec probes only; those checks MUST NOT install
packages and MUST NOT be treated as hardware performance. Public Extropic,
THRML, XGrammar, llguidance, and Kona/Aleph pages MAY be cited as bounded
ecosystem context but MUST NOT promote TSU, Kona, board, local GGUF, sampler, or
speedup claims. Any speedup claim MUST remain blocked unless checked-in
authenticated local evidence records command transcripts, device identity,
baseline comparator, artifact checksum, workload, and reproducibility notes.

**Acceptance criteria:**
- `results/experiment_3174_hardware_tooling_boundary_v8.json` is generated with
  `hardware_tooling_boundary_v8_ready`, `authenticated_speedup_claim_allowed`,
  `hardware_commands_run`, `local_tooling_checks`, `cuda_status`,
  `kv260_status`, `gatemate_status`, `polarfire_status`,
  `extropic_thrml_status`, `kona_status`, `speedup_claim_made`,
  `source_artifacts`, `inference_substrate`, and `honest_verdict`.
- `hardware_commands_run=[]`, `authenticated_speedup_claim_allowed=false`, and
  `speedup_claim_made=false` for Exp 3174 itself.
- `local_tooling_checks` records THRML, XGrammar, and llguidance package/import
  availability separately from hardware performance and marks every check as
  no hardware-performance evidence.
- The ledger partitions evidence into public ecosystem references, local
  tooling checks, and authenticated performance evidence instead of mixing
  those classes.
- CUDA/local GGUF readiness remains a runtime/tooling status and not a sampler
  speedup claim.
- KV260, GateMate, and PolarFire statuses require checked-in command
  transcripts for any board/workload claim and do not run fresh hardware
  commands.
- THRML/Extropic TSU and Kona/Aleph remain public ecosystem or architecture
  context unless authenticated local execution evidence exists.

**Implementation status:** Planned (Exp 3174)

---

### SCENARIO-HW-098

**Scenario:** Hardware/tooling boundary v8 records ecosystem and import status without promoting speedups.

**Given:** The repository contains the Exp 3160 and Exp 3146 boundary ledgers,
hardware wishlist/status documents, research references, and any checked-in
historical KV260, GateMate, or PolarFire transcripts.
**When:** Exp 3174 builds the v8 ledger from checked-in files and optional local
Python import metadata probes only.
**Then:** It writes the v8 JSON with `hardware_tooling_boundary_v8_ready=true`,
`authenticated_speedup_claim_allowed=false`, `hardware_commands_run=[]`,
`speedup_claim_made=false`, separate status strings for CUDA, KV260, GateMate,
PolarFire, THRML/Extropic, and Kona/Aleph, `local_tooling_checks` for THRML,
XGrammar, and llguidance, source-artifact/page provenance, no-live-inference
substrate metadata, and a terminal verdict beginning with `complete:`.

**Implementation status:** Planned (Exp 3174)

---

### REQ-HW-099

**Title:** Exp 3188 THRML factor-graph API boundary v1 MUST translate tiny exact rows without hardware claims

**Description:**
Experiment 3188 MUST produce
`results/experiment_3188_thrml_factor_graph_api_boundary_v1.json` as a local
API-boundary artifact for translating tiny exact-row constraints into
THRML-compatible factor-graph structures where the installed THRML package
exposes suitable software APIs. The experiment MUST check local `thrml` import
availability, version, and import path, select one or two deterministic
exact-authority rows from checked-in exact-row artifacts, build a simple
internal factor-graph representation, and attempt a construction-only mapping
to THRML node/block/factor concepts.

The experiment MUST NOT run board commands, use retired KV260 host-storage
checks, install packages, run synthesis/place-and-route, execute TSU/Kona
hardware, call live models, or report sampler latency/speedup. Any THRML import
or API mapping failure MUST materialize as a blocked/preflight artifact with
missing symbols and next adapter steps rather than a fabricated success.

**Acceptance criteria:**
- `results/experiment_3188_thrml_factor_graph_api_boundary_v1.json` is
  generated with `thrml_factor_graph_api_boundary_v1_ready`,
  `thrml_import_available`, `thrml_version`, `selected_exact_rows`,
  `factor_graph_translation_records`, `api_gap_records`,
  `local_api_smoke_passed`, `hardware_speedup_claim_allowed`,
  `kona_or_tsu_execution_claimed`, `inference_substrate`, and
  `honest_verdict`.
- `selected_exact_rows` contains deterministic exact-authority row metadata
  from checked-in exact-row artifacts and records the row id, exact label,
  candidate answers, known false-accept status, and source artifact.
- `factor_graph_translation_records` exposes both the internal factor-graph
  variables/factors and the attempted THRML API mapping for each selected row.
- THRML import availability records the local package version and import path
  when import succeeds, and records an import error when it does not.
- `local_api_smoke_passed` is true only for construction-only local API checks
  that instantiate suitable THRML software structures; it MUST NOT represent a
  sampler benchmark or hardware run.
- `hardware_speedup_claim_allowed=false` and
  `kona_or_tsu_execution_claimed=false` for every outcome.
- `inference_substrate` records no hardware commands, no live model inference,
  no package installation, no TSU/Kona execution, and no sampler speedup claim.

**Implementation status:** Planned (Exp 3188)

---

### REQ-HW-100

**Title:** Exp 3202 sparse Potts/PAOA/THRML factor boundary MUST materialize q-state factor records without hardware claims

**Description:**
Experiment 3202 MUST produce
`results/experiment_3202_sparse_potts_paoa_thrml_factor_boundary_v1.json` as
a representation-only hardware-boundary artifact. The experiment MUST load the
checked-in Exp 3188 THRML factor-graph API boundary and Exp 3197 inductive
certificate invariant artifact, derive sparse q-state Potts factor records
from exact rows and invariant certificates, attach PAOA-ready coupling/update
metadata, estimate graph density and sparse-vs-dense representation deltas, and
optionally report whether local THRML API construction evidence is already
available.

The experiment MUST NOT execute KV260, GateMate, PolarFire, TSU, Z1, XTR-0,
Kona, or any other hardware. It MUST NOT run synthesis, place-and-route, board
commands, sampler benchmarks, live model calls, package installation, or
speedup measurement. It MUST explicitly deny KV260, GateMate, PolarFire, TSU,
Z1, XTR-0, Kona, and speedup claims unless authenticated hardware transcripts
are present in checked-in source artifacts.

**Acceptance criteria:**
- `results/experiment_3202_sparse_potts_paoa_thrml_factor_boundary_v1.json`
  is generated with `schema_version`, `experiment_id`, `source_artifacts`,
  `factor_record_schema`, `factor_record_count`, `q_state_count_summary`,
  `graph_density_summary`, `paoa_metadata_schema`,
  `thrml_local_api_checked`, `authenticated_hardware_transcript_present`,
  `speedup_claim_allowed`, `hardware_claims_denied`, and `honest_verdict`.
- `factor_record_schema` documents q-state Potts factor records with stable
  row/invariant lineage, categorical variables, sparse edge/factor scopes,
  energy tables, and PAOA metadata attachment points.
- `factor_record_count` equals the number of materialized sparse factor
  records, and each record derives from an exact row or invariant certificate.
- `q_state_count_summary` reports per-record and aggregate q-state accounting
  without converting categorical states into binary-only claims.
- `graph_density_summary` reports node counts, possible dense factor/edge
  counts, actual sparse factor/edge counts, density ratios, and sparse-vs-dense
  storage deltas.
- `paoa_metadata_schema` documents coupling tensor shape, update order, beta or
  annealing schedule metadata, constraint family, and boundary-only sampler
  fields needed for a future PAOA adapter.
- `thrml_local_api_checked` is a transparent local API evidence flag only; it
  MUST NOT imply TSU, Kona, hardware execution, latency, or speedup.
- `authenticated_hardware_transcript_present=false`,
  `speedup_claim_allowed=false`, and `hardware_claims_denied` lists at least
  KV260, GateMate, PolarFire, TSU, Z1, XTR-0, Kona, and speedup claims when no
  authenticated hardware transcript is present.
- `honest_verdict` begins with `complete:` when the representation artifact is
  materialized from local artifacts, or `blocked_` when required local sources
  are missing or malformed.

**Implementation status:** Planned (Exp 3202)

---

### SCENARIO-HW-100

**Scenario:** Sparse Potts/PAOA factor records prepare exact certificates without reviving THRML or hardware speedup claims.

**Given:** Exp 3188 exposes local THRML factor-graph boundary rows, Exp 3197
exposes invariant certificate records, and no authenticated hardware transcript
is present.
**When:** Exp 3202 builds sparse q-state Potts factor records and PAOA metadata
from those checked-in artifacts only.
**Then:** It writes the schema-versioned v1 JSON artifact with materialized
factor records, q-state accounting, graph density and sparse-vs-dense deltas,
PAOA-ready metadata schema, local THRML transparency, explicit denial of KV260,
GateMate, PolarFire, TSU, Z1, XTR-0, Kona, and speedup claims, no hardware
commands, and a terminal verdict beginning with `complete:`.

**Implementation status:** Planned (Exp 3202)

---

### SCENARIO-HW-099

**Scenario:** THRML factor-graph boundary maps exact rows only as local API construction evidence.

**Given:** Checked-in exact-row artifacts expose deterministic exact labels and
known false-accept rows, and the active Python environment may or may not
import `thrml`.
**When:** Exp 3188 builds the THRML factor-graph API boundary from local
artifacts and import metadata only.
**Then:** It writes the v1 JSON artifact with selected exact rows, internal
factor graphs, THRML construction records or actionable API gaps, a
construction-only smoke result separated from performance, denied hardware
speedup and TSU/Kona execution claims, no board commands, and a terminal
verdict beginning with `complete:` or an honest blocked-precondition prefix.

**Implementation status:** Planned (Exp 3188)

---

### SCENARIO-HW-091

**Scenario:** Matrix v21 GateMate/SSQA refresh carries forward no-rerun blockers when operator evidence is absent.

**Given:** Exp 3048 reports `gatemate_output_contract_ready=false`, Exp 3063
reports `gatemate_rerun_allowed=false`, Exp 3064 reports
`ssqa_readback_allowed=false`, and no committed Exp 3050 host-visible smoke
artifact is present.
**When:** Exp 3078 builds the matrix v21 GateMate/SSQA refresh from local
artifacts only.
**Then:** It writes the v1 JSON with
`gatemate_ssqa_refresh_ready=true`, `gatemate_rerun_allowed=false`,
`ssqa_readback_allowed=false`, explicit missing operator actions, empty
hardware execution and speedup claims, and a terminal verdict beginning with
`complete:`.

**Implementation status:** Pending (Exp 3078)

---

### SCENARIO-HW-088

**Scenario:** GateMate operator package blocks cleanly when Exp 3034 still lacks pinout and reader authority.

**Given:** Exp 3034 reports `done` and `spin_out[15:0]` as candidate outputs
but blocks because the current GateMate CCF is build-only and no concrete host
reader command exists.
**When:** Exp 3048 inspects local docs, committed GateMate files, and tool
availability without flashing the board.
**Then:** It writes the v1 operator package with
`selected_output_signal="done"`,
`gatemate_output_contract_ready=false`,
`host_visible_io_plan_ready=false`, empty `ccf_binding`, empty
`host_reader_command`, empty `expected_transcript`, explicit missing operator
actions, no hardware or speedup claims, and a terminal verdict beginning with
`complete:`.

**Implementation status:** Implemented (Exp 3048)

---

### REQ-HW-101

**Title:** Exp 3346 KV260 MMD-vs-CPU continuity rerun MUST use the live SSH board path and never the retired host SD-card check

**Description:**
Experiment 3346 is the deferred continuity rerun of the Exp 2938 MMD-vs-CPU
sequential-Gibbs comparison, executed against the *current* KV260 board access
path. It MUST establish board reachability with
`ssh -o ConnectTimeout=5 -o BatchMode=yes kria true` as its sole hardware
precondition and MUST NOT reference any host block device (for example
`/dev/mmcblk*`) — the host SD-card branch is permanently retired per the
CLAUDE.md "KV260 SSH-Not-SD-Card Discipline". When the board is reachable it
MUST collect board-local evidence (`uname`, `xmutil` overlay status, and the
visible `/dev/uio*` device list), reproduce the same Exp 2898 dense n=64 Ising
problems for seeds 42, 137, and 271, exercise the existing KV260 synchronous
Glauber sampler over UIO without modifying the bitstream, and run a matched CPU
sequential-Gibbs baseline on the same problem distribution. It MUST compute the
repo's established energy-distribution distance (RBF MMD² plus the
Kolmogorov-Smirnov statistic) per seed and summarise whether the board and CPU
energy distributions are distinguishable. If any board step cannot execute, it
MUST preserve the command transcript and the precise blocked reason rather than
fabricate a result. The continuity smoke MAY use a reduced sample count (the
"smallest available sampler path"); the count actually used MUST be recorded.

**Acceptance criteria:**
- `results/experiment_3346_kv260_mmd_vs_cpu_sequential_gibbs_v1.json` is written
  with `inference_substrate="hardware_smoke"` and `honest_verdict` beginning with
  a terminal prefix (`complete:`/`success:`) on a successful board run or with a
  `blocked_<resource>` prefix when a board step fails.
- The artifact includes `honest_verdict`, `inference_substrate`, `random_seed`,
  `reproducibility_checksum`, `duration_s`, `files_updated`, and the hardware
  evidence fields `ssh_reachable`, `board_uname`, `xmutil_status`, `uio_status`,
  `cpu_baseline_summary`, `kv260_summary`, `mmd_vs_cpu`, `sample_count_cpu`,
  `sample_count_kv260`, `command_transcript`, and `blocked_reasons`.
- The hardware precondition is SSH reachability; no host block-device check
  appears anywhere in the experiment.
- On a successful run `mmd_vs_cpu` records exactly three per-seed MMD² and KS
  results and `sample_count_cpu == sample_count_kv260`.
- A blocked run leaves `mmd_vs_cpu` empty, sets `blocked_reasons` to a non-empty
  list, and keeps the command transcript for audit.

**Implementation status:** Implemented (Exp 3346)

---

### SCENARIO-HW-101

**Scenario:** KV260 continuity rerun compares board synchronous Glauber energies to a matched CPU sequential-Gibbs baseline over the live SSH path.

**Given:** `ssh kria` is reachable, Exp 2898 is present and its three dense Ising
problems reproduce from seeds 42, 137, and 271, and the KV260 sampler overlay is
loaded with a readable UIO device.
**When:** Exp 3346 collects board evidence, runs the KV260 synchronous Glauber
sampler and a matched CPU sequential-Gibbs baseline at the continuity sample
count, and compares the two energy distributions.
**Then:** It writes the v1 artifact with the board evidence fields populated, the
three per-seed MMD²/KS results in `mmd_vs_cpu`, equal CPU and KV260 sample
counts, an empty `blocked_reasons`, and a terminal `honest_verdict`. If SSH or
any board step fails, it instead records the blocked reason and the transcript
and emits a `blocked_<resource>` verdict.

**Implementation status:** Implemented (Exp 3346)

---

### REQ-HW-102

**Title:** Exp 3347 GateMate n=16 Ising tile build + detect smoke v2 MUST use the himbaechel flow and never fabricate a host-visible readback

**Description:**
Experiment 3347 resumes the operator-deferred GateMate A1-EVB-2M n=16 Ising
tile build/detect smoke. It MUST build the `ising_n16_gatemate` tile through the
current open-source GateMate flow — yosys `synth_gatemate`, then
`nextpnr-himbaechel --device CCGM1A1`, then `gmpack` — reusing the deterministic
Exp 2955 constraints package and the established Exp 2956 build module. The
experiment MUST NOT invoke the obsolete `nextpnr-gatemate` binary, which does
not exist in the current OSS CAD Suite.

The DirtyJTAG board detect (`openFPGALoader -c dirtyJtag --detect`) gates only
the optional flash/detect smoke, not the build itself: a build tool that is
missing blocks the build, while an unreachable board only skips the smoke. The
experiment MUST NOT program the board's SRAM with the build-only,
pin-unconstrained bitstream, and MUST NOT claim any host-visible readback that
the captured transcript does not prove. The DirtyJTAG IDCODE returned by
`--detect` is the only board-side readback this experiment may claim. If a build
tool is missing or any build stage fails, the experiment MUST write an honest
blocked artifact with the command transcript and the precise blocked reason
rather than fabricate a passing result.

**Acceptance criteria:**
- `results/experiment_3347_gatemate_n16_ising_tile_bitstream_build_smoke_v2.json`
  is written with `inference_substrate="hardware_build"` and an `honest_verdict`
  beginning with a terminal prefix (`complete:`) on a successful build or a
  `blocked_` prefix when a build precondition fails.
- The artifact includes `honest_verdict`, `inference_substrate`, `random_seed`,
  `reproducibility_checksum`, `duration_s`, `files_updated`, and the hardware
  evidence fields `yosys_version`, `nextpnr_himbaechel_version`,
  `gmpack_version`, `dirtyjtag_detected`, `build_succeeded`, `bitstream_path`,
  `bitstream_checksum`, `resource_summary`, `flash_smoke_status`,
  `command_transcript`, and `blocked_reasons`.
- The place-and-route command references `nextpnr-himbaechel --device CCGM1A1`
  and never `nextpnr-gatemate`.
- `flash_smoke_status` reflects detect-only board evidence (IDCODE readback)
  when the board is detected and a skipped status when it is not; it never
  records an SRAM programming or host-visible readback claim.
- A blocked build leaves `build_succeeded=false`, sets `blocked_reasons` to a
  non-empty list, and preserves the command transcript for audit.

**Implementation status:** Implemented (Exp 3347)

---

### SCENARIO-HW-102

**Scenario:** GateMate n=16 Ising tile builds through the himbaechel flow and records a detect-only board smoke without a fabricated readback.

**Given:** The Exp 2955 constraints package is ready, the OSS CAD Suite provides
yosys, `nextpnr-himbaechel`, `gmpack`, and `openFPGALoader`, and the DirtyJTAG
adapter may or may not expose the GateMate IDCODE.
**When:** Exp 3347 builds the `ising_n16_gatemate` tile through the
synth_gatemate → nextpnr-himbaechel → gmpack flow and then runs the smallest
board detect.
**Then:** It writes the v2 artifact with the toolchain version fields, a hashed
bitstream, the synthesis/pnr/pack command transcript, a detect-only
`flash_smoke_status` when the board IDCODE is read (or a skipped status when it
is not), an empty `blocked_reasons`, and a terminal `honest_verdict` beginning
with `complete:`. If a build tool is missing or any build stage fails, it
instead records the blocked reason and the transcript and emits a `blocked_`
verdict.

**Implementation status:** Implemented (Exp 3347)

---

### REQ-HW-103

**Title:** Exp 3351 GateMate n=16 Ising tile latency benchmark MUST record a blocked artifact if hardware is unreachable or lacks a host communication interface

**Description:**
Experiment 3351 MUST measure the live GateMate n=16 Ising tile sampling latency.
However, because the current GateMate RTL lacks a host communication interface (e.g. AXI, UIO, or dedicated USB/JTAG endpoints), the experiment MUST record a blocked artifact and preserve physical-board evidence without attempting to fabricate samples.
The artifact MUST include the fields `honest_verdict`, `gatemate_latency_us`, `speedup_vs_cpu`, and `blocked_reasons`. If the board lacks a communication interface, the verdict MUST reflect this state (e.g. `blocked_no_io_interface_in_rtl`) and both latency and speedup MUST be `null`.

**Acceptance criteria:**
- `results/experiment_3351_gatemate_latency_benchmark.json` is written with `inference_substrate="hardware_smoke"` and `honest_verdict` beginning with a terminal prefix (`blocked_` if no IO).
- The artifact includes `honest_verdict`, `gatemate_latency_us`, `speedup_vs_cpu`, `blocked_reasons`, and `duration_s`.
- Both latency and speedup are `null` if the benchmark cannot be executed.
- `blocked_reasons` contains a clear description of the blocker.

**Implementation status:** Implemented (Exp 3351)

---

### SCENARIO-HW-103

**Scenario:** GateMate n=16 latency benchmark blocks on missing communication interface.

**Given:** The GateMate A1-EVB-2M n=16 Ising tile RTL exists but has no host-visible read/write data interface over USB/JTAG.
**When:** Experiment 3351 executes to measure real sampling latency.
**Then:** It correctly identifies the missing IO constraint, skips polling, writes the artifact with `gatemate_latency_us=null`, and assigns the verdict `blocked_no_io_interface_in_rtl`.

**Implementation status:** Implemented (Exp 3351)

---

### REQ-HW-104

**Title:** KV260 SSH recovery and connectivity diagnosis

**Description:**
Experiment 3365 MUST diagnose and attempt to restore network/SSH connectivity to the KV260 board, which was unreachable in milestone 310 (exp3350). The script MUST check local network routes and the ARP cache, attempt a serial console connection if SSH fails, and verify simple command execution upon restoring SSH access. The diagnostic findings MUST be emitted in a JSON artifact.

**Acceptance criteria:**
- `results/experiment_3365_kv260_ssh_recovery.json` is generated with `inference_substrate="hardware_smoke"`.
- The artifact includes `honest_verdict`, `ssh_reachable`, `routes_checked`, `arp_cache_checked`, `serial_connection_attempted`, `connectivity_restored`, `command_execution_verified`, and `duration_s`.
- The script checks local routing and ARP.
- `honest_verdict` reflects the final state (e.g., `complete: ssh_restored` or `blocked: ssh_and_serial_failed`).

**Implementation status:** Pending (Exp 3365)

---

### SCENARIO-HW-104

**Scenario:** KV260 SSH connectivity is diagnosed and an artifact is written.

**Given:** The KV260 board may or may not be reachable via SSH or serial console.
**When:** Experiment 3365 executes.
**Then:** It checks the routing and ARP, attempts SSH, falls back to serial if needed, and writes `results/experiment_3365_kv260_ssh_recovery.json` detailing the connectivity status.

**Implementation status:** Pending (Exp 3365)

### REQ-HW-105

**Title:** GateMate n=16 RTL must expose a basic AXI or UIO compatible interface

**Description:**
Experiment 3366 MUST add a memory-mapped interface (such as AXI4-Lite) to the GateMate n=16 Ising tile RTL.
The previous experiment (Exp 3351) was blocked because the RTL lacked a host communication interface.
The AXI4-Lite interface must allow reading and writing the `h` vector and `spins` vector via memory-mapped registers.
The artifact MUST include the synthesis status of this updated RTL.

**Acceptance criteria:**
- `rtl/gatemate_ising_n16.v` contains an AXI4-Lite interface.
- Python mock driver or tests verify the synthesis and interface behavior.
- `results/experiment_3366_gatemate_axi_uio.json` is generated containing the synthesis result.

**Implementation status:** Pending (Exp 3366)

---

### SCENARIO-HW-105

**Scenario:** GateMate n=16 RTL synthesizes successfully with AXI4-Lite interface.

**Given:** The `rtl/gatemate_ising_n16.v` contains an AXI4-Lite wrapper for the Ising core.
**When:** Experiment 3366 executes.
**Then:** It performs yosys synthesis and writes the artifact with the synthesis success status and resource usage.

**Implementation status:** Pending (Exp 3366)


### REQ-HW-106

**Title:** Synthesize, flash, and run a simple test on GateMate board via dirtyJtag

**Description:**
Experiment 3382 MUST use yosys and nextpnr-himbaechel to build the N=16 GateMate design,
flash the resulting bitstream using openFPGALoader -c dirtyJtag, and attempt to
execute a basic test vector across the new host interface to verify hardware connectivity.

**Acceptance criteria:**
- Script scripts/experiment_3382_gatemate_n16_smoke.py attempts synthesis, routing, and flashing.
- Script writes results/experiment_3382_gatemate_n16_smoke.json with hardware verification status.
- Test tests/python/test_experiment_3382_gatemate_n16_smoke.py covers the script.

**Implementation status:** Pending (Exp 3382)

---

### SCENARIO-HW-106

**Scenario:** GateMate n=16 design is flashed and basic test executed.

**Given:** An experiment script that drives yosys, nextpnr-himbaechel, and openFPGALoader.
**When:** The script is executed.
**Then:** It logs the build/flash/test attempts and emits the artifact JSON with the final success or blocked status.

**Implementation status:** Pending (Exp 3382)

---

### REQ-HW-107

**Title:** Root-cause why the GateMate N=16 bootstrap artifact resolves `unspecified`

**Description:**
The GateMate N=16 bootstrap flash (Exps 3382, 3392, 3404) landed an artifact whose
reconciled `honest_verdict` was `unspecified` for three consecutive milestones. Re-running
the identical flash is a doomed rerun (Failed-Experiment Rerun Discipline) and GateMate is
opportunistic per north-star §3, so the honest move is a ROOT-CAUSE DIAGNOSTIC rather than a
fourth flash attempt. Experiment 3421 MUST run three diagnostic checks — (a) toolchain
presence (`yosys`, `nextpnr-himbaechel`, `openFPGALoader`), (b) GateMate IDCODE detection via
`openFPGALoader -c dirtyJtag --detect`, and (c) a static scan of the Exp 3404 script for any
code path that assigns `honest_verdict` — and classify the actionable root cause. The
diagnostic MUST NOT attempt synthesis, place-and-route, or any flash command.

**Acceptance criteria:**
- `results/experiment_3421_gatemate_bootstrap_rootcause_diagnostic_v1.json` is generated with
  `inference_substrate="hardware_smoke"`.
- The artifact includes `honest_verdict`, `preconditions_checked`, `rootcause_classification`,
  `recommended_fix`, and `duration_s`.
- `rootcause_classification` is one of `toolchain_missing`, `board_unreachable`, or
  `script_never_sets_verdict`.
- `honest_verdict` starts with a terminal prefix (`complete:` / `success:` / `passed:` /
  `shipped_`) when the diagnostic completes, or a `blocked_gatemate_*` prefix only if the
  diagnostic itself could not run.
- No flash command is attempted (`no_flash_attempted=true`).

**Implementation status:** Pending (Exp 3421)

---

### SCENARIO-HW-107

**Scenario:** The bootstrap artifact's `unspecified` verdict is diagnosed to the producing script.

**Given:** The Exp 3404 script assigns `status` but never assigns `honest_verdict`, and
`build_result()` does not auto-populate `honest_verdict`.
**When:** Experiment 3421 runs the three diagnostic checks.
**Then:** It classifies `rootcause_classification="script_never_sets_verdict"`, records each
precondition check result in `preconditions_checked`, recommends adding an explicit
`honest_verdict` to the producing script's `build_result()` calls, and emits a terminal
`honest_verdict` without attempting any flash.

**Implementation status:** Pending (Exp 3421)

---

### REQ-HW-108

**Title:** Apply the GateMate bootstrap verdict fix and attempt one flash

**Description:**
Exp 3421 (REQ-HW-107) root-caused the recurring `unspecified` reconciled verdict: the
GateMate bootstrap script never assigns `honest_verdict`, and `build_result()` does not
auto-populate it, so the reconciler reads `unspecified`. Experiment 3432 MUST apply that fix
by guaranteeing that EVERY artifact-emission path assigns a terminal `honest_verdict` (one
starting `complete:` / `success:`, or a `complete: blocked_gatemate_*` prefix when a
precondition is hard-missing) — `unspecified` MUST NOT be emittable. Because GateMate is
opportunistic per north-star §3 (it does NOT block milestones), the experiment attempts the
N=16 Ising tile flash AT MOST ONCE — it is NOT a fourth identical re-flash. The toolchain
(`yosys`, `nextpnr-himbaechel`, `openFPGALoader`, `gmpack`) lives under the oss-cad-suite
install, which MUST be resolved onto PATH before the precondition checks.

**Acceptance criteria:**
- `scripts/experiment_3432_gatemate_apply_verdict_fix_and_flash_v1.py` resolves the oss-cad-suite
  toolchain onto PATH, runs the PRECONDITIONS (toolchain presence + GateMate IDCODE detect), and
  on pass attempts the synth -> place-and-route -> pack -> flash flow exactly once.
- The verdict classifier returns a CONCRETE terminal verdict for every (toolchain, board, flash)
  outcome — never `unspecified`.
- `results/experiment_3432_gatemate_apply_verdict_fix_and_flash_v1.json` is emitted with
  `inference_substrate="hardware_smoke"`, `honest_verdict` (terminal prefix or
  `complete: blocked_gatemate_*`), `preconditions_checked`, `verdict_fix_applied`,
  `gatemate_bitstream_flashed`, and `duration_s`.
- Test `tests/python/test_experiment_3432_gatemate_apply_verdict_fix_and_flash_v1.py` covers the
  classifier, the PATH-resolution helper, and the precondition builder.

**Implementation status:** Implemented (Exp 3432)

---

### SCENARIO-HW-108

**Scenario:** The bootstrap verdict fix is applied and a single flash is attempted.

**Given:** The producing script previously emitted no `honest_verdict`, so the reconciler read
`unspecified`.
**When:** Experiment 3432 resolves the toolchain, checks preconditions, and (if they pass)
attempts the N=16 flash once.
**Then:** It emits the artifact with a concrete terminal `honest_verdict` — `success: ...flashed`
when the bitstream reaches the board, `complete: ...not_flashed` when the flow runs but does not
flash, or `complete: blocked_gatemate_toolchain_missing` / `complete: blocked_gatemate_board_unreachable`
when a precondition is hard-missing — and never `unspecified`.

**Implementation status:** Implemented (Exp 3432)

---

### REQ-HW-109

**Title:** Exp 3866 GateMate n=16 Ising tile terminal flash v2 MUST precondition-gate hardware and preserve real resource evidence

**Description:**
Experiment 3866 MUST drive the GateMate A1-EVB-2M n=16 Carnot Ising tile toward
terminal state using the current open-source GateMate flow:
`yosys synth_gatemate`, `nextpnr-himbaechel --device CCGM1A1`, `gmpack`, and
`openFPGALoader -c dirtyJtag -b olimex_gatemateevb`. Before any synthesis or
fabrication step, it MUST check the exact toolchain and board resources:
`yosys -V` with `synth_gatemate` support, `nextpnr-himbaechel --help`,
`gmpack`, `openFPGALoader`, and `openFPGALoader -c dirtyJtag --detect` proving
the Cologne Chip GateMate/GM1Ax IDCODE. If any precondition fails, the
experiment MUST write a blocked artifact and MUST NOT run synthesis, P&R,
packing, or flashing.

When the preconditions pass, the experiment MUST synthesize, place-and-route,
pack, and attempt exactly one real board flash. The artifact MUST record real
LUT/DFF utilization and Fmax parsed from the actual tool transcripts. If the
bitstream flashes but the current RTL/tooling exposes no host-visible sample
readback path, `sample_timing_us` MUST be null with an explanatory note rather
than a fabricated timing value. The terminal gate is the real board accepting
the bitstream: `gatemate_bitstream_flashed=true`.

**Acceptance criteria:**
- `scripts/experiments/experiment_3866_gatemate_ising_tile_flash_v2.py` writes
  `results/experiment_3866_gatemate_ising_tile_flash_v2.json`.
- The artifact includes `honest_verdict`, `gatemate_bitstream_flashed`,
  `synth_pnr_pack_succeeded`, `lut_dff_utilization`, `fmax_mhz`,
  `sample_timing_us`, `sample_timing_note`, `preconditions_checked`,
  `thermal_note`, `run_duration_s`, `duration_s`, `inference_substrate`, and
  `reproducibility_checksum`.
- The required artifact fields have principle annotations in
  `field_provenance`, while `gatemate_bitstream_flashed` and
  `synth_pnr_pack_succeeded` remain bare booleans.
- A missing precondition emits an `honest_verdict` beginning with
  `blocked_gatemate_<resource>` and exits before synthesis.
- A successful synth/P&R/pack but failed flash emits an `honest_verdict`
  beginning with `complete: gatemate_synth_pnr_pack_succeeded_flash_pending`.
- A board-accepted bitstream emits an `honest_verdict` beginning with
  `success: gatemate_ising_tile_n16_flashed_terminal`.

**Implementation status:** Implemented (Exp 3866)

---

### SCENARIO-HW-109

**Scenario:** GateMate n=16 Ising tile flash v2 records terminal, partial, or blocked state without fabrication.

**Given:** The GateMate toolchain may or may not be installed and the
Olimex GateMate A1-EVB-2M may or may not be detected over onboard DirtyJTAG.
**When:** Experiment 3866 runs its preconditions and, only on pass, runs the
synth_gatemate -> nextpnr-himbaechel -> gmpack -> openFPGALoader flow.
**Then:** It writes the v2 artifact with precondition evidence, real
utilization/Fmax values when the flow reaches P&R, a real flash result, and
null sample timing with a note whenever no host-visible timing/readback path is
available.

**Implementation status:** Implemented (Exp 3866)

---

### REQ-HW-3867

**Title:** PolarFire SoC Ising CPU dispatch v4 MUST be SSH-precondition-gated and hash-verified

**Description:**
Experiment 3867 MUST dispatch a deterministic Carnot Ising energy-evaluation
workload to the Microchip PolarFire SoC Discovery Kit over SSH and verify that
the board reproduces the exact CPU reference result. The experiment MUST first
check `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'` and then
`ssh polarfire 'python3 --version'`. If either precondition fails, it MUST write
a blocked artifact (`blocked_polarfire_ssh_timeout` or
`blocked_polarfire_no_python`) and MUST NOT SCP or run the workload.

When preconditions pass, the experiment MUST build a seed-controlled fixed Ising
coupling matrix and fixed spin configurations, compute the CPU reference
energies locally, SCP the same workload and a self-contained Python runner to
the board, run it on the PolarFire Linux riscv64 CPU substrate, capture stdout,
and compare the board result SHA256 to the CPU reference SHA256. The terminal
gate requires `polarfire_workload_validated=true`, `result_hash_match=true`,
and a real SSH round-trip `run_duration_s >= 5`.

**Acceptance criteria:**
- `scripts/experiments/experiment_3867_polarfire_soc_smoke_v4.py` writes
  `results/experiment_3867_polarfire_soc_smoke_v4.json`.
- The artifact includes `polarfire_workload_validated`, `result_hash_match`,
  `board_result_sha256`, `cpu_reference_sha256`, `run_duration_s`,
  `soc_temp_max_c`, `thermal_note`, `preconditions_checked`, `random_seed`,
  `reproducibility_checksum`, `inference_substrate`, and `honest_verdict`.
- Each required artifact field has a `principle` note in `field_provenance`,
  while `polarfire_workload_validated` and `result_hash_match` remain bare
  booleans.
- A failed SSH precondition emits `blocked_polarfire_ssh_timeout` and exits
  before Python probing, SCP, or workload execution.
- A missing board Python precondition emits `blocked_polarfire_no_python` and
  exits before SCP or workload execution.
- A hash match with `run_duration_s >= 5` emits an `honest_verdict` beginning
  with `success: polarfire_carnot_dispatch_hash_verified_terminal`.
- A hash mismatch emits an `honest_verdict` beginning with
  `complete: polarfire_dispatch_ran_hash_MISMATCH` and does not validate the
  workload.

**Implementation status:** Pending (Exp 3867)

---

### SCENARIO-HW-3867

**Scenario:** PolarFire runs a deterministic Ising energy workload and matches the CPU hash.

**Given:** `ssh polarfire` is reachable and `python3` is available on the board.
**When:** Experiment 3867 copies the deterministic Ising workload plus runner to
the board, runs the workload over SSH, captures stdout, and probes SoC
temperature if readable.
**Then:** It writes `results/experiment_3867_polarfire_soc_smoke_v4.json` with
precondition evidence, CPU and board result hashes, the hash-match boolean,
real `run_duration_s`, thermal fields, principle annotations, and a terminal,
partial, or blocked verdict without fabricating board evidence.

**Implementation status:** Pending (Exp 3867)

---

### REQ-HW-3889

**Title:** GateMate continuity corrigendum MUST de-flag Exp 3866 with distinct timers and honest readback

**Description:**
Experiment 3889 MUST rerun the GateMate A1-EVB-2M n=16 Carnot Ising tile
continuity flow as a corrigendum for Exp 3866. Before any synthesis, packing,
flashing, or readback attempt, it MUST check `command -v nextpnr-himbaechel`,
`command -v yosys`, `command -v openFPGALoader`, and
`openFPGALoader -c dirtyJtag --detect` proving the Cologne Chip GateMate/GM1Ax
IDCODE. Missing tools MUST emit `blocked_gatemate_toolchain_missing`; missing
board contact MUST emit `blocked_gatemate_board_unreachable`.

When preconditions pass, the experiment MUST run the current open-source
GateMate flow (`yosys synth_gatemate`, `nextpnr-himbaechel --device CCGM1A1`,
`gmpack`, and `openFPGALoader -c dirtyJtag -b olimex_gatemateevb`) against the
n=16 tile. The artifact MUST record `duration_s` as total task wall-clock and
`run_duration_s` as the on-board flash/readback interval only; these two fields
MUST be measured by distinct timers and MUST NOT be bit-identical. The artifact
MUST probe whether the installed `openFPGALoader` advertises GateMate-compatible
readback support, attempt readback only when supported, and compare any captured
readback SHA256 to the flashed bitstream SHA256.

**Acceptance criteria:**
- `scripts/experiments/experiment_3889_gatemate_continuity_corrigendum.py`
  writes `results/experiment_3889_gatemate_continuity_corrigendum.json`.
- The artifact includes bare `duration_s`, `run_duration_s`,
  `readback_verified`, `readback_supported`, `gatemate_bitstream_flashed`,
  `fmax_mhz`, `lut_used`, `dff_used`, `preconditions_checked`,
  `reproducibility_checksum`, and `inference_substrate` fields.
- `duration_s != run_duration_s` for every non-blocked flash artifact.
- A clean terminal artifact sets `gatemate_bitstream_flashed=true` and either
  `readback_verified=true` or `readback_supported=false`, with
  `honest_verdict` beginning
  `success: gatemate_continuity_CLEAN_terminal_fmax`.
- A flashed artifact with supported but unverified readback emits an
  `honest_verdict` beginning
  `success: gatemate_continuity_flashed_readback_inconclusive_fmax`.
- Required fields are annotated through `field_provenance`; the fields
  themselves MUST remain bare booleans, numbers, strings, or lists rather than
  `{value, principle}` wrappers.

**Implementation status:** Pending (Exp 3889)

---

### SCENARIO-HW-3889

**Scenario:** GateMate corrigendum re-flashes the n=16 tile without timer tautology and records readback support.

**Given:** The GateMate toolchain may or may not be installed, the Olimex
GateMate A1-EVB-2M may or may not be reachable over DirtyJTAG, and the installed
`openFPGALoader` may or may not expose a compatible `--readback` command.
**When:** Experiment 3889 runs its preconditions and, only on pass, runs the
synth_gatemate -> nextpnr-himbaechel -> gmpack -> openFPGALoader flow plus the
readback probe.
**Then:** It writes the corrigendum artifact with terminal, inconclusive, or
blocked verdict evidence; distinct task and on-board timers; real Fmax and
LUT/DFF counts; and an honest readback-supported/readback-verified record.

**Implementation status:** Pending (Exp 3889)

---

### REQ-HW-3900

**Title:** GateMate terminal confirmation MUST re-confirm the n=16 flash state with distinct timers and honest readback

**Description:**
Experiment 3900 MUST re-confirm the GateMate A1-EVB-2M n=16 Carnot Ising tile
state after the Exp 3889 timer corrigendum. Before any synthesis, packing,
flashing, or readback attempt, it MUST check the GateMate toolchain with
`command -v nextpnr-himbaechel && command -v yosys && command -v
openFPGALoader` and MUST check board contact with `openFPGALoader -c dirtyJtag
--detect` proving the Cologne Chip GateMate/GM1Ax IDCODE. Missing tools MUST
emit `blocked_gatemate_toolchain_missing`; missing board contact MUST emit
`blocked_gatemate_board_unreachable`.

When preconditions pass, the experiment MUST run the current open-source
GateMate flow (`yosys synth_gatemate`, `nextpnr-himbaechel --device CCGM1A1`,
`gmpack`, and `openFPGALoader -c dirtyJtag -b olimex_gatemateevb`) against the
n=16 tile or otherwise confirm the same flashed state with equivalent board
evidence. The artifact MUST record `duration_s` as total task wall-clock and
`run_duration_s` as on-board run time only; the two timer fields MUST be
distinct and MUST NOT repeat the Exp 3866 tautology. The artifact MUST honestly
record whether JTAG readback is supported, compare readback bytes to the flashed
bitstream hash when supported, and set a bare `terminal_state_reached` boolean
only when the bitstream is flashed, smoke evidence is OK, the readback gate is
either verified or unsupported, and no timer tautology is present.

**Acceptance criteria:**
- `scripts/experiments/experiment_3900_gatemate_terminal_confirmation.py`
  writes `results/experiment_3900_gatemate_terminal_confirmation.json`.
- The artifact includes bare `duration_s`, `run_duration_s`,
  `readback_verified`, `readback_supported`, `terminal_state_reached`,
  `gatemate_bitstream_flashed`, `fmax_mhz`, `lut_used`, `dff_used`,
  `preconditions_checked`, `reproducibility_checksum`, and
  `inference_substrate` fields.
- `duration_s != run_duration_s` for every flashed artifact, and
  `terminal_state_reached=false` whenever the two timer fields are equal.
- A terminal artifact emits an `honest_verdict` beginning
  `success: gatemate_TERMINAL_reached_fmax` and ending
  `_can_graduate_to_opportunistic`.
- A flashed artifact with supported but unverified readback emits an
  `honest_verdict` beginning
  `success: gatemate_flashed_readback_inconclusive_fmax` and ending
  `_stays_mandatory`.
- Failed preconditions or failed flash/build flow artifacts emit
  `honest_verdict` values beginning with `blocked_`.
- Required fields are annotated through `field_provenance`; the fields
  themselves MUST remain bare booleans, numbers, strings, or lists rather than
  `{value, principle}` wrappers.

**Implementation status:** Pending (Exp 3900)

---

### SCENARIO-HW-3900

**Scenario:** GateMate terminal confirmation records graduation or mandatory state without timer tautology.

**Given:** The GateMate toolchain may or may not be installed, the Olimex
GateMate A1-EVB-2M may or may not be reachable over DirtyJTAG, and the installed
`openFPGALoader` may or may not expose a compatible `--readback` command.
**When:** Experiment 3900 runs its preconditions and, only on pass, re-confirms
the n=16 tile through the synth_gatemate -> nextpnr-himbaechel -> gmpack ->
openFPGALoader flow plus the readback probe.
**Then:** It writes the terminal-confirmation artifact with distinct task and
on-board timers, real Fmax and LUT/DFF counts, readback-supported/readback-
verified evidence, a bare terminal-state boolean, and a terminal, caveated, or
blocked verdict.

**Implementation status:** Pending (Exp 3900)

---

### REQ-HW-3890

**Title:** PolarFire plus KV260 consolidated continuity audit MUST preserve SSH-only preconditions and no fabric-acceleration claim

**Description:**
Experiment 3890 MUST consolidate the PolarFire and KV260 continuity checks into
one hardware-smoke artifact. It MUST check each board independently and MUST NOT
fail the whole experiment when only one board is unreachable. The PolarFire
precondition is `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
The KV260 precondition is exactly SSH reachability,
`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; the retired host
SD-card precondition MUST NOT be used.

If PolarFire is reachable, the experiment MUST re-confirm Exp 3867's terminal
hash-verified Ising dispatch boundary and record it honestly as a soft-CPU SSH
dispatch with hash verification, not FPGA fabric compute acceleration. If KV260
is reachable, the experiment MUST run `ssh kria 'xmutil listapps'` and
`ssh kria 'ls /dev/uio*'`, record the loaded overlay, UIO presence, and whether
the Carnot Ising bitstream is active. The artifact MUST keep
`fabric_acceleration_claimed=false` because neither board demonstrates compute
acceleration in this continuity audit.

**Acceptance criteria:**
- `scripts/experiments/experiment_3890_polarfire_kv260_continuity.py` writes
  `results/experiment_3890_polarfire_kv260_continuity.json`.
- The artifact includes bare `polarfire_reachable`, `kv260_reachable`,
  `polarfire_state`, `kv260_state`, `fabric_acceleration_claimed`,
  `preconditions_checked`, `reproducibility_checksum`, `inference_substrate`,
  and `honest_verdict` fields.
- Required fields are annotated through `field_principles`; the required fields
  themselves MUST remain bare booleans, strings, or lists rather than
  `{value, principle}` wrappers.
- If one board is SSH-unreachable, the artifact records
  `blocked_<board>_ssh_unreachable` for that board and still audits the other
  board.
- If at least one board is reachable and `fabric_acceleration_claimed=false`,
  `honest_verdict` MUST begin
  `success: polarfire_kv260_continuity_pf` and end `_no_fabric_claim`.
- If neither board is reachable, `honest_verdict` MUST be exactly
  `blocked_polarfire_and_kv260_ssh_unreachable`.
- The artifact MUST set `inference_substrate="hardware_smoke"` and MUST NOT
  include GGUF, CUDA, live-inference, hardware-speedup, thermalization, or host
  SD-card markers.

**Implementation status:** Pending (Exp 3890)

---

### SCENARIO-HW-3890

**Scenario:** Consolidated PolarFire and KV260 continuity records terminal or blocked state without acceleration claims.

**Given:** PolarFire and KV260 may independently be reachable or unreachable
over SSH.
**When:** Experiment 3890 runs both SSH preconditions, dispatches the
hash-verified PolarFire soft-CPU Ising check only when PolarFire is reachable,
and runs KV260 `xmutil listapps` plus `/dev/uio*` listing only when KV260 is
reachable.
**Then:** It writes `results/experiment_3890_polarfire_kv260_continuity.json`
with per-board state, raw command evidence, scalar required fields, a false
fabric-acceleration claim flag, a terminal success verdict when either board is
reachable, or the both-boards blocked verdict when neither SSH precondition
passes.

**Implementation status:** Pending (Exp 3890)

---

### REQ-HW-3901

**Title:** PolarFire plus KV260 continuity audit MUST independently record SSH reachability and no fabric-acceleration claim

**Description:**
Experiment 3901 MUST produce a consolidated PolarFire plus KV260 continuity
artifact for the Hardware-Task Continuity Discipline. It MUST check each board
independently and MUST continue when only one board is SSH-unreachable. The
PolarFire precondition is `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire
'true'`. The KV260 precondition is exactly
`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; host SD-card checks are
retired and MUST NOT be used.

If PolarFire is reachable, the experiment MUST re-confirm the Exp 3867 terminal
hash-verified Ising dispatch and record the state as a soft-CPU SSH dispatch
with hash verification only, with no FPGA fabric compute-acceleration claim. If
KV260 is reachable, the experiment MUST run `ssh kria 'xmutil listapps'` and
`ssh kria 'ls /dev/uio*'`, then record the loaded overlay, UIO presence, and
whether a Carnot Ising bitstream is active. The artifact MUST keep
`fabric_acceleration_claimed=false` because neither board demonstrates compute
acceleration in this continuity audit.

**Acceptance criteria:**
- `scripts/experiments/experiment_3901_polarfire_kv260_continuity.py` writes
  `results/experiment_3901_polarfire_kv260_continuity.json`.
- The artifact includes bare `polarfire_reachable`, `kv260_reachable`,
  `polarfire_state`, `kv260_state`, `fabric_acceleration_claimed`,
  `preconditions_checked`, `reproducibility_checksum`, `inference_substrate`,
  and `honest_verdict` fields.
- Required fields are annotated through `field_principles`; the required fields
  themselves MUST remain bare booleans, strings, or lists rather than
  `{value, principle}` wrappers.
- If one board is SSH-unreachable, the artifact records
  `blocked_<board>_ssh_unreachable` for that board and still audits the other
  board.
- If at least one board is reachable and `fabric_acceleration_claimed=false`,
  `honest_verdict` MUST begin
  `success: polarfire_kv260_continuity_pf` and end `_no_fabric_claim`.
- If neither board is reachable, `honest_verdict` MUST be exactly
  `blocked_polarfire_and_kv260_ssh_unreachable`.
- The artifact MUST set `inference_substrate="hardware_smoke"` and MUST NOT
  include GGUF, CUDA, live-inference, hardware-speedup, thermalization, or host
  SD-card markers.

**Implementation status:** Pending (Exp 3901)

---

### SCENARIO-HW-3901

**Scenario:** PolarFire and KV260 continuity is recorded with per-board blocked states and no acceleration claim.

**Given:** PolarFire and KV260 may independently be reachable or unreachable
over SSH.
**When:** Experiment 3901 runs both SSH preconditions, dispatches the
hash-verified PolarFire soft-CPU Ising check only when PolarFire is reachable,
and runs KV260 `xmutil listapps` plus `/dev/uio*` listing only when KV260 is
reachable.
**Then:** It writes `results/experiment_3901_polarfire_kv260_continuity.json`
with per-board state, raw command evidence, scalar required fields, a false
fabric-acceleration claim flag, a terminal success verdict when either board is
reachable, or the both-boards blocked verdict when neither SSH precondition
passes.

**Implementation status:** Pending (Exp 3901)

---

### REQ-HW-3922

**Title:** Consolidated hardware continuity MUST audit GateMate, PolarFire, and KV260 independently

**Description:**
Experiment 3922 MUST produce
`results/experiment_3922_hardware_continuity_consolidated.json` as the
consolidated continuity audit for the attached GateMate, PolarFire, and KV260
boards. The experiment MUST check each board independently and MUST NOT cascade
one board's failure to the other boards. The GateMate precondition is
`command -v nextpnr-himbaechel && command -v yosys && command -v
openFPGALoader` plus `openFPGALoader -c dirtyJtag --detect`. PolarFire MUST use
`ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`. KV260 MUST use
`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; retired host SD-card
preconditions MUST NOT be used.

If GateMate is reachable, the experiment MUST re-confirm the n=16 tile flash
state, record bare `duration_s` and `run_duration_s` fields that are distinct,
attempt JTAG readback when the installed tool flow supports it, and set
`gatemate_terminal_state_reached` only when flashed plus smoke evidence passes,
the readback gate is verified or unsupported, and the timer tautology is absent.
If PolarFire is reachable, the experiment MUST re-confirm the Exp 3867 terminal
hash-verified Ising dispatch and record `polarfire_state` honestly as soft-CPU
SSH dispatch with hash verification, not FPGA fabric compute acceleration. If
KV260 is reachable, the experiment MUST run `ssh kria 'xmutil listapps'` and
`ssh kria 'ls /dev/uio*'`, then record the loaded overlay, UIO presence, and
whether the Carnot Ising bitstream is active. The artifact MUST keep
`fabric_acceleration_claimed=false`.

**Acceptance criteria:**
- `scripts/experiments/experiment_3922_hardware_continuity_consolidated.py`
  writes `results/experiment_3922_hardware_continuity_consolidated.json`.
- The artifact includes bare `gatemate_reachable`, `polarfire_reachable`,
  `kv260_reachable`, `gatemate_terminal_state_reached`, `duration_s`,
  `run_duration_s`, `polarfire_state`, `kv260_state`,
  `fabric_acceleration_claimed`, `preconditions_checked`,
  `reproducibility_checksum`, `inference_substrate`, and `honest_verdict`
  fields.
- Required fields are annotated through `field_principles`; the required fields
  themselves MUST remain bare booleans, numbers, strings, or lists rather than
  `{value, principle}` wrappers.
- GateMate blockers MUST be recorded per board as
  `blocked_gatemate_toolchain_missing` or `blocked_gatemate_board_unreachable`
  and MUST NOT prevent PolarFire or KV260 checks.
- `duration_s != run_duration_s` whenever
  `gatemate_terminal_state_reached=true`; equal GateMate timers MUST keep that
  field false.
- If at least one board is reachable and `fabric_acceleration_claimed=false`,
  `honest_verdict` MUST begin `success: hardware_continuity_gatemate`, include
  `pf` and `kv` state tokens, and end `_no_fabric_claim`.
- If no board is reachable, `honest_verdict` MUST be exactly
  `blocked_all_boards_unreachable`.
- The artifact MUST set `inference_substrate="hardware_smoke"` and MUST NOT
  include GGUF, CUDA, live-inference, hardware-speedup, thermalization, or host
  SD-card markers.

**Implementation status:** Planned (Exp 3922)

---

### SCENARIO-HW-3922

**Scenario:** Three-board continuity records terminal, non-terminal, or blocked state without acceleration claims.

**Given:** GateMate, PolarFire, and KV260 may independently be reachable or
unreachable through their allowed board-specific preconditions.
**When:** Experiment 3922 checks GateMate through JTAG detect and the current
open-source GateMate flow, checks PolarFire through SSH and the Exp 3867
hash-verified soft-CPU dispatch, and checks KV260 through SSH `xmutil listapps`
plus `/dev/uio*` listing.
**Then:** It writes
`results/experiment_3922_hardware_continuity_consolidated.json` with per-board
state, scalar required fields, field principles, a false fabric-acceleration
claim flag, a terminal success verdict when any board is reachable, or the
all-boards blocked verdict when no board precondition passes.

**Implementation status:** Planned (Exp 3922)

---

### REQ-HW-3931

**Title:** Clean hardware continuity rerun MUST repair Exp 3922 zero-duration and substrate-marker flags

**Description:**
Experiment 3931 MUST produce
`results/experiment_3931_hardware_continuity_clean_rerun.json` as the clean
rerun of the Exp 3922 consolidated GateMate, PolarFire, and KV260 continuity
audit. The experiment MUST check the same per-board preconditions independently:
GateMate uses `command -v nextpnr-himbaechel && command -v yosys && command -v
openFPGALoader` plus `openFPGALoader -c dirtyJtag --detect`, PolarFire uses
`ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`, and KV260 uses
`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. KV260 host SD-card
preconditions MUST NOT be used.

The artifact MUST repair the Exp 3922 flag by recording two honest distinct
top-level timers whenever any board is reachable: `duration_s` for the full
clean-rerun wall clock and `run_duration_s` for measured board-command or
board-dispatch time. If no board is reachable, the experiment MUST emit
`blocked_all_boards_unreachable` while still recording measured precondition
timers. The artifact MUST keep all required fields as bare scalar or list values, keep
`fabric_acceleration_claimed=false`, declare `inference_substrate` as
`hardware_smoke`, and avoid model-inference substrate markers.

If GateMate is reachable, the experiment MUST re-confirm the n=16 tile flash
state, attempt JTAG readback when supported, record readback unsupported when
that is the honest tool-flow result, and set `gatemate_terminal_state_reached`
only when flash plus smoke plus readback-or-unsupported evidence passes and the
timer tautology is absent. If PolarFire is reachable, the experiment MUST
re-confirm the Exp 3867 hash-verified dispatch as soft-CPU SSH dispatch only.
If KV260 is reachable, the experiment MUST run `ssh kria 'xmutil listapps'` and
`ssh kria 'ls /dev/uio*'`, then record the loaded overlay, UIO devices, and
whether the Carnot Ising bitstream is active.

**Acceptance criteria:**
- `scripts/experiments/experiment_3931_hardware_continuity_clean_rerun.py`
  writes `results/experiment_3931_hardware_continuity_clean_rerun.json`.
- The artifact includes bare `gatemate_reachable`, `polarfire_reachable`,
  `kv260_reachable`, `gatemate_terminal_state_reached`, `duration_s`,
  `run_duration_s`, `polarfire_state`, `kv260_state`,
  `fabric_acceleration_claimed`, `preconditions_checked`,
  `reproducibility_checksum`, `inference_substrate`, and `honest_verdict`
  fields.
- If at least one board is reachable, `duration_s > 0`, `run_duration_s > 0`,
  and `duration_s != run_duration_s`.
- If at least one board is reachable and `fabric_acceleration_claimed=false`,
  `honest_verdict` MUST begin
  `success: hardware_continuity_clean_gatemate`, include `pf` and `kv` state
  tokens, and end `_distinct_timers_no_fabric_claim`.
- If no board is reachable, `honest_verdict` MUST be exactly
  `blocked_all_boards_unreachable`.
- The artifact MUST NOT include model-inference substrate markers or retired
  KV260 host-storage markers.

**Implementation status:** Planned (Exp 3931)

---

### SCENARIO-HW-3931

**Scenario:** Clean three-board continuity records distinct timers without fabric claims.

**Given:** GateMate, PolarFire, and KV260 may independently be reachable or
unreachable through their allowed board-specific preconditions.
**When:** Experiment 3931 runs the clean rerun, recording full wall-clock time
separately from measured board-operation time.
**Then:** It writes
`results/experiment_3931_hardware_continuity_clean_rerun.json` with per-board
state, scalar required fields, distinct success timers, field principles, a
false fabric-acceleration claim flag, a terminal success verdict when any board
is reachable, or the all-boards blocked verdict when no board precondition
passes.

**Implementation status:** Planned (Exp 3931)

---

### REQ-HW-3972

**Title:** Hardware continuity owed rerun MUST keep KV260, GateMate, and PolarFire visible with per-board timers

**Description:**
Experiment 3972 MUST produce
`results/experiment_3972_hardware_continuity.json` as the owed rerun for the
Exp 3961 hardware-continuity slot that produced no artifact. The experiment is
a reachability and continuity check only; ARC work remains the milestone focus.
The three attached boards MUST be checked independently and honestly:
KV260 MUST use only `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'` as
the board precondition, GateMate MUST use
`openFPGALoader -c dirtyJtag --detect` and require a colognechip/GateMate
IDCODE in the output, and PolarFire MUST use
`ssh -o ConnectTimeout=5 polarfire 'true'`. KV260 host SD-card block-device
checks MUST NOT be used.

For each reachable board, the artifact MUST record the detected or loaded
continuity evidence and the next concrete forward step. KV260 evidence MUST
include the loaded overlay from `ssh kria 'xmutil listapps'` plus `/dev/uio*`
state from `ssh kria 'ls /dev/uio*'`. GateMate evidence MUST include the
`openFPGALoader --detect` output. PolarFire evidence MUST include the live SSH
continuity output gathered after the precondition. For any unreachable board,
the board's next step MUST be exactly `blocked_<board>_unreachable`, where
`<board>` is `kv260`, `gatemate`, or `polarfire`.

The artifact MUST include bare `kv260_reachable`, `gatemate_reachable`,
`polarfire_reachable`, `per_board_next_step`, `per_board_duration_s`,
`preconditions_checked`, `honest_verdict`, `duration_s`, and
`inference_substrate` fields. `per_board_duration_s` MUST contain one measured
wall-clock timer per board, and all three values MUST be positive and distinct
so the Exp 3866 timer-tautology failure mode cannot recur. The artifact MUST
set `inference_substrate="hardware_smoke"`. `honest_verdict` MUST start with
`success:` or `complete:` when at least one board is reachable, or with
`blocked_` when no board is reachable.

**Acceptance criteria:**
- `scripts/experiments/experiment_3972_hardware_continuity.py` writes
  `results/experiment_3972_hardware_continuity.json`.
- The required artifact fields are present as bare values; field principles
  annotate those fields separately through `field_principles`.
- `preconditions_checked` records which resource checks ran, with at least
  `{resource, available}` for `kv260_ssh`, `gatemate_jtag_detect`, and
  `polarfire_ssh`.
- Reachable boards have concrete next steps derived from their recorded
  continuity evidence; unreachable boards have the required
  `blocked_<board>_unreachable` next-step string.
- The artifact contains no KV260 host SD-card marker such as `/dev/mmcblk`.

**Implementation status:** Planned (Exp 3972)

---

### SCENARIO-HW-3972

**Scenario:** Owed three-board continuity rerun records reachability, evidence, and distinct board timers.

**Given:** KV260, GateMate, and PolarFire may independently be reachable or
unreachable through their allowed SSH/JTAG preconditions.
**When:** Experiment 3972 runs the hardware-smoke continuity rerun.
**Then:** It writes `results/experiment_3972_hardware_continuity.json` with
the bare required fields, per-board precondition records, per-board next steps,
distinct positive per-board durations, a hardware-smoke substrate declaration,
and an honest terminal-prefix or blocked verdict.

**Implementation status:** Planned (Exp 3972)

---

### REQ-HW-3983

**Title:** ARC hardware continuity check MUST keep KV260, GateMate, and PolarFire visible without acceleration claims

**Description:**
Experiment 3983 MUST produce
`results/experiment_3983_hardware_continuity.json` as a reachability-only
hardware-continuity artifact while ARC remains the milestone focus. The
hardware target is continuity, not model inference or acceleration; the north
star relaxes KV260 work to the next terminal-confirming step. The experiment
MUST check the three attached boards independently:
KV260 MUST use only `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'` as
the board precondition, GateMate MUST use
`openFPGALoader -c dirtyJtag --detect` and require a colognechip/GateMate
IDCODE in the output, and PolarFire MUST use
`ssh -o ConnectTimeout=5 polarfire 'true'`. KV260 host SD-card block-device
checks MUST NOT be used.

For each reachable board, the artifact MUST record the loaded-overlay or
detect/continuity output and the next concrete forward step. KV260 evidence
MUST include the loaded overlay from `ssh kria 'xmutil listapps'` plus
`/dev/uio*` state from `ssh kria 'ls /dev/uio*'`, and the KV260 next step MUST
point toward terminal confirmation. GateMate evidence MUST include the
`openFPGALoader --detect` output. PolarFire evidence MUST include live SSH
continuity output gathered after the precondition. For any unreachable board,
the board's next step MUST be exactly `blocked_<board>_unreachable`, where
`<board>` is `kv260`, `gatemate`, or `polarfire`.

The artifact MUST include bare `kv260_reachable`, `gatemate_reachable`,
`polarfire_reachable`, `per_board_next_step`, `per_board_duration_s`,
`preconditions_checked`, `honest_verdict`, `duration_s`, and
`inference_substrate` fields. `preconditions_checked` MUST record which checks
ran with at least `{resource, available}` for the three board preconditions.
`per_board_duration_s` MUST contain one measured wall-clock timer per board,
and all three values MUST be positive and distinct so the Exp 3866
timer-tautology failure mode cannot recur. The artifact MUST set
`inference_substrate="hardware_smoke"`. `honest_verdict` MUST start with
`success:` or `complete:` when at least one board is reachable, or with
`blocked_` when no board is reachable.

**Acceptance criteria:**
- `scripts/experiments/experiment_3983_hardware_continuity.py` writes
  `results/experiment_3983_hardware_continuity.json`.
- The required artifact fields are present as bare values; field principles
  annotate those fields separately through `field_principles`.
- Reachable boards record the requested overlay/detect/continuity evidence and
  concrete next forward steps; unreachable boards record the required
  `blocked_<board>_unreachable` next-step string.
- The artifact contains no KV260 host SD-card marker such as `/dev/mmcblk`.

**Implementation status:** Planned (Exp 3983)

---

### SCENARIO-HW-3983

**Scenario:** ARC continuity check records board reachability, evidence, and distinct timers.

**Given:** KV260, GateMate, and PolarFire may independently be reachable or
unreachable through their allowed SSH/JTAG preconditions.
**When:** Experiment 3983 runs the hardware-smoke continuity check.
**Then:** It writes `results/experiment_3983_hardware_continuity.json` with
the bare required fields, per-board precondition records, per-board next steps,
distinct positive per-board durations, a hardware-smoke substrate declaration,
and an honest terminal-prefix or blocked verdict.

**Implementation status:** Planned (Exp 3983)

---

### REQ-HW-3995

**Title:** ARC hardware continuity check MUST keep all three attached boards visible with honest per-board blockers

**Description:**
Experiment 3995 MUST produce
`results/experiment_3995_hardware_continuity.json` as a reachability-only
hardware-continuity artifact while ARC remains the milestone focus. The check is
limited to board reachability and continuity evidence; it MUST NOT claim model
inference or acceleration. KV260 work is relaxed to the next terminal-confirming
step per north-star section 3. The experiment MUST check the three attached
boards independently:
KV260 MUST use only `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'` as
the board precondition, GateMate MUST use
`openFPGALoader -c dirtyJtag --detect` and require a colognechip/GateMate
IDCODE in the output, and PolarFire MUST use
`ssh -o ConnectTimeout=5 polarfire 'true'`. KV260 host SD-card block-device
checks MUST NOT be used.

For each reachable board, the artifact MUST record the loaded-overlay or
detect/continuity output and the next concrete forward step. KV260 evidence
MUST include the loaded overlay from `ssh kria 'xmutil listapps'` plus
`/dev/uio*` state from `ssh kria 'ls /dev/uio*'`, and the KV260 next step MUST
point toward terminal confirmation. GateMate evidence MUST include the
`openFPGALoader --detect` output. PolarFire evidence MUST include live SSH
continuity output gathered after the precondition. For any unreachable board,
the board's next step MUST be exactly `blocked_<board>_unreachable`, where
`<board>` is `kv260`, `gatemate`, or `polarfire`.

The artifact MUST include bare `kv260_reachable`, `gatemate_reachable`,
`polarfire_reachable`, `per_board_next_step`, `per_board_duration_s`,
`preconditions_checked`, `honest_verdict`, `duration_s`, and
`inference_substrate` fields. `preconditions_checked` MUST record which checks
ran with at least `{resource, available}` for the three board preconditions.
`per_board_duration_s` MUST contain one measured wall-clock timer per board,
and all three values MUST be positive and distinct so the Exp 3866
timer-tautology failure mode cannot recur. The artifact MUST set
`inference_substrate="hardware_smoke"`. `honest_verdict` MUST start with
`success:` or `complete:` when at least one board is reachable, or with
`blocked_` when no board is reachable.

**Acceptance criteria:**
- `scripts/experiments/experiment_3995_hardware_continuity.py` writes
  `results/experiment_3995_hardware_continuity.json`.
- The required artifact fields are present as bare values; field principles
  annotate those fields separately through `field_principles`.
- Reachable boards record the requested overlay/detect/continuity evidence and
  concrete next forward steps; unreachable boards record the required
  `blocked_<board>_unreachable` next-step string.
- The artifact contains no KV260 host SD-card marker such as `/dev/mmcblk`.

**Implementation status:** Planned (Exp 3995)

---

### SCENARIO-HW-3995

**Scenario:** ARC continuity check records board reachability, evidence, and distinct timers.

**Given:** KV260, GateMate, and PolarFire may independently be reachable or
unreachable through their allowed SSH/JTAG preconditions.
**When:** Experiment 3995 runs the hardware-smoke continuity check.
**Then:** It writes `results/experiment_3995_hardware_continuity.json` with
the bare required fields, per-board precondition records, per-board next steps,
distinct positive per-board durations, a hardware-smoke substrate declaration,
and an honest terminal-prefix or blocked verdict.

**Implementation status:** Planned (Exp 3995)

---

### REQ-HW-4006

**Title:** ARC hardware continuity check MUST preserve KV260, GateMate, and PolarFire visibility

**Description:**
Experiment 4006 MUST produce
`results/experiment_4006_hardware_continuity.json` as a reachability-only
hardware-continuity artifact while ARC remains the focus. The check is limited
to board reachability and continuity evidence; it MUST NOT claim model
inference or acceleration. KV260 work is relaxed to the next terminal-confirming
step per north-star section 3. The experiment MUST check the three attached
boards independently:
KV260 MUST use only `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'` as
the board precondition, GateMate MUST use
`openFPGALoader -c dirtyJtag --detect` and require a colognechip/GateMate
IDCODE in the output, and PolarFire MUST use
`ssh -o ConnectTimeout=5 polarfire 'true'`. KV260 host SD-card block-device
checks MUST NOT be used.

For each reachable board, the artifact MUST record the loaded-overlay or
detect/continuity output and the next concrete forward step. KV260 evidence
MUST include the loaded overlay from `ssh kria 'xmutil listapps'` plus
`/dev/uio*` state from `ssh kria 'ls /dev/uio*'`, and the KV260 next step MUST
point toward terminal confirmation. GateMate evidence MUST include the
`openFPGALoader --detect` output. PolarFire evidence MUST include live SSH
continuity output gathered after the precondition. For any unreachable board,
the board's next step MUST be exactly `blocked_<board>_unreachable`, where
`<board>` is `kv260`, `gatemate`, or `polarfire`.

The artifact MUST include bare `kv260_reachable`, `gatemate_reachable`,
`polarfire_reachable`, `per_board_next_step`, `per_board_duration_s`,
`preconditions_checked`, `honest_verdict`, `duration_s`, and
`inference_substrate` fields. `preconditions_checked` MUST record which checks
ran with at least `{resource, available}` for the three board preconditions.
`per_board_duration_s` MUST contain one measured wall-clock timer per board,
and all three values MUST be positive and distinct so the Exp 3866
timer-tautology failure mode cannot recur. The artifact MUST set
`inference_substrate="hardware_smoke"`. `honest_verdict` MUST start with
`success:` or `complete:` when at least one board is reachable, or with
`blocked_` when no board is reachable.

**Acceptance criteria:**
- `scripts/experiments/experiment_4006_hardware_continuity.py` writes
  `results/experiment_4006_hardware_continuity.json`.
- The required artifact fields are present as bare values; field principles
  annotate those fields separately through `field_principles`.
- Reachable boards record the requested overlay/detect/continuity evidence and
  concrete next forward steps; unreachable boards record the required
  `blocked_<board>_unreachable` next-step string.
- The artifact contains no KV260 host SD-card marker such as `/dev/mmcblk`.

**Implementation status:** Implemented (Exp 4006)

---

### SCENARIO-HW-4006

**Scenario:** ARC continuity check records board reachability, evidence, and distinct timers.

**Given:** KV260, GateMate, and PolarFire may independently be reachable or
unreachable through their allowed SSH/JTAG preconditions.
**When:** Experiment 4006 runs the hardware-smoke continuity check.
**Then:** It writes `results/experiment_4006_hardware_continuity.json` with
the bare required fields, per-board precondition records, per-board next steps,
distinct positive per-board durations, a hardware-smoke substrate declaration,
and an honest terminal-prefix or blocked verdict.

**Implementation status:** Implemented (Exp 4006)

---

### REQ-HW-3577

**Title:** KV260 hardware-task continuity verification

**Description:**
To satisfy hardware-task continuity, one KV260 task per milestone must run. KV260 lives on SSH.
The task MUST check SSH reachability via `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.
If reachable, it MUST query loaded overlays using `xmutil listapps` via SSH.
It MUST output `results/experiment_3577_kv260_continuity_v16.json` with fields:
`honest_verdict` (terminal prefix), `inference_substrate="hardware_smoke"`, `preconditions_checked`,
`kv260_ssh_reachable` (bool), `kv260_overlay_loaded` (string or null), `random_seed`,
`reproducibility_checksum`, `duration_s`.

**Implementation status:** Implemented (Exp 3577)

---

### SCENARIO-HW-3577

**Scenario:** KV260 continuity is checked via SSH.

**Given:** A KV260 continuity check is required.
**When:** The script runs `ssh kria 'true'`.
**Then:** If the check fails, it produces a verdict `complete: blocked_kv260_ssh_unreachable`. If it succeeds, it runs `xmutil listapps` and produces `complete: kv260_continuity_confirmed_reachable` with the overlay state.

**Implementation status:** Implemented (Exp 3577)


### REQ-HW-3578

**Title:** PolarFire hardware-task continuity verification

**Description:**
To satisfy hardware-task continuity, one PolarFire task per milestone must run. PolarFire lives on SSH.
The task MUST check SSH reachability via `ssh -o ConnectTimeout=5 polarfire 'true'`.
If reachable, it MUST confirm continuity (uptime, carnot dispatch path) deflagged.
It MUST output `results/experiment_3578_polarfire_continuity_v16.json` with fields:
`honest_verdict` (terminal prefix), `inference_substrate="hardware_smoke"`, `preconditions_checked`,
`polarfire_ssh_reachable` (bool), `random_seed`, `reproducibility_checksum`, `duration_s`.

**Implementation status:** Implemented (Exp 3578)

---

### SCENARIO-HW-3578

**Scenario:** PolarFire continuity is checked via SSH.

**Given:** A PolarFire continuity check is required.
**When:** The script runs `ssh -o ConnectTimeout=5 polarfire 'true'`.
**Then:** If the check fails, it produces a verdict `complete: blocked_polarfire_ssh_timeout`. If it succeeds, it produces `complete: polarfire_continuity_confirmed_reachable`.

**Implementation status:** Implemented (Exp 3578)

---

### REQ-HW-3579

**Title:** GateMate continuity audit recorded flash smoke host-IO hang known-blocker

**Description:**
GateMate visibility each milestone, but the flash/smoke host-IO script hangs (known-issues). Documentation / reachability audit only — do NOT run the hanging flash/smoke path.
PRECONDITIONS (step 0): `openFPGALoader -c dirtyJtag --detect`. If it hangs/fails, record the known-issue state honestly and STOP.
REQUIRED ARTIFACT FIELDS (principle-annotated): honest_verdict (terminal prefix), inference_substrate=hardware_smoke, gatemate_idcode_detected (bool or null), known_blocker (string), random_seed, reproducibility_checksum, duration_s.
TERMINAL VERDICT: "complete: gatemate_continuity_audit_recorded_flash_smoke_host_io_hang_known_blocker"

**Implementation status:** Implemented (Exp 3579)

---

### SCENARIO-HW-3579

**Scenario:** GateMate continuity is audited via detect without running hanging host-IO path.

**Given:** A GateMate continuity audit is required.
**When:** The script checks `openFPGALoader -c dirtyJtag --detect` and avoids hanging paths.
**Then:** It writes `results/experiment_3579_gatemate_continuity_audit_v16.json` with the required artifact fields.

**Implementation status:** Implemented (Exp 3579)

---

### REQ-HW-3609

**Title:** GateMate continuity audit recorded flash smoke host-IO hang known-blocker

**Description:**
GateMate visibility each milestone, but the flash/smoke host-IO script hangs (known-issues). Documentation / reachability audit only — do NOT run the hanging flash/smoke path.
PRECONDITIONS (step 0): `openFPGALoader -c dirtyJtag --detect` (with a short timeout). If it hangs/fails, record the known-issue state honestly and STOP.
REQUIRED ARTIFACT FIELDS (principle-annotated): honest_verdict (terminal prefix), inference_substrate=hardware_smoke, gatemate_idcode_detected (bool or null), known_blocker (string), random_seed, reproducibility_checksum, duration_s.
TERMINAL VERDICT: "complete: gatemate_continuity_audit_recorded_flash_smoke_host_io_hang_known_blocker"

**Implementation status:** Proposed

---

### SCENARIO-HW-3609

**Scenario:** GateMate continuity is audited via detect without running hanging host-IO path for v18.

**Given:** A GateMate continuity audit is required.
**When:** The script checks `openFPGALoader -c dirtyJtag --detect` and avoids hanging paths.
**Then:** It writes `results/experiment_3609_gatemate_continuity_audit_v18.json` with the required artifact fields.

**Implementation status:** Proposed

### REQ-FPGA-KV260-CONTINUITY-V17

**Title:** KV260 continuity MUST be audited using ONLY SSH reachability

**Description:**
Experiment 3593 MUST perform a KV260 hardware-task continuity audit. It MUST use ONLY SSH reachability (`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`) as the precondition. The host SD-card device checks are strictly forbidden per the KV260 SSH-Not-SD-Card Discipline.

- If reachable, it MUST run `ssh kria 'xmutil listapps'` and record the overlay continuity state.
- If unreachable, it MUST record `honest_verdict: "complete: blocked_kv260_ssh_unreachable"` and not fail the test script.

**Acceptance Criteria:**
- `results/experiment_3593_kv260_continuity_v17.json` is generated.
- The artifact records `honest_verdict`, `inference_substrate: "hardware_smoke"`, `preconditions_checked`, `kv260_ssh_reachable`, `kv260_overlay_loaded` (if reachable), `random_seed`, `reproducibility_checksum`, and `duration_s`.

### SCENARIO-HW-3593

**Scenario:** KV260 continuity is audited via SSH reachability only.

**Given:** A KV260 continuity audit is required for the milestone.
**When:** The script checks SSH reachability (`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`) and, if reachable, records the overlay state.
**Then:** It writes `results/experiment_3593_kv260_continuity_v17.json` with the required artifact fields.

**Implementation status:** Pending

---

### REQ-HW-079

**Title:** PolarFire Continuity Check v17

**Description:**
Experiment 3594 MUST perform a continuity check for the PolarFire SoC board. Hardware-Task Continuity requires one PolarFire task per milestone. The experiment MUST first execute `ssh -o ConnectTimeout=5 polarfire 'true'` to verify board reachability. If reachable, it MUST confirm continuity by retrieving board uptime and checking the carnot dispatch path. The result MUST be written to `results/experiment_3594_polarfire_continuity_v17.json` using distinct field names.

**Acceptance criteria:**
- `results/experiment_3594_polarfire_continuity_v17.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate` equal to "hardware_smoke", `preconditions_checked`, `polarfire_ssh_reachable`, `random_seed`, `reproducibility_checksum`, and `duration_s`.
- If `ssh` succeeds, the verdict MUST be "complete: polarfire_continuity_confirmed_reachable".
- If `ssh` fails, the verdict MUST be "complete: blocked_polarfire_ssh_timeout".

**Implementation status:** Pending (Exp 3594)

---

### SCENARIO-HW-079

**Scenario:** PolarFire continuity check executes reachability and records artifact.

**Given:** PolarFire is checked via SSH.
**When:** Experiment 3594 runs the continuity check script.
**Then:** It writes the results artifact with all required artifact fields, and if reachable, records uptime and dispatch path.

**Implementation status:** Pending (Exp 3594)

---

### REQ-HW-080

**Title:** KV260 Continuity Check v18

**Description:**
Experiment 3607 MUST perform a continuity check for the KV260 board. Hardware-Task Continuity requires one KV260 task per milestone. The experiment MUST first execute `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'` to verify board reachability. The host SD-card device check is permanently retired. If reachable, it MUST execute `ssh kria 'xmutil listapps'` to determine the overlay loaded. The result MUST be written to `results/experiment_3607_kv260_continuity_v18.json` using distinct field names.

**Acceptance criteria:**
- `results/experiment_3607_kv260_continuity_v18.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate` equal to "hardware_smoke", `preconditions_checked`, `kv260_ssh_reachable`, `kv260_overlay_loaded`, `random_seed`, `reproducibility_checksum`, and `duration_s`.
- If `ssh` succeeds, the verdict MUST be "complete: kv260_continuity_confirmed_reachable".
- If `ssh` fails, the verdict MUST be "complete: blocked_kv260_ssh_unreachable".

**Implementation status:** Pending (Exp 3607)

---

### SCENARIO-HW-080

**Scenario:** KV260 continuity check executes reachability and records artifact.

**Given:** KV260 is checked via SSH.
**When:** Experiment 3607 runs the continuity check script.
**Then:** It writes the results artifact with all required artifact fields, and if reachable, records the overlay loaded.

**Implementation status:** Pending (Exp 3607)


---

### REQ-HW-3648

**Title:** KV260 Continuity Check v21 uses SSH reachability only and records the .331-.334 unreachable streak

**Description:**
Experiment 3648 MUST perform the milestone KV260 hardware-task continuity check.
KV260 lives on SSH, so the ONLY permitted precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and record the overlay continuity state. If SSH is unreachable, the experiment
MUST write a terminal blocked artifact rather than fabricating a pass, and MUST
record that .331, .332, .333, and .334 now form four consecutive unreachable
milestones requiring operator action.

**Acceptance criteria:**
- `results/experiment_3648_kv260_continuity_v21.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `kv260_ssh_reachable`, `kv260_overlay_loaded`,
  `consecutive_unreachable_milestones`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for the required fields.
- If `ssh` succeeds, the verdict MUST be
  `"complete: kv260_continuity_confirmed_reachable"`.
- If `ssh` fails, the verdict MUST be
  `"complete: blocked_kv260_ssh_unreachable"` and
  `consecutive_unreachable_milestones` MUST be `4`.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, or any
  host SD-card device-node check as a KV260 precondition.

**Implementation status:** Pending (Exp 3648)

---

### SCENARIO-HW-3648

**Scenario:** KV260 continuity v21 records reachable or blocked state from SSH only.

**Given:** A KV260 continuity check is required for milestone .334.
**When:** Experiment 3648 runs the SSH reachability precondition and, only when
reachable, runs `ssh kria 'xmutil listapps'`.
**Then:** It writes `results/experiment_3648_kv260_continuity_v21.json` with the
terminal verdict, SSH state, overlay state, unreachable streak count, field
principles, deterministic seed, checksum, and duration.

**Implementation status:** Pending (Exp 3648)

---

### REQ-HW-3661

**Title:** KV260 Continuity Check v22 uses SSH reachability only and records the .331-.335 unreachable streak

**Description:**
Experiment 3661 MUST perform the milestone KV260 hardware-task continuity check.
KV260 lives on SSH, so the ONLY permitted precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and record the overlay continuity state using distinct KV260 field names. If SSH
is unreachable, the experiment MUST write a terminal blocked artifact rather
than fabricating a pass, and MUST record that .331, .332, .333, .334, and .335
now form five consecutive unreachable milestones requiring operator action.

**Acceptance criteria:**
- `results/experiment_3661_kv260_continuity_v22.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `kv260_ssh_reachable`, `kv260_overlay_loaded`,
  `consecutive_unreachable_milestones`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for the required fields.
- If `ssh` succeeds, the verdict MUST be
  `"complete: kv260_continuity_confirmed_reachable"`.
- If `ssh` fails, the verdict MUST be
  `"complete: blocked_kv260_ssh_unreachable"` and
  `consecutive_unreachable_milestones` MUST be `5`.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, or any
  host SD-card device-node check as a KV260 precondition.

**Implementation status:** Pending (Exp 3661)

---

### SCENARIO-HW-3661

**Scenario:** KV260 continuity v22 records reachable or blocked state from SSH only.

**Given:** A KV260 continuity check is required for milestone .335.
**When:** Experiment 3661 runs the SSH reachability precondition and, only when
reachable, runs `ssh kria 'xmutil listapps'`.
**Then:** It writes `results/experiment_3661_kv260_continuity_v22.json` with the
terminal verdict, SSH state, overlay state, five-milestone unreachable streak
count, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Pending (Exp 3661)


---

### REQ-HW-3674

**Title:** KV260 Continuity Check v23 uses SSH reachability only and records the .331-.336 unreachable streak

**Description:**
Experiment 3674 MUST perform the milestone KV260 hardware-task continuity check.
KV260 lives on SSH, so the ONLY permitted precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and record the overlay continuity state using distinct KV260 field names. If SSH
is unreachable, the experiment MUST write a terminal blocked artifact rather
than fabricating a pass, and MUST record that .331, .332, .333, .334, .335, and
.336 now form six consecutive unreachable milestones requiring operator action.

**Acceptance criteria:**
- `results/experiment_3674_kv260_continuity_v23.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `kv260_ssh_reachable`, `kv260_overlay_loaded`,
  `consecutive_unreachable_milestones`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for the required fields.
- If `ssh` succeeds, the verdict MUST be
  `"complete: kv260_continuity_confirmed_reachable"`.
- If `ssh` fails, the verdict MUST be
  `"complete: blocked_kv260_ssh_unreachable"` and
  `consecutive_unreachable_milestones` MUST be `6`.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, or any
  host SD-card device-node check as a KV260 precondition.

**Implementation status:** Pending (Exp 3674)

---

### SCENARIO-HW-3674

**Scenario:** KV260 continuity v23 records reachable or blocked state from SSH only.

**Given:** A KV260 continuity check is required for milestone .336.
**When:** Experiment 3674 runs the SSH reachability precondition and, only when
reachable, runs `ssh kria 'xmutil listapps'`.
**Then:** It writes `results/experiment_3674_kv260_continuity_v23.json` with the
terminal verdict, SSH state, overlay state, six-milestone unreachable streak
count, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Pending (Exp 3674)


---

### REQ-HW-3686

**Title:** KV260 Continuity Check v24 uses SSH reachability only and records the .331-.337 unreachable streak

**Description:**
Experiment 3686 MUST perform the milestone KV260 hardware-task continuity check.
KV260 lives on SSH, so the ONLY permitted precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and record the overlay continuity state using distinct KV260 field names. If SSH
is unreachable, the experiment MUST write a terminal blocked artifact rather
than fabricating a pass, and MUST record that .331, .332, .333, .334, .335,
.336, and .337 now form seven consecutive unreachable milestones requiring
operator action.

**Acceptance criteria:**
- `results/experiment_3686_kv260_continuity_v24.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `kv260_ssh_reachable`, `kv260_overlay_loaded`,
  `consecutive_unreachable_milestones`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for the required fields.
- If `ssh` succeeds, the verdict MUST be
  `"complete: kv260_continuity_confirmed_reachable"`.
- If `ssh` fails, the verdict MUST be
  `"complete: blocked_kv260_ssh_unreachable"` and
  `consecutive_unreachable_milestones` MUST be `7`.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, or any
  host SD-card device-node check as a KV260 precondition.

**Implementation status:** Pending (Exp 3686)

---

### SCENARIO-HW-3686

**Scenario:** KV260 continuity v24 records reachable or blocked state from SSH only.

**Given:** A KV260 continuity check is required for milestone .337.
**When:** Experiment 3686 runs the SSH reachability precondition and, only when
reachable, runs `ssh kria 'xmutil listapps'`.
**Then:** It writes `results/experiment_3686_kv260_continuity_v24.json` with the
terminal verdict, SSH state, overlay state, seven-milestone unreachable streak
count, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Pending (Exp 3686)


---

### REQ-HW-3700

**Title:** GateMate continuity audit v25 records openFPGALoader state without running flash smoke host IO

**Description:**
Experiment 3700 MUST perform the milestone GateMate visibility audit without
running the known hanging flash/smoke host-IO path. The only permitted hardware
probe is the JTAG detect precondition:

1. Run `command -v openFPGALoader`.
2. Only if that command succeeds, run
   `timeout --kill-after=2s 5s openFPGALoader -c dirtyJtag --detect`.
3. Stop after recording the installed/detect state and the known blocker.

The artifact MUST NOT claim a flash or host-visible smoke pass. A successful
GateMate IDCODE detect only proves JTAG visibility for the audit.

**Acceptance criteria:**
- `results/experiment_3700_gatemate_continuity_audit_v25.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `openfpgaloader_installed`, `gatemate_idcode_detected`, `known_blocker`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists and storing the actual artifact value
  separately from the principle.
- `honest_verdict` MUST be
  `"complete: gatemate_continuity_audit_recorded_openfpgaloader_state_flash_smoke_host_io_hang_known_blocker"`.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF or CUDA markers.
- `openfpgaloader_installed` MUST record whether `command -v openFPGALoader`
  succeeded in the current environment.
- If `command -v openFPGALoader` fails, `openfpgaloader_installed` MUST be
  `false`, `gatemate_idcode_detected` MUST be `null`, and the detect command
  MUST NOT run.
- If the detect command runs and reports a GateMate IDCODE, the artifact MUST
  set `gatemate_idcode_detected=true` while still preserving the
  flash/smoke host-IO hang as a known blocker.
- If the detect command runs and exits non-zero, times out, or does not report
  a GateMate IDCODE, `gatemate_idcode_detected` MUST be `false` and the artifact
  MUST name the missing-tool or detect blocker together with the flash/smoke
  host-IO hang.
- No implementation or test path may invoke GateMate flash programming or the
  hanging host-IO smoke script.

**Implementation status:** Implemented (Exp 3700)

---

### SCENARIO-HW-3700

**Scenario:** GateMate continuity v25 audits JTAG detect state and preserves the host-IO blocker.

**Given:** A GateMate continuity audit is required for milestone .338.
**When:** Experiment 3700 checks for `openFPGALoader`, optionally runs the bounded
DirtyJTAG detect command, and avoids all flash/smoke host-IO commands.
**Then:** It writes `results/experiment_3700_gatemate_continuity_audit_v25.json`
with the terminal verdict, openFPGALoader installed state, IDCODE detect result,
known blocker, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Implemented (Exp 3700)


---

### REQ-HW-3699

**Title:** PolarFire continuity v25 MUST confirm live SSH continuity for milestone .338

**Description:**
Experiment 3699 MUST perform the milestone PolarFire hardware-task continuity
check. Hardware-Task Continuity requires one PolarFire task per milestone, and
the board was last recorded reachable in .337. The experiment MUST first execute
the SSH precondition:

`ssh -o ConnectTimeout=5 polarfire 'true'`

If that precondition exits zero, the experiment MUST collect live continuity
evidence from the board by recording uptime and the Carnot dispatch path with
distinct, deflagged PolarFire field names. If the precondition exits non-zero,
the experiment MUST write the terminal blocked verdict and MUST NOT claim board
continuity values.

**Acceptance criteria:**
- `results/experiment_3699_polarfire_continuity_v25.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `polarfire_ssh_reachable`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while storing the bare value separately.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF or CUDA markers.
- If `ssh -o ConnectTimeout=5 polarfire 'true'` succeeds, the verdict MUST be
  `"complete: polarfire_continuity_confirmed_reachable"` and the artifact MUST
  include PolarFire uptime and Carnot dispatch-path values.
- If the SSH precondition fails, the verdict MUST be
  `"complete: blocked_polarfire_ssh_timeout"` and the artifact MUST record the
  failed precondition before reporting the blocked board state.

**Implementation status:** Implemented (Exp 3699)

---

### SCENARIO-HW-3699

**Scenario:** PolarFire continuity v25 records reachable or blocked state from SSH.

**Given:** A PolarFire continuity check is required for milestone .338 after the
board was reachable in .337.
**When:** Experiment 3699 runs the SSH reachability precondition and, only when
reachable, asks the board for uptime and the Carnot dispatch path.
**Then:** It writes `results/experiment_3699_polarfire_continuity_v25.json` with
the terminal verdict, SSH state, precondition record, live continuity values
when reachable, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Implemented (Exp 3699)


---

### REQ-HW-3698

**Title:** KV260 Continuity Check v25 uses SSH reachability only and records the .331-.338 unreachable streak

**Description:**
Experiment 3698 MUST perform the milestone KV260 hardware-task continuity check.
KV260 lives on SSH, so the ONLY permitted precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and record the overlay continuity state using distinct KV260 field names. If SSH
is unreachable, the experiment MUST write a terminal blocked artifact rather
than fabricating a pass, and MUST record that .331, .332, .333, .334, .335,
.336, .337, and the current milestone now form eight consecutive unreachable
milestones requiring operator action.

**Acceptance criteria:**
- `results/experiment_3698_kv260_continuity_v25.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `kv260_ssh_reachable`, `kv260_overlay_loaded`,
  `consecutive_unreachable_milestones`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for the required fields,
  while the required fields store bare values.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF or CUDA markers.
- If `ssh` succeeds, the verdict MUST be
  `"complete: kv260_continuity_confirmed_reachable"`.
- If `ssh` fails, the verdict MUST be
  `"complete: blocked_kv260_ssh_unreachable"` and
  `consecutive_unreachable_milestones` MUST be `8`.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, or any
  host SD-card device-node check as a KV260 precondition.

**Implementation status:** Pending (Exp 3698)

---

### SCENARIO-HW-3698

**Scenario:** KV260 continuity v25 records reachable or blocked state from SSH only.

**Given:** A KV260 continuity check is required after milestones .331 through
.337 were SSH-unreachable.
**When:** Experiment 3698 runs the SSH reachability precondition and, only when
reachable, runs `ssh kria 'xmutil listapps'`.
**Then:** It writes `results/experiment_3698_kv260_continuity_v25.json` with the
terminal verdict, SSH state, overlay state, eight-milestone unreachable streak
count if still blocked, field principles, deterministic seed, checksum, and
duration.

**Implementation status:** Pending (Exp 3698)


---

### REQ-HW-3709

**Title:** KV260 terminal POC latency transcript uses SSH-only reachability and records raw on-board samples

**Description:**
Experiment 3709 MUST drive the KV260 hardware-task continuity mandate to a
terminal-candidate board-latency transcript. KV260 lives on SSH, so the ONLY
permitted precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and, when the Carnot Ising overlay is not already listed, MUST run
`ssh kria 'xmutil loadapp carnot_ising_v2_n64'` before re-checking the overlay.
If the board-side `xmutil` command reports that root privileges are required,
the experiment MAY retry the same `xmutil` operation as `sudo xmutil` over SSH,
while preserving the original non-sudo probe in the command transcript.
The experiment MUST then run the on-board Ising sampler over SSH and record
board-measured per-sample and per-batch wall-clock latency using a fixed compute
budget. The transcript is a POC functional latency anchor only; it MUST NOT make
thermalization, equilibrium-sampling, or hardware-speedup-over-CPU claims.

**Acceptance criteria:**
- `results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `kv260_ssh_reachable`, `kv260_overlay_loaded`,
  `board_latency_samples`, `board_latency_median_ms`,
  `terminal_condition_met`, `speedup_claim_avoided_assert`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for the required fields,
  while the required fields store bare values.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF or CUDA markers.
- If `ssh` fails, the verdict MUST be
  `"complete: blocked_kv260_ssh_unreachable"` and no overlay, host storage, or
  sampler command may be run.
- If `ssh` succeeds and at least 30 positive on-board latency samples are
  captured with a confirmed Carnot Ising overlay, the verdict MUST be
  `"complete: kv260_board_latency_transcript_captured_poc_anchor_terminal_candidate"`
  and `terminal_condition_met` MUST be `true`.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, or any
  host SD-card device-node check as a KV260 precondition.

**Implementation status:** Pending (Exp 3709)

---

### SCENARIO-HW-3709

**Scenario:** KV260 terminal-candidate latency transcript records reachable or blocked state honestly.

**Given:** The KV260 board is the current hardware-task continuity target and
must be checked by SSH reachability only.
**When:** Experiment 3709 runs the SSH precondition, confirms or loads the
Carnot Ising overlay, and runs the board sampler timing harness over SSH.
**Then:** It writes
`results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json` with
the terminal-prefixed verdict, SSH state, overlay state, raw latency
distribution, fixed compute budget, terminal-condition flag, field principles,
deterministic seed, checksum, and duration.

**Implementation status:** Pending (Exp 3709)


---

### REQ-HW-3710

**Title:** PolarFire continuity v26 MUST confirm live SSH continuity for milestone .339

**Description:**
Experiment 3710 MUST perform the milestone PolarFire hardware-task continuity
check. Hardware-Task Continuity requires one PolarFire task per milestone, and
the board was last recorded reachable in .338. The experiment MUST first execute
the SSH precondition:

`ssh -o ConnectTimeout=5 polarfire 'true'`

If that precondition exits zero, the experiment MUST collect live continuity
evidence from the board by recording uptime and the Carnot dispatch path with
distinct, deflagged PolarFire field names. If the precondition exits non-zero,
the experiment MUST write the terminal blocked verdict and MUST NOT claim board
continuity values.

**Acceptance criteria:**
- `results/experiment_3710_polarfire_continuity_v26.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `polarfire_ssh_reachable`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while storing the bare value separately.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF or CUDA markers.
- If `ssh -o ConnectTimeout=5 polarfire 'true'` succeeds, the verdict MUST be
  `"complete: polarfire_continuity_confirmed_reachable"` and the artifact MUST
  include PolarFire uptime and Carnot dispatch-path values.
- If the SSH precondition fails, the verdict MUST be
  `"complete: blocked_polarfire_ssh_timeout"` and the artifact MUST record the
  failed precondition before reporting the blocked board state.

**Implementation status:** Implemented (Exp 3710)

---

### SCENARIO-HW-3710

**Scenario:** PolarFire continuity v26 records reachable or blocked state from SSH.

**Given:** A PolarFire continuity check is required for milestone .339 after the
board was reachable in .338.
**When:** Experiment 3710 runs the SSH reachability precondition and, only when
reachable, asks the board for uptime and the Carnot dispatch path.
**Then:** It writes `results/experiment_3710_polarfire_continuity_v26.json` with
the terminal verdict, SSH state, precondition record, live continuity values
when reachable, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Implemented (Exp 3710)


---

### REQ-HW-3711

**Title:** GateMate continuity audit v26 records openFPGALoader state without running flash smoke host IO

**Description:**
Experiment 3711 MUST perform the milestone GateMate visibility audit without
running the known hanging flash/smoke host-IO path. The only permitted hardware
probe is the JTAG detect precondition:

1. Run `command -v openFPGALoader`.
2. Only if that command succeeds, run
   `timeout --kill-after=2s 5s openFPGALoader -c dirtyJtag --detect`.
3. Stop after recording the installed/detect state and the known blocker.

The artifact MUST NOT claim a flash or host-visible smoke pass. A successful
GateMate IDCODE detect only proves JTAG visibility for the audit.

**Acceptance criteria:**
- `results/experiment_3711_gatemate_continuity_audit_v26.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `openfpgaloader_installed`, `gatemate_idcode_detected`, `known_blocker`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while storing the bare value separately.
- `honest_verdict` MUST be
  `"complete: gatemate_continuity_audit_recorded_openfpgaloader_state_flash_smoke_host_io_hang_known_blocker"`.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF or CUDA markers.
- `openfpgaloader_installed` MUST record whether `command -v openFPGALoader`
  succeeded in the current environment.
- If `command -v openFPGALoader` fails, `openfpgaloader_installed` MUST be
  `false`, `gatemate_idcode_detected` MUST be `null`, and the detect command
  MUST NOT run.
- If the detect command runs and reports a GateMate IDCODE, the artifact MUST
  set `gatemate_idcode_detected=true` while still preserving the
  flash/smoke host-IO hang as a known blocker.
- If the detect command runs and exits non-zero, times out, or does not report
  a GateMate IDCODE, `gatemate_idcode_detected` MUST be `false` and the artifact
  MUST name the missing-tool or detect blocker together with the flash/smoke
  host-IO hang.
- No implementation or test path may invoke GateMate flash programming or the
  hanging host-IO smoke script.

**Implementation status:** Implemented (Exp 3711)

---

### SCENARIO-HW-3711

**Scenario:** GateMate continuity v26 audits JTAG detect state and preserves the host-IO blocker.

**Given:** A GateMate continuity audit is required for milestone .339.
**When:** Experiment 3711 checks for `openFPGALoader`, optionally runs the bounded
DirtyJTAG detect command, and avoids all flash/smoke host-IO commands.
**Then:** It writes `results/experiment_3711_gatemate_continuity_audit_v26.json`
with the terminal verdict, openFPGALoader installed state, IDCODE detect result,
known blocker, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Implemented (Exp 3711)


---

### REQ-HW-3721

**Title:** Consolidated hardware continuity confirms KV260 terminal state and audits PolarFire plus GateMate

**Description:**
Experiment 3721 MUST perform the consolidated hardware-task continuity slot for
milestone .340. KV260 reached its terminal-candidate board latency transcript in
Exp 3709, so this experiment MUST confirm the terminal condition rather than
rerun the latency harness. The ONLY permitted KV260 precondition is SSH
reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If KV260 SSH is reachable, the experiment MUST run
`ssh kria 'xmutil listapps'` to confirm that a `carnot_ising` overlay is listed.
If `xmutil` reports that root privileges are required, the experiment MAY retry
the same list operation as `sudo xmutil listapps` over SSH while preserving the
original non-sudo probe. The experiment MUST inspect the existing
`results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json`
artifact on disk and only accept it as terminal evidence when it carries at
least 30 positive board-latency samples plus board-harness command evidence for
real timings. If the overlay is confirmed and the Exp 3709 transcript evidence
is non-fabricated, `kv260_terminal_condition_confirmed` MUST be a bare boolean
`true` and the artifact MUST recommend that the operator lift the per-milestone
KV260 mandate.

The same artifact MUST opportunistically audit PolarFire and GateMate without
making either board a milestone blocker. PolarFire MUST use
`ssh -o ConnectTimeout=5 polarfire 'true'` as its precondition and, only when
reachable, record uptime and the Carnot dispatch path with distinct PolarFire
field names. GateMate MUST only record `command -v openFPGALoader` state and the
known flash/smoke host-IO hang blocker; it MUST NOT run flash programming,
host-visible smoke, or the hanging GateMate path.

The transcript is a POC functional latency anchor only; the artifact MUST NOT
make thermalization, equilibrium-sampling, or hardware-speedup-over-CPU claims.

**Acceptance criteria:**
- `results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json`
  is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `kv260_ssh_reachable`, `kv260_overlay_loaded`,
  `kv260_terminal_transcript_present`,
  `kv260_terminal_condition_confirmed`,
  `kv260_mandate_lift_recommendation`, `polarfire_ssh_reachable`,
  `gatemate_openfpgaloader_installed`, `speedup_claim_avoided_assert`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while the required field stores the bare
  value.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF or CUDA markers.
- If KV260 SSH is reachable, the artifact MUST include the `xmutil listapps`
  probe transcript and set `kv260_overlay_loaded=true` only when a
  `carnot_ising` overlay is present.
- If the Exp 3709 artifact is missing, has fewer than 30 positive samples, or
  lacks board-harness timing evidence, `kv260_terminal_transcript_present` MUST
  be `false`.
- If KV260 SSH is unreachable, the verdict MUST be
  `"complete: kv260_unreachable_polarfire_gatemate_audited"` while PolarFire
  and GateMate are still audited.
- If KV260 SSH is reachable, the Carnot overlay is confirmed, and the Exp 3709
  terminal transcript is present, the verdict MUST be
  `"complete: kv260_terminal_confirmed_mandate_lift_recommended_polarfire_gatemate_audited"`.
- If the consolidated audit completes but KV260 is reachable without satisfying
  the terminal confirmation, the verdict MUST be
  `"complete: hardware_continuity_partial_recorded"`.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`,
  GateMate flash programming, or the hanging GateMate host-visible smoke path.

**Implementation status:** Pending (Exp 3721)

---

### SCENARIO-HW-3721

**Scenario:** Consolidated hardware continuity records terminal, unreachable, and partial states honestly.

**Given:** KV260 terminal confirmation and opportunistic PolarFire/GateMate
continuity are required for milestone .340.
**When:** Experiment 3721 runs the KV260 SSH precondition, checks the KV260
overlay and Exp 3709 transcript when possible, audits PolarFire SSH continuity,
and records GateMate openFPGALoader state without flash or smoke commands.
**Then:** It writes
`results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json`
with one of the terminal-prefixed verdicts, bare required values, field
principles, deterministic seed, checksum, duration, and no forbidden substrate
markers or retired host-storage checks.

**Implementation status:** Pending (Exp 3721)


---

### REQ-HW-3730

**Title:** KV260 opportunistic terminal-state continuity audit uses SSH-only board evidence

**Description:**
Experiment 3730 MUST perform a light, documentation-only KV260 terminal-state
continuity audit after Exp 3709 established the non-fabricated board-latency
terminal transcript and the per-milestone hardware mandate was relaxed to
opportunistic. The ONLY permitted KV260 precondition is SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If the SSH precondition exits non-zero, the experiment MUST stop after
writing an honest blocked artifact with
`honest_verdict="blocked_kv260_ssh_unreachable"` and MUST NOT run `xmutil` or
any other board operation. If SSH is reachable, the experiment MUST run
`ssh kria 'xmutil listapps'` to confirm whether a Carnot accelerator overlay is
present/loadable. If `xmutil` reports that root privileges are required, the
experiment MAY retry the same list operation as `sudo xmutil listapps` over SSH
while preserving the original non-sudo probe. The artifact MUST record whether
the .340 terminal state still holds, or note a regression for the operator when
the accelerator overlay is not listed.

The audit is a hardware smoke check only. It MUST NOT claim live model
inference, thermalization, equilibrium sampling, or hardware speedup.

**Acceptance criteria:**
- `results/experiment_3730_kv260_opportunistic_continuity_audit.json` is
  generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `kv260_ssh_reachable`, `terminal_state_holds`, `preconditions_checked`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while the required field stores the bare
  value.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF, CUDA, or live-inference markers.
- `preconditions_checked` MUST show that the SSH reachability check ran before
  any `xmutil` operation.
- If KV260 SSH is unreachable, `honest_verdict` MUST be exactly
  `"blocked_kv260_ssh_unreachable"`, `kv260_ssh_reachable=false`, and no overlay
  command may run.
- If KV260 SSH is reachable and a `carnot_ising` overlay appears in the
  `xmutil listapps` transcript, `terminal_state_holds=true` and
  `honest_verdict` MUST be exactly
  `"complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"`.
- If KV260 SSH is reachable but no Carnot overlay appears, the artifact MUST
  record `terminal_state_holds=false` and an operator regression note.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, a
  host storage-slot check, a latency-harness rerun, or a live-inference command.

**Implementation status:** Pending (Exp 3730)

---

### SCENARIO-HW-3730

**Scenario:** KV260 opportunistic audit records terminal, overlay-regressed, and SSH-blocked states honestly.

**Given:** KV260 terminal state was established by Exp 3709 and the hardware
mandate is now opportunistic.
**When:** Experiment 3730 runs the SSH precondition and, only when reachable,
checks `xmutil listapps` over SSH for a Carnot accelerator overlay.
**Then:** It writes
`results/experiment_3730_kv260_opportunistic_continuity_audit.json` with the
terminal or blocked verdict, bare required values, field principles,
deterministic seed, checksum, duration, and no retired host-storage or live
inference checks.

**Implementation status:** Pending (Exp 3730)


---

### REQ-HW-3741

**Title:** KV260 opportunistic terminal-state continuity audit reconfirms .340 terminal state through SSH-only board evidence

**Description:**
Experiment 3741 MUST perform a light, documentation-only KV260 terminal-state
continuity audit after Exp 3709 established the non-fabricated board-latency
terminal transcript and Exp 3730 re-confirmed it. The per-milestone hardware
mandate is relaxed to opportunistic, so this audit MUST only check whether the
terminal state still holds. The ONLY permitted KV260 precondition is SSH
reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If the SSH precondition exits non-zero, the experiment MUST stop after
writing an honest blocked artifact with
`honest_verdict="blocked_kv260_ssh_unreachable"` and MUST NOT run `xmutil` or
any other board operation. If SSH is reachable, the experiment MUST run
`ssh kria 'xmutil listapps'` to confirm whether a Carnot accelerator overlay is
present/loadable. If `xmutil` reports that root privileges are required, the
experiment MAY retry the same list operation as `sudo xmutil listapps` over SSH
while preserving the original non-sudo probe. The artifact MUST record whether
the .340 terminal state still holds, or note a regression for the operator when
the accelerator overlay is not listed.

The audit is a hardware smoke check only. It MUST NOT claim live model
inference, thermalization, equilibrium sampling, or hardware speedup.

**Acceptance criteria:**
- `results/experiment_3741_kv260_opportunistic_continuity_audit.json` is
  generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `kv260_ssh_reachable`, `terminal_state_holds`, `preconditions_checked`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while the required field stores the bare
  value.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF, CUDA, or live-inference markers.
- `preconditions_checked` MUST show that the SSH reachability check ran before
  any `xmutil` operation.
- If KV260 SSH is unreachable, `honest_verdict` MUST be exactly
  `"blocked_kv260_ssh_unreachable"`, `kv260_ssh_reachable=false`, and no overlay
  command may run.
- If KV260 SSH is reachable and a `carnot_ising` overlay appears in the
  `xmutil listapps` transcript, `terminal_state_holds=true` and
  `honest_verdict` MUST be exactly
  `"complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"`.
- If KV260 SSH is reachable but no Carnot overlay appears, the artifact MUST
  record `terminal_state_holds=false` and an operator regression note.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, a
  host storage-slot check, a latency-harness rerun, or a live-inference command.

**Implementation status:** Implemented (Exp 3741)

---

### SCENARIO-HW-3741

**Scenario:** KV260 opportunistic audit records terminal, overlay-regressed, and SSH-blocked states honestly.

**Given:** KV260 terminal state was established by Exp 3709, re-confirmed by
Exp 3730, and the hardware mandate is now opportunistic.
**When:** Experiment 3741 runs the SSH precondition and, only when reachable,
checks `xmutil listapps` over SSH for a Carnot accelerator overlay.
**Then:** It writes
`results/experiment_3741_kv260_opportunistic_continuity_audit.json` with the
terminal or blocked verdict, bare required values, field principles,
deterministic seed, checksum, duration, and no retired host-storage or live
inference checks.

**Implementation status:** Implemented (Exp 3741)


---

### REQ-HW-3762

**Title:** KV260 opportunistic continuity audit confirms .340 terminal state still holds through SSH-only board evidence

**Description:**
Experiment 3762 MUST perform a light, documentation-only KV260 terminal-state
continuity audit after Exp 3709 established the non-fabricated board-latency
terminal transcript and Exps 3730 and 3741 re-confirmed it. The per-milestone
hardware mandate is relaxed to opportunistic, so this audit MUST only check
whether the terminal state still holds. The ONLY permitted KV260 precondition is
SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If the SSH precondition exits non-zero, the experiment MUST stop after
writing an honest blocked artifact with
`honest_verdict="blocked_kv260_ssh_unreachable"` and MUST NOT run `xmutil` or
any other board operation. If SSH is reachable, the experiment MUST run
`ssh kria 'xmutil listapps'` to confirm whether a Carnot accelerator overlay is
present/loadable. If `xmutil` reports that root privileges are required, the
experiment MAY retry the same list operation as `sudo xmutil listapps` over SSH
while preserving the original non-sudo probe. The artifact MUST record whether
the .340 terminal state still holds, or note a regression for the operator when
the accelerator overlay is not listed.

The audit is a hardware smoke check only. It MUST NOT claim live model
inference, thermalization, equilibrium sampling, or hardware speedup.

**Acceptance criteria:**
- `results/experiment_3762_kv260_opportunistic_continuity_audit.json` is
  generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `kv260_ssh_reachable`, `terminal_state_holds`, `preconditions_checked`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while the required field stores the bare
  value.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF, CUDA, or live-inference markers.
- `preconditions_checked` MUST show that the SSH reachability check ran before
  any `xmutil` operation.
- If KV260 SSH is unreachable, `honest_verdict` MUST be exactly
  `"blocked_kv260_ssh_unreachable"`, `kv260_ssh_reachable=false`, and no overlay
  command may run.
- If KV260 SSH is reachable and a `carnot_ising` overlay appears in the
  `xmutil listapps` transcript, `terminal_state_holds=true` and
  `honest_verdict` MUST be exactly
  `"complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"`.
- If KV260 SSH is reachable but no Carnot overlay appears, the artifact MUST
  record `terminal_state_holds=false` and an operator regression note.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, a
  host storage-slot check, a latency-harness rerun, or a live-inference command.

**Implementation status:** Pending (Exp 3762)

---

### SCENARIO-HW-3762

**Scenario:** KV260 opportunistic audit records terminal, overlay-regressed, and SSH-blocked states honestly.

**Given:** KV260 terminal state was established by Exp 3709 and re-confirmed by
Exps 3730 and 3741, with the hardware mandate now opportunistic.
**When:** Experiment 3762 runs the SSH precondition and, only when reachable,
checks `xmutil listapps` over SSH for a Carnot accelerator overlay.
**Then:** It writes
`results/experiment_3762_kv260_opportunistic_continuity_audit.json` with the
terminal or blocked verdict, bare required values, field principles,
deterministic seed, checksum, duration, and no retired host-storage or live
inference checks.

**Implementation status:** Pending (Exp 3762)


---

### REQ-HW-3774

**Title:** KV260 opportunistic terminal-state continuity audit reconfirms .340 through SSH-only board evidence after .342/.343 and .344 partial audit

**Description:**
Experiment 3774 MUST perform a light, documentation-only KV260 terminal-state
continuity audit after Exp 3709 established the non-fabricated board-latency
terminal transcript, Exps 3730 and 3741 re-confirmed it, and the .344 partial
Exp 3762 audit checked it again. The per-milestone hardware mandate is relaxed
to opportunistic, so this audit MUST only check whether the terminal state still
holds. The ONLY permitted KV260 precondition is SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If the SSH precondition exits non-zero, the experiment MUST stop after
writing an honest blocked artifact with
`honest_verdict="blocked_kv260_ssh_unreachable"` and MUST NOT run `xmutil` or
any other board operation. If SSH is reachable, the experiment MUST run
`ssh kria 'xmutil listapps'` to confirm whether a Carnot accelerator overlay is
present/loadable. If `xmutil` reports that root privileges are required, the
experiment MAY retry the same list operation as `sudo xmutil listapps` over SSH
while preserving the original non-sudo probe. The artifact MUST record whether
the .340 terminal state still holds, or note a regression for the operator when
the accelerator overlay is not listed.

The audit is a hardware smoke check only. It MUST NOT claim live model
inference, thermalization, equilibrium sampling, or hardware speedup.

**Acceptance criteria:**
- `results/experiment_3774_kv260_opportunistic_continuity_audit.json` is
  generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `kv260_ssh_reachable`, `terminal_state_holds`, `preconditions_checked`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while the required field stores the bare
  value.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF, CUDA, or live-inference markers.
- `preconditions_checked` MUST show that the SSH reachability check ran before
  any `xmutil` operation.
- If KV260 SSH is unreachable, `honest_verdict` MUST be exactly
  `"blocked_kv260_ssh_unreachable"`, `kv260_ssh_reachable=false`, and no overlay
  command may run.
- If KV260 SSH is reachable and a `carnot_ising` overlay appears in the
  `xmutil listapps` transcript, `terminal_state_holds=true` and
  `honest_verdict` MUST be exactly
  `"complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"`.
- If KV260 SSH is reachable but no Carnot overlay appears, the artifact MUST
  record `terminal_state_holds=false` and an operator regression note.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, a
  host storage-slot check, a latency-harness rerun, or a live-inference command.

**Implementation status:** Implemented (Exp 3774)

---

### SCENARIO-HW-3774

**Scenario:** KV260 opportunistic audit records terminal, overlay-regressed, and SSH-blocked states honestly.

**Given:** KV260 terminal state was established by Exp 3709 and re-confirmed by
Exps 3730 and 3741, with the .344 partial Exp 3762 audit adding another
opportunistic confirmation and the hardware mandate now opportunistic.
**When:** Experiment 3774 runs the SSH precondition and, only when reachable,
checks `xmutil listapps` over SSH for a Carnot accelerator overlay.
**Then:** It writes
`results/experiment_3774_kv260_opportunistic_continuity_audit.json` with the
terminal or blocked verdict, bare required values, field principles,
deterministic seed, checksum, duration, and no retired host-storage or live
inference checks.

**Implementation status:** Implemented (Exp 3774)


---

### REQ-HW-3784

**Title:** KV260 opportunistic terminal-state continuity audit confirms .340 still holds after .345 reconfirmation

**Description:**
Experiment 3784 MUST perform a light, documentation-only KV260 terminal-state
continuity audit after Exp 3709 established the non-fabricated board-latency
terminal transcript and Exps 3730, 3741, 3762, and 3774 re-confirmed it
through milestone .345. The per-milestone hardware mandate is relaxed to
opportunistic, so this audit MUST only check whether the terminal state still
holds. The ONLY permitted KV260 precondition is SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If the SSH precondition exits non-zero, the experiment MUST stop after
writing an honest blocked artifact with
`honest_verdict="blocked_kv260_ssh_unreachable"` and MUST NOT run `xmutil` or
any other board operation. If SSH is reachable, the experiment MUST run
`ssh kria 'xmutil listapps'` to confirm whether a Carnot accelerator overlay is
present/loadable. If `xmutil` reports that root privileges are required, the
experiment MAY retry the same list operation as `sudo xmutil listapps` over SSH
while preserving the original non-sudo probe. The artifact MUST record whether
the .340 terminal state still holds, or note a regression for the operator when
the accelerator overlay is not listed.

The audit is a hardware smoke check only. It MUST NOT claim live model
inference, thermalization, equilibrium sampling, or hardware speedup.

**Acceptance criteria:**
- `results/experiment_3784_kv260_opportunistic_continuity_audit.json` is
  generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `kv260_ssh_reachable`, `terminal_state_holds`, `preconditions_checked`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while the required field stores the bare
  value.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF, CUDA, or live-inference markers.
- `preconditions_checked` MUST show that the SSH reachability check ran before
  any `xmutil` operation.
- If KV260 SSH is unreachable, `honest_verdict` MUST be exactly
  `"blocked_kv260_ssh_unreachable"`, `kv260_ssh_reachable=false`, and no overlay
  command may run.
- If KV260 SSH is reachable and a `carnot_ising` overlay appears in the
  `xmutil listapps` transcript, `terminal_state_holds=true` and
  `honest_verdict` MUST be exactly
  `"complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"`.
- If KV260 SSH is reachable but no Carnot overlay appears, the artifact MUST
  record `terminal_state_holds=false` and an operator regression note.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, a
  host storage-slot check, a latency-harness rerun, or a live-inference command.

**Implementation status:** Pending (Exp 3784)

---

### SCENARIO-HW-3784

**Scenario:** KV260 opportunistic audit records terminal, overlay-regressed, and SSH-blocked states honestly.

**Given:** KV260 terminal state was established by Exp 3709 and re-confirmed by
Exps 3730, 3741, 3762, and 3774, with the hardware mandate now opportunistic.
**When:** Experiment 3784 runs the SSH precondition and, only when reachable,
checks `xmutil listapps` over SSH for a Carnot accelerator overlay.
**Then:** It writes
`results/experiment_3784_kv260_opportunistic_continuity_audit.json` with the
terminal or blocked verdict, bare required values, field principles,
deterministic seed, checksum, duration, and no retired host-storage or live
inference checks.

**Implementation status:** Pending (Exp 3784)


---

### REQ-HW-3795

**Title:** KV260 opportunistic terminal-state continuity audit confirms .340 still holds after .346 reconfirmation

**Description:**
Experiment 3795 MUST perform a light, documentation-only KV260 terminal-state
continuity audit after Exp 3709 established the non-fabricated board-latency
terminal transcript and Exps 3730, 3741, 3762, 3774, and 3784 re-confirmed it
through milestone .346. The per-milestone hardware mandate is relaxed to
opportunistic, so this audit MUST only check whether the terminal state still
holds. The ONLY permitted KV260 precondition is SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If the SSH precondition exits non-zero, the experiment MUST stop after
writing an honest blocked artifact with
`honest_verdict="blocked_kv260_ssh_unreachable"` and MUST NOT run `xmutil` or
any other board operation. If SSH is reachable, the experiment MUST run
`ssh kria 'xmutil listapps'` to confirm whether a Carnot accelerator overlay is
present/loadable. If `xmutil` reports that root privileges are required, the
experiment MAY retry the same list operation as `sudo xmutil listapps` over SSH
while preserving the original non-sudo probe. The artifact MUST record whether
the .340 terminal state still holds, or note a regression for the operator when
the accelerator overlay is not listed.

The audit is a hardware smoke check only. It MUST NOT claim live model
inference, thermalization, equilibrium sampling, or hardware speedup.

**Acceptance criteria:**
- `results/experiment_3795_kv260_opportunistic_continuity_audit.json` is
  generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `kv260_ssh_reachable`, `terminal_state_holds`, `preconditions_checked`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while the required field stores the bare
  value.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF, CUDA, or live-inference markers.
- `preconditions_checked` MUST show that the SSH reachability check ran before
  any `xmutil` operation.
- If KV260 SSH is unreachable, `honest_verdict` MUST be exactly
  `"blocked_kv260_ssh_unreachable"`, `kv260_ssh_reachable=false`, and no overlay
  command may run.
- If KV260 SSH is reachable and a `carnot_ising` overlay appears in the
  `xmutil listapps` transcript, `terminal_state_holds=true` and
  `honest_verdict` MUST be exactly
  `"complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"`.
- If KV260 SSH is reachable but no Carnot overlay appears, the artifact MUST
  record `terminal_state_holds=false` and an operator regression note.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, a
  host storage-slot check, a latency-harness rerun, or a live-inference command.

**Implementation status:** Pending (Exp 3795)

---

### SCENARIO-HW-3795

**Scenario:** KV260 opportunistic audit records terminal, overlay-regressed, and SSH-blocked states honestly.

**Given:** KV260 terminal state was established by Exp 3709 and re-confirmed by
Exps 3730, 3741, 3762, 3774, and 3784, with the hardware mandate now
opportunistic.
**When:** Experiment 3795 runs the SSH precondition and, only when reachable,
checks `xmutil listapps` over SSH for a Carnot accelerator overlay.
**Then:** It writes
`results/experiment_3795_kv260_opportunistic_continuity_audit.json` with the
terminal or blocked verdict, bare required values, field principles,
deterministic seed, checksum, duration, and no retired host-storage or live
inference checks.

**Implementation status:** Pending (Exp 3795)


---

### REQ-HW-3806

**Title:** KV260 opportunistic terminal-state continuity audit confirms .340 still holds after .347 reconfirmation

**Description:**
Experiment 3806 MUST perform a light, documentation-only KV260 terminal-state
continuity audit after Exp 3709 established the non-fabricated board-latency
terminal transcript and Exps 3730, 3741, 3762, 3774, 3784, and 3795
re-confirmed it through milestone .347. The per-milestone hardware mandate is
relaxed to opportunistic, so this audit MUST only check whether the terminal
state still holds. The ONLY permitted KV260 precondition is SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If the SSH precondition exits non-zero, the experiment MUST stop after
writing an honest blocked artifact with
`honest_verdict="blocked_kv260_ssh_unreachable"` and MUST NOT run `xmutil` or
any other board operation. If SSH is reachable, the experiment MUST run
`ssh kria 'xmutil listapps'` to confirm whether a Carnot accelerator overlay is
present/loadable. If `xmutil` reports that root privileges are required, the
experiment MAY retry the same list operation as `sudo xmutil listapps` over SSH
while preserving the original non-sudo probe. The artifact MUST record whether
the .340 terminal state still holds, or note a regression for the operator when
the accelerator overlay is not listed.

The audit is a hardware smoke check only. It MUST NOT claim live model
inference, thermalization, equilibrium sampling, or hardware speedup.

**Acceptance criteria:**
- `results/experiment_3806_kv260_opportunistic_continuity_audit.json` is
  generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `kv260_ssh_reachable`, `terminal_state_holds`, `preconditions_checked`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while the required field stores the bare
  value.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF, CUDA, or live-inference markers.
- `preconditions_checked` MUST show that the SSH reachability check ran before
  any `xmutil` operation.
- If KV260 SSH is unreachable, `honest_verdict` MUST be exactly
  `"blocked_kv260_ssh_unreachable"`, `kv260_ssh_reachable=false`, and no overlay
  command may run.
- If KV260 SSH is reachable and a `carnot_ising` overlay appears in the
  `xmutil listapps` transcript, `terminal_state_holds=true` and
  `honest_verdict` MUST be exactly
  `"complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"`.
- If KV260 SSH is reachable but no Carnot overlay appears, the artifact MUST
  record `terminal_state_holds=false` and an operator regression note.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, a
  host storage-slot check, a latency-harness rerun, or a live-inference command.

**Implementation status:** Pending (Exp 3806)

---

### SCENARIO-HW-3806

**Scenario:** KV260 opportunistic audit records terminal, overlay-regressed, and SSH-blocked states honestly.

**Given:** KV260 terminal state was established by Exp 3709 and re-confirmed by
Exps 3730, 3741, 3762, 3774, 3784, and 3795, with the hardware mandate now
opportunistic.
**When:** Experiment 3806 runs the SSH precondition and, only when reachable,
checks `xmutil listapps` over SSH for a Carnot accelerator overlay.
**Then:** It writes
`results/experiment_3806_kv260_opportunistic_continuity_audit.json` with the
terminal or blocked verdict, bare required values, field principles,
deterministic seed, checksum, duration, and no retired host-storage or live
inference checks.

**Implementation status:** Pending (Exp 3806)


---

### REQ-HW-3817

**Title:** KV260 opportunistic terminal-state continuity audit confirms .340 still holds after .348 reconfirmation

**Description:**
Experiment 3817 MUST perform a light, documentation-only KV260 terminal-state
continuity audit after Exp 3709 established the non-fabricated board-latency
terminal transcript and Exps 3730, 3741, 3762, 3774, 3784, 3795, and 3806
re-confirmed it through milestone .348. The per-milestone hardware mandate is
relaxed to opportunistic, so this audit MUST only check whether the terminal
state still holds. The ONLY permitted KV260 precondition is SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be
used. If the SSH precondition exits non-zero, the experiment MUST stop after
writing an honest blocked artifact with
`honest_verdict="blocked_kv260_ssh_unreachable"` and MUST NOT run `xmutil` or
any other board operation. If SSH is reachable, the experiment MUST run
`ssh kria 'xmutil listapps'` to confirm whether a Carnot accelerator overlay is
present/loadable. If `xmutil` reports that root privileges are required, the
experiment MAY retry the same list operation as `sudo xmutil listapps` over SSH
while preserving the original non-sudo probe. The artifact MUST record whether
the .340 terminal state still holds, or note a regression for the operator when
the accelerator overlay is not listed.

The audit is a hardware smoke check only. It MUST NOT claim live model
inference, thermalization, equilibrium sampling, or hardware speedup.

**Acceptance criteria:**
- `results/experiment_3817_kv260_opportunistic_continuity_audit.json` is
  generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `kv260_ssh_reachable`, `terminal_state_holds`, `preconditions_checked`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while the required field stores the bare
  value.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include
  GGUF, CUDA, or live-inference markers.
- `preconditions_checked` MUST show that the SSH reachability check ran before
  any `xmutil` operation.
- If KV260 SSH is unreachable, `honest_verdict` MUST be exactly
  `"blocked_kv260_ssh_unreachable"`, `kv260_ssh_reachable=false`, and no overlay
  command may run.
- If KV260 SSH is reachable and a `carnot_ising` overlay appears in the
  `xmutil listapps` transcript, `terminal_state_holds=true` and
  `honest_verdict` MUST be exactly
  `"complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"`.
- If KV260 SSH is reachable but no Carnot overlay appears, the artifact MUST
  record `terminal_state_holds=false` and an operator regression note.
- No implementation or test path may invoke `/dev/mmcblk*`, `/dev/disk*`, a
  host storage-slot check, a latency-harness rerun, or a live-inference command.

**Implementation status:** Pending (Exp 3817)

---

### SCENARIO-HW-3817

**Scenario:** KV260 opportunistic audit records terminal, overlay-regressed, and SSH-blocked states honestly.

**Given:** KV260 terminal state was established by Exp 3709 and re-confirmed by
Exps 3730, 3741, 3762, 3774, 3784, 3795, and 3806, with the hardware mandate now
opportunistic.
**When:** Experiment 3817 runs the SSH precondition and, only when reachable,
checks `xmutil listapps` over SSH for a Carnot accelerator overlay.
**Then:** It writes
`results/experiment_3817_kv260_opportunistic_continuity_audit.json` with the
terminal or blocked verdict, bare required values, field principles,
deterministic seed, checksum, duration, and no retired host-storage or live
inference checks.

**Implementation status:** Pending (Exp 3817)


---

### REQ-HW-3687

**Title:** PolarFire continuity v24 MUST confirm live SSH continuity for milestone .337

**Description:**
Experiment 3687 MUST perform the milestone PolarFire hardware-task continuity
check. Hardware-Task Continuity requires one PolarFire task per milestone, and
the board was last recorded reachable in .331. The experiment MUST first execute
the SSH precondition:

`ssh -o ConnectTimeout=5 polarfire 'true'`

If that precondition exits zero, the experiment MUST collect live continuity
evidence from the board by recording uptime and the Carnot dispatch path with
distinct, deflagged PolarFire field names. If the precondition exits non-zero,
the experiment MUST write the terminal blocked verdict and MUST NOT claim board
continuity values.

**Acceptance criteria:**
- `results/experiment_3687_polarfire_continuity_v24.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `polarfire_ssh_reachable`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while storing the bare value separately.
- `inference_substrate` MUST be `"hardware_smoke"`.
- If `ssh -o ConnectTimeout=5 polarfire 'true'` succeeds, the verdict MUST be
  `"complete: polarfire_continuity_confirmed_reachable"` and the artifact MUST
  include PolarFire uptime and Carnot dispatch-path values.
- If the SSH precondition fails, the verdict MUST be
  `"complete: blocked_polarfire_ssh_timeout"` and the artifact MUST record the
  failed precondition before reporting the blocked board state.

**Implementation status:** Implemented (Exp 3687)

---

### SCENARIO-HW-3687

**Scenario:** PolarFire continuity v24 records reachable or blocked state from SSH.

**Given:** A PolarFire continuity check is required for milestone .337.
**When:** Experiment 3687 runs the SSH reachability precondition and, only when
reachable, asks the board for uptime and the Carnot dispatch path.
**Then:** It writes `results/experiment_3687_polarfire_continuity_v24.json` with
the terminal verdict, SSH state, precondition record, live continuity values
when reachable, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Implemented (Exp 3687)


---

### REQ-HW-3688

**Title:** GateMate continuity audit v24 records openFPGALoader state without running flash smoke host IO

**Description:**
Experiment 3688 MUST perform the milestone GateMate visibility audit without
running the known hanging flash/smoke host-IO path. The only permitted hardware
probe is the JTAG detect precondition:

1. Run `command -v openFPGALoader`.
2. Only if that command succeeds, run
   `timeout --kill-after=2s 5s openFPGALoader -c dirtyJtag --detect`.
3. Stop after recording the installed/detect state and the known blocker.

The artifact MUST NOT claim a flash or host-visible smoke pass. A successful
GateMate IDCODE detect only proves JTAG visibility for the audit.

**Acceptance criteria:**
- `results/experiment_3688_gatemate_continuity_audit_v24.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `openfpgaloader_installed`, `gatemate_idcode_detected`, `known_blocker`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists and storing the actual artifact value
  separately from the principle.
- `honest_verdict` MUST be
  `"complete: gatemate_continuity_audit_recorded_openfpgaloader_state_flash_smoke_host_io_hang_known_blocker"`.
- `inference_substrate` MUST be `"hardware_smoke"`.
- `openfpgaloader_installed` MUST record whether `command -v openFPGALoader`
  succeeded in the current environment.
- If `command -v openFPGALoader` fails, `openfpgaloader_installed` MUST be
  `false`, `gatemate_idcode_detected` MUST be `null`, and the detect command
  MUST NOT run.
- If the detect command runs and reports a GateMate IDCODE, the artifact MUST
  set `gatemate_idcode_detected=true` while still preserving the
  flash/smoke host-IO hang as a known blocker.
- If the detect command runs and exits non-zero, times out, or does not report
  a GateMate IDCODE, `gatemate_idcode_detected` MUST be `false` and the artifact
  MUST name the missing-tool or detect blocker together with the flash/smoke
  host-IO hang.
- No implementation or test path may invoke GateMate flash programming or the
  hanging host-IO smoke script.

**Implementation status:** Implemented (Exp 3688)

---

### SCENARIO-HW-3688

**Scenario:** GateMate continuity v24 audits JTAG detect state and preserves the host-IO blocker.

**Given:** A GateMate continuity audit is required for milestone .337.
**When:** Experiment 3688 checks for `openFPGALoader`, optionally runs the bounded
DirtyJTAG detect command, and avoids all flash/smoke host-IO commands.
**Then:** It writes `results/experiment_3688_gatemate_continuity_audit_v24.json`
with the terminal verdict, openFPGALoader installed state, IDCODE detect result,
known blocker, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Implemented (Exp 3688)


---

### REQ-HW-3675

**Title:** PolarFire continuity v23 MUST confirm live SSH continuity for milestone .336

**Description:**
Experiment 3675 MUST perform the milestone PolarFire hardware-task continuity
check. Hardware-Task Continuity requires one PolarFire task per milestone, and
the board was last recorded reachable in .331. The experiment MUST first execute
the SSH precondition:

`ssh -o ConnectTimeout=5 polarfire 'true'`

If that precondition exits zero, the experiment MUST collect live continuity
evidence from the board by recording uptime and the Carnot dispatch path with
distinct, deflagged PolarFire field names. If the precondition exits non-zero,
the experiment MUST write the terminal blocked verdict and MUST NOT claim board
continuity values.

**Acceptance criteria:**
- `results/experiment_3675_polarfire_continuity_v23.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `polarfire_ssh_reachable`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists while storing the bare value separately.
- `inference_substrate` MUST be `"hardware_smoke"`.
- If `ssh -o ConnectTimeout=5 polarfire 'true'` succeeds, the verdict MUST be
  `"complete: polarfire_continuity_confirmed_reachable"` and the artifact MUST
  include PolarFire uptime and Carnot dispatch-path values.
- If the SSH precondition fails, the verdict MUST be
  `"complete: blocked_polarfire_ssh_timeout"` and the artifact MUST record the
  failed precondition before reporting the blocked board state.

**Implementation status:** Pending (Exp 3675)

---

### SCENARIO-HW-3675

**Scenario:** PolarFire continuity v23 records reachable or blocked state from SSH.

**Given:** A PolarFire continuity check is required for milestone .336.
**When:** Experiment 3675 runs the SSH reachability precondition and, only when
reachable, asks the board for uptime and the Carnot dispatch path.
**Then:** It writes `results/experiment_3675_polarfire_continuity_v23.json` with
the terminal verdict, SSH state, precondition record, live continuity values
when reachable, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Pending (Exp 3675)


---

### REQ-HW-3676

**Title:** GateMate continuity audit v23 records openFPGALoader state without running flash smoke host IO

**Description:**
Experiment 3676 MUST perform the milestone GateMate visibility audit without
running the known hanging flash/smoke host-IO path. The only permitted hardware
probe is the JTAG detect precondition:

1. Run `command -v openFPGALoader`.
2. Only if that command succeeds, run
   `timeout --kill-after=2s 5s openFPGALoader -c dirtyJtag --detect`.
3. Stop after recording the installed/detect state and the known blocker.

The artifact MUST NOT claim a flash or host-visible smoke pass. A successful
GateMate IDCODE detect only proves JTAG visibility for the audit.

**Acceptance criteria:**
- `results/experiment_3676_gatemate_continuity_audit_v23.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `openfpgaloader_installed`, `gatemate_idcode_detected`, `known_blocker`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists and storing the actual artifact value
  separately from the principle.
- `honest_verdict` MUST be
  `"complete: gatemate_continuity_audit_recorded_openfpgaloader_state_flash_smoke_host_io_hang_known_blocker"`.
- `inference_substrate` MUST be `"hardware_smoke"`.
- `openfpgaloader_installed` MUST record whether `command -v openFPGALoader`
  succeeded in the current environment.
- If `command -v openFPGALoader` fails, `openfpgaloader_installed` MUST be
  `false`, `gatemate_idcode_detected` MUST be `null`, and the detect command
  MUST NOT run.
- If the detect command runs and reports a GateMate IDCODE, the artifact MUST
  set `gatemate_idcode_detected=true` while still preserving the
  flash/smoke host-IO hang as a known blocker.
- If the detect command runs and exits non-zero, times out, or does not report
  a GateMate IDCODE, `gatemate_idcode_detected` MUST be `false` and the artifact
  MUST name the missing-tool or detect blocker together with the flash/smoke
  host-IO hang.
- No implementation or test path may invoke GateMate flash programming or the
  hanging host-IO smoke script.

**Implementation status:** Pending (Exp 3676)

---

### SCENARIO-HW-3676

**Scenario:** GateMate continuity v23 audits JTAG detect state and preserves the host-IO blocker.

**Given:** A GateMate continuity audit is required for milestone 2026.06.336.
**When:** Experiment 3676 checks for `openFPGALoader`, optionally runs the bounded
DirtyJTAG detect command, and avoids all flash/smoke host-IO commands.
**Then:** It writes `results/experiment_3676_gatemate_continuity_audit_v23.json`
with the terminal verdict, openFPGALoader installed state, IDCODE detect result,
known blocker, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Pending (Exp 3676)


---

### REQ-HW-3649

**Title:** PolarFire continuity v21 MUST gate continuity evidence on live SSH reachability

**Description:**
Experiment 3649 MUST perform the milestone PolarFire hardware-task continuity
check for milestone 2026.06.334. PolarFire lives on SSH and was reachable in
milestone .331, so the experiment MUST first execute the SSH precondition:

`ssh -o ConnectTimeout=5 polarfire 'true'`

If that precondition exits zero, the experiment MUST proceed to collect live
continuity evidence from the board by recording uptime and the Carnot dispatch
path with distinct PolarFire field names. If the precondition exits non-zero,
the experiment MUST stop at a terminal blocked artifact and MUST NOT claim
continuity evidence.

**Acceptance criteria:**
- `results/experiment_3649_polarfire_continuity_v21.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `polarfire_ssh_reachable`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists rather than only restating the field name.
- `inference_substrate` MUST be `"hardware_smoke"`.
- If `ssh -o ConnectTimeout=5 polarfire 'true'` succeeds, the verdict MUST be
  `"complete: polarfire_continuity_confirmed_reachable"` and the artifact MUST
  include PolarFire uptime and Carnot dispatch-path values.
- If the SSH precondition fails, the verdict MUST be
  `"complete: blocked_polarfire_ssh_timeout"` and the artifact MUST record the
  failed precondition before reporting the blocked board state.

**Implementation status:** Pending (Exp 3649)

---

### SCENARIO-HW-3649

**Scenario:** PolarFire continuity v21 records reachable or blocked state from SSH.

**Given:** A PolarFire continuity check is required for milestone 2026.06.334.
**When:** Experiment 3649 runs the SSH reachability precondition and, only when
reachable, asks the board for uptime and the Carnot dispatch path.
**Then:** It writes `results/experiment_3649_polarfire_continuity_v21.json` with
the terminal verdict, SSH state, precondition record, live continuity values
when reachable, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Pending (Exp 3649)


---

### REQ-HW-3662

**Title:** PolarFire continuity v22 MUST confirm live SSH continuity for milestone .335

**Description:**
Experiment 3662 MUST perform the milestone PolarFire hardware-task continuity
check for milestone 2026.06.335. Hardware-Task Continuity requires one
PolarFire task per milestone, and the board was last known reachable in .331.
The experiment MUST first execute the SSH precondition:

`ssh -o ConnectTimeout=5 polarfire 'true'`

If that precondition exits zero, the experiment MUST collect live continuity
evidence from the board by recording uptime and the Carnot dispatch path with
distinct PolarFire field names. If the precondition exits non-zero, the
experiment MUST write the terminal blocked verdict and MUST NOT claim board
continuity values.

**Acceptance criteria:**
- `results/experiment_3662_polarfire_continuity_v22.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `preconditions_checked`, `polarfire_ssh_reachable`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists and storing the actual artifact value
  separately from the principle.
- `inference_substrate` MUST be `"hardware_smoke"`.
- If `ssh -o ConnectTimeout=5 polarfire 'true'` succeeds, the verdict MUST be
  `"complete: polarfire_continuity_confirmed_reachable"` and the artifact MUST
  include PolarFire uptime and Carnot dispatch-path values.
- If the SSH precondition fails, the verdict MUST be
  `"complete: blocked_polarfire_ssh_timeout"` and the artifact MUST record the
  failed precondition before reporting the blocked board state.

**Implementation status:** Pending (Exp 3662)

---

### SCENARIO-HW-3662

**Scenario:** PolarFire continuity v22 records reachable or blocked state from SSH.

**Given:** A PolarFire continuity check is required for milestone 2026.06.335.
**When:** Experiment 3662 runs the SSH reachability precondition and, only when
reachable, asks the board for uptime and the Carnot dispatch path.
**Then:** It writes `results/experiment_3662_polarfire_continuity_v22.json` with
the terminal verdict, SSH state, precondition record, live continuity values
when reachable, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Pending (Exp 3662)


---

### REQ-HW-3663

**Title:** GateMate continuity audit v22 records openFPGALoader state without running flash smoke host IO

**Description:**
Experiment 3663 MUST perform the milestone GateMate visibility audit without
running the known hanging flash/smoke host-IO path. The only permitted hardware
probe is the JTAG detect precondition:

1. Run `command -v openFPGALoader`.
2. Only if that command succeeds, run
   `timeout --kill-after=2s 5s openFPGALoader -c dirtyJtag --detect`.
3. Stop after recording the installed/detect state and the known blocker.

The artifact MUST NOT claim a flash or host-visible smoke pass. A successful
GateMate IDCODE detect only proves JTAG visibility for the audit.

**Acceptance criteria:**
- `results/experiment_3663_gatemate_continuity_audit_v22.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `openfpgaloader_installed`, `gatemate_idcode_detected`, `known_blocker`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists and storing the actual artifact value
  separately from the principle.
- `honest_verdict` MUST be
  `"complete: gatemate_continuity_audit_recorded_openfpgaloader_state_flash_smoke_host_io_hang_known_blocker"`.
- `inference_substrate` MUST be `"hardware_smoke"`.
- `openfpgaloader_installed` MUST record whether `command -v openFPGALoader`
  succeeded in the current environment.
- If `command -v openFPGALoader` fails, `openfpgaloader_installed` MUST be
  `false`, `gatemate_idcode_detected` MUST be `null`, and the detect command
  MUST NOT run.
- If the detect command runs and reports a GateMate IDCODE, the artifact MUST
  set `gatemate_idcode_detected=true` while still preserving the
  flash/smoke host-IO hang as a known blocker.
- If the detect command runs and exits non-zero, times out, or does not report
  a GateMate IDCODE, `gatemate_idcode_detected` MUST be `false` and the artifact
  MUST name the missing-tool or detect blocker together with the flash/smoke
  host-IO hang.
- No implementation or test path may invoke GateMate flash programming or the
  hanging host-IO smoke script.

**Implementation status:** Pending (Exp 3663)

---

### SCENARIO-HW-3663

**Scenario:** GateMate continuity v22 audits JTAG detect state and preserves the host-IO blocker.

**Given:** A GateMate continuity audit is required for milestone 2026.06.335.
**When:** Experiment 3663 checks for `openFPGALoader`, optionally runs the bounded
DirtyJTAG detect command, and avoids all flash/smoke host-IO commands.
**Then:** It writes `results/experiment_3663_gatemate_continuity_audit_v22.json`
with the terminal verdict, openFPGALoader installed state, IDCODE detect result,
known blocker, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Pending (Exp 3663)


---

### REQ-HW-3650

**Title:** GateMate continuity audit v21 records openFPGALoader state without running flash smoke host IO

**Description:**
Experiment 3650 MUST perform the milestone GateMate visibility audit without
running the known hanging flash/smoke host-IO path. The only permitted hardware
probe is the JTAG detect precondition:

1. Run `command -v openFPGALoader`.
2. Only if that command succeeds, run
   `timeout --kill-after=2s 5s openFPGALoader -c dirtyJtag --detect`.
3. Stop after recording the installed/detect state and the known blocker.

The artifact MUST NOT claim a flash or host-visible smoke pass. A successful
GateMate IDCODE detect only proves JTAG visibility for the audit.

**Acceptance criteria:**
- `results/experiment_3650_gatemate_continuity_audit_v21.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`,
  `openfpgaloader_installed`, `gatemate_idcode_detected`, `known_blocker`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field,
  documenting why the value exists rather than only restating the field name.
- `honest_verdict` MUST be
  `"complete: gatemate_continuity_audit_recorded_openfpgaloader_state_flash_smoke_host_io_hang_known_blocker"`.
- `inference_substrate` MUST be `"hardware_smoke"`.
- If `command -v openFPGALoader` fails, `openfpgaloader_installed` MUST be
  `false`, `gatemate_idcode_detected` MUST be `null`, and the detect command
  MUST NOT run.
- If the detect command runs and exits non-zero or times out,
  `gatemate_idcode_detected` MUST be `false` and the artifact MUST name the
  missing-tool or detect blocker together with the flash/smoke host-IO hang.
- No implementation or test path may invoke GateMate flash programming or the
  hanging host-IO smoke script.

**Implementation status:** Pending (Exp 3650)

---

### SCENARIO-HW-3650

**Scenario:** GateMate continuity v21 audits JTAG detect state and preserves the host-IO blocker.

**Given:** A GateMate continuity audit is required for milestone 2026.06.334.
**When:** Experiment 3650 checks for `openFPGALoader`, optionally runs the bounded
DirtyJTAG detect command, and avoids all flash/smoke host-IO commands.
**Then:** It writes `results/experiment_3650_gatemate_continuity_audit_v21.json`
with the terminal verdict, openFPGALoader installed state, IDCODE detect result,
known blocker, field principles, deterministic seed, checksum, and duration.

**Implementation status:** Pending (Exp 3650)


---

### REQ-HW-081

**Title:** PolarFire Continuity Check v18

**Description:**
Experiment 3608 MUST perform a continuity check for the PolarFire SoC board. Hardware-Task Continuity requires one PolarFire task per milestone. The experiment MUST first execute `ssh -o ConnectTimeout=5 polarfire 'true'` to verify board reachability. If reachable, it MUST confirm continuity by retrieving board uptime and checking the carnot dispatch path. The result MUST be written to `results/experiment_3608_polarfire_continuity_v18.json` using distinct field names.

**Acceptance criteria:**
- `results/experiment_3608_polarfire_continuity_v18.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate` equal to "hardware_smoke", `preconditions_checked`, `polarfire_ssh_reachable`, `polarfire_uptime`, `polarfire_dispatch_path`, `random_seed`, `reproducibility_checksum`, and `duration_s`.
- If `ssh` succeeds, the verdict MUST be "complete: polarfire_continuity_confirmed_reachable".
- If `ssh` fails, the verdict MUST be "complete: blocked_polarfire_ssh_timeout".

**Implementation status:** Pending (Exp 3608)

---

### SCENARIO-HW-081

**Scenario:** PolarFire continuity check executes reachability and records artifact for v18.

**Given:** PolarFire is checked via SSH.
**When:** Experiment 3608 runs the continuity check script.
**Then:** It writes the results artifact with all required artifact fields, and if reachable, records uptime and dispatch path.

**Implementation status:** Pending (Exp 3608)

---

### REQ-HW-3842

**Title:** KV260 opportunistic terminal-state continuity audit

**Description:**
Experiment 3842 MUST perform a light, documentation-only KV260 terminal-state continuity audit. The per-milestone hardware mandate is relaxed to opportunistic, so this audit MUST only check whether the terminal state still holds. The ONLY permitted KV260 precondition is SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be used. If the SSH precondition exits non-zero, the experiment MUST stop after recording the failure and emit the honest_verdict `blocked_kv260_ssh_unreachable`.
If the SSH precondition exits 0, the experiment MUST run `ssh kria 'xmutil listapps'` (or with sudo if necessary) to confirm the accelerator overlay is listable/loadable.
The audit is a hardware smoke check only. It MUST NOT claim live model inference, thermalization, equilibrium sampling, or hardware speedup.

**Acceptance criteria:**
- `results/experiment_3842_kv260_opportunistic_continuity_audit.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`, `kv260_ssh_reachable`, `accelerator_overlay_loadable`, `preconditions_checked`, `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field, documenting why the value exists while the required field stores the bare value.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include GGUF, CUDA, or live-inference markers.
- `preconditions_checked` MUST show that the SSH reachability check ran before any `xmutil` operation.
- If KV260 SSH is unreachable, `honest_verdict` MUST be exactly `"blocked_kv260_ssh_unreachable"`, `kv260_ssh_reachable=false`, and no overlay command may run.
- If KV260 SSH is reachable and a Carnot overlay appears in the `xmutil listapps` transcript, `accelerator_overlay_loadable=true` and `honest_verdict` MUST be exactly `"complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"`.

**Implementation status:** Pending (Exp 3842)

---

### SCENARIO-HW-3842

**Scenario:** KV260 SSH reachability gating for opportunistic audit.

**Given:** The SSH reachability check is the only permitted gate.
**When:** Experiment 3842 runs the SSH precondition and, only when reachable, checks `xmutil listapps` over SSH for a Carnot accelerator overlay.
**Then:** It writes `results/experiment_3842_kv260_opportunistic_continuity_audit.json` with the terminal or blocked verdict, bare required values, field principles, deterministic seed, checksum, duration, and no retired host-storage or live inference checks.

**Implementation status:** Pending (Exp 3842)

---

### REQ-HW-4017

**Title:** ARC hardware continuity reachability check records all attached boards

**Description:**
Experiment 4017 MUST keep the KV260, GateMate, and PolarFire boards visible
during ARC-focused work by running only the permitted per-board reachability
preconditions and recording honest forward steps. The required preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`
- GateMate: `openFPGALoader -c dirtyJtag --detect`
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`

KV260 MUST NOT use a host SD-card block-device precondition such as `/dev/mmcblk`.
For each reachable board, the artifact MUST include loaded-overlay or detect /
continuity output and the next concrete forward step. For each unreachable board,
the next step MUST be `blocked_<board>_unreachable` while the other boards
continue. KV260 reachable next steps MUST point toward terminal confirmation per
north-star section 3.

**Acceptance criteria:**
- `results/experiment_4017_hardware_continuity.json` is generated.
- The artifact includes bare scalar fields `kv260_reachable`,
  `gatemate_reachable`, and `polarfire_reachable`.
- The artifact includes bare `per_board_next_step`, `per_board_duration_s`,
  `preconditions_checked`, `honest_verdict`, `duration_s`, and
  `inference_substrate` fields plus matching field principles.
- `per_board_duration_s` MUST contain distinct wall-clock measurements for
  KV260, GateMate, and PolarFire.
- `preconditions_checked` MUST list which preconditions ran and their boolean
  availability.
- `honest_verdict` MUST begin with `complete:`, `success:`, or
  `blocked_<resource>`.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and the artifact MUST
  NOT claim fabric acceleration.

**Implementation status:** Pending (Exp 4017)

---

### REQ-HW-4027

**Title:** Hardware continuity MUST record per-board reachability and terminal-state checks

**Description:**
Experiment 4027 MUST produce
`results/experiment_4027_hardware_continuity.json` as a hardware-smoke
continuity artifact for the KV260, GateMate, and PolarFire boards. The
experiment MUST check each board through its board-specific access path before
recording any terminal-state evidence:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`, followed only
  when reachable by `ssh kria 'xmutil listapps'` and `ssh kria 'ls /dev/uio*'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`, followed only when
  reachable by `ssh polarfire 'uname -a && uptime'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`. The artifact MUST record a bare `per_board_reachability` dict
with keys `kv260`, `gatemate`, and `polarfire`; a bare
`preconditions_checked` list whose entries include at least `{resource,
available}`; a bare `per_board_next_step` dict; a bare
`per_board_duration_s` dict with a distinct positive wall-clock timer per
board; `inference_substrate="hardware_smoke"`; and a terminal-prefix
`honest_verdict`.

For every board, the artifact MUST record the terminal-state check result that
was actually observed or mark it skipped when the board access precondition was
unavailable. Reachable KV260 next steps MUST point toward terminal
confirmation; unreachable boards MUST use the exact
`blocked_<board>_unreachable` next step. Field principles MUST annotate the
required fields separately while those required fields remain bare values.

**Acceptance criteria:**
- `results/experiment_4027_hardware_continuity.json` is generated.
- `per_board_reachability` is a dict of booleans for KV260, GateMate, and
  PolarFire and matches the scalar reachability fields.
- `preconditions_checked` records which board accesses were verified before the
  smoke, with at least `{resource, available}` per board, pre-empting the
  fabricate-when-unreachable failure mode.
- `per_board_duration_s` contains distinct positive wall-clock measurements for
  the three boards.
- `honest_verdict` starts with `complete:`, `success:`, or `blocked_`.
- `inference_substrate` is exactly `"hardware_smoke"` and no fabric
  acceleration or host SD-card precondition is claimed.

**Implementation status:** Pending (Exp 4027)

---

### SCENARIO-HW-4027

**Scenario:** Hardware continuity records reachability, terminal-state checks, and next steps per board.

**Given:** KV260, GateMate, and PolarFire may independently be reachable or
unreachable through their SSH or DirtyJTAG preconditions.
**When:** Experiment 4027 runs the hardware-smoke continuity check.
**Then:** It writes `results/experiment_4027_hardware_continuity.json` with
terminal-prefix verdict, `inference_substrate="hardware_smoke"`, bare
`per_board_reachability`, precondition evidence, terminal-state check evidence,
per-board next steps, distinct per-board timers, deterministic seed, checksum,
and no KV260 host SD-card precondition.

**Implementation status:** Pending (Exp 4027)

---

### REQ-HW-4040

**Title:** Hardware continuity MUST drive KV260 toward terminal overlay and latency evidence

**Description:**
Experiment 4040 MUST produce
`results/experiment_4040_hardware_continuity.json` as a hardware-smoke
continuity artifact for KV260, GateMate, and PolarFire. The experiment MUST
check each board independently and continue the remaining boards when one board
is unavailable. The required preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`. If KV260 SSH is reachable, the experiment MUST run
`ssh kria 'xmutil listapps'` and may fall back to `ssh kria 'sudo xmutil listapps'`
when xmutil requires root. If no `carnot_ising` overlay is active in the
transcript, it MUST attempt to load the canonical Carnot overlay with
`ssh kria 'xmutil loadapp carnot_ising_v2_n64'`, falling back to
`ssh kria 'sudo xmutil loadapp carnot_ising_v2_n64'` only when xmutil asks for
root privileges, then re-check listapps. If a Carnot overlay is confirmed, the
experiment MUST take a board-latency transcript step using the KV260 board path
and record the command transcript, sample summary, and whether the step landed.

The artifact MUST include bare `honest_verdict`, `per_board_reachability`,
`kv260_overlay_loaded`, `preconditions_checked`, and `inference_substrate`
fields with separate principle annotations. `kv260_overlay_loaded` MUST be a
boolean terminal-state gate: true only when an active Carnot overlay is
confirmed through the KV260 SSH/xmutil path. `per_board_duration_s` MUST use a
distinct positive wall-clock timer for each board. Unreachable boards MUST use
`blocked_<board>_unreachable` for their next step and terminal state while the
other boards continue. `inference_substrate` MUST be exactly `"hardware_smoke"`,
and the artifact MUST avoid fabric speedup or live inference claims.

**Acceptance criteria:**
- `results/experiment_4040_hardware_continuity.json` is generated.
- `per_board_reachability` is a dict of booleans for KV260, GateMate, and
  PolarFire and matches the scalar reachability fields.
- `kv260_overlay_loaded` is a bare boolean and is true only after a confirmed
  active Carnot overlay appears in a KV260 `xmutil listapps` transcript.
- `preconditions_checked` records the three board preconditions before any
  board smoke step, with at least `{resource, available}` per entry.
- If KV260 is reachable and the overlay is absent initially, a Carnot overlay
  load command is preserved in the KV260 command transcripts before any latency
  transcript step is claimed.
- If KV260 overlay confirmation succeeds, the artifact records a board-latency
  transcript step with positive on-board latency samples or records an honest
  blocked latency state without claiming terminal completion.
- `honest_verdict` starts with `complete:`, `success:`, or `blocked_`;
  `inference_substrate` is exactly `"hardware_smoke"`; and no KV260 host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.

**Implementation status:** Pending (Exp 4040)

---

### SCENARIO-HW-4040

**Scenario:** KV260 continuity loads the Carnot overlay when absent and records a latency transcript step.

**Given:** KV260, GateMate, and PolarFire may independently be reachable or
unreachable through their SSH or DirtyJTAG preconditions.
**When:** Experiment 4040 runs the hardware-smoke continuity check.
**Then:** It writes `results/experiment_4040_hardware_continuity.json` with a
terminal-prefix verdict, `inference_substrate="hardware_smoke"`, bare
`per_board_reachability`, precondition evidence, bare boolean
`kv260_overlay_loaded`, KV260 xmutil/loadapp/latency transcripts when reachable,
per-board next steps, distinct per-board timers, deterministic seed, checksum,
and no KV260 host SD-card precondition.

**Implementation status:** Pending (Exp 4040)

---

### REQ-HW-4052

**Title:** Hardware continuity MUST complete the KV260 latency-transcript terminal step

**Description:**
Experiment 4052 MUST produce
`results/experiment_4052_hardware_continuity.json` as a hardware-smoke
continuity artifact for KV260, GateMate, and PolarFire. The experiment MUST
check each board independently with separate wall-clock timers and continue the
remaining boards when one board is unavailable. The required preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`. If KV260 SSH is reachable, the experiment MUST run
`xmutil listapps` over SSH, may fall back to `sudo xmutil listapps` when xmutil
requires root, MUST confirm or load a `carnot_ising` overlay from a listed
`id_ok` or active handle row, and MUST take the board-level latency-transcript
terminal step by reading a UIO register or timing an energy-eval round trip over
SSH. GateMate and PolarFire MUST record reachability plus the next concrete step
opportunistically.

The artifact MUST include principle-annotated top-level fields
`honest_verdict`, `per_board_reachability`, `kv260_latency_step_taken`,
`kv260_overlay_loaded`, `preconditions_checked`, and
`inference_substrate`. `honest_verdict` MUST use a terminal prefix such as
`complete: hardware_continuity_kv260_latency_transcript_landed_4052` when the
KV260 overlay and latency transcript land. Unreachable boards MUST use
`blocked_<board>_unreachable` for that board and MUST NOT block the other board
checks. `kv260_latency_step_taken` MUST be true only when the KV260 SSH path
records a successful board-latency transcript after overlay confirmation.
`inference_substrate` MUST be exactly `"hardware_smoke"`, and the artifact MUST
avoid fabric speedup or live inference claims.

**Acceptance criteria:**
- `results/experiment_4052_hardware_continuity.json` is generated.
- `per_board_reachability` is a dict of booleans for KV260, GateMate, and
  PolarFire and matches scalar reachability fields.
- `kv260_overlay_loaded` is a bare boolean and is true only after a confirmed
  listed or active Carnot overlay appears in a KV260 `xmutil listapps`
  transcript.
- `kv260_latency_step_taken` is a bare boolean and is true only after a
  successful KV260 board-latency transcript command.
- `preconditions_checked` records the three board preconditions before any
  board smoke step, with at least `{resource, available}` per entry.
- `honest_verdict` starts with `complete:`, `success:`, or `blocked_`;
  terminal KV260 success uses the `kv260_latency_transcript_landed_4052`
  wording; `inference_substrate` is exactly `"hardware_smoke"`; and no KV260
  host SD-card marker such as `/dev/mmcblk` appears in the artifact.

**Implementation status:** Pending (Exp 4052)

---

### SCENARIO-HW-4052

**Scenario:** KV260 latency-transcript terminal step records board evidence while other boards remain visible.

**Given:** KV260, GateMate, and PolarFire may independently be reachable or
unreachable through their SSH or DirtyJTAG preconditions.
**When:** Experiment 4052 runs the hardware-smoke continuity check.
**Then:** It writes `results/experiment_4052_hardware_continuity.json` with a
terminal-prefix verdict, `inference_substrate="hardware_smoke"`, bare
`per_board_reachability`, precondition evidence, bare boolean
`kv260_overlay_loaded`, bare boolean `kv260_latency_step_taken`, KV260
xmutil/loadapp/latency transcripts when reachable, per-board next steps,
distinct per-board timers, deterministic seed, checksum, and no KV260 host
SD-card precondition.

**Implementation status:** Pending (Exp 4052)

---

### REQ-HW-4064

**Title:** Hardware continuity MUST keep KV260 opportunistic and drive GateMate/PolarFire one concrete step

**Description:**
Experiment 4064 MUST produce
`results/experiment_4064_hardware_continuity.json` as a hardware-smoke
continuity artifact for KV260, GateMate, and PolarFire. The experiment MUST
check each board independently with separate wall-clock timers, record all
preconditions before any forward smoke step, and continue the remaining boards
when one board is unavailable. The required preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; after Exp 4052, KV260 is terminal and MUST only receive an
opportunistic SSH confirmation. If GateMate is reachable, the experiment MUST
take one concrete drive-to-terminal step by either flashing an existing n=16
Ising tile bitstream and re-running a DirtyJTAG smoke detect, or by running the
`yosys synth_gatemate -> nextpnr-himbaechel --device CCGM1A1 -> gmpack` build
path. If PolarFire is reachable, the experiment MUST take one concrete
drive-to-terminal step by running a CPU-only Ising/Carnot dispatch smoke with a
hash-match verification. Unreachable boards MUST record
`blocked_<board>_unreachable` for that board and MUST NOT block the other board
checks.

The artifact MUST include principle-annotated top-level fields
`honest_verdict`, `per_board_reachability`, `gatemate_step_taken`,
`polarfire_step_taken`, `kv260_terminal_confirmed`, `preconditions_checked`,
and `inference_substrate`. `honest_verdict` MUST use a terminal prefix such as
`complete: hardware_continuity_gatemate_<...>_polarfire_<...>` and name the
observed GateMate and PolarFire forward-step states. `inference_substrate` MUST
be exactly `"hardware_smoke"`, and the artifact MUST avoid fabric speedup,
model-inference, or KV260 host SD-card claims.

**Acceptance criteria:**
- `results/experiment_4064_hardware_continuity.json` is generated.
- `per_board_reachability` is a dict of booleans for KV260, GateMate, and
  PolarFire and matches scalar reachability fields.
- `preconditions_checked` records the three board preconditions before any
  GateMate or PolarFire smoke step, with at least `{resource, available}` per
  entry.
- `gatemate_step_taken` is a bare string naming the GateMate drive-to-terminal
  step or `blocked_gatemate_unreachable`.
- `polarfire_step_taken` is a bare string naming the PolarFire
  drive-to-terminal step or `blocked_polarfire_unreachable`.
- `kv260_terminal_confirmed` is a bare boolean true only for the SSH-only
  opportunistic confirmation and no KV260 overlay load, latency rerun, or host
  SD-card precondition appears in the artifact.
- `honest_verdict` starts with `complete:` or `blocked_`;
  `inference_substrate` is exactly `"hardware_smoke"`; and no KV260 host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.

**Implementation status:** Pending (Exp 4064)

---

### SCENARIO-HW-4064

**Scenario:** GateMate and PolarFire advance while KV260 remains opportunistic.

**Given:** Exp 4052 established KV260 terminal state, while GateMate and
PolarFire may independently be reachable or unreachable through their DirtyJTAG
or SSH preconditions.
**When:** Experiment 4064 runs the hardware-smoke continuity check.
**Then:** It writes `results/experiment_4064_hardware_continuity.json` with a
terminal-prefix verdict, `inference_substrate="hardware_smoke"`, bare
`per_board_reachability`, precondition evidence recorded before smoke steps,
bare string `gatemate_step_taken`, bare string `polarfire_step_taken`, bare
boolean `kv260_terminal_confirmed`, distinct per-board timers, deterministic
seed, checksum, and no KV260 host SD-card precondition.

**Implementation status:** Pending (Exp 4064)

---

### REQ-HW-4074

**Title:** Hardware continuity MUST keep KV260 opportunistic while driving GateMate and PolarFire forward

**Description:**
Experiment 4074 MUST produce
`results/experiment_4074_hardware_continuity.json` as a hardware-smoke
continuity artifact for KV260, GateMate, and PolarFire. The experiment MUST
read the previous hardware-continuity state and hardware bring-up notes, check
each board independently with separate wall-clock timers, record all
preconditions before any forward smoke step, and continue the remaining boards
when one board is unavailable. The required preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; after Exp 4052 and Exp 4064, KV260 is terminal and MUST only
receive an opportunistic SSH confirmation. If GateMate is reachable, the
experiment MUST take one concrete drive-to-terminal step by either flashing an
existing n=16 Ising tile bitstream and re-running a DirtyJTAG smoke detect, or
by running the `yosys synth_gatemate -> nextpnr-himbaechel --device CCGM1A1 ->
gmpack` build path. If PolarFire is reachable, the experiment MUST take one
concrete drive-to-terminal step by running a CPU-only Ising/Carnot dispatch
smoke with hash-match verification. Unreachable boards MUST record
`blocked_<board>_unreachable` for that board and MUST NOT block the other board
checks.

The artifact MUST include principle-annotated top-level fields
`honest_verdict`, `per_board_reachability`, `gatemate_step_taken`,
`polarfire_step_taken`, `kv260_terminal_confirmed`, `preconditions_checked`,
and `inference_substrate`. `honest_verdict` MUST use a terminal prefix such as
`complete: hardware_continuity_gatemate_<...>_polarfire_<...>` and name the
observed GateMate and PolarFire forward-step states. `inference_substrate` MUST
be exactly `"hardware_smoke"`, and the artifact MUST avoid fabric speedup,
model-inference, or KV260 host SD-card claims.

**Acceptance criteria:**
- `results/experiment_4074_hardware_continuity.json` is generated.
- `per_board_reachability` is a dict of booleans for KV260, GateMate, and
  PolarFire and matches scalar reachability fields.
- `preconditions_checked` records the three board preconditions before any
  GateMate or PolarFire smoke step, with at least `{resource, available}` per
  entry.
- `gatemate_step_taken` is a bare string naming the GateMate drive-to-terminal
  step or `blocked_gatemate_unreachable`.
- `polarfire_step_taken` is a bare string naming the PolarFire
  drive-to-terminal step or `blocked_polarfire_unreachable`.
- `kv260_terminal_confirmed` is a bare boolean true only for the SSH-only
  opportunistic confirmation and no KV260 overlay load, latency rerun, or host
  SD-card precondition appears in the artifact.
- `honest_verdict` starts with `complete:` or `blocked_`;
  `inference_substrate` is exactly `"hardware_smoke"`; and no KV260 host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.

**Implementation status:** Pending (Exp 4074)

---

### SCENARIO-HW-4074

**Scenario:** GateMate and PolarFire advance while KV260 remains opportunistic.

**Given:** Exp 4064 records KV260 as terminal and GateMate/PolarFire as
reachable-but-not-terminal boards, while each board may independently be
reachable or unreachable through its DirtyJTAG or SSH precondition.
**When:** Experiment 4074 runs the hardware-smoke continuity check.
**Then:** It writes `results/experiment_4074_hardware_continuity.json` with a
terminal-prefix verdict, `inference_substrate="hardware_smoke"`, bare
`per_board_reachability`, precondition evidence recorded before smoke steps,
bare string `gatemate_step_taken`, bare string `polarfire_step_taken`, bare
boolean `kv260_terminal_confirmed`, distinct per-board timers, deterministic
seed, checksum, and no KV260 host SD-card precondition.

**Implementation status:** Pending (Exp 4074)

---

### REQ-HW-4084

**Title:** Hardware continuity MUST record per-board reachability and next steps after the GateMate re-plug

**Description:**
Experiment 4084 MUST produce
`results/experiment_4084_hardware_continuity.json` as a hardware-smoke
continuity artifact for KV260, GateMate, and PolarFire. The experiment MUST
read the previous hardware-continuity state, check each board independently
with separate wall-clock timers, record all preconditions before any forward
smoke step, and continue the remaining boards when one board is unavailable.
The required preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`, with the result recorded as
  the post-2026-06-11 re-plug continuity check.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; KV260 is terminal and MUST only receive an opportunistic SSH
confirmation. If GateMate is reachable, the experiment MUST record a concrete
GateMate forward step from the DirtyJTAG detect result and any available
existing n=16 bitstream flash smoke. If PolarFire is reachable, the experiment
MUST record a concrete PolarFire forward step from an SSH-backed CPU dispatch
smoke with hash-match verification. Unreachable boards MUST record
`blocked_<board>_unreachable` for that board and MUST NOT block the other board
checks.

The artifact MUST include principle-annotated top-level fields
`honest_verdict`, `per_board_reachability`, `preconditions_checked`, and
`inference_substrate`, plus the concrete per-board next step for KV260,
GateMate, and PolarFire. `preconditions_checked` MUST be a list of entries
containing at least `{resource, available}` and its principle MUST state that
it records which board accesses were verified before the smoke. `honest_verdict`
MUST use a terminal prefix such as `complete:` or `blocked_`.
`inference_substrate` MUST be exactly `"hardware_smoke"`, and the artifact MUST
avoid fabric speedup, model-inference, or KV260 host SD-card claims.

**Acceptance criteria:**
- `results/experiment_4084_hardware_continuity.json` is generated.
- `per_board_reachability` is a dict of booleans for KV260, GateMate, and
  PolarFire and matches scalar reachability fields.
- `preconditions_checked` records the three board preconditions before any
  GateMate or PolarFire smoke step, with at least `{resource, available}` per
  entry and the required principle text.
- `per_board_next_step` is a dict keyed by KV260, GateMate, and PolarFire that
  records each board's next concrete forward step or
  `blocked_<board>_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`;
  `inference_substrate` is exactly `"hardware_smoke"`; and no KV260 host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.

**Implementation status:** Pending (Exp 4084)

---

### SCENARIO-HW-4084

**Scenario:** Per-board hardware continuity records reachability and next steps.

**Given:** GateMate was re-plugged on 2026-06-11, PolarFire may be reachable
over SSH, and KV260 remains terminal with opportunistic SSH confirmation only.
**When:** Experiment 4084 runs the hardware-smoke continuity check.
**Then:** It writes `results/experiment_4084_hardware_continuity.json` with a
terminal-prefix verdict, `inference_substrate="hardware_smoke"`, bare
`per_board_reachability`, precondition evidence recorded before smoke steps,
bare `per_board_next_step`, distinct per-board timers, deterministic seed,
checksum, and no KV260 host SD-card precondition.

**Implementation status:** Pending (Exp 4084)

---

### REQ-HW-4096

**Title:** Hardware continuity MUST confirm per-board reachability and record the GateMate n=16 flash blocker

**Description:**
Experiment 4096 MUST produce
`results/experiment_4096_hardware_continuity.json` as a hardware-smoke
continuity artifact for KV260, GateMate, and PolarFire. The experiment MUST
read `results/experiment_4084_hardware_continuity.json` as the prior .377 state,
check each board independently with separate wall-clock timers, and record all
preconditions before any forward smoke step. The required preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`, confirming the GM1Ax
  IDCODE (`0x20000001`) before treating the GateMate board as reachable.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; KV260 is terminal and MUST only receive an opportunistic SSH
confirmation. If GateMate is reachable, the experiment MUST investigate the
prior .377 n=16 flash return-code-1 blocker by re-running or reading the
`openFPGALoader -c dirtyJtag -b olimex_gatemateevb <n16 bitstream>` flash
transcript, extracting the concrete flash error, and recording the next concrete
GateMate forward step. If the current GateMate detect does not expose the GM1Ax
IDCODE, the artifact MUST still carry the prior .377 flash error and the next
concrete GateMate step inside `gatemate_step` while leaving
`per_board_next_step["gatemate"]` as `blocked_gatemate_unreachable`. If
PolarFire is reachable, the experiment MUST take the next SSH-backed step toward
end-to-end Carnot dispatch by running the existing hash-verified CPU dispatch
smoke. Unreachable boards MUST record `blocked_<board>_unreachable` for that
board and MUST NOT block the other board checks.

The artifact MUST include principle-annotated top-level fields
`honest_verdict`, `per_board_reachability`, `preconditions_checked`, and
`inference_substrate`, plus concrete per-board next steps. `preconditions_checked`
MUST be a list of entries containing at least `{resource, available}` and its
principle MUST state that it records which board accesses were verified before
the smoke. `honest_verdict` MUST use a terminal prefix such as `complete:` or
`blocked_`. `inference_substrate` MUST be exactly `"hardware_smoke"`, and the
artifact MUST avoid fabric speedup, model-inference, or KV260 host SD-card claims.

**Acceptance criteria:**
- `results/experiment_4096_hardware_continuity.json` is generated.
- `per_board_reachability` is a dict of booleans for KV260, GateMate, and
  PolarFire and matches scalar reachability fields.
- `preconditions_checked` records the three board preconditions before any
  GateMate or PolarFire smoke step, with at least `{resource, available}` per
  entry and the required principle text.
- GateMate reachability requires a GM1Ax/`0x20000001` IDCODE signal from
  `openFPGALoader -c dirtyJtag --detect`.
- A blocked GateMate n=16 flash records `flash_error` and
  `next_concrete_step` inside the GateMate step details; if current GateMate
  reachability is blocked before flash, `gatemate_step` still records
  `previous_377_flash_error` and `next_concrete_step`.
- `per_board_next_step` is a dict keyed by KV260, GateMate, and PolarFire that
  records each board's next concrete forward step or
  `blocked_<board>_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`;
  `inference_substrate` is exactly `"hardware_smoke"`; and no KV260 host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.

**Implementation status:** Pending (Exp 4096)

---

### SCENARIO-HW-4096

**Scenario:** Per-board hardware continuity records reachability, flash error,
and next concrete steps.

**Given:** The .377 hardware-continuity artifact recorded a GateMate n=16 flash
return-code-1 blocker, PolarFire may be reachable over SSH, and KV260 remains
terminal with opportunistic SSH confirmation only.
**When:** Experiment 4096 runs the hardware-smoke continuity check.
**Then:** It writes `results/experiment_4096_hardware_continuity.json` with a
terminal-prefix verdict, `inference_substrate="hardware_smoke"`, bare
`per_board_reachability`, precondition evidence recorded before smoke steps,
bare `per_board_next_step`, distinct per-board timers, a GateMate flash error
and next concrete step when flash is blocked, deterministic seed, checksum, and
no KV260 host SD-card precondition.

**Implementation status:** Pending (Exp 4096)

---

### REQ-HW-4104

**Title:** Hardware continuity MUST record per-board status with SSH/USB-detect preconditions only

**Description:**
Experiment 4104 MUST produce
`results/experiment_4104_hardware_continuity.json` as the next hardware-smoke
continuity artifact for KV260, GateMate, and PolarFire. The experiment MUST keep
one explicit reachability plus next-step record for every board in
`per_board_status`, use distinct wall-clock timers per board, and continue
checking the remaining boards when any one board is unreachable. KV260 is
terminal, so it receives only opportunistic SSH confirmation. GateMate and
PolarFire are non-terminal and MUST each name the next concrete forward step.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's step
MUST be the next concrete step toward the n=16 Ising tile flash from the most
recent GateMate artifact. PolarFire's step MUST be the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and that blocked status is
an honest complete per-board verdict rather than a reason to skip other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. Per-board honest status; an honest blocked_<board> is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten (the failure mode this discipline exists to prevent).`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions actually run; enforces SSH-Not-SD-Card and pre-empts fabricated hardware results.`

**Acceptance criteria:**
- `results/experiment_4104_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, and `duration_s`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- PolarFire reachable runs or records the hash-match Carnot dispatch step; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Pending (Exp 4104)

---

### SCENARIO-HW-4104

**Scenario:** Hardware continuity records one honest status and next step per board.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to be recorded without host SD-card
preconditions.
**When:** Experiment 4104 runs the GateMate DirtyJTAG detect, PolarFire SSH
BatchMode, and KV260 SSH BatchMode preconditions.
**Then:** It writes `results/experiment_4104_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timers, deterministic seed, checksum,
and no KV260 host SD-card precondition.

**Implementation status:** Pending (Exp 4104)

---

### REQ-HW-4113

**Title:** Hardware continuity MUST preserve one SSH/USB-detect reachability and next-step record per board

**Description:**
Experiment 4113 MUST produce
`results/experiment_4113_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. As long as the
boards are attached, every milestone MUST reserve one task record per
non-terminal board. KV260 is terminal and receives only opportunistic SSH
confirmation. GateMate and PolarFire are non-terminal and MUST each name the
next concrete forward step even when another board is unreachable.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash. PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. Per-board honest status; an honest blocked_<board> is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten (the failure mode this discipline exists to prevent).`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions actually run; enforces SSH-Not-SD-Card and pre-empts fabricated hardware results.`

**Acceptance criteria:**
- `results/experiment_4113_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- PolarFire reachable runs or records the hash-match Carnot dispatch step; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Pending (Exp 4113)

---

### SCENARIO-HW-4113

**Scenario:** Hardware continuity preserves per-board timers and next steps.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4113 runs the GateMate DirtyJTAG detect, PolarFire SSH
BatchMode, and KV260 SSH BatchMode preconditions.
**Then:** It writes `results/experiment_4113_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Pending (Exp 4113)

---

### REQ-HW-4123

**Title:** Hardware continuity MUST reserve one SSH/USB-detect status record per attached board

**Description:**
Experiment 4123 MUST produce
`results/experiment_4123_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. As long as the
boards are attached, every milestone MUST reserve one task record per
non-terminal board. KV260 is terminal and receives only opportunistic SSH
confirmation. GateMate and PolarFire are non-terminal and MUST each name the
next concrete forward step even when another board is unreachable.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash. PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. Per-board honest status; an honest blocked_<board> is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card and pre-empts fabricated hardware results.`

**Acceptance criteria:**
- `results/experiment_4123_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- PolarFire reachable runs or records the hash-match Carnot dispatch step; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Pending (Exp 4123)

---

### SCENARIO-HW-4123

**Scenario:** Hardware continuity records per-board reachability and concrete next steps.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4123 runs the GateMate DirtyJTAG detect, PolarFire SSH
BatchMode, and KV260 SSH BatchMode preconditions.
**Then:** It writes `results/experiment_4123_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Pending (Exp 4123)

---

### REQ-HW-4132

**Title:** Hardware continuity MUST continue SSH/USB-only per-board status with distinct board timers

**Description:**
Experiment 4132 MUST produce
`results/experiment_4132_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. As long as the
boards are attached, every milestone MUST reserve one task record per
non-terminal board. KV260 is terminal and receives only opportunistic SSH
confirmation. GateMate and PolarFire are non-terminal and MUST each name the
next concrete forward step even when another board is unreachable.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash. PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. Per-board honest status; an honest blocked_<board> is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card and pre-empts fabricated hardware results.`

**Acceptance criteria:**
- `results/experiment_4132_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- PolarFire reachable runs or records the hash-match Carnot dispatch step; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Pending (Exp 4132)

---

### SCENARIO-HW-4132

**Scenario:** Hardware continuity preserves reachability, timers, and next work per board.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4132 runs the GateMate DirtyJTAG detect, PolarFire SSH
BatchMode, and KV260 SSH BatchMode preconditions.
**Then:** It writes `results/experiment_4132_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Pending (Exp 4132)

---

### REQ-HW-4143

**Title:** Hardware continuity MUST preserve per-board status after Exp 4132

**Description:**
Experiment 4143 MUST produce
`results/experiment_4143_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4132_hardware_continuity.json` as prior context, preserve
one reachability plus next-step record per attached board, and continue with
reachable boards when another board is blocked. KV260 is terminal and receives
only opportunistic SSH confirmation. GateMate and PolarFire are non-terminal
and MUST each name the next concrete forward step.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward investigating the
recurring post-flash detect block before the n=16 Ising tile flash can be
closed. PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. Per-board honest status; an honest blocked_<board> is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card and pre-empts fabricated hardware results.`

**Acceptance criteria:**
- `results/experiment_4143_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- PolarFire reachable runs or records the hash-match Carnot dispatch step; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Pending (Exp 4143)

---

### SCENARIO-HW-4143

**Scenario:** Hardware continuity records SSH/USB reachability and next work per board after Exp 4132.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4143 reads Exp 4132 context, runs the GateMate DirtyJTAG
detect, PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions, and
records the next concrete step per non-terminal board.
**Then:** It writes `results/experiment_4143_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Pending (Exp 4143)

---

### REQ-HW-4154

**Title:** Hardware continuity MUST produce per-board SSH/USB status for Exp 4154

**Description:**
Experiment 4154 MUST produce
`results/experiment_4154_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4143_hardware_continuity.json` as prior context, preserve
one reachability plus next-step record per attached board, and continue checking
reachable boards when another board is blocked. KV260 is terminal and receives
only opportunistic SSH confirmation. GateMate and PolarFire are non-terminal
and MUST each name the next concrete forward step.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash. PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. Per-board honest status; an honest blocked_<board> is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card and pre-empts fabricated hardware results.`

**Acceptance criteria:**
- `results/experiment_4154_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- PolarFire reachable runs or records the hash-match Carnot dispatch step; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4154)

---

### SCENARIO-HW-4154

**Scenario:** Exp 4154 records SSH/USB reachability and concrete next work per board.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4154 reads Exp 4143 context, runs the GateMate DirtyJTAG
detect, PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions, and
records the next concrete step per non-terminal board.
**Then:** It writes `results/experiment_4154_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4154)

---

### REQ-HW-4164

**Title:** Hardware continuity MUST produce per-board SSH/USB status for Exp 4164

**Description:**
Experiment 4164 MUST produce
`results/experiment_4164_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4154_hardware_continuity.json` as prior context, preserve
one reachability plus next-step record per attached board, and continue checking
reachable boards when another board is blocked. KV260 is terminal and receives
only opportunistic SSH confirmation. GateMate and PolarFire are non-terminal
and MUST each name the next concrete forward step.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash. PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. Per-board honest status; an honest blocked_<board> is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card and pre-empts fabricated hardware results.`

**Acceptance criteria:**
- `results/experiment_4164_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- PolarFire reachable runs or records the hash-match Carnot dispatch step; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4164)

---

### SCENARIO-HW-4164

**Scenario:** Exp 4164 records SSH/USB reachability and concrete next work per board.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4164 reads Exp 4154 context, runs the GateMate DirtyJTAG
detect, PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions, and
records the next concrete step per non-terminal board.
**Then:** It writes `results/experiment_4164_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4164)

---

### REQ-HW-4172

**Title:** Hardware continuity MUST produce per-board SSH/USB status for Exp 4172

**Description:**
Experiment 4172 MUST produce
`results/experiment_4172_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4164_hardware_continuity.json` as prior context, preserve
one reachability plus next-step record per attached board, and continue checking
reachable boards when another board is blocked. KV260 is terminal and receives
only opportunistic SSH confirmation. GateMate and PolarFire are non-terminal
and MUST each name the next concrete forward step.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash. PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. Per-board honest status; an honest blocked_<board> is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card.`

**Acceptance criteria:**
- `results/experiment_4172_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- PolarFire reachable runs or records the hash-match Carnot dispatch step; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4172)

---

### SCENARIO-HW-4172

**Scenario:** Exp 4172 records SSH/USB reachability and concrete next work per board.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4172 reads Exp 4164 context, runs the GateMate DirtyJTAG
detect, PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions, and
records the next concrete step per non-terminal board.
**Then:** It writes `results/experiment_4172_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4172)

---

### REQ-HW-4182

**Title:** Hardware continuity MUST produce per-board SSH/USB status for Exp 4182

**Description:**
Experiment 4182 MUST produce
`results/experiment_4182_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4172_hardware_continuity.json` as prior context, preserve
one reachability plus next-step record per attached board, and continue checking
reachable boards when another board is blocked. KV260 is terminal and receives
only opportunistic SSH confirmation. GateMate and PolarFire are non-terminal
and MUST each name the next concrete forward step.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash. PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. An honest blocked_<board> for an unreachable board is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card.`

**Acceptance criteria:**
- `results/experiment_4182_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- PolarFire reachable runs or records the hash-match Carnot dispatch step; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4182)

---

### SCENARIO-HW-4182

**Scenario:** Exp 4182 records SSH/USB reachability and concrete next work per board.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4182 reads Exp 4172 context, runs the GateMate DirtyJTAG
detect, PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions, and
records the next concrete step per non-terminal board.
**Then:** It writes `results/experiment_4182_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4182)

---

### REQ-HW-4194

**Title:** Hardware continuity MUST produce per-board SSH/USB status for Exp 4194

**Description:**
Experiment 4194 MUST produce
`results/experiment_4194_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4182_hardware_continuity.json` as prior context, preserve
one reachability plus next-step record per attached board, and continue checking
reachable boards when another board is blocked. KV260 is terminal and receives
only opportunistic SSH confirmation. GateMate and PolarFire are non-terminal
and MUST each name the next concrete forward step.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash. PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. An honest blocked_<board> for an unreachable board is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card.`

**Acceptance criteria:**
- `results/experiment_4194_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- GateMate reachable runs or records the next concrete step toward flashing the
  n=16 Ising tile; if unreachable, `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- PolarFire reachable runs or records the next concrete step toward end-to-end
  Carnot dispatch with a hash match; if unreachable,
  `per_board_status["polarfire"]["status"]` is `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4194)

---

### SCENARIO-HW-4194

**Scenario:** Exp 4194 records SSH/USB reachability and concrete next work per board.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4194 reads Exp 4182 context, runs the GateMate DirtyJTAG
detect, PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions, and
records the next concrete step per non-terminal board.
**Then:** It writes `results/experiment_4194_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4194)

---

### REQ-HW-4205

**Title:** Hardware continuity MUST produce per-board SSH/USB status for Exp 4205

**Description:**
Experiment 4205 MUST produce
`results/experiment_4205_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4194_hardware_continuity.json` as prior context, preserve
one reachability plus next-step record per attached board, and continue checking
reachable boards when another board is blocked. KV260 is terminal and receives
only opportunistic SSH confirmation. GateMate and PolarFire are non-terminal
and MUST each name the next concrete forward step.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash. PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. An honest blocked_<board> for an unreachable board is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card.`

**Acceptance criteria:**
- `results/experiment_4205_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- GateMate reachable runs or records the next concrete step toward flashing the
  n=16 Ising tile; if unreachable, `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- PolarFire reachable runs or records the next concrete step toward end-to-end
  Carnot dispatch with a hash match; if unreachable,
  `per_board_status["polarfire"]["status"]` is `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4205)

---

### REQ-HW-4334

**Title:** Hardware continuity MUST report opportunistic per-board reachability for Exp 4334

**Description:**
Experiment 4334 MUST produce
`results/experiment_4334_hardware_continuity.json` as an opportunistic
per-board hardware-continuity artifact for KV260, GateMate, and PolarFire. It
MUST read `results/experiment_4322_hardware_continuity.json` as the prior .399
hardware state and preserve the continuity discipline: KV260 is terminal and
SSH-only, PolarFire is opportunistic hash-verified CPU dispatch, and GateMate may
be honestly blocked when DirtyJTAG detect does not expose the board.

The experiment MUST run exactly the per-board reachability preconditions below,
and no board being unreachable may block the milestone:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. This is the
  only valid KV260 precondition; host SD-card block-device checks such as
  `/dev/mmcblk` are forbidden.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).

When KV260 is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and record SSH-only terminal status without making a fabric speedup claim. When
PolarFire is reachable, the experiment MUST run or inject a CPU-dispatch
hash-verify smoke and record the result hash status. When GateMate is detected,
the experiment MUST record the IDCODE; otherwise the GateMate status MUST be
`blocked_gatemate_unreachable`. For any unreachable board, the board's status
MUST be `blocked_<board>_unreachable`, and the experiment MUST continue checking
the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. A per-board reachability report (including honest blocked_<board>) is COMPLETE; no board blocks the milestone.`
- `per_board_status`: `Per-board reachability + any opportunistic result -- keeps the boards visible (continuity) without blocking on operator hardware actions.`
- `preconditions_checked`: `Records the SSH/USB reachability checks (KV260 via SSH NOT SD-card); pre-empts the wrong-mechanism + fabrication modes.`
- `reproducibility_checksum`: `Hash of the per-board probe outputs; auditable.`

**Acceptance criteria:**
- `python3 results/experiment_4334_hardware_continuity.py` writes
  `results/experiment_4334_hardware_continuity.json`.
- `per_board_status` is a dict keyed by `kv260`, `polarfire`, and `gatemate`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records the KV260 SSH BatchMode, PolarFire SSH, and
  GateMate DirtyJTAG detect preconditions; no host SD-card marker such as
  `/dev/mmcblk` appears in the artifact.
- KV260 reachable records `ssh_only_terminal_status` and an `xmutil listapps`
  command transcript; if unreachable, `per_board_status["kv260"]["status"]` is
  `blocked_kv260_unreachable`.
- PolarFire reachable records a CPU-dispatch hash-verify smoke result; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- GateMate reachable records the detected IDCODE; if unreachable,
  `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4334)

---

### REQ-HW-4345

**Title:** Opportunistic hardware continuity MUST record KV260 SSH state and board reachability

**Description:**
Experiment 4345 MUST produce
`results/experiment_4345_hardware_continuity.json` as the opportunistic
hardware-continuity artifact after the `.400` baseline
`results/experiment_4334_hardware_continuity.json`. The experiment MUST treat
KV260 as SSH-only, per north-star §3 and the KV260 SSH-Not-SD-Card discipline:
the only valid KV260 precondition is
`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. Host SD-card block-device checks
are a retired mechanism and MUST NOT appear in the artifact.

If KV260 SSH is unreachable, the experiment MUST write an honest
`blocked_kv260_ssh_unreachable` verdict, record the failed SSH precondition,
set `kv260_reachable` to bare `false`, and stop without fabricating GateMate or
PolarFire state. If KV260 SSH is reachable, the experiment MUST run
`ssh kria 'xmutil listapps'` to record whether the Carnot Ising bitstream is
loadable, then opportunistically probe GateMate with
`openFPGALoader -c dirtyJtag --detect` and PolarFire with
`ssh -o ConnectTimeout=5 polarfire 'true'`. The result MUST preserve board
visibility with a `boards_probed` list, where each entry has `name`,
`reachable`, and `state` fields, and MUST include a board-state transcript of
the commands that were actually run.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. Records the KV260 SSH continuity + opportunistic board reachability (or blocked_kv260_ssh_unreachable -- an honest non-fabrication).`
- `kv260_reachable`: `BARE bool: KV260 SSH-reachable (the sovereignty story's board-state; the SSH precondition, NEVER host SD-card).`
- `boards_probed`: `list: each board {name, reachable, state} -- keeps the boards visible in the milestone retro per the continuity discipline.`
- `preconditions_checked`: `Records the SSH reachability checks (NEVER a host-SD-card device-node check); pre-empts the silent-missing-resource fabrication mode + the retired-mechanism trap.`

**Acceptance criteria:**
- `python3 results/experiment_4345_hardware_continuity.py` writes
  `results/experiment_4345_hardware_continuity.json`.
- If KV260 is unreachable, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, `kv260_reachable` is a bare boolean false,
  and `preconditions_checked` records the KV260 SSH command and exit code.
- If KV260 is reachable, `kv260_reachable` is a bare boolean true,
  `boards_probed` includes KV260, GateMate, and PolarFire entries, and the
  board-state transcript includes `ssh kria 'xmutil listapps'`,
  `openFPGALoader -c dirtyJtag --detect`, and
  `ssh -o ConnectTimeout=5 polarfire 'true'` results.
- `preconditions_checked` records the actual SSH/USB reachability probes that
  were run before any board state is claimed, and no host SD-card marker appears
  in the JSON artifact.
- `honest_verdict` starts with `complete:` for reachable KV260 continuity or is
  exactly `blocked_kv260_ssh_unreachable`.

**Implementation status:** Pending (Exp 4345)

---

### SCENARIO-HW-4345

**Scenario:** Exp 4345 records KV260 SSH continuity and opportunistic board reachability.

**Given:** KV260 is the terminal sovereignty board and must be checked by SSH
only, while GateMate and PolarFire are opportunistic continuity probes.
**When:** Experiment 4345 runs the KV260 SSH precondition and, only when it
succeeds, runs KV260 `xmutil listapps`, GateMate DirtyJTAG detect, and PolarFire
SSH reachability.
**Then:** It writes `results/experiment_4345_hardware_continuity.json` with the
required field principles, bare `kv260_reachable`, `boards_probed`,
`preconditions_checked`, a board-state transcript, deterministic seed,
checksum, no fabric speedup claim, and no KV260 host SD-card precondition.

**Implementation status:** Pending (Exp 4345)

---

### REQ-HW-4356

**Title:** KV260 continuity check MUST use SSH reachability and report loaded bitstream state

**Description:**
Experiment 4356 MUST produce
`results/experiment_4356_hardware_continuity_kv260.json` as a lightweight
KV260-only continuity artifact. Per north-star §3, KV260 remains the
sovereignty story, but this experiment is only an SSH-reachability +
loaded-bitstream continuity check, not a deep hardware build. The only valid
KV260 precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card presence is NEVER a valid precondition for this task; host block
device probes such as `/dev/mmcblk*` MUST NOT appear in the JSON artifact.

If the SSH precondition fails, the experiment MUST write an honest
`blocked_kv260_ssh_unreachable` verdict, set `kv260_reachable` to bare
`false`, populate `preconditions_checked` with the failed SSH command, and stop
without running the overlay or UIO probes. If SSH succeeds, the experiment MUST
record the loaded overlay with:

`ssh kria 'xmutil listapps'`

and the UIO device presence with:

`ssh kria 'ls /dev/uio*'`

The artifact MUST also report `kv260_terminal_state_reached` as a bare boolean.
That boolean is true only when the terminal evidence shows both a
board-latency transcript and `kv260_synthesis_succeeded=true`, matching
north-star §3. Historical terminal evidence may be read from
`results/experiment_2742_kv260_latency_transcript_terminal.json`,
`results/experiment_2465_kv260_rtl_synthesis_fix_v6.json`,
`results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json`, and
`results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json`;
the experiment MUST NOT fabricate terminal status when that evidence is absent.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed (success_kv260_continuity_<state> or blocked_kv260_ssh_unreachable). A reachable continuity check and a clean unreachable skip are BOTH honest terminal states.`
- `kv260_reachable`: `BARE bool: SSH reachability -- the only valid KV260 precondition (NEVER host SD-card presence).`
- `loaded_overlay`: `The xmutil listapps result -- records whether the carnot_ising bitstream is loaded (board continuity).`
- `kv260_terminal_state_reached`: `BARE bool: true iff the board-latency transcript + kv260_synthesis_succeeded terminal state is met (north-star §3) -- after which the per-milestone KV260 mandate lifts.`
- `preconditions_checked`: `Records the SSH-reachability check (NOT host SD card); pre-empts the wrong-mechanism SD-card confusion + the silent-missing-resource fabrication mode.`

**Acceptance criteria:**
- `python3 results/experiment_4356_hardware_continuity_kv260.py` writes
  `results/experiment_4356_hardware_continuity_kv260.json`.
- If KV260 is unreachable, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, `kv260_reachable` is a bare boolean false,
  `preconditions_checked` records the KV260 SSH command and exit code, and no
  overlay/UIO probe is run.
- If KV260 is reachable, `kv260_reachable` is a bare boolean true,
  `loaded_overlay` records the `ssh kria 'xmutil listapps'` transcript,
  `uio_device_presence` records the `ssh kria 'ls /dev/uio*'` transcript, and
  `kv260_terminal_state_reached` is reported as a bare boolean.
- The JSON artifact includes `preconditions_checked`, the required
  `field_principles`, deterministic seed, checksum, `hardware_smoke`
  substrate, no fabric speedup claim, and no KV260 host SD-card precondition.
- `honest_verdict` is either exactly `blocked_kv260_ssh_unreachable` or starts
  with `success_kv260_continuity_`.

**Implementation status:** Pending (Exp 4356)

---

### SCENARIO-HW-4356

**Scenario:** Exp 4356 records KV260 SSH reachability, loaded overlay, and terminal continuity.

**Given:** KV260 is checked only through board SSH reachability and never
through host SD-card presence.
**When:** Experiment 4356 runs the KV260 SSH precondition and, only when it
succeeds, runs `xmutil listapps` and `/dev/uio*` listing over SSH.
**Then:** It writes `results/experiment_4356_hardware_continuity_kv260.json`
with a terminal-prefixed verdict, bare `kv260_reachable`, `loaded_overlay`,
bare `kv260_terminal_state_reached`, `preconditions_checked`, UIO evidence,
deterministic seed, checksum, no speedup claim, and no KV260 host SD-card
precondition.

**Implementation status:** Pending (Exp 4356)

---

### REQ-HW-4367

**Title:** KV260 continuity rerun MUST preserve SSH-only reachability and loaded-bitstream evidence

**Description:**
Experiment 4367 MUST produce
`results/experiment_4367_hardware_continuity_kv260.json` as a lightweight
KV260-only continuity artifact. Per north-star §3, KV260 remains the
sovereignty story until the board-latency transcript plus
`kv260_synthesis_succeeded` terminal state is met, but this experiment is only
an SSH-reachability plus loaded-bitstream continuity check, not a deep hardware
build. The only valid KV260 precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card presence is NEVER a valid precondition for this task; host block
device probes such as `/dev/mmcblk*` MUST NOT appear in the JSON artifact.

If the SSH precondition fails, the experiment MUST write an honest
`blocked_kv260_ssh_unreachable` verdict, set `kv260_reachable` to bare
`false`, populate `preconditions_checked` with the failed SSH command, and stop
without running the overlay or UIO probes. If SSH succeeds, the experiment MUST
record the loaded overlay with:

`ssh kria 'xmutil listapps'`

and the UIO device presence with:

`ssh kria 'ls /dev/uio*'`

The artifact MUST also report `kv260_terminal_state_reached` as a bare boolean.
That boolean is true only when the terminal evidence shows both a
board-latency transcript and `kv260_synthesis_succeeded=true`, matching
north-star §3. The artifact MUST read the prior KV260 continuity context from
`results/experiment_4356_hardware_continuity_kv260.json` when available and
MUST NOT fabricate terminal status when evidence is absent.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed (success_kv260_continuity_<state> or blocked_kv260_ssh_unreachable). A reachable continuity check and a clean unreachable skip are BOTH honest terminal states.`
- `kv260_reachable`: `BARE bool: SSH reachability -- the only valid KV260 precondition (NEVER host SD-card presence).`
- `loaded_overlay`: `The xmutil listapps result -- records whether the carnot_ising bitstream is loaded (board continuity).`
- `kv260_terminal_state_reached`: `BARE bool: true iff the board-latency transcript + kv260_synthesis_succeeded terminal state is met (north-star §3) -- after which the per-milestone KV260 mandate lifts.`
- `preconditions_checked`: `Records the SSH-reachability check (NOT host SD card); pre-empts the wrong-mechanism SD-card confusion + the silent-missing-resource fabrication mode.`

**Acceptance criteria:**
- `python3 results/experiment_4367_hardware_continuity_kv260.py` writes
  `results/experiment_4367_hardware_continuity_kv260.json`.
- If KV260 is unreachable, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, `kv260_reachable` is a bare boolean false,
  `preconditions_checked` records the KV260 SSH command and exit code, and no
  overlay/UIO probe is run.
- If KV260 is reachable, `kv260_reachable` is a bare boolean true,
  `loaded_overlay` records the `ssh kria 'xmutil listapps'` transcript,
  `uio_device_presence` records the `ssh kria 'ls /dev/uio*'` transcript, and
  `kv260_terminal_state_reached` is reported as a bare boolean.
- The JSON artifact includes `preconditions_checked`, the required
  `field_principles`, deterministic seed, checksum, `hardware_smoke`
  substrate, no fabric speedup claim, and no KV260 host SD-card precondition.
- `honest_verdict` is either exactly `blocked_kv260_ssh_unreachable` or starts
  with `success_kv260_continuity_`.

**Implementation status:** Pending (Exp 4367)

---

### SCENARIO-HW-4367

**Scenario:** Exp 4367 records KV260 SSH reachability, loaded overlay, and terminal continuity.

**Given:** KV260 is checked only through board SSH reachability and never
through host SD-card presence.
**When:** Experiment 4367 runs the KV260 SSH precondition and, only when it
succeeds, runs `xmutil listapps` and `/dev/uio*` listing over SSH.
**Then:** It writes `results/experiment_4367_hardware_continuity_kv260.json`
with a terminal-prefixed verdict, bare `kv260_reachable`, `loaded_overlay`,
bare `kv260_terminal_state_reached`, `preconditions_checked`, UIO evidence,
prior Exp 4356 context, deterministic seed, checksum, no speedup claim, and no
KV260 host SD-card precondition.

**Implementation status:** Pending (Exp 4367)

---

### REQ-HW-4378

**Title:** KV260 continuity follow-up MUST use SSH reachability and record loaded-bitstream continuity

**Description:**
Experiment 4378 MUST produce
`results/experiment_4378_hardware_continuity_kv260.json` as a lightweight
KV260-only continuity artifact. Per north-star §3, KV260 remains the
sovereignty story, but this experiment is only an SSH-reachability plus
loaded-bitstream continuity check, not a deep hardware build. The only valid
KV260 precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card presence is NEVER a valid precondition for this task; host block
device probes such as `/dev/mmcblk*` MUST NOT appear in the JSON artifact.

If the SSH precondition fails, the experiment MUST write an honest
`blocked_kv260_ssh_unreachable` verdict, set `kv260_reachable` to bare
`false`, populate `preconditions_checked` with the failed SSH command, and stop
without running the overlay or UIO probes. If SSH succeeds, the experiment MUST
record the loaded overlay with:

`ssh kria 'xmutil listapps'`

and the UIO device presence with:

`ssh kria 'ls /dev/uio*'`

The artifact MUST also report `kv260_terminal_state_reached` as a bare boolean.
That boolean is true only when the terminal evidence shows both a
board-latency transcript and `kv260_synthesis_succeeded=true`, matching
north-star §3. The artifact MUST read the prior KV260 continuity context from
`results/experiment_4367_hardware_continuity_kv260.json` when available and
MUST NOT fabricate terminal status when evidence is absent.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed (success_kv260_continuity_<state> or blocked_kv260_ssh_unreachable). A reachable continuity check and a clean unreachable skip are BOTH honest terminal states.`
- `kv260_reachable`: `BARE bool: SSH reachability -- the only valid KV260 precondition (NEVER host SD-card presence).`
- `loaded_overlay`: `The xmutil listapps result -- records whether the carnot_ising bitstream is loaded (board continuity).`
- `kv260_terminal_state_reached`: `BARE bool: true iff the board-latency transcript + kv260_synthesis_succeeded terminal state is met (north-star §3) -- after which the per-milestone KV260 mandate lifts.`
- `preconditions_checked`: `Records the SSH-reachability check (NOT host SD card); pre-empts the wrong-mechanism SD-card confusion + the silent-missing-resource fabrication mode.`

**Acceptance criteria:**
- `python3 results/experiment_4378_hardware_continuity_kv260.py` writes
  `results/experiment_4378_hardware_continuity_kv260.json`.
- If KV260 is unreachable, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, `kv260_reachable` is a bare boolean false,
  `preconditions_checked` records the KV260 SSH command and exit code, and no
  overlay/UIO probe is run.
- If KV260 is reachable, `kv260_reachable` is a bare boolean true,
  `loaded_overlay` records the `ssh kria 'xmutil listapps'` transcript,
  `uio_device_presence` records the `ssh kria 'ls /dev/uio*'` transcript, and
  `kv260_terminal_state_reached` is reported as a bare boolean.
- The JSON artifact includes `preconditions_checked`, the required
  `field_principles`, deterministic seed, checksum, `hardware_smoke`
  substrate, no fabric speedup claim, and no KV260 host SD-card precondition.
- `honest_verdict` is either exactly `blocked_kv260_ssh_unreachable` or starts
  with `success_kv260_continuity_`.

**Implementation status:** Implemented (Exp 4378)

---

### SCENARIO-HW-4378

**Scenario:** Exp 4378 records KV260 SSH reachability, loaded overlay, and terminal continuity.

**Given:** KV260 is checked only through board SSH reachability and never
through host SD-card presence.
**When:** Experiment 4378 runs the KV260 SSH precondition and, only when it
succeeds, runs `xmutil listapps` and `/dev/uio*` listing over SSH.
**Then:** It writes `results/experiment_4378_hardware_continuity_kv260.json`
with a terminal-prefixed verdict, bare `kv260_reachable`, `loaded_overlay`,
bare `kv260_terminal_state_reached`, `preconditions_checked`, UIO evidence,
prior Exp 4367 context, deterministic seed, checksum, no speedup claim, and no
KV260 host SD-card precondition.

**Implementation status:** Implemented (Exp 4378)

---

### REQ-HW-4389

**Title:** KV260 continuity follow-up MUST use SSH reachability and record loaded-bitstream continuity

**Description:**
Experiment 4389 MUST produce
`results/experiment_4389_hardware_continuity_kv260.json` as a lightweight
KV260-only continuity artifact. Per north-star §3, KV260 remains the
sovereignty story, but this experiment is only an SSH-reachability plus
loaded-bitstream continuity check, not a deep hardware build. The only valid
KV260 precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card presence is NEVER a valid precondition for this task; host block
device probes such as `/dev/mmcblk*` MUST NOT appear in the JSON artifact.

If the SSH precondition fails, the experiment MUST write an honest
`blocked_kv260_ssh_unreachable` verdict, set `kv260_reachable` to bare
`false`, populate `preconditions_checked` with the failed SSH command, and stop
without running the overlay or UIO probes. If SSH succeeds, the experiment MUST
record the loaded overlay with:

`ssh kria 'xmutil listapps'`

and the UIO device presence with:

`ssh kria 'ls /dev/uio*'`

The artifact MUST also report `kv260_terminal_state_reached` as a bare boolean.
That boolean is true only when the terminal evidence shows both a
board-latency transcript and `kv260_synthesis_succeeded=true`, matching
north-star §3. The artifact MUST read the prior KV260 continuity context from
`results/experiment_4378_hardware_continuity_kv260.json` when available and
MUST NOT fabricate terminal status when evidence is absent.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed (success_kv260_continuity_<state> or blocked_kv260_ssh_unreachable). A reachable continuity check and a clean unreachable skip are BOTH honest terminal states.`
- `kv260_reachable`: `BARE bool: SSH reachability -- the only valid KV260 precondition (NEVER host SD-card presence).`
- `loaded_overlay`: `The xmutil listapps result -- records whether the carnot_ising bitstream is loaded (board continuity).`
- `kv260_terminal_state_reached`: `BARE bool: true iff the board-latency transcript + kv260_synthesis_succeeded terminal state is met (north-star §3) -- after which the per-milestone KV260 mandate lifts.`
- `preconditions_checked`: `Records the SSH-reachability check (NOT host SD card); pre-empts the wrong-mechanism SD-card confusion + the silent-missing-resource fabrication mode.`

**Acceptance criteria:**
- `python3 results/experiment_4389_hardware_continuity_kv260.py` writes
  `results/experiment_4389_hardware_continuity_kv260.json`.
- If KV260 is unreachable, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, `kv260_reachable` is a bare boolean false,
  `preconditions_checked` records the KV260 SSH command and exit code, and no
  overlay/UIO probe is run.
- If KV260 is reachable, `kv260_reachable` is a bare boolean true,
  `loaded_overlay` records the `ssh kria 'xmutil listapps'` transcript,
  `uio_device_presence` records the `ssh kria 'ls /dev/uio*'` transcript, and
  `kv260_terminal_state_reached` is reported as a bare boolean.
- The JSON artifact includes `preconditions_checked`, the required
  `field_principles`, deterministic seed, checksum, `hardware_smoke`
  substrate, no fabric speedup claim, and no KV260 host SD-card precondition.
- `honest_verdict` is either exactly `blocked_kv260_ssh_unreachable` or starts
  with `success_kv260_continuity_`.

**Implementation status:** Implemented (Exp 4389)

---

### SCENARIO-HW-4389

**Scenario:** Exp 4389 records KV260 SSH reachability, loaded overlay, and terminal continuity.

**Given:** KV260 is checked only through board SSH reachability and never
through host SD-card presence.
**When:** Experiment 4389 runs the KV260 SSH precondition and, only when it
succeeds, runs `xmutil listapps` and `/dev/uio*` listing over SSH.
**Then:** It writes `results/experiment_4389_hardware_continuity_kv260.json`
with a terminal-prefixed verdict, bare `kv260_reachable`, `loaded_overlay`,
bare `kv260_terminal_state_reached`, `preconditions_checked`, UIO evidence,
prior Exp 4378 context, deterministic seed, checksum, no speedup claim, and no
KV260 host SD-card precondition.

**Implementation status:** Implemented (Exp 4389)

---

### REQ-HW-4400

**Title:** KV260 continuity check MUST preserve SSH-only reachability and loaded-bitstream reporting

**Description:**
Experiment 4400 MUST produce
`results/experiment_4400_hardware_continuity_kv260.json` as a lightweight
KV260-only continuity artifact. Per north-star §3, KV260 remains the
sovereignty story, but this experiment is only an SSH-reachability plus
loaded-bitstream continuity check, not a deep hardware build. The only valid
KV260 precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card presence is NEVER a valid precondition for this task; host block
device probes such as `/dev/mmcblk*` MUST NOT appear in the JSON artifact.

If the SSH precondition fails, the experiment MUST write an honest
`blocked_kv260_ssh_unreachable` verdict, set `kv260_reachable` to bare
`false`, populate `preconditions_checked` with the failed SSH command, and stop
without running the overlay or UIO probes. If SSH succeeds, the experiment MUST
record the loaded overlay with:

`ssh kria 'xmutil listapps'`

and the UIO device presence with:

`ssh kria 'ls /dev/uio*'`

The artifact MUST also report `kv260_terminal_state_reached` as a bare boolean.
That boolean is true only when the terminal evidence shows both a
board-latency transcript and `kv260_synthesis_succeeded=true`, matching
north-star §3. The artifact MUST read the prior KV260 continuity context from
`results/experiment_4389_hardware_continuity_kv260.json` when available and
MUST NOT fabricate terminal status when evidence is absent.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed (success_kv260_continuity_<state> or blocked_kv260_ssh_unreachable). A reachable continuity check and a clean unreachable skip are BOTH honest terminal states.`
- `kv260_reachable`: `BARE bool: SSH reachability -- the only valid KV260 precondition (NEVER host SD-card presence).`
- `loaded_overlay`: `The xmutil listapps result -- records whether the carnot_ising bitstream is loaded (board continuity).`
- `kv260_terminal_state_reached`: `BARE bool: true iff the board-latency transcript + kv260_synthesis_succeeded terminal state is met (north-star §3) -- after which the per-milestone KV260 mandate lifts.`
- `preconditions_checked`: `Records the SSH-reachability check (NOT host SD card); pre-empts the wrong-mechanism SD-card confusion + the silent-missing-resource fabrication mode.`

**Acceptance criteria:**
- `python3 results/experiment_4400_hardware_continuity_kv260.py` writes
  `results/experiment_4400_hardware_continuity_kv260.json`.
- If KV260 is unreachable, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, `kv260_reachable` is a bare boolean false,
  `preconditions_checked` records the KV260 SSH command and exit code, and no
  overlay/UIO probe is run.
- If KV260 is reachable, `kv260_reachable` is a bare boolean true,
  `loaded_overlay` records the `ssh kria 'xmutil listapps'` transcript,
  `uio_device_presence` records the `ssh kria 'ls /dev/uio*'` transcript, and
  `kv260_terminal_state_reached` is reported as a bare boolean.
- The JSON artifact includes `preconditions_checked`, the required
  `field_principles`, deterministic seed, checksum, `hardware_smoke`
  substrate, no fabric speedup claim, and no KV260 host SD-card precondition.
- `honest_verdict` is either exactly `blocked_kv260_ssh_unreachable` or starts
  with `success_kv260_continuity_`.

**Implementation status:** Implemented (Exp 4400)

---

### SCENARIO-HW-4400

**Scenario:** Exp 4400 records KV260 SSH reachability, loaded overlay, and terminal continuity.

**Given:** KV260 is checked only through board SSH reachability and never
through host SD-card presence.
**When:** Experiment 4400 runs the KV260 SSH precondition and, only when it
succeeds, runs `xmutil listapps` and `/dev/uio*` listing over SSH.
**Then:** It writes `results/experiment_4400_hardware_continuity_kv260.json`
with a terminal-prefixed verdict, bare `kv260_reachable`, `loaded_overlay`,
bare `kv260_terminal_state_reached`, `preconditions_checked`, UIO evidence,
prior Exp 4389 context, deterministic seed, checksum, no speedup claim, and no
KV260 host SD-card precondition.

**Implementation status:** Implemented (Exp 4400)

---

### REQ-HW-4411

**Title:** KV260 continuity check MUST preserve SSH-only reachability and loaded-bitstream reporting

**Description:**
Experiment 4411 MUST produce
`results/experiment_4411_hardware_continuity_kv260.json` as a lightweight
KV260-only continuity artifact. Per north-star §3, KV260 remains the
sovereignty story, but this experiment is only an SSH-reachability plus
loaded-bitstream continuity check, not a deep hardware build. The only valid
KV260 precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card presence is NEVER a valid precondition for this task; host block
device probes such as `/dev/mmcblk*` MUST NOT appear in the JSON artifact.

If the SSH precondition fails, the experiment MUST write an honest
`blocked_kv260_ssh_unreachable` verdict, set `kv260_reachable` to bare
`false`, populate `preconditions_checked` with the failed SSH command, and stop
without running the overlay or UIO probes. If SSH succeeds, the experiment MUST
record the loaded overlay with:

`ssh kria 'xmutil listapps'`

and the UIO device presence with:

`ssh kria 'ls /dev/uio*'`

The artifact MUST read the prior KV260 continuity context from
`results/experiment_4400_hardware_continuity_kv260.json` when available and
MUST NOT fabricate terminal status when evidence is absent. If the terminal
board-latency transcript is available, the artifact MUST capture it through the
existing terminal-state evidence fields.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed (success_kv260_continuity_... or blocked_kv260_ssh_unreachable) -- a clean documented skip is honest, NOT a fabrication.`
- `kv260_reachable`: `BARE bool: SSH reachability (the ONLY valid precondition; host SD-card presence is the retired wrong mechanism).`
- `loaded_overlay`: `string: the xmutil listapps overlay + UIO liveness -- the board-state record that keeps the sovereignty board visible without a deep task slot.`
- `preconditions_checked`: `Records the SSH reachability check (NOT a host SD-card check); pre-empts the retired-mechanism + fabrication modes.`
- `random_seed`: `Determinism precondition for any sampled probe ordering.`

**Acceptance criteria:**
- `python3 results/experiment_4411_hardware_continuity_kv260.py` writes
  `results/experiment_4411_hardware_continuity_kv260.json`.
- If KV260 is unreachable, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, `kv260_reachable` is a bare boolean false,
  `preconditions_checked` records the KV260 SSH command and exit code, and no
  overlay/UIO probe is run.
- If KV260 is reachable, `kv260_reachable` is a bare boolean true,
  `loaded_overlay` records the `ssh kria 'xmutil listapps'` transcript,
  `uio_device_presence` records the `ssh kria 'ls /dev/uio*'` transcript, and
  the loaded bitstream/UIO liveness state is reported.
- The JSON artifact includes `preconditions_checked`, the required
  `field_principles`, deterministic seed, checksum, `hardware_smoke`
  substrate, no fabric speedup claim, and no KV260 host SD-card precondition.
- `honest_verdict` is either exactly `blocked_kv260_ssh_unreachable` or starts
  with `success_kv260_continuity_`.

**Implementation status:** Implemented (Exp 4411)

---

### SCENARIO-HW-4411

**Scenario:** Exp 4411 records KV260 SSH reachability, loaded overlay, and UIO liveness.

**Given:** KV260 is checked only through board SSH reachability and never
through host SD-card presence.
**When:** Experiment 4411 runs the KV260 SSH precondition and, only when it
succeeds, runs `xmutil listapps` and `/dev/uio*` listing over SSH.
**Then:** It writes `results/experiment_4411_hardware_continuity_kv260.json`
with a terminal-prefixed verdict, bare `kv260_reachable`, `loaded_overlay`,
`preconditions_checked`, UIO evidence, prior Exp 4400 context, deterministic
seed, checksum, no speedup claim, and no KV260 host SD-card precondition.

**Implementation status:** Implemented (Exp 4411)

---

### REQ-HW-4422

**Title:** KV260 continuity check MUST report SSH reachability, loaded overlay, and UIO liveness as bare fields

**Description:**
Experiment 4422 MUST produce
`results/experiment_4422_hardware_continuity_kv260.json` as an opportunistic
KV260-only continuity artifact. Per north-star section 3, KV260 is the
sovereignty-story board, but this experiment is a lightweight continuity check:
SSH reachability first, then loaded-bitstream and UIO liveness only when SSH is
reachable.

The only valid KV260 precondition is:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card presence is NEVER a valid precondition for this task; host block
device probes such as `/dev/mmcblk*` MUST NOT appear in the JSON artifact.

If the SSH precondition fails, the experiment MUST write an honest
`blocked_kv260_ssh_unreachable` verdict, set `kv260_reachable` to bare
`false`, set `loaded_overlay` to `null`, set `uio_present` to `null`, populate
`preconditions_checked` with the failed SSH command, and stop without running
the overlay or UIO probes. If SSH succeeds, the experiment MUST record the
loaded overlay with:

`ssh kria 'xmutil listapps'`

and the UIO device presence with:

`ssh kria 'ls /dev/uio*'`

The artifact MUST read the prior KV260 continuity context from
`results/experiment_4411_hardware_continuity_kv260.json` when available and
MUST NOT fabricate terminal status, speedup, or fabric acceleration claims.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed (success_kv260_reachable_overlay_<name> or blocked_kv260_ssh_unreachable -- a clean documented skip is honest, not a fabrication).`
- `kv260_reachable`: `BARE bool: SSH-reachability (the ONLY valid KV260 precondition; never host SD-card presence).`
- `loaded_overlay`: `str|null: the xmutil listapps loaded overlay if reachable (the sovereignty-story continuity record).`
- `uio_present`: `bool|null: the carnot_ising bitstream UIO device present if reachable (the liveness check).`
- `preconditions_checked`: `Records the SSH-reachability check (NOT a host SD-card check); pre-empts the retired-mechanism + fabrication modes.`

**Acceptance criteria:**
- `python3 results/experiment_4422_hardware_continuity_kv260.py` writes
  `results/experiment_4422_hardware_continuity_kv260.json`.
- If KV260 is unreachable, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, `kv260_reachable` is a bare boolean false,
  `loaded_overlay` is null, `uio_present` is null, `preconditions_checked`
  records the KV260 SSH command and exit code, and no overlay/UIO probe is run.
- If KV260 is reachable, `kv260_reachable` is a bare boolean true,
  `loaded_overlay` is the parsed `xmutil listapps` overlay string or null when
  no overlay can be parsed, `uio_present` is a bare boolean derived from
  `ssh kria 'ls /dev/uio*'`, and the overlay/UIO command transcripts are
  retained for audit.
- The JSON artifact includes the required `field_principles`, deterministic
  seed, checksum, `hardware_smoke` substrate, prior Exp 4411 source context,
  no fabric speedup claim, and no KV260 host SD-card precondition.
- `honest_verdict` is either exactly `blocked_kv260_ssh_unreachable` or starts
  with `success_kv260_reachable_overlay_`.

**Implementation status:** Implemented (Exp 4422)

---

### SCENARIO-HW-4422

**Scenario:** Exp 4422 records KV260 SSH reachability, loaded overlay, and UIO liveness.

**Given:** KV260 is checked only through board SSH reachability and never
through host SD-card presence.
**When:** Experiment 4422 runs the KV260 SSH precondition and, only when it
succeeds, runs `xmutil listapps` and `/dev/uio*` listing over SSH.
**Then:** It writes `results/experiment_4422_hardware_continuity_kv260.json`
with a terminal-prefixed verdict, bare `kv260_reachable`, `loaded_overlay` as
`str|null`, `uio_present` as `bool|null`, `preconditions_checked`, command
transcripts, prior Exp 4411 context, deterministic seed, checksum, no speedup
claim, and no KV260 host SD-card precondition.

**Implementation status:** Implemented (Exp 4422)

---

### REQ-HW-4439

**Title:** Hardware continuity MUST preserve SSH-gated attached-board visibility for Exp 4439

**Description:**
Experiment 4439 MUST produce
`results/experiment_4439_hardware_continuity.json` as a precondition-gated
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4428_hardware_continuity.json` when available, but every
live board claim remains valid only after the board-specific preconditions
below:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. SSH
  reachability is the ONLY valid KV260 precondition. Retired host SD-card-device
  mechanisms are forbidden.
- GateMate: `command -v nextpnr-himbaechel` and
  `openFPGALoader -c dirtyJtag --detect`. The obsolete `nextpnr-gatemate`
  executable MUST NOT be used.
- PolarFire: `ssh polarfire 'true'`.

For each reachable board, the experiment MUST take exactly one concrete next
step for that board's track: KV260 records a latency transcript, GateMate
performs a bitstream pack step through the himbaechel/gmpack flow, and PolarFire
records a sampler smoke. For any blocked board, the artifact MUST emit that
board's status as `blocked_<resource>` and include a one-line audit instead of
fabricating a step. KV260 SSH failure is terminal for the overall verdict and
MUST produce `honest_verdict="blocked_kv260_ssh_unreachable"` even when other
board preconditions and steps are still recorded.

The artifact MUST include `field_principles` entries for the required fields:

- `preconditions_checked`: `list of {resource, available}; principle: pre-empts the fabricate-when-resource-missing mode.`
- `per_board_status`: `one status per attached board so each stays visible in the milestone retro (Hardware-Task Continuity)`
- `honest_verdict`: `terminal-prefixed; a clean SSH-unreachable is blocked_, an audit-only step is complete_`

**Acceptance criteria:**
- `python3 results/experiment_4439_hardware_continuity.py` writes
  `results/experiment_4439_hardware_continuity.json`.
- `preconditions_checked` is a list whose entries include bare `resource` and
  bare boolean `available` fields for KV260 SSH, GateMate
  nextpnr-himbaechel, GateMate DirtyJTAG detect, and PolarFire SSH.
- KV260 uses only `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; no
  host SD-card marker such as `/dev/mmcblk*` appears in the artifact.
- GateMate uses `nextpnr-himbaechel` and `openFPGALoader -c dirtyJtag --detect`
  and never uses `nextpnr-gatemate`.
- PolarFire uses `ssh polarfire 'true'`.
- Reachable boards record a concrete step transcript (`latency_transcript`,
  `bitstream_pack`, or `sampler_smoke`); blocked boards record
  `blocked_<resource>` plus one audit sentence.
- `honest_verdict` starts with `blocked_kv260_ssh_unreachable` when KV260 SSH
  is unreachable, otherwise with `complete:` once the attached-board audit is
  written.

**Implementation status:** Pending (Exp 4439)

---

### REQ-HW-4451

**Title:** Hardware continuity MUST produce a precondition-gated attached-board artifact for Exp 4451

**Description:**
Experiment 4451 MUST produce
`results/experiment_4451_hardware_continuity.json` as a precondition-gated
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4439_hardware_continuity.json` when available, but every
live board claim remains valid only after the board-specific preconditions
below:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. SSH
  reachability is the ONLY valid KV260 precondition. Retired host SD-card-device
  mechanisms are forbidden.
- GateMate: `command -v nextpnr-himbaechel` and
  `openFPGALoader -c dirtyJtag --detect`. The obsolete `nextpnr-gatemate`
  executable MUST NOT be used.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

For each reachable board, the experiment MUST take one concrete next step for
that board's track: KV260 records a latency transcript, GateMate performs a
himbaechel/gmpack bitstream pack step toward the P&R/flash-smoke track, and
PolarFire records a sampler smoke. For any blocked board, the artifact MUST
emit that board's status as `blocked_<resource>` and include a one-line audit
instead of fabricating a step. KV260 SSH failure is terminal for the overall
verdict and MUST produce `honest_verdict="blocked_kv260_ssh_unreachable"` even
when other board preconditions and steps are still recorded.

The artifact MUST include `field_principles` entries for the required fields:

- `preconditions_checked`: `list of {resource, available}; pre-empts the fabricate-when-resource-missing mode`
- `per_board_status`: `one status per attached board so each stays visible in the milestone retro (Hardware-Task Continuity)`
- `honest_verdict`: `terminal-prefixed; a clean SSH-unreachable is blocked_, an audit-only step is complete_`
- `inference_substrate`: `hardware_smoke -- SSH-attached board test; per-board floor`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4451_hardware_continuity.py`
  writes `results/experiment_4451_hardware_continuity.json`.
- `preconditions_checked` is a list whose entries include bare `resource` and
  bare boolean `available` fields for KV260 SSH, GateMate
  nextpnr-himbaechel, GateMate DirtyJTAG detect, and PolarFire SSH.
- KV260 uses only `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; no
  host SD-card marker such as `/dev/mmcblk*` appears in the artifact.
- GateMate uses `nextpnr-himbaechel` and `openFPGALoader -c dirtyJtag --detect`
  and never uses `nextpnr-gatemate`.
- PolarFire uses `ssh -o ConnectTimeout=5 polarfire 'true'`.
- Reachable boards record a concrete step transcript (`latency_transcript`,
  `bitstream_pack`, or `sampler_smoke`); blocked boards record
  `blocked_<resource>` plus one audit sentence.
- `inference_substrate` is `hardware_smoke`.
- `honest_verdict` starts with `blocked_kv260_ssh_unreachable` when KV260 SSH
  is unreachable, otherwise with `complete:` once the attached-board audit is
  written.

**Implementation status:** Implemented (Exp 4451)

---

### SCENARIO-HW-4451

**Scenario:** Exp 4451 checks all attached-board continuity without fabricating missing hardware.

**Given:** KV260, GateMate, and PolarFire may each be reachable or blocked, and
the conductor requires one forward step or one precondition-gated audit per
attached board.
**When:** Experiment 4451 runs the KV260 SSH precondition, the GateMate
nextpnr-himbaechel plus DirtyJTAG preconditions, and the PolarFire SSH
precondition, then runs only the reachable boards' next-step commands.
**Then:** It writes `results/experiment_4451_hardware_continuity.json` with
required field principles, `preconditions_checked`, `per_board_status`,
`honest_verdict`, `inference_substrate="hardware_smoke"`, command transcripts,
deterministic seed, checksum, prior Exp 4439 context, no speedup claim, no host
SD-card precondition, and no `nextpnr-gatemate` command.

**Implementation status:** Implemented (Exp 4451)

---

### REQ-HW-4463

**Title:** Hardware continuity MUST produce a precondition-gated attached-board artifact for Exp 4463

**Description:**
Experiment 4463 MUST produce
`results/experiment_4463_hardware_continuity.json` as a precondition-gated
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4451_hardware_continuity.json` when available, but every
live board claim remains valid only after the board-specific preconditions
below:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. SSH
  reachability is the ONLY valid KV260 precondition. Retired host SD-card-device
  mechanisms are forbidden.
- GateMate: `command -v nextpnr-himbaechel` and
  `openFPGALoader -c dirtyJtag --detect`. The obsolete `nextpnr-gatemate`
  executable MUST NOT be used.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

For each reachable board, the experiment MUST take one concrete next step for
that board's track: KV260 records a latency transcript, GateMate performs a
himbaechel/gmpack bitstream pack step toward the P&R/flash-smoke track, and
PolarFire records a sampler smoke. For any blocked board, the artifact MUST
emit that board's status as `blocked_<resource>` and include a one-line audit
instead of fabricating a step. KV260 SSH failure is terminal for the overall
verdict and MUST produce `honest_verdict="blocked_kv260_ssh_unreachable"` even
when other board preconditions and steps are still recorded.

The artifact MUST include `field_principles` entries for the required fields:

- `preconditions_checked`: `list of {resource, available}; pre-empts the fabricate-when-resource-missing mode`
- `per_board_status`: `one status per attached board so each stays visible in the milestone retro (Hardware-Task Continuity)`
- `honest_verdict`: `terminal-prefixed; a clean SSH-unreachable is blocked_, an audit-only step is complete_`
- `inference_substrate`: `hardware_smoke -- SSH-attached board test; per-board floor`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4463_hardware_continuity.py`
  writes `results/experiment_4463_hardware_continuity.json`.
- `preconditions_checked` is a list whose entries include bare `resource` and
  bare boolean `available` fields for KV260 SSH, GateMate
  nextpnr-himbaechel, GateMate DirtyJTAG detect, and PolarFire SSH.
- KV260 uses only `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; no
  host SD-card marker such as `/dev/mmcblk*` appears in the artifact.
- GateMate uses `nextpnr-himbaechel` and `openFPGALoader -c dirtyJtag --detect`
  and never uses `nextpnr-gatemate`.
- PolarFire uses `ssh -o ConnectTimeout=5 polarfire 'true'`.
- Reachable boards record a concrete step transcript (`latency_transcript`,
  `bitstream_pack`, or `sampler_smoke`); blocked boards record
  `blocked_<resource>` plus one audit sentence.
- `inference_substrate` is `hardware_smoke`.
- `honest_verdict` starts with `blocked_kv260_ssh_unreachable` when KV260 SSH
  is unreachable, otherwise with `complete:` once the attached-board audit is
  written.

**Implementation status:** Implemented (Exp 4463)

---

### SCENARIO-HW-4463

**Scenario:** Exp 4463 checks all attached-board continuity without fabricating missing hardware.

**Given:** KV260, GateMate, and PolarFire may each be reachable or blocked, and
the conductor requires one forward step or one precondition-gated audit per
attached board.
**When:** Experiment 4463 runs the KV260 SSH precondition, the GateMate
nextpnr-himbaechel plus DirtyJTAG preconditions, and the PolarFire SSH
precondition, then runs only the reachable boards' next-step commands.
**Then:** It writes `results/experiment_4463_hardware_continuity.json` with
required field principles, `preconditions_checked`, `per_board_status`,
`honest_verdict`, `inference_substrate="hardware_smoke"`, command transcripts,
deterministic seed, checksum, prior Exp 4451 context, no speedup claim, no host
SD-card precondition, and no `nextpnr-gatemate` command.

**Implementation status:** Implemented (Exp 4463)

---

### REQ-HW-4476

**Title:** Hardware continuity MUST produce a precondition-gated attached-board artifact for Exp 4476

**Description:**
Experiment 4476 MUST produce
`results/experiment_4476_hardware_continuity.json` as a precondition-gated
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4463_hardware_continuity.json` when available, but every
live board claim remains valid only after the board-specific preconditions
below:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. SSH
  reachability is the ONLY valid KV260 precondition. Retired host SD-card-device
  mechanisms are forbidden.
- GateMate: `command -v nextpnr-himbaechel` and
  `openFPGALoader -c dirtyJtag --detect`. The obsolete `nextpnr-gatemate`
  executable MUST NOT be used.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

For each reachable board, the experiment MUST take one concrete next step for
that board's track: KV260 records a latency transcript, GateMate performs a
himbaechel/gmpack bitstream pack step toward the P&R/flash-smoke track, and
PolarFire records a sampler smoke. For any blocked board, the artifact MUST
emit that board's status as `blocked_<resource>` and include a one-line audit
instead of fabricating a step. KV260 SSH failure is terminal for the overall
verdict and MUST produce `honest_verdict="blocked_kv260_ssh_unreachable"` even
when other board preconditions and steps are still recorded.

The artifact MUST include `field_principles` entries for the required fields:

- `preconditions_checked`: `list of {resource, available}; pre-empts the fabricate-when-resource-missing mode`
- `per_board_status`: `one status per attached board so each stays visible in the milestone retro (Hardware-Task Continuity)`
- `honest_verdict`: `terminal-prefixed; a clean SSH-unreachable is blocked_, an audit-only step is complete_`
- `inference_substrate`: `hardware_smoke -- SSH-attached board test; per-board floor`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4476_hardware_continuity.py`
  writes `results/experiment_4476_hardware_continuity.json`.
- `preconditions_checked` is a list whose entries include bare `resource` and
  bare boolean `available` fields for KV260 SSH, GateMate
  nextpnr-himbaechel, GateMate DirtyJTAG detect, and PolarFire SSH.
- KV260 uses only `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; no
  host SD-card marker such as `/dev/mmcblk*` appears in the artifact.
- GateMate uses `nextpnr-himbaechel` and `openFPGALoader -c dirtyJtag --detect`
  and never uses `nextpnr-gatemate`.
- PolarFire uses `ssh -o ConnectTimeout=5 polarfire 'true'`.
- Reachable boards record a concrete step transcript (`latency_transcript`,
  `bitstream_pack`, or `sampler_smoke`); blocked boards record
  `blocked_<resource>` plus one audit sentence.
- `inference_substrate` is `hardware_smoke`.
- `honest_verdict` starts with `blocked_kv260_ssh_unreachable` when KV260 SSH
  is unreachable, otherwise with `complete:` once the attached-board audit is
  written.

**Implementation status:** Implemented (Exp 4476)

---

### SCENARIO-HW-4476

**Scenario:** Exp 4476 checks all attached-board continuity without fabricating missing hardware.

**Given:** KV260, GateMate, and PolarFire may each be reachable or blocked, and
the conductor requires one forward step or one precondition-gated audit per
attached board.
**When:** Experiment 4476 runs the KV260 SSH precondition, the GateMate
nextpnr-himbaechel plus DirtyJTAG preconditions, and the PolarFire SSH
precondition, then runs only the reachable boards' next-step commands.
**Then:** It writes `results/experiment_4476_hardware_continuity.json` with
required field principles, `preconditions_checked`, `per_board_status`,
`honest_verdict`, `inference_substrate="hardware_smoke"`, command transcripts,
deterministic seed, checksum, prior Exp 4463 context, no speedup claim, no host
SD-card precondition, and no `nextpnr-gatemate` command.

**Implementation status:** Implemented (Exp 4476)

---

### SCENARIO-HW-4439

**Scenario:** Exp 4439 checks all attached-board continuity without fabricating missing hardware.

**Given:** KV260, GateMate, and PolarFire may each be reachable or blocked, and
the conductor requires one forward step or one precondition-gated audit per
attached board.
**When:** Experiment 4439 runs the KV260 SSH precondition, the GateMate
nextpnr-himbaechel plus DirtyJTAG preconditions, and the PolarFire SSH
precondition, then runs only the reachable boards' next-step commands.
**Then:** It writes `results/experiment_4439_hardware_continuity.json` with
required field principles, `preconditions_checked`, `per_board_status`,
`honest_verdict`, command transcripts, deterministic seed, checksum, prior Exp
4428 context, no speedup claim, no host SD-card precondition, and no
`nextpnr-gatemate` command.

**Implementation status:** Pending (Exp 4439)

---

### REQ-HW-4428

**Title:** Hardware continuity MUST produce precondition-gated per-board status for Exp 4428

**Description:**
Experiment 4428 MUST produce
`results/experiment_4428_hardware_continuity.json` as the next hardware
continuity artifact for KV260, GateMate, and PolarFire. The artifact MUST read
`results/experiment_4422_hardware_continuity_kv260.json` when available, but
live board claims are valid only after the board-specific preconditions below:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. SSH
  reachability is the ONLY valid KV260 precondition. Host SD-card-device
  mechanisms are forbidden.
- GateMate: `command -v nextpnr-himbaechel` and
  `openFPGALoader -c dirtyJtag --detect`. The obsolete `nextpnr-gatemate`
  executable MUST NOT be used.
- PolarFire: `ssh polarfire 'true'`.

For each reachable board, the experiment MUST take exactly one concrete next
step for that board's current track: KV260 records a latency transcript,
GateMate performs a bitstream pack step through the himbaechel/gmpack flow, and
PolarFire records a sampler smoke. For any unreachable board, the artifact MUST
emit that board's status as `blocked_<resource>` and include a one-line audit
instead of fabricating a step. In particular, KV260 SSH failure MUST produce
`blocked_kv260_ssh_unreachable`.

The artifact MUST include `field_principles` entries for the required fields:

- `preconditions_checked`: `list of {resource, available}; principle: pre-empts the fabricate-when-resource-missing mode.`
- `per_board_status`: `One status record per board with reachable/blocked outcome and the concrete next step or one-line audit.`
- `honest_verdict`: `Terminal-prefixed. A complete multi-board continuity result or blocked_<resource> outcome is honest when precondition-gated.`

**Acceptance criteria:**
- `python3 results/experiment_4428_hardware_continuity.py` writes
  `results/experiment_4428_hardware_continuity.json`.
- `preconditions_checked` is a list whose entries include bare `resource` and
  bare boolean `available` fields for KV260 SSH, GateMate
  nextpnr-himbaechel, GateMate DirtyJTAG detect, and PolarFire SSH.
- KV260 uses only `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; no
  host SD-card marker such as `/dev/mmcblk*` appears in the artifact.
- GateMate uses `nextpnr-himbaechel` and `openFPGALoader -c dirtyJtag --detect`
  and never uses `nextpnr-gatemate`.
- PolarFire uses `ssh polarfire 'true'`.
- Reachable boards record a concrete step transcript (`latency_transcript`,
  `bitstream_pack`, or `sampler_smoke`); blocked boards record
  `blocked_<resource>` plus one audit sentence.
- `honest_verdict` starts with `complete:` when at least one board is reachable,
  otherwise with `blocked_`.

**Implementation status:** Pending (Exp 4428)

---

### SCENARIO-HW-4428

**Scenario:** Exp 4428 checks all attached-board continuity without fabricating missing hardware.

**Given:** KV260, GateMate, and PolarFire may each be reachable or blocked, and
the conductor requires one forward step or one precondition-gated audit per
board.
**When:** Experiment 4428 runs the KV260 SSH precondition, the GateMate
nextpnr-himbaechel plus DirtyJTAG preconditions, and the PolarFire SSH
precondition, then runs only the reachable boards' next-step commands.
**Then:** It writes `results/experiment_4428_hardware_continuity.json` with
`preconditions_checked`, `per_board_status`, `honest_verdict`, command
transcripts, deterministic seed, checksum, prior Exp 4422 context, no speedup
claim, no host SD-card precondition, and no `nextpnr-gatemate` command.

**Implementation status:** Pending (Exp 4428)

---

### SCENARIO-HW-4334

**Scenario:** Exp 4334 reports KV260, PolarFire, and GateMate continuity without milestone blocking.

**Given:** KV260 is terminal and SSH-only per north-star §3, PolarFire and
GateMate are opportunistic, and the milestone requires board continuity to use
only SSH/USB reachability checks.
**When:** Experiment 4334 reads Exp 4322 context, runs the KV260 SSH, PolarFire
SSH, and GateMate DirtyJTAG detect preconditions, performs reachable-board
opportunistic checks, and records honest `blocked_<board>` statuses for
unreachable boards.
**Then:** It writes `results/experiment_4334_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4334)

---

### REQ-HW-4322

**Title:** Hardware continuity MUST report opportunistic per-board reachability for Exp 4322

**Description:**
Experiment 4322 MUST produce
`results/experiment_4322_hardware_continuity.json` as an opportunistic
per-board hardware-continuity artifact for KV260, GateMate, and PolarFire. It
MUST read `results/experiment_4311_hardware_continuity.json` as the prior .398
hardware state and preserve the continuity discipline: KV260 is terminal and
SSH-only, PolarFire is opportunistic hash-verified CPU dispatch, and GateMate may
be honestly blocked when DirtyJTAG detect does not expose the board.

The experiment MUST run exactly the per-board reachability preconditions below,
and no board being unreachable may block the milestone:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. This is the
  only valid KV260 precondition; host SD-card block-device checks such as
  `/dev/mmcblk` are forbidden.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).

When KV260 is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and record SSH-only terminal status without making a fabric speedup claim. When
PolarFire is reachable, the experiment MUST run or inject a CPU-dispatch
hash-verify smoke and record the result hash status. When GateMate is detected,
the experiment MUST record the IDCODE; otherwise the GateMate status MUST be
`blocked_gatemate_unreachable`. For any unreachable board, the board's status
MUST be `blocked_<board>_unreachable`, and the experiment MUST continue checking
the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. A per-board reachability report (including honest blocked_<board>) is COMPLETE; no board blocks the milestone.`
- `per_board_status`: `Per-board reachability + any opportunistic result -- keeps the boards visible (continuity) without blocking on operator hardware actions.`
- `preconditions_checked`: `Records the SSH/USB reachability checks (KV260 via SSH NOT SD-card); pre-empts the wrong-mechanism + fabrication modes.`
- `reproducibility_checksum`: `Hash of the per-board probe outputs; auditable.`

**Acceptance criteria:**
- `python3 results/experiment_4322_hardware_continuity.py` writes
  `results/experiment_4322_hardware_continuity.json`.
- `per_board_status` is a dict keyed by `kv260`, `polarfire`, and `gatemate`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records the KV260 SSH BatchMode, PolarFire SSH, and
  GateMate DirtyJTAG detect preconditions; no host SD-card marker such as
  `/dev/mmcblk` appears in the artifact.
- KV260 reachable records `ssh_only_terminal_status` and an `xmutil listapps`
  command transcript; if unreachable, `per_board_status["kv260"]["status"]` is
  `blocked_kv260_unreachable`.
- PolarFire reachable records a CPU-dispatch hash-verify smoke result; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- GateMate reachable records the detected IDCODE; if unreachable,
  `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4322)

---

### SCENARIO-HW-4322

**Scenario:** Exp 4322 reports KV260, PolarFire, and GateMate continuity without milestone blocking.

**Given:** KV260 is terminal and SSH-only per north-star §3, PolarFire and
GateMate are opportunistic, and the milestone requires board continuity to use
only SSH/USB reachability checks.
**When:** Experiment 4322 reads Exp 4311 context, runs the KV260 SSH, PolarFire
SSH, and GateMate DirtyJTAG detect preconditions, performs reachable-board
opportunistic checks, and records honest `blocked_<board>` statuses for
unreachable boards.
**Then:** It writes `results/experiment_4322_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4322)

---

### REQ-HW-4311

**Title:** Hardware continuity MUST report opportunistic per-board reachability for Exp 4311

**Description:**
Experiment 4311 MUST produce
`results/experiment_4311_hardware_continuity.json` as an opportunistic
per-board hardware-continuity artifact for KV260, GateMate, and PolarFire. It
MUST read `results/experiment_4300_hardware_continuity.json` as the prior .397
hardware state and preserve the continuity discipline: KV260 is terminal and
SSH-only, PolarFire is opportunistic hash-verified CPU dispatch, and GateMate may
be honestly blocked when DirtyJTAG detect does not expose the board.

The experiment MUST run exactly the per-board reachability preconditions below,
and no board being unreachable may block the milestone:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. This is the
  only valid KV260 precondition; host SD-card block-device checks such as
  `/dev/mmcblk` are forbidden.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).

When KV260 is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and record SSH-only terminal status without making a fabric speedup claim. When
PolarFire is reachable, the experiment MUST run or inject a CPU-dispatch
hash-verify smoke and record the result hash status. When GateMate is detected,
the experiment MUST record the IDCODE; otherwise the GateMate status MUST be
`blocked_gatemate_unreachable`. For any unreachable board, the board's status
MUST be `blocked_<board>_unreachable`, and the experiment MUST continue checking
the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. A per-board reachability report (including honest blocked_<board>) is COMPLETE; no board blocks the milestone.`
- `per_board_status`: `Per-board reachability + any opportunistic result -- keeps the boards visible (continuity) without blocking on operator hardware actions.`
- `preconditions_checked`: `Records the SSH/USB reachability checks (KV260 via SSH NOT SD-card); pre-empts the wrong-mechanism + fabrication modes.`
- `reproducibility_checksum`: `Hash of the per-board probe outputs; auditable.`

**Acceptance criteria:**
- `python3 results/experiment_4311_hardware_continuity.py` writes
  `results/experiment_4311_hardware_continuity.json`.
- `per_board_status` is a dict keyed by `kv260`, `polarfire`, and `gatemate`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records the KV260 SSH BatchMode, PolarFire SSH, and
  GateMate DirtyJTAG detect preconditions; no host SD-card marker such as
  `/dev/mmcblk` appears in the artifact.
- KV260 reachable records `ssh_only_terminal_status` and an `xmutil listapps`
  command transcript; if unreachable, `per_board_status["kv260"]["status"]` is
  `blocked_kv260_unreachable`.
- PolarFire reachable records a CPU-dispatch hash-verify smoke result; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- GateMate reachable records the detected IDCODE; if unreachable,
  `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4311)

---

### SCENARIO-HW-4311

**Scenario:** Exp 4311 reports KV260, PolarFire, and GateMate continuity without milestone blocking.

**Given:** KV260 is terminal and SSH-only per north-star §3, PolarFire and
GateMate are opportunistic, and the milestone requires board continuity to use
only SSH/USB reachability checks.
**When:** Experiment 4311 reads Exp 4300 context, runs the KV260 SSH, PolarFire
SSH, and GateMate DirtyJTAG detect preconditions, performs reachable-board
opportunistic checks, and records honest `blocked_<board>` statuses for
unreachable boards.
**Then:** It writes `results/experiment_4311_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4311)

---

### REQ-HW-4300

**Title:** Hardware continuity MUST report opportunistic per-board reachability for Exp 4300

**Description:**
Experiment 4300 MUST produce
`results/experiment_4300_hardware_continuity.json` as an opportunistic
per-board hardware-continuity artifact for KV260, GateMate, and PolarFire. It
MUST read `results/experiment_4288_hardware_continuity.json` as the prior .396
hardware state and preserve the continuity discipline: KV260 is terminal and
SSH-only, PolarFire is opportunistic hash-verified CPU dispatch, and GateMate may
be honestly blocked when DirtyJTAG detect does not expose the board.

The experiment MUST run exactly the per-board reachability preconditions below,
and no board being unreachable may block the milestone:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. This is the
  only valid KV260 precondition; host SD-card block-device checks such as
  `/dev/mmcblk` are forbidden.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).

When KV260 is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and record SSH-only terminal status without making a fabric speedup claim. When
PolarFire is reachable, the experiment MUST run or inject a CPU-dispatch
hash-verify smoke and record the result hash status. When GateMate is detected,
the experiment MUST record the IDCODE; otherwise the GateMate status MUST be
`blocked_gatemate_unreachable`. For any unreachable board, the board's status
MUST be `blocked_<board>_unreachable`, and the experiment MUST continue checking
the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. A per-board reachability report (including honest blocked_<board>) is COMPLETE; no board blocks the milestone.`
- `per_board_status`: `Per-board reachability + any opportunistic result -- keeps the boards visible (continuity) without blocking on operator hardware actions.`
- `preconditions_checked`: `Records the SSH/USB reachability checks (KV260 via SSH NOT SD-card); pre-empts the wrong-mechanism + fabrication modes.`
- `reproducibility_checksum`: `Hash of the per-board probe outputs; auditable.`

**Acceptance criteria:**
- `python3 results/experiment_4300_hardware_continuity.py` writes
  `results/experiment_4300_hardware_continuity.json`.
- `per_board_status` is a dict keyed by `kv260`, `polarfire`, and `gatemate`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records the KV260 SSH BatchMode, PolarFire SSH, and
  GateMate DirtyJTAG detect preconditions; no host SD-card marker such as
  `/dev/mmcblk` appears in the artifact.
- KV260 reachable records `ssh_only_terminal_status` and an `xmutil listapps`
  command transcript; if unreachable, `per_board_status["kv260"]["status"]` is
  `blocked_kv260_unreachable`.
- PolarFire reachable records a CPU-dispatch hash-verify smoke result; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- GateMate reachable records the detected IDCODE; if unreachable,
  `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4300)

---

### SCENARIO-HW-4300

**Scenario:** Exp 4300 reports KV260, PolarFire, and GateMate continuity without milestone blocking.

**Given:** KV260 is terminal and SSH-only per north-star §3, PolarFire and
GateMate are opportunistic, and the milestone requires board continuity to use
only SSH/USB reachability checks.
**When:** Experiment 4300 reads Exp 4288 context, runs the KV260 SSH, PolarFire
SSH, and GateMate DirtyJTAG detect preconditions, performs reachable-board
opportunistic checks, and records honest `blocked_<board>` statuses for
unreachable boards.
**Then:** It writes `results/experiment_4300_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4300)

---

### REQ-HW-4288

**Title:** Hardware continuity MUST report opportunistic per-board reachability for Exp 4288

**Description:**
Experiment 4288 MUST produce
`results/experiment_4288_hardware_continuity.json` as an opportunistic
per-board hardware-continuity artifact for KV260, GateMate, and PolarFire. It
MUST read `results/experiment_4278_hardware_continuity.json` as the prior .395
hardware state and preserve the continuity discipline: KV260 is terminal and
SSH-only, PolarFire is opportunistic hash-verified CPU dispatch, and GateMate may
be honestly blocked when DirtyJTAG detect does not expose the board.

The experiment MUST run exactly the per-board reachability preconditions below,
and no board being unreachable may block the milestone:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. This is the
  only valid KV260 precondition; host SD-card block-device checks such as
  `/dev/mmcblk` are forbidden.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).

When KV260 is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and record SSH-only terminal status without making a fabric speedup claim. When
PolarFire is reachable, the experiment MUST run or inject a CPU-dispatch
hash-verify smoke and record the result hash status. When GateMate is detected,
the experiment MUST record the IDCODE; otherwise the GateMate status MUST be
`blocked_gatemate_unreachable`. For any unreachable board, the board's status
MUST be `blocked_<board>_unreachable`, and the experiment MUST continue checking
the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. A per-board reachability report (including honest blocked_<board>) is COMPLETE; no board blocks the milestone.`
- `per_board_status`: `Per-board reachability + any opportunistic result -- keeps the boards visible (continuity) without blocking on operator hardware actions.`
- `preconditions_checked`: `Records the SSH/USB reachability checks (KV260 via SSH NOT SD-card); pre-empts the wrong-mechanism + fabrication modes.`
- `reproducibility_checksum`: `Hash of the per-board probe outputs; auditable.`

**Acceptance criteria:**
- `python3 results/experiment_4288_hardware_continuity.py` writes
  `results/experiment_4288_hardware_continuity.json`.
- `per_board_status` is a dict keyed by `kv260`, `polarfire`, and `gatemate`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records the KV260 SSH BatchMode, PolarFire SSH, and
  GateMate DirtyJTAG detect preconditions; no host SD-card marker such as
  `/dev/mmcblk` appears in the artifact.
- KV260 reachable records `ssh_only_terminal_status` and an `xmutil listapps`
  command transcript; if unreachable, `per_board_status["kv260"]["status"]` is
  `blocked_kv260_unreachable`.
- PolarFire reachable records a CPU-dispatch hash-verify smoke result; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- GateMate reachable records the detected IDCODE; if unreachable,
  `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4288)

---

### SCENARIO-HW-4288

**Scenario:** Exp 4288 reports KV260, PolarFire, and GateMate continuity without milestone blocking.

**Given:** KV260 is terminal and SSH-only per north-star §3, PolarFire and
GateMate are opportunistic, and the milestone requires board continuity to use
only SSH/USB reachability checks.
**When:** Experiment 4288 reads Exp 4278 context, runs the KV260 SSH, PolarFire
SSH, and GateMate DirtyJTAG detect preconditions, performs reachable-board
opportunistic checks, and records honest `blocked_<board>` statuses for
unreachable boards.
**Then:** It writes `results/experiment_4288_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4288)

---

### REQ-HW-4278

**Title:** Hardware continuity MUST report opportunistic per-board reachability for Exp 4278

**Description:**
Experiment 4278 MUST produce
`results/experiment_4278_hardware_continuity.json` as an opportunistic
per-board hardware-continuity artifact for KV260, GateMate, and PolarFire. It
MUST read `results/experiment_4267_hardware_continuity.json` as the prior .394
hardware state and preserve the continuity discipline: KV260 is terminal and
SSH-only, PolarFire is opportunistic hash-verified CPU dispatch, and GateMate may
be honestly blocked when DirtyJTAG detect does not expose the board.

The experiment MUST run exactly the per-board reachability preconditions below,
and no board being unreachable may block the milestone:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. This is the
  only valid KV260 precondition; host SD-card block-device checks such as
  `/dev/mmcblk` are forbidden.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).

When KV260 is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and record SSH-only terminal status without making a fabric speedup claim. When
PolarFire is reachable, the experiment MUST run or inject a CPU-dispatch
hash-verify smoke and record the result hash status. When GateMate is detected,
the experiment MUST record the IDCODE; otherwise the GateMate status MUST be
`blocked_gatemate_unreachable`. For any unreachable board, the board's status
MUST be `blocked_<board>_unreachable`, and the experiment MUST continue checking
the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. A per-board reachability report (including honest blocked_<board>) is COMPLETE; no board blocks the milestone.`
- `per_board_status`: `Per-board reachability + any opportunistic result -- keeps the boards visible (continuity) without blocking on operator hardware actions.`
- `preconditions_checked`: `Records the SSH/USB reachability checks (KV260 via SSH NOT SD-card); pre-empts the wrong-mechanism + fabrication modes.`
- `reproducibility_checksum`: `Hash of the per-board probe outputs; auditable.`

**Acceptance criteria:**
- `python3 results/experiment_4278_hardware_continuity.py` writes
  `results/experiment_4278_hardware_continuity.json`.
- `per_board_status` is a dict keyed by `kv260`, `polarfire`, and `gatemate`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records the KV260 SSH BatchMode, PolarFire SSH, and
  GateMate DirtyJTAG detect preconditions; no host SD-card marker such as
  `/dev/mmcblk` appears in the artifact.
- KV260 reachable records `ssh_only_terminal_status` and an `xmutil listapps`
  command transcript; if unreachable, `per_board_status["kv260"]["status"]` is
  `blocked_kv260_unreachable`.
- PolarFire reachable records a CPU-dispatch hash-verify smoke result; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- GateMate reachable records the detected IDCODE; if unreachable,
  `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4278)

---

### SCENARIO-HW-4278

**Scenario:** Exp 4278 reports KV260, PolarFire, and GateMate continuity without milestone blocking.

**Given:** KV260 is terminal and SSH-only per north-star §3, PolarFire and
GateMate are opportunistic, and the milestone requires board continuity to use
only SSH/USB reachability checks.
**When:** Experiment 4278 reads Exp 4267 context, runs the KV260 SSH, PolarFire
SSH, and GateMate DirtyJTAG detect preconditions, performs reachable-board
opportunistic checks, and records honest `blocked_<board>` statuses for
unreachable boards.
**Then:** It writes `results/experiment_4278_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4278)

---

### REQ-HW-4267

**Title:** Hardware continuity MUST report opportunistic per-board reachability for Exp 4267

**Description:**
Experiment 4267 MUST produce
`results/experiment_4267_hardware_continuity.json` as an opportunistic
per-board hardware-continuity artifact for KV260, GateMate, and PolarFire. It
MUST read `results/experiment_4253_hardware_continuity.json` as the prior
hardware state and preserve the .393 discipline: KV260 is terminal and SSH-only,
PolarFire is opportunistic hash-verified CPU dispatch, and GateMate may be
honestly blocked when DirtyJTAG detect does not expose the board.

The experiment MUST run exactly the per-board reachability preconditions below,
and no board being unreachable may block the milestone:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. This is the
  only valid KV260 precondition; host SD-card block-device checks such as
  `/dev/mmcblk` are forbidden.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).

When KV260 is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and record SSH-only terminal status without making a fabric speedup claim. When
PolarFire is reachable, the experiment MUST run or inject a CPU-dispatch
hash-verify smoke and record the result hash status. When GateMate is detected,
the experiment MUST record the IDCODE; otherwise the GateMate status MUST be
`blocked_gatemate_unreachable`. For any unreachable board, the board's status
MUST be `blocked_<board>_unreachable`, and the experiment MUST continue checking
the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. A per-board reachability report (including honest blocked_<board>) is COMPLETE; no board blocks the milestone.`
- `per_board_status`: `Per-board reachability + any opportunistic result -- keeps the boards visible (continuity) without blocking on operator hardware actions.`
- `preconditions_checked`: `Records the SSH/USB reachability checks (KV260 via SSH NOT SD-card); pre-empts the wrong-mechanism + fabrication modes.`
- `reproducibility_checksum`: `Hash of the per-board probe outputs; auditable.`

**Acceptance criteria:**
- `python3 results/experiment_4267_hardware_continuity.py` writes
  `results/experiment_4267_hardware_continuity.json`.
- `per_board_status` is a dict keyed by `kv260`, `polarfire`, and `gatemate`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records the KV260 SSH BatchMode, PolarFire SSH, and
  GateMate DirtyJTAG detect preconditions; no host SD-card marker such as
  `/dev/mmcblk` appears in the artifact.
- KV260 reachable records `ssh_only_terminal_status` and an `xmutil listapps`
  command transcript; if unreachable, `per_board_status["kv260"]["status"]` is
  `blocked_kv260_unreachable`.
- PolarFire reachable records a CPU-dispatch hash-verify smoke result; if
  unreachable, `per_board_status["polarfire"]["status"]` is
  `blocked_polarfire_unreachable`.
- GateMate reachable records the detected IDCODE; if unreachable,
  `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4267)

---

### SCENARIO-HW-4267

**Scenario:** Exp 4267 reports KV260, PolarFire, and GateMate continuity without milestone blocking.

**Given:** KV260 is terminal and SSH-only per north-star §3, PolarFire and
GateMate are opportunistic, and the milestone requires board continuity to use
only SSH/USB reachability checks.
**When:** Experiment 4267 reads Exp 4253 context, runs the KV260 SSH, PolarFire
SSH, and GateMate DirtyJTAG detect preconditions, performs reachable-board
opportunistic checks, and records honest `blocked_<board>` statuses for
unreachable boards.
**Then:** It writes `results/experiment_4267_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4267)

---

### REQ-HW-4253

**Title:** Hardware continuity MUST produce per-board SSH/USB status for Exp 4253

**Description:**
Experiment 4253 MUST produce
`results/experiment_4253_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4240_hardware_continuity.json` as prior .392 context,
preserve one reachability plus next-step record per attached board, and
continue checking reachable boards when another board is blocked. As long as
boards are attached, each milestone reserves one task for every non-terminal
board. KV260 is terminal and receives only opportunistic SSH confirmation.
GateMate and PolarFire are non-terminal and MUST each name the next concrete
forward step.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash, including IDCODE visibility recovery when .392 reported GateMate as
unreachable. PolarFire's non-terminal status MUST name the next concrete step
toward end-to-end Carnot dispatch with a hash match. For any unreachable board,
the board's status MUST be `blocked_<board>_unreachable`, and the experiment
MUST continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. An honest blocked_<board> for an unreachable board is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card.`

**Acceptance criteria:**
- `python3 results/experiment_4253_hardware_continuity.py` writes
  `results/experiment_4253_hardware_continuity.json`.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- GateMate reachable runs or records the next concrete step toward flashing the
  n=16 Ising tile; if unreachable, `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- PolarFire reachable runs or records the next concrete step toward end-to-end
  Carnot dispatch with a hash match; if unreachable,
  `per_board_status["polarfire"]["status"]` is `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4253)

---

### SCENARIO-HW-4253

**Scenario:** Exp 4253 records SSH/USB reachability and concrete next work per board.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4253 reads Exp 4240 context, runs the GateMate DirtyJTAG
detect, PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions, and
records the next concrete step per non-terminal board.
**Then:** It writes `results/experiment_4253_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4253)

---

### REQ-HW-4240

**Title:** Hardware continuity MUST produce per-board SSH/USB status for Exp 4240

**Description:**
Experiment 4240 MUST produce
`results/experiment_4240_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4228_hardware_continuity.json` as prior .391 context,
preserve one reachability plus next-step record per attached board, and
continue checking reachable boards when another board is blocked. As long as
boards are attached, each milestone reserves one task for every non-terminal
board. KV260 is terminal and receives only opportunistic SSH confirmation.
GateMate and PolarFire are non-terminal and MUST each name the next concrete
forward step.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash, including IDCODE visibility recovery when the board is unreachable.
PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. An honest blocked_<board> for an unreachable board is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card.`

**Acceptance criteria:**
- `python3 results/experiment_4240_hardware_continuity.py` writes
  `results/experiment_4240_hardware_continuity.json`.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- GateMate reachable runs or records the next concrete step toward flashing the
  n=16 Ising tile; if unreachable, `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- PolarFire reachable runs or records the next concrete step toward end-to-end
  Carnot dispatch with a hash match; if unreachable,
  `per_board_status["polarfire"]["status"]` is `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4240)

---

### SCENARIO-HW-4240

**Scenario:** Exp 4240 records SSH/USB reachability and concrete next work per board.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4240 reads Exp 4228 context, runs the GateMate DirtyJTAG
detect, PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions, and
records the next concrete step per non-terminal board.
**Then:** It writes `results/experiment_4240_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4240)

---

### REQ-HW-4228

**Title:** Hardware continuity MUST produce per-board SSH/USB status for Exp 4228

**Description:**
Experiment 4228 MUST produce
`results/experiment_4228_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4217_hardware_continuity.json` as prior context, preserve
one reachability plus next-step record per attached board, and continue checking
reachable boards when another board is blocked. As long as boards are attached,
each milestone reserves one task for every non-terminal board. KV260 is terminal
and receives only opportunistic SSH confirmation. GateMate and PolarFire are
non-terminal and MUST each name the next concrete forward step.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash. PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. An honest blocked_<board> for an unreachable board is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card.`

**Acceptance criteria:**
- `results/experiment_4228_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- GateMate reachable runs or records the next concrete step toward flashing the
  n=16 Ising tile; if unreachable, `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- PolarFire reachable runs or records the next concrete step toward end-to-end
  Carnot dispatch with a hash match; if unreachable,
  `per_board_status["polarfire"]["status"]` is `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4228)

---

### SCENARIO-HW-4228

**Scenario:** Exp 4228 records SSH/USB reachability and concrete next work per board.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4228 reads Exp 4217 context, runs the GateMate DirtyJTAG
detect, PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions, and
records the next concrete step per non-terminal board.
**Then:** It writes `results/experiment_4228_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4228)

---

### REQ-HW-4217

**Title:** Hardware continuity MUST produce per-board SSH/USB status for Exp 4217

**Description:**
Experiment 4217 MUST produce
`results/experiment_4217_hardware_continuity.json` as the next per-board
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST read
`results/experiment_4205_hardware_continuity.json` as prior context, preserve
one reachability plus next-step record per attached board, and continue checking
reachable boards when another board is blocked. As long as boards are attached,
each milestone reserves one task for every non-terminal board. KV260 is terminal
and receives only opportunistic SSH confirmation. GateMate and PolarFire are
non-terminal and MUST each name the next concrete forward step.

The only valid preconditions are:

- GateMate: `openFPGALoader -c dirtyJtag --detect`, accepted only when it
  reports the GM1Ax IDCODE (`0x20000001`).
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`.

KV260 MUST NOT use a host SD-card block-device precondition such as
`/dev/mmcblk`; the SSH-Not-SD-Card discipline is mandatory. GateMate's
non-terminal status MUST name the next concrete step toward the n=16 Ising tile
flash. PolarFire's non-terminal status MUST name the next concrete step toward
end-to-end Carnot dispatch with a hash match. For any unreachable board, the
board's status MUST be `blocked_<board>_unreachable`, and the experiment MUST
continue checking the other boards.

The artifact MUST include `field_principles` entries for the required fields:

- `honest_verdict`: `Terminal-prefixed. An honest blocked_<board> for an unreachable board is a COMPLETE verdict for that board.`
- `per_board_status`: `One reachability + next-step record per board so a board is never silently forgotten.`
- `preconditions_checked`: `Records the SSH/USB-detect preconditions run; enforces SSH-Not-SD-Card.`

**Acceptance criteria:**
- `results/experiment_4217_hardware_continuity.json` is generated.
- `per_board_status` is a dict keyed by `kv260`, `gatemate`, and `polarfire`;
  each entry records `reachable`, `status`, `next_concrete_step`,
  `precondition_resource`, `precondition_command`, `duration_s`, and a distinct
  board-specific `timer_id`.
- `preconditions_checked` records exactly the GateMate DirtyJTAG detect,
  PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions; no host
  SD-card marker such as `/dev/mmcblk` appears in the artifact.
- GateMate reachability requires the GM1Ax/`0x20000001` IDCODE from
  `openFPGALoader -c dirtyJtag --detect`.
- GateMate reachable runs or records the next concrete step toward flashing the
  n=16 Ising tile; if unreachable, `per_board_status["gatemate"]["status"]` is
  `blocked_gatemate_unreachable`.
- PolarFire reachable runs or records the next concrete step toward end-to-end
  Carnot dispatch with a hash match; if unreachable,
  `per_board_status["polarfire"]["status"]` is `blocked_polarfire_unreachable`.
- KV260 terminal state is confirmed via SSH when reachable; if unreachable,
  `per_board_status["kv260"]["status"]` is `blocked_kv260_unreachable`.
- `honest_verdict` starts with `complete:` or `blocked_`.

**Implementation status:** Implemented (Exp 4217)

---

### SCENARIO-HW-4217

**Scenario:** Exp 4217 records SSH/USB reachability and concrete next work per board.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4217 reads Exp 4205 context, runs the GateMate DirtyJTAG
detect, PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions, and
records the next concrete step per non-terminal board.
**Then:** It writes `results/experiment_4217_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4217)

---

### SCENARIO-HW-4205

**Scenario:** Exp 4205 records SSH/USB reachability and concrete next work per board.

**Given:** GateMate and PolarFire remain non-terminal, KV260 is terminal, and
the milestone requires board continuity to use only SSH/USB-detect
preconditions.
**When:** Experiment 4205 reads Exp 4194 context, runs the GateMate DirtyJTAG
detect, PolarFire SSH BatchMode, and KV260 SSH BatchMode preconditions, and
records the next concrete step per non-terminal board.
**Then:** It writes `results/experiment_4205_hardware_continuity.json` with a
terminal-prefixed verdict, `inference_substrate="hardware_smoke"`, required
field principles, `per_board_status`, precondition evidence for the three
allowed board checks, distinct per-board timer identifiers and wall-clock
durations, deterministic seed, checksum, and no KV260 host SD-card
precondition.

**Implementation status:** Implemented (Exp 4205)

---

### SCENARIO-HW-4017

**Scenario:** ARC continuity records reachable boards and blocked board next steps.

**Given:** The ARC milestone requires all three attached boards to remain visible.
**When:** Experiment 4017 runs the KV260 SSH, GateMate DirtyJTAG detect, and
PolarFire SSH preconditions, then gathers continuity evidence only for reachable
boards.
**Then:** It writes `results/experiment_4017_hardware_continuity.json` with
terminal-prefix verdict, hardware_smoke substrate, bare reachability values,
per-board next steps, distinct per-board durations, precondition evidence,
checksum, deterministic seed, and no host SD-card precondition.

**Implementation status:** Pending (Exp 4017)

---

### REQ-HW-4484

**Title:** Hardware continuity audit MUST record live reachability for KV260, GateMate, and PolarFire

**Description:**
Experiment 4484 MUST produce
`results/experiment_4484_hardware_continuity_audit.json` as an audit-only
hardware-continuity artifact for the three attached board tracks. It MUST run
exactly the board reachability preconditions requested by the operator and MUST
record one reachability plus next-forward-step row per board. The audit MUST NOT
claim fabric acceleration, speedup, flashing, latency transcript, sampler smoke,
or ARC solve progress; it records only the precondition result and the next
concrete step, or `blocked_<board>_unreachable` when the board is not reachable.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this is an
  SSH-only continuity check and host storage probing is out of scope.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GateMate/GM1Ax/IDCODE-style observation.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `MUST start with a terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ so the reconciler classifies it as terminal (Verdict Terminal-Prefix Discipline).`
- `inference_substrate`: `explicit declaration (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right floor.`
- `offline_reproduced`: `a solve not reproducible offline is wasted effort -- only reproduced levels count (ARC Solve Reproducibility).`
- `reproduced_levels`: `headline metric reproducible_total_levels grows monotonically; report the count banked, real-env-confirmed.`
- `preconditions_checked`: `records WHICH resources were verified before launching; pre-empts the silent-missing-resource fabrication mode.`

**Acceptance criteria:**
- `results/experiment_4484_hardware_continuity_audit.json` is generated.
- `inference_substrate` is exactly `hardware_smoke`.
- `offline_reproduced` is `false` and `reproduced_levels` is `0` because this
  audit does not reproduce ARC levels.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  GateMate DirtyJTAG detect command, and PolarFire SSH command listed above.
- `per_board_status` is keyed by `kv260`, `gatemate`, and `polarfire`; each row
  records `reachable`, `status`, `next_forward_step`, `precondition_resource`,
  `precondition_command`, and a bounded command transcript.
- Any unreachable board row uses `blocked_<board>_unreachable`.
- `honest_verdict` starts with one of the terminal prefixes listed in the field
  principle and remains terminal even when individual board rows are blocked.
- The artifact includes `random_seed=4484`, `spec_refs`, `field_principles`,
  `duration_s`, and a stable `reproducibility_checksum`.

**Implementation status:** Pending (Exp 4484)

---

### REQ-HW-4497

**Title:** Hardware continuity audit MUST record live reachability for KV260, GateMate, and PolarFire

**Description:**
Experiment 4497 MUST produce
`results/experiment_4497_hardware_continuity_audit.json` as an audit-only
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST run
only the operator-requested reachability preconditions and MUST record one
reachability plus next-forward-step row per board. The audit MUST NOT claim
fabric acceleration, speedup, flashing, latency transcript, sampler smoke, or
ARC solve progress; it records only the precondition result and the next
concrete step, or `blocked_<board>_unreachable` when the board is not reachable.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this is an
  SSH-only continuity check, and host storage probing is out of scope.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GateMate/GM1Ax/IDCODE-style observation.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).`
- `inference_substrate`: `explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor.`
- `preconditions_checked`: `records WHICH resources were verified; pre-empts silent-missing-resource fabrication.`

**Acceptance criteria:**
- `results/experiment_4497_hardware_continuity_audit.json` is generated.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  GateMate DirtyJTAG detect command, and PolarFire SSH command listed above.
- `per_board_status` is keyed by `kv260`, `gatemate`, and `polarfire`; each row
  records `reachable`, `status`, `next_forward_step`, `precondition_resource`,
  `precondition_command`, and a bounded command transcript.
- Any unreachable board row uses `blocked_<board>_unreachable`.
- `honest_verdict` starts with one of the terminal prefixes listed in the field
  principle and remains terminal even when individual board rows are blocked.
- The artifact includes `random_seed=4497`, `spec_refs`, `field_principles`,
  `duration_s`, and a stable `reproducibility_checksum`.

**Implementation status:** Pending (Exp 4497)

---

### SCENARIO-HW-4497

**Scenario:** Exp 4497 audits board reachability without fabricating progress.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
GateMate DirtyJTAG detect, and PolarFire SSH reachability.
**When:** Experiment 4497 runs the three required commands and builds the
artifact.
**Then:** It writes `results/experiment_4497_hardware_continuity_audit.json`
with terminal-prefixed `honest_verdict`, `inference_substrate="hardware_smoke"`,
principle-annotated required fields, one reachability/next-step row per board,
`blocked_<board>_unreachable` for unavailable boards, and no unsupported
hardware progress claims.

**Implementation status:** Pending (Exp 4497)

---

### REQ-HW-4507

**Title:** Hardware-task continuity audit MUST record live reachability and next steps for all attached boards

**Description:**
Experiment 4507 MUST produce
`results/experiment_4507_hardware_continuity_audit.json` as an audit-only
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST run
only the operator-requested reachability preconditions and MUST record one
reachability plus next-forward-step row per board. The audit MUST NOT claim
fabric acceleration, speedup, flashing, latency transcript, sampler smoke, or
ARC solve progress; it records only the precondition result and the next
concrete step, or `blocked_<board>_unreachable` when the board is not reachable.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this is an
  SSH-only continuity check, with no alternate host-side storage probe.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GateMate/GM1Ax/IDCODE-style observation.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_.`
- `inference_substrate`: `explicit substrate so adversarial_verify applies the right duration floor.`
- `preconditions_checked`: `records WHICH resources were verified; pre-empts silent-missing-resource fabrication.`

**Acceptance criteria:**
- `results/experiment_4507_hardware_continuity_audit.json` is generated.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  GateMate DirtyJTAG detect command, and PolarFire SSH command listed above.
- `per_board_status` is keyed by `kv260`, `gatemate`, and `polarfire`; each row
  records `reachable`, `status`, `next_forward_step`, `precondition_resource`,
  `precondition_command`, and a bounded command transcript.
- Any unreachable board row uses `blocked_<board>_unreachable`.
- `honest_verdict` starts with one of the terminal prefixes listed in the field
  principle and remains terminal even when individual board rows are blocked.
- The artifact includes `random_seed=4507`, `spec_refs`, `field_principles`,
  `duration_s`, and a stable `reproducibility_checksum`.

**Implementation status:** Implemented (Exp 4507)

---

### SCENARIO-HW-4507

**Scenario:** Exp 4507 audits board reachability without fabricating progress.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
GateMate DirtyJTAG detect, and PolarFire SSH reachability.
**When:** Experiment 4507 runs the three required commands and builds the
artifact.
**Then:** It writes `results/experiment_4507_hardware_continuity_audit.json`
with terminal-prefixed `honest_verdict`, `inference_substrate="hardware_smoke"`,
principle-annotated required fields, one reachability/next-step row per board,
`blocked_<board>_unreachable` for unavailable boards, and no unsupported
hardware progress claims.

**Implementation status:** Implemented (Exp 4507)

---

### SCENARIO-HW-4484

**Scenario:** Exp 4484 audits board reachability without fabricating progress.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
GateMate DirtyJTAG detect, and PolarFire SSH reachability.
**When:** Experiment 4484 runs the three required commands and builds the
artifact.
**Then:** It writes `results/experiment_4484_hardware_continuity_audit.json`
with terminal-prefixed `honest_verdict`, `inference_substrate="hardware_smoke"`,
principle-annotated required fields, one reachability/next-step row per board,
`blocked_<board>_unreachable` for unavailable boards, and no unsupported hardware
progress claims.

**Implementation status:** Pending (Exp 4484)

---

### REQ-HW-4519

**Title:** Hardware-task continuity audit MUST record per-board reachability and next forward steps

**Description:**
Experiment 4519 MUST produce
`results/experiment_4519_hardware_continuity_audit.json` as an audit-only
hardware-continuity artifact for KV260, GateMate, and PolarFire. It MUST run
only the operator-requested reachability preconditions and MUST record one
reachability plus next-forward-step row per board. The audit MUST NOT claim
fabric acceleration, speedup, flashing, latency transcript, sampler smoke, or
ARC solve progress; it records only the precondition result and the next
concrete step, or `blocked_<board>_unreachable` when the board is not reachable.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this is the
  only valid KV260 precondition because the SD-card mechanism is retired.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a colognechip GateMate/GM1Ax/IDCODE-style observation.
- PolarFire: `ssh -o ConnectTimeout=5 polarfire 'true'`.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; e.g. complete: hardware_continuity_audit_<per-board summary> (a board blocked is an honest non-terminal state, not a task failure).`
- `inference_substrate`: `hardware_smoke -- SSH/USB reachability, per-board duration floor.`
- `kv260_reachable`: `SSH-reachability is the ONLY valid KV260 precondition (SD-card mechanism retired).`
- `gatemate_detected`: `honest USB-detect result -- DirtyJTAG-unreachable is recorded, never fabricated.`
- `polarfire_reachable`: `SSH reachability of the board.`
- `preconditions_checked`: `records WHICH boards were probed and how; the audit IS the precondition set.`

**Acceptance criteria:**
- `results/experiment_4519_hardware_continuity_audit.json` is generated.
- `inference_substrate` is exactly `hardware_smoke`.
- Top-level `kv260_reachable`, `gatemate_detected`, and
  `polarfire_reachable` are bare booleans matching per-board reachability.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  GateMate DirtyJTAG detect command, and PolarFire SSH command listed above.
- `per_board_status` is keyed by `kv260`, `gatemate`, and `polarfire`; each row
  records `reachable`, `status`, `next_forward_step`, `precondition_resource`,
  `precondition_command`, and a bounded command transcript.
- KV260 reachable records the latency transcript / bitstream next step;
  PolarFire reachable records the sampler smoke next step.
- Any unreachable board row uses `blocked_<board>_unreachable`.
- `honest_verdict` starts with one of the terminal prefixes listed in the field
  principle and remains terminal even when individual board rows are blocked.
- The artifact includes `random_seed=4519`, `spec_refs`, `field_principles`,
  `duration_s`, and a stable `reproducibility_checksum`.
- The artifact contains no host SD-card device-path precondition such as
  `mmcblk`.

**Implementation status:** Pending (Exp 4519)

---

### SCENARIO-HW-4519

**Scenario:** Exp 4519 audits board reachability without fabricating progress.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
GateMate DirtyJTAG detect, and PolarFire SSH reachability.
**When:** Experiment 4519 runs the three required commands and builds the
artifact.
**Then:** It writes `results/experiment_4519_hardware_continuity_audit.json`
with terminal-prefixed `honest_verdict`, `inference_substrate="hardware_smoke"`,
principle-annotated required reachability fields, one reachability/next-step row
per board, `blocked_<board>_unreachable` for unavailable boards, and no
unsupported hardware progress claims.

**Implementation status:** Pending (Exp 4519)

---

### REQ-HW-4529

**Title:** Hardware continuity audit MUST capture reachable-board state for KV260, GateMate, and PolarFire

**Description:**
Experiment 4529 MUST produce `results/experiment_4529_hardware_continuity.json`
as an audit-only hardware-continuity artifact for KV260, GateMate, and
PolarFire. It MUST run only the operator-requested board reachability
preconditions and MUST treat each board independently: an unavailable board is
reported as its own `blocked_<board>` state while the experiment still completes
with a terminal `honest_verdict`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this is the
  only valid KV260 precondition because the host SD-card mechanism is retired.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GM1Ax IDCODE observation.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.

For each reachable board, the experiment MUST record current board state and
MUST NOT build, flash, or claim any new bitstream:

- KV260 reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'sudo -n xmutil listapps'`.
- GateMate reachable state: the GM1Ax IDCODE observed from DirtyJTAG detect.
- PolarFire reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire uptime`.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; complete: hardware_continuity_audit_<n>_boards_reachable (blocked_<board> per-board is honest, not terminal failure).`
- `inference_substrate`: `hardware_smoke -- SSH/USB board checks, per-board duration floors.`
- `per_board_status`: `each board's reachability + state -- keeps every attached board visible in the milestone retro (the forget-pattern guard).`
- `preconditions_checked`: `records WHICH board resources were verified; SSH-not-SD-card for KV260.`

**Acceptance criteria:**
- `results/experiment_4529_hardware_continuity.json` is generated.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `complete: hardware_continuity_audit_<n>_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  GateMate DirtyJTAG detect command, and PolarFire SSH BatchMode command listed
  above.
- `per_board_status` is keyed by `kv260`, `gatemate`, and `polarfire`; each row
  records `reachable`, `status`, `precondition_resource`, `precondition_command`,
  reachability transcript, state transcript, and total board duration.
- Unreachable board rows use `blocked_kv260_ssh_unreachable`,
  `blocked_gatemate_usb_undetected`, or `blocked_polarfire_ssh_timeout` and do
  not fabricate a state transcript.
- Reachable KV260 rows include the `xmutil listapps` transcript, reachable
  GateMate rows include the GM1Ax IDCODE, and reachable PolarFire rows include
  the `uptime` transcript.
- The artifact includes `random_seed=4529`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4529)

---

### SCENARIO-HW-4529

**Scenario:** Exp 4529 records live board reachability and reachable-board state.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
GateMate DirtyJTAG GM1Ax detection, and PolarFire SSH reachability.
**When:** Experiment 4529 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4529_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one reachability plus
state row per attached board, precise per-board blocked statuses for unavailable
boards, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4529)

---

### REQ-HW-4540

**Title:** Hardware continuity audit MUST keep KV260, GateMate, and PolarFire visible without build work

**Description:**
Experiment 4540 MUST produce `results/experiment_4540_hardware_continuity.json`
as an audit-only hardware-continuity artifact for KV260, GateMate, and
PolarFire. It MUST run only the operator-requested board reachability
preconditions and MUST treat each board independently: an unavailable board is
reported as its own `blocked_<board>` state while the experiment still completes
with a terminal `honest_verdict`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this is the
  only valid KV260 precondition because the host SD-card mechanism is retired.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GM1Ax IDCODE observation.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.

For each reachable board, the experiment MUST record current board state and
MUST NOT build, flash, or claim any new bitstream:

- KV260 reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'sudo -n xmutil listapps'`.
- GateMate reachable state: the GM1Ax IDCODE observed from DirtyJTAG detect.
- PolarFire reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire uptime`.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; complete: hardware_continuity_audit_<n>_boards_reachable (blocked_<board> per-board is honest, not terminal failure).`
- `inference_substrate`: `hardware_smoke -- SSH/USB board checks, per-board duration floors.`
- `per_board_status`: `each board's reachability + state -- keeps every attached board visible in the milestone retro (the forget-pattern guard).`
- `preconditions_checked`: `records WHICH board resources were verified; SSH-not-SD-card for KV260.`

**Acceptance criteria:**
- `results/experiment_4540_hardware_continuity.json` is generated.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `complete: hardware_continuity_audit_<n>_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  GateMate DirtyJTAG detect command, and PolarFire SSH BatchMode command listed
  above.
- `per_board_status` is keyed by `kv260`, `gatemate`, and `polarfire`; each row
  records `reachable`, `status`, `precondition_resource`, `precondition_command`,
  reachability transcript, state transcript, and total board duration.
- Unreachable board rows use `blocked_kv260_ssh_unreachable`,
  `blocked_gatemate_usb_undetected`, or `blocked_polarfire_ssh_timeout` and do
  not fabricate a state transcript.
- Reachable KV260 rows include the `xmutil listapps` transcript, reachable
  GateMate rows include the GM1Ax IDCODE, and reachable PolarFire rows include
  the `uptime` transcript.
- The artifact includes `random_seed=4540`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4540)

---

### SCENARIO-HW-4540

**Scenario:** Exp 4540 records live board reachability and reachable-board state.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
GateMate DirtyJTAG GM1Ax detection, and PolarFire SSH reachability.
**When:** Experiment 4540 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4540_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one reachability plus
state row per attached board, precise per-board blocked statuses for unavailable
boards, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4540)

---

### REQ-HW-4552

**Title:** Hardware continuity audit MUST check KV260 by SSH, GateMate by USB detect, and PolarFire by SSH

**Description:**
Experiment 4552 MUST produce `results/experiment_4552_hardware_continuity.json`
as an audit-only hardware-continuity artifact for KV260, GateMate, and
PolarFire. It MUST run only the operator-requested board reachability
preconditions and MUST treat each board independently: an unavailable board is
reported as its own `blocked_<board>` state while the experiment still completes
with a terminal `honest_verdict`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this is the
  only valid KV260 precondition because the host SD-card block-device mechanism
  is retired.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GM1Ax IDCODE observation.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.

For each reachable board, the experiment MUST record current board state and
MUST NOT build, flash, or claim any new bitstream:

- KV260 reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'sudo -n xmutil listapps'`.
- GateMate reachable state: the GM1Ax IDCODE observed from DirtyJTAG detect.
- PolarFire reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire uptime`.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; complete: hardware_continuity_audit_<n>_boards_reachable (blocked_<board> per-board is honest, not terminal failure).`
- `inference_substrate`: `hardware_smoke -- SSH/USB board checks, per-board duration floors.`
- `per_board_status`: `each board's reachability + state -- keeps every attached board visible in the milestone retro (the forget-pattern guard).`
- `preconditions_checked`: `records WHICH board resources were verified; SSH-not-SD-card for KV260.`

**Acceptance criteria:**
- `results/experiment_4552_hardware_continuity.json` is generated.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `complete: hardware_continuity_audit_<n>_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  GateMate DirtyJTAG detect command, and PolarFire SSH BatchMode command listed
  above.
- `per_board_status` is keyed by `kv260`, `gatemate`, and `polarfire`; each row
  records `reachable`, `status`, `precondition_resource`, `precondition_command`,
  reachability transcript, state transcript, and total board duration.
- Unreachable board rows use `blocked_kv260_ssh_unreachable`,
  `blocked_gatemate_usb_undetected`, or `blocked_polarfire_ssh_timeout` and do
  not fabricate a state transcript.
- Reachable KV260 rows include the `xmutil listapps` transcript, reachable
  GateMate rows include the GM1Ax IDCODE, and reachable PolarFire rows include
  the `uptime` transcript.
- The artifact includes `random_seed=4552`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4552)

---

### SCENARIO-HW-4552

**Scenario:** Exp 4552 records live board reachability and reachable-board state.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
GateMate DirtyJTAG GM1Ax detection, and PolarFire SSH reachability.
**When:** Experiment 4552 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4552_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one reachability plus
state row per attached board, precise per-board blocked statuses for unavailable
boards, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4552)

---

### REQ-HW-4564

**Title:** Hardware continuity audit MUST independently check KV260, GateMate, and PolarFire state

**Description:**
Experiment 4564 MUST produce `results/experiment_4564_hardware_continuity.json`
as an audit-only hardware-continuity artifact for KV260, GateMate, and
PolarFire. It MUST run only the operator-requested board reachability
preconditions and MUST treat each board independently: an unavailable board is
reported as its own `blocked_<board>` state while the experiment still completes
with a terminal `honest_verdict`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this is the
  only valid KV260 precondition because the host SD-card block-device mechanism
  is retired.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GM1Ax IDCODE observation.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.

For each reachable board, the experiment MUST record current board state and
MUST NOT build, flash, or claim any new bitstream:

- KV260 reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'xmutil listapps'`.
- GateMate reachable state: the GM1Ax IDCODE observed from DirtyJTAG detect.
- PolarFire reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire uptime`.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; complete: hardware_continuity_audit_<n>_boards_reachable (blocked_<board> per-board is honest, not terminal failure).`
- `inference_substrate`: `hardware_smoke -- SSH/USB board checks, per-board duration floors.`
- `per_board_status`: `each board's reachability + state -- keeps every attached board visible in the milestone retro (the forget-pattern guard).`
- `preconditions_checked`: `records WHICH board resources were verified; SSH-not-SD-card for KV260.`

**Acceptance criteria:**
- `results/experiment_4564_hardware_continuity.json` is generated.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `complete: hardware_continuity_audit_<n>_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  GateMate DirtyJTAG detect command, and PolarFire SSH BatchMode command listed
  above.
- `per_board_status` is keyed by `kv260`, `gatemate`, and `polarfire`; each row
  records `reachable`, `status`, `precondition_resource`, `precondition_command`,
  reachability transcript, state transcript, and total board duration.
- Unreachable board rows use `blocked_kv260_ssh_unreachable`,
  `blocked_gatemate_usb_undetected`, or `blocked_polarfire_ssh_timeout` and do
  not fabricate a state transcript.
- Reachable KV260 rows include the `xmutil listapps` transcript from SSH,
  reachable GateMate rows include the GM1Ax IDCODE, and reachable PolarFire rows
  include the `uptime` transcript.
- The artifact includes `random_seed=4564`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4564)

---

### SCENARIO-HW-4564

**Scenario:** Exp 4564 records live board reachability and reachable-board state.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
GateMate DirtyJTAG GM1Ax detection, and PolarFire SSH reachability.
**When:** Experiment 4564 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4564_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one reachability plus
state row per attached board, precise per-board blocked statuses for unavailable
boards, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4564)

---

### REQ-HW-4576

**Title:** Hardware continuity audit MUST preserve per-board visibility without host SD-card checks

**Description:**
Experiment 4576 MUST produce `results/experiment_4576_hardware_continuity.json`
as an audit-only hardware-continuity artifact for KV260, GateMate, and
PolarFire. It MUST run only the operator-requested board reachability
preconditions and MUST treat each board independently: an unavailable board is
reported as its own `blocked_<board>` state while the experiment still completes
with a terminal `honest_verdict`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this is the
  only valid KV260 precondition because the host SD-card block-device mechanism
  is retired.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GM1Ax IDCODE observation.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.

For each reachable board, the experiment MUST record current board state and
MUST NOT build, flash, or claim any new bitstream:

- KV260 reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'xmutil listapps'`.
- GateMate reachable state: the GM1Ax IDCODE observed from DirtyJTAG detect.
- PolarFire reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire uptime`.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; complete: hardware_continuity_audit_<n>_boards_reachable (blocked_<board> per-board is honest, not terminal failure).`
- `inference_substrate`: `hardware_smoke -- SSH/USB board checks, per-board duration floors.`
- `per_board_status`: `each board's reachability + state -- keeps every attached board visible in the milestone retro (the forget-pattern guard).`
- `preconditions_checked`: `records WHICH board resources were verified; SSH-not-SD-card for KV260.`

**Acceptance criteria:**
- `results/experiment_4576_hardware_continuity.json` is generated.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `complete: hardware_continuity_audit_<n>_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  GateMate DirtyJTAG detect command, and PolarFire SSH BatchMode command listed
  above.
- `per_board_status` is keyed by `kv260`, `gatemate`, and `polarfire`; each row
  records `reachable`, `status`, `precondition_resource`, `precondition_command`,
  reachability transcript, state transcript, and total board duration.
- Unreachable board rows use `blocked_kv260_ssh_unreachable`,
  `blocked_gatemate_usb_undetected`, or `blocked_polarfire_ssh_timeout` and do
  not fabricate a state transcript.
- Reachable KV260 rows include the `xmutil listapps` transcript from SSH,
  reachable GateMate rows include the GM1Ax IDCODE, and reachable PolarFire rows
  include the `uptime` transcript.
- The artifact includes `random_seed=4576`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4576)

---

### SCENARIO-HW-4576

**Scenario:** Exp 4576 records live board reachability and reachable-board state.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
GateMate DirtyJTAG GM1Ax detection, and PolarFire SSH reachability.
**When:** Experiment 4576 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4576_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one reachability plus
state row per attached board, precise per-board blocked statuses for unavailable
boards, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4576)

---

### REQ-HW-4588

**Title:** Hardware continuity audit MUST report KV260, GateMate, and PolarFire without retired host-storage checks

**Description:**
Experiment 4588 MUST produce `results/experiment_4588_hardware_continuity.json`
as an audit-only hardware-continuity artifact for KV260, GateMate, and
PolarFire. It MUST run only the operator-requested board reachability
preconditions and MUST treat each board independently: an unavailable board is
reported as its own `blocked_<board>` state while the experiment still completes
with a terminal `honest_verdict`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this is the
  only valid KV260 precondition because the host SD-card block-device mechanism
  is retired.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GM1Ax IDCODE observation.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.

For each reachable board, the experiment MUST record current board state and
MUST NOT build, flash, or claim any new bitstream:

- KV260 reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'xmutil listapps'`.
- GateMate reachable state: the GM1Ax IDCODE observed from DirtyJTAG detect.
- PolarFire reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire uptime`.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; complete: hardware_continuity_audit_<n>_boards_reachable (blocked_<board> per-board is honest, not terminal failure).`
- `inference_substrate`: `hardware_smoke -- SSH/USB board checks, per-board duration floors.`
- `per_board_status`: `each board's reachability + state -- keeps every attached board visible in the milestone retro (the forget-pattern guard).`
- `preconditions_checked`: `records WHICH board resources were verified; SSH-not-SD-card for KV260.`

**Acceptance criteria:**
- `results/experiment_4588_hardware_continuity.json` is generated.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `complete: hardware_continuity_audit_<n>_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  GateMate DirtyJTAG detect command, and PolarFire SSH BatchMode command listed
  above.
- `per_board_status` is keyed by `kv260`, `gatemate`, and `polarfire`; each row
  records `reachable`, `status`, `precondition_resource`, `precondition_command`,
  reachability transcript, state transcript, and total board duration.
- Unreachable board rows use `blocked_kv260_ssh_unreachable`,
  `blocked_gatemate_usb_undetected`, or `blocked_polarfire_ssh_timeout` and do
  not fabricate a state transcript.
- Reachable KV260 rows include the `xmutil listapps` transcript from SSH,
  reachable GateMate rows include the GM1Ax IDCODE, and reachable PolarFire rows
  include the `uptime` transcript.
- The artifact includes `random_seed=4588`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4588)

---

### SCENARIO-HW-4588

**Scenario:** Exp 4588 records live board reachability and reachable-board state.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
GateMate DirtyJTAG GM1Ax detection, and PolarFire SSH reachability.
**When:** Experiment 4588 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4588_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one reachability plus
state row per attached board, precise per-board blocked statuses for unavailable
boards, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4588)

---

### REQ-HW-4600

**Title:** Hardware continuity audit MUST report KV260, GateMate, and PolarFire without retired host-storage checks

**Description:**
Experiment 4600 MUST produce `results/experiment_4600_hardware_continuity.json`
as an audit-only hardware-continuity artifact for KV260, GateMate, and
PolarFire. It MUST run only the operator-requested board reachability
preconditions and MUST treat each board independently: an unavailable board is
reported as its own `blocked_<board>` state while the experiment still completes
with a terminal `honest_verdict`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this is the
  only valid KV260 precondition because the host SD-card block-device mechanism
  is retired.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GM1Ax IDCODE observation.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.

For each reachable board, the experiment MUST record current board state and
MUST NOT build, flash, or claim any new bitstream:

- KV260 reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'xmutil listapps'`.
- GateMate reachable state: the GM1Ax IDCODE observed from DirtyJTAG detect.
- PolarFire reachable state: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire uptime`.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; complete: hardware_continuity_audit_<n>_boards_reachable (blocked_<board> per-board is honest, not terminal failure).`
- `inference_substrate`: `hardware_smoke -- SSH/USB board checks, per-board duration floors.`
- `per_board_status`: `each board's reachability + state -- keeps every attached board visible in the milestone retro (the forget-pattern guard).`
- `preconditions_checked`: `records WHICH board resources were verified; SSH-not-SD-card for KV260.`

**Acceptance criteria:**
- `results/experiment_4600_hardware_continuity.json` is generated.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `complete: hardware_continuity_audit_<n>_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  GateMate DirtyJTAG detect command, and PolarFire SSH BatchMode command listed
  above.
- `per_board_status` is keyed by `kv260`, `gatemate`, and `polarfire`; each row
  records `reachable`, `status`, `precondition_resource`, `precondition_command`,
  reachability transcript, state transcript, and total board duration.
- Unreachable board rows use `blocked_kv260_ssh_unreachable`,
  `blocked_gatemate_usb_undetected`, or `blocked_polarfire_ssh_timeout` and do
  not fabricate a state transcript.
- Reachable KV260 rows include the `xmutil listapps` transcript from SSH,
  reachable GateMate rows include the GM1Ax IDCODE, and reachable PolarFire rows
  include the `uptime` transcript.
- The artifact includes `random_seed=4600`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4600)

---

### SCENARIO-HW-4600

**Scenario:** Exp 4600 records live board reachability and reachable-board state.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
GateMate DirtyJTAG GM1Ax detection, and PolarFire SSH reachability.
**When:** Experiment 4600 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4600_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one reachability plus
state row per attached board, precise per-board blocked statuses for unavailable
boards, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4600)

---

### REQ-HW-4612

**Title:** Hardware continuity audit MUST keep KV260, PolarFire, and GateMate visible with SSH/USB-only preconditions

**Description:**
Experiment 4612 MUST produce `results/experiment_4612_hardware_continuity.json`
as a lightweight hardware-continuity audit for the three attached boards. The
audit is not a bring-up, flash, or heavy benchmark. It MUST run the operator
requested reachability checks, record state only for reachable boards, and
continue after an individual board fails by recording a per-board
`blocked_<board>_<reason>` state.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this MUST be
  the SSH-reachability check and MUST NOT be replaced by a host SD-card-slot
  device check.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GateMate GM1Ax IDCODE such as `0x20000001`.

For each reachable board, the experiment MUST record the current board state:

- KV260: `ssh kria 'xmutil listapps'` to record the loaded overlay, and
  optionally `ssh kria 'ls /dev/uio*'` as a light energy-eval smoke when the
  Carnot Ising overlay is loaded.
- PolarFire: `ssh polarfire uptime`.
- GateMate: the detected GateMate IDCODE.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; success: hardware_continuity_<n>_of_3_boards_reachable (or complete:/blocked_ per board state).`
- `inference_substrate`: `hardware_smoke -- SSH/USB reachability checks, per-board floor; declared so a fast real check is not DURATION_TOO_SHORT false-flagged.`
- `boards_reachable`: `per-board reachability (kv260/polarfire/gatemate) -- the continuity record that keeps each attached board visible per the discipline.`
- `kv260_precondition`: `MUST be the SSH-reachability check, NEVER a host SD-card-slot device check (the KV260 SSH-Not-SD-Card Discipline).`
- `per_board_state`: `the recorded state per reachable board (loaded overlay / uptime / IDCODE) or blocked_<board>_<reason>.`
- `preconditions_checked`: `records WHICH board checks ran + their results; pre-empts silent-missing-hardware fabrication.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4612_hardware_continuity.py`
  writes `results/experiment_4612_hardware_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `success: hardware_continuity_<n>_of_3_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  PolarFire SSH BatchMode command, and GateMate DirtyJTAG detect command listed
  above.
- `boards_reachable` is keyed by `kv260`, `polarfire`, and `gatemate`.
- `kv260_precondition` records the KV260 SSH command and contains no host
  `/dev/mmcblk*` storage check.
- `per_board_state` records the KV260 loaded overlay transcript, PolarFire
  uptime transcript, and GateMate IDCODE for reachable boards; unreachable
  boards are recorded as `blocked_kv260_ssh_unreachable`,
  `blocked_polarfire_ssh_timeout`, or `blocked_gatemate_usb_undetected`.
- A reachable KV260 with a Carnot Ising overlay records the optional light
  energy-eval UIO smoke command `ssh kria 'ls /dev/uio*'`.
- The artifact includes `random_seed=4612`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4612)

---

### SCENARIO-HW-4612

**Scenario:** Exp 4612 records attached-board reachability and light state without heavy bring-up.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
PolarFire SSH reachability, and GateMate DirtyJTAG GateMate IDCODE detection.
**When:** Experiment 4612 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4612_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one state or blocked row
per attached board, optional KV260 UIO smoke only when a Carnot Ising overlay is
loaded, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4612)

---

### REQ-HW-4624

**Title:** Hardware continuity audit MUST record KV260, PolarFire, and GateMate reachability for Exp 4624

**Description:**
Experiment 4624 MUST produce `results/experiment_4624_hardware_continuity.json`
as a lightweight hardware-continuity audit for the three attached boards during
the ARC sprint. The audit is a reachability plus light state record, not a
heavy bring-up, flash, or benchmark. It MUST run the operator-requested
reachability checks, record state only for reachable boards, and continue after
an individual board fails by recording that board as
`blocked_<board>_<reason>`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this MUST be
  the SSH-reachability check and MUST NOT be replaced by a host SD-card-slot
  device check.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GateMate GM1Ax IDCODE such as `0x20000001`.

For each reachable board, the experiment MUST record current board state:

- KV260: `ssh kria 'xmutil listapps'` to record the loaded overlay, and
  optionally `ssh kria 'ls /dev/uio*'` as a light energy-eval smoke when the
  Carnot Ising overlay is loaded.
- PolarFire: `ssh polarfire uptime`.
- GateMate: the detected GateMate IDCODE.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; success: hardware_continuity_<n>_of_3_boards_reachable (or complete:/blocked_ per board state).`
- `inference_substrate`: `hardware_smoke -- SSH/USB reachability checks, per-board floor; declared so a fast real check is not DURATION_TOO_SHORT false-flagged.`
- `boards_reachable`: `per-board reachability (kv260/polarfire/gatemate) -- the continuity record that keeps each attached board visible per the discipline.`
- `kv260_precondition`: `MUST be the SSH-reachability check, NEVER a host SD-card-slot device check (the KV260 SSH-Not-SD-Card Discipline).`
- `per_board_state`: `the recorded state per reachable board (loaded overlay / uptime / IDCODE) or blocked_<board>_<reason>.`
- `preconditions_checked`: `records WHICH board checks ran + their results; pre-empts silent-missing-hardware fabrication.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4624_hardware_continuity.py`
  writes `results/experiment_4624_hardware_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `success: hardware_continuity_<n>_of_3_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  PolarFire SSH BatchMode command, and GateMate DirtyJTAG detect command listed
  above.
- `boards_reachable` is keyed by `kv260`, `polarfire`, and `gatemate`.
- `kv260_precondition` records the KV260 SSH command and contains no host
  `/dev/mmcblk*` storage check.
- `per_board_state` records the KV260 loaded overlay transcript, PolarFire
  uptime transcript, and GateMate IDCODE for reachable boards; unreachable
  boards are recorded as `blocked_kv260_ssh_unreachable`,
  `blocked_polarfire_ssh_timeout`, or `blocked_gatemate_usb_undetected`.
- A reachable KV260 with a Carnot Ising overlay records the optional light
  energy-eval UIO smoke command `ssh kria 'ls /dev/uio*'`.
- The artifact includes `random_seed=4624`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Implemented (Exp 4624)

---

### REQ-HW-4636

**Title:** Hardware continuity audit MUST record KV260, PolarFire, and GateMate reachability for Exp 4636

**Description:**
Experiment 4636 MUST produce `results/experiment_4636_hardware_continuity.json`
as a lightweight hardware-continuity audit for the three attached boards during
the ARC sprint. The audit is a reachability plus light state record, not a
heavy bring-up, flash, or benchmark. It MUST run the operator-requested
reachability checks, record state only for reachable boards, and continue after
an individual board fails by recording that board as
`blocked_<board>_<reason>`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this MUST be
  the SSH-reachability check and MUST NOT be replaced by a host SD-card-slot
  device check.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GateMate GM1Ax IDCODE such as `0x20000001`.

For each reachable board, the experiment MUST record current board state:

- KV260: `ssh kria 'xmutil listapps'` to record the loaded overlay, and
  optionally `ssh kria 'ls /dev/uio*'` as a light energy-eval smoke when the
  Carnot Ising overlay is loaded.
- PolarFire: `ssh polarfire uptime`.
- GateMate: the detected GateMate IDCODE.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; success: hardware_continuity_<n>_of_3_boards_reachable (or complete:/blocked_ per board state).`
- `inference_substrate`: `hardware_smoke -- SSH/USB reachability checks, per-board floor; declared so a fast real check is not DURATION_TOO_SHORT false-flagged.`
- `boards_reachable`: `per-board reachability (kv260/polarfire/gatemate) -- the continuity record that keeps each attached board visible per the discipline.`
- `kv260_precondition`: `MUST be the SSH-reachability check, NEVER a host SD-card-slot device check (the KV260 SSH-Not-SD-Card Discipline).`
- `per_board_state`: `the recorded state per reachable board (loaded overlay / uptime / IDCODE) or blocked_<board>_<reason>.`
- `preconditions_checked`: `records WHICH board checks ran + their results; pre-empts silent-missing-hardware fabrication.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4636_hardware_continuity.py`
  writes `results/experiment_4636_hardware_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `success: hardware_continuity_<n>_of_3_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  PolarFire SSH BatchMode command, and GateMate DirtyJTAG detect command listed
  above.
- `boards_reachable` is keyed by `kv260`, `polarfire`, and `gatemate`.
- `kv260_precondition` records the KV260 SSH command and contains no host
  `/dev/mmcblk*` storage check.
- `per_board_state` records the KV260 loaded overlay transcript, PolarFire
  uptime transcript, and GateMate IDCODE for reachable boards; unreachable
  boards are recorded as `blocked_kv260_ssh_unreachable`,
  `blocked_polarfire_ssh_timeout`, or `blocked_gatemate_usb_undetected`.
- A reachable KV260 with a Carnot Ising overlay records the optional light
  energy-eval UIO smoke command `ssh kria 'ls /dev/uio*'`.
- The artifact includes `random_seed=4636`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Implemented (Exp 4636)

---

### SCENARIO-HW-4636

**Scenario:** Exp 4636 records attached-board reachability and light state without heavy bring-up.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
PolarFire SSH reachability, and GateMate DirtyJTAG GateMate IDCODE detection.
**When:** Experiment 4636 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4636_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one state or blocked row
per attached board, optional KV260 UIO smoke only when a Carnot Ising overlay is
loaded, and no bitstream-build or speedup claim.

**Implementation status:** Implemented (Exp 4636)

---

### REQ-HW-4648

**Title:** Hardware continuity audit MUST record KV260, PolarFire, and GateMate reachability for Exp 4648

**Description:**
Experiment 4648 MUST produce `results/experiment_4648_hardware_continuity.json`
as a lightweight hardware-continuity audit for the three attached boards during
the ARC sprint. The audit is a reachability plus light state record, not a
heavy bring-up, flash, or benchmark. It MUST run the operator-requested
reachability checks, record state only for reachable boards, and continue after
an individual board fails by recording that board as
`blocked_<board>_<reason>`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this MUST be
  the SSH-reachability check and MUST NOT be replaced by a host SD-card-slot
  device check.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GateMate GM1Ax IDCODE such as `0x20000001`.

For each reachable board, the experiment MUST record current board state:

- KV260: `ssh kria 'xmutil listapps'` to record the loaded overlay, and
  optionally `ssh kria 'ls /dev/uio*'` as a light energy-eval smoke when the
  Carnot Ising overlay is loaded.
- PolarFire: `ssh polarfire uptime`.
- GateMate: the detected GateMate IDCODE.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; success: hardware_continuity_<n>_of_3_boards_reachable (or complete:/blocked_ per board state).`
- `inference_substrate`: `hardware_smoke -- SSH/USB reachability checks, per-board floor; declared so a fast real check is not DURATION_TOO_SHORT false-flagged.`
- `boards_reachable`: `per-board reachability (kv260/polarfire/gatemate) -- the continuity record that keeps each attached board visible per the discipline.`
- `kv260_precondition`: `MUST be the SSH-reachability check, NEVER a host SD-card-slot device check (the KV260 SSH-Not-SD-Card Discipline).`
- `per_board_state`: `the recorded state per reachable board (loaded overlay / uptime / IDCODE) or blocked_<board>_<reason>.`
- `preconditions_checked`: `records WHICH board checks ran + their results; pre-empts silent-missing-hardware fabrication.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4648_hardware_continuity.py`
  writes `results/experiment_4648_hardware_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `success: hardware_continuity_<n>_of_3_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  PolarFire SSH BatchMode command, and GateMate DirtyJTAG detect command listed
  above.
- `boards_reachable` is keyed by `kv260`, `polarfire`, and `gatemate`.
- `kv260_precondition` records the KV260 SSH command and contains no host
  `/dev/mmcblk*` storage check.
- `per_board_state` records the KV260 loaded overlay transcript, PolarFire
  uptime transcript, and GateMate IDCODE for reachable boards; unreachable
  boards are recorded as `blocked_kv260_ssh_unreachable`,
  `blocked_polarfire_ssh_timeout`, or `blocked_gatemate_usb_undetected`.
- A reachable KV260 with a Carnot Ising overlay records the optional light
  energy-eval UIO smoke command `ssh kria 'ls /dev/uio*'`.
- The artifact includes `random_seed=4648`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4648)

---

### SCENARIO-HW-4648

**Scenario:** Exp 4648 records attached-board reachability and light state without heavy bring-up.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
PolarFire SSH reachability, and GateMate DirtyJTAG GateMate IDCODE detection.
**When:** Experiment 4648 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4648_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one state or blocked row
per attached board, optional KV260 UIO smoke only when a Carnot Ising overlay is
loaded, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4648)

---

### REQ-HW-4660

**Title:** Hardware continuity audit MUST record KV260, PolarFire, and GateMate reachability for Exp 4660

**Description:**
Experiment 4660 MUST produce `results/experiment_4660_hardware_continuity.json`
as a lightweight hardware-continuity audit for the three attached boards during
the ARC sprint. The audit is a reachability plus light state record, not a
bitstream build, flash attempt, benchmark, or fabric-acceleration claim. It MUST
run each board's reachability probe, record exit codes, capture state only for
reachable boards, and continue after an individual board fails by recording that
board as `blocked_<board>_<reason>`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this MUST be
  the SSH-reachability check and MUST NOT be replaced by a host SD-card-slot
  device check.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GateMate GM1Ax IDCODE such as `0x20000001`.

For each reachable board, the experiment MUST record current board state:

- KV260: `ssh kria 'xmutil listapps'` to record the loaded overlay, and
  optionally `ssh kria 'ls /dev/uio*'` as a light energy-eval smoke when the
  Carnot Ising overlay is loaded.
- PolarFire: `ssh polarfire uptime`.
- GateMate: the detected GateMate IDCODE.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; success: hardware_continuity_<n>_of_3_boards_reachable (or complete:/blocked_ per board state).`
- `inference_substrate`: `hardware_smoke -- SSH/USB reachability checks, per-board floor; declared so a fast real check is not DURATION_TOO_SHORT false-flagged.`
- `kv260_precondition`: `MUST be the SSH-reachability check (ssh kria true), NEVER a host SD-card-slot device check (KV260 SSH-Not-SD-Card Discipline).`
- `boards_reachable`: `per-board reachability (kv260/polarfire/gatemate) -- the continuity record that keeps each attached board visible.`
- `per_board_state`: `the recorded state per reachable board (loaded overlay / uptime / IDCODE) or blocked_<board>_<reason>.`
- `reachable_board_count`: `the integer count of reachable boards (the at-a-glance continuity number).`
- `speedup_claim_made`: `MUST be false -- this is a reachability audit, not a fabric-acceleration claim (no fabrication).`
- `preconditions_checked`: `records WHICH board checks ran + their results; pre-empts silent-missing-hardware fabrication.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4660_hardware_continuity.py`
  writes `results/experiment_4660_hardware_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `success: hardware_continuity_<n>_of_3_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  PolarFire SSH BatchMode command, and GateMate DirtyJTAG detect command listed
  above.
- `boards_reachable` is keyed by `kv260`, `polarfire`, and `gatemate`.
- `kv260_precondition` records the KV260 SSH command and contains no host
  `/dev/mmcblk*` storage check.
- `per_board_state` records the KV260 loaded overlay transcript, PolarFire
  uptime transcript, and GateMate IDCODE for reachable boards; unreachable
  boards are recorded as `blocked_kv260_ssh_unreachable`,
  `blocked_polarfire_ssh_timeout`, or `blocked_gatemate_usb_undetected`.
- A reachable KV260 with a Carnot Ising overlay records the optional light
  energy-eval UIO smoke command `ssh kria 'ls /dev/uio*'`.
- The artifact includes `random_seed=4660`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4660)

---

### SCENARIO-HW-4660

**Scenario:** Exp 4660 records attached-board reachability and light state without heavy bring-up.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
PolarFire SSH reachability, and GateMate DirtyJTAG GateMate IDCODE detection.
**When:** Experiment 4660 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4660_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one state or blocked row
per attached board, optional KV260 UIO smoke only when a Carnot Ising overlay is
loaded, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4660)

---

### REQ-HW-4672

**Title:** Hardware continuity audit MUST record KV260, PolarFire, and GateMate reachability for Exp 4672

**Description:**
Experiment 4672 MUST produce `results/experiment_4672_hardware_continuity.json`
as a lightweight hardware-continuity audit for the three attached boards during
the ARC sprint. The audit is a reachability plus light state record, not a
bitstream build, flash attempt, benchmark, or fabric-acceleration claim. It MUST
run each board's reachability probe, record exit codes, capture state only for
reachable boards, and continue after an individual board fails by recording that
board as `blocked_<board>_<reason>`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this MUST be
  the SSH-reachability check and MUST NOT be replaced by a host SD-card-slot
  device check.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GateMate GM1Ax IDCODE such as `0x20000001`.

For each reachable board, the experiment MUST record current board state:

- KV260: `ssh kria 'xmutil listapps'` to record the loaded overlay, and
  optionally `ssh kria 'ls /dev/uio*'` as a light energy-eval smoke when the
  Carnot Ising overlay is loaded.
- PolarFire: `ssh polarfire uptime`.
- GateMate: the detected GateMate IDCODE.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; success: hardware_continuity_<n>_of_3_boards_reachable (or complete:/blocked_ per board state).`
- `inference_substrate`: `hardware_smoke -- SSH/USB reachability checks, per-board floor; declared so a fast real check is not DURATION_TOO_SHORT false-flagged.`
- `kv260_precondition`: `MUST be the SSH-reachability check (ssh kria true), NEVER a host SD-card-slot device check (KV260 SSH-Not-SD-Card Discipline).`
- `boards_reachable`: `per-board reachability (kv260/polarfire/gatemate) -- the continuity record that keeps each attached board visible.`
- `per_board_state`: `the recorded state per reachable board (loaded overlay / uptime / IDCODE) or blocked_<board>_<reason>.`
- `reachable_board_count`: `the integer count of reachable boards (the at-a-glance continuity number).`
- `speedup_claim_made`: `MUST be false -- this is a reachability audit, not a fabric-acceleration claim (no fabrication).`
- `preconditions_checked`: `records WHICH board checks ran + their results; pre-empts silent-missing-hardware fabrication.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4672_hardware_continuity.py`
  writes `results/experiment_4672_hardware_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `success: hardware_continuity_<n>_of_3_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  PolarFire SSH BatchMode command, and GateMate DirtyJTAG detect command listed
  above.
- `boards_reachable` is keyed by `kv260`, `polarfire`, and `gatemate`.
- `kv260_precondition` records the KV260 SSH command and contains no host
  `/dev/mmcblk*` storage check.
- `per_board_state` records the KV260 loaded overlay transcript, PolarFire
  uptime transcript, and GateMate IDCODE for reachable boards; unreachable
  boards are recorded as `blocked_kv260_ssh_unreachable`,
  `blocked_polarfire_ssh_timeout`, or `blocked_gatemate_usb_undetected`.
- A reachable KV260 with a Carnot Ising overlay records the optional light
  energy-eval UIO smoke command `ssh kria 'ls /dev/uio*'`.
- The artifact includes `random_seed=4672`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4672)

---

### SCENARIO-HW-4672

**Scenario:** Exp 4672 records attached-board reachability and light state without heavy bring-up.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
PolarFire SSH reachability, and GateMate DirtyJTAG GateMate IDCODE detection.
**When:** Experiment 4672 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4672_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one state or blocked row
per attached board, optional KV260 UIO smoke only when a Carnot Ising overlay is
loaded, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4672)

---

### REQ-HW-4684

**Title:** Hardware continuity audit MUST record KV260, PolarFire, and GateMate reachability for Exp 4684

**Description:**
Experiment 4684 MUST produce `results/experiment_4684_hardware_continuity.json`
as a lightweight hardware-continuity audit for the three attached boards during
the ARC sprint. The audit is a reachability plus light state record, not a
bitstream build, flash attempt, benchmark, or fabric-acceleration claim. It MUST
run each board's reachability probe, record exit codes, capture state only for
reachable boards, and continue after an individual board fails by recording that
board as `blocked_<board>_<reason>`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this MUST be
  the SSH-reachability check and MUST NOT be replaced by a host SD-card-slot
  device check.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GateMate GM1Ax IDCODE such as `0x20000001`.

For each reachable board, the experiment MUST record current board state:

- KV260: `ssh kria 'xmutil listapps'` to record the loaded overlay, and
  optionally `ssh kria 'ls /dev/uio*'` as a light energy-eval smoke when the
  Carnot Ising overlay is loaded.
- PolarFire: `ssh polarfire uptime`.
- GateMate: the detected GateMate IDCODE.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; success: hardware_continuity_<n>_of_3_boards_reachable (or complete:/blocked_ per board state).`
- `inference_substrate`: `hardware_smoke -- SSH/USB reachability checks, per-board floor; declared so a fast real check is not DURATION_TOO_SHORT false-flagged.`
- `kv260_precondition`: `MUST be the SSH-reachability check (ssh kria true), NEVER a host SD-card-slot device check (KV260 SSH-Not-SD-Card Discipline).`
- `boards_reachable`: `per-board reachability (kv260/polarfire/gatemate) -- the continuity record that keeps each attached board visible.`
- `per_board_state`: `the recorded state per reachable board (loaded overlay / uptime / IDCODE) or blocked_<board>_<reason>.`
- `reachable_board_count`: `the integer count of reachable boards (the at-a-glance continuity number).`
- `speedup_claim_made`: `MUST be false -- this is a reachability audit, not a fabric-acceleration claim (no fabrication).`
- `preconditions_checked`: `records WHICH board checks ran + their results; pre-empts silent-missing-hardware fabrication.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4684_hardware_continuity.py`
  writes `results/experiment_4684_hardware_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `success: hardware_continuity_<n>_of_3_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  PolarFire SSH BatchMode command, and GateMate DirtyJTAG detect command listed
  above.
- `boards_reachable` is keyed by `kv260`, `polarfire`, and `gatemate`.
- `kv260_precondition` records the KV260 SSH command and contains no host
  `/dev/mmcblk*` storage check.
- `per_board_state` records the KV260 loaded overlay transcript, PolarFire
  uptime transcript, and GateMate IDCODE for reachable boards; unreachable
  boards are recorded as `blocked_kv260_ssh_unreachable`,
  `blocked_polarfire_ssh_timeout`, or `blocked_gatemate_usb_undetected`.
- A reachable KV260 with a Carnot Ising overlay records the optional light
  energy-eval UIO smoke command `ssh kria 'ls /dev/uio*'`.
- The artifact includes `random_seed=4684`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4684)

---

### SCENARIO-HW-4684

**Scenario:** Exp 4684 records attached-board reachability and light state without heavy bring-up.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
PolarFire SSH reachability, and GateMate DirtyJTAG GateMate IDCODE detection.
**When:** Experiment 4684 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4684_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one state or blocked row
per attached board, optional KV260 UIO smoke only when a Carnot Ising overlay is
loaded, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4684)

---

### REQ-HW-4696

**Title:** Hardware continuity audit MUST record KV260, PolarFire, and GateMate reachability for Exp 4696

**Description:**
Experiment 4696 MUST produce `results/experiment_4696_hardware_continuity.json`
as a lightweight hardware-continuity audit for the three attached boards during
the ARC sprint. The audit is a reachability plus light state record, not a
bitstream build, flash attempt, benchmark, or fabric-acceleration claim. It MUST
run each board's reachability probe, record exit codes, capture state only for
reachable boards, and continue after an individual board fails by recording that
board as `blocked_<board>_<reason>`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this MUST be
  the SSH-reachability check and MUST NOT be replaced by a host SD-card-slot
  device check.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GateMate GM1Ax IDCODE such as `0x20000001`.

For each reachable board, the experiment MUST record current board state:

- KV260: `ssh kria 'xmutil listapps'` to record the loaded overlay, and
  optionally `ssh kria 'ls /dev/uio*'` as a light energy-eval smoke when the
  Carnot Ising overlay is loaded.
- PolarFire: `ssh polarfire uptime`.
- GateMate: the detected GateMate IDCODE.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; success: hardware_continuity_<n>_of_3_boards_reachable (or complete:/blocked_ per board state).`
- `inference_substrate`: `hardware_smoke -- SSH/USB reachability checks, per-board floor; declared so a fast real check is not DURATION_TOO_SHORT false-flagged.`
- `kv260_precondition`: `MUST be the SSH-reachability check (ssh kria true), NEVER a host SD-card-slot device check (KV260 SSH-Not-SD-Card Discipline).`
- `boards_reachable`: `per-board reachability (kv260/polarfire/gatemate) -- the continuity record that keeps each attached board visible.`
- `per_board_state`: `the recorded state per reachable board (loaded overlay / uptime / IDCODE) or blocked_<board>_<reason>.`
- `reachable_board_count`: `the integer count of reachable boards (the at-a-glance continuity number).`
- `speedup_claim_made`: `MUST be false -- this is a reachability audit, not a fabric-acceleration claim (no fabrication).`
- `preconditions_checked`: `records WHICH board checks ran + their results; pre-empts silent-missing-hardware fabrication.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4696_hardware_continuity.py`
  writes `results/experiment_4696_hardware_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `success: hardware_continuity_<n>_of_3_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  PolarFire SSH BatchMode command, and GateMate DirtyJTAG detect command listed
  above.
- `boards_reachable` is keyed by `kv260`, `polarfire`, and `gatemate`.
- `kv260_precondition` records the KV260 SSH command and contains no host
  `/dev/mmcblk*` storage check.
- `per_board_state` records the KV260 loaded overlay transcript, PolarFire
  uptime transcript, and GateMate IDCODE for reachable boards; unreachable
  boards are recorded as `blocked_kv260_ssh_unreachable`,
  `blocked_polarfire_ssh_timeout`, or `blocked_gatemate_usb_undetected`.
- A reachable KV260 with a Carnot Ising overlay records the optional light
  energy-eval UIO smoke command `ssh kria 'ls /dev/uio*'`.
- The artifact includes `random_seed=4696`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4696)

---

### SCENARIO-HW-4696

**Scenario:** Exp 4696 records attached-board reachability and light state without heavy bring-up.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
PolarFire SSH reachability, and GateMate DirtyJTAG GateMate IDCODE detection.
**When:** Experiment 4696 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4696_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one state or blocked row
per attached board, optional KV260 UIO smoke only when a Carnot Ising overlay is
loaded, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4696)

---

### REQ-HW-4708

**Title:** Hardware continuity audit MUST record KV260, PolarFire, and GateMate reachability for Exp 4708

**Description:**
Experiment 4708 MUST produce `results/experiment_4708_hardware_continuity.json`
as a lightweight hardware-continuity audit for the three attached boards during
the ARC sprint. The audit is a reachability plus light state record, not a
bitstream build, flash attempt, benchmark, or fabric-acceleration claim. It MUST
run each board's reachability probe, record exit codes, capture state only for
reachable boards, and continue after an individual board fails by recording that
board as `blocked_<board>_<reason>`.

The only valid preconditions are:

- KV260: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; this MUST be
  the SSH-reachability check and MUST NOT be replaced by a host SD-card-slot
  device check.
- PolarFire: `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`.
- GateMate: `openFPGALoader -c dirtyJtag --detect`; reachability requires a
  successful command and a GateMate GM1Ax IDCODE such as `0x20000001`.

For each reachable board, the experiment MUST record current board state:

- KV260: `ssh kria 'xmutil listapps'` to record the loaded overlay, and
  optionally `ssh kria 'ls /dev/uio*'` as a light energy-eval smoke when the
  Carnot Ising overlay is loaded.
- PolarFire: `ssh polarfire uptime`.
- GateMate: the detected GateMate IDCODE.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; success: hardware_continuity_<n>_of_3_boards_reachable (or complete:/blocked_ per board state).`
- `inference_substrate`: `hardware_smoke -- SSH/USB reachability checks, per-board floor; declared so a fast real check is not DURATION_TOO_SHORT false-flagged.`
- `kv260_precondition`: `MUST be the SSH-reachability check (ssh kria true), NEVER a host SD-card-slot device check (KV260 SSH-Not-SD-Card Discipline).`
- `boards_reachable`: `per-board reachability (kv260/polarfire/gatemate) -- the continuity record that keeps each attached board visible.`
- `per_board_state`: `the recorded state per reachable board (loaded overlay / uptime / IDCODE) or blocked_<board>_<reason>.`
- `reachable_board_count`: `the integer count of reachable boards (the at-a-glance continuity number).`
- `speedup_claim_made`: `MUST be false -- this is a reachability audit, not a fabric-acceleration claim (no fabrication).`
- `preconditions_checked`: `records WHICH board checks ran + their results; pre-empts silent-missing-hardware fabrication.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4708_hardware_continuity.py`
  writes `results/experiment_4708_hardware_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `honest_verdict` is `success: hardware_continuity_<n>_of_3_boards_reachable`
  where `<n>` is the number of reachable boards.
- `preconditions_checked` records exactly the KV260 SSH BatchMode command,
  PolarFire SSH BatchMode command, and GateMate DirtyJTAG detect command listed
  above.
- `boards_reachable` is keyed by `kv260`, `polarfire`, and `gatemate`.
- `kv260_precondition` records the KV260 SSH command and contains no host
  `/dev/mmcblk*` storage check.
- `per_board_state` records the KV260 loaded overlay transcript, PolarFire
  uptime transcript, and GateMate IDCODE for reachable boards; unreachable
  boards are recorded as `blocked_kv260_ssh_unreachable`,
  `blocked_polarfire_ssh_timeout`, or `blocked_gatemate_usb_undetected`.
- A reachable KV260 with a Carnot Ising overlay records the optional light
  energy-eval UIO smoke command `ssh kria 'ls /dev/uio*'`.
- The artifact includes `random_seed=4708`, `spec_refs`, `field_principles`,
  `duration_s`, `fabric_acceleration_claimed=false`,
  `speedup_claim_made=false`, `bitstream_build_attempted=false`, and a stable
  `reproducibility_checksum`.
- The artifact contains no host SD-card block-device precondition.

**Implementation status:** Pending (Exp 4708)

---

### REQ-HW-4733

**Title:** KV260 continuity MUST capture an SSH-gated on-board Ising latency transcript for Exp 4733

**Description:**
Experiment 4733 MUST produce `results/experiment_4733_kv260_continuity.json`
as the next KV260 hardware-continuity artifact for the ARC sprint. The
precondition MUST be board SSH reachability via
`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; a host SD-card device
node check is not a valid precondition. If that SSH command returns non-zero,
the artifact MUST stop honestly with
`honest_verdict=complete:/blocked_kv260_ssh_unreachable`, record the failed
SSH precondition in `preconditions_checked`, set `kv260_ssh_reachable=false`,
set `kv260_latency_numbers=null`, set `kv260_synthesis_succeeded=false`, and
avoid fabricated latency values.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'` and
preserve that transcript. If the non-sudo `xmutil` probe reports that root is
required, the experiment MAY retry with `ssh kria 'sudo xmutil listapps'`.
When no Carnot Ising overlay is listed as loaded, the experiment MUST load the
current deployable Carnot Ising app over SSH, preferring
`sudo xmutil loadapp carnot_ising_v4_n64` when available and otherwise using
the current `carnot_ising_v2_n64`/`carnot_ising_v4` app. The experiment MUST
then run an on-board Python UIO harness with `ssh kria 'sudo python3 -'` to
capture per-sample wall-clock latency numbers for a deterministic n=64 Ising
problem. The harness MUST use the UIO register interface and must report at
least 30 positive per-sample latency values.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; success: kv260_latency_transcript_captured OR complete:/blocked_kv260_ssh_unreachable (an honest non-terminal continuity record).`
- `inference_substrate`: `hardware_smoke -- an SSH-attached board test; the duration floor is per-board, not the 60s live-model floor.`
- `kv260_ssh_reachable`: `the SSH-reachability result (the CORRECT precondition; NEVER host SD-card device nodes).`
- `kv260_latency_numbers`: `the on-board per-sample Ising latency transcript (the terminal-state deliverable) -- present only if reachable; null + blocked verdict otherwise.`
- `kv260_synthesis_succeeded`: `true if the carnot Ising overlay loaded + ran on-board -- part of the terminal-state definition (north-star §3).`
- `preconditions_checked`: `records the SSH-reachability check (NEVER host SD-card); pre-empts the wrong-mechanism SD-card confusion + missing-resource fabrication.`
- `verifier_is_oracle`: `false -- a hardware latency measurement invokes no verifier oracle.`
- `random_seed`: `determinism precondition for reproducibility (the sampler seed).`
- `reproducibility_checksum`: `content-addressed hash of the bitstream + transcript inputs.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4733_kv260_continuity.py` writes
  `results/experiment_4733_kv260_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `complete:/blocked_kv260_ssh_unreachable`, `kv260_latency_numbers` is null,
  `kv260_synthesis_succeeded=false`, and no board latency transcript is
  fabricated.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success: kv260_latency_transcript_captured`, `kv260_latency_numbers`
  contains at least 30 positive per-sample wall-clock values with summary
  statistics, `kv260_synthesis_succeeded=true`, and the command transcript
  proves the Carnot Ising overlay was listed or loaded and the UIO harness ran
  on-board.
- `verifier_is_oracle=false`, `random_seed=4733`, `field_principles`,
  `duration_s`, `command_probes`, `overlay_loaded`, `bitstream_sha256`, and a
  stable `reproducibility_checksum` are present.

**Implementation status:** Planned (Exp 4733)

---

### SCENARIO-HW-4733

**Scenario:** Exp 4733 records the KV260 terminal latency transcript or an
honest SSH-unreachable continuity record.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4733 checks SSH, confirms or loads a Carnot Ising overlay
with `xmutil`, and runs the UIO latency harness only when the board is
reachable.
**Then:** It writes `results/experiment_4733_kv260_continuity.json` with a
terminal-prefixed `honest_verdict`, `inference_substrate=hardware_smoke`,
principle-annotated required fields, `verifier_is_oracle=false`, deterministic
`random_seed=4733`, and either a real on-board per-sample latency transcript or
`complete:/blocked_kv260_ssh_unreachable` with null latency numbers.

**Implementation status:** Planned (Exp 4733)

---

### SCENARIO-HW-4708

**Scenario:** Exp 4708 records attached-board reachability and light state without heavy bring-up.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
PolarFire SSH reachability, and GateMate DirtyJTAG GateMate IDCODE detection.
**When:** Experiment 4708 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4708_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one state or blocked row
per attached board, optional KV260 UIO smoke only when a Carnot Ising overlay is
loaded, and no bitstream-build or speedup claim.

**Implementation status:** Pending (Exp 4708)

---

### SCENARIO-HW-4624

**Scenario:** Exp 4624 records attached-board reachability and light state without heavy bring-up.

**Given:** The hardware-continuity task requires KV260 SSH reachability only,
PolarFire SSH reachability, and GateMate DirtyJTAG GateMate IDCODE detection.
**When:** Experiment 4624 runs the three required preconditions and then records
state only for boards that are reachable.
**Then:** It writes `results/experiment_4624_hardware_continuity.json` with a
terminal-prefixed board-count `honest_verdict`, `inference_substrate` equal to
`hardware_smoke`, principle-annotated required fields, one state or blocked row
per attached board, optional KV260 UIO smoke only when a Carnot Ising overlay is
loaded, and no bitstream-build or speedup claim.

**Implementation status:** Implemented (Exp 4624)

---

### REQ-HW-4745

**Title:** KV260 continuity MUST capture the ARC sprint SSH-gated Ising latency transcript for Exp 4745

**Description:**
Experiment 4745 MUST produce `results/experiment_4745_kv260_continuity.json`
as the next KV260 hardware-continuity artifact. The precondition MUST be board
SSH reachability via `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`;
a host SD-card device-node check is not a valid precondition. If that SSH
command returns non-zero, the artifact MUST stop honestly with
`honest_verdict=complete:/blocked_kv260_ssh_unreachable`, record the failed
SSH precondition in `preconditions_checked`, set `kv260_ssh_reachable=false`,
set `kv260_latency_numbers=null`, set `kv260_synthesis_succeeded=false`, and
avoid fabricated latency values.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'` and
preserve that transcript. If the non-sudo `xmutil` probe reports that root is
required, the experiment MAY retry with `ssh kria 'sudo xmutil listapps'`.
When no Carnot Ising overlay is listed as loaded, the experiment MUST load the
current deployable Carnot Ising app over SSH, preferring
`sudo xmutil loadapp carnot_ising_v4_n64` when available and otherwise using
the current `carnot_ising_v2_n64`/`carnot_ising_v4` app. The experiment MUST
then run an on-board Python UIO harness with `ssh kria 'sudo python3 -'` to
capture per-sample wall-clock latency numbers for a deterministic n=64 Ising
problem. The harness MUST use the UIO register interface and must report at
least 30 positive per-sample latency values.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; success: kv260_latency_transcript_captured OR complete:/blocked_kv260_ssh_unreachable (an honest non-terminal continuity record).`
- `inference_substrate`: `hardware_smoke -- an SSH-attached board test; the duration floor is per-board, not the 60s live-model floor.`
- `kv260_ssh_reachable`: `the SSH-reachability result (the CORRECT precondition; NEVER host SD-card device nodes).`
- `kv260_latency_numbers`: `the on-board per-sample Ising latency transcript (the terminal-state deliverable) -- present only if reachable; null + blocked verdict otherwise.`
- `kv260_synthesis_succeeded`: `true if the carnot Ising overlay loaded + ran on-board -- part of the terminal-state definition (north-star §3).`
- `preconditions_checked`: `records the SSH-reachability check (NEVER host SD-card); pre-empts the wrong-mechanism SD-card confusion + missing-resource fabrication.`
- `verifier_is_oracle`: `false -- a hardware latency measurement invokes no verifier oracle.`
- `random_seed`: `determinism precondition for reproducibility (the sampler seed).`
- `reproducibility_checksum`: `content-addressed hash of the bitstream + transcript inputs.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4745_kv260_continuity.py` writes
  `results/experiment_4745_kv260_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `complete:/blocked_kv260_ssh_unreachable`, `kv260_latency_numbers` is null,
  `kv260_synthesis_succeeded=false`, and no board latency transcript is
  fabricated.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success: kv260_latency_transcript_captured`, `kv260_latency_numbers`
  contains at least 30 positive per-sample wall-clock values with summary
  statistics, `kv260_synthesis_succeeded=true`, and the command transcript
  proves the Carnot Ising overlay was listed or loaded and the UIO harness ran
  on-board.
- `verifier_is_oracle=false`, `random_seed=4745`, `field_principles`,
  `duration_s`, `command_probes`, `overlay_loaded`, `bitstream_sha256`, and a
  stable `reproducibility_checksum` are present.

**Implementation status:** Implemented (Exp 4745)

---

### SCENARIO-HW-4745

**Scenario:** Exp 4745 records the KV260 terminal latency transcript or an
honest SSH-unreachable continuity record.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4745 checks SSH, confirms or loads a Carnot Ising overlay
with `xmutil`, and runs the UIO latency harness only when the board is
reachable.
**Then:** It writes `results/experiment_4745_kv260_continuity.json` with a
terminal-prefixed `honest_verdict`, `inference_substrate=hardware_smoke`,
principle-annotated required fields, `verifier_is_oracle=false`, deterministic
`random_seed=4745`, and either a real on-board per-sample latency transcript or
`complete:/blocked_kv260_ssh_unreachable` with null latency numbers.

**Implementation status:** Implemented (Exp 4745)

---

### REQ-HW-4757

**Title:** KV260 per-milestone continuity MUST use SSH-only reachability and record the next board step

**Description:**
Experiment 4757 MUST produce `results/experiment_4757_kv260_continuity.json`
as the per-milestone KV260 continuity artifact. The precondition MUST be board
SSH reachability via `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`;
the retired host SD-card device-node mechanism MUST NOT be used. If that SSH
command returns non-zero, the artifact MUST stop honestly with
`honest_verdict=complete:/blocked_kv260_ssh_unreachable`, record the failed
SSH precondition in `preconditions_checked`, set `kv260_ssh_reachable=false`,
and record the next forward step as restoring SSH and rerunning the SSH-only
continuity check.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
to report the loaded overlay and MUST preserve the direct command transcript.
If the non-sudo `xmutil` probe reports that root is required, the experiment MAY
retry with `ssh kria 'sudo xmutil listapps'` only to read board state; it MUST
not use host SD-card device nodes. The experiment MUST also record lightweight
board state over SSH, including hostname, kernel/OS line, uptime, and UIO device
count. Per `ops/hardware-bringup-prep.md`, the reachable-board
`next_forward_step` MUST preserve the KV260 terminal state as:
`GRADUATED: KV260 terminal criteria met; continue per-milestone SSH-only continuity checks.`

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; blocked_kv260_ssh_unreachable is an honest non-terminal block, a continuity-recorded run is complete_.`
- `inference_substrate`: `hardware_smoke (SSH-attached board).`
- `preconditions_checked`: `records the SSH-reachability check (NEVER host SD-card) -- the KV260 wrong-mechanism guard.`
- `kv260_ssh_reachable`: `the board's SSH reachability -- the only valid KV260 precondition.`
- `next_forward_step`: `the next concrete board step recorded so the board stays visible in retros (the forget-pattern guard).`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4757_kv260_continuity.py` writes
  `results/experiment_4757_kv260_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `complete:/blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command
  is run, and `next_forward_step` tells the operator to restore SSH and rerun
  the SSH-only continuity check.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success: kv260_continuity_recorded`, the direct `ssh kria 'xmutil listapps'`
  transcript is preserved, a sudo read-only fallback transcript is preserved
  when root is required, the loaded Carnot overlay is reported when present,
  lightweight board state is recorded, and `next_forward_step` records the
  terminal KV260 continuity action above.
- `verifier_is_oracle=false`, `random_seed=4757`, `field_principles`,
  `duration_s`, `command_probes`, and a stable `reproducibility_checksum` are
  present.

**Implementation status:** Planned (Exp 4757)

---

### SCENARIO-HW-4757

**Scenario:** Exp 4757 records KV260 continuity or an honest SSH-unreachable block.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4757 checks SSH and, only when reachable, records the
`xmutil listapps` transcript plus lightweight board state over SSH.
**Then:** It writes `results/experiment_4757_kv260_continuity.json` with a
terminal-prefixed `honest_verdict`, `inference_substrate=hardware_smoke`,
principle-annotated required fields, `kv260_ssh_reachable`, deterministic
`random_seed=4757`, a concrete `next_forward_step`, and no host SD-card
precondition.

**Implementation status:** Planned (Exp 4757)

---

### REQ-HW-4767

**Title:** KV260 hardware continuity MUST use SSH-only reachability and preserve the board state for Exp 4767

**Description:**
Experiment 4767 MUST produce `results/experiment_4767_kv260_continuity.json`
as the next KV260 hardware-continuity artifact. The precondition MUST be board
SSH reachability via `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`;
host SD-card device nodes are permanently retired and MUST NOT be used as a
KV260 precondition. If that SSH command returns non-zero, the artifact MUST stop
honestly with `honest_verdict=blocked_kv260_ssh_unreachable`, record the failed
SSH precondition in `preconditions_checked`, set `kv260_ssh_reachable=false`,
skip `xmutil` and board-state commands, and record the next forward step as
restoring SSH and rerunning the SSH-only continuity check.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'` and
preserve that direct transcript to report the loaded overlay. If non-sudo
`xmutil` reports that root privileges are required, the experiment MAY retry
with `ssh kria 'sudo xmutil listapps'` as a read-only board-state fallback and
MUST preserve both transcripts. The experiment MUST also record lightweight
board state over SSH, including hostname, kernel/OS line, uptime, and UIO device
count. The reachable-board `next_forward_step` MUST preserve the terminal-state
definition from the most recent KV260 artifact:
`GRADUATED: KV260 terminal criteria met; continue per-milestone SSH-only continuity checks.`

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; reachable + state recorded is success_/complete_; unreachable is blocked_kv260_ssh_unreachable.`
- `inference_substrate`: `hardware_smoke -- SSH-attached board test.`
- `kv260_ssh_reachable`: `the SSH reachability check -- the ONLY valid KV260 precondition (not host SD-card presence).`
- `preconditions_checked`: `records the SSH check so a wrong-mechanism (SD-card) precondition cannot escalate the operator for a no-op.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4767_kv260_continuity.py` writes
  `results/experiment_4767_kv260_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command is run,
  and `next_forward_step` tells the operator to restore SSH and rerun the
  SSH-only continuity check.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success: kv260_continuity_recorded`, the direct `ssh kria 'xmutil listapps'`
  transcript is preserved, a sudo read-only fallback transcript is preserved
  when root is required, the loaded overlay is reported when present, lightweight
  board state is recorded, and `next_forward_step` records the terminal KV260
  continuity action above.
- `verifier_is_oracle=false`, `random_seed=4767`, `field_principles`,
  `duration_s`, `command_probes`, and a stable `reproducibility_checksum` are
  present.

**Implementation status:** Implemented (Exp 4767)

---

### SCENARIO-HW-4767

**Scenario:** Exp 4767 records SSH-attached KV260 continuity or an honest SSH-unreachable block.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4767 checks SSH and, only when reachable, records the
direct `xmutil listapps` transcript plus lightweight board state over SSH.
**Then:** It writes `results/experiment_4767_kv260_continuity.json` with a
terminal-prefixed reachable `honest_verdict`, `inference_substrate=hardware_smoke`,
principle-annotated required fields, `kv260_ssh_reachable`, deterministic
`random_seed=4767`, a concrete `next_forward_step`, and no host SD-card
precondition.

**Implementation status:** Implemented (Exp 4767)

---

### REQ-HW-4777

**Title:** KV260 hardware continuity MUST preserve SSH-only board state for Exp 4777

**Description:**
Experiment 4777 MUST produce `results/experiment_4777_kv260_continuity.json`
as the next KV260 hardware-continuity artifact. The precondition MUST be board
SSH reachability via `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`;
host SD-card device nodes are permanently retired for KV260 and MUST NOT be used
as a precondition. If that SSH command returns non-zero, the artifact MUST stop
honestly with `honest_verdict=blocked_kv260_ssh_unreachable`, record the failed
SSH precondition in `preconditions_checked`, set `kv260_ssh_reachable=false`,
skip `xmutil` and board-state commands, and record the next forward step as
restoring SSH and rerunning the SSH-only continuity check.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'` and
preserve that direct transcript to report the loaded overlay. If non-sudo
`xmutil` reports that root privileges are required, the experiment MAY retry
with `ssh kria 'sudo xmutil listapps'` as a read-only board-state fallback and
MUST preserve both transcripts. The experiment MUST also record lightweight
board state over SSH, including hostname, kernel/OS line, uptime, and UIO device
count. The reachable-board `next_forward_step` MUST preserve the terminal-state
definition from the most recent KV260 artifact:
`GRADUATED: KV260 terminal criteria met; continue per-milestone SSH-only continuity checks.`
If a later operator action is needed, the artifact MUST queue a documentation or
audit step instead of dropping the KV260 continuity thread.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; reachable + state recorded is success_/complete_; unreachable is blocked_kv260_ssh_unreachable.`
- `inference_substrate`: `hardware_smoke -- SSH-attached board test.`
- `kv260_ssh_reachable`: `the SSH reachability check -- the ONLY valid KV260 precondition (not host SD-card presence).`
- `preconditions_checked`: `records the SSH check so a wrong-mechanism precondition cannot escalate the operator for a no-op.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4777_kv260_continuity.py` writes
  `results/experiment_4777_kv260_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command is run,
  and `next_forward_step` tells the operator to restore SSH and rerun the
  SSH-only continuity check.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success: kv260_continuity_recorded`, the direct `ssh kria 'xmutil listapps'`
  transcript is preserved, a sudo read-only fallback transcript is preserved
  when root is required, the loaded overlay is reported when present, lightweight
  board state is recorded, and `next_forward_step` records the terminal KV260
  continuity action above.
- `verifier_is_oracle=false`, `random_seed=4777`, `field_principles`,
  `duration_s`, `command_probes`, and a stable `reproducibility_checksum` are
  present.

**Implementation status:** Implemented (Exp 4777)

---

### SCENARIO-HW-4777

**Scenario:** Exp 4777 records SSH-attached KV260 continuity or an honest SSH-unreachable block.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4777 checks SSH and, only when reachable, records the
direct `xmutil listapps` transcript plus lightweight board state over SSH.
**Then:** It writes `results/experiment_4777_kv260_continuity.json` with a
terminal-prefixed reachable `honest_verdict`, `inference_substrate=hardware_smoke`,
principle-annotated required fields, `kv260_ssh_reachable`, deterministic
`random_seed=4777`, a concrete `next_forward_step`, and no host SD-card
precondition.

**Implementation status:** Implemented (Exp 4777)

---

### REQ-HW-4787

**Title:** KV260 hardware continuity MUST preserve SSH-only board state and next-step discipline for Exp 4787

**Description:**
Experiment 4787 MUST produce `results/experiment_4787_kv260_continuity.json`
as the next KV260 hardware-continuity artifact. The precondition MUST be board
SSH reachability via `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`;
host SD-card device nodes are permanently retired for KV260 and MUST NOT be used
as a precondition. If that SSH command returns non-zero, the artifact MUST stop
honestly with `honest_verdict=blocked_kv260_ssh_unreachable`, record the failed
SSH precondition in `preconditions_checked`, set `kv260_ssh_reachable=false`,
skip `xmutil` and board-state commands, and record a concrete operator/audit
step to restore SSH and rerun the SSH-only continuity check.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'` and
preserve that direct transcript to report the loaded overlay. If non-sudo
`xmutil` reports that root privileges are required, the experiment MAY retry
with `ssh kria 'sudo xmutil listapps'` as a read-only board-state fallback and
MUST preserve both transcripts. The experiment MUST also record lightweight
board state over SSH, including hostname, kernel/OS line, uptime, and UIO device
count. The reachable-board `next_forward_step` MUST record the next concrete
KV260 track step as:
`GRADUATED: KV260 terminal criteria met; continue per-milestone SSH-only continuity checks. Next concrete step: keep the board in the continuity rotation and queue an overlay-state audit only if a future SSH or xmutil probe changes.`
If SSH is unreachable, the blocked `next_forward_step` MUST queue the operator
and audit action as:
`OPERATOR ACTION: restore KV260 SSH reachability, then rerun the SSH-only continuity check; operator/audit queue: document the SSH outage and keep host SD-card preconditions retired.`

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; reachable + state recorded is success_/complete_; unreachable is blocked_kv260_ssh_unreachable.`
- `inference_substrate`: `hardware_smoke (SSH-attached board test).`
- `kv260_ssh_reachable`: `the SSH reachability check -- the ONLY valid KV260 precondition (not host SD-card presence).`
- `preconditions_checked`: `records the SSH check so a wrong-mechanism precondition cannot escalate the operator for a no-op.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4787_kv260_continuity.py` writes
  `results/experiment_4787_kv260_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command is run,
  and `next_forward_step` queues an operator/audit step to restore SSH and
  rerun the SSH-only continuity check.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success: kv260_continuity_recorded`, the direct `ssh kria 'xmutil listapps'`
  transcript is preserved, a sudo read-only fallback transcript is preserved
  when root is required, the loaded overlay is reported when present, lightweight
  board state is recorded, and `next_forward_step` records the next concrete
  KV260 continuity action.
- `verifier_is_oracle=false`, `random_seed=4787`, `field_principles`,
  `duration_s`, `command_probes`, and a stable `reproducibility_checksum` are
  present.

**Implementation status:** Implemented (Exp 4787)

---

### SCENARIO-HW-4787

**Scenario:** Exp 4787 records SSH-attached KV260 continuity or an honest SSH-unreachable block.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4787 checks SSH and, only when reachable, records the
direct `xmutil listapps` transcript plus lightweight board state over SSH.
**Then:** It writes `results/experiment_4787_kv260_continuity.json` with a
terminal-prefixed reachable `honest_verdict`, `inference_substrate=hardware_smoke`,
principle-annotated required fields, `kv260_ssh_reachable`, deterministic
`random_seed=4787`, a concrete `next_forward_step`, and no host SD-card
precondition.

**Implementation status:** Implemented (Exp 4787)

---

### REQ-HW-4797

**Title:** KV260 hardware continuity MUST preserve SSH-only board state and next-step discipline for Exp 4797

**Description:**
Experiment 4797 MUST produce `results/experiment_4797_kv260_continuity.json`
as the next KV260 hardware-continuity artifact. The precondition MUST be board
SSH reachability via `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`;
host SD-card device nodes are permanently retired for KV260 and MUST NOT be used
as a precondition. If that SSH command returns non-zero, the artifact MUST stop
honestly with `honest_verdict=blocked_kv260_ssh_unreachable`, record the failed
SSH precondition in `preconditions_checked`, set `kv260_ssh_reachable=false`,
skip `xmutil` and board-state commands, and record a concrete operator/audit
step to restore SSH and rerun the SSH-only continuity check.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'` and
preserve that direct transcript to report the loaded overlay. If non-sudo
`xmutil` reports that root privileges are required, the experiment MAY retry
with `ssh kria 'sudo xmutil listapps'` as a read-only board-state fallback and
MUST preserve both transcripts. The experiment MUST also record lightweight
board state over SSH, including hostname, kernel/OS line, uptime, and UIO device
count. The reachable-board `next_forward_step` MUST record the next concrete
KV260 track step as:
`GRADUATED: KV260 terminal criteria met; continue per-milestone SSH-only continuity checks. Next concrete step: keep the board in the continuity rotation and queue an overlay-state audit only if a future SSH or xmutil probe changes.`
If SSH is unreachable, the blocked `next_forward_step` MUST queue the operator
and audit action as:
`OPERATOR ACTION: restore KV260 SSH reachability, then rerun the SSH-only continuity check; operator/audit queue: document the SSH outage and keep host SD-card preconditions retired.`

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; reachable is success_/complete_; unreachable is blocked_kv260_ssh_unreachable.`
- `inference_substrate`: `hardware_smoke (SSH-attached board test).`
- `kv260_ssh_reachable`: `the SSH reachability check -- the ONLY valid KV260 precondition.`
- `preconditions_checked`: `records the SSH check so a wrong-mechanism precondition cannot escalate the operator for a no-op.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4797_kv260_continuity.py` writes
  `results/experiment_4797_kv260_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command is run,
  and `next_forward_step` queues an operator/audit step to restore SSH and
  rerun the SSH-only continuity check.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success: kv260_continuity_recorded`, the direct `ssh kria 'xmutil listapps'`
  transcript is preserved, a sudo read-only fallback transcript is preserved
  when root is required, the loaded overlay is reported when present, lightweight
  board state is recorded, and `next_forward_step` records the next concrete
  KV260 continuity action.
- `verifier_is_oracle=false`, `random_seed=4797`, `field_principles`,
  `duration_s`, `command_probes`, and a stable `reproducibility_checksum` are
  present.

**Implementation status:** Implemented (Exp 4797)

---

### REQ-HW-4807

**Title:** KV260 hardware continuity MUST preserve SSH-only board state and next-step discipline for Exp 4807

**Description:**
Experiment 4807 MUST produce `results/experiment_4807_kv260_continuity.json`
as the next KV260 hardware-continuity artifact. The precondition MUST be board
SSH reachability via `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`;
host SD-card device nodes are permanently retired for KV260 and MUST NOT be used
as a precondition. If that SSH command returns non-zero, the artifact MUST stop
honestly with `honest_verdict=blocked_kv260_ssh_unreachable`, record the failed
SSH precondition in `preconditions_checked`, set `kv260_ssh_reachable=false`,
skip `xmutil` and board-state commands, and record a concrete operator/audit
step to restore SSH and rerun the SSH-only continuity check.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'` and
preserve that direct transcript to report the loaded overlay. If non-sudo
`xmutil` reports that root privileges are required, the experiment MAY retry
with `ssh kria 'sudo xmutil listapps'` as a read-only board-state fallback and
MUST preserve both transcripts. The experiment MUST also record lightweight
board state over SSH, including hostname, kernel/OS line, uptime, and UIO device
count. The reachable-board `next_forward_step` MUST record the next concrete
KV260 track step as:
`GRADUATED: KV260 terminal criteria met; continue per-milestone SSH-only continuity checks. Next concrete step: keep the board in the continuity rotation and queue an overlay-state audit only if a future SSH or xmutil probe changes.`
If SSH is unreachable, the blocked `next_forward_step` MUST queue the operator
and audit action as:
`OPERATOR ACTION: restore KV260 SSH reachability, then rerun the SSH-only continuity check; operator/audit queue: document the SSH outage and keep host SD-card preconditions retired.`

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; reachable is success_/complete_; unreachable is blocked_kv260_ssh_unreachable.`
- `inference_substrate`: `hardware_smoke (SSH-attached board test).`
- `kv260_ssh_reachable`: `the SSH reachability check -- the ONLY valid KV260 precondition.`
- `preconditions_checked`: `records the SSH check so a wrong-mechanism precondition cannot escalate the operator for a no-op.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4807_kv260_continuity.py` writes
  `results/experiment_4807_kv260_continuity.json`.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command is run,
  and `next_forward_step` queues an operator/audit step to restore SSH and
  rerun the SSH-only continuity check.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success: kv260_continuity_recorded`, the direct `ssh kria 'xmutil listapps'`
  transcript is preserved, a sudo read-only fallback transcript is preserved
  when root is required, the loaded overlay is reported when present, lightweight
  board state is recorded, and `next_forward_step` records the next concrete
  KV260 continuity action.
- `verifier_is_oracle=false`, `random_seed=4807`, `field_principles`,
  `duration_s`, `command_probes`, and a stable `reproducibility_checksum` are
  present.

**Implementation status:** Implemented (Exp 4807)

---

### SCENARIO-HW-4807

**Scenario:** Exp 4807 records SSH-attached KV260 continuity or an honest SSH-unreachable block.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4807 checks SSH and, only when reachable, records the
direct `xmutil listapps` transcript plus lightweight board state over SSH.
**Then:** It writes `results/experiment_4807_kv260_continuity.json` with a
terminal-prefixed reachable `honest_verdict`, `inference_substrate=hardware_smoke`,
principle-annotated required fields, `kv260_ssh_reachable`, deterministic
`random_seed=4807`, a concrete `next_forward_step`, and no host SD-card
precondition.

**Implementation status:** Implemented (Exp 4807)

---

### SCENARIO-HW-4797

**Scenario:** Exp 4797 records SSH-attached KV260 continuity or an honest SSH-unreachable block.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4797 checks SSH and, only when reachable, records the
direct `xmutil listapps` transcript plus lightweight board state over SSH.
**Then:** It writes `results/experiment_4797_kv260_continuity.json` with a
terminal-prefixed reachable `honest_verdict`, `inference_substrate=hardware_smoke`,
principle-annotated required fields, `kv260_ssh_reachable`, deterministic
`random_seed=4797`, a concrete `next_forward_step`, and no host SD-card
precondition.

**Implementation status:** Implemented (Exp 4797)

---

### REQ-HW-4827

**Title:** KV260 continuity MUST write the Exp 4827 deliverable even when SSH is unreachable

**Description:**
Experiment 4827 MUST produce `results/experiment_4827_kv260_continuity.json`
as the corrected `.444` KV260 hardware-continuity artifact. The first
precondition MUST be board SSH reachability via
`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; host SD-card device
nodes are permanently retired for KV260 and MUST NOT be used as a precondition.
If that SSH command returns non-zero, the experiment MUST still write the
artifact with `honest_verdict=blocked_kv260_ssh_unreachable`,
`kv260_ssh_reachable=false`, `inference_substrate=hardware_smoke`, and
`preconditions_checked` recording the failed SSH check. A blocked artifact is
the expected correct output while the board is offline; the experiment MUST NOT
exit with no file changes.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and preserve that direct transcript to report the loaded overlay. If non-sudo
`xmutil` reports that root privileges are required, the experiment MAY retry
with `ssh kria 'sudo xmutil listapps'` as a read-only board-state fallback and
MUST preserve both transcripts. The experiment MUST also record lightweight
board state over SSH, including hostname, kernel/OS line, uptime, and UIO device
count. The reachable-board `next_forward_step` MUST record the next concrete
KV260 track step as:
`GRADUATED: KV260 terminal criteria met; continue per-milestone SSH-only continuity checks. Next concrete step: keep the board in the continuity rotation and queue an overlay-state audit only if a future SSH or xmutil probe changes.`
If SSH is unreachable, the blocked `next_forward_step` MUST queue the operator
and audit action as:
`OPERATOR ACTION: restore KV260 SSH reachability, then rerun the SSH-only continuity check; operator/audit queue: document the SSH outage and keep host SD-card preconditions retired.`

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; reachable is success_/complete_; unreachable is blocked_kv260_ssh_unreachable -- and the artifact MUST be written either way (no 'No file changes' 3-fail-skip).`
- `inference_substrate`: `hardware_smoke (SSH-attached board test).`
- `kv260_ssh_reachable`: `the SSH reachability check -- false when the board is offline; the artifact is still written.`
- `preconditions_checked`: `records the SSH check; a blocked artifact is the correct output when the board is offline, NOT a no-file-changes failure.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4827_kv260_continuity.py` writes
  `results/experiment_4827_kv260_continuity.json` whether SSH is reachable or
  unreachable.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command is run,
  and `next_forward_step` queues an operator/audit step to restore SSH and
  rerun the SSH-only continuity check.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success: kv260_continuity_recorded`, the direct `ssh kria 'xmutil listapps'`
  transcript is preserved, a sudo read-only fallback transcript is preserved
  when root is required, the loaded overlay is reported when present, lightweight
  board state is recorded, and `next_forward_step` records the next concrete
  KV260 continuity action.
- `verifier_is_oracle=false`, `random_seed=4827`, `field_principles`,
  `duration_s`, `command_probes`, and a stable `reproducibility_checksum` are
  present.

**Implementation status:** Implemented (Exp 4827)

---

### REQ-HW-4837

**Title:** KV260 continuity MUST always write the Exp 4837 deliverable after the SSH precondition

**Description:**
Experiment 4837 MUST produce `results/experiment_4837_kv260_continuity.json`
for the KV260 hardware-continuity slot. The only valid precondition is board
SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card device nodes are permanently retired for KV260 and MUST NOT be
used. If the SSH command returns non-zero, the experiment MUST still write the
artifact with `honest_verdict=blocked_kv260_ssh_unreachable`,
`kv260_ssh_reachable=false`, `inference_substrate=hardware_smoke`, and
`preconditions_checked` recording the failed SSH check. A blocked artifact is
the expected correct outcome while the board is offline; the experiment MUST
NOT exit with no file changes.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and preserve that transcript to report the loaded overlay. If non-sudo
`xmutil` reports that root privileges are required, the experiment MAY retry
with `ssh kria 'sudo xmutil listapps'` as a read-only board-state fallback and
MUST preserve both transcripts. The experiment MUST also record lightweight
board state over SSH and a concrete next-forward step.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; reachable is success_/complete_; unreachable is blocked_kv260_ssh_unreachable -- artifact MUST be written either way (no 'No file changes' 3-fail-skip).`
- `inference_substrate`: `hardware_smoke (SSH-attached board test).`
- `kv260_ssh_reachable`: `the SSH check -- false when offline; the artifact is still written.`
- `preconditions_checked`: `records the SSH check; a blocked artifact is correct when the board is offline.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4837_kv260_continuity.py` writes
  `results/experiment_4837_kv260_continuity.json` whether SSH is reachable or
  unreachable.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command is run,
  and the blocked artifact is the terminal output rather than a no-file-changes
  skip.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success: kv260_continuity_recorded`, the loaded overlay and lightweight
  board state are recorded, and `next_forward_step` records the next concrete
  KV260 continuity action.
- `verifier_is_oracle=false`, `random_seed=4837`, `field_principles`,
  `duration_s`, `command_probes`, and a stable `reproducibility_checksum` are
  present.

**Implementation status:** Implemented (Exp 4837)

---

### SCENARIO-HW-4837

**Scenario:** Exp 4837 writes SSH-attached KV260 continuity or an honest SSH-unreachable block.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4837 checks SSH and, only when reachable, records the
direct `xmutil listapps` transcript plus lightweight board state over SSH.
**Then:** It always writes `results/experiment_4837_kv260_continuity.json` with
`honest_verdict=blocked_kv260_ssh_unreachable` when SSH is unreachable, or
`success: kv260_continuity_recorded` plus overlay and board state when SSH is
reachable, `inference_substrate=hardware_smoke`, principle-annotated required
fields, `kv260_ssh_reachable`, deterministic `random_seed=4837`, a concrete
`next_forward_step`, and no host SD-card precondition.

**Implementation status:** Implemented (Exp 4837)

---

### SCENARIO-HW-4827

**Scenario:** Exp 4827 writes SSH-attached KV260 continuity or an honest SSH-unreachable block.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4827 checks SSH and, only when reachable, records the
direct `xmutil listapps` transcript plus lightweight board state over SSH.
**Then:** It always writes `results/experiment_4827_kv260_continuity.json` with
a terminal-prefixed reachable `honest_verdict` or
`blocked_kv260_ssh_unreachable`, `inference_substrate=hardware_smoke`,
principle-annotated required fields, `kv260_ssh_reachable`, deterministic
`random_seed=4827`, a concrete `next_forward_step`, and no host SD-card
precondition.

**Implementation status:** Implemented (Exp 4827)

---

### REQ-HW-4847

**Title:** KV260 continuity MUST always write the Exp 4847 deliverable after the SSH precondition

**Description:**
Experiment 4847 MUST produce `results/experiment_4847_kv260_continuity.json`
for the KV260 hardware-continuity slot. The only valid precondition is board
SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card device nodes are permanently retired for KV260 and MUST NOT be
used. If the SSH command returns non-zero, the experiment MUST still write the
artifact with `honest_verdict=blocked_kv260_ssh_unreachable`,
`kv260_ssh_reachable=false`, `inference_substrate=hardware_smoke`, and
`preconditions_checked` recording the failed SSH check. A blocked artifact is
the expected correct outcome while the board is offline; the experiment MUST
NOT exit with no file changes, including the prior "No file changes"
3-fail-skip failure mode.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and preserve that transcript to report the loaded overlay. If non-sudo
`xmutil` reports that root privileges are required, the experiment MAY retry
with `ssh kria 'sudo xmutil listapps'` as a read-only board-state fallback and
MUST preserve both transcripts. The experiment MUST also record lightweight
board state over SSH and a concrete next-forward step.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; reachable is success_/complete_; unreachable is blocked_kv260_ssh_unreachable -- artifact MUST be written either way (no 'No file changes' 3-fail-skip).`
- `inference_substrate`: `hardware_smoke (SSH-attached board test).`
- `kv260_ssh_reachable`: `the SSH check -- false when offline; the artifact is still written.`
- `preconditions_checked`: `records the SSH check; a blocked artifact is correct when the board is offline.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4847_kv260_continuity.py` writes
  `results/experiment_4847_kv260_continuity.json` whether SSH is reachable or
  unreachable.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command is run,
  and the blocked artifact is the terminal output rather than a no-file-changes
  skip.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success: kv260_continuity_recorded`, the loaded overlay and lightweight
  board state are recorded, and `next_forward_step` records the next concrete
  KV260 continuity action.
- `verifier_is_oracle=false`, `random_seed=4847`, `field_principles`,
  `duration_s`, `command_probes`, and a stable `reproducibility_checksum` are
  present.

**Implementation status:** Implemented (Exp 4847)

---

### SCENARIO-HW-4847

**Scenario:** Exp 4847 writes SSH-attached KV260 continuity or an honest SSH-unreachable block.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4847 checks SSH and, only when reachable, records the
direct `xmutil listapps` transcript plus lightweight board state over SSH.
**Then:** It always writes `results/experiment_4847_kv260_continuity.json` with
`honest_verdict=blocked_kv260_ssh_unreachable` when SSH is unreachable, or
`success: kv260_continuity_recorded` plus overlay and board state when SSH is
reachable, `inference_substrate=hardware_smoke`, principle-annotated required
fields, `kv260_ssh_reachable`, deterministic `random_seed=4847`, a concrete
`next_forward_step`, and no host SD-card precondition.

**Implementation status:** Implemented (Exp 4847)

---

### REQ-HW-4857

**Title:** KV260 continuity MUST always write the Exp 4857 deliverable after the SSH precondition

**Description:**
Experiment 4857 MUST produce `results/experiment_4857_kv260_continuity.json`
for the KV260 hardware-continuity slot. The only valid precondition is board
SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card device nodes are permanently retired for KV260 and MUST NOT be
used. If the SSH command returns non-zero, the experiment MUST still write the
artifact with `honest_verdict=blocked_kv260_ssh_unreachable`,
`kv260_ssh_reachable=false`, `inference_substrate=hardware_smoke`, and
`preconditions_checked` recording the failed SSH check. A blocked artifact is
the expected correct outcome while the board is offline; the experiment MUST
NOT exit with no file changes, including the prior "No file changes"
3-fail-skip failure mode.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and preserve that transcript to report the loaded overlay. If non-sudo
`xmutil` reports that root privileges are required, the experiment MAY retry
with `ssh kria 'sudo xmutil listapps'` as a read-only board-state fallback and
MUST preserve both transcripts. The experiment MUST also record lightweight
board state over SSH and a concrete next-forward step.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; reachable is success_kv260_continuity_ok; unreachable is blocked_kv260_ssh_unreachable -- artifact MUST be written either way (no 'No file changes' 3-fail-skip).`
- `inference_substrate`: `hardware_smoke (SSH-attached board test).`
- `kv260_ssh_reachable`: `the SSH check -- false when offline; the artifact is still written.`
- `preconditions_checked`: `records the SSH check; a blocked artifact is correct when the board is offline.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4857_kv260_continuity.py` writes
  `results/experiment_4857_kv260_continuity.json` whether SSH is reachable or
  unreachable.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command is run,
  and the blocked artifact is the terminal output rather than a no-file-changes
  skip.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success_kv260_continuity_ok`, the loaded overlay and lightweight board state
  are recorded, and `next_forward_step` records the next concrete KV260
  continuity action.
- `verifier_is_oracle=false`, `random_seed=4857`, `field_principles`,
  `duration_s`, `command_probes`, and a stable `reproducibility_checksum` are
  present.

**Implementation status:** Implemented (Exp 4857)

---

### SCENARIO-HW-4857

**Scenario:** Exp 4857 writes SSH-attached KV260 continuity or an honest SSH-unreachable block.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4857 checks SSH and, only when reachable, records the
direct `xmutil listapps` transcript plus lightweight board state over SSH.
**Then:** It always writes `results/experiment_4857_kv260_continuity.json` with
`honest_verdict=blocked_kv260_ssh_unreachable` when SSH is unreachable, or
`success_kv260_continuity_ok` plus overlay and board state when SSH is
reachable, `inference_substrate=hardware_smoke`, principle-annotated required
fields, `kv260_ssh_reachable`, deterministic `random_seed=4857`, a concrete
`next_forward_step`, and no host SD-card precondition.

**Implementation status:** Implemented (Exp 4857)

---

### REQ-HW-4867

**Title:** KV260 continuity MUST always write the Exp 4867 deliverable after the SSH precondition

**Description:**
Experiment 4867 MUST produce `results/experiment_4867_kv260_continuity.json`
for the KV260 hardware-continuity slot. The only valid precondition is board
SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card device nodes are permanently retired for KV260 and MUST NOT be
used. If the SSH command returns non-zero, the experiment MUST still write the
artifact with `honest_verdict=blocked_kv260_ssh_unreachable`,
`kv260_ssh_reachable=false`, `inference_substrate=hardware_smoke`, and
`preconditions_checked` recording the failed SSH check. A blocked artifact is
the expected correct outcome while the board is offline; the experiment MUST
NOT exit with no file changes, including the prior "No file changes"
3-fail-skip failure mode.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and preserve that transcript to report the loaded overlay. If non-sudo
`xmutil` reports that root privileges are required, the experiment MAY retry
with `ssh kria 'sudo xmutil listapps'` as a read-only board-state fallback and
MUST preserve both transcripts. The experiment MUST also record lightweight
board state over SSH and a concrete next-forward step.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; reachable is success_kv260_continuity_ok; unreachable is blocked_kv260_ssh_unreachable -- artifact MUST be written either way (no 'No file changes' 3-fail-skip).`
- `inference_substrate`: `hardware_smoke (SSH-attached board test).`
- `kv260_ssh_reachable`: `the SSH check -- false when offline; the artifact is still written.`
- `preconditions_checked`: `records the SSH check; a blocked artifact is correct when the board is offline.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4867_kv260_continuity.py` writes
  `results/experiment_4867_kv260_continuity.json` whether SSH is reachable or
  unreachable.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command is run,
  and the blocked artifact is the terminal output rather than a no-file-changes
  skip.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success_kv260_continuity_ok`, the loaded overlay and lightweight board state
  are recorded, and `next_forward_step` records the next concrete KV260
  continuity action.
- `verifier_is_oracle=false`, `random_seed=4867`, `field_principles`,
  `duration_s`, `command_probes`, and a stable `reproducibility_checksum` are
  present.

**Implementation status:** Implemented (Exp 4867)

---

### SCENARIO-HW-4867

**Scenario:** Exp 4867 writes SSH-attached KV260 continuity or an honest SSH-unreachable block.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4867 checks SSH and, only when reachable, records the
direct `xmutil listapps` transcript plus lightweight board state over SSH.
**Then:** It always writes `results/experiment_4867_kv260_continuity.json` with
`honest_verdict=blocked_kv260_ssh_unreachable` when SSH is unreachable, or
`success_kv260_continuity_ok` plus overlay and board state when SSH is
reachable, `inference_substrate=hardware_smoke`, principle-annotated required
fields, `kv260_ssh_reachable`, deterministic `random_seed=4867`, a concrete
`next_forward_step`, and no host SD-card precondition.

**Implementation status:** Implemented (Exp 4867)

---

### REQ-HW-4878

**Title:** KV260 continuity MUST always write the Exp 4878 deliverable after the SSH precondition

**Description:**
Experiment 4878 MUST produce `results/experiment_4878_kv260_continuity.json`
for the KV260 hardware-continuity slot. The only valid precondition is board
SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card device nodes are permanently retired for KV260 and MUST NOT be
used. If the SSH command returns non-zero, the experiment MUST still write the
artifact with `honest_verdict=blocked_kv260_ssh_unreachable`,
`kv260_ssh_reachable=false`, `inference_substrate=hardware_smoke`, and
`preconditions_checked` recording the failed SSH check. A blocked artifact is
the expected correct outcome while the board is offline; the experiment MUST
NOT exit with no file changes, including the prior "No file changes"
3-fail-skip failure mode.

When SSH is reachable, the experiment MUST run `ssh kria 'xmutil listapps'`
and preserve that transcript to report the loaded overlay. If non-sudo
`xmutil` reports that root privileges are required, the experiment MAY retry
with `ssh kria 'sudo xmutil listapps'` as a read-only board-state fallback and
MUST preserve both transcripts. The experiment MUST also record lightweight
board state over SSH and a concrete next-forward step.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; reachable is success_kv260_continuity_ok; unreachable is blocked_kv260_ssh_unreachable -- artifact MUST be written either way (no 'No file changes' 3-fail-skip).`
- `inference_substrate`: `hardware_smoke (SSH-attached board test).`
- `kv260_ssh_reachable`: `the SSH check -- false when offline; the artifact is still written.`
- `preconditions_checked`: `records the SSH check; a blocked artifact is correct when the board is offline.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4878_kv260_continuity.py` writes
  `results/experiment_4878_kv260_continuity.json` whether SSH is reachable or
  unreachable.
- `inference_substrate` is exactly `hardware_smoke`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command is run,
  and the blocked artifact is the terminal output rather than a no-file-changes
  skip.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success_kv260_continuity_ok`, the loaded overlay and lightweight board state
  are recorded, and `next_forward_step` records the next concrete KV260
  continuity action.
- `verifier_is_oracle=false`, `random_seed=4878`, `field_principles`,
  `duration_s`, `command_probes`, and a stable `reproducibility_checksum` are
  present.

**Implementation status:** Implemented (Exp 4878)

---

### SCENARIO-HW-4878

**Scenario:** Exp 4878 writes SSH-attached KV260 continuity or an honest SSH-unreachable block.

**Given:** The KV260 continuity task requires SSH reachability of the board and
must not use a host SD-card device-node precondition.
**When:** Experiment 4878 checks SSH and, only when reachable, records the
direct `xmutil listapps` transcript plus lightweight board state over SSH.
**Then:** It always writes `results/experiment_4878_kv260_continuity.json` with
`honest_verdict=blocked_kv260_ssh_unreachable` when SSH is unreachable, or
`success_kv260_continuity_ok` plus overlay and board state when SSH is
reachable, `inference_substrate=hardware_smoke`, principle-annotated required
fields, `kv260_ssh_reachable`, deterministic `random_seed=4878`, a concrete
`next_forward_step`, and no host SD-card precondition.

**Implementation status:** Implemented (Exp 4878)

---

### REQ-HW-4889

**Title:** KV260 continuity MUST always write the Exp 4889 SSH-only deliverable

**Description:**
Experiment 4889 MUST produce `results/experiment_4889_kv260_continuity.json`
for the KV260 hardware-continuity slot. The only valid precondition is board
SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

Host SD-card device nodes are permanently retired for KV260 and MUST NOT be
used. If the SSH command returns non-zero, the experiment MUST still write the
artifact with `honest_verdict=blocked_kv260_ssh_unreachable`,
`kv260_ssh_reachable=false`, `inference_substrate=aggregation_from_upstream_artifacts`,
and `preconditions_checked` recording the failed SSH check. A blocked artifact
is the expected correct outcome while the board is offline; the experiment MUST
NOT exit with no file changes, including the prior "No file changes"
3-fail-skip failure mode.

When SSH is reachable, the experiment MUST capture board state over SSH:
hostname, kernel, uptime, `/dev/uio*` count, and `xmutil listapps` if
available. If non-sudo `xmutil` reports that root privileges are required, the
experiment MAY retry with `ssh kria 'sudo xmutil listapps'` as a read-only
board-state fallback and MUST preserve both transcripts. The next forward step
MUST be continuity-only because the KV260 has already graduated to terminal.

The artifact MUST include bare values plus `field_principles` entries for these
required fields:

- `honest_verdict`: `terminal prefix; reachable is success_kv260_continuity_ok; unreachable is blocked_kv260_ssh_unreachable (still WRITES).`
- `kv260_ssh_reachable`: `true with board state, or false -- the SSH-only continuity signal (NEVER host SD-card presence).`
- `board_state`: `captured hostname/kernel/uptime/uio-count over SSH -- the terminal-state continuity record.`
- `next_forward_step`: `continuity-only (graduated terminal); records the next concrete step only if a probe changes.`
- `inference_substrate`: `aggregation_from_upstream_artifacts (an SSH state read; 0.0001s floor).`
- `preconditions_checked`: `records the SSH reachability check; an unreachable board emits blocked_kv260_ssh_unreachable, never a fabricated state.`

**Acceptance criteria:**
- `.venv/bin/python python/carnot/experiment_4889_kv260_continuity.py` writes
  `results/experiment_4889_kv260_continuity.json` whether SSH is reachable or
  unreachable.
- `inference_substrate` is exactly `aggregation_from_upstream_artifacts`.
- `preconditions_checked` records the SSH BatchMode command listed above and
  contains no host SD-card device-node precondition.
- If `kv260_ssh_reachable=false`, `honest_verdict` is exactly
  `blocked_kv260_ssh_unreachable`, no `xmutil` or board-state command is run,
  and the blocked artifact is the terminal output rather than a no-file-changes
  skip.
- If `kv260_ssh_reachable=true`, `honest_verdict` is exactly
  `success_kv260_continuity_ok`, board state is captured over SSH, `xmutil
  listapps` is preserved when available, and `next_forward_step` remains
  continuity-only for the terminal KV260.
- `verifier_is_oracle=false`, `random_seed=4889`, `field_principles`,
  `duration_s >= 0.0001`, `command_probes`, and a stable
  `reproducibility_checksum` are present.

**Implementation status:** Implemented (Exp 4889)

---

### SCENARIO-HW-4889

**Scenario:** Exp 4889 writes SSH-attached KV260 terminal continuity or an honest SSH-unreachable block.

**Given:** The KV260 terminal board is in the per-milestone SSH-only continuity
rotation and host SD-card device-node checks are retired.
**When:** Experiment 4889 checks SSH and, only when reachable, records `xmutil
listapps` plus hostname, kernel, uptime, and UIO count over SSH.
**Then:** It always writes `results/experiment_4889_kv260_continuity.json` with
`honest_verdict=blocked_kv260_ssh_unreachable` when SSH is unreachable, or
`success_kv260_continuity_ok` plus board state when SSH is reachable,
`inference_substrate=aggregation_from_upstream_artifacts`,
principle-annotated required fields, `kv260_ssh_reachable`, deterministic
`random_seed=4889`, a terminal continuity-only `next_forward_step`, and no host
SD-card precondition.

**Implementation status:** Implemented (Exp 4889)
