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
