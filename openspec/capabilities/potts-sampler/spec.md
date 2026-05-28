# Potts Sampler Capability Spec

## REQ-POTTS-001: q=3 Synchronous Checkerboard Sampler

Carnot MUST provide a NumPy `PottsSampler` for q-state Potts machines with
default `q=3`, default `beta=2.0`, and spin states in `{0, 1, 2}` for
correct/partial/violated constraint scoring.

Sub-requirements:
- REQ-POTTS-001-1: `PottsSampler.sample(J, n_steps)` SHALL return a
  one-dimensional NumPy array of length `n_spins`.
- REQ-POTTS-001-2: Returned spins SHALL be integers in `[0, q-1]`.
- REQ-POTTS-001-3: The sampler SHALL update even-indexed spins and odd-indexed
  spins as two synchronous checkerboard phases per sweep.

## REQ-POTTS-002: Potts Energy and Convergence Validation

The sampler energy MUST implement `E = -sum_ij J_ij * delta(s_i, s_j)`.
On a positive ferromagnetic coupling graph, repeated q=3 sampling MUST reduce
mean final energy relative to random initialization and reach a low-energy
same-state configuration.

Sub-requirements:
- REQ-POTTS-002-1: `PottsSampler.energy(J, spins)` SHALL return a Python float.
- REQ-POTTS-002-2: On a 16-spin q=3 validation graph, expected final energy
  SHALL be no greater than expected random-initialization energy.
- REQ-POTTS-002-3: On an unconstrained q=3 validation graph, final samples
  SHALL exercise all three states across repeated runs.

## REQ-POTTS-003: q=2 Boundary Compatibility

When constructed with `q=2`, `PottsSampler` MUST remain on the two-state
boundary and preserve the same low-energy preference as an Ising alignment
objective under positive couplings.

## REQ-POTTS-004: KV260 Potts RTL

Carnot MUST provide `hardware/kv260/potts_sampler_v1.v` as a q=3 Potts
extension of the synchronous KV260 Ising sampler.

Sub-requirements:
- REQ-POTTS-004-1: The top-level RTL SHALL expose `N_SPINS=64`,
  `Q_STATES=3`, and `BETA_FIXED=8'h40` parameters by default.
- REQ-POTTS-004-2: The RTL SHALL store each spin in 2 bits.
- REQ-POTTS-004-3: The RTL SHALL compute a 3-entry fixed-point softmax for
  `exp(-beta * E_i(a))` over states `{0, 1, 2}`.
- REQ-POTTS-004-4: The RTL SHALL sample the categorical update using per-spin
  2-bit LFSR state.
- REQ-POTTS-004-5: The RTL SHALL keep the AXI-Lite control/status/register-map
  shape compatible with `ising_sampler_v2.v`.

## REQ-POTTS-005: Experiment 1098 Artifact

Experiment 1098 MUST write `results/experiment_1098_potts_machine_q3_verilog.json`
with the requested Potts simulation and RTL completion fields.

Sub-requirements:
- REQ-POTTS-005-1: The artifact SHALL include `python_sim_written`,
  `python_sim_validated`, `verilog_file_written`,
  `verilog_synthesis_area_estimate_lut`, `verilog_fits_kv260_budget`,
  `tests_passing`, and `honest_verdict`.
- REQ-POTTS-005-2: `honest_verdict` SHALL be one of
  `potts_sim_and_rtl_complete`, `potts_sim_only_rtl_stub`, or `failed`.

## REQ-POTTS-006: Experiment 1649 Vivado Synthesis

Experiment 1649 MUST write `results/experiment_1649_vivado_synthesis.json`
with the required synthesis status fields.

Sub-requirements:
- REQ-POTTS-006-1: The script SHALL attempt Vivado synthesis for the q=3 Potts machine (`hardware/kv260/potts_sampler_v1.v`).
- REQ-POTTS-006-2: The artifact SHALL include `synthesis_success`, `vivado_available`, and `honest_verdict`.
- REQ-POTTS-006-3: If Vivado is not available, `honest_verdict` SHALL report "vivado_not_installed" and `synthesis_success` SHALL be false.

## REQ-POTTS-007: Experiment 1692 Potts v2 RTL

Experiment 1692 MUST provide `rtl/potts_machine_v2.v` as a synthesizable q=3 Potts block suitable for Vivado.

Sub-requirements:
- REQ-POTTS-007-1: The top-level RTL SHALL be located at `rtl/potts_machine_v2.v`.
- REQ-POTTS-007-2: The RTL SHALL use standard synchronous design constraints.
- REQ-POTTS-007-3: The task SHALL write `results/experiment_1692_potts_export.json` with the experiment artifact.

## REQ-POTTS-008: Experiment 1704 KV260

Experiment 1704 MUST write `results/experiment_1704_kv260.json` with the required artifact fields.

Sub-requirements:
- REQ-POTTS-008-1: The artifact SHALL include `vivado_available`, `synthesis_success`, `performance`, `resource_utilization`, and `honest_verdict`.
- REQ-POTTS-008-2: If Vivado is not available, `honest_verdict` SHALL report "vivado_not_installed", `synthesis_success` SHALL be false, and `vivado_available` SHALL be false.
- REQ-POTTS-008-3: `crates/carnot-webgpu-gateway/src/kv260_bindings.rs` SHALL expose a Rust KV260 q=3 Potts sampler binding that writes the `potts_sampler_v1.v` AXI-Lite register map, polls `STATUS.DONE`, and unpacks 2-bit Potts states.
- REQ-POTTS-008-4: `carnot-python` SHALL expose the KV260 Potts sampler through PyO3 without requiring KV260 hardware at import time.
- REQ-POTTS-008-5: The Exp 1704 artifact SHALL record the Rust binding path, Python binding name, driver interface, register map, spec traces, tests run, and whether the binding is ready.

## REQ-POTTS-009: Experiment 3256 P-Dit/Potts Partial-Credit Diagnostic

Experiment 3256 MUST write
`results/experiment_3256_pdit_potts_multistate_sampler_diagnostic_v1.json` as
a CPU/simulation-only diagnostic manifest mapping p-dit/Potts multi-state
variables to Carnot partial-credit verifier rows while preserving exact
fallback authority.

Sub-requirements:
- REQ-POTTS-009-1: The artifact SHALL include `experiment_id`, `task_id`,
  `milestone`, `inference_substrate`, `principle_annotations`,
  `pdit_potts_mapping_ready`, `candidate_verifier_row_types`,
  `q_state_energy_mapping`, `exact_fallback_preserved`,
  `hardware_speedup_claim_allowed`, `retired_pimi_scope_reopened`,
  `thrml_scaling_sweep_reopened`, `future_gated_experiment_contract`,
  `random_seed`, `reproducibility_checksum`, and `honest_verdict`.
- REQ-POTTS-009-2: The selected verifier row types SHALL include q-state
  partial-credit labels that are more natural than binary Ising spins and SHALL
  define Potts/p-dit variables, state labels, and deterministic energy tables.
- REQ-POTTS-009-3: The diagnostic SHALL set
  `hardware_speedup_claim_allowed=false`, `retired_pimi_scope_reopened=false`,
  and `thrml_scaling_sweep_reopened=false`.
- REQ-POTTS-009-4: Exact fallback SHALL be preserved for every candidate row
  type through explicit verifier checks, and any future experiment contract
  SHALL be gated on exact fallback plus no retired-scope reopening.

### SCENARIO-POTTS-009

**Given** the p-dit and Potts references plus prior p-dit accounting artifacts
are available in the repository
**When** the Exp 3256 diagnostic manifest builder runs
**Then** it writes the required JSON artifact with q-state partial-credit row
mappings, exact fallback gates, denied hardware/speedup/retired-scope claims,
a stable reproducibility checksum, and an `honest_verdict` beginning with
`complete:` that does not claim live hardware, THRML, Kona, or speedup evidence.

