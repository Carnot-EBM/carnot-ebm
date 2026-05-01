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
