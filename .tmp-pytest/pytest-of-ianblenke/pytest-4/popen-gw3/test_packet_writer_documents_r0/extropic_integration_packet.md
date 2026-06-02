# Extropic Z1/XTR-0 Integration Packet

Spec: REQ-SAMPLE-040, SCENARIO-SAMPLE-066, SCENARIO-SAMPLE-067, SCENARIO-SAMPLE-068.

## Hardware Context

Carnot treats KV260 as a proof-of-concept tier. The planned production hardware path
for thermodynamic sampling is Extropic Z1 when access opens, with XTR-0 as the
development bridge between a traditional processor and Extropic probabilistic
circuits. THRML is the public CPU/GPU JAX simulation stack for TSU-style PGMs and
EBMs.

Local THRML status for this run:
- availability: not available
- version: not importable
- install command: `pip install thrml`
- local pip probe: not available on PATH
- 100-spin THRML latency: not measured because THRML import did not succeed

## Minimal Carnot Workload Spec

Z1 must support these EBM operations for Carnot's first integration:

1. Ising sampling
   - Input: dense or sparse pairwise Ising couplings `J`, bias vector `h`, inverse
     temperature `beta`, sample count, and sweep or schedule controls.
   - Output: boolean or {-1,+1} spin states with shape `(n_samples, n_spins)`.
   - Required first target: 128 spins, sparse or dense pairwise couplings.

2. spin state read/write
   - Host can upload an initial spin state or request hardware-native randomized
     initialization.
   - Host can read sampled spin states without lossy packing ambiguity.
   - The convention must be explicit: Carnot accepts {0,1} booleans at the
     SamplerBackend boundary and can convert to {-1,+1} for TSU APIs.

3. energy evaluation
   - Hardware or SDK must expose the Ising Hamiltonian energy for returned states:
     `E(s) = -sum_i h_i s_i - sum_{i<j} J_ij s_i s_j`.
   - Energy values must include enough precision for KL and acceptance-test
     diagnostics against CPU Gibbs.

## ThermoSamplerBackend Interface Requirements

`ThermoSamplerBackend` should preserve the current Carnot `SamplerBackend` protocol:

- `backend_name -> str`
- `minimize_energy(biases, couplings, n_samples, n_steps, beta) -> np.ndarray`
- `sample(biases, couplings, n_samples, config) -> np.ndarray`

The production class should report whether it is using Z1/XTR-0 hardware, THRML CPU
simulation, or Carnot CPU Gibbs fallback. It must not label THRML or CPU fallback as
live Extropic hardware.

Backend-specific `config` keys expected by Carnot:

- `beta`: fixed inverse temperature for `sample()`.
- `steps_per_sample`: simulation or device update stride.
- `n_warmup`: host-side THRML/CPU warmup where applicable.
- `initial_state`: optional host-provided spin state.
- `read_energy`: optional request to return or log sampled-state energies.

## Acceptance Tests

The first hardware acceptance gate is intentionally small and distributional:

- KL(Z1 || CPU_Gibbs) < 0.05 on small exactly enumerable Ising systems.
- latency < 1ms for 128 spins, including host-device round trip for one sample batch.
- Returned state shape and dtype match Carnot `SamplerBackend` expectations.
- State convention is round-trip checked between {0,1} and {-1,+1}.
- Energy sign is validated on a ferromagnetic ground-state smoke test.

## THRML Parity Benchmark Plan

When THRML is installed, Carnot runs a 100-spin ring Ising workload through the public
THRML `IsingEBM` plus block Gibbs `sample_states` API and records `thrml_latency_us`.
That CPU/JAX latency is the parity baseline, not a hardware claim.

Expected speedup profile when Z1 replaces CPU simulation:

- CPU Gibbs / THRML simulation: useful for API parity and correctness, bounded by host
  JAX execution and PRNG overhead.
- XTR-0 development bridge: expected to reduce sampling latency once host-device
  transfer and SDK overhead are controlled.
- Z1 production card: expected to move the bottleneck from stochastic sampling to
  workload packing, readback, and verifier orchestration. Carnot should report both
  raw sampler latency and end-to-end SamplerBackend latency.

## Fallback Plan

If Z1/XTR-0 hardware is unavailable, Carnot uses THRML CPU simulation for parity work.
If THRML is not installed, Carnot uses the existing CPU Gibbs `CpuBackend` and records
the missing THRML dependency honestly in the artifact.

## Early-Access Checklist For Extropic

- SDK install steps and Python package name/version.
- Device discovery mechanism equivalent to `CARNOT_TSU_DEVICE`.
- Maximum spins per program and maximum non-zero pairwise couplings.
- Bias/coupling numeric ranges and quantization rules.
- Required spin convention and packed readback format.
- Warmup/schedule controls exposed through the SDK.
- Energy readback support and precision.
- Batch size limits, host-device transfer model, and async execution semantics.
- Recommended benchmark harness for KL and latency validation.

## References

- `python/carnot/samplers/backend.py` for the current `SamplerBackend` protocol.
- `python/carnot/samplers/thrml_backend.py` for the Exp 1150 backend stub.
- `research-references.md` Extropic XTR-0 / Z1 Hardware Status section.
- `_bmad/architecture.md` hardware path and portability sections.
- THRML docs: https://docs.thrml.ai/en/latest/
