#!/usr/bin/env python3
"""Experiment 1150: Extropic Z1/XTR-0 integration packet.

This experiment writes the formal Carnot hardware integration packet for
Extropic-class thermodynamic sampling, checks whether THRML is importable, and
emits the roadmap-required artifact. If THRML is installed, it runs a small
100-spin Ising sampling benchmark using the public THRML API.

Spec: REQ-SAMPLE-040, SCENARIO-SAMPLE-068.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKET_PATH = REPO_ROOT / "docs" / "hardware" / "extropic_integration_packet.md"
BACKEND_PATH = REPO_ROOT / "python" / "carnot" / "samplers" / "thrml_backend.py"
DELIVERABLE = REPO_ROOT / "results" / "experiment_1150_extropic_integration_packet.json"

PACKET_PATH_STR = "docs/hardware/extropic_integration_packet.md"
BACKEND_PATH_STR = "python/carnot/samplers/thrml_backend.py"
THRML_INSTALL_COMMAND = "pip install thrml"

HONEST_VERDICTS = {
    "thrml_available_benchmark_run",
    "thrml_not_available_packet_written",
    "partial_packet_written",
}

REQUIRED_ARTIFACT_FIELDS = {
    "thrml_available",
    "thrml_version",
    "integration_packet_written",
    "packet_path",
    "thrml_backend_stub_written",
    "thrml_backend_path",
    "thrml_latency_us",
    "sampler_backend_interface_documented",
    "extropic_integration_packet_written",
    "honest_verdict",
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def probe_thrml() -> tuple[bool, str | None, ModuleType | None]:
    """Return THRML import availability, version string, and imported module."""
    try:
        module = importlib.import_module("thrml")
    except ModuleNotFoundError:
        return False, None, None

    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version("thrml")
        except Exception:
            version = "unknown"
    return True, str(version), module


def _block_until_ready(value: Any) -> None:
    """Block on a JAX PyTree if leaves expose block_until_ready()."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _block_until_ready(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready(item)


def run_thrml_ising_benchmark(
    thrml_module: ModuleType | None = None,
    *,
    n_spins: int = 100,
    n_samples: int = 32,
    n_warmup: int = 50,
    steps_per_sample: int = 2,
) -> float:
    """Run a 100-spin ring Ising benchmark through THRML CPU/JAX simulation.

    The implementation follows the public THRML quick-example API: build
    `SpinNode` objects, split them into two free blocks, wrap an `IsingEBM` in
    an `IsingSamplingProgram`, initialize with `hinton_init`, and call
    `sample_states`.

    Spec: REQ-SAMPLE-040
    """
    if thrml_module is None:
        thrml_module = importlib.import_module("thrml")

    import jax
    import jax.numpy as jnp
    from thrml import Block, SamplingSchedule, SpinNode, sample_states
    from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

    nodes = [SpinNode() for _ in range(n_spins)]
    edges = [(nodes[i], nodes[(i + 1) % n_spins]) for i in range(n_spins)]
    biases = jnp.zeros((n_spins,), dtype=jnp.float32)
    weights = jnp.ones((len(edges),), dtype=jnp.float32) * 0.25
    beta = jnp.array(1.0, dtype=jnp.float32)
    model = IsingEBM(nodes, edges, biases, weights, beta)

    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    readout_blocks = [Block(nodes)]
    schedule = SamplingSchedule(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample,
    )

    key = jax.random.PRNGKey(1150)
    k_init, k_warm, k_init_2, k_sample = jax.random.split(key, 4)

    init_state = hinton_init(k_init, model, free_blocks, ())
    warmup_samples = sample_states(k_warm, program, schedule, init_state, [], readout_blocks)
    _block_until_ready(warmup_samples)

    init_state_2 = hinton_init(k_init_2, model, free_blocks, ())
    start = time.perf_counter()
    samples = sample_states(k_sample, program, schedule, init_state_2, [], readout_blocks)
    _block_until_ready(samples)
    return float((time.perf_counter() - start) * 1_000_000.0)


def write_integration_packet(
    *,
    packet_path: Path = PACKET_PATH,
    thrml_available: bool,
    thrml_version: str | None,
    thrml_latency_us: float | None,
) -> None:
    """Write the Extropic early-access packet required by Exp 1150."""
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    pip_status = "available on PATH" if shutil.which("pip") else "not available on PATH"
    latency_text = (
        f"{thrml_latency_us:.3f} us for the measured 100-spin THRML CPU/JAX call"
        if thrml_latency_us is not None
        else "not measured because THRML import did not succeed"
    )
    version_text = thrml_version if thrml_version is not None else "not importable"
    availability_text = "available" if thrml_available else "not available"

    packet_path.write_text(
        f"""# Extropic Z1/XTR-0 Integration Packet

Spec: REQ-SAMPLE-040, SCENARIO-SAMPLE-066, SCENARIO-SAMPLE-067, SCENARIO-SAMPLE-068.

## Hardware Context

Carnot treats KV260 as a proof-of-concept tier. The planned production hardware path
for thermodynamic sampling is Extropic Z1 when access opens, with XTR-0 as the
development bridge between a traditional processor and Extropic probabilistic
circuits. THRML is the public CPU/GPU JAX simulation stack for TSU-style PGMs and
EBMs.

Local THRML status for this run:
- availability: {availability_text}
- version: {version_text}
- install command: `{THRML_INSTALL_COMMAND}`
- local pip probe: {pip_status}
- 100-spin THRML latency: {latency_text}

## Minimal Carnot Workload Spec

Z1 must support these EBM operations for Carnot's first integration:

1. Ising sampling
   - Input: dense or sparse pairwise Ising couplings `J`, bias vector `h`, inverse
     temperature `beta`, sample count, and sweep or schedule controls.
   - Output: boolean or {{-1,+1}} spin states with shape `(n_samples, n_spins)`.
   - Required first target: 128 spins, sparse or dense pairwise couplings.

2. spin state read/write
   - Host can upload an initial spin state or request hardware-native randomized
     initialization.
   - Host can read sampled spin states without lossy packing ambiguity.
   - The convention must be explicit: Carnot accepts {{0,1}} booleans at the
     SamplerBackend boundary and can convert to {{-1,+1}} for TSU APIs.

3. energy evaluation
   - Hardware or SDK must expose the Ising Hamiltonian energy for returned states:
     `E(s) = -sum_i h_i s_i - sum_{{i<j}} J_ij s_i s_j`.
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
- State convention is round-trip checked between {{0,1}} and {{-1,+1}}.
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
""",
    )


def _classify_verdict(
    *,
    thrml_available: bool,
    thrml_latency_us: float | None,
    packet_written: bool,
    backend_stub_written: bool,
) -> str:
    if not packet_written or not backend_stub_written:
        return "partial_packet_written"
    if thrml_available and thrml_latency_us is not None:
        return "thrml_available_benchmark_run"
    if not thrml_available:
        return "thrml_not_available_packet_written"
    return "partial_packet_written"


def build_artifact(
    *,
    thrml_available: bool,
    thrml_version: str | None,
    thrml_latency_us: float | None,
    packet_written: bool,
    backend_stub_written: bool,
    run_date: str | None = None,
) -> dict[str, Any]:
    """Build the Exp 1150 result artifact."""
    honest_verdict = _classify_verdict(
        thrml_available=thrml_available,
        thrml_latency_us=thrml_latency_us,
        packet_written=packet_written,
        backend_stub_written=backend_stub_written,
    )
    interface_documented = bool(packet_written)
    return {
        "experiment_id": 1150,
        "schema": "extropic_integration_packet_v1",
        "run_date": run_date or _utc_now_iso(),
        "thrml_available": bool(thrml_available),
        "thrml_version": thrml_version,
        "thrml_install_command": THRML_INSTALL_COMMAND,
        "thrml_install_alternatives": ["uv pip install thrml"],
        "thrml_latency_us": thrml_latency_us,
        "integration_packet_written": bool(packet_written),
        "packet_path": PACKET_PATH_STR,
        "thrml_backend_stub_written": bool(backend_stub_written),
        "thrml_backend_path": BACKEND_PATH_STR,
        "sampler_backend_interface_documented": interface_documented,
        "extropic_integration_packet_written": bool(packet_written),
        "acceptance_kl_threshold": 0.05,
        "acceptance_latency_ms_128_spins": 1.0,
        "fallback_plan": "Use THRML CPU simulation when Z1 is unavailable; use CpuBackend if THRML is not installed.",
        "honest_verdict": honest_verdict,
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate the roadmap-required artifact fields before writing."""
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["honest_verdict"] not in HONEST_VERDICTS:
        raise ValueError(f"invalid honest_verdict: {artifact['honest_verdict']!r}")
    if artifact["packet_path"] != PACKET_PATH_STR:
        raise ValueError("packet_path must point to the Extropic integration packet")
    if artifact["thrml_backend_path"] != BACKEND_PATH_STR:
        raise ValueError("thrml_backend_path must point to the THRML backend stub")
    if not artifact["integration_packet_written"]:
        raise ValueError("integration packet was not written")
    if not artifact["thrml_backend_stub_written"]:
        raise ValueError("THRML backend stub was not written")


def write_artifact(artifact: dict[str, Any], output_path: Path = DELIVERABLE) -> None:
    """Write a stable JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def main() -> None:
    start = time.perf_counter()
    thrml_available, thrml_version, thrml_module = probe_thrml()
    thrml_latency_us: float | None = None
    if thrml_available:
        try:
            thrml_latency_us = run_thrml_ising_benchmark(thrml_module)
        except Exception:
            thrml_latency_us = None

    write_integration_packet(
        packet_path=PACKET_PATH,
        thrml_available=thrml_available,
        thrml_version=thrml_version,
        thrml_latency_us=thrml_latency_us,
    )
    packet_written = PACKET_PATH.exists()
    backend_stub_written = BACKEND_PATH.exists()

    artifact = build_artifact(
        thrml_available=thrml_available,
        thrml_version=thrml_version,
        thrml_latency_us=thrml_latency_us,
        packet_written=packet_written,
        backend_stub_written=backend_stub_written,
        run_date=_utc_now_iso(),
    )
    artifact["duration_s"] = float(time.perf_counter() - start)
    validate_artifact(artifact)
    write_artifact(artifact, DELIVERABLE)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
