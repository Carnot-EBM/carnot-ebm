"""Tests for EqM GPU backend and Exp 1740.

Spec traces: REQ-SAMPLE-1740, SCENARIO-SAMPLE-1740.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from carnot.core.energy import AutoGradMixin
from carnot.samplers.equilibrium_matching import EquilibriumMatchingSampler
import scripts.experiment_1740_eqm_gpu as exp


class GpuBackendBowl(AutoGradMixin):
    """Convex energy used to test REQ-SAMPLE-1740 backend parity."""

    def __init__(self, target: jax.Array) -> None:
        self.target = target

    @property
    def input_dim(self) -> int:
        return int(self.target.shape[0])

    def energy(self, x: jax.Array) -> jax.Array:
        residual = x - self.target
        return 0.5 * jnp.sum(residual**2)


def _platform(value: jax.Array) -> str:
    device = getattr(value, "device", None)
    return str(getattr(device, "platform", "unknown"))


def test_req_sample_1740_spec_entry_exists() -> None:
    spec = Path("openspec/capabilities/samplers/spec.md").read_text()
    assert "REQ-SAMPLE-1740" in spec
    assert "SCENARIO-SAMPLE-1740" in spec


def test_req_sample_1740_cpu_jit_matches_unjitted_reference() -> None:
    """REQ-SAMPLE-1740-3: JIT backend preserves CPU EqM update behavior."""
    model = GpuBackendBowl(target=jnp.linspace(-0.5, 0.5, 8))
    init = jnp.linspace(3.0, -3.0, 8)

    reference = EquilibriumMatchingSampler(
        step_size=0.12,
        learning_rate=0.4,
        matching_strength=0.7,
        clip_norm=8.0,
        backend="cpu",
        jit=False,
    )
    accelerated = EquilibriumMatchingSampler(
        step_size=0.12,
        learning_rate=0.4,
        matching_strength=0.7,
        clip_norm=8.0,
        backend="cpu",
        jit=True,
    )

    ref_chain = reference.sample_chain(model, init, n_steps=24)
    accelerated_chain = accelerated.sample_chain(model, init, n_steps=24)

    assert _platform(accelerated_chain) == "cpu"
    assert jnp.allclose(accelerated_chain, ref_chain, atol=1e-6)
    assert jnp.allclose(accelerated.sample(model, init, n_steps=24), accelerated_chain[-1])


def test_req_sample_1740_auto_backend_uses_cpu_when_pytest_disables_gpu() -> None:
    """REQ-SAMPLE-1740-1: auto remains runnable without a visible GPU."""
    sampler = EquilibriumMatchingSampler(backend="auto")
    summary = sampler.backend_summary()

    assert summary["requested_backend"] == "auto"
    assert summary["selected_platform"] == "cpu"
    assert summary["accelerated"] is False


def test_req_sample_1740_gpu_backend_requires_visible_gpu() -> None:
    """REQ-SAMPLE-1740-2: explicit gpu backend fails clearly when unavailable."""
    model = GpuBackendBowl(target=jnp.zeros(2))
    sampler = EquilibriumMatchingSampler(backend="gpu")

    with pytest.raises(RuntimeError, match="JAX GPU backend is not available"):
        sampler.sample(model, jnp.ones(2), n_steps=1)


def test_req_sample_1740_backend_errors_are_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SAMPLE-1740-1: invalid or unavailable device choices fail explicitly."""
    with pytest.raises(ValueError, match="Unsupported EqM backend"):
        EquilibriumMatchingSampler(backend="cuda").sample_chain(  # type: ignore[arg-type]
            GpuBackendBowl(target=jnp.zeros(1)),
            jnp.zeros(1),
            n_steps=1,
        )

    with pytest.raises(RuntimeError, match="device_index=999"):
        EquilibriumMatchingSampler(backend="cpu", device_index=999)._select_device()

    sampler = EquilibriumMatchingSampler(backend="cpu")
    monkeypatch.setattr(sampler, "_devices_for_platform", lambda _platform: [])
    with pytest.raises(RuntimeError, match="JAX CPU backend is not available"):
        sampler._select_device()


def test_req_sample_1740_negative_steps_rejected() -> None:
    """REQ-SAMPLE-1740-3: accelerated runner rejects nonsensical step counts."""
    sampler = EquilibriumMatchingSampler(backend="cpu")
    with pytest.raises(ValueError, match="n_steps must be non-negative"):
        sampler.sample(GpuBackendBowl(target=jnp.zeros(1)), jnp.zeros(1), n_steps=-1)


def test_scenario_sample_1740_helper_edges() -> None:
    """SCENARIO-SAMPLE-1740: artifact helpers cover no-data and verdict edges."""
    device = jax.devices("cpu")[0]
    energy_fn, _init = exp._make_problem(dimension=4, device=device)

    assert energy_fn.input_dim == 4
    exp._block_until_ready((jnp.array([1.0]), [jnp.array([2.0])]))
    assert exp._latency_summary([]) == {"mean": 0.0, "min": 0.0, "max": 0.0}
    assert exp._derive_verdict(True, 1.0, 0.5, None) == "gpu_available_parity_failed"
    assert exp._derive_verdict(True, 1.0, 2.0, 0.0) == "gpu_available_but_not_faster"


def test_scenario_sample_1740_experiment_writes_backend_aware_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-1740: benchmark writes honest GPU availability fields."""
    output_path = tmp_path / "experiment_1740_eqm_gpu.json"
    artifact = exp.run_experiment(output_path=output_path, dimension=32, n_steps=6, repeats=1)

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written == artifact

    assert artifact["experiment_id"] == "1740"
    assert artifact["spec_refs"] == ["REQ-SAMPLE-1740", "SCENARIO-SAMPLE-1740"]
    assert artifact["backend"]["cpu"]["available"] is True
    assert artifact["latency_ms"]["cpu"]["mean"] >= 0.0
    assert artifact["problem"]["dimension"] == 32
    assert artifact["problem"]["n_steps"] == 6

    if artifact["backend"]["gpu"]["available"]:
        assert artifact["latency_ms"]["gpu"]["mean"] >= 0.0
        assert artifact["parity"]["max_abs_delta"] <= exp.PARITY_TOLERANCE
        assert artifact["honest_verdict"] in {
            "gpu_faster",
            "gpu_available_but_not_faster",
        }
    else:
        assert artifact["latency_ms"]["gpu"] is None
        assert artifact["parity"]["max_abs_delta"] is None
        assert artifact["honest_verdict"] == "gpu_unavailable"
