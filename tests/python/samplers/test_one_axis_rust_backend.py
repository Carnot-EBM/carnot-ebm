"""Tests for the one-axis Rust SamplerBackend adapter.

Spec coverage: REQ-SAMPLE-5723, SCENARIO-SAMPLE-5723
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_5714_one_axis_tempering_rust_parity as exp5714
from carnot.samplers.backend import CpuBackend, SamplerBackend, get_backend
from carnot.samplers.one_axis_rust_backend import (
    ONE_AXIS_ALGORITHM,
    ONE_AXIS_TOPOLOGY,
    CHECKPOINT_SCHEMA_VERSION,
    OneAxisRustBackend,
    canonical_json,
    checkpoint_checksum,
    descriptor_for_run,
)


REPO = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"


def _fixture_inputs(dtype: np.dtype = np.float64) -> tuple[np.ndarray, np.ndarray]:
    config = exp5714.default_config()
    return (
        np.asarray(config.fields, dtype=dtype),
        np.asarray(config.couplings, dtype=dtype),
    )


def _descriptor(seed: int = 5723) -> dict[str, object]:
    state = exp5714.default_state(seed=seed)
    return descriptor_for_run(
        seed=seed,
        initial_states=state.states.astype(int).tolist(),
        initial_labels=list(state.labels),
        burn_in_sweeps=1,
    )


def test_req_sample_5723_spec_declares_production_backend_contract() -> None:
    """REQ-SAMPLE-5723: OpenSpec anchors backend exposure, fallback, and gates."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5723") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        "OneAxisRustBackend",
        "SamplerBackend",
        "one_axis_rust",
        "default factory selection as `cpu`",
        ONE_AXIS_ALGORITHM,
        ONE_AXIS_TOPOLOGY,
        'active_backend="python_exact_fallback"',
        "malformed descriptors",
        "unsupported topology",
        "seed mismatches",
        str(Path("results/experiment_5723_one_axis_rust_samplerbackend_integration.json")),
        "one_axis_samplerbackend_ready_score",
        "two_axis_code_added=false",
        "timing_claimed=false",
        "hardware_speedup_claimed=false",
        "production_python_samplerbackend_plus_rust_pyo3_one_axis",
    ):
        assert marker in section or marker in normalized


def test_req_sample_5723_protocol_factory_and_default_preservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLE-5723: one_axis_rust is explicit and the default remains cpu."""
    monkeypatch.delenv("CARNOT_BACKEND", raising=False)
    backend = get_backend("one_axis_rust")
    default_backend = get_backend()

    assert isinstance(backend, OneAxisRustBackend)
    assert isinstance(backend, SamplerBackend)
    assert backend.backend_name == "one_axis_rust"
    assert isinstance(default_backend, CpuBackend)
    assert default_backend.backend_name == "cpu"
    assert backend.set_constraints(lambda state: state) is None
    assert backend.dual_update_step(0.1) is None


def test_req_sample_5723_protocol_sample_and_minimize_methods() -> None:
    """REQ-SAMPLE-5723: protocol sample/minimize paths use the same descriptor runner."""
    biases, couplings = _fixture_inputs()
    descriptor = descriptor_for_run(seed=5723, burn_in_sweeps=0, force_python_fallback=True)
    backend = OneAxisRustBackend(seed=5723)

    samples = backend.sample(biases, couplings, n_samples=2, config=descriptor)
    minimized = backend.minimize_energy(biases, couplings, n_samples=2, n_steps=1, beta=1.25)

    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert descriptor["force_python_fallback"] is True
    assert samples.shape == (2, biases.size)
    assert samples.dtype == np.bool_
    assert minimized.shape == (2, biases.size)
    assert minimized.dtype == np.bool_


def test_req_sample_5723_rust_path_matches_exact_python_fallback() -> None:
    """REQ-SAMPLE-5723: Rust adapter samples and decision logs match fallback."""
    biases, couplings = _fixture_inputs()
    descriptor = _descriptor(seed=5723)

    rust_result = OneAxisRustBackend(seed=5723).run_descriptor(
        biases,
        couplings,
        n_samples=3,
        config=descriptor,
    )
    python_result = OneAxisRustBackend(seed=5723, prefer_rust=False).run_descriptor(
        biases,
        couplings,
        n_samples=3,
        config=descriptor,
    )

    assert rust_result["receipt"]["active_backend"] == "rust_pyo3"
    assert python_result["receipt"]["active_backend"] == "python_exact_fallback"
    np.testing.assert_array_equal(rust_result["samples"], python_result["samples"])
    assert rust_result["samples"].dtype == np.bool_
    assert rust_result["samples_spin"] == python_result["samples_spin"]
    assert rust_result["checkpoint"]["state"] == python_result["checkpoint"]["state"]
    assert (
        rust_result["receipt"]["transition_budget"] == python_result["receipt"]["transition_budget"]
    )
    assert rust_result["decision_log"] == python_result["decision_log"]
    assert rust_result["receipt"]["energy_convention"] == "-0.5*x^T*J*x - h^T*x"


def test_req_sample_5723_checkpoint_roundtrip_and_restart_cross_language() -> None:
    """REQ-SAMPLE-5723: adapter checkpoints duplicate and restart across backends."""
    biases, couplings = _fixture_inputs()
    descriptor = _descriptor(seed=5724)
    rust_backend = OneAxisRustBackend(seed=5724)
    prefix = rust_backend.run_descriptor(biases, couplings, n_samples=2, config=descriptor)
    checkpoint_a = rust_backend.save_checkpoint()
    checkpoint_b = rust_backend.save_checkpoint()

    assert checkpoint_a == checkpoint_b
    assert checkpoint_a["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint_a["payload_checksum"] == checkpoint_checksum(checkpoint_a)
    assert (
        rust_backend.load_checkpoint(checkpoint_a, biases, couplings, config=descriptor)
        == (checkpoint_a["state"])
    )

    suffix_descriptor = {**descriptor, "checkpoint": checkpoint_a}
    rust_suffix = OneAxisRustBackend(seed=5724).run_descriptor(
        biases,
        couplings,
        n_samples=2,
        config=suffix_descriptor,
    )
    python_suffix = OneAxisRustBackend(seed=5724, prefer_rust=False).run_descriptor(
        biases,
        couplings,
        n_samples=2,
        config=suffix_descriptor,
    )

    assert prefix["checkpoint"]["payload_checksum"] == checkpoint_a["payload_checksum"]
    np.testing.assert_array_equal(rust_suffix["samples"], python_suffix["samples"])
    assert rust_suffix["checkpoint"]["state"] == python_suffix["checkpoint"]["state"]

    corrupt = deepcopy(checkpoint_a)
    corrupt["state"]["sweep"] = int(corrupt["state"]["sweep"]) + 1
    with pytest.raises(ValueError, match="checksum"):
        rust_backend.load_checkpoint(corrupt, biases, couplings, config=descriptor)

    mismatched_seed = {**descriptor, "seed": int(descriptor["seed"]) + 1}
    with pytest.raises(ValueError, match="seed mismatch"):
        rust_backend.load_checkpoint(checkpoint_a, biases, couplings, config=mismatched_seed)


def test_req_sample_5723_checkpoint_rejection_edges_fail_closed() -> None:
    """REQ-SAMPLE-5723: checkpoint schema, identity, input, and state edges reject."""
    biases, couplings = _fixture_inputs()
    descriptor = _descriptor(seed=5730)
    backend = OneAxisRustBackend(seed=5730)

    backend.run_descriptor(biases, couplings, n_samples=1, config=descriptor)
    checkpoint = backend.save_checkpoint()
    with pytest.raises(ValueError, match="no one-axis checkpoint"):
        OneAxisRustBackend().save_checkpoint()
    with pytest.raises(ValueError, match="object"):
        backend.load_checkpoint([], biases, couplings, config=descriptor)

    mutations = [
        ("schema_version", lambda data: data.__setitem__("schema_version", "bad")),
        ("algorithm", lambda data: data.__setitem__("algorithm", "bad")),
        (
            "source_algorithm_hash",
            lambda data: data.__setitem__("source_algorithm_hash", "bad"),
        ),
        ("beta_ladder_hash", lambda data: data.__setitem__("beta_ladder_hash", "bad")),
        ("input_hash", lambda data: data.__setitem__("input_hash", "bad")),
        ("state", lambda data: data.pop("state")),
        ("checkpoint state invalid", lambda data: data.__setitem__("state", {"states": [[1]]})),
    ]

    for match, mutate in mutations:
        bad = deepcopy(checkpoint)
        mutate(bad)
        if match in {"state", "checkpoint state invalid"}:
            bad["payload_checksum"] = checkpoint_checksum(bad)
        with pytest.raises(ValueError, match=match):
            backend.load_checkpoint(bad, biases, couplings, config=descriptor)


def test_req_sample_5723_explicit_fallback_cases_are_exact() -> None:
    """REQ-SAMPLE-5723: missing, broken, wrong-symbol, dtype, layout fallback exactly."""
    biases, couplings = _fixture_inputs()
    descriptor = _descriptor(seed=5725)

    fallback_backends = [
        OneAxisRustBackend(
            seed=5725,
            rust_module_loader=lambda: (_ for _ in ()).throw(ImportError("missing")),
        ),
        OneAxisRustBackend(
            seed=5725,
            rust_module_loader=lambda: (_ for _ in ()).throw(RuntimeError("broken")),
        ),
        OneAxisRustBackend(seed=5725, rust_module_loader=lambda: SimpleNamespace()),
        OneAxisRustBackend(seed=5725),
        OneAxisRustBackend(seed=5725),
        OneAxisRustBackend(seed=5725),
    ]
    fallback_inputs = [
        (biases, couplings, descriptor, "rust_extension_missing"),
        (biases, couplings, descriptor, "rust_extension_error"),
        (biases, couplings, descriptor, "rust_symbol_missing"),
        (
            _fixture_inputs(np.float32)[0],
            _fixture_inputs(np.float32)[1],
            descriptor,
            "unsupported_dtype_or_layout",
        ),
        (
            biases.copy()[::-1][::-1],
            np.asfortranarray(couplings),
            descriptor,
            "unsupported_dtype_or_layout",
        ),
        (
            biases,
            couplings,
            {**descriptor, "force_python_fallback": True},
            "declared_python_compatibility",
        ),
    ]

    for backend, (case_biases, case_couplings, case_descriptor, reason) in zip(
        fallback_backends,
        fallback_inputs,
        strict=True,
    ):
        baseline = OneAxisRustBackend(seed=5725, prefer_rust=False).run_descriptor(
            case_biases,
            case_couplings,
            n_samples=2,
            config=case_descriptor,
        )
        result = backend.run_descriptor(case_biases, case_couplings, 2, case_descriptor)
        assert result["receipt"]["active_backend"] == "python_exact_fallback"
        assert reason in result["receipt"]["fallback_reason"]
        np.testing.assert_array_equal(result["samples"], baseline["samples"])
        assert result["decision_log"] == baseline["decision_log"]

    class BadConfigModule:
        RustOneAxisTemperingCore = object
        RustOneAxisTemperingState = object

        class RustOneAxisTemperingConfig:
            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise ValueError("bad rust config")

    result = OneAxisRustBackend(
        seed=5725,
        rust_module_loader=lambda: BadConfigModule,
    ).run_descriptor(biases, couplings, 2, descriptor)
    assert result["receipt"]["active_backend"] == "python_exact_fallback"
    assert "rust_extension_error" in result["receipt"]["fallback_reason"]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda data: data.pop("algorithm"), "descriptor algorithm"),
        (lambda data: data.__setitem__("algorithm", "wrong"), "descriptor algorithm"),
        (
            lambda data: data.__setitem__("topology", "two_axis_temperature_penalty_exchange"),
            "unsupported topology",
        ),
        (lambda data: data.__setitem__("beta_ladder", [0.8, 0.8, 1.25]), "beta_ladder"),
        (lambda data: data.__setitem__("initial_labels", [0, 0, 2]), "labels"),
    ],
)
def test_req_sample_5723_malformed_descriptors_fail_closed(mutate, match: str) -> None:
    """REQ-SAMPLE-5723: malformed descriptors and two-axis requests fail closed."""
    biases, couplings = _fixture_inputs()
    descriptor = _descriptor(seed=5726)
    mutate(descriptor)

    with pytest.raises(ValueError, match=match):
        OneAxisRustBackend(seed=5726).run_descriptor(biases, couplings, 2, descriptor)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda data: data.__setitem__("source_algorithm_hash", "bad"),
            "source_algorithm_hash",
        ),
        (lambda data: data.__setitem__("proposal_std", 0.5), "proposal_std"),
        (lambda data: data.__setitem__("drift_scale", 0.5), "drift_scale"),
        (lambda data: data.__setitem__("seed", -1), "seed"),
        (lambda data: data.__setitem__("burn_in_sweeps", -1), "burn_in_sweeps"),
    ],
)
def test_req_sample_5723_descriptor_identity_edges_fail_closed(mutate, match: str) -> None:
    """REQ-SAMPLE-5723: descriptor identity, seed, and budget edges reject."""
    biases, couplings = _fixture_inputs()
    descriptor = _descriptor(seed=5728)
    mutate(descriptor)

    with pytest.raises(ValueError, match=match):
        OneAxisRustBackend(seed=5728).run_descriptor(biases, couplings, 2, descriptor)


def test_req_sample_5723_invalid_inputs_and_energy_sign_control_fail_closed() -> None:
    """REQ-SAMPLE-5723: invalid shapes and wrong energy sign controls reject."""
    biases, couplings = _fixture_inputs()
    descriptor = _descriptor(seed=5727)
    backend = OneAxisRustBackend(seed=5727)

    with pytest.raises(ValueError, match="couplings"):
        backend.run_descriptor(biases, np.zeros((2, 2), dtype=np.float64), 2, descriptor)
    with pytest.raises(ValueError, match="biases"):
        backend.run_descriptor(biases.reshape(1, -1), couplings, 2, descriptor)
    with pytest.raises(ValueError, match="finite"):
        backend.run_descriptor(np.array([0.0, np.nan]), np.zeros((2, 2)), 2, descriptor)
    with pytest.raises(ValueError, match="finite"):
        backend.run_descriptor(np.zeros(2), np.array([[0.0, np.inf], [0.0, 0.0]]), 2, descriptor)
    with pytest.raises(ValueError, match="descriptor config"):
        backend.run_descriptor(biases, couplings, 2, [])
    with pytest.raises(ValueError, match="n_samples"):
        backend.run_descriptor(biases, couplings, 0, descriptor)

    assert backend.energy_sign_control(biases, couplings, descriptor)["rejected"] is True
