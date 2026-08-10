"""Tests for the mode-jump runtime SamplerBackend adapter.

Spec coverage: REQ-SAMPLE-6208, SCENARIO-SAMPLE-6208-DEFAULT-OFF-FALLBACK,
SCENARIO-SAMPLE-6208-RUNTIME-PARITY, SCENARIO-SAMPLE-6208-BOUNDARY-ERRORS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot.samplers.backend import CpuBackend, SamplerBackend, get_backend, get_sampler_backend
from carnot.samplers.mode_jump_rust_backend import (
    ACTIVE_PYTHON_FALLBACK,
    ACTIVE_RUST_BACKEND,
    CHECKPOINT_SCHEMA_VERSION,
    MODE_JUMP_ALGORITHM,
    MODE_JUMP_TOPOLOGY,
    TYPED_STATE_METADATA_SCHEMA_VERSION,
    VARIABLE_CARDINALITY_TOPOLOGY,
    ModeJumpRustBackend,
    checkpoint_checksum,
    descriptor_for_run,
    frozen_mode_jump_inputs,
    mode_jump_inputs_from_fixture_receipt,
    normalize_typed_state_metadata,
    typed_state_metadata_from_fixture_receipt,
)


REPO = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
EXP6268_PATH = REPO / "results/experiment_6268_multimodal_sampler_fixture_suite.json"


def _fixture_inputs(dtype: np.dtype = np.float64) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels, target, proposal = frozen_mode_jump_inputs()
    return labels, np.asarray(target, dtype=dtype), np.asarray(proposal, dtype=dtype)


def _exp6268_receipts() -> list[dict[str, object]]:
    artifact = json.loads(EXP6268_PATH.read_text(encoding="utf-8"))
    return list(artifact["exact_enumeration_receipts"])


def test_req_sample_6208_spec_declares_runtime_adapter_contract() -> None:
    """REQ-SAMPLE-6208: OpenSpec anchors default-off runtime integration."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-6208") :]
    normalized = " ".join(section.split())

    for marker in (
        "mode_jump_rust",
        "enable_mode_jump_runtime=true",
        "CARNOT_ENABLE_MODE_JUMP_RUNTIME=1",
        "REQ-SAMPLE-6208-DEFAULT-OFF",
        "REQ-SAMPLE-6208-FIXED-KERNEL",
        "REQ-SAMPLE-6208-SHAPE-CONTRACT",
        "REQ-SAMPLE-6208-EXACT-FALLBACK",
        "REQ-SAMPLE-6208-RUNTIME-ACCOUNTING",
        "REQ-SAMPLE-6208-ARTIFACT",
        "SCENARIO-SAMPLE-6208-DEFAULT-OFF-FALLBACK",
        "SCENARIO-SAMPLE-6208-RUNTIME-PARITY",
        "SCENARIO-SAMPLE-6208-BOUNDARY-ERRORS",
        "production_python_samplerbackend_plus_rust_pyo3_mode_jump_cpu",
        "hardware_or_speed_power_energy_claimed",
    ):
        assert marker in section or marker in normalized


def test_req_sampler_6280_spec_declares_variable_cardinality_contract() -> None:
    """REQ-SAMPLER-6280: OpenSpec anchors typed metadata and controls."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLER-6280") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAMPLER-6280-SPEC-FIRST",
        "REQ-SAMPLER-6280-METADATA",
        "REQ-SAMPLER-6280-COMPATIBILITY",
        "REQ-SAMPLER-6280-PARITY",
        "REQ-SAMPLER-6280-ACTIVATION",
        "REQ-SAMPLER-6280-CONTROLS",
        "SCENARIO-SAMPLER-6280-METADATA-ROUNDTRIP",
        "SCENARIO-SAMPLER-6280-PROPOSAL-PARITY",
        "SCENARIO-SAMPLER-6280-NO-AB-VALUE-CLAIM",
        "typed_state_metadata_schema",
        "variable_cardinality_backend_ready_score",
        "local_cpu_rust_python_variable_cardinality_sampler_abi",
    ):
        assert marker in section or marker in normalized


def test_req_sample_6208_factory_preserves_default_cpu_and_requires_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLE-6208-DEFAULT-OFF: default factories remain cpu."""

    monkeypatch.delenv("CARNOT_BACKEND", raising=False)
    monkeypatch.delenv("CARNOT_SAMPLER", raising=False)
    monkeypatch.delenv("CARNOT_ENABLE_MODE_JUMP_RUNTIME", raising=False)
    labels, target, proposal = _fixture_inputs()

    default_backend = get_backend()
    default_sampler = get_sampler_backend()
    mode_jump = get_backend("mode_jump_rust")

    assert isinstance(default_backend, CpuBackend)
    assert default_backend.backend_name == "cpu"
    assert isinstance(default_sampler, CpuBackend)
    assert isinstance(mode_jump, ModeJumpRustBackend)
    assert isinstance(mode_jump, SamplerBackend)
    assert mode_jump.backend_name == "mode_jump_rust"

    result = mode_jump.run_descriptor(
        target,
        proposal,
        n_samples=4,
        config=descriptor_for_run(labels=labels, seed=6208),
    )

    assert result["receipt"]["active_backend"] == ACTIVE_PYTHON_FALLBACK
    assert result["receipt"]["fallback_reason"] == "feature_flag_disabled"
    assert result["samples"].shape == (4, 6)
    assert result["samples"].dtype == np.bool_


@pytest.mark.parametrize("receipt", _exp6268_receipts(), ids=lambda row: row["fixture_name"])
def test_req_sampler_6280_typed_metadata_roundtrips_exp6268_fixtures(
    receipt: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-6280-METADATA-ROUNDTRIP: each fixture has explicit metadata."""

    labels, target, proposal, metadata = mode_jump_inputs_from_fixture_receipt(receipt)
    normalized = normalize_typed_state_metadata(metadata, label_count=len(labels))
    direct_metadata = typed_state_metadata_from_fixture_receipt(receipt)

    assert direct_metadata == metadata
    assert normalized["schema"] == TYPED_STATE_METADATA_SCHEMA_VERSION
    assert normalized["shape"] == [len(normalized["cardinalities"])]
    assert normalized["support_count"] == len(labels) == target.shape[0] == proposal.shape[0]
    assert normalized["state_space_size"] >= normalized["support_count"]
    assert normalized["proposal_domain"] in {
        "explicit_support_complete_no_self",
        "explicit_support_table",
    }
    assert normalized["state_labels"] == labels
    assert target.shape == (len(labels),)
    assert proposal.shape == (len(labels), len(labels))
    assert np.allclose(proposal.sum(axis=1), 1.0)

    from carnot._rust import RustModeJumpStateMetadata

    rust_metadata = RustModeJumpStateMetadata(
        normalized["schema"],
        normalized["shape"],
        normalized["cardinalities"],
        normalized["encoding"],
        normalized["state_labels"],
        normalized["state_values"],
        normalized["proposal_domain"],
        normalized["state_space_size"],
    )
    for index, label in enumerate(labels):
        assert normalized["label_to_index"][label] == index
        assert rust_metadata.encode_label(label) == index
        assert rust_metadata.decode_index(index) == label
        assert rust_metadata.state_value(label) == normalized["state_values"][index]


@pytest.mark.parametrize("receipt", _exp6268_receipts(), ids=lambda row: row["fixture_name"])
def test_scenario_sampler_6280_rust_python_proposal_parity_and_seed_replay(
    receipt: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-6280-PROPOSAL-PARITY: Rust and Python traces match."""

    labels, target, proposal, metadata = mode_jump_inputs_from_fixture_receipt(receipt)
    descriptor = {
        "algorithm": MODE_JUMP_ALGORITHM,
        "topology": VARIABLE_CARDINALITY_TOPOLOGY,
        "labels": labels,
        "typed_state_metadata": metadata,
        "seed": 6280,
        "initial_label": labels[0],
        "burn_in": 3,
        "enable_mode_jump_runtime": True,
        "return_trace": True,
    }

    rust_a = ModeJumpRustBackend(seed=6280).run_descriptor(target, proposal, 24, descriptor)
    rust_b = ModeJumpRustBackend(seed=6280).run_descriptor(target, proposal, 24, descriptor)
    fallback = ModeJumpRustBackend(seed=6280, prefer_rust=False).run_descriptor(
        target,
        proposal,
        24,
        descriptor,
    )

    assert rust_a["receipt"]["active_backend"] == ACTIVE_RUST_BACKEND
    assert fallback["receipt"]["active_backend"] == ACTIVE_PYTHON_FALLBACK
    assert rust_a["receipt"]["topology"] == VARIABLE_CARDINALITY_TOPOLOGY
    assert rust_a["receipt"]["typed_state_metadata_hash"] == fallback["receipt"][
        "typed_state_metadata_hash"
    ]
    np.testing.assert_array_equal(rust_a["samples"], fallback["samples"])
    assert rust_a["sample_labels"] == fallback["sample_labels"]
    assert rust_a["decision_log"] == fallback["decision_log"]
    assert rust_a["checkpoint"]["state"] == fallback["checkpoint"]["state"]
    assert rust_a["checkpoint"]["state"] == rust_b["checkpoint"]["state"]
    assert rust_a["sample_labels"] == rust_b["sample_labels"]
    assert any(event["accepted"] for event in rust_a["decision_log"])


def test_req_sampler_6280_original_six_state_compatibility_path_is_unchanged() -> None:
    """REQ-SAMPLER-6280-COMPATIBILITY: old six-state descriptor still replays."""

    labels, target, proposal = _fixture_inputs()
    descriptor = descriptor_for_run(
        labels=labels,
        seed=6280,
        burn_in=3,
        enable_mode_jump_runtime=True,
    )
    legacy = ModeJumpRustBackend(seed=6280).run_descriptor(target, proposal, 24, descriptor)
    receipt = next(
        row
        for row in _exp6268_receipts()
        if row["fixture_name"] == "exp6237_original_six_state"
    )
    labels2, target2, proposal2, metadata = mode_jump_inputs_from_fixture_receipt(receipt)
    typed = ModeJumpRustBackend(seed=6280).run_descriptor(
        target2,
        proposal2,
        24,
        {
            **descriptor,
            "topology": VARIABLE_CARDINALITY_TOPOLOGY,
            "labels": labels2,
            "typed_state_metadata": metadata,
        },
    )

    np.testing.assert_array_equal(legacy["samples"], typed["samples"])
    assert legacy["sample_labels"] == typed["sample_labels"]
    assert legacy["decision_log"] == typed["decision_log"]
    assert legacy["checkpoint"]["state"] == typed["checkpoint"]["state"]


def test_req_sampler_6280_malformed_metadata_and_domains_fail_closed() -> None:
    """REQ-SAMPLER-6280-CONTROLS: malformed metadata and proposals fail closed."""

    receipt = next(row for row in _exp6268_receipts() if row["fixture_name"] == "potts_chain3_q3")
    labels, target, proposal, metadata = mode_jump_inputs_from_fixture_receipt(receipt)
    descriptor = {
        "algorithm": MODE_JUMP_ALGORITHM,
        "topology": VARIABLE_CARDINALITY_TOPOLOGY,
        "labels": labels,
        "typed_state_metadata": metadata,
        "seed": 6280,
        "initial_label": labels[0],
        "enable_mode_jump_runtime": True,
    }

    bad_cardinality = deepcopy(metadata)
    bad_cardinality["cardinalities"][0] = 2
    with pytest.raises(ValueError, match="cardinality"):
        ModeJumpRustBackend().run_descriptor(
            target,
            proposal,
            2,
            {**descriptor, "typed_state_metadata": bad_cardinality},
        )

    bad_shape = deepcopy(metadata)
    bad_shape["shape"] = [1, 3]
    with pytest.raises(ValueError, match="rank-1"):
        ModeJumpRustBackend().run_descriptor(
            target,
            proposal,
            2,
            {**descriptor, "typed_state_metadata": bad_shape},
        )

    bad_domain = proposal.copy()
    bad_domain[0, :] = 0.0
    bad_domain[0, 0] = 1.0
    with pytest.raises(ValueError, match="proposal support"):
        ModeJumpRustBackend().run_descriptor(target, bad_domain, 2, descriptor)

    permuted = deepcopy(metadata)
    permuted["state_labels"] = list(reversed(permuted["state_labels"]))
    with pytest.raises(ValueError, match="labels"):
        ModeJumpRustBackend().run_descriptor(
            target,
            proposal,
            2,
            {**descriptor, "typed_state_metadata": permuted},
        )

    with pytest.raises(ValueError, match="target probabilities"):
        ModeJumpRustBackend().run_descriptor(target.reshape(3, 9), proposal, 2, descriptor)


def test_scenario_sample_6208_enabled_rust_path_matches_exact_python_fallback() -> None:
    """SCENARIO-SAMPLE-6208-RUNTIME-PARITY: enabled PyO3 path matches fallback."""

    labels, target, proposal = _fixture_inputs()
    descriptor = descriptor_for_run(
        labels=labels,
        seed=6208,
        burn_in=3,
        enable_mode_jump_runtime=True,
    )

    rust_result = ModeJumpRustBackend(seed=6208).run_descriptor(
        target,
        proposal,
        n_samples=32,
        config=descriptor,
    )
    fallback_result = ModeJumpRustBackend(seed=6208, prefer_rust=False).run_descriptor(
        target,
        proposal,
        n_samples=32,
        config=descriptor,
    )

    assert rust_result["receipt"]["active_backend"] == ACTIVE_RUST_BACKEND
    assert fallback_result["receipt"]["active_backend"] == ACTIVE_PYTHON_FALLBACK
    np.testing.assert_array_equal(rust_result["samples"], fallback_result["samples"])
    assert rust_result["sample_labels"] == fallback_result["sample_labels"]
    assert rust_result["decision_log"] == fallback_result["decision_log"]
    assert rust_result["checkpoint"]["state"] == fallback_result["checkpoint"]["state"]
    assert rust_result["metrics"] == fallback_result["metrics"]


def test_req_sample_6208_protocol_sample_and_minimize_shapes() -> None:
    """REQ-SAMPLE-6208-SHAPE-CONTRACT: protocol returns one-hot label samples."""

    labels, target, proposal = _fixture_inputs()
    backend = ModeJumpRustBackend(seed=6209, prefer_rust=False)
    descriptor = descriptor_for_run(labels=labels, seed=6209, burn_in=1)

    samples = backend.sample(target, proposal, n_samples=5, config=descriptor)
    minimized = backend.minimize_energy(target, proposal, n_samples=3, n_steps=2, beta=1.0)

    assert samples.shape == (5, len(labels))
    assert samples.dtype == np.bool_
    assert np.all(samples.sum(axis=1) == 1)
    assert minimized.shape == (3, len(labels))
    assert minimized.dtype == np.bool_


def test_req_sample_6208_fallback_cases_are_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SAMPLE-6208-EXACT-FALLBACK: compatibility fallback stays exact."""

    labels, target, proposal = _fixture_inputs()
    descriptor = descriptor_for_run(
        labels=labels,
        seed=6210,
        burn_in=2,
        enable_mode_jump_runtime=True,
    )
    baseline = ModeJumpRustBackend(seed=6210, prefer_rust=False).run_descriptor(
        target,
        proposal,
        n_samples=10,
        config=descriptor,
    )

    fallback_cases = [
        (
            ModeJumpRustBackend(seed=6210),
            np.asarray(target, dtype=np.float32),
            np.asarray(proposal, dtype=np.float32),
            "unsupported_dtype_or_layout",
        ),
        (
            ModeJumpRustBackend(seed=6210),
            target.copy(),
            np.asfortranarray(proposal),
            "unsupported_dtype_or_layout",
        ),
        (
            ModeJumpRustBackend(seed=6210),
            target,
            proposal,
            "declared_python_compatibility",
            {**descriptor, "force_python_fallback": True},
        ),
        (
            ModeJumpRustBackend(
                seed=6210,
                rust_module_loader=lambda: (_ for _ in ()).throw(ImportError("missing")),
            ),
            target,
            proposal,
            "rust_extension_missing",
        ),
        (
            ModeJumpRustBackend(
                seed=6210,
                rust_module_loader=lambda: (_ for _ in ()).throw(RuntimeError("broken")),
            ),
            target,
            proposal,
            "rust_extension_error",
        ),
        (
            ModeJumpRustBackend(seed=6210, rust_module_loader=lambda: SimpleNamespace()),
            target,
            proposal,
            "rust_symbol_missing",
        ),
    ]

    for row in fallback_cases:
        backend, case_target, case_proposal, reason, *override = row
        case_descriptor = override[0] if override else descriptor
        result = backend.run_descriptor(case_target, case_proposal, 10, case_descriptor)
        assert result["receipt"]["active_backend"] == ACTIVE_PYTHON_FALLBACK
        assert reason in result["receipt"]["fallback_reason"]
        np.testing.assert_array_equal(result["samples"], baseline["samples"])
        assert result["sample_labels"] == baseline["sample_labels"]
        assert result["checkpoint"]["state"] == baseline["checkpoint"]["state"]

    monkeypatch.setenv("CARNOT_ENABLE_MODE_JUMP_RUNTIME", "1")
    env_result = ModeJumpRustBackend(seed=6210).run_descriptor(
        target,
        proposal,
        2,
        descriptor_for_run(labels=labels, seed=6210),
    )
    assert env_result["receipt"]["active_backend"] == ACTIVE_RUST_BACKEND


def test_scenario_sample_6208_checkpoint_roundtrip_and_cross_backend_restart() -> None:
    """SCENARIO-SAMPLE-6208-RUNTIME-PARITY: checkpoints resume across paths."""

    labels, target, proposal = _fixture_inputs()
    descriptor = descriptor_for_run(
        labels=labels,
        seed=6211,
        burn_in=1,
        enable_mode_jump_runtime=True,
    )
    backend = ModeJumpRustBackend(seed=6211)
    prefix = backend.run_descriptor(target, proposal, n_samples=8, config=descriptor)
    checkpoint_a = backend.save_checkpoint()
    checkpoint_b = backend.save_checkpoint()

    assert prefix["checkpoint"]["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint_a == checkpoint_b
    assert checkpoint_a["payload_checksum"] == checkpoint_checksum(checkpoint_a)
    assert (
        backend.load_checkpoint(checkpoint_a, target, proposal, config=descriptor)
        == (checkpoint_a["state"])
    )

    resumed_descriptor = {**descriptor, "checkpoint": checkpoint_a}
    rust_suffix = ModeJumpRustBackend(seed=6211).run_descriptor(
        target,
        proposal,
        n_samples=8,
        config=resumed_descriptor,
    )
    fallback_suffix = ModeJumpRustBackend(seed=6211, prefer_rust=False).run_descriptor(
        target,
        proposal,
        n_samples=8,
        config=resumed_descriptor,
    )

    np.testing.assert_array_equal(rust_suffix["samples"], fallback_suffix["samples"])
    assert rust_suffix["checkpoint"]["state"] == fallback_suffix["checkpoint"]["state"]
    assert rust_suffix["decision_log"] == fallback_suffix["decision_log"]

    corrupt = deepcopy(checkpoint_a)
    corrupt["state"]["step"] = int(corrupt["state"]["step"]) + 1
    with pytest.raises(ValueError, match="checksum"):
        backend.load_checkpoint(corrupt, target, proposal, config=descriptor)

    mismatched_seed = {**descriptor, "seed": int(descriptor["seed"]) + 1}
    with pytest.raises(ValueError, match="seed mismatch"):
        backend.load_checkpoint(checkpoint_a, target, proposal, config=mismatched_seed)


@pytest.mark.parametrize(
    "bad_target,bad_proposal,match",
    [
        (np.ones((1, 6), dtype=np.float64), np.zeros((6, 6), dtype=np.float64), "target"),
        (np.ones(6, dtype=np.float64), np.zeros((5, 5), dtype=np.float64), "proposal"),
        (np.array([0.36, 0.24, 0.025, 0.025, 0.245, 0.0]), None, "target"),
        (np.array([0.36, 0.24, 0.025, 0.025, 0.245, np.inf]), None, "finite"),
    ],
)
def test_scenario_sample_6208_invalid_shapes_and_values_fail_closed(
    bad_target: np.ndarray,
    bad_proposal: np.ndarray | None,
    match: str,
) -> None:
    """SCENARIO-SAMPLE-6208-BOUNDARY-ERRORS: malformed inputs raise errors."""

    labels, target, proposal = _fixture_inputs()
    with pytest.raises(ValueError, match=match):
        ModeJumpRustBackend(seed=6212).run_descriptor(
            bad_target,
            proposal if bad_proposal is None else bad_proposal,
            n_samples=2,
            config=descriptor_for_run(
                labels=labels,
                seed=6212,
                enable_mode_jump_runtime=True,
            ),
        )

    bad_descriptor = descriptor_for_run(labels=labels, seed=6212)
    bad_descriptor["algorithm"] = "wrong"
    with pytest.raises(ValueError, match="algorithm"):
        ModeJumpRustBackend().run_descriptor(target, proposal, 2, bad_descriptor)

    bad_descriptor = descriptor_for_run(labels=labels, seed=6212)
    bad_descriptor["topology"] = "two_axis"
    with pytest.raises(ValueError, match="topology"):
        ModeJumpRustBackend().run_descriptor(target, proposal, 2, bad_descriptor)

    bad_descriptor = descriptor_for_run(labels=labels, seed=6212)
    bad_descriptor["labels"] = [*labels[:-1], "shadow"]
    with pytest.raises(ValueError, match="labels"):
        ModeJumpRustBackend().run_descriptor(target, proposal, 2, bad_descriptor)


def test_scenario_sample_6208_cancellation_timeout_and_runtime_edges() -> None:
    """SCENARIO-SAMPLE-6208-BOUNDARY-ERRORS: interruption controls fail closed."""

    labels, target, proposal = _fixture_inputs()
    descriptor = descriptor_for_run(labels=labels, seed=6213, enable_mode_jump_runtime=True)

    with pytest.raises(TimeoutError, match="cancelled"):
        ModeJumpRustBackend(seed=6213).run_descriptor(
            target,
            proposal,
            n_samples=4,
            config={**descriptor, "cancel_after_steps": 0},
        )
    with pytest.raises(TimeoutError, match="timeout"):
        ModeJumpRustBackend(seed=6213).run_descriptor(
            target,
            proposal,
            n_samples=4,
            config={**descriptor, "timeout_s": 0.0},
        )
    with pytest.raises(ValueError, match="n_samples"):
        ModeJumpRustBackend(seed=6213).run_descriptor(target, proposal, 0, descriptor)
    with pytest.raises(ValueError, match="initial_label"):
        ModeJumpRustBackend(seed=6213).run_descriptor(
            target,
            proposal,
            2,
            {**descriptor, "initial_label": "shadow"},
        )
    with pytest.raises(ValueError, match="no mode-jump checkpoint"):
        ModeJumpRustBackend().save_checkpoint()

    bad_checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": MODE_JUMP_ALGORITHM,
        "topology": MODE_JUMP_TOPOLOGY,
        "state": {"current_label": "left_peak"},
        "payload_checksum": "bad",
    }
    with pytest.raises(ValueError, match="checksum"):
        ModeJumpRustBackend().load_checkpoint(bad_checkpoint, target, proposal, config=descriptor)
