"""Tests for Exp5723 one-axis Rust SamplerBackend integration.

Spec coverage: REQ-SAMPLE-5723, SCENARIO-SAMPLE-5723
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5723_one_axis_rust_samplerbackend_integration as mod


REPO = Path(__file__).resolve().parents[2]
TEST_PATHS = [
    "tests/python/samplers/test_one_axis_rust_backend.py",
    "tests/python/test_experiment_5723_one_axis_rust_samplerbackend_integration.py",
]


def test_req_sample_5723_artifact_builder_records_all_required_gates() -> None:
    """REQ-SAMPLE-5723: artifact builder records lineage, exposure, parity, and controls."""
    artifact = mod.build_artifact(root=REPO, random_seeds=mod.DEFAULT_RANDOM_SEEDS[:2])

    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["upstream_gate_receipts"]["exp5714"]["ready"] is True
    assert artifact["upstream_gate_receipts"]["exp5715"]["ready"] is True
    assert artifact["upstream_gate_receipts"]["exp5714"]["algorithm_hash_matches"] is True
    assert artifact["upstream_gate_receipts"]["exp5715"]["checkpoint_schema_matches"] is True
    assert artifact["source_algorithm_hash"] == mod.exp5714.source_algorithm_hash()
    assert len(artifact["rust_source_hash"]) == 64
    assert len(artifact["pyo3_binding_hash"]) == 64
    assert len(artifact["python_adapter_hash"]) == 64
    assert artifact["sampler_backend_protocol"]["is_sampler_backend"] is True
    assert artifact["factory_registration_receipt"]["explicit_backend_name"] == "one_axis_rust"
    assert artifact["factory_registration_receipt"]["default_backend_preserved"] is True
    assert artifact["supported_descriptor_schema"]["algorithm"] == mod.ONE_AXIS_ALGORITHM
    assert artifact["unsupported_case_policy"]["unsupported_topology"] == "fail_closed"
    assert artifact["seed_semantics"]["rng"] == "portable_lcg_u64"
    assert artifact["temperature_ladder_receipt"]["matches_exp5714"] is True
    assert artifact["swap_schedule_receipt"]["label_only_adjacent_swaps"] is True
    assert artifact["transition_budget_receipt"]["matched"] is True
    assert artifact["energy_parity_max_error"] <= mod.FROZEN_TOLERANCES["energy"]
    assert artifact["proposal_parity_max_error"] <= mod.FROZEN_TOLERANCES["proposal"]
    assert artifact["swap_parity_max_error"] <= mod.FROZEN_TOLERANCES["swap"]
    assert artifact["decision_log_parity"] is True
    assert artifact["checkpoint_schema_version"] == mod.CHECKPOINT_SCHEMA_VERSION
    assert artifact["checkpoint_roundtrip_pass"] is True
    assert artifact["python_to_rust_restart_pass"] is True
    assert artifact["rust_to_python_restart_pass"] is True
    assert artifact["fallback_equivalence_pass"] is True
    assert artifact["exact_fallback_equivalence_score"] == 1.0
    assert {row["control_id"] for row in artifact["broken_control_results"]} == set(
        mod.BROKEN_CONTROL_IDS
    )
    assert all(row["passed"] is True for row in artifact["broken_control_results"])
    assert artifact["one_axis_samplerbackend_ready_score"] == 1.0
    assert artifact["two_axis_code_added"] is False
    assert artifact["timing_claimed"] is False
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_sample_5723_writes_valid_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5723: runner writes the required JSON artifact."""
    artifact = mod.build_artifact(
        root=REPO,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS[:2],
        tests_added_or_reused=TEST_PATHS,
    )
    output_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert saved["tests_added_or_reused"] == TEST_PATHS
    assert saved["one_axis_samplerbackend_ready_score"] == 1.0
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_sample_5723_validation_and_ready_score_fail_closed() -> None:
    """REQ-SAMPLE-5723: manual edits, missing controls, and claims block readiness."""
    artifact = mod.build_artifact(root=REPO, random_seeds=mod.DEFAULT_RANDOM_SEEDS[:1])
    mutations = [
        ("missing required field", lambda data: data.pop("field_principles")),
        ("field_principles", lambda data: data["field_principles"].__setitem__("x", "y")),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "bad")),
        ("two_axis_code_added", lambda data: data.__setitem__("two_axis_code_added", True)),
        ("timing_claimed", lambda data: data.__setitem__("timing_claimed", True)),
        (
            "hardware_speedup_claimed",
            lambda data: data.__setitem__("hardware_speedup_claimed", True),
        ),
        (
            "one_axis_samplerbackend_ready_score",
            lambda data: data.__setitem__("one_axis_samplerbackend_ready_score", 0.0),
        ),
        (
            "honest_verdict mismatch",
            lambda data: data.__setitem__("honest_verdict", "blocked: stale"),
        ),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        (
            "reproducibility_checksum",
            lambda data: data.__setitem__("reproducibility_checksum", "bad"),
        ),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {"missing required field", "reproducibility_checksum"}:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    blocked = deepcopy(artifact)
    blocked["fallback_equivalence_pass"] = False
    blocked["exact_fallback_equivalence_score"] = 0.0
    blocked["one_axis_samplerbackend_ready_score"] = mod.ready_score(blocked)
    assert blocked["one_axis_samplerbackend_ready_score"] == 0.0
    assert mod.honest_verdict(blocked).startswith("blocked:")


def test_req_sample_5723_helper_edges_fail_closed() -> None:
    """REQ-SAMPLE-5723: helper edge cases report deterministic failures."""
    assert mod.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    with pytest.raises(ValueError, match="random_seeds"):
        mod.build_artifact(root=REPO, random_seeds=[])

    mismatch = mod._decision_log_errors([{"current_energy": 0.0}], [])
    assert mismatch == {
        "energy_parity_max_error": 1.0,
        "proposal_parity_max_error": 1.0,
        "swap_parity_max_error": 1.0,
    }
    assert mod._raises_value_error(lambda: None) is False


def test_req_sample_5723_main_delegates_artifact_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-SAMPLE-5723: CLI entrypoint delegates build and write steps."""
    calls: list[tuple[str, object]] = []

    def fake_build(**kwargs):
        calls.append(("build", kwargs))
        return {"ok": True}

    def fake_write(root, artifact):
        calls.append(("write", (root, artifact)))
        return Path("results/fake.json")

    monkeypatch.setattr(mod, "build_artifact", fake_build)
    monkeypatch.setattr(mod, "write_output", fake_write)

    mod.main()

    assert calls == [
        ("build", {"root": mod.REPO_ROOT, "random_seeds": mod.DEFAULT_RANDOM_SEEDS}),
        ("write", (mod.REPO_ROOT, {"ok": True})),
    ]
