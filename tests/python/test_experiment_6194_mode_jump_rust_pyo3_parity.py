"""Tests for Exp6194 fixed mode-jump Rust/PyO3 parity.

Spec refs: REQ-SAMPLE-6194, REQ-RUSTPY-6194,
SCENARIO-SAMPLE-6194-EXACT-TRANSITION-PARITY,
SCENARIO-SAMPLE-6194-DISTRIBUTION-QUALITY-PARITY,
SCENARIO-SAMPLE-6194-SERIALIZATION-ERROR-PRESERVATION,
SCENARIO-RUSTPY-6194-BOUNDARY-PARITY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6194_mode_jump_rust_pyo3_parity as mod
from carnot._rust import RustModeJumpConfig, RustModeJumpCore, RustModeJumpState


REPO = Path(__file__).resolve().parents[2]
SAMPLER_SPEC = REPO / "openspec/capabilities/samplers/spec.md"
BOUNDARY_SPEC = REPO / "openspec/capabilities/rust-python-boundary/spec.md"


def _passing_receipts() -> list[dict[str, object]]:
    commands = (
        "cargo test -p carnot-samplers --test mode_jump",
        "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python",
        ".venv/bin/pytest tests/python/test_experiment_6194_mode_jump_rust_pyo3_parity.py -q --no-cov -n 0",
        ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6194_mode_jump_rust_pyo3_parity.py crates/carnot-samplers/tests/mode_jump.rs",
        ".venv/bin/python -c \"import carnot._rust as r; assert hasattr(r, 'RustModeJumpCore')\"",
    )
    return [
        {
            "name": f"cmd_{index}",
            "command": command,
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
        }
        for index, command in enumerate(commands)
    ]


def test_req_sample_6194_specs_declare_sampler_and_boundary_contracts() -> None:
    """REQ-SAMPLE-6194/REQ-RUSTPY-6194: OpenSpec anchors Exp6194."""

    sampler = SAMPLER_SPEC.read_text(encoding="utf-8")
    section = sampler[sampler.index("### REQ-SAMPLE-6194") :]
    boundary = BOUNDARY_SPEC.read_text(encoding="utf-8")
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAMPLE-6194-PROPOSAL",
        "REQ-SAMPLE-6194-ACCEPTANCE",
        "REQ-SAMPLE-6194-RNG-DETERMINISM",
        "REQ-SAMPLE-6194-SERIALIZATION",
        "REQ-SAMPLE-6194-PYO3-BINDING",
        "REQ-SAMPLE-6194-EXACT-PARITY",
        "REQ-SAMPLE-6194-DISTRIBUTION-PARITY",
        "REQ-SAMPLE-6194-ERROR-HANDLING",
        "REQ-SAMPLE-6194-NO-HARDWARE",
        "REQ-SAMPLE-6194-DETERMINATION-PRESERVATION",
        "SCENARIO-SAMPLE-6194-EXACT-TRANSITION-PARITY",
        "SCENARIO-SAMPLE-6194-DISTRIBUTION-QUALITY-PARITY",
        "SCENARIO-SAMPLE-6194-SERIALIZATION-ERROR-PRESERVATION",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized

    assert "REQ-RUSTPY-6194" in boundary
    assert "SCENARIO-RUSTPY-6194-BOUNDARY-PARITY" in boundary
    assert "silent fallback" in boundary


def test_scenario_sample_6194_python_fixture_is_derived_from_immutable_evidence() -> None:
    """SCENARIO-SAMPLE-6194-EXACT-TRANSITION-PARITY: fixture is frozen."""

    config = mod.fixed_algorithm_equations_config_and_seed(REPO)
    fixture = mod.build_exact_transition_fixture(config, step_count=10)
    first = fixture["events"][0]
    seventh = fixture["events"][6]

    assert config["source_result_hashes"]["exp6166_result"] == mod.sha256_file(
        REPO / mod.EXP6166_RESULT_RELATIVE_PATH
    )
    assert config["source_result_hashes"]["exp6180_result"] == mod.sha256_file(
        REPO / mod.EXP6180_RESULT_RELATIVE_PATH
    )
    assert config["hardware_scope_excluded"] == ["FPGA", "TSU", "THRML scaling", "two-axis"]
    assert first["current_label"] == "left_peak"
    assert first["proposed_label"] == "left_shoulder"
    assert first["accepted"] is False
    assert first["state_after"]["current_label"] == "left_peak"
    assert first["rng_state_after"] == 8287729265405344948
    assert seventh["proposed_label"] == "right_peak"
    assert seventh["accepted"] is True
    assert seventh["state_after"]["current_label"] == "right_peak"
    assert fixture["fixture_sha256"].startswith("sha256:")


def test_scenario_rustpy_6194_exact_step_parity_and_serialization() -> None:
    """SCENARIO-RUSTPY-6194-BOUNDARY-PARITY: PyO3 replays exact steps."""

    parity = mod.compare_exact_transition_parity(REPO, step_count=10)

    assert RustModeJumpConfig is not None
    assert RustModeJumpCore is not None
    assert RustModeJumpState is not None
    assert parity["all_fields_match"] is True
    assert parity["mismatch_count"] == 0
    assert parity["final_python_state"] == parity["final_rust_state"]
    assert parity["serialized_state_match"] is True
    assert parity["fixture_sha256"].startswith("sha256:")


def test_scenario_sample_6194_distribution_quality_parity() -> None:
    """SCENARIO-SAMPLE-6194-DISTRIBUTION-QUALITY-PARITY: long run matches target."""

    metrics = mod.compare_distribution_metrics(REPO)

    assert metrics["sample_count"] == mod.LONG_RUN_SAMPLE_COUNT
    assert metrics["burn_in"] == mod.LONG_RUN_BURN_IN
    assert metrics["python"]["tv_to_target"] <= mod.TOLERANCES["target_tv"]
    assert metrics["rust"]["tv_to_target"] <= mod.TOLERANCES["target_tv"]
    assert metrics["python"]["kl_target_to_empirical"] <= mod.TOLERANCES["target_kl"]
    assert metrics["rust"]["kl_target_to_empirical"] <= mod.TOLERANCES["target_kl"]
    assert metrics["python_rust_frequency_delta_max"] <= mod.TOLERANCES["python_rust_freq_delta"]
    assert metrics["rust"]["effective_sample_size"] > mod.TOLERANCES["ess_min"]


def test_scenario_sample_6194_artifact_schema_and_error_controls(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-6194-SERIALIZATION-ERROR-PRESERVATION: artifact validates."""

    before = mod.snapshot_preconditions(REPO, exp6184_preflight_exit_code=0)
    artifact = mod.write_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        before_snapshot=before,
    )
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["mode_jump_rust_pyo3_ready_score"] == 1.0
    assert artifact["hardware_or_speedup_claimed"] is False
    assert artifact["timing_diagnostic_only"] is True
    assert artifact["historical_artifacts_unchanged"]["unchanged"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert (
        artifact["serialization_snapshot_restore_and_error_receipts"]["all_error_controls_passed"]
        is True
    )

    non_task_receipts = [
        *_passing_receipts(),
        {
            "name": "full_python_suite",
            "command": ".venv/bin/pytest tests/python -q",
            "exit_code": 2,
            "classification": "unrelated_preexisting",
            "task_owned": False,
        },
    ]
    non_task_artifact = mod.build_artifact(
        root=REPO,
        command_receipts=non_task_receipts,
        duration_s=0.0,
        before_snapshot=before,
    )
    assert non_task_artifact["status"] == "complete_ready"
    assert non_task_artifact["mode_jump_rust_pyo3_ready_score"] == 1.0
    assert "full_python_suite" in non_task_artifact["honest_verdict"]
    assert "unrelated_preexisting" in non_task_artifact["honest_verdict"]

    bad = deepcopy(artifact)
    bad["hardware_or_speedup_claimed"] = True
    bad["mode_jump_rust_pyo3_ready_score"] = mod.ready_score(bad)
    bad["status"] = mod.status(bad)
    bad["honest_verdict"] = mod.honest_verdict(bad)
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    with pytest.raises(ValueError, match="hardware_or_speedup_claimed"):
        mod.validate_artifact(bad)

    missing = deepcopy(artifact)
    del missing["status"]
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum)


def test_req_sample_6194_defensive_edges_and_cli_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-SAMPLE-6194-ERROR-HANDLING: defensive branches are covered."""

    non_object = tmp_path / "non_object.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object expected"):
        mod._read_json(non_object)

    missing_command = mod._run_text(["/definitely/missing/carnot-command"], REPO)
    assert missing_command["available"] is False
    assert mod._draw_label(["a"], [0.0], 1.0) == "a"
    with pytest.raises(ValueError, match="step_count"):
        mod.build_exact_transition_fixture(step_count=0)
    assert mod._quality_from_indicator([1.0, 1.0]) == (0.0, 1.0, 2.0)
    assert mod._expect_value_error(lambda: None)["raised"] is False
    assert mod.status(
        {"exact_transition_fixture_hash_and_parity_matrix": {"all_fields_match": False}}
    ) == ("blocked")
    assert mod.honest_verdict(
        {"exact_transition_fixture_hash_and_parity_matrix": {"all_fields_match": False}}
    ).startswith("blocked:")

    before = mod.snapshot_preconditions(REPO, exp6184_preflight_exit_code=0)
    artifact = mod.build_artifact(
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        before_snapshot=before,
    )
    mutations = [
        ("timing_diagnostic_only", lambda data: data.__setitem__("timing_diagnostic_only", False)),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "gpu")),
        (
            "mode_jump_rust_pyo3_ready_score",
            lambda data: data.__setitem__("mode_jump_rust_pyo3_ready_score", 0.0),
        ),
        ("status", lambda data: data.__setitem__("status", "blocked")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "blocked: wrong")),
        ("field_provenance", lambda data: data.__setitem__("field_provenance", [])),
        (
            "field_provenance:status",
            lambda data: data["field_provenance"]["status"].__setitem__("principle", "wrong"),
        ),
    ]
    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {"field_provenance"}:
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "snapshot_preconditions", lambda *_args, **_kwargs: {"fake": True})
    monkeypatch.setattr(
        mod,
        "write_artifact",
        lambda **_kwargs: {"status": "complete_ready"},
    )
    assert mod.main() == 0
    assert "complete_ready" in capsys.readouterr().out
