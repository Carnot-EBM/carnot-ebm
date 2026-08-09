"""Tests for Exp6220 mode-jump runtime A/B artifact.

Spec refs: REQ-SAMPLE-6220,
SCENARIO-SAMPLE-6220-MATCHED-RUNTIME-QUALITY,
SCENARIO-SAMPLE-6220-UNSUPPORTED-FIXTURE-BOUNDARY,
SCENARIO-SAMPLE-6220-STATE-FALLBACK-NOCLAIM.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6220_mode_jump_runtime_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"


def _passing_receipts() -> list[dict[str, object]]:
    commands = (
        ".venv/bin/pytest tests/python/test_experiment_6220_mode_jump_runtime_ab.py -q -o addopts=",
        ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6220_mode_jump_runtime_ab.py -m pytest tests/python/test_experiment_6220_mode_jump_runtime_ab.py -q --no-cov -o addopts=",
        ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6220_mode_jump_runtime_ab.py --fail-under=100",
        ".venv/bin/python -m carnot.experiment_6220_mode_jump_runtime_ab --date 20260809",
        ".venv/bin/pytest tests/python -q",
    )
    return [
        {
            "name": f"task_cmd_{index}",
            "command": command,
            "exit_code": 0,
            "task_owned": True,
            "classification": "task_owned",
        }
        for index, command in enumerate(commands)
    ]


def test_req_sample_6220_spec_declares_fields_and_principles() -> None:
    """REQ-SAMPLE-6220-ARTIFACT: OpenSpec anchors every required field."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-6220") :]
    normalized = " ".join(section.split())

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized
    for marker in (
        "SCENARIO-SAMPLE-6220-MATCHED-RUNTIME-QUALITY",
        "SCENARIO-SAMPLE-6220-UNSUPPORTED-FIXTURE-BOUNDARY",
        "SCENARIO-SAMPLE-6220-STATE-FALLBACK-NOCLAIM",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section


def test_req_sample_6220_artifact_schema_and_no_claim_gates(tmp_path: Path) -> None:
    """REQ-SAMPLE-6220-ARTIFACT: JSON validates and stays software-only."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.write_artifact(
        output_path=output,
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        run_date="20260809",
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_partial"
    assert artifact["honest_verdict"].startswith("complete_partial:")
    assert artifact["default_off_preserved"] is True
    assert artifact["fpga_tsu_power_hardware_claim_count"] == 0
    assert artifact["timing_claim_allowed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"]["value"] is True


def test_scenario_sample_6220_support_quality_and_unsupported_boundary() -> None:
    """REQ-SAMPLE-6220-SUPPORT: accepted and unsupported fixtures are separated."""

    artifact = mod.build_artifact(
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        run_date="20260809",
    )

    support = artifact["support_validity_by_fixture_arm"]
    multimodal = support["fixtures"]["multimodal_exp6194"]
    unimodal = support["fixtures"]["unimodal_contract_probe"]
    assert multimodal["fallback_exact"]["support_valid"] is True
    assert multimodal["mode_jump_runtime"]["support_valid"] is True
    assert multimodal["mode_jump_runtime"]["active_backend"] == "rust_pyo3"
    assert unimodal["fallback_exact"]["support_valid"] is False
    assert unimodal["mode_jump_runtime"]["support_valid"] is False
    assert unimodal["fallback_exact"]["fail_closed"] is True
    assert unimodal["mode_jump_runtime"]["fail_closed"] is True

    errors = artifact["observable_and_energy_error_by_fixture_arm"]
    assert errors["all_supported_errors_within_tolerance"] is True
    assert errors["fixtures"]["multimodal_exp6194"]["mode_jump_runtime"]["samples_match_fallback"]
    assert artifact["quality_gate_passed"] is False


def test_scenario_sample_6220_state_fallback_and_timing_receipts() -> None:
    """SCENARIO-SAMPLE-6220-STATE-FALLBACK-NOCLAIM: state and fallback gates pass."""

    artifact = mod.build_artifact(
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        run_date="20260809",
    )

    assert artifact["serialization_and_restart_receipts"]["serialization_pass"] is True
    assert artifact["fallback_trigger_and_exactness"]["all_fallbacks_exact"] is True
    assert artifact["ess_and_autocorrelation_by_fixture_arm"]["all_supported_mixing_pass"] is True
    assert artifact["transition_and_mode_occupancy_counts"]["all_supported_counts_recorded"]
    assert artifact["cpu_thread_and_wall_time_receipts"]["timing_is_diagnostic"] is True
    assert artifact["cpu_thread_and_wall_time_receipts"]["uncertainty_excludes_parity"] in {
        True,
        False,
    }


def test_req_sample_6220_task_owned_failure_blocks_without_relabeling() -> None:
    """REQ-SAMPLE-6220-ARTIFACT: task-owned failures are not called pre-existing."""

    receipts = [
        *_passing_receipts(),
        {
            "name": "focused_failure",
            "command": "pytest failed",
            "exit_code": 1,
            "task_owned": True,
            "classification": "task_owned",
        },
        {
            "name": "old_workspace_failure",
            "command": "cargo fmt --all -- --check",
            "exit_code": 1,
            "task_owned": False,
            "classification": "preexisting_repository_wide_nonzero",
        },
    ]
    artifact = mod.build_artifact(
        root=REPO,
        command_receipts=receipts,
        duration_s=0.0,
        run_date="20260809",
    )

    classified = artifact["task_owned_and_preexisting_test_classification"]
    assert artifact["status"] == "blocked"
    assert classified["task_owned_failure_count"] == 1
    assert classified["preexisting_nonzero_count"] >= 1
    assert classified["task_owned_failures"][0]["name"] == "focused_failure"
    assert "focused_failure" in artifact["honest_verdict"]


def test_req_sample_6220_validate_artifact_rejects_bad_required_fields() -> None:
    """REQ-SAMPLE-6220-ARTIFACT: schema mutations fail closed."""

    artifact = mod.build_artifact(
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        run_date="20260809",
    )
    mutations = [
        (
            "fpga_tsu_power_hardware_claim_count",
            lambda data: data.__setitem__("fpga_tsu_power_hardware_claim_count", 1),
        ),
        ("default_off_preserved", lambda data: data.__setitem__("default_off_preserved", False)),
        ("timing_claim_allowed", lambda data: data.__setitem__("timing_claim_allowed", True)),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "gpu")),
        ("status", lambda data: data.__setitem__("status", "complete_ready")),
        ("quality_gate_passed", lambda data: data.__setitem__("quality_gate_passed", True)),
        (
            "sampler_runtime_ready_score",
            lambda data: data.__setitem__("sampler_runtime_ready_score", 1.0),
        ),
        (
            "honest_verdict",
            lambda data: data.__setitem__("honest_verdict", "complete_ready: wrong"),
        ),
        (
            "field_principles",
            lambda data: data["field_principles"].__setitem__("status", "wrong"),
        ),
        (
            "field_provenance",
            lambda data: data["field_provenance"]["status"].__setitem__("source", ""),
        ),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum)


def test_req_sample_6220_helper_edges_for_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLE-6220-ARTIFACT: helper edge cases stay deterministic."""

    assert mod.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json(bad_json)  # noqa: SLF001

    with monkeypatch.context() as scoped:
        scoped.setattr(
            mod,
            "_run_arm",
            lambda *_args, **_kwargs: {"success": False, "message": "unsupported"},
        )
        with pytest.raises(RuntimeError, match="unsupported"):
            mod._time_one_run(REPO, "fallback_exact")  # noqa: SLF001

    assert mod._interval([0.5]) == [0.5, 0.5]  # noqa: SLF001
    assert mod._expect_error(lambda: None, ValueError)["raised"] is False  # noqa: SLF001
    assert mod.status({}) == "blocked"

    receipts_path = tmp_path / "receipts.json"
    receipts_path.write_text(json.dumps(_passing_receipts()), encoding="utf-8")
    monkeypatch.setenv("CARNOT_6220_COMMAND_RECEIPTS", str(receipts_path))
    assert mod._external_command_receipts() == _passing_receipts()  # noqa: SLF001

    missing_default = tmp_path / "missing.json"
    monkeypatch.delenv("CARNOT_6220_COMMAND_RECEIPTS", raising=False)
    monkeypatch.setattr(mod, "DEFAULT_RECEIPT_PATH", missing_default)
    assert mod._external_command_receipts() is None  # noqa: SLF001

    bad_receipts = tmp_path / "bad_receipts.json"
    bad_receipts.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CARNOT_6220_COMMAND_RECEIPTS", str(bad_receipts))
    with pytest.raises(ValueError, match="command receipt payload"):
        mod._external_command_receipts()  # noqa: SLF001

    readyish = mod.build_artifact(
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
        run_date="20260809",
    )
    readyish["support_validity_by_fixture_arm"]["all_required_fixture_arms_supported"] = True
    readyish["quality_gate_passed"] = True
    readyish["cpu_thread_and_wall_time_receipts"]["uncertainty_excludes_parity"] = True
    readyish["timing_claim_allowed"] = True
    assert mod.sampler_runtime_ready_score(readyish) == 1.0
    assert mod.status(readyish) == "complete_ready"

    bad = deepcopy(readyish)
    bad["field_provenance"] = []
    bad["sampler_runtime_ready_score"] = mod.sampler_runtime_ready_score(bad)
    bad["status"] = mod.status(bad)
    bad["honest_verdict"] = mod.honest_verdict(bad)
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad)


def test_req_sample_6220_cli_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-SAMPLE-6220-ARTIFACT: CLI accepts the required date argument."""

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        mod,
        "write_artifact",
        lambda **_kwargs: {
            "status": "complete_partial",
            "reproducibility_checksum": "sha256:" + "0" * 64,
        },
    )

    assert mod.main(["--date", "20260809"]) == 0
    assert "complete_partial" in capsys.readouterr().out
