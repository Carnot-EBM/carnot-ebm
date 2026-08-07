"""Tests for Exp6208 mode-jump runtime integration artifact.

Spec coverage: REQ-SAMPLE-6208, SCENARIO-SAMPLE-6208-DEFAULT-OFF-FALLBACK,
SCENARIO-SAMPLE-6208-RUNTIME-PARITY, SCENARIO-SAMPLE-6208-BOUNDARY-ERRORS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6208_mode_jump_runtime_integration as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"


def _passing_receipts() -> list[dict[str, object]]:
    commands = (
        "cargo test -p carnot-samplers --test mode_jump",
        "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python",
        ".venv/bin/pytest tests/python/samplers/test_mode_jump_rust_backend.py tests/python/test_experiment_6208_mode_jump_runtime_integration.py -q",
        ".venv/bin/python scripts/check_spec_coverage.py tests/python/samplers/test_mode_jump_rust_backend.py tests/python/test_experiment_6208_mode_jump_runtime_integration.py crates/carnot-samplers/tests/mode_jump.rs",
    )
    return [
        {
            "name": f"cmd_{index}",
            "command": command,
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
            "task_owned": True,
        }
        for index, command in enumerate(commands)
    ]


def test_req_sample_6208_spec_declares_artifact_fields_and_principles() -> None:
    """REQ-SAMPLE-6208-ARTIFACT: OpenSpec anchors every required field."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-6208") :]
    normalized = " ".join(section.split())

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert mod.INFERENCE_SUBSTRATE in section


def test_req_sample_6208_artifact_schema_and_ready_receipts(tmp_path: Path) -> None:
    """REQ-SAMPLE-6208-ARTIFACT: terminal JSON validates and stays no-hardware."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.write_artifact(
        output_path=output,
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["hardware_or_speed_power_energy_claimed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["default_off_receipt"]["default_backend_name"] == "cpu"
    assert artifact["default_off_receipt"]["mode_jump_backend_name"] == "mode_jump_rust"
    assert artifact["config_and_feature_flag_contract"]["feature_env_var"] == (
        "CARNOT_ENABLE_MODE_JUMP_RUNTIME"
    )
    assert artifact["seeded_quality_parity"]["seeded_samples_match"] is True
    assert artifact["serialization_roundtrip"]["checkpoint_roundtrip_pass"] is True
    assert artifact["cancellation_timeout_and_error_receipts"]["all_controls_passed"] is True
    assert artifact["exact_fallback_receipts"]["all_fallbacks_exact"] is True
    assert artifact["verifier_is_oracle"]["value"] is True


def test_req_sample_6208_artifact_classifies_unrelated_nonzero_without_blocking() -> None:
    """REQ-SAMPLE-6208-ARTIFACT: unrelated nonzero commands stay classified."""

    receipts = [
        *_passing_receipts(),
        {
            "name": "exploratory_rg_typo",
            "command": "rg crates/Cargo.toml",
            "exit_code": 2,
            "classification": "unrelated_preexisting",
            "task_owned": False,
        },
    ]
    artifact = mod.build_artifact(root=REPO, command_receipts=receipts, duration_s=0.0)

    assert artifact["status"] == "complete_ready"
    assert artifact["task_owned_test_commands_and_exit_codes"]["all_task_owned_commands_passed"]
    assert artifact["unrelated_nonzero_command_classifications"] == [
        {
            "name": "exploratory_rg_typo",
            "command": "rg crates/Cargo.toml",
            "exit_code": 2,
            "classification": "unrelated_preexisting",
            "task_owned": False,
        }
    ]
    assert "exploratory_rg_typo" in artifact["honest_verdict"]


def test_req_sample_6208_validate_artifact_rejects_bad_required_fields() -> None:
    """REQ-SAMPLE-6208-ARTIFACT: schema mutations fail closed."""

    artifact = mod.build_artifact(
        root=REPO,
        command_receipts=_passing_receipts(),
        duration_s=0.0,
    )
    mutations = [
        (
            "hardware_or_speed_power_energy_claimed",
            lambda data: data.__setitem__("hardware_or_speed_power_energy_claimed", True),
        ),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "gpu")),
        ("status", lambda data: data.__setitem__("status", "blocked")),
        (
            "honest_verdict",
            lambda data: data.__setitem__("honest_verdict", "blocked: wrong"),
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


def test_req_sample_6208_cli_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-SAMPLE-6208-ARTIFACT: CLI accepts the required date argument."""

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        mod,
        "write_artifact",
        lambda **_kwargs: {
            "status": "complete_ready",
            "reproducibility_checksum": "sha256:" + "0" * 64,
        },
    )

    assert mod.main(["--date", "20260807"]) == 0
    assert "complete_ready" in capsys.readouterr().out
