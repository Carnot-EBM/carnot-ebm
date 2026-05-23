"""Tests for Exp 2901 THRML local import repair.

Spec traces: REQ-SAMPLE-096, SCENARIO-SAMPLE-096.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import carnot.samplers.thrml_local_import_repair_v1 as exp2901
from carnot.samplers.thrml_installability_preflight import CommandResult


def _result(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> CommandResult:
    return CommandResult(
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        duration_s=0.01,
    )


def _terminal_import_stdout(version: str = "0.7.0") -> str:
    return json.dumps(
        {
            "metadata_version": version,
            "path": "/tmp/project/.venv/lib/python/site-packages/thrml/__init__.py",
            "version": version,
        },
        sort_keys=True,
    )


def _traceback() -> str:
    return (
        "Traceback (most recent call last):\n"
        '  File "<stdin>", line 3, in <module>\n'
        "ModuleNotFoundError: No module named 'thrml'\n"
    )


def _parity_metrics(delta: float = 0.0) -> dict[str, Any]:
    return {
        "case_id": "exp2901:n16_signed_ring_chord:exact_distribution",
        "state_count": 65536,
        "carnot_mean_energy": -3.25,
        "thrml_mean_energy": -3.25 + float(delta),
        "mean_energy_delta": float(delta),
        "max_energy_abs_delta": float(delta),
    }


def test_req_sample_096_spec_anchor_exists() -> None:
    """REQ-SAMPLE-096, SCENARIO-SAMPLE-096: Exp2901 is spec-anchored."""

    spec = (exp2901.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-096" in spec
    assert "SCENARIO-SAMPLE-096" in spec
    assert "experiment_2901_thrml_local_import_repair_v1.json" in spec
    assert "pip install -U thrml" in spec


def test_req_sample_096_probes_jax_and_captures_full_thrml_traceback() -> None:
    """REQ-SAMPLE-096: JAX version and pre-repair THRML traceback are captured."""

    def runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        code = command[-1]
        if "import jax" in code:
            return _result(command, stdout="0.10.0\n")
        if "import thrml" in code:
            return _result(command, returncode=1, stderr=_traceback())
        raise AssertionError(f"unexpected command: {command}")

    jax_probe = exp2901.probe_jax_version(python_executable="python", runner=runner)
    thrml_probe = exp2901.probe_thrml_import(python_executable="python", runner=runner)

    assert jax_probe.version == "0.10.0"
    assert jax_probe.command_result["returncode"] == 0
    assert thrml_probe.import_succeeded is False
    assert thrml_probe.version is None
    assert "Traceback (most recent call last):" in thrml_probe.traceback_text
    assert "ModuleNotFoundError: No module named 'thrml'" in thrml_probe.traceback_text


def test_req_sample_096_probe_error_paths_are_explicit() -> None:
    """REQ-SAMPLE-096: failed JAX probes and malformed THRML metadata are explicit."""

    def failed_jax(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        return _result(command, returncode=1, stderr="jax exploded")

    def empty_jax(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        return _result(command, stdout="\n")

    def malformed_thrml(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        return _result(command, stdout="not-json\n")

    with pytest.raises(RuntimeError, match="JAX precondition failed"):
        exp2901.probe_jax_version(python_executable="python", runner=failed_jax)
    with pytest.raises(RuntimeError, match="no version output"):
        exp2901.probe_jax_version(python_executable="python", runner=empty_jax)

    probe = exp2901.probe_thrml_import(python_executable="python", runner=malformed_thrml)
    assert probe.import_succeeded is False
    assert "metadata JSON parsing failed" in probe.traceback_text


def test_req_sample_096_repair_scope_and_failed_install_are_recorded(tmp_path: Path) -> None:
    """REQ-SAMPLE-096: repair is scoped to .venv and failed pip is preserved."""

    initial = exp2901.ThrmlImportProbe(
        import_succeeded=False,
        version=None,
        import_path=None,
        traceback_text=_traceback(),
        command_result={"returncode": 1},
    )

    with pytest.raises(RuntimeError, match="outside the project .venv"):
        exp2901.repair_thrml_import_if_needed(
            initial,
            project_root=tmp_path,
            python_executable="/usr/bin/python",
            runner=lambda command, timeout_s: _result(command),
        )

    def failed_pip(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        return _result(command, returncode=2, stderr="pip resolver failed")

    actions = exp2901.repair_thrml_import_if_needed(
        initial,
        project_root=tmp_path,
        python_executable=str(tmp_path / ".venv" / "bin" / "python"),
        runner=failed_pip,
    )

    assert actions[0]["status"] == "repair_install_failed"
    assert actions[0]["mutating_install_performed"] is True
    assert actions[0]["scope"] == "project_virtualenv"


def test_scenario_sample_096_missing_thrml_repairs_and_writes_success_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAMPLE-096: missing THRML is repaired before n=16 parity runs."""

    commands: list[list[str]] = []
    import_calls = 0

    def runner(command: list[str], timeout_s: float) -> CommandResult:
        nonlocal import_calls
        del timeout_s
        commands.append(command)
        code = command[-1]
        if "import jax" in code:
            return _result(command, stdout="0.10.0\n")
        if command[1:4] == ["-m", "pip", "--disable-pip-version-check"]:
            return _result(command, stdout="Successfully installed thrml-0.7.0\n")
        if "import thrml" in code:
            import_calls += 1
            if import_calls == 1:
                return _result(command, returncode=1, stderr=_traceback())
            return _result(command, stdout=_terminal_import_stdout())
        raise AssertionError(f"unexpected command: {command}")

    output_path = tmp_path / "results" / "experiment_2901_thrml_local_import_repair_v1.json"
    python_executable = tmp_path / ".venv" / "bin" / "python"

    artifact = exp2901.run_local_import_repair(
        output_path=output_path,
        project_root=tmp_path,
        python_executable=str(python_executable),
        runner=runner,
        parity_runner=lambda: _parity_metrics(0.0),
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    exp2901.validate_artifact(artifact)
    assert payload == artifact
    assert exp2901.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["honest_verdict"] == (
        "complete: thrml_import_repaired_n16_parity_passed_no_hardware_claim"
    )
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["thrml_import_succeeded"] is True
    assert artifact["thrml_version_installed"] == "0.7.0"
    assert artifact["jax_version"] == "0.10.0"
    assert artifact["parity_energy_delta"] == pytest.approx(0.0)
    assert artifact["random_seed"] == exp2901.DEFAULT_RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["metadata"]["initial_thrml_import_succeeded"] is False
    assert "ModuleNotFoundError" in artifact["metadata"]["initial_thrml_import_traceback"]
    assert artifact["metadata"]["parity_metrics"]["state_count"] == 65536
    assert any("-U" in command and "thrml" in command for command in commands)


def test_req_sample_096_n16_parity_runner_uses_thrml_energy_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLE-096: n=16 parity runner compares Carnot and THRML energies."""

    class FakeCase:
        n_spins = 4
        topology = "signed_ring_chord"
        j_matrix = np.zeros((4, 4), dtype=float)
        bias = np.zeros(4, dtype=float)
        beta = 1.0

    monkeypatch.setattr(
        exp2901.parity_n8,
        "_import_thrml",
        lambda importer: (("thrml", "models"), {"thrml_version": "fake"}, None),
    )
    monkeypatch.setattr(exp2901.parity_n16, "n16_signed_ring_chord_case", lambda: FakeCase())
    monkeypatch.setattr(
        exp2901.parity_n8,
        "_build_thrml_model",
        lambda thrml_modules, case: ("model", ["nodes"], "thrml"),
    )
    monkeypatch.setattr(
        exp2901.parity_n8,
        "_thrml_energy_for_state",
        lambda model, nodes, thrml_module, state: 0.0,
    )

    metrics = exp2901.run_n16_thrml_carnot_energy_parity(importer=lambda name: object())

    assert metrics["case_id"] == "exp2901:n16_signed_ring_chord:bounded_energy_smoke"
    assert metrics["n_spins"] == 4
    assert metrics["state_count"] == 4
    assert metrics["mean_energy_delta"] == pytest.approx(0.0)
    assert metrics["thrml_details"]["thrml_version"] == "fake"
    assert metrics["no_tsu_hardware_claim"] is True


def test_req_sample_096_n16_parity_runner_reports_import_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLE-096: THRML import blockers prevent fake parity."""

    monkeypatch.setattr(
        exp2901.parity_n8,
        "_import_thrml",
        lambda importer: (None, {}, {"blocker": "thrml_local_import_unavailable"}),
    )

    with pytest.raises(RuntimeError, match="thrml_local_import_unavailable"):
        exp2901.run_n16_thrml_carnot_energy_parity(importer=lambda name: object())


def test_req_sample_096_existing_thrml_skips_install_and_still_runs_parity(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLE-096: an already-importable THRML package is not reinstalled."""

    commands: list[list[str]] = []

    def runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        commands.append(command)
        code = command[-1]
        if "import jax" in code:
            return _result(command, stdout="0.10.0\n")
        if "import thrml" in code:
            return _result(command, stdout=_terminal_import_stdout("0.7.1"))
        raise AssertionError(f"unexpected command: {command}")

    artifact = exp2901.run_local_import_repair(
        output_path=None,
        project_root=tmp_path,
        python_executable=str(tmp_path / ".venv" / "bin" / "python"),
        runner=runner,
        parity_runner=lambda: _parity_metrics(0.001),
    )

    exp2901.validate_artifact(artifact)
    assert artifact["thrml_version_installed"] == "0.7.1"
    assert artifact["parity_energy_delta"] == pytest.approx(0.001)
    assert artifact["metadata"]["repair_actions"] == []
    assert not any("-m" in command and "pip" in command for command in commands)


def test_req_sample_096_terminal_import_failure_stops_before_artifact(tmp_path: Path) -> None:
    """REQ-SAMPLE-096: failed terminal import cannot produce a valid artifact."""

    import_calls = 0

    def runner(command: list[str], timeout_s: float) -> CommandResult:
        nonlocal import_calls
        del timeout_s
        code = command[-1]
        if "import jax" in code:
            return _result(command, stdout="0.10.0\n")
        if command[1:4] == ["-m", "pip", "--disable-pip-version-check"]:
            return _result(command)
        if "import thrml" in code:
            import_calls += 1
            return _result(command, returncode=1, stderr=_traceback())
        raise AssertionError(f"unexpected command: {command}")

    with pytest.raises(RuntimeError, match="THRML import still failed"):
        exp2901.run_local_import_repair(
            output_path=None,
            project_root=tmp_path,
            python_executable=str(tmp_path / ".venv" / "bin" / "python"),
            runner=runner,
            parity_runner=lambda: _parity_metrics(0.0),
        )

    assert import_calls == 2


def test_req_sample_096_validate_rejects_incomplete_or_dishonest_artifacts(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLE-096: validation enforces required fields and success gates."""

    initial = exp2901.ThrmlImportProbe(
        import_succeeded=False,
        version=None,
        import_path=None,
        traceback_text=_traceback(),
        command_result={"returncode": 1},
    )
    terminal = exp2901.ThrmlImportProbe(
        import_succeeded=True,
        version="0.7.0",
        import_path="/tmp/thrml/__init__.py",
        traceback_text="",
        command_result={"returncode": 0},
    )
    artifact = exp2901.build_artifact(
        project_root=tmp_path,
        jax_version="0.10.0",
        initial_import=initial,
        terminal_import=terminal,
        repair_actions=[{"status": "repair_install_succeeded"}],
        parity_metrics=_parity_metrics(0.0),
        duration_s=0.25,
    )

    exp2901.validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("jax_version")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp2901.validate_artifact(missing)

    wrong_substrate = dict(artifact, inference_substrate="offline_fixture")
    with pytest.raises(ValueError, match="inference_substrate"):
        exp2901.validate_artifact(wrong_substrate)

    not_imported = dict(artifact, thrml_import_succeeded=False)
    with pytest.raises(ValueError, match="thrml_import_succeeded"):
        exp2901.validate_artifact(not_imported)

    bad_delta = dict(artifact, parity_energy_delta=-1.0)
    with pytest.raises(ValueError, match="parity_energy_delta"):
        exp2901.validate_artifact(bad_delta)

    bad_checksum = dict(artifact, reproducibility_checksum="not-a-sha256")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp2901.validate_artifact(bad_checksum)

    mismatched_checksum = dict(artifact, reproducibility_checksum="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        exp2901.validate_artifact(mismatched_checksum)

    empty_thrml_version = dict(artifact, thrml_version_installed="")
    with pytest.raises(ValueError, match="thrml_version_installed"):
        exp2901.validate_artifact(empty_thrml_version)

    empty_jax_version = dict(artifact, jax_version="")
    with pytest.raises(ValueError, match="jax_version"):
        exp2901.validate_artifact(empty_jax_version)

    bad_seed = dict(artifact, random_seed="202605232901")
    with pytest.raises(ValueError, match="random_seed"):
        exp2901.validate_artifact(bad_seed)

    bad_duration = dict(artifact, duration_s=-0.1)
    with pytest.raises(ValueError, match="duration_s"):
        exp2901.validate_artifact(bad_duration)

    bad_principles = dict(artifact)
    bad_principles["metadata"] = {
        **artifact["metadata"],
        "field_principles": {
            **artifact["metadata"]["field_principles"],
            "no_hardware_acceleration_claim": False,
        },
    }
    with pytest.raises(ValueError, match="no-hardware-claim"):
        exp2901.validate_artifact(bad_principles)


def test_req_sample_096_checksum_excludes_runtime_duration(tmp_path: Path) -> None:
    """REQ-SAMPLE-096: checksum is stable for the same repair and parity evidence."""

    initial = exp2901.ThrmlImportProbe(
        import_succeeded=False,
        version=None,
        import_path=None,
        traceback_text=_traceback(),
        command_result={"returncode": 1},
    )
    terminal = exp2901.ThrmlImportProbe(
        import_succeeded=True,
        version="0.7.0",
        import_path="/tmp/thrml/__init__.py",
        traceback_text="",
        command_result={"returncode": 0},
    )

    first = exp2901.build_artifact(
        project_root=tmp_path,
        jax_version="0.10.0",
        initial_import=initial,
        terminal_import=terminal,
        repair_actions=[{"status": "repair_install_succeeded"}],
        parity_metrics=_parity_metrics(0.0),
        duration_s=0.1,
    )
    second = exp2901.build_artifact(
        project_root=tmp_path,
        jax_version="0.10.0",
        initial_import=initial,
        terminal_import=terminal,
        repair_actions=[{"status": "repair_install_succeeded"}],
        parity_metrics=_parity_metrics(0.0),
        duration_s=9.9,
    )

    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]


def test_req_sample_096_parity_delta_accepts_max_energy_fallback(tmp_path: Path) -> None:
    """REQ-SAMPLE-096: artifact accepts parity metrics named by max energy delta."""

    initial = exp2901.ThrmlImportProbe(
        import_succeeded=True,
        version="0.7.0",
        import_path="/tmp/thrml/__init__.py",
        traceback_text="",
        command_result={"returncode": 0},
    )

    artifact = exp2901.build_artifact(
        project_root=tmp_path,
        jax_version="0.10.0",
        initial_import=initial,
        terminal_import=initial,
        repair_actions=[],
        parity_metrics={"max_energy_abs_delta": 0.125},
        duration_s=0.2,
    )

    assert artifact["honest_verdict"] == (
        "complete: thrml_import_already_available_n16_parity_passed_no_hardware_claim"
    )
    assert artifact["parity_energy_delta"] == pytest.approx(0.125)
