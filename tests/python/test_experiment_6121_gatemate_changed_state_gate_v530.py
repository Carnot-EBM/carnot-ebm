"""Tests for Exp6121 GateMate changed-state gate.

Spec refs: REQ-HW-6121, SCENARIO-HW-6121.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "hardware" / "spec.md"
MODULE_PATH = REPO_ROOT / "python" / "carnot" / "experiment_6121_gatemate_changed_state_gate_v530.py"
SPEC = importlib.util.spec_from_file_location(
    "experiment_6121_gatemate_changed_state_gate_v530",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


class RecordingRunner:
    """REQ-HW-6121 command runner that records every attempted hardware command."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]] | None = None) -> None:
        self.probes = {command: list(values) for command, values in (probes or {}).items()}
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        assert timeout_s > 0.0
        command = tuple(command)
        self.calls.append(command)
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


class StepClock:
    """Deterministic clock that gives stable artifact duration fields."""

    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def _probe(
    command: tuple[str, ...],
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.012,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _changed_receipt() -> dict:
    receipt = copy.deepcopy(mod.DEFAULT_DATED_OPERATOR_PHYSICAL_RECEIPT)
    receipt.update(
        {
            "receipt_date": "20260804",
            "source": "unit-test dated operator physical receipt",
            "cable": "reseated USB-C cable on a different host root port",
            "port": "host root port changed from cached 3-2.3 to test-root-port-7",
            "power": "power LED visually confirmed after reseat",
            "changes": [
                {
                    "field": "cable",
                    "before": mod.LAST_ATTEMPT_PHYSICAL_STATE["cable"],
                    "after": "reseated USB-C cable on a different host root port",
                },
                {
                    "field": "port",
                    "before": mod.LAST_ATTEMPT_PHYSICAL_STATE["port"],
                    "after": "host root port changed from cached 3-2.3 to test-root-port-7",
                },
                {
                    "field": "power",
                    "before": mod.LAST_ATTEMPT_PHYSICAL_STATE["power"],
                    "after": "power LED visually confirmed after reseat",
                },
            ],
        }
    )
    return receipt


def _detect_miss_stdout() -> str:
    return "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\nfound 0 devices\n"


def _detect_hit_stdout() -> str:
    return (
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
        "index 0:\n"
        "\tidcode 0x20000001\n"
        "\tmanufacturer colognechip\n"
        "\tfamily GateMate Series\n"
        "\tmodel  GM1Ax\n"
    )


def _smoke_stdout() -> str:
    return json.dumps(
        {
            "prebuilt_smoke_sha256": mod.PREBUILT_SMOKE_EXPECTED_HASH,
            "read_only": True,
            "host_io_observed": True,
        },
        sort_keys=True,
    )


def test_spec_defines_req_and_scenario() -> None:
    """REQ-HW-6121: OpenSpec anchors the changed-state gate before code."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert "### REQ-HW-6121" in text
    assert "### SCENARIO-HW-6121" in text
    assert "experiment_6121_gatemate_changed_state_gate_v530.json" in text


def test_unchanged_state_runs_no_jtag_and_emits_exact_packet() -> None:
    """SCENARIO-HW-6121: unchanged physical state blocks without a detect command."""
    runner = RecordingRunner()
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(10.0, 11.25),
        run_date="20260804",
        protected_before_hashes=mod.protected_file_hashes(REPO_ROOT),
    )

    assert runner.calls == []
    assert artifact["status"] == "blocked_physical_action"
    assert artifact["physical_state_changed"] is False
    assert artifact["prior_and_current_physical_state_hashes"]["prior"] == (
        artifact["prior_and_current_physical_state_hashes"]["current"]
    )
    detect = artifact["detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code"]
    assert detect["allowed"] is False
    assert detect["attempt_count"] == 0
    assert detect["command"] is None
    assert detect["stdout"] == ""
    assert detect["stderr"] == ""
    assert artifact["operator_action_packet"] == mod.EXACT_OPERATOR_ACTION_PACKET
    assert artifact["retirement_triggered"] is True
    assert artifact["hardware_execution_authenticated"]["authenticated"] is False
    assert artifact["speed_power_and_terminal_claim_counts"] == mod.ZERO_CLAIM_COUNTS
    assert artifact["flash_synthesis_place_route_pack_and_firmware_mutation_counts"] == (
        mod.ZERO_MUTATION_COUNTS
    )
    assert artifact["honest_verdict"].startswith("blocked_physical_action:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert "scripts/research_conductor.py" in artifact["protected_files_unchanged"]["after_hashes"]
    mod.validate_artifact(artifact)


def test_preconditions_hash_required_receipt_categories_without_hardware() -> None:
    """REQ-HW-6121: preconditions hash artifacts, receipts, output path, and worktree."""
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=RecordingRunner(),
        clock=StepClock(1.0, 1.5),
        run_date="20260804",
    )

    preconditions = artifact["preconditions_checked"]
    required = {
        "prior_board_artifacts",
        "cable_port_power_descriptions",
        "usb_dirtyjtag_descriptors",
        "tool_versions",
        "operator_receipts",
        "bitstream_hashes",
        "output_paths",
        "protected_files",
        "dirty_worktree",
    }
    assert required <= set(preconditions)
    assert preconditions["prior_board_artifacts"]["all_hashed"] is True
    assert "results/experiment_5201_hardware_continuity_gatemate_diagnostic_v476.json" in (
        preconditions["prior_board_artifacts"]["hashes"]
    )
    assert "results/experiment_3866_gatemate_ising_tile_flash_v2.json" in (
        preconditions["prior_board_artifacts"]["hashes"]
    )
    assert preconditions["dirty_worktree"]["output_path_excluded_from_hash"] == (
        mod.OUTPUT_REL_PATH.as_posix()
    )


def test_changed_state_runs_exactly_one_allowlisted_non_destructive_detect() -> None:
    """REQ-HW-6121: changed physical receipt permits one detect command."""
    runner = RecordingRunner(
        {mod.DETECT_COMMAND: [_probe(mod.DETECT_COMMAND, stdout=_detect_miss_stdout())]}
    )
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(20.0, 20.75),
        run_date="20260804",
        dated_operator_physical_receipt=_changed_receipt(),
    )

    assert runner.calls == [mod.DETECT_COMMAND]
    assert artifact["physical_state_changed"] is True
    assert artifact["retirement_triggered"] is False
    detect = artifact["detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code"]
    assert detect["allowed"] is True
    assert detect["attempt_count"] == 1
    assert detect["command"] == "openFPGALoader -c dirtyJtag --detect"
    assert detect["exit_code"] == 0
    assert "found 0 devices" in detect["stdout"]
    assert artifact["expected_and_observed_idcode"]["expected_idcode"] == "0x20000001"
    assert artifact["expected_and_observed_idcode"]["observed_idcode"] is None
    assert artifact["prebuilt_bitstream_and_smoke_hashes"]["smoke_attempt"]["attempted"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_wrong_idcode_prevents_prebuilt_smoke_even_when_command_is_configured() -> None:
    """REQ-HW-6121: prebuilt smoke is gated by expected GateMate IDCODE."""
    runner = RecordingRunner(
        {
            mod.DETECT_COMMAND: [
                _probe(mod.DETECT_COMMAND, stdout="index 0:\n\tidcode 0x12345678\n")
            ]
        }
    )
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(30.0, 31.0),
        run_date="20260804",
        dated_operator_physical_receipt=_changed_receipt(),
        prebuilt_smoke_command=mod.PREBUILT_READ_ONLY_SMOKE_COMMAND,
    )

    assert runner.calls == [mod.DETECT_COMMAND]
    assert artifact["expected_and_observed_idcode"]["observed_idcode"] == "0x12345678"
    smoke = artifact["prebuilt_bitstream_and_smoke_hashes"]["smoke_attempt"]
    assert smoke["attempted"] is False
    assert smoke["reason"] == "expected_idcode_not_observed"
    assert artifact["hardware_execution_authenticated"]["authenticated"] is False


def test_expected_idcode_allows_only_matching_prebuilt_read_only_smoke() -> None:
    """REQ-HW-6121: expected IDCODE plus matching hash permits one read-only smoke."""
    runner = RecordingRunner(
        {
            mod.DETECT_COMMAND: [_probe(mod.DETECT_COMMAND, stdout=_detect_hit_stdout())],
            mod.PREBUILT_READ_ONLY_SMOKE_COMMAND: [
                _probe(mod.PREBUILT_READ_ONLY_SMOKE_COMMAND, stdout=_smoke_stdout())
            ],
        }
    )
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(40.0, 41.0),
        run_date="20260804",
        dated_operator_physical_receipt=_changed_receipt(),
        prebuilt_smoke_command=mod.PREBUILT_READ_ONLY_SMOKE_COMMAND,
    )

    assert runner.calls == [mod.DETECT_COMMAND, mod.PREBUILT_READ_ONLY_SMOKE_COMMAND]
    assert artifact["expected_and_observed_idcode"]["observed_idcode"] == "0x20000001"
    smoke = artifact["prebuilt_bitstream_and_smoke_hashes"]["smoke_attempt"]
    assert smoke["attempted"] is True
    assert smoke["exit_code"] == 0
    assert smoke["hash_matches_prior_receipt"] is True
    assert smoke["read_only"] is True
    assert artifact["hardware_execution_authenticated"]["authenticated"] is True
    assert artifact["honest_verdict"].startswith("complete_changed_state:")
    mod.validate_artifact(artifact)


def test_expected_idcode_without_configured_smoke_stays_blocked() -> None:
    """REQ-HW-6121: an IDCODE alone is not host-I/O smoke evidence."""
    runner = RecordingRunner(
        {mod.DETECT_COMMAND: [_probe(mod.DETECT_COMMAND, stdout=_detect_hit_stdout())]}
    )
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(42.0, 43.0),
        run_date="20260804",
        dated_operator_physical_receipt=_changed_receipt(),
    )

    assert runner.calls == [mod.DETECT_COMMAND]
    assert artifact["status"] == "blocked"
    assert artifact["prebuilt_bitstream_and_smoke_hashes"]["smoke_attempt"]["reason"] == (
        "no_prebuilt_smoke_command_configured"
    )
    assert artifact["honest_verdict"].startswith(
        "blocked: expected GateMate IDCODE observed but"
    )


def test_helper_fallbacks_hash_missing_paths_and_failed_git(tmp_path: Path) -> None:
    """REQ-HW-6121: helper fallbacks are explicit instead of fabricating evidence."""
    probe = _probe(mod.DETECT_COMMAND, stdout="ok\n")
    assert probe.as_dict()["command"] == "openFPGALoader -c dirtyJtag --detect"
    assert mod.run_command(("printf", "ok"), 5.0).stdout == "ok"
    assert mod.path_receipt(tmp_path, "missing.json") == {
        "path": "missing.json",
        "present": False,
        "bytes": 0,
        "sha256": None,
    }
    assert mod.read_json_if_present(tmp_path, "missing.json") == {}
    assert mod._find_line_with("alpha\nbeta\n", "gamma") == ""
    dirty = mod.dirty_worktree_receipt(tmp_path)
    assert dirty["status_porcelain_sha256"].startswith("sha256:")
    assert dirty["tracked_diff_sha256"].startswith("sha256:")


def test_invalid_smoke_stdout_does_not_authenticate() -> None:
    """REQ-HW-6121: malformed read-only smoke output is not authenticated."""
    runner = RecordingRunner(
        {
            mod.DETECT_COMMAND: [_probe(mod.DETECT_COMMAND, stdout=_detect_hit_stdout())],
            mod.PREBUILT_READ_ONLY_SMOKE_COMMAND: [
                _probe(mod.PREBUILT_READ_ONLY_SMOKE_COMMAND, stdout="not-json")
            ],
        }
    )
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(44.0, 45.0),
        run_date="20260804",
        dated_operator_physical_receipt=_changed_receipt(),
        prebuilt_smoke_command=mod.PREBUILT_READ_ONLY_SMOKE_COMMAND,
    )

    smoke = artifact["prebuilt_bitstream_and_smoke_hashes"]["smoke_attempt"]
    assert smoke["attempted"] is True
    assert smoke["hash_matches_prior_receipt"] is False
    assert artifact["hardware_execution_authenticated"]["authenticated"] is False


def test_schema_rejects_mutation_counts_bad_attempt_count_and_disallowed_command() -> None:
    """REQ-HW-6121: schema rejects flash/mutation, rerun, and command allowlist breaks."""
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=RecordingRunner(),
        clock=StepClock(2.0, 2.5),
        run_date="20260804",
    )

    mutated = copy.deepcopy(artifact)
    mutated["flash_synthesis_place_route_pack_and_firmware_mutation_counts"]["flash"] = 1
    assert any("mutation count" in err for err in mod.artifact_schema_errors(mutated))
    with pytest.raises(ValueError, match="mutation count"):
        mod.validate_artifact(mutated)

    rerun = copy.deepcopy(artifact)
    rerun["physical_state_changed"] = True
    rerun["detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code"][
        "attempt_count"
    ] = 2
    assert any("at most one detect" in err for err in mod.artifact_schema_errors(rerun))

    bad_command = copy.deepcopy(artifact)
    detect = bad_command["detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code"]
    detect.update({"allowed": True, "attempt_count": 1, "command": "openFPGALoader --flash x"})
    assert any("detect command allowlist" in err for err in mod.artifact_schema_errors(bad_command))


def test_schema_defensive_errors_cover_required_contracts() -> None:
    """REQ-HW-6121: defensive schema checks reject malformed receipts."""
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=RecordingRunner(),
        clock=StepClock(3.0, 3.5),
        run_date="20260804",
    )

    missing = copy.deepcopy(artifact)
    del missing["status"]
    assert any("missing required fields" in err for err in mod.artifact_schema_errors(missing))

    schema = copy.deepcopy(artifact)
    schema["schema"] = "wrong"
    schema["spec_refs"] = []
    schema["random_seed"] = 0
    schema["inference_substrate"] = "wrong"
    errors = mod.artifact_schema_errors(schema)
    assert "schema mismatch" in errors
    assert "spec_refs mismatch" in errors
    assert "random_seed mismatch" in errors
    assert "inference_substrate mismatch" in errors

    detect_not_mapping = copy.deepcopy(artifact)
    detect_not_mapping[
        "detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code"
    ] = []
    assert any("detect receipt" in err for err in mod.artifact_schema_errors(detect_not_mapping))

    packet = copy.deepcopy(artifact)
    packet["operator_action_packet"] = {}
    assert any("exact operator action packet" in err for err in mod.artifact_schema_errors(packet))

    claims = copy.deepcopy(artifact)
    claims["speed_power_and_terminal_claim_counts"]["speedup"] = 1
    assert any("claim counts" in err for err in mod.artifact_schema_errors(claims))

    smoke = copy.deepcopy(artifact)
    smoke["prebuilt_bitstream_and_smoke_hashes"]["smoke_attempt"]["attempted"] = True
    assert any("prebuilt smoke attempted" in err for err in mod.artifact_schema_errors(smoke))

    auth = copy.deepcopy(artifact)
    auth["hardware_execution_authenticated"]["authenticated"] = True
    assert any("hardware execution authenticated" in err for err in mod.artifact_schema_errors(auth))

    protected = copy.deepcopy(artifact)
    protected["protected_files_unchanged"]["all_unchanged"] = False
    assert any("protected files" in err for err in mod.artifact_schema_errors(protected))

    verdict = copy.deepcopy(artifact)
    verdict["honest_verdict"] = "success: wrong-prefix"
    assert any("honest_verdict prefix" in err for err in mod.artifact_schema_errors(verdict))


def test_run_experiment_writes_artifact_to_requested_output_root(tmp_path: Path) -> None:
    """SCENARIO-HW-6121: run_experiment writes the deliverable JSON without hardware."""
    out = mod.run_experiment(
        repo_root=tmp_path,
        source_root=REPO_ROOT,
        command_runner=RecordingRunner(),
        clock=StepClock(50.0, 50.5),
        run_date="20260804",
    )
    assert out == tmp_path / mod.OUTPUT_REL_PATH
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["status"] == "blocked_physical_action"
    assert artifact["detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code"][
        "attempt_count"
    ] == 0
    mod.validate_artifact(artifact)


def test_main_prints_changed_state_gate_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    """REQ-HW-6121: CLI reports the artifact path and terminal verdict."""

    real_run_experiment = mod.run_experiment

    def fake_run_experiment(*, repo_root: Path, run_date: str) -> Path:
        return real_run_experiment(
            repo_root=tmp_path,
            source_root=REPO_ROOT,
            command_runner=RecordingRunner(),
            clock=StepClock(60.0, 60.5),
            run_date=run_date,
        )

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    rc = mod.main(["--date", "20260804", "--repo-root", str(tmp_path)])
    captured = capsys.readouterr().out

    assert rc == 0
    assert "experiment_6121_gatemate_changed_state_gate_v530.json" in captured
    assert "physical_state_changed: False" in captured
    assert "detect_attempt_count: 0" in captured
