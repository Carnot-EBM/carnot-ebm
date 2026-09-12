"""Tests for authenticated three-board continuity.

Spec: REQ-ISING-7231 and SCENARIO-ISING-7231-PREFLIGHT through
SCENARIO-ISING-7231-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import base64
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from carnot import experiment_7231_v636_board_continuity as experiment


ROOT = Path(__file__).resolve().parents[2]


def _transport(*, available: bool = True, exit_code: int = 0) -> dict[str, Any]:
    """SCENARIO-ISING-7231-POLARFIRE: make independent raw transport bytes."""

    if not available:
        text = "\n".join(
            (
                experiment.REMOTE_BEGIN,
                "availability=missing",
                f"input_sha256={experiment.sha256_bytes(experiment.POLARFIRE_INPUT_BYTES)}",
                experiment.REMOTE_END,
                "",
            )
        ).encode()
        return {
            "command": list(experiment.POLARFIRE_COMMAND),
            "exit_code": 3,
            "stdout": text,
            "stderr": b"",
            "timed_out": False,
            "transport_duration_s": 0.25,
        }
    output = b"carnot deployed workload help\n"
    binary_hash = "1" * 64
    lines = (
        experiment.REMOTE_BEGIN,
        "availability=available",
        f"binary_sha256={binary_hash}",
        f"input_sha256={experiment.sha256_bytes(experiment.POLARFIRE_INPUT_BYTES)}",
        f"workload_exit_code={exit_code}",
        f"output_sha256={experiment.sha256_bytes(output)}",
        f"output_bytes={len(output)}",
        f"output_base64={base64.b64encode(output).decode()}",
        "elapsed_ns=12345",
        experiment.REMOTE_END,
        "",
    )
    return {
        "command": list(experiment.POLARFIRE_COMMAND),
        "exit_code": exit_code,
        "stdout": "\n".join(lines).encode(),
        "stderr": b"transport stderr\n",
        "timed_out": False,
        "transport_duration_s": 0.5,
    }


def test_req_ising_7231_preconditions_authenticate_exact_upstreams(tmp_path: Path) -> None:
    """REQ-ISING-7231: authenticate gates before reading board evidence."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    checks, hashes, upstreams = experiment.collect_preconditions(ROOT, paths)
    by_name = {row["check"]: row for row in checks}
    assert all(row["passed"] for row in checks)
    assert by_name["exp7217_producer_authentication"]["observed_value"] == []
    assert by_name["exp7217_board_gate"]["observed_value"]["score"] == 1
    assert by_name["exp7226_producer_authentication"]["observed_value"] == []
    assert by_name["exp7226_controller_gate"]["observed_value"] == {
        "belief_compiler_ready_score": 1,
        "cpu_packed_kernel_bytes": 152,
        "fpga_table_bits": 1216,
    }
    assert set(upstreams) == {"board", "controller"}
    assert set(hashes) == {path.as_posix() for path in experiment.REQUIRED_SOURCE_PATHS}


def test_scenario_ising_7231_preflight_rejects_quarantine_before_gate(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7231-PREFLIGHT: quarantine prevents gate consumption."""

    upstream = json.loads((ROOT / experiment.BOARD_UPSTREAM_PATH).read_text(encoding="utf-8"))
    upstream["flagged_adversarial"] = True
    path = tmp_path / "quarantined.json"
    path.write_text(json.dumps(upstream), encoding="utf-8")
    checks, _, _ = experiment.collect_preconditions(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path / "outputs"),
        board_upstream_path=path,
    )
    by_name = {row["check"]: row for row in checks}
    assert by_name["exp7217_not_quarantined"]["passed"] is False
    assert by_name["exp7217_board_gate"]["observed_value"] == ("not_consumed_due_to_quarantine")
    exact = {"principle": "why", "value": 1}
    extra = {"principle": "why", "value": 1, "evidence": "untrusted"}
    assert experiment.unwrap_principled_value(exact) == 1
    assert experiment.unwrap_principled_value(extra) is extra


def test_scenario_ising_7231_boards_preserve_exact_terminal_states() -> None:
    """SCENARIO-ISING-7231-BOARDS: select and authenticate each board row."""

    upstream = json.loads((ROOT / experiment.BOARD_UPSTREAM_PATH).read_text(encoding="utf-8"))
    rows, operator = experiment.load_latest_board_rows(ROOT, upstream)
    by_board = {row["board"]: row for row in rows}
    assert set(by_board) == {"KV260", "GateMate", "PolarFire"}
    assert by_board["KV260"]["terminal_criterion"] == (
        "board-level programmable-logic latency transcript and successful KV260 synthesis"
    )
    assert by_board["KV260"]["disposition"] == "graduated_preserved"
    assert by_board["KV260"]["processor_class"] == "fpga_fabric"
    assert by_board["GateMate"]["disposition"] == ("blocked_inherited_no_new_physical_state")
    assert by_board["GateMate"]["hardware_command_count"] == 0
    assert operator["newer_than_exp6559"] is False
    assert operator["hardware_operations_issued"] == []
    assert by_board["PolarFire"]["processor_class"] == "cpu"
    assert by_board["PolarFire"]["programmable_logic_sampling_observed"] is False
    assert all(row["latest_receipt_authenticated"] for row in rows)


def test_scenario_ising_7231_polarfire_captures_and_verifies_raw_bytes(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7231-POLARFIRE: retain exact bounded dispatch evidence."""

    seen: list[tuple[tuple[str, ...], float]] = []

    def runner(command: tuple[str, ...], timeout_s: float) -> dict[str, Any]:
        seen.append((command, timeout_s))
        return _transport()

    receipt = experiment.run_polarfire_smoke(tmp_path / "raw.json", command_runner=runner)
    assert seen == [(experiment.POLARFIRE_COMMAND, experiment.LOCAL_TIMEOUT_S)]
    assert receipt["dispatch_completed"] is True
    assert receipt["binary_sha256"] == "sha256:" + "1" * 64
    assert receipt["input_hash_matches"] is True
    assert receipt["output_hash_matches"] is True
    assert receipt["processor_class"] == "cpu"
    assert receipt["programmable_logic_sampling_observed"] is False
    raw = json.loads((tmp_path / "raw.json").read_text(encoding="utf-8"))
    assert base64.b64decode(raw["transport_stdout_base64"]) == _transport()["stdout"]
    assert base64.b64decode(raw["workload_output_base64"]) == (b"carnot deployed workload help\n")
    assert raw["transport_stderr"] == "transport stderr\n"


@pytest.mark.parametrize(
    ("transport", "reason"),
    (
        (_transport(available=False), "missing_deployed_workload"),
        (_transport(exit_code=2), "deployed_workload_failed"),
        (
            {
                **_transport(),
                "stdout": b"not a structured receipt",
            },
            "invalid_dispatch_transcript",
        ),
        (
            {
                **_transport(),
                "exit_code": 124,
                "timed_out": True,
            },
            "transport_timeout",
        ),
    ),
)
def test_scenario_ising_7231_polarfire_blocks_without_inventing_results(
    tmp_path: Path,
    transport: dict[str, Any],
    reason: str,
) -> None:
    """SCENARIO-ISING-7231-POLARFIRE: missing evidence stays a board block."""

    receipt = experiment.run_polarfire_smoke(
        tmp_path / f"{reason}.json",
        command_runner=lambda _command, _timeout: transport,
    )
    assert receipt["dispatch_completed"] is False
    assert receipt["block_reason"] == reason
    assert receipt["programmable_logic_sampling_observed"] is False


def test_scenario_ising_7231_placement_uses_current_compact_contract() -> None:
    """SCENARIO-ISING-7231-PLACEMENT: retain footprint and unknown device data."""

    controller = json.loads(
        (ROOT / experiment.CONTROLLER_UPSTREAM_PATH).read_text(encoding="utf-8")
    )
    mapping = experiment.operation_map(controller, polarfire_executed=True)
    by_target = {row["target"]: row for row in mapping}
    assert by_target["host_cpu"]["memory_table_bytes"] == 152
    assert by_target["polarfire_cpu"]["executed_here"] is True
    assert by_target["fpga_fabric"]["memory_table_bits"] == 1216
    assert by_target["fpga_fabric"]["topology_fit"] == "unknown"
    assert by_target["tsu"]["power"] == "unknown"
    assert all(row["speed"] == "unknown" for row in mapping)


def test_scenario_ising_7231_artifact_complete_with_remote_cpu_smoke(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7231-ARTIFACT: three dispositions complete the receipt."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.build_artifact(
        ROOT,
        paths,
        command_runner=lambda _command, _timeout: _transport(),
    )
    assert experiment.validate_artifact(artifact) == []
    assert artifact["board_continuity_complete_score"] == 1
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert len(artifact["hardware_operations_issued"]) == 1
    assert artifact["raw_dispatch_transcript_path"] == str(paths.raw_transcript)
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_ising_7231_artifact_complete_with_blocked_board(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7231-ARTIFACT: a board block does not erase dispositions."""

    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        command_runner=lambda _command, _timeout: _transport(available=False),
    )
    assert experiment.validate_artifact(artifact) == []
    assert artifact["board_continuity_complete_score"] == 1
    assert artifact["inference_substrate_class"] == "aggregation"
    polarfire = next(row for row in artifact["board_rows"] if row["board"] == "PolarFire")
    assert polarfire["disposition"] == "blocked_missing_deployed_workload"
    assert polarfire["abstention"] is True
    assert polarfire["hardware_command_count"] == 1


def test_scenario_ising_7231_preflight_block_has_exact_gate_and_no_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ISING-7231-PREFLIGHT: repository blocks stop board access."""

    failed = experiment.check("missing", "repository", "source", True, False, False)
    monkeypatch.setattr(
        experiment,
        "collect_preconditions",
        lambda *_args, **_kwargs: ([failed], {}, {}),
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("board command must not run")

    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        command_runner=forbidden,
    )
    assert experiment.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked_external_precondition"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["hardware_operations_issued"] == []
    assert artifact["raw_dispatch_transcript_path"] is None
    assert artifact["gate_check_summary"]["failed_check"] == "missing"


def test_req_ising_7231_validator_recomputes_claim_boundaries(tmp_path: Path) -> None:
    """REQ-ISING-7231: mutations cannot self-report a complete safe receipt."""

    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        command_runner=lambda _command, _timeout: _transport(),
    )
    for mutation, expected in (
        ({"board_continuity_complete_score": 0}, "completion_score"),
        ({"execution_venue": "polarfire"}, "execution_identity"),
        ({"MODEL_SPECS": [{}]}, "model_declaration"),
        ({"topology_fit": "fits"}, "hardware_claim_boundary"),
    ):
        changed = deepcopy(artifact)
        changed.update(mutation)
        changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
        assert expected in experiment.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["board_rows"][0]["terminal_criterion"] = "weaker criterion"
    changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    assert "board_rows" in experiment.validate_artifact(changed)


def test_req_ising_7231_atomic_write_subprocess_and_cli_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ISING-7231: writes are atomic and subprocesses have bounded outcomes."""

    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path / "build"),
        command_runner=lambda _command, _timeout: _transport(available=False),
    )
    output = tmp_path / "artifact.json"
    receipt = experiment.atomic_write(output, artifact)
    assert receipt["atomic_replace"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert experiment.main(["--validate", str(output)]) == 0
    assert "validation_passed" in capsys.readouterr().out
    assert experiment.main(["--date", "20260911", "--output", str(output)]) == 2
    assert "run date must be 20260912" in capsys.readouterr().out

    success = experiment._run_subprocess(
        (sys.executable, "-c", "print('ok')"),
        timeout_s=2,
        heartbeat_interval_s=0.001,
    )
    assert success["exit_code"] == 0
    assert success["stdout"] == b"ok\n"
    timeout = experiment._run_subprocess(
        (sys.executable, "-c", "import time; time.sleep(1)"),
        timeout_s=0.01,
        heartbeat_interval_s=0.001,
    )
    assert timeout["timed_out"] is True

    monkeypatch.setattr(experiment, "run_experiment", lambda *_args, **_kwargs: artifact)
    assert experiment.main(["--output", str(output)]) == 0
    assert "experiment_complete" in capsys.readouterr().out
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    assert experiment.main(["--validate", str(invalid)]) == 2


def test_req_ising_7231_entrypoint_is_thin_and_has_no_other_board_tools() -> None:
    """REQ-ISING-7231: the executable delegates to the production module."""

    path = ROOT / "scripts/experiments/experiment_7231_v636_board_continuity.py"
    source = path.read_text(encoding="utf-8")
    assert "experiment_7231_v636_board_continuity import main" in source
    assert "openFPGALoader" not in source
    assert "/dev/mmc" not in source


def test_scenario_ising_7231_preflight_defensive_input_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ISING-7231-PREFLIGHT: malformed local inputs fail closed."""

    assert experiment.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert experiment._read_json(missing) == {}
    assert experiment._read_json(invalid) == {}
    assert experiment._read_json(scalar) == {}

    assert experiment._task_contract(tmp_path) is None
    (tmp_path / experiment.ROADMAP_PATH).write_text("[", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    (tmp_path / experiment.ROADMAP_PATH).write_text("tasks: scalar\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    (tmp_path / experiment.ROADMAP_PATH).write_text("tasks: []\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None

    monkeypatch.setattr(
        experiment.tempfile,
        "mkstemp",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("unwritable")),
    )
    assert experiment._writable_destination(tmp_path / "output.json") is False


def test_scenario_ising_7231_preflight_missing_manifest_is_observed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ISING-7231-PREFLIGHT: an unreadable exclusion file cannot authorize."""

    roadmap = experiment.yaml.safe_load((ROOT / experiment.ROADMAP_PATH).read_text())
    calls = iter((roadmap, experiment.yaml.YAMLError("missing manifest")))

    def safe_load(_text: str) -> Any:
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(experiment.yaml, "safe_load", safe_load)
    checks, _, _ = experiment.collect_preconditions(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
    )
    assert all(row["passed"] for row in checks)


def test_scenario_ising_7231_boards_reject_malformed_authenticated_rows() -> None:
    """SCENARIO-ISING-7231-BOARDS: malformed latest receipts cannot become rows."""

    upstream = json.loads((ROOT / experiment.BOARD_UPSTREAM_PATH).read_text(encoding="utf-8"))
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({**upstream, "board_rows": {}}, "must be a list"),
        ({**upstream, "board_rows": [7]}, "row is malformed"),
        (
            {
                **upstream,
                "board_rows": [{**upstream["board_rows"][0], "source_hash": "sha256:bad"}],
            },
            "evidence hash mismatch",
        ),
        ({**upstream, "board_rows": upstream["board_rows"][:2]}, "exactly three"),
        (
            {
                **upstream,
                "board_rows": [
                    {**upstream["board_rows"][0], "disposition": "changed"},
                    *upstream["board_rows"][1:],
                ],
            },
            "disposition changed",
        ),
        ({**upstream, "operator_state_receipt": []}, "operator_state_receipt"),
    )
    for payload, message in cases:
        with pytest.raises(ValueError, match=message):
            experiment.load_latest_board_rows(ROOT, payload)


def test_scenario_ising_7231_polarfire_parser_rejects_bad_hashes_and_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ISING-7231-POLARFIRE: malformed hashes and encodings stay blocked."""

    assert experiment._tag_hash(None) is None
    assert experiment._tag_hash("not-a-hash") is None
    malformed = f"{experiment.REMOTE_BEGIN}\nno-separator\n{experiment.REMOTE_END}\n"
    assert experiment._remote_fields(malformed.encode()) == {}

    transport = _transport()
    transport["stdout"] = transport["stdout"].replace(
        b"output_base64=Y2Fybm90IGRlcGxveWVkIHdvcmtsb2FkIGhlbHAK",
        b"output_base64=%%%",
    )
    receipt = experiment.run_polarfire_smoke(
        tmp_path / "invalid-b64.json",
        command_runner=lambda _command, _timeout: transport,
    )
    assert receipt["block_reason"] == "invalid_dispatch_transcript"

    monkeypatch.setattr(
        experiment.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )
    target = tmp_path / "replace-failure.json"
    with pytest.raises(OSError, match="replace failed"):
        experiment._atomic_json(target, {"value": 1})
    assert not list(tmp_path.glob(".replace-failure.json.*.tmp"))


def test_scenario_ising_7231_gatemate_new_receipt_only_names_later_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ISING-7231-GATEMATE: new state still issues no GateMate command."""

    board = json.loads((ROOT / experiment.BOARD_UPSTREAM_PATH).read_text(encoding="utf-8"))
    board["operator_state_receipt"] = {
        **board["operator_state_receipt"],
        "newer_than_exp6559": True,
    }
    controller = json.loads(
        (ROOT / experiment.CONTROLLER_UPSTREAM_PATH).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        experiment,
        "collect_preconditions",
        lambda *_args, **_kwargs: (
            [experiment.check("ok", "fixture", "field", True, True, True)],
            {},
            {"board": board, "controller": controller},
        ),
    )
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        command_runner=lambda _command, _timeout: _transport(available=False),
    )
    row = next(item for item in artifact["board_rows"] if item["board"] == "GateMate")
    assert row["disposition"] == "authorized_later_action"
    assert row["hardware_operations_issued"] == []
    assert row["hardware_command_count"] == 0
    assert experiment.validate_artifact(artifact) == []


def test_req_ising_7231_validator_failure_and_publish_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ISING-7231: validation covers malformed and source-bound publication."""

    assert experiment.validate_artifact({})[0].startswith("missing_fields:")
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path / "build"),
        command_runner=lambda _command, _timeout: _transport(available=False),
    )
    changed = deepcopy(artifact)
    changed["started_at_utc"] = "not-a-time"
    changed["completed_at_utc"] = "not-a-time"
    changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    errors = experiment.validate_artifact(changed)
    assert "started_at_utc" in errors
    assert "completed_at_utc" in errors

    changed = deepcopy(artifact)
    changed["random_seed"] = {1}
    assert "reproducibility_checksum" in experiment.validate_artifact(changed)
    assert experiment.validate_artifact(artifact, root=ROOT) == []

    changed = deepcopy(artifact)
    changed["status"] = "running"
    changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    with pytest.raises(ValueError, match="invalid Exp7231 artifact"):
        experiment.atomic_write(tmp_path / "invalid.json", changed)

    paths = experiment.ExperimentPaths.under(tmp_path / "publish")
    paths.raw_transcript.parent.mkdir(parents=True)
    paths.raw_transcript.write_text("{}\n", encoding="utf-8")
    publish = deepcopy(artifact)
    publish["source_artifact_hashes"] = {
        str(paths.raw_transcript): experiment.sha256_file(paths.raw_transcript)
    }
    publish["reproducibility_checksum"] = experiment.artifact_checksum(publish)
    monkeypatch.setattr(experiment, "build_artifact", lambda *_args, **_kwargs: publish)
    result = experiment.run_experiment(ROOT, paths)
    assert result == publish
    assert paths.artifact.is_file()

    monkeypatch.setattr(experiment, "validate_artifact", lambda *_args, **_kwargs: ["bad"])
    with pytest.raises(ValueError, match="invalid Exp7231 artifact"):
        experiment.run_experiment(ROOT, paths)
