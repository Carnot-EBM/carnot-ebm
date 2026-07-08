"""Tests for Exp5424 comparable hardware timing receipts.

Spec refs: REQ-HW-5424, SCENARIO-HW-5424.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5424_hardware_comparable_timing_receipts_v493 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5424_hardware_comparable_timing_receipts_v493.py "
    "-q --no-cov"
)


class RecordingRunner:
    """SCENARIO-HW-5424 runner with exact command expectations."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        assert timeout_s > 0.0
        command = tuple(command)
        self.commands.append(command)
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


class StepClock:
    """Deterministic clock for repeat timing assertions."""

    def __init__(self) -> None:
        self.value = 5424.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


def _probe(
    command: tuple[str, ...],
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _tests_run() -> list[dict[str, object]]:
    return [{"command": TEST_COMMAND, "outcome": "passed"}]


def _board_stdout(
    workload: dict[str, object],
    *,
    workload_hash: str | None = None,
    result_hash: str | None = None,
    wall_time_s: float = 0.123,
) -> str:
    payload = {
        "board_local": True,
        "exact_best_sequence": workload["exact_result"]["exact_best_sequence"],
        "exact_min_energy": workload["exact_result"]["exact_min_energy"],
        "exact_result_hash": result_hash or workload["exact_result_hash"],
        "seed": mod.WORKLOAD_SEED,
        "wall_time_s": wall_time_s,
        "workload_hash": workload_hash or workload["workload_hash"],
    }
    return json.dumps(payload, sort_keys=True) + "\n"


def _runner(
    *,
    polarfire_status_exit: int = 0,
    board_stdout: list[str] | None = None,
    gatemate_exit: int = 0,
) -> RecordingRunner:
    workload = mod.select_workload(REPO)
    board_command = mod.polarfire_workload_command(workload)
    probes: dict[tuple[str, ...], list[mod.CommandProbe]] = {
        mod.KV260_SSH_COMMAND: [
            _probe(
                mod.KV260_SSH_COMMAND,
                exit_code=255,
                stderr="ssh: Could not resolve hostname kria: Name or service not known\n",
            )
        ],
        mod.POLARFIRE_STATUS_COMMAND: [
            _probe(
                mod.POLARFIRE_STATUS_COMMAND,
                exit_code=polarfire_status_exit,
                stdout="polarfire reachable\n" if polarfire_status_exit == 0 else "",
                stderr="" if polarfire_status_exit == 0 else "ssh: no route to host\n",
            )
        ],
        mod.GATEMATE_DETECT_COMMAND: [
            _probe(
                mod.GATEMATE_DETECT_COMMAND,
                exit_code=gatemate_exit,
                stdout=(
                    "GateMate Series GM1Ax IDCODE 0x20000001\n"
                    if gatemate_exit == 0
                    else "Jtag frequency : requested 6000000 Hz\n"
                ),
                stderr="" if gatemate_exit == 0 else "detect failed\n",
            )
        ],
    }
    if polarfire_status_exit == 0:
        outputs = board_stdout or [
            _board_stdout(workload, wall_time_s=0.101),
            _board_stdout(workload, wall_time_s=0.102),
            _board_stdout(workload, wall_time_s=0.103),
        ]
        probes[board_command] = [
            _probe(board_command, stdout=stdout, duration_s=0.2) for stdout in outputs
        ]
    return RecordingRunner(probes)


def test_req_hw_5424_spec_declares_comparable_timing_contract() -> None:
    """REQ-HW-5424: OpenSpec anchors comparable timing and speedup refusal."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5424") : spec.index("### SCENARIO-HW-5424")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-HW-5424",
        "SCENARIO-HW-5424",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp 5420",
        "workload_hash",
        "cpu_result_hash",
        "board_result_hash",
        "same_workload_hash_match",
        "same_result_hash_match",
        "measurement_access_complete",
        "comparable_timing_receipts_ready",
        "hardware_speedup_claim",
        "hardware_timing_with_cpu_reference",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_scenario_hw_5424_writes_hash_matched_comparable_receipts(tmp_path: Path) -> None:
    """SCENARIO-HW-5424: hash/result matched timing receipts are ready, not speedup."""

    runner = _runner()
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    out_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    board_command = mod.polarfire_workload_command(mod.select_workload(REPO))

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.POLARFIRE_STATUS_COMMAND,
        board_command,
        board_command,
        board_command,
        mod.GATEMATE_DETECT_COMMAND,
    ]
    assert saved["preconditions_checked"] is True
    assert saved["workload_hash"] == mod.select_workload(REPO)["workload_hash"]
    assert saved["cpu_repeat_count"] == mod.REPEAT_TARGET
    assert saved["board_repeat_count"] == mod.REPEAT_TARGET
    assert saved["cpu_result_hash"] == mod.select_workload(REPO)["exact_result_hash"]
    assert saved["board_result_hash"] == saved["cpu_result_hash"]
    assert saved["same_workload_hash_match"] is True
    assert saved["same_result_hash_match"] is True
    assert saved["polarfire_reachable"] is True
    assert saved["kv260_ssh_checked"] is True
    assert saved["gatemate_diagnostic_checked"] is True
    assert saved["measurement_access_complete"] is True
    assert saved["comparable_timing_receipts_ready"] is True
    assert saved["hardware_speedup_claim"] is False
    assert saved["timing_comparison"]["comparison_performed"] is True
    assert saved["timing_comparison"]["speedup_reported"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_hw_5424_unreachable_polarfire_records_blocked_precondition() -> None:
    """REQ-HW-5424: missing board access records the attempted SSH precondition."""

    runner = _runner(polarfire_status_exit=255)
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert mod.polarfire_workload_command(mod.select_workload(REPO)) not in runner.commands
    assert artifact["polarfire_reachable"] is False
    assert artifact["board_repeat_count"] == 0
    assert artifact["board_result_hash"] == ""
    assert artifact["same_workload_hash_match"] is False
    assert artifact["same_result_hash_match"] is False
    assert artifact["measurement_access_complete"] is False
    assert artifact["comparable_timing_receipts_ready"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["blocked_hardware_precondition"]["command"] == mod.command_to_string(
        mod.POLARFIRE_STATUS_COMMAND
    )
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_hw_5424_result_hash_drift_refuses_comparability_and_speedup() -> None:
    """REQ-HW-5424: result-hash drift prevents comparison and speedup claims."""

    workload = mod.select_workload(REPO)
    runner = _runner(
        board_stdout=[
            _board_stdout(workload, wall_time_s=0.101),
            _board_stdout(workload, result_hash="1" * 64, wall_time_s=0.102),
            _board_stdout(workload, wall_time_s=0.103),
        ]
    )
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert artifact["board_repeat_count"] == 2
    assert artifact["same_workload_hash_match"] is False
    assert artifact["same_result_hash_match"] is False
    assert artifact["measurement_access_complete"] is False
    assert artifact["comparable_timing_receipts_ready"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert "same_result_hash_mismatch" in artifact["readiness_blockers"]
    assert artifact["timing_receipts"][1]["invalid_repeat_count"] == 1
    mod.validate_artifact(artifact)


def test_req_hw_5424_validation_rejects_speedup_low_repeats_and_unsafe_commands() -> None:
    """REQ-HW-5424: validator fails closed on speedup, thresholds, and unsafe probes."""

    artifact = mod.build_artifact(
        root=REPO,
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    speedup = deepcopy(artifact)
    speedup["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(speedup)

    low_repeats = deepcopy(artifact)
    low_repeats["cpu_repeat_count"] = 2
    low_repeats["comparable_timing_receipts_ready"] = False
    with pytest.raises(ValueError, match="cpu_repeat_count"):
        mod.validate_artifact(low_repeats)

    unsafe_kv260 = deepcopy(artifact)
    unsafe_kv260["command_receipts"][0]["command"] = "ssh kria 'ls /dev/mmcblk*'"
    unsafe_kv260["comparable_timing_receipts_ready"] = False
    with pytest.raises(ValueError, match="host block-device"):
        mod.validate_artifact(unsafe_kv260)

    destructive = deepcopy(artifact)
    destructive["command_receipts"].append(
        {
            "kind": "gatemate_bad_write",
            "command": "openFPGALoader --write flash.bit",
            "command_sha256": mod.sha256_text("openFPGALoader --write flash.bit"),
            "exit_code": 0,
            "duration_s": 0.1,
            "timeout_s": 1.0,
            "outcome": "bad",
            "stdout_sha256": mod.sha256_text(""),
            "stderr_sha256": mod.sha256_text(""),
            "stdout_excerpt": "",
            "stderr_excerpt": "",
        }
    )
    destructive["comparable_timing_receipts_ready"] = False
    with pytest.raises(ValueError, match="destructive"):
        mod.validate_artifact(destructive)

    missing_tests = deepcopy(artifact)
    missing_tests["tests_run"] = []
    missing_tests["comparable_timing_receipts_ready"] = False
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(missing_tests)


def test_req_hw_5424_defensive_helpers_and_run_experiment(tmp_path: Path) -> None:
    """REQ-HW-5424: helper paths and run_experiment preserve stable artifacts."""

    ok = mod.run_command(("sh", "-lc", "printf ok"), timeout_s=1.0)
    missing = mod.run_command(("definitely_missing_carnot_5424",), timeout_s=0.1)
    timeout = mod.run_command(("sh", "-lc", "sleep 0.2"), timeout_s=0.01)

    assert ok.exit_code == 0
    assert ok.stdout == "ok"
    assert missing.exit_code == 127
    assert timeout.exit_code == 124
    assert mod.timing_distribution([]) == {"count": 0}
    assert mod.readiness_blockers(
        polarfire_reachable=True,
        cpu_repeat_count=2,
        board_repeat_count=3,
        same_workload_hash_match=True,
        same_result_hash_match=True,
    ) == ["cpu_repeat_count_below_threshold"]
    assert mod.default_tests_run()[0]["outcome"] == "pending_external_test_run"

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        workload_root=REPO,
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert saved["spec_refs"] == list(mod.SPEC_REFS)
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert saved["hardware_speedup_claim"] is False
    mod.validate_artifact(saved)
