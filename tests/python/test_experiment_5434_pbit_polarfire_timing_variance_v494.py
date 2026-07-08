"""Tests for Exp5434 gated p-bit/PolarFire timing variance receipts.

Spec refs: REQ-HW-5434, SCENARIO-HW-5434.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5434_pbit_polarfire_timing_variance_v494 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5434_pbit_polarfire_timing_variance_v494.py "
    "-q --no-cov"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5434_pbit_polarfire_timing_variance_v494.py "
    "-m pytest tests/python/test_experiment_5434_pbit_polarfire_timing_variance_v494.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report "
    "--include=python/carnot/experiment_5434_pbit_polarfire_timing_variance_v494.py "
    "--fail-under=100"
)


class RecordingRunner:
    """SCENARIO-HW-5434 runner with exact command expectations."""

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


class VariableClock:
    """Deterministic clock that gives nonzero repeat variance."""

    def __init__(self) -> None:
        self.value = 5434.0
        self.increments = [0.0003 + 0.00001 * index for index in range(80)]
        self.index = 0

    def __call__(self) -> float:
        increment = self.increments[self.index % len(self.increments)]
        self.index += 1
        self.value += increment
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
    return [
        {"command": TEST_COMMAND, "outcome": "passed"},
        {"command": COVERAGE_COMMAND, "outcome": "passed"},
        {"command": COVERAGE_REPORT_COMMAND, "outcome": "passed"},
    ]


def _board_stdout(
    workload: dict[str, object],
    *,
    workload_hash: str | None = None,
    result_hash: str | None = None,
    wall_time_s: float = 0.123,
) -> str:
    exact_result = workload["exact_result"]
    assert isinstance(exact_result, dict)
    payload = {
        "active_constraint_ids": workload["active_constraint_ids"],
        "board_local": True,
        "exact_best_sequence": exact_result["exact_best_sequence"],
        "exact_min_energy": exact_result["exact_min_energy"],
        "result_hash": result_hash or workload["result_hash"],
        "seed": mod.RANDOM_SEED,
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
            _board_stdout(workload, wall_time_s=0.101 + 0.001 * index)
            for index in range(mod.REPEAT_TARGET)
        ]
        probes[board_command] = [
            _probe(board_command, stdout=stdout, duration_s=0.2) for stdout in outputs
        ]
    return RecordingRunner(probes)


def test_req_hw_5434_spec_declares_variance_receipt_contract() -> None:
    """REQ-HW-5434: OpenSpec anchors gated variance receipts and no speedup."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5434") : spec.index("### SCENARIO-HW-5434")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-HW-5434",
        "SCENARIO-HW-5434",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp 5433",
        "active_constraint_diversity_ready=true",
        "at least ten times",
        "workload_hash",
        "cpu_result_hash",
        "board_result_hash",
        "cpu_timing_variance",
        "board_timing_variance",
        "timing_variance_receipts_ready",
        "hardware_speedup_claim",
        "hardware_timing_with_cpu_reference",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_scenario_hw_5434_writes_hash_matched_variance_receipts(tmp_path: Path) -> None:
    """SCENARIO-HW-5434: hash-matched repeated receipts are ready, not speedup."""

    runner = _runner()
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=runner,
        clock=VariableClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    out_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    workload = mod.select_workload(REPO)
    board_command = mod.polarfire_workload_command(workload)

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.POLARFIRE_STATUS_COMMAND,
        *([board_command] * mod.REPEAT_TARGET),
        mod.GATEMATE_DETECT_COMMAND,
    ]
    assert saved["preconditions_checked"] is True
    assert saved["gated_upstream_ready"] is True
    assert saved["selected_workload"]["source_experiment_id"].startswith("exp5433-")
    assert saved["selected_workload"]["exact_solver_validity"] is True
    assert saved["workload_hash"] == workload["workload_hash"]
    assert saved["cpu_repeat_count"] == mod.REPEAT_TARGET
    assert saved["board_repeat_count"] == mod.REPEAT_TARGET
    assert saved["cpu_result_hash"] == workload["result_hash"]
    assert saved["board_result_hash"] == saved["cpu_result_hash"]
    assert saved["same_workload_hash_match"] is True
    assert saved["same_result_hash_match"] is True
    assert isinstance(saved["cpu_timing_variance"], float)
    assert isinstance(saved["board_timing_variance"], float)
    assert saved["timing_comparison"]["board_cpu_ratio"] > 0.0
    assert saved["timing_comparison"]["hardware_speedup_claim"] is False
    assert saved["polarfire_reachable"] is True
    assert saved["kv260_ssh_checked"] is True
    assert saved["gatemate_diagnostic_checked"] is True
    assert saved["measurement_access_complete"] is True
    assert saved["timing_variance_receipts_ready"] is True
    assert saved["hardware_speedup_claim"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_hw_5434_unreachable_polarfire_records_blocked_precondition() -> None:
    """REQ-HW-5434: unreachable PolarFire records blocked measurement access."""

    runner = _runner(polarfire_status_exit=255)
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=runner,
        clock=VariableClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert mod.polarfire_workload_command(mod.select_workload(REPO)) not in runner.commands
    assert artifact["gated_upstream_ready"] is True
    assert artifact["polarfire_reachable"] is False
    assert artifact["board_repeat_count"] == 0
    assert artifact["board_result_hash"] == ""
    assert artifact["same_workload_hash_match"] is False
    assert artifact["same_result_hash_match"] is False
    assert artifact["board_timing_variance"] == 0.0
    assert artifact["measurement_access_complete"] is False
    assert artifact["timing_variance_receipts_ready"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["blocked_hardware_precondition"]["command"] == mod.command_to_string(
        mod.POLARFIRE_STATUS_COMMAND
    )
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_hw_5434_hash_mismatch_and_low_repeats_refuse_comparison() -> None:
    """REQ-HW-5434: result-hash drift blocks comparison and speedup claims."""

    workload = mod.select_workload(REPO)
    outputs = [
        _board_stdout(workload, wall_time_s=0.101 + 0.001 * index)
        for index in range(mod.REPEAT_TARGET)
    ]
    outputs[3] = _board_stdout(workload, result_hash="1" * 64, wall_time_s=0.104)
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=_runner(board_stdout=outputs),
        clock=VariableClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert artifact["board_repeat_count"] == mod.REPEAT_TARGET - 1
    assert artifact["same_workload_hash_match"] is False
    assert artifact["same_result_hash_match"] is False
    assert artifact["measurement_access_complete"] is False
    assert artifact["timing_variance_receipts_ready"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert "same_result_hash_mismatch" in artifact["readiness_blockers"]
    assert artifact["timing_receipts"][1]["invalid_repeat_count"] == 1
    mod.validate_artifact(artifact)


def test_req_hw_5434_validation_rejects_speedup_thresholds_and_unsafe_commands() -> None:
    """REQ-HW-5434: validator fails closed on speedup, thresholds, and probes."""

    artifact = mod.build_artifact(
        root=REPO,
        command_runner=_runner(),
        clock=VariableClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    speedup = deepcopy(artifact)
    speedup["hardware_speedup_claim"] = True
    speedup["timing_comparison"]["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(speedup)

    low_repeats = deepcopy(artifact)
    low_repeats["cpu_repeat_count"] = mod.REPEAT_TARGET - 1
    low_repeats["timing_variance_receipts_ready"] = False
    with pytest.raises(ValueError, match="cpu_repeat_count"):
        mod.validate_artifact(low_repeats)

    unsafe_kv260 = deepcopy(artifact)
    unsafe_kv260["command_receipts"][0]["command"] = "ssh kria 'ls /dev/mmcblk*'"
    unsafe_kv260["timing_variance_receipts_ready"] = False
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
    destructive["timing_variance_receipts_ready"] = False
    with pytest.raises(ValueError, match="destructive"):
        mod.validate_artifact(destructive)

    missing_tests = deepcopy(artifact)
    missing_tests["tests_run"] = []
    missing_tests["timing_variance_receipts_ready"] = False
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(missing_tests)


def test_req_hw_5434_helpers_run_experiment_and_repository_artifact(tmp_path: Path) -> None:
    """REQ-HW-5434: helper paths, writer, and checked-in artifact stay valid."""

    workload = mod.select_workload(REPO)
    ok = mod.run_command(("sh", "-lc", "printf ok"), timeout_s=1.0)
    missing = mod.run_command(("definitely_missing_carnot_5434",), timeout_s=0.1)
    timeout = mod.run_command(("sh", "-lc", "sleep 0.2"), timeout_s=0.01)

    assert ok.exit_code == 0
    assert ok.stdout == "ok"
    assert missing.exit_code == 127
    assert timeout.exit_code == 124
    assert mod.precedence_energy(workload, ["not-a-valid-sequence"]) > 0
    assert mod.timing_distribution([]) == {
        "count": 0,
        "mean_s": 0.0,
        "median_s": 0.0,
        "p95_s": 0.0,
        "variance_s2": 0.0,
    }
    assert mod.parse_board_workload_stdout("\nnot json\n[]\n", workload) == (
        None,
        "workload stdout is not valid JSON",
    )
    bad_payload = json.loads(_board_stdout(workload))
    bad_payload.update(
        {
            "active_constraint_ids": ["wrong->constraint"],
            "board_local": False,
            "exact_best_sequence": ["wrong"],
            "exact_min_energy": 99,
            "seed": -1,
            "wall_time_s": -0.1,
            "workload_hash": "2" * 64,
        }
    )
    _, parse_error = mod.parse_board_workload_stdout(
        json.dumps(bad_payload, sort_keys=True),
        workload,
    )
    assert parse_error is not None
    for marker in (
        "workload_hash mismatch",
        "seed mismatch",
        "active_constraint_ids mismatch",
        "exact_min_energy mismatch",
        "exact_best_sequence mismatch",
        "board_local missing",
        "wall_time_s invalid",
    ):
        assert marker in parse_error
    assert mod.readiness_blockers(
        gated_upstream_ready=True,
        polarfire_reachable=True,
        cpu_repeat_count=mod.REPEAT_TARGET - 1,
        board_repeat_count=mod.REPEAT_TARGET,
        same_workload_hash_match=True,
        same_result_hash_match=True,
    ) == ["cpu_repeat_count_below_threshold"]
    assert "active_constraint_diversity_not_ready" in mod.readiness_blockers(
        gated_upstream_ready=False,
        polarfire_reachable=False,
        cpu_repeat_count=mod.REPEAT_TARGET,
        board_repeat_count=0,
        same_workload_hash_match=False,
        same_result_hash_match=False,
    )
    assert mod.default_tests_run()[0]["outcome"] == "pending_external_test_run"

    fallback_source = json.loads((REPO / mod.UPSTREAM_DIVERSITY_RELATIVE_PATH).read_text())
    for row in fallback_source["row_records"]:
        if row["hint_mode"] == "lns_guided_hint":
            row["exact_min_energy"] = 1
    fallback_path = tmp_path / mod.UPSTREAM_DIVERSITY_RELATIVE_PATH
    fallback_path.parent.mkdir(parents=True)
    fallback_path.write_text(json.dumps(fallback_source), encoding="utf-8")
    fallback_workload = mod.select_workload(tmp_path)
    assert fallback_workload["fixture_id"] == workload["fixture_id"]

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        workload_root=REPO,
        command_runner=_runner(),
        clock=VariableClock(),
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

    if RESULT_PATH.exists():
        checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        assert checked_in["experiment_id"] == mod.EXPERIMENT_ID
        assert checked_in["reproducibility_checksum"] == mod.payload_checksum(checked_in)
        assert checked_in["hardware_speedup_claim"] is False
        mod.validate_artifact(checked_in)
