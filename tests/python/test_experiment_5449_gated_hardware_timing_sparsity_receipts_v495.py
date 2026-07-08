"""Tests for Exp5449 gated hardware timing sparsity receipts.

Spec refs: REQ-HW-5449, SCENARIO-HW-5449.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5449_gated_hardware_timing_sparsity_receipts_v495 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5449_gated_hardware_timing_sparsity_receipts_v495.py "
    "-q --no-cov"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.py "
    "-m pytest tests/python/test_experiment_5449_gated_hardware_timing_sparsity_receipts_v495.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report "
    "--include=python/carnot/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.py "
    "--fail-under=100"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
E2E_COMMAND = (
    "ops/e2e-test-plan.md review: Exp5449 is a deterministic receipt replay; "
    "no live training or PyO3 e2e path applies"
)


class RecordingRunner:
    """SCENARIO-HW-5449 runner with exact command expectations."""

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
    """Deterministic clock that gives nonzero CPU repeat variance."""

    def __init__(self) -> None:
        self.value = 5449.0
        self.increments = [0.00021 + 0.00001 * index for index in range(120)]
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
        {"command": FULL_SUITE_COMMAND, "outcome": "passed"},
        {"command": E2E_COMMAND, "outcome": "not_applicable"},
    ]


def _board_stdout(
    board: str,
    workload: dict[str, object],
    *,
    workload_hash: str | None = None,
    result_hash: str | None = None,
    wall_time_s: float = 0.25,
) -> str:
    payload = {
        "board_identity": board,
        "board_local": True,
        "fixture_subset": workload["fixture_subset_ids"],
        "result_hash": result_hash or workload["expected_result_hashes"]["aggregate"],
        "seed": workload["seeds"]["upstream_random_seed"],
        "wall_time_s": wall_time_s,
        "workload_hash": workload_hash or workload["workload_hash"],
    }
    return json.dumps(payload, sort_keys=True) + "\n"


def _runner(
    *,
    kv260_reachable: bool = True,
    polarfire_reachable: bool = True,
    kv260_outputs: list[str] | None = None,
    polarfire_outputs: list[str] | None = None,
    gatemate_detected: bool = False,
) -> RecordingRunner:
    workload = mod.extract_workload(REPO)
    kv_command = mod.board_workload_command("kv260", workload)
    pf_command = mod.board_workload_command("polarfire", workload)
    probes: dict[tuple[str, ...], list[mod.CommandProbe]] = {
        mod.KV260_SSH_COMMAND: [
            _probe(
                mod.KV260_SSH_COMMAND,
                exit_code=0 if kv260_reachable else 255,
                stdout="kv260 reachable\n" if kv260_reachable else "",
                stderr="" if kv260_reachable else "ssh: no route to host\n",
            )
        ],
        mod.POLARFIRE_STATUS_COMMAND: [
            _probe(
                mod.POLARFIRE_STATUS_COMMAND,
                exit_code=0 if polarfire_reachable else 255,
                stdout="polarfire reachable\n" if polarfire_reachable else "",
                stderr="" if polarfire_reachable else "ssh: no route to host\n",
            )
        ],
        mod.GATEMATE_DETECT_COMMAND: [
            _probe(
                mod.GATEMATE_DETECT_COMMAND,
                exit_code=0 if gatemate_detected else 1,
                stdout=(
                    "GateMate Series GM1Ax IDCODE 0x20000001\n"
                    if gatemate_detected
                    else "Jtag frequency : requested 6000000 Hz\n"
                ),
                stderr="" if gatemate_detected else "detect failed\n",
            )
        ],
    }
    if kv260_reachable:
        outputs = kv260_outputs or [
            _board_stdout("kv260", workload, wall_time_s=0.31 + 0.001 * index)
            for index in range(mod.REPEAT_TARGET)
        ]
        probes[kv_command] = [_probe(kv_command, stdout=stdout) for stdout in outputs]
    if polarfire_reachable:
        outputs = polarfire_outputs or [
            _board_stdout("polarfire", workload, wall_time_s=0.41 + 0.001 * index)
            for index in range(mod.REPEAT_TARGET)
        ]
        probes[pf_command] = [_probe(pf_command, stdout=stdout) for stdout in outputs]
    return RecordingRunner(probes)


def test_req_hw_5449_spec_declares_gated_receipt_contract() -> None:
    """REQ-HW-5449: OpenSpec anchors hash-matched hardware receipts."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5449") : spec.index("### SCENARIO-HW-5449")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-HW-5449",
        "SCENARIO-HW-5449",
        str(mod.RESULT_RELATIVE_PATH),
        "pbit_assumption_bridge_ready=true",
        "blocked_board_unreachable",
        "workload hash",
        "result hash",
        "KV260 through SSH-only commands",
        "GateMate SHALL remain diagnostic-only",
        "cpu_and_reachable_board_timing_receipts",
        "hardware_speedup_claim=false",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_scenario_hw_5449_writes_hash_matched_reachable_board_receipts(tmp_path: Path) -> None:
    """SCENARIO-HW-5449: repeated CPU and board receipts match before timing."""

    runner = _runner(gatemate_detected=True)
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
    workload = mod.extract_workload(REPO)
    kv_command = mod.board_workload_command("kv260", workload)
    pf_command = mod.board_workload_command("polarfire", workload)

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        *([kv_command] * mod.REPEAT_TARGET),
        mod.POLARFIRE_STATUS_COMMAND,
        *([pf_command] * mod.REPEAT_TARGET),
        mod.GATEMATE_DETECT_COMMAND,
    ]
    assert saved["preconditions_checked"] is True
    assert saved["gated_upstream_ready"] is True
    assert saved["workload_hash"] == workload["workload_hash"]
    assert saved["selected_workload"]["fixture_subset_ids"] == workload["fixture_subset_ids"]
    assert saved["selected_workload"]["expected_result_hashes"]["aggregate"] == saved[
        "cpu_result_hash"
    ]
    assert saved["cpu_result_hash"] == mod.workload_result_hash(mod.replay_workload(workload))
    assert saved["board_result_hashes"]["kv260"] == saved["cpu_result_hash"]
    assert saved["board_result_hashes"]["polarfire"] == saved["cpu_result_hash"]
    assert saved["board_result_hashes"]["gatemate"] == ""
    assert saved["board_reachability"]["kv260"]["check_method"] == "ssh_only"
    assert saved["board_reachability"]["gatemate"]["diagnostic_only"] is True
    assert saved["board_reachability"]["gatemate"]["workload_execution_claim"] is False
    assert saved["kv260_ssh_only_checked"] is True
    assert saved["timing_repeat_counts"] == {
        "cpu": mod.REPEAT_TARGET,
        "kv260": mod.REPEAT_TARGET,
        "polarfire": mod.REPEAT_TARGET,
        "gatemate": 0,
    }
    assert saved["timing_summary"]["cpu"]["count"] == mod.REPEAT_TARGET
    assert saved["timing_summary"]["kv260"]["variance_s2"] >= 0.0
    assert saved["timing_comparison"]["comparison_performed"] is True
    assert saved["hashes_match_before_timing_compare"] is True
    assert saved["hardware_speedup_claim"] is False
    assert saved["hardware_receipts_ready"] is True
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_hw_5449_unreachable_boards_emit_blocked_board_unreachable() -> None:
    """REQ-HW-5449: unreachable boards keep CPU receipts and block honestly."""

    runner = _runner(kv260_reachable=False, polarfire_reachable=False)
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=runner,
        clock=VariableClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.POLARFIRE_STATUS_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
    ]
    assert artifact["gated_upstream_ready"] is True
    assert artifact["timing_repeat_counts"]["cpu"] == mod.REPEAT_TARGET
    assert artifact["timing_repeat_counts"]["kv260"] == 0
    assert artifact["timing_repeat_counts"]["polarfire"] == 0
    assert artifact["board_result_hashes"] == {"kv260": "", "polarfire": "", "gatemate": ""}
    assert artifact["hashes_match_before_timing_compare"] is False
    assert artifact["hardware_receipts_ready"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert "blocked_board_unreachable" in artifact["readiness_blockers"]
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_hw_5449_hash_mismatch_blocks_timing_comparison() -> None:
    """REQ-HW-5449: board result drift blocks comparison and speedup claims."""

    workload = mod.extract_workload(REPO)
    outputs = [
        _board_stdout("polarfire", workload, wall_time_s=0.41 + 0.001 * index)
        for index in range(mod.REPEAT_TARGET)
    ]
    outputs[2] = _board_stdout("polarfire", workload, result_hash="1" * 64, wall_time_s=0.43)
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=_runner(kv260_reachable=False, polarfire_outputs=outputs),
        clock=VariableClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert artifact["board_result_hashes"]["polarfire"] == ""
    assert artifact["timing_repeat_counts"]["polarfire"] == mod.REPEAT_TARGET - 1
    assert artifact["hashes_match_before_timing_compare"] is False
    assert artifact["hardware_receipts_ready"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert "board_hash_or_repeat_mismatch" in artifact["readiness_blockers"]
    mod.validate_artifact(artifact)


def test_req_hw_5449_gate_false_fails_closed_without_board_probes(tmp_path: Path) -> None:
    """REQ-HW-5449: false Exp5448 gate blocks before hardware probing."""

    source = json.loads((REPO / mod.UPSTREAM_EXP5448_RELATIVE_PATH).read_text())
    source["pbit_assumption_bridge_ready"] = False
    target = tmp_path / mod.UPSTREAM_EXP5448_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(source), encoding="utf-8")
    runner = RecordingRunner({})

    artifact = mod.build_artifact(
        root=tmp_path,
        command_runner=runner,
        clock=VariableClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert runner.commands == []
    assert artifact["preconditions_checked"] is True
    assert artifact["gated_upstream_ready"] is False
    assert artifact["workload_hash"] == ""
    assert artifact["cpu_result_hash"] == ""
    assert artifact["kv260_ssh_only_checked"] is False
    assert artifact["timing_repeat_counts"]["cpu"] == 0
    assert artifact["hardware_receipts_ready"] is False
    assert "pbit_assumption_bridge_not_ready" in artifact["readiness_blockers"]
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_hw_5449_validation_rejects_speedup_storage_and_bad_commands() -> None:
    """REQ-HW-5449: validator fails closed on unsafe schema drift."""

    artifact = mod.build_artifact(
        root=REPO,
        command_runner=_runner(),
        clock=VariableClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    mod.validate_artifact(artifact)

    speedup = deepcopy(artifact)
    speedup["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(speedup)

    bad_storage = deepcopy(artifact)
    bad_storage["command_receipts"][0]["command"] = "ssh kria 'ls /dev/mmcblk*'"
    bad_storage["hardware_receipts_ready"] = False
    with pytest.raises(ValueError, match="host block-device"):
        mod.validate_artifact(bad_storage)

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
    destructive["hardware_receipts_ready"] = False
    with pytest.raises(ValueError, match="destructive"):
        mod.validate_artifact(destructive)

    missing_tests = deepcopy(artifact)
    missing_tests["tests_run"] = []
    missing_tests["hardware_receipts_ready"] = False
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(missing_tests)


def test_req_hw_5449_helpers_and_repository_artifact(tmp_path: Path) -> None:
    """REQ-HW-5449: helpers, writer, and checked-in artifact stay valid."""

    workload = mod.extract_workload(REPO)
    replay = mod.replay_workload(workload)
    fixture = workload["fixture_subset"][0]
    ok = mod.run_command(("sh", "-lc", "printf ok"), timeout_s=1.0)
    missing = mod.run_command(("definitely_missing_carnot_5449",), timeout_s=0.1)
    timeout = mod.run_command(("sh", "-lc", "sleep 0.2"), timeout_s=0.01)

    assert ok.exit_code == 0
    assert ok.stdout == "ok"
    assert missing.exit_code == 127
    assert timeout.exit_code == 124
    assert mod.workload_result_hash(replay) == workload["expected_result_hashes"]["aggregate"]
    assert mod.load_upstream_gate(tmp_path)["source_status"] == "missing"
    unreadable = tmp_path / mod.UPSTREAM_EXP5448_RELATIVE_PATH
    unreadable.parent.mkdir(parents=True, exist_ok=True)
    unreadable.write_text("{not-json", encoding="utf-8")
    assert mod.load_upstream_gate(tmp_path)["source_status"] == "unreadable"
    assert mod._serialize_solution(fixture, None) is None
    assert mod._solution_valid(fixture, {"status": "unsat", "solution": None}) is False
    assert mod._solution_valid(fixture, {"status": "sat", "solution": None}) is False
    assert mod.timing_distribution([]) == {
        "count": 0,
        "min_s": 0.0,
        "max_s": 0.0,
        "mean_s": 0.0,
        "median_s": 0.0,
        "variance_s2": 0.0,
    }
    assert mod.parse_board_workload_stdout("\nnot json\n[]\n", workload, "kv260") == (
        None,
        "workload stdout is not valid JSON",
    )
    bad_payload = json.loads(_board_stdout("kv260", workload))
    bad_payload.update(
        {
            "board_identity": "wrong",
            "board_local": False,
            "fixture_subset": ["wrong"],
            "result_hash": "2" * 64,
            "seed": -1,
            "wall_time_s": -0.1,
            "workload_hash": "3" * 64,
        }
    )
    _, parse_error = mod.parse_board_workload_stdout(
        json.dumps(bad_payload, sort_keys=True),
        workload,
        "kv260",
    )
    assert parse_error is not None
    for marker in (
        "board_identity mismatch",
        "workload_hash mismatch",
        "seed mismatch",
        "fixture_subset mismatch",
        "result_hash mismatch",
        "board_local missing",
        "wall_time_s invalid",
    ):
        assert marker in parse_error

    assert mod.readiness_blockers(
        gated_upstream_ready=True,
        cpu_repeat_count=mod.REPEAT_TARGET,
        executable_board_count=0,
        hashes_match_before_timing_compare=False,
    ) == ["blocked_board_unreachable"]
    assert "cpu_repeat_count_below_threshold" in mod.readiness_blockers(
        gated_upstream_ready=False,
        cpu_repeat_count=mod.REPEAT_TARGET - 1,
        executable_board_count=0,
        hashes_match_before_timing_compare=False,
    )
    assert mod.default_tests_run()[0]["outcome"] == "pending_external_test_run"

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        workload_root=REPO,
        command_runner=_runner(kv260_reachable=False, polarfire_reachable=False),
        clock=VariableClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert saved["spec_refs"] == list(mod.SPEC_REFS)
    assert saved["hardware_speedup_claim"] is False
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)

    if RESULT_PATH.exists():
        checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        assert checked_in["experiment_id"] == mod.EXPERIMENT_ID
        assert checked_in["reproducibility_checksum"] == mod.payload_checksum(checked_in)
        assert checked_in["hardware_speedup_claim"] is False
        mod.validate_artifact(checked_in)
