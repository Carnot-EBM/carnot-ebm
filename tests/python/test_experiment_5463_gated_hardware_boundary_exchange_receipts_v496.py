"""Tests for Exp5463 gated hardware boundary-exchange receipts.

Spec refs: REQ-VERIFY-5463, SCENARIO-VERIFY-5463.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5463_gated_hardware_boundary_exchange_receipts_v496 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5463_gated_hardware_boundary_exchange_receipts_v496.py "
    "-q --no-cov"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.py "
    "-m pytest "
    "tests/python/test_experiment_5463_gated_hardware_boundary_exchange_receipts_v496.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report "
    "--include=python/carnot/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.py "
    "--fail-under=100"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
E2E_COMMAND = (
    "ops/e2e-test-plan.md review: Exp5463 replays deterministic Exp5462 fixtures "
    "and performs SSH-only board receipts; no live training or PyO3 path applies"
)


class RecordingRunner:
    """SCENARIO-VERIFY-5463 fake command runner with exact command receipts."""

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
    """Deterministic nonzero clock so repeat timing variance is testable."""

    def __init__(self) -> None:
        self.value = 5463.0
        self.increments = [0.00031 + 0.00001 * index for index in range(160)]
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
        {"command": SPEC_COVERAGE_COMMAND, "outcome": "passed"},
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
        "boundary_exchange_supported": False,
    }
    return json.dumps(payload, sort_keys=True) + "\n"


def _runner(
    *,
    kv260_reachable: bool = True,
    polarfire_reachable: bool = True,
    kv260_outputs: list[str] | None = None,
    polarfire_outputs: list[str] | None = None,
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
    }
    if kv260_reachable:
        outputs = kv260_outputs or [
            _board_stdout("kv260", workload, wall_time_s=0.35 + 0.001 * index)
            for index in range(mod.REPEAT_TARGET)
        ]
        probes[kv_command] = [_probe(kv_command, stdout=stdout) for stdout in outputs]
    if polarfire_reachable:
        outputs = polarfire_outputs or [
            _board_stdout("polarfire", workload, wall_time_s=0.45 + 0.001 * index)
            for index in range(mod.REPEAT_TARGET)
        ]
        probes[pf_command] = [_probe(pf_command, stdout=stdout) for stdout in outputs]
    return RecordingRunner(probes)


def test_req_verify_5463_spec_declares_gated_boundary_receipt_contract() -> None:
    """REQ-VERIFY-5463: OpenSpec anchors hash-first hardware receipts."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5463") : spec.index("### REQ-VERIFY-5433")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5463",
        "SCENARIO-VERIFY-5463",
        str(mod.RESULT_RELATIVE_PATH),
        "minimal_core_pbit_bridge_ready=true",
        "blocked_board_unreachable",
        "KV260 only through SSH-safe commands",
        "boundary_exchange_ratio_summary",
        mod.INFERENCE_SUBSTRATE,
        "hardware_speedup_claim",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_scenario_verify_5463_writes_hash_matched_polarfire_receipts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5463: reachable board hashes match before timing."""

    runner = _runner(kv260_reachable=False, polarfire_reachable=True)
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=runner,
        clock=VariableClock(),
        run_date="20260709",
        commit="abc123",
        tests_run=_tests_run(),
    )
    out_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    workload = mod.extract_workload(REPO)
    pf_command = mod.board_workload_command("polarfire", workload)

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.POLARFIRE_STATUS_COMMAND,
        *([pf_command] * mod.REPEAT_TARGET),
    ]
    assert saved["preconditions_checked"] is True
    assert saved["gated_upstream_ready"] is True
    assert saved["workload_hash"] == workload["workload_hash"]
    assert saved["selected_workload"]["fixture_subset_ids"] == workload["fixture_subset_ids"]
    assert saved["selected_workload"]["expected_result_hashes"]["aggregate"] == saved[
        "cpu_result_hash"
    ]
    assert saved["cpu_result_hash"] == mod.workload_result_hash(mod.replay_workload(workload))
    assert saved["board_result_hashes"]["kv260"] == ""
    assert saved["board_result_hashes"]["polarfire"] == saved["cpu_result_hash"]
    assert saved["board_reachability"]["kv260"]["check_method"] == "ssh_only"
    assert saved["board_reachability"]["kv260"]["blocked_reason"] == "blocked_kv260_ssh"
    assert saved["board_reachability"]["polarfire"]["workload_execution_attempted"] is True
    assert saved["kv260_ssh_only_checked"] is True
    assert saved["boundary_exchange_ratio_summary"]["source_artifact"].endswith(
        "experiment_5371_pbit_boundary_exchange_schedule_v489.json"
    )
    assert saved["boundary_exchange_ratio_summary"]["simulation_only"] is True
    assert saved["boundary_exchange_ratio_summary"]["eta_values"] == [0.25, 0.5, 1.0]
    assert saved["boundary_exchange_ratio_summary"]["repeat_counts"]["cpu_simulated"] > 0
    assert saved["timing_repeat_counts"] == {
        "cpu": mod.REPEAT_TARGET,
        "kv260": 0,
        "polarfire": mod.REPEAT_TARGET,
    }
    assert saved["timing_summary"]["cpu"]["count"] == mod.REPEAT_TARGET
    assert saved["timing_summary"]["polarfire"]["variance_s2"] >= 0.0
    assert saved["timing_comparison"]["comparison_performed"] is True
    assert saved["hashes_match_before_timing_compare"] is True
    assert saved["hardware_speedup_claim"] is False
    assert saved["hardware_receipts_ready"] is True
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert "/dev/mmcblk" not in json.dumps(saved)
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_verify_5463_unreachable_boards_keep_cpu_receipts() -> None:
    """REQ-VERIFY-5463: blocked boards emit blocked_board_unreachable honestly."""

    runner = _runner(kv260_reachable=False, polarfire_reachable=False)
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=runner,
        clock=VariableClock(),
        run_date="20260709",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert runner.commands == [mod.KV260_SSH_COMMAND, mod.POLARFIRE_STATUS_COMMAND]
    assert artifact["gated_upstream_ready"] is True
    assert artifact["timing_repeat_counts"]["cpu"] == mod.REPEAT_TARGET
    assert artifact["timing_repeat_counts"]["kv260"] == 0
    assert artifact["timing_repeat_counts"]["polarfire"] == 0
    assert artifact["board_result_hashes"] == {"kv260": "", "polarfire": ""}
    assert artifact["hashes_match_before_timing_compare"] is False
    assert artifact["hardware_receipts_ready"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert "blocked_board_unreachable" in artifact["readiness_blockers"]
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_verify_5463_hash_mismatch_blocks_timing_comparison() -> None:
    """REQ-VERIFY-5463: board result drift blocks comparison and claims."""

    workload = mod.extract_workload(REPO)
    outputs = [
        _board_stdout("polarfire", workload, wall_time_s=0.45 + 0.001 * index)
        for index in range(mod.REPEAT_TARGET)
    ]
    outputs[2] = _board_stdout("polarfire", workload, result_hash="1" * 64, wall_time_s=0.47)
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=_runner(kv260_reachable=False, polarfire_outputs=outputs),
        clock=VariableClock(),
        run_date="20260709",
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


def test_req_verify_5463_gate_false_fails_closed_without_board_probes(tmp_path: Path) -> None:
    """REQ-VERIFY-5463: false Exp5462 gate blocks before hardware probing."""

    source = json.loads((REPO / mod.UPSTREAM_EXP5462_RELATIVE_PATH).read_text())
    source["minimal_core_pbit_bridge_ready"] = False
    target = tmp_path / mod.UPSTREAM_EXP5462_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(source), encoding="utf-8")
    runner = RecordingRunner({})

    artifact = mod.build_artifact(
        root=tmp_path,
        command_runner=runner,
        clock=VariableClock(),
        run_date="20260709",
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
    assert artifact["boundary_exchange_ratio_summary"]["eta_values"] == []
    assert artifact["hardware_receipts_ready"] is False
    assert "minimal_core_pbit_bridge_not_ready" in artifact["readiness_blockers"]
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_verify_5463_validation_rejects_unsafe_schema_drift() -> None:
    """REQ-VERIFY-5463: validator fails closed on unsafe receipt drift."""

    artifact = mod.build_artifact(
        root=REPO,
        command_runner=_runner(kv260_reachable=False, polarfire_reachable=True),
        clock=VariableClock(),
        run_date="20260709",
        commit="abc123",
        tests_run=_tests_run(),
    )
    mod.validate_artifact(artifact)

    missing = deepcopy(artifact)
    missing.pop("boundary_exchange_ratio_summary")
    with pytest.raises(ValueError, match="missing required field"):
        mod.validate_artifact(missing)

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
            "kind": "bad_write",
            "command": "ssh polarfire 'rm -rf /tmp/carnot'",
            "command_sha256": mod.sha256_text("ssh polarfire 'rm -rf /tmp/carnot'"),
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

    bad_boundary = deepcopy(artifact)
    bad_boundary["boundary_exchange_ratio_summary"]["eta_values"] = []
    bad_boundary["hardware_receipts_ready"] = False
    with pytest.raises(ValueError, match="boundary_exchange_ratio_summary"):
        mod.validate_artifact(bad_boundary)


def test_req_verify_5463_helpers_and_repository_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-5463: helpers, writer, and checked-in artifact stay valid."""

    workload = mod.extract_workload(REPO)
    replay = mod.replay_workload(workload)
    fixture = workload["fixture_subset"][0]
    ok = mod.run_command(("sh", "-lc", "printf ok"), timeout_s=1.0)
    missing = mod.run_command(("definitely_missing_carnot_5463",), timeout_s=0.1)
    timeout = mod.run_command(("sh", "-lc", "sleep 0.2"), timeout_s=0.01)

    assert ok.exit_code == 0
    assert ok.stdout == "ok"
    assert missing.exit_code == 127
    assert timeout.exit_code == 124
    assert mod.workload_result_hash(replay) == workload["expected_result_hashes"]["aggregate"]
    assert len(workload["expected_result_hashes"]["rows"]) == (
        mod.EXPECTED_FIXTURE_COUNT * len(mod.ASSUMPTION_SOURCES)
    )
    assert mod.load_upstream_gate(tmp_path)["source_status"] == "missing"
    unreadable = tmp_path / mod.UPSTREAM_EXP5462_RELATIVE_PATH
    unreadable.parent.mkdir(parents=True, exist_ok=True)
    unreadable.write_text("{not-json", encoding="utf-8")
    assert mod.load_upstream_gate(tmp_path)["source_status"] == "unreadable"
    assert mod.timing_distribution([]) == {
        "count": 0,
        "min_s": 0.0,
        "max_s": 0.0,
        "mean_s": 0.0,
        "median_s": 0.0,
        "variance_s2": 0.0,
    }
    assert mod._pdit_value_to_assumption(fixture, "x1", "unknown") == ""
    assert mod._serialize_solution(fixture, None) is None
    assert mod._solution_valid(fixture, {"status": "unsat", "solution": None}) is False
    assert mod._solution_valid(fixture, {"status": "sat", "solution": None}) is False
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
        boundary_exchange_present=True,
    ) == ["blocked_board_unreachable"]
    assert "boundary_exchange_ratio_missing" in mod.readiness_blockers(
        gated_upstream_ready=True,
        cpu_repeat_count=mod.REPEAT_TARGET - 1,
        executable_board_count=1,
        hashes_match_before_timing_compare=False,
        boundary_exchange_present=False,
    )
    assert mod.default_tests_run()[0]["outcome"] == "pending_external_test_run"

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        workload_root=REPO,
        command_runner=_runner(kv260_reachable=False, polarfire_reachable=False),
        clock=VariableClock(),
        run_date="20260709",
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
