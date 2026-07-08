"""Tests for Exp5420 p-bit/QUBO hardware-transfer preflight.

Spec refs: REQ-HW-5420, SCENARIO-HW-5420.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5420_pbit_hardware_transfer_preflight_v493 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5420_pbit_hardware_transfer_preflight_v493.py "
    "-q --no-cov"
)


class RecordingRunner:
    """SCENARIO-HW-5420 runner with exact safe hardware command expectations."""

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
    """Deterministic clock for CPU timing receipt checks."""

    def __init__(self) -> None:
        self.value = 5420.0

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
    exact_result_hash: str | None = None,
    wall_time_s: float = 0.123,
) -> str:
    payload = {
        "board_local": True,
        "exact_best_sequence": workload["exact_result"]["exact_best_sequence"],
        "exact_min_energy": workload["exact_result"]["exact_min_energy"],
        "exact_result_hash": exact_result_hash or workload["exact_result_hash"],
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
            _board_stdout(workload, wall_time_s=0.101),
            _board_stdout(workload, wall_time_s=0.102),
            _board_stdout(workload, wall_time_s=0.103),
        ]
        probes[board_command] = [
            _probe(board_command, stdout=stdout, duration_s=0.2) for stdout in outputs
        ]
    return RecordingRunner(probes)


def test_req_hw_5420_spec_declares_preflight_contract() -> None:
    """REQ-HW-5420: OpenSpec anchors the p-bit hardware preflight contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5420") : spec.index("### SCENARIO-HW-5420")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-HW-5420",
        "SCENARIO-HW-5420",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.UPSTREAM_LNS_RELATIVE_PATH),
        str(mod.UPSTREAM_PBIT_RELATIVE_PATH),
        "active_constraint_lns_scale_ready=true",
        "PolarFire over authenticated SSH",
        "KV260 over SSH-only reachability",
        "GateMate non-destructive DirtyJTAG diagnostics",
        "hardware_speedup_claim",
        "pbit_transfer_preflight_ready",
        "hardware_preflight_with_cpu_reference",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_scenario_hw_5420_cpu_receipt_preserves_exact_enumeration() -> None:
    """SCENARIO-HW-5420: CPU reference repeats match the Exp5407 exact result."""

    assert mod.load_upstream_gate(REPO)["gate_value"] is True
    workload = mod.select_workload(REPO)
    receipt = mod.cpu_reference_receipt(workload, repeat_count=3, clock=StepClock())

    assert workload["fixture_id"] == "stress_synthetic_linear_review"
    assert workload["source_artifact"] == str(mod.UPSTREAM_PBIT_RELATIVE_PATH)
    assert workload["exact_result"]["exact_min_energy"] == 0
    assert workload["exact_result"]["exact_best_sequence"] == [
        "outline",
        "draft",
        "review",
        "submit",
    ]
    assert receipt["kind"] == "cpu_reference"
    assert receipt["seed"] == mod.RANDOM_SEED
    assert receipt["repeat_count"] == 3
    assert receipt["workload_hash"] == workload["workload_hash"]
    assert receipt["exact_result"] == workload["exact_result"]
    assert receipt["exact_enumeration_match"] is True
    assert len(receipt["repeat_timings_s"]) == 3
    assert all(row["workload_hash"] == workload["workload_hash"] for row in receipt["repeats"])


def test_scenario_hw_5420_hash_matched_receipts_ready_for_timing_not_speedup(tmp_path: Path) -> None:
    """SCENARIO-HW-5420: hash-matched CPU/board receipts gate only Exp5424 timing."""

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
    assert saved["gated_upstream_ready"] is True
    assert saved["cpu_repeat_count"] == 3
    assert saved["board_repeat_count"] == 3
    assert saved["exact_enumeration_match"] is True
    assert saved["same_workload_hash_match"] is True
    assert saved["polarfire_reachable"] is True
    assert saved["kv260_ssh_checked"] is True
    assert saved["gatemate_diagnostic_checked"] is True
    assert saved["hardware_speedup_claim"] is False
    assert saved["pbit_transfer_preflight_ready"] is True
    assert saved["inference_substrate"] == "hardware_preflight_with_cpu_reference"
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["timing_receipts"][0]["kind"] == "cpu_reference"
    assert saved["timing_receipts"][1]["kind"] == "polarfire_board_preflight"
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_hw_5420_unreachable_polarfire_blocks_board_timing_without_claim() -> None:
    """REQ-HW-5420: a missing board receipt is an honest hardware precondition block."""

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
    assert artifact["same_workload_hash_match"] is False
    assert artifact["exact_enumeration_match"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["pbit_transfer_preflight_ready"] is False
    assert artifact["readiness_blockers"] == ["polarfire_unreachable"]
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_hw_5420_hash_drift_refuses_timing_readiness_and_speedup() -> None:
    """REQ-HW-5420: repeated board timing without hash agreement cannot claim speedup."""

    workload = mod.select_workload(REPO)
    runner = _runner(
        board_stdout=[
            _board_stdout(workload, wall_time_s=0.101),
            _board_stdout(workload, workload_hash="0" * 64, wall_time_s=0.102),
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
    assert artifact["exact_enumeration_match"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["pbit_transfer_preflight_ready"] is False
    assert "same_workload_hash_mismatch" in artifact["readiness_blockers"]
    assert artifact["timing_receipts"][1]["invalid_repeat_count"] == 1
    mod.validate_artifact(artifact)


def test_req_hw_5420_validation_rejects_claim_and_storage_drift() -> None:
    """REQ-HW-5420: validation rejects speedup, exactness, and unsafe KV260 evidence."""

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

    exact = deepcopy(artifact)
    exact["exact_enumeration_match"] = False
    exact["pbit_transfer_preflight_ready"] = False
    with pytest.raises(ValueError, match="exact_enumeration_match"):
        mod.validate_artifact(exact)

    storage = deepcopy(artifact)
    storage["command_receipts"][0]["command"] = "ssh kria 'ls /dev/mmcblk*'"
    with pytest.raises(ValueError, match="host block-device"):
        mod.validate_artifact(storage)

    tests_missing = deepcopy(artifact)
    tests_missing["tests_run"] = []
    tests_missing["pbit_transfer_preflight_ready"] = False
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(tests_missing)


def test_req_hw_5420_defensive_command_and_parser_paths(tmp_path: Path) -> None:
    """REQ-HW-5420: safe wrappers preserve blocked command and parser evidence."""

    ok = mod.run_command(("sh", "-lc", "printf ok"), timeout_s=1.0)
    missing = mod.run_command(("definitely_missing_carnot_5420",), timeout_s=0.1)
    timeout = mod.run_command(("sh", "-lc", "sleep 0.2"), timeout_s=0.01)

    assert ok.exit_code == 0
    assert ok.stdout == "ok"
    assert missing.exit_code == 127
    assert timeout.exit_code == 124

    missing_gate = mod.load_upstream_gate(tmp_path)
    assert missing_gate["source_status"] == "missing"
    gate_path = tmp_path / mod.UPSTREAM_LNS_RELATIVE_PATH
    gate_path.parent.mkdir(parents=True)
    gate_path.write_text("{not-json", encoding="utf-8")
    unreadable_gate = mod.load_upstream_gate(tmp_path)
    assert unreadable_gate["source_status"] == "unreadable"

    workload = mod.select_workload(REPO)
    assert mod.precedence_energy(workload, ["outline"]) > 0
    receipt, error = mod.parse_board_workload_stdout("\nnot-json\n[]\n", workload)
    assert receipt is None
    assert error == "workload stdout is not valid JSON"

    bad_payload = {
        "board_local": False,
        "exact_best_sequence": ["submit", "review", "draft", "outline"],
        "exact_min_energy": 10,
        "exact_result_hash": "1" * 64,
        "seed": 0,
        "wall_time_s": -1.0,
        "workload_hash": "2" * 64,
    }
    receipt, error = mod.parse_board_workload_stdout(
        json.dumps(bad_payload, sort_keys=True),
        workload,
    )
    assert isinstance(receipt, dict)
    assert error is not None
    for marker in (
        "workload_hash mismatch",
        "seed mismatch",
        "exact_min_energy mismatch",
        "exact_best_sequence mismatch",
        "exact_result_hash mismatch",
        "board_local missing",
        "wall_time_s invalid",
    ):
        assert marker in error
    assert mod._board_exact_matches([{"receipt": receipt, "exact_match": False}]) is False


def test_req_hw_5420_false_gate_blocks_before_hardware_and_defaults_tests(tmp_path: Path) -> None:
    """REQ-HW-5420: a false upstream gate fails fast without board commands."""

    gate_path = tmp_path / mod.UPSTREAM_LNS_RELATIVE_PATH
    gate_path.parent.mkdir(parents=True)
    gate_path.write_text(
        json.dumps({"active_constraint_lns_scale_ready": False, "status": "blocked"}),
        encoding="utf-8",
    )
    runner = RecordingRunner({})

    artifact = mod.build_artifact(root=tmp_path, command_runner=runner, clock=StepClock())

    assert runner.commands == []
    assert artifact["gated_upstream_ready"] is False
    assert artifact["kv260_ssh_checked"] is False
    assert artifact["gatemate_diagnostic_checked"] is False
    assert artifact["tests_run"] == mod.default_tests_run()
    assert "active_constraint_lns_scale_not_ready" in artifact["readiness_blockers"]
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_scenario_hw_5420_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-5420: run_experiment writes the requested deliverable path."""

    out_path = mod.run_experiment(
        repo_root=tmp_path,
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
