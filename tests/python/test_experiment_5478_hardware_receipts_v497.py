"""Tests for Exp5478 Exp5477 hardware receipts.

Spec refs: REQ-VERIFY-5478, SCENARIO-VERIFY-5478.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5478_hardware_receipts_v497 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5478_hardware_receipts_v497.py -q --no-cov"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5478_hardware_receipts_v497.py "
    "-m pytest tests/python/test_experiment_5478_hardware_receipts_v497.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report "
    "--include=python/carnot/experiment_5478_hardware_receipts_v497.py "
    "--fail-under=100"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
E2E_COMMAND = (
    "ops/e2e-test-plan.md review: Exp5478 collects local CPU and SSH-safe "
    "reachable-board receipts only; no live training, PyO3, or destructive "
    "hardware path applies"
)


class RecordingRunner:
    """SCENARIO-VERIFY-5478 fake runner preserving exact command order."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        assert timeout_s > 0.0
        self.commands.append(command)
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


class VariableClock:
    """Deterministic nonzero clock so timing distributions have variance."""

    def __init__(self) -> None:
        self.value = 5478.0
        self.increments = [0.00021 + 0.00001 * index for index in range(80)]
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
    result_hashes: list[str] | None = None,
    workload_hashes: list[str] | None = None,
    wall_time_s: float = 0.25,
) -> str:
    hashes = result_hashes or list(workload["cpu_reference_hashes"])
    payload = {
        "aggregate_result_hash": mod.aggregate_result_hash(hashes),
        "board_identity": board,
        "board_local": True,
        "result_hashes": hashes,
        "wall_time_s": wall_time_s,
        "workload_hashes": workload_hashes or workload["workload_hashes"],
    }
    return json.dumps(payload, sort_keys=True) + "\n"


def _runner(
    *,
    kv260_reachable: bool = False,
    polarfire_reachable: bool = True,
    kv260_outputs: list[str] | None = None,
    polarfire_outputs: list[str] | None = None,
) -> RecordingRunner:
    workload = mod.load_exp5477_workloads(REPO)
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
            _board_stdout("kv260", workload, wall_time_s=0.33 + 0.001 * index)
            for index in range(mod.REPEAT_TARGET)
        ]
        probes[kv_command] = [_probe(kv_command, stdout=stdout) for stdout in outputs]
    if polarfire_reachable:
        outputs = polarfire_outputs or [
            _board_stdout("polarfire", workload, wall_time_s=0.43 + 0.001 * index)
            for index in range(mod.REPEAT_TARGET)
        ]
        probes[pf_command] = [_probe(pf_command, stdout=stdout) for stdout in outputs]
    return RecordingRunner(probes)


def test_req_verify_5478_spec_declares_exp5477_hardware_receipt_contract() -> None:
    """REQ-VERIFY-5478: OpenSpec anchors Exp5477 receipt collection."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5478") : spec.index("### REQ-VERIFY-5477")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5478",
        "SCENARIO-VERIFY-5478",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp5477 workload hashes",
        "KV260 by SSH reachability only",
        "PolarFire by SSH reachability",
        "GateMate SHALL be diagnostic-only",
        mod.INFERENCE_SUBSTRATE,
        "hardware_speedup_claim",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_req_verify_5478_loads_exp5477_hashes_and_cpu_reference_hashes() -> None:
    """REQ-VERIFY-5478: CPU hashes are stable and tied to Exp5477 workloads."""

    workload = mod.load_exp5477_workloads(REPO)
    second = mod.load_exp5477_workloads(REPO)
    cpu = mod.cpu_reference_receipt(workload, repeat_count=mod.REPEAT_TARGET, clock=VariableClock())

    assert workload == second
    assert workload["workload_hashes"] == workload["upstream_workload_hashes"]
    assert len(workload["workload_hashes"]) == mod.EXPECTED_WORKLOAD_COUNT
    assert len(workload["cpu_reference_hashes"]) == mod.EXPECTED_WORKLOAD_COUNT
    assert all(len(item) == 64 for item in workload["workload_hashes"])
    assert all(len(item) == 64 for item in workload["cpu_reference_hashes"])
    assert cpu["result_hashes"] == workload["cpu_reference_hashes"]
    assert cpu["repeat_count"] == mod.REPEAT_TARGET
    assert cpu["stable_result_hashes"] is True
    assert cpu["timing_distribution"]["count"] == mod.REPEAT_TARGET


def test_scenario_verify_5478_polarfire_receipts_match_cpu_hashes(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5478: reachable board receipts match CPU before readiness."""

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
    workload = mod.load_exp5477_workloads(REPO)
    pf_command = mod.board_workload_command("polarfire", workload)

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.POLARFIRE_STATUS_COMMAND,
        *([pf_command] * mod.REPEAT_TARGET),
    ]
    assert saved["upstream_workload_hashes"] == workload["workload_hashes"]
    assert saved["cpu_reference_hashes"] == workload["cpu_reference_hashes"]
    assert saved["reachable_boards"] == ["polarfire"]
    assert {item["board_identity"] for item in saved["unreachable_boards"]} == {
        "kv260",
        "gatemate",
    }
    assert len(saved["board_receipts"]) == 1
    receipt = saved["board_receipts"][0]
    assert receipt["board_identity"] == "polarfire"
    assert receipt["workload_hashes"] == workload["workload_hashes"]
    assert receipt["result_hashes"] == workload["cpu_reference_hashes"]
    assert receipt["repeat_count"] == mod.REPEAT_TARGET
    assert receipt["timing_distribution"]["count"] == mod.REPEAT_TARGET
    assert receipt["stdout_sha256"]
    assert receipt["stderr_sha256"] == mod.sha256_text("")
    assert saved["repeat_count"] == mod.REPEAT_TARGET
    assert saved["timing_summary"]["cpu"]["count"] == mod.REPEAT_TARGET
    assert saved["timing_summary"]["polarfire"]["count"] == mod.REPEAT_TARGET
    assert saved["result_hash_match_rate"] == pytest.approx(1.0)
    assert saved["hardware_receipts_ready"] is True
    assert saved["hardware_speedup_claim"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["research_conductor_modified"] is False
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_verify_5478_unreachable_boards_keep_cpu_receipts() -> None:
    """REQ-VERIFY-5478: unreachable boards block readiness without losing CPU receipts."""

    artifact = mod.build_artifact(
        root=REPO,
        command_runner=_runner(kv260_reachable=False, polarfire_reachable=False),
        clock=VariableClock(),
        run_date="20260709",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert artifact["cpu_reference_hashes"] == mod.load_exp5477_workloads(REPO)[
        "cpu_reference_hashes"
    ]
    assert artifact["reachable_boards"] == []
    assert {item["board_identity"] for item in artifact["unreachable_boards"]} == {
        "kv260",
        "polarfire",
        "gatemate",
    }
    assert artifact["board_receipts"] == []
    assert artifact["result_hash_match_rate"] == 0.0
    assert artifact["hardware_receipts_ready"] is False
    assert "no_reachable_workload_board" in artifact["readiness_blockers"]
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_verify_5478_hash_mismatch_blocks_receipt_readiness() -> None:
    """REQ-VERIFY-5478: board result drift blocks readiness and speedup claims."""

    workload = mod.load_exp5477_workloads(REPO)
    outputs = [
        _board_stdout("polarfire", workload, wall_time_s=0.43 + 0.001 * index)
        for index in range(mod.REPEAT_TARGET)
    ]
    bad_hashes = list(workload["cpu_reference_hashes"])
    bad_hashes[1] = "1" * 64
    outputs[1] = _board_stdout("polarfire", workload, result_hashes=bad_hashes, wall_time_s=0.44)

    artifact = mod.build_artifact(
        root=REPO,
        command_runner=_runner(kv260_reachable=False, polarfire_outputs=outputs),
        clock=VariableClock(),
        run_date="20260709",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert artifact["reachable_boards"] == ["polarfire"]
    assert artifact["board_receipts"][0]["matched_repeat_count"] == mod.REPEAT_TARGET - 1
    assert artifact["result_hash_match_rate"] == pytest.approx(2 / 3)
    assert artifact["hardware_receipts_ready"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert "board_hash_mismatch" in artifact["readiness_blockers"]
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_verify_5478_validation_rejects_unsafe_schema_drift() -> None:
    """REQ-VERIFY-5478: validator fails closed on unsafe receipt drift."""

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
    missing.pop("cpu_reference_hashes")
    with pytest.raises(ValueError, match="missing required field"):
        mod.validate_artifact(missing)

    speedup = deepcopy(artifact)
    speedup["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(speedup)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "hardware_sampler"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    promoted_gatemate = deepcopy(artifact)
    promoted_gatemate["board_receipts"][0]["board_identity"] = "gatemate"
    promoted_gatemate["reachable_boards"] = ["gatemate"]
    promoted_gatemate["hardware_receipts_ready"] = False
    with pytest.raises(ValueError, match="diagnostic-only"):
        mod.validate_artifact(promoted_gatemate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["board_receipts"][0]["stdout_sha256"] = "2" * 64
    bad_checksum["hardware_receipts_ready"] = False
    with pytest.raises(ValueError, match="stdout_sha256"):
        mod.validate_artifact(bad_checksum)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    bad_tests["hardware_receipts_ready"] = False
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(bad_tests)


def test_req_verify_5478_helpers_writer_and_repository_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-5478: helpers, writer, and checked-in artifact stay valid."""

    workload = mod.load_exp5477_workloads(REPO)
    command = mod.board_workload_command("polarfire", workload)
    ok = mod.run_command(("sh", "-lc", "printf ok"), timeout_s=1.0)
    missing = mod.run_command(("definitely_missing_carnot_5478",), timeout_s=0.1)
    timeout = mod.run_command(("sh", "-lc", "sleep 0.2"), timeout_s=0.01)

    assert ok.exit_code == 0
    assert ok.stdout == "ok"
    assert missing.exit_code == 127
    assert timeout.exit_code == 124
    assert command[0] == "ssh"
    assert mod.timing_distribution([]) == {
        "count": 0,
        "min_s": 0.0,
        "max_s": 0.0,
        "mean_s": 0.0,
        "median_s": 0.0,
        "variance_s2": 0.0,
    }
    assert mod.parse_board_workload_stdout("\nnot json\n[]\n", workload, "polarfire") == (
        None,
        "workload stdout is not valid JSON",
    )

    bad_payload = json.loads(_board_stdout("polarfire", workload))
    bad_payload.update(
        {
            "aggregate_result_hash": "4" * 64,
            "board_identity": "wrong",
            "board_local": False,
            "result_hashes": ["2" * 64],
            "wall_time_s": -0.1,
            "workload_hashes": ["3" * 64],
        }
    )
    _, parse_error = mod.parse_board_workload_stdout(
        json.dumps(bad_payload, sort_keys=True),
        workload,
        "polarfire",
    )
    assert parse_error is not None
    for marker in (
        "board_identity mismatch",
        "workload_hashes mismatch",
        "result_hashes mismatch",
        "aggregate_result_hash mismatch",
        "board_local missing",
        "wall_time_s invalid",
    ):
        assert marker in parse_error

    assert mod.readiness_blockers(
        cpu_stable=False,
        reachable_workload_count=0,
        match_rate=0.0,
    ) == ["cpu_reference_unstable", "no_reachable_workload_board"]
    assert mod.readiness_blockers(
        cpu_stable=True,
        reachable_workload_count=1,
        match_rate=1.0,
        upstream_hash_match=False,
    ) == ["upstream_workload_hash_mismatch"]
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
