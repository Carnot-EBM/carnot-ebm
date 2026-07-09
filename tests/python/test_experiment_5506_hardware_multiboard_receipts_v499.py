"""Tests for Exp5506 multi-board hardware smoke receipts.

Spec refs: REQ-VERIFY-5506, SCENARIO-VERIFY-5506.
"""

from __future__ import annotations

from copy import deepcopy
import builtins
import json
from pathlib import Path
import sys

import pytest

from carnot import experiment_5506_hardware_multiboard_receipts_v499 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5506_hardware_multiboard_receipts_v499.py")


class RecordingRunner:
    """SCENARIO-VERIFY-5506 fake command runner preserving command transcripts."""

    def __init__(self, probes: dict[tuple[str, ...], mod.CommandProbe]) -> None:
        self.probes = dict(probes)
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        assert timeout_s > 0.0
        self.commands.append(command)
        if command not in self.probes:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command]


class VariableClock:
    """Deterministic nonzero clock so smoke receipt durations are stable in tests."""

    def __init__(self) -> None:
        self.value = 5506.0

    def __call__(self) -> float:
        self.value += 0.001
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


def _tests_run() -> list[dict[str, str]]:
    return [{"command": TEST_PATH.as_posix(), "outcome": "passed"}]


def _identity_stdout(board: str, machine: str) -> str:
    return f"board_identity={board}\nhostname={board}-local\nmachine={machine}\n"


def _runner_for_reachable_polarfire() -> RecordingRunner:
    source = mod.load_descriptor_source(REPO)
    workload = mod.build_smoke_workload(source)
    cpu_command = mod.local_smoke_command("cpu", workload)
    cuda_command = mod.local_smoke_command("cuda", workload)
    polar_command = mod.remote_smoke_command("polarfire", workload)
    return RecordingRunner(
        {
            cpu_command: _probe(
                cpu_command,
                stdout=mod.smoke_receipt_stdout("cpu", workload, runtime="python_cpu"),
            ),
            cuda_command: _probe(
                cuda_command,
                exit_code=43,
                stdout="cuda_available=false\n",
                stderr="torch CUDA device unavailable\n",
            ),
            mod.KV260_IDENTITY_COMMAND: _probe(
                mod.KV260_IDENTITY_COMMAND,
                exit_code=255,
                stderr="ssh: no route to host\n",
            ),
            mod.GATEMATE_DETECT_COMMAND: _probe(
                mod.GATEMATE_DETECT_COMMAND,
                exit_code=127,
                stderr="openFPGALoader: not found\n",
            ),
            mod.POLARFIRE_IDENTITY_COMMAND: _probe(
                mod.POLARFIRE_IDENTITY_COMMAND,
                stdout=_identity_stdout("polarfire", "riscv64"),
            ),
            polar_command: _probe(
                polar_command,
                stdout=mod.smoke_receipt_stdout("polarfire", workload, runtime="remote_python"),
            ),
        }
    )


def test_req_verify_5506_spec_declares_multiboard_receipt_contract() -> None:
    """REQ-VERIFY-5506: OpenSpec anchors required fields and safe identity paths."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5506") : spec.index("### REQ-VERIFY-5462")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-VERIFY-5506",
        str(mod.RESULT_RELATIVE_PATH),
        "results/experiment_5505_active_constraint_milp_descriptor_v499.json",
        "Exp 5491 descriptor",
        "ssh polarfire",
        "ssh kria",
        "openFPGALoader -c dirtyJtag --detect",
        "host `/dev/mmcblk*`",
        "hardware_smoke",
        "hardware_speedup_claim",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for status in mod.STATUS_VALUES:
        assert status in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_verify_5506_selects_exp5505_when_ready_and_exp5491_fallback(tmp_path: Path) -> None:
    """REQ-VERIFY-5506: descriptor source selection records primary and fallback sources."""

    primary = mod.load_descriptor_source(REPO)
    primary_workload = mod.build_smoke_workload(primary)

    assert primary["descriptor_source"] == mod.EXP5505_RELATIVE_PATH.as_posix()
    assert primary["descriptor_source_ready"] is True
    assert primary["fallback_used"] is False
    assert len(primary_workload["descriptor_smokes"]) == mod.SMOKE_DESCRIPTOR_LIMIT
    assert primary_workload["descriptor_source"] == primary["descriptor_source"]
    assert len(primary_workload["aggregate_input_hash"]) == 64
    assert len(primary_workload["aggregate_expected_output_hash"]) == 64

    fallback_payload = json.loads(
        (REPO / mod.EXP5491_FALLBACK_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    fallback_path = tmp_path / mod.EXP5491_FALLBACK_RELATIVE_PATH
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_text(json.dumps(fallback_payload), encoding="utf-8")

    fallback = mod.load_descriptor_source(tmp_path)
    assert fallback["descriptor_source"] == mod.EXP5491_FALLBACK_RELATIVE_PATH.as_posix()
    assert fallback["descriptor_source_ready"] is True
    assert fallback["fallback_used"] is True
    assert (
        len(mod.build_smoke_workload(fallback)["descriptor_smokes"]) == mod.SMOKE_DESCRIPTOR_LIMIT
    )


def test_scenario_verify_5506_builds_honest_multiboard_receipts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5506: reachable PolarFire and CPU match hashes without speedup."""

    runner = _runner_for_reachable_polarfire()
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=runner,
        clock=VariableClock(),
        tests_run=_tests_run(),
    )
    out_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    source = mod.load_descriptor_source(REPO)
    workload = mod.build_smoke_workload(source)

    assert saved == artifact
    assert runner.commands == [
        mod.local_smoke_command("cpu", workload),
        mod.local_smoke_command("cuda", workload),
        mod.KV260_IDENTITY_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_IDENTITY_COMMAND,
        mod.remote_smoke_command("polarfire", workload),
    ]
    assert all("/dev/mmcblk" not in mod.command_to_string(command) for command in runner.commands)
    assert all("flash" not in mod.command_to_string(command).lower() for command in runner.commands)
    assert saved["descriptor_source"] == mod.EXP5505_RELATIVE_PATH.as_posix()
    assert saved["descriptor_source_ready"] is True
    assert saved["cpu_status"] == "reachable"
    assert saved["cuda_status"] == "blocked_toolchain"
    assert saved["kv260_status"] == "blocked_identity"
    assert saved["gatemate_status"] == "blocked_toolchain"
    assert saved["polar_fire_status"] == "reachable"
    assert saved["matched_timing_available"] is False
    assert saved["hardware_speedup_claim"] is False
    assert saved["conductor_unchanged"] is True
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert {row["substrate"] for row in saved["matched_hashes"]} == {"cpu", "polarfire"}
    assert all(row["matched"] is True for row in saved["matched_hashes"])
    assert all("exit_code" in receipt for receipt in saved["command_receipts"])
    assert all("stdout_sha256" in receipt for receipt in saved["command_receipts"])
    assert any(
        receipt.get("blocked_reason") == "cuda_unavailable" for receipt in saved["command_receipts"]
    )
    assert any(
        receipt.get("blocked_reason") == "blocked_kv260_ssh_identity"
        for receipt in saved["command_receipts"]
    )
    assert any(
        receipt.get("blocked_reason") == "gatemate_toolchain_unavailable"
        for receipt in saved["command_receipts"]
    )
    mod.validate_artifact(saved)


def test_req_verify_5506_validation_fails_closed_on_overclaim_and_unsafe_commands() -> None:
    """REQ-VERIFY-5506: validation rejects speedup, conductor drift, and unsafe probes."""

    artifact = mod.build_artifact(
        root=REPO,
        command_runner=_runner_for_reachable_polarfire(),
        clock=VariableClock(),
        tests_run=_tests_run(),
    )
    mod.validate_artifact(artifact)

    speedup = deepcopy(artifact)
    speedup["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(speedup)

    timing = deepcopy(artifact)
    timing["matched_timing_available"] = True
    with pytest.raises(ValueError, match="matched_timing_available"):
        mod.validate_artifact(timing)

    conductor = deepcopy(artifact)
    conductor["conductor_unchanged"] = False
    with pytest.raises(ValueError, match="conductor_unchanged"):
        mod.validate_artifact(conductor)

    storage_probe = deepcopy(artifact)
    storage_probe["command_receipts"][0]["command"] = "ls /dev/mmcblk0"
    storage_probe["command_receipts"][0]["command_sha256"] = mod.sha256_text("ls /dev/mmcblk0")
    storage_probe["reproducibility_checksum"] = mod.payload_checksum(storage_probe)
    with pytest.raises(ValueError, match="host storage command"):
        mod.validate_artifact(storage_probe)

    flash_probe = deepcopy(artifact)
    flash_probe["command_receipts"][0]["command"] = "openFPGALoader --write flash.bit"
    flash_probe["command_receipts"][0]["command_sha256"] = mod.sha256_text(
        "openFPGALoader --write flash.bit"
    )
    flash_probe["reproducibility_checksum"] = mod.payload_checksum(flash_probe)
    with pytest.raises(ValueError, match="destructive command"):
        mod.validate_artifact(flash_probe)


def test_req_verify_5506_blocked_descriptor_source_still_writes_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5506: missing descriptors produce blocked statuses instead of overclaims."""

    runner = RecordingRunner({})
    artifact = mod.build_artifact(
        root=tmp_path,
        command_runner=runner,
        clock=VariableClock(),
        tests_run=_tests_run(),
    )

    assert runner.commands == []
    assert artifact["descriptor_source"] == "missing"
    assert artifact["descriptor_source_ready"] is False
    assert artifact["cpu_status"] == "blocked_descriptor"
    assert artifact["cuda_status"] == "blocked_descriptor"
    assert artifact["kv260_status"] == "blocked_descriptor"
    assert artifact["gatemate_status"] == "blocked_descriptor"
    assert artifact["polar_fire_status"] == "blocked_descriptor"
    assert artifact["command_receipts"] == []
    assert artifact["matched_hashes"] == []
    assert artifact["matched_timing_available"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_verify_5506_covers_defensive_command_and_parser_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5506: defensive branches preserve blocked receipts without claims."""

    ok_probe = mod.run_command((sys.executable, "-c", "print('ok')"), timeout_s=5.0)
    missing_probe = mod.run_command(("definitely-missing-carnot-exp5506-bin",), timeout_s=0.01)
    timeout_probe = mod.run_command(
        (sys.executable, "-c", "import time; time.sleep(1)"),
        timeout_s=0.01,
    )
    assert ok_probe.exit_code == 0
    assert "ok" in ok_probe.stdout
    assert missing_probe.exit_code == 127
    assert timeout_probe.exit_code == 124

    malformed_path = tmp_path / mod.EXP5505_RELATIVE_PATH
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_path.write_text("{not-json", encoding="utf-8")
    assert mod.load_descriptor_source(tmp_path)["descriptor_source"] == "missing"

    bad_primary = {"descriptor_ready_for_hardware": False, "readiness_blockers": []}
    bad_fallback = {"subproblem_descriptor_ready": False, "readiness_blockers": []}
    malformed_path.write_text(json.dumps(bad_primary), encoding="utf-8")
    fallback_path = tmp_path / mod.EXP5491_FALLBACK_RELATIVE_PATH
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_text(json.dumps(bad_fallback), encoding="utf-8")
    blocked_source = mod.load_descriptor_source(tmp_path)
    assert blocked_source["descriptor_source"] == "missing"
    assert "exp5491_fallback_not_ready" in blocked_source["readiness_blockers"]
    assert "exp5505_descriptor_not_ready" in blocked_source["readiness_blockers"]

    source = mod.load_descriptor_source(REPO)
    workload = mod.build_smoke_workload(source)
    assert (
        mod.smoke_receipt("cpu", workload, runtime="python_cpu", wall_time_s=0.1, extra={"x": 1})[
            "x"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="local substrate"):
        mod.local_smoke_command("fpga", workload)
    with pytest.raises(ValueError, match="remote board"):
        mod.remote_smoke_command("kv260", workload)

    assert mod.parse_smoke_stdout("", substrate="cpu", workload=workload) == (
        None,
        "smoke stdout is not valid JSON",
    )
    assert (
        mod.parse_smoke_stdout(
            "\n" + mod.smoke_receipt_stdout("cpu", workload, runtime="python_cpu"),
            substrate="cpu",
            workload=workload,
        )[1]
        is None
    )
    mismatch = json.loads(mod.smoke_receipt_stdout("cpu", workload, runtime="python_cpu"))
    mismatch["substrate"] = "cuda"
    mismatch["descriptor_count"] = 999
    mismatch["wall_time_s"] = -1
    mismatch["aggregate_input_hash"] = "bad"
    receipt, parse_error = mod.parse_smoke_stdout(
        "not-json\n" + json.dumps(mismatch),
        substrate="cpu",
        workload=workload,
    )
    assert receipt is not None
    assert "substrate mismatch" in str(parse_error)
    assert "aggregate_input_hash mismatch" in str(parse_error)
    assert "descriptor_count mismatch" in str(parse_error)
    assert "wall_time_s invalid" in str(parse_error)

    cpu_bad = _probe(mod.local_smoke_command("cpu", workload), exit_code=127)
    assert mod._classify_smoke_probe(  # noqa: SLF001 - tests pin defensive branch behavior.
        cpu_bad,
        substrate="cpu",
        workload=workload,
        unavailable_reason="cpu_missing",
    ) == ("blocked_toolchain", None, "cpu_missing")
    cpu_parse_bad = _probe(mod.local_smoke_command("cpu", workload), exit_code=1, stdout="bad")
    status, parsed, blocker = mod._classify_smoke_probe(  # noqa: SLF001
        cpu_parse_bad,
        substrate="cpu",
        workload=workload,
        unavailable_reason="cpu_missing",
    )
    assert status == "blocked_toolchain"
    assert parsed is None
    assert blocker == "smoke stdout is not valid JSON"
    assert (
        mod._cuda_blocked_reason(  # noqa: SLF001
            _probe(
                mod.local_smoke_command("cuda", workload),
                exit_code=42,
                stdout="torch_import_failed=ImportError",
            )
        )
        == "cuda_toolchain_unavailable"
    )
    assert (
        mod._cuda_blocked_reason(  # noqa: SLF001
            _probe(mod.local_smoke_command("cuda", workload), exit_code=99, stderr="other")
        )
        == "cuda_smoke_failed"
    )
    assert mod.parse_identity_stdout("noise\nboard_identity=kv260\n") == {"board_identity": "kv260"}
    assert mod._identity_status(  # noqa: SLF001
        _probe(mod.KV260_IDENTITY_COMMAND, stdout="board_identity=wrong\n"),
        board="kv260",
    ) == ("blocked_identity", "blocked_kv260_ssh_identity")
    assert mod._gatemate_status(  # noqa: SLF001
        _probe(mod.GATEMATE_DETECT_COMMAND, stdout="IDCODE 0x1 GateMate\n")
    ) == ("reachable", None)
    assert mod._gatemate_status(  # noqa: SLF001
        _probe(mod.GATEMATE_DETECT_COMMAND, exit_code=1, stderr="scan failed")
    ) == ("blocked_identity", "blocked_gatemate_dirtyjtag_identity")
    assert mod._matched_hash_rows("cpu", None, workload) == []  # noqa: SLF001
    mismatch_receipt = json.loads(mod.smoke_receipt_stdout("cpu", workload, runtime="python_cpu"))
    mismatch_receipt["aggregate_expected_output_hash"] = "bad"
    assert mod._matched_hash_rows("cpu", mismatch_receipt, workload) == []  # noqa: SLF001

    polar_blocked_runner = RecordingRunner(
        {
            mod.local_smoke_command("cpu", workload): _probe(
                mod.local_smoke_command("cpu", workload),
                stdout=mod.smoke_receipt_stdout("cpu", workload, runtime="python_cpu"),
            ),
            mod.local_smoke_command("cuda", workload): _probe(
                mod.local_smoke_command("cuda", workload),
                exit_code=43,
                stdout="cuda_available=false\n",
            ),
            mod.KV260_IDENTITY_COMMAND: _probe(
                mod.KV260_IDENTITY_COMMAND,
                stdout=_identity_stdout("kv260", "aarch64"),
            ),
            mod.GATEMATE_DETECT_COMMAND: _probe(
                mod.GATEMATE_DETECT_COMMAND,
                stdout="IDCODE 0x1 GateMate\n",
            ),
            mod.POLARFIRE_IDENTITY_COMMAND: _probe(
                mod.POLARFIRE_IDENTITY_COMMAND,
                exit_code=255,
                stderr="ssh failed\n",
            ),
        }
    )
    collected = mod.collect_receipts(workload=workload, command_runner=polar_blocked_runner)
    assert collected["polar_fire_status"] == "blocked_identity"
    assert mod.readiness_blockers(
        descriptor_ready=True,
        cpu_status="blocked_toolchain",
        source_blockers=[],
    ) == ["cpu_descriptor_smoke_not_reachable"]
    assert (
        mod.honest_verdict(
            ready=False,
            descriptor_ready=True,
            statuses={"cpu_status": "blocked_toolchain"},
            blockers=[],
        )
        == "blocked: cpu_descriptor_smoke_not_reachable; hardware_speedup_claim=false"
    )
    assert mod._normalize_tests(None) == [  # noqa: SLF001
        {"command": "verification not yet attached", "outcome": "pending"}
    ]

    class DirtyResult:
        returncode = 1

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: DirtyResult())
    assert mod.conductor_unchanged(REPO) is False


def test_req_verify_5506_emit_local_smoke_branches(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5506: local smoke CLI emits bounded receipts and blocked exits."""

    source = mod.load_descriptor_source(REPO)
    workload = mod.build_smoke_workload(source)
    payload = mod._workload_command_payload(workload)  # noqa: SLF001

    assert mod.emit_local_smoke("cpu", payload) == 0
    cpu_receipt = json.loads(capsys.readouterr().out)
    assert cpu_receipt["substrate"] == "cpu"
    assert cpu_receipt["runtime"] == "python_cpu"

    assert mod.emit_local_smoke("unknown", payload) == 2
    assert "unknown_substrate" in capsys.readouterr().out

    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert mod.emit_local_smoke("cuda", payload) == 42
    assert "gpu_runtime_import_failed=ImportError" in capsys.readouterr().out
    monkeypatch.setattr(builtins, "__import__", real_import)

    class FakeUnavailableCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeUnavailableTorch:
        cuda = FakeUnavailableCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeUnavailableTorch())
    assert mod.emit_local_smoke("cuda", payload) == 43
    assert "gpu_available=false" in capsys.readouterr().out

    class FakeTensor:
        def __init__(self, value: int) -> None:
            self.value = value

        def cpu(self) -> "FakeTensor":
            return self

        def item(self) -> int:
            return self.value

    class FakeAvailableCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def get_device_name(device: str) -> str:
            assert device == "cuda"
            return "fake-cuda"

    class FakeAvailableTorch:
        cuda = FakeAvailableCuda()

        @staticmethod
        def device(name: str) -> str:
            assert name == "cuda"
            return name

        @staticmethod
        def tensor(values: list[int], *, device: str) -> FakeTensor:
            assert device == "cuda"
            return FakeTensor(values[0])

    monkeypatch.setitem(sys.modules, "torch", FakeAvailableTorch())
    assert mod.emit_local_smoke("cuda", payload) == 0
    cuda_receipt = json.loads(capsys.readouterr().out)
    assert cuda_receipt["substrate"] == "cuda"
    assert cuda_receipt["runtime"] == "local_gpu_smoke"
    assert cuda_receipt["gpu_device"] == "fake-cuda"
    assert cuda_receipt["gpu_tensor_value"] == len(payload["descriptor_smokes"])


def test_req_verify_5506_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-5506: run_experiment writes the terminal artifact path."""

    artifact_path = mod.run_experiment(
        repo_root=tmp_path,
        descriptor_root=REPO,
        command_runner=_runner_for_reachable_polarfire(),
        clock=VariableClock(),
        tests_run=_tests_run(),
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert artifact["descriptor_source"] == mod.EXP5505_RELATIVE_PATH.as_posix()
    assert artifact["hardware_speedup_claim"] is False
    mod.validate_artifact(artifact)
