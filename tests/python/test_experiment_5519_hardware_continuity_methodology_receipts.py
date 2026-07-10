"""Tests for Exp5519 hardware continuity and timing-methodology receipts.

Spec refs: REQ-VERIFY-5519, SCENARIO-VERIFY-5519.
"""

from __future__ import annotations

from copy import deepcopy
import builtins
import json
from pathlib import Path
import sys

import pytest

from carnot import experiment_5519_hardware_continuity_methodology_receipts as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path(
    "tests/python/test_experiment_5519_hardware_continuity_methodology_receipts.py"
)


class RecordingRunner:
    """SCENARIO-VERIFY-5519 fake command runner preserving command order."""

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
    """Deterministic clock so artifact durations are stable in tests."""

    def __init__(self) -> None:
        self.value = 5519.0

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


def _json_line(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True) + "\n"


def _tests_run() -> list[dict[str, str]]:
    return [{"command": TEST_PATH.as_posix(), "outcome": "passed"}]


def _reachable_runner() -> RecordingRunner:
    cpu_payload = {
        "status": "reachable",
        "device_names": ["AMD Ryzen AI 9 HX 370"],
        "driver_versions": {},
        "runtime_versions": {"python": "3.12.0", "platform": "Linux"},
        "metadata": {"machine": "x86_64"},
    }
    cuda_payload = {
        "status": "reachable",
        "device_names": ["NVIDIA GeForce RTX 3090"],
        "driver_versions": {},
        "runtime_versions": {"torch": "2.9.0+cu128", "cuda": "12.8"},
        "metadata": {"device_count": 1},
    }
    return RecordingRunner(
        {
            mod.CPU_INFO_COMMAND: _probe(mod.CPU_INFO_COMMAND, stdout=_json_line(cpu_payload)),
            mod.CUDA_INFO_COMMAND: _probe(mod.CUDA_INFO_COMMAND, stdout=_json_line(cuda_payload)),
            mod.NVIDIA_SMI_QUERY_COMMAND: _probe(
                mod.NVIDIA_SMI_QUERY_COMMAND,
                stdout="NVIDIA GeForce RTX 3090, 575.57.08\n",
            ),
            mod.POLARFIRE_IDENTITY_COMMAND: _probe(
                mod.POLARFIRE_IDENTITY_COMMAND,
                stdout=(
                    "board_identity=polarfire\n"
                    "hostname=mpfs-disco-kit\n"
                    "machine=riscv64\n"
                    "kernel=6.18.17-linux4microchip-2026.04.1\n"
                    "model=Microchip PolarFire SoC Discovery Kit\n"
                    "firmware_sha256="
                    + "a" * 64
                    + " /lib/firmware/polarfire/carnot.bit\n"
                ),
            ),
            mod.KV260_IDENTITY_COMMAND: _probe(
                mod.KV260_IDENTITY_COMMAND,
                stdout="board_identity=kv260\nhostname=kria\nmachine=aarch64\n",
            ),
            mod.KV260_XMUTIL_COMMAND: _probe(
                mod.KV260_XMUTIL_COMMAND,
                stdout="carnot_ising_v2_n64 loaded\n",
            ),
            mod.KV260_UIO_COMMAND: _probe(
                mod.KV260_UIO_COMMAND,
                stdout="/dev/uio0\n/dev/uio4\n",
            ),
            mod.GATEMATE_DETECT_COMMAND: _probe(
                mod.GATEMATE_DETECT_COMMAND,
                stdout="IDCODE 0x20000000 colognechip GateMate Series GM1A\n",
            ),
            mod.YOSYS_VERSION_COMMAND: _probe(mod.YOSYS_VERSION_COMMAND, stdout="Yosys 0.64\n"),
            mod.NEXTPNR_VERSION_COMMAND: _probe(
                mod.NEXTPNR_VERSION_COMMAND,
                stdout="nextpnr-himbaechel 0.8\n",
            ),
            mod.GMPACK_VERSION_COMMAND: _probe(mod.GMPACK_VERSION_COMMAND, stdout="gmpack 2026\n"),
        }
    )


def test_req_verify_5519_spec_declares_receipt_contract() -> None:
    """REQ-VERIFY-5519: OpenSpec anchors fields, safe paths, and no-speedup gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5519") : spec.index("### REQ-VERIFY-5506")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-VERIFY-5519",
        str(mod.RESULT_RELATIVE_PATH),
        "ssh polarfire",
        "ssh kria",
        "xmutil",
        "remote `/dev/uio*`",
        "host `/dev/mmcblk*`",
        "openFPGALoader -c dirtyJtag --detect",
        "hardware_receipts",
        "hardware_speedup_claim_allowed",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5519_builds_required_receipts_without_speedup(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5519: reachable commands produce required receipt fields."""

    runner = _reachable_runner()
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=VariableClock(),
        tests_run=_tests_run(),
    )
    out_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert runner.commands == [
        mod.CPU_INFO_COMMAND,
        mod.CUDA_INFO_COMMAND,
        mod.NVIDIA_SMI_QUERY_COMMAND,
        mod.POLARFIRE_IDENTITY_COMMAND,
        mod.KV260_IDENTITY_COMMAND,
        mod.KV260_XMUTIL_COMMAND,
        mod.KV260_UIO_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.YOSYS_VERSION_COMMAND,
        mod.NEXTPNR_VERSION_COMMAND,
        mod.GMPACK_VERSION_COMMAND,
    ]
    assert all("/dev/mmcblk" not in mod.command_to_string(command) for command in runner.commands)
    assert all("--write" not in mod.command_to_string(command) for command in runner.commands)
    assert saved["cpu_receipt"]["status"] == "reachable"
    assert saved["cuda_receipt"]["driver_versions"]["nvidia_driver"] == "575.57.08"
    assert saved["polar_fire_receipt"]["metadata"]["machine"] == "riscv64"
    assert saved["polar_fire_receipt"]["hash_identity"]["firmware_sha256"] == ["a" * 64]
    assert saved["kv260_receipt"]["metadata"]["loaded_overlay"] == "carnot_ising_v2_n64"
    assert saved["kv260_receipt"]["metadata"]["uio_devices"] == ["/dev/uio0", "/dev/uio4"]
    assert saved["gatemate_receipt"]["status"] == "reachable"
    assert saved["forbidden_kv260_host_sdcard_used"] is False
    assert saved["timing_methodology"]["workload"] == mod.TIMING_WORKLOAD
    assert saved["timing_methodology"]["warmup"] == mod.TIMING_WARMUP
    assert saved["timing_methodology"]["repetitions"] == mod.TIMING_REPETITIONS
    assert saved["timing_methodology"]["matched_cpu_gpu_fpga_timing_exists"] is False
    assert saved["matched_timing_available"] is False
    assert saved["hardware_speedup_claim"] is False
    assert saved["hardware_speedup_claim_allowed"] is False
    assert saved["blocked_devices"] == []
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert "speedup_claim_allowed=false" in saved["honest_verdict"]
    assert len(saved["receipt_commands"]) == len(runner.commands)
    assert all("command_sha256" in receipt for receipt in saved["receipt_commands"])
    mod.validate_artifact(saved)


def test_req_verify_5519_records_blocked_devices_and_fails_closed() -> None:
    """REQ-VERIFY-5519: unavailable devices are blockers, not speedup claims."""

    runner = RecordingRunner(
        {
            mod.CPU_INFO_COMMAND: _probe(
                mod.CPU_INFO_COMMAND,
                stdout=_json_line(
                    {
                        "status": "reachable",
                        "device_names": ["CPU"],
                        "driver_versions": {},
                        "runtime_versions": {"python": "3.12"},
                        "metadata": {},
                    }
                ),
            ),
            mod.CUDA_INFO_COMMAND: _probe(
                mod.CUDA_INFO_COMMAND,
                stdout=_json_line(
                    {
                        "status": "blocked_runtime",
                        "device_names": [],
                        "driver_versions": {},
                        "runtime_versions": {"torch": "missing"},
                        "metadata": {"reason": "cuda_unavailable"},
                    }
                ),
            ),
            mod.NVIDIA_SMI_QUERY_COMMAND: _probe(
                mod.NVIDIA_SMI_QUERY_COMMAND,
                exit_code=127,
                stderr="nvidia-smi: not found\n",
            ),
            mod.POLARFIRE_IDENTITY_COMMAND: _probe(
                mod.POLARFIRE_IDENTITY_COMMAND,
                exit_code=255,
                stderr="ssh: no route to host\n",
            ),
            mod.KV260_IDENTITY_COMMAND: _probe(
                mod.KV260_IDENTITY_COMMAND,
                exit_code=124,
                stderr="timeout\n",
            ),
            mod.GATEMATE_DETECT_COMMAND: _probe(
                mod.GATEMATE_DETECT_COMMAND,
                exit_code=127,
                stderr="openFPGALoader: not found\n",
            ),
            mod.YOSYS_VERSION_COMMAND: _probe(mod.YOSYS_VERSION_COMMAND, exit_code=127),
            mod.NEXTPNR_VERSION_COMMAND: _probe(mod.NEXTPNR_VERSION_COMMAND, exit_code=127),
            mod.GMPACK_VERSION_COMMAND: _probe(mod.GMPACK_VERSION_COMMAND, exit_code=127),
        }
    )
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=VariableClock(),
        tests_run=_tests_run(),
    )

    assert mod.KV260_XMUTIL_COMMAND not in runner.commands
    assert mod.KV260_UIO_COMMAND not in runner.commands
    blocked = {row["device"]: row["blocked_reason"] for row in artifact["blocked_devices"]}
    assert blocked["cuda"] == "cuda_unavailable"
    assert blocked["polar_fire"] == "blocked_polarfire_ssh_identity"
    assert blocked["kv260"] == "blocked_kv260_ssh_identity"
    assert blocked["gatemate"] == "gatemate_toolchain_unavailable"
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)

    speedup = deepcopy(artifact)
    speedup["hardware_speedup_claim"] = True
    speedup["reproducibility_checksum"] = mod.payload_checksum(speedup)
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(speedup)

    allowed = deepcopy(artifact)
    allowed["hardware_speedup_claim_allowed"] = True
    allowed["reproducibility_checksum"] = mod.payload_checksum(allowed)
    with pytest.raises(ValueError, match="hardware_speedup_claim_allowed"):
        mod.validate_artifact(allowed)

    storage = deepcopy(artifact)
    storage["forbidden_kv260_host_sdcard_used"] = True
    storage["reproducibility_checksum"] = mod.payload_checksum(storage)
    with pytest.raises(ValueError, match="forbidden_kv260_host_sdcard_used"):
        mod.validate_artifact(storage)

    unsafe = deepcopy(artifact)
    unsafe["receipt_commands"][0]["command"] = "ls /dev/mmcblk0"
    unsafe["receipt_commands"][0]["command_sha256"] = mod.sha256_text("ls /dev/mmcblk0")
    unsafe["reproducibility_checksum"] = mod.payload_checksum(unsafe)
    with pytest.raises(ValueError, match="host storage command"):
        mod.validate_artifact(unsafe)

    flash = deepcopy(artifact)
    flash["receipt_commands"][0]["command"] = "openFPGALoader --write tile.bit"
    flash["receipt_commands"][0]["command_sha256"] = mod.sha256_text(
        "openFPGALoader --write tile.bit"
    )
    flash["reproducibility_checksum"] = mod.payload_checksum(flash)
    with pytest.raises(ValueError, match="destructive command"):
        mod.validate_artifact(flash)


def test_req_verify_5519_helper_branches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5519: parser, runtime, and command branches stay receipt-safe."""

    assert mod.parse_json_stdout("noise\n{\"status\":\"reachable\"}\n") == {
        "status": "reachable"
    }
    assert mod.parse_json_stdout("no json") is None
    assert mod.parse_key_value_stdout("a=1\nnoise\nb = two\n") == {"a": "1", "b": "two"}
    assert mod.parse_nvidia_smi("GPU A, 570.1\n") == {
        "device_names": ["GPU A"],
        "driver_versions": {"nvidia_driver": "570.1"},
    }
    assert mod.parse_nvidia_smi("") == {"device_names": [], "driver_versions": {}}
    assert mod.loaded_overlay_from_xmutil("foo loaded\n") == "foo"
    assert mod.loaded_overlay_from_xmutil("no overlays\n") is None
    assert mod.parse_uio_devices("/dev/uio4\n/dev/uio4\n/dev/uio0\n") == [
        "/dev/uio4",
        "/dev/uio0",
    ]
    assert mod._normalize_tests(None) == [  # noqa: SLF001 - tests pin defensive default.
        {"command": "verification not yet attached", "outcome": "pending"}
    ]

    class MissingPath:
        def __init__(self, _path: str) -> None:
            return None

        @staticmethod
        def exists() -> bool:
            return False

    monkeypatch.setattr(mod, "Path", MissingPath)
    monkeypatch.setattr(mod.platform, "processor", lambda: "fallback-cpu")
    assert mod._cpu_model_name() == "fallback-cpu"  # noqa: SLF001
    monkeypatch.undo()

    ok_probe = mod.run_command((sys.executable, "-c", "print('ok')"), timeout_s=5.0)
    missing_probe = mod.run_command(("definitely-missing-carnot-exp5519-bin",), timeout_s=0.01)
    timeout_probe = mod.run_command(
        (sys.executable, "-c", "import time; time.sleep(1)"),
        timeout_s=0.01,
    )
    assert ok_probe.exit_code == 0
    assert missing_probe.exit_code == 127
    assert timeout_probe.exit_code == 124

    assert mod.emit_cpu_info() == 0
    cpu_payload = json.loads(capsys.readouterr().out)
    assert cpu_payload["status"] == "reachable"
    assert cpu_payload["device_names"]

    assert mod.cuda_info_from_runtime(torch_module=None, import_torch=lambda: None)["status"] in {
        "blocked_toolchain",
        "blocked_runtime",
    }

    class FakeUnavailableCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

    class FakeUnavailableTorch:
        __version__ = "fake"
        version = type("Version", (), {"cuda": "12.8"})()
        cuda = FakeUnavailableCuda()

    unavailable = mod.cuda_info_from_runtime(torch_module=FakeUnavailableTorch())
    assert unavailable["status"] == "blocked_runtime"

    class FakeAvailableCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "fake-gpu"

    class FakeAvailableTorch:
        __version__ = "fake+cu"
        version = type("Version", (), {"cuda": "12.8"})()
        cuda = FakeAvailableCuda()

    available = mod.cuda_info_from_runtime(torch_module=FakeAvailableTorch())
    assert available["status"] == "reachable"
    assert available["device_names"] == ["fake-gpu"]

    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    fake_torch = object()

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            return fake_torch
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert mod._import_torch() is fake_torch  # noqa: SLF001

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert mod.emit_cuda_info() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "blocked_toolchain"
    monkeypatch.setattr(builtins, "__import__", real_import)

    cuda_receipts: list[dict[str, object]] = []
    cuda_fallback = mod.collect_cuda_receipt(
        RecordingRunner(
            {
                mod.CUDA_INFO_COMMAND: _probe(
                    mod.CUDA_INFO_COMMAND,
                    stdout=_json_line(
                        {
                            "status": "reachable",
                            "device_names": [],
                            "driver_versions": {},
                            "runtime_versions": {"torch": "fake"},
                            "metadata": {},
                        }
                    ),
                ),
                mod.NVIDIA_SMI_QUERY_COMMAND: _probe(
                    mod.NVIDIA_SMI_QUERY_COMMAND,
                    stdout="fallback-gpu, 570.1\n",
                ),
            }
        ),
        cuda_receipts,
    )
    assert cuda_fallback["device_names"] == ["fallback-gpu"]
    assert mod._gatemate_identity_status(  # noqa: SLF001
        _probe(mod.GATEMATE_DETECT_COMMAND, stdout="Jtag frequency ok\n")
    ) == ("blocked_identity", "blocked_gatemate_dirtyjtag_identity")

    artifact_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_reachable_runner(),
        clock=VariableClock(),
        tests_run=_tests_run(),
    )
    assert artifact_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["hardware_speedup_claim"] is False
