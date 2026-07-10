"""Tests for Exp5532 hardware receipt parser repeatability repair.

Spec refs: REQ-VERIFY-5532, SCENARIO-VERIFY-5532.
"""

from __future__ import annotations

from copy import deepcopy
import builtins
import json
from pathlib import Path
import sys

import pytest

from carnot import experiment_5532_hardware_receipt_parser_repeatability as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5532_hardware_receipt_parser_repeatability.py")


class RecordingRunner:
    """SCENARIO-VERIFY-5532 fake runner preserving safe command order."""

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
    """Deterministic clock for stable artifact duration checks."""

    def __init__(self) -> None:
        self.value = 5532.0

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


def _tests_run() -> list[str]:
    return [TEST_PATH.as_posix()]


def _cpu_payload() -> dict[str, object]:
    return {
        "status": "reachable",
        "device_names": ["AMD Ryzen AI 9 HX 370"],
        "driver_versions": {},
        "runtime_versions": {"python": "3.12.13", "platform": "Linux-test"},
        "versions": {"machine": "x86_64"},
        "memory": {"mem_total_kib": 1024, "mem_available_kib": 512},
        "metadata": {"python_executable": sys.executable},
    }


def _cuda_payload() -> dict[str, object]:
    return {
        "status": "reachable",
        "device_names": ["runtime-gpu"],
        "driver_versions": {},
        "runtime_versions": {"torch": "2.9.0+cu128", "cuda": "12.8"},
        "versions": {"device_count": 1},
        "memory": {"device_memory": [{"index": 0, "total_mib": 24576, "reserved_mib": 12}]},
        "metadata": {"device_count": 1},
    }


def _reachable_runner() -> RecordingRunner:
    return RecordingRunner(
        {
            mod.CPU_INFO_COMMAND: _probe(mod.CPU_INFO_COMMAND, stdout=_json_line(_cpu_payload())),
            mod.CUDA_INFO_COMMAND: _probe(
                mod.CUDA_INFO_COMMAND,
                stdout=_json_line(_cuda_payload()),
            ),
            mod.NVIDIA_SMI_QUERY_COMMAND: _probe(
                mod.NVIDIA_SMI_QUERY_COMMAND,
                stdout="NVIDIA GeForce RTX 3090, 610.43.02, 24576, 22000\n",
            ),
            mod.POLARFIRE_IDENTITY_COMMAND: _probe(
                mod.POLARFIRE_IDENTITY_COMMAND,
                stdout=(
                    "board_identity=polarfire\n"
                    "hostname=mpfs-disco-kit\n"
                    "machine=riscv64\n"
                    "kernel=6.18.17\n"
                    "model=Microchip PolarFire-SoC Discovery Kit\n"
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
                stdout="IDCODE 0x20000000 Cologne Chip GateMate Series GM1A\n",
            ),
            mod.YOSYS_VERSION_COMMAND: _probe(mod.YOSYS_VERSION_COMMAND, stdout="Yosys 0.64\n"),
            mod.NEXTPNR_VERSION_COMMAND: _probe(
                mod.NEXTPNR_VERSION_COMMAND,
                stdout="nextpnr-himbaechel 0.10\n",
            ),
            mod.GMPACK_VERSION_COMMAND: _probe(mod.GMPACK_VERSION_COMMAND, stdout="gmpack v1.13\n"),
        }
    )


def _write_prior_repeat_artifact(root: Path) -> None:
    path = root / mod.REPEAT_SOURCE_RELATIVE_PATHS[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": "exp5492-test",
                "run_date": "2026-07-10",
                "repeat_count": 3,
                "workload_hashes": ["a" * 64, "b" * 64],
                "cpu_baseline_receipts": [
                    {
                        "substrate": "local_cpu_exp5491_descriptor_exact_reference",
                        "workload_hashes": ["a" * 64, "b" * 64],
                        "repeat_count": 3,
                        "aggregate_output_hash": "c" * 64,
                    }
                ],
                "board_receipts": [
                    {
                        "board_identity": "polarfire",
                        "workload_hashes": ["a" * 64, "b" * 64],
                        "repeat_count": 3,
                        "aggregate_output_hash": "d" * 64,
                    }
                ],
                "hardware_speedup_claim": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_req_verify_5532_spec_declares_parser_repeatability_contract() -> None:
    """REQ-VERIFY-5532: OpenSpec anchors fields, classes, safe paths, and no-speedup."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5532") : spec.index("### REQ-VERIFY-5519")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-VERIFY-5532",
        str(mod.RESULT_RELATIVE_PATH),
        "ssh polarfire",
        "ssh kria",
        "xmutil",
        "board-local `/dev/uio*`",
        "`/dev/mmcblk*`",
        "hardware_receipt_parser_repeatability",
        "hardware_speedup_claim_allowed",
    ):
        assert marker in section
    for class_name in mod.REPEATABILITY_CLASSES:
        assert class_name in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5532_builds_repaired_receipts_and_bound_repeats(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5532: repaired parsers produce safe receipts without speedup."""

    _write_prior_repeat_artifact(tmp_path)
    runner = _reachable_runner()
    artifact = mod.build_artifact(
        root=tmp_path,
        command_runner=runner,
        clock=VariableClock(),
        timestamp=lambda: "2026-07-10T00:00:00Z",
        tests_added_or_reused=_tests_run(),
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
    assert all("/dev/disk" not in mod.command_to_string(command) for command in runner.commands)
    assert all("--write" not in mod.command_to_string(command) for command in runner.commands)
    assert saved["devices_checked"] == list(mod.DEVICES_CHECKED)
    assert saved["cpu_receipt_parseable"] is True
    assert saved["cuda_receipt_parseable"] is True
    assert saved["polarfire_reachable"] is True
    assert saved["kv260_safe_path_used"] is True
    assert saved["forbidden_kv260_host_sdcard_used"] is False
    assert saved["gatemate_identity_ok"] is True
    assert saved["device_receipts"]["cpu"]["memory"]["mem_total_kib"] == 1024
    assert saved["device_receipts"]["cuda"]["driver_versions"]["nvidia_driver"] == "610.43.02"
    assert saved["device_receipts"]["cuda"]["memory"]["nvidia_smi"][0]["total_mib"] == 24576
    assert saved["device_receipts"]["kv260"]["metadata"]["loaded_overlay"] == "carnot_ising_v2_n64"
    assert saved["device_receipts"]["kv260"]["metadata"]["uio_devices"] == ["/dev/uio0", "/dev/uio4"]
    assert saved["device_receipts"]["gatemate"]["classification"] == "workload_blocked"
    assert saved["parser_failures"] == {}
    assert saved["repeated_workload_hashes"] == ["a" * 64, "b" * 64]
    assert {
        (row["device"], row["workload_hash"], row["parser_version"])
        for row in saved["repeated_workload_receipts"]
    } == {
        ("cpu", "a" * 64, mod.PARSER_VERSION),
        ("cpu", "b" * 64, mod.PARSER_VERSION),
        ("polarfire", "a" * 64, mod.PARSER_VERSION),
        ("polarfire", "b" * 64, mod.PARSER_VERSION),
    }
    assert saved["matched_timing_available"] is False
    assert saved["hardware_speedup_claim"] is False
    assert saved["hardware_speedup_claim_allowed"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    mod.validate_artifact(saved)


def test_req_verify_5532_records_precise_blockers_and_fails_closed() -> None:
    """REQ-VERIFY-5532: malformed rows become parser blockers, not speedup claims."""

    runner = RecordingRunner(
        {
            mod.CPU_INFO_COMMAND: _probe(mod.CPU_INFO_COMMAND, stdout="not json\n"),
            mod.CUDA_INFO_COMMAND: _probe(mod.CUDA_INFO_COMMAND, stdout="not json\n"),
            mod.NVIDIA_SMI_QUERY_COMMAND: _probe(
                mod.NVIDIA_SMI_QUERY_COMMAND,
                stdout="fallback-gpu, 610.43.02, 24576, 22000\n",
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
                stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
            ),
            mod.YOSYS_VERSION_COMMAND: _probe(mod.YOSYS_VERSION_COMMAND, exit_code=127),
            mod.NEXTPNR_VERSION_COMMAND: _probe(mod.NEXTPNR_VERSION_COMMAND, exit_code=127),
            mod.GMPACK_VERSION_COMMAND: _probe(mod.GMPACK_VERSION_COMMAND, exit_code=127),
        }
    )
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=VariableClock(),
        timestamp=lambda: "2026-07-10T00:00:00Z",
        tests_added_or_reused=_tests_run(),
    )

    assert mod.KV260_XMUTIL_COMMAND not in runner.commands
    assert mod.KV260_UIO_COMMAND not in runner.commands
    assert artifact["cpu_receipt_parseable"] is False
    assert artifact["cuda_receipt_parseable"] is True
    assert artifact["device_receipts"]["cpu"]["classification"] == "parser_blocked"
    assert artifact["device_receipts"]["cuda"]["classification"] == "workload_blocked"
    assert artifact["device_receipts"]["polarfire"]["classification"] == "identity_blocked"
    assert artifact["device_receipts"]["kv260"]["classification"] == "identity_blocked"
    assert artifact["device_receipts"]["gatemate"]["classification"] == "identity_blocked"
    assert artifact["parser_failures"]["cpu"] == "cpu_info_unparseable"
    assert artifact["parser_failures"]["cuda_runtime"] == "cuda_runtime_info_unparseable"
    assert artifact["device_receipts"]["cuda"]["device_names"] == ["fallback-gpu"]
    assert artifact["device_receipts"]["cuda"]["blocked_reason"] == "cuda_runtime_info_unparseable"
    assert artifact["gatemate_identity_ok"] is False
    assert artifact["hardware_speedup_claim"] is False
    mod.validate_artifact(artifact)

    speedup = deepcopy(artifact)
    speedup["hardware_speedup_claim"] = True
    speedup["reproducibility_checksum"] = mod.payload_checksum(speedup)
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(speedup)

    storage = deepcopy(artifact)
    storage["command_receipts"][0]["command"] = "ls /dev/mmcblk0"
    storage["command_receipts"][0]["command_sha256"] = mod.sha256_text("ls /dev/mmcblk0")
    storage["reproducibility_checksum"] = mod.payload_checksum(storage)
    with pytest.raises(ValueError, match="host storage command"):
        mod.validate_artifact(storage)

    bad_class = deepcopy(artifact)
    bad_class["device_receipts"]["cpu"]["classification"] = "maybe"
    bad_class["reproducibility_checksum"] = mod.payload_checksum(bad_class)
    with pytest.raises(ValueError, match="classification"):
        mod.validate_artifact(bad_class)


def test_req_verify_5532_parser_and_runtime_helpers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5532: parser helpers cover malformed, unavailable, and CLI paths."""

    assert mod.parse_json_stdout("noise\n{\"status\":\"reachable\"}\n") == {
        "status": "reachable"
    }
    assert mod.parse_json_stdout("no json") is None
    assert mod.parse_key_value_stdout("a=1\nnoise\nb = two\n") == {"a": "1", "b": "two"}
    assert mod.parse_nvidia_smi("GPU A, 570.1, 100, 40\n") == {
        "device_names": ["GPU A"],
        "driver_versions": {"nvidia_driver": "570.1"},
        "memory": {"nvidia_smi": [{"index": 0, "total_mib": 100, "free_mib": 40}]},
    }
    assert mod.parse_nvidia_smi("bad row\n") == {
        "device_names": [],
        "driver_versions": {},
        "memory": {"nvidia_smi": []},
    }
    assert mod.parse_nvidia_smi("GPU B, 570.1, bad, also-bad\n")["memory"] == {
        "nvidia_smi": [{"index": 0, "total_mib": None, "free_mib": None}]
    }
    assert mod.parse_meminfo("MemTotal:       2048 kB\nMemAvailable:   1024 kB\n") == {
        "mem_total_kib": 2048,
        "mem_available_kib": 1024,
    }
    assert mod.parse_meminfo("noise\nSwapTotal: 10 kB\n") == {}
    assert mod.loaded_overlay_from_xmutil("overlay loaded\n") == "overlay"
    assert mod.loaded_overlay_from_xmutil("no loaded apps\n") is None
    assert mod.parse_uio_devices("/dev/uio4\n/dev/uio4\n/dev/uio0\n") == [
        "/dev/uio4",
        "/dev/uio0",
    ]
    assert mod.classify_receipt(status="blocked_toolchain", parseable=False) == "parser_blocked"
    assert mod.classify_receipt(status="blocked_toolchain", parseable=True) == "unavailable"
    assert mod.classify_receipt(status="blocked_identity", parseable=True) == "identity_blocked"
    assert mod.classify_receipt(status="reachable", parseable=True, repeated=True) == "timing_blocked"
    assert mod.classify_receipt(status="reachable", parseable=True) == "workload_blocked"
    assert mod.classify_receipt(status="mystery", parseable=True) == "unavailable"
    assert mod.now_utc().endswith("Z")
    assert mod._normalize_tests(None) == [  # noqa: SLF001
        "tests/python/test_experiment_5532_hardware_receipt_parser_repeatability.py"
    ]

    class MissingPath:
        def __init__(self, _path: str) -> None:
            return None

        @staticmethod
        def exists() -> bool:
            return False

    monkeypatch.setattr(mod, "Path", MissingPath)
    monkeypatch.setattr(mod.platform, "processor", lambda: "fallback-cpu")
    assert mod._read_meminfo() == ""  # noqa: SLF001
    assert mod._cpu_model_name() == "fallback-cpu"  # noqa: SLF001
    monkeypatch.undo()

    ok_probe = mod.run_command((sys.executable, "-c", "print('ok')"), timeout_s=5.0)
    missing_probe = mod.run_command(("definitely-missing-carnot-exp5532-bin",), timeout_s=0.01)
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
    assert cpu_payload["memory"]

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

        @staticmethod
        def mem_get_info(index: int) -> tuple[int, int]:
            assert index == 0
            return (256 * 1024**2, 512 * 1024**2)

        @staticmethod
        def memory_reserved(index: int) -> int:
            assert index == 0
            return 64 * 1024**2

    class FakeAvailableTorch:
        __version__ = "fake+cu"
        version = type("Version", (), {"cuda": "12.8"})()
        cuda = FakeAvailableCuda()

    available = mod.cuda_info_from_runtime(torch_module=FakeAvailableTorch())
    assert available["status"] == "reachable"
    assert available["memory"]["device_memory"][0]["free_mib"] == 256

    class FakeMemoryGapCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "memory-gap-gpu"

        @staticmethod
        def mem_get_info(index: int) -> tuple[int, int]:
            assert index == 0
            raise RuntimeError("no memory hook")

        @staticmethod
        def memory_reserved(index: int) -> int:
            assert index == 0
            raise RuntimeError("no reserved hook")

    class FakeMemoryGapTorch:
        __version__ = "fake+cu"
        version = type("Version", (), {"cuda": "12.8"})()
        cuda = FakeMemoryGapCuda()

    memory_gap = mod.cuda_info_from_runtime(torch_module=FakeMemoryGapTorch())
    assert memory_gap["memory"]["device_memory"] == [{"index": 0}]

    assert mod.cuda_info_from_runtime(torch_module=None, import_torch=lambda: None)["status"] == (
        "blocked_toolchain"
    )

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
    parser_failures: dict[str, object] = {}
    cuda_blocked = mod.collect_cuda_receipt(
        RecordingRunner(
            {
                mod.CUDA_INFO_COMMAND: _probe(
                    mod.CUDA_INFO_COMMAND,
                    stdout=_json_line(
                        {
                            "status": "blocked_runtime",
                            "device_names": [],
                            "driver_versions": {},
                            "runtime_versions": {"torch": "fake", "cuda": "unknown"},
                            "versions": {},
                            "memory": {},
                            "metadata": {"reason": "cuda_unavailable"},
                        }
                    ),
                ),
                mod.NVIDIA_SMI_QUERY_COMMAND: _probe(
                    mod.NVIDIA_SMI_QUERY_COMMAND,
                    exit_code=127,
                    stderr="not found",
                ),
            }
        ),
        cuda_receipts,
        parser_failures,
    )
    assert cuda_blocked["status"] == "blocked_runtime"
    assert cuda_blocked["blocked_reason"] == "cuda_unavailable"
    assert mod._gatemate_identity_status(  # noqa: SLF001
        _probe(mod.GATEMATE_DETECT_COMMAND, exit_code=127)
    ) == ("unavailable", "gatemate_toolchain_unavailable")

    malformed_path = tmp_path / mod.REPEAT_SOURCE_RELATIVE_PATHS[0]
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_path.write_text("{not-json", encoding="utf-8")
    assert mod.collect_repeated_workload_receipts(tmp_path, {}, "2026-07-10T00:00:00Z") == []
    malformed_path.write_text(
        json.dumps(
            {
                "cpu_baseline_receipts": [{"workload_hashes": ["a" * 64], "repeat_count": 1}],
                "board_receipts": [
                    "not-a-mapping",
                    {
                        "board_identity": "polar_fire",
                        "workload_hashes": ["b" * 64],
                        "repeat_count": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    polar_rows = mod.collect_repeated_workload_receipts(
        tmp_path,
        {"polarfire": {"device": "polarfire"}},
        "2026-07-10T00:00:00Z",
    )
    assert [(row["device"], row["workload_hash"]) for row in polar_rows] == [
        ("polarfire", "b" * 64)
    ]
    artifact_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_reachable_runner(),
        clock=VariableClock(),
        timestamp=lambda: "2026-07-10T00:00:00Z",
        tests_added_or_reused=_tests_run(),
    )
    assert artifact_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["hardware_speedup_claim"] is False
