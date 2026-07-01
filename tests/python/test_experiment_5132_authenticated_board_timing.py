"""Tests for Exp 5132 authenticated board timing continuity.

Spec refs: REQ-HW-5132, SCENARIO-HW-5132.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from carnot import experiment_5132_authenticated_board_timing as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "fpga" / "spec.md"


class RecordingRunner:
    """SCENARIO-HW-5132 runner with explicit authenticated command outputs."""

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


class FlatClock:
    """Deterministic clock for REQ-HW-5132 duration accounting."""

    def __call__(self) -> float:
        return 5132.0


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _ready_probes(
    *,
    kv260_timing_probe: mod.CommandProbe | None = None,
    gatemate_detect_stdout: str = "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
) -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    probes: dict[tuple[str, ...], list[mod.CommandProbe]] = {
        mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
        mod.KV260_UIO_LIST_COMMAND: [
            _probe(mod.KV260_UIO_LIST_COMMAND, stdout="/dev/uio0\n/dev/uio1\n", duration_s=0.2)
        ],
        mod.KV260_UIO_SYSFS_COMMAND: [
            _probe(
                mod.KV260_UIO_SYSFS_COMMAND,
                stdout="uio0 carnot_ising\nuio1 axi_timer\n",
                duration_s=0.2,
            )
        ],
        mod.GATEMATE_COMMAND_AVAILABLE_COMMAND: [
            _probe(
                mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
                stdout="/opt/oss-cad-suite/bin/openFPGALoader\n",
                duration_s=0.1,
            )
        ],
        mod.GATEMATE_YOSYS_VERSION_COMMAND: [
            _probe(mod.GATEMATE_YOSYS_VERSION_COMMAND, stdout="Yosys 0.64\n")
        ],
        mod.GATEMATE_NEXTPNR_VERSION_COMMAND: [
            _probe(mod.GATEMATE_NEXTPNR_VERSION_COMMAND, stdout="nextpnr-himbaechel 0.8\n")
        ],
        mod.GATEMATE_GMPACK_VERSION_COMMAND: [
            _probe(mod.GATEMATE_GMPACK_VERSION_COMMAND, stdout="gmpack 2026.04\n")
        ],
        mod.GATEMATE_USB_EVIDENCE_COMMAND: [
            _probe(
                mod.GATEMATE_USB_EVIDENCE_COMMAND,
                stdout=(
                    "1209:c0ca (bus 3, device 6) path: 2.3\n"
                    "1514:2008 (bus 3, device 5) path: 2.1\n"
                ),
            )
        ],
        mod.GATEMATE_DETECT_COMMAND: [
            _probe(mod.GATEMATE_DETECT_COMMAND, stdout=gatemate_detect_stdout, duration_s=0.3)
        ],
        mod.POLARFIRE_SSH_COMMAND: [_probe(mod.POLARFIRE_SSH_COMMAND, duration_s=0.4)],
        mod.POLARFIRE_ARCH_COMMAND: [
            _probe(mod.POLARFIRE_ARCH_COMMAND, stdout="riscv64\n", duration_s=0.2)
        ],
        mod.POLARFIRE_PYTHON_COMMAND: [
            _probe(mod.POLARFIRE_PYTHON_COMMAND, stdout="Python 3.12.12\n", duration_s=0.2)
        ],
        mod.POLARFIRE_DISPATCH_PRECHECK_COMMAND: [
            _probe(
                mod.POLARFIRE_DISPATCH_PRECHECK_COMMAND,
                stdout=json.dumps(
                    {
                        "duration_s": 0.000231,
                        "energy": 4,
                        "sample_quality": {"finite_energy": True, "sample_count": 4},
                        "workload_sha256": mod.POLARFIRE_DISPATCH_WORKLOAD_HASH,
                    },
                    sort_keys=True,
                )
                + "\n",
                duration_s=0.25,
            )
        ],
    }
    if kv260_timing_probe is not None:
        probes[kv260_timing_probe.command] = [kv260_timing_probe]
    return probes


def _write_safe_kv260_workload(root: Path) -> dict[str, str]:
    path = root / mod.KV260_SAFE_WORKLOAD_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "# READ_ONLY_UIO_WORKLOAD\n"
        "# safe_for_continuity_audit\n"
        "# read_only\n"
        "import json\n"
        "print(json.dumps({'sample_quality': {'register_read_only': True, 'sample_count': 2}}))\n"
    )
    path.write_text(text, encoding="utf-8")
    loaded = mod.load_safe_kv260_workload(root)
    assert loaded is not None
    return loaded


def test_req_hw_5132_spec_declares_authenticated_timing_contract() -> None:
    """REQ-HW-5132: OpenSpec anchors v470 authenticated blocker fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-HW-5132",
        "SCENARIO-HW-5132",
        "experiment_5132_authenticated_board_timing_v470.json",
        "hardware_smoke_or_authenticated_blockers",
        "kv260_timing_transcript",
        "gatemate_transcript",
        "polarfire_transcript",
        "extropic_tsu_execution_claimed=false",
        "no_speedup_claim=true",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_hw_5132_blockers_write_authenticated_continuity_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-5132: missing safe board workloads become precise blockers."""

    runner = RecordingRunner(_ready_probes())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=FlatClock(),
        run_date="20260701",
    )
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_UIO_LIST_COMMAND,
        mod.KV260_UIO_SYSFS_COMMAND,
        mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
        mod.GATEMATE_YOSYS_VERSION_COMMAND,
        mod.GATEMATE_NEXTPNR_VERSION_COMMAND,
        mod.GATEMATE_GMPACK_VERSION_COMMAND,
        mod.GATEMATE_USB_EVIDENCE_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_COMMAND,
        mod.POLARFIRE_ARCH_COMMAND,
        mod.POLARFIRE_PYTHON_COMMAND,
        mod.POLARFIRE_DISPATCH_PRECHECK_COMMAND,
    ]
    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["experiment_id"] == "exp5132-authenticated-board-timing-v470"
    assert saved["milestone"] == "2026.07.470"
    assert saved["honest_verdict"] == (
        "complete_authenticated_board_blockers_cpu_residual_no_speedup_claim"
    )
    assert saved["inference_substrate"] == "hardware_smoke_or_authenticated_blockers"
    assert saved["duration_s"] == 0.0001
    assert saved["kv260_ssh_checked"] is True
    assert saved["kv260_host_block_devices_touched"] is False
    assert saved["kv260_timing_transcript"]["uio_timing_attempted"] is False
    assert saved["kv260_timing_transcript"]["blockers"] == [
        "no_checked_in_safe_kv260_uio_timing_workload"
    ]
    assert saved["gatemate_checked"] is True
    assert saved["gatemate_transcript"]["flash_attempted"] is False
    assert saved["gatemate_transcript"]["flash_precheck"]["blocker"] == (
        "no_safe_gatemate_flash_manifest"
    )
    assert saved["polarfire_checked"] is True
    assert saved["polarfire_transcript"]["dispatch_precheck_attempted"] is True
    assert saved["polarfire_transcript"]["workload_hash"] == (
        mod.POLARFIRE_DISPATCH_WORKLOAD_HASH
    )
    assert saved["timing_measurements"]["kv260_authenticated_workload_s"] is None
    assert saved["timing_measurements"]["polarfire_dispatch_precheck_s"] == 0.25
    assert saved["residual_energy_by_sweep"] == mod.compute_cpu_residual_sweep()[0]
    assert saved["sample_quality_evidence"]["board_speedup_evidence_complete"] is False
    assert saved["sample_quality_evidence"]["cpu_residual_sample_count"] == 8
    assert saved["no_speedup_claim"] is True
    assert saved["extropic_tsu_execution_claimed"] is False
    assert saved["conductor_modified"] is False
    assert saved["flagged_adversarial"] is False
    assert saved["tests_run"] == mod.DEFAULT_TESTS_RUN
    assert saved["workload_hashes"]["kv260_timing_workload"] is None
    assert saved["workload_hashes"]["polarfire_dispatch_precheck_workload"] == (
        mod.POLARFIRE_DISPATCH_WORKLOAD_HASH
    )
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    encoded = json.dumps(saved, sort_keys=True).lower()
    assert "/dev/mmcblk" not in encoded
    assert "/dev/disk" not in encoded
    mod.validate_artifact(saved)


def test_req_hw_5132_safe_kv260_workload_is_hashed_and_timed(tmp_path: Path) -> None:
    """REQ-HW-5132: read-only KV260 UIO workload evidence is hash-matched."""

    safe_workload = _write_safe_kv260_workload(tmp_path)
    timing_command = mod.kv260_timing_command(safe_workload["text"], safe_workload["sha256"])
    timing_probe = _probe(
        timing_command,
        stdout=json.dumps(
            {
                "duration_s": 0.000321,
                "sample_quality": {"register_read_only": True, "sample_count": 2},
                "workload_sha256": safe_workload["sha256"],
            },
            sort_keys=True,
        )
        + "\n",
        duration_s=0.42,
    )
    runner = RecordingRunner(_ready_probes(kv260_timing_probe=timing_probe))

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=FlatClock(),
        run_date="20260701",
    )

    assert timing_command in runner.commands
    assert artifact["kv260_timing_transcript"]["uio_timing_attempted"] is True
    assert artifact["kv260_timing_transcript"]["blockers"] == []
    assert artifact["kv260_timing_transcript"]["workload_hash"] == safe_workload["sha256"]
    assert artifact["timing_measurements"]["kv260_authenticated_workload_s"] == 0.42
    assert artifact["workload_hashes"]["kv260_timing_workload"] == safe_workload["sha256"]
    assert artifact["sample_quality_evidence"]["kv260_read_only_sample_quality"] == {
        "register_read_only": True,
        "sample_count": 2,
    }
    assert mod.safe_kv260_workload_text(safe_workload["text"]) is True
    assert mod.safe_kv260_workload_text(safe_workload["text"] + "write_u32") is False
    mod.validate_artifact(artifact)


def test_req_hw_5132_validation_rejects_overclaims_and_schema_drift(tmp_path: Path) -> None:
    """REQ-HW-5132: validation rejects speedup, TSU, conductor, and storage drift."""

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=RecordingRunner(_ready_probes()),
        clock=FlatClock(),
        run_date="20260701",
    )

    for field in (
        "no_speedup_claim",
        "extropic_tsu_execution_claimed",
        "conductor_modified",
        "kv260_host_block_devices_touched",
        "flagged_adversarial",
    ):
        bad = dict(artifact, **{field: not artifact[field]})
        bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=field):
            mod.validate_artifact(bad)

    bad_verdict = dict(artifact, honest_verdict="kv260_blocked_no_prefix")
    bad_verdict["reproducibility_checksum"] = mod.payload_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_storage = dict(artifact)
    bad_storage["command_transcripts"] = {"unsafe": {"command": "touch /dev/" + "disk"}}
    bad_storage["reproducibility_checksum"] = mod.payload_checksum(bad_storage)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(bad_storage)

    bad_checksum = dict(artifact, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)

    assert "missing required fields" in "; ".join(mod.artifact_schema_errors({}))
    assert "command_transcripts must be a dict" in mod.artifact_schema_errors(
        dict(artifact, command_transcripts=[])
    )
    assert "workload_hashes must be a dict" in mod.artifact_schema_errors(
        dict(artifact, workload_hashes=[])
    )
    assert "timing_measurements must be a dict" in mod.artifact_schema_errors(
        dict(artifact, timing_measurements=[])
    )
    assert "residual_energy_by_sweep must be a list" in mod.artifact_schema_errors(
        dict(artifact, residual_energy_by_sweep={})
    )
    assert "tests_run must be non-empty" in mod.artifact_schema_errors(
        dict(artifact, tests_run=[])
    )


def test_req_hw_5132_blocker_helper_branches_are_precise(tmp_path: Path) -> None:
    """REQ-HW-5132: helper blockers classify unsafe hardware paths precisely."""

    kv_unreachable = mod.run_kv260_checks(
        repo_root=tmp_path,
        command_runner=RecordingRunner(
            {
                mod.KV260_SSH_COMMAND: [
                    _probe(mod.KV260_SSH_COMMAND, exit_code=255, stderr="timeout\n")
                ]
            }
        ),
    )
    assert kv_unreachable["blockers"] == ["blocked_kv260_ssh_unreachable"]

    unsafe_path = tmp_path / mod.KV260_SAFE_WORKLOAD_REL_PATH
    unsafe_path.parent.mkdir(parents=True, exist_ok=True)
    unsafe_path.write_text("# READ_ONLY_UIO_WORKLOAD\nwrite_u32\n", encoding="utf-8")
    assert mod.load_safe_kv260_workload(tmp_path) is None

    safe_workload = _write_safe_kv260_workload(tmp_path)
    kv_timing_fail_command = mod.kv260_timing_command(
        safe_workload["text"], safe_workload["sha256"]
    )
    kv_timing_fail = mod.run_kv260_checks(
        repo_root=tmp_path,
        command_runner=RecordingRunner(
            {
                mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND)],
                mod.KV260_UIO_LIST_COMMAND: [_probe(mod.KV260_UIO_LIST_COMMAND)],
                mod.KV260_UIO_SYSFS_COMMAND: [_probe(mod.KV260_UIO_SYSFS_COMMAND)],
                kv_timing_fail_command: [
                    _probe(kv_timing_fail_command, exit_code=1, stderr="read failed\n")
                ],
            }
        ),
    )
    assert kv_timing_fail["blockers"] == ["kv260_safe_uio_timing_workload_failed"]

    manifest_path = tmp_path / mod.GATEMATE_SAFE_FLASH_MANIFEST_REL_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{bad json", encoding="utf-8")
    invalid_manifest = mod.gatemate_flash_precheck(tmp_path)
    assert invalid_manifest["manifest_present"] is True
    assert invalid_manifest["blocker"] == "safe_gatemate_flash_manifest_invalid"
    manifest_path.write_text(
        json.dumps({"flash_allowed": True, "design_scope": "tiny_readback_only"}),
        encoding="utf-8",
    )
    safe_manifest = mod.gatemate_flash_precheck(tmp_path)
    assert safe_manifest["blocker"] is None
    assert mod.parse_probe_json(_probe(mod.KV260_SSH_COMMAND, stdout="not-json\n")) == {}

    pf_unreachable = mod.run_polarfire_checks(
        command_runner=RecordingRunner(
            {
                mod.POLARFIRE_SSH_COMMAND: [
                    _probe(mod.POLARFIRE_SSH_COMMAND, exit_code=255, stderr="timeout\n")
                ]
            }
        )
    )
    assert pf_unreachable["blockers"] == ["blocked_polarfire_ssh_unreachable"]

    pf_bad_platform = mod.run_polarfire_checks(
        command_runner=RecordingRunner(
            {
                mod.POLARFIRE_SSH_COMMAND: [_probe(mod.POLARFIRE_SSH_COMMAND)],
                mod.POLARFIRE_ARCH_COMMAND: [
                    _probe(mod.POLARFIRE_ARCH_COMMAND, stdout="x86_64\n")
                ],
                mod.POLARFIRE_PYTHON_COMMAND: [
                    _probe(mod.POLARFIRE_PYTHON_COMMAND, stdout="Python 3.9.18\n")
                ],
            }
        )
    )
    assert pf_bad_platform["blockers"] == [
        "polarfire_arch_not_riscv64",
        "polarfire_python_precheck_failed",
    ]

    pf_dispatch_fail = mod.run_polarfire_checks(
        command_runner=RecordingRunner(
            {
                mod.POLARFIRE_SSH_COMMAND: [_probe(mod.POLARFIRE_SSH_COMMAND)],
                mod.POLARFIRE_ARCH_COMMAND: [
                    _probe(mod.POLARFIRE_ARCH_COMMAND, stdout="riscv64\n")
                ],
                mod.POLARFIRE_PYTHON_COMMAND: [
                    _probe(mod.POLARFIRE_PYTHON_COMMAND, stdout="Python 3.12.12\n")
                ],
                mod.POLARFIRE_DISPATCH_PRECHECK_COMMAND: [
                    _probe(
                        mod.POLARFIRE_DISPATCH_PRECHECK_COMMAND,
                        exit_code=1,
                        stderr="dispatch failed\n",
                    )
                ],
            }
        )
    )
    assert pf_dispatch_fail["blockers"] == ["polarfire_dispatch_precheck_failed"]


def test_scenario_hw_5132_run_experiment_and_cli_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-HW-5132: run_experiment and script entrypoints write the artifact."""

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=RecordingRunner(_ready_probes()),
        clock=FlatClock(),
        run_date="20260701",
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["spec_refs"] == ["REQ-HW-5132", "SCENARIO-HW-5132"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)

    called: dict[str, object] = {}

    def fake_run_experiment(**kwargs: object) -> Path:
        called["kwargs"] = kwargs
        path = Path(kwargs["repo_root"]) / mod.OUTPUT_REL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
        return path

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    assert mod.main(["--date", "20260701", "--repo-root", str(tmp_path)]) == 0
    assert called["kwargs"]["run_date"] == "20260701"
    assert "honest_verdict" in capsys.readouterr().out

    script = importlib.import_module("scripts.experiment_5132_authenticated_board_timing_v470")
    monkeypatch.setattr(script, "experiment_main", lambda argv: 19)
    assert script.main(["--date", "20260701"]) == 19
