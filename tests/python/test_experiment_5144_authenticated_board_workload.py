"""Tests for Exp 5144 authenticated board workload transcripts.

Spec refs: REQ-HW-5144, SCENARIO-HW-5144.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from carnot import experiment_5144_authenticated_board_workload as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "fpga" / "spec.md"


class RecordingRunner:
    """SCENARIO-HW-5144 runner with exact command transcripts."""

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
    """Deterministic REQ-HW-5144 clock for duration accounting."""

    def __call__(self) -> float:
        return 5144.0


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _base_probes(
    *,
    gatemate_detect_stdout: str = "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\nIDCode : 0x20000001\n",
    extra: dict[tuple[str, ...], list[mod.CommandProbe]] | None = None,
) -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    probes: dict[tuple[str, ...], list[mod.CommandProbe]] = {
        mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
        mod.GATEMATE_COMMAND_AVAILABLE_COMMAND: [
            _probe(
                mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
                stdout="/opt/oss-cad-suite/bin/openFPGALoader\n",
                duration_s=0.1,
            )
        ],
        mod.GATEMATE_DETECT_COMMAND: [
            _probe(mod.GATEMATE_DETECT_COMMAND, stdout=gatemate_detect_stdout, duration_s=0.3)
        ],
        mod.POLARFIRE_SSH_COMMAND: [_probe(mod.POLARFIRE_SSH_COMMAND, duration_s=0.4)],
    }
    if extra:
        probes.update(extra)
    return probes


def _write_exp5141_descriptors(root: Path) -> None:
    path = root / mod.EXP5141_RESULT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptors = []
    for board in ("kv260", "gatemate", "polarfire"):
        descriptors.append(
            {
                "descriptor_id": f"exp5141_{board}_unit_descriptor",
                "target_board": board,
                "workload_family": "partitioned_hubo_2dpt_boundary_refresh",
                "workload_hash": mod.sha256_json({"board": board, "unit": True}),
                "hardware_executed": False,
                "hardware_speedup_claimed": False,
            }
        )
    path.write_text(
        json.dumps({"board_ready_workload_descriptors": descriptors}, sort_keys=True),
        encoding="utf-8",
    )


def _write_manifest(root: Path, workloads: dict[str, object]) -> dict[str, object]:
    manifest = {
        "schema": mod.SAFE_WORKLOAD_MANIFEST_SCHEMA,
        "manifest_id": "unit-exp5144-safe-workloads",
        "workloads": workloads,
    }
    path = root / mod.SAFE_WORKLOAD_MANIFEST_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _polarfire_entry() -> dict[str, object]:
    payload = {
        "descriptor_id": "exp5141_polarfire_unit_descriptor",
        "expected_energy": 4,
        "spins": [1, -1, 1, -1],
    }
    return {
        "enabled": True,
        "command_kind": mod.POLARFIRE_COMMAND_KIND,
        "payload": payload,
        "workload_sha256": mod.sha256_json(payload),
        "executable_sha256": mod.POLARFIRE_INLINE_EXECUTABLE_HASH,
    }


def _kv260_entry(root: Path) -> tuple[dict[str, object], str]:
    rel_path = Path("hardware") / "kv260" / "exp5144_unit_read_only.py"
    text = (
        "# EXP5144_SAFE_READ_ONLY_UIO_WORKLOAD\n"
        "# safe_for_continuity_audit\n"
        "# read_only\n"
        "import json\n"
        "print(json.dumps({'workload_sha256': 'filled-by-test'}))\n"
    )
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    digest = mod.sha256_text(text)
    return (
        {
            "enabled": True,
            "command_kind": mod.KV260_COMMAND_KIND,
            "workload_path": str(rel_path),
            "workload_sha256": digest,
            "executable_path": str(rel_path),
            "executable_sha256": digest,
        },
        text,
    )


def _gatemate_entry(root: Path) -> dict[str, object]:
    rel_path = Path("hardware") / "gatemate" / "exp5144_unit.bit"
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unit safe gatemate bitstream\n")
    digest = mod.sha256_file(path)
    return {
        "enabled": True,
        "command_kind": mod.GATEMATE_COMMAND_KIND,
        "bitstream_path": str(rel_path),
        "bitstream_sha256": digest,
        "workload_sha256": digest,
        "board_profile": "olimex_gatemateevb",
        "flash_allowed": True,
    }


def test_req_hw_5144_spec_declares_authenticated_workload_contract() -> None:
    """REQ-HW-5144: OpenSpec anchors v471 board workload transcript fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5144") :]

    for marker in (
        "REQ-HW-5144",
        "SCENARIO-HW-5144",
        "experiment_5144_authenticated_board_workload_v471.json",
        "local_board_transcripts_or_blocked",
        "safe_workload_manifest",
        "hardware_workload_transcripts_ready=true",
        "extropic_tsu_execution_claimed=false",
        "no_speedup_claim=true",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_hw_5144_missing_manifest_blocks_without_speedup(tmp_path: Path) -> None:
    """SCENARIO-HW-5144: reachable boards without safe workloads stay blocked."""

    _write_exp5141_descriptors(tmp_path)
    runner = RecordingRunner(_base_probes())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=FlatClock(),
        run_date="20260702",
    )
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_COMMAND,
    ]
    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["experiment_id"] == mod.EXPERIMENT_ID
    assert saved["milestone"] == "2026.07.471"
    assert saved["honest_verdict"] == "blocked_no_safe_board_workload_manifest_no_speedup_claim"
    assert saved["inference_substrate"] == "local_board_transcripts_or_blocked"
    assert saved["safe_workload_manifest"]["present"] is False
    assert saved["safe_workload_manifest"]["exp5141_descriptors_loaded"] is True
    assert saved["safe_workload_manifest"]["descriptor_counts"] == {
        "gatemate": 1,
        "kv260": 1,
        "polarfire": 1,
    }
    assert saved["kv260_ssh_checked"] is True
    assert saved["kv260_host_block_devices_touched"] is False
    assert saved["kv260_timing_transcript"]["blockers"] == ["no_safe_kv260_workload_manifest"]
    assert saved["gatemate_transcript"]["blockers"] == ["no_safe_gatemate_workload_manifest"]
    assert saved["polarfire_transcript"]["blockers"] == ["no_safe_polarfire_workload_manifest"]
    assert saved["hardware_workload_transcripts_ready"] is False
    assert saved["no_speedup_claim"] is True
    assert saved["extropic_tsu_execution_claimed"] is False
    assert saved["conductor_modified"] is False
    assert saved["tests_run"] == mod.DEFAULT_TESTS_RUN
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    encoded = json.dumps(saved, sort_keys=True).lower()
    assert "/dev/mmcblk" not in encoded
    assert "/dev/disk" not in encoded
    mod.validate_artifact(saved)


def test_req_hw_5144_polarfire_manifest_dispatch_is_hash_matched(tmp_path: Path) -> None:
    """REQ-HW-5144: PolarFire dispatch transcript can make readiness true."""

    _write_exp5141_descriptors(tmp_path)
    polarfire_entry = _polarfire_entry()
    _write_manifest(tmp_path, {"polarfire": polarfire_entry})
    dispatch_command = mod.polarfire_dispatch_command(polarfire_entry)
    dispatch_probe = _probe(
        dispatch_command,
        stdout=json.dumps(
            {
                "correctness": {"energy_matches_expected": True},
                "duration_s": 0.000231,
                "energy": 4,
                "executable_sha256": polarfire_entry["executable_sha256"],
                "sample_quality": {"finite_energy": True, "sample_count": 4},
                "workload_sha256": polarfire_entry["workload_sha256"],
            },
            sort_keys=True,
        )
        + "\n",
        duration_s=0.25,
    )
    runner = RecordingRunner(_base_probes(extra={dispatch_command: [dispatch_probe]}))

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=FlatClock(),
        run_date="20260702",
    )

    assert dispatch_command in runner.commands
    assert artifact["honest_verdict"] == "complete_authenticated_board_workload_transcripts_no_speedup_claim"
    assert artifact["hardware_workload_transcripts_ready"] is True
    assert artifact["polarfire_transcript"]["dispatch_attempted"] is True
    assert artifact["polarfire_transcript"]["hash_matched"] is True
    assert artifact["workload_hashes"]["polarfire_workload"] == polarfire_entry["workload_sha256"]
    assert artifact["workload_hashes"]["polarfire_executable"] == polarfire_entry["executable_sha256"]
    assert artifact["timing_measurements"]["polarfire_dispatch_s"] == 0.25
    assert artifact["sample_quality_evidence"]["ready_evidence_boards"] == ["polarfire"]
    assert artifact["sample_quality_evidence"]["polarfire"]["correctness"] == {
        "energy_matches_expected": True
    }
    assert artifact["no_speedup_claim"] is True
    mod.validate_artifact(artifact)


def test_req_hw_5144_kv260_and_gatemate_manifest_paths_are_safe(tmp_path: Path) -> None:
    """REQ-HW-5144: manifest-backed KV260/GateMate commands require hash safety."""

    _write_exp5141_descriptors(tmp_path)
    kv260_entry, kv260_text = _kv260_entry(tmp_path)
    gatemate_entry = _gatemate_entry(tmp_path)
    _write_manifest(tmp_path, {"kv260": kv260_entry, "gatemate": gatemate_entry})
    kv260_command = mod.kv260_workload_command(kv260_entry, kv260_text)
    gatemate_command = mod.gatemate_workload_command(tmp_path, gatemate_entry)
    extra = {
        kv260_command: [
            _probe(
                kv260_command,
                stdout=json.dumps(
                    {
                        "correctness": {"read_only_register_probe": True},
                        "duration_s": 0.000321,
                        "executable_sha256": kv260_entry["executable_sha256"],
                        "sample_quality": {"read_only": True, "sample_count": 2},
                        "workload_sha256": kv260_entry["workload_sha256"],
                    },
                    sort_keys=True,
                )
                + "\n",
                duration_s=0.42,
            )
        ],
        gatemate_command: [
            _probe(gatemate_command, stdout="Programming: Success\nVerify: Success\n", duration_s=0.55)
        ],
    }
    runner = RecordingRunner(_base_probes(extra=extra))

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=FlatClock(),
        run_date="20260702",
    )

    assert kv260_command in runner.commands
    assert gatemate_command in runner.commands
    assert "nextpnr-gatemate" not in json.dumps(artifact)
    assert artifact["kv260_timing_transcript"]["hash_matched"] is True
    assert artifact["gatemate_transcript"]["hash_matched"] is True
    assert artifact["gatemate_transcript"]["flash_attempted"] is True
    assert artifact["workload_hashes"]["kv260_workload"] == kv260_entry["workload_sha256"]
    assert artifact["workload_hashes"]["gatemate_bitstream"] == gatemate_entry["bitstream_sha256"]
    assert set(artifact["sample_quality_evidence"]["ready_evidence_boards"]) == {
        "gatemate",
        "kv260",
    }
    assert mod.safe_kv260_workload_text(kv260_text) is True
    assert mod.safe_kv260_workload_text(kv260_text + "write_u32") is False
    mod.validate_artifact(artifact)


def test_req_hw_5144_validation_rejects_overclaims_and_schema_drift(tmp_path: Path) -> None:
    """REQ-HW-5144: validation rejects speedup, TSU, conductor, and storage drift."""

    _write_exp5141_descriptors(tmp_path)
    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=RecordingRunner(_base_probes()),
        clock=FlatClock(),
        run_date="20260702",
    )

    for field in (
        "no_speedup_claim",
        "extropic_tsu_execution_claimed",
        "conductor_modified",
        "kv260_host_block_devices_touched",
    ):
        bad = dict(artifact, **{field: not artifact[field]})
        bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=field):
            mod.validate_artifact(bad)

    bad_verdict = dict(artifact, honest_verdict="workload_blocked_no_prefix")
    bad_verdict["reproducibility_checksum"] = mod.payload_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_storage = dict(artifact)
    bad_storage["command_transcripts"] = {"unsafe": {"command": "touch /dev/" + "disk"}}
    bad_storage["reproducibility_checksum"] = mod.payload_checksum(bad_storage)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(bad_storage)

    bad_ready = dict(artifact, hardware_workload_transcripts_ready=True)
    bad_ready["sample_quality_evidence"] = dict(
        artifact["sample_quality_evidence"], ready_evidence_boards=[]
    )
    bad_ready["reproducibility_checksum"] = mod.payload_checksum(bad_ready)
    with pytest.raises(ValueError, match="ready gate"):
        mod.validate_artifact(bad_ready)

    bad_checksum = dict(artifact, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)

    assert "missing required fields" in "; ".join(mod.artifact_schema_errors({}))
    assert "safe_workload_manifest must be a dict" in mod.artifact_schema_errors(
        dict(artifact, safe_workload_manifest=[])
    )
    assert "workload_hashes must be a dict" in mod.artifact_schema_errors(
        dict(artifact, workload_hashes=[])
    )
    assert "timing_measurements must be a dict" in mod.artifact_schema_errors(
        dict(artifact, timing_measurements=[])
    )
    assert "tests_run must be non-empty" in mod.artifact_schema_errors(
        dict(artifact, tests_run=[])
    )


def test_req_hw_5144_blocker_helpers_are_precise(tmp_path: Path) -> None:
    """REQ-HW-5144: helper blockers distinguish unsafe workload paths."""

    manifest_path = tmp_path / mod.SAFE_WORKLOAD_MANIFEST_REL_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{bad json", encoding="utf-8")
    malformed = mod.load_safe_workload_manifest(tmp_path)
    assert malformed["valid"] is False
    assert malformed["blockers"] == ["safe_workload_manifest_malformed_json"]

    unsafe_entry, unsafe_text = _kv260_entry(tmp_path)
    unsafe_path = tmp_path / unsafe_entry["workload_path"]
    unsafe_path.write_text(unsafe_text + "write_u32\n", encoding="utf-8")
    _write_manifest(tmp_path, {"kv260": unsafe_entry})
    unsafe_manifest = mod.load_safe_workload_manifest(tmp_path)
    assert unsafe_manifest["workloads"]["kv260"]["safe"] is False
    assert unsafe_manifest["workloads"]["kv260"]["blocker"] == "kv260_workload_hash_mismatch"

    missing_tool_runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [
                _probe(mod.KV260_SSH_COMMAND, exit_code=255, stderr="timeout\n")
            ],
            mod.GATEMATE_COMMAND_AVAILABLE_COMMAND: [
                _probe(
                    mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
                    exit_code=127,
                    stderr="openFPGALoader missing\n",
                )
            ],
            mod.POLARFIRE_SSH_COMMAND: [
                _probe(mod.POLARFIRE_SSH_COMMAND, exit_code=255, stderr="timeout\n")
            ],
        }
    )
    blocked = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=missing_tool_runner,
        clock=FlatClock(),
        run_date="20260702",
    )
    assert blocked["kv260_timing_transcript"]["blockers"][0] == "blocked_kv260_ssh_unreachable"
    assert blocked["gatemate_transcript"]["blockers"][0] == "blocked_gatemate_openfpgaloader_missing"
    assert blocked["polarfire_transcript"]["blockers"][0] == "blocked_polarfire_ssh_unreachable"


def test_req_hw_5144_defensive_helper_branches_are_covered(tmp_path: Path) -> None:
    """REQ-HW-5144: malformed manifests and transcripts fail closed."""

    bad_schema = {
        "schema": "bad.schema",
        "manifest_id": "bad-schema",
        "workloads": {},
    }
    path = tmp_path / mod.SAFE_WORKLOAD_MANIFEST_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bad_schema), encoding="utf-8")
    assert mod.load_safe_workload_manifest(tmp_path)["blockers"] == [
        "safe_workload_manifest_schema_mismatch"
    ]

    assert mod._normalize_kv260_workload(  # noqa: SLF001
        tmp_path, {"enabled": True, "command_kind": "bad"}
    )["blocker"] == "kv260_workload_command_kind_invalid"
    assert mod._normalize_kv260_workload(  # noqa: SLF001
        tmp_path,
        {
            "enabled": True,
            "command_kind": mod.KV260_COMMAND_KIND,
            "workload_path": "missing.py",
        },
    )["blocker"] == "kv260_workload_file_missing"
    unsafe_text = (
        "# EXP5144_SAFE_READ_ONLY_UIO_WORKLOAD\n"
        "# safe_for_continuity_audit\n"
        "# read_only\n"
        "write_u32\n"
    )
    unsafe_path = tmp_path / "unsafe_kv260.py"
    unsafe_path.write_text(unsafe_text, encoding="utf-8")
    unsafe_hash = mod.sha256_text(unsafe_text)
    assert mod._normalize_kv260_workload(  # noqa: SLF001
        tmp_path,
        {
            "enabled": True,
            "command_kind": mod.KV260_COMMAND_KIND,
            "workload_path": "unsafe_kv260.py",
            "workload_sha256": unsafe_hash,
            "executable_sha256": unsafe_hash,
        },
    )["blocker"] == "kv260_workload_not_read_only_safe"

    assert mod._normalize_gatemate_workload(  # noqa: SLF001
        tmp_path, {"enabled": True, "command_kind": "bad"}
    )["blocker"] == "gatemate_workload_command_kind_invalid"
    assert mod._normalize_gatemate_workload(  # noqa: SLF001
        tmp_path,
        {
            "enabled": True,
            "command_kind": mod.GATEMATE_COMMAND_KIND,
            "flash_allowed": False,
            "board_profile": "olimex_gatemateevb",
        },
    )["blocker"] == "gatemate_flash_manifest_not_safe"
    assert mod._normalize_gatemate_workload(  # noqa: SLF001
        tmp_path,
        {
            "enabled": True,
            "command_kind": mod.GATEMATE_COMMAND_KIND,
            "flash_allowed": True,
            "board_profile": "olimex_gatemateevb",
            "bitstream_path": "missing.bit",
        },
    )["blocker"] == "gatemate_bitstream_missing"
    bit_path = tmp_path / "bad_hash.bit"
    bit_path.write_bytes(b"bitstream")
    assert mod._normalize_gatemate_workload(  # noqa: SLF001
        tmp_path,
        {
            "enabled": True,
            "command_kind": mod.GATEMATE_COMMAND_KIND,
            "flash_allowed": True,
            "board_profile": "olimex_gatemateevb",
            "bitstream_path": "bad_hash.bit",
            "bitstream_sha256": "0" * 64,
            "workload_sha256": "0" * 64,
        },
    )["blocker"] == "gatemate_bitstream_hash_mismatch"

    assert mod._normalize_polarfire_workload(  # noqa: SLF001
        {"enabled": True, "command_kind": "bad"}
    )["blocker"] == "polarfire_workload_command_kind_invalid"
    assert mod._normalize_polarfire_workload(  # noqa: SLF001
        {"enabled": True, "command_kind": mod.POLARFIRE_COMMAND_KIND, "payload": {}}
    )["blocker"] == "polarfire_payload_invalid"
    payload = {"spins": [1], "expected_energy": 1}
    assert mod._normalize_polarfire_workload(  # noqa: SLF001
        {
            "enabled": True,
            "command_kind": mod.POLARFIRE_COMMAND_KIND,
            "payload": payload,
            "workload_sha256": "0" * 64,
            "executable_sha256": mod.POLARFIRE_INLINE_EXECUTABLE_HASH,
        }
    )["blocker"] == "polarfire_workload_hash_mismatch"
    assert mod._normalize_polarfire_workload(  # noqa: SLF001
        {
            "enabled": True,
            "command_kind": mod.POLARFIRE_COMMAND_KIND,
            "payload": payload,
            "workload_sha256": mod.sha256_json(payload),
            "executable_sha256": "0" * 64,
        }
    )["blocker"] == "polarfire_executable_hash_mismatch"

    no_detect = mod.run_gatemate_checks(
        tmp_path,
        mod.load_safe_workload_manifest(tmp_path),
        RecordingRunner(
            {
                mod.GATEMATE_COMMAND_AVAILABLE_COMMAND: [
                    _probe(mod.GATEMATE_COMMAND_AVAILABLE_COMMAND, stdout="/bin/openFPGALoader\n")
                ],
                mod.GATEMATE_DETECT_COMMAND: [_probe(mod.GATEMATE_DETECT_COMMAND, stdout="no id\n")],
            }
        ),
    )
    assert no_detect["blockers"] == ["blocked_gatemate_dirtyjtag_not_detected"]

    gate_entry = _gatemate_entry(tmp_path)
    _write_manifest(tmp_path, {"gatemate": gate_entry})
    safe_manifest = mod.load_safe_workload_manifest(tmp_path)
    gate_command = mod.gatemate_workload_command(tmp_path, safe_manifest["workloads"]["gatemate"])
    failed_gate = mod.run_gatemate_checks(
        tmp_path,
        safe_manifest,
        RecordingRunner(
            {
                mod.GATEMATE_COMMAND_AVAILABLE_COMMAND: [
                    _probe(mod.GATEMATE_COMMAND_AVAILABLE_COMMAND, stdout="/bin/openFPGALoader\n")
                ],
                mod.GATEMATE_DETECT_COMMAND: [
                    _probe(mod.GATEMATE_DETECT_COMMAND, stdout="IDCode : 0x20000001\n")
                ],
                gate_command: [_probe(gate_command, exit_code=1, stderr="program failed\n")],
            }
        ),
    )
    assert failed_gate["blockers"] == ["gatemate_workload_command_failed"]

    assert mod.parse_probe_json(None) == {}
    assert mod.parse_probe_json(_probe(mod.POLARFIRE_SSH_COMMAND, stdout="not-json\n")) == {}
    assert mod.output_blockers("unit", {}, None, {}) == ["unit_workload_not_attempted"]
    assert mod.output_blockers(
        "unit", {}, _probe(mod.POLARFIRE_SSH_COMMAND, exit_code=1), {}
    ) == ["unit_workload_command_failed"]
    safe_workload = {"safe": True, "workload_hash": "a" * 64, "executable_hash": None}
    assert mod.output_blockers(
        "unit", safe_workload, _probe(mod.POLARFIRE_SSH_COMMAND), {"workload_sha256": "b" * 64}
    ) == ["unit_workload_output_hash_mismatch"]
    assert mod.output_blockers(
        "unit", safe_workload, _probe(mod.POLARFIRE_SSH_COMMAND), {"workload_sha256": "a" * 64}
    ) == ["unit_sample_quality_or_correctness_missing"]

    (tmp_path / mod.SAFE_WORKLOAD_MANIFEST_REL_PATH).unlink()
    _write_exp5141_descriptors(tmp_path)
    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=RecordingRunner(_base_probes()),
        clock=FlatClock(),
        run_date="20260702",
    )
    assert "sample_quality_evidence must be a dict" in mod.artifact_schema_errors(
        dict(artifact, sample_quality_evidence=[])
    )


def test_scenario_hw_5144_run_experiment_and_cli_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-HW-5144: run_experiment and script entrypoints write the artifact."""

    _write_exp5141_descriptors(tmp_path)
    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=RecordingRunner(_base_probes()),
        clock=FlatClock(),
        run_date="20260702",
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["spec_refs"] == ["REQ-HW-5144", "SCENARIO-HW-5144"]
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
    assert mod.main(["--date", "20260702", "--repo-root", str(tmp_path)]) == 0
    assert called["kwargs"]["run_date"] == "20260702"
    assert "hardware_workload_transcripts_ready" in capsys.readouterr().out

    script = importlib.import_module("scripts.experiment_5144_authenticated_board_workload_v471")
    monkeypatch.setattr(script, "experiment_main", lambda argv: 23)
    assert script.main(["--date", "20260702"]) == 23
