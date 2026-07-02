"""Tests for Exp 5166 hardware continuity board timing.

Spec refs: REQ-HW-5166, SCENARIO-HW-5166.
"""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

from carnot import experiment_5166_hardware_continuity_board_timing as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "fpga" / "spec.md"


class RecordingRunner:
    """SCENARIO-HW-5166 runner that returns exact command transcripts."""

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
    """Deterministic REQ-HW-5166 clock for checksum-stable tests."""

    def __call__(self) -> float:
        return 5166.0


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _write_wishlist_update(root: Path) -> None:
    path = root / "research-hardware-wishlist.md"
    path.write_text(
        "# Hardware Wishlist\n\n"
        "### Active hardware tracks (test fixture)\n\n"
        "| Track | Why active now | Boundary |\n"
        "|---|---|---|\n"
        + "\n".join(f"| {marker} | test status | no speedup claim |" for marker in mod.WISHLIST_MARKERS)
        + "\n",
        encoding="utf-8",
    )


def _board_stdout(workload_hash: str, executable_hash: str, expected_energy: int) -> str:
    return (
        json.dumps(
            {
                "correctness": {"energy_matches_expected": True},
                "duration_s": 0.000123,
                "energy": expected_energy,
                "executable_sha256": executable_hash,
                "inference_substrate": mod.INFERENCE_SUBSTRATE,
                "sample_quality": {"finite_energy": True, "sample_count": 8},
                "workload_sha256": workload_hash,
            },
            sort_keys=True,
        )
        + "\n"
    )


def _all_reachable_probes() -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    kv260_command = mod.kv260_workload_command()
    polarfire_command = mod.polarfire_workload_command()
    return {
        mod.KV260_PRECONDITION_COMMAND: [
            _probe(mod.KV260_PRECONDITION_COMMAND, duration_s=0.2)
        ],
        kv260_command: [
            _probe(
                kv260_command,
                stdout=_board_stdout(
                    mod.KV260_WORKLOAD_HASH,
                    mod.INLINE_EXECUTABLE_HASH,
                    mod.KV260_EXPECTED_ENERGY,
                ),
                duration_s=0.31,
            )
        ],
        mod.GATEMATE_DETECT_COMMAND: [
            _probe(
                mod.GATEMATE_DETECT_COMMAND,
                stdout="IDCode : 0x20000001 colognechip GateMate Series GM1Ax\n",
                duration_s=0.4,
            ),
            _probe(
                mod.GATEMATE_DETECT_COMMAND,
                stdout="IDCode : 0x20000001 colognechip GateMate Series GM1Ax\n",
                duration_s=0.41,
            ),
        ],
        mod.POLARFIRE_PRECONDITION_COMMAND: [
            _probe(mod.POLARFIRE_PRECONDITION_COMMAND, duration_s=0.5)
        ],
        polarfire_command: [
            _probe(
                polarfire_command,
                stdout=_board_stdout(
                    mod.POLARFIRE_WORKLOAD_HASH,
                    mod.INLINE_EXECUTABLE_HASH,
                    mod.POLARFIRE_EXPECTED_ENERGY,
                ),
                duration_s=0.61,
            )
        ],
    }


def _valid_artifact(tmp_path: Path) -> dict[str, object]:
    _write_wishlist_update(tmp_path)
    return mod.build_artifact(
        repo_root=tmp_path,
        command_runner=RecordingRunner(_all_reachable_probes()),
        clock=FlatClock(),
        run_date="20260702",
    )


def _with_checksum(artifact: dict[str, object], **updates: object) -> dict[str, object]:
    bad = copy.deepcopy(artifact)
    bad.update(updates)
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    return bad


def test_req_hw_5166_spec_declares_board_timing_contract() -> None:
    """REQ-HW-5166: OpenSpec anchors the v473 combined board artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5166") :]

    for marker in (
        "REQ-HW-5166",
        "SCENARIO-HW-5166",
        "experiment_5166_hardware_continuity_board_timing_v473.json",
        "inference_substrate=hardware_smoke",
        "blocked_gatemate_dirtyjtag_idcode",
        "boards_reachable_count",
        "hardware_wishlist_updated=true",
        "no_speedup_claim=true",
        "hardware_speedup_claimed=false",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_hw_5166_all_reachable_writes_hash_verified_transcripts(tmp_path: Path) -> None:
    """SCENARIO-HW-5166: reachable boards get hash-verified timing transcripts."""

    _write_wishlist_update(tmp_path)
    runner = RecordingRunner(_all_reachable_probes())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=FlatClock(),
        run_date="20260702",
    )
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [
        mod.KV260_PRECONDITION_COMMAND,
        mod.kv260_workload_command(),
        mod.GATEMATE_DETECT_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_PRECONDITION_COMMAND,
        mod.polarfire_workload_command(),
    ]
    assert saved["experiment_id"] == mod.EXPERIMENT_ID
    assert saved["milestone"] == "2026.07.473"
    assert saved["honest_verdict"].startswith("complete_")
    assert saved["inference_substrate"] == "hardware_smoke"
    assert saved["boards_reachable_count"] == 3
    assert saved["hardware_wishlist_updated"] is True
    assert saved["kv260_result"]["reachable"] is True
    assert saved["kv260_result"]["workload_hash"] == mod.KV260_WORKLOAD_HASH
    assert saved["kv260_result"]["hash_verified"] is True
    assert saved["gatemate_result"]["reachable"] is True
    assert saved["gatemate_result"]["workload_hash"] == mod.GATEMATE_WORKLOAD_HASH
    assert saved["gatemate_result"]["hash_verified"] is True
    assert saved["polarfire_result"]["reachable"] is True
    assert saved["polarfire_result"]["workload_hash"] == mod.POLARFIRE_WORKLOAD_HASH
    assert saved["sample_quality_evidence"]["speedup_evidence_claimed"] is False
    assert saved["no_speedup_claim"] is True
    assert saved["hardware_speedup_claimed"] is False
    assert saved["kv260_host_block_devices_touched"] is False
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "/dev/mmcblk" not in json.dumps(saved, sort_keys=True).lower()
    mod.validate_artifact(saved)


def test_scenario_hw_5166_per_board_blockers_continue(tmp_path: Path) -> None:
    """SCENARIO-HW-5166: blocked GateMate/PolarFire do not block reachable KV260."""

    _write_wishlist_update(tmp_path)
    kv260_command = mod.kv260_workload_command()
    runner = RecordingRunner(
        {
            mod.KV260_PRECONDITION_COMMAND: [_probe(mod.KV260_PRECONDITION_COMMAND)],
            kv260_command: [
                _probe(
                    kv260_command,
                    stdout=_board_stdout(
                        mod.KV260_WORKLOAD_HASH,
                        mod.INLINE_EXECUTABLE_HASH,
                        mod.KV260_EXPECTED_ENERGY,
                    ),
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, stdout="Jtag frequency : requested 6000000 Hz\n")
            ],
            mod.POLARFIRE_PRECONDITION_COMMAND: [
                _probe(mod.POLARFIRE_PRECONDITION_COMMAND, exit_code=255, stderr="timeout\n")
            ],
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=FlatClock(),
        run_date="20260702",
    )

    assert runner.commands == [
        mod.KV260_PRECONDITION_COMMAND,
        mod.kv260_workload_command(),
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_PRECONDITION_COMMAND,
    ]
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["boards_reachable_count"] == 1
    assert artifact["kv260_result"]["reachable"] is True
    assert artifact["gatemate_result"]["reachable"] is False
    assert artifact["gatemate_result"]["blocked_reason"] == "blocked_gatemate_dirtyjtag_idcode"
    assert artifact["gatemate_result"]["latency_transcript"] is None
    assert artifact["polarfire_result"]["reachable"] is False
    assert artifact["polarfire_result"]["blocked_reason"] == "blocked_polarfire_ssh"
    assert artifact["polarfire_result"]["latency_transcript"] is None
    assert artifact["sample_quality_evidence"]["reachable_boards"] == ["kv260"]
    mod.validate_artifact(artifact)


def test_req_hw_5166_validation_rejects_overclaims_and_schema_drift(tmp_path: Path) -> None:
    """REQ-HW-5166: validation rejects overclaims, bad reachability, and unsafe storage."""

    artifact = _valid_artifact(tmp_path)

    for field, value in (
        ("inference_substrate", "local_board_transcripts_or_blocked"),
        ("no_speedup_claim", False),
        ("hardware_speedup_claimed", True),
        ("kv260_host_block_devices_touched", True),
        ("hardware_wishlist_updated", False),
    ):
        bad = _with_checksum(artifact, **{field: value})
        with pytest.raises(ValueError, match=field):
            mod.validate_artifact(bad)

    bad_verdict = _with_checksum(artifact, honest_verdict="blocked_one_board_unreachable")
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_count = _with_checksum(artifact, boards_reachable_count=2)
    with pytest.raises(ValueError, match="boards_reachable_count"):
        mod.validate_artifact(bad_count)

    bad_result = copy.deepcopy(artifact)
    bad_result["kv260_result"]["reachable"] = "yes"
    bad_result["reproducibility_checksum"] = mod.payload_checksum(bad_result)
    with pytest.raises(ValueError, match="kv260_result"):
        mod.validate_artifact(bad_result)

    bad_storage = copy.deepcopy(artifact)
    bad_storage["command_transcripts"] = {"unsafe": {"command": "ls /dev/" + "mmcblk0"}}
    bad_storage["reproducibility_checksum"] = mod.payload_checksum(bad_storage)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(bad_storage)

    bad_checksum = dict(artifact, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)

    assert "missing required fields" in "; ".join(mod.artifact_schema_errors({}))
    assert "kv260_result must be a dict" in mod.artifact_schema_errors(
        _with_checksum(artifact, kv260_result=[])
    )


def test_req_hw_5166_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-HW-5166: helper parsing and hash checks fail closed."""

    assert mod.ising_energy(mod.KV260_WORKLOAD) == mod.KV260_EXPECTED_ENERGY
    assert mod.parse_probe_json(None) == {}
    assert mod.parse_probe_json(_probe(mod.KV260_PRECONDITION_COMMAND, stdout="not-json\n")) == {}
    assert mod.parse_probe_json(_probe(mod.KV260_PRECONDITION_COMMAND, stdout="[1, 2]\n")) == {}
    assert mod.wishlist_has_update(tmp_path) is False

    output = {
        "workload_sha256": "b" * 64,
        "executable_sha256": mod.INLINE_EXECUTABLE_HASH,
        "sample_quality": {"finite_energy": True},
        "correctness": {"energy_matches_expected": True},
    }
    assert mod.ssh_output_hash_verified("a" * 64, output) is False
    assert mod.ssh_output_has_evidence({}) is False
    assert mod.ssh_workload_blocker("unit", None, "a" * 64, {}) == "blocked_unit_workload_missing"
    assert (
        mod.ssh_workload_blocker(
            "unit",
            _probe(mod.KV260_PRECONDITION_COMMAND, exit_code=1),
            "a" * 64,
            {},
        )
        == "blocked_unit_workload_command"
    )
    evidence_missing = {
        "workload_sha256": "a" * 64,
        "executable_sha256": mod.INLINE_EXECUTABLE_HASH,
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
    }
    assert (
        mod.ssh_workload_blocker(
            "unit",
            _probe(mod.KV260_PRECONDITION_COMMAND),
            "a" * 64,
            evidence_missing,
        )
        == "blocked_unit_workload_evidence"
    )

    gate_command_failed = mod.run_gatemate_board(
        RecordingRunner(
            {
                mod.GATEMATE_DETECT_COMMAND: [
                    _probe(
                        mod.GATEMATE_DETECT_COMMAND,
                        stdout="IDCode : 0x20000001 colognechip GateMate Series GM1Ax\n",
                    ),
                    _probe(mod.GATEMATE_DETECT_COMMAND, exit_code=1, stderr="detect failed\n"),
                ]
            }
        )
    )
    assert gate_command_failed["blocked_reason"] == "blocked_gatemate_workload_command"

    gate_idcode_failed = mod.run_gatemate_board(
        RecordingRunner(
            {
                mod.GATEMATE_DETECT_COMMAND: [
                    _probe(
                        mod.GATEMATE_DETECT_COMMAND,
                        stdout="IDCode : 0x20000001 colognechip GateMate Series GM1Ax\n",
                    ),
                    _probe(mod.GATEMATE_DETECT_COMMAND, stdout="IDCode : 0x00000000\n"),
                ]
            }
        )
    )
    assert gate_idcode_failed["blocked_reason"] == "blocked_gatemate_workload_idcode"

    _write_wishlist_update(tmp_path)
    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=RecordingRunner(
            {
                mod.KV260_PRECONDITION_COMMAND: [
                    _probe(mod.KV260_PRECONDITION_COMMAND, exit_code=255, stderr="no route\n")
                ],
                mod.GATEMATE_DETECT_COMMAND: [
                    _probe(mod.GATEMATE_DETECT_COMMAND, exit_code=1, stderr="fails to open\n")
                ],
                mod.POLARFIRE_PRECONDITION_COMMAND: [_probe(mod.POLARFIRE_PRECONDITION_COMMAND)],
                mod.polarfire_workload_command(): [
                    _probe(mod.polarfire_workload_command(), stdout=json.dumps(output) + "\n")
                ],
            }
        ),
        clock=FlatClock(),
        run_date="20260702",
    )
    assert artifact["kv260_result"]["blocked_reason"] == "blocked_kv260_ssh"
    assert artifact["gatemate_result"]["blocked_reason"] == "blocked_gatemate_dirtyjtag"
    assert artifact["polarfire_result"]["blocked_reason"] == "blocked_polarfire_workload_hash"
    assert artifact["boards_reachable_count"] == 1

    partial_root = tmp_path / "partial"
    partial_root.mkdir()
    partial_doc = partial_root / "research-hardware-wishlist.md"
    partial_doc.write_text("# Hardware Wishlist\n", encoding="utf-8")
    assert mod.ensure_hardware_wishlist_update(partial_root) == partial_doc
    assert mod.wishlist_has_update(partial_root) is True
    already = partial_doc.read_text(encoding="utf-8")
    assert mod.ensure_hardware_wishlist_update(partial_root) == partial_doc
    assert partial_doc.read_text(encoding="utf-8") == already


def test_scenario_hw_5166_run_experiment_and_cli_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-HW-5166: run_experiment and script entrypoints write the artifact."""

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=RecordingRunner(_all_reachable_probes()),
        clock=FlatClock(),
        run_date="20260702",
        update_wishlist=True,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["spec_refs"] == ["REQ-HW-5166", "SCENARIO-HW-5166"]
    assert artifact["hardware_wishlist_updated"] is True
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
    assert "boards_reachable_count" in capsys.readouterr().out

    script = importlib.import_module(
        "scripts.experiment_5166_hardware_continuity_board_timing_v473"
    )
    monkeypatch.setattr(script, "experiment_main", lambda argv: 19)
    assert script.main(["--date", "20260702"]) == 19
