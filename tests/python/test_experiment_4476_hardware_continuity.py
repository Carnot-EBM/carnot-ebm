"""Tests for Exp 4476 attached-board hardware continuity.

Spec refs: REQ-HW-4476, SCENARIO-HW-4476.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4476_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4476 command runner with queued board transcripts."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        del timeout_s
        command = tuple(command)
        self.commands.append(command)
        if command in self.probes and self.probes[command]:
            return self.probes[command].pop(0)
        if command and command[0].endswith("gmpack"):
            output_path = Path(command[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"REQ-HW-4476 packed bitstream\n")
            return _probe(command, stdout="GateMate pack ok\n", duration_s=0.21)
        raise AssertionError(f"unexpected command: {command!r}")


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _write_previous_4463(repo_root: Path, per_board_status: object | None = None) -> None:
    results = repo_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "experiment_4463_hardware_continuity.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete: hardware_continuity_4463",
                "per_board_status": (
                    per_board_status
                    if per_board_status is not None
                    else {
                        "kv260": {"status": "kv260_latency_transcript_recorded"},
                        "gatemate": {"status": "blocked_gatemate_dirtyjtag_unreachable"},
                        "polarfire": {"status": "polarfire_sampler_smoke_recorded"},
                    }
                ),
            }
        ),
        encoding="utf-8",
    )


def _seed_gatemate_cfg(repo_root: Path) -> Path:
    cfg = (
        repo_root
        / "build"
        / "gatemate"
        / "experiment_3866_gatemate_ising_tile_flash_v2"
        / "gatemate_ising_n16.cfg.bit"
    )
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("# REQ-HW-4476 fake routed config\n", encoding="utf-8")
    return cfg


def _reachable_runner() -> RecordingRunner:
    return RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION, duration_s=0.11)],
            mod.GATEMATE_NEXTPNR_PRECONDITION: [
                _probe(
                    mod.GATEMATE_NEXTPNR_PRECONDITION,
                    stdout="/opt/oss-cad-suite/bin/nextpnr-himbaechel\n",
                    duration_s=0.12,
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="idcode 0x20000001 colognechip GateMate GM1Ax\n",
                    duration_s=0.13,
                )
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, duration_s=0.14)
            ],
            mod.KV260_LATENCY_TRANSCRIPT_COMMAND: [
                _probe(
                    mod.KV260_LATENCY_TRANSCRIPT_COMMAND,
                    stdout=(
                        '{"schema":"carnot.kv260.ssh_latency_transcript.v1",'
                        '"sample_count":3,"per_sample_us":[1.0,2.0,3.0]}\n'
                    ),
                    duration_s=0.15,
                )
            ],
            mod.GATEMATE_GMPACK_LOOKUP_COMMAND: [
                _probe(
                    mod.GATEMATE_GMPACK_LOOKUP_COMMAND,
                    stdout="/opt/oss-cad-suite/bin/gmpack\n",
                    duration_s=0.16,
                )
            ],
            mod.POLARFIRE_SAMPLER_SMOKE_COMMAND: [
                _probe(
                    mod.POLARFIRE_SAMPLER_SMOKE_COMMAND,
                    stdout='{"schema":"carnot.polarfire.sampler_smoke.v1","sample_count":4}\n',
                    duration_s=0.17,
                )
            ],
        }
    )


def test_req_hw_4476_spec_entry_declares_required_artifact_contract() -> None:
    """REQ-HW-4476: OpenSpec anchors fields, commands, and blocked fallbacks."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4476" in spec
    assert "SCENARIO-HW-4476" in spec
    assert "experiment_4476_hardware_continuity.json" in spec
    assert "experiment_4463_hardware_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "command -v nextpnr-himbaechel" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "nextpnr-gatemate" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "blocked_kv260_ssh_unreachable" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_hw_4476_reachable_boards_take_one_forward_step(tmp_path: Path) -> None:
    """SCENARIO-HW-4476: reachable boards record latency, pack, and smoke steps."""
    _write_previous_4463(tmp_path)
    cfg = _seed_gatemate_cfg(tmp_path)
    runner = _reachable_runner()

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner)
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands[:4] == [
        mod.KV260_SSH_PRECONDITION,
        mod.GATEMATE_NEXTPNR_PRECONDITION,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
    ]
    assert mod.KV260_LATENCY_TRANSCRIPT_COMMAND in runner.commands
    assert mod.GATEMATE_GMPACK_LOOKUP_COMMAND in runner.commands
    assert mod.POLARFIRE_SAMPLER_SMOKE_COMMAND in runner.commands
    assert all("nextpnr-gatemate" not in " ".join(command) for command in runner.commands)
    assert saved["schema"] == mod.SCHEMA
    assert saved["experiment"] == mod.EXPERIMENT_ID
    assert saved["spec_refs"] == mod.SPEC_REFS
    assert saved["random_seed"] == mod.RANDOM_SEED
    assert saved["source_context"]["previous_experiment"] == 4463
    assert saved["source_context"]["previous_artifact_read"] is True
    assert saved["preconditions_checked"] == [
        {
            "resource": "kv260_ssh",
            "available": True,
            "command": mod.command_to_string(mod.KV260_SSH_PRECONDITION),
            "exit_code": 0,
            "duration_s": 0.11,
            "observed": "returncode=0",
        },
        {
            "resource": "gatemate_nextpnr_himbaechel",
            "available": True,
            "command": mod.command_to_string(mod.GATEMATE_NEXTPNR_PRECONDITION),
            "exit_code": 0,
            "duration_s": 0.12,
            "observed": "/opt/oss-cad-suite/bin/nextpnr-himbaechel",
        },
        {
            "resource": "gatemate_dirtyjtag_detect",
            "available": True,
            "command": mod.command_to_string(mod.GATEMATE_DETECT_COMMAND),
            "exit_code": 0,
            "duration_s": 0.13,
            "observed": "idcode 0x20000001 colognechip GateMate GM1Ax",
        },
        {
            "resource": "polarfire_ssh",
            "available": True,
            "command": mod.command_to_string(mod.POLARFIRE_SSH_PRECONDITION),
            "exit_code": 0,
            "duration_s": 0.14,
            "observed": "returncode=0",
        },
    ]
    assert saved["field_principles"]["inference_substrate"] == (
        "hardware_smoke -- SSH-attached board test; per-board floor"
    )
    assert saved["inference_substrate"] == "hardware_smoke"
    assert saved["per_board_status"]["kv260"]["status"] == "kv260_latency_transcript_recorded"
    assert saved["per_board_status"]["kv260"]["step"] == "latency_transcript"
    assert saved["per_board_status"]["gatemate"]["status"] == "gatemate_bitstream_pack_succeeded"
    assert saved["per_board_status"]["gatemate"]["step"] == "bitstream_pack"
    assert saved["per_board_status"]["gatemate"]["config_path"] == str(cfg)
    assert "experiment_4476_hardware_continuity" in saved["per_board_status"]["gatemate"][
        "bitstream_path"
    ]
    assert saved["per_board_status"]["polarfire"]["status"] == "polarfire_sampler_smoke_recorded"
    assert saved["per_board_status"]["polarfire"]["step"] == "sampler_smoke"
    assert saved["honest_verdict"].startswith("complete: hardware_continuity_4476_")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "mmcblk" not in json.dumps(saved).lower()
    assert "nextpnr-gatemate" not in json.dumps(saved)
    mod.validate_artifact(saved)


def test_req_hw_4476_kv260_ssh_block_is_global_but_other_boards_continue(
    tmp_path: Path,
) -> None:
    """REQ-HW-4476: KV260 SSH failure gives the required blocked_ terminal verdict."""
    _seed_gatemate_cfg(tmp_path)
    runner = _reachable_runner()
    runner.probes[mod.KV260_SSH_PRECONDITION] = [
        _probe(mod.KV260_SSH_PRECONDITION, 255, stderr="ssh timeout", duration_s=0.2)
    ]

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner)

    assert artifact["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert mod.KV260_LATENCY_TRANSCRIPT_COMMAND not in runner.commands
    assert mod.GATEMATE_GMPACK_LOOKUP_COMMAND in runner.commands
    assert mod.POLARFIRE_SAMPLER_SMOKE_COMMAND in runner.commands
    assert artifact["per_board_status"]["kv260"]["status"] == "blocked_kv260_ssh_unreachable"
    assert artifact["per_board_status"]["gatemate"]["status"] == "gatemate_bitstream_pack_succeeded"
    assert artifact["per_board_status"]["polarfire"]["status"] == "polarfire_sampler_smoke_recorded"
    mod.validate_artifact(artifact)


def test_req_hw_4476_blocks_unavailable_resources_without_steps(tmp_path: Path) -> None:
    """REQ-HW-4476: failed preconditions emit blocked statuses and one-line audits."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, 255, stderr="ssh timeout", duration_s=0.2)
            ],
            mod.GATEMATE_NEXTPNR_PRECONDITION: [
                _probe(
                    mod.GATEMATE_NEXTPNR_PRECONDITION,
                    1,
                    stderr="missing nextpnr-himbaechel",
                    duration_s=0.3,
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, 1, stdout="no idcode", duration_s=0.4)
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout", duration_s=0.5)
            ],
        }
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.GATEMATE_NEXTPNR_PRECONDITION,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
    ]
    assert saved["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert saved["per_board_status"]["kv260"]["status"] == "blocked_kv260_ssh_unreachable"
    assert saved["per_board_status"]["gatemate"]["status"] == (
        "blocked_gatemate_nextpnr_himbaechel_unavailable"
    )
    assert saved["per_board_status"]["polarfire"]["status"] == "blocked_polarfire_ssh_unreachable"
    assert all("\n" not in board["audit"] for board in saved["per_board_status"].values())
    assert all(board["step"] == "precondition_audit" for board in saved["per_board_status"].values())
    assert [entry["available"] for entry in saved["preconditions_checked"]] == [
        False,
        False,
        False,
        False,
    ]
    mod.validate_artifact(saved)

    detect_block_runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, stdout="", duration_s=0.01)
            ],
            mod.GATEMATE_NEXTPNR_PRECONDITION: [
                _probe(
                    mod.GATEMATE_NEXTPNR_PRECONDITION,
                    stdout="/opt/oss-cad-suite/bin/nextpnr-himbaechel\n",
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, 1, stdout="no idcode")
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout")
            ],
            mod.KV260_LATENCY_TRANSCRIPT_COMMAND: [
                _probe(
                    mod.KV260_LATENCY_TRANSCRIPT_COMMAND,
                    stdout='{"schema":"carnot.kv260.ssh_latency_transcript.v1"}\n',
                )
            ],
        }
    )

    detect_block = mod.build_artifact(repo_root=tmp_path, command_runner=detect_block_runner)

    assert detect_block["per_board_status"]["gatemate"]["status"] == (
        "blocked_gatemate_dirtyjtag_unreachable"
    )
    assert detect_block["honest_verdict"].startswith("complete:")

    _write_previous_4463(tmp_path, per_board_status="bad")
    malformed_prior = mod.build_artifact(repo_root=tmp_path, command_runner=_reachable_runner())

    assert malformed_prior["source_context"]["previous_per_board_status"] == {}


def test_req_hw_4476_reachable_board_step_failures_are_honest_blocks(
    tmp_path: Path,
) -> None:
    """REQ-HW-4476: reachable preconditions can still block during a concrete step."""
    _seed_gatemate_cfg(tmp_path)
    runner = _reachable_runner()
    runner.probes[mod.KV260_LATENCY_TRANSCRIPT_COMMAND] = [
        _probe(mod.KV260_LATENCY_TRANSCRIPT_COMMAND, 1, stderr="python missing", duration_s=0.15)
    ]
    runner.probes[mod.GATEMATE_GMPACK_LOOKUP_COMMAND] = [
        _probe(mod.GATEMATE_GMPACK_LOOKUP_COMMAND, 1, stderr="gmpack missing", duration_s=0.16)
    ]
    runner.probes[mod.POLARFIRE_SAMPLER_SMOKE_COMMAND] = [
        _probe(mod.POLARFIRE_SAMPLER_SMOKE_COMMAND, 1, stderr="smoke failed", duration_s=0.17)
    ]

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner)

    assert artifact["per_board_status"]["kv260"]["status"] == (
        "blocked_kv260_latency_transcript_failed"
    )
    assert artifact["per_board_status"]["gatemate"]["status"] == (
        "blocked_gatemate_gmpack_unavailable"
    )
    assert artifact["per_board_status"]["polarfire"]["status"] == (
        "blocked_polarfire_sampler_smoke_failed"
    )
    assert artifact["honest_verdict"].startswith("complete: hardware_continuity_4476_")
    mod.validate_artifact(artifact)

    cfg = _seed_gatemate_cfg(tmp_path)
    bitstream = mod._gatemate_output_bitstream(tmp_path)
    pack_command = ("/opt/oss-cad-suite/bin/gmpack", str(cfg), str(bitstream))
    pack_fail_runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, stdout="", duration_s=0.01)
            ],
            mod.GATEMATE_NEXTPNR_PRECONDITION: [
                _probe(
                    mod.GATEMATE_NEXTPNR_PRECONDITION,
                    stdout="/opt/oss-cad-suite/bin/nextpnr-himbaechel\n",
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="idcode 0x20000001 colognechip GateMate GM1Ax\n",
                )
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout")
            ],
            mod.KV260_LATENCY_TRANSCRIPT_COMMAND: [
                _probe(
                    mod.KV260_LATENCY_TRANSCRIPT_COMMAND,
                    stdout='{"schema":"carnot.kv260.ssh_latency_transcript.v1"}\n',
                )
            ],
            mod.GATEMATE_GMPACK_LOOKUP_COMMAND: [
                _probe(
                    mod.GATEMATE_GMPACK_LOOKUP_COMMAND,
                    stdout="/opt/oss-cad-suite/bin/gmpack\n",
                )
            ],
            pack_command: [_probe(pack_command, 1, stderr="pack failed")],
        }
    )

    pack_fail = mod.build_artifact(repo_root=tmp_path, command_runner=pack_fail_runner)

    assert pack_fail["per_board_status"]["gatemate"]["status"] == (
        "blocked_gatemate_bitstream_pack_failed"
    )


def test_req_hw_4476_validation_rejects_fabrication_markers(tmp_path: Path) -> None:
    """REQ-HW-4476: validation catches wrappers, SD-card markers, and stale checksums."""
    _seed_gatemate_cfg(tmp_path)
    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=_reachable_runner())
    artifact["preconditions_checked"] = {
        "value": artifact["preconditions_checked"],
        "principle": mod.FIELD_PRINCIPLES["preconditions_checked"],
    }
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)

    with pytest.raises(ValueError, match="preconditions_checked must remain a bare value"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=_reachable_runner())
    artifact["per_board_status"]["kv260"]["audit"] = "checked /dev/mmcblk0"
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)

    with pytest.raises(ValueError, match="SD-card"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=_reachable_runner())
    artifact["per_board_status"]["gatemate"]["audit"] = "used nextpnr-gatemate"
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)

    with pytest.raises(ValueError, match="nextpnr-gatemate"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=_reachable_runner())
    artifact["inference_substrate"] = "cpu"
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)

    with pytest.raises(ValueError, match="wrong substrate"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=_reachable_runner())
    artifact["reproducibility_checksum"] = "stale"

    with pytest.raises(ValueError, match="bad checksum"):
        mod.validate_artifact(artifact)


def test_req_hw_4476_main_script_lives_under_python_package() -> None:
    """REQ-HW-4476: the required package script entry point is present."""
    script = Path("python/carnot/experiment_4476_hardware_continuity.py")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "REQ-HW-4476" in text
    assert "run_experiment" in text
