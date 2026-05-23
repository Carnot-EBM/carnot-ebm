"""Tests for Exp 2929 GateMate flash smoke and timing boundary.

Spec refs: REQ-HW-069, SCENARIO-HW-069.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from carnot.experiment_2929_gatemate_flash_timing_boundary import (
    ARTIFACT_FILENAME,
    EXP2928_FILENAME,
    CommandResult,
    build_artifact,
    command_result_text,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"

REQUIRED_FIELDS = (
    "honest_verdict",
    "gatemate_flash_smoke_ready",
    "board_detected",
    "bitstream_sha256_verified",
    "flash_attempted",
    "flash_transcript_path",
    "board_contact_transcript_path",
    "timing_boundary_ready",
    "timing_claim_allowed",
    "speedup_claim_allowed",
    "blocker",
    "inference_substrate",
    "duration_s",
    "run_date",
)


def _clock(values: list[float]):
    state = iter(values)

    def monotonic() -> float:
        return next(state)

    return monotonic


def _runner(results: dict[tuple[str, ...], CommandResult]):
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        key = tuple(args)
        calls.append(key)
        return results[key]

    return run, calls


def _write_exp2928(repo_root: Path, payload: dict) -> Path:
    path = repo_root / "results" / EXP2928_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exp2929_spec_entry_present() -> None:
    """REQ-HW-069: the FPGA capability spec anchors the physical-board gate."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-069" in spec
    assert "SCENARIO-HW-069" in spec
    assert ARTIFACT_FILENAME in spec


def test_exp2929_missing_exp2928_blocks_before_board_contact(tmp_path: Path) -> None:
    """SCENARIO-HW-069: absent bitstream evidence stops before any board command."""
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([10.0, 10.25]),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_bitstream_missing"
    assert artifact["gatemate_flash_smoke_ready"] is False
    assert artifact["board_detected"] is False
    assert artifact["bitstream_sha256_verified"] is False
    assert artifact["flash_attempted"] is False
    assert artifact["flash_transcript_path"] == ""
    assert artifact["board_contact_transcript_path"] == ""
    assert artifact["timing_boundary_ready"] is False
    assert artifact["timing_claim_allowed"] is False
    assert artifact["speedup_claim_allowed"] is False
    assert artifact["inference_substrate"] == "physical_board_smoke"
    assert artifact["duration_s"] == 0.25
    assert artifact["run_date"] == "20260523"
    assert EXP2928_FILENAME in artifact["blocker"]
    assert calls == []


def test_exp2929_false_exp2928_gate_blocks_before_board_contact(tmp_path: Path) -> None:
    """REQ-HW-069: gatemate_bitstream_built=false is not enough to touch hardware."""
    _write_exp2928(
        tmp_path,
        {
            "honest_verdict": "blocked_constraints_missing",
            "gatemate_bitstream_built": False,
        },
    )
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([20.0, 20.5]),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_bitstream_missing"
    assert artifact["blocker"] == "exp2928 gatemate_bitstream_built is false"
    assert artifact["flash_attempted"] is False
    assert calls == []


def test_exp2929_missing_bitstream_path_or_sha_blocks_before_board_contact(
    tmp_path: Path,
) -> None:
    """REQ-HW-069: built=true still requires a concrete bitstream path and SHA."""
    _write_exp2928(
        tmp_path,
        {
            "gatemate_bitstream_built": True,
            "bitstream_path": "build/gatemate/missing.bit",
            "bitstream_sha256": "",
        },
    )
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([25.0, 25.5]),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_bitstream_missing"
    assert "build/gatemate/missing.bit" in artifact["blocker"]
    assert artifact["board_detected"] is False
    assert artifact["flash_attempted"] is False
    assert calls == []


def test_exp2929_board_detect_failure_records_raw_contact_transcript(tmp_path: Path) -> None:
    """REQ-HW-069: failed board detection is preserved as a transcript, not a claim."""
    bitstream = tmp_path / "build" / "gatemate" / "ising.bit"
    bitstream.parent.mkdir(parents=True)
    bitstream.write_bytes(b"ready")
    _write_exp2928(
        tmp_path,
        {
            "gatemate_bitstream_built": True,
            "bitstream_path": str(bitstream),
            "bitstream_sha256": hashlib.sha256(b"ready").hexdigest(),
        },
    )
    detect = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
    runner, calls = _runner({detect: CommandResult(1, "", "no device found\n")})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([27.0, 28.0]),
    )

    contact_path = Path(artifact["board_contact_transcript_path"])
    assert artifact["honest_verdict"] == "blocked_board_not_detected"
    assert artifact["board_detected"] is False
    assert artifact["flash_attempted"] is False
    assert "no device found" in contact_path.read_text(encoding="utf-8")
    assert calls == [detect]


def test_exp2929_detects_board_then_blocks_on_sha_mismatch(tmp_path: Path) -> None:
    """REQ-HW-069: a flash attempt requires the local bitstream hash to match Exp 2928."""
    bitstream = tmp_path / "build" / "gatemate" / "ising.bit"
    bitstream.parent.mkdir(parents=True)
    bitstream.write_bytes(b"test-bitstream")
    _write_exp2928(
        tmp_path,
        {
            "gatemate_bitstream_built": True,
            "bitstream_path": str(bitstream),
            "bitstream_sha256": "0" * 64,
        },
    )
    detect = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
    runner, calls = _runner(
        {
            detect: CommandResult(0, "IDCODE: 0x20000001 GateMate GM1Ax\n", ""),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([30.0, 31.0]),
    )

    contact_path = Path(artifact["board_contact_transcript_path"])
    assert artifact["honest_verdict"] == "blocked_bitstream_sha256_mismatch"
    assert artifact["board_detected"] is True
    assert artifact["bitstream_sha256_verified"] is False
    assert artifact["flash_attempted"] is False
    assert artifact["flash_transcript_path"] == ""
    assert contact_path.exists()
    transcript = contact_path.read_text(encoding="utf-8")
    assert "COMMAND: openFPGALoader -c dirtyJtag --detect" in transcript
    assert "IDCODE: 0x20000001 GateMate GM1Ax" in transcript
    assert calls == [detect]


def test_exp2929_blocks_when_documented_flash_path_is_missing(tmp_path: Path) -> None:
    """REQ-HW-069: openFPGALoader flash is allowed only when the board path is documented."""
    bitstream = tmp_path / "ising.bit"
    bitstream.write_bytes(b"hash-matches")
    digest = hashlib.sha256(b"hash-matches").hexdigest()
    _write_exp2928(
        tmp_path,
        {
            "gatemate_bitstream_built": True,
            "bitstream_path": str(bitstream),
            "bitstream_sha256": digest,
        },
    )
    (tmp_path / "research-hardware-wishlist.md").write_text(
        "GateMate board present, flash command intentionally absent.\n",
        encoding="utf-8",
    )
    detect = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
    runner, _calls = _runner(
        {
            detect: CommandResult(0, "GateMate Series GM1Ax IDCODE 0x20000001\n", ""),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([40.0, 40.75]),
    )

    assert artifact["honest_verdict"] == "blocked_flash_path_undocumented"
    assert artifact["bitstream_sha256_verified"] is True
    assert artifact["flash_attempted"] is False
    assert artifact["blocker"] == "documented GateMate openFPGALoader flash path missing"


def test_exp2929_flash_failure_preserves_flash_transcript(tmp_path: Path) -> None:
    """REQ-HW-069: a failed flash is reported as contact evidence, not success."""
    bitstream = tmp_path / "ising.bit"
    bitstream.write_bytes(b"flash-fails")
    digest = hashlib.sha256(b"flash-fails").hexdigest()
    _write_exp2928(
        tmp_path,
        {
            "gatemate_bitstream_built": True,
            "bitstream_path": str(bitstream),
            "bitstream_sha256": digest,
        },
    )
    (tmp_path / "research-hardware-wishlist.md").write_text(
        "Use `openFPGALoader -c dirtyJtag -b olimex_gatemateevb <bit>`.\n",
        encoding="utf-8",
    )
    detect = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
    flash = (
        "openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        "olimex_gatemateevb",
        str(bitstream),
    )
    runner, calls = _runner(
        {
            detect: CommandResult(0, "GateMate IDCODE 0x20000001\n", ""),
            flash: CommandResult(2, "", "flash parse error\n"),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([45.0, 46.0]),
    )

    assert artifact["honest_verdict"] == "blocked_flash_failed"
    assert artifact["flash_attempted"] is True
    assert artifact["gatemate_flash_smoke_ready"] is False
    assert "flash parse error" in Path(artifact["flash_transcript_path"]).read_text(
        encoding="utf-8"
    )
    assert calls == [detect, flash]


def test_exp2929_flash_contact_success_keeps_timing_boundary_false(tmp_path: Path) -> None:
    """REQ-HW-069: a flash/contact smoke still cannot invent a timing benchmark."""
    bitstream = tmp_path / "ising.bit"
    bitstream.write_bytes(b"flashable")
    digest = hashlib.sha256(b"flashable").hexdigest()
    _write_exp2928(
        tmp_path,
        {
            "gatemate_bitstream_built": True,
            "bitstream_path": str(bitstream),
            "bitstream_sha256": digest,
        },
    )
    (tmp_path / "research-hardware-wishlist.md").write_text(
        "Use `openFPGALoader -c dirtyJtag -b olimex_gatemateevb <bit>`.\n",
        encoding="utf-8",
    )
    detect = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
    flash = (
        "openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        "olimex_gatemateevb",
        str(bitstream),
    )
    runner, calls = _runner(
        {
            detect: CommandResult(0, "GateMate IDCODE 0x20000001\n", ""),
            flash: CommandResult(0, "Load SRAM via JTAG: 100%\n", ""),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([50.0, 51.5]),
    )

    assert artifact["honest_verdict"] == "gatemate_flash_contact_smoke_no_timing_counter"
    assert artifact["gatemate_flash_smoke_ready"] is True
    assert artifact["board_detected"] is True
    assert artifact["bitstream_sha256_verified"] is True
    assert artifact["flash_attempted"] is True
    assert Path(artifact["flash_transcript_path"]).read_text(encoding="utf-8").endswith(
        "STDERR:\n"
    )
    assert artifact["board_contact_transcript_path"] != artifact["flash_transcript_path"]
    assert artifact["timing_boundary_ready"] is False
    assert artifact["timing_claim_allowed"] is False
    assert artifact["speedup_claim_allowed"] is False
    assert calls == [detect, flash, detect]


def test_exp2929_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """SCENARIO-HW-069: run_experiment writes the required v1 deliverable JSON."""
    artifact_path = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=artifact_path,
        run_command=lambda args, timeout_s: CommandResult(99, "", "unexpected"),
        monotonic=_clock([60.0, 60.125]),
    )

    written = json.loads(artifact_path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in written]
    assert not missing
    assert written == artifact
    assert artifact_path.name == ARTIFACT_FILENAME


def test_exp2929_command_result_text_preserves_stdout_and_stderr() -> None:
    """REQ-HW-069: transcripts preserve raw command output content."""
    assert command_result_text(CommandResult(1, "out\n", "err\n")) == "out\nerr\n"
    assert command_result_text(CommandResult(0, "", "")) == ""
