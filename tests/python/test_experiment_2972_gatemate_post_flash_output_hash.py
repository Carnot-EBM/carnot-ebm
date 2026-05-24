"""Tests for Exp 2972 GateMate post-flash output-hash smoke.

Spec refs: REQ-HW-079, SCENARIO-HW-079.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from carnot.experiment_2972_gatemate_post_flash_output_hash import (
    ARTIFACT_FILENAME,
    EXP2971_FILENAME,
    CommandResult,
    build_artifact,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"

REQUIRED_FIELDS = (
    "honest_verdict",
    "preconditions_checked",
    "board_detected",
    "bitstream_sha256_verified",
    "flash_attempted",
    "flash_succeeded",
    "smoke_vector_passed",
    "observed_output_sha256",
    "timing_observation",
    "transcript_paths",
    "failure_command",
    "failure_excerpt",
    "no_speedup_claim",
    "no_boltzmann_claim",
    "no_thermalization_claim",
    "inference_substrate",
    "duration_s",
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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_loader(repo_root: Path) -> Path:
    loader = repo_root / "suite" / "bin" / "openFPGALoader"
    loader.parent.mkdir(parents=True, exist_ok=True)
    loader.write_text("#!/bin/sh\n", encoding="utf-8")
    loader.chmod(0o755)
    return loader


def _write_exp2971(
    repo_root: Path,
    *,
    board_detected: bool = True,
    bitstream_sha256_verified: bool = True,
    bitstream_bytes: bytes = b"gate-mate-exp2972-bitstream",
    sha_override: str | None = None,
    flash_command_override: str | None = None,
) -> Path:
    loader = _write_loader(repo_root)
    bitstream = (
        repo_root
        / "build"
        / "gatemate"
        / "experiment_2956_gatemate_n16"
        / "ising_n16_gatemate.bit"
    )
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    bitstream.write_bytes(bitstream_bytes)
    sha = _sha256_bytes(bitstream_bytes) if sha_override is None else sha_override
    payload = {
        "honest_verdict": "complete: gatemate_flash_preconditions_ready",
        "gatemate_board_detected": board_detected,
        "bitstream_sha256_verified": bitstream_sha256_verified,
        "bitstream_path": str(bitstream),
        "bitstream_sha256": sha,
        "flash_command": (
            f"{loader} -c dirtyJtag -b olimex_gatemateevb {bitstream}"
            if flash_command_override is None
            else flash_command_override
        ),
        "detection_commands": [f"{loader} -c dirtyJtag --detect"],
        "inference_substrate": "hardware_preflight",
    }
    path = repo_root / "results" / EXP2971_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_exp2972_spec_entry_present() -> None:
    """REQ-HW-079: the FPGA spec anchors the post-flash smoke artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-079" in spec
    assert "SCENARIO-HW-079" in spec
    assert ARTIFACT_FILENAME in spec


def test_exp2972_missing_exp2971_blocks_before_hardware_contact(tmp_path: Path) -> None:
    """REQ-HW-079: missing Exp 2971 evidence blocks before any board command."""
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([10.0, 10.25]),
    )

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["honest_verdict"] == "blocked_exp2971_preconditions_missing"
    assert artifact["board_detected"] is False
    assert artifact["bitstream_sha256_verified"] is False
    assert artifact["flash_attempted"] is False
    assert artifact["flash_succeeded"] is False
    assert artifact["smoke_vector_passed"] is False
    assert artifact["observed_output_sha256"] == ""
    assert artifact["transcript_paths"] == []
    assert artifact["no_speedup_claim"] is True
    assert artifact["no_boltzmann_claim"] is True
    assert artifact["no_thermalization_claim"] is True
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert calls == []


def test_exp2972_false_exp2971_gate_blocks_before_hardware_contact(tmp_path: Path) -> None:
    """REQ-HW-079: Exp 2971 must prove board detection and SHA verification first."""
    _write_exp2971(tmp_path, board_detected=False, bitstream_sha256_verified=False)
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([11.0, 11.5]),
    )

    assert artifact["honest_verdict"] == "blocked_exp2971_preconditions_missing"
    assert artifact["failure_command"].endswith(EXP2971_FILENAME)
    assert "gatemate_board_detected=false" in artifact["failure_excerpt"]
    assert artifact["flash_attempted"] is False
    assert calls == []


def test_exp2972_missing_exp2971_flash_command_blocks_before_parse(tmp_path: Path) -> None:
    """REQ-HW-079: Exp 2971 must supply the concrete flash command."""
    exp2971 = _write_exp2971(tmp_path)
    payload = json.loads(exp2971.read_text(encoding="utf-8"))
    payload["flash_command"] = ""
    exp2971.write_text(json.dumps(payload), encoding="utf-8")
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([11.75, 12.0]),
    )

    assert artifact["honest_verdict"] == "blocked_exp2971_preconditions_missing"
    assert "flash_command missing" in artifact["failure_excerpt"]
    assert artifact["flash_attempted"] is False
    assert calls == []


def test_exp2972_malformed_exp2971_flash_command_blocks_before_detection(
    tmp_path: Path,
) -> None:
    """REQ-HW-079: a non-flash command from Exp 2971 cannot be guessed into shape."""
    _write_exp2971(tmp_path, flash_command_override="openFPGALoader -c dirtyJtag")
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([12.25, 12.5]),
    )

    assert artifact["honest_verdict"] == "blocked_exp2971_flash_command_missing"
    assert artifact["failure_command"] == "openFPGALoader -c dirtyJtag"
    assert "missing required" in artifact["failure_excerpt"]
    assert artifact["flash_attempted"] is False
    assert calls == []


def test_exp2972_missing_bitstream_path_blocks_before_detection(tmp_path: Path) -> None:
    """REQ-HW-079: the Exp 2971 command must point at an existing bitstream file."""
    exp2971 = _write_exp2971(tmp_path)
    bitstream = Path(json.loads(exp2971.read_text(encoding="utf-8"))["bitstream_path"])
    bitstream.unlink()
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([12.6, 12.9]),
    )

    assert artifact["honest_verdict"] == "blocked_exp2971_bitstream_missing"
    assert artifact["failure_command"] == str(bitstream)
    assert artifact["flash_attempted"] is False
    assert calls == []


def test_exp2972_bitstream_hash_mismatch_blocks_before_detection(tmp_path: Path) -> None:
    """REQ-HW-079: the bitstream SHA256 must be rechecked immediately before flash."""
    _write_exp2971(tmp_path, sha_override="0" * 64)
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([12.0, 12.5]),
    )

    assert artifact["honest_verdict"] == "blocked_bitstream_sha256_mismatch"
    assert artifact["bitstream_sha256_verified"] is False
    assert artifact["board_detected"] is False
    assert artifact["flash_attempted"] is False
    assert artifact["failure_command"].startswith("sha256sum ")
    assert calls == []


def test_exp2972_detection_failure_records_transcript_and_command(tmp_path: Path) -> None:
    """SCENARIO-HW-079: failed pre-flash detection preserves raw transcript evidence."""
    _write_exp2971(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    detect = (loader, "-c", "dirtyJtag", "--detect")
    runner, calls = _runner({detect: CommandResult(1, "", "no device found\n")})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([20.0, 20.1, 20.4, 20.5]),
    )

    transcript = Path(artifact["transcript_paths"][0])
    assert artifact["honest_verdict"] == "blocked_board_not_detected"
    assert artifact["board_detected"] is False
    assert artifact["bitstream_sha256_verified"] is True
    assert artifact["flash_attempted"] is False
    assert artifact["failure_command"] == f"{loader} -c dirtyJtag --detect"
    assert "no device found" in artifact["failure_excerpt"]
    assert "RETURNCODE: 1" in transcript.read_text(encoding="utf-8")
    assert artifact["timing_observation"]["command_durations_s"]["pre_flash_detect"] == 0.3
    assert calls == [detect]


def test_exp2972_flash_failure_preserves_transcript(tmp_path: Path) -> None:
    """SCENARIO-HW-079: failed flashes are honest blocked artifacts."""
    _write_exp2971(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    bitstream = json.loads((tmp_path / "results" / EXP2971_FILENAME).read_text())[
        "bitstream_path"
    ]
    detect = (loader, "-c", "dirtyJtag", "--detect")
    flash = (loader, "-c", "dirtyJtag", "-b", "olimex_gatemateevb", bitstream)
    runner, calls = _runner(
        {
            detect: CommandResult(0, "idcode 0x20000001\nfamily GateMate Series\n", ""),
            flash: CommandResult(2, "", "flash failed\n"),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([30.0, 30.1, 30.2, 30.3, 30.9, 31.0]),
    )

    assert artifact["honest_verdict"] == "blocked_flash_failed"
    assert artifact["board_detected"] is True
    assert artifact["flash_attempted"] is True
    assert artifact["flash_succeeded"] is False
    assert artifact["failure_command"] == " ".join(flash)
    assert len(artifact["transcript_paths"]) == 2
    assert "flash failed" in Path(artifact["transcript_paths"][1]).read_text(encoding="utf-8")
    assert artifact["timing_observation"]["command_durations_s"]["flash"] == 0.6
    assert calls == [detect, flash]


def test_exp2972_successful_flash_records_post_flash_hash_without_sampler_claim(
    tmp_path: Path,
) -> None:
    """REQ-HW-079: successful no-readback smoke hashes the post-flash transcript only."""
    _write_exp2971(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    bitstream = json.loads((tmp_path / "results" / EXP2971_FILENAME).read_text())[
        "bitstream_path"
    ]
    detect = (loader, "-c", "dirtyJtag", "--detect")
    flash = (loader, "-c", "dirtyJtag", "-b", "olimex_gatemateevb", bitstream)
    runner, calls = _runner(
        {
            detect: CommandResult(0, "idcode 0x20000001\nfamily GateMate Series\n", ""),
            flash: CommandResult(0, "Load SRAM via JTAG: 100.00%\nDone\n", ""),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        monotonic=_clock([40.0, 40.1, 40.3, 40.4, 40.9, 41.0, 41.1, 41.25]),
    )

    transcript_paths = [Path(path) for path in artifact["transcript_paths"]]
    post_flash_hash = _sha256_file(transcript_paths[2])
    assert artifact["honest_verdict"] == "complete: gatemate_flash_contact_smoke_no_readback"
    assert artifact["board_detected"] is True
    assert artifact["bitstream_sha256_verified"] is True
    assert artifact["flash_attempted"] is True
    assert artifact["flash_succeeded"] is True
    assert artifact["smoke_vector_passed"] is False
    assert artifact["observed_output_sha256"] == post_flash_hash
    assert artifact["transcript_sha256"][str(transcript_paths[2])] == post_flash_hash
    assert artifact["timing_observation"]["command_durations_s"] == {
        "pre_flash_detect": 0.2,
        "flash": 0.5,
        "post_flash_detect": 0.1,
    }
    assert artifact["timing_observation"]["readback_supported"] is False
    assert artifact["timing_observation"]["post_flash_contact_detected"] is True
    assert artifact["no_speedup_claim"] is True
    assert artifact["no_boltzmann_claim"] is True
    assert artifact["no_thermalization_claim"] is True
    assert calls == [detect, flash, detect]


def test_exp2972_post_flash_detection_failure_is_blocked_but_flash_succeeded(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-079: post-flash contact is checked before completion."""
    _write_exp2971(tmp_path)
    loader = str(tmp_path / "suite" / "bin" / "openFPGALoader")
    bitstream = json.loads((tmp_path / "results" / EXP2971_FILENAME).read_text())[
        "bitstream_path"
    ]
    detect = (loader, "-c", "dirtyJtag", "--detect")
    flash = (loader, "-c", "dirtyJtag", "-b", "olimex_gatemateevb", bitstream)
    calls_seen: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        key = tuple(args)
        calls_seen.append(key)
        if key == detect and calls_seen.count(detect) == 1:
            return CommandResult(0, "GateMate Series GM1Ax\n", "")
        if key == detect:
            return CommandResult(1, "", "post-flash detect timeout\n")
        if key == flash:
            return CommandResult(0, "Done\n", "")
        raise AssertionError(key)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run,
        monotonic=_clock([50.0, 50.1, 50.2, 50.3, 50.4, 50.5, 50.6, 50.7]),
    )

    assert artifact["honest_verdict"] == "blocked_post_flash_board_contact_missing"
    assert artifact["board_detected"] is True
    assert artifact["flash_attempted"] is True
    assert artifact["flash_succeeded"] is True
    assert artifact["smoke_vector_passed"] is False
    assert artifact["failure_command"] == f"{loader} -c dirtyJtag --detect"
    assert artifact["observed_output_sha256"] == _sha256_file(Path(artifact["transcript_paths"][2]))
    assert artifact["timing_observation"]["post_flash_contact_detected"] is False


def test_exp2972_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """SCENARIO-HW-079: run_experiment writes the required v3 deliverable JSON."""
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=lambda args, timeout_s: CommandResult(99, "", "unexpected"),
        monotonic=_clock([60.0, 60.125]),
    )

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in loaded]
    assert missing == []
    assert loaded == artifact
    assert destination.name == ARTIFACT_FILENAME
