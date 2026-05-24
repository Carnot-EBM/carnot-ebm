"""Tests for Exp 2957 GateMate n=16 flash/timing smoke.

Spec refs: REQ-HW-077, SCENARIO-HW-077.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from carnot.experiment_2957_gatemate_flash_timing_smoke import (
    ARTIFACT_FILENAME,
    EXP2956_FILENAME,
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
    "no_speedup_claim",
    "no_boltzmann_claim",
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


def _which_from(paths: dict[str, str]):
    def which(name: str) -> str | None:
        return paths.get(name)

    return which


def _write_exp2956(
    repo_root: Path,
    *,
    built: bool = True,
    bitstream_bytes: bytes = b"gate-mate-real-bitstream",
    sha_override: str | None = None,
    loader_path: str = "/suite/bin/openFPGALoader",
) -> Path:
    bitstream = repo_root / "build" / "gatemate" / "experiment_2956_gatemate_n16" / "ising_n16_gatemate.bit"
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    bitstream.write_bytes(bitstream_bytes)
    ccf = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.ccf"
    ccf.parent.mkdir(parents=True, exist_ok=True)
    ccf.write_text(
        "# build-only constraints\n# allow-unconstrained\n# no physical Pin_in/Pin_out locations\n",
        encoding="utf-8",
    )
    (repo_root / "research-hardware-wishlist.md").write_text(
        "GateMate flash: openFPGALoader -c dirtyJtag -b olimex_gatemateevb <bit>\n",
        encoding="utf-8",
    )
    sha = hashlib.sha256(bitstream_bytes).hexdigest() if sha_override is None else sha_override
    payload = {
        "honest_verdict": "complete: gatemate_n16_bitstream_built",
        "gatemate_bitstream_built": built,
        "bitstream_path": str(bitstream),
        "bitstream_sha256": sha,
        "constraints_path": str(ccf),
        "inference_substrate": "hardware_build",
        "preconditions_checked": [
            {
                "resource": "openFPGALoader",
                "available": bool(loader_path),
                "path": loader_path,
            }
        ],
    }
    path = repo_root / "results" / EXP2956_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_exp2957_spec_entry_present() -> None:
    """REQ-HW-077: the FPGA spec anchors the flash/timing smoke artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-077" in spec
    assert "SCENARIO-HW-077" in spec
    assert ARTIFACT_FILENAME in spec


def test_exp2957_missing_exp2956_blocks_before_hardware_contact(tmp_path: Path) -> None:
    """REQ-HW-077: missing bitstream evidence blocks before any board command."""
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({}),
        monotonic=_clock([10.0, 10.25]),
    )

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["honest_verdict"] == "blocked_exp2956_bitstream_missing"
    assert artifact["board_detected"] is False
    assert artifact["bitstream_sha256_verified"] is False
    assert artifact["flash_attempted"] is False
    assert artifact["flash_succeeded"] is False
    assert artifact["smoke_vector_passed"] is False
    assert artifact["observed_output_sha256"] == ""
    assert artifact["transcript_paths"] == []
    assert artifact["no_speedup_claim"] is True
    assert artifact["no_boltzmann_claim"] is True
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["duration_s"] == 0.25
    assert calls == []


def test_exp2957_false_exp2956_gate_blocks_before_hardware_contact(tmp_path: Path) -> None:
    """REQ-HW-077: gatemate_bitstream_built=false is not flashable evidence."""
    _write_exp2956(tmp_path, built=False)
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": "/suite/bin/openFPGALoader"}),
        monotonic=_clock([11.0, 11.5]),
    )

    assert artifact["honest_verdict"] == "blocked_exp2956_bitstream_missing"
    assert artifact["failure_excerpt"] == "exp2956 gatemate_bitstream_built is false"
    assert artifact["flash_attempted"] is False
    assert calls == []


def test_exp2957_sha_mismatch_blocks_before_board_detection(tmp_path: Path) -> None:
    """REQ-HW-077: a flash attempt requires the Exp 2956 bitstream hash to match."""
    _write_exp2956(tmp_path, sha_override="0" * 64)
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": "/suite/bin/openFPGALoader"}),
        monotonic=_clock([12.0, 12.75]),
    )

    assert artifact["honest_verdict"] == "blocked_bitstream_sha256_mismatch"
    assert artifact["bitstream_sha256_verified"] is False
    assert artifact["board_detected"] is False
    assert artifact["flash_attempted"] is False
    assert artifact["failure_command"] == "sha256sum"
    assert calls == []


def test_exp2957_missing_openfpgaloader_blocks_before_detection(tmp_path: Path) -> None:
    """REQ-HW-077: openFPGALoader must exist before detection or flash."""
    _write_exp2956(tmp_path, loader_path="")
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({}),
        monotonic=_clock([13.0, 13.5]),
    )

    assert artifact["honest_verdict"] == "blocked_openfpgaloader_missing"
    assert artifact["bitstream_sha256_verified"] is True
    assert artifact["failure_command"] == "command -v openFPGALoader"
    assert artifact["flash_attempted"] is False
    assert calls == []


def test_exp2957_records_generic_readback_reason_without_constraints_path(
    tmp_path: Path,
) -> None:
    """REQ-HW-077: absent IO metadata still blocks sampler-output claims honestly."""
    exp_path = _write_exp2956(tmp_path, loader_path="")
    payload = json.loads(exp_path.read_text(encoding="utf-8"))
    payload.pop("constraints_path")
    exp_path.write_text(json.dumps(payload), encoding="utf-8")
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({}),
        monotonic=_clock([13.75, 14.0]),
    )

    assert artifact["honest_verdict"] == "blocked_openfpgaloader_missing"
    assert artifact["timing_observation"]["readback_reason"] == (
        "No GateMate board output/readback capture command is defined for this bitstream."
    )
    assert artifact["smoke_vector_passed"] is False
    assert calls == []


def test_exp2957_board_detect_failure_records_transcript_and_command(tmp_path: Path) -> None:
    """SCENARIO-HW-077: failed board detection preserves the raw transcript."""
    _write_exp2956(tmp_path)
    detect = ("/suite/bin/openFPGALoader", "-c", "dirtyJtag", "--detect")
    runner, calls = _runner({detect: CommandResult(1, "", "no device found\n")})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({}),
        monotonic=_clock([20.0, 20.1, 20.4, 20.5]),
    )

    transcript = Path(artifact["transcript_paths"][0])
    assert artifact["honest_verdict"] == "blocked_board_not_detected"
    assert artifact["board_detected"] is False
    assert artifact["bitstream_sha256_verified"] is True
    assert artifact["flash_attempted"] is False
    assert artifact["failure_command"] == "/suite/bin/openFPGALoader -c dirtyJtag --detect"
    assert "no device found" in transcript.read_text(encoding="utf-8")
    assert artifact["timing_observation"]["command_durations_s"]["detect"] == 0.3
    assert calls == [detect]


def test_exp2957_undocumented_flash_path_blocks_after_detection(tmp_path: Path) -> None:
    """REQ-HW-077: flashing requires the documented openFPGALoader board path."""
    _write_exp2956(tmp_path)
    (tmp_path / "research-hardware-wishlist.md").write_text(
        "GateMate exists, but no flash command is documented here.\n",
        encoding="utf-8",
    )
    detect = ("/suite/bin/openFPGALoader", "-c", "dirtyJtag", "--detect")
    runner, calls = _runner({detect: CommandResult(0, "GateMate Series GM1Ax\n", "")})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({}),
        monotonic=_clock([30.0, 30.25, 30.75, 31.0]),
    )

    assert artifact["honest_verdict"] == "blocked_flash_path_undocumented"
    assert artifact["board_detected"] is True
    assert artifact["flash_attempted"] is False
    assert artifact["failure_command"] == "openFPGALoader -c dirtyJtag -b olimex_gatemateevb <bitstream>"
    assert calls == [detect]


def test_exp2957_flash_failure_preserves_transcript(tmp_path: Path) -> None:
    """SCENARIO-HW-077: failed flash attempts are transcript-backed blockers."""
    _write_exp2956(tmp_path)
    bitstream = json.loads((tmp_path / "results" / EXP2956_FILENAME).read_text(encoding="utf-8"))[
        "bitstream_path"
    ]
    detect = ("/suite/bin/openFPGALoader", "-c", "dirtyJtag", "--detect")
    flash = (
        "/suite/bin/openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        "olimex_gatemateevb",
        bitstream,
    )
    runner, calls = _runner(
        {
            detect: CommandResult(0, "idcode 0x20000001\nfamily GateMate Series\n", ""),
            flash: CommandResult(2, "", "flash failed\n"),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({}),
        monotonic=_clock([40.0, 40.1, 40.2, 40.3, 40.9, 41.0]),
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


def test_exp2957_successful_flash_records_contact_hashes_without_smoke_claim(
    tmp_path: Path,
) -> None:
    """REQ-HW-077: flash/contact evidence must not become a sampler or speedup claim."""
    _write_exp2956(tmp_path)
    bitstream = json.loads((tmp_path / "results" / EXP2956_FILENAME).read_text(encoding="utf-8"))[
        "bitstream_path"
    ]
    detect = ("/suite/bin/openFPGALoader", "-c", "dirtyJtag", "--detect")
    flash = (
        "/suite/bin/openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        "olimex_gatemateevb",
        bitstream,
    )
    runner, calls = _runner(
        {
            detect: CommandResult(0, "idcode 0x20000001\nfamily GateMate Series\n", ""),
            flash: CommandResult(0, "Load SRAM via JTAG: 100.00%\nDone\n", ""),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({}),
        monotonic=_clock([50.0, 50.1, 50.3, 50.4, 50.9, 51.0, 51.1, 51.25, 51.5]),
    )

    transcript_paths = [Path(path) for path in artifact["transcript_paths"]]
    post_detect_hash = _sha256_file(transcript_paths[2])
    assert artifact["honest_verdict"] == "complete: gatemate_flash_contact_smoke_no_readback"
    assert artifact["board_detected"] is True
    assert artifact["bitstream_sha256_verified"] is True
    assert artifact["flash_attempted"] is True
    assert artifact["flash_succeeded"] is True
    assert artifact["smoke_vector_passed"] is False
    assert artifact["observed_output_sha256"] == post_detect_hash
    assert artifact["transcript_sha256"][str(transcript_paths[2])] == post_detect_hash
    assert artifact["timing_observation"]["timing_source"] == "host_wall_clock_command_duration"
    assert artifact["timing_observation"]["command_durations_s"] == {
        "detect": 0.2,
        "flash": 0.5,
        "post_flash_detect": 0.1,
    }
    assert artifact["timing_observation"]["readback_supported"] is False
    assert "no physical" in artifact["timing_observation"]["readback_reason"]
    assert artifact["timing_observation"]["post_flash_idcode_detected"] is True
    assert artifact["timing_observation"]["post_flash_contact_detected"] is True
    assert artifact["no_speedup_claim"] is True
    assert artifact["no_boltzmann_claim"] is True
    assert calls == [detect, flash, detect]


def test_exp2957_post_flash_jtag_frequency_counts_as_contact_without_idcode(
    tmp_path: Path,
) -> None:
    """REQ-HW-077: post-flash contact can be timing output without sampler readback."""
    _write_exp2956(tmp_path)
    bitstream = json.loads((tmp_path / "results" / EXP2956_FILENAME).read_text(encoding="utf-8"))[
        "bitstream_path"
    ]
    detect = ("/suite/bin/openFPGALoader", "-c", "dirtyJtag", "--detect")
    flash = (
        "/suite/bin/openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        "olimex_gatemateevb",
        bitstream,
    )
    calls_seen: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        key = tuple(args)
        calls_seen.append(key)
        if key == detect and calls_seen.count(detect) == 1:
            return CommandResult(0, "GateMate Series GM1Ax\n", "")
        if key == detect:
            return CommandResult(0, "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n", "")
        if key == flash:
            return CommandResult(0, "Done\n", "")
        raise AssertionError(key)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run,
        which_func=_which_from({}),
        monotonic=_clock([55.0, 55.1, 55.2, 55.3, 55.4, 55.5, 55.6, 55.7, 55.8]),
    )

    assert artifact["honest_verdict"] == "complete: gatemate_flash_contact_smoke_no_readback"
    assert artifact["board_detected"] is True
    assert artifact["flash_succeeded"] is True
    assert artifact["smoke_vector_passed"] is False
    assert artifact["timing_observation"]["post_flash_idcode_detected"] is False
    assert artifact["timing_observation"]["post_flash_contact_detected"] is True


def test_exp2957_post_flash_detection_failure_is_blocked_but_flash_succeeded(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-077: post-flash contact must be rechecked before completion."""
    _write_exp2956(tmp_path)
    bitstream = json.loads((tmp_path / "results" / EXP2956_FILENAME).read_text(encoding="utf-8"))[
        "bitstream_path"
    ]
    detect = ("/suite/bin/openFPGALoader", "-c", "dirtyJtag", "--detect")
    flash = (
        "/suite/bin/openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        "olimex_gatemateevb",
        bitstream,
    )
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
        which_func=_which_from({}),
        monotonic=_clock([60.0, 60.1, 60.2, 60.3, 60.4, 60.5, 60.6, 60.7, 60.8]),
    )

    assert artifact["honest_verdict"] == "blocked_post_flash_board_contact_missing"
    assert artifact["board_detected"] is True
    assert artifact["flash_succeeded"] is True
    assert artifact["flash_attempted"] is True
    assert artifact["timing_observation"]["post_flash_contact_detected"] is False
    assert artifact["observed_output_sha256"] == _sha256_file(Path(artifact["transcript_paths"][2]))


def test_exp2957_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """SCENARIO-HW-077: run_experiment writes the required v2 deliverable JSON."""
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=lambda args, timeout_s: CommandResult(99, "", "unexpected"),
        which_func=_which_from({}),
        monotonic=_clock([70.0, 70.125]),
    )

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in loaded]
    assert missing == []
    assert loaded == artifact
    assert destination.name == ARTIFACT_FILENAME
