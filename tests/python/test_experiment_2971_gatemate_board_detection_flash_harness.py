"""Tests for Exp 2971 GateMate detection/flash-command preflight.

Spec refs: REQ-HW-078, SCENARIO-HW-078.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from carnot.experiment_2971_gatemate_board_detection_flash_harness import (
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
    "gatemate_board_detected",
    "bitstream_sha256_verified",
    "gatemate_flash_preconditions_ready",
    "detection_commands",
    "detection_transcript_paths",
    "bitstream_path",
    "bitstream_sha256",
    "flash_command",
    "failure_command",
    "failure_excerpt",
    "files_changed",
    "inference_substrate",
    "duration_s",
)


def _clock() -> object:
    value = 100.0

    def monotonic() -> float:
        nonlocal value
        value += 0.25
        return value

    return monotonic


def _runner(results: dict[tuple[str, ...], CommandResult]):
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        key = tuple(args)
        calls.append(key)
        if "-b" in key or "olimex_gatemateevb" in key:
            raise AssertionError(f"Exp 2971 must not flash: {key}")
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
) -> Path:
    bitstream = (
        repo_root / "build" / "gatemate" / "experiment_2956_gatemate_n16" / "ising_n16_gatemate.bit"
    )
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    bitstream.write_bytes(bitstream_bytes)
    sha = hashlib.sha256(bitstream_bytes).hexdigest() if sha_override is None else sha_override
    payload = {
        "honest_verdict": "complete: gatemate_n16_bitstream_built",
        "gatemate_bitstream_built": built,
        "bitstream_path": str(bitstream),
        "bitstream_sha256": sha,
    }
    path = repo_root / "results" / EXP2956_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_exp2971_spec_entry_present() -> None:
    """REQ-HW-078: the FPGA spec anchors the detection preflight artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-078" in spec
    assert "SCENARIO-HW-078" in spec
    assert ARTIFACT_FILENAME in spec


def test_exp2971_ready_path_prepares_flash_command_without_flashing(tmp_path: Path) -> None:
    """SCENARIO-HW-078: stable detection and a matching hash prepare a command only."""
    _write_exp2956(tmp_path)
    device_node_root = tmp_path / "devbus"
    dirtyjtag_node = device_node_root / "003" / "014"
    dirtyjtag_node.parent.mkdir(parents=True)
    dirtyjtag_node.write_bytes(b"")
    dirtyjtag_node.chmod(0o660)
    loader = "/suite/bin/openFPGALoader"
    bitstream = json.loads((tmp_path / "results" / EXP2956_FILENAME).read_text(encoding="utf-8"))[
        "bitstream_path"
    ]
    lsusb = ("lsusb",)
    detect = (loader, "-c", "dirtyJtag", "--detect")
    runner, calls = _runner(
        {
            lsusb: CommandResult(0, "Bus 003 Device 014: ID 1209:c0ca DirtyJTAG\n", ""),
            detect: CommandResult(
                0,
                "idcode 0x20000001\nmanufacturer colognechip\nfamily GateMate Series\nmodel GM1Ax\n",
                "",
            ),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock(),
        device_node_root=device_node_root,
    )

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["honest_verdict"] == "complete: gatemate_flash_preconditions_ready"
    assert artifact["gatemate_board_detected"] is True
    assert artifact["bitstream_sha256_verified"] is True
    assert artifact["gatemate_flash_preconditions_ready"] is True
    assert artifact["flash_command"] == f"{loader} -c dirtyJtag -b olimex_gatemateevb {bitstream}"
    assert artifact["failure_command"] == ""
    assert artifact["failure_excerpt"] == ""
    assert artifact["detection_commands"] == [f"{loader} -c dirtyJtag --detect"] * 2
    assert len(artifact["detection_transcript_paths"]) == 2
    assert all(Path(path).exists() for path in artifact["detection_transcript_paths"])
    assert artifact["inference_substrate"] == "hardware_preflight"
    node_info = next(
        item
        for item in artifact["preconditions_checked"]
        if item["resource"] == "dirtyjtag_usb_device_node"
    )
    assert node_info["available"] is True
    assert node_info["path"] == str(dirtyjtag_node)
    assert node_info["mode_octal"] == "0o660"
    assert (
        "python/carnot/experiment_2971_gatemate_board_detection_flash_harness.py"
        in artifact["files_changed"]
    )
    assert calls == [lsusb, detect, detect]


def test_exp2971_missing_loader_blocks_before_usb_or_detection(tmp_path: Path) -> None:
    """REQ-HW-078: openFPGALoader discovery is the first executable gate."""
    _write_exp2956(tmp_path)
    runner, calls = _runner({})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({}),
        monotonic=_clock(),
    )

    assert artifact["honest_verdict"] == "blocked_openfpgaloader_missing"
    assert artifact["gatemate_board_detected"] is False
    assert artifact["gatemate_flash_preconditions_ready"] is False
    assert artifact["failure_command"] == "command -v openFPGALoader"
    assert artifact["detection_commands"] == []
    assert artifact["detection_transcript_paths"] == []
    assert calls == []


def test_exp2971_missing_exp2956_blocks_after_usb_check(tmp_path: Path) -> None:
    """REQ-HW-078: no Exp 2956 bitstream evidence means no flash command."""
    loader = "/suite/bin/openFPGALoader"
    runner, calls = _runner({("lsusb",): CommandResult(0, "ID 1209:c0ca DirtyJTAG\n", "")})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock(),
    )

    assert artifact["honest_verdict"] == "blocked_exp2956_bitstream_missing"
    assert "missing exp2956 artifact" in artifact["failure_excerpt"]
    assert artifact["failure_command"].endswith(EXP2956_FILENAME)
    assert artifact["bitstream_sha256_verified"] is False
    assert artifact["detection_commands"] == []
    assert calls == [("lsusb",)]


def test_exp2971_usb_missing_blocks_before_detection(tmp_path: Path) -> None:
    """REQ-HW-078: missing DirtyJTAG USB visibility is an actionable cable/power block."""
    _write_exp2956(tmp_path)
    loader = "/suite/bin/openFPGALoader"
    runner, calls = _runner(
        {("lsusb",): CommandResult(0, "Bus 003 Device 001: Linux root hub\n", "")}
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock(),
    )

    assert artifact["honest_verdict"] == "blocked_dirtyjtag_usb_missing"
    assert artifact["failure_command"] == "lsusb"
    assert "1209:c0ca" in artifact["failure_excerpt"]
    assert artifact["bitstream_sha256_verified"] is True
    assert artifact["detection_transcript_paths"] == []
    assert calls == [("lsusb",)]


def test_exp2971_sha_mismatch_blocks_before_detection(tmp_path: Path) -> None:
    """REQ-HW-078: the Exp 2956 SHA gate must match before a flash command is emitted."""
    _write_exp2956(tmp_path, sha_override="0" * 64)
    loader = "/suite/bin/openFPGALoader"
    runner, calls = _runner({("lsusb",): CommandResult(0, "ID 1209:c0ca DirtyJTAG\n", "")})

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock(),
    )

    assert artifact["honest_verdict"] == "blocked_bitstream_sha256_mismatch"
    assert artifact["bitstream_sha256_verified"] is False
    assert artifact["gatemate_board_detected"] is False
    assert artifact["flash_command"] == ""
    assert artifact["failure_command"].startswith("sha256sum ")
    assert calls == [("lsusb",)]


def test_exp2971_detection_failure_records_transcript_and_command(tmp_path: Path) -> None:
    """SCENARIO-HW-078: failed DirtyJTAG detection preserves raw transcript evidence."""
    _write_exp2956(tmp_path)
    loader = "/suite/bin/openFPGALoader"
    lsusb = ("lsusb",)
    detect = (loader, "-c", "dirtyJtag", "--detect")
    runner, calls = _runner(
        {
            lsusb: CommandResult(0, "ID 1209:c0ca DirtyJTAG\n", ""),
            detect: CommandResult(
                0, "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n", ""
            ),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({"openFPGALoader": loader}),
        monotonic=_clock(),
        detection_attempts=1,
    )

    transcript = Path(artifact["detection_transcript_paths"][0])
    assert artifact["honest_verdict"] == "blocked_board_not_detected"
    assert artifact["gatemate_board_detected"] is False
    assert artifact["gatemate_flash_preconditions_ready"] is False
    assert artifact["failure_command"] == f"{loader} -c dirtyJtag --detect"
    assert "Jtag frequency" in artifact["failure_excerpt"]
    assert "RETURNCODE: 0" in transcript.read_text(encoding="utf-8")
    assert calls == [lsusb, detect]


def test_exp2971_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """SCENARIO-HW-078: run_experiment writes the required v3 deliverable JSON."""
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=lambda args, timeout_s: CommandResult(99, "", "unexpected"),
        which_func=_which_from({}),
        monotonic=_clock(),
    )

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in loaded]
    assert missing == []
    assert loaded == artifact
    assert destination.name == ARTIFACT_FILENAME
