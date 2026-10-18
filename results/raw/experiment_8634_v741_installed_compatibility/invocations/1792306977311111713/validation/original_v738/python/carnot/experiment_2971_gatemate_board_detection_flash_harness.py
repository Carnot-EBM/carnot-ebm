"""GateMate board-detection preflight and flash-command harness for Exp 2971.

Spec: REQ-HW-078, SCENARIO-HW-078.

This module deliberately stops one step short of programming the board. Exp 2956
already produced the n=16 GateMate bitstream, and Exp 2957 showed why contact
evidence has to be captured honestly: a DirtyJTAG command can return success
while only reporting JTAG clock negotiation. Here we verify the bitstream hash,
check USB visibility, run one or more read-only `--detect` probes, and emit the
exact flash command only when those preconditions are ready.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ARTIFACT_FILENAME = "experiment_2971_gatemate_board_detection_flash_harness_v3.json"
EXP2956_FILENAME = "experiment_2956_gatemate_n16_bitstream_build_v4.json"
RUN_DATE = "20260524"
INFERENCE_SUBSTRATE = "hardware_preflight"
DETECT_ARGS = ("-c", "dirtyJtag", "--detect")
FLASH_ARGS = ("-c", "dirtyJtag", "-b", "olimex_gatemateevb")
FILES_CHANGED = [
    "openspec/capabilities/fpga/spec.md",
    "python/carnot/experiment_2971_gatemate_board_detection_flash_harness.py",
    "tests/python/test_experiment_2971_gatemate_board_detection_flash_harness.py",
    f"results/{ARTIFACT_FILENAME}",
]


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result so tests can inject hardware transcripts."""

    returncode: int
    stdout: str
    stderr: str


RunCommand = Callable[[list[str], float], CommandResult]
WhichFunc = Callable[[str], str | None]
ClockFunc = Callable[[], float]


def _default_run_command(args: list[str], timeout_s: float) -> CommandResult:  # pragma: no cover
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _quote(args: list[str]) -> str:
    return shlex.join(args)


def _command_text(result: CommandResult) -> str:
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else repo_root / path


def _load_exp2956(repo_root: Path, exp2956_path: Path | None) -> tuple[dict, dict, Path]:
    path = exp2956_path or repo_root / "results" / EXP2956_FILENAME
    if not path.exists():
        return (
            {
                "resource": "exp2956_gatemate_bitstream",
                "available": False,
                "ready": False,
                "path": str(path),
                "reason": f"missing exp2956 artifact: {path}",
            },
            {},
            Path(),
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    built = bool(payload.get("gatemate_bitstream_built", False))
    bitstream = _existing_path(str(payload.get("bitstream_path", "")), repo_root)
    expected_sha = str(payload.get("bitstream_sha256", ""))
    ready = built and bool(expected_sha) and bitstream.exists()
    reason = "" if ready else "exp2956 bitstream path, SHA256, or ready flag is missing"
    return (
        {
            "resource": "exp2956_gatemate_bitstream",
            "available": ready,
            "ready": ready,
            "path": str(path),
            "bitstream_path": str(bitstream),
            "expected_sha256": expected_sha,
            "reason": reason,
        },
        payload,
        bitstream,
    )


def _looks_like_dirtyjtag(result: CommandResult) -> bool:
    text = _command_text(result).lower()
    return result.returncode == 0 and (
        "1209:c0ca" in text or "dirtyjtag" in text or "gatemate" in text
    )


def _looks_like_gatemate(result: CommandResult) -> bool:
    text = _command_text(result).lower()
    has_id = "idcode 0x20000001" in text or "gm1a" in text
    has_family = "gatemate" in text or "colognechip" in text
    return result.returncode == 0 and has_id and has_family


def _parse_dirtyjtag_bus_device(text: str) -> tuple[int, int] | None:
    for line in text.splitlines():
        lowered = line.lower()
        if "1209:c0ca" in lowered or "dirtyjtag" in lowered or "gatemate" in lowered:
            match = re.search(r"bus\s+0*(\d+).*device\s+0*(\d+)", line, flags=re.IGNORECASE)
            if match:
                return int(match.group(1)), int(match.group(2))
    return None


def _usb_device_node_precondition(result: CommandResult, device_node_root: Path) -> dict:
    parsed = _parse_dirtyjtag_bus_device(_command_text(result))
    if parsed is None:
        return {
            "resource": "dirtyjtag_usb_device_node",
            "available": False,
            "path": "",
            "reason": "DirtyJTAG bus/device not parseable from lsusb output",
        }

    bus, device = parsed
    path = device_node_root / f"{bus:03d}" / f"{device:03d}"
    exists = path.exists()
    info = {
        "resource": "dirtyjtag_usb_device_node",
        "available": exists,
        "path": str(path),
        "reason": "" if exists else f"USB device node is absent: {path}",
    }
    if exists:
        stat_result = path.stat()
        info.update(
            {
                "mode_octal": oct(stat_result.st_mode & 0o777),
                "uid": stat_result.st_uid,
                "gid": stat_result.st_gid,
                "current_user_rw": os.access(path, os.R_OK | os.W_OK),
            }
        )
    return info


def _write_transcript(path: Path, command: list[str], result: CommandResult) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"COMMAND: {_quote(command)}",
                f"RETURNCODE: {result.returncode}",
                "STDOUT:",
                result.stdout,
                "STDERR:",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


def _base_artifact(
    *,
    honest_verdict: str,
    start: float,
    monotonic: ClockFunc,
    preconditions_checked: list[dict],
    gatemate_board_detected: bool = False,
    bitstream_sha256_verified: bool = False,
    gatemate_flash_preconditions_ready: bool = False,
    detection_commands: list[str] | None = None,
    detection_transcript_paths: list[str] | None = None,
    bitstream_path: str = "",
    bitstream_sha256: str = "",
    flash_command: str = "",
    failure_command: str = "",
    failure_excerpt: str = "",
) -> dict:
    return {
        "honest_verdict": honest_verdict,
        "preconditions_checked": preconditions_checked,
        "gatemate_board_detected": gatemate_board_detected,
        "bitstream_sha256_verified": bitstream_sha256_verified,
        "gatemate_flash_preconditions_ready": gatemate_flash_preconditions_ready,
        "detection_commands": detection_commands or [],
        "detection_transcript_paths": detection_transcript_paths or [],
        "bitstream_path": bitstream_path,
        "bitstream_sha256": bitstream_sha256,
        "flash_command": flash_command,
        "failure_command": failure_command,
        "failure_excerpt": failure_excerpt,
        "files_changed": FILES_CHANGED,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(monotonic() - start, 6),
        "run_date": RUN_DATE,
    }


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
    exp2956_path: Path | None = None,
    transcript_dir: Path | None = None,
    device_node_root: Path = Path("/dev/bus/usb"),
    detection_attempts: int = 2,
) -> dict:
    start = monotonic()
    transcripts = transcript_dir or (
        repo_root / "logs" / "experiment_2971_gatemate_board_detection_flash_harness_v3"
    )
    preconditions: list[dict] = []
    detection_commands: list[str] = []
    detection_transcript_paths: list[str] = []

    loader = which_func("openFPGALoader") or ""
    preconditions.append(
        {
            "resource": "openFPGALoader",
            "available": bool(loader),
            "command": "command -v openFPGALoader",
            "path": loader,
            "reason": "" if loader else "openFPGALoader not found in PATH",
        }
    )
    if not loader:
        return _base_artifact(
            honest_verdict="blocked_openfpgaloader_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            failure_command="command -v openFPGALoader",
            failure_excerpt="openFPGALoader not found in PATH",
        )

    lsusb_command = ["lsusb"]
    lsusb_result = run_command(lsusb_command, 10.0)
    usb_visible = _looks_like_dirtyjtag(lsusb_result)
    preconditions.append(
        {
            "resource": "dirtyjtag_usb_visibility",
            "available": usb_visible,
            "command": _quote(lsusb_command),
            "returncode": lsusb_result.returncode,
            "output": _command_text(lsusb_result),
            "reason": "" if usb_visible else "DirtyJTAG/GateMate USB ID 1209:c0ca not visible",
        }
    )
    preconditions.append(_usb_device_node_precondition(lsusb_result, device_node_root))

    exp_precondition, exp2956, bitstream = _load_exp2956(repo_root, exp2956_path)
    preconditions.append(exp_precondition)
    if not exp_precondition["available"]:
        return _base_artifact(
            honest_verdict="blocked_exp2956_bitstream_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            failure_command=str(exp_precondition["path"]),
            failure_excerpt=str(exp_precondition["reason"]),
        )

    expected_sha = str(exp2956.get("bitstream_sha256", ""))
    actual_sha = _sha256_file(bitstream)
    sha_verified = actual_sha == expected_sha
    preconditions.append(
        {
            "resource": "exp2956_bitstream_sha256",
            "available": True,
            "path": str(bitstream),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "verified": sha_verified,
        }
    )

    if not usb_visible:
        excerpt = "DirtyJTAG/GateMate USB ID 1209:c0ca not visible in lsusb output"
        return _base_artifact(
            honest_verdict="blocked_dirtyjtag_usb_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            bitstream_sha256_verified=sha_verified,
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            failure_command="lsusb",
            failure_excerpt=f"{excerpt}: {_command_text(lsusb_result)}",
        )

    if not sha_verified:
        return _base_artifact(
            honest_verdict="blocked_bitstream_sha256_mismatch",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            failure_command=f"sha256sum {bitstream}",
            failure_excerpt=f"bitstream sha256 mismatch: expected {expected_sha}, got {actual_sha}",
        )

    detect_command = [loader, *DETECT_ARGS]
    detect_results: list[CommandResult] = []
    for attempt in range(max(1, detection_attempts)):
        result = run_command(detect_command, 20.0)
        detect_results.append(result)
        detection_commands.append(_quote(detect_command))
        detection_transcript_paths.append(
            _write_transcript(transcripts / f"detect_{attempt + 1}.txt", detect_command, result)
        )

    detection_ready = all(_looks_like_gatemate(result) for result in detect_results)
    preconditions.append(
        {
            "resource": "gatemate_board_detection",
            "available": detection_ready,
            "command": _quote(detect_command),
            "attempts": len(detect_results),
            "transcript_paths": detection_transcript_paths,
        }
    )
    if not detection_ready:
        failing = next(
            (result for result in detect_results if not _looks_like_gatemate(result)),
            detect_results[0],
        )
        return _base_artifact(
            honest_verdict="blocked_board_not_detected",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            bitstream_sha256_verified=True,
            detection_commands=detection_commands,
            detection_transcript_paths=detection_transcript_paths,
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            failure_command=_quote(detect_command),
            failure_excerpt=_command_text(failing),
        )

    flash_command = [loader, *FLASH_ARGS, str(bitstream)]
    return _base_artifact(
        honest_verdict="complete: gatemate_flash_preconditions_ready",
        start=start,
        monotonic=monotonic,
        preconditions_checked=preconditions,
        gatemate_board_detected=True,
        bitstream_sha256_verified=True,
        gatemate_flash_preconditions_ready=True,
        detection_commands=detection_commands,
        detection_transcript_paths=detection_transcript_paths,
        bitstream_path=str(bitstream),
        bitstream_sha256=actual_sha,
        flash_command=_quote(flash_command),
    )


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
    exp2956_path: Path | None = None,
) -> dict:
    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(
        repo_root=root,
        run_command=run_command,
        which_func=which_func,
        monotonic=monotonic,
        exp2956_path=exp2956_path,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
