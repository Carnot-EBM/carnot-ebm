"""GateMate post-flash output-hash smoke for Exp 2972.

Spec: REQ-HW-079, SCENARIO-HW-079.

Exp 2971 proved that the GateMate board could be detected and emitted the exact
flash command for the Exp 2956 n=16 bitstream. This module consumes that
preflight artifact, repeats the detection and SHA256 checks immediately before
flash, executes the Exp 2971 flash command, and records the smallest available
post-flash smoke evidence. The current GateMate bitstream has no host-visible
sampler output or readback protocol, so the observed output hash is the
post-flash contact transcript hash, not a sampler-quality or timing-speedup
claim.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ARTIFACT_FILENAME = "experiment_2972_gatemate_post_flash_output_hash_v3.json"
EXP2971_FILENAME = "experiment_2971_gatemate_board_detection_flash_harness_v3.json"
RUN_DATE = "20260524"
INFERENCE_SUBSTRATE = "hardware_smoke"


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result so tests can inject physical-board transcripts."""

    returncode: int
    stdout: str
    stderr: str


RunCommand = Callable[[list[str], float], CommandResult]
ClockFunc = Callable[[], float]


def _default_run_command(args: list[str], timeout_s: float) -> CommandResult:  # pragma: no cover
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
    except FileNotFoundError as exc:
        return CommandResult(127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else str(exc)
        return CommandResult(124, stdout, stderr)


def _duration(start: float, monotonic: ClockFunc) -> float:
    return round(monotonic() - start, 6)


def _quote(args: list[str]) -> str:
    return shlex.join(args)


def _command_text(result: CommandResult) -> str:
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def _existing_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else repo_root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _transcript_hashes(paths: list[str]) -> dict[str, str]:
    return {path: _sha256_file(Path(path)) for path in paths}


def _looks_like_gatemate(result: CommandResult) -> bool:
    text = _command_text(result).lower()
    has_id = "idcode 0x20000001" in text or "gm1a" in text
    has_family = "gatemate" in text or "colognechip" in text
    return result.returncode == 0 and has_id and has_family


def _looks_like_post_flash_contact(result: CommandResult) -> bool:
    return _looks_like_gatemate(result) or (
        result.returncode == 0 and "jtag frequency" in _command_text(result).lower()
    )


def _load_exp2971(repo_root: Path, exp2971_path: Path | None) -> tuple[dict, dict]:
    path = exp2971_path or repo_root / "results" / EXP2971_FILENAME
    if not path.exists():
        return (
            {
                "resource": "exp2971_preconditions",
                "available": False,
                "path": str(path),
                "reason": f"missing exp2971 artifact: {path}",
            },
            {},
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    reasons: list[str] = []
    if not payload.get("gatemate_board_detected", False):
        reasons.append("gatemate_board_detected=false")
    if not payload.get("bitstream_sha256_verified", False):
        reasons.append("bitstream_sha256_verified=false")
    if not str(payload.get("flash_command", "")):
        reasons.append("flash_command missing")
    ready = not reasons
    return (
        {
            "resource": "exp2971_preconditions",
            "available": ready,
            "path": str(path),
            "gatemate_board_detected": bool(payload.get("gatemate_board_detected", False)),
            "bitstream_sha256_verified": bool(payload.get("bitstream_sha256_verified", False)),
            "reason": "; ".join(reasons),
        },
        payload,
    )


def _parse_exp2971_commands(payload: dict, repo_root: Path) -> tuple[dict, list[str], list[str], Path]:
    flash_text = str(payload.get("flash_command", ""))
    flash_command = shlex.split(flash_text)
    if len(flash_command) < 5:
        return (
            {
                "resource": "exp2971_flash_command",
                "available": False,
                "command": flash_text,
                "reason": "Exp 2971 flash command is missing required openFPGALoader arguments",
            },
            [],
            [],
            Path(),
        )

    detection_commands = payload.get("detection_commands", [])
    if detection_commands:
        detect_command = shlex.split(str(detection_commands[0]))
    else:  # pragma: no cover - Exp 2971 artifacts always include detection commands.
        detect_command = [flash_command[0], "-c", "dirtyJtag", "--detect"]

    bitstream = _existing_path(flash_command[-1], repo_root)
    return (
        {
            "resource": "exp2971_flash_command",
            "available": True,
            "command": flash_text,
            "bitstream_path": str(bitstream),
            "reason": "",
        },
        flash_command,
        detect_command,
        bitstream,
    )


def _base_artifact(
    *,
    honest_verdict: str,
    duration_s: float,
    preconditions_checked: list[dict],
    timing_observation: dict,
    transcript_paths: list[str] | None = None,
    board_detected: bool = False,
    bitstream_sha256_verified: bool = False,
    flash_attempted: bool = False,
    flash_succeeded: bool = False,
    observed_output_sha256: str = "",
    failure_command: str = "",
    failure_excerpt: str = "",
    bitstream_path: str = "",
    bitstream_sha256: str = "",
    flash_command: str = "",
) -> dict:
    paths = transcript_paths or []
    return {
        "honest_verdict": honest_verdict,
        "preconditions_checked": preconditions_checked,
        "board_detected": board_detected,
        "bitstream_sha256_verified": bitstream_sha256_verified,
        "flash_attempted": flash_attempted,
        "flash_succeeded": flash_succeeded,
        "smoke_vector_passed": False,
        "observed_output_sha256": observed_output_sha256,
        "timing_observation": timing_observation,
        "transcript_paths": paths,
        "transcript_sha256": _transcript_hashes(paths),
        "failure_command": failure_command,
        "failure_excerpt": failure_excerpt,
        "no_speedup_claim": True,
        "no_boltzmann_claim": True,
        "no_thermalization_claim": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "run_date": RUN_DATE,
        "bitstream_path": bitstream_path,
        "bitstream_sha256": bitstream_sha256,
        "flash_command": flash_command,
    }


def _blocked(
    *,
    verdict: str,
    start: float,
    monotonic: ClockFunc,
    preconditions_checked: list[dict],
    timing_observation: dict,
    transcript_paths: list[str] | None = None,
    board_detected: bool = False,
    bitstream_sha256_verified: bool = False,
    flash_attempted: bool = False,
    flash_succeeded: bool = False,
    observed_output_sha256: str = "",
    failure_command: str,
    failure_excerpt: str,
    bitstream_path: str = "",
    bitstream_sha256: str = "",
    flash_command: str = "",
) -> dict:
    return _base_artifact(
        honest_verdict=verdict,
        duration_s=_duration(start, monotonic),
        preconditions_checked=preconditions_checked,
        board_detected=board_detected,
        bitstream_sha256_verified=bitstream_sha256_verified,
        flash_attempted=flash_attempted,
        flash_succeeded=flash_succeeded,
        observed_output_sha256=observed_output_sha256,
        timing_observation=timing_observation,
        transcript_paths=transcript_paths,
        failure_command=failure_command,
        failure_excerpt=failure_excerpt,
        bitstream_path=bitstream_path,
        bitstream_sha256=bitstream_sha256,
        flash_command=flash_command,
    )


def _run_recorded(
    *,
    label: str,
    command: list[str],
    timeout_s: float,
    run_command: RunCommand,
    monotonic: ClockFunc,
    transcript_dir: Path,
    transcript_paths: list[str],
    command_durations: dict[str, float],
) -> CommandResult:
    command_start = monotonic()
    result = run_command(command, timeout_s)
    command_durations[label] = round(monotonic() - command_start, 6)
    transcript_paths.append(_write_transcript(transcript_dir / f"{label}.txt", command, result))
    return result


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    monotonic: ClockFunc = time.monotonic,
    exp2971_path: Path | None = None,
    transcript_dir: Path | None = None,
) -> dict:
    start = monotonic()
    transcripts = transcript_dir or repo_root / "logs" / "experiment_2972_gatemate_post_flash_output_hash_v3"
    transcript_paths: list[str] = []
    command_durations: dict[str, float] = {}
    timing_observation = {
        "timing_source": "host_wall_clock_command_duration",
        "command_durations_s": command_durations,
        "readback_supported": False,
        "readback_reason": (
            "No GateMate n=16 host-visible sampler output/readback path is defined; "
            "post-flash detect is contact smoke only."
        ),
        "smoke_path": "post_flash_detect",
    }

    exp2971_precondition, exp2971 = _load_exp2971(repo_root, exp2971_path)
    preconditions = [exp2971_precondition]
    if not exp2971_precondition["available"]:
        return _blocked(
            verdict="blocked_exp2971_preconditions_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            timing_observation=timing_observation,
            failure_command=str(exp2971_precondition["path"]),
            failure_excerpt=str(exp2971_precondition["reason"]),
        )

    command_precondition, flash_command, detect_command, bitstream = _parse_exp2971_commands(
        exp2971, repo_root
    )
    preconditions.append(command_precondition)
    if not command_precondition["available"]:
        return _blocked(
            verdict="blocked_exp2971_flash_command_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            timing_observation=timing_observation,
            failure_command=str(command_precondition["command"]),
            failure_excerpt=str(command_precondition["reason"]),
        )

    if not bitstream.exists():
        return _blocked(
            verdict="blocked_exp2971_bitstream_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            timing_observation=timing_observation,
            failure_command=str(bitstream),
            failure_excerpt=f"bitstream path from Exp 2971 does not exist: {bitstream}",
            bitstream_path=str(bitstream),
            flash_command=_quote(flash_command),
        )

    expected_sha = str(exp2971.get("bitstream_sha256", ""))
    actual_sha = _sha256_file(bitstream)
    sha_verified = actual_sha == expected_sha
    preconditions.append(
        {
            "resource": "exp2971_bitstream_sha256_recheck",
            "available": True,
            "path": str(bitstream),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "verified": sha_verified,
        }
    )
    if not sha_verified:
        return _blocked(
            verdict="blocked_bitstream_sha256_mismatch",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            timing_observation=timing_observation,
            failure_command=f"sha256sum {bitstream}",
            failure_excerpt=f"bitstream sha256 mismatch: expected {expected_sha}, got {actual_sha}",
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            flash_command=_quote(flash_command),
        )

    detect_result = _run_recorded(
        label="pre_flash_detect",
        command=detect_command,
        timeout_s=20.0,
        run_command=run_command,
        monotonic=monotonic,
        transcript_dir=transcripts,
        transcript_paths=transcript_paths,
        command_durations=command_durations,
    )
    board_detected = _looks_like_gatemate(detect_result)
    preconditions.append(
        {
            "resource": "pre_flash_gatemate_board_detection",
            "available": board_detected,
            "command": _quote(detect_command),
            "returncode": detect_result.returncode,
            "transcript_path": transcript_paths[-1],
        }
    )
    if not board_detected:
        return _blocked(
            verdict="blocked_board_not_detected",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            bitstream_sha256_verified=True,
            timing_observation=timing_observation,
            transcript_paths=transcript_paths,
            failure_command=_quote(detect_command),
            failure_excerpt=_command_text(detect_result),
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            flash_command=_quote(flash_command),
        )

    flash_result = _run_recorded(
        label="flash",
        command=flash_command,
        timeout_s=120.0,
        run_command=run_command,
        monotonic=monotonic,
        transcript_dir=transcripts,
        transcript_paths=transcript_paths,
        command_durations=command_durations,
    )
    if flash_result.returncode != 0:
        return _blocked(
            verdict="blocked_flash_failed",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            board_detected=True,
            bitstream_sha256_verified=True,
            flash_attempted=True,
            flash_succeeded=False,
            timing_observation=timing_observation,
            transcript_paths=transcript_paths,
            failure_command=_quote(flash_command),
            failure_excerpt=_command_text(flash_result),
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            flash_command=_quote(flash_command),
        )

    post_detect_result = _run_recorded(
        label="post_flash_detect",
        command=detect_command,
        timeout_s=20.0,
        run_command=run_command,
        monotonic=monotonic,
        transcript_dir=transcripts,
        transcript_paths=transcript_paths,
        command_durations=command_durations,
    )
    observed_output_sha256 = _sha256_file(Path(transcript_paths[-1]))
    post_flash_idcode = _looks_like_gatemate(post_detect_result)
    post_flash_contact = _looks_like_post_flash_contact(post_detect_result)
    timing_observation["post_flash_idcode_detected"] = post_flash_idcode
    timing_observation["post_flash_contact_detected"] = post_flash_contact
    if not post_flash_contact:
        return _blocked(
            verdict="blocked_post_flash_board_contact_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            board_detected=True,
            bitstream_sha256_verified=True,
            flash_attempted=True,
            flash_succeeded=True,
            observed_output_sha256=observed_output_sha256,
            timing_observation=timing_observation,
            transcript_paths=transcript_paths,
            failure_command=_quote(detect_command),
            failure_excerpt=_command_text(post_detect_result),
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            flash_command=_quote(flash_command),
        )

    return _base_artifact(
        honest_verdict="complete: gatemate_flash_contact_smoke_no_readback",
        duration_s=_duration(start, monotonic),
        preconditions_checked=preconditions,
        board_detected=True,
        bitstream_sha256_verified=True,
        flash_attempted=True,
        flash_succeeded=True,
        observed_output_sha256=observed_output_sha256,
        timing_observation=timing_observation,
        transcript_paths=transcript_paths,
        bitstream_path=str(bitstream),
        bitstream_sha256=actual_sha,
        flash_command=_quote(flash_command),
    )


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand = _default_run_command,
    monotonic: ClockFunc = time.monotonic,
    exp2971_path: Path | None = None,
) -> dict:
    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(
        repo_root=root,
        run_command=run_command,
        monotonic=monotonic,
        exp2971_path=exp2971_path,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
