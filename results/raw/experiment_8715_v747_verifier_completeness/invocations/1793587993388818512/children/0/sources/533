"""GateMate n=16 flash/timing smoke for Exp 2957.

This experiment is deliberately narrower than a sampler benchmark. Exp 2956
proved that a real GateMate `.bit` file exists; this module verifies that exact
file, contacts the physical DirtyJTAG/GateMate chain, flashes the bitstream, and
records raw transcripts. The current GateMate constraints are build-only and do
not expose a host readback path, so a successful flash remains a contact smoke,
not evidence for speedup, Boltzmann sampling, or thermodynamic behavior.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ARTIFACT_FILENAME = "experiment_2957_gatemate_flash_timing_smoke_v2.json"
EXP2956_FILENAME = "experiment_2956_gatemate_n16_bitstream_build_v4.json"
RUN_DATE = "20260524"
INFERENCE_SUBSTRATE = "hardware_smoke"
FLASH_SNIPPET = "openFPGALoader -c dirtyJtag -b olimex_gatemateevb"
DETECT_ARGS = ("-c", "dirtyJtag", "--detect")
FLASH_ARGS = ("-c", "dirtyJtag", "-b", "olimex_gatemateevb")


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result so tests can inject exact hardware transcripts."""

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


def _duration(start: float, monotonic: ClockFunc) -> float:
    return round(monotonic() - start, 6)


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
    missing_bitstream = not bitstream.exists() or not expected_sha
    reason = ""
    if not built:
        reason = "exp2956 gatemate_bitstream_built is false"
    elif missing_bitstream:  # pragma: no cover - defensive gate; covered by earlier 2956 tests.
        reason = f"exp2956 bitstream path or sha missing: {bitstream}"

    return (
        {
            "resource": "exp2956_gatemate_bitstream",
            "available": not reason,
            "ready": built and not missing_bitstream,
            "path": str(path),
            "bitstream_path": str(bitstream),
            "expected_sha256": expected_sha,
            "reason": reason,
        },
        payload,
        bitstream,
    )


def _loader_path_from_exp2956(exp2956: dict) -> str:
    for item in exp2956.get("preconditions_checked", []):
        if item.get("resource") == "openFPGALoader" and item.get("path"):
            return str(item["path"])
    return ""


def _locate_openfpgaloader(exp2956: dict, which_func: WhichFunc) -> dict:
    path = _loader_path_from_exp2956(exp2956) or which_func("openFPGALoader") or ""
    return {
        "resource": "openFPGALoader",
        "available": bool(path),
        "path": path,
        "reason": "" if path else "openFPGALoader not found in Exp 2956 or PATH",
    }


def _documented_flash_path(repo_root: Path) -> bool:
    wishlist = repo_root / "research-hardware-wishlist.md"
    return wishlist.exists() and FLASH_SNIPPET in wishlist.read_text(encoding="utf-8")


def _looks_like_gatemate(result: CommandResult) -> bool:
    text = _command_text(result)
    tokens = ("GateMate", "GM1A", "colognechip", "idcode 0x20000001", "IDCODE")
    return result.returncode == 0 and any(token in text for token in tokens)


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


def _readback_reason(exp2956: dict, repo_root: Path) -> str:
    constraints = exp2956.get("constraints_path", "")
    if constraints:
        path = _existing_path(str(constraints), repo_root)
        if path.exists():
            text = path.read_text(encoding="utf-8").lower()
            if "no physical" in text or "allow-unconstrained" in text:
                return "Exp 2956 uses build-only constraints with no physical board output/readback path."
    return "No GateMate board output/readback capture command is defined for this bitstream."


def _transcript_hashes(paths: list[str]) -> dict[str, str]:
    return {path: _sha256_file(Path(path)) for path in paths}


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
        "no_speedup_claim": True,
        "no_boltzmann_claim": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "run_date": RUN_DATE,
        "failure_command": failure_command,
        "failure_excerpt": failure_excerpt,
        "bitstream_path": bitstream_path,
        "bitstream_sha256": bitstream_sha256,
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
    )


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
    exp2956_path: Path | None = None,
    transcript_dir: Path | None = None,
) -> dict:
    start = monotonic()
    transcripts = transcript_dir or repo_root / "logs" / "experiment_2957_gatemate_flash_timing_smoke_v2"
    transcript_paths: list[str] = []
    command_durations: dict[str, float] = {}
    timing_observation = {
        "timing_source": "host_wall_clock_command_duration",
        "command_durations_s": command_durations,
        "readback_supported": False,
        "readback_reason": "",
    }

    exp_precondition, exp2956, bitstream = _load_exp2956(repo_root, exp2956_path)
    preconditions = [exp_precondition]
    if not exp_precondition["available"]:
        return _blocked(
            verdict="blocked_exp2956_bitstream_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            timing_observation=timing_observation,
            failure_command="",
            failure_excerpt=str(exp_precondition["reason"]),
        )

    expected_sha = str(exp2956.get("bitstream_sha256", ""))
    actual_sha = _sha256_file(bitstream)
    sha_precondition = {
        "resource": "exp2956_bitstream_sha256",
        "available": True,
        "path": str(bitstream),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "verified": actual_sha == expected_sha,
    }
    preconditions.append(sha_precondition)
    timing_observation["readback_reason"] = _readback_reason(exp2956, repo_root)
    if actual_sha != expected_sha:
        return _blocked(
            verdict="blocked_bitstream_sha256_mismatch",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            timing_observation=timing_observation,
            failure_command="sha256sum",
            failure_excerpt=f"bitstream sha256 mismatch: expected {expected_sha}, got {actual_sha}",
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
        )

    loader_precondition = _locate_openfpgaloader(exp2956, which_func)
    preconditions.append(loader_precondition)
    if not loader_precondition["available"]:
        return _blocked(
            verdict="blocked_openfpgaloader_missing",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            bitstream_sha256_verified=True,
            timing_observation=timing_observation,
            failure_command="command -v openFPGALoader",
            failure_excerpt=str(loader_precondition["reason"]),
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
        )

    loader = str(loader_precondition["path"])
    detect_command = [loader, *DETECT_ARGS]
    detect_result = _run_recorded(
        label="detect",
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
            "resource": "gatemate_board_detection",
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
        )

    documented = _documented_flash_path(repo_root)
    preconditions.append(
        {
            "resource": "documented_gatemate_flash_path",
            "available": documented,
            "command_shape": f"{FLASH_SNIPPET} <bitstream>",
        }
    )
    if not documented:
        return _blocked(
            verdict="blocked_flash_path_undocumented",
            start=start,
            monotonic=monotonic,
            preconditions_checked=preconditions,
            board_detected=True,
            bitstream_sha256_verified=True,
            timing_observation=timing_observation,
            transcript_paths=transcript_paths,
            failure_command=f"{FLASH_SNIPPET} <bitstream>",
            failure_excerpt="GateMate flash command shape is not documented in research-hardware-wishlist.md",
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
        )

    flash_command = [loader, *FLASH_ARGS, str(bitstream)]
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
    post_flash_detected = _looks_like_gatemate(post_detect_result)
    post_flash_contact = post_flash_detected or (
        post_detect_result.returncode == 0 and "Jtag frequency" in _command_text(post_detect_result)
    )
    timing_observation["post_flash_idcode_detected"] = post_flash_detected
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
