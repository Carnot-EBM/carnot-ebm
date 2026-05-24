"""GateMate readback and smoke-vector evidence collector for Exp 2984.

Spec: REQ-HW-080, SCENARIO-HW-080.

The prior GateMate artifacts prove USB/JTAG board contact and a successful SRAM
load, but they do not prove sampler output. This module keeps those boundaries
explicit: it rechecks the bitstream hash, records tool versions and live board
detection, attempts readback only when the installed tool advertises a compatible
readback command, and refuses to mark the n=16 smoke vector as passed unless a
host-visible IO path exists and returns the expected value.
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


ARTIFACT_FILENAME = "experiment_2984_gatemate_readback_smoke_vector_v4.json"
EXP2971_FILENAME = "experiment_2971_gatemate_board_detection_flash_harness_v3.json"
EXP2972_FILENAME = "experiment_2972_gatemate_post_flash_output_hash_v3.json"
RUN_DATE = "20260524"
INFERENCE_SUBSTRATE = "physical_gatemate_board"


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result so tests can inject board/tool transcripts."""

    returncode: int
    stdout: str
    stderr: str


RunCommand = Callable[[list[str], float], CommandResult]
WhichFunc = Callable[[str], str | None]
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def _first_version_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return next((line for line in lines if "version" in line.lower()), lines[0] if lines else "")


def _loader_from_flash_command(payload: dict) -> str:
    command = str(payload.get("flash_command", ""))
    parts = shlex.split(command) if command else []
    return parts[0] if parts else ""


def _tool_path(name: str, which_func: WhichFunc, fallback: str = "") -> str:
    candidate = which_func(name) or fallback
    return candidate


def _collect_tool_versions(
    *,
    run_command: RunCommand,
    which_func: WhichFunc,
    loader_hint: str,
) -> dict[str, dict]:
    tools = {
        "openFPGALoader": (_tool_path("openFPGALoader", which_func, loader_hint), ["-V"]),
        "yosys": (_tool_path("yosys", which_func), ["-V"]),
        "nextpnr-himbaechel": (_tool_path("nextpnr-himbaechel", which_func), ["--version"]),
        "gmpack": (_tool_path("gmpack", which_func), ["--help"]),
    }
    versions: dict[str, dict] = {}
    for name, (path, args) in tools.items():
        if not path:
            versions[name] = {"available": False, "path": "", "version": "", "returncode": None}
            continue
        command = [path, *args]
        result = run_command(command, 10.0)
        output = _command_text(result)
        versions[name] = {
            "available": True,
            "path": path,
            "command": _quote(command),
            "returncode": result.returncode,
            "version": _first_version_line(output),
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        }
    return versions


def _extract_board_id(result: CommandResult) -> str:
    text = _command_text(result)
    lowered = text.lower()
    if result.returncode != 0 or ("gatemate" not in lowered and "colognechip" not in lowered):
        return ""
    parts: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("idcode"):
            parts.append(stripped)
        elif "manufacturer" in low and "colognechip" in low:
            parts.append("colognechip")
        elif "family" in low and "gatemate" in low:
            parts.append("GateMate Series")
        elif low.startswith("model"):
            model = stripped.split()[-1]
            parts.append(model)
    return "; ".join(parts)


def _readback_decision(help_result: CommandResult) -> tuple[bool, str]:
    text = _command_text(help_result)
    lowered = text.lower()
    if "--readback" in lowered:
        return True, "openFPGALoader help exposes --readback for this installed tool."
    if "spi flash only" in text or "--dump-flash" in lowered or "--verify" in lowered:
        return (
            False,
            "openFPGALoader exposes flash dump/verify options, but verify is SPI Flash only; "
            "no GateMate SRAM or sampler-output readback command is advertised.",
        )
    return False, "openFPGALoader help does not advertise a GateMate-compatible readback command."


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


def _smoke_vector_blocker(repo_root: Path) -> tuple[str, str]:
    ccf = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.ccf"
    rtl = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.v"
    test_vector = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate_test_vector.json"
    ccf_text = ccf.read_text(encoding="utf-8") if ccf.exists() else ""
    reason = (
        "GateMate n=16 CCF assigns no physical Pin_in/Pin_out locations and requires "
        "allow-unconstrained; RTL/test-vector ports have no JTAG, UART, GPIO, or host "
        "register protocol for observing spin_out."
        if "no physical" in ccf_text.lower() or "allow-unconstrained" in ccf_text.lower()
        else "No host-visible GateMate IO protocol is documented for the n=16 smoke vector."
    )
    evidence = {
        "constraints_path": str(ccf),
        "rtl_path": str(rtl),
        "test_vector_path": str(test_vector),
        "constraints_sha256": _sha256_file(ccf) if ccf.exists() else "",
        "rtl_sha256": _sha256_file(rtl) if rtl.exists() else "",
        "test_vector_sha256": _sha256_file(test_vector) if test_vector.exists() else "",
    }
    return reason, json.dumps(evidence, sort_keys=True)


def _base_artifact(
    *,
    honest_verdict: str,
    start: float,
    monotonic: ClockFunc,
    board_detected: bool,
    board_id: str,
    tool_versions: dict,
    bitstream_path: str,
    bitstream_sha256: str,
    flash_succeeded: bool,
    readback_supported: bool,
    readback_attempted: bool,
    readback_hash: str,
    smoke_vector_attempted: bool,
    smoke_vector_passed: bool,
    observed_smoke_output: str,
    expected_smoke_output: str,
    timing_observation: dict,
) -> dict:
    return {
        "honest_verdict": honest_verdict,
        "board_detected": board_detected,
        "board_id": board_id,
        "tool_versions": tool_versions,
        "bitstream_path": bitstream_path,
        "bitstream_sha256": bitstream_sha256,
        "flash_succeeded": flash_succeeded,
        "readback_supported": readback_supported,
        "readback_attempted": readback_attempted,
        "readback_hash": readback_hash,
        "smoke_vector_attempted": smoke_vector_attempted,
        "smoke_vector_passed": smoke_vector_passed,
        "observed_smoke_output": observed_smoke_output,
        "expected_smoke_output": expected_smoke_output,
        "timing_observation": timing_observation,
        "sampler_claim_allowed": False,
        "speedup_claim_allowed": False,
        "thermodynamic_claim_allowed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _duration(start, monotonic),
        "run_date": RUN_DATE,
    }


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
    exp2971_path: Path | None = None,
    exp2972_path: Path | None = None,
    transcript_dir: Path | None = None,
) -> dict:
    start = monotonic()
    transcripts = transcript_dir or repo_root / "logs" / "experiment_2984_gatemate_readback_smoke_vector_v4"
    transcript_paths: list[str] = []
    command_durations: dict[str, float] = {}
    timing_observation: dict = {
        "timing_source": "host_wall_clock_command_duration",
        "command_durations_s": command_durations,
        "transcript_paths": transcript_paths,
        "transcript_sha256": {},
    }

    exp2971_file = exp2971_path or repo_root / "results" / EXP2971_FILENAME
    exp2972_file = exp2972_path or repo_root / "results" / EXP2972_FILENAME
    exp2971 = _read_json(exp2971_file)
    exp2972 = _read_json(exp2972_file)
    if not exp2971 or not exp2972:
        timing_observation["missing_artifacts"] = [
            str(path) for path, payload in ((exp2971_file, exp2971), (exp2972_file, exp2972)) if not payload
        ]
        return _base_artifact(
            honest_verdict="blocked_prior_gatemate_artifact_missing",
            start=start,
            monotonic=monotonic,
            board_detected=False,
            board_id="",
            tool_versions={},
            bitstream_path="",
            bitstream_sha256="",
            flash_succeeded=False,
            readback_supported=False,
            readback_attempted=False,
            readback_hash="",
            smoke_vector_attempted=False,
            smoke_vector_passed=False,
            observed_smoke_output="",
            expected_smoke_output="unavailable_prior_artifact_missing",
            timing_observation=timing_observation,
        )

    loader_hint = _loader_from_flash_command(exp2971) or _loader_from_flash_command(exp2972)
    tool_versions = _collect_tool_versions(
        run_command=run_command, which_func=which_func, loader_hint=loader_hint
    )
    loader = str(tool_versions.get("openFPGALoader", {}).get("path", ""))
    bitstream = _existing_path(
        str(exp2972.get("bitstream_path") or exp2971.get("bitstream_path", "")), repo_root
    )
    expected_sha = str(exp2972.get("bitstream_sha256") or exp2971.get("bitstream_sha256", ""))
    flash_succeeded = bool(exp2972.get("flash_succeeded", False))
    actual_sha = _sha256_file(bitstream) if bitstream.exists() else ""
    timing_observation["flash_command"] = str(
        exp2972.get("flash_command") or exp2971.get("flash_command", "")
    )
    timing_observation["prior_exp2972_transcript_sha256"] = exp2972.get("transcript_sha256", {})

    if not bitstream.exists() or actual_sha != expected_sha:
        timing_observation["failure_command"] = f"sha256sum {bitstream}"
        timing_observation["failure_excerpt"] = (
            f"bitstream missing or sha256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
        return _base_artifact(
            honest_verdict="blocked_bitstream_sha256_mismatch",
            start=start,
            monotonic=monotonic,
            board_detected=False,
            board_id="",
            tool_versions=tool_versions,
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            flash_succeeded=flash_succeeded,
            readback_supported=False,
            readback_attempted=False,
            readback_hash="",
            smoke_vector_attempted=False,
            smoke_vector_passed=False,
            observed_smoke_output="",
            expected_smoke_output="unavailable_bitstream_hash_mismatch",
            timing_observation=timing_observation,
        )

    if not flash_succeeded:
        timing_observation["failure_command"] = str(
            exp2972.get("flash_command") or exp2971.get("flash_command", "")
        )
        timing_observation["failure_excerpt"] = "Exp 2972 did not record flash_succeeded=true."
        return _base_artifact(
            honest_verdict="blocked_prior_flash_not_succeeded",
            start=start,
            monotonic=monotonic,
            board_detected=False,
            board_id="",
            tool_versions=tool_versions,
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            flash_succeeded=False,
            readback_supported=False,
            readback_attempted=False,
            readback_hash="",
            smoke_vector_attempted=False,
            smoke_vector_passed=False,
            observed_smoke_output="",
            expected_smoke_output="unavailable_prior_flash_failed",
            timing_observation=timing_observation,
        )

    if not loader:
        timing_observation["failure_command"] = "command -v openFPGALoader"
        timing_observation["failure_excerpt"] = "openFPGALoader not found in PATH or prior flash command."
        return _base_artifact(
            honest_verdict="blocked_openfpgaloader_missing",
            start=start,
            monotonic=monotonic,
            board_detected=False,
            board_id="",
            tool_versions=tool_versions,
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            flash_succeeded=True,
            readback_supported=False,
            readback_attempted=False,
            readback_hash="",
            smoke_vector_attempted=False,
            smoke_vector_passed=False,
            observed_smoke_output="",
            expected_smoke_output="unavailable_openfpgaloader_missing",
            timing_observation=timing_observation,
        )

    smoke_reason, smoke_evidence = _smoke_vector_blocker(repo_root)
    timing_observation["smoke_vector_reason"] = smoke_reason
    timing_observation["smoke_vector_evidence"] = smoke_evidence

    help_result = run_command([loader, "--help"], 10.0)
    readback_supported, readback_reason = _readback_decision(help_result)
    timing_observation["readback_reason"] = readback_reason

    detect_command = [loader, "-c", "dirtyJtag", "--detect"]
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
    board_id = _extract_board_id(detect_result)
    board_detected = bool(board_id)
    timing_observation["transcript_sha256"] = _transcript_hashes(transcript_paths)
    timing_observation["dirtyjtag_contact_detected"] = (
        detect_result.returncode == 0 and "jtag frequency" in _command_text(detect_result).lower()
    )
    if not board_detected:
        timing_observation["failure_command"] = _quote(detect_command)
        timing_observation["failure_excerpt"] = _command_text(detect_result)
        return _base_artifact(
            honest_verdict="blocked_board_not_detected",
            start=start,
            monotonic=monotonic,
            board_detected=False,
            board_id="",
            tool_versions=tool_versions,
            bitstream_path=str(bitstream),
            bitstream_sha256=actual_sha,
            flash_succeeded=True,
            readback_supported=readback_supported,
            readback_attempted=False,
            readback_hash="",
            smoke_vector_attempted=False,
            smoke_vector_passed=False,
            observed_smoke_output="",
            expected_smoke_output="unavailable_board_not_detected",
            timing_observation=timing_observation,
        )

    readback_attempted = False
    readback_hash = ""
    if readback_supported:
        readback_attempted = True
        readback_path = transcripts / "readback.bin"
        readback_command = [loader, "-c", "dirtyJtag", "--readback", str(readback_path)]
        _run_recorded(
            label="readback",
            command=readback_command,
            timeout_s=120.0,
            run_command=run_command,
            monotonic=monotonic,
            transcript_dir=transcripts,
            transcript_paths=transcript_paths,
            command_durations=command_durations,
        )
        readback_hash = _sha256_file(readback_path) if readback_path.exists() else ""
        timing_observation["readback_path"] = str(readback_path)

    timing_observation["transcript_sha256"] = _transcript_hashes(transcript_paths)
    verdict = (
        "complete: gatemate_readback_captured_no_host_smoke_io"
        if readback_hash
        else "complete: gatemate_no_readback_no_host_smoke_io"
    )
    return _base_artifact(
        honest_verdict=verdict,
        start=start,
        monotonic=monotonic,
        board_detected=True,
        board_id=board_id,
        tool_versions=tool_versions,
        bitstream_path=str(bitstream),
        bitstream_sha256=actual_sha,
        flash_succeeded=True,
        readback_supported=readback_supported,
        readback_attempted=readback_attempted,
        readback_hash=readback_hash,
        smoke_vector_attempted=False,
        smoke_vector_passed=False,
        observed_smoke_output="",
        expected_smoke_output="unavailable_no_host_visible_io_path",
        timing_observation=timing_observation,
    )


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc = shutil.which,
    monotonic: ClockFunc = time.monotonic,
    exp2971_path: Path | None = None,
    exp2972_path: Path | None = None,
) -> dict:
    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(
        repo_root=root,
        run_command=run_command,
        which_func=which_func,
        monotonic=monotonic,
        exp2971_path=exp2971_path,
        exp2972_path=exp2972_path,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
