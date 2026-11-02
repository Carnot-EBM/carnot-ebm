"""GateMate physical-board flash smoke and timing boundary for Exp 2929.

This experiment is intentionally gate-heavy because a physical FPGA flash is
not an ordinary software benchmark. The module first proves that Exp 2928
produced a hash-addressed n=16 GateMate bitstream. Only then may it contact the
board, preserve raw command transcripts, and attempt a bounded flash smoke.
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


ARTIFACT_FILENAME = "experiment_2929_gatemate_flash_timing_boundary_v1.json"
EXP2928_FILENAME = "experiment_2928_gatemate_n16_himbaechel_bitstream_build_v3.json"
RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "physical_board_smoke"
DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
OPENFPGALOADER_FLASH_PREFIX = (
    "openFPGALoader",
    "-c",
    "dirtyJtag",
    "-b",
    "olimex_gatemateevb",
)
DOCUMENTED_FLASH_SNIPPET = "openFPGALoader -c dirtyJtag -b olimex_gatemateevb"
FLASH_SCRIPT_CANDIDATES = (
    Path("hardware") / "gatemate" / "flash_gatemate.sh",
    Path("hardware") / "gatemate" / "flash.sh",
    Path("hardware") / "flash_gatemate.sh",
)


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result so tests can inject exact board transcripts."""

    returncode: int
    stdout: str
    stderr: str


RunCommand = Callable[[list[str], float], CommandResult]
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


def command_result_text(result: CommandResult) -> str:
    return f"{result.stdout}{result.stderr}"


def _quote_command(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def _write_transcript(path: Path, command: list[str], result: CommandResult) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"COMMAND: {_quote_command(command)}",
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


def _duration(start: float, monotonic: ClockFunc) -> float:
    return round(monotonic() - start, 6)


def _base_artifact(
    *,
    honest_verdict: str,
    blocker: str,
    duration_s: float,
    gatemate_flash_smoke_ready: bool = False,
    board_detected: bool = False,
    bitstream_sha256_verified: bool = False,
    flash_attempted: bool = False,
    flash_transcript_path: str = "",
    board_contact_transcript_path: str = "",
    timing_boundary_ready: bool = False,
) -> dict:
    return {
        "honest_verdict": honest_verdict,
        "gatemate_flash_smoke_ready": gatemate_flash_smoke_ready,
        "board_detected": board_detected,
        "bitstream_sha256_verified": bitstream_sha256_verified,
        "flash_attempted": flash_attempted,
        "flash_transcript_path": flash_transcript_path,
        "board_contact_transcript_path": board_contact_transcript_path,
        "timing_boundary_ready": timing_boundary_ready,
        "timing_claim_allowed": False,
        "speedup_claim_allowed": False,
        "blocker": blocker,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "run_date": RUN_DATE,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _board_detected(result: CommandResult) -> bool:
    text = command_result_text(result)
    return result.returncode == 0 and any(token in text for token in ("GateMate", "GM1A", "IDCODE"))


def _documented_openfpgaloader_path(wishlist_path: Path) -> bool:
    return wishlist_path.exists() and DOCUMENTED_FLASH_SNIPPET in wishlist_path.read_text(
        encoding="utf-8"
    )


def _existing_flash_script(repo_root: Path) -> Path | None:
    return next((repo_root / candidate for candidate in FLASH_SCRIPT_CANDIDATES if (repo_root / candidate).exists()), None)


def _bitstream_path(repo_root: Path, exp2928: dict) -> Path:
    candidate = Path(str(exp2928.get("bitstream_path", ""))).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    monotonic: ClockFunc = time.monotonic,
    exp2928_path: Path | None = None,
    wishlist_path: Path | None = None,
    transcript_dir: Path | None = None,
) -> dict:
    start = monotonic()
    exp_path = exp2928_path or repo_root / "results" / EXP2928_FILENAME
    wishlist = wishlist_path or repo_root / "research-hardware-wishlist.md"
    transcripts = transcript_dir or repo_root / "logs" / "experiment_2929_gatemate_flash_timing_boundary_v1"

    if not exp_path.exists():
        return _base_artifact(
            honest_verdict="blocked_gatemate_bitstream_missing",
            blocker=f"missing exp2928 artifact: {exp_path}",
            duration_s=_duration(start, monotonic),
        )

    exp2928 = json.loads(exp_path.read_text(encoding="utf-8"))
    if not exp2928.get("gatemate_bitstream_built", False):
        return _base_artifact(
            honest_verdict="blocked_gatemate_bitstream_missing",
            blocker="exp2928 gatemate_bitstream_built is false",
            duration_s=_duration(start, monotonic),
        )

    bitstream = _bitstream_path(repo_root, exp2928)
    expected_sha = str(exp2928.get("bitstream_sha256", ""))
    if not bitstream.exists() or not expected_sha:
        return _base_artifact(
            honest_verdict="blocked_gatemate_bitstream_missing",
            blocker=f"exp2928 bitstream path or sha missing: {bitstream}",
            duration_s=_duration(start, monotonic),
        )

    detect_command = list(DETECT_COMMAND)
    detect_result = run_command(detect_command, 20.0)
    initial_contact_path = _write_transcript(
        transcripts / "initial_board_contact.txt",
        detect_command,
        detect_result,
    )
    detected = _board_detected(detect_result)
    if not detected:
        return _base_artifact(
            honest_verdict="blocked_board_not_detected",
            blocker="GateMate DirtyJTAG detect command did not report board contact",
            board_contact_transcript_path=initial_contact_path,
            duration_s=_duration(start, monotonic),
        )

    actual_sha = _sha256(bitstream)
    sha_verified = actual_sha == expected_sha
    if not sha_verified:
        return _base_artifact(
            honest_verdict="blocked_bitstream_sha256_mismatch",
            blocker=f"bitstream sha256 mismatch: expected {expected_sha}, got {actual_sha}",
            board_detected=True,
            board_contact_transcript_path=initial_contact_path,
            duration_s=_duration(start, monotonic),
        )

    flash_script = _existing_flash_script(repo_root)
    if flash_script is None and not _documented_openfpgaloader_path(wishlist):
        return _base_artifact(
            honest_verdict="blocked_flash_path_undocumented",
            blocker="documented GateMate openFPGALoader flash path missing",
            board_detected=True,
            bitstream_sha256_verified=True,
            board_contact_transcript_path=initial_contact_path,
            duration_s=_duration(start, monotonic),
        )

    flash_command = (
        [str(flash_script), str(bitstream)]
        if flash_script is not None
        else [*OPENFPGALOADER_FLASH_PREFIX, str(bitstream)]
    )
    flash_result = run_command(flash_command, 120.0)
    flash_transcript_path = _write_transcript(
        transcripts / "flash.txt",
        flash_command,
        flash_result,
    )
    if flash_result.returncode != 0:
        return _base_artifact(
            honest_verdict="blocked_flash_failed",
            blocker="GateMate flash command failed",
            board_detected=True,
            bitstream_sha256_verified=True,
            flash_attempted=True,
            flash_transcript_path=flash_transcript_path,
            board_contact_transcript_path=initial_contact_path,
            duration_s=_duration(start, monotonic),
        )

    post_contact_result = run_command(detect_command, 20.0)
    post_contact_path = _write_transcript(
        transcripts / "post_flash_board_contact.txt",
        detect_command,
        post_contact_result,
    )
    post_flash_detected = _board_detected(post_contact_result)
    return _base_artifact(
        honest_verdict=(
            "gatemate_flash_contact_smoke_no_timing_counter"
            if post_flash_detected
            else "blocked_post_flash_board_contact_missing"
        ),
        blocker=("" if post_flash_detected else "post-flash board contact was not detected"),
        gatemate_flash_smoke_ready=post_flash_detected,
        board_detected=post_flash_detected,
        bitstream_sha256_verified=True,
        flash_attempted=True,
        flash_transcript_path=flash_transcript_path,
        board_contact_transcript_path=post_contact_path,
        duration_s=_duration(start, monotonic),
    )


def run_experiment(
    *,
    repo_root: Path | None = None,
    artifact_path: Path | None = None,
    run_command: RunCommand = _default_run_command,
    monotonic: ClockFunc = time.monotonic,
) -> dict:
    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(
        repo_root=root,
        run_command=run_command,
        monotonic=monotonic,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
