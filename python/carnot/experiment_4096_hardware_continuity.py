"""Exp 4096 per-board hardware continuity and GateMate flash-blocker record.

This experiment keeps the same board-smoke contract as Exp 4084, but tightens
the GateMate reachability evidence back to the chip identity requested by the
operator: `openFPGALoader -c dirtyJtag --detect` must expose the GM1Ax IDCODE
before the board is marked reachable. When the existing n=16 GateMate flash is
blocked, the artifact carries the concrete loader error and the next actionable
hardware step instead of only recording a return code.

Spec refs: REQ-HW-4096, SCENARIO-HW-4096.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from carnot import experiment_4064_hardware_continuity as _base


EXPERIMENT_ID = 4096
SCHEMA = _base.SCHEMA
SPEC_REFS = ["REQ-HW-4096", "SCENARIO-HW-4096"]
OUTPUT_REL_PATH = Path("results") / "experiment_4096_hardware_continuity.json"
RANDOM_SEED = 4096
INFERENCE_SUBSTRATE = _base.INFERENCE_SUBSTRATE
BOARD_NAMES = _base.BOARD_NAMES

KV260_SSH_PRECONDITION = _base.KV260_SSH_PRECONDITION
GATEMATE_DETECT_COMMAND = _base.GATEMATE_DETECT_COMMAND
POLARFIRE_SSH_PRECONDITION = _base.POLARFIRE_SSH_PRECONDITION
GATEMATE_FLASH_BOARD = _base.GATEMATE_FLASH_BOARD

REQUIRED_ARTIFACT_FIELDS = _base.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = dict(_base.FIELD_PRINCIPLES)

CommandProbe = _base.CommandProbe
StepOutcome = _base.StepOutcome
CommandRunner = _base.CommandRunner
Clock = _base.Clock
StepRunner = _base.StepRunner

run_command = _base.run_command
prepend_oss_cad_suite = _base.prepend_oss_cad_suite
command_to_string = _base.command_to_string
payload_checksum = _base.payload_checksum
state_token = _base.state_token


def _gatemate_available(probe: CommandProbe) -> bool:
    """Return true only when the DirtyJTAG detect transcript identifies GM1Ax."""
    text = probe.combined_output.lower()
    return probe.exit_code == 0 and (
        "0x20000001" in text
        or "gm1ax" in text
        or ("gm1a" in text and ("gatemate" in text or "colognechip" in text))
    )


_CONFIG_OVERRIDES = {
    "EXPERIMENT_ID": EXPERIMENT_ID,
    "SPEC_REFS": SPEC_REFS,
    "OUTPUT_REL_PATH": OUTPUT_REL_PATH,
    "RANDOM_SEED": RANDOM_SEED,
    "_gatemate_available": _gatemate_available,
    "run_gatemate_forward_step": None,
}


@contextmanager
def _configured_base() -> Iterator[Any]:
    """Temporarily apply Exp 4096 constants to the shared continuity engine."""
    overrides = dict(_CONFIG_OVERRIDES)
    overrides["run_gatemate_forward_step"] = run_gatemate_forward_step
    original = {name: getattr(_base, name) for name in overrides}
    for name, value in overrides.items():
        setattr(_base, name, value)
    try:
        yield _base
    finally:
        for name, value in original.items():
            setattr(_base, name, value)


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = _base.time.perf_counter,
    gatemate_step_runner: StepRunner | None = None,
    polarfire_step_runner: StepRunner | None = None,
) -> dict[str, Any]:
    """Build the Exp 4096 artifact from live or injected board-access probes."""
    root = Path(repo_root)
    with _configured_base() as base:
        artifact = base.build_artifact(
            repo_root=root,
            command_runner=command_runner,
            clock=clock,
            gatemate_step_runner=gatemate_step_runner,
            polarfire_step_runner=polarfire_step_runner,
        )
        artifact["source_context"] = _source_context(root)
        _enrich_blocked_gatemate_step(artifact)
        artifact["reproducibility_checksum"] = base.payload_checksum(artifact)
        validate_artifact(artifact)
        return artifact


def run_gatemate_forward_step(
    *,
    repo_root: Path,
    command_runner: CommandRunner,
    clock: Clock = _base.time.perf_counter,
) -> StepOutcome:
    """Flash the existing n=16 bitstream and preserve the exact flash blocker."""
    started = clock()
    previous = _source_context(repo_root)
    candidates = _base._candidate_gatemate_bitstreams(repo_root)
    bitstream = next((candidate for candidate in candidates if candidate.exists()), None)
    if bitstream is None:
        return StepOutcome(
            step_taken="blocked_gatemate_no_existing_n16_bitstream",
            terminal_state="reachable_step_blocked_no_existing_n16_bitstream",
            success=False,
            duration_s=_base._elapsed(started, clock),
            details={
                "step": "flash_existing_n16_bitstream_plus_dirtyjtag_detect_smoke",
                "candidate_paths_checked": [str(path) for path in candidates],
                "blocker": "blocked_gatemate_no_existing_n16_bitstream",
                "next_concrete_step": "Rebuild the GateMate n=16 bitstream, then rerun the DirtyJTAG flash smoke.",
                "previous_377_flash_error": previous["previous_gatemate_flash_error"],
            },
        )

    flash_command = (
        "openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        GATEMATE_FLASH_BOARD,
        str(bitstream),
    )
    flash_probe = command_runner(flash_command, 120.0)
    detect_probe = command_runner(GATEMATE_DETECT_COMMAND, 30.0)
    flash_ok = flash_probe.exit_code == 0
    detect_ok = _gatemate_available(detect_probe)
    success = flash_ok and detect_ok
    if success:
        step_taken = "gatemate_existing_n16_bitstream_flash_detect_smoke_succeeded"
        terminal = "reachable_n16_bitstream_flashed_detect_smoke_recorded"
    elif not flash_ok:
        step_taken = f"gatemate_existing_n16_bitstream_flash_blocked_returncode_{flash_probe.exit_code}"
        terminal = "reachable_n16_bitstream_flash_blocked"
    else:
        step_taken = "gatemate_existing_n16_bitstream_post_flash_detect_blocked"
        terminal = "reachable_n16_bitstream_post_flash_detect_blocked"

    return StepOutcome(
        step_taken=step_taken,
        terminal_state=terminal,
        success=success,
        duration_s=_base._elapsed(started, clock),
        details={
            "step": "flash_existing_n16_bitstream_plus_dirtyjtag_detect_smoke",
            "bitstream_path": str(bitstream),
            "bitstream_sha256": _base._sha256_file(bitstream),
            "flash_command": command_to_string(flash_command),
            "post_flash_detect_command": command_to_string(GATEMATE_DETECT_COMMAND),
            "flash_exit_code": flash_probe.exit_code,
            "post_flash_detect_exit_code": detect_probe.exit_code,
            "post_flash_detected_gatemate": detect_ok,
            "flash_error": _extract_flash_error(flash_probe.combined_output),
            "next_concrete_step": _next_gatemate_flash_step(flash_probe, detect_probe),
            "previous_377_flash_error": previous["previous_gatemate_flash_error"],
            "command_transcripts": [
                _base._command_transcript("flash_existing_bitstream", flash_probe),
                _base._command_transcript("post_flash_detect_smoke", detect_probe),
            ],
        },
    )


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = _base.time.perf_counter,
) -> Path:
    """Run Exp 4096 and write `results/experiment_4096_hardware_continuity.json`."""
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    with _configured_base() as base:
        return base.write_artifact(repo_root, artifact)


def validate_artifact(artifact: dict[str, Any]) -> None:
    with _configured_base() as base:
        base.validate_artifact(artifact)
    _validate_4096_fields(artifact)


def _validate_4096_fields(artifact: dict[str, Any]) -> None:
    source = artifact.get("source_context")
    _require(isinstance(source, dict), "source_context must be present")
    _require(source.get("previous_experiment") == 4084, "source_context must read Exp 4084")
    gate_step = artifact.get("gatemate_step", {})
    if (
        artifact.get("gatemate_reachable") is True
        and isinstance(gate_step, dict)
        and gate_step.get("flash_exit_code") not in (None, 0)
    ):
        _require(bool(gate_step.get("flash_error")), "blocked GateMate flash must record flash_error")
        _require(
            bool(gate_step.get("next_concrete_step")),
            "blocked GateMate flash must record next_concrete_step",
        )
    if artifact.get("gatemate_step_taken") == "blocked_gatemate_unreachable":
        _require(
            isinstance(gate_step, dict) and bool(gate_step.get("next_concrete_step")),
            "blocked GateMate reachability must record next_concrete_step",
        )


def _enrich_blocked_gatemate_step(artifact: dict[str, Any]) -> None:
    if artifact.get("gatemate_step_taken") != "blocked_gatemate_unreachable":
        return
    source = artifact["source_context"]
    step = dict(artifact.get("gatemate_step", {}))
    step["previous_377_flash_error"] = source["previous_gatemate_flash_error"]
    step["previous_377_flash_exit_code"] = source["previous_gatemate_flash_exit_code"]
    step["next_concrete_step"] = (
        "Recover GM1Ax IDCODE visibility with `openFPGALoader -c dirtyJtag --detect`, "
        "then rerun the n=16 GateMate flash; if detect still reports only JTAG "
        "frequency, inspect DirtyJTAG/GateMate power, cable, and openFPGALoader "
        "board-profile matching before any flash retry."
    )
    artifact["gatemate_step"] = step


def _source_context(repo_root: Path) -> dict[str, Any]:
    prior_payload = _read_previous_artifact(repo_root)
    gate_step = prior_payload.get("gatemate_step", {}) if isinstance(prior_payload, dict) else {}
    flash_error = _extract_flash_error_from_transcripts(gate_step.get("command_transcripts", []))
    return {
        "previous_experiment": 4084,
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_per_board_reachability": prior_payload.get("per_board_reachability"),
        "previous_gatemate_step_taken": prior_payload.get("gatemate_step_taken"),
        "previous_gatemate_flash_exit_code": gate_step.get("flash_exit_code"),
        "previous_gatemate_flash_error": flash_error,
    }


def _read_previous_artifact(repo_root: Path) -> dict[str, Any]:
    prior_path = repo_root / "results" / "experiment_4084_hardware_continuity.json"
    if not prior_path.exists():
        return {}
    try:
        return json.loads(prior_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt local artifact.
        return {}


def _extract_flash_error_from_transcripts(transcripts: Any) -> str | None:
    if not isinstance(transcripts, list):
        return None
    for transcript in transcripts:
        if not isinstance(transcript, dict):
            continue
        if transcript.get("stage") != "flash_existing_bitstream":
            continue
        error = _extract_flash_error(str(transcript.get("output_excerpt", "")))
        if error:
            return error
    return None


def _extract_flash_error(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("error:"):
            return line
    for line in lines:
        if "no device found" in line.lower() or "fails to open device" in line.lower():
            return line
    return lines[-1] if lines else ""


def _next_gatemate_flash_step(flash_probe: CommandProbe, detect_probe: CommandProbe) -> str:
    if flash_probe.exit_code == 0 and _gatemate_available(detect_probe):
        return "GateMate n=16 flash smoke succeeded; next run host-visible readback or dispatch smoke."
    flash_error = _extract_flash_error(flash_probe.combined_output)
    if "no device found" in flash_error.lower():
        return (
            "Detect confirms GM1Ax but the n=16 flash says no device found; next run "
            "`openFPGALoader -c dirtyJtag --detect` immediately followed by "
            "`openFPGALoader -c dirtyJtag -b olimex_gatemateevb <bitstream>` with "
            "verbose cable/board-profile diagnostics, then update or bypass the "
            "Olimex GateMate board profile if the mismatch persists."
        )
    return (
        f"GateMate n=16 flash returned rc={flash_probe.exit_code}; next preserve the "
        "full openFPGALoader transcript and retry after confirming DirtyJTAG still "
        "reports GM1Ax on the same USB path."
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)  # pragma: no cover


def main() -> None:  # pragma: no cover
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"per_board_reachability: {artifact['per_board_reachability']}")
    print(f"gatemate_step_taken: {artifact['gatemate_step_taken']}")
    print(f"polarfire_step_taken: {artifact['polarfire_step_taken']}")
    print(f"per_board_duration_s: {artifact['per_board_duration_s']}")


if __name__ == "__main__":  # pragma: no cover
    main()
