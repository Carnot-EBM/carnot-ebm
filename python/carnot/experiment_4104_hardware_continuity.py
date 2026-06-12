"""Exp 4104 per-board hardware continuity status artifact.

This experiment keeps the existing hardware-smoke engine but records the newer
Hardware-Task Continuity Discipline surface: one honest status and next-step
record per board. GateMate and PolarFire remain non-terminal; KV260 is terminal
and receives only the SSH confirmation allowed by the SSH-Not-SD-Card rule.

Spec refs: REQ-HW-4104, SCENARIO-HW-4104.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from carnot import experiment_4064_hardware_continuity as _base


EXPERIMENT_ID = 4104
SCHEMA = "carnot.hardware_continuity_per_board_status.v1"
SPEC_REFS = ["REQ-HW-4104", "SCENARIO-HW-4104"]
OUTPUT_REL_PATH = Path("results") / "experiment_4104_hardware_continuity.json"
RANDOM_SEED = 4104
INFERENCE_SUBSTRATE = _base.INFERENCE_SUBSTRATE
BOARD_NAMES = _base.BOARD_NAMES

KV260_SSH_PRECONDITION = _base.KV260_SSH_PRECONDITION
GATEMATE_DETECT_COMMAND = _base.GATEMATE_DETECT_COMMAND
POLARFIRE_SSH_PRECONDITION = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)
GATEMATE_FLASH_BOARD = _base.GATEMATE_FLASH_BOARD

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "per_board_status",
    "preconditions_checked",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Per-board honest status; an honest blocked_<board> "
        "is a COMPLETE verdict for that board."
    ),
    "per_board_status": (
        "One reachability + next-step record per board so a board is never "
        "silently forgotten (the failure mode this discipline exists to prevent)."
    ),
    "preconditions_checked": (
        "Records the SSH/USB-detect preconditions actually run; enforces "
        "SSH-Not-SD-Card and pre-empts fabricated hardware results."
    ),
}

_BASE_FIELD_PRINCIPLES = dict(_base.FIELD_PRINCIPLES)
_BASE_FIELD_PRINCIPLES.update(FIELD_PRINCIPLES)

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
    """Return true only when DirtyJTAG detect exposes the GateMate GM1Ax IDCODE."""
    text = probe.combined_output.lower()
    return probe.exit_code == 0 and (
        "0x20000001" in text
        or "gm1ax" in text
        or ("gm1a" in text and ("gatemate" in text or "colognechip" in text))
    )


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return a terminal-prefixed verdict that keeps every board visible."""
    return (
        "complete: hardware_continuity_4104_"
        f"gatemate_{state_token(str(artifact['gatemate_step_taken']))}_"
        f"polarfire_{state_token(str(artifact['polarfire_step_taken']))}_"
        f"kv260_{state_token(str(artifact['per_board_terminal_state']['kv260']))}_"
        "ssh_usb_detect_only"
    )


@contextmanager
def _configured_base() -> Iterator[Any]:
    """Temporarily apply Exp 4104 constants to the shared continuity engine."""
    overrides = {
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "SCHEMA": SCHEMA,
        "SPEC_REFS": SPEC_REFS,
        "OUTPUT_REL_PATH": OUTPUT_REL_PATH,
        "RANDOM_SEED": RANDOM_SEED,
        "FIELD_PRINCIPLES": _BASE_FIELD_PRINCIPLES,
        "POLARFIRE_SSH_PRECONDITION": POLARFIRE_SSH_PRECONDITION,
        "_gatemate_available": _gatemate_available,
        "honest_verdict": honest_verdict,
        "run_gatemate_forward_step": run_gatemate_forward_step,
    }
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
    """Build the Exp 4104 artifact from live or injected board probes."""
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
    artifact["field_principles"].update(FIELD_PRINCIPLES)
    artifact["per_board_status"] = _build_per_board_status(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run_gatemate_forward_step(
    *,
    repo_root: Path,
    command_runner: CommandRunner,
    clock: Clock = _base.time.perf_counter,
) -> StepOutcome:
    """Flash the most recent n=16 GateMate bitstream and preserve blockers."""
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
                "next_concrete_step": (
                    "Rebuild the GateMate n=16 bitstream from the most recent "
                    "GateMate artifact, then rerun the DirtyJTAG flash smoke."
                ),
                "previous_4096_gatemate_step_taken": previous["previous_gatemate_step_taken"],
                "previous_4096_flash_error": previous["previous_gatemate_flash_error"],
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
            "previous_4096_gatemate_step_taken": previous["previous_gatemate_step_taken"],
            "previous_4096_flash_error": previous["previous_gatemate_flash_error"],
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
    """Run Exp 4104 and write `results/experiment_4104_hardware_continuity.json`."""
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4104")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4104")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4104")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4104")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _require(
        "mmcblk" not in json.dumps(artifact, sort_keys=True, default=str).lower(),
        "SD-card precondition marker is forbidden",
    )
    _validate_principles(artifact)
    _validate_preconditions(artifact)
    _validate_per_board_status(artifact)
    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be hardware_smoke",
    )
    verdict = str(artifact.get("honest_verdict", ""))
    _require(
        verdict.startswith("complete:") or verdict.startswith("blocked_"),
        "honest_verdict must have a terminal prefix",
    )
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "stale verdict")
    _require(artifact.get("fabric_acceleration_claimed") is False, "no fabric acceleration claim")
    _require(artifact.get("speedup_claim_made") is False, "no speedup claim")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _build_per_board_status(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    preconditions = _preconditions_by_board(artifact["preconditions_checked"])
    statuses: dict[str, dict[str, Any]] = {}
    for board in BOARD_NAMES:
        precondition = preconditions[board]
        record: dict[str, Any] = {
            "reachable": bool(artifact["per_board_reachability"][board]),
            "status": _board_status(board, artifact),
            "terminal_state": artifact["per_board_terminal_state"][board],
            "next_concrete_step": _next_concrete_step(board, artifact),
            "precondition_resource": precondition["resource"],
            "precondition_command": precondition["command"],
            "precondition_exit_code": precondition["exit_code"],
            "precondition_available": precondition["available"],
            "duration_s": artifact["per_board_duration_s"][board],
            "timer_scope": "precondition_plus_forward_step_wall_clock",
        }
        if board == "polarfire" and "result_hash_match" in artifact.get("polarfire_step", {}):
            record["hash_match"] = bool(artifact["polarfire_step"]["result_hash_match"])
        statuses[board] = record
    return statuses


def _preconditions_by_board(preconditions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    resource_to_board = {
        "kv260_ssh": "kv260",
        "gatemate_jtag_detect": "gatemate",
        "polarfire_ssh": "polarfire",
    }
    return {
        resource_to_board[entry["resource"]]: entry
        for entry in preconditions
        if entry.get("resource") in resource_to_board
    }


def _board_status(board: str, artifact: dict[str, Any]) -> str:
    if board == "kv260":
        return (
            "kv260_terminal_confirmed_ssh_only"
            if artifact["per_board_reachability"]["kv260"]
            else "blocked_kv260_unreachable"
        )
    if board == "gatemate":
        return str(artifact["gatemate_step_taken"])
    return str(artifact["polarfire_step_taken"])


def _next_concrete_step(board: str, artifact: dict[str, Any]) -> str:
    if board == "kv260":
        return (
            "kv260_terminal_state_confirmed_via_ssh"
            if artifact["per_board_reachability"]["kv260"]
            else "blocked_kv260_unreachable"
        )
    if board == "gatemate":
        step = artifact.get("gatemate_step", {})
        if isinstance(step, dict) and step.get("next_concrete_step"):
            return str(step["next_concrete_step"])
        return str(artifact["gatemate_step_taken"])
    if not artifact["per_board_reachability"]["polarfire"]:
        return (
            "Restore `ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'`, "
            "then rerun the hash-match Carnot dispatch smoke."
        )
    step = artifact.get("polarfire_step", {})
    if isinstance(step, dict) and step.get("next_concrete_step"):
        return str(step["next_concrete_step"])
    return str(artifact["polarfire_step_taken"])


def _source_context(repo_root: Path) -> dict[str, Any]:
    prior_path = repo_root / "results" / "experiment_4096_hardware_continuity.json"
    prior_payload = _read_json(prior_path)
    gate_step = prior_payload.get("gatemate_step", {}) if isinstance(prior_payload, dict) else {}
    flash_error = None
    flash_exit_code = None
    if isinstance(gate_step, dict):
        flash_error = (
            gate_step.get("flash_error")
            or gate_step.get("previous_4096_flash_error")
            or gate_step.get("previous_377_flash_error")
            or _extract_flash_error_from_transcripts(gate_step.get("command_transcripts", []))
        )
        flash_exit_code = gate_step.get("flash_exit_code") or gate_step.get(
            "previous_377_flash_exit_code"
        )
    return {
        "previous_experiment": 4096,
        "most_recent_gatemate_artifact": str(prior_path),
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_per_board_reachability": prior_payload.get("per_board_reachability"),
        "previous_gatemate_step_taken": prior_payload.get("gatemate_step_taken"),
        "previous_gatemate_flash_exit_code": flash_exit_code,
        "previous_gatemate_flash_error": flash_error,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt local artifact.
        return {}


def _enrich_blocked_gatemate_step(artifact: dict[str, Any]) -> None:
    if artifact.get("gatemate_step_taken") != "blocked_gatemate_unreachable":
        return
    source = artifact["source_context"]
    step = dict(artifact.get("gatemate_step", {}))
    step["previous_4096_gatemate_step_taken"] = source["previous_gatemate_step_taken"]
    step["previous_4096_flash_error"] = source["previous_gatemate_flash_error"]
    step["next_concrete_step"] = (
        "Recover GM1Ax IDCODE visibility with `openFPGALoader -c dirtyJtag --detect`, "
        "then rerun the n=16 GateMate flash from the most recent GateMate artifact."
    )
    artifact["gatemate_step"] = step


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
        lowered = line.lower()
        if "no device found" in lowered or "fails to open device" in lowered:
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
            "verbose cable/board-profile diagnostics."
        )
    return (
        f"GateMate n=16 flash returned rc={flash_probe.exit_code}; next preserve the "
        "full openFPGALoader transcript and retry after confirming DirtyJTAG still "
        "reports GM1Ax on the same USB path."
    )


def _validate_principles(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    _require(isinstance(principles, dict), "field_principles must be dict")
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(principles.get(field) == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
        _require(
            not (isinstance(artifact[field], dict) and set(artifact[field]) == {"value", "principle"}),
            f"{field} must remain a bare value",
        )


def _validate_preconditions(artifact: dict[str, Any]) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(isinstance(preconditions, list), "preconditions_checked must be a list")
    _require(len(preconditions) == 3, "exactly three preconditions are required")
    expected = {
        "kv260_ssh": command_to_string(KV260_SSH_PRECONDITION),
        "gatemate_jtag_detect": command_to_string(GATEMATE_DETECT_COMMAND),
        "polarfire_ssh": command_to_string(POLARFIRE_SSH_PRECONDITION),
    }
    resources = [entry.get("resource") for entry in preconditions if isinstance(entry, dict)]
    _require(resources == ["kv260_ssh", "gatemate_jtag_detect", "polarfire_ssh"], "")
    for entry in preconditions:
        _require(isinstance(entry, dict), "precondition entries must be dicts")
        _require({"resource", "available", "command", "exit_code"} <= set(entry), "precondition missing keys")
        _require(isinstance(entry["available"], bool), "available must be bool")
        _require(entry["command"] == expected[entry["resource"]], "invalid precondition command")


def _validate_per_board_status(artifact: dict[str, Any]) -> None:
    status = artifact.get("per_board_status")
    reachability = artifact.get("per_board_reachability")
    durations = artifact.get("per_board_duration_s")
    _require(isinstance(status, dict), "per_board_status must be dict")
    _require(set(status) == set(BOARD_NAMES), "per_board_status must include all boards")
    _require(isinstance(reachability, dict) and set(reachability) == set(BOARD_NAMES), "")
    _require(isinstance(durations, dict) and set(durations) == set(BOARD_NAMES), "")
    for board, record in status.items():
        _require(isinstance(record, dict), "per-board record must be dict")
        _require(
            {
                "reachable",
                "status",
                "next_concrete_step",
                "precondition_resource",
                "precondition_command",
                "duration_s",
            }
            <= set(record),
            "per-board status missing keys",
        )
        _require(record["reachable"] is reachability[board], "per-board reachability mismatch")
        _require(isinstance(record["status"], str) and record["status"], "missing board status")
        _require(isinstance(record["next_concrete_step"], str) and record["next_concrete_step"], "")
        _require(float(record["duration_s"]) > 0.0, "duration_s must be positive")
        _require(float(record["duration_s"]) == float(durations[board]), "duration mismatch")
        if not record["reachable"]:
            _require(record["status"] == f"blocked_{board}_unreachable", "blocked status mismatch")
    _require(
        status["polarfire"]["precondition_command"] == command_to_string(POLARFIRE_SSH_PRECONDITION),
        "PolarFire BatchMode precondition required",
    )
    _require(
        status["gatemate"]["precondition_command"] == command_to_string(GATEMATE_DETECT_COMMAND),
        "GateMate DirtyJTAG detect precondition required",
    )
    _require(
        status["kv260"]["precondition_command"] == command_to_string(KV260_SSH_PRECONDITION),
        "KV260 SSH precondition required",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"per_board_status: {artifact['per_board_status']}")
    print(f"preconditions_checked: {artifact['preconditions_checked']}")


if __name__ == "__main__":  # pragma: no cover
    main()
