"""ARC-focused KV260, GateMate, and PolarFire continuity check for Exp 4017.

This run keeps the three attached boards visible while ARC remains the focus.
It only checks the required SSH or USB reachability paths, records continuity
evidence for reachable boards, and records explicit blocked next steps for
unreachable boards. KV260 stays SSH-only; host SD-card devices are not a gate.

Spec refs: REQ-HW-4017, SCENARIO-HW-4017.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_3972_hardware_continuity as _hardware_base
from carnot import experiment_4006_hardware_continuity as _previous


EXPERIMENT_ID = 4017
SCHEMA = "carnot.hardware_continuity_arc_reachability.v3"
SPEC_REFS = ["REQ-HW-4017", "SCENARIO-HW-4017"]
OUTPUT_REL_PATH = Path("results") / "experiment_4017_hardware_continuity.json"
RANDOM_SEED = 4017
INFERENCE_SUBSTRATE = _previous.INFERENCE_SUBSTRATE

CommandResult = _previous.CommandResult
CommandRunner = _previous.CommandRunner
Clock = _previous.Clock

KV260_SSH_PRECONDITION = _previous.KV260_SSH_PRECONDITION
GATEMATE_DETECT_COMMAND = _previous.GATEMATE_DETECT_COMMAND
POLARFIRE_SSH_PRECONDITION = _previous.POLARFIRE_SSH_PRECONDITION
KV260_LISTAPPS_COMMAND = _previous.KV260_LISTAPPS_COMMAND
KV260_UIO_COMMAND = _previous.KV260_UIO_COMMAND
POLARFIRE_CONTINUITY_COMMAND = _previous.POLARFIRE_CONTINUITY_COMMAND

BOARD_NAMES = _previous.BOARD_NAMES
REQUIRED_ARTIFACT_FIELDS = _previous.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = {
    "kv260_reachable": (
        "BARE BOOLs -- per-board SSH/USB reachability "
        "(keeps boards from being silently dropped)."
    ),
    "gatemate_reachable": (
        "BARE BOOLs -- per-board SSH/USB reachability "
        "(keeps boards from being silently dropped)."
    ),
    "polarfire_reachable": (
        "BARE BOOLs -- per-board SSH/USB reachability "
        "(keeps boards from being silently dropped)."
    ),
    "per_board_next_step": (
        "The next concrete forward step per reachable board "
        "(KV260 toward terminal per north-star \u00a73)."
    ),
    "per_board_duration_s": (
        "DISTINCT wall-clock per board -- identical durations is a "
        "TAUTOLOGY flag (exp3866)."
    ),
    "preconditions_checked": (
        "list of {resource, available} -- records WHICH checks ran "
        "(pre-empts the fabrication mode)."
    ),
    "honest_verdict": "Terminal-prefix verdict + hardware_smoke substrate.",
    "duration_s": "Terminal-prefix verdict + hardware_smoke substrate.",
    "inference_substrate": "Terminal-prefix verdict + hardware_smoke substrate.",
}

run_command = _previous.run_command
payload_checksum = _previous.payload_checksum


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4017 artifact from live or injected board command results."""
    artifact = _previous.build_artifact(
        repo_root=repo_root,
        command_runner=command_runner,
        clock=clock,
    )
    artifact.update(
        schema=SCHEMA,
        experiment=EXPERIMENT_ID,
        spec_refs=list(SPEC_REFS),
        random_seed=RANDOM_SEED,
        field_principles=dict(FIELD_PRINCIPLES),
    )
    artifact["per_board_next_step"]["kv260"] = kv260_next_step(
        bool(artifact["kv260_reachable"]),
        artifact.get("kv260_loaded_overlay"),
        list(artifact.get("kv260_uio_devices") or []),
    )
    artifact["honest_verdict"] = honest_verdict(
        kv260_reachable=bool(artifact["kv260_reachable"]),
        gatemate_reachable=bool(artifact["gatemate_reachable"]),
        polarfire_reachable=bool(artifact["polarfire_reachable"]),
        kv260_state=str(artifact["kv260_state"]),
        gatemate_state=str(artifact["gatemate_state"]),
        polarfire_state=str(artifact["polarfire_state"]),
    )
    artifact["reproducibility_checksum"] = ""
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def kv260_next_step(reachable: bool, overlay: str | None, uio_devices: list[str]) -> str:
    if not reachable:
        return "blocked_kv260_unreachable"
    if overlay and uio_devices:
        return "kv260_forward_step_run_terminal_overlay_latency_smoke"
    if overlay:
        return "kv260_forward_step_restore_uio_binding_then_terminal_latency_smoke"
    return "kv260_forward_step_load_terminal_overlay_per_north_star_section_3"


def honest_verdict(
    *,
    kv260_reachable: bool,
    gatemate_reachable: bool,
    polarfire_reachable: bool,
    kv260_state: str,
    gatemate_state: str,
    polarfire_state: str,
) -> str:
    if not (kv260_reachable or gatemate_reachable or polarfire_reachable):
        return "blocked_all_boards_unreachable"
    return (
        "complete: hardware_continuity_4017_"
        f"kv{_hardware_base.state_token(kv260_state)}_"
        f"gm{_hardware_base.state_token(gatemate_state)}_"
        f"pf{_hardware_base.state_token(polarfire_state)}"
    )


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema") != SCHEMA:
        raise ValueError("schema must identify the Exp 4017 continuity artifact")
    if artifact.get("experiment") != EXPERIMENT_ID:
        raise ValueError("experiment must be 4017")
    if artifact.get("spec_refs") != SPEC_REFS:
        raise ValueError("spec_refs must reference REQ-HW-4017 and SCENARIO-HW-4017")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 4017")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):
        raise ValueError("field_principles must be a dict")
    missing_principles = set(REQUIRED_ARTIFACT_FIELDS) - set(principles)
    if missing_principles:
        raise ValueError(f"field_principles missing required fields: {sorted(missing_principles)}")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if principles[field] != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} principle must match the Exp 4017 contract")
        value = artifact[field]
        if isinstance(value, dict) and set(value) == {"value", "principle"}:
            raise ValueError(f"{field} must remain a bare value, not a principle wrapper")
    for field in ("kv260_reachable", "gatemate_reachable", "polarfire_reachable"):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be bool")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be hardware_smoke")
    if artifact.get("fabric_acceleration_claimed") is not False:
        raise ValueError("fabric_acceleration_claimed must be false")
    _hardware_base.validate_durations(artifact)
    _hardware_base.validate_next_steps(artifact)
    validate_kv260_terminal_next_step(artifact)
    _hardware_base.validate_preconditions(artifact)
    _hardware_base.validate_verdict(artifact)
    text = json.dumps(artifact, sort_keys=True, default=str).lower()
    if "/dev/mmcblk" in text:
        raise ValueError("artifact contains forbidden KV260 host storage marker")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum does not match artifact content")


def validate_kv260_terminal_next_step(artifact: dict[str, Any]) -> None:
    if artifact.get("kv260_reachable") and "terminal" not in str(
        artifact["per_board_next_step"]["kv260"]
    ):
        raise ValueError("reachable KV260 next step must point toward terminal confirmation")


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def main() -> None:  # pragma: no cover - CLI wrapper
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"kv260_reachable: {artifact['kv260_reachable']}")
    print(f"gatemate_reachable: {artifact['gatemate_reachable']}")
    print(f"polarfire_reachable: {artifact['polarfire_reachable']}")
    print(f"duration_s: {artifact['duration_s']}")
    print(f"per_board_duration_s: {artifact['per_board_duration_s']}")


if __name__ == "__main__":  # pragma: no cover
    main()
