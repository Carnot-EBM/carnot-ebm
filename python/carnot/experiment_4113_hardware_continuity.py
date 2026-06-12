"""Exp 4113 per-board hardware continuity status artifact.

This run preserves the Hardware-Task Continuity Discipline after Exp 4104:
every attached board receives a reachability record, a board-local wall-clock
timer, and an honest next step. KV260 is already terminal, so the only allowed
KV260 action is SSH confirmation. GateMate and PolarFire remain non-terminal.

Spec refs: REQ-HW-4113, SCENARIO-HW-4113.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4104_hardware_continuity as _previous


EXPERIMENT_ID = 4113
SCHEMA = "carnot.hardware_continuity_per_board_status.v2"
SPEC_REFS = ["REQ-HW-4113", "SCENARIO-HW-4113"]
OUTPUT_REL_PATH = Path("results") / "experiment_4113_hardware_continuity.json"
RANDOM_SEED = 4113
INFERENCE_SUBSTRATE = _previous.INFERENCE_SUBSTRATE
BOARD_NAMES = _previous.BOARD_NAMES

KV260_SSH_PRECONDITION = _previous.KV260_SSH_PRECONDITION
GATEMATE_DETECT_COMMAND = _previous.GATEMATE_DETECT_COMMAND
POLARFIRE_SSH_PRECONDITION = _previous.POLARFIRE_SSH_PRECONDITION
GATEMATE_FLASH_BOARD = _previous.GATEMATE_FLASH_BOARD

REQUIRED_ARTIFACT_FIELDS = _previous.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = dict(_previous.FIELD_PRINCIPLES)

CommandProbe = _previous.CommandProbe
StepOutcome = _previous.StepOutcome
CommandRunner = _previous.CommandRunner
Clock = _previous.Clock
StepRunner = _previous.StepRunner

run_command = _previous.run_command
prepend_oss_cad_suite = _previous.prepend_oss_cad_suite
payload_checksum = _previous.payload_checksum
state_token = _previous.state_token


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = _previous._base.time.perf_counter,
    gatemate_step_runner: StepRunner | None = None,
    polarfire_step_runner: StepRunner | None = None,
) -> dict[str, Any]:
    """Build the Exp 4113 artifact from live or injected board probes."""
    root = Path(repo_root)
    artifact = _previous.build_artifact(
        repo_root=root,
        command_runner=command_runner,
        clock=clock,
        gatemate_step_runner=gatemate_step_runner,
        polarfire_step_runner=polarfire_step_runner,
    )
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "random_seed": RANDOM_SEED,
            "source_context": _source_context(root),
        }
    )
    _add_distinct_timer_ids(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return a terminal-prefixed verdict that keeps all board statuses visible."""
    return (
        "complete: hardware_continuity_4113_"
        f"gatemate_{state_token(str(artifact['gatemate_step_taken']))}_"
        f"polarfire_{state_token(str(artifact['polarfire_step_taken']))}_"
        f"kv260_{state_token(str(artifact['per_board_terminal_state']['kv260']))}_"
        "ssh_usb_detect_only"
    )


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = _previous._base.time.perf_counter,
) -> Path:
    """Run Exp 4113 and write `results/experiment_4113_hardware_continuity.json`."""
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4113")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4113")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4113")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4113")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _validate_principles(artifact)
    _validate_preconditions(artifact)
    _validate_per_board_status(artifact)
    _validate_source_context(artifact)
    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be hardware_smoke",
    )
    _require(
        "mmcblk" not in json.dumps(artifact, sort_keys=True, default=str).lower(),
        "SD-card precondition marker is forbidden",
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


def _add_distinct_timer_ids(artifact: dict[str, Any]) -> None:
    statuses = artifact["per_board_status"]
    for board in BOARD_NAMES:
        statuses[board]["timer_id"] = f"{board}_precondition_plus_forward_step_wall_clock"


def _source_context(repo_root: Path) -> dict[str, Any]:
    prior_path = repo_root / "results" / "experiment_4104_hardware_continuity.json"
    prior_payload = _read_json(prior_path)
    gate_step = prior_payload.get("gatemate_step", {}) if isinstance(prior_payload, dict) else {}
    return {
        "previous_experiment": 4104,
        "most_recent_hardware_continuity_artifact": str(prior_path),
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_per_board_reachability": prior_payload.get("per_board_reachability"),
        "previous_gatemate_step_taken": prior_payload.get("gatemate_step_taken"),
        "previous_gatemate_flash_exit_code": (
            gate_step.get("flash_exit_code") if isinstance(gate_step, dict) else None
        ),
        "previous_gatemate_flash_error": (
            gate_step.get("flash_error") if isinstance(gate_step, dict) else None
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt local artifact.
        return {}


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
    resources = [entry.get("resource") for entry in preconditions if isinstance(entry, dict)]
    _require(resources == ["kv260_ssh", "gatemate_jtag_detect", "polarfire_ssh"], "")
    for entry in preconditions:
        _require(isinstance(entry, dict), "precondition entries must be dicts")
        _require({"resource", "available", "command", "exit_code"} <= set(entry), "precondition missing keys")
        _require(isinstance(entry["available"], bool), "available must be bool")


def _validate_per_board_status(artifact: dict[str, Any]) -> None:
    statuses = artifact.get("per_board_status")
    _require(isinstance(statuses, dict), "per_board_status must be dict")
    _require(set(statuses) == set(BOARD_NAMES), "per_board_status must be keyed by all boards")
    timer_ids: list[str] = []
    for board in BOARD_NAMES:
        record = statuses[board]
        _require(isinstance(record, dict), "per_board_status entries must be dicts")
        _require(record.get("reachable") is artifact["per_board_reachability"][board], "reachable mismatch")
        _require(isinstance(record.get("status"), str) and record["status"], "missing status")
        _require(
            isinstance(record.get("next_concrete_step"), str) and record["next_concrete_step"],
            "missing next concrete step",
        )
        _require(float(record.get("duration_s", 0.0)) > 0.0, "duration_s must be positive")
        _require(record.get("precondition_resource"), "missing precondition resource")
        _require(record.get("precondition_command"), "missing precondition command")
        timer_id = record.get("timer_id")
        _require(
            isinstance(timer_id, str) and timer_id.startswith(f"{board}_"),
            "missing distinct timer_id",
        )
        timer_ids.append(timer_id)
        if not artifact["per_board_reachability"][board]:
            _require(record["status"] == f"blocked_{board}_unreachable", "blocked board status mismatch")
    _require(len(set(timer_ids)) == len(timer_ids), "timer_id values must be distinct")


def _validate_source_context(artifact: dict[str, Any]) -> None:
    source = artifact.get("source_context")
    _require(isinstance(source, dict), "source_context must be present")
    _require(source.get("previous_experiment") == 4104, "source_context must read Exp 4104")
    _require(
        str(source.get("most_recent_hardware_continuity_artifact", "")).endswith(
            "experiment_4104_hardware_continuity.json"
        ),
        "source_context must point at Exp 4104",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)  # pragma: no cover


def main() -> None:  # pragma: no cover
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"per_board_status: {artifact['per_board_status']}")


if __name__ == "__main__":  # pragma: no cover
    main()
