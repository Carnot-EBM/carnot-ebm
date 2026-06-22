"""Exp 4588 hardware-continuity audit across attached boards.

Spec refs: REQ-HW-4588, SCENARIO-HW-4588.

This module keeps the milestone hardware audit deliberately small: KV260 is
checked only through SSH, GateMate through DirtyJTAG USB detect, and PolarFire
through SSH. A reachable board gets one state transcript. An unreachable board
gets an honest blocked row, which preserves board visibility without fabricating
progress.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4576_hardware_continuity as _previous


EXPERIMENT_ID = 4588
SCHEMA = "carnot.hardware_continuity_state_audit.v6"
SPEC_REFS = ["REQ-HW-4588", "SCENARIO-HW-4588"]
OUTPUT_REL_PATH = Path("results") / "experiment_4588_hardware_continuity.json"
RANDOM_SEED = 4588

INFERENCE_SUBSTRATE = _previous.INFERENCE_SUBSTRATE
BOARD_NAMES = _previous.BOARD_NAMES
KV260_REACHABILITY_COMMAND = _previous.KV260_REACHABILITY_COMMAND
GATEMATE_REACHABILITY_COMMAND = _previous.GATEMATE_REACHABILITY_COMMAND
POLARFIRE_REACHABILITY_COMMAND = _previous.POLARFIRE_REACHABILITY_COMMAND
KV260_STATE_COMMAND = _previous.KV260_STATE_COMMAND
POLARFIRE_STATE_COMMAND = _previous.POLARFIRE_STATE_COMMAND

REQUIRED_OPERATOR_FIELDS = _previous.REQUIRED_OPERATOR_FIELDS
REQUIRED_ARTIFACT_FIELDS = _previous.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = dict(_previous.FIELD_PRINCIPLES)
BLOCKED_STATUSES = dict(_previous.BLOCKED_STATUSES)
REACHABLE_STATUSES = dict(_previous.REACHABLE_STATUSES)

CommandProbe = _previous.CommandProbe
CommandRunner = _previous.CommandRunner
Clock = _previous.Clock

run_command = _previous.run_command
prepend_oss_cad_suite = _previous.prepend_oss_cad_suite
command_to_string = _previous.command_to_string
payload_checksum = _previous.payload_checksum
check_preconditions = _previous.check_preconditions
build_per_board_status = _previous.build_per_board_status
honest_verdict = _previous.honest_verdict


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4588 artifact from live or injected hardware probes."""
    started = clock()
    preconditions = check_preconditions(command_runner)
    by_resource = {entry["resource"]: entry for entry in preconditions}
    state_records = capture_board_states(command_runner, by_resource)
    per_board_status = build_per_board_status(by_resource, state_records)
    reachability = {board: bool(per_board_status[board]["reachable"]) for board in BOARD_NAMES}
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "honest_verdict": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "per_board_reachability": reachability,
        "reachable_board_count": sum(1 for reachable in reachability.values() if reachable),
        "per_board_status": per_board_status,
        "duration_s": _previous._previous._state_audit._round_duration(clock() - started),
        "fabric_acceleration_claimed": False,
        "speedup_claim_made": False,
        "bitstream_build_attempted": False,
        "reproducibility_checksum": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def capture_board_states(
    command_runner: CommandRunner,
    preconditions_by_resource: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Capture board state only after that board's reachability check succeeds."""
    records: dict[str, dict[str, Any]] = {}
    if preconditions_by_resource["kv260_ssh"]["available"]:
        records["kv260"] = _previous._previous._state_audit._state_record_from_probe(
            "xmutil_listapps",
            command_runner(KV260_STATE_COMMAND, 20.0),
        )
    else:
        records["kv260"] = _previous._previous._state_audit._blocked_state_record(
            BLOCKED_STATUSES["kv260"]
        )
    if preconditions_by_resource["gatemate_usb_detect"]["available"]:
        records["gatemate"] = _previous._previous._state_audit._gatemate_state_record(
            preconditions_by_resource["gatemate_usb_detect"]
        )
    else:
        records["gatemate"] = _previous._previous._state_audit._blocked_state_record(
            BLOCKED_STATUSES["gatemate"]
        )
    if preconditions_by_resource["polarfire_ssh"]["available"]:
        records["polarfire"] = _previous._previous._state_audit._state_record_from_probe(
            "uptime",
            command_runner(POLARFIRE_STATE_COMMAND, 20.0),
        )
    else:
        records["polarfire"] = _previous._previous._state_audit._blocked_state_record(
            BLOCKED_STATUSES["polarfire"]
        )
    return records


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4588 and write `results/experiment_4588_hardware_continuity.json`."""
    prepend_oss_cad_suite()
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    """Write stable JSON so board-state audit diffs stay easy to inspect."""
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Reject artifacts that drift from the Exp 4588 audit-only contract."""
    _previous._previous._state_audit._require(
        artifact.get("schema") == SCHEMA,
        "schema must identify Exp 4588",
    )
    _previous._previous._state_audit._require(
        artifact.get("experiment") == EXPERIMENT_ID,
        "experiment must be 4588",
    )
    _previous._previous._state_audit._require(
        artifact.get("spec_refs") == SPEC_REFS,
        "spec_refs must name REQ/SCENARIO 4588",
    )
    _previous._previous._state_audit._require(
        artifact.get("random_seed") == RANDOM_SEED,
        "random_seed must be 4588",
    )
    _previous._previous._state_audit._require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "bad checksum",
    )
    _previous.validate_artifact(_as_4576_metadata(artifact))
    _validate_4588_state_commands(artifact)


def _validate_4588_state_commands(artifact: dict[str, Any]) -> None:
    statuses = artifact["per_board_status"]
    if statuses["kv260"]["reachable"]:
        _previous._previous._state_audit._require(
            statuses["kv260"]["state"].get("command") == command_to_string(KV260_STATE_COMMAND),
            "kv260 state command mismatch",
        )
    if statuses["polarfire"]["reachable"]:
        _previous._previous._state_audit._require(
            statuses["polarfire"]["state"].get("command")
            == command_to_string(POLARFIRE_STATE_COMMAND),
            "polarfire state command mismatch",
        )


def _as_4576_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    previous_artifact = dict(artifact)
    previous_artifact.update(
        {
            "schema": _previous.SCHEMA,
            "experiment": _previous.EXPERIMENT_ID,
            "spec_refs": list(_previous.SPEC_REFS),
            "random_seed": _previous.RANDOM_SEED,
        }
    )
    previous_artifact["reproducibility_checksum"] = payload_checksum(previous_artifact)
    return previous_artifact


def main() -> None:  # pragma: no cover - live hardware entrypoint.
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
