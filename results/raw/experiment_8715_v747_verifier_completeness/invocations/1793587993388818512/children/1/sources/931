"""Exp 4529 hardware-continuity audit across attached boards.

Spec refs: REQ-HW-4529, SCENARIO-HW-4529.

This run is deliberately a board-state audit, not a bring-up task. It checks
KV260 only over SSH, checks GateMate through DirtyJTAG detect, checks PolarFire
over SSH, and records a small state transcript for every board that is
reachable. A board that is down stays visible as its own blocked row instead of
turning the whole audit into a failure.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

from carnot import experiment_4439_hardware_continuity as _base


EXPERIMENT_ID = 4529
SCHEMA = "carnot.hardware_continuity_state_audit.v1"
SPEC_REFS = ["REQ-HW-4529", "SCENARIO-HW-4529"]
OUTPUT_REL_PATH = Path("results") / "experiment_4529_hardware_continuity.json"
RANDOM_SEED = 4529
INFERENCE_SUBSTRATE = "hardware_smoke"
BOARD_NAMES = ("kv260", "gatemate", "polarfire")

KV260_REACHABILITY_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
GATEMATE_REACHABILITY_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
POLARFIRE_REACHABILITY_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)
KV260_STATE_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "sudo -n xmutil listapps",
)
POLARFIRE_STATE_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "uptime",
)

REQUIRED_OPERATOR_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "per_board_status",
    "preconditions_checked",
)
REQUIRED_ARTIFACT_FIELDS = (
    *REQUIRED_OPERATOR_FIELDS,
    "per_board_reachability",
    "reachable_board_count",
    "field_principles",
    "spec_refs",
    "random_seed",
    "duration_s",
    "fabric_acceleration_claimed",
    "speedup_claim_made",
    "bitstream_build_attempted",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; complete: hardware_continuity_audit_<n>_boards_reachable "
        "(blocked_<board> per-board is honest, not terminal failure)."
    ),
    "inference_substrate": (
        "hardware_smoke -- SSH/USB board checks, per-board duration floors."
    ),
    "per_board_status": (
        "each board's reachability + state -- keeps every attached board visible "
        "in the milestone retro (the forget-pattern guard)."
    ),
    "preconditions_checked": (
        "records WHICH board resources were verified; SSH-not-SD-card for KV260."
    ),
}

BLOCKED_STATUSES = {
    "kv260": "blocked_kv260_ssh_unreachable",
    "gatemate": "blocked_gatemate_usb_undetected",
    "polarfire": "blocked_polarfire_ssh_timeout",
}
REACHABLE_STATUSES = {
    "kv260": "kv260_reachable_state_recorded",
    "gatemate": "gatemate_reachable_idcode_recorded",
    "polarfire": "polarfire_reachable_state_recorded",
}

CommandProbe = _base.CommandProbe
CommandRunner = _base.CommandRunner
Clock = _base.Clock

run_command = _base.run_command
prepend_oss_cad_suite = _base.prepend_oss_cad_suite
command_to_string = _base.command_to_string
payload_checksum = _base.payload_checksum


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4529 artifact from live or injected hardware probes."""
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
        "duration_s": _round_duration(clock() - started),
        "fabric_acceleration_claimed": False,
        "speedup_claim_made": False,
        "bitstream_build_attempted": False,
        "reproducibility_checksum": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def check_preconditions(command_runner: CommandRunner) -> list[dict[str, Any]]:
    """Run exactly the three independent reachability checks from REQ-HW-4529."""
    specs = (
        ("kv260_ssh", KV260_REACHABILITY_COMMAND, _exit_zero, 10.0),
        ("gatemate_usb_detect", GATEMATE_REACHABILITY_COMMAND, _gatemate_detected, 30.0),
        ("polarfire_ssh", POLARFIRE_REACHABILITY_COMMAND, _exit_zero, 10.0),
    )
    return [
        _precondition_entry(resource, probe, predicate(probe))
        for resource, command, predicate, timeout_s in specs
        for probe in (command_runner(command, timeout_s),)
    ]


def capture_board_states(
    command_runner: CommandRunner,
    preconditions_by_resource: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Capture state only after the matching board reachability check succeeds."""
    records: dict[str, dict[str, Any]] = {}
    if preconditions_by_resource["kv260_ssh"]["available"]:
        records["kv260"] = _state_record_from_probe(
            "xmutil_listapps",
            command_runner(KV260_STATE_COMMAND, 20.0),
        )
    else:
        records["kv260"] = _blocked_state_record(BLOCKED_STATUSES["kv260"])
    if preconditions_by_resource["gatemate_usb_detect"]["available"]:
        records["gatemate"] = _gatemate_state_record(
            preconditions_by_resource["gatemate_usb_detect"]
        )
    else:
        records["gatemate"] = _blocked_state_record(BLOCKED_STATUSES["gatemate"])
    if preconditions_by_resource["polarfire_ssh"]["available"]:
        records["polarfire"] = _state_record_from_probe(
            "uptime",
            command_runner(POLARFIRE_STATE_COMMAND, 20.0),
        )
    else:
        records["polarfire"] = _blocked_state_record(BLOCKED_STATUSES["polarfire"])
    return records


def build_per_board_status(
    preconditions_by_resource: dict[str, dict[str, Any]],
    state_records: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Combine reachability and board-state transcripts into one row per board."""
    specs = (
        ("kv260", "kv260_ssh"),
        ("gatemate", "gatemate_usb_detect"),
        ("polarfire", "polarfire_ssh"),
    )
    return {
        board: _board_record(board, preconditions_by_resource[resource], state_records[board])
        for board, resource in specs
    }


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return the terminal board-count verdict required by this audit task."""
    return f"complete: hardware_continuity_audit_{artifact['reachable_board_count']}_boards_reachable"


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4529 and write `results/experiment_4529_hardware_continuity.json`."""
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
    """Reject artifacts that drift from the Exp 4529 audit-only contract."""
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4529")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4529")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4529")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4529")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _validate_operator_fields(artifact)
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    _require("mmcblk" not in encoded, "forbidden retired host-storage marker")
    _validate_preconditions(artifact)
    _validate_per_board_status(artifact)
    reachability = artifact["per_board_reachability"]
    reachable_count = sum(1 for reachable in reachability.values() if reachable)
    _require(
        artifact.get("reachable_board_count") == reachable_count,
        "reachable_board_count mismatch",
    )
    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be hardware_smoke",
    )
    _require(artifact.get("fabric_acceleration_claimed") is False, "no fabric claim")
    _require(artifact.get("speedup_claim_made") is False, "no speedup claim")
    _require(artifact.get("bitstream_build_attempted") is False, "no bitstream build")
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "stale honest_verdict")
    _require(artifact.get("honest_verdict", "").startswith("complete:"), "honest_verdict prefix")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _precondition_entry(resource: str, probe: CommandProbe, available: bool) -> dict[str, Any]:
    return {
        "resource": resource,
        "available": bool(available),
        "command": command_to_string(probe.command),
        "exit_code": int(probe.exit_code),
        "duration_s": _round_duration(probe.duration_s),
        "observed": _observed(probe),
    }


def _board_record(
    board: str,
    precondition: dict[str, Any],
    state_record: dict[str, Any],
) -> dict[str, Any]:
    reachable = bool(precondition["available"])
    status = REACHABLE_STATUSES[board] if reachable else BLOCKED_STATUSES[board]
    state_duration = float(state_record.get("duration_s", 0.0)) if reachable else 0.0
    return {
        "board": board,
        "reachable": reachable,
        "status": status,
        "precondition_resource": precondition["resource"],
        "precondition_command": precondition["command"],
        "reachability_transcript": {
            "exit_code": precondition["exit_code"],
            "duration_s": precondition["duration_s"],
            "observed": precondition["observed"],
        },
        "state": state_record,
        "duration_s": _round_duration(float(precondition["duration_s"]) + state_duration),
    }


def _state_record_from_probe(state_type: str, probe: CommandProbe) -> dict[str, Any]:
    return {
        "captured": True,
        "state_type": state_type,
        "command": command_to_string(probe.command),
        "exit_code": int(probe.exit_code),
        "duration_s": _round_duration(probe.duration_s),
        "observed": _observed(probe),
    }


def _gatemate_state_record(precondition: dict[str, Any]) -> dict[str, Any]:
    observed = str(precondition["observed"])
    return {
        "captured": True,
        "state_type": "gm1ax_idcode",
        "command": precondition["command"],
        "exit_code": int(precondition["exit_code"]),
        "duration_s": 0.0,
        "observed": observed,
        "idcode": _idcode_from_observed(observed),
    }


def _blocked_state_record(reason: str) -> dict[str, Any]:
    return {"captured": False, "reason": reason}


def _exit_zero(probe: CommandProbe) -> bool:
    return probe.exit_code == 0


def _gatemate_detected(probe: CommandProbe) -> bool:
    text = _observed(probe).lower()
    return probe.exit_code == 0 and "idcode" in text and "gm1a" in text


def _idcode_from_observed(observed: str) -> str:
    match = re.search(r"0x[0-9a-fA-F]+", observed)
    return match.group(0) if match else "gm1ax_idcode_observed"


def _validate_operator_fields(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    _require(isinstance(principles, dict), "field_principles must be a dict")
    for field in REQUIRED_OPERATOR_FIELDS:
        _require(principles.get(field) == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
        _require(
            not (isinstance(artifact[field], dict) and set(artifact[field]) == {"value", "principle"}),
            f"{field} must remain a bare value",
        )


def _validate_preconditions(artifact: dict[str, Any]) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(isinstance(preconditions, list), "preconditions_checked must be a list")
    _require(len(preconditions) == 3, "exactly three preconditions required")
    expected_commands = {
        "kv260_ssh": command_to_string(KV260_REACHABILITY_COMMAND),
        "gatemate_usb_detect": command_to_string(GATEMATE_REACHABILITY_COMMAND),
        "polarfire_ssh": command_to_string(POLARFIRE_REACHABILITY_COMMAND),
    }
    resources = [entry.get("resource") for entry in preconditions if isinstance(entry, dict)]
    _require(
        resources == ["kv260_ssh", "gatemate_usb_detect", "polarfire_ssh"],
        "precondition resources mismatch",
    )
    for entry in preconditions:
        _require(isinstance(entry, dict), "precondition entries must be dicts")
        _require(entry.get("resource") in expected_commands, "unknown precondition resource")
        _require(isinstance(entry.get("available"), bool), "available must be bool")
        _require(entry.get("command") == expected_commands[entry["resource"]], "command mismatch")
        _require(isinstance(entry.get("exit_code"), int), "exit_code must be int")
        _require(float(entry.get("duration_s", -1.0)) >= 0.0, "duration_s must be non-negative")
        _require(isinstance(entry.get("observed"), str) and entry["observed"], "observed required")


def _validate_per_board_status(artifact: dict[str, Any]) -> None:
    statuses = artifact.get("per_board_status")
    reachability = artifact.get("per_board_reachability")
    _require(isinstance(statuses, dict), "per_board_status must be a dict")
    _require(isinstance(reachability, dict), "per_board_reachability must be a dict")
    _require(set(statuses) == set(BOARD_NAMES), "per_board_status keys mismatch")
    _require(set(reachability) == set(BOARD_NAMES), "per_board_reachability keys mismatch")
    by_resource = {entry["resource"]: entry for entry in artifact["preconditions_checked"]}
    expected_resources = {
        "kv260": "kv260_ssh",
        "gatemate": "gatemate_usb_detect",
        "polarfire": "polarfire_ssh",
    }
    for board in BOARD_NAMES:
        record = statuses[board]
        _require(isinstance(record, dict), "board status entries must be dicts")
        _require(record.get("board") == board, "board name mismatch")
        _require(isinstance(record.get("reachable"), bool), "reachable must be bool")
        _require(reachability[board] is record["reachable"], "reachability mismatch")
        _require(record.get("precondition_resource") == expected_resources[board], "resource mismatch")
        precondition = by_resource[record["precondition_resource"]]
        _require(record.get("precondition_command") == precondition["command"], "command mismatch")
        _require(isinstance(record.get("reachability_transcript"), dict), "transcript required")
        _require(float(record.get("duration_s", -1.0)) >= 0.0, "board duration required")
        state = record.get("state")
        _require(isinstance(state, dict), "state must be a dict")
        if record["reachable"]:
            _require(record.get("status") == REACHABLE_STATUSES[board], "reachable status mismatch")
            _require(state.get("captured") is True, "reachable state must be captured")
            _require(state.get("state_type") in _expected_state_types(board), "state type mismatch")
            _require(isinstance(state.get("observed"), str) and state["observed"], "state observed")
        else:
            _require(record.get("status") == BLOCKED_STATUSES[board], "blocked status mismatch")
            _require(state == _blocked_state_record(BLOCKED_STATUSES[board]), "blocked state mismatch")


def _expected_state_types(board: str) -> tuple[str, ...]:
    return {
        "kv260": ("xmutil_listapps",),
        "gatemate": ("gm1ax_idcode",),
        "polarfire": ("uptime",),
    }[board]


def _observed(probe: CommandProbe) -> str:
    text = probe.combined_output.strip()
    return text[:500] if text else f"returncode={probe.exit_code}"


def _round_duration(value: float) -> float:
    return round(max(float(value), 0.0), 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover - live hardware entrypoint.
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
