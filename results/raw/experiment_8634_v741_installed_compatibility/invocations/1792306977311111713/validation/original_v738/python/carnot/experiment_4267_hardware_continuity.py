"""Exp 4267 opportunistic per-board hardware continuity status artifact.

This run keeps KV260, PolarFire, and GateMate visible without turning any one
board into a milestone blocker. KV260 remains terminal and SSH-only: the
precondition is board SSH reachability, followed by `xmutil listapps` only when
SSH succeeds. PolarFire receives an opportunistic hash-verified CPU dispatch
smoke when reachable. GateMate records the DirtyJTAG IDCODE when detected, or an
honest blocked status when not detected.

Spec refs: REQ-HW-4267, SCENARIO-HW-4267.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

from carnot import experiment_4064_hardware_continuity as _base


EXPERIMENT_ID = 4267
SCHEMA = "carnot.hardware_continuity_per_board_status.v16"
SPEC_REFS = ["REQ-HW-4267", "SCENARIO-HW-4267"]
OUTPUT_REL_PATH = Path("results") / "experiment_4267_hardware_continuity.json"
RANDOM_SEED = 4267
INFERENCE_SUBSTRATE = _base.INFERENCE_SUBSTRATE
BOARD_NAMES = ("kv260", "polarfire", "gatemate")

KV260_SSH_PRECONDITION = _base.KV260_SSH_PRECONDITION
POLARFIRE_SSH_PRECONDITION = ("ssh", "-o", "ConnectTimeout=5", "polarfire", "true")
GATEMATE_DETECT_COMMAND = _base.GATEMATE_DETECT_COMMAND
KV260_LISTAPPS_COMMAND = ("ssh", "kria", "xmutil listapps")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "per_board_status",
    "preconditions_checked",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A per-board reachability report (including honest "
        "blocked_<board>) is COMPLETE; no board blocks the milestone."
    ),
    "per_board_status": (
        "Per-board reachability + any opportunistic result -- keeps the boards "
        "visible (continuity) without blocking on operator hardware actions."
    ),
    "preconditions_checked": (
        "Records the SSH/USB reachability checks (KV260 via SSH NOT SD-card); "
        "pre-empts the wrong-mechanism + fabrication modes."
    ),
    "reproducibility_checksum": "Hash of the per-board probe outputs; auditable.",
}

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


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    polarfire_step_runner: StepRunner | None = None,
) -> dict[str, Any]:
    """Build the Exp 4267 artifact from live or injected board probes."""
    root = Path(repo_root)
    started = clock()
    preconditions, reachability, precondition_durations = check_preconditions(command_runner)
    per_board_duration_s = dict(precondition_durations)

    kv260_probe: CommandProbe | None = None
    if reachability["kv260"]:
        kv260_probe = command_runner(KV260_LISTAPPS_COMMAND, 30.0)
        per_board_duration_s["kv260"] = _round_duration(
            per_board_duration_s["kv260"] + kv260_probe.duration_s
        )

    polarfire_step = _blocked_step("polarfire")
    polarfire_step_taken = "blocked_polarfire_unreachable"
    polarfire_terminal_state = "blocked_polarfire_unreachable"
    if reachability["polarfire"]:
        polar_runner = polarfire_step_runner or _base.run_polarfire_forward_step
        polarfire_outcome = polar_runner(
            repo_root=root,
            command_runner=command_runner,
            clock=clock,
        )
        polarfire_step = dict(polarfire_outcome.details)
        polarfire_step_taken = polarfire_outcome.step_taken
        polarfire_terminal_state = polarfire_outcome.terminal_state
        per_board_duration_s["polarfire"] = _round_duration(
            per_board_duration_s["polarfire"] + polarfire_outcome.duration_s
        )

    gate_idcode = _extract_gatemate_idcode(_precondition_for("gatemate_jtag_detect", preconditions))
    gatemate_step_taken = "gatemate_idcode_detected" if gate_idcode else "blocked_gatemate_unreachable"
    gatemate_terminal_state = (
        "reachable_idcode_detected_opportunistic_only"
        if gate_idcode
        else "blocked_gatemate_unreachable"
    )
    gatemate_step = _gatemate_step(gate_idcode)

    kv260_status, kv260_terminal_state = _kv260_status_and_terminal(kv260_probe, reachability["kv260"])
    per_board_terminal_state = {
        "kv260": kv260_terminal_state,
        "polarfire": polarfire_terminal_state,
        "gatemate": gatemate_terminal_state,
    }
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "honest_verdict": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_context": _source_context(root),
        "preconditions_checked": preconditions,
        "per_board_reachability": dict(reachability),
        "kv260_reachable": bool(reachability["kv260"]),
        "polarfire_reachable": bool(reachability["polarfire"]),
        "gatemate_reachable": bool(reachability["gatemate"]),
        "kv260_terminal_confirmed": kv260_status == "kv260_terminal_confirmed_ssh_only",
        "kv260_step_taken": kv260_status,
        "polarfire_step_taken": polarfire_step_taken,
        "gatemate_step_taken": gatemate_step_taken,
        "polarfire_step": polarfire_step,
        "gatemate_step": gatemate_step,
        "per_board_terminal_state": per_board_terminal_state,
        "per_board_next_step": _per_board_next_step(kv260_status, polarfire_step, gatemate_step),
        "per_board_duration_s": {
            board: _round_duration(per_board_duration_s[board]) for board in BOARD_NAMES
        },
        "duration_s": _elapsed(started, clock),
        "fabric_acceleration_claimed": False,
        "speedup_claim_made": False,
        "per_board_status": {},
        "reproducibility_checksum": "",
    }
    artifact["per_board_status"] = _build_per_board_status(artifact, kv260_probe, gate_idcode)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def check_preconditions(
    command_runner: CommandRunner,
) -> tuple[list[dict[str, Any]], dict[str, bool], dict[str, float]]:
    """Run the SSH/USB reachability checks before opportunistic board work."""
    specs = (
        ("kv260", "kv260_ssh", KV260_SSH_PRECONDITION, _ssh_available, 10.0),
        ("polarfire", "polarfire_ssh", POLARFIRE_SSH_PRECONDITION, _ssh_available, 10.0),
        ("gatemate", "gatemate_jtag_detect", GATEMATE_DETECT_COMMAND, _gatemate_available, 30.0),
    )
    entries: list[dict[str, Any]] = []
    reachability: dict[str, bool] = {}
    durations: dict[str, float] = {}
    for board, resource, command, predicate, timeout_s in specs:
        probe = command_runner(command, timeout_s)
        available = bool(predicate(probe))
        entries.append(_precondition_entry(resource, probe, available))
        reachability[board] = available
        durations[board] = _round_duration(probe.duration_s)
    return entries, reachability, durations


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return a terminal-prefixed verdict with all board states visible."""
    return (
        "complete: hardware_continuity_4267_"
        f"kv260_{state_token(str(artifact['kv260_step_taken']))}_"
        f"polarfire_{state_token(str(artifact['polarfire_step_taken']))}_"
        f"gatemate_{state_token(str(artifact['gatemate_step_taken']))}_"
        "ssh_usb_reachability_only"
    )


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    polarfire_step_runner: StepRunner | None = None,
) -> Path:
    """Run Exp 4267 and write `results/experiment_4267_hardware_continuity.json`."""
    prepend_oss_cad_suite()
    artifact = build_artifact(
        repo_root=repo_root,
        command_runner=command_runner,
        clock=clock,
        polarfire_step_runner=polarfire_step_runner,
    )
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4267")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4267")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4267")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4267")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _validate_principles(artifact)
    _require(
        "mmcblk" not in json.dumps(artifact, sort_keys=True, default=str).lower(),
        "SD-card precondition marker is forbidden",
    )
    _validate_preconditions(artifact)
    _validate_per_board_status(artifact)
    _validate_source_context(artifact)
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


def _build_per_board_status(
    artifact: dict[str, Any],
    kv260_probe: CommandProbe | None,
    gate_idcode: str | None,
) -> dict[str, dict[str, Any]]:
    preconditions = _preconditions_by_board(artifact["preconditions_checked"])
    statuses: dict[str, dict[str, Any]] = {}
    for board in BOARD_NAMES:
        precondition = preconditions[board]
        record: dict[str, Any] = {
            "reachable": bool(artifact["per_board_reachability"][board]),
            "status": _board_status(board, artifact),
            "terminal_state": artifact["per_board_terminal_state"][board],
            "next_concrete_step": artifact["per_board_next_step"][board],
            "precondition_resource": precondition["resource"],
            "precondition_command": precondition["command"],
            "precondition_exit_code": precondition["exit_code"],
            "precondition_available": precondition["available"],
            "duration_s": artifact["per_board_duration_s"][board],
            "timer_id": f"{board}_precondition_plus_opportunistic_check_wall_clock",
            "timer_scope": "precondition_plus_opportunistic_check_wall_clock",
        }
        if board == "kv260":
            record.update(_kv260_status_details(kv260_probe, bool(artifact["per_board_reachability"][board])))
        if board == "polarfire" and "result_hash_match" in artifact.get("polarfire_step", {}):
            record["hash_match"] = bool(artifact["polarfire_step"]["result_hash_match"])
            record["board_result_sha256"] = artifact["polarfire_step"].get("board_result_sha256")
            record["cpu_reference_sha256"] = artifact["polarfire_step"].get("cpu_reference_sha256")
        if board == "gatemate" and gate_idcode:
            record["idcode"] = gate_idcode
        statuses[board] = record
    return statuses


def _preconditions_by_board(preconditions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    resource_to_board = {
        "kv260_ssh": "kv260",
        "polarfire_ssh": "polarfire",
        "gatemate_jtag_detect": "gatemate",
    }
    return {
        resource_to_board[entry["resource"]]: entry
        for entry in preconditions
        if entry.get("resource") in resource_to_board
    }


def _board_status(board: str, artifact: dict[str, Any]) -> str:
    if board == "kv260":
        return str(artifact["kv260_step_taken"])
    if board == "polarfire":
        return str(artifact["polarfire_step_taken"])
    return str(artifact["gatemate_step_taken"])


def _per_board_next_step(
    kv260_status: str,
    polarfire_step: dict[str, Any],
    gatemate_step: dict[str, Any],
) -> dict[str, str]:
    return {
        "kv260": (
            "kv260_terminal_state_confirmed_via_ssh_xmutil_listapps"
            if kv260_status == "kv260_terminal_confirmed_ssh_only"
            else kv260_status
        ),
        "polarfire": str(polarfire_step.get("next_concrete_step") or polarfire_step.get("step")),
        "gatemate": str(gatemate_step.get("next_concrete_step") or gatemate_step.get("step")),
    }


def _kv260_status_and_terminal(probe: CommandProbe | None, reachable: bool) -> tuple[str, str]:
    if not reachable:
        return "blocked_kv260_unreachable", "blocked_kv260_unreachable"
    if probe is not None and probe.exit_code == 0:
        return "kv260_terminal_confirmed_ssh_only", "opportunistic_terminal_confirmed_ssh_only"
    code = probe.exit_code if probe is not None else 127
    return (
        f"kv260_terminal_xmutil_listapps_blocked_returncode_{code}",
        "reachable_xmutil_listapps_blocked_ssh_only",
    )


def _kv260_status_details(probe: CommandProbe | None, reachable: bool) -> dict[str, Any]:
    if not reachable:
        return {"ssh_only_terminal_status": "blocked_kv260_unreachable_ssh_only"}
    status = (
        "terminal_confirmed_via_xmutil_listapps_ssh_only"
        if probe is not None and probe.exit_code == 0
        else "ssh_reachable_xmutil_listapps_blocked_ssh_only"
    )
    return {
        "ssh_only_terminal_status": status,
        "xmutil_listapps_command": command_to_string(KV260_LISTAPPS_COMMAND),
        "xmutil_listapps_exit_code": probe.exit_code if probe is not None else 127,
        "xmutil_listapps_observed": _observed(probe) if probe is not None else "not_run",
        "xmutil_listapps_duration_s": _round_duration(probe.duration_s if probe is not None else 0.0),
    }


def _gatemate_step(idcode: str | None) -> dict[str, Any]:
    if idcode:
        return {
            "step": "record_gatemate_dirtyjtag_idcode",
            "idcode": idcode,
            "next_concrete_step": "gatemate_idcode_detected_opportunistic_only",
        }
    return {
        "step": "blocked_gatemate_unreachable",
        "blocker": "blocked_gatemate_unreachable",
        "next_concrete_step": (
            "Recover GM1Ax IDCODE visibility with `openFPGALoader -c dirtyJtag --detect`, "
            "then rerun the opportunistic GateMate continuity check."
        ),
    }


def _blocked_step(board: str) -> dict[str, Any]:
    return {
        "step": f"blocked_{board}_unreachable",
        "blocker": f"blocked_{board}_unreachable",
        "next_concrete_step": f"blocked_{board}_unreachable",
    }


def _precondition_for(resource: str, preconditions: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in preconditions:
        if entry["resource"] == resource:
            return entry
    return {}  # pragma: no cover - validate_preconditions catches malformed artifacts.


def _source_context(repo_root: Path) -> dict[str, Any]:
    prior_path = repo_root / "results" / "experiment_4253_hardware_continuity.json"
    prior_payload = _read_json(prior_path)
    return {
        "previous_experiment": 4253,
        "most_recent_hardware_continuity_artifact": str(prior_path),
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_per_board_reachability": prior_payload.get("per_board_reachability"),
        "previous_per_board_status": prior_payload.get("per_board_status"),
        "previous_gatemate_step_taken": prior_payload.get("gatemate_step_taken"),
        "previous_polarfire_step_taken": prior_payload.get("polarfire_step_taken"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}  # pragma: no cover - live script usually has the prior artifact.
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt local artifact.
        return {}


def _precondition_entry(resource: str, probe: CommandProbe, available: bool) -> dict[str, Any]:
    return {
        "resource": resource,
        "available": bool(available),
        "command": command_to_string(probe.command),
        "duration_s": _round_duration(probe.duration_s),
        "exit_code": probe.exit_code,
        "observed": _observed(probe),
    }


def _ssh_available(probe: CommandProbe) -> bool:
    return probe.exit_code == 0


def _gatemate_available(probe: CommandProbe) -> bool:
    return _extract_gatemate_idcode(_precondition_entry("gatemate_jtag_detect", probe, False)) is not None


def _extract_gatemate_idcode(entry: dict[str, Any]) -> str | None:
    text = str(entry.get("observed", ""))
    match = re.search(r"0x[0-9a-fA-F]{8}", text)
    if match and match.group(0).lower() == "0x20000001":
        return match.group(0).lower()
    return None


def _observed(probe: CommandProbe) -> str:
    text = probe.combined_output.strip()
    return text if text else f"returncode={probe.exit_code}"


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
        "polarfire_ssh": command_to_string(POLARFIRE_SSH_PRECONDITION),
        "gatemate_jtag_detect": command_to_string(GATEMATE_DETECT_COMMAND),
    }
    resources = [entry.get("resource") for entry in preconditions if isinstance(entry, dict)]
    _require(resources == ["kv260_ssh", "polarfire_ssh", "gatemate_jtag_detect"], "")
    for entry in preconditions:
        _require(isinstance(entry, dict), "precondition entries must be dicts")
        _require({"resource", "available", "command", "exit_code"} <= set(entry), "precondition missing keys")
        _require(isinstance(entry["available"], bool), "available must be bool")
        _require(entry["command"] == expected[entry["resource"]], "invalid precondition command")


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
        _require(isinstance(timer_id, str) and timer_id.startswith(f"{board}_"), "missing timer_id")
        timer_ids.append(timer_id)
        if not artifact["per_board_reachability"][board]:
            _require(record["status"] == f"blocked_{board}_unreachable", "blocked board status mismatch")
    _require(len(set(timer_ids)) == len(timer_ids), "timer_id values must be distinct")


def _validate_source_context(artifact: dict[str, Any]) -> None:
    source = artifact.get("source_context")
    _require(isinstance(source, dict), "source_context must be present")
    _require(source.get("previous_experiment") == 4253, "source_context must read Exp 4253")
    _require(
        str(source.get("most_recent_hardware_continuity_artifact", "")).endswith(
            "experiment_4253_hardware_continuity.json"
        ),
        "source_context must point at Exp 4253",
    )


def _round_duration(value: float) -> float:
    return _base._round_duration(value)


def _elapsed(started: float, clock: Clock) -> float:
    return _base._elapsed(started, clock)


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
