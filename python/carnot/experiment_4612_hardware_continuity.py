"""Exp 4612 hardware-continuity audit across the three attached boards.

Spec refs: REQ-HW-4612, SCENARIO-HW-4612.

This module is deliberately a light continuity check. It keeps KV260, PolarFire,
and GateMate visible in the milestone record without attempting bring-up,
flashing, or a heavy benchmark. KV260 is checked only through SSH because the
board is already booted; host SD-card checks would measure the wrong machine.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

from carnot import experiment_4600_hardware_continuity as _previous


EXPERIMENT_ID = 4612
SCHEMA = "carnot.hardware_continuity_state_audit.v8"
SPEC_REFS = ["REQ-HW-4612", "SCENARIO-HW-4612"]
OUTPUT_REL_PATH = Path("results") / "experiment_4612_hardware_continuity.json"
RANDOM_SEED = 4612
INFERENCE_SUBSTRATE = "hardware_smoke"
BOARD_NAMES = ("kv260", "polarfire", "gatemate")

KV260_PRECONDITION_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
POLARFIRE_PRECONDITION_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)
GATEMATE_PRECONDITION_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
KV260_STATE_COMMAND = ("ssh", "kria", "xmutil listapps")
KV260_UIO_SMOKE_COMMAND = ("ssh", "kria", "ls /dev/uio*")
POLARFIRE_STATE_COMMAND = ("ssh", "polarfire", "uptime")

REQUIRED_OPERATOR_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "boards_reachable",
    "kv260_precondition",
    "per_board_state",
    "preconditions_checked",
)
REQUIRED_ARTIFACT_FIELDS = (
    *REQUIRED_OPERATOR_FIELDS,
    "schema",
    "experiment",
    "spec_refs",
    "random_seed",
    "field_principles",
    "per_board_reachability",
    "reachable_board_count",
    "duration_s",
    "fabric_acceleration_claimed",
    "speedup_claim_made",
    "bitstream_build_attempted",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: hardware_continuity_<n>_of_3_boards_reachable "
        "(or complete:/blocked_ per board state)."
    ),
    "inference_substrate": (
        "hardware_smoke -- SSH/USB reachability checks, per-board floor; declared so "
        "a fast real check is not DURATION_TOO_SHORT false-flagged."
    ),
    "boards_reachable": (
        "per-board reachability (kv260/polarfire/gatemate) -- the continuity record "
        "that keeps each attached board visible per the discipline."
    ),
    "kv260_precondition": (
        "MUST be the SSH-reachability check, NEVER a host SD-card-slot device check "
        "(the KV260 SSH-Not-SD-Card Discipline)."
    ),
    "per_board_state": (
        "the recorded state per reachable board (loaded overlay / uptime / IDCODE) "
        "or blocked_<board>_<reason>."
    ),
    "preconditions_checked": (
        "records WHICH board checks ran + their results; pre-empts "
        "silent-missing-hardware fabrication."
    ),
}

BLOCKED_STATUSES = {
    "kv260": "blocked_kv260_ssh_unreachable",
    "polarfire": "blocked_polarfire_ssh_timeout",
    "gatemate": "blocked_gatemate_usb_undetected",
}
REACHABLE_STATUSES = {
    "kv260": "kv260_reachable_loaded_overlay_recorded",
    "polarfire": "polarfire_reachable_uptime_recorded",
    "gatemate": "gatemate_reachable_idcode_recorded",
}

CommandProbe = _previous.CommandProbe
CommandRunner = _previous.CommandRunner
Clock = _previous.Clock

run_command = _previous.run_command
prepend_oss_cad_suite = _previous.prepend_oss_cad_suite
command_to_string = _previous.command_to_string
payload_checksum = _previous.payload_checksum


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4612 artifact from real or injected hardware probes."""
    started = clock()
    preconditions = check_preconditions(command_runner)
    by_resource = {entry["resource"]: entry for entry in preconditions}
    per_board_state = capture_board_states(command_runner, by_resource)
    boards_reachable = {
        board: bool(per_board_state[board]["reachable"]) for board in BOARD_NAMES
    }
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "honest_verdict": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "boards_reachable": boards_reachable,
        "per_board_reachability": dict(boards_reachable),
        "reachable_board_count": sum(1 for reachable in boards_reachable.values() if reachable),
        "kv260_precondition": {
            "command": command_to_string(KV260_PRECONDITION_COMMAND),
            "result": by_resource["kv260_ssh"],
            "discipline": "ssh_only_no_host_sd_card",
        },
        "per_board_state": per_board_state,
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
    """Run exactly the SSH/USB checks required by REQ-HW-4612."""
    specs = (
        ("kv260_ssh", KV260_PRECONDITION_COMMAND, _exit_zero, 10.0),
        ("polarfire_ssh", POLARFIRE_PRECONDITION_COMMAND, _exit_zero, 10.0),
        ("gatemate_usb_detect", GATEMATE_PRECONDITION_COMMAND, _gatemate_detected, 30.0),
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
    """Capture state only after a board has passed its own reachability check."""
    records: dict[str, dict[str, Any]] = {}
    if preconditions_by_resource["kv260_ssh"]["available"]:
        records["kv260"] = _reachable_kv260_record(
            preconditions_by_resource["kv260_ssh"],
            command_runner,
        )
    else:
        records["kv260"] = _blocked_board_record(
            "kv260",
            preconditions_by_resource["kv260_ssh"],
        )
    if preconditions_by_resource["polarfire_ssh"]["available"]:
        records["polarfire"] = _reachable_state_record(
            "polarfire",
            preconditions_by_resource["polarfire_ssh"],
            _state_record_from_probe(
                "uptime",
                command_runner(POLARFIRE_STATE_COMMAND, 20.0),
            ),
        )
    else:
        records["polarfire"] = _blocked_board_record(
            "polarfire",
            preconditions_by_resource["polarfire_ssh"],
        )
    if preconditions_by_resource["gatemate_usb_detect"]["available"]:
        records["gatemate"] = _reachable_state_record(
            "gatemate",
            preconditions_by_resource["gatemate_usb_detect"],
            _gatemate_state_record(preconditions_by_resource["gatemate_usb_detect"]),
        )
    else:
        records["gatemate"] = _blocked_board_record(
            "gatemate",
            preconditions_by_resource["gatemate_usb_detect"],
        )
    return records


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return the board-count verdict while treating per-board blocks as honest data."""
    return f"success: hardware_continuity_{artifact['reachable_board_count']}_of_3_boards_reachable"


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4612 and write `results/experiment_4612_hardware_continuity.json`."""
    prepend_oss_cad_suite()
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    """Write stable JSON so hardware continuity diffs are easy to audit."""
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Reject artifacts that drift from the Exp 4612 continuity contract."""
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4612")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4612")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4612")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4612")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _validate_field_principles(artifact)
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    _require("mmcblk" not in encoded, "forbidden retired host-storage marker")
    _validate_preconditions(artifact)
    _validate_kv260_precondition(artifact)
    _validate_board_maps(artifact)
    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be hardware_smoke",
    )
    _require(artifact.get("fabric_acceleration_claimed") is False, "no fabric claim")
    _require(artifact.get("speedup_claim_made") is False, "no speedup claim")
    _require(artifact.get("bitstream_build_attempted") is False, "no bitstream build")
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "stale honest_verdict")
    _require(str(artifact.get("honest_verdict", "")).startswith("success:"), "honest_verdict prefix")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _reachable_kv260_record(
    precondition: dict[str, Any],
    command_runner: CommandRunner,
) -> dict[str, Any]:
    state = _state_record_from_probe(
        "loaded_overlay",
        command_runner(KV260_STATE_COMMAND, 20.0),
    )
    if _kv260_carnot_overlay_loaded(state):
        smoke = _state_record_from_probe(
            "uio_devices",
            command_runner(KV260_UIO_SMOKE_COMMAND, 20.0),
        )
    else:
        smoke = {
            "captured": False,
            "reason": "carnot_ising_overlay_not_loaded",
        }
    record = _reachable_state_record("kv260", precondition, state)
    record["energy_eval_smoke"] = smoke
    record["duration_s"] = _round_duration(
        float(record["duration_s"]) + float(smoke.get("duration_s", 0.0))
    )
    return record


def _reachable_state_record(
    board: str,
    precondition: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    status = REACHABLE_STATUSES[board]
    if state.get("succeeded") is not True:
        status = f"blocked_{board}_{state['state_type']}_probe_failed_returncode_{state['exit_code']}"
    return {
        "board": board,
        "reachable": True,
        "status": status,
        "precondition_resource": precondition["resource"],
        "precondition_command": precondition["command"],
        "reachability_transcript": _reachability_transcript(precondition),
        "state": state,
        "duration_s": _round_duration(
            float(precondition["duration_s"]) + float(state.get("duration_s", 0.0))
        ),
    }


def _blocked_board_record(board: str, precondition: dict[str, Any]) -> dict[str, Any]:
    status = BLOCKED_STATUSES[board]
    return {
        "board": board,
        "reachable": False,
        "status": status,
        "precondition_resource": precondition["resource"],
        "precondition_command": precondition["command"],
        "reachability_transcript": _reachability_transcript(precondition),
        "state": {"captured": False, "reason": status},
        "duration_s": _round_duration(float(precondition["duration_s"])),
    }


def _state_record_from_probe(state_type: str, probe: CommandProbe) -> dict[str, Any]:
    return {
        "captured": True,
        "succeeded": probe.exit_code == 0,
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
        "succeeded": True,
        "state_type": "gatemate_idcode",
        "command": precondition["command"],
        "exit_code": int(precondition["exit_code"]),
        "duration_s": float(precondition["duration_s"]),
        "observed": observed,
        "idcode": _extract_idcode(observed),
    }


def _precondition_entry(resource: str, probe: CommandProbe, available: bool) -> dict[str, Any]:
    return {
        "resource": resource,
        "available": bool(available),
        "command": command_to_string(probe.command),
        "exit_code": int(probe.exit_code),
        "duration_s": _round_duration(probe.duration_s),
        "observed": _observed(probe),
    }


def _reachability_transcript(precondition: dict[str, Any]) -> dict[str, Any]:
    return {
        "exit_code": precondition["exit_code"],
        "duration_s": precondition["duration_s"],
        "observed": precondition["observed"],
    }


def _exit_zero(probe: CommandProbe) -> bool:
    return probe.exit_code == 0


def _gatemate_detected(probe: CommandProbe) -> bool:
    observed = _observed(probe)
    return probe.exit_code == 0 and _extract_idcode(observed) is not None


def _extract_idcode(text: str) -> str | None:
    match = re.search(r"0x[0-9a-fA-F]{8}", text)
    return match.group(0) if match else None


def _kv260_carnot_overlay_loaded(state: dict[str, Any]) -> bool:
    return "carnot_ising" in str(state.get("observed", "")).lower()


def _observed(probe: CommandProbe) -> str:
    transcript = (probe.stdout or probe.stderr).strip()
    if transcript:
        return transcript
    return f"returncode={probe.exit_code}"


def _round_duration(value: float) -> float:
    return round(max(float(value), 0.0), 6)


def _validate_field_principles(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    _require(principles == FIELD_PRINCIPLES, "field_principles mismatch")
    for field in REQUIRED_OPERATOR_FIELDS:
        _require(field in artifact, f"missing operator field: {field}")
        _require(
            not (isinstance(artifact[field], dict) and set(artifact[field]) == {"value", "principle"}),
            f"{field} must remain a bare value",
        )


def _validate_preconditions(artifact: dict[str, Any]) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(isinstance(preconditions, list), "preconditions_checked must be a list")
    _require(len(preconditions) == 3, "exactly three preconditions are required")
    expected = [
        ("kv260_ssh", command_to_string(KV260_PRECONDITION_COMMAND)),
        ("polarfire_ssh", command_to_string(POLARFIRE_PRECONDITION_COMMAND)),
        ("gatemate_usb_detect", command_to_string(GATEMATE_PRECONDITION_COMMAND)),
    ]
    for entry, (resource, command) in zip(preconditions, expected, strict=True):
        _require(isinstance(entry, dict), "precondition entries must be dicts")
        _require(entry.get("resource") == resource, "precondition resource mismatch")
        _require(entry.get("command") == command, "precondition command mismatch")
        _require(isinstance(entry.get("available"), bool), "precondition available must be bool")
        _require({"exit_code", "duration_s", "observed"} <= set(entry), "precondition missing keys")


def _validate_kv260_precondition(artifact: dict[str, Any]) -> None:
    kv260 = artifact.get("kv260_precondition")
    _require(isinstance(kv260, dict), "kv260_precondition must be dict")
    _require(
        kv260.get("command") == command_to_string(KV260_PRECONDITION_COMMAND),
        "kv260 precondition command mismatch",
    )
    _require(kv260.get("result") == artifact["preconditions_checked"][0], "kv260 result mismatch")
    _require("mmcblk" not in json.dumps(kv260, sort_keys=True).lower(), "kv260 host storage check")


def _validate_board_maps(artifact: dict[str, Any]) -> None:
    boards_reachable = artifact.get("boards_reachable")
    states = artifact.get("per_board_state")
    _require(isinstance(boards_reachable, dict), "boards_reachable must be dict")
    _require(isinstance(states, dict), "per_board_state must be dict")
    _require(set(boards_reachable.keys()) == set(BOARD_NAMES), "boards_reachable board mismatch")
    _require(set(states.keys()) == set(BOARD_NAMES), "per_board_state board mismatch")
    _require(artifact.get("per_board_reachability") == boards_reachable, "reachability alias mismatch")
    reachable_count = sum(1 for reachable in boards_reachable.values() if reachable)
    _require(artifact.get("reachable_board_count") == reachable_count, "reachable count mismatch")
    for board in BOARD_NAMES:
        record = states[board]
        _require(record.get("reachable") is boards_reachable[board], f"{board} reachability mismatch")
        _require(record.get("precondition_command"), f"{board} precondition command missing")
        _require(float(record.get("duration_s", 0.0)) >= 0.0, f"{board} duration invalid")
        if boards_reachable[board]:
            _require(record.get("state", {}).get("captured") is True, f"{board} state missing")
            if record.get("state", {}).get("succeeded") is True:
                _require(record.get("status") == REACHABLE_STATUSES[board], f"{board} status mismatch")
            else:
                _require(
                    str(record.get("status", "")).startswith(f"blocked_{board}_"),
                    f"{board} failed state status mismatch",
                )
        else:
            _require(record.get("status") == BLOCKED_STATUSES[board], f"{board} blocked status mismatch")
            _require(
                record.get("state") == {"captured": False, "reason": BLOCKED_STATUSES[board]},
                f"{board} blocked state mismatch",
            )
    if boards_reachable["kv260"]:
        _require(
            "energy_eval_smoke" in states["kv260"],
            "reachable kv260 must report UIO smoke status",
        )


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
