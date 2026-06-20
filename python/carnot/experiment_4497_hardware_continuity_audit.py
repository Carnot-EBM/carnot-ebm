"""Exp 4497 audit-only hardware continuity across attached boards.

Spec refs: REQ-HW-4497, SCENARIO-HW-4497.

This module keeps the hardware task deliberately narrow: it asks whether each
board is reachable through the operator-approved path, records the transcript,
and names the next practical hardware step. It does not flash bitstreams,
measure latency, or turn a missing board into claimed progress.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4439_hardware_continuity as _base


EXPERIMENT_ID = 4497
SCHEMA = "carnot.hardware_continuity_audit.v1"
SPEC_REFS = ["REQ-HW-4497", "SCENARIO-HW-4497"]
OUTPUT_REL_PATH = Path("results") / "experiment_4497_hardware_continuity_audit.json"
RANDOM_SEED = 4497
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
POLARFIRE_REACHABILITY_COMMAND = ("ssh", "-o", "ConnectTimeout=5", "polarfire", "true")

REQUIRED_OPERATOR_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
)
REQUIRED_ARTIFACT_FIELDS = (
    *REQUIRED_OPERATOR_FIELDS,
    "per_board_status",
    "per_board_reachability",
    "field_principles",
    "spec_refs",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with terminal prefix complete:/complete_/success:/success_/"
        "passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit (live_llm_inference | verifier_ensemble_against_cached_candidates "
        "| aggregation_from_upstream_artifacts) so adversarial_verify applies the "
        "right duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource "
        "fabrication."
    ),
}

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

NEXT_FORWARD_STEPS = {
    "kv260": (
        "Keep KV260 continuity on SSH-only confirmation, then run any board-level "
        "latency or overlay transcript in a separate hardware task."
    ),
    "gatemate": (
        "Use the reachable DirtyJTAG path to flash the n=16 Ising tile bitstream "
        "in a separate GateMate hardware task."
    ),
    "polarfire": (
        "Run an end-to-end Carnot dispatch hash-match smoke over SSH in a separate "
        "PolarFire hardware task."
    ),
}

CommandProbe = _base.CommandProbe
CommandRunner = _base.CommandRunner
Clock = _base.Clock

run_command = _base.run_command
command_to_string = _base.command_to_string
payload_checksum = _base.payload_checksum
state_token = _base.state_token


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the audit artifact from live or injected reachability probes."""
    started = clock()
    preconditions = check_preconditions(command_runner)
    by_resource = {entry["resource"]: entry for entry in preconditions}
    per_board_status = {
        "kv260": _board_status("kv260", by_resource["kv260_ssh"]),
        "gatemate": _board_status("gatemate", by_resource["gatemate_dirtyjtag_detect"]),
        "polarfire": _board_status("polarfire", by_resource["polarfire_ssh"]),
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
        "per_board_reachability": {
            board: bool(per_board_status[board]["reachable"]) for board in BOARD_NAMES
        },
        "per_board_status": per_board_status,
        "duration_s": _round_duration(clock() - started),
        "fabric_acceleration_claimed": False,
        "speedup_claim_made": False,
        "reproducibility_checksum": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def check_preconditions(command_runner: CommandRunner) -> list[dict[str, Any]]:
    """Run exactly the three reachability probes required by REQ-HW-4497."""
    specs = (
        ("kv260_ssh", KV260_REACHABILITY_COMMAND, _exit_zero, 10.0),
        ("gatemate_dirtyjtag_detect", GATEMATE_REACHABILITY_COMMAND, _gatemate_detected, 30.0),
        ("polarfire_ssh", POLARFIRE_REACHABILITY_COMMAND, _exit_zero, 10.0),
    )
    return [
        _precondition_entry(resource, probe, predicate(probe))
        for resource, command, predicate, timeout_s in specs
        for probe in (command_runner(command, timeout_s),)
    ]


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return the terminal-prefixed audit verdict required by the reconciler."""
    statuses = artifact["per_board_status"]
    return (
        "complete: hardware_continuity_audit_4497_"
        f"kv260_{state_token(str(statuses['kv260']['status']))}_"
        f"gatemate_{state_token(str(statuses['gatemate']['status']))}_"
        f"polarfire_{state_token(str(statuses['polarfire']['status']))}"
    )


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4497 and write the requested JSON artifact under results/."""
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    """Write stable JSON so hardware audit diffs stay inspectable."""
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Reject artifacts that drift from the Exp 4497 audit-only contract."""
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4497")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4497")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4497")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4497")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _validate_operator_fields(artifact)
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    _require("mmcblk" not in encoded, "forbidden retired host-storage marker")
    _validate_preconditions(artifact)
    _validate_per_board_status(artifact)
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "wrong substrate")
    _require(artifact.get("fabric_acceleration_claimed") is False, "no fabric claim")
    _require(artifact.get("speedup_claim_made") is False, "no speedup claim")
    verdict = str(artifact.get("honest_verdict", ""))
    _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict terminal prefix")
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "stale verdict")
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


def _board_status(board: str, precondition: dict[str, Any]) -> dict[str, Any]:
    reachable = bool(precondition["available"])
    status = _reachable_status(board) if reachable else f"blocked_{board}_unreachable"
    return {
        "board": board,
        "reachable": reachable,
        "status": status,
        "next_forward_step": NEXT_FORWARD_STEPS[board] if reachable else status,
        "precondition_resource": precondition["resource"],
        "precondition_command": precondition["command"],
        "reachability_transcript": {
            "exit_code": precondition["exit_code"],
            "duration_s": precondition["duration_s"],
            "observed": precondition["observed"],
        },
    }


def _reachable_status(board: str) -> str:
    return {
        "kv260": "kv260_reachable_ssh",
        "gatemate": "gatemate_reachable_dirtyjtag_detect",
        "polarfire": "polarfire_reachable_ssh",
    }[board]


def _exit_zero(probe: CommandProbe) -> bool:
    return probe.exit_code == 0


def _gatemate_detected(probe: CommandProbe) -> bool:
    observed = _observed(probe).lower()
    markers = ("gatemate", "gm1a", "idcode", "0x20000001")
    return probe.exit_code == 0 and any(marker in observed for marker in markers)


def _observed(probe: CommandProbe) -> str:
    observed = probe.combined_output.strip()
    return observed[:500] if observed else f"returncode={probe.exit_code}"


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
        "gatemate_dirtyjtag_detect": command_to_string(GATEMATE_REACHABILITY_COMMAND),
        "polarfire_ssh": command_to_string(POLARFIRE_REACHABILITY_COMMAND),
    }
    resources = [entry.get("resource") for entry in preconditions if isinstance(entry, dict)]
    _require(
        resources == ["kv260_ssh", "gatemate_dirtyjtag_detect", "polarfire_ssh"],
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
    expected_resource = {
        "kv260": "kv260_ssh",
        "gatemate": "gatemate_dirtyjtag_detect",
        "polarfire": "polarfire_ssh",
    }
    for board in BOARD_NAMES:
        record = statuses[board]
        _require(isinstance(record, dict), "board status entries must be dicts")
        _require(record.get("board") == board, "board name mismatch")
        _require(isinstance(record.get("reachable"), bool), "reachable must be bool")
        _require(reachability[board] is record["reachable"], "reachability mismatch")
        _require(record.get("precondition_resource") == expected_resource[board], "resource mismatch")
        precondition = by_resource[record["precondition_resource"]]
        _require(record.get("precondition_command") == precondition["command"], "command mismatch")
        _require(isinstance(record.get("next_forward_step"), str), "next step must be string")
        _require(record["next_forward_step"], "next step required")
        _require(isinstance(record.get("reachability_transcript"), dict), "transcript required")
        if record["reachable"]:
            _require(record.get("status") == _reachable_status(board), "reachable status mismatch")
        else:
            blocked = f"blocked_{board}_unreachable"
            _require(record.get("status") == blocked, "blocked status mismatch")
            _require(record.get("next_forward_step") == blocked, "blocked next step mismatch")


def _round_duration(value: float) -> float:
    return round(max(float(value), 0.0), 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
