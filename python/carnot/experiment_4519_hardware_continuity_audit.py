"""Exp 4519 audit-only hardware-task continuity across attached boards.

Spec refs: REQ-HW-4519, SCENARIO-HW-4519.

This experiment does exactly the operator-approved reachability audit: KV260 by
SSH only, GateMate by DirtyJTAG detect, and PolarFire by SSH. It records the
observed result for each board and names the next forward step without claiming
that a blocked board made progress.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4497_hardware_continuity_audit as _shared
from carnot import experiment_4507_hardware_continuity_audit as _prior


EXPERIMENT_ID = 4519
SCHEMA = "carnot.hardware_continuity_audit.v3"
SPEC_REFS = ["REQ-HW-4519", "SCENARIO-HW-4519"]
OUTPUT_REL_PATH = Path("results") / "experiment_4519_hardware_continuity_audit.json"
RANDOM_SEED = 4519
INFERENCE_SUBSTRATE = "hardware_smoke"
BOARD_NAMES = _prior.BOARD_NAMES

KV260_REACHABILITY_COMMAND = _prior.KV260_REACHABILITY_COMMAND
GATEMATE_REACHABILITY_COMMAND = _prior.GATEMATE_REACHABILITY_COMMAND
POLARFIRE_REACHABILITY_COMMAND = _prior.POLARFIRE_REACHABILITY_COMMAND

REQUIRED_OPERATOR_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "kv260_reachable",
    "gatemate_detected",
    "polarfire_reachable",
    "preconditions_checked",
)
REQUIRED_ARTIFACT_FIELDS = (
    *REQUIRED_OPERATOR_FIELDS,
    "per_board_status",
    "per_board_reachability",
    "field_principles",
    "spec_refs",
    "random_seed",
    "duration_s",
    "fabric_acceleration_claimed",
    "speedup_claim_made",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; e.g. complete: hardware_continuity_audit_<per-board "
        "summary> (a board blocked is an honest non-terminal state, not a task "
        "failure)."
    ),
    "inference_substrate": (
        "hardware_smoke -- SSH/USB reachability, per-board duration floor."
    ),
    "kv260_reachable": (
        "SSH-reachability is the ONLY valid KV260 precondition (SD-card "
        "mechanism retired)."
    ),
    "gatemate_detected": (
        "honest USB-detect result -- DirtyJTAG-unreachable is recorded, never "
        "fabricated."
    ),
    "polarfire_reachable": "SSH reachability of the board.",
    "preconditions_checked": (
        "records WHICH boards were probed and how; the audit IS the precondition set."
    ),
}

TERMINAL_PREFIXES = _prior.TERMINAL_PREFIXES

NEXT_FORWARD_STEPS = {
    "kv260": (
        "Run the SSH-only KV260 latency transcript against the active bitstream "
        "or overlay in a separate hardware task."
    ),
    "gatemate": (
        "Use the reachable DirtyJTAG path to flash the n=16 Ising tile bitstream "
        "in a separate GateMate hardware task."
    ),
    "polarfire": (
        "Run the PolarFire sampler smoke over SSH in a separate hardware task."
    ),
}

CommandProbe = _prior.CommandProbe
CommandRunner = _prior.CommandRunner
Clock = _prior.Clock

run_command = _prior.run_command
command_to_string = _prior.command_to_string
payload_checksum = _prior.payload_checksum
state_token = _prior.state_token
check_preconditions = _prior.check_preconditions


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4519 artifact from live or injected reachability probes."""
    artifact = _prior.build_artifact(command_runner=command_runner, clock=clock)
    _refresh_reachable_next_steps(artifact)
    reachability = artifact["per_board_reachability"]
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "random_seed": RANDOM_SEED,
            "field_principles": dict(FIELD_PRINCIPLES),
            "kv260_reachable": bool(reachability["kv260"]),
            "gatemate_detected": bool(reachability["gatemate"]),
            "polarfire_reachable": bool(reachability["polarfire"]),
        }
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return the terminal-prefixed audit verdict required by the conductor."""
    statuses = artifact["per_board_status"]
    return (
        "complete: hardware_continuity_audit_4519_"
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
    """Run Exp 4519 and write the requested JSON artifact under results/."""
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    """Write stable JSON so hardware audit diffs stay inspectable."""
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Reject artifacts that drift from the Exp 4519 audit-only contract."""
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4519")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4519")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4519")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4519")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _validate_operator_fields(artifact)
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    _require("mmcblk" not in encoded, "forbidden retired host-storage marker")
    _shared._validate_preconditions(artifact)
    _shared._validate_per_board_status(artifact)
    _validate_top_level_reachability(artifact)
    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be hardware_smoke",
    )
    _require(artifact.get("fabric_acceleration_claimed") is False, "no fabric claim")
    _require(artifact.get("speedup_claim_made") is False, "no speedup claim")
    verdict = str(artifact.get("honest_verdict", ""))
    _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict terminal prefix")
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "stale verdict")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _refresh_reachable_next_steps(artifact: dict[str, Any]) -> None:
    for board, next_step in NEXT_FORWARD_STEPS.items():
        record = artifact["per_board_status"][board]
        if record["reachable"]:
            record["next_forward_step"] = next_step


def _validate_operator_fields(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    _require(isinstance(principles, dict), "field_principles must be a dict")
    for field in REQUIRED_OPERATOR_FIELDS:
        _require(principles.get(field) == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
        _require(
            not (isinstance(artifact[field], dict) and set(artifact[field]) == {"value", "principle"}),
            f"{field} must remain a bare value",
        )


def _validate_top_level_reachability(artifact: dict[str, Any]) -> None:
    reachability = artifact.get("per_board_reachability")
    _require(isinstance(reachability, dict), "per_board_reachability must be a dict")
    _require(isinstance(artifact.get("kv260_reachable"), bool), "kv260_reachable must be bool")
    _require(isinstance(artifact.get("gatemate_detected"), bool), "gatemate_detected must be bool")
    _require(
        isinstance(artifact.get("polarfire_reachable"), bool),
        "polarfire_reachable must be bool",
    )
    _require(artifact["kv260_reachable"] is reachability["kv260"], "kv260_reachable mismatch")
    _require(artifact["gatemate_detected"] is reachability["gatemate"], "gatemate_detected mismatch")
    _require(
        artifact["polarfire_reachable"] is reachability["polarfire"],
        "polarfire_reachable mismatch",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    print(run_experiment())
