"""Exp 4507 audit-only hardware-task continuity across attached boards.

Spec refs: REQ-HW-4507, SCENARIO-HW-4507.

This module intentionally does less than a hardware bring-up experiment. It
checks whether each board can be reached through the operator-approved command,
records the observed transcript, and names the next forward step. That boundary
keeps a missing or unreachable board from being mistaken for real hardware
progress.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4497_hardware_continuity_audit as _prior


EXPERIMENT_ID = 4507
SCHEMA = "carnot.hardware_continuity_audit.v2"
SPEC_REFS = ["REQ-HW-4507", "SCENARIO-HW-4507"]
OUTPUT_REL_PATH = Path("results") / "experiment_4507_hardware_continuity_audit.json"
RANDOM_SEED = 4507
INFERENCE_SUBSTRATE = "hardware_smoke"
BOARD_NAMES = _prior.BOARD_NAMES

KV260_REACHABILITY_COMMAND = _prior.KV260_REACHABILITY_COMMAND
GATEMATE_REACHABILITY_COMMAND = _prior.GATEMATE_REACHABILITY_COMMAND
POLARFIRE_REACHABILITY_COMMAND = _prior.POLARFIRE_REACHABILITY_COMMAND

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
        "passed:/passed_/shipped:/shipped_."
    ),
    "inference_substrate": (
        "explicit substrate so adversarial_verify applies the right duration floor."
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
    """Build the Exp 4507 artifact from live or injected reachability probes."""
    artifact = _prior.build_artifact(command_runner=command_runner, clock=clock)
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "random_seed": RANDOM_SEED,
            "field_principles": dict(FIELD_PRINCIPLES),
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
        "complete: hardware_continuity_audit_4507_"
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
    """Run Exp 4507 and write the requested JSON artifact under results/."""
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    """Write stable JSON so hardware audit diffs stay inspectable."""
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Reject artifacts that drift from the Exp 4507 audit-only contract."""
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4507")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4507")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4507")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4507")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _validate_operator_fields(artifact)
    _prior._validate_preconditions(artifact)
    _prior._validate_per_board_status(artifact)
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


def _validate_operator_fields(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    _require(isinstance(principles, dict), "field_principles must be a dict")
    for field in REQUIRED_OPERATOR_FIELDS:
        _require(principles.get(field) == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
        _require(
            not (isinstance(artifact[field], dict) and set(artifact[field]) == {"value", "principle"}),
            f"{field} must remain a bare value",
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
