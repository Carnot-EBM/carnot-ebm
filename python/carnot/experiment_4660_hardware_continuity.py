"""Exp 4660 hardware-continuity audit across the three attached boards.

Spec refs: REQ-HW-4660, SCENARIO-HW-4660.

This is a lightweight continuity record, not a bring-up or benchmark. KV260 is
checked by SSH only, PolarFire by SSH, and GateMate by DirtyJTAG IDCODE detect.
Reachable boards get a small state read; blocked boards stay visible as their
own blocked_<board> row.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any


if __package__ in {None, ""}:  # pragma: no cover - direct script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot import experiment_4612_hardware_continuity as _base  # noqa: E402


EXPERIMENT_ID = 4660
SCHEMA = "carnot.hardware_continuity_state_audit.v12"
SPEC_REFS = ["REQ-HW-4660", "SCENARIO-HW-4660"]
OUTPUT_REL_PATH = Path("results") / "experiment_4660_hardware_continuity.json"
RANDOM_SEED = 4660
INFERENCE_SUBSTRATE = _base.INFERENCE_SUBSTRATE
BOARD_NAMES = _base.BOARD_NAMES

KV260_PRECONDITION_COMMAND = _base.KV260_PRECONDITION_COMMAND
POLARFIRE_PRECONDITION_COMMAND = _base.POLARFIRE_PRECONDITION_COMMAND
GATEMATE_PRECONDITION_COMMAND = _base.GATEMATE_PRECONDITION_COMMAND
KV260_STATE_COMMAND = _base.KV260_STATE_COMMAND
KV260_UIO_SMOKE_COMMAND = _base.KV260_UIO_SMOKE_COMMAND
POLARFIRE_STATE_COMMAND = _base.POLARFIRE_STATE_COMMAND

REQUIRED_OPERATOR_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "kv260_precondition",
    "boards_reachable",
    "per_board_state",
    "reachable_board_count",
    "speedup_claim_made",
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
    "duration_s",
    "fabric_acceleration_claimed",
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
    "kv260_precondition": (
        "MUST be the SSH-reachability check (ssh kria true), NEVER a host SD-card-slot "
        "device check (KV260 SSH-Not-SD-Card Discipline)."
    ),
    "boards_reachable": (
        "per-board reachability (kv260/polarfire/gatemate) -- the continuity record "
        "that keeps each attached board visible."
    ),
    "per_board_state": (
        "the recorded state per reachable board (loaded overlay / uptime / IDCODE) "
        "or blocked_<board>_<reason>."
    ),
    "reachable_board_count": (
        "the integer count of reachable boards (the at-a-glance continuity number)."
    ),
    "speedup_claim_made": (
        "MUST be false -- this is a reachability audit, not a fabric-acceleration "
        "claim (no fabrication)."
    ),
    "preconditions_checked": (
        "records WHICH board checks ran + their results; pre-empts "
        "silent-missing-hardware fabrication."
    ),
}

BLOCKED_STATUSES = _base.BLOCKED_STATUSES
REACHABLE_STATUSES = _base.REACHABLE_STATUSES

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
    """Build the Exp 4660 artifact from real or injected hardware probes."""
    artifact = _base.build_artifact(command_runner=command_runner, clock=clock)
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
    """Return the board-count verdict required by REQ-HW-4660."""
    return f"success: hardware_continuity_{artifact['reachable_board_count']}_of_3_boards_reachable"


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4660 and write `results/experiment_4660_hardware_continuity.json`."""
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
    """Reject artifacts that drift from the Exp 4660 continuity contract."""
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4660")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4660")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4660")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4660")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _validate_field_principles(artifact)
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    _require("mmcblk" not in encoded, "forbidden retired host-storage marker")
    _base._validate_preconditions(artifact)
    _base._validate_kv260_precondition(artifact)
    _base._validate_board_maps(artifact)
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


def _validate_field_principles(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    _require(principles == FIELD_PRINCIPLES, "field_principles mismatch")
    for field in REQUIRED_OPERATOR_FIELDS:
        _require(field in artifact, f"missing operator field: {field}")
        _require(
            not (isinstance(artifact[field], dict) and set(artifact[field]) == {"value", "principle"}),
            f"{field} must remain a bare value",
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
