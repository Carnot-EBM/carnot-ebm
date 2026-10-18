"""Exp 4552 hardware-continuity audit across attached boards.

Spec refs: REQ-HW-4552, SCENARIO-HW-4552.

This module is intentionally small because the milestone asks for a continuity
audit, not bring-up. It checks KV260 only through SSH, checks GateMate through
DirtyJTAG USB detect, checks PolarFire through SSH, and records one state
transcript for each reachable board. Blocked boards stay visible as blocked
rows, so a missing board does not erase it from the milestone record.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4529_hardware_continuity as _state_audit


EXPERIMENT_ID = 4552
SCHEMA = "carnot.hardware_continuity_state_audit.v3"
SPEC_REFS = ["REQ-HW-4552", "SCENARIO-HW-4552"]
OUTPUT_REL_PATH = Path("results") / "experiment_4552_hardware_continuity.json"
RANDOM_SEED = 4552

INFERENCE_SUBSTRATE = _state_audit.INFERENCE_SUBSTRATE
BOARD_NAMES = _state_audit.BOARD_NAMES
KV260_REACHABILITY_COMMAND = _state_audit.KV260_REACHABILITY_COMMAND
GATEMATE_REACHABILITY_COMMAND = _state_audit.GATEMATE_REACHABILITY_COMMAND
POLARFIRE_REACHABILITY_COMMAND = _state_audit.POLARFIRE_REACHABILITY_COMMAND
KV260_STATE_COMMAND = _state_audit.KV260_STATE_COMMAND
POLARFIRE_STATE_COMMAND = _state_audit.POLARFIRE_STATE_COMMAND

REQUIRED_OPERATOR_FIELDS = _state_audit.REQUIRED_OPERATOR_FIELDS
REQUIRED_ARTIFACT_FIELDS = _state_audit.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = dict(_state_audit.FIELD_PRINCIPLES)
BLOCKED_STATUSES = dict(_state_audit.BLOCKED_STATUSES)
REACHABLE_STATUSES = dict(_state_audit.REACHABLE_STATUSES)

CommandProbe = _state_audit.CommandProbe
CommandRunner = _state_audit.CommandRunner
Clock = _state_audit.Clock

run_command = _state_audit.run_command
prepend_oss_cad_suite = _state_audit.prepend_oss_cad_suite
command_to_string = _state_audit.command_to_string
payload_checksum = _state_audit.payload_checksum


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4552 artifact from live or injected hardware probes."""
    artifact = _state_audit.build_artifact(command_runner=command_runner, clock=clock)
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "random_seed": RANDOM_SEED,
        }
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return the terminal board-count verdict required by this audit task."""
    return _state_audit.honest_verdict(artifact)


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4552 and write `results/experiment_4552_hardware_continuity.json`."""
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
    """Reject artifacts that drift from the Exp 4552 audit-only contract."""
    _state_audit._require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4552")
    _state_audit._require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4552")
    _state_audit._require(
        artifact.get("spec_refs") == SPEC_REFS,
        "spec_refs must name REQ/SCENARIO 4552",
    )
    _state_audit._require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4552")
    _state_audit._require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "bad checksum",
    )
    _state_audit.validate_artifact(_as_4529_metadata(artifact))


def _as_4529_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    base_artifact = dict(artifact)
    base_artifact.update(
        {
            "schema": _state_audit.SCHEMA,
            "experiment": _state_audit.EXPERIMENT_ID,
            "spec_refs": list(_state_audit.SPEC_REFS),
            "random_seed": _state_audit.RANDOM_SEED,
        }
    )
    base_artifact["reproducibility_checksum"] = payload_checksum(base_artifact)
    return base_artifact


def main() -> None:  # pragma: no cover - live hardware entrypoint.
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
