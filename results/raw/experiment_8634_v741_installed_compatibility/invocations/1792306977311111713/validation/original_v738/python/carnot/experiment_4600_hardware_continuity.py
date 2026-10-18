"""Exp 4600 hardware-continuity audit across attached boards.

Spec refs: REQ-HW-4600, SCENARIO-HW-4600.

This milestone is another continuity audit, not a bring-up or synthesis run.
The board checks intentionally stay identical to Exp 4588: KV260 is checked
only through SSH, GateMate through DirtyJTAG USB detect, and PolarFire through
SSH. Keeping the same mechanics prevents the retired KV260 host-storage
precondition from creeping back in while still giving every attached board a
visible row in the milestone artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4588_hardware_continuity as _previous


EXPERIMENT_ID = 4600
SCHEMA = "carnot.hardware_continuity_state_audit.v7"
SPEC_REFS = ["REQ-HW-4600", "SCENARIO-HW-4600"]
OUTPUT_REL_PATH = Path("results") / "experiment_4600_hardware_continuity.json"
RANDOM_SEED = 4600

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
capture_board_states = _previous.capture_board_states


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4600 artifact from live or injected hardware probes."""
    artifact = dict(_previous.build_artifact(command_runner=command_runner, clock=clock))
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
        }
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4600 and write `results/experiment_4600_hardware_continuity.json`."""
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
    """Reject artifacts that drift from the Exp 4600 audit-only contract."""
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4600")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4600")
    _require(
        artifact.get("spec_refs") == SPEC_REFS,
        "spec_refs must name REQ/SCENARIO 4600",
    )
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4600")
    _require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "bad checksum",
    )
    _previous.validate_artifact(_as_4588_metadata(artifact))


def _as_4588_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
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
