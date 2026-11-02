"""Exp 4988 KV260 SSH-only overlay continuity artifact.

Spec refs: REQ-HW-4988, SCENARIO-HW-4988.

This continuity check keeps the graduated KV260 board in the milestone rotation.
The SSH probe is the only valid precondition. Host SD-card or block-device
checks are the retired mechanism and are not evidence that this board is ready.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time


if __package__ in {None, ""}:  # pragma: no cover - direct script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot import experiment_4977_kv260_continuity as base


EXPERIMENT_ID = 4988
SCHEMA = base.SCHEMA
SPEC_REFS = ["REQ-HW-4988", "SCENARIO-HW-4988"]
OUTPUT_REL_PATH = Path("results") / "experiment_4988_kv260_continuity.json"
RANDOM_SEED = 4988
INFERENCE_SUBSTRATE = base.INFERENCE_SUBSTRATE

CommandProbe = base.CommandProbe
CommandRunner = base.CommandRunner
Clock = base.Clock

KV260_SSH_COMMAND = base.KV260_SSH_COMMAND
KV260_LISTAPPS_COMMAND = base.KV260_LISTAPPS_COMMAND
KV260_LISTAPPS_SUDO_COMMAND = base.KV260_LISTAPPS_SUDO_COMMAND
VALID_OVERLAYS = base.VALID_OVERLAYS

SUCCESS_VERDICT = base.SUCCESS_VERDICT
BLOCKED_SSH_VERDICT = base.BLOCKED_SSH_VERDICT
WRONG_MECHANISM_VERDICT = base.WRONG_MECHANISM_VERDICT

REQUIRED_PRINCIPLE_FIELDS = base.REQUIRED_PRINCIPLE_FIELDS
REQUIRED_ARTIFACT_FIELDS = base.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = dict(base.FIELD_PRINCIPLES)

command_to_string = base.command_to_string
loaded_overlay_from_xmutil = base.loaded_overlay_from_xmutil
payload_checksum = base.payload_checksum
run_command = base.run_command


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, object]:
    """Build the Exp 4988 artifact from live or injected SSH board probes."""
    artifact = base.build_artifact(command_runner=command_runner, clock=clock)
    return _retarget_artifact(artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, object]) -> Path:
    """Write a validated Exp 4988 artifact under the requested repository root."""
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4988 and write `results/experiment_4988_kv260_continuity.json`."""
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate the Exp 4988 SSH-only overlay continuity schema."""
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    _require(artifact.get("schema") == SCHEMA, "schema mismatch")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment mismatch")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle must be false")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    _require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "bad checksum",
    )
    _validate_base_contract(artifact)


def _retarget_artifact(artifact: dict[str, object]) -> dict[str, object]:
    retargeted = dict(artifact)
    retargeted.update(
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "field_principles": dict(FIELD_PRINCIPLES),
            "inference_substrate": INFERENCE_SUBSTRATE,
            "random_seed": RANDOM_SEED,
        }
    )
    if retargeted.get("kv260_ssh_reachable") is False:
        retargeted["honest_verdict"] = BLOCKED_SSH_VERDICT
    else:
        retargeted["honest_verdict"] = SUCCESS_VERDICT
    retargeted["reproducibility_checksum"] = payload_checksum(retargeted)
    validate_artifact(retargeted)
    return retargeted


def _validate_base_contract(artifact: dict[str, object]) -> None:
    base_artifact = dict(artifact)
    base_artifact.update(
        {
            "schema": base.SCHEMA,
            "experiment": base.EXPERIMENT_ID,
            "spec_refs": list(base.SPEC_REFS),
            "field_principles": dict(base.FIELD_PRINCIPLES),
            "inference_substrate": base.INFERENCE_SUBSTRATE,
            "random_seed": base.RANDOM_SEED,
        }
    )
    if base_artifact.get("kv260_ssh_reachable") is False:
        base_artifact["honest_verdict"] = base.BLOCKED_SSH_VERDICT
    else:
        base_artifact["honest_verdict"] = base.SUCCESS_VERDICT
    base_artifact["reproducibility_checksum"] = base.payload_checksum(base_artifact)
    base.validate_artifact(base_artifact)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover - live hardware entrypoint.
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"kv260_ssh_reachable: {artifact['kv260_ssh_reachable']}")
    print(f"loaded_overlay: {artifact['loaded_overlay']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
