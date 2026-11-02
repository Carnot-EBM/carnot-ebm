"""Exp 5037 KV260 SSH-only overlay/UIO and energy continuity artifact.

Spec refs: REQ-HW-5037, SCENARIO-HW-5037.

This check repeats the Exp 5023 KV260 continuity pattern for the current
hardware-continuity slot. The board is reached only over SSH. A failed SSH
precondition still writes the blocked artifact; a reachable board records
overlay/UIO state and runs the tiny on-board energy smoke only when a Carnot
Ising overlay is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time


if __package__ in {None, ""}:  # pragma: no cover - direct script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot import experiment_5023_kv260_continuity as base


EXPERIMENT_ID = 5037
SCHEMA = base.SCHEMA
SPEC_REFS = ["REQ-HW-5037", "SCENARIO-HW-5037"]
OUTPUT_REL_PATH = Path("results") / "experiment_5037_kv260_continuity.json"
RANDOM_SEED = 5037
INFERENCE_SUBSTRATE = base.INFERENCE_SUBSTRATE

CommandProbe = base.CommandProbe
CommandRunner = base.CommandRunner
Clock = base.Clock

KV260_SSH_COMMAND = base.KV260_SSH_COMMAND
KV260_LISTAPPS_COMMAND = base.KV260_LISTAPPS_COMMAND
KV260_LISTAPPS_SUDO_COMMAND = base.KV260_LISTAPPS_SUDO_COMMAND
KV260_UIO_COMMAND = base.KV260_UIO_COMMAND
KV260_ENERGY_COMMAND = base.KV260_ENERGY_COMMAND
VALID_OVERLAYS = base.VALID_OVERLAYS

ENERGY_SMOKE_PROBLEM = base.ENERGY_SMOKE_PROBLEM
ENERGY_SMOKE_EXPECTED = base.ENERGY_SMOKE_EXPECTED
BLOCKED_SSH_VERDICT = base.BLOCKED_SSH_VERDICT

REQUIRED_PRINCIPLE_FIELDS = base.REQUIRED_PRINCIPLE_FIELDS
REQUIRED_ARTIFACT_FIELDS = base.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = base.FIELD_PRINCIPLES

command_to_string = base.command_to_string
payload_checksum = base.payload_checksum
run_command = base.run_command
parse_uio_devices = base.parse_uio_devices
loaded_overlay_from_xmutil = base.loaded_overlay_from_xmutil
parse_energy_smoke_stdout = base.parse_energy_smoke_stdout


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, object]:
    """Build the Exp 5037 artifact from live or injected SSH board probes."""
    return _retarget_artifact(base.build_artifact(command_runner=command_runner, clock=clock))


def write_artifact(repo_root: str | Path, artifact: dict[str, object]) -> Path:
    """Write a validated Exp 5037 artifact under the requested repository root."""
    validate_artifact(artifact)
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
    """Run Exp 5037 and write `results/experiment_5037_kv260_continuity.json`."""
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate the Exp 5037 SSH-only overlay/UIO energy schema."""
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    _require(artifact.get("schema") == SCHEMA, "schema mismatch")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment mismatch")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle must be false")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    _validate_base_contract(artifact)
    _require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "bad checksum",
    )


def _retarget_artifact(artifact: dict[str, object]) -> dict[str, object]:
    retargeted = dict(artifact)
    retargeted.update(
        {
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
        }
    )
    retargeted["reproducibility_checksum"] = payload_checksum(retargeted)
    validate_artifact(retargeted)
    return retargeted


def _validate_base_contract(artifact: dict[str, object]) -> None:
    base.validate_artifact(_base_contract_artifact(artifact))


def _base_contract_artifact(artifact: dict[str, object]) -> dict[str, object]:
    base_artifact = dict(artifact)
    base_artifact.update(
        {
            "experiment": base.EXPERIMENT_ID,
            "spec_refs": list(base.SPEC_REFS),
            "random_seed": base.RANDOM_SEED,
            "reproducibility_checksum": "",
        }
    )
    base_artifact["reproducibility_checksum"] = base.payload_checksum(base_artifact)
    return base_artifact


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover - live hardware entrypoint.
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"kv260_ssh_reachable: {artifact['kv260_ssh_reachable']}")
    print(f"loaded_overlay: {artifact['overlay_state']['loaded_overlay']}")
    print(f"on_board_energy_duration_s: {artifact['on_board_energy_duration_s']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
