"""Exp 4378 KV260 SSH reachability and loaded-bitstream continuity check.

This continuation keeps the KV260 path visible with the same lightweight board
probe as Exp 4367: SSH reachability first, then loaded-overlay and UIO evidence
only when the board is reachable. The precondition is the board SSH command,
never a host storage-device check.

Spec refs: REQ-HW-4378, SCENARIO-HW-4378.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4356_hardware_continuity_kv260 as _base
from carnot import experiment_4367_hardware_continuity_kv260 as _previous


EXPERIMENT_ID = 4378
SCHEMA = _previous.SCHEMA
SPEC_REFS = ["REQ-HW-4378", "SCENARIO-HW-4378"]
OUTPUT_REL_PATH = Path("results") / "experiment_4378_hardware_continuity_kv260.json"
RANDOM_SEED = 4378
INFERENCE_SUBSTRATE = _previous.INFERENCE_SUBSTRATE

KV260_SSH_PRECONDITION = _previous.KV260_SSH_PRECONDITION
KV260_LISTAPPS_COMMAND = _previous.KV260_LISTAPPS_COMMAND
KV260_UIO_COMMAND = _previous.KV260_UIO_COMMAND

REQUIRED_ARTIFACT_FIELDS = _previous.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = dict(_previous.FIELD_PRINCIPLES)

CommandProbe = _previous.CommandProbe
CommandRunner = _previous.CommandRunner
Clock = _previous.Clock

run_command = _previous.run_command
prepend_oss_cad_suite = _previous.prepend_oss_cad_suite
command_to_string = _previous.command_to_string
payload_checksum = _previous.payload_checksum


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4378 artifact from live or injected KV260 SSH probes."""
    root = Path(repo_root)
    artifact = _previous.build_artifact(
        repo_root=root,
        command_runner=command_runner,
        clock=clock,
    )
    artifact.update(
        {
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "random_seed": RANDOM_SEED,
            "source_context": _source_context(root),
        }
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = ""
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return the terminal-prefixed Exp 4378 verdict."""
    return _previous.honest_verdict(artifact)


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4378 and write the requested KV260 continuity JSON artifact."""
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("schema") == SCHEMA, "schema must identify KV260 continuity")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4378")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4378")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4378")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _base._validate_principles(artifact)
    _base._validate_bare_bool(artifact, "kv260_reachable")
    _base._validate_bare_bool(artifact, "kv260_terminal_state_reached")
    _base._validate_preconditions(artifact)
    _base._validate_overlay(artifact)
    _base._validate_uio(artifact)
    _base._validate_transcript(artifact)
    _validate_source_context(artifact)
    _require(
        "mmcblk" not in json.dumps(artifact, sort_keys=True, default=str).lower(),
        "SD-card precondition marker is forbidden",
    )
    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be hardware_smoke",
    )
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "stale verdict")
    _require(artifact.get("fabric_acceleration_claimed") is False, "no fabric acceleration claim")
    _require(artifact.get("speedup_claim_made") is False, "no speedup claim")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _source_context(repo_root: Path) -> dict[str, Any]:
    prior_path = repo_root / "results" / "experiment_4367_hardware_continuity_kv260.json"
    prior_payload = _base._read_json(prior_path)
    return {
        "previous_experiment": 4367,
        "most_recent_hardware_continuity_artifact": str(prior_path),
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_kv260_reachable": prior_payload.get("kv260_reachable"),
    }


def _validate_source_context(artifact: dict[str, Any]) -> None:
    source = artifact.get("source_context")
    _require(isinstance(source, dict), "source_context must be present")
    _require(source.get("previous_experiment") == 4367, "source_context must read Exp 4367")
    _require(
        str(source.get("most_recent_hardware_continuity_artifact", "")).endswith(
            "experiment_4367_hardware_continuity_kv260.json"
        ),
        "source_context must point at Exp 4367",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"kv260_reachable: {artifact['kv260_reachable']}")
    print(f"loaded_overlay: {artifact['loaded_overlay']}")
    print(f"kv260_terminal_state_reached: {artifact['kv260_terminal_state_reached']}")


if __name__ == "__main__":  # pragma: no cover
    main()
