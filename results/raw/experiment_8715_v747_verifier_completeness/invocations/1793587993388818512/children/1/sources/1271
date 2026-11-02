"""Exp 4878 KV260 SSH-only continuity artifact.

Spec refs: REQ-HW-4878, SCENARIO-HW-4878.

This continuity check exists to prevent a quiet "no file changes" skip when the
board is offline. SSH is the only valid KV260 precondition; a failed SSH probe
still produces the blocked artifact that the conductor expects.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time


if __package__ in {None, ""}:  # pragma: no cover - direct script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot import experiment_4867_kv260_continuity as base


EXPERIMENT_ID = 4878
SCHEMA = "carnot.kv260_ssh_continuity.v12"
SPEC_REFS = ["REQ-HW-4878", "SCENARIO-HW-4878"]
OUTPUT_REL_PATH = Path("results") / "experiment_4878_kv260_continuity.json"
RANDOM_SEED = 4878
INFERENCE_SUBSTRATE = base.INFERENCE_SUBSTRATE

CommandProbe = base.CommandProbe
CommandRunner = base.CommandRunner
Clock = base.Clock

KV260_SSH_COMMAND = base.KV260_SSH_COMMAND
KV260_LISTAPPS_COMMAND = base.KV260_LISTAPPS_COMMAND
KV260_LISTAPPS_SUDO_COMMAND = base.KV260_LISTAPPS_SUDO_COMMAND
KV260_BOARD_STATE_COMMAND = base.KV260_BOARD_STATE_COMMAND
VALID_OVERLAYS = base.VALID_OVERLAYS

SUCCESS_VERDICT = "success_kv260_continuity_ok"
BLOCKED_SSH_VERDICT = "blocked_kv260_ssh_unreachable"
REACHABLE_NEXT_FORWARD_STEP = base.REACHABLE_NEXT_FORWARD_STEP
BLOCKED_NEXT_FORWARD_STEP = base.BLOCKED_NEXT_FORWARD_STEP

REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "kv260_ssh_reachable",
    "preconditions_checked",
)
REQUIRED_ARTIFACT_FIELDS = (
    *REQUIRED_PRINCIPLE_FIELDS,
    "schema",
    "experiment",
    "spec_refs",
    "field_principles",
    "duration_s",
    "command_probes",
    "loaded_overlay",
    "xmutil_requires_sudo",
    "board_state",
    "next_forward_step",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; reachable is success_kv260_continuity_ok; unreachable is "
        "blocked_kv260_ssh_unreachable -- artifact MUST be written either way "
        "(no 'No file changes' 3-fail-skip)."
    ),
    "inference_substrate": "hardware_smoke (SSH-attached board test).",
    "kv260_ssh_reachable": (
        "the SSH check -- false when offline; the artifact is still written."
    ),
    "preconditions_checked": (
        "records the SSH check; a blocked artifact is correct when the board is offline."
    ),
}

command_to_string = base.command_to_string
loaded_overlay_from_xmutil = base.loaded_overlay_from_xmutil
payload_checksum = base.payload_checksum
run_command = base.run_command


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, object]:
    """Build the Exp 4878 artifact from live or injected SSH board probes."""
    artifact = base.build_artifact(command_runner=command_runner, clock=clock)
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "field_principles": dict(FIELD_PRINCIPLES),
            "random_seed": RANDOM_SEED,
        }
    )
    if artifact.get("kv260_ssh_reachable") is False:
        artifact["honest_verdict"] = BLOCKED_SSH_VERDICT
        artifact["next_forward_step"] = BLOCKED_NEXT_FORWARD_STEP
    else:
        artifact["honest_verdict"] = SUCCESS_VERDICT
        artifact["next_forward_step"] = REACHABLE_NEXT_FORWARD_STEP
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(repo_root: str | Path, artifact: dict[str, object]) -> Path:
    """Write a validated Exp 4878 artifact under the requested repository root."""
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
    """Run Exp 4878 and write `results/experiment_4878_kv260_continuity.json`."""
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate the SSH-only continuity schema and checksum."""
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    _require(artifact.get("schema") == SCHEMA, "schema mismatch")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment mismatch")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle must be false")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    _require(float(artifact.get("duration_s", 0.0)) > 0.0, "duration_s must be positive")
    for field in REQUIRED_PRINCIPLE_FIELDS:
        _require(field in artifact, f"missing principle field: {field}")
        _require(
            not (isinstance(artifact[field], dict) and "principle" in artifact[field]),
            f"{field} must remain a bare value",
        )
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    _require(
        "mmcblk" not in encoded and "/dev/disk" not in encoded,
        "forbidden host storage marker",
    )
    _validate_verdict(artifact)
    _validate_base_contract(artifact)
    _require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "bad checksum",
    )


def _validate_verdict(artifact: dict[str, object]) -> None:
    if artifact.get("kv260_ssh_reachable") is False:
        _require(artifact.get("honest_verdict") == BLOCKED_SSH_VERDICT, "bad blocked verdict")
    else:
        _require(artifact.get("honest_verdict") == SUCCESS_VERDICT, "bad success verdict")


def _validate_base_contract(artifact: dict[str, object]) -> None:
    base_artifact = dict(artifact)
    base_artifact.update(
        {
            "schema": base.SCHEMA,
            "experiment": base.EXPERIMENT_ID,
            "spec_refs": list(base.SPEC_REFS),
            "field_principles": dict(base.FIELD_PRINCIPLES),
            "random_seed": base.RANDOM_SEED,
        }
    )
    if base_artifact.get("kv260_ssh_reachable") is False:
        base_artifact["honest_verdict"] = base.BLOCKED_SSH_VERDICT
        base_artifact["next_forward_step"] = base.BLOCKED_NEXT_FORWARD_STEP
    else:
        base_artifact["honest_verdict"] = base.SUCCESS_VERDICT
        base_artifact["next_forward_step"] = base.REACHABLE_NEXT_FORWARD_STEP
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
    print(f"loaded_overlay: {artifact['loaded_overlay']}")
    print(f"next_forward_step: {artifact['next_forward_step']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
