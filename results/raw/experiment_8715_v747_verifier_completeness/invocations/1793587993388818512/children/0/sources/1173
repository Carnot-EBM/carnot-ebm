"""Exp 4777 KV260 SSH-only continuity artifact.

Spec refs: REQ-HW-4777, SCENARIO-HW-4777.

This experiment is deliberately a board-state continuity record rather than a
new performance claim. The useful evidence is that the KV260 is reachable over
SSH, the currently visible overlay state is preserved through `xmutil`, and the
next board step is explicit. Host SD-card device nodes are retired for KV260, so
they never appear as a precondition here.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any


if __package__ in {None, ""}:  # pragma: no cover - direct script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot import experiment_4767_kv260_continuity as base


EXPERIMENT_ID = 4777
SCHEMA = "carnot.kv260_ssh_continuity.v3"
SPEC_REFS = ["REQ-HW-4777", "SCENARIO-HW-4777"]
OUTPUT_REL_PATH = Path("results") / "experiment_4777_kv260_continuity.json"
RANDOM_SEED = 4777
INFERENCE_SUBSTRATE = base.INFERENCE_SUBSTRATE

CommandProbe = base.CommandProbe
CommandRunner = base.CommandRunner
Clock = base.Clock

KV260_SSH_COMMAND = base.KV260_SSH_COMMAND
KV260_LISTAPPS_COMMAND = base.KV260_LISTAPPS_COMMAND
KV260_LISTAPPS_SUDO_COMMAND = base.KV260_LISTAPPS_SUDO_COMMAND
KV260_BOARD_STATE_COMMAND = base.KV260_BOARD_STATE_COMMAND
VALID_OVERLAYS = base.VALID_OVERLAYS

SUCCESS_VERDICT = base.SUCCESS_VERDICT
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
        "terminal prefix; reachable + state recorded is success_/complete_; "
        "unreachable is blocked_kv260_ssh_unreachable."
    ),
    "inference_substrate": "hardware_smoke -- SSH-attached board test.",
    "kv260_ssh_reachable": (
        "the SSH reachability check -- the ONLY valid KV260 precondition "
        "(not host SD-card presence)."
    ),
    "preconditions_checked": (
        "records the SSH check so a wrong-mechanism precondition cannot "
        "escalate the operator for a no-op."
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
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(repo_root: str | Path, artifact: dict[str, object]) -> Path:
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def validate_artifact(artifact: dict[str, object]) -> None:
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
    _require("mmcblk" not in encoded and "/dev/disk" not in encoded, "forbidden host storage marker")
    _validate_precondition(artifact)

    if artifact.get("kv260_ssh_reachable") is False:
        _validate_blocked_artifact(artifact)
        return

    _validate_reachable_artifact(artifact)


def _validate_blocked_artifact(artifact: dict[str, object]) -> None:
    _require(artifact.get("honest_verdict") == BLOCKED_SSH_VERDICT, "bad blocked SSH verdict")
    _require(artifact.get("loaded_overlay") is None, "blocked SSH cannot report overlay")
    _require(artifact.get("next_forward_step") == BLOCKED_NEXT_FORWARD_STEP, "bad blocked next step")
    probes = _command_probes(artifact)
    _require(probes.get("kv260_xmutil_listapps") is None, "blocked SSH cannot run xmutil")
    _require(probes.get("kv260_board_state") is None, "blocked SSH cannot run board state")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _validate_reachable_artifact(artifact: dict[str, object]) -> None:
    _require(artifact.get("kv260_ssh_reachable") is True, "kv260_ssh_reachable must be bool")
    _require(artifact.get("honest_verdict") == SUCCESS_VERDICT, "bad success verdict")
    _require(artifact.get("next_forward_step") == REACHABLE_NEXT_FORWARD_STEP, "bad reachable next step")
    loaded_overlay = artifact.get("loaded_overlay")
    _require(loaded_overlay is None or loaded_overlay in VALID_OVERLAYS, "invalid loaded overlay")
    _require(isinstance(artifact.get("xmutil_requires_sudo"), bool), "xmutil_requires_sudo must be bool")
    probes = _command_probes(artifact)
    _require(probes.get("kv260_xmutil_listapps") is not None, "success requires xmutil probe")
    if artifact.get("xmutil_requires_sudo"):
        _require(probes.get("kv260_xmutil_listapps_sudo") is not None, "sudo fallback missing")
    _require(probes.get("kv260_board_state") is not None, "success requires board state probe")
    board_state = artifact.get("board_state")
    _require(isinstance(board_state, dict), "board_state must be a dict")
    _require(board_state.get("captured") is True, "success requires captured board state")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _validate_precondition(artifact: dict[str, object]) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(isinstance(preconditions, list) and len(preconditions) == 1, "bad preconditions_checked")
    entry = preconditions[0]
    _require(isinstance(entry, dict), "bad precondition entry")
    _require(entry.get("resource") == "kv260_ssh", "bad KV260 precondition resource")
    _require(entry.get("command") == command_to_string(KV260_SSH_COMMAND), "bad KV260 SSH command")
    _require(entry.get("discipline") == "ssh_only_no_host_sd_card", "bad KV260 discipline")
    _require(entry.get("available") is artifact.get("kv260_ssh_reachable"), "precondition mismatch")


def _command_probes(artifact: dict[str, object]) -> dict[str, Any]:
    probes = artifact.get("command_probes")
    _require(isinstance(probes, dict), "command_probes must be a dict")
    return probes


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
