"""Consolidated PolarFire plus KV260 continuity audit for Exp 3901.

This experiment keeps the same deliberately narrow claim boundary as Exp 3890.
PolarFire evidence is a soft-CPU SSH dispatch re-confirmed through Exp 3867's
hash verifier. KV260 evidence is SSH reachability, `xmutil listapps`, and
`/dev/uio*` presence. Neither board demonstrates FPGA compute acceleration in
this audit, so the fabric acceleration claim stays false.

Spec refs: REQ-HW-3901, SCENARIO-HW-3901.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_3890_polarfire_kv260_continuity as _base


EXPERIMENT_ID = 3901
SCHEMA = "carnot.polarfire_kv260_continuity.v2"
SPEC_REFS = ["REQ-HW-3901", "SCENARIO-HW-3901"]
OUTPUT_REL_PATH = Path("results") / "experiment_3901_polarfire_kv260_continuity.json"
RANDOM_SEED = 3901
INFERENCE_SUBSTRATE = _base.INFERENCE_SUBSTRATE

POLARFIRE_SSH_PRECONDITION = _base.POLARFIRE_SSH_PRECONDITION
KV260_SSH_PRECONDITION = _base.KV260_SSH_PRECONDITION
KV260_LISTAPPS_COMMAND = _base.KV260_LISTAPPS_COMMAND
KV260_UIO_COMMAND = _base.KV260_UIO_COMMAND
VALID_KV260_OVERLAYS = _base.VALID_KV260_OVERLAYS

REQUIRED_ARTIFACT_FIELDS = _base.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = {
    "polarfire_reachable": (
        "Per-board SSH reachability is the honest precondition; a PolarFire miss "
        "does not suppress the KV260 audit."
    ),
    "kv260_reachable": (
        "Per-board SSH reachability is the honest KV260 precondition; no retired "
        "host-storage check is used."
    ),
    "polarfire_state": (
        "Terminal hash-verified dispatch, soft-CPU only; no fabric-acceleration "
        "claim."
    ),
    "kv260_state": (
        "Loaded overlay plus UIO presence records the terminal or non-terminal "
        "state for Hardware-Task Continuity."
    ),
    "fabric_acceleration_claimed": (
        "Must be false because neither board demonstrates compute acceleration "
        "in this audit."
    ),
    "preconditions_checked": (
        "Hardware-smoke methodology records real SSH board interaction before any "
        "board-specific command."
    ),
    "reproducibility_checksum": (
        "Hardware-smoke methodology gives a content hash so later drift is "
        "visible without model-inference substrate markers."
    ),
    "inference_substrate": (
        "Hardware-smoke methodology identifies this as SSH board interaction, not "
        "a model-inference substrate."
    ),
}

CommandResult = _base.CommandResult
CommandRunner = _base.CommandRunner
Clock = _base.Clock
PolarfireDispatcher = _base.PolarfireDispatcher

command_to_string = _base.command_to_string
run_command = _base.run_command
payload_checksum = _base.payload_checksum
parse_kv260_listapps = _base.parse_kv260_listapps
parse_uio_devices = _base.parse_uio_devices
classify_kv260_state = _base.classify_kv260_state
run_polarfire_dispatch = _base.run_polarfire_dispatch
summarize_polarfire_dispatch = _base.summarize_polarfire_dispatch


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    polarfire_dispatcher: PolarfireDispatcher = run_polarfire_dispatch,
    clock: Clock = time.perf_counter,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Build the Exp 3901 artifact using the audited Exp 3890 board sequence."""
    artifact = _base.build_artifact(
        repo_root=repo_root,
        command_runner=command_runner,
        polarfire_dispatcher=polarfire_dispatcher,
        clock=clock,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "field_principles": dict(FIELD_PRINCIPLES),
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
        }
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema") != SCHEMA:
        raise ValueError("schema must identify the Exp 3901 continuity artifact")
    if artifact.get("experiment") != EXPERIMENT_ID:
        raise ValueError("experiment must be 3901")
    if artifact.get("spec_refs") != SPEC_REFS:
        raise ValueError("spec_refs must reference REQ-HW-3901 and SCENARIO-HW-3901")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 3901")
    _base.validate_artifact(artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    polarfire_dispatcher: PolarfireDispatcher = run_polarfire_dispatch,
    clock: Clock = time.perf_counter,
    duration_s: float | None = None,
) -> Path:
    artifact = build_artifact(
        repo_root=repo_root,
        command_runner=command_runner,
        polarfire_dispatcher=polarfire_dispatcher,
        clock=clock,
        duration_s=duration_s,
    )
    return write_artifact(repo_root, artifact)


def main() -> None:  # pragma: no cover
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"polarfire_state: {artifact['polarfire_state']}")
    print(f"kv260_state: {artifact['kv260_state']}")
    print(f"fabric_acceleration_claimed: {artifact['fabric_acceleration_claimed']}")


if __name__ == "__main__":  # pragma: no cover
    main()
