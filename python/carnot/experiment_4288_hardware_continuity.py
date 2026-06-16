"""Exp 4288 opportunistic per-board hardware continuity status artifact.

This module keeps KV260, PolarFire, and GateMate visible without making any
single board a milestone blocker. KV260 is checked only through board SSH
reachability, then `xmutil listapps` when reachable. PolarFire receives an
opportunistic hash-verified CPU dispatch smoke when reachable. GateMate records
the DirtyJTAG IDCODE when detected, or an honest per-board blocked status.

Spec refs: REQ-HW-4288, SCENARIO-HW-4288.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4267_hardware_continuity as _state
from carnot import experiment_4278_hardware_continuity as _prev


EXPERIMENT_ID = 4288
SCHEMA = _prev.SCHEMA
SPEC_REFS = ["REQ-HW-4288", "SCENARIO-HW-4288"]
OUTPUT_REL_PATH = Path("results") / "experiment_4288_hardware_continuity.json"
RANDOM_SEED = 4288
INFERENCE_SUBSTRATE = _prev.INFERENCE_SUBSTRATE
BOARD_NAMES = _prev.BOARD_NAMES

KV260_SSH_PRECONDITION = _prev.KV260_SSH_PRECONDITION
POLARFIRE_SSH_PRECONDITION = _prev.POLARFIRE_SSH_PRECONDITION
GATEMATE_DETECT_COMMAND = _prev.GATEMATE_DETECT_COMMAND
KV260_LISTAPPS_COMMAND = _prev.KV260_LISTAPPS_COMMAND

REQUIRED_ARTIFACT_FIELDS = _prev.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = dict(_prev.FIELD_PRINCIPLES)

CommandProbe = _prev.CommandProbe
StepOutcome = _prev.StepOutcome
CommandRunner = _prev.CommandRunner
Clock = _prev.Clock
StepRunner = _prev.StepRunner

run_command = _prev.run_command
prepend_oss_cad_suite = _prev.prepend_oss_cad_suite
command_to_string = _prev.command_to_string
payload_checksum = _prev.payload_checksum
state_token = _state.state_token


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    polarfire_step_runner: StepRunner | None = None,
) -> dict[str, Any]:
    """Build the Exp 4288 artifact from live or injected board probes."""
    root = Path(repo_root)
    artifact = _prev.build_artifact(
        repo_root=root,
        command_runner=command_runner,
        clock=clock,
        polarfire_step_runner=polarfire_step_runner,
    )
    artifact.update(
        {
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "random_seed": RANDOM_SEED,
            "source_context": _source_context(root),
            "honest_verdict": "",
            "reproducibility_checksum": "",
        }
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return a terminal-prefixed verdict with all board states visible."""
    return (
        "complete: hardware_continuity_4288_"
        f"kv260_{state_token(str(artifact['kv260_step_taken']))}_"
        f"polarfire_{state_token(str(artifact['polarfire_step_taken']))}_"
        f"gatemate_{state_token(str(artifact['gatemate_step_taken']))}_"
        "ssh_usb_reachability_only"
    )


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    polarfire_step_runner: StepRunner | None = None,
) -> Path:
    """Run Exp 4288 and write `results/experiment_4288_hardware_continuity.json`."""
    prepend_oss_cad_suite()
    artifact = build_artifact(
        repo_root=repo_root,
        command_runner=command_runner,
        clock=clock,
        polarfire_step_runner=polarfire_step_runner,
    )
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4288")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4288")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4288")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4288")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _state._validate_principles(artifact)
    _require(
        "mmcblk" not in json.dumps(artifact, sort_keys=True, default=str).lower(),
        "SD-card precondition marker is forbidden",
    )
    _state._validate_preconditions(artifact)
    _state._validate_per_board_status(artifact)
    _validate_source_context(artifact)
    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be hardware_smoke",
    )
    verdict = str(artifact.get("honest_verdict", ""))
    _require(
        verdict.startswith("complete:") or verdict.startswith("blocked_"),
        "honest_verdict must have a terminal prefix",
    )
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "stale verdict")
    _require(artifact.get("fabric_acceleration_claimed") is False, "no fabric acceleration claim")
    _require(artifact.get("speedup_claim_made") is False, "no speedup claim")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _source_context(repo_root: Path) -> dict[str, Any]:
    prior_path = repo_root / "results" / "experiment_4278_hardware_continuity.json"
    prior_payload = _read_json(prior_path)
    return {
        "previous_experiment": 4278,
        "most_recent_hardware_continuity_artifact": str(prior_path),
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_per_board_reachability": prior_payload.get("per_board_reachability"),
        "previous_per_board_status": prior_payload.get("per_board_status"),
        "previous_kv260_step_taken": prior_payload.get("kv260_step_taken"),
        "previous_polarfire_step_taken": prior_payload.get("polarfire_step_taken"),
        "previous_gatemate_step_taken": prior_payload.get("gatemate_step_taken"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}  # pragma: no cover - live script usually has the prior artifact.
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt local artifact.
        return {}


def _validate_source_context(artifact: dict[str, Any]) -> None:
    source = artifact.get("source_context")
    _require(isinstance(source, dict), "source_context must be present")
    _require(source.get("previous_experiment") == 4278, "source_context must read Exp 4278")
    _require(
        str(source.get("most_recent_hardware_continuity_artifact", "")).endswith(
            "experiment_4278_hardware_continuity.json"
        ),
        "source_context must point at Exp 4278",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)  # pragma: no cover


def main() -> None:  # pragma: no cover
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"per_board_status: {artifact['per_board_status']}")


if __name__ == "__main__":  # pragma: no cover
    main()
