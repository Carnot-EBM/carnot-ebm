"""Exp 4240 per-board hardware continuity status artifact.

This continuation keeps every attached board visible with exactly one SSH or
USB-detect precondition, a board-local wall-clock timer, and a concrete next
step. GateMate and PolarFire remain non-terminal; KV260 is terminal and is only
confirmed opportunistically over SSH. Host SD-card checks are deliberately
excluded by the KV260 SSH-Not-SD-Card discipline.

Spec refs: REQ-HW-4240, SCENARIO-HW-4240.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4194_hardware_continuity as _validation
from carnot import experiment_4228_hardware_continuity as _previous


EXPERIMENT_ID = 4240
SCHEMA = "carnot.hardware_continuity_per_board_status.v14"
SPEC_REFS = ["REQ-HW-4240", "SCENARIO-HW-4240"]
OUTPUT_REL_PATH = Path("results") / "experiment_4240_hardware_continuity.json"
RANDOM_SEED = 4240
INFERENCE_SUBSTRATE = _previous.INFERENCE_SUBSTRATE
BOARD_NAMES = _previous.BOARD_NAMES

KV260_SSH_PRECONDITION = _previous.KV260_SSH_PRECONDITION
GATEMATE_DETECT_COMMAND = _previous.GATEMATE_DETECT_COMMAND
POLARFIRE_SSH_PRECONDITION = _previous.POLARFIRE_SSH_PRECONDITION
GATEMATE_FLASH_BOARD = _previous.GATEMATE_FLASH_BOARD

REQUIRED_ARTIFACT_FIELDS = _previous.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = dict(_previous.FIELD_PRINCIPLES)

CommandProbe = _previous.CommandProbe
StepOutcome = _previous.StepOutcome
CommandRunner = _previous.CommandRunner
Clock = _previous.Clock
StepRunner = _previous.StepRunner

run_command = _previous.run_command
prepend_oss_cad_suite = _previous.prepend_oss_cad_suite
payload_checksum = _previous.payload_checksum
state_token = _previous.state_token


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    gatemate_step_runner: StepRunner | None = None,
    polarfire_step_runner: StepRunner | None = None,
) -> dict[str, Any]:
    """Build the Exp 4240 artifact from live or injected SSH/USB probes."""
    root = Path(repo_root)
    artifact = _previous.build_artifact(
        repo_root=root,
        command_runner=command_runner,
        clock=clock,
        gatemate_step_runner=gatemate_step_runner,
        polarfire_step_runner=polarfire_step_runner,
    )
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "random_seed": RANDOM_SEED,
            "source_context": _source_context(root),
        }
    )
    artifact["field_principles"].update(FIELD_PRINCIPLES)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return a terminal-prefixed verdict that keeps all board statuses visible."""
    return (
        "complete: hardware_continuity_4240_"
        f"gatemate_{state_token(str(artifact['gatemate_step_taken']))}_"
        f"polarfire_{state_token(str(artifact['polarfire_step_taken']))}_"
        f"kv260_{state_token(str(artifact['per_board_terminal_state']['kv260']))}_"
        "ssh_usb_detect_only"
    )


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4240 and write `results/experiment_4240_hardware_continuity.json`."""
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4240")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4240")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4240")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4240")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _validation._validate_principles(artifact)
    _require(
        "mmcblk" not in json.dumps(artifact, sort_keys=True, default=str).lower(),
        "SD-card precondition marker is forbidden",
    )
    _validation._validate_preconditions(artifact)
    _validation._validate_per_board_status(artifact)
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
    prior_path = repo_root / "results" / "experiment_4228_hardware_continuity.json"
    prior_payload = _read_json(prior_path)
    gate_step = prior_payload.get("gatemate_step", {}) if isinstance(prior_payload, dict) else {}
    return {
        "previous_experiment": 4228,
        "most_recent_hardware_continuity_artifact": str(prior_path),
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_per_board_reachability": prior_payload.get("per_board_reachability"),
        "previous_gatemate_step_taken": prior_payload.get("gatemate_step_taken"),
        "previous_polarfire_step_taken": prior_payload.get("polarfire_step_taken"),
        "previous_gatemate_next_concrete_step": (
            gate_step.get("next_concrete_step") if isinstance(gate_step, dict) else None
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt local artifact.
        return {}


def _validate_source_context(artifact: dict[str, Any]) -> None:
    source = artifact.get("source_context")
    _require(isinstance(source, dict), "source_context must be present")
    _require(source.get("previous_experiment") == 4228, "source_context must read Exp 4228")
    _require(
        str(source.get("most_recent_hardware_continuity_artifact", "")).endswith(
            "experiment_4228_hardware_continuity.json"
        ),
        "source_context must point at Exp 4228",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"per_board_status: {artifact['per_board_status']}")


if __name__ == "__main__":  # pragma: no cover
    main()
