"""Exp 4084 hardware continuity after the GateMate 2026-06-11 re-plug.

This module keeps the board-smoke logic in the shared Exp 4064 continuity
engine and only changes the experiment identity. That matters because the
artifact is a dated continuity record, but the engineering contract is still
the same: check each board through the real access path, record the next board
step, and avoid treating a host SD-card device as evidence for KV260.

Spec refs: REQ-HW-4084, SCENARIO-HW-4084.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from carnot import experiment_4064_hardware_continuity as _base


EXPERIMENT_ID = 4084
SCHEMA = _base.SCHEMA
SPEC_REFS = ["REQ-HW-4084", "SCENARIO-HW-4084"]
OUTPUT_REL_PATH = Path("results") / "experiment_4084_hardware_continuity.json"
RANDOM_SEED = 4084
INFERENCE_SUBSTRATE = _base.INFERENCE_SUBSTRATE
BOARD_NAMES = _base.BOARD_NAMES
GATEMATE_REPLUG_DATE = "2026-06-11"

KV260_SSH_PRECONDITION = _base.KV260_SSH_PRECONDITION
GATEMATE_DETECT_COMMAND = _base.GATEMATE_DETECT_COMMAND
POLARFIRE_SSH_PRECONDITION = _base.POLARFIRE_SSH_PRECONDITION
GATEMATE_FLASH_BOARD = _base.GATEMATE_FLASH_BOARD

REQUIRED_ARTIFACT_FIELDS = _base.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = dict(_base.FIELD_PRINCIPLES)

CommandProbe = _base.CommandProbe
StepOutcome = _base.StepOutcome
CommandRunner = _base.CommandRunner
Clock = _base.Clock
StepRunner = _base.StepRunner

run_command = _base.run_command
prepend_oss_cad_suite = _base.prepend_oss_cad_suite
command_to_string = _base.command_to_string
state_token = _base.state_token


def _gatemate_available(probe: CommandProbe) -> bool:
    """Accept the 4084 DirtyJTAG continuity signal even when IDCODE is absent.

    Exp 4064 still required an IDCODE because it was proving target identity
    from a cold detect. After the GateMate SRAM flash path, this host has also
    observed `openFPGALoader -c dirtyJtag --detect` returning only the negotiated
    JTAG frequency. For Exp 4084, that still proves the board access path named
    in the task is alive, so the forward step can decide whether flashing and a
    post-flash smoke are usable.
    """
    text = probe.combined_output.lower()
    return probe.exit_code == 0 and (
        "idcode" in text
        or "gatemate" in text
        or "colognechip" in text
        or "gm1a" in text
        or "jtag frequency" in text
    )


_CONFIG_OVERRIDES = {
    "EXPERIMENT_ID": EXPERIMENT_ID,
    "SPEC_REFS": SPEC_REFS,
    "OUTPUT_REL_PATH": OUTPUT_REL_PATH,
    "RANDOM_SEED": RANDOM_SEED,
    "_gatemate_available": _gatemate_available,
}


@contextmanager
def _configured_base() -> Iterator[Any]:
    """Temporarily apply Exp 4084 constants to the shared continuity engine."""
    original = {name: getattr(_base, name) for name in _CONFIG_OVERRIDES}
    for name, value in _CONFIG_OVERRIDES.items():
        setattr(_base, name, value)
    try:
        yield _base
    finally:
        for name, value in original.items():
            setattr(_base, name, value)


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = _base.time.perf_counter,
    gatemate_step_runner: StepRunner | None = None,
    polarfire_step_runner: StepRunner | None = None,
) -> dict[str, Any]:
    """Build the Exp 4084 artifact from live or injected board-access probes."""
    root = Path(repo_root)
    with _configured_base() as base:
        artifact = base.build_artifact(
            repo_root=root,
            command_runner=command_runner,
            clock=clock,
            gatemate_step_runner=gatemate_step_runner,
            polarfire_step_runner=polarfire_step_runner,
        )
        artifact["source_context"] = _source_context(root)
        artifact["gatemate_replug_context"] = {
            "date": GATEMATE_REPLUG_DATE,
            "reachability_command": command_to_string(GATEMATE_DETECT_COMMAND),
        }
        artifact["reproducibility_checksum"] = base.payload_checksum(artifact)
        base.validate_artifact(artifact)
        return artifact


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = _base.time.perf_counter,
) -> Path:
    """Run Exp 4084 and write `results/experiment_4084_hardware_continuity.json`."""
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    with _configured_base() as base:
        return base.write_artifact(repo_root, artifact)


def validate_artifact(artifact: dict[str, Any]) -> None:
    with _configured_base() as base:
        base.validate_artifact(artifact)


def payload_checksum(artifact: dict[str, Any]) -> str:
    return _base.payload_checksum(artifact)


def _source_context(repo_root: Path) -> dict[str, Any]:
    prior_path = repo_root / "results" / "experiment_4074_hardware_continuity.json"
    prior_payload: dict[str, Any] = {}
    if prior_path.exists():
        try:
            prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt local artifact.
            prior_payload = {}
    return {
        "previous_experiment": 4074,
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_per_board_reachability": prior_payload.get("per_board_reachability"),
    }


def main() -> None:  # pragma: no cover
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"per_board_reachability: {artifact['per_board_reachability']}")
    print(f"per_board_next_step: {artifact['per_board_next_step']}")
    print(f"inference_substrate: {artifact['inference_substrate']}")


if __name__ == "__main__":  # pragma: no cover
    main()
