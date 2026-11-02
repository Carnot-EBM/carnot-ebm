"""Exp 4745 KV260 SSH-gated Ising latency continuity artifact.

Spec refs: REQ-HW-4745, SCENARIO-HW-4745.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:  # pragma: no cover - direct script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot import experiment_4733_kv260_continuity as _base  # noqa: E402


EXPERIMENT_ID = 4745
SCHEMA = _base.SCHEMA
SPEC_REFS = ["REQ-HW-4745", "SCENARIO-HW-4745"]
OUTPUT_REL_PATH = Path("results") / "experiment_4745_kv260_continuity.json"
RANDOM_SEED = 4745
INFERENCE_SUBSTRATE = _base.INFERENCE_SUBSTRATE

KV260_SSH_COMMAND = _base.KV260_SSH_COMMAND
KV260_LISTAPPS_COMMAND = _base.KV260_LISTAPPS_COMMAND
KV260_LISTAPPS_SUDO_COMMAND = _base.KV260_LISTAPPS_SUDO_COMMAND
KV260_LATENCY_COMMAND = _base.KV260_LATENCY_COMMAND
KV260_BITSTREAM_SHA_COMMAND = _base.KV260_BITSTREAM_SHA_COMMAND

VALID_OVERLAYS = _base.VALID_OVERLAYS
LOAD_APP_PREFERENCE = _base.LOAD_APP_PREFERENCE

BOARD_SAMPLE_COUNT = _base.BOARD_SAMPLE_COUNT
BOARD_SPIN_COUNT = _base.BOARD_SPIN_COUNT
BOARD_MAX_DEGREE = _base.BOARD_MAX_DEGREE
BOARD_BETA_FINAL_Q88 = _base.BOARD_BETA_FINAL_Q88

SUCCESS_VERDICT = _base.SUCCESS_VERDICT
BLOCKED_SSH_VERDICT = _base.BLOCKED_SSH_VERDICT

REQUIRED_OPERATOR_FIELDS = _base.REQUIRED_OPERATOR_FIELDS
REQUIRED_ARTIFACT_FIELDS = _base.REQUIRED_ARTIFACT_FIELDS
FIELD_PRINCIPLES = dict(_base.FIELD_PRINCIPLES)
DEFAULT_FIXED_COMPUTE_BUDGET = dict(_base.DEFAULT_FIXED_COMPUTE_BUDGET)
BOARD_HARNESS_SOURCE = (
    _base.BOARD_HARNESS_SOURCE.replace("exp4733", "exp4745")
    .replace("+ 4733", "+ 4745")
    .replace(" 4733)", " 4745)")
)

CommandProbe = _base.CommandProbe
CommandRunner = _base.CommandRunner

command_to_string = _base.command_to_string
extract_board_payload = _base.extract_board_payload
loadapp_command = _base.loadapp_command
payload_checksum = _base.payload_checksum
run_command = _base.run_command
validate_board_payload = _base.validate_board_payload


@contextmanager
def _base_overrides() -> Iterator[None]:
    """Temporarily retarget the shared KV260 harness to Exp 4745 metadata."""
    overrides = {
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "SPEC_REFS": list(SPEC_REFS),
        "OUTPUT_REL_PATH": OUTPUT_REL_PATH,
        "RANDOM_SEED": RANDOM_SEED,
        "FIELD_PRINCIPLES": dict(FIELD_PRINCIPLES),
        "DEFAULT_FIXED_COMPUTE_BUDGET": dict(DEFAULT_FIXED_COMPUTE_BUDGET),
        "BOARD_HARNESS_SOURCE": BOARD_HARNESS_SOURCE,
    }
    previous = {name: getattr(_base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(_base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


def build_artifact(
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Build the Exp 4745 artifact from real or injected KV260 SSH commands."""
    with _base_overrides():
        return _base.build_artifact(command_runner=command_runner, duration_s=duration_s)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    """Write the Exp 4745 continuity artifact to the expected results path."""
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> Path:
    """Run Exp 4745 and write `results/experiment_4745_kv260_continuity.json`."""
    artifact = build_artifact(command_runner=command_runner, duration_s=duration_s)
    return write_artifact(repo_root, artifact)


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Reject artifacts that drift from the Exp 4745 continuity contract."""
    with _base_overrides():
        _base.validate_artifact(artifact)


def main() -> None:  # pragma: no cover - live hardware entrypoint.
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
