"""Run the second V635 ARC selfparse session with isolated task identity.

Experiment 7206 already owns the audited CUDA, llama.cpp, heartbeat, ARC, and
receipt logic. This module changes only immutable session-B inputs while that
runtime executes. It restores the imported module afterward so one test process
cannot leak session-B identity into session-A checks.

Spec refs: REQ-ARC-WMTE-7207 and SCENARIO-ARC-WMTE-7207-*.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from carnot import experiment_7206_v635_arc_volume_a as base


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]

TASK_ID = "exp7207-arc-volume-b"
EXPERIMENT_ID = 7207
MILESTONE = "2026.09.635"
RUN_DATE = "20260911"
GAME = "r11l"
RANDOM_SEED = 7_207_001
ACTION_BUDGET = 4000
SESSION_TIMEOUT_S = 3600
INDUCTION_TIMEOUT_S = 2400
HARD_CAP_S = 4800
N_CTX = 49152
COMPLETION_BUDGET = 4096
EVIDENCE_TARGET = 10
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]
EXPECTED_PRIOR_VERDICT = "blocked_required_source_bytes"

SCHEMA = "carnot.experiment_7207.arc_volume_b.v1"
DRIVING_REQUIREMENT = "REQ-ARC-WMTE-7207"
MODULE_PATH = Path("python/carnot/experiment_7207_v635_arc_volume_b.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7207_v635_arc_volume_b.py")
TEST_PATH = Path("tests/python/test_experiment_7207_v635_arc_volume_b.py")
RESULT_PATH = Path("results/experiment_7207_v635_arc_volume_b.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7207_v635_arc_volume_b/running.json")
CHECKPOINT_SCHEMA = "carnot.experiment_7207.checkpoint.v1"
RAW_DIR = Path("results/raw/experiment_7207")
SIBLING_PATH = Path("results/experiment_7206_v635_arc_volume_a.json")
SIBLING_TASK_ID = "exp7206-arc-volume-a"
BASE_MODULE_PATH = Path("python/carnot/experiment_7206_v635_arc_volume_a.py")

EXPECTED_TASK_CONTRACT = {
    "id": TASK_ID,
    "milestone": MILESTONE,
    "deliverable": RESULT_PATH.as_posix(),
    "gated_on": None,
    "prior_failures": [
        {
            "experiment_id": "exp7186-arc-withheld-transfer",
            "verdict": EXPECTED_PRIOR_VERDICT,
            "addressed_by": (
                "Use the shipped direct run_game path and the successful Exp7193 runtime; "
                "no invented runner is required."
            ),
            "retire_if_same_verdict": True,
        }
    ],
    "operator_override": (
        "2026-09-11 operator directive in ops/known-issues.md: accumulate separate selfparse "
        "sessions after Exp7193; each new seed contributes unique induction evidence under "
        "its own wall cap."
    ),
}

_REPLACED_SOURCES = {base.MODULE_PATH, base.WRAPPER_PATH, base.TEST_PATH}
REQUIRED_SOURCE_PATHS = tuple(
    path for path in base.REQUIRED_SOURCE_PATHS if path not in _REPLACED_SOURCES
) + (BASE_MODULE_PATH, MODULE_PATH, WRAPPER_PATH, TEST_PATH)

_BASE_OVERRIDES: dict[str, Any] = {
    "TASK_ID": TASK_ID,
    "EXPERIMENT_ID": EXPERIMENT_ID,
    "MILESTONE": MILESTONE,
    "RUN_DATE": RUN_DATE,
    "GAME": GAME,
    "RANDOM_SEED": RANDOM_SEED,
    "ACTION_BUDGET": ACTION_BUDGET,
    "SESSION_TIMEOUT_S": SESSION_TIMEOUT_S,
    "INDUCTION_TIMEOUT_S": INDUCTION_TIMEOUT_S,
    "HARD_CAP_S": HARD_CAP_S,
    "N_CTX": N_CTX,
    "COMPLETION_BUDGET": COMPLETION_BUDGET,
    "EVIDENCE_TARGET": EVIDENCE_TARGET,
    "MODEL_ID": MODEL_ID,
    "QUANTIZATION": QUANTIZATION,
    "MODEL_SPECS": MODEL_SPECS,
    "EXPECTED_PRIOR_VERDICT": EXPECTED_PRIOR_VERDICT,
    "SCHEMA": SCHEMA,
    "DRIVING_REQUIREMENT": DRIVING_REQUIREMENT,
    "MODULE_PATH": MODULE_PATH,
    "WRAPPER_PATH": WRAPPER_PATH,
    "TEST_PATH": TEST_PATH,
    "RESULT_PATH": RESULT_PATH,
    "CHECKPOINT_PATH": CHECKPOINT_PATH,
    "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
    "RAW_DIR": RAW_DIR,
    "SIBLING_PATH": SIBLING_PATH,
    "SIBLING_TASK_ID": SIBLING_TASK_ID,
    "EXPECTED_TASK_CONTRACT": EXPECTED_TASK_CONTRACT,
    "REQUIRED_SOURCE_PATHS": REQUIRED_SOURCE_PATHS,
}

_REUSED_FIELDS = (
    "TASK_ID",
    "MILESTONE",
    "RUN_DATE",
    "GAME",
    "RANDOM_SEED",
    "ACTION_BUDGET",
    "SESSION_TIMEOUT_S",
    "INDUCTION_TIMEOUT_S",
    "N_CTX",
    "COMPLETION_BUDGET",
    "RESULT_PATH",
    "CHECKPOINT_PATH",
    "RAW_DIR",
    "WRAPPER_PATH",
)


@contextmanager
def configured_runtime() -> Iterator[Any]:
    """Apply session-B constants only while the reused runtime is active."""

    base_before = {name: getattr(base, name) for name in _BASE_OVERRIDES}
    reused_before = {name: getattr(base.reused, name) for name in _REUSED_FIELDS}
    try:
        for name, value in _BASE_OVERRIDES.items():
            setattr(base, name, deepcopy(value))
        yield base
    finally:
        for name, value in base_before.items():
            setattr(base, name, value)
        for name, value in reused_before.items():
            setattr(base.reused, name, value)


def run_experiment(
    *,
    root: Path,
    run_date: str,
    result_path: Path,
    checkpoint_path: Path,
    raw_dir: Path,
    live_runner: Callable[..., Any] | None = None,
) -> JsonDict:
    """Run one bounded session through the audited session-A implementation."""

    with configured_runtime() as configured:
        return configured.run_experiment(
            root=root,
            run_date=run_date,
            result_path=result_path,
            checkpoint_path=checkpoint_path,
            raw_dir=raw_dir,
            live_runner=live_runner,
        )


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Validate an Exp7207 artifact against the configured shared schema."""

    with configured_runtime() as configured:
        return configured.validate_artifact(value)


def run_session_child(args: Any) -> int:
    """Run the isolated native-model child with session-B paths and seed."""

    with configured_runtime() as configured:
        return int(configured.run_session_child(args))


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the configured driver, child, or cold-validation command."""

    with configured_runtime() as configured:
        return int(configured.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
