"""Experiment 4824: held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4824, SCENARIO-CAPSTONE-4824,
SCENARIO-CAPSTONE-4824-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4824-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4814_heldout_first_win_readiness as previous


JsonDict = dict[str, Any]
PreconditionsChecker = previous.PreconditionsChecker
ParityCheck = previous.ParityCheck
ProxyRunner = previous.ProxyRunner
PriorBestLoader = previous.PriorBestLoader
PartialProxyLoader = previous.PartialProxyLoader

base = previous.base
REPO_ROOT = previous.REPO_ROOT
EXPERIMENT = "experiment_4824_heldout_first_win_readiness"
EXPERIMENT_ID = 4824
SCHEMA = "carnot.arc.heldout_first_win_readiness_4824.v1"
RESULT_RELATIVE_PATH = "results/experiment_4824_heldout_first_win_readiness.json"
PARTIAL_RESULT_RELATIVE_PATH = "results/experiment_4824_heldout_first_win_readiness.partial.json"
PROXY_RESULT_RELATIVE_PATH = previous.PROXY_RESULT_RELATIVE_PATH
FIRST_WIN_BASELINE = previous.FIRST_WIN_BASELINE
MIN_HELD_OUT_VARIANT_ATTEMPTS = previous.MIN_HELD_OUT_VARIANT_ATTEMPTS
HELD_OUT_VARIANT_ATTEMPT_FLOOR = previous.HELD_OUT_VARIANT_ATTEMPT_FLOOR
HELD_OUT_VARIANT_IDS = previous.HELD_OUT_VARIANT_IDS
RANDOM_SEED = 4824
SOFT_BUDGET_ENV = previous.SOFT_BUDGET_ENV
DEFAULT_SOFT_BUDGET_S = previous.DEFAULT_SOFT_BUDGET_S
TERMINAL_PREFIXES = previous.TERMINAL_PREFIXES
LIVE_SUBSTRATE = previous.LIVE_SUBSTRATE
AGGREGATION_SUBSTRATE = previous.AGGREGATION_SUBSTRATE
LIVE_DURATION_FLOOR_S = previous.LIVE_DURATION_FLOOR_S
AGGREGATION_DURATION_FLOOR_S = previous.AGGREGATION_DURATION_FLOOR_S

SPEC_REFS = [
    "REQ-CAPSTONE-4824",
    "SCENARIO-CAPSTONE-4824",
    "SCENARIO-CAPSTONE-4824-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4824-FIELD-PRINCIPLES",
]

PRIOR_READINESS_RESULT_PATHS = previous.PRIOR_READINESS_RESULT_PATHS + (
    "results/experiment_4814_heldout_first_win_readiness.json",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; a measured rate is complete_/success_."
    },
    "heldout_first_win_rate": {
        "principle": "the deadline-relevant generalization signal -- held-out, not in-sample."
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference (60s floor) ONLY if the agent ran live; "
            "aggregation_from_upstream_artifacts if a checkpoint/cache hit -- declare what "
            "actually ran."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "for a flat null (rate == baseline 0.04), explains why the agreement is a "
            "genuine no-improvement result, not a TAUTOLOGY bug."
        )
    },
    "checkpoint_emitted": {
        "principle": "a capped run must still emit a usable partial artifact."
    },
    "preconditions_checked": {
        "principle": (
            "records generator/harness checks; a missing resource emits blocked_, never a "
            "fabricated rate."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + tuple(
    field for field in previous.REQUIRED_ARTIFACT_FIELDS if field not in FIELD_PRINCIPLES
)

_PATCHED_PREVIOUS_CONSTANTS: dict[str, Any] = {
    "EXPERIMENT": EXPERIMENT,
    "EXPERIMENT_ID": EXPERIMENT_ID,
    "SCHEMA": SCHEMA,
    "RESULT_RELATIVE_PATH": RESULT_RELATIVE_PATH,
    "PARTIAL_RESULT_RELATIVE_PATH": PARTIAL_RESULT_RELATIVE_PATH,
    "RANDOM_SEED": RANDOM_SEED,
    "SPEC_REFS": SPEC_REFS,
    "PRIOR_READINESS_RESULT_PATHS": PRIOR_READINESS_RESULT_PATHS,
    "FIELD_PRINCIPLES": FIELD_PRINCIPLES,
    "REQUIRED_ARTIFACT_FIELDS": REQUIRED_ARTIFACT_FIELDS,
}


@contextmanager
def _patched_previous_constants() -> Iterator[None]:
    old_constants = {
        name: getattr(previous, name)
        for name in _PATCHED_PREVIOUS_CONSTANTS
    }
    old_patched_previous_constants = previous._PATCHED_PREVIOUS_CONSTANTS
    try:
        for name, value in _PATCHED_PREVIOUS_CONSTANTS.items():
            setattr(previous, name, value)
        previous._PATCHED_PREVIOUS_CONSTANTS = dict(_PATCHED_PREVIOUS_CONSTANTS)
        yield
    finally:
        previous._PATCHED_PREVIOUS_CONSTANTS = old_patched_previous_constants
        for name, value in old_constants.items():
            setattr(previous, name, value)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    with _patched_previous_constants():
        return previous.payload_checksum(artifact)


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    proxy_artifact: Mapping[str, Any],
    prior_best: Mapping[str, Any],
    partial: bool,
    checkpoint_emitted: bool,
    live_agent_ran: bool,
    duration_s: float,
    budget_exceeded: base._BudgetExceeded | None = None,
    blocked_reason: str | None = None,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    with _patched_previous_constants():
        return dict(
            previous.build_artifact(
                preconditions_checked=preconditions_checked,
                parity_test=parity_test,
                proxy_artifact=proxy_artifact,
                prior_best=prior_best,
                partial=partial,
                checkpoint_emitted=checkpoint_emitted,
                live_agent_ran=live_agent_ran,
                duration_s=duration_s,
                budget_exceeded=budget_exceeded,
                blocked_reason=blocked_reason,
                random_seed=random_seed,
            )
        )


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    with _patched_previous_constants():
        return previous.artifact_schema_errors(artifact)


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    with _patched_previous_constants():
        return previous.write_artifact(root, artifact)


def run_held_out_proxy_checkpointed(
    root: Path,
    parity_test: Mapping[str, Any],
    *,
    now: Callable[[], float] = time.time,
    soft_budget_s: float | None = None,
    public_games: Sequence[str] | None = None,
) -> JsonDict:
    with _patched_previous_constants():
        return dict(
            previous.run_held_out_proxy_checkpointed(
                root,
                parity_test,
                now=now,
                soft_budget_s=soft_budget_s,
                public_games=public_games,
            )
        )


def check_preconditions(root: Path) -> JsonDict:  # pragma: no cover - subprocess/cache boundary.
    with _patched_previous_constants():
        return dict(previous.check_preconditions(root))


def run_parity_test(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary.
    with _patched_previous_constants():
        return dict(previous.run_parity_test(root))


def load_cached_or_run_held_out_proxy(root: Path, parity_test: Mapping[str, Any]) -> JsonDict:
    with _patched_previous_constants():
        return dict(previous.load_cached_or_run_held_out_proxy(root, parity_test))


def load_prior_best(root: Path) -> JsonDict:
    with _patched_previous_constants():
        return dict(previous.load_prior_best(root))


def _partial_proxy_from_budget(
    root: Path,
    budget_exceeded: base._BudgetExceeded,
    parity_test: Mapping[str, Any],
) -> JsonDict:
    with _patched_previous_constants():
        return dict(previous._partial_proxy_from_budget(root, budget_exceeded, parity_test))


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checker: PreconditionsChecker = check_preconditions,
    parity_check: ParityCheck = run_parity_test,
    proxy_runner: ProxyRunner | None = None,
    prior_best_loader: PriorBestLoader = load_prior_best,
    partial_proxy_loader: PartialProxyLoader = _partial_proxy_from_budget,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    with _patched_previous_constants():
        return dict(
            previous.run(
                root=root,
                preconditions_checker=preconditions_checker,
                parity_check=parity_check,
                proxy_runner=proxy_runner,
                prior_best_loader=prior_best_loader,
                partial_proxy_loader=partial_proxy_loader,
                now=now,
                sleep_fn=sleep_fn,
            )
        )


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(f"heldout_first_win_rate={artifact['heldout_first_win_rate']}")
    print(f"heldout_first_win_ci={json.dumps(artifact['heldout_first_win_ci'], sort_keys=True)}")
    print(f"prior_best_heldout_first_win_rate={artifact['prior_best_heldout_first_win_rate']}")
    print(f"inference_substrate={artifact['inference_substrate']}")
    print(f"checkpoint_emitted={artifact['checkpoint_emitted']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
