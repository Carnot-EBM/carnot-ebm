"""Exp 4191: ARC-AGI-3 live-env grounding probe.

Spec refs: REQ-PHASE4-056, SCENARIO-PHASE4-056.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from carnot.agentic.arc_agi3_live_adapter import (
    BASE_URL,
    DEFAULT_ACTION_BUDGET,
    RANDOM_SEED,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIRED_FIELD_PRINCIPLES,
    REQUIREMENTS,
    RESULT_NAME,
    ArcLivePreconditions,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    check_live_preconditions,
    open_online_arcade,
    run_live_reachability_probe,
    validate_recorded_fixture,
)


REPO = Path(__file__).resolve().parents[2]


def _write_artifact(artifact: dict[str, Any]) -> None:
    output = REPO / "results" / RESULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    write: bool = True,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    base_url: str = BASE_URL,
) -> dict[str, Any]:
    """Run the bounded live grounding probe or write an honest blocked verdict."""

    started = time.time()
    preconditions = check_live_preconditions(base_url=base_url)
    offline_validation: dict[str, Any] = {"passed": False, "skipped": True}

    if not preconditions.ok:
        artifact = blocked_artifact(preconditions=preconditions, duration_s=time.time() - started)
        if write:
            _write_artifact(artifact)
        return artifact

    try:
        offline_validation = validate_recorded_fixture()
        if offline_validation.get("passed") is not True:
            raise RuntimeError("recorded fixture adapter validation failed")
        arcade = open_online_arcade(base_url=base_url)
        environment_count, outcome = run_live_reachability_probe(
            arcade,
            action_budget=action_budget,
            random_seed=RANDOM_SEED,
        )
        artifact = build_artifact(
            outcome=outcome,
            preconditions=preconditions,
            offline_validation=offline_validation,
            environment_count=environment_count,
            duration_s=time.time() - started,
        )
    except Exception as exc:
        blocked_preconditions = ArcLivePreconditions(
            sdk_importable=preconditions.sdk_importable,
            sdk_version=preconditions.sdk_version,
            network_reachable=preconditions.network_reachable,
            base_url=preconditions.base_url,
            error=f"{preconditions.error}; live_probe_error={type(exc).__name__}: {exc}".strip("; "),
        )
        artifact = blocked_artifact(preconditions=blocked_preconditions, duration_s=time.time() - started)
        artifact["offline_validation"] = offline_validation
        errors = artifact_schema_errors(artifact)
        if errors:
            raise ValueError("; ".join(errors)) from exc

    if write:
        _write_artifact(artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--action-budget", type=int, default=DEFAULT_ACTION_BUDGET)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()
    artifact = run(
        write=not args.no_write,
        action_budget=args.action_budget,
        base_url=args.base_url,
    )
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
