"""Experiment 4783: ARC self-play verifier checkpoint refresh.

Spec refs: REQ-ARC-WMTE-4783,
SCENARIO-ARC-WMTE-4783-CHECKPOINT-REFRESHED,
SCENARIO-ARC-WMTE-4783-RESIDUAL-NO-FABRICATION,
SCENARIO-ARC-WMTE-4783-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from carnot import experiment_4763_self_play_verifier_checkpoint as base


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
ARTIFACT = RESULTS / "experiment_4783_self_play_verifier_checkpoint.json"

EXPERIMENT = "experiment_4783_self_play_verifier_checkpoint"
SCHEMA = "carnot.exp4783.self_play_verifier_checkpoint.v1"
RESULT_RELATIVE_PATH = "results/experiment_4783_self_play_verifier_checkpoint.json"
REGISTRY_RELATIVE_PATH = base.REGISTRY_RELATIVE_PATH
RANDOM_SEED = 4783
INFERENCE_SUBSTRATE = base.INFERENCE_SUBSTRATE
SOLVE_PROVENANCE = base.SOLVE_PROVENANCE

SPEC_REFS = [
    "REQ-ARC-WMTE-4783",
    "SCENARIO-ARC-WMTE-4783-CHECKPOINT-REFRESHED",
    "SCENARIO-ARC-WMTE-4783-RESIDUAL-NO-FABRICATION",
    "SCENARIO-ARC-WMTE-4783-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES = dict(base.FIELD_PRINCIPLES)
REQUIRED_FIELDS = base.REQUIRED_FIELDS + ("checkpoint_mtime_delta_ns",)
DEFAULT_TARGET_ORDER = base.DEFAULT_TARGET_ORDER


@contextmanager
def _base_scope() -> Iterator[None]:
    saved = {
        "REPO": base.REPO,
        "RESULTS": base.RESULTS,
        "REGISTRY": base.REGISTRY,
        "ARTIFACT": base.ARTIFACT,
    }
    base.REPO = REPO
    base.RESULTS = RESULTS
    base.REGISTRY = REGISTRY
    base.ARTIFACT = ARTIFACT
    try:
        yield
    finally:
        base.REPO = saved["REPO"]
        base.RESULTS = saved["RESULTS"]
        base.REGISTRY = saved["REGISTRY"]
        base.ARTIFACT = saved["ARTIFACT"]


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_checkpoint_mtimes(artifact: dict[str, Any]) -> None:
    before = _as_optional_int(artifact.get("checkpoint_mtime_before_ns"))
    after = _as_optional_int(artifact.get("checkpoint_mtime_after_ns"))
    artifact["checkpoint_mtime_before_ns"] = str(before) if before is not None else None
    artifact["checkpoint_mtime_after_ns"] = str(after) if after is not None else None
    artifact["checkpoint_mtime_delta_ns"] = (
        int(after) - int(before) if before is not None and after is not None else None
    )


def stable_checksum(payload: dict[str, Any]) -> str:
    return base.stable_checksum(payload)


def load_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = REGISTRY if path is None else Path(path)
    return base.load_registry(registry_path)


def registry_levels(registry: dict[str, Any]) -> dict[str, int]:
    return base.registry_levels(registry)


def select_banked_target(
    registry: dict[str, Any],
    preferred_game: str | None = None,
    candidate_games: tuple[str, ...] = DEFAULT_TARGET_ORDER,
) -> dict[str, Any]:
    return base.select_banked_target(registry, preferred_game, candidate_games)


def summarize_loop_result(
    *,
    game: str,
    loop_result: dict[str, Any] | None,
    loop_result_path: str,
    checkpoint_mtime_before_ns: int | None,
) -> dict[str, Any]:
    with _base_scope():
        return base.summarize_loop_result(
            game=game,
            loop_result=loop_result,
            loop_result_path=loop_result_path,
            checkpoint_mtime_before_ns=checkpoint_mtime_before_ns,
        )


def _retarget_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["experiment"] = EXPERIMENT
    artifact["schema"] = SCHEMA
    artifact["spec_refs"] = list(SPEC_REFS)
    artifact["field_principles"] = dict(FIELD_PRINCIPLES)
    artifact["random_seed"] = RANDOM_SEED
    _normalize_checkpoint_mtimes(artifact)
    artifact["reproducibility_checksum"] = ""
    artifact["schema_errors"] = []
    artifact["reproducibility_checksum"] = stable_checksum(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def build_artifact(
    *,
    target_selection: dict[str, Any],
    loop_summary: dict[str, Any],
    preconditions_checked: dict[str, Any],
) -> dict[str, Any]:
    with _base_scope():
        artifact = base.build_artifact(
            target_selection=target_selection,
            loop_summary=loop_summary,
            preconditions_checked=preconditions_checked,
        )
    return _retarget_artifact(artifact)


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: dict[str, Any],
    target_game: str | None = None,
) -> dict[str, Any]:
    with _base_scope():
        artifact = base.build_blocked_artifact(
            reason=reason,
            preconditions_checked=preconditions_checked,
            target_game=target_game,
        )
    return _retarget_artifact(artifact)


def artifact_schema_errors(payload: dict[str, Any]) -> list[str]:
    errors = base.artifact_schema_errors(payload)
    for field in REQUIRED_FIELDS:
        if field not in payload and f"missing_field:{field}" not in errors:
            errors.append(f"missing_field:{field}")
    if payload.get("experiment") != EXPERIMENT:
        errors.append("experiment_mismatch")
    if payload.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs_mismatch")
    if payload.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")

    if bool(payload.get("verifier_checkpoint_refreshed")):
        delta_ns = _as_optional_int(payload.get("checkpoint_mtime_delta_ns"))
        if delta_ns is None:
            errors.append("refreshed_checkpoint_missing_mtime_delta")
        elif delta_ns <= 0:
            errors.append("refreshed_checkpoint_nonpositive_mtime_delta")
    return errors


def write_artifact(payload: dict[str, Any], path: Path | None = None) -> Path:
    output = ARTIFACT if path is None else Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def check_preconditions(game: str | None = None) -> dict[str, Any]:  # pragma: no cover - live environment boundary.
    with _base_scope():
        return base.check_preconditions(game)


def _precondition_failure(preconditions: dict[str, Any]) -> str | None:
    return base._precondition_failure(preconditions)


def _relative_path(path: Path) -> str:
    with _base_scope():
        return base._relative_path(path)


def _read_loop_result(path: Path) -> dict[str, Any] | None:
    with _base_scope():
        return base._read_loop_result(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game")
    parser.add_argument("--loop-result")
    parser.add_argument("--checkpoint-mtime-before-ns", type=int)
    args = parser.parse_args(argv)

    preconditions = check_preconditions(args.game)
    selection = dict(preconditions.get("target_selection") or {})
    failure = _precondition_failure(preconditions)
    if failure is not None:
        artifact = build_blocked_artifact(
            reason=failure,
            preconditions_checked=preconditions,
            target_game=args.game or selection.get("game"),
        )
    else:
        game = str(selection["game"])
        loop_path = Path(args.loop_result) if args.loop_result else RESULTS / f"arc_loop_solve_{game}.json"
        loop_summary = summarize_loop_result(
            game=game,
            loop_result=_read_loop_result(loop_path),
            loop_result_path=_relative_path(loop_path),
            checkpoint_mtime_before_ns=args.checkpoint_mtime_before_ns,
        )
        artifact = build_artifact(
            target_selection=selection,
            loop_summary=loop_summary,
            preconditions_checked=preconditions,
        )

    write_artifact(artifact, ARTIFACT)
    print(f"honest_verdict={artifact['honest_verdict']}")
    print(f"verifier_checkpoint_refreshed={artifact['verifier_checkpoint_refreshed']}")
    print(f"states_expanded={artifact['states_expanded']}")
    print(f"schema_errors={artifact['schema_errors']}")
    return 0 if not artifact["schema_errors"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
