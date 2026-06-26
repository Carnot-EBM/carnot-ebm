"""Experiment 4803: ARC self-play verifier checkpoint refresh.

Spec refs: REQ-ARC-WMTE-4803,
SCENARIO-ARC-WMTE-4803-CHECKPOINT-REFRESHED,
SCENARIO-ARC-WMTE-4803-RESIDUAL-NO-FABRICATION,
SCENARIO-ARC-WMTE-4803-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from carnot import experiment_4793_self_play_verifier_checkpoint as previous


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
ARTIFACT = RESULTS / "experiment_4803_self_play_verifier_checkpoint.json"

EXPERIMENT = "experiment_4803_self_play_verifier_checkpoint"
SCHEMA = "carnot.exp4803.self_play_verifier_checkpoint.v1"
RESULT_RELATIVE_PATH = "results/experiment_4803_self_play_verifier_checkpoint.json"
RANDOM_SEED = 4803
INFERENCE_SUBSTRATE = previous.INFERENCE_SUBSTRATE
SOLVE_PROVENANCE = previous.SOLVE_PROVENANCE

SPEC_REFS = [
    "REQ-ARC-WMTE-4803",
    "SCENARIO-ARC-WMTE-4803-CHECKPOINT-REFRESHED",
    "SCENARIO-ARC-WMTE-4803-RESIDUAL-NO-FABRICATION",
    "SCENARIO-ARC-WMTE-4803-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; checkpoint refreshed + gate green is success_.",
    "verifier_checkpoint_refreshed": "the self-improvement signal.",
    "inference_substrate": "live_llm_inference (60s floor).",
    "solve_provenance": (
        "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
    ),
    "preconditions_checked": (
        "records arcade/registry checks; a missing target emits blocked_, never a fabricated checkpoint."
    ),
}

REQUIRED_FIELDS = previous.REQUIRED_FIELDS


@contextmanager
def _previous_scope() -> Iterator[None]:
    saved = {
        "REPO": previous.REPO,
        "RESULTS": previous.RESULTS,
        "REGISTRY": previous.REGISTRY,
        "ARTIFACT": previous.ARTIFACT,
    }
    previous.REPO = REPO
    previous.RESULTS = RESULTS
    previous.REGISTRY = REGISTRY
    previous.ARTIFACT = ARTIFACT
    try:
        yield
    finally:
        previous.REPO = saved["REPO"]
        previous.RESULTS = saved["RESULTS"]
        previous.REGISTRY = saved["REGISTRY"]
        previous.ARTIFACT = saved["ARTIFACT"]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _checksum_is_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def stable_checksum(payload: dict[str, Any]) -> str:
    return previous.stable_checksum(payload)


def summarize_loop_result(
    *,
    game: str,
    loop_result: dict[str, Any] | None,
    loop_result_path: str,
    checkpoint_mtime_before_ns: int | None,
) -> dict[str, Any]:
    with _previous_scope():
        summary = previous.summarize_loop_result(
            game=game,
            loop_result=loop_result,
            loop_result_path=loop_result_path,
            checkpoint_mtime_before_ns=checkpoint_mtime_before_ns,
        )
    summary["search_state_count"] = _as_int(summary.get("states_expanded"), 0)
    return summary


def _retarget_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["experiment"] = EXPERIMENT
    artifact["schema"] = SCHEMA
    artifact["spec_refs"] = list(SPEC_REFS)
    artifact["field_principles"] = dict(FIELD_PRINCIPLES)
    artifact["random_seed"] = RANDOM_SEED
    artifact["search_state_count"] = _as_int(
        artifact.get("search_state_count", artifact.get("states_expanded")), 0
    )
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
    with _previous_scope():
        artifact = previous.build_artifact(
            target_selection=target_selection,
            loop_summary=loop_summary,
            preconditions_checked=preconditions_checked,
        )
    artifact["search_state_count"] = _as_int(
        loop_summary.get("search_state_count", loop_summary.get("states_expanded")), 0
    )
    return _retarget_artifact(artifact)


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: dict[str, Any],
    target_game: str | None = None,
) -> dict[str, Any]:
    with _previous_scope():
        artifact = previous.build_blocked_artifact(
            reason=reason,
            preconditions_checked=preconditions_checked,
            target_game=target_game,
        )
    artifact["search_state_count"] = 0
    return _retarget_artifact(artifact)


def artifact_schema_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")

    principles = payload.get("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if not isinstance(principles, dict) or principles.get(field) != principle:
            errors.append(f"missing_principle:{field}")

    verdict = str(payload.get("honest_verdict") or "")
    if not verdict.startswith(("success_", "complete_", "blocked_")):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("experiment") != EXPERIMENT:
        errors.append("experiment_mismatch")
    if payload.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs_mismatch")
    if payload.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance_mismatch")
    if not isinstance(payload.get("verifier_checkpoint_refreshed"), bool):
        errors.append("verifier_checkpoint_refreshed_must_be_bool")

    checksum = payload.get("reproducibility_checksum")
    if not _checksum_is_hex(checksum):
        errors.append("invalid_reproducibility_checksum")
    elif checksum != stable_checksum(dict(payload)):
        errors.append("checksum_mismatch")

    refreshed = bool(payload.get("verifier_checkpoint_refreshed"))
    success = verdict.startswith("success_")
    before_ns = _as_optional_int(payload.get("checkpoint_mtime_before_ns"))
    after_ns = _as_optional_int(payload.get("checkpoint_mtime_after_ns"))
    delta_ns = _as_optional_int(payload.get("checkpoint_mtime_delta_ns"))
    if success and not refreshed:
        errors.append("success_without_refreshed_checkpoint")
    if success and not (
        payload.get("offline_reproduced") is True and _as_int(payload.get("reproduced_levels")) >= 1
    ):
        errors.append("success_without_reproduction_gate")
    if success and _as_int(payload.get("states_expanded")) < 1:
        errors.append("success_without_search_states")
    if success and _as_int(payload.get("search_state_count")) < 1:
        errors.append("success_without_search_state_count")
    if refreshed and not payload.get("checkpoint_path"):
        errors.append("refreshed_checkpoint_missing_path")
    if refreshed and after_ns is None:
        errors.append("refreshed_checkpoint_missing_mtime")
    if refreshed and before_ns is not None and after_ns is not None and after_ns <= before_ns:
        errors.append("refreshed_checkpoint_without_mtime_advance")
    if refreshed and delta_ns is None:
        errors.append("refreshed_checkpoint_missing_mtime_delta")
    if refreshed and delta_ns is not None and delta_ns <= 0:
        errors.append("refreshed_checkpoint_nonpositive_mtime_delta")
    return errors


def write_artifact(payload: dict[str, Any], path: Path | None = None) -> Path:
    output = ARTIFACT if path is None else Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def check_preconditions(game: str | None = None) -> dict[str, Any]:  # pragma: no cover - live boundary.
    with _previous_scope():
        return previous.check_preconditions(game)


def _precondition_failure(preconditions: dict[str, Any]) -> str | None:
    return previous._precondition_failure(preconditions)


def _relative_path(path: Path) -> str:
    with _previous_scope():
        return previous._relative_path(path)


def _read_loop_result(path: Path) -> dict[str, Any] | None:
    with _previous_scope():
        return previous._read_loop_result(path)


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
        loop_path = (
            Path(args.loop_result) if args.loop_result else RESULTS / f"arc_loop_solve_{game}.json"
        )
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
    print(f"search_state_count={artifact['search_state_count']}")
    print(f"schema_errors={artifact['schema_errors']}")
    return 0 if not artifact["schema_errors"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
