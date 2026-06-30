"""Experiment 5025: preferred lf52 ARC self-play checkpoint refresh.

Spec refs: REQ-LEARN-5025,
SCENARIO-LEARN-5025-CHECKPOINT-REFRESHED,
SCENARIO-LEARN-5025-SUBSTRATE-FIX,
SCENARIO-LEARN-5025-RESIDUAL-NO-FABRICATION,
SCENARIO-LEARN-5025-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from carnot import experiment_5011_self_play_verifier_checkpoint as previous


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
ARTIFACT = RESULTS / "experiment_5025_self_play_verifier_checkpoint.json"

EXPERIMENT = "experiment_5025_self_play_verifier_checkpoint"
SCHEMA = "carnot.exp5025.self_play_verifier_checkpoint.v1"
RESULT_RELATIVE_PATH = "results/experiment_5025_self_play_verifier_checkpoint.json"
RANDOM_SEED = 5025
DEFAULT_TARGET_GAME = "lf52"
DISALLOWED_TARGET_GAMES = (
    "r11l",
    "su15",
    "ft09",
    "sp80",
    "dc22",
    "lp85",
    "ar25",
    "bp35",
    "vc33",
    "sk48",
    "ls20",
    "re86",
)
INFERENCE_SUBSTRATE = previous.INFERENCE_SUBSTRATE
LIVE_LLM_SUBSTRATE = previous.LIVE_LLM_SUBSTRATE
SOLVE_PROVENANCE = previous.SOLVE_PROVENANCE
SUCCESS_VERDICT = previous.SUCCESS_VERDICT
VERIFIER_MIN_DURATION_S = previous.VERIFIER_MIN_DURATION_S
LIVE_LLM_MIN_DURATION_S = previous.LIVE_LLM_MIN_DURATION_S

SPEC_REFS = [
    "REQ-LEARN-5025",
    "SCENARIO-LEARN-5025-CHECKPOINT-REFRESHED",
    "SCENARIO-LEARN-5025-SUBSTRATE-FIX",
    "SCENARIO-LEARN-5025-RESIDUAL-NO-FABRICATION",
    "SCENARIO-LEARN-5025-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
    ),
    "verifier_checkpoint_refreshed": (
        "the self-improvement signal -- the learned verifier trained on this run's traces "
        "(FR-11 / continuous self-learning)."
    ),
    "checkpoint_path": (
        "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
    ),
    "offline_reproduced": (
        "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
    ),
    "reproduced_levels": "the depth confirmed this run.",
    "target_game": (
        "the rotated self-play target (differs from .461 r11l / .460 su15 / "
        ".459 ft09 / .458 sp80 / .457 dc22 / .456 lp85 / .455 ar25 / "
        ".453 bp35 / .452 vc33 / .451 sk48 / .450 ls20 / .449 re86)."
    ),
    "solve_provenance": (
        "live_agent_self_discovery -- self-play is the agent improving on its own attempts "
        "(NOT outer_loop_re)."
    ),
    "inference_substrate": (
        "the HONEST substrate -- verifier_ensemble_against_cached_candidates (1s floor) "
        "for an offline gate run; live_llm_inference (60s floor) ONLY if the generator "
        "actually ran >=60s (maintains the .456 substrate fix)."
    ),
    "duration_s": "measured wall-clock; consistent with inference_substrate.",
    "flag_resolved": "true iff NOT flagged true_live_recheck=critical.",
    "preconditions_checked": (
        "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."
    ),
}

REQUIRED_FIELDS = previous.REQUIRED_FIELDS
_PATCHED_PREVIOUS_GLOBALS = (
    "REPO",
    "RESULTS",
    "REGISTRY",
    "ARTIFACT",
    "EXPERIMENT",
    "SCHEMA",
    "RESULT_RELATIVE_PATH",
    "RANDOM_SEED",
    "DEFAULT_TARGET_GAME",
    "DISALLOWED_TARGET_GAMES",
    "INFERENCE_SUBSTRATE",
    "LIVE_LLM_SUBSTRATE",
    "SOLVE_PROVENANCE",
    "SUCCESS_VERDICT",
    "VERIFIER_MIN_DURATION_S",
    "LIVE_LLM_MIN_DURATION_S",
    "SPEC_REFS",
    "FIELD_PRINCIPLES",
)


@contextmanager
def _previous_scope() -> Iterator[None]:
    saved = {name: getattr(previous, name) for name in _PATCHED_PREVIOUS_GLOBALS}
    for name in _PATCHED_PREVIOUS_GLOBALS:
        setattr(previous, name, globals()[name])
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(previous, name, value)


def _call_previous(function_name: str, *args: Any, **kwargs: Any) -> Any:
    with _previous_scope():
        return getattr(previous, function_name)(*args, **kwargs)


def _target_game(game: str | None) -> str:
    return game or DEFAULT_TARGET_GAME


def _rotation_ok(game: str | None) -> bool:
    return game not in DISALLOWED_TARGET_GAMES


def stable_checksum(payload: dict[str, Any]) -> str:
    return _call_previous("stable_checksum", payload)


def summarize_loop_result(
    *,
    game: str,
    loop_result: dict[str, Any] | None,
    loop_result_path: str,
    checkpoint_mtime_before_ns: int | None,
) -> dict[str, Any]:
    return _call_previous(
        "summarize_loop_result",
        game=game,
        loop_result=loop_result,
        loop_result_path=loop_result_path,
        checkpoint_mtime_before_ns=checkpoint_mtime_before_ns,
    )


def build_artifact(
    *,
    target_selection: dict[str, Any],
    loop_summary: dict[str, Any],
    preconditions_checked: dict[str, Any],
    duration_s: float | None = None,
    flag_resolved: bool = True,
) -> dict[str, Any]:
    return _call_previous(
        "build_artifact",
        target_selection=target_selection,
        loop_summary=loop_summary,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        flag_resolved=flag_resolved,
    )


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: dict[str, Any],
    target_game: str | None = None,
    duration_s: float | None = None,
    flag_resolved: bool = True,
) -> dict[str, Any]:
    return _call_previous(
        "build_blocked_artifact",
        reason=reason,
        preconditions_checked=preconditions_checked,
        target_game=target_game,
        duration_s=duration_s,
        flag_resolved=flag_resolved,
    )


def artifact_schema_errors(payload: dict[str, Any]) -> list[str]:
    return _call_previous("artifact_schema_errors", payload)


def write_artifact(payload: dict[str, Any], path: Path | None = None) -> Path:
    output = ARTIFACT if path is None else Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def check_preconditions(game: str | None = None) -> dict[str, Any]:
    return _call_previous("check_preconditions", game)


def _precondition_failure(preconditions: dict[str, Any]) -> str | None:
    return _call_previous("_precondition_failure", preconditions)


def _relative_path(path: Path) -> str:
    return _call_previous("_relative_path", path)


def _read_loop_result(path: Path) -> dict[str, Any] | None:
    return _call_previous("_read_loop_result", path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game")
    parser.add_argument("--loop-result")
    parser.add_argument("--checkpoint-mtime-before-ns", type=int)
    parser.add_argument("--duration-s", type=float, default=VERIFIER_MIN_DURATION_S)
    parser.add_argument("--flag-unresolved", action="store_true")
    args = parser.parse_args(argv)

    preconditions = check_preconditions(args.game)
    selection = dict(preconditions.get("target_selection") or {})
    flag_resolved = not args.flag_unresolved
    failure = _precondition_failure(preconditions)
    if failure is not None:
        artifact = build_blocked_artifact(
            reason=failure,
            preconditions_checked=preconditions,
            target_game=args.game or selection.get("game") or DEFAULT_TARGET_GAME,
            duration_s=args.duration_s,
            flag_resolved=flag_resolved,
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
            duration_s=args.duration_s,
            flag_resolved=flag_resolved,
        )

    write_artifact(artifact, ARTIFACT)
    print(f"honest_verdict={artifact['honest_verdict']}")
    print(f"verifier_checkpoint_refreshed={artifact['verifier_checkpoint_refreshed']}")
    print(f"checkpoint_path={artifact['checkpoint_path']}")
    print(f"target_game={artifact['target_game']}")
    print(f"inference_substrate={artifact['inference_substrate']}")
    print(f"duration_s={artifact['duration_s']}")
    print(f"flag_resolved={artifact['flag_resolved']}")
    print(f"schema_errors={artifact['schema_errors']}")
    return 0 if not artifact["schema_errors"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
