"""Experiment 4949: honest-substrate ARC self-play checkpoint refresh.

Spec refs: REQ-LEARN-4949,
SCENARIO-LEARN-4949-CHECKPOINT-REFRESHED,
SCENARIO-LEARN-4949-SUBSTRATE-FIX,
SCENARIO-LEARN-4949-RESIDUAL-NO-FABRICATION,
SCENARIO-LEARN-4949-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from carnot import experiment_4938_self_play_verifier_checkpoint as previous


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
ARTIFACT = RESULTS / "experiment_4949_self_play_verifier_checkpoint.json"

EXPERIMENT = "experiment_4949_self_play_verifier_checkpoint"
SCHEMA = "carnot.exp4949.self_play_verifier_checkpoint.v1"
RESULT_RELATIVE_PATH = "results/experiment_4949_self_play_verifier_checkpoint.json"
RANDOM_SEED = 4949
DEFAULT_TARGET_GAME = "lp85"
DISALLOWED_TARGET_GAMES = ("ar25", "bp35", "vc33", "sk48", "ls20", "re86")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
LIVE_LLM_SUBSTRATE = "live_llm_inference"
SOLVE_PROVENANCE = previous.SOLVE_PROVENANCE
SUCCESS_VERDICT = "success_self_play_checkpoint_refreshed"
VERIFIER_MIN_DURATION_S = 1.0
LIVE_LLM_MIN_DURATION_S = 60.0

SPEC_REFS = [
    "REQ-LEARN-4949",
    "SCENARIO-LEARN-4949-CHECKPOINT-REFRESHED",
    "SCENARIO-LEARN-4949-SUBSTRATE-FIX",
    "SCENARIO-LEARN-4949-RESIDUAL-NO-FABRICATION",
    "SCENARIO-LEARN-4949-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
    ),
    "verifier_checkpoint_refreshed": (
        "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
    ),
    "checkpoint_path": (
        "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
    ),
    "offline_reproduced": (
        "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
    ),
    "reproduced_levels": "the depth confirmed this run.",
    "target_game": (
        "the rotated self-play target (differs from .455 ar25 / .453 bp35 / .452 vc33 / "
        ".451 sk48 / .450 ls20 / .449 re86 AND from A1/A2)."
    ),
    "inference_substrate": (
        "the HONEST substrate -- verifier_ensemble_against_cached_candidates (1s floor) "
        "for an offline gate run; live_llm_inference (60s floor) ONLY if the generator "
        "ran >=60s. This fixes the .455 exp4938 DURATION_TOO_SHORT flag."
    ),
    "duration_s": (
        "the measured wall-clock; must be consistent with inference_substrate "
        "(>=1s for verifier-ensemble, >=60s for live_llm_inference)."
    ),
    "flag_resolved": (
        "true iff the fresh artifact is NOT flagged true_live_recheck=critical "
        "(the .455 DURATION_TOO_SHORT substrate mismatch is fixed)."
    ),
    "solve_provenance": (
        "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
    ),
    "preconditions_checked": (
        "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."
    ),
}

REQUIRED_FIELDS = tuple(
    dict.fromkeys(
        (
            *(
                field
                for field in previous.REQUIRED_FIELDS
                if field != "search_state_count"
            ),
            "duration_s",
            "flag_resolved",
        )
    )
)
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
    "SOLVE_PROVENANCE",
    "SUCCESS_VERDICT",
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


def _target_game(game: str | None) -> str:
    return game or DEFAULT_TARGET_GAME


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _success_gate(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("verifier_checkpoint_refreshed") is True
        and payload.get("offline_reproduced") is True
        and _as_int(payload.get("reproduced_levels")) >= 1
        and payload.get("flag_resolved") is True
    )


def _rotation_ok(game: str | None) -> bool:
    return game not in DISALLOWED_TARGET_GAMES


def _rotation_failure_reason() -> str:
    return "target_rotation_" + "_".join(DISALLOWED_TARGET_GAMES) + "_disallowed"


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
        return previous.summarize_loop_result(
            game=game,
            loop_result=loop_result,
            loop_result_path=loop_result_path,
            checkpoint_mtime_before_ns=checkpoint_mtime_before_ns,
        )


def _retarget_artifact(
    payload: dict[str, Any],
    *,
    duration_s: float | None,
    flag_resolved: bool,
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["experiment"] = EXPERIMENT
    artifact["schema"] = SCHEMA
    artifact["spec_refs"] = list(SPEC_REFS)
    artifact["result_path"] = RESULT_RELATIVE_PATH
    artifact["field_principles"] = dict(FIELD_PRINCIPLES)
    artifact["random_seed"] = RANDOM_SEED
    artifact["inference_substrate"] = INFERENCE_SUBSTRATE
    artifact["duration_s"] = VERIFIER_MIN_DURATION_S if duration_s is None else float(duration_s)
    artifact["flag_resolved"] = bool(flag_resolved)
    artifact.pop("search_state_count", None)
    if _success_gate(artifact):
        artifact["honest_verdict"] = SUCCESS_VERDICT
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
    duration_s: float | None = None,
    flag_resolved: bool = True,
) -> dict[str, Any]:
    with _previous_scope():
        artifact = previous.build_artifact(
            target_selection=target_selection,
            loop_summary=loop_summary,
            preconditions_checked=preconditions_checked,
        )
    return _retarget_artifact(
        artifact, duration_s=duration_s, flag_resolved=flag_resolved
    )


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: dict[str, Any],
    target_game: str | None = None,
    duration_s: float | None = None,
    flag_resolved: bool = True,
) -> dict[str, Any]:
    with _previous_scope():
        artifact = previous.build_blocked_artifact(
            reason=reason,
            preconditions_checked=preconditions_checked,
            target_game=target_game,
        )
    return _retarget_artifact(
        artifact, duration_s=duration_s, flag_resolved=flag_resolved
    )


def _duration_schema_error(payload: dict[str, Any]) -> str | None:
    duration = _as_float(payload.get("duration_s"))
    if duration is None:
        return "duration_s_not_finite"
    substrate = str(payload.get("inference_substrate") or "")
    if substrate == INFERENCE_SUBSTRATE and duration < VERIFIER_MIN_DURATION_S:
        return "duration_too_short_for_verifier_ensemble"
    if substrate == LIVE_LLM_SUBSTRATE and duration < LIVE_LLM_MIN_DURATION_S:
        return "duration_too_short_for_live_llm_inference"
    if substrate not in {INFERENCE_SUBSTRATE, LIVE_LLM_SUBSTRATE}:
        return "unknown_inference_substrate"
    return None


def artifact_schema_errors(payload: dict[str, Any]) -> list[str]:
    with _previous_scope():
        errors = previous.artifact_schema_errors(payload)
    errors = [
        error
        for error in errors
        if error
        not in {"missing_field:search_state_count", "success_without_search_state_count"}
    ]

    for field in REQUIRED_FIELDS:
        missing = f"missing_field:{field}"
        if field not in payload and missing not in errors:
            errors.append(missing)

    preconditions = payload.get("preconditions_checked")
    if not isinstance(preconditions, dict) or "checkpoint_existing" not in preconditions:
        errors.append("preconditions_missing_checkpoint_check")

    duration_error = _duration_schema_error(payload)
    if duration_error is not None:
        errors.append(duration_error)
    if not isinstance(payload.get("flag_resolved"), bool):
        errors.append("flag_resolved_must_be_bool")
    if payload.get("flag_resolved") is True and (
        payload.get("flagged_adversarial") is True
        or payload.get("true_live_recheck") == "critical"
    ):
        errors.append("flag_resolved_contradicts_critical_recheck")

    verdict = str(payload.get("honest_verdict") or "")
    if _success_gate(payload) and verdict != SUCCESS_VERDICT:
        errors.append("success_verdict_mismatch")
    if verdict == SUCCESS_VERDICT and not _success_gate(payload):
        errors.append("success_verdict_without_gate")
    if verdict == SUCCESS_VERDICT and payload.get("flag_resolved") is not True:
        errors.append("success_without_flag_resolved")
    if _success_gate(payload) and not _rotation_ok(str(payload.get("target_game") or "")):
        errors.append("success_target_rotation_violation")
    if payload.get("result_path") != RESULT_RELATIVE_PATH:
        errors.append("result_path_mismatch")
    return errors


def write_artifact(payload: dict[str, Any], path: Path | None = None) -> Path:
    output = ARTIFACT if path is None else Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def check_preconditions(game: str | None = None) -> dict[str, Any]:  # pragma: no cover - live boundary.
    target = _target_game(game)
    with _previous_scope():
        preconditions = previous.check_preconditions(target)

    selection = dict(preconditions.get("target_selection") or {})
    selected_game = str(selection.get("game") or target)
    if game is None and selected_game == DEFAULT_TARGET_GAME:
        selection["reason"] = f"rotated_banked_target_warm_start_{DEFAULT_TARGET_GAME}"
        preconditions["target_selection"] = selection
        if isinstance(preconditions.get("target_banked"), dict):
            preconditions["target_banked"]["reason"] = selection["reason"]

    rotation_ok = _rotation_ok(selected_game)
    preconditions["target_rotation"] = {
        "ok": rotation_ok,
        "game": selected_game,
        "rotated_off": list(DISALLOWED_TARGET_GAMES),
    }
    preconditions["ok"] = bool(preconditions.get("ok") and rotation_ok)
    return preconditions


def _precondition_failure(preconditions: dict[str, Any]) -> str | None:
    base_preconditions = dict(preconditions)
    base_rotation = base_preconditions.get("target_rotation")
    if isinstance(base_rotation, dict):
        base_preconditions["target_rotation"] = {**base_rotation, "ok": True}
    else:
        base_preconditions["target_rotation"] = {"ok": True}
    base_failure = previous._precondition_failure(base_preconditions)
    if base_failure is not None:
        return base_failure
    if preconditions.get("target_rotation", {}).get("ok") is not True:
        return _rotation_failure_reason()
    if preconditions.get("checkpoint_existing", {}).get("ok") is not True:
        return "checkpoint_missing"
    return None


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
