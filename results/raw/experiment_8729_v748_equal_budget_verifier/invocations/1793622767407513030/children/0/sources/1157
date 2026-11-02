"""Experiment 4763: ARC self-play verifier checkpoint refresh.

Spec refs: REQ-ARC-WMTE-4763,
SCENARIO-ARC-WMTE-4763-CHECKPOINT-REFRESHED,
SCENARIO-ARC-WMTE-4763-RESIDUAL-NO-FABRICATION,
SCENARIO-ARC-WMTE-4763-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
ARTIFACT = RESULTS / "experiment_4763_self_play_verifier_checkpoint.json"

EXPERIMENT = "experiment_4763_self_play_verifier_checkpoint"
SCHEMA = "carnot.exp4763.self_play_verifier_checkpoint.v1"
RESULT_RELATIVE_PATH = "results/experiment_4763_self_play_verifier_checkpoint.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4763
INFERENCE_SUBSTRATE = "live_llm_inference"
SOLVE_PROVENANCE = "live_agent_self_discovery"

SPEC_REFS = [
    "REQ-ARC-WMTE-4763",
    "SCENARIO-ARC-WMTE-4763-CHECKPOINT-REFRESHED",
    "SCENARIO-ARC-WMTE-4763-RESIDUAL-NO-FABRICATION",
    "SCENARIO-ARC-WMTE-4763-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; checkpoint refreshed + gate green is success_.",
    "verifier_checkpoint_refreshed": (
        "the self-improvement signal -- the learned verifier improved from its own play."
    ),
    "inference_substrate": "live_llm_inference; 60s floor.",
    "solve_provenance": (
        "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
    ),
    "preconditions_checked": (
        "records arcade/registry checks; a missing target emits blocked_, never a fabricated checkpoint."
    ),
}

REQUIRED_FIELDS = (
    "experiment",
    "schema",
    "spec_refs",
    "field_principles",
    "honest_verdict",
    "verifier_checkpoint_refreshed",
    "inference_substrate",
    "solve_provenance",
    "preconditions_checked",
    "target_game",
    "checkpoint_path",
    "checkpoint_mtime_before_ns",
    "checkpoint_mtime_after_ns",
    "states_expanded",
    "offline_reproduced",
    "reproduced_levels",
    "reproduction_gate",
    "self_play_residual",
    "loop_result_path",
    "random_seed",
    "reproducibility_checksum",
    "schema_errors",
)

DEFAULT_TARGET_ORDER = (
    "re86",
    "sb26",
    "bp35",
    "lf52",
    "sk48",
    "vc33",
    "ls20",
    "dc22",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _checksum_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"reproducibility_checksum", "schema_errors"}
    }


def stable_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(_checksum_payload(payload)).encode("utf-8")).hexdigest()


def _checksum_is_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


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


def load_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = REGISTRY if path is None else Path(path)
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def registry_levels(registry: dict[str, Any]) -> dict[str, int]:
    levels: dict[str, int] = {}
    for row in registry.get("games", []) or []:
        if isinstance(row, dict) and row.get("game"):
            levels[str(row["game"])] = _as_int(row.get("levels_reproduced"), 0)
    return levels


def select_banked_target(
    registry: dict[str, Any],
    preferred_game: str | None = None,
    candidate_games: tuple[str, ...] = DEFAULT_TARGET_ORDER,
) -> dict[str, Any]:
    levels = registry_levels(registry)
    if preferred_game:
        prior = int(levels.get(preferred_game, 0))
        return {
            "game": preferred_game,
            "prior_level": prior,
            "banked": prior >= 1,
            "reason": "preferred_banked_target" if prior >= 1 else "preferred_target_not_banked",
        }

    for game in candidate_games:
        prior = int(levels.get(game, 0))
        if prior >= 1:
            return {
                "game": game,
                "prior_level": prior,
                "banked": True,
                "reason": "first_banked_rotation_target",
            }

    return {
        "game": None,
        "prior_level": 0,
        "banked": False,
        "reason": "no_banked_target",
    }


def _gate(loop_result: dict[str, Any]) -> dict[str, Any]:
    gate = loop_result.get("reproduction_gate")
    return dict(gate) if isinstance(gate, dict) else {}


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _checkpoint_status(checkpoint_path: Path | None, before_mtime_ns: int | None) -> dict[str, Any]:
    after_mtime_ns = checkpoint_path.stat().st_mtime_ns if checkpoint_path and checkpoint_path.exists() else None
    refreshed = bool(
        after_mtime_ns is not None
        and (before_mtime_ns is None or int(after_mtime_ns) > int(before_mtime_ns))
    )
    return {
        "checkpoint_mtime_before_ns": before_mtime_ns,
        "checkpoint_mtime_after_ns": after_mtime_ns,
        "verifier_checkpoint_refreshed": refreshed,
    }


def _checkpoint_path_from_loop(loop_result: dict[str, Any]) -> Path | None:
    checkpoint = loop_result.get("learned_verifier_checkpoint")
    if not checkpoint:
        return None
    path = Path(str(checkpoint))
    return path if path.is_absolute() else REPO / path


def summarize_loop_result(
    *,
    game: str,
    loop_result: dict[str, Any] | None,
    loop_result_path: str,
    checkpoint_mtime_before_ns: int | None,
) -> dict[str, Any]:
    before_ns = _as_optional_int(checkpoint_mtime_before_ns)
    if not isinstance(loop_result, dict):
        return {
            "game": game,
            "loop_result_path": loop_result_path,
            "checkpoint_path": None,
            "checkpoint_mtime_before_ns": before_ns,
            "checkpoint_mtime_after_ns": None,
            "verifier_checkpoint_refreshed": False,
            "states_expanded": 0,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "reproduction_gate": {},
            "self_play_residual": "loop_result_missing",
            "loop_solve_provenance": None,
        }

    gate = _gate(loop_result)
    reproduced = bool(loop_result.get("offline_reproduced") and gate.get("reproduced", True))
    reproduced_levels = _as_int(loop_result.get("reproduced_levels") or loop_result.get("reached_level"), 0)
    states_expanded = _as_int(loop_result.get("states_expanded"), 0)
    checkpoint_path = _checkpoint_path_from_loop(loop_result)
    checkpoint = _checkpoint_status(checkpoint_path, before_ns)

    if not reproduced or reproduced_levels < 1:
        residual = "reproduction_gate_failed"
    elif checkpoint_path is None:
        residual = "checkpoint_not_reported"
    elif not checkpoint["verifier_checkpoint_refreshed"]:
        residual = "checkpoint_mtime_not_advanced"
    else:
        residual = "checkpoint_refreshed_gate_passed"

    return {
        "game": game,
        "loop_result_path": loop_result_path,
        "checkpoint_path": _relative_path(checkpoint_path) if checkpoint_path else None,
        "checkpoint_mtime_before_ns": checkpoint["checkpoint_mtime_before_ns"],
        "checkpoint_mtime_after_ns": checkpoint["checkpoint_mtime_after_ns"],
        "verifier_checkpoint_refreshed": bool(checkpoint["verifier_checkpoint_refreshed"]),
        "states_expanded": states_expanded,
        "offline_reproduced": reproduced,
        "reproduced_levels": reproduced_levels if reproduced else 0,
        "reproduction_gate": gate,
        "self_play_residual": residual,
        "loop_solve_provenance": loop_result.get("solve_provenance"),
    }


def build_artifact(
    *,
    target_selection: dict[str, Any],
    loop_summary: dict[str, Any],
    preconditions_checked: dict[str, Any],
) -> dict[str, Any]:
    target_game = str(target_selection.get("game") or loop_summary.get("game") or "none")
    reproduced_levels = _as_int(loop_summary.get("reproduced_levels"), 0)
    checkpoint_refreshed = bool(loop_summary.get("verifier_checkpoint_refreshed"))
    gate_green = bool(loop_summary.get("offline_reproduced") and reproduced_levels >= 1)
    if checkpoint_refreshed and gate_green:
        verdict = f"success_{target_game}_L{reproduced_levels}_checkpoint_refreshed"
    else:
        verdict = f"complete_{target_game}_self_play_residual_{loop_summary.get('self_play_residual')}"

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": verdict,
        "verifier_checkpoint_refreshed": checkpoint_refreshed,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "solve_provenance": SOLVE_PROVENANCE,
        "preconditions_checked": dict(preconditions_checked),
        "target_game": target_game,
        "target_registry_prior_level": _as_int(target_selection.get("prior_level"), 0),
        "target_selection_reason": target_selection.get("reason"),
        "checkpoint_path": loop_summary.get("checkpoint_path"),
        "checkpoint_mtime_before_ns": loop_summary.get("checkpoint_mtime_before_ns"),
        "checkpoint_mtime_after_ns": loop_summary.get("checkpoint_mtime_after_ns"),
        "states_expanded": _as_int(loop_summary.get("states_expanded"), 0),
        "offline_reproduced": bool(loop_summary.get("offline_reproduced")),
        "reproduced_levels": reproduced_levels,
        "reproduction_gate": dict(loop_summary.get("reproduction_gate") or {}),
        "self_play_residual": str(loop_summary.get("self_play_residual")),
        "loop_result_path": loop_summary.get("loop_result_path"),
        "loop_solve_provenance": loop_summary.get("loop_solve_provenance"),
        "model_specs": {
            "standing_loop": "scripts/arc_loop_solve.py",
            "checkpoint_template": "models/arc_verifier_<game>.json",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "schema_errors": [],
    }
    artifact["reproducibility_checksum"] = stable_checksum(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: dict[str, Any],
    target_game: str | None = None,
) -> dict[str, Any]:
    target = target_game or "none"
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": f"blocked_{reason}",
        "verifier_checkpoint_refreshed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "solve_provenance": SOLVE_PROVENANCE,
        "preconditions_checked": dict(preconditions_checked),
        "target_game": target,
        "target_registry_prior_level": 0,
        "target_selection_reason": reason,
        "checkpoint_path": None,
        "checkpoint_mtime_before_ns": None,
        "checkpoint_mtime_after_ns": None,
        "states_expanded": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "reproduction_gate": {},
        "self_play_residual": reason,
        "loop_result_path": None,
        "loop_solve_provenance": None,
        "model_specs": {
            "standing_loop": "scripts/arc_loop_solve.py",
            "checkpoint_template": "models/arc_verifier_<game>.json",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "schema_errors": [],
    }
    artifact["reproducibility_checksum"] = stable_checksum(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


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
    if success and not refreshed:
        errors.append("success_without_refreshed_checkpoint")
    if success and not (payload.get("offline_reproduced") is True and _as_int(payload.get("reproduced_levels")) >= 1):
        errors.append("success_without_reproduction_gate")
    if success and _as_int(payload.get("states_expanded")) < 1:
        errors.append("success_without_search_states")
    if refreshed and not payload.get("checkpoint_path"):
        errors.append("refreshed_checkpoint_missing_path")
    if refreshed and after_ns is None:
        errors.append("refreshed_checkpoint_missing_mtime")
    if refreshed and before_ns is not None and after_ns is not None and after_ns <= before_ns:
        errors.append("refreshed_checkpoint_without_mtime_advance")
    return errors


def write_artifact(payload: dict[str, Any], path: Path | None = None) -> Path:
    output = ARTIFACT if path is None else Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def check_preconditions(game: str | None = None) -> dict[str, Any]:  # pragma: no cover - live environment boundary.
    preconditions: dict[str, Any] = {
        "AGENTS.md": (REPO / "AGENTS.md").exists(),
        "CODEX.md": (REPO / "CODEX.md").exists(),
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        preconditions["offline_arcade"] = {"ok": True, "check": "arc_solver_kit.offline_arcade()"}
    except Exception as exc:
        preconditions["offline_arcade"] = {
            "ok": False,
            "check": "arc_solver_kit.offline_arcade()",
            "error": repr(exc),
        }
        preconditions["ok"] = False
        return preconditions

    try:
        registry = load_registry(REGISTRY)
        preconditions["registry_loadable"] = {"ok": True, "path": REGISTRY_RELATIVE_PATH}
    except Exception as exc:
        preconditions["registry_loadable"] = {
            "ok": False,
            "path": REGISTRY_RELATIVE_PATH,
            "error": repr(exc),
        }
        preconditions["ok"] = False
        return preconditions

    selection = select_banked_target(registry, preferred_game=game)
    preconditions["target_selection"] = selection
    preconditions["target_banked"] = {
        "ok": bool(selection.get("banked")),
        "game": selection.get("game"),
        "prior_level": selection.get("prior_level"),
        "reason": selection.get("reason"),
    }
    preconditions["ok"] = bool(selection.get("banked"))
    return preconditions


def _precondition_failure(preconditions: dict[str, Any]) -> str | None:
    if preconditions.get("offline_arcade", {}).get("ok") is not True:
        return "offline_arcade_missing"
    if preconditions.get("registry_loadable", {}).get("ok") is not True:
        return "registry_missing"
    if preconditions.get("target_banked", {}).get("ok") is not True:
        return "target_not_banked"
    return None


def _read_loop_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


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
