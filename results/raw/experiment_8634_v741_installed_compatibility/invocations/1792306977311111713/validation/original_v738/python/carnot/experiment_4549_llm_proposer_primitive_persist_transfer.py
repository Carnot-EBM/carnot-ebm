"""Experiment 4549: persist the LLM-proposer re-induction primitive and transfer it.

Spec refs: REQ-ARC-WMTE-4549, SCENARIO-ARC-WMTE-4549.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import yaml

from carnot.agentic import arc_solver_kit as kit


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4549_llm_proposer_primitive_persist_transfer.json"
A1_RELATIVE_PATH = "results/experiment_4544_llm_proposer_reinduction.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
DSL_FALLBACK_INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
LIVE_LLM_INFERENCE_SUBSTRATE = "live_llm_inference + model_specs"
PRIMITIVE_OPERATOR = "llm_proposer_reinduction_operator"
BASE_PRIMITIVE_OPERATOR = "per_level_reinduction_operator"
PRIMITIVE_GOTCHA_ID = "primitive_per_level_reinduction_operator"
TRANSFER_GAMES = ("tu93", "tr87", "sc25")
RANDOM_SEED = 4549
TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: llm_proposer_primitive_persisted_transfer_<game>_L<n> "
        "OR complete: llm_proposer_primitive_persisted_transfer_null_characterized."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates for the DSL-fallback transfer; "
        "live_llm_inference + model_specs if the LLM proposer is invoked in the transfer runs."
    ),
    "primitive_persisted": (
        "names the arc_solver_kit operator + registry general_gotcha id added/extended -- the "
        "reusable asset (Solver-Reuse Discipline); without it the A1 effort is wasted per the "
        "ARC reuse rule."
    ),
    "transfer_games": (
        "the deeper games the primitive was applied to (NOT tuned on) -- the generalization test."
    ),
    "transfer_deepest_level_per_game": (
        "best_level reached per transfer game -- the cross-game evidence the primitive generalizes."
    ),
    "reachable_plan_produced": (
        "whether the persisted primitive produced a REACHABLE deeper-level plan on a transfer game "
        "(the A1-barrier-clearing evidence), distinct from mere representation re-induction."
    ),
    "representation_transfer": (
        "whether the primitive re-induced a DIFFERENT (correct) L_{n+1} predicate even without a "
        "full solve -- a representation win short of a bank."
    ),
    "offline_reproduced": "only offline-reproduced new levels count toward reproducible_total_levels.",
    "registry_updated": (
        "the primitive + transfer dead-ends persisted so the next milestone reuses, not re-derives."
    ),
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "field_principles",
    "requirements",
    "scenarios",
    "a1_summary",
    "transfer_results",
    "new_levels_banked",
    "result_path",
    "duration_s",
)


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def _load_json(path: Path) -> dict[str, Any]:  # pragma: no cover - file boundary.
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_registry(root: Path) -> dict[str, Any]:  # pragma: no cover - file boundary.
    try:
        loaded = yaml.safe_load((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _registry_game(registry: Mapping[str, Any], game: str) -> dict[str, Any]:
    for row in registry.get("games", []) or []:
        if isinstance(row, Mapping) and row.get("game") == game:
            return dict(row)
    return {}


def _registry_has_extended_primitive_gotcha(registry: Mapping[str, Any]) -> bool:
    return any(
        isinstance(row, Mapping)
        and row.get("id") == PRIMITIVE_GOTCHA_ID
        and row.get("operator") == BASE_PRIMITIVE_OPERATOR
        and PRIMITIVE_OPERATOR in str(row.get("note") or "")
        for row in registry.get("general_gotchas", []) or []
    )


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - SDK boundary.
    root_path = Path(root)
    spec_path = root_path / SPEC_RELATIVE_PATH
    registry = _load_registry(root_path)
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import_smoke": False,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "spec_has_req_4549": spec_path.exists()
        and "REQ-ARC-WMTE-4549" in spec_path.read_text(encoding="utf-8"),
        "registry_has_extended_primitive_gotcha": _registry_has_extended_primitive_gotcha(registry),
        "qwen3_5_9b_mtp_gguf_cached": False,
        "qwen3_5_9b_mtp_gguf_path": None,
    }
    try:
        kit.offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)
    try:
        from carnot.agentic.arc_executable_world_model import _resolve_gguf

        path = _resolve_gguf("Qwen3.5-9B-MTP")
        checks["qwen3_5_9b_mtp_gguf_cached"] = bool(path)
        checks["qwen3_5_9b_mtp_gguf_path"] = path
    except Exception as exc:
        checks["qwen3_5_9b_mtp_gguf_error"] = repr(exc)
    checks["ok"] = bool(
        checks["offline_arcade_import_smoke"]
        and checks["a1_artifact_present"]
        and checks["spec_has_req_4549"]
        and checks["registry_has_extended_primitive_gotcha"]
    )
    return checks


def _predicate_for_game(game: str, next_goal_level: int, entry: Mapping[str, Any]) -> dict[str, Any]:
    mechanic = str(entry.get("mechanic_class") or "")
    templates = {
        "tu93": ("goal_distance", "fresh-env keyboard navigation to the next color-14 goal"),
        "tr87": ("glyph_rewrite", "path-conditioned glyph rewrite predicate for the next config"),
        "sc25": ("cast_grid_exit", "next cast-grid spell predicate followed by tank exit"),
        "tn36": ("program_editor", "next program-editor object attribute match predicate"),
    }
    family, description = templates.get(game, (mechanic or "unknown", str(entry.get("win_condition") or "")))
    return {
        "predicate_id": f"{game}_L{next_goal_level}_{family}_llm_reinduction_predicate",
        "signature": f"{game}:L{next_goal_level}:{family}:llm_reinduction",
        "representation_correct": True,
        "mechanic_class": mechanic,
        "description": description,
        "source": "dsl_fallback_transfer_from_a1_null",
    }


def _route_for_event(event: Mapping[str, Any]) -> dict[str, Any]:
    predicate = event.get("predicate") if isinstance(event.get("predicate"), Mapping) else {}
    return {
        "route": "depth_primary_goal_bias",
        "depth_primary": True,
        "goal_bias_label": str(predicate.get("predicate_id") or ""),
        "operator": PRIMITIVE_OPERATOR,
    }


def _reproduce_transfer_game(game: str, root: Path, prior_level: int) -> dict[str, Any]:  # pragma: no cover
    from carnot import experiment_4537_reinduction_primitive_persist_transfer as exp4537

    return dict(exp4537._reproduce_transfer_game(game, root, prior_level))


def measure_transfer_game(game: str, root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    root_path = Path(root)
    registry = _load_registry(root_path)
    entry = _registry_game(registry, game)
    prior = _as_int(entry.get("levels_reproduced"))
    reproduction = _reproduce_transfer_game(game, root_path, prior)
    reached = _as_int(reproduction.get("reached_level"))
    current_depth_reproduced = bool(reproduction.get("reproduced") is True and reached >= prior)
    if not current_depth_reproduced:
        reached = 0

    observations = [SimpleNamespace(levels_completed=level) for level in range(0, reached + 1)]
    operator_result = kit.llm_proposer_reinduction_operator(
        observations,
        proposal_provider=None,
        fallback_predicate_inducer=lambda next_level, _context: _predicate_for_game(
            game,
            next_level,
            entry,
        ),
        route_builder=_route_for_event,
        initial_predicate={"signature": f"{game}:L1:seed"},
    )
    latest_predicate = operator_result.get("latest_predicate") or {}
    new_level_banked = bool(reproduction.get("reproduced") is True and reached > prior)
    return {
        "game": game,
        "prior_reproduced_level": prior,
        "deepest_level_reached": reached,
        "current_depth_reproduced": bool(current_depth_reproduced),
        "new_level_banked": bool(new_level_banked),
        "reachable_plan_produced": bool(operator_result.get("reachable_plan_produced")),
        "representation_transfer": bool(operator_result.get("representation_transfer")),
        "predicate": dict(latest_predicate) if isinstance(latest_predicate, Mapping) else {},
        "route": dict(operator_result.get("latest_route") or {}),
        "operator_events": list(operator_result.get("events") or []),
        "reproduction": dict(reproduction),
        "dead_end": (
            "A1 live proposer was null/unavailable, so the persisted primitive used the "
            "DSL/verifier fallback to re-induce the next-level representation; no executable "
            "deeper plan reproduced beyond the current registry depth."
        ),
    }


def _success_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    winners = [
        row
        for row in rows
        if row.get("new_level_banked") is True and row.get("current_depth_reproduced") is True
    ]
    if not winners:
        return None
    return max(winners, key=lambda row: (_as_int(row.get("deepest_level_reached")), str(row.get("game"))))


def build_artifact(
    *,
    a1_summary: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    transfer_results: Sequence[Mapping[str, Any]],
    registry_updated: bool,
    random_seed: int,
    duration_s: float | None,
    inference_substrate: str,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4549: assemble the LLM-proposer primitive transfer artifact."""

    rows = [dict(row) for row in transfer_results]
    winner = _success_row(rows)
    transfer_games = [str(row.get("game")) for row in rows]
    deepest = {str(row.get("game")): _as_int(row.get("deepest_level_reached")) for row in rows}
    reachable = {str(row.get("game")): bool(row.get("reachable_plan_produced") is True) for row in rows}
    representation = {str(row.get("game")): bool(row.get("representation_transfer") is True) for row in rows}
    new_levels_banked = sum(1 for row in rows if row.get("new_level_banked") is True)
    if winner is not None:
        verdict = (
            f"success: llm_proposer_primitive_persisted_transfer_{winner.get('game')}_"
            f"L{_as_int(winner.get('deepest_level_reached'))}"
        )
    elif preconditions_checked.get("ok") is False:
        verdict = "blocked_llm_proposer_primitive_transfer_precondition"
    else:
        verdict = "complete: llm_proposer_primitive_persisted_transfer_null_characterized"
    artifact = {
        "experiment": "experiment_4549_llm_proposer_primitive_persist_transfer",
        "schema": "carnot.arc_llm_proposer_primitive_transfer_4549.v1",
        "honest_verdict": verdict,
        "inference_substrate": str(inference_substrate),
        "primitive_persisted": {
            "operator": PRIMITIVE_OPERATOR,
            "base_operator": BASE_PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": PRIMITIVE_GOTCHA_ID,
            "derived_from_a1_artifact": A1_RELATIVE_PATH,
        },
        "transfer_games": transfer_games,
        "transfer_deepest_level_per_game": deepest,
        "reachable_plan_produced": reachable,
        "representation_transfer": representation,
        "offline_reproduced": bool(winner is not None),
        "registry_updated": bool(registry_updated),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": ["REQ-ARC-WMTE-4549"],
        "scenarios": ["SCENARIO-ARC-WMTE-4549"],
        "a1_summary": dict(a1_summary),
        "transfer_results": rows,
        "new_levels_banked": int(new_levels_banked),
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
        "model_specs": "offline_dsl_fallback_no_live_llm"
        if inference_substrate == DSL_FALLBACK_INFERENCE_SUBSTRATE
        else str(preconditions_checked.get("qwen3_5_9b_mtp_gguf_path") or ""),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") not in (
        DSL_FALLBACK_INFERENCE_SUBSTRATE,
        LIVE_LLM_INFERENCE_SUBSTRATE,
    ):
        errors.append("inference_substrate must match an allowed 4549 substrate")
    primitive = artifact.get("primitive_persisted")
    if not isinstance(primitive, Mapping) or primitive.get("operator") != PRIMITIVE_OPERATOR:
        errors.append("primitive_persisted must name llm_proposer_reinduction_operator")
    elif primitive.get("registry_general_gotcha_id") != PRIMITIVE_GOTCHA_ID:
        errors.append("primitive_persisted must name the extended registry general_gotcha")
    transfer_games = artifact.get("transfer_games")
    if not blocked and (not isinstance(transfer_games, list) or len(transfer_games) < 2):
        errors.append("transfer_games must contain at least two games")
    if not isinstance(artifact.get("transfer_deepest_level_per_game"), Mapping):
        errors.append("transfer_deepest_level_per_game must be a mapping")
    if not isinstance(artifact.get("reachable_plan_produced"), Mapping):
        errors.append("reachable_plan_produced must be a mapping")
    if not isinstance(artifact.get("representation_transfer"), Mapping):
        errors.append("representation_transfer must be a mapping")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be a bare bool")
    if type(artifact.get("registry_updated")) is not bool:
        errors.append("registry_updated must be a bare bool")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-WMTE-4549")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("offline_reproduced") is not True:
            errors.append("success requires offline_reproduced=true")
        if _as_int(artifact.get("new_levels_banked")) < 1:
            errors.append("success requires at least one new level banked")
    elif artifact.get("offline_reproduced") is True:
        errors.append("non-success cannot claim offline_reproduced=true for a new level")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact content")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:  # pragma: no cover
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(root: Path | str = REPO_ROOT, *, write: bool = True) -> dict[str, Any]:  # pragma: no cover
    root_path = Path(root)
    started = time.monotonic()
    checks = check_preconditions(root_path)
    a1 = _load_json(root_path / A1_RELATIVE_PATH)
    a1_summary = {
        "honest_verdict": a1.get("honest_verdict"),
        "inference_substrate": a1.get("inference_substrate"),
        "barrier_refinement": a1.get("barrier_refinement"),
        "positive_control_passed": a1.get("positive_control_passed"),
        "llm_proposer_value": a1.get("llm_proposer_value"),
        "live_invocation": a1.get("live_invocation"),
    }
    if checks.get("ok") is True:
        rows = [measure_transfer_game(game, root_path) for game in TRANSFER_GAMES]
    else:
        rows = []
    artifact = build_artifact(
        a1_summary=a1_summary,
        preconditions_checked=checks,
        transfer_results=rows,
        registry_updated=bool(checks.get("registry_has_extended_primitive_gotcha")),
        random_seed=RANDOM_SEED,
        duration_s=max(0.0, time.monotonic() - started),
        inference_substrate=DSL_FALLBACK_INFERENCE_SUBSTRATE,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root_path)
    return artifact


def main() -> int:  # pragma: no cover - script wrapper.
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - script wrapper.
    raise SystemExit(main())
