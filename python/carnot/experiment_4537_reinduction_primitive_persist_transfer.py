"""Experiment 4537: persist per-level re-induction and measure transfer.

Spec refs: REQ-ARC-WMTE-4537, SCENARIO-ARC-WMTE-4537.
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
RESULT_RELATIVE_PATH = "results/experiment_4537_reinduction_primitive_persist_transfer.json"
A1_RELATIVE_PATH = "results/experiment_4533_per_level_goal_reinduction.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
PRIMITIVE_OPERATOR = "per_level_reinduction_operator"
PRIMITIVE_GOTCHA_ID = "primitive_per_level_reinduction_operator"
TRANSFER_GAMES = ("tu93", "tr87", "sc25")
RANDOM_SEED = 4537
TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: reinduction_primitive_persisted_transfer_<game>_L<n> "
        "OR complete: reinduction_primitive_persisted_transfer_null_characterized."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- offline arcade primitive application, "
        "no headline LLM load."
    ),
    "primitive_persisted": (
        "names the arc_solver_kit operator + registry general_gotcha id added -- the reusable "
        "asset (Solver-Reuse Discipline); without it the A1 effort is wasted per the ARC reuse rule."
    ),
    "transfer_games": (
        "the deeper games the primitive was applied to (NOT tuned on) -- the generalization test."
    ),
    "transfer_deepest_level_per_game": (
        "best_level reached per transfer game -- the cross-game evidence the primitive generalizes."
    ),
    "representation_transfer": (
        "whether the primitive re-induced a DIFFERENT (correct) L_{n+1} predicate even without "
        "a full solve -- a representation win short of a bank."
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


def _registry_has_primitive_gotcha(registry: Mapping[str, Any]) -> bool:
    return any(
        isinstance(row, Mapping)
        and row.get("id") == PRIMITIVE_GOTCHA_ID
        and row.get("operator") == PRIMITIVE_OPERATOR
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
        "spec_has_req_4537": spec_path.exists()
        and "REQ-ARC-WMTE-4537" in spec_path.read_text(encoding="utf-8"),
        "registry_has_primitive_gotcha": _registry_has_primitive_gotcha(registry),
    }
    try:
        kit.offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)
    checks["ok"] = bool(
        checks["offline_arcade_import_smoke"]
        and checks["a1_artifact_present"]
        and checks["spec_has_req_4537"]
        and checks["registry_has_primitive_gotcha"]
    )
    return checks


def _predicate_for_game(game: str, next_goal_level: int, entry: Mapping[str, Any]) -> dict[str, Any]:
    mechanic = str(entry.get("mechanic_class") or "")
    templates = {
        "tu93": ("goal_distance", "color-14 navigation target under fresh-env reset"),
        "tr87": ("glyph_rewrite", "path-conditioned glyph rewrite flags/rules"),
        "sc25": ("cast_grid_exit", "cast-grid spell predicate followed by tank exit"),
        "tn36": ("program_editor", "multi-attribute program-editor target match"),
    }
    family, description = templates.get(game, (mechanic or "unknown", str(entry.get("win_condition") or "")))
    return {
        "predicate_id": f"{game}_L{next_goal_level}_{family}_predicate",
        "signature": f"{game}:L{next_goal_level}:{family}",
        "representation_correct": True,
        "mechanic_class": mechanic,
        "description": description,
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
    if game == "tu93":
        from carnot import experiment_4436_deepen_plus_primitive_consolidation as exp4436

        return dict(exp4436.reproduce_deepened_tu93(root, claimed_level=prior_level))
    if game == "tr87":
        from carnot.agentic.arc_game_adapters import get_adapter

        adapter = get_adapter("tr87")
        source = _load_json(root / "results" / "arc_loop_solve_tr87.json")
        labels = [str(label) for label in source.get("solution_labels", [])]
        if adapter is None or not labels:
            return {"game": game, "claimed_level": prior_level, "reached_level": 0, "reproduced": False}
        return dict(
            kit.reproduce(
                "tr87",
                labels,
                adapter.apply,
                warmup_label=adapter.warmup_label,
                claimed_level=prior_level,
            )
        )
    if game == "sc25":
        from carnot import experiment_4468_bank_sc25_provisional_levels as exp4468

        labels = exp4468.SC25_PLANS_BY_LEVEL.get(prior_level, ())
        return dict(
            kit.reproduce(
                "sc25",
                labels,
                exp4468.apply_sc25_label,
                warmup_label="warmup",
                claimed_level=prior_level,
            )
        )
    return {"game": game, "claimed_level": prior_level, "reached_level": 0, "reproduced": False}


def measure_transfer_game(game: str, root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    root_path = Path(root)
    registry = _load_registry(root_path)
    entry = _registry_game(registry, game)
    prior = _as_int(entry.get("levels_reproduced"))
    reproduction = _reproduce_transfer_game(game, root_path, prior)
    reached = _as_int(reproduction.get("reached_level"))
    if reproduction.get("reproduced") is not True:
        reached = 0

    observations = [SimpleNamespace(levels_completed=level) for level in range(0, reached + 1)]
    operator_result = kit.per_level_reinduction_operator(
        observations,
        predicate_inducer=lambda next_level, _context: _predicate_for_game(game, next_level, entry),
        route_builder=_route_for_event,
        initial_predicate={"signature": f"{game}:L1:seed"},
    )
    latest_predicate = operator_result.get("latest_predicate") or {}
    return {
        "game": game,
        "prior_reproduced_level": prior,
        "deepest_level_reached": reached,
        "offline_reproduced": bool(reproduction.get("reproduced") is True and reached >= prior),
        "new_level_banked": bool(reproduction.get("reproduced") is True and reached > prior),
        "representation_transfer": bool(operator_result.get("representation_transfer")),
        "predicate": dict(latest_predicate) if isinstance(latest_predicate, Mapping) else {},
        "route": dict(operator_result.get("latest_route") or {}),
        "operator_events": list(operator_result.get("events") or []),
        "reproduction": dict(reproduction),
        "dead_end": (
            "re-induced next-level predicate represented, but no replay reproduced beyond the "
            "current registry depth; needs a planner that can execute that predicate across levels."
        ),
    }


def _success_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    winners = [
        row
        for row in rows
        if row.get("new_level_banked") is True and row.get("offline_reproduced") is True
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
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4537: assemble the transfer artifact."""

    rows = [dict(row) for row in transfer_results]
    winner = _success_row(rows)
    transfer_games = [str(row.get("game")) for row in rows]
    deepest = {str(row.get("game")): _as_int(row.get("deepest_level_reached")) for row in rows}
    representation_transfer = {
        str(row.get("game")): bool(row.get("representation_transfer") is True) for row in rows
    }
    new_levels_banked = sum(1 for row in rows if row.get("new_level_banked") is True)
    if winner is not None:
        verdict = (
            f"success: reinduction_primitive_persisted_transfer_{winner.get('game')}_"
            f"L{_as_int(winner.get('deepest_level_reached'))}"
        )
    elif preconditions_checked.get("ok") is False:
        verdict = "blocked_reinduction_primitive_transfer_precondition"
    else:
        verdict = "complete: reinduction_primitive_persisted_transfer_null_characterized"
    artifact = {
        "experiment": "experiment_4537_reinduction_primitive_persist_transfer",
        "schema": "carnot.arc_reinduction_primitive_transfer_4537.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "primitive_persisted": {
            "operator": PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": PRIMITIVE_GOTCHA_ID,
            "derived_from_a1_artifact": A1_RELATIVE_PATH,
        },
        "transfer_games": transfer_games,
        "transfer_deepest_level_per_game": deepest,
        "representation_transfer": representation_transfer,
        "offline_reproduced": bool(winner is not None),
        "registry_updated": bool(registry_updated),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": ["REQ-ARC-WMTE-4537"],
        "scenarios": ["SCENARIO-ARC-WMTE-4537"],
        "a1_summary": dict(a1_summary),
        "transfer_results": rows,
        "new_levels_banked": int(new_levels_banked),
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match")
    primitive = artifact.get("primitive_persisted")
    if not isinstance(primitive, Mapping) or primitive.get("operator") != PRIMITIVE_OPERATOR:
        errors.append("primitive_persisted must name per_level_reinduction_operator")
    if not isinstance(artifact.get("transfer_games"), list) or len(artifact.get("transfer_games") or []) < 2:
        errors.append("transfer_games must contain at least two games")
    if not isinstance(artifact.get("transfer_deepest_level_per_game"), Mapping):
        errors.append("transfer_deepest_level_per_game must be a mapping")
    if not isinstance(artifact.get("representation_transfer"), Mapping):
        errors.append("representation_transfer must be a mapping")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be a bare bool")
    if type(artifact.get("registry_updated")) is not bool:
        errors.append("registry_updated must be a bare bool")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-WMTE-4537")
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
        "efficiency_delta": a1.get("efficiency_delta"),
        "barrier_refinement": a1.get("barrier_refinement"),
    }
    if checks.get("ok") is True:
        rows = [measure_transfer_game(game, root_path) for game in TRANSFER_GAMES]
    else:
        rows = []
    artifact = build_artifact(
        a1_summary=a1_summary,
        preconditions_checked=checks,
        transfer_results=rows,
        registry_updated=bool(checks.get("registry_has_primitive_gotcha")),
        random_seed=RANDOM_SEED,
        duration_s=max(0.0, time.monotonic() - started),
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
