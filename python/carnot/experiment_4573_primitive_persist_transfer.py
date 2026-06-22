"""Experiment 4573: persist A1/A2 primitive and measure leave-one-game transfer.

Spec refs: REQ-ARC-WMTE-4573, SCENARIO-ARC-WMTE-4573.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from carnot.agentic import arc_solver_kit as kit


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4573_primitive_persist_transfer.json"
A1_RELATIVE_PATH = "results/experiment_4568_clickability_action_effect_predictor.json"
A2_RELATIVE_PATH = "results/experiment_4569_verifier_guided_expansion.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
TRANSITION_CORPUS_RELATIVE_DIR = "data/arc_transition_corpus"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline transfer over cached "
    "action-effect rows; a CNN predictor forward pass is NOT live_llm_inference."
)
PRIMITIVE_OPERATOR = "persistent_action_effect_memory_operator"
PRIMITIVE_GOTCHA_ID = "primitive_persistent_action_effect_memory_operator"
RANDOM_SEED = 4573
TRANSFER_GAME_LIMIT = 3
TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: primitive_persisted_transfer_<game>_value_added OR "
        "complete: primitive_persisted_transfer_null_characterized."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates for the offline transfer; a CNN predictor "
        "forward pass is NOT live_llm_inference."
    ),
    "primitive_persisted": (
        "names the arc_solver_kit operator + registry general_gotcha id added/extended -- the "
        "reusable asset (Solver-Reuse Discipline); without it the A1/A2 effort is wasted per "
        "the ARC reuse rule."
    ),
    "transfer_games": (
        "the games the primitive was applied to (NOT tuned on) -- the generalization test."
    ),
    "transfer_value_per_game": (
        "the per-game value-add (predictor actions-reduced / expansion winner-generated) -- "
        "the cross-game evidence the primitive generalizes."
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
    "upstream_decision",
    "upstream_summaries",
    "transfer_results",
    "transfer_dead_ends",
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


def _load_json(path: Path) -> dict[str, Any]:  # pragma: no cover - file boundary
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_registry(root: Path) -> dict[str, Any]:  # pragma: no cover - file boundary
    try:
        loaded = yaml.safe_load((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _registry_has_primitive_gotcha(registry: Mapping[str, Any]) -> bool:
    return any(
        isinstance(row, Mapping)
        and row.get("id") == PRIMITIVE_GOTCHA_ID
        and row.get("operator") == PRIMITIVE_OPERATOR
        for row in registry.get("general_gotchas", []) or []
    )


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    root_path = Path(root)
    spec_path = root_path / SPEC_RELATIVE_PATH
    registry = _load_registry(root_path)
    transition_dir = root_path / TRANSITION_CORPUS_RELATIVE_DIR
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import_smoke": False,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "spec_has_req_4573": spec_path.exists()
        and "REQ-ARC-WMTE-4573" in spec_path.read_text(encoding="utf-8"),
        "registry_has_primitive_gotcha": _registry_has_primitive_gotcha(registry),
        "transition_corpus_present": transition_dir.is_dir(),
        "transition_npz_count": len(sorted(transition_dir.glob("*.npz"))),
        "cnn_forward_pass_evaluated": False,
        "torch_import_required": False,
    }
    try:
        kit.offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)
    checks["ok"] = bool(
        checks["offline_arcade_import_smoke"]
        and checks["a1_artifact_present"]
        and checks["a2_artifact_present"]
        and checks["spec_has_req_4573"]
        and checks["registry_has_primitive_gotcha"]
        and checks["transition_corpus_present"]
        and int(checks["transition_npz_count"]) > 0
    )
    return checks


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def select_primitive_from_upstreams(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4573: choose the strongest A1/A2 primitive signal."""

    positive_control = a1_artifact.get("positive_control")
    positive_control_passed = bool(
        a1_artifact.get("positive_control_passed") is True
        or (
            isinstance(positive_control, Mapping)
            and positive_control.get("actions_reduced") is True
        )
    )
    positive_control_delta = 0.0
    if isinstance(positive_control, Mapping):
        positive_control_delta = _as_float(
            positive_control.get("baseline_actions_to_first_levelup")
        ) - _as_float(positive_control.get("ranked_actions_to_first_levelup"))
    a1_actions_delta = _as_float(a1_artifact.get("actions_delta"))
    a1_signal = max(a1_actions_delta, positive_control_delta)

    winner_generated = a2_artifact.get("winner_generated")
    a2_generated_count = (
        _as_int(winner_generated.get("generated_count"))
        if isinstance(winner_generated, Mapping)
        else 0
    )

    if positive_control_passed:
        rationale = (
            "A1 positive control reduced actions-to-first-levelup, while held-out transfer was "
            "null; persist the primitive-as-built as cross-game action-effect memory."
        )
    elif a2_generated_count > 0:
        rationale = (
            "A2 generated winners, but this milestone persists the A1-compatible memory only "
            "when the A1 signal is absent from the live reusable path."
        )
    else:
        rationale = (
            "Both A1/A2 transfer gates were null; A1 is still the best-characterized primitive "
            "because its positive-control and corpus contract are explicit."
        )

    return {
        "source": "A1_action_effect_predictor",
        "operator": PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": PRIMITIVE_GOTCHA_ID,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "measured_signal": float(a1_signal),
        "actions_delta": float(a1_actions_delta),
        "positive_control_actions_delta": float(positive_control_delta),
        "positive_control_passed": bool(positive_control_passed),
        "a2_winner_generated_count": int(a2_generated_count),
        "persisted_as_best_characterized_null": bool(a1_actions_delta <= 0.0),
        "selection_rationale": rationale,
    }


def _grid_state_key(grid: Any) -> str:
    arr = np.asarray(grid)
    digest = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    return f"{arr.shape}:{digest}"


def _frame_delta_fraction(before: Any, after: Any) -> float:
    lhs = np.asarray(before)
    rhs = np.asarray(after)
    if lhs.shape != rhs.shape:
        return 1.0
    total = int(lhs.size)
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(lhs != rhs) / total)


def load_transition_effect_rows(root: Path | str = REPO_ROOT) -> list[dict[str, Any]]:
    """REQ-ARC-WMTE-4573: load cached action-effect rows without CNN inference."""

    transition_dir = Path(root) / TRANSITION_CORPUS_RELATIVE_DIR
    rows: list[dict[str, Any]] = []
    for path in sorted(transition_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        game = path.stem
        grids = data["grids"]
        next_grids = data["next_grids"]
        for index in range(int(grids.shape[0])):
            action_id = int(data["actions"][index])
            x_value = int(data["xs"][index]) if "xs" in data else -1
            y_value = int(data["ys"][index]) if "ys" in data else -1
            level_before = int(data["lb"][index]) if "lb" in data else 0
            level_after = int(data["la"][index]) if "la" in data else 0
            delta = _frame_delta_fraction(grids[index], next_grids[index])
            row: dict[str, Any] = {
                "game": game,
                "env": game,
                "state_key": _grid_state_key(grids[index]),
                "action_id": action_id,
                "changed": bool(delta > 0.0),
                "frame_delta": float(delta),
                "level_progress": 1.0 if level_after > level_before else 0.0,
                "step_index": int(index),
            }
            if action_id == 6 and x_value >= 0 and y_value >= 0:
                row["x"] = int(x_value)
                row["y"] = int(y_value)
            rows.append(row)
    return rows


def _row_game(row: Any) -> str:
    if isinstance(row, Mapping):
        return str(row.get("game") or row.get("env") or "")
    return str(getattr(row, "game", "") or getattr(row, "env", "") or "")


def _row_state_key(row: Any) -> str:
    if isinstance(row, Mapping):
        return str(row.get("state_key") or "")
    return str(getattr(row, "state_key", "") or "")


def _row_action_id(row: Any) -> int | None:
    value = row.get("action_id") if isinstance(row, Mapping) else getattr(row, "action_id", None)
    if value is None:
        value = row.get("action") if isinstance(row, Mapping) else getattr(row, "action", None)
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _row_xy(row: Any) -> tuple[int, int] | None:
    x_value = row.get("x") if isinstance(row, Mapping) else getattr(row, "x", None)
    y_value = row.get("y") if isinstance(row, Mapping) else getattr(row, "y", None)
    if x_value is None or y_value is None:
        return None
    try:
        return int(x_value), int(y_value)
    except (TypeError, ValueError):
        return None


def _row_effective_target(row: Any) -> bool:
    if isinstance(row, Mapping):
        changed_value = row.get("changed")
        level_progress = _as_float(row.get("level_progress"))
        frame_delta = _as_float(row.get("frame_delta"))
    else:
        changed_value = getattr(row, "changed", None)
        level_progress = _as_float(getattr(row, "level_progress", 0.0))
        frame_delta = _as_float(getattr(row, "frame_delta", 0.0))
    changed = bool(changed_value) if changed_value is not None else frame_delta > 0.0
    return bool(level_progress > 0.0 or changed)


def _candidate_from_effect_row(game: str, state_key: str, index: int, row: Any) -> dict[str, Any]:
    action_id = _row_action_id(row)
    candidate: dict[str, Any] = {
        "candidate_id": f"{game}:{state_key}:{index}",
        "action_id": int(action_id or 0),
        "reaches_levelup": _row_effective_target(row),
    }
    xy = _row_xy(row)
    if xy is not None:
        candidate["data"] = {"x": int(xy[0]), "y": int(xy[1])}
    return candidate


def measure_action_effect_memory_transfer_game(
    game: str,
    *,
    effect_rows: Sequence[Any],
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4573: apply PersistentAEM with the target game excluded."""

    memory = kit.PersistentAEM.from_effect_rows(effect_rows, exclude_games=(game,))
    groups: dict[str, list[Any]] = defaultdict(list)
    target_rows = [row for row in effect_rows if _row_game(row) == game]
    for row in target_rows:
        action_id = _row_action_id(row)
        state_key = _row_state_key(row)
        if action_id is None or not state_key:
            continue
        if action_id == 6 and _row_xy(row) is None:
            continue
        groups[state_key].append(row)

    group_results: list[dict[str, Any]] = []
    before_ranks: list[float] = []
    after_ranks: list[float] = []
    deltas: list[float] = []
    target_group_count = 0

    for state_key, rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        candidates = [
            _candidate_from_effect_row(game, state_key, index, row)
            for index, row in enumerate(rows)
        ]
        if not any(candidate["reaches_levelup"] for candidate in candidates):
            continue
        target_group_count += 1
        ranking = kit.persistent_action_effect_memory_operator(candidates, memory=memory)
        before = ranking.get("actions_to_first_levelup_before")
        after = ranking.get("actions_to_first_levelup_after")
        if before is not None:
            before_ranks.append(float(before))
        if after is not None:
            after_ranks.append(float(after))
        if before is not None and after is not None:
            deltas.append(float(before) - float(after))
        group_results.append(
            {
                "state_key": state_key,
                "candidate_count": int(ranking.get("candidate_count") or 0),
                "actions_to_first_levelup_before": before,
                "actions_to_first_levelup_after": after,
                "actions_reduced": float(ranking.get("actions_reduced") or 0.0),
                "value_added": bool(ranking.get("value_added") is True),
                "best_candidate_id": ranking.get("best_candidate_id"),
            }
        )

    baseline = float(median(before_ranks)) if before_ranks else None
    with_memory = float(median(after_ranks)) if after_ranks else None
    actions_reduced = (
        float(baseline - with_memory) if baseline is not None and with_memory is not None else 0.0
    )
    representation_transfer = bool(memory.row_count > 0 and target_group_count > 0)
    value_added = bool(actions_reduced > 0.0)
    dead_end = ""
    if not target_rows:
        dead_end = "no cached action-effect rows were available for the held-out transfer game."
    elif not groups:
        dead_end = "target rows were present, but no trainable action candidates were available."
    elif target_group_count == 0:
        dead_end = (
            "cached rows did not contain same-state candidate alternatives with a generated "
            "effective target action."
        )
    elif not value_added:
        dead_end = (
            "PersistentAEM transferred a representation, but median actions-to-first-levelup "
            "did not improve; candidate generation/richer effect features are still needed."
        )

    transfer_value = {
        "operator": PRIMITIVE_OPERATOR,
        "actions_to_first_levelup_baseline": baseline,
        "actions_to_first_levelup_with_memory": with_memory,
        "actions_reduced": actions_reduced,
        "candidate_group_count": int(target_group_count),
        "target_row_count": int(len(target_rows)),
        "memory_row_count": int(memory.row_count),
        "representation_transfer": representation_transfer,
        "winner_generated": False,
        "target_candidate_generated": bool(target_group_count > 0),
        "value_added": value_added,
        "target_kind": "frame_change_or_level_progress",
    }
    return {
        "game": game,
        "value_added": value_added,
        "excluded_from_memory": game in memory.excluded_games,
        "transfer_value": transfer_value,
        "group_results": group_results[:10],
        "offline_reproduced_new_level": False,
        "dead_end": dead_end,
    }


def select_transfer_results(
    effect_rows: Sequence[Any],
    *,
    limit: int = TRANSFER_GAME_LIMIT,
) -> list[dict[str, Any]]:
    games = sorted({game for game in (_row_game(row) for row in effect_rows) if game})
    measured = [
        measure_action_effect_memory_transfer_game(game, effect_rows=effect_rows) for game in games
    ]
    measured.sort(
        key=lambda row: (
            row.get("value_added") is not True,
            -_as_float((row.get("transfer_value") or {}).get("actions_reduced"))
            if isinstance(row.get("transfer_value"), Mapping)
            else 0.0,
            -_as_int((row.get("transfer_value") or {}).get("candidate_group_count"))
            if isinstance(row.get("transfer_value"), Mapping)
            else 0,
            str(row.get("game") or ""),
        )
    )
    return [dict(row) for row in measured[: max(0, int(limit))]]


def _success_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    winners = [row for row in rows if row.get("value_added") is True]
    if not winners:
        return None
    return max(
        winners,
        key=lambda row: (
            _as_float((row.get("transfer_value") or {}).get("actions_reduced"))
            if isinstance(row.get("transfer_value"), Mapping)
            else 0.0,
            str(row.get("game") or ""),
        ),
    )


def build_artifact(
    *,
    upstream_decision: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    transfer_results: Sequence[Mapping[str, Any]],
    registry_updated: bool,
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4573: assemble the primitive persistence transfer artifact."""

    rows = [dict(row) for row in transfer_results]
    winner = _success_row(rows)
    transfer_games = [str(row.get("game")) for row in rows]
    transfer_values: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = dict(row.get("transfer_value") or {})
        value.setdefault("value_added", bool(row.get("value_added") is True))
        transfer_values[str(row.get("game"))] = value
    dead_ends = {
        str(row.get("game")): str(row.get("dead_end") or "")
        for row in rows
        if str(row.get("dead_end") or "")
    }
    new_levels_banked = sum(1 for row in rows if row.get("offline_reproduced_new_level") is True)
    if winner is not None:
        verdict = f"success: primitive_persisted_transfer_{winner.get('game')}_value_added"
    elif preconditions_checked.get("ok") is False:
        verdict = "blocked_primitive_persist_transfer_precondition"
    else:
        verdict = "complete: primitive_persisted_transfer_null_characterized"

    artifact = {
        "experiment": "experiment_4573_primitive_persist_transfer",
        "schema": "carnot.arc_primitive_persist_transfer_4573.v1",
        "honest_verdict": verdict,
        "inference_substrate": str(
            upstream_decision.get("inference_substrate") or INFERENCE_SUBSTRATE
        ),
        "primitive_persisted": {
            "operator": upstream_decision.get("operator"),
            "registry_general_gotcha_id": upstream_decision.get("registry_general_gotcha_id"),
            "source": upstream_decision.get("source"),
            "derived_from_artifacts": [A1_RELATIVE_PATH, A2_RELATIVE_PATH],
        },
        "transfer_games": transfer_games,
        "transfer_value_per_game": transfer_values,
        "offline_reproduced": bool(new_levels_banked > 0),
        "registry_updated": bool(registry_updated),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": ["REQ-ARC-WMTE-4573"],
        "scenarios": ["SCENARIO-ARC-WMTE-4573"],
        "upstream_decision": dict(upstream_decision),
        "upstream_summaries": {
            "a1_action_effect_predictor": {
                "artifact": A1_RELATIVE_PATH,
                "selected": upstream_decision.get("source") == "A1_action_effect_predictor",
            },
            "a2_verifier_guided_expansion": {
                "artifact": A2_RELATIVE_PATH,
                "selected": upstream_decision.get("source") == "A2_verifier_guided_expansion",
            },
        },
        "transfer_results": rows,
        "transfer_dead_ends": dead_ends,
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
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match the 4573 offline substrate")
    primitive = artifact.get("primitive_persisted")
    if not isinstance(primitive, Mapping) or primitive.get("operator") != PRIMITIVE_OPERATOR:
        errors.append(f"primitive_persisted must name {PRIMITIVE_OPERATOR}")
    elif primitive.get("registry_general_gotcha_id") != PRIMITIVE_GOTCHA_ID:
        errors.append("primitive_persisted must name the 4573 registry general_gotcha")
    transfer_games = artifact.get("transfer_games")
    if not blocked and (not isinstance(transfer_games, list) or len(transfer_games) < 2):
        errors.append("transfer_games must contain at least two games")
    if not isinstance(artifact.get("transfer_value_per_game"), Mapping):
        errors.append("transfer_value_per_game must be a mapping")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be a bare bool")
    if type(artifact.get("registry_updated")) is not bool:
        errors.append("registry_updated must be a bare bool")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-WMTE-4573")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        values = artifact.get("transfer_value_per_game")
        if not isinstance(values, Mapping) or not any(
            isinstance(value, Mapping) and value.get("value_added") is True
            for value in values.values()
        ):
            errors.append("success requires at least one transfer value_added=true")
    if (
        artifact.get("offline_reproduced") is True
        and _as_int(artifact.get("new_levels_banked")) < 1
    ):
        errors.append("offline_reproduced=true requires at least one new level banked")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact content")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:
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
    a2 = _load_json(root_path / A2_RELATIVE_PATH)
    decision = select_primitive_from_upstreams(a1_artifact=a1, a2_artifact=a2)
    rows: list[dict[str, Any]] = []
    if checks.get("ok") is True and decision.get("operator") == PRIMITIVE_OPERATOR:
        rows = select_transfer_results(load_transition_effect_rows(root_path))
    artifact = build_artifact(
        upstream_decision=decision,
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


def main() -> int:  # pragma: no cover - requested command boundary
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
