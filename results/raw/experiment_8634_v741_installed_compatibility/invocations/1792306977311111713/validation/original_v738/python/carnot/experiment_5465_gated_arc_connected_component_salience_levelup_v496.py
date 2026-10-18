"""Experiment 5465: gated ARC connected-component salience level-up attempt.

Spec refs: REQ-ARC-FCP-5465, SCENARIO-ARC-FCP-5465.

The credited path here is deliberately narrow: pick a target from Exp5464's
metric-clean shortlist, use the submitted live agent's connected-component
salience path, and count progress only if the official reproduction gate
confirms a level deeper than the registry precheck. Public-game source reading,
offline ground-truth BFS, and hand-adapter credit stay false in the artifact.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from carnot.agentic.arc_color_blob_salience import (
    ColorBlobSaliencePrior,
    connected_color_blobs,
)


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5465
EXPERIMENT = "experiment_5465_gated_arc_connected_component_salience_levelup_v496"
MILESTONE = "2026.07.496"
RESULT_RELATIVE_PATH = (
    "results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json"
)
PRECHECK_RELATIVE_PATH = (
    "results/experiment_5464_arc_metric_integrity_perception_precheck_v496.json"
)
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5465", "SCENARIO-ARC-FCP-5465"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "arc_live_agent_self_discovery"
DEFAULT_BUDGET = 36
REQUIRED_FEATURES = (
    "connected_component",
    "color_blob",
    "changed_pixel",
    "salience_tier",
    "action_effect",
)
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5465_gated_arc_connected_component_salience_levelup_v496.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5465_gated_arc_connected_component_salience_levelup_v496.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5465_gated_arc_connected_component_salience_levelup_v496.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "solve_provenance": {
        "principle": "live_agent_self_discovery; credited path is the live agent's own attempts plus runtime reverse engineering."
    },
    "registry_precheck_performed": {
        "principle": "bare bool proving the registry was re-read before selecting a target from Exp5464's shortlist."
    },
    "target_game": {
        "principle": "game selected from Exp5464 target_shortlist after the rerun precheck."
    },
    "target_level_before": {
        "principle": "registry-precheck reproduced depth for the selected target before this attempt."
    },
    "target_level_attempted": {
        "principle": "one deeper level attempted by the live/offline agent."
    },
    "live_attempt_count": {
        "principle": "bounded count of live-agent attempts actually executed."
    },
    "perception_features_used": {
        "principle": "auditable list containing connected_component, color_blob, changed_pixel, salience_tier, and action_effect when exercised."
    },
    "source_reading_used": {
        "principle": "must be false; hidden/public game source is not credited in this live self-discovery path."
    },
    "offline_bfs_used": {
        "principle": "must be false; exhaustive offline ground-truth BFS is not the credited solve path."
    },
    "hand_adapter_credited": {
        "principle": "must be false; a hand GameAdapter may not receive live-agent self-discovery credit."
    },
    "offline_reproduced": {
        "principle": "true only when the live-agent candidate reproduces through the official reproduction gate beyond the precheck depth."
    },
    "reproduced_levels": {
        "principle": "absolute reproduced level count after the gate; success requires this to exceed target_level_before."
    },
    "new_level_banked": {
        "principle": "true only when offline_reproduced=true and reproduced_levels > target_level_before."
    },
    "arc_registry_update_required": {
        "principle": "true only when a newly banked level should update ops/arc_solve_registry.yaml."
    },
    "inference_substrate": {"principle": "must equal arc_live_agent_self_discovery."},
    "honest_verdict": {
        "principle": "one-line verdict starting complete:, honest_null:, or blocked:."
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


class _NoOpProposer:  # pragma: no cover - ARC runtime boundary
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5465_no_live_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:
        return []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(root: Path = REPO) -> dict[str, Any]:
    path = root / REGISTRY_RELATIVE_PATH
    if not path.exists():  # pragma: no cover - defensive missing-repo path
        return {"reproducible_total_levels": 0, "games": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def registry_precheck(registry: Mapping[str, Any]) -> dict[str, Any]:
    """REQ-ARC-FCP-5465: re-read registry depth before target selection."""

    rows = _registry_rows(registry)
    return {
        "performed": True,
        "reproducible_total_levels": _as_int(registry.get("reproducible_total_levels")),
        "games_checked": len(rows),
        "depths": {
            game: _as_int(row.get("levels_reproduced"))
            for game, row in sorted(rows.items())
        },
    }


def select_target_from_precheck(
    precheck: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-5465: select a non-duplicate target from Exp5464's shortlist."""

    summary = registry_precheck(registry)
    if precheck.get("arc_metric_integrity_ready") is not True:
        return {
            "blocked": True,
            "blocker": "exp5464_precheck_not_ready",
            "registry_precheck_performed": bool(summary["performed"]),
            "target_from_exp5464_shortlist": False,
            "target_game": "none",
            "target_level_before": 0,
            "target_level_attempted": 1,
            "registry_precheck": summary,
        }
    rows = _registry_rows(registry)
    for index, target in enumerate(precheck.get("target_shortlist") or []):
        if not isinstance(target, Mapping):
            continue
        game = str(target.get("game") or "")
        row = rows.get(game)
        if row is None:
            continue
        current = _as_int(row.get("levels_reproduced"))
        listed_level = _as_int(target.get("target_level"))
        if current < 1 or listed_level <= current:
            continue
        return {
            "blocked": False,
            "registry_precheck_performed": bool(summary["performed"]),
            "target_from_exp5464_shortlist": True,
            "target_shortlist_index": int(index),
            "target_game": game,
            "target_level_before": current,
            "target_level_attempted": current + 1,
            "precheck_target": dict(target),
            "selection_reason": "exp5464_shortlist_first_nonduplicate_after_registry_rerun",
            "registry_precheck": summary,
        }
    return {
        "blocked": True,
        "blocker": "no_exp5464_shortlist_target_survived_registry_rerun",
        "registry_precheck_performed": bool(summary["performed"]),
        "target_from_exp5464_shortlist": False,
        "target_game": "none",
        "target_level_before": 0,
        "target_level_attempted": 1,
        "registry_precheck": summary,
    }


def _diagnostic_frames() -> tuple[SimpleNamespace, SimpleNamespace]:
    before = np.zeros((20, 20), dtype=np.int16)
    before[0, :] = 16
    before[2:10, 2:18] = 8
    before[14:16, 14:16] = 9
    after = before.copy()
    after[14:16, 14:16] = 10
    return (
        SimpleNamespace(frame=before, available_actions=[6]),
        SimpleNamespace(frame=after, available_actions=[6]),
    )


def _changed_pixel_rows(before: Any, after: Any) -> list[dict[str, int]]:
    before_arr = np.asarray(before.frame if hasattr(before, "frame") else before)
    after_arr = np.asarray(after.frame if hasattr(after, "frame") else after)
    ys, xs = np.nonzero(before_arr != after_arr)
    return [
        {
            "y": int(y),
            "x": int(x),
            "before": int(before_arr[y, x]),
            "after": int(after_arr[y, x]),
        }
        for y, x in zip(ys.tolist(), xs.tolist(), strict=False)
    ]


def build_live_feature_receipts() -> dict[str, Any]:
    """SCENARIO-ARC-FCP-5465: exercise the live salience feature path."""

    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    before, after = _diagnostic_frames()
    prior = ColorBlobSaliencePrior()
    components = connected_color_blobs(before, max_component_fraction=1.0)
    component_rows = [
        {
            "color": int(blob.color),
            "pixel_count": int(blob.pixel_count),
            "bbox": [int(value) for value in blob.bbox],
            "centroid_y": float(blob.centroid[0]),
            "centroid_x": float(blob.centroid[1]),
        }
        for blob in components
    ]
    candidates = [
        {"action": 6, "data": {"x": int(x), "y": int(y)}, "source": "color_blob_prior"}
        for x, y in prior.click_points(before, max_points=4)
    ]
    changed_pixels = _changed_pixel_rows(before, after)
    policy = E3AgentPolicy(
        "bp35",
        proposer=None,
        value_head=None,
        frame_change_scorer=None,
        candidate_router=None,
        action_effect_expansion_prior=False,
        goal_bias=None,
        goal_candidate_guidance=False,
        active_probe_controller=False,
    )
    generated = policy.explorer._candidates(before)  # noqa: SLF001 - live hook fixture
    salience_diagnostics = policy.explorer.action_salience_diagnostics()
    receipts = {
        "spec_refs": list(SPEC_REFS),
        "live_agent_policy_reachable": bool(
            salience_diagnostics.get("connected_component_salience_enabled")
        ),
        "connected_component_rows": component_rows,
        "color_blob_rows": prior.tier_rows(before),
        "changed_pixel_rows": changed_pixels,
        "salience_tier_rows": salience_diagnostics.get("tier_rows", []),
        "action_tier_rows": prior.action_tier_rows(before, candidates),
        "action_effect_observations": [
            {
                "action": 6,
                "data": {"x": 14, "y": 14},
                "changed_pixels": len(changed_pixels),
                "before_color": 9,
                "after_color": 10,
                "source": "synthetic_live_path_frame_delta",
            }
        ],
        "first_live_candidate": dict(generated[0]) if generated else None,
        "live_salience_diagnostics": dict(salience_diagnostics),
    }
    features: list[str] = []
    if receipts["connected_component_rows"]:
        features.append("connected_component")
    if receipts["color_blob_rows"]:
        features.append("color_blob")
    if receipts["changed_pixel_rows"]:
        features.append("changed_pixel")
    if receipts["salience_tier_rows"]:
        features.append("salience_tier")
    if receipts["action_effect_observations"]:
        features.append("action_effect")
    receipts["perception_features_used"] = features
    return receipts


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
    except Exception:
        return False
    return True


def _action_label(action: int | str, data: Any) -> str:
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def run_live_salience_attempt(  # pragma: no cover - ARC runtime boundary
    *,
    root: Path,
    selection: Mapping[str, Any],
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:
    """Run one bounded live-agent attempt and reproduction-gate only reached candidates."""

    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    game = str(selection["target_game"])
    target_level_before = _as_int(selection.get("target_level_before"))
    attempted_level = _as_int(selection.get("target_level_attempted"))
    prior = ColorBlobSaliencePrior()
    try:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        policy = E3AgentPolicy(
            game,
            proposer=_NoOpProposer(),
            explore_budget=max(1, int(budget)),
            target_levels=max(1, attempted_level),
            value_head=None,
            frame_change_scorer=None,
            candidate_router=None,
            action_effect_expansion_prior=False,
            action_prior=prior,
            goal_bias=None,
            goal_candidate_guidance=False,
            active_probe_controller=False,
            go_explore_archive=False,
        )
        frames: list[Any] = []
        latest = None
        labels: list[str] = []
        actions_taken = 0
        max_level = target_level_before
        for _index in range(max(1, int(budget))):
            if policy.is_done(frames, latest):
                break
            kind, data = policy.next_move(frames, latest)
            if kind == "RESET":
                latest = env.reset()
                if labels:
                    labels.append("RESET")
            elif kind is None:
                break
            else:
                latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
                labels.append(_action_label(int(kind), data))
                actions_taken += 1
            observed_level = _as_int(_level_of(latest), default=max_level)
            max_level = max(max_level, observed_level)
            frames.append(latest)
            if latest is None or max_level >= attempted_level:
                break

        gate: dict[str, Any] = {
            "game": game,
            "claimed_level": max_level,
            "reached_level": target_level_before,
            "reproduced": False,
            "mode": "official_reproduction_gate_not_run_no_new_level_candidate",
        }
        if max_level > target_level_before and labels:
            gate = dict(
                kit.reproduce(game, labels, _apply_action_label, claimed_level=max_level)
            )
        reproduced_level = _as_int(gate.get("reached_level"), default=max_level)
        offline_reproduced = bool(gate.get("reproduced")) and reproduced_level > target_level_before
        return {
            "target_game": game,
            "target_level_before": target_level_before,
            "target_level_attempted": attempted_level,
            "live_attempt_count": 1,
            "actions_taken": int(actions_taken),
            "max_level_reached": int(max_level),
            "offline_reproduced": bool(offline_reproduced),
            "reproduced_levels": int(reproduced_level if offline_reproduced else target_level_before),
            "new_level_banked": bool(offline_reproduced),
            "failure_mode": None if offline_reproduced else "bounded_budget_no_levelup",
            "solution_labels": list(labels) if offline_reproduced else [],
            "reproduction_gate": gate,
            "runtime_self_discovery": True,
            "source_reading_used": False,
            "offline_bfs_used": False,
            "hand_adapter_credited": False,
            "root": str(root),
            "salience_diagnostics": policy.explorer.action_salience_diagnostics(),
            "blob_prior_diagnostics": prior.as_dict(),
        }
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _attempt_live_count(attempt: Mapping[str, Any], blocked: bool) -> int:
    if "live_attempt_count" in attempt:
        return _as_int(attempt.get("live_attempt_count"))
    return 0 if blocked else 1


def build_artifact(
    *,
    selection: Mapping[str, Any],
    feature_receipts: Mapping[str, Any],
    attempt: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    target_game = str(selection.get("target_game") or "")
    target_level_before = _as_int(selection.get("target_level_before"))
    target_level_attempted = _as_int(selection.get("target_level_attempted"))
    blocked = bool(selection.get("blocked")) or bool(attempt.get("blocked"))
    attempt_reproduced_levels = _as_int(attempt.get("reproduced_levels"), target_level_before)
    no_prohibited_inputs = (
        attempt.get("source_reading_used", False) is False
        and attempt.get("offline_bfs_used", False) is False
        and attempt.get("hand_adapter_credited", False) is False
    )
    new_level_banked = bool(
        attempt.get("offline_reproduced") is True
        and attempt_reproduced_levels > target_level_before
        and no_prohibited_inputs
        and not blocked
    )
    status = "complete" if new_level_banked else "blocked" if blocked else "honest_null"
    failure_mode = None if new_level_banked else str(attempt.get("failure_mode") or "")
    if status == "honest_null" and not failure_mode:
        failure_mode = "bounded_budget_no_levelup"
    if status == "blocked" and not failure_mode:
        failure_mode = str(selection.get("blocker") or "missing_harness_access")
    reproduced_levels = (
        attempt_reproduced_levels if new_level_banked else max(target_level_before, attempt_reproduced_levels)
    )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5465_gated_arc_connected_component_salience_levelup_v496.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "solve_provenance": SOLVE_PROVENANCE,
        "registry_precheck_performed": selection.get("registry_precheck_performed") is True,
        "target_game": target_game,
        "target_level_before": target_level_before,
        "target_level_attempted": target_level_attempted,
        "live_attempt_count": _attempt_live_count(attempt, blocked),
        "perception_features_used": list(feature_receipts.get("perception_features_used") or []),
        "source_reading_used": bool(attempt.get("source_reading_used", False)),
        "offline_bfs_used": bool(attempt.get("offline_bfs_used", False)),
        "hand_adapter_credited": bool(attempt.get("hand_adapter_credited", False)),
        "offline_reproduced": bool(new_level_banked),
        "reproduced_levels": int(reproduced_levels),
        "new_level_banked": bool(new_level_banked),
        "arc_registry_update_required": bool(new_level_banked),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {target_game} L{target_level_attempted} live connected-component salience reproduced"
            if new_level_banked
            else f"blocked: {failure_mode}"
            if status == "blocked"
            else f"honest_null: {target_game} L{target_level_attempted} {failure_mode}"
        ),
        "failure_mode": failure_mode,
        "target_selection": dict(selection),
        "feature_receipts": dict(feature_receipts),
        "attempt": dict(attempt),
        "preconditions_checked": dict(preconditions_checked),
        "tests_run": list(tests_run),
        "duration_s": float(duration_s),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if type(artifact.get("registry_precheck_performed")) is not bool:
        errors.append("registry_precheck_performed must be bare bool")
    elif artifact.get("registry_precheck_performed") is not True:
        errors.append("registry_precheck_performed must be true")
    if not isinstance(artifact.get("target_game"), str) or not artifact.get("target_game"):
        errors.append("target_game must be a non-empty string")
    for field in (
        "target_level_before",
        "target_level_attempted",
        "live_attempt_count",
        "reproduced_levels",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    if (
        type(artifact.get("target_level_attempted")) is int
        and artifact.get("target_level_attempted")
        != _as_int(artifact.get("target_level_before")) + 1
    ):
        errors.append("target_level_attempted must be target_level_before + 1")
    features = artifact.get("perception_features_used")
    if not isinstance(features, list):
        errors.append("perception_features_used must be a list")
        features = []
    for feature in REQUIRED_FEATURES:
        if feature not in features:
            errors.append(f"perception_features_used missing {feature}")
    for field in ("source_reading_used", "offline_bfs_used", "hand_adapter_credited"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
        elif artifact.get(field) is not False:
            errors.append(f"{field} must be false")
    for field in ("offline_reproduced", "new_level_banked", "arc_registry_update_required"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    if (
        artifact.get("offline_reproduced") is True
        and type(artifact.get("reproduced_levels")) is int
        and artifact["reproduced_levels"] <= _as_int(artifact.get("target_level_before"))
    ):
        errors.append("offline_reproduced requires reproduced_levels > target_level_before")
    if artifact.get("new_level_banked") is True:
        if artifact.get("offline_reproduced") is not True:
            errors.append("new_level_banked requires offline_reproduced true")
        if artifact.get("arc_registry_update_required") is not True:
            errors.append("new_level_banked requires arc_registry_update_required true")
    if artifact.get("arc_registry_update_required") is True and artifact.get(
        "new_level_banked"
    ) is not True:
        errors.append("arc_registry_update_required requires new_level_banked true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "honest_null:", "blocked:")):
        errors.append("honest_verdict must start with complete:, honest_null:, or blocked:")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def _write_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    root: Path = REPO,
    budget: int = DEFAULT_BUDGET,
    attempt_runner: Callable[..., Mapping[str, Any]] = run_live_salience_attempt,
    offline_arcade_check: Callable[[], bool] = offline_arcade_available,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    precheck_path = root / PRECHECK_RELATIVE_PATH
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "spec_has_req_5465": (
            "REQ-ARC-FCP-5465" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "exp5464_precheck_present": precheck_path.exists(),
        "exp5464_ready": False,
        "offline_arcade_available": False,
        "no_source_reading": True,
        "no_offline_bfs": True,
        "no_hand_adapter_credit": True,
    }
    registry = load_registry(root)
    precheck = load_json(precheck_path)
    preconditions["exp5464_ready"] = precheck.get("arc_metric_integrity_ready") is True
    selection = select_target_from_precheck(precheck, registry)
    feature_receipts = build_live_feature_receipts()
    ready_without_arcade = (
        preconditions["AGENTS.md"]
        and preconditions["CODEX.md"]
        and preconditions["spec_has_req_5465"]
        and preconditions["registry_present"]
        and preconditions["exp5464_precheck_present"]
        and preconditions["exp5464_ready"]
        and not selection.get("blocked")
    )
    if not ready_without_arcade:
        attempt: Mapping[str, Any] = {
            "blocked": True,
            "failure_mode": str(selection.get("blocker") or "missing_precondition"),
            "live_attempt_count": 0,
            "source_reading_used": False,
            "offline_bfs_used": False,
            "hand_adapter_credited": False,
            "reproduced_levels": _as_int(selection.get("target_level_before")),
        }
    else:
        preconditions["offline_arcade_available"] = bool(offline_arcade_check())
        if not preconditions["offline_arcade_available"]:
            attempt = {
                "blocked": True,
                "failure_mode": "missing_harness_access",
                "live_attempt_count": 0,
                "source_reading_used": False,
                "offline_bfs_used": False,
                "hand_adapter_credited": False,
                "reproduced_levels": _as_int(selection.get("target_level_before")),
            }
        else:
            attempt = attempt_runner(root=root, selection=selection, budget=budget)
    artifact = build_artifact(
        selection=selection,
        feature_receipts=feature_receipts,
        attempt=attempt,
        preconditions_checked=preconditions,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(artifact)
    _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
