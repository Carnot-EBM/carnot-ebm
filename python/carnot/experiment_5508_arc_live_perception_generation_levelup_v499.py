"""Experiment 5508: live ARC perception-generation level-up attempt.

Spec refs: REQ-ARC-FCP-5508, SCENARIO-ARC-FCP-5508.

This module is the accounting shell around one bounded live-agent attempt. The
mechanism under test is generation-stage perception grounding: connected
components, color blobs, sprite overlays, salient motion, and action
affordances are extracted from runtime frames and handed to the existing
`E3AgentPolicy`. A level only counts when the live-discovered candidate passes
the standard reproduction gate; source reading, offline BFS, and per-game
adapters stay outside the credited path.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from carnot.agentic.arc_perception_generation import (
    ClassicalPerceptionGenerator,
    REQUIRED_PERCEPTION_FEATURES,
    TRAJECTORY_TAXONOMY_KEYS,
)


JsonDict = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5508
EXPERIMENT = "experiment_5508_arc_live_perception_generation_levelup_v499"
MILESTONE = "2026.07.499"
RESULT_RELATIVE_PATH = "results/experiment_5508_arc_live_perception_generation_levelup_v499.json"
PRECHECK_RELATIVE_PATH = "results/experiment_5507_arc_null_coordinate_perception_precheck_v499.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
KNOWN_ISSUES_RELATIVE_PATH = "ops/known-issues.md"
LEVERS_NOTE_RELATIVE_PATH = "docs/research-notes/arc-agi3-levers-tried-x-verdict-2026-06-25.md"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5508", "SCENARIO-ARC-FCP-5508"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5508
DEFAULT_BUDGET = 48
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5508_arc_live_perception_generation_levelup_v499.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5508_arc_live_perception_generation_levelup_v499.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/agentic/arc_perception_generation.py "
        "python/carnot/experiment_5508_arc_live_perception_generation_levelup_v499.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "selected_game": {
        "principle": "Exp5507-selected game id, or empty string only when the precheck blocks before a target can be attempted."
    },
    "selected_level": {
        "principle": "Exp5507-selected level label such as L3; success must be strictly deeper than the re-read registry depth."
    },
    "registry_before_levels": {
        "principle": "authoritative `ops/arc_solve_registry.yaml` total immediately before the live attempt."
    },
    "registry_after_levels": {
        "principle": "authoritative registry total after the attempt; unchanged on honest null or blocked runs."
    },
    "arc_registry_delta": {
        "principle": "bare int delta between after and before totals; success requires this to equal the newly reproduced levels."
    },
    "offline_reproduced": {
        "principle": "true only when the live-discovered candidate passes the standard reproduction gate for a new level."
    },
    "reproduced_levels": {
        "principle": "new reproduced levels banked beyond the selected game's pre-run depth; success requires >=1."
    },
    "solve_provenance": {"principle": "must equal live_agent_self_discovery."},
    "live_agent_attempts": {
        "principle": "bare int count of runtime actions actually executed by the live agent."
    },
    "runtime_observation_steps": {
        "principle": "bare int count of runtime frame/action observations available to perception generation."
    },
    "perception_features_enabled": {
        "principle": "list containing connected_components, color_blobs, sprite_overlays, salient_motion, and action_affordances when the pass is active."
    },
    "trajectory_taxonomy_counts": {
        "principle": "dict with factual, referential, logical, procedural, and scope_based failure counts."
    },
    "offline_bfs_used": {
        "principle": "must be false; offline ground-truth BFS is not part of the credited path."
    },
    "game_source_read": {
        "principle": "must be false; game source reading is outside live self-discovery credit."
    },
    "hand_built_per_game_adapter_used": {
        "principle": "must be false; no hand per-game adapter is credited."
    },
    "methodology_receipt": {
        "principle": "string receipt naming the bounded live runtime, candidate-generation mechanism, reproduction gate, and prohibited-input flags."
    },
    "inference_substrate": {
        "principle": "must equal offline_arcade_live_agent_runtime_self_discovery_no_llm."
    },
    "honest_verdict": {
        "principle": "one-line verdict starting complete:, honest_null:, or blocked:."
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


class _NoOpProposer:  # pragma: no cover - ARC runtime boundary
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5508_no_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:
        return []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _level_label(level: int) -> str:
    return f"L{max(0, int(level))}"


def _parse_level_label(value: Any) -> int:
    text = str(value or "").strip().upper()
    if text.startswith("L") and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    return 0


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> JsonDict:
    if not path.exists():
        return {"reproducible_total_levels": 0, "games": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {
        "reproducible_total_levels": 0,
        "games": [],
    }


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }


def _registry_depth(registry: Mapping[str, Any], game: str) -> int:
    return _as_int((_registry_rows(registry).get(game) or {}).get("levels_reproduced"))


def _registry_total(registry: Mapping[str, Any]) -> int:
    return _as_int(registry.get("reproducible_total_levels"))


def load_registry(root: Path = REPO) -> JsonDict:
    return _read_yaml(Path(root) / REGISTRY_RELATIVE_PATH)


def select_target_from_precheck(
    precheck: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> JsonDict:
    """REQ-ARC-FCP-5508: re-read registry and block duplicate targets."""

    total = _registry_total(registry)
    game = str(precheck.get("selected_game") or "")
    level_label = str(precheck.get("selected_level") or "")
    target_level = _parse_level_label(level_label)
    if not precheck:
        return _blocked_target("precheck_target_missing", total, game, level_label, target_level)
    if precheck.get("levelup_attempt_ready") is not True:
        return _blocked_target("precheck_target_not_ready", total, game, level_label, target_level)
    if not game:
        return _blocked_target("precheck_target_missing", total, game, level_label, target_level)
    if target_level <= 0:
        return _blocked_target("precheck_selected_level_malformed", total, game, level_label, target_level)
    prior = _registry_depth(registry, game)
    if prior >= target_level:
        return _blocked_target(
            "selected_level_already_reproducible",
            total,
            game,
            _level_label(target_level),
            target_level,
            prior,
        )
    return {
        "blocked": False,
        "selected_game": game,
        "selected_level": _level_label(target_level),
        "target_level": int(target_level),
        "prior_levels_reproduced": int(prior),
        "registry_before_levels": int(total),
        "selection_reason": "exp5507_ready_target_survived_registry_reread",
    }


def _blocked_target(
    blocker: str,
    total: int,
    game: str,
    level_label: str,
    target_level: int,
    prior: int = 0,
) -> JsonDict:
    return {
        "blocked": True,
        "blocker": str(blocker),
        "selected_game": str(game or ""),
        "selected_level": str(level_label or ""),
        "target_level": int(max(0, target_level)),
        "prior_levels_reproduced": int(max(0, prior)),
        "registry_before_levels": int(total),
        "selection_reason": str(blocker),
    }


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
    except Exception:
        return False
    return True


def run_live_perception_generation_attempt(  # pragma: no cover - ARC runtime boundary
    *,
    root: Path,
    target: Mapping[str, Any],
    budget: int = DEFAULT_BUDGET,
) -> JsonDict:
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of
    from carnot.experiment_5494_arc_live_trajectory_option_induction_v498 import (
        _action_label,
        _apply_action_label,
    )

    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    game = str(target["selected_game"])
    prior = _as_int(target.get("prior_levels_reproduced"))
    target_level = _as_int(target.get("target_level"))
    labels: list[str] = []
    frames: list[Any] = []
    action_count = 0
    latest = None
    max_level = prior
    generator = ClassicalPerceptionGenerator(max_candidates=8)
    try:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        policy = E3AgentPolicy(
            game,
            proposer=_NoOpProposer(),
            explore_budget=max(1, int(budget)),
            target_levels=max(1, target_level),
            value_head=None,
            frame_change_scorer=None,
            candidate_router=None,
            action_effect_expansion_prior=False,
            action_prior=generator,
            qd_generator=generator,
            goal_bias=None,
            goal_candidate_guidance=False,
            active_probe_controller=False,
            go_explore_archive=False,
        )
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
                action_count += 1
            observed_level = _as_int(_level_of(latest), default=max_level)
            max_level = max(max_level, observed_level)
            frames.append(latest)
            if latest is None or max_level >= target_level:
                break

        gate: JsonDict = {
            "game": game,
            "claimed_level": max_level,
            "reached_level": prior,
            "reproduced": False,
            "mode": "standard_reproduction_gate_not_run_no_new_target_candidate",
        }
        if max_level > prior and labels:
            gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=max_level))
        reached = _as_int(gate.get("reached_level"), default=max_level)
        reproduced = bool(gate.get("reproduced")) and reached >= target_level
        if not reproduced:
            generator.record_scope_failure("bounded_budget_no_target_level_reproduction")
        diagnostics = generator.diagnostics()
        return {
            "live_agent_attempts": int(action_count),
            "runtime_observation_steps": max(
                int(len(frames)),
                _as_int(diagnostics.get("runtime_observation_steps")),
            ),
            "post_levels_reproduced": int(reached if reproduced else prior),
            "offline_reproduced": bool(reproduced),
            "reproduced_levels": max(0, int(reached) - int(prior)) if reproduced else 0,
            "perception_features_enabled": list(diagnostics["perception_features_enabled"]),
            "trajectory_taxonomy_counts": dict(diagnostics["trajectory_taxonomy_counts"]),
            "trajectory_taxonomy_steps": list(diagnostics["trajectory_taxonomy_steps"]),
            "candidate_action_count": _as_int(diagnostics.get("candidate_generation_count")),
            "failure_mode": "" if reproduced else "bounded_budget_no_target_level_reproduction",
            "reproduction_gate": gate,
            "solution_labels": list(labels) if reproduced else [],
            "offline_bfs_used": False,
            "game_source_read": False,
            "hand_built_per_game_adapter_used": False,
            "methodology_receipt": (
                f"bounded_live_runtime budget={int(budget)} mechanism=classical_perception_generation "
                "gate=standard_reproduction prohibited_inputs=false"
            ),
            "root": str(root),
        }
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _normalize_taxonomy_counts(value: Mapping[str, Any] | None) -> dict[str, int]:
    raw = value or {}
    return {key: _as_int(raw.get(key)) for key in TRAJECTORY_TAXONOMY_KEYS}


def _accepted_reproduced_levels(target: Mapping[str, Any], attempt: Mapping[str, Any]) -> int:
    if attempt.get("offline_reproduced") is not True:
        return 0
    if (
        attempt.get("offline_bfs_used", False)
        or attempt.get("game_source_read", False)
        or attempt.get("hand_built_per_game_adapter_used", False)
    ):
        return 0
    prior = _as_int(target.get("prior_levels_reproduced"))
    target_level = _as_int(target.get("target_level"))
    post = _as_int(attempt.get("post_levels_reproduced"), prior + _as_int(attempt.get("reproduced_levels")))
    if post <= prior or post < target_level:
        return 0
    return max(0, _as_int(attempt.get("reproduced_levels"), post - prior))


def _blocked_attempt(reason: str, features: Sequence[str] = ()) -> JsonDict:
    return {
        "blocked": True,
        "live_agent_attempts": 0,
        "runtime_observation_steps": 0,
        "post_levels_reproduced": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "perception_features_enabled": list(features),
        "trajectory_taxonomy_counts": _normalize_taxonomy_counts({}),
        "trajectory_taxonomy_steps": [{"failure_kind": "scope_based", "reason": str(reason)}],
        "candidate_action_count": 0,
        "failure_mode": str(reason),
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "methodology_receipt": (
            f"blocked_before_live_runtime reason={reason} mechanism=classical_perception_generation "
            "gate=standard_reproduction prohibited_inputs=false"
        ),
    }


def build_artifact(
    *,
    target: Mapping[str, Any],
    attempt: Mapping[str, Any],
    registry_updated: bool,
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> JsonDict:
    """SCENARIO-ARC-FCP-5508: build the required deliverable artifact."""

    blocked = bool(target.get("blocked")) or bool(attempt.get("blocked"))
    accepted_delta = _accepted_reproduced_levels(target, attempt)
    can_bank = bool(accepted_delta >= 1 and registry_updated)
    before_total = _as_int(target.get("registry_before_levels"))
    after_total = before_total + accepted_delta if can_bank else before_total
    selected_game = str(target.get("selected_game") or "") if not blocked else ""
    selected_level = str(target.get("selected_level") or "") if not blocked else ""
    failure_mode = str(
        attempt.get("failure_mode")
        or target.get("blocker")
        or "bounded_budget_no_target_level_reproduction"
    )
    features = [str(item) for item in attempt.get("perception_features_enabled") or []]
    taxonomy = _normalize_taxonomy_counts(
        attempt.get("trajectory_taxonomy_counts")
        if isinstance(attempt.get("trajectory_taxonomy_counts"), Mapping)
        else {}
    )
    status = "complete" if can_bank else "blocked" if blocked else "honest_null"
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5508_arc_live_perception_generation_levelup_v499.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "selected_game": selected_game,
        "selected_level": selected_level,
        "registry_before_levels": int(before_total),
        "registry_after_levels": int(after_total),
        "arc_registry_delta": int(after_total - before_total),
        "offline_reproduced": bool(can_bank),
        "reproduced_levels": int(accepted_delta if can_bank else 0),
        "solve_provenance": SOLVE_PROVENANCE,
        "live_agent_attempts": _as_int(attempt.get("live_agent_attempts")),
        "runtime_observation_steps": _as_int(attempt.get("runtime_observation_steps")),
        "perception_features_enabled": features,
        "trajectory_taxonomy_counts": taxonomy,
        "offline_bfs_used": bool(attempt.get("offline_bfs_used", False)),
        "game_source_read": bool(attempt.get("game_source_read", False)),
        "hand_built_per_game_adapter_used": bool(
            attempt.get("hand_built_per_game_adapter_used", False)
        ),
        "methodology_receipt": str(attempt.get("methodology_receipt") or ""),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {target.get('selected_game')} {target.get('selected_level')} live perception-generation reproduced and banked"
            if can_bank
            else f"blocked: {failure_mode}"
            if status == "blocked"
            else f"honest_null: {target.get('selected_game')} {target.get('selected_level')} {failure_mode}; registry_delta=0"
        ),
        "target_level": _as_int(target.get("target_level")),
        "prior_levels_reproduced": _as_int(target.get("prior_levels_reproduced")),
        "post_levels_reproduced": (
            _as_int(attempt.get("post_levels_reproduced"))
            if can_bank
            else _as_int(target.get("prior_levels_reproduced"))
        ),
        "registry_updated": bool(registry_updated),
        "candidate_action_count": _as_int(attempt.get("candidate_action_count")),
        "trajectory_taxonomy_steps": [
            dict(row) for row in attempt.get("trajectory_taxonomy_steps") or [] if isinstance(row, Mapping)
        ],
        "target_selection": dict(target),
        "attempt": dict(attempt),
        "preconditions_checked": dict(preconditions_checked),
        "tests_run": list(tests_run),
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    for field in ("selected_game", "selected_level"):
        if not isinstance(artifact.get(field), str):
            errors.append(f"{field} must be a string")
    for field in (
        "registry_before_levels",
        "registry_after_levels",
        "arc_registry_delta",
        "reproduced_levels",
        "live_agent_attempts",
        "runtime_observation_steps",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
        elif field in ("reproduced_levels", "live_agent_attempts", "runtime_observation_steps") and artifact[field] < 0:
            errors.append(f"{field} must be non-negative")
    if (
        type(artifact.get("registry_before_levels")) is int
        and type(artifact.get("registry_after_levels")) is int
        and artifact["registry_after_levels"] < artifact["registry_before_levels"]
    ):
        errors.append("registry_after_levels must be >= registry_before_levels")
    elif (
        artifact.get("registry_before_levels") is not None
        and artifact.get("registry_after_levels") is not None
        and _as_int(artifact.get("registry_after_levels")) < _as_int(artifact.get("registry_before_levels"))
    ):
        errors.append("registry_after_levels must be >= registry_before_levels")
    if (
        type(artifact.get("registry_before_levels")) is int
        and type(artifact.get("registry_after_levels")) is int
        and type(artifact.get("arc_registry_delta")) is int
        and artifact["arc_registry_delta"]
        != artifact["registry_after_levels"] - artifact["registry_before_levels"]
    ):
        errors.append("arc_registry_delta must equal registry_after_levels - registry_before_levels")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if artifact.get("offline_reproduced") is True:
        if _as_int(artifact.get("reproduced_levels")) < 1:
            errors.append("offline_reproduced requires reproduced_levels >= 1")
        if _as_int(artifact.get("arc_registry_delta")) != _as_int(artifact.get("reproduced_levels")):
            errors.append("offline_reproduced requires arc_registry_delta == reproduced_levels")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if not isinstance(artifact.get("perception_features_enabled"), list):
        errors.append("perception_features_enabled must be a list")
    elif artifact.get("status") != "blocked":
        missing = [
            feature
            for feature in REQUIRED_PERCEPTION_FEATURES
            if feature not in set(artifact.get("perception_features_enabled") or [])
        ]
        if missing:
            errors.append("perception_features_enabled missing required feature")
    taxonomy = artifact.get("trajectory_taxonomy_counts")
    if not isinstance(taxonomy, dict):
        errors.append("trajectory_taxonomy_counts must be a dict")
    else:
        for key in TRAJECTORY_TAXONOMY_KEYS:
            if type(taxonomy.get(key)) is not int:
                errors.append(f"trajectory_taxonomy_counts.{key} must be bare int")
    for field in ("offline_bfs_used", "game_source_read", "hand_built_per_game_adapter_used"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
        elif artifact.get(field) is not False:
            errors.append(f"{field} must be false")
    if not isinstance(artifact.get("methodology_receipt"), str) or not artifact.get(
        "methodology_receipt"
    ):
        errors.append("methodology_receipt must be a non-empty string")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "honest_null:", "blocked:")):
        errors.append("honest_verdict must start with complete:, honest_null:, or blocked:")
    if artifact.get("registry_updated") is True and artifact.get("offline_reproduced") is not True:
        errors.append("registry_updated requires offline_reproduced true")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def update_registry_if_banked(
    *,
    root: Path,
    artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> bool:
    if artifact.get("offline_reproduced") is not True:
        return False
    updated = dict(registry)
    games = list(updated.get("games") or [])
    game = str(artifact["selected_game"])
    post = _as_int(artifact.get("post_levels_reproduced"))
    delta = _as_int(artifact.get("reproduced_levels"))
    found = False
    for row in games:
        if isinstance(row, dict) and row.get("game") == game:
            row["reproducibility"] = "reproduced"
            row["levels_reproduced"] = post
            row["latest_exp5508_levelup_attempt"] = {
                "artifact": RESULT_RELATIVE_PATH,
                "offline_reproduced": True,
                "reproduced_levels": delta,
                "prior_levels_reproduced": _as_int(artifact.get("prior_levels_reproduced")),
                "post_levels_reproduced": post,
                "registry_before_levels": _as_int(artifact.get("registry_before_levels")),
                "registry_after_levels": _as_int(artifact.get("registry_after_levels")),
                "solve_provenance": SOLVE_PROVENANCE,
            }
            found = True
            break
    if not found:
        games.append(
            {
                "game": game,
                "reproducibility": "reproduced",
                "levels_reproduced": post,
                "latest_exp5508_levelup_attempt": {
                    "artifact": RESULT_RELATIVE_PATH,
                    "offline_reproduced": True,
                    "reproduced_levels": delta,
                    "post_levels_reproduced": post,
                    "solve_provenance": SOLVE_PROVENANCE,
                },
            }
        )
    updated["games"] = games
    updated["reproducible_total_levels"] = _registry_total(updated) + delta
    path = root / REGISTRY_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
    return True


def _write_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    root: Path = REPO,
    budget: int = DEFAULT_BUDGET,
    attempt_runner: Callable[..., Mapping[str, Any]] = run_live_perception_generation_attempt,
    offline_arcade_check: Callable[[], bool] = offline_arcade_available,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    precheck_path = root / PRECHECK_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "CLAUDE.md": (root / "CLAUDE.md").exists(),
        "spec_has_req_5508": "REQ-ARC-FCP-5508" in spec_text,
        "registry_present": registry_path.exists(),
        "precheck_present": precheck_path.exists(),
        "known_issues_present": (root / KNOWN_ISSUES_RELATIVE_PATH).exists(),
        "levers_note_present": (root / LEVERS_NOTE_RELATIVE_PATH).exists(),
        "offline_arcade_available": False,
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
    }
    registry = load_registry(root)
    precheck = _read_json(precheck_path)
    target = select_target_from_precheck(precheck, registry)
    ready_without_arcade = (
        preconditions["AGENTS.md"]
        and preconditions["CODEX.md"]
        and preconditions["CLAUDE.md"]
        and preconditions["spec_has_req_5508"]
        and preconditions["registry_present"]
        and preconditions["precheck_present"]
        and not target.get("blocked")
    )
    if not ready_without_arcade:
        reason = str(target.get("blocker") or "missing_exp5508_precondition")
        attempt: Mapping[str, Any] = _blocked_attempt(reason)
    else:
        preconditions["offline_arcade_available"] = bool(offline_arcade_check())
        if not preconditions["offline_arcade_available"]:
            attempt = _blocked_attempt("missing_harness_access", REQUIRED_PERCEPTION_FEATURES)
        else:
            attempt = attempt_runner(root=root, target=target, budget=budget)

    registry_updated = False
    if _accepted_reproduced_levels(target, attempt) >= 1:
        preliminary = build_artifact(
            target=target,
            attempt=attempt,
            registry_updated=True,
            preconditions_checked=preconditions,
            tests_run=tests_run,
            duration_s=time.monotonic() - started,
        )
        registry_updated = update_registry_if_banked(
            root=root,
            artifact=preliminary,
            registry=registry,
        )
    artifact = build_artifact(
        target=target,
        attempt=attempt,
        registry_updated=registry_updated,
        preconditions_checked=preconditions,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(artifact)
    _write_artifact(root, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = parser.parse_args(argv)
    artifact = run_experiment(budget=args.budget)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
