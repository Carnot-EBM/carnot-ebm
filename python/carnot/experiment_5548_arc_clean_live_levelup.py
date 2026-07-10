"""Exp5548: clean no-LLM ARC live level-up attempt.

Spec refs: REQ-ARC-FCP-5548, SCENARIO-ARC-FCP-5548.

This module is the credit-bearing follow-up to Exp5547. Exp5534 exercised the
right live-agent path but carried stale model metadata, which made the artifact
look like a model-inference run. Exp5548 keeps the same bounded live attempt
shape while making the substrate explicit: no LLM strategy proposer, no model
load, and no model specs unless a model is actually invoked.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from carnot.agentic.arc_bounded_strategy_router import BoundedStrategyCandidateRouter
from carnot.experiment_5534_arc_strategy_routed_levelup import (
    offline_arcade_available,
    run_live_strategy_routed_attempt,
    trajectory_metrics,
)
from carnot.experiment_5547_arc_no_llm_substrate_precheck import DEFAULT_TARGET_CANDIDATES


JsonDict = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5548
EXPERIMENT = "experiment_5548_arc_clean_live_levelup"
MILESTONE = "2026.07.502"
RESULT_RELATIVE_PATH = "results/experiment_5548_arc_clean_live_levelup.json"
TRAJECTORY_RELATIVE_PATH = "results/experiment_5548_arc_clean_live_levelup_trajectory.json"
UPSTREAM_RELATIVE_PATH = "results/experiment_5547_arc_no_llm_substrate_precheck.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5548", "SCENARIO-ARC-FCP-5548"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
DEFAULT_BUDGET = 48
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5548_arc_clean_live_levelup.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5548_arc_clean_live_levelup.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5548_arc_clean_live_levelup.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "selected_game": "registry-safe game id used for the clean live attempt after Exp5547 and duplicate checks.",
    "selected_level": "adjacent unreproduced frontier level label attempted by the live agent.",
    "solve_provenance": "must equal live_agent_self_discovery so only the live runtime's own attempt can receive credit.",
    "llm_strategy_proposer_used": "bare bool false proving no LLM strategy proposer or model path was invoked.",
    "no_model_specs_required": "bare bool true because the declared no-LLM substrate has no model invocation to name.",
    "random_seed": "Exp5547-recorded deterministic seed reused for target rotation, trajectory gating, and checksum replay.",
    "reproducibility_checksum": "content-addressed hash over target, seed, trajectory metrics, and banking gate to catch silent drift.",
    "attempts": "bare int count of runtime actions executed during the live attempt.",
    "trajectory_path": "path to the detailed trajectory log containing actions, route evidence, suppression events, terminal state, and replay gate.",
    "action_entropy": "Shannon entropy over executed live action/coordinate choices as a bare float.",
    "repeated_coordinate_rate": "fraction of executed coordinate actions that repeated an earlier executed coordinate.",
    "offline_reproduced": "true only when the live-discovered trajectory passes the standard offline replay gate.",
    "reproduced_levels": "integer new levels banked from the live-discovered trajectory; success requires at least one.",
    "registry_delta": "bare int registry total delta; nonzero only when the accepted reproduction gate passes.",
    "arc_live_levelup_ready": "bare bool proving Exp5547, registry reread, no-LLM metadata, and live harness preconditions allowed runtime.",
    "tests_added_or_reused": "list of focused tests covering clean schema, target rotation, trajectory metrics, checksum, and banking gate.",
    "field_principles": "mapping of one-line principle annotations for each headline and gate field.",
    "inference_substrate": "must equal offline_arcade_live_agent_runtime_self_discovery_no_llm.",
    "honest_verdict": "one-line verdict starting complete:, honest_null:, or blocked:.",
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive parsing.
        return int(default)


def _parse_level_label(value: Any) -> int:
    text = str(value or "").strip().upper()
    if text.startswith("L") and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    return 0


def _level_label(level: int) -> str:
    return f"L{max(0, int(level))}"


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


def _row_data(row: Mapping[str, Any]) -> Mapping[str, Any]:
    data = row.get("data")
    return data if isinstance(data, Mapping) else {}


def _row_coordinate(row: Mapping[str, Any]) -> tuple[int, int] | None:
    data = _row_data(row)
    if "x" in data and "y" in data:
        return _as_int(data.get("x")), _as_int(data.get("y"))
    if "x" in row and "y" in row:
        return _as_int(row.get("x")), _as_int(row.get("y"))
    return None


def _row_signature(row: Mapping[str, Any]) -> str:
    coord = _row_coordinate(row)
    if coord is not None:
        return f"A{_as_int(row.get('action'))}@{coord[0]},{coord[1]}"
    return f"A{_as_int(row.get('action'))}"


def _target_marker(game: str, level: int) -> str:
    return f"{game}:{_level_label(level)}"


def _default_strategy_portfolio() -> list[JsonDict]:
    return BoundedStrategyCandidateRouter().portfolio_descriptors()


def _strategy_portfolio_from_precheck(precheck: Mapping[str, Any]) -> list[JsonDict]:
    probe = precheck.get("strategy_probe")
    diagnostics = probe.get("diagnostics") if isinstance(probe, Mapping) else None
    rows = diagnostics.get("strategy_portfolio") if isinstance(diagnostics, Mapping) else None
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        portfolio = [dict(row) for row in rows if isinstance(row, Mapping)]
        if portfolio:
            return portfolio
    return _default_strategy_portfolio()


def _blocked_target(
    blocker: str,
    registry: Mapping[str, Any],
    *,
    game: str = "",
    selected_level: str = "",
    target_level: int = 0,
    prior: int = 0,
    precheck: Mapping[str, Any] | None = None,
) -> JsonDict:
    return {
        "blocked": True,
        "blocker": str(blocker),
        "selected_game": str(game or ""),
        "selected_level": str(selected_level or ""),
        "target_level": int(max(0, target_level)),
        "prior_levels_reproduced": int(max(0, prior)),
        "registry_before_levels": _registry_total(registry),
        "random_seed": _as_int((precheck or {}).get("random_seed"), EXPERIMENT_ID),
        "suppress_repeated_coordinates": bool(
            (precheck or {}).get("repeated_coordinate_suppression_enabled", False)
        ),
        "strategy_portfolio": _strategy_portfolio_from_precheck(precheck or {}),
        "target_rotation": {
            "rotated": False,
            "reason": str(blocker),
            "original_game": str(game or ""),
            "original_level": str(selected_level or ""),
        },
    }


def _fresh_adjacent_target(
    registry: Mapping[str, Any],
    *,
    exclude: tuple[str, int],
    target_candidates: Sequence[tuple[str, int]],
) -> JsonDict | None:
    rows = _registry_rows(registry)
    excluded_game, excluded_level = exclude
    for game, target_level in target_candidates:
        target_level = int(target_level)
        if game == excluded_game and target_level == excluded_level:
            continue
        depth = _registry_depth(registry, game)
        if game in rows and depth + 1 == target_level:
            return {
                "selected_game": game,
                "selected_level": _level_label(target_level),
                "target_level": int(target_level),
                "prior_levels_reproduced": int(depth),
            }
    return None


def _precheck_clean(precheck: Mapping[str, Any]) -> bool:
    return bool(
        precheck
        and precheck.get("arc_clean_precheck_ready") is True
        and precheck.get("llm_strategy_proposer_used") is False
        and precheck.get("no_model_specs_required") is True
        and precheck.get("inference_substrate") == INFERENCE_SUBSTRATE
        and precheck.get("solve_provenance") == SOLVE_PROVENANCE
        and precheck.get("repeated_coordinate_suppression_enabled") is True
    )


def select_target_from_precheck(
    precheck: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    target_candidates: Sequence[tuple[str, int]] = DEFAULT_TARGET_CANDIDATES,
) -> JsonDict:
    """REQ-ARC-FCP-5548: reuse Exp5547's target unless the registry banks it."""

    if not precheck:
        return _blocked_target("exp5547_precheck_missing", registry, precheck=precheck)
    game = str(precheck.get("selected_game") or "")
    selected_level = str(precheck.get("selected_level") or "")
    target_level = _parse_level_label(selected_level)
    prior = _registry_depth(registry, game)
    if not _precheck_clean(precheck):
        return _blocked_target(
            "exp5547_precheck_not_ready",
            registry,
            game=game,
            selected_level=selected_level,
            target_level=target_level,
            prior=prior,
            precheck=precheck,
        )
    if not game:
        return _blocked_target(
            "exp5547_target_missing",
            registry,
            selected_level=selected_level,
            target_level=target_level,
            precheck=precheck,
        )
    if target_level <= 0:
        return _blocked_target(
            "exp5547_selected_level_malformed",
            registry,
            game=game,
            selected_level=selected_level,
            target_level=target_level,
            prior=prior,
            precheck=precheck,
        )
    if game not in _registry_rows(registry):
        return _blocked_target(
            "no_fresh_adjacent_frontier_target",
            registry,
            game=game,
            selected_level=_level_label(target_level),
            target_level=target_level,
            prior=0,
            precheck=precheck,
        )

    rotation = {
        "rotated": False,
        "reason": "exp5547_target_survived_registry_reread",
        "original_game": game,
        "original_level": _level_label(target_level),
    }
    if prior >= target_level:
        fresh = _fresh_adjacent_target(
            registry,
            exclude=(game, target_level),
            target_candidates=target_candidates,
        )
        if fresh is None:
            return _blocked_target(
                "no_fresh_adjacent_frontier_target",
                registry,
                game=game,
                selected_level=_level_label(target_level),
                target_level=target_level,
                prior=prior,
                precheck=precheck,
            )
        rotation = {
            "rotated": True,
            "reason": "exp5547_target_already_banked_rotated_to_adjacent_frontier",
            "original_game": game,
            "original_level": _level_label(target_level),
        }
        game = str(fresh["selected_game"])
        selected_level = str(fresh["selected_level"])
        target_level = _as_int(fresh["target_level"])
        prior = _as_int(fresh["prior_levels_reproduced"])
    elif prior + 1 != target_level:
        return _blocked_target(
            "exp5547_target_not_adjacent_frontier",
            registry,
            game=game,
            selected_level=_level_label(target_level),
            target_level=target_level,
            prior=prior,
            precheck=precheck,
        )

    return {
        "blocked": False,
        "selected_game": game,
        "selected_level": _level_label(target_level),
        "target_level": int(target_level),
        "prior_levels_reproduced": int(prior),
        "registry_before_levels": _registry_total(registry),
        "random_seed": _as_int(precheck.get("random_seed"), EXPERIMENT_ID),
        "suppress_repeated_coordinates": True,
        "strategy_portfolio": _strategy_portfolio_from_precheck(precheck),
        "target_rotation": rotation,
        "target_audit": {
            _target_marker(game, target_level): {
                "game": game,
                "target_level": int(target_level),
                "registry_depth": int(prior),
                "already_reproduced": False,
                "decision": "selected",
            }
        },
    }


def _blocked_attempt(reason: str, *, target: Mapping[str, Any] | None = None) -> JsonDict:
    target_level = _as_int((target or {}).get("target_level"))
    prior = _as_int((target or {}).get("prior_levels_reproduced"))
    return {
        "blocked": True,
        "attempts": 0,
        "post_levels_reproduced": prior,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "action_rows": [],
        "proposed_action_rows": [],
        "observations": [{"event": "blocked", "reason": str(reason)}],
        "verifier_routes": [],
        "suppression_events": [],
        "level_counter_changes": [],
        "reproduction_gate": {"reproduced": False, "reason": str(reason)},
        "verifier_feedback": {"reproduction_gate": {"reproduced": False, "reason": str(reason)}},
        "solution_labels": [],
        "failure_mode": str(reason),
        "terminal_state": {
            "status": "blocked",
            "reason": str(reason),
            "max_level": int(prior),
            "target_level": int(target_level),
        },
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "llm_strategy_proposer_used": False,
    }


def run_clean_live_attempt(  # pragma: no cover - live arcade boundary.
    *,
    root: Path,
    target: Mapping[str, Any],
    precheck: Mapping[str, Any],
    strategy_portfolio: Sequence[Mapping[str, Any]],
    budget: int = DEFAULT_BUDGET,
    random_seed: int,
    suppress_repeated_coordinates: bool,
) -> JsonDict:
    """SCENARIO-ARC-FCP-5548: execute the existing live path with no LLM proposer."""

    del precheck, suppress_repeated_coordinates
    random.seed(int(random_seed))
    return dict(
        run_live_strategy_routed_attempt(
            root=root,
            target=target,
            exp5533={},
            strategy_portfolio=strategy_portfolio,
            budget=budget,
        )
    )


def _accepted_reproduced_levels(target: Mapping[str, Any], attempt: Mapping[str, Any]) -> int:
    if attempt.get("offline_reproduced") is not True:
        return 0
    if (
        attempt.get("offline_bfs_used", False)
        or attempt.get("game_source_read", False)
        or attempt.get("hand_built_per_game_adapter_used", False)
        or attempt.get("llm_strategy_proposer_used", False)
    ):
        return 0
    prior = _as_int(target.get("prior_levels_reproduced"))
    target_level = _as_int(target.get("target_level"))
    post = _as_int(attempt.get("post_levels_reproduced"), prior)
    if post <= prior or post < target_level:
        return 0
    return max(0, _as_int(attempt.get("reproduced_levels"), post - prior))


def compute_reproducibility_checksum(
    *,
    selected_game: str,
    selected_level: str,
    random_seed: int,
    attempts: int,
    action_entropy: float,
    repeated_coordinate_rate: float,
    offline_reproduced: bool,
    reproduced_levels: int,
    registry_delta: int,
    terminal_state: Mapping[str, Any],
    reproduction_gate: Mapping[str, Any],
) -> str:
    """Hash the fields that define the replayable clean live-attempt claim."""

    payload = {
        "action_entropy": float(action_entropy),
        "attempts": int(attempts),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "offline_reproduced": bool(offline_reproduced),
        "random_seed": int(random_seed),
        "registry_delta": int(registry_delta),
        "repeated_coordinate_rate": float(repeated_coordinate_rate),
        "reproduced_levels": int(reproduced_levels),
        "reproduction_gate": dict(reproduction_gate),
        "selected_game": str(selected_game),
        "selected_level": str(selected_level),
        "solve_provenance": SOLVE_PROVENANCE,
        "terminal_state": dict(terminal_state),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    target: Mapping[str, Any],
    attempt: Mapping[str, Any],
    registry_updated: bool,
    trajectory_path: str,
    precheck: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
    attempt_budget: int,
) -> JsonDict:
    """SCENARIO-ARC-FCP-5548: build the clean no-LLM live-attempt artifact."""

    action_rows = [row for row in attempt.get("action_rows") or [] if isinstance(row, Mapping)]
    proposed_rows = [
        row for row in attempt.get("proposed_action_rows") or [] if isinstance(row, Mapping)
    ]
    metrics = trajectory_metrics(action_rows, proposed_rows)
    blocked = bool(target.get("blocked")) or bool(attempt.get("blocked"))
    accepted_delta = _accepted_reproduced_levels(target, attempt)
    can_bank = bool(accepted_delta >= 1 and registry_updated and not blocked)
    registry_delta = int(accepted_delta if can_bank else 0)
    before_total = _as_int(target.get("registry_before_levels"))
    selected_game = str(target.get("selected_game") or "")
    selected_level = str(target.get("selected_level") or "")
    attempts = _as_int(attempt.get("attempts"), len(action_rows))
    terminal_state = dict(
        attempt.get("terminal_state")
        or {
            "status": "reproduced" if can_bank else "budget_exhausted",
            "max_level": _as_int(attempt.get("post_levels_reproduced")),
            "target_level": _as_int(target.get("target_level")),
        }
    )
    reproduction_gate = dict(attempt.get("reproduction_gate") or {})
    status = "complete" if can_bank else "blocked" if blocked else "honest_null"
    reproduced_levels = int(accepted_delta if can_bank else 0)
    offline_reproduced = bool(can_bank)
    random_seed = _as_int(target.get("random_seed"), _as_int(precheck.get("random_seed"), EXPERIMENT_ID))
    checksum = compute_reproducibility_checksum(
        selected_game=selected_game,
        selected_level=selected_level,
        random_seed=random_seed,
        attempts=attempts,
        action_entropy=float(metrics["action_entropy"]),
        repeated_coordinate_rate=float(metrics["repeated_coordinate_rate"]),
        offline_reproduced=offline_reproduced,
        reproduced_levels=reproduced_levels,
        registry_delta=registry_delta,
        terminal_state=terminal_state,
        reproduction_gate=reproduction_gate,
    )
    failure_mode = str(
        attempt.get("failure_mode")
        or target.get("blocker")
        or "bounded_budget_no_target_level_reproduction"
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5548_arc_clean_live_levelup.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "status": status,
        "selected_game": selected_game,
        "selected_level": selected_level,
        "solve_provenance": SOLVE_PROVENANCE,
        "llm_strategy_proposer_used": False,
        "no_model_specs_required": True,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "attempts": int(attempts),
        "trajectory_path": str(trajectory_path),
        "action_entropy": float(metrics["action_entropy"]),
        "repeated_coordinate_rate": float(metrics["repeated_coordinate_rate"]),
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "registry_delta": int(registry_delta),
        "arc_live_levelup_ready": bool(not blocked),
        "tests_added_or_reused": list(tests_run or DEFAULT_TESTS_RUN),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {selected_game} {selected_level} clean no-LLM live trajectory reproduced and banked"
            if can_bank
            else f"blocked: {failure_mode}"
            if status == "blocked"
            else (
                f"honest_null: {selected_game} {selected_level} {failure_mode}; "
                f"entropy={metrics['action_entropy']:.3f}; "
                f"repeat_rate={metrics['repeated_coordinate_rate']:.3f}; registry_delta=0"
            )
        ),
        "attempt_budget": int(attempt_budget),
        "terminal_state": terminal_state,
        "reproduction_gate": reproduction_gate,
        "exact_replay_result": {
            "offline_reproduced": offline_reproduced,
            "reproduced_levels": reproduced_levels,
            "registry_delta": int(registry_delta),
            "gate": reproduction_gate,
        },
        "selected_target_level": _as_int(target.get("target_level")),
        "prior_levels_reproduced": _as_int(target.get("prior_levels_reproduced")),
        "post_levels_reproduced": (
            _as_int(attempt.get("post_levels_reproduced"))
            if can_bank
            else _as_int(target.get("prior_levels_reproduced"))
        ),
        "registry_before_levels": int(before_total),
        "registry_after_levels": int(before_total + registry_delta),
        "registry_updated": bool(registry_updated),
        "target_selection": dict(target),
        "target_rotation": dict(target.get("target_rotation") or {}),
        "strategy_portfolio_used": list(target.get("strategy_portfolio") or []),
        "repeated_coordinate_suppression_enabled": bool(
            target.get("suppress_repeated_coordinates", False)
        ),
        "repeated_coordinate_suppression_events": sum(
            max(0, _as_int(event.get("suppressed_coordinate_count")))
            for event in attempt.get("suppression_events") or []
            if isinstance(event, Mapping)
        ),
        "offline_bfs_used": bool(attempt.get("offline_bfs_used", False)),
        "game_source_read": bool(attempt.get("game_source_read", False)),
        "hand_built_per_game_adapter_used": bool(
            attempt.get("hand_built_per_game_adapter_used", False)
        ),
        "upstream_precheck_path": UPSTREAM_RELATIVE_PATH,
        "upstream_arc_clean_precheck_ready": precheck.get("arc_clean_precheck_ready") is True,
        "input_artifacts": [UPSTREAM_RELATIVE_PATH, REGISTRY_RELATIVE_PATH],
        "preconditions_checked": dict(preconditions_checked),
        "duration_s": float(duration_s),
    }


def build_trajectory_log(
    *,
    target: Mapping[str, Any],
    attempt: Mapping[str, Any],
    artifact: Mapping[str, Any],
    precheck: Mapping[str, Any],
) -> JsonDict:
    """SCENARIO-ARC-FCP-5548: persist trajectory evidence for replay audit."""

    return {
        "schema": "carnot.experiment_5548_arc_clean_live_levelup.trajectory.v1",
        "experiment": EXPERIMENT,
        "selected_game": artifact.get("selected_game") or target.get("selected_game") or "",
        "selected_level": artifact.get("selected_level") or target.get("selected_level") or "",
        "random_seed": artifact.get("random_seed"),
        "attempt_budget": artifact.get("attempt_budget"),
        "target_selection": dict(target),
        "target_rotation": dict(target.get("target_rotation") or {}),
        "observations": list(attempt.get("observations") or []),
        "proposed_actions": list(attempt.get("proposed_action_rows") or []),
        "executed_actions": list(attempt.get("action_rows") or []),
        "verifier_routes": list(attempt.get("verifier_routes") or []),
        "suppression_events": list(attempt.get("suppression_events") or []),
        "level_counter_changes": list(attempt.get("level_counter_changes") or []),
        "verifier_feedback": dict(attempt.get("verifier_feedback") or {}),
        "solution_labels": list(attempt.get("solution_labels") or []),
        "terminal_state": dict(artifact.get("terminal_state") or {}),
        "exact_replay_result": dict(artifact.get("exact_replay_result") or {}),
        "metrics": {
            "action_entropy": artifact.get("action_entropy"),
            "repeated_coordinate_rate": artifact.get("repeated_coordinate_rate"),
            "repeated_coordinate_suppression_events": artifact.get(
                "repeated_coordinate_suppression_events"
            ),
        },
        "prohibited_inputs": {
            "offline_bfs_used": bool(attempt.get("offline_bfs_used", False)),
            "game_source_read": bool(attempt.get("game_source_read", False)),
            "hand_built_per_game_adapter_used": bool(
                attempt.get("hand_built_per_game_adapter_used", False)
            ),
        },
        "upstream_precheck": {
            "selected_game": precheck.get("selected_game"),
            "selected_level": precheck.get("selected_level"),
            "arc_clean_precheck_ready": precheck.get("arc_clean_precheck_ready"),
            "random_seed": precheck.get("random_seed"),
        },
    }


def write_trajectory_log(path: Path, log: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checksum_looks_valid(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    if not isinstance(artifact.get("selected_game"), str):
        errors.append("selected_game must be a string")
    if not isinstance(artifact.get("selected_level"), str):
        errors.append("selected_level must be a string")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if artifact.get("llm_strategy_proposer_used") is not False:
        errors.append("llm_strategy_proposer_used must be false")
    if artifact.get("no_model_specs_required") is not True:
        errors.append("no_model_specs_required must be true")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be an int")
    if not _checksum_looks_valid(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be a sha256 string")
    if type(artifact.get("attempts")) is not int:
        errors.append("attempts must be bare int")
    elif artifact["attempts"] < 0:
        errors.append("attempts must be non-negative")
    if not isinstance(artifact.get("trajectory_path"), str) or not artifact.get(
        "trajectory_path"
    ):
        errors.append("trajectory_path must be a non-empty string")
    if type(artifact.get("action_entropy")) is not float:
        errors.append("action_entropy must be bare float")
    if type(artifact.get("repeated_coordinate_rate")) is not float:
        errors.append("repeated_coordinate_rate must be bare float")
    elif not (0.0 <= artifact["repeated_coordinate_rate"] <= 1.0):
        errors.append("repeated_coordinate_rate must be in [0, 1]")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    elif artifact["reproduced_levels"] < 0:
        errors.append("reproduced_levels must be non-negative")
    if type(artifact.get("registry_delta")) is not int:
        errors.append("registry_delta must be bare int")
    elif artifact["registry_delta"] < 0:
        errors.append("registry_delta must be non-negative")
    if artifact.get("offline_reproduced") is True:
        if _as_int(artifact.get("reproduced_levels")) < 1:
            errors.append("offline_reproduced requires reproduced_levels >= 1")
        if _as_int(artifact.get("registry_delta")) != _as_int(artifact.get("reproduced_levels")):
            errors.append("offline_reproduced true requires registry_delta == reproduced_levels")
    if type(artifact.get("arc_live_levelup_ready")) is not bool:
        errors.append("arc_live_levelup_ready must be bare bool")
    tests = artifact.get("tests_added_or_reused")
    if not isinstance(tests, list) or not tests:
        errors.append("tests_added_or_reused must be a non-empty list")
    if not isinstance(artifact.get("field_principles"), Mapping):
        errors.append("field_principles must be a mapping")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(
            "inference_substrate must be offline_arcade_live_agent_runtime_self_discovery_no_llm"
        )
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "honest_null:", "blocked:")):
        errors.append("honest_verdict must start with complete:, honest_null:, or blocked:")
    if "model_specs" in artifact:
        errors.append("model_specs must be omitted for no-LLM substrate")
    if "target_model" in artifact:
        errors.append("target_model must be omitted for no-LLM substrate")
    for field in ("offline_bfs_used", "game_source_read", "hand_built_per_game_adapter_used"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
        elif artifact.get(field) is not False:
            errors.append(f"{field} must be false")
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
    if artifact.get("offline_reproduced") is not True or _as_int(artifact.get("registry_delta")) <= 0:
        return False
    updated = dict(registry)
    games = list(updated.get("games") or [])
    game = str(artifact["selected_game"])
    post = _as_int(artifact.get("post_levels_reproduced"))
    delta = _as_int(artifact.get("reproduced_levels"))
    evidence = {
        "artifact": RESULT_RELATIVE_PATH,
        "trajectory_path": artifact.get("trajectory_path"),
        "offline_reproduced": True,
        "reproduced_levels": delta,
        "prior_levels_reproduced": _as_int(artifact.get("prior_levels_reproduced")),
        "post_levels_reproduced": post,
        "registry_delta": delta,
        "solve_provenance": SOLVE_PROVENANCE,
        "random_seed": _as_int(artifact.get("random_seed")),
        "reproducibility_checksum": artifact.get("reproducibility_checksum"),
    }
    found = False
    for row in games:
        if isinstance(row, dict) and row.get("game") == game:
            row["reproducibility"] = "reproduced"
            row["levels_reproduced"] = post
            row["latest_exp5548_clean_live_levelup"] = evidence
            found = True
            break
    if not found:
        games.append(
            {
                "game": game,
                "reproducibility": "reproduced",
                "levels_reproduced": post,
                "latest_exp5548_clean_live_levelup": evidence,
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
    attempt_runner: Callable[..., Mapping[str, Any]] = run_clean_live_attempt,
    offline_arcade_check: Callable[[], bool] = offline_arcade_available,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    upstream_path = root / UPSTREAM_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    registry = _read_yaml(registry_path)
    precheck = _read_json(upstream_path)
    target = select_target_from_precheck(precheck, registry)
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "OPENCODE.md": (root / "OPENCODE.md").exists(),
        "CLAUDE.md": (root / "CLAUDE.md").exists(),
        "spec_has_req_5548": "REQ-ARC-FCP-5548" in spec_text,
        "registry_present": registry_path.exists(),
        "exp5547_precheck_present": upstream_path.exists(),
        "exp5547_ready": precheck.get("arc_clean_precheck_ready") is True,
        "arc_clean_precheck_ready": precheck.get("arc_clean_precheck_ready") is True,
        "offline_arcade_available": False,
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "llm_strategy_proposer_used": False,
        "model_specs_present": False,
    }
    ready_without_arcade = (
        preconditions["AGENTS.md"]
        and (preconditions["CODEX.md"] or preconditions["OPENCODE.md"])
        and preconditions["CLAUDE.md"]
        and preconditions["spec_has_req_5548"]
        and preconditions["registry_present"]
        and preconditions["exp5547_precheck_present"]
        and _precheck_clean(precheck)
        and not target.get("blocked")
    )
    if not ready_without_arcade:
        attempt: Mapping[str, Any] = _blocked_attempt(
            str(target.get("blocker") or "missing_exp5548_precondition"),
            target=target,
        )
    else:
        preconditions["offline_arcade_available"] = bool(offline_arcade_check())
        if not preconditions["offline_arcade_available"]:
            attempt = _blocked_attempt("missing_harness_access", target=target)
        else:
            attempt = attempt_runner(
                root=root,
                target=target,
                precheck=precheck,
                strategy_portfolio=target.get("strategy_portfolio") or _default_strategy_portfolio(),
                budget=budget,
                random_seed=_as_int(target.get("random_seed"), EXPERIMENT_ID),
                suppress_repeated_coordinates=bool(target.get("suppress_repeated_coordinates")),
            )

    trajectory_path = TRAJECTORY_RELATIVE_PATH
    registry_updated = False
    if _accepted_reproduced_levels(target, attempt) >= 1:
        preliminary = build_artifact(
            target=target,
            attempt=attempt,
            registry_updated=True,
            trajectory_path=trajectory_path,
            precheck=precheck,
            preconditions_checked=preconditions,
            tests_run=tests_run,
            duration_s=time.monotonic() - started,
            attempt_budget=budget,
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
        trajectory_path=trajectory_path,
        precheck=precheck,
        preconditions_checked=preconditions,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
        attempt_budget=budget,
    )
    validate_artifact(artifact)
    log = build_trajectory_log(
        target=target,
        attempt=attempt,
        artifact=artifact,
        precheck=precheck,
    )
    write_trajectory_log(root / trajectory_path, log)
    _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
