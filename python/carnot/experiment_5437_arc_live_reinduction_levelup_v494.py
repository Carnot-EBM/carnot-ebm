"""Experiment 5437: ARC registry-guided live reinduction level-up attempt.

Spec refs: REQ-ARC-FCP-5437, SCENARIO-ARC-FCP-5437.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import yaml

from carnot.agentic.arc_live_trajectory_frontier import LiveCoExLandmarkFrontierGenerator


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5437
EXPERIMENT = "experiment_5437_arc_live_reinduction_levelup_v494"
MILESTONE = "2026.07.494"
RESULT_RELATIVE_PATH = "results/experiment_5437_arc_live_reinduction_levelup_v494.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5437", "SCENARIO-ARC-FCP-5437"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "live_arc_agent_runtime"
DEFAULT_BUDGET = 52
DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5437_arc_live_reinduction_levelup_v494.py -q --no-cov",
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5437_arc_live_reinduction_levelup_v494.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5437_arc_live_reinduction_levelup_v494.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "registry_precheck": {"principle": "duplicate-solve avoidance"},
    "target_game": {"principle": "target provenance"},
    "target_level": {"principle": "target provenance"},
    "duplicate_solve_avoided": {"principle": "no already-banked headline"},
    "solve_provenance": {"principle": "credited path"},
    "offline_reproduced": {"principle": "reproducible solve gate"},
    "reproduced_levels": {"principle": "level-up gate; must be >=1 for a banked solve"},
    "arc_new_level_banked": {"principle": "north-star metric"},
    "attempt_count": {"principle": "effort accounting"},
    "frontier_expansion_count": {"principle": "mechanism evidence"},
    "runtime_predicate_count": {"principle": "reinduction evidence"},
    "action_sequence_receipts": {"principle": "reproducibility"},
    "no_offline_bfs": {"principle": "live-path discipline"},
    "no_per_game_adapter": {"principle": "live-path discipline"},
    "arc_levelup_lint_passed": {"principle": "roadmap guarantee evidence"},
    "inference_substrate": {"principle": "actual live agent"},
    "honest_verdict": {
        "principle": "terminal status; start with complete: or honest_null: or blocked:"
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


class _NoOpProposer:  # pragma: no cover - ARC runtime boundary
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5437_no_live_llm"

    def refactor(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5437_no_live_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:
        return []


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }


def _registry_total(registry: Mapping[str, Any]) -> int:
    return int(registry.get("reproducible_total_levels") or 0)


def _level_label(level: int) -> str:
    return f"L{max(1, int(level))}"


def _levels_reproduced(row: Mapping[str, Any] | None) -> int:
    return int((row or {}).get("levels_reproduced") or 0)


def _selected(
    *,
    game: str,
    before: int,
    target: int,
    reason: str,
    eligible_reason: str,
) -> dict[str, Any]:
    return {
        "status": "selected",
        "registry_precheck": True,
        "target_game": str(game),
        "target_level_before": int(before),
        "target_level_number": int(target),
        "target_level": _level_label(target),
        "duplicate_solve_avoided": True,
        "selection_reason": str(reason),
        "target_eligible_reason": str(eligible_reason),
    }


def select_target_after_precheck(
    registry: Mapping[str, Any],
    *,
    preferred: Sequence[tuple[str, int, int]] = (("cn04", 3, 4), ("vc33", 2, 3)),
    alternates: Sequence[str] | None = None,
    avoid_recent_no_bank: Sequence[str] = ("lf52", "re86"),
) -> dict[str, Any]:
    """REQ-ARC-FCP-5437: choose a next unbanked frontier before attempting."""

    rows = _registry_rows(registry)
    notes: list[str] = []
    for game, required_before, target in preferred:
        before = _levels_reproduced(rows.get(game))
        label = f"{game} {_level_label(target)}"
        if before == int(required_before):
            reason = "; ".join(notes + [f"preferred {label} eligible after registry precheck"])
            return _selected(
                game=game,
                before=before,
                target=target,
                reason=reason,
                eligible_reason=f"{label} is next unbanked frontier",
            )
        if before >= int(target):
            notes.append(f"preferred {label} already banked")
        else:
            notes.append(
                f"preferred {label} not next frontier; registry has {_level_label(before)}"
            )

    if alternates is None:
        avoided = set(avoid_recent_no_bank) | {game for game, _, _ in preferred}
        alternate_games = sorted(
            (game for game in rows if game not in avoided),
            key=lambda game: (-_levels_reproduced(rows.get(game)), game),
        )
    else:
        alternate_games = list(alternates)

    for game in alternate_games:
        row = rows.get(str(game))
        if row is None or str(row.get("reproducibility")) != "reproduced":
            continue
        before = _levels_reproduced(row)
        if before < 1:
            continue
        return _selected(
            game=str(game),
            before=before,
            target=before + 1,
            reason="; ".join(notes + ["rotated to nearest eligible unbanked frontier"]),
            eligible_reason=f"{game} {_level_label(before + 1)} is next after registry depth {_level_label(before)}",
        )

    before = _levels_reproduced(rows.get("cn04"))
    return {
        "status": "blocked_duplicate_solve",
        "registry_precheck": True,
        "target_game": "cn04",
        "target_level_before": before,
        "target_level_number": 4,
        "target_level": "L4",
        "duplicate_solve_avoided": True,
        "selection_reason": "; ".join(notes + ["no eligible unbanked frontier available"]),
        "target_eligible_reason": "no eligible unbanked frontier available",
    }


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
    except Exception:
        return False
    return True


def run_arc_levelup_lint(root: Path) -> dict[str, Any]:  # pragma: no cover - subprocess boundary
    roadmap = root / "research-roadmap.yaml"
    script = root / "scripts" / "arc_levelup_guarantee_lint.py"
    command = [sys.executable, str(script), str(roadmap), "--min", "1"]
    if not roadmap.exists() or not script.exists():
        return {
            "command": " ".join(command),
            "passed": False,
            "returncode": 2,
            "stdout": "",
            "stderr": "roadmap_or_lint_missing",
        }
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    return {
        "command": " ".join(command),
        "passed": completed.returncode == 0,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def _action_label(action: int | str, data: Any) -> str:
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def _fallback_sequence_receipt(labels: Sequence[str]) -> list[dict[str, Any]]:
    sequence = []
    for label in labels:
        if label == "RESET":
            continue
        try:
            sequence.append(json.loads(label))
        except Exception:
            continue
    if not sequence:
        return []
    return [
        {
            "sequence": sequence,
            "measurement_receipts": [],
            "replayable": True,
            "source": "executed_live_action_labels",
        }
    ]


def runtime_predicates_from_diagnostics(
    diagnostics: Mapping[str, Any],
    induction_attempts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """SCENARIO-ARC-FCP-5437: summarize predicates induced from live observations."""

    predicates: list[dict[str, Any]] = []
    for row in diagnostics.get("verifier_observations", []) or []:
        if not isinstance(row, Mapping):
            continue
        support = int(row.get("support_count") or 0)
        if support <= 0:
            continue
        predicates.append(
            {
                "predicate": "observed_action_effect",
                "action": int(row.get("action") or 0),
                "data": row.get("data"),
                "support_count": support,
                "effect_count": int(row.get("effect_count") or 0),
                "accepted": bool(row.get("accepted")),
                "salience_route": str(row.get("salience_route") or "unknown"),
                "source": "runtime_observation_cluster",
            }
        )
    for attempt in induction_attempts:
        if not isinstance(attempt, Mapping):
            continue
        reason = str(attempt.get("reason") or "")
        transition_count = int(attempt.get("transition_count") or 0)
        if not reason and transition_count <= 0:
            continue
        predicates.append(
            {
                "predicate": "level_up_reinduction_route",
                "reason": reason,
                "transition_count": transition_count,
                "planned": bool(attempt.get("planned")),
                "accepted": bool(attempt.get("planned")),
                "skipped": str(attempt.get("skipped") or ""),
                "source": "generic_verifier_routing",
            }
        )
    return predicates


def generic_verifier_routes_from_predicates(
    predicates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """REQ-ARC-FCP-5437: compact route evidence for the artifact."""

    routes: list[dict[str, Any]] = []
    for row in predicates:
        route = str(row.get("source") or row.get("predicate") or "unknown")
        routes.append({"route": route, "accepted": bool(row.get("accepted", False))})
    return routes


def run_live_reinduction_attempt(  # pragma: no cover - ARC runtime boundary
    *,
    root: Path,
    selection: Mapping[str, Any],
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    game = str(selection["target_game"])
    target_before = int(selection["target_level_before"])
    target_number = int(selection["target_level_number"])
    generator = LiveCoExLandmarkFrontierGenerator(min_support=1, max_uncertainty=0.51)
    reset_count = 0
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(
        game,
        proposer=_NoOpProposer(),
        explore_budget=max(1, int(budget)),
        target_levels=max(1, target_number),
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
    frames: list[Any] = []
    latest = None
    labels: list[str] = []
    max_level = target_before
    newly_reached: list[str] = []
    for _index in range(max(1, int(budget))):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
            reset_count += 1
            generator.record_reset(level=max_level)
            if labels:
                labels.append("RESET")
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
            labels.append(_action_label(int(kind), data))
        observed_level = int(_level_of(latest))
        if observed_level > max_level:
            newly_reached.extend(
                _level_label(level) for level in range(max_level + 1, observed_level + 1)
            )
        max_level = max(max_level, observed_level)
        frames.append(latest)
        if latest is None or max_level >= target_number:
            break

    gate: dict[str, Any] = {
        "game": game,
        "claimed_level": 0,
        "reached_level": 0,
        "reproduced": False,
        "mode": "offline_reproduction_gate_no_new_level_claim",
    }
    if max_level > target_before and labels:
        gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=max_level))
    reproduced = (
        bool(gate.get("reproduced")) and int(gate.get("reached_level") or 0) > target_before
    )
    diagnostics = generator.diagnostics()
    predicates = runtime_predicates_from_diagnostics(diagnostics, policy.induction_attempts)
    receipts = list(diagnostics.get("action_sequence_receipts") or [])
    if not receipts:
        receipts = _fallback_sequence_receipt(labels)
    new_reproduced = (
        max(0, int(gate.get("reached_level") or max_level) - target_before) if reproduced else 0
    )
    return {
        "target_game": game,
        "target_level_before": target_before,
        "target_level": _level_label(target_number),
        "attempt_count": len([label for label in labels if label != "RESET"]),
        "reset_count": int(reset_count),
        "max_level_reached": int(max_level),
        "offline_reproduced": bool(reproduced),
        "new_reproduced_levels": int(new_reproduced),
        "failure_mode": None if reproduced else "bounded_budget_no_levelup",
        "frontier_expansion_count": int(diagnostics.get("frontier_expansion_count") or 0),
        "runtime_predicate_count": int(len(predicates)),
        "runtime_predicates": predicates,
        "frontier_transitions": list(diagnostics.get("frontier_transitions") or []),
        "action_sequence_receipts": receipts,
        "runtime_observations": list(diagnostics.get("runtime_observations") or []),
        "measurement_access_receipts": list(diagnostics.get("measurement_access_receipts") or []),
        "action_history_clusters": list(diagnostics.get("action_history_clusters") or []),
        "generic_verifier_routes": generic_verifier_routes_from_predicates(predicates),
        "level_induction_events": list(policy.level_induction_events),
        "induction_attempts": list(policy.induction_attempts),
        "newly_reached_levels": newly_reached,
        "solution_labels": list(labels) if reproduced else [],
        "reproduction_gate": gate,
        "runtime_self_discovery": True,
        "no_offline_bfs": True,
        "no_per_game_adapter": True,
        "root": str(root),
    }


def _new_reproduced_levels(attempt: Mapping[str, Any]) -> int:
    if attempt.get("offline_reproduced") is not True:
        return 0
    return max(0, int(attempt.get("new_reproduced_levels") or 0))


def build_artifact(
    *,
    selection: Mapping[str, Any],
    registry_total_before: int,
    attempt: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    lint_result: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    blocked = bool(attempt.get("blocked")) or selection.get("status") == "blocked_duplicate_solve"
    target_game = str(selection.get("target_game") or "cn04")
    target_level = str(selection.get("target_level") or "L1")
    no_offline_bfs = attempt.get("no_offline_bfs", True) is True
    no_per_game_adapter = attempt.get("no_per_game_adapter", True) is True
    receipts = list(attempt.get("action_sequence_receipts") or [])
    new_reproduced = _new_reproduced_levels(attempt)
    runtime_predicate_count = int(attempt.get("runtime_predicate_count") or 0)
    frontier_expansion_count = int(attempt.get("frontier_expansion_count") or 0)
    runtime_evidence = bool(
        runtime_predicate_count > 0
        or frontier_expansion_count > 0
        or attempt.get("frontier_transitions")
    )
    can_bank = bool(
        new_reproduced >= 1
        and receipts
        and runtime_evidence
        and attempt.get("runtime_self_discovery") is True
        and bool(selection.get("duplicate_solve_avoided"))
        and no_offline_bfs
        and no_per_game_adapter
    )
    status = "complete" if can_bank else "blocked" if blocked else "honest_null"
    failure_mode = attempt.get("failure_mode")
    if status == "blocked" and not failure_mode:
        failure_mode = (
            "duplicate_solve_precheck"
            if selection.get("status") == "blocked_duplicate_solve"
            else "missing_harness_access"
        )
    if status == "honest_null" and not failure_mode:
        failure_mode = "bounded_budget_no_levelup"
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5437_arc_live_reinduction_levelup_v494.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "registry_precheck": bool(selection.get("registry_precheck")),
        "target_game": target_game,
        "target_level": target_level,
        "target_eligible_reason": str(selection.get("target_eligible_reason") or ""),
        "duplicate_solve_avoided": bool(selection.get("duplicate_solve_avoided")),
        "solve_provenance": SOLVE_PROVENANCE,
        "offline_reproduced": bool(can_bank),
        "reproduced_levels": int(new_reproduced if can_bank else 0),
        "arc_new_level_banked": bool(can_bank),
        "attempt_count": int(attempt.get("attempt_count") or 0),
        "reset_count": int(attempt.get("reset_count") or 0),
        "frontier_expansion_count": frontier_expansion_count,
        "runtime_predicate_count": runtime_predicate_count,
        "action_sequence_receipts": receipts,
        "no_offline_bfs": bool(no_offline_bfs),
        "no_per_game_adapter": bool(no_per_game_adapter),
        "arc_levelup_lint_passed": bool(lint_result.get("passed")),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {target_game} {target_level} live reinduction reproduced"
            if can_bank
            else f"blocked: {failure_mode}"
            if status == "blocked"
            else f"honest_null: {target_game} {target_level} {failure_mode}"
        ),
        "registry_total_before": int(registry_total_before),
        "registry_total_after": int(registry_total_before) + int(new_reproduced if can_bank else 0),
        "target_selection": dict(selection),
        "attempts": [dict(attempt)] if attempt else [],
        "runtime_predicates": list(attempt.get("runtime_predicates") or []),
        "frontier_transitions": list(attempt.get("frontier_transitions") or []),
        "runtime_observations": list(attempt.get("runtime_observations") or []),
        "measurement_access_receipts": list(attempt.get("measurement_access_receipts") or []),
        "action_history_clusters": list(attempt.get("action_history_clusters") or []),
        "generic_verifier_routes": list(attempt.get("generic_verifier_routes") or []),
        "level_induction_events": list(attempt.get("level_induction_events") or []),
        "induction_attempts": list(attempt.get("induction_attempts") or []),
        "newly_reached_levels": list(attempt.get("newly_reached_levels") or []),
        "solution_labels": list(attempt.get("solution_labels") or []),
        "reproduction_gate": dict(attempt.get("reproduction_gate") or {}),
        "failure_mode": None if can_bank else str(failure_mode or ""),
        "preconditions_checked": dict(preconditions_checked),
        "arc_levelup_lint": dict(lint_result),
        "tests_run": list(tests_run),
        "duration_s": float(duration_s),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    if artifact.get("status") not in {"complete", "honest_null", "blocked"}:
        errors.append("status must be complete, honest_null, or blocked")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    for field in (
        "registry_precheck",
        "duplicate_solve_avoided",
        "offline_reproduced",
        "arc_new_level_banked",
        "no_offline_bfs",
        "no_per_game_adapter",
        "arc_levelup_lint_passed",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    for field in (
        "registry_precheck",
        "duplicate_solve_avoided",
        "no_offline_bfs",
        "no_per_game_adapter",
    ):
        if artifact.get(field) is not True:
            errors.append(f"{field} must be true")
    for field in (
        "attempt_count",
        "frontier_expansion_count",
        "runtime_predicate_count",
        "reproduced_levels",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    for field in ("target_game", "target_level"):
        if not isinstance(artifact.get(field), str) or not artifact.get(field):
            errors.append(f"{field} must be non-empty string")
    if not isinstance(artifact.get("action_sequence_receipts"), list):
        errors.append("action_sequence_receipts must be list")
    elif artifact.get("status") != "blocked" and not artifact.get("action_sequence_receipts"):
        errors.append("action_sequence_receipts must be a non-empty list")
    if (
        artifact.get("offline_reproduced") is True
        and artifact.get("solve_provenance") != SOLVE_PROVENANCE
    ):
        errors.append("offline_reproduced true requires live_agent_self_discovery")
    if artifact.get("status") == "complete":
        if artifact.get("offline_reproduced") is not True:
            errors.append("complete artifact requires offline_reproduced true")
        if artifact.get("arc_new_level_banked") is not True:
            errors.append("complete artifact requires arc_new_level_banked true")
        if type(artifact.get("reproduced_levels")) is int and artifact["reproduced_levels"] < 1:
            errors.append("complete artifact requires reproduced_levels >= 1")
        if (
            type(artifact.get("runtime_predicate_count")) is int
            and type(artifact.get("frontier_expansion_count")) is int
            and artifact["runtime_predicate_count"] < 1
            and artifact["frontier_expansion_count"] < 1
            and not artifact.get("frontier_transitions")
        ):
            errors.append("complete artifact requires runtime predicate or frontier evidence")
    else:
        if artifact.get("offline_reproduced") is True:
            errors.append("non-complete artifact cannot set offline_reproduced true")
        if artifact.get("arc_new_level_banked") is True:
            errors.append("arc_new_level_banked requires complete status")
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
    attempt_runner: Callable[..., Mapping[str, Any]] = run_live_reinduction_attempt,
    offline_arcade_check: Callable[[], bool] = offline_arcade_available,
    lint_runner: Callable[[Path], Mapping[str, Any]] = run_arc_levelup_lint,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "spec_has_req_5437": (
            "REQ-ARC-FCP-5437" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "offline_arcade_available": False,
        "no_offline_bfs": True,
        "no_per_game_adapter": True,
    }
    registry = (
        yaml.safe_load(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    )
    registry = registry or {}
    selection = select_target_after_precheck(registry)
    lint_result = dict(lint_runner(root))
    ready_without_arcade = (
        preconditions["AGENTS.md"]
        and preconditions["CODEX.md"]
        and preconditions["spec_has_req_5437"]
        and preconditions["registry_present"]
        and selection.get("status") != "blocked_duplicate_solve"
    )
    if not ready_without_arcade:
        attempt: Mapping[str, Any] = {
            "blocked": True,
            "failure_mode": "duplicate_solve_precheck"
            if selection.get("status") == "blocked_duplicate_solve"
            else "missing_harness_access",
            "attempt_count": 0,
            "reset_count": 0,
            "action_sequence_receipts": [],
            "no_offline_bfs": True,
            "no_per_game_adapter": True,
        }
    else:
        preconditions["offline_arcade_available"] = bool(offline_arcade_check())
        if not preconditions["offline_arcade_available"]:
            attempt = {
                "blocked": True,
                "failure_mode": "missing_harness_access",
                "attempt_count": 0,
                "reset_count": 0,
                "action_sequence_receipts": [],
                "no_offline_bfs": True,
                "no_per_game_adapter": True,
            }
        else:
            attempt = attempt_runner(root=root, selection=selection, budget=budget)
    artifact = build_artifact(
        selection=selection,
        registry_total_before=_registry_total(registry),
        attempt=attempt,
        preconditions_checked=preconditions,
        lint_result=lint_result,
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
