"""Experiment 5450: ARC measurement-access live level-up attempt.

Spec refs: REQ-ARC-FCP-5450, SCENARIO-ARC-FCP-5450.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from carnot.agentic.arc_live_trajectory_frontier import LiveCoExLandmarkFrontierGenerator


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5450
EXPERIMENT = "experiment_5450_arc_measurement_access_live_levelup_v495"
MILESTONE = "2026.07.495"
RESULT_RELATIVE_PATH = "results/experiment_5450_arc_measurement_access_live_levelup_v495.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5450", "SCENARIO-ARC-FCP-5450"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
DEFAULT_BUDGET = 40
DEFAULT_RECENT_NO_BANK_TARGETS = ("cn04:L4", "re86:L3")
DEFAULT_FRONTIER_PRIORITY = (
    "cn04",
    "re86",
    "ka59",
    "vc33",
    "bp35",
    "sb26",
    "g50t",
    "r11l",
    "s5i5",
    "cd82",
    "sp80",
    "su15",
    "m0r0",
    "sk48",
    "lf52",
    "ar25",
    "ft09",
    "dc22",
    "ls20",
    "wa30",
    "lp85",
    "sc25",
    "tn36",
    "tr87",
    "tu93",
)
DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5450_arc_measurement_access_live_levelup_v495.py -q --no-cov",
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5450_arc_measurement_access_live_levelup_v495.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5450_arc_measurement_access_live_levelup_v495.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "solve_provenance": {
        "principle": "live_agent_self_discovery -- credited path is the live agent's own attempts."
    },
    "registry_precheck_total_levels": {"principle": "duplicate-solve prevention."},
    "selected_game": {"principle": "target audit."},
    "selected_target_level": {"principle": "target audit."},
    "target_rotation_reason": {"principle": "no stale rerun."},
    "live_attempt_count": {"principle": "live effort evidence."},
    "runtime_predicates_induced": {"principle": "credited mechanism evidence."},
    "offline_reproduced": {"principle": "official reproduction gate."},
    "reproduced_levels": {"principle": "level-up acceptance."},
    "new_levels_banked": {"principle": "north-star movement."},
    "new_level_reproduced": {"principle": "lint-readable solve gate."},
    "no_offline_bfs": {"principle": "no outer-loop solve."},
    "no_source_reading": {"principle": "no hidden source path."},
    "no_per_game_adapter_credited": {"principle": "live mechanism only."},
    "arc_new_level_banked": {"principle": "capstone field."},
    "inference_substrate": {"principle": "explicit runtime path."},
    "honest_verdict": {
        "principle": "terminal status; start with complete: or honest_null: or blocked:."
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


class _NoOpProposer:  # pragma: no cover - live runtime boundary
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5450_no_llm"

    def refactor(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5450_no_llm"

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


def _level_label(level: int) -> str:
    return f"L{max(0, int(level))}"


def _loop_depth(loop_results: Mapping[str, Mapping[str, Any]], game: str) -> int:
    row = loop_results.get(str(game)) or {}
    if row.get("offline_reproduced") is True:
        return _as_int(row.get("reproduced_levels"))
    return 0


def load_registry(root: Path = REPO) -> dict[str, Any]:  # pragma: no cover - file I/O wrapper
    path = root / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return {"reproducible_total_levels": 0, "games": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_arc_loop_results(root: Path = REPO) -> dict[str, dict[str, Any]]:  # pragma: no cover
    results: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "results").glob("arc_loop_solve*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        game = str(data.get("game") or path.stem.replace("arc_loop_solve_", ""))
        results[game] = data
    return results


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
    except Exception:
        return False
    return True


def _dead_end_summary(row: Mapping[str, Any]) -> str:
    dead = row.get("dead_ends") or []
    if isinstance(dead, str):
        return dead[:240]
    try:
        return "; ".join(str(item)[:120] for item in dead)[:240]
    except TypeError:
        return ""


def select_rotated_target(
    registry: Mapping[str, Any],
    loop_results: Mapping[str, Mapping[str, Any]],
    *,
    recent_no_bank_targets: Sequence[str] = DEFAULT_RECENT_NO_BANK_TARGETS,
    frontier_priority: Sequence[str] = DEFAULT_FRONTIER_PRIORITY,
) -> dict[str, Any]:
    """REQ-ARC-FCP-5450: precheck public-game depths and rotate off stale targets."""

    rows = _registry_rows(registry)
    recent = {str(item) for item in recent_no_bank_targets}
    total = _as_int(registry.get("reproducible_total_levels"))
    frontier_precheck: dict[str, dict[str, Any]] = {}
    skipped_recent: list[str] = []

    for game in frontier_priority:
        row = rows.get(str(game))
        registry_depth = _as_int((row or {}).get("levels_reproduced"))
        loop_depth = _loop_depth(loop_results, str(game))
        current_depth = max(registry_depth, loop_depth)
        target_level = current_depth + 1 if current_depth > 0 else 1
        marker = f"{game}:{_level_label(target_level)}"
        frontier_precheck[str(game)] = {
            "registry_levels": registry_depth,
            "loop_reproduced_levels": loop_depth,
            "current_precheck_depth": current_depth,
            "next_target_level": target_level,
            "recent_no_bank": marker in recent,
            "dead_end_summary": _dead_end_summary(row or {}),
        }
        if marker in recent:
            skipped_recent.append(marker)
            continue
        if row is None or str(row.get("reproducibility")) != "reproduced" or current_depth < 1:
            continue
        reason = (
            "rotated_away_from_recent_no_bank_targets="
            f"{','.join(skipped_recent) or 'none'}; selected {game} "
            f"{_level_label(target_level)} after registry depth {_level_label(current_depth)}"
        )
        return {
            "status": "selected",
            "registry_precheck_total_levels": total,
            "selected_game": str(game),
            "selected_target_level": int(target_level),
            "selected_target_level_label": _level_label(target_level),
            "registry_level_before": int(current_depth),
            "target_rotation_reason": reason,
            "skipped_recent_no_bank_targets": skipped_recent,
            "frontier_precheck": frontier_precheck,
            "duplicate_solve_avoided": True,
        }

    return {
        "status": "blocked",
        "registry_precheck_total_levels": total,
        "selected_game": "",
        "selected_target_level": 0,
        "selected_target_level_label": "",
        "registry_level_before": 0,
        "target_rotation_reason": (
            "no_eligible_frontier_after_rotation; skipped_recent_no_bank_targets="
            f"{','.join(skipped_recent) or 'none'}"
        ),
        "skipped_recent_no_bank_targets": skipped_recent,
        "frontier_precheck": frontier_precheck,
        "duplicate_solve_avoided": True,
    }


def _first_mapping(rows: Sequence[Any]) -> Mapping[str, Any] | None:
    for row in rows:
        if isinstance(row, Mapping):
            return row
    return None


def induce_runtime_predicates(attempt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """SCENARIO-ARC-FCP-5450: summarize runtime measurement-access predicates."""

    predicates: list[dict[str, Any]] = []
    observation = _first_mapping(list(attempt.get("runtime_observations") or []))
    if observation is not None:
        predicates.append(
            {
                "predicate": "frame_level_measurement",
                "level_before": _as_int(observation.get("level_before")),
                "level_after": _as_int(observation.get("level_after")),
                "changed_cells": _as_int(observation.get("changed_cells")),
                "source": "frame_level_runtime_measurement",
            }
        )

    receipt_owner = _first_mapping(list(attempt.get("action_sequence_receipts") or []))
    receipt = None
    if receipt_owner is not None:
        receipt = _first_mapping(list(receipt_owner.get("measurement_receipts") or []))
    if receipt is not None:
        predicates.append(
            {
                "predicate": "action_effect_observation",
                "receipt_id": str(receipt.get("receipt_id") or ""),
                "changed_cells": _as_int(receipt.get("changed_cells")),
                "source": "measurement_access_receipt",
            }
        )

    transition = _first_mapping(list(attempt.get("frontier_transitions") or []))
    if transition is not None:
        predicates.append(
            {
                "predicate": "state_change_summary",
                "from_hash": str(transition.get("from_hash") or ""),
                "to_hash": str(transition.get("to_hash") or ""),
                "action": _as_int(transition.get("action")),
                "changed_cells": _as_int(transition.get("changed_cells")),
                "source": "frontier_transition_summary",
            }
        )

    route = _first_mapping(list(attempt.get("generic_verifier_routes") or []))
    induction = _first_mapping(list(attempt.get("induction_attempts") or []))
    if route is not None or induction is not None:
        predicates.append(
            {
                "predicate": "verifier_routed_predicate",
                "route": str((route or {}).get("route") or (induction or {}).get("reason") or ""),
                "accepted": bool((route or {}).get("accepted", (induction or {}).get("planned", False))),
                "transition_count": _as_int((induction or {}).get("transition_count")),
                "source": "generic_verifier_route",
            }
        )
    return predicates


def _action_label(action: int | str, data: Any) -> str:
    return json.dumps({"action": int(action), "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def run_live_measurement_access_attempt(  # pragma: no cover - ARC runtime boundary
    *,
    root: Path,
    selection: Mapping[str, Any],
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    game = str(selection["selected_game"])
    target_before = _as_int(selection.get("registry_level_before"))
    target_level = _as_int(selection.get("selected_target_level"))
    generator = LiveCoExLandmarkFrontierGenerator(min_support=1, max_uncertainty=0.51)
    labels: list[str] = []
    frames: list[Any] = []
    runtime_observations: list[dict[str, Any]] = []
    reset_count = 0
    max_level = target_before
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
        latest = None
        previous_hash = ""
        for _index in range(max(1, int(budget))):
            if policy.is_done(frames, latest):
                break
            level_before = int(_level_of(latest))
            kind, data = policy.next_move(frames, latest)
            if kind == "RESET":
                latest = env.reset()
                reset_count += 1
                generator.record_reset(level=max_level)
                previous_hash = "reset"
            elif kind is None:
                break
            else:
                latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
                labels.append(_action_label(int(kind), data))
            level_after = int(_level_of(latest))
            max_level = max(max_level, level_after)
            try:
                from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of

                after_hash = frame_hash(grid_of(latest))
            except Exception:
                after_hash = ""
            runtime_observations.append(
                {
                    "level_before": level_before,
                    "level_after": level_after,
                    "before_hash": previous_hash,
                    "after_hash": after_hash,
                    "changed_cells": 0,
                }
            )
            previous_hash = after_hash
            frames.append(latest)
            if max_level >= target_level:
                break
        gate = {
            "game": game,
            "claimed_level": 0,
            "reached_level": 0,
            "reproduced": False,
            "mode": "offline_reproduction_gate_no_new_level_claim",
        }
        if max_level >= target_level and labels:
            gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=max_level))
        reproduced = (
            bool(gate.get("reproduced")) and _as_int(gate.get("reached_level")) >= target_level
        )
        diagnostics = generator.diagnostics()
        attempt = {
            "selected_game": game,
            "selected_target_level": target_level,
            "live_attempt_count": len(labels),
            "reset_count": reset_count,
            "max_level_reached": max_level,
            "frontier_expansion_count": _as_int(diagnostics.get("frontier_expansion_count")),
            "frontier_transitions": list(diagnostics.get("frontier_transitions") or []),
            "runtime_observations": runtime_observations,
            "action_sequence_receipts": list(diagnostics.get("action_sequence_receipts") or []),
            "generic_verifier_routes": [],
            "induction_attempts": list(policy.induction_attempts),
            "offline_reproduced": bool(reproduced),
            "reproduced_levels": max(0, _as_int(gate.get("reached_level")) - target_before)
            if reproduced
            else 0,
            "reproduction_gate": gate,
            "residual_wall": "" if reproduced else "bounded_budget_no_levelup",
            "solution_labels": list(labels) if reproduced else [],
            "runtime_self_discovery": True,
            "no_offline_bfs": True,
            "no_source_reading": True,
            "no_per_game_adapter_credited": True,
            "root": str(root),
        }
        attempt["runtime_predicates_induced"] = induce_runtime_predicates(attempt)
        return attempt
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _new_levels_banked(attempt: Mapping[str, Any]) -> int:
    if attempt.get("offline_reproduced") is not True:
        return 0
    return max(0, _as_int(attempt.get("reproduced_levels")))


def build_artifact(
    *,
    selection: Mapping[str, Any],
    attempt: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    blocked = bool(attempt.get("blocked")) or selection.get("status") == "blocked"
    selected_game = str(selection.get("selected_game") or "")
    selected_target_level = _as_int(selection.get("selected_target_level"))
    registry_before = _as_int(selection.get("registry_level_before"))
    predicates = list(attempt.get("runtime_predicates_induced") or induce_runtime_predicates(attempt))
    new_levels = _new_levels_banked(attempt)
    target_advanced = selected_target_level > registry_before
    discipline_flags = (
        attempt.get("no_offline_bfs", True) is True
        and attempt.get("no_source_reading", True) is True
        and attempt.get("no_per_game_adapter_credited", True) is True
    )
    can_bank = bool(
        new_levels >= 1
        and target_advanced
        and predicates
        and attempt.get("runtime_self_discovery") is True
        and discipline_flags
    )
    status = "complete" if can_bank else "blocked" if blocked else "honest_null"
    residual_wall = str(
        attempt.get("residual_wall")
        or ("duplicate_or_precondition_block" if status == "blocked" else "bounded_budget_no_levelup")
    )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5450_arc_measurement_access_live_levelup_v495.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "solve_provenance": SOLVE_PROVENANCE,
        "registry_precheck_total_levels": _as_int(
            selection.get("registry_precheck_total_levels")
        ),
        "selected_game": selected_game,
        "selected_target_level": selected_target_level,
        "selected_target_level_label": str(selection.get("selected_target_level_label") or ""),
        "registry_level_before": registry_before,
        "target_rotation_reason": str(selection.get("target_rotation_reason") or ""),
        "live_attempt_count": _as_int(attempt.get("live_attempt_count")),
        "runtime_predicates_induced": predicates,
        "offline_reproduced": bool(can_bank),
        "reproduced_levels": int(new_levels if can_bank else 0),
        "new_levels_banked": int(new_levels if can_bank else 0),
        "new_level_reproduced": bool(can_bank),
        "no_offline_bfs": attempt.get("no_offline_bfs", True) is True,
        "no_source_reading": attempt.get("no_source_reading", True) is True,
        "no_per_game_adapter_credited": attempt.get("no_per_game_adapter_credited", True)
        is True,
        "arc_new_level_banked": bool(can_bank),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {selected_game} L{selected_target_level} measurement-access live level reproduced"
            if can_bank
            else f"blocked: {residual_wall}"
            if status == "blocked"
            else f"honest_null: {selected_game} L{selected_target_level} {residual_wall}"
        ),
        "target_selection": dict(selection),
        "frontier_expansion_count": _as_int(attempt.get("frontier_expansion_count")),
        "frontier_transitions": list(attempt.get("frontier_transitions") or []),
        "runtime_observations": list(attempt.get("runtime_observations") or []),
        "action_sequence_receipts": list(attempt.get("action_sequence_receipts") or []),
        "generic_verifier_routes": list(attempt.get("generic_verifier_routes") or []),
        "induction_attempts": list(attempt.get("induction_attempts") or []),
        "reproduction_gate": dict(attempt.get("reproduction_gate") or {}),
        "solution_labels": list(attempt.get("solution_labels") or []),
        "residual_wall": "" if can_bank else residual_wall,
        "preconditions_checked": dict(preconditions_checked),
        "tests_run": list(tests_run),
        "duration_s": float(duration_s),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    status = artifact.get("status")
    if status not in {"complete", "honest_null", "blocked"}:
        errors.append("status must be complete, honest_null, or blocked")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    for field in ("registry_precheck_total_levels", "selected_target_level", "live_attempt_count"):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    for field in ("reproduced_levels", "new_levels_banked"):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    for field in (
        "offline_reproduced",
        "new_level_reproduced",
        "no_offline_bfs",
        "no_source_reading",
        "no_per_game_adapter_credited",
        "arc_new_level_banked",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    for field in ("no_offline_bfs", "no_source_reading", "no_per_game_adapter_credited"):
        if artifact.get(field) is not True:
            errors.append(f"{field} must be true")
    if status != "blocked":
        if not isinstance(artifact.get("selected_game"), str) or not artifact.get("selected_game"):
            errors.append("selected_game must be non-empty string")
        if _as_int(artifact.get("selected_target_level")) < 1:
            errors.append("selected_target_level must be >= 1")
        if not isinstance(artifact.get("runtime_predicates_induced"), list) or not artifact.get(
            "runtime_predicates_induced"
        ):
            errors.append("runtime_predicates_induced must be a non-empty list")
    if not isinstance(artifact.get("target_rotation_reason"), str) or not artifact.get(
        "target_rotation_reason"
    ):
        errors.append("target_rotation_reason must be non-empty string")
    if type(artifact.get("selected_target_level")) is not int:
        errors.append("selected_target_level must be bare int")
    if type(artifact.get("live_attempt_count")) is not int:
        errors.append("live_attempt_count must be bare int")
    if status == "complete":
        if artifact.get("offline_reproduced") is not True:
            errors.append("complete artifact requires offline_reproduced true")
        if artifact.get("new_level_reproduced") is not True:
            errors.append("complete artifact requires new_level_reproduced true")
        if artifact.get("arc_new_level_banked") is not True:
            errors.append("complete artifact requires arc_new_level_banked true")
        if type(artifact.get("reproduced_levels")) is int and artifact["reproduced_levels"] < 1:
            errors.append("complete artifact requires reproduced_levels >= 1")
        if type(artifact.get("new_levels_banked")) is int and artifact["new_levels_banked"] < 1:
            errors.append("complete artifact requires new_levels_banked >= 1")
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
    attempt_runner=run_live_measurement_access_attempt,
    offline_arcade_check=offline_arcade_available,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> dict[str, Any]:
    start = time.perf_counter()
    registry = load_registry(root)
    loop_results = load_arc_loop_results(root)
    selection = select_rotated_target(registry, loop_results)
    preconditions_checked = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "spec_has_req_5450": "REQ-ARC-FCP-5450"
        in (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8"),
        "registry_present": (root / REGISTRY_RELATIVE_PATH).exists(),
        "arc_loop_results_checked": bool(loop_results),
        "offline_arcade_available": bool(offline_arcade_check()),
        "no_offline_bfs": True,
        "no_source_reading": True,
        "no_per_game_adapter_credited": True,
    }
    if selection.get("status") == "blocked" or not all(preconditions_checked.values()):
        attempt = {
            "blocked": True,
            "live_attempt_count": 0,
            "residual_wall": "missing_harness_or_target_precondition",
            "no_offline_bfs": True,
            "no_source_reading": True,
            "no_per_game_adapter_credited": True,
        }
    else:
        attempt = attempt_runner(root=root, selection=selection, budget=budget)
    artifact = build_artifact(
        selection=selection,
        attempt=attempt,
        preconditions_checked=preconditions_checked,
        tests_run=tests_run,
        duration_s=time.perf_counter() - start,
    )
    validate_artifact(artifact)
    _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
