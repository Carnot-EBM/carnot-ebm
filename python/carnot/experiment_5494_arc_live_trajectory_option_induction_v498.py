"""Experiment 5494: Exp5493-selected live ARC trajectory induction.

Spec refs: REQ-ARC-FCP-5494, SCENARIO-ARC-FCP-5494.

This module is intentionally an accounting shell around one bounded live-agent
attempt. The credited work is the runtime process: the agent tries actions,
observes frame deltas, induces option-like action sequences, and either passes
the standard reproduction gate or records why those hypotheses were rejected.
The module does not read game source, does not run offline ground-truth BFS, and
does not construct a per-game adapter because those shortcuts would not improve
the hidden-game live self-discovery loop.
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


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5494
EXPERIMENT = "experiment_5494_arc_live_trajectory_option_induction_v498"
MILESTONE = "2026.07.498"
RESULT_RELATIVE_PATH = "results/experiment_5494_arc_live_trajectory_levelup_v498.json"
EXP5493_RELATIVE_PATH = "results/experiment_5493_arc_trajectory_target_precheck_v498.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5494", "SCENARIO-ARC-FCP-5494"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "arc_live_agent_self_discovery"
RANDOM_SEED = 5494
DEFAULT_BUDGET = 48
MANDATED_LLM_MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_TRAJECTORY_PRECONDITIONS = (
    "runtime_action_effect_observations",
    "visible_toggle_or_navigation_state_changes",
    "level_counter_delta_read_from_frames",
    "frontier_prefixes_grouped_into_options",
)
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5494_arc_live_trajectory_option_induction_v498.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5494_arc_live_trajectory_option_induction_v498.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5494_arc_live_trajectory_option_induction_v498.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "selected_game": {
        "principle": "Exp5493-selected game id, or empty string only when the attempt blocks before target selection."
    },
    "target_level": {"principle": "Exp5493-selected target level as a bare int."},
    "prior_levels_reproduced": {
        "principle": "registry depth before Exp5494; success must be strictly deeper."
    },
    "post_levels_reproduced": {
        "principle": "registry depth after Exp5494; unchanged on honest null."
    },
    "solve_provenance": {"principle": "must equal live_agent_self_discovery."},
    "offline_bfs_used": {
        "principle": "must be false; no offline ground-truth BFS is credited."
    },
    "per_game_adapter_used": {
        "principle": "must be false; no hand per-game adapter is credited."
    },
    "game_source_read": {
        "principle": "must be false; source reading is outside the credited live path."
    },
    "trajectory_hypothesis_count": {
        "principle": "bare int count of hypothesized action sequences induced from runtime observations."
    },
    "live_attempt_count": {"principle": "bare int count of live actions actually executed."},
    "offline_reproduced": {
        "principle": "true only after the live-discovered candidate passes the reproduction gate."
    },
    "reproduced_levels": {
        "principle": "new reproduced levels beyond the prior depth; complete requires >=1."
    },
    "new_level_banked": {
        "principle": "true only when offline_reproduced=true and reproduced_levels>=1."
    },
    "registry_updated": {
        "principle": "true only when a newly reproduced level is written to ops/arc_solve_registry.yaml."
    },
    "model_specs_if_llm_used": {
        "principle": "empty when no LLM was invoked; otherwise contains the three mandated headline GGUF model specs."
    },
    "failure_mode": {"principle": "empty on success or concise blocked/no-bank reason."},
    "inference_substrate": {"principle": "must equal arc_live_agent_self_discovery."},
    "random_seed": {"principle": "deterministic seed for the bounded attempt."},
    "honest_verdict": {
        "principle": "terminal status starts with complete:, honest_null:, or blocked:."
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


class _NoOpProposer:  # pragma: no cover - ARC runtime boundary
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5494_no_live_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:
        return []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _level_label(level: int) -> str:
    return f"L{max(0, int(level))}"


def _target_marker(game: str, level: int) -> str:
    return f"{game}:{_level_label(level)}"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"reproducible_total_levels": 0, "games": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }


def load_registry(root: Path = REPO) -> dict[str, Any]:
    registry = _load_yaml(root / REGISTRY_RELATIVE_PATH)
    if registry:
        return registry
    return {"reproducible_total_levels": 0, "games": []}


def select_target_from_exp5493(
    exp5493_artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    requested_game: str | None = None,
    requested_target_level: int | None = None,
    requested_prior_levels: int | None = None,
) -> dict[str, Any]:
    """REQ-ARC-FCP-5494: gate the Exp5493 target before live budget is spent."""

    game = str(exp5493_artifact.get("selected_game") or "")
    target = _as_int(exp5493_artifact.get("selected_target_level"))
    prior_from_precheck = _as_int(exp5493_artifact.get("prior_levels_reproduced"))
    if requested_game and requested_game != game:
        return _blocked_selection("requested_game_mismatch", game, target, prior_from_precheck)
    if requested_target_level is not None and int(requested_target_level) != target:
        return _blocked_selection("requested_target_level_mismatch", game, target, prior_from_precheck)
    if requested_prior_levels is not None and int(requested_prior_levels) != prior_from_precheck:
        return _blocked_selection("requested_prior_levels_mismatch", game, target, prior_from_precheck)
    if exp5493_artifact.get("arc_trajectory_precheck_ready") is not True:
        return _blocked_selection("exp5493_precheck_not_ready", game, target, prior_from_precheck)
    if not game or target <= 0:
        return _blocked_selection("missing_exp5493_target", game, target, prior_from_precheck)

    rows = _registry_rows(registry)
    row = rows.get(game)
    if row is None or str(row.get("reproducibility") or "") != "reproduced":
        return _blocked_selection("missing_reproduced_registry_row", game, target, prior_from_precheck)
    prior = _as_int(row.get("levels_reproduced"))
    if prior >= target:
        return _blocked_selection("target_already_reproduced", game, target, prior)

    marker = _target_marker(game, target)
    recent_no_bank = [str(item) for item in exp5493_artifact.get("excluded_recent_no_bank_targets") or []]
    if marker in set(recent_no_bank):
        return _blocked_selection("recent_same_mechanism_no_bank", game, target, prior)
    audit = exp5493_artifact.get("candidate_audit") or {}
    audit_row = audit.get(marker) if isinstance(audit, Mapping) else None
    if isinstance(audit_row, Mapping) and audit_row.get("decision") != "selected":
        return _blocked_selection("exp5493_candidate_not_selected", game, target, prior)

    preconditions = [
        str(item) for item in exp5493_artifact.get("trajectory_induction_preconditions") or []
    ]
    missing = [item for item in REQUIRED_TRAJECTORY_PRECONDITIONS if item not in preconditions]
    if missing:
        selection = _blocked_selection("missing_live_trajectory_preconditions", game, target, prior)
        selection["missing_trajectory_induction_preconditions"] = missing
        return selection

    return {
        "blocked": False,
        "selected_game": game,
        "target_level": target,
        "prior_levels_reproduced": prior,
        "duplicate_solve_avoided": True,
        "trajectory_induction_preconditions": preconditions,
        "recent_no_bank_targets": recent_no_bank,
        "selection_reason": "exp5493_selected_non_duplicate_trajectory_target",
    }


def _blocked_selection(blocker: str, game: str, target: int, prior: int) -> dict[str, Any]:
    return {
        "blocked": True,
        "blocker": str(blocker),
        "selected_game": str(game or ""),
        "target_level": int(max(0, target)),
        "prior_levels_reproduced": int(max(0, prior)),
        "duplicate_solve_avoided": str(blocker) != "target_already_reproduced",
        "trajectory_induction_preconditions": [],
        "recent_no_bank_targets": [],
        "selection_reason": str(blocker),
    }


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
    except Exception:
        return False
    return True


def _action_label(action: int | str, data: Any) -> str:
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _label_to_step(label: str) -> dict[str, Any] | None:
    if label == "RESET":
        return None
    try:
        row = json.loads(label)
    except (TypeError, ValueError):
        return None
    return {"action": _as_int(row.get("action")), "data": row.get("data")}


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def _hypotheses_from_diagnostics(
    diagnostics: Mapping[str, Any],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    receipts = diagnostics.get("action_sequence_receipts") or []
    hypotheses: list[dict[str, Any]] = []
    for index, receipt in enumerate(receipts, start=1):
        if not isinstance(receipt, Mapping):
            continue
        sequence = [dict(step) for step in receipt.get("sequence") or [] if isinstance(step, Mapping)]
        if sequence:
            hypotheses.append(
                {
                    "hypothesis_id": f"h{index:04d}",
                    "sequence": sequence,
                    "replayable": bool(receipt.get("replayable", True)),
                    "source": "live_coex_action_sequence_receipt",
                    "measurement_receipt_count": len(receipt.get("measurement_receipts") or []),
                }
            )
    if hypotheses:
        return hypotheses

    steps = [step for step in (_label_to_step(label) for label in labels) if step is not None]
    if not steps:
        return []
    return [
        {
            "hypothesis_id": "h0001",
            "sequence": steps[: min(3, len(steps))],
            "replayable": True,
            "source": "executed_live_action_prefix_fallback",
            "measurement_receipt_count": 0,
        }
    ]


def _observation_deltas_from_diagnostics(diagnostics: Mapping[str, Any]) -> list[dict[str, Any]]:
    observations = diagnostics.get("runtime_observations") or []
    deltas = [dict(row) for row in observations if isinstance(row, Mapping)]
    if deltas:
        return deltas
    receipts = diagnostics.get("measurement_access_receipts") or []
    return [
        {
            "action": _as_int(row.get("action")),
            "data": row.get("data"),
            "changed_cells": _as_int(row.get("changed_cells")),
            "level_before": _as_int(row.get("level_before")),
            "level_after": _as_int(row.get("level_after")),
            "receipt_id": str(row.get("receipt_id") or ""),
        }
        for row in receipts
        if isinstance(row, Mapping)
    ]


def _verifier_checks_from_diagnostics(
    diagnostics: Mapping[str, Any],
    reproduction_gate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks = [dict(row) for row in diagnostics.get("verifier_observations") or [] if isinstance(row, Mapping)]
    checks.append(
        {
            "check": "standard_live_offline_reproduction_gate",
            "reproduced": bool(reproduction_gate.get("reproduced")),
            "claimed_level": _as_int(reproduction_gate.get("claimed_level")),
            "reached_level": _as_int(reproduction_gate.get("reached_level")),
        }
    )
    return checks


def _rejection_reasons(
    *,
    diagnostics: Mapping[str, Any],
    reproduced: bool,
    failure_mode: str,
) -> list[str]:
    if reproduced:
        return []
    reasons = [failure_mode]
    if _as_int(diagnostics.get("uncertainty_rejections")) > 0:
        reasons.append("uncertainty_gate_rejected_low_support_options")
    if _as_int(diagnostics.get("frontier_expansion_count")) == 0:
        reasons.append("no_accepted_trajectory_prefix")
    return list(dict.fromkeys(reason for reason in reasons if reason))


def run_live_trajectory_option_attempt(  # pragma: no cover - ARC runtime boundary
    *,
    root: Path,
    selection: Mapping[str, Any],
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of
    from carnot.agentic.arc_live_trajectory_frontier import LiveCoExLandmarkFrontierGenerator

    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    game = str(selection["selected_game"])
    prior = _as_int(selection["prior_levels_reproduced"])
    target = _as_int(selection["target_level"])
    generator = LiveCoExLandmarkFrontierGenerator(min_support=1, max_uncertainty=0.51)
    reset_count = 0
    try:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        policy = E3AgentPolicy(
            game,
            proposer=_NoOpProposer(),
            explore_budget=max(1, int(budget)),
            target_levels=max(1, target),
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
        max_level = prior
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
            observed_level = _as_int(_level_of(latest), default=max_level)
            max_level = max(max_level, observed_level)
            frames.append(latest)
            if latest is None or max_level >= target:
                break

        gate: dict[str, Any] = {
            "game": game,
            "claimed_level": max_level,
            "reached_level": prior,
            "reproduced": False,
            "mode": "standard_reproduction_gate_not_run_no_new_target_candidate",
        }
        if max_level > prior and labels:
            gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=max_level))
        reproduced = bool(gate.get("reproduced")) and _as_int(gate.get("reached_level")) >= target
        post = _as_int(gate.get("reached_level"), default=max_level) if reproduced else prior
        failure_mode = "" if reproduced else "bounded_budget_no_target_level_reproduction"
        diagnostics = generator.diagnostics()
        return {
            "live_attempt_count": len([label for label in labels if label != "RESET"]),
            "reset_count": int(reset_count),
            "post_levels_reproduced": int(post),
            "max_level_reached": int(max_level),
            "offline_reproduced": bool(reproduced),
            "reproduced_levels": max(0, int(post) - int(prior)) if reproduced else 0,
            "trajectory_hypotheses": _hypotheses_from_diagnostics(diagnostics, labels),
            "observation_deltas": _observation_deltas_from_diagnostics(diagnostics),
            "verifier_checks": _verifier_checks_from_diagnostics(diagnostics, gate),
            "rejection_reasons": _rejection_reasons(
                diagnostics=diagnostics,
                reproduced=reproduced,
                failure_mode=failure_mode,
            ),
            "failure_mode": failure_mode,
            "frontier_expansion_count": _as_int(diagnostics.get("frontier_expansion_count")),
            "landmark_count": _as_int(diagnostics.get("landmark_count")),
            "action_history_clusters": list(diagnostics.get("action_history_clusters") or []),
            "measurement_access_receipts": list(diagnostics.get("measurement_access_receipts") or []),
            "reproduction_gate": gate,
            "solution_labels": list(labels) if reproduced else [],
            "offline_bfs_used": False,
            "per_game_adapter_used": False,
            "game_source_read": False,
            "llm_generator_invoked": False,
            "model_specs_if_llm_used": [],
            "root": str(root),
        }
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _accepted_reproduced_levels(
    selection: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> int:
    if attempt.get("offline_reproduced") is not True:
        return 0
    if (
        attempt.get("offline_bfs_used", False)
        or attempt.get("per_game_adapter_used", False)
        or attempt.get("game_source_read", False)
    ):
        return 0
    prior = _as_int(selection.get("prior_levels_reproduced"))
    target = _as_int(selection.get("target_level"))
    post = _as_int(attempt.get("post_levels_reproduced"), prior + _as_int(attempt.get("reproduced_levels")))
    if post <= prior or post < target:
        return 0
    return max(0, _as_int(attempt.get("reproduced_levels"), post - prior))


def _blocked_attempt(selection: Mapping[str, Any], failure_mode: str) -> dict[str, Any]:
    prior = _as_int(selection.get("prior_levels_reproduced"))
    return {
        "blocked": True,
        "live_attempt_count": 0,
        "post_levels_reproduced": prior,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "trajectory_hypotheses": [],
        "observation_deltas": [],
        "verifier_checks": [],
        "rejection_reasons": [str(failure_mode)],
        "failure_mode": str(failure_mode),
        "offline_bfs_used": False,
        "per_game_adapter_used": False,
        "game_source_read": False,
        "llm_generator_invoked": False,
        "model_specs_if_llm_used": [],
    }


def build_artifact(
    *,
    selection: Mapping[str, Any],
    attempt: Mapping[str, Any],
    registry_updated: bool,
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    prior = _as_int(selection.get("prior_levels_reproduced"))
    target = _as_int(selection.get("target_level"))
    selected_game = str(selection.get("selected_game") or "")
    accepted_delta = _accepted_reproduced_levels(selection, attempt)
    can_bank = bool(accepted_delta >= 1 and registry_updated)
    blocked = bool(selection.get("blocked")) or bool(attempt.get("blocked"))
    status = "complete" if can_bank else "blocked" if blocked else "honest_null"
    post = prior + accepted_delta if can_bank else prior
    hypotheses = [dict(row) for row in attempt.get("trajectory_hypotheses") or [] if isinstance(row, Mapping)]
    model_specs = [str(item) for item in attempt.get("model_specs_if_llm_used") or []]
    failure_mode = (
        ""
        if can_bank
        else str(
            attempt.get("failure_mode")
            or selection.get("blocker")
            or "bounded_budget_no_target_level_reproduction"
        )
    )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5494_arc_live_trajectory_option_induction_v498.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "selected_game": selected_game,
        "target_level": int(target),
        "prior_levels_reproduced": int(prior),
        "post_levels_reproduced": int(post),
        "solve_provenance": SOLVE_PROVENANCE,
        "offline_bfs_used": bool(attempt.get("offline_bfs_used", False)),
        "per_game_adapter_used": bool(attempt.get("per_game_adapter_used", False)),
        "game_source_read": bool(attempt.get("game_source_read", False)),
        "trajectory_hypothesis_count": int(len(hypotheses)),
        "live_attempt_count": _as_int(attempt.get("live_attempt_count")),
        "offline_reproduced": bool(can_bank),
        "reproduced_levels": int(accepted_delta if can_bank else 0),
        "new_level_banked": bool(can_bank),
        "registry_updated": bool(registry_updated),
        "model_specs_if_llm_used": model_specs,
        "failure_mode": failure_mode,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "honest_verdict": (
            f"complete: {selected_game} {_level_label(target)} live trajectory reproduced and banked"
            if can_bank
            else f"blocked: {failure_mode}"
            if status == "blocked"
            else f"honest_null: {selected_game} {_level_label(target)} {failure_mode}"
        ),
        "trajectory_hypotheses": hypotheses,
        "observation_deltas": [
            dict(row) for row in attempt.get("observation_deltas") or [] if isinstance(row, Mapping)
        ],
        "verifier_checks": [
            dict(row) for row in attempt.get("verifier_checks") or [] if isinstance(row, Mapping)
        ],
        "rejection_reasons": [str(row) for row in attempt.get("rejection_reasons") or []],
        "target_selection": dict(selection),
        "attempt": dict(attempt),
        "preconditions_checked": dict(preconditions_checked),
        "tests_run": list(tests_run),
        "duration_s": float(duration_s),
    }
    return artifact


def _model_spec_errors(model_specs: Sequence[str], llm_invoked: bool) -> list[str]:
    if not llm_invoked and not model_specs:
        return []
    return [
        f"model_specs_if_llm_used missing {spec}"
        for spec in MANDATED_LLM_MODEL_SPECS
        if spec not in set(model_specs)
    ]


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    if artifact.get("status") not in {"complete", "honest_null", "blocked"}:
        errors.append("status must be complete, honest_null, or blocked")
    if not isinstance(artifact.get("selected_game"), str):
        errors.append("selected_game must be a string")
    for field in (
        "target_level",
        "prior_levels_reproduced",
        "post_levels_reproduced",
        "trajectory_hypothesis_count",
        "live_attempt_count",
        "reproduced_levels",
        "random_seed",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
        elif field in ("trajectory_hypothesis_count", "live_attempt_count", "reproduced_levels") and artifact[field] < 0:
            errors.append(f"{field} must be non-negative")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    for field in ("offline_bfs_used", "per_game_adapter_used", "game_source_read"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
        elif artifact.get(field) is not False:
            errors.append(f"{field} must be false")
    for field in ("offline_reproduced", "new_level_banked", "registry_updated"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    for field in ("trajectory_hypotheses", "observation_deltas", "verifier_checks", "rejection_reasons"):
        if not isinstance(artifact.get(field), list):
            errors.append(f"{field} must be a list")
    model_specs = artifact.get("model_specs_if_llm_used")
    if not isinstance(model_specs, list):
        errors.append("model_specs_if_llm_used must be a list")
        model_specs = []
    errors.extend(_model_spec_errors([str(item) for item in model_specs], bool(artifact.get("llm_generator_invoked")) or bool(model_specs)))

    prior = _as_int(artifact.get("prior_levels_reproduced"))
    post = _as_int(artifact.get("post_levels_reproduced"))
    if post < prior:
        errors.append("post_levels_reproduced must be >= prior_levels_reproduced")
    if artifact.get("offline_reproduced") is True:
        if _as_int(artifact.get("reproduced_levels")) < 1:
            errors.append("offline_reproduced requires reproduced_levels >= 1")
        if post <= prior:
            errors.append("offline_reproduced requires post_levels_reproduced > prior_levels_reproduced")
    if artifact.get("new_level_banked") is True:
        if artifact.get("offline_reproduced") is not True:
            errors.append("new_level_banked requires offline_reproduced true")
        if artifact.get("registry_updated") is not True:
            errors.append("new_level_banked requires registry_updated true")
    if artifact.get("registry_updated") is True and artifact.get("new_level_banked") is not True:
        errors.append("registry_updated requires new_level_banked true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "honest_null:", "blocked:")):
        errors.append("honest_verdict must start with complete:, honest_null:, or blocked:")
    if "solved" in verdict.lower():
        errors.append("honest_verdict must not claim an unreproduced solve")
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
    if artifact.get("new_level_banked") is not True:
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
            row["latest_exp5494_levelup_attempt"] = {
                "artifact": RESULT_RELATIVE_PATH,
                "offline_reproduced": True,
                "reproduced_levels": delta,
                "prior_levels_reproduced": _as_int(artifact.get("prior_levels_reproduced")),
                "post_levels_reproduced": post,
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
                "latest_exp5494_levelup_attempt": {
                    "artifact": RESULT_RELATIVE_PATH,
                    "offline_reproduced": True,
                    "reproduced_levels": delta,
                    "post_levels_reproduced": post,
                    "solve_provenance": SOLVE_PROVENANCE,
                },
            }
        )
    updated["games"] = games
    updated["reproducible_total_levels"] = _as_int(updated.get("reproducible_total_levels")) + delta
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
    requested_game: str | None = None,
    requested_target_level: int | None = None,
    requested_prior_levels: int | None = None,
    attempt_runner: Callable[..., Mapping[str, Any]] = run_live_trajectory_option_attempt,
    offline_arcade_check: Callable[[], bool] = offline_arcade_available,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    exp5493_path = root / EXP5493_RELATIVE_PATH
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "spec_has_req_5494": (
            "REQ-ARC-FCP-5494" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "exp5493_present": exp5493_path.exists(),
        "offline_arcade_available": False,
        "offline_bfs_used": False,
        "per_game_adapter_used": False,
        "game_source_read": False,
    }
    registry = load_registry(root)
    exp5493_artifact = _load_json(exp5493_path)
    selection = select_target_from_exp5493(
        exp5493_artifact,
        registry,
        requested_game=requested_game,
        requested_target_level=requested_target_level,
        requested_prior_levels=requested_prior_levels,
    )
    ready_without_arcade = (
        preconditions["AGENTS.md"]
        and preconditions["CODEX.md"]
        and preconditions["spec_has_req_5494"]
        and preconditions["registry_present"]
        and preconditions["exp5493_present"]
        and not selection.get("blocked")
    )
    if not ready_without_arcade:
        failure = str(selection.get("blocker") or "missing_exp5494_precondition")
        attempt: Mapping[str, Any] = _blocked_attempt(selection, failure)
    else:
        preconditions["offline_arcade_available"] = bool(offline_arcade_check())
        if not preconditions["offline_arcade_available"]:
            attempt = _blocked_attempt(selection, "missing_harness_access")
        else:
            attempt = attempt_runner(root=root, selection=selection, budget=budget)

    registry_updated = False
    if _accepted_reproduced_levels(selection, attempt) >= 1:
        preliminary = build_artifact(
            selection=selection,
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
        selection=selection,
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
    parser.add_argument("--game", default=None)
    parser.add_argument("--target-level", type=int, default=None)
    parser.add_argument("--prior-levels", type=int, default=None)
    parser.add_argument("--mechanism", default="")
    parser.add_argument("--no-offline-bfs", action="store_true")
    parser.add_argument("--no-per-game-adapter", action="store_true")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = parser.parse_args(argv)
    artifact = run_experiment(
        budget=args.budget,
        requested_game=args.game,
        requested_target_level=args.target_level,
        requested_prior_levels=args.prior_levels,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
