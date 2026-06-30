"""Experiment 5054: bounded ARC live-path self-discovery attempt.

Spec refs: REQ-ARC-WMTE-5054,
SCENARIO-ARC-WMTE-5054-DUPLICATE-TARGET-GUARD,
SCENARIO-ARC-WMTE-5054-PROVENANCE-GATE,
SCENARIO-ARC-WMTE-5054-STABLE-ARTIFACT.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO / "python"))

EXPERIMENT = "experiment_5054_arc_live_path_self_discovery"
EXPERIMENT_ID = 5054
SCHEMA = "carnot.arc_live_path_self_discovery_5054.v1"
RESULT_RELATIVE_PATH = "results/experiment_5054_arc_live_path_self_discovery.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 5054
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
CURRENT_TARGET = ("lp85", 6)
DEFAULT_CANDIDATE_GAMES = ("lp85", "tu93", "s5i5", "bp35", "re86", "sb26", "lf52")
HIDDEN_STATE_TARGETS = ("ka59", "wa30")
DEFAULT_BUDGET = 36
GO_EXPLORE_CONFIG = {"enabled": True, "bins": 16, "max_cells": 128}
ALLOWED_PROVENANCE = ("live_agent_self_discovery", "development_proxy", "outer_loop_re")
SPEC_REFS = [
    "REQ-ARC-WMTE-5054",
    "SCENARIO-ARC-WMTE-5054-DUPLICATE-TARGET-GUARD",
    "SCENARIO-ARC-WMTE-5054-PROVENANCE-GATE",
    "SCENARIO-ARC-WMTE-5054-STABLE-ARTIFACT",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success_<game>_levelup_banked only for a strict "
            "live-agent reproduced level, complete_<game>_no_new_level_residual_* "
            "for honest no-bank, blocked_* for failed preconditions."
        )
    },
    "solve_provenance": {
        "principle": (
            "one of live_agent_self_discovery, development_proxy, outer_loop_re; "
            "headline credit requires live_agent_self_discovery with runtime trace evidence."
        )
    },
    "target_game": {
        "principle": "selected unsolved next-level target after registry duplicate/dead-end precheck."
    },
    "registry_precheck_passed": {
        "principle": "bare bool: false only when duplicate or malformed registry preconditions block."
    },
    "live_agent_attempts": {
        "principle": (
            "the bounded E3AgentPolicy runtime attempts; these are action traces from the live path, "
            "not offline source-reading, per-game BFS, or hand-built adapters."
        )
    },
    "new_levels_banked": {
        "principle": "bare int: increments only for strict reproduction-gated progress beyond registry depth."
    },
    "offline_reproduced": {
        "principle": "bare bool: true only when the live-agent trace passes arc_solver_kit.reproduce."
    },
    "duplicate_solve_avoided": {
        "principle": "bare bool: true when already-banked or duplicate-depth outcomes are not credited."
    },
    "reproducible_total_levels_before": {
        "principle": "registry reproducible_total_levels before the attempt."
    },
    "reproducible_total_levels_after": {
        "principle": "before + new_levels_banked; unchanged for no-bank and blocked artifacts."
    },
    "reproducibility_checksum": {
        "principle": "content hash of target selection, provenance, live attempts, and bank summary."
    },
}

REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "field_principles",
    "prior_reproduced_level",
    "target_level",
    "inference_substrate",
    "preconditions_checked",
    "candidate_selection",
    "solve_claim",
    "random_seed",
    "duration_s",
)


def _load_registry_text(root: Path) -> str:
    return (root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")


def _load_registry(registry_text: str) -> dict[str, Any]:
    loaded = yaml.safe_load(registry_text)
    return loaded if isinstance(loaded, dict) else {}


def _game_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }


def _registry_total(registry: Mapping[str, Any]) -> int:
    return int(registry.get("reproducible_total_levels") or 0)


def _dead_end_strings(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        notes: list[str] = []
        for key, item in value.items():
            if "dead_end" in str(key):
                notes.append(str(item))
            else:
                notes.extend(_dead_end_strings(item))
        return notes
    if isinstance(value, list):
        notes = []
        for item in value:
            notes.extend(_dead_end_strings(item))
        return notes
    return [str(value)] if value else []


def _row_dead_ends(row: Mapping[str, Any]) -> list[str]:
    return _dead_end_strings(row.get("dead_ends") or [])


def _has_next_level_dry_dead_end(row: Mapping[str, Any], game: str, target_level: int) -> bool:
    tokens = (
        f"no_grounded_l{int(target_level)}_delta",
        f"no grounded l{int(target_level)} delta",
        f"target-level {int(target_level)} replays to l{int(target_level) - 1}",
        "no_grounded_next_level",
        "no grounded next-level",
        "duplicate_depth",
        "same-depth",
        "same depth",
    )
    for note in _row_dead_ends(row):
        lowered = note.lower()
        if game not in lowered:
            continue
        if any(token in lowered for token in tokens):
            return True
    return False


def _candidate_audit_row(
    game: str,
    row: Mapping[str, Any] | None,
    *,
    status: str,
    reason: str,
    target_level: int = 0,
) -> dict[str, Any]:
    prior = int((row or {}).get("levels_reproduced") or 0)
    return {
        "game": game,
        "prior_reproduced_level": prior,
        "target_level": int(target_level or prior + 1),
        "status": status,
        "reason": reason,
        "dead_ends_consulted": _row_dead_ends(row or {}),
    }


def select_target(
    registry: Mapping[str, Any],
    *,
    current_target: tuple[str, int] = CURRENT_TARGET,
    candidate_games: Sequence[str] = DEFAULT_CANDIDATE_GAMES,
    hidden_state_targets: Sequence[str] = HIDDEN_STATE_TARGETS,
) -> dict[str, Any]:
    rows = _game_rows(registry)
    current_game, current_level = str(current_target[0]), int(current_target[1])
    hidden = {str(game) for game in hidden_state_targets}
    audit: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    duplicate_current: dict[str, Any] | None = None

    for raw_game in candidate_games:
        game = str(raw_game)
        row = rows.get(game)
        if row is None:
            audit.append(
                _candidate_audit_row(
                    game,
                    None,
                    status="skip_missing_registry_row",
                    reason="missing_registry_row",
                )
            )
            continue
        prior = int(row.get("levels_reproduced") or 0)
        target_level = prior + 1
        if game in hidden:
            audit.append(
                _candidate_audit_row(
                    game,
                    row,
                    status="skip_hidden_state_target",
                    reason="hidden_state_bound_target",
                    target_level=target_level,
                )
            )
            continue
        if game == current_game and current_level <= prior:
            duplicate_current = {
                "game": game,
                "prior_reproduced_level": prior,
                "target_level": current_level,
            }
            audit.append(
                _candidate_audit_row(
                    game,
                    row,
                    status="skip_duplicate_current_target",
                    reason="current_target_already_reproduced",
                    target_level=current_level,
                )
            )
            continue
        if _has_next_level_dry_dead_end(row, game, target_level):
            audit.append(
                _candidate_audit_row(
                    game,
                    row,
                    status="skip_recorded_dry_next_level",
                    reason=f"recorded_l{target_level}_dry_dead_end",
                    target_level=target_level,
                )
            )
            continue
        if selected is None:
            selected = {
                "game": game,
                "prior_reproduced_level": prior,
                "target_level": target_level,
                "status": "selected",
                "reason": "unsolved_next_level_live_path_target",
            }
            audit.append(
                _candidate_audit_row(
                    game,
                    row,
                    status="candidate_selected",
                    reason="unsolved_next_level_live_path_target",
                    target_level=target_level,
                )
            )
        else:
            audit.append(
                _candidate_audit_row(
                    game,
                    row,
                    status="alternate_not_selected",
                    reason="higher_priority_candidate_selected",
                    target_level=target_level,
                )
            )

    base = {
        "candidate_audit": audit,
        "candidate_order": [str(game) for game in candidate_games],
        "current_target": {"game": current_game, "target_level": current_level},
        "hidden_state_targets_avoided": sorted(hidden),
        "duplicate_solve_avoided": True,
    }
    if selected is not None:
        return {**selected, "registry_precheck_passed": True, **base}
    if duplicate_current is not None:
        return {
            **duplicate_current,
            "status": "blocked_duplicate_target",
            "reason": "current_target_already_reproduced_and_no_rotation_available",
            "registry_precheck_passed": False,
            **base,
        }
    return {
        "game": "none",
        "prior_reproduced_level": 0,
        "target_level": 0,
        "status": "blocked_no_unsolved_target",
        "reason": "no_unsolved_live_path_target_available",
        "registry_precheck_passed": False,
        **base,
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        "schema": artifact.get("schema"),
        "experiment": artifact.get("experiment"),
        "honest_verdict": artifact.get("honest_verdict"),
        "solve_provenance": artifact.get("solve_provenance"),
        "target_game": artifact.get("target_game"),
        "prior_reproduced_level": artifact.get("prior_reproduced_level"),
        "target_level": artifact.get("target_level"),
        "registry_precheck_passed": artifact.get("registry_precheck_passed"),
        "live_agent_attempts": artifact.get("live_agent_attempts"),
        "new_levels_banked": artifact.get("new_levels_banked"),
        "offline_reproduced": artifact.get("offline_reproduced"),
        "duplicate_solve_avoided": artifact.get("duplicate_solve_avoided"),
        "reproducible_total_levels_before": artifact.get("reproducible_total_levels_before"),
        "reproducible_total_levels_after": artifact.get("reproducible_total_levels_after"),
        "candidate_selection": artifact.get("candidate_selection"),
        "solve_claim": artifact.get("solve_claim"),
        "random_seed": artifact.get("random_seed"),
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _attempt_gate(live_attempt: Mapping[str, Any]) -> Mapping[str, Any]:
    gate = live_attempt.get("reproduction_gate")
    return gate if isinstance(gate, Mapping) else {}


def _bank_summary(
    *,
    prior_level: int,
    live_attempt: Mapping[str, Any],
) -> tuple[bool, int, str]:
    gate = _attempt_gate(live_attempt)
    reproduced = gate.get("reproduced") is True
    gate_level = int(gate.get("reached_level") or 0)
    max_level = int(live_attempt.get("max_level_reached") or 0)
    if reproduced and gate_level > prior_level and max_level > prior_level:
        return True, gate_level - prior_level, "banked_offline_reproduced_level"
    if max_level <= prior_level:
        return False, 0, "duplicate_depth"
    if not reproduced:
        return False, 0, "offline_reproduction_failed"
    return False, 0, "reproduction_not_strictly_deeper"


def _solve_claim(
    *,
    success: bool,
    live_attempt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    attempt = dict(live_attempt or {})
    evidence = {
        "runtime_self_discovery": bool(attempt.get("runtime_self_discovery")),
        "solution_labels_from_live_run": bool(attempt.get("solution_labels")) and success,
        "offline_source_reading_used": bool(attempt.get("offline_source_reading_used")),
        "per_game_bfs_used": bool(attempt.get("per_game_bfs_used")),
        "hand_built_adapter_used": bool(attempt.get("hand_built_adapter_used")),
        "llm_reasoning_invoked": bool(attempt.get("llm_reasoning_invoked")),
    }
    return {
        "claimed": bool(success),
        "provenance": SOLVE_PROVENANCE,
        "provenance_evidence": evidence,
        "reproduction_gate": dict(_attempt_gate(attempt)),
    }


def build_artifact(
    *,
    selection: Mapping[str, Any],
    registry_total: int,
    live_attempt: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    game = str(selection.get("game") or "none")
    prior_level = int(selection.get("prior_reproduced_level") or 0)
    target_level = int(selection.get("target_level") or prior_level + 1)
    success, banked, residual = _bank_summary(prior_level=prior_level, live_attempt=live_attempt)
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": (
            f"success_{game}_levelup_banked"
            if success
            else f"complete_{game}_no_new_level_residual_{residual}"
        ),
        "solve_provenance": SOLVE_PROVENANCE,
        "target_game": game,
        "prior_reproduced_level": prior_level,
        "target_level": target_level,
        "registry_precheck_passed": bool(selection.get("registry_precheck_passed")),
        "live_agent_attempts": [dict(live_attempt)],
        "new_levels_banked": int(banked),
        "offline_reproduced": bool(success),
        "duplicate_solve_avoided": not success,
        "reproducible_total_levels_before": int(registry_total),
        "reproducible_total_levels_after": int(registry_total) + int(banked),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "candidate_selection": dict(selection),
        "solve_claim": _solve_claim(success=success, live_attempt=live_attempt),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def blocked_artifact(
    *,
    reason: str,
    selection: Mapping[str, Any],
    registry_total: int,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    game = str(selection.get("game") or "none")
    prior_level = int(selection.get("prior_reproduced_level") or 0)
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": f"blocked_{reason}",
        "solve_provenance": SOLVE_PROVENANCE,
        "target_game": game,
        "prior_reproduced_level": prior_level,
        "target_level": int(selection.get("target_level") or 0),
        "registry_precheck_passed": False,
        "live_agent_attempts": [],
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "duplicate_solve_avoided": True,
        "reproducible_total_levels_before": int(registry_total),
        "reproducible_total_levels_after": int(registry_total),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "candidate_selection": dict(selection),
        "solve_claim": _solve_claim(success=False, live_attempt=None),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_bool(value: Any) -> bool:
    return type(value) is bool


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if artifact.get("experiment") != EXPERIMENT:
        errors.append("experiment mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs mismatch")
    provenance = artifact.get("solve_provenance")
    if provenance not in ALLOWED_PROVENANCE:
        errors.append("solve_provenance must be one of live_agent_self_discovery, development_proxy, outer_loop_re")
    if not _is_bool(artifact.get("registry_precheck_passed")):
        errors.append("registry_precheck_passed must be bare bool")
    if not isinstance(artifact.get("live_agent_attempts"), list):
        errors.append("live_agent_attempts must be a list")
    for field in (
        "new_levels_banked",
        "reproducible_total_levels_before",
        "reproducible_total_levels_after",
        "prior_reproduced_level",
        "target_level",
    ):
        if not _is_int(artifact.get(field)):
            errors.append(f"{field} must be bare int")
    for field in ("offline_reproduced", "duplicate_solve_avoided"):
        if not _is_bool(artifact.get(field)):
            errors.append(f"{field} must be bare bool")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (
        verdict.startswith("success_")
        or verdict.startswith("complete_")
        or verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict must use a terminal prefix")
    before = artifact.get("reproducible_total_levels_before")
    after = artifact.get("reproducible_total_levels_after")
    banked = artifact.get("new_levels_banked")
    if _is_int(before) and _is_int(after) and _is_int(banked) and after != before + banked:
        errors.append("reproducible_total_levels_after must equal before + new_levels_banked")
    if verdict.startswith("success_"):
        if provenance != "live_agent_self_discovery":
            errors.append("success requires live_agent_self_discovery provenance")
        if artifact.get("offline_reproduced") is not True:
            errors.append("success requires offline_reproduced true")
        if _is_int(banked) and banked < 1:
            errors.append("success requires new_levels_banked >= 1")
        claim = artifact.get("solve_claim")
        evidence = claim.get("provenance_evidence") if isinstance(claim, Mapping) else {}
        if not isinstance(evidence, Mapping) or evidence.get("solution_labels_from_live_run") is not True:
            errors.append("success requires live-agent solution label evidence")
        if isinstance(evidence, Mapping) and (
            evidence.get("offline_source_reading_used")
            or evidence.get("per_game_bfs_used")
            or evidence.get("hand_built_adapter_used")
        ):
            errors.append("success cannot use offline source-reading, per-game BFS, or hand adapter")
    elif artifact.get("offline_reproduced") is True:
        errors.append("non-success cannot set offline_reproduced true")
    if _is_int(banked) and banked == 0 and artifact.get("duplicate_solve_avoided") is not True:
        errors.append("no-bank artifacts must set duplicate_solve_avoided true")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        errors.append("reproducibility_checksum must be 64 hex chars")
    elif checksum != reproducibility_checksum(artifact):
        errors.append("checksum mismatch")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
    except Exception:
        return False
    return True


def _action_label(action: int | str, data: Any) -> str:  # pragma: no cover - ARC runtime
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


class _NoOpProposer:  # pragma: no cover - ARC runtime
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "disabled_exp5054_no_live_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:
        return []


def run_live_agent_attempt(
    *,
    root: Path,
    selection: Mapping[str, Any],
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:  # pragma: no cover - ARC runtime boundary
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    game = str(selection["game"])
    prior_level = int(selection.get("prior_reproduced_level") or 0)
    target_level = int(selection.get("target_level") or prior_level + 1)
    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    try:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        policy = E3AgentPolicy(
            game,
            proposer=_NoOpProposer(),
            explore_budget=max(1, int(budget)),
            target_levels=max(1, int(target_level)),
            value_head=None,
            frame_change_scorer=None,
            candidate_router=None,
            action_effect_expansion_prior=False,
            goal_bias=None,
            goal_candidate_guidance=False,
            go_explore_archive=dict(GO_EXPLORE_CONFIG),
            active_probe_controller=False,
        )
        frames: list[Any] = []
        latest = None
        labels: list[str] = []
        actions = 0
        max_level = 0
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
                actions += 1
            max_level = max(max_level, int(_level_of(latest)))
            frames.append(latest)
            if max_level >= target_level or latest is None:
                break
        claimed = max_level if max_level > prior_level else 0
        gate: dict[str, Any] = {
            "game": game,
            "claimed_level": int(claimed),
            "reached_level": 0,
            "reproduced": False,
            "mode": "offline_reproduction_gate_no_new_level_claim",
        }
        if claimed and labels:
            gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=claimed))
        go_diag = {}
        try:
            go_diag = policy.explorer.go_explore_archive_diagnostics()
        except Exception:
            go_diag = {"enabled": True, "diagnostics_error": "unavailable"}
        return {
            "attempt_id": f"{game}_live_go_explore_archive_budget_{int(budget)}",
            "target_game": game,
            "prior_reproduced_level": prior_level,
            "target_level": target_level,
            "budget": int(budget),
            "actions_taken": int(actions),
            "max_level_reached": int(max_level),
            "exceeded_registry_depth": bool(max_level > prior_level),
            "runtime_self_discovery": True,
            "policy": "E3AgentPolicy",
            "self_discovery_lever": "go_explore_archive",
            "solution_labels": list(labels) if claimed else [],
            "reproduction_gate": gate,
            "offline_source_reading_used": False,
            "per_game_bfs_used": False,
            "hand_built_adapter_used": False,
            "llm_reasoning_invoked": False,
            "model_specs": {
                "live_reasoning_model": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "fallback_reasoning_model": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "invoked": False,
            },
            "go_explore_archive": go_diag,
            "root": str(root),
        }
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable


def _write_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _blocked_and_write(
    *,
    root: Path,
    reason: str,
    selection: Mapping[str, Any],
    registry_total: int,
    preconditions: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    artifact = blocked_artifact(
        reason=reason,
        selection=selection,
        registry_total=registry_total,
        preconditions_checked=preconditions,
        duration_s=time.monotonic() - started,
    )
    _write_artifact(root, artifact)
    return artifact


def run_experiment(
    *,
    root: Path = REPO,
    current_target: tuple[str, int] = CURRENT_TARGET,
    candidate_games: Sequence[str] = DEFAULT_CANDIDATE_GAMES,
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    empty_selection = {
        "game": "none",
        "prior_reproduced_level": 0,
        "target_level": 0,
        "registry_precheck_passed": False,
        "candidate_audit": [],
    }
    preconditions: dict[str, Any] = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "arc_world_model_trust_energy_spec_has_req_5054": (
            "REQ-ARC-WMTE-5054" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "registry_loadable": False,
        "offline_arcade_available": False,
        "llm_reasoning_invoked": False,
        "offline_source_reading_used": False,
        "per_game_bfs_used": False,
        "hand_built_adapter_used": False,
    }
    if not preconditions["arc_world_model_trust_energy_spec_has_req_5054"]:
        return _blocked_and_write(
            root=root,
            reason="spec_missing",
            selection=empty_selection,
            registry_total=0,
            preconditions=preconditions,
            started=started,
        )
    try:
        registry_text = _load_registry_text(root)
        registry = _load_registry(registry_text)
        preconditions["registry_loadable"] = bool(registry)
    except (OSError, yaml.YAMLError):
        return _blocked_and_write(
            root=root,
            reason="arc_solve_registry_unreadable",
            selection=empty_selection,
            registry_total=0,
            preconditions=preconditions,
            started=started,
        )
    if not preconditions["registry_loadable"]:
        return _blocked_and_write(
            root=root,
            reason="arc_solve_registry_unreadable",
            selection=empty_selection,
            registry_total=0,
            preconditions=preconditions,
            started=started,
        )

    registry_total = _registry_total(registry)
    selection = select_target(
        registry,
        current_target=current_target,
        candidate_games=candidate_games,
    )
    if str(selection.get("status") or "").startswith("blocked_"):
        reason = str(selection.get("status")).removeprefix("blocked_")
        return _blocked_and_write(
            root=root,
            reason=reason,
            selection=selection,
            registry_total=registry_total,
            preconditions=preconditions,
            started=started,
        )
    preconditions["offline_arcade_available"] = offline_arcade_available()
    if not preconditions["offline_arcade_available"]:
        return _blocked_and_write(
            root=root,
            reason="offline_arcade_missing",
            selection=selection,
            registry_total=registry_total,
            preconditions=preconditions,
            started=started,
        )
    live_attempt = run_live_agent_attempt(root=root, selection=selection, budget=budget)
    artifact = build_artifact(
        selection=selection,
        registry_total=registry_total,
        live_attempt=live_attempt,
        preconditions_checked=preconditions,
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
