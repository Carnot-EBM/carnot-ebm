"""Exp 5176: B1/B2-gated deepen live level-up attempt.

Spec refs: REQ-REPORT-5176,
SCENARIO-REPORT-5176-BLOCKED-NO-VALIDATED-LEVER,
SCENARIO-REPORT-5176-VALIDATED-LEVER-SELECTION.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPERIMENT = "experiment_5176_deepen_live_levelup_attempt_v474"
SCHEMA = "carnot.experiment_5176_deepen_live_levelup_attempt_v474.v1"
RESULT_RELATIVE_PATH = "results/experiment_5176_deepen_live_levelup_attempt_v474.json"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
DEEPENED_BUT_STUCK = ("ar25", "bp35", "cd82", "cn04", "dc22", "ft09")
SPEC_REFS = [
    "REQ-REPORT-5176",
    "SCENARIO-REPORT-5176-BLOCKED-NO-VALIDATED-LEVER",
    "SCENARIO-REPORT-5176-VALIDATED-LEVER-SELECTION",
]
MAP_NEXT_DIRECTION = (
    "MAP (arXiv:2605.13037) map-then-act / hierarchical pre-search before "
    "flat frontier enumeration."
)
FIELD_PRINCIPLES = {
    "lever_used": "one of exp5174 / exp5175 / both / none_available, determined from actual upstream artifacts.",
    "target_games": "list of {game, level_before, level_attempted} from an immediate registry precheck.",
    "levels_banked": "Only reproduce()-confirmed levels count.",
    "reproducible_levels_delta": "the load-bearing ARC metric; zero must be reported plainly.",
    "live_path_reachable": "true only after the live import closure/orphan lint is clean.",
    "solve_provenance": "live_agent_self_discovery only if the live E3AgentPolicy path itself made the discovery at runtime.",
    "next_direction_if_null": "zero-bank artifacts must explicitly name MAP (arXiv:2605.13037) or another evidenced .475 direction.",
    "verifier_is_oracle": "false for this attempt; reproduction is the executable gate, not a claimed verifier win.",
    "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm unless a live LLM is actually invoked.",
    "honest_verdict": "Must start with complete:/complete_/success:/success_ and state plainly how many levels were banked.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def _json_checksum(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _json_checksum(payload)


def load_registry_levels(registry_text: str) -> dict[str, int]:
    loaded = yaml.safe_load(registry_text) or {}
    rows = loaded.get("games") or []
    return {
        str(row["game"]): int(row.get("levels_reproduced", 0) or 0)
        for row in rows
        if isinstance(row, Mapping) and row.get("game") in DEEPENED_BUT_STUCK
    }


def _exp5174_has_fixable_gap(exp5174: Mapping[str, Any]) -> bool:
    recommendation = exp5174.get("gap_status_recommendation") or {}
    value = str(recommendation.get("value", "") if isinstance(recommendation, Mapping) else "")
    scope = str(recommendation.get("new_scope", "") if isinstance(recommendation, Mapping) else "")
    combined = f"{value} {scope}".lower()
    return "fixable" in combined and "wiring" in combined and "re-scoped" not in combined


def _exp5175_banked_levels(exp5175: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "game": str(row.get("game")),
            "new_level": int(row.get("new_level", 0) or 0),
            "offline_reproduced": True,
            "reproducibility_checksum": str(row.get("reproducibility_checksum", "")),
        }
        for row in exp5175.get("levels_banked", []) or []
        if isinstance(row, Mapping) and row.get("offline_reproduced") is True
    ]


def _exp5175_has_state_win(exp5175: Mapping[str, Any]) -> bool:
    reductions = exp5175.get("states_expanded_reduction_pct") or {}
    return any(float(value or 0.0) > 0.0 for value in reductions.values())


def determine_lever(exp5174: Mapping[str, Any], exp5175: Mapping[str, Any]) -> str:
    has_5174 = _exp5174_has_fixable_gap(exp5174)
    has_5175 = bool(_exp5175_banked_levels(exp5175) or _exp5175_has_state_win(exp5175))
    if has_5174 and has_5175:
        return "both"
    if has_5174:
        return "exp5174"
    if has_5175:
        return "exp5175"
    return "none_available"


def select_target_games(
    registry_levels: Mapping[str, int],
    exp5175: Mapping[str, Any],
    *,
    limit: int = 3,
) -> list[dict[str, int | str]]:
    move_pruned_edges = exp5175.get("move_pruned_edges") or {}
    exp5175_targets = set(exp5175.get("target_games") or [])
    negative_control = str(exp5175.get("negative_control_game", ""))
    rows = []
    for game in DEEPENED_BUT_STUCK:
        level = int(registry_levels.get(game, 0) or 0)
        if level <= 0:
            continue
        pruned = int(move_pruned_edges.get(game, 0) or 0)
        direct_signal = game in move_pruned_edges
        target_priority = 1 if game in exp5175_targets else 0
        control_penalty = 1 if game == negative_control else 0
        rows.append((direct_signal, target_priority, pruned, -control_penalty, game, level))
    direct_rows = [row for row in rows if row[0]]
    selected = sorted(direct_rows or rows, reverse=True)[: int(limit)]
    return [
        {"game": game, "level_before": level, "level_attempted": level + 1}
        for _direct_signal, _target_priority, _pruned, _control, game, level in selected
    ]


def _verdict(delta: int, lever_used: str) -> str:
    if delta == 0:
        return "complete_blocked_no_validated_lever_from_b1_b2_zero_levels_banked"
    if delta == 1:
        return f"success_one_level_banked_with_{lever_used}"
    return f"success_{delta}_level_banked_with_{lever_used}"


def build_artifact(
    *,
    exp5174: Mapping[str, Any],
    exp5175: Mapping[str, Any],
    registry_levels: Mapping[str, int],
    live_path_reachable: bool,
    arc_orphan_solver_lint: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    lever_used = determine_lever(exp5174, exp5175)
    levels_banked = _exp5175_banked_levels(exp5175) if lever_used in {"exp5175", "both"} else []
    delta = len(levels_banked)
    blocked = lever_used == "none_available"
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "lever_used": lever_used,
        "target_games": select_target_games(registry_levels, exp5175, limit=3),
        "levels_banked": levels_banked,
        "reproducible_levels_delta": delta,
        "live_path_reachable": bool(live_path_reachable),
        "solve_provenance": (
            "not_applicable_blocked_no_runtime_discovery" if blocked else "development_proxy"
        ),
        "next_direction_if_null": MAP_NEXT_DIRECTION if delta == 0 else "",
        "verifier_is_oracle": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _verdict(delta, lever_used),
        "preconditions_checked": {
            "exp5174_read_directly": True,
            "exp5175_read_directly": True,
            "registry_prechecked": True,
            "stopped_before_levelup_attempt": blocked,
        },
        "upstream_lever_assessment": {
            "exp5174_fixable_live_wiring_gap": _exp5174_has_fixable_gap(exp5174),
            "exp5175_reproduce_gated_levels": len(_exp5175_banked_levels(exp5175)),
            "exp5175_state_expansion_win": _exp5175_has_state_win(exp5175),
            "exp5175_gap4891_status_recommendation": str(
                exp5175.get("gap4891_status_recommendation", "")
            ),
        },
        "arc_orphan_solver_lint": dict(arc_orphan_solver_lint),
        "in_scope_candidates": list(DEEPENED_BUT_STUCK),
        "duration_s": round(float(duration_s), 2),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
