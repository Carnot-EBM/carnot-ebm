"""Task-aware transition-admission energy for ARC live-path measurements.

The helper is intentionally small: it calibrates an admission threshold from
other games' live transition rows and scores held rows after the agent has
observed the action outcome. It does not plan, solve, read adapters, or read
game source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any


CALIBRATION_MODULE_ID = "carnot.agentic.arc_task_aware_energy.v1"
GLOBAL_SCORE_NAME = "global_existing_admit_all"
TASK_AWARE_SCORE_NAME = "task_aware_training_game_change_threshold"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    stable = dict(manifest)
    stable["manifest_hash"] = ""
    return _sha256_json(stable)


def global_freeze_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "calibration_module": CALIBRATION_MODULE_ID,
        "score_name": GLOBAL_SCORE_NAME,
        "decision_rule": "existing_live_default_admit_observed_transition",
        "threshold": 0,
        "abstention_policy": "none",
        "training_games": [],
        "training_row_count": 0,
        "held_row_count_used_for_fit": 0,
        "hand_calibrated_per_game": False,
        "uses_registry_gotchas": False,
        "uses_game_source": False,
        "uses_offline_bfs": False,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def fit_task_aware_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    held_game: str,
) -> dict[str, Any]:
    training = [dict(row) for row in rows if str(row.get("game")) != str(held_game)]
    changed = [int(row.get("changed_cell_count") or 0) for row in training]
    positive = [value for value in changed if value > 0]
    threshold = max(1, min(positive)) if positive else 1
    manifest: dict[str, Any] = {
        "calibration_module": CALIBRATION_MODULE_ID,
        "score_name": TASK_AWARE_SCORE_NAME,
        "held_game": str(held_game),
        "decision_rule": "admit_when_observed_changed_cells_meet_training_floor_else_abstain",
        "min_changed_cells": int(threshold),
        "abstention_policy": "abstain_when_observed_changed_cells_below_training_floor",
        "training_games": sorted({str(row.get("game")) for row in training}),
        "training_row_count": len(training),
        "training_changed_row_count": len(positive),
        "training_changed_rate": (len(positive) / len(training)) if training else 0.0,
        "held_row_count_used_for_fit": 0,
        "hand_calibrated_per_game": False,
        "uses_registry_gotchas": False,
        "uses_game_source": False,
        "uses_offline_bfs": False,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def score_transition(
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    arm: str,
) -> dict[str, Any]:
    changed_cells = int(row.get("changed_cell_count") or 0)
    frame_changed = bool(row.get("frame_changed") or changed_cells > 0)
    if arm == "global":
        admitted = True
        abstained = False
        score = 1.0
        confidence = 1.0
        reason = "existing_global_default_admit"
    elif arm == "task_aware":
        threshold = max(1, int(manifest.get("min_changed_cells") or 1))
        admitted = changed_cells >= threshold
        abstained = not admitted
        score = changed_cells / threshold
        confidence = min(1.0, abs(changed_cells - threshold) / threshold if changed_cells else 1.0)
        reason = "training_game_threshold_admit" if admitted else "training_game_threshold_abstain"
    else:
        raise ValueError(f"unknown arm: {arm}")
    return {
        "row_id": str(row.get("row_id")),
        "game": str(row.get("game")),
        "seed": int(row.get("seed") or 0),
        "action_index": int(row.get("action_index") or 0),
        "arm": arm,
        "score": round(float(score), 6),
        "confidence": round(float(confidence), 6),
        "triggered": True,
        "admitted": bool(admitted),
        "abstained": bool(abstained),
        "frame_changed": frame_changed,
        "changed_cell_count": changed_cells,
        "false_confident_admission": bool(admitted and not frame_changed and confidence >= 0.9),
        "safe_abstention": bool(abstained and not frame_changed),
        "decision_reason": reason,
    }


def live_entrypoint_reachability_witness() -> dict[str, str]:
    return {
        "calibration_module": CALIBRATION_MODULE_ID,
        "score_name": TASK_AWARE_SCORE_NAME,
    }
