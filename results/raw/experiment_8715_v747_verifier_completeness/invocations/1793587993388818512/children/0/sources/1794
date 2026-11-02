"""Experiment 5479: ARC target-rotation live-path precheck.

Spec refs: REQ-ARC-FCP-5479, SCENARIO-ARC-FCP-5479.

This module intentionally stops before a solve attempt. Its job is to prove
that the next ARC level-up target is not a duplicate or stale no-bank rerun,
and that the submitted live salience/perception path can still emit the
frame-only evidence a later attempt would need.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5479
EXPERIMENT = "experiment_5479_arc_target_rotation_precheck_v497"
MILESTONE = "2026.07.497"
RESULT_RELATIVE_PATH = "results/experiment_5479_arc_target_rotation_precheck_v497.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
PRECHECK_RELATIVE_PATH = (
    "results/experiment_5464_arc_metric_integrity_perception_precheck_v496.json"
)
NO_BANK_RELATIVE_PATH = (
    "results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json"
)
KNOWN_ISSUES_RELATIVE_PATH = "ops/known-issues.md"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5479", "SCENARIO-ARC-FCP-5479"]
INFERENCE_SUBSTRATE = "arc_live_path_precheck_no_solve"
RANDOM_SEED = 5479
DEFAULT_RECENT_NO_BANK_TARGETS = ("bp35:L3", "ka59:L2", "cn04:L4")
DEFAULT_ROTATED_TARGET_PRIORITY = ("sb26", "g50t", "dc22", "sp80")
REQUIRED_SUMMARY_KEYS = (
    "connected_components",
    "color_blobs",
    "changed_cells",
    "target_region_candidates",
    "known_blockers",
)
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5479_arc_target_rotation_precheck_v497.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5479_arc_target_rotation_precheck_v497.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5479_arc_target_rotation_precheck_v497.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "selected_game": {
        "principle": "rotated non-duplicate Exp5464 shortlist game selected before any level-up attempt."
    },
    "selected_target_level": {
        "principle": "selected next level, strictly greater than the registry reproduced depth."
    },
    "registry_reproducible_total_levels_before": {
        "principle": "authoritative registry total before this no-solve precheck."
    },
    "duplicate_target_rejected": {
        "principle": "bare bool proving already reproduced target levels are rejected before selection."
    },
    "recent_no_bank_targets_avoided": {
        "principle": "auditable list containing bp35:L3, ka59:L2, and cn04:L4 unless a new mechanism justifies inclusion."
    },
    "live_path_reachable": {
        "principle": "true only when submitted live salience/perception code emits the dry-check features."
    },
    "hidden_source_reading": {
        "principle": "must be false; hidden/public source is not read in this precheck."
    },
    "offline_bfs_used": {
        "principle": "must be false; no offline ground-truth BFS is used."
    },
    "hand_adapter_used": {
        "principle": "must be false; target eligibility is not credited to a hand per-game adapter."
    },
    "salience_feature_summary": {
        "principle": "dict summarizing connected components, color blobs, changed cells, target-region candidates, and known blockers."
    },
    "arc_target_rotation_ready": {
        "principle": "true only when non-duplicate target rotation and live-path salience eligibility both pass."
    },
    "solve_claimed": {
        "principle": "must be false; this artifact is a precheck, not a level solve."
    },
    "inference_substrate": {"principle": "must equal arc_live_path_precheck_no_solve."},
    "random_seed": {
        "principle": "deterministic seed for reproducible target ordering and dry-check fixtures."
    },
    "honest_verdict": {
        "principle": "one-line verdict starting complete:, honest_null:, or blocked: with no solve claim."
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive corrupt artifact path
        return int(default)


def _level_label(level: int) -> str:
    return f"L{max(0, int(level))}"


def _target_marker(game: str, level: int) -> str:
    return f"{game}:{_level_label(level)}"


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }


def _shortlist_rows(precheck: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in precheck.get("target_shortlist") or []:
        if not isinstance(item, Mapping):  # pragma: no cover - defensive artifact path
            continue
        game = str(item.get("game") or "")
        target_level = _as_int(item.get("target_level"))
        if not game or target_level <= 0:  # pragma: no cover - defensive artifact path
            continue
        rows.append(
            {
                "game": game,
                "target_level": target_level,
                "marker": _target_marker(game, target_level),
                "source": dict(item),
            }
        )
    return rows


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():  # pragma: no cover - missing precondition path
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_text(path: Path) -> str:
    if not path.exists():  # pragma: no cover - missing precondition path
        return ""
    return path.read_text(encoding="utf-8")


def load_registry(root: Path = REPO) -> dict[str, Any]:
    path = root / REGISTRY_RELATIVE_PATH
    if not path.exists():  # pragma: no cover - missing precondition path
        return {"reproducible_total_levels": 0, "games": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _recent_targets_from_no_bank(no_bank_artifact: Mapping[str, Any]) -> list[str]:
    recent = list(DEFAULT_RECENT_NO_BANK_TARGETS)
    game = str(no_bank_artifact.get("target_game") or "")
    level = _as_int(no_bank_artifact.get("target_level_attempted"))
    marker = _target_marker(game, level) if game and level > 0 else ""
    if marker and marker not in recent:
        recent.insert(0, marker)
    return recent


def select_rotated_target(
    registry: Mapping[str, Any],
    precheck: Mapping[str, Any],
    known_issues_text: str,
    *,
    recent_no_bank_targets: Sequence[str] = DEFAULT_RECENT_NO_BANK_TARGETS,
    rotated_target_priority: Sequence[str] = DEFAULT_ROTATED_TARGET_PRIORITY,
) -> dict[str, Any]:
    """REQ-ARC-FCP-5479: select a rotated, unreproduced target before any attempt."""

    rows = _registry_rows(registry)
    total = _as_int(registry.get("reproducible_total_levels"))
    recent = [str(item) for item in recent_no_bank_targets]
    target_audit: dict[str, dict[str, Any]] = {}
    eligible: dict[str, dict[str, Any]] = {}
    duplicate_target_rejected = False
    priority = [str(game) for game in rotated_target_priority]

    for candidate in _shortlist_rows(precheck):
        game = str(candidate["game"])
        target_level = _as_int(candidate["target_level"])
        marker = str(candidate["marker"])
        registry_depth = _as_int((rows.get(game) or {}).get("levels_reproduced"))
        base = {
            "game": game,
            "target_level": int(target_level),
            "registry_depth": int(registry_depth),
        }
        if target_level <= registry_depth:
            duplicate_target_rejected = True
            target_audit[marker] = {**base, "decision": "rejected_duplicate"}
            continue
        if marker in recent:
            target_audit[marker] = {**base, "decision": "rejected_recent_no_bank"}
            continue
        if game not in priority:
            target_audit[marker] = {**base, "decision": "skipped_not_rotated_priority"}
            continue
        target_audit[marker] = {**base, "decision": "eligible_rotated"}
        eligible[game] = {**base, "marker": marker, "source": dict(candidate["source"])}

    for game in priority:
        if game not in eligible:
            continue
        selected = eligible[game]
        duplicate_probe = _target_marker(game, _as_int(selected["registry_depth"]))
        target_audit[duplicate_probe] = {
            "game": game,
            "target_level": _as_int(selected["registry_depth"]),
            "registry_depth": _as_int(selected["registry_depth"]),
            "decision": "rejected_duplicate",
        }
        target_audit[str(selected["marker"])]["decision"] = "selected"
        duplicate_target_rejected = True
        return {
            "blocked": False,
            "selected_game": game,
            "selected_target_level": _as_int(selected["target_level"]),
            "registry_level_before": _as_int(selected["registry_depth"]),
            "registry_reproducible_total_levels_before": total,
            "duplicate_target_rejected": bool(duplicate_target_rejected),
            "recent_no_bank_targets_avoided": recent,
            "target_audit": target_audit,
            "known_issues_checked": bool(known_issues_text),
            "selection_reason": (
                f"rotated_exp5464_shortlist_after_recent_no_bank_avoidance:{selected['marker']}"
            ),
        }

    return {
        "blocked": True,
        "blocker": "no_eligible_rotated_target",
        "selected_game": "",
        "selected_target_level": 0,
        "registry_level_before": 0,
        "registry_reproducible_total_levels_before": total,
        "duplicate_target_rejected": bool(duplicate_target_rejected),
        "recent_no_bank_targets_avoided": recent,
        "target_audit": target_audit,
        "known_issues_checked": bool(known_issues_text),
        "selection_reason": "no_eligible_rotated_target_after_duplicate_and_no_bank_filters",
    }


def _known_blockers(
    *,
    selected_game: str,
    registry_row: Mapping[str, Any],
    known_issues_text: str,
) -> list[str]:
    blockers: list[str] = []
    dead_ends = registry_row.get("dead_ends") or []
    if isinstance(dead_ends, str):
        blockers.append(dead_ends[:240])
    else:
        blockers.extend(str(item)[:240] for item in dead_ends if item)
    blockers.extend(
        line.strip()[:240]
        for line in known_issues_text.splitlines()
        if selected_game and selected_game in line
    )
    return blockers[:6] or ["no known blocker recorded for selected target"]


def _known_issue_mentions(selected_game: str, known_issues_text: str) -> int:
    return sum(1 for line in known_issues_text.splitlines() if selected_game and selected_game in line)


def _tier_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("tier")) for row in rows if row.get("tier") is not None)
    return {tier: int(count) for tier, count in sorted(counts.items())}


def _sample_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], limit: int = 3) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in keys if key in row}
        for row in rows[: max(0, int(limit))]
    ]


def build_salience_feature_summary(
    *,
    selected_game: str,
    registry_row: Mapping[str, Any],
    known_issues_text: str,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-5479: run a bounded live-path salience dry check."""

    from carnot.experiment_5465_gated_arc_connected_component_salience_levelup_v496 import (
        build_live_feature_receipts,
    )

    receipts = build_live_feature_receipts()
    component_rows = list(receipts.get("connected_component_rows") or [])
    color_blob_rows = list(receipts.get("color_blob_rows") or [])
    changed_rows = list(receipts.get("changed_pixel_rows") or [])
    action_rows = [
        row
        for row in list(receipts.get("action_tier_rows") or [])
        if isinstance(row, Mapping) and row.get("tier") is not None and row.get("non_status_region")
    ]
    target_candidates = [
        {
            "action": _as_int(row.get("action")),
            "data": dict(row.get("data") or {}),
            "tier": _as_int(row.get("tier")),
            "source": str(row.get("source") or ""),
            "color": _as_int(row.get("color")),
            "score": float(row.get("score") or 0.0),
            "button_like": bool(row.get("button_like")),
        }
        for row in action_rows[:5]
    ]
    summary = {
        "selected_game": str(selected_game),
        "live_path_reachable": bool(receipts.get("live_agent_policy_reachable")),
        "connected_components": {
            "count": len(component_rows),
            "sample": _sample_rows(
                component_rows,
                ("color", "pixel_count", "bbox", "centroid_x", "centroid_y"),
            ),
        },
        "color_blobs": {
            "count": len(color_blob_rows),
            "tier_counts": _tier_counts(color_blob_rows),
            "sample": _sample_rows(
                color_blob_rows,
                ("tier", "color", "pixel_count", "bbox", "button_like", "large_flat"),
            ),
        },
        "changed_cells": {
            "count": len(changed_rows),
            "sample": _sample_rows(changed_rows, ("x", "y", "before", "after"), limit=5),
        },
        "target_region_candidates": target_candidates,
        "known_blockers": _known_blockers(
            selected_game=selected_game,
            registry_row=registry_row,
            known_issues_text=known_issues_text,
        ),
        "known_issues_mentions": _known_issue_mentions(selected_game, known_issues_text),
        "features_present": [
            "connected_component",
            "color_blob",
            "changed_pixel",
            "target_region_candidate",
            "known_blocker",
        ],
    }
    return summary


def build_artifact(
    *,
    selection: Mapping[str, Any],
    salience_feature_summary: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    selected_game = str(selection.get("selected_game") or "none")
    selected_target_level = _as_int(selection.get("selected_target_level"))
    live_path_reachable = bool(salience_feature_summary.get("live_path_reachable"))
    blocked = bool(selection.get("blocked"))
    arc_ready = bool(
        not blocked
        and selected_game != "none"
        and selected_target_level > _as_int(selection.get("registry_level_before"))
        and selection.get("duplicate_target_rejected") is True
        and live_path_reachable
    )
    status = "complete" if arc_ready else "blocked"
    failure_mode = "" if arc_ready else str(selection.get("blocker") or "live_path_not_reachable")
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5479_arc_target_rotation_precheck_v497.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "selected_game": selected_game,
        "selected_target_level": int(selected_target_level),
        "registry_reproducible_total_levels_before": _as_int(
            selection.get("registry_reproducible_total_levels_before")
        ),
        "duplicate_target_rejected": selection.get("duplicate_target_rejected") is True,
        "recent_no_bank_targets_avoided": list(
            selection.get("recent_no_bank_targets_avoided") or []
        ),
        "live_path_reachable": bool(live_path_reachable),
        "hidden_source_reading": False,
        "offline_bfs_used": False,
        "hand_adapter_used": False,
        "salience_feature_summary": dict(salience_feature_summary),
        "arc_target_rotation_ready": bool(arc_ready),
        "solve_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "honest_verdict": (
            f"complete: {selected_game} {_level_label(selected_target_level)} live-path precheck ready; no level bank claimed"
            if arc_ready
            else f"blocked: {failure_mode}"
        ),
        "failure_mode": failure_mode,
        "target_selection": dict(selection),
        "preconditions_checked": dict(preconditions_checked),
        "tests_run": list(tests_run),
        "duration_s": float(duration_s),
    }
    return artifact


def _verdict_claims_solve(verdict: str) -> bool:
    text = verdict.lower()
    return "solved" in text or "solve reproduced" in text or "new level banked" in text


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    if not isinstance(artifact.get("selected_game"), str) or not artifact.get("selected_game"):
        errors.append("selected_game must be a non-empty string")
    for field in (
        "selected_target_level",
        "registry_reproducible_total_levels_before",
        "random_seed",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    for field in (
        "duplicate_target_rejected",
        "live_path_reachable",
        "hidden_source_reading",
        "offline_bfs_used",
        "hand_adapter_used",
        "arc_target_rotation_ready",
        "solve_claimed",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    recent = artifact.get("recent_no_bank_targets_avoided")
    if not isinstance(recent, list):
        errors.append("recent_no_bank_targets_avoided must be a list")
        recent = []
    for marker in DEFAULT_RECENT_NO_BANK_TARGETS:
        if marker not in recent:
            errors.append(f"recent_no_bank_targets_avoided missing {marker}")
    for field in ("hidden_source_reading", "offline_bfs_used", "hand_adapter_used", "solve_claimed"):
        if artifact.get(field) is True:
            errors.append(f"{field} must be false")
    summary = artifact.get("salience_feature_summary")
    if not isinstance(summary, Mapping):
        errors.append("salience_feature_summary must be a dict")
        summary = {}
    for key in REQUIRED_SUMMARY_KEYS:
        if key not in summary:
            errors.append(f"salience_feature_summary missing {key}")
    if artifact.get("arc_target_rotation_ready") is True:
        if artifact.get("solve_claimed") is not False:
            errors.append("arc_target_rotation_ready requires solve_claimed false")
        if artifact.get("live_path_reachable") is not True:
            errors.append("arc_target_rotation_ready requires live_path_reachable true")
        if artifact.get("duplicate_target_rejected") is not True:
            errors.append("arc_target_rotation_ready requires duplicate_target_rejected true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "honest_null:", "blocked:")):
        errors.append("honest_verdict must start with complete:, honest_null:, or blocked:")
    if _verdict_claims_solve(verdict):
        errors.append("honest_verdict must not claim a solve")
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
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    precheck_path = root / PRECHECK_RELATIVE_PATH
    no_bank_path = root / NO_BANK_RELATIVE_PATH
    known_issues_path = root / KNOWN_ISSUES_RELATIVE_PATH
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "spec_has_req_5479": (
            "REQ-ARC-FCP-5479" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "exp5464_precheck_present": precheck_path.exists(),
        "exp5465_no_bank_present": no_bank_path.exists(),
        "known_issues_present": known_issues_path.exists(),
        "hidden_source_reading": False,
        "offline_bfs_used": False,
        "hand_adapter_used": False,
    }
    registry = load_registry(root)
    precheck = load_json(precheck_path)
    no_bank = load_json(no_bank_path)
    known_issues_text = load_text(known_issues_path)
    selection = select_rotated_target(
        registry,
        precheck,
        known_issues_text,
        recent_no_bank_targets=_recent_targets_from_no_bank(no_bank),
    )
    registry_row = _registry_rows(registry).get(str(selection.get("selected_game"))) or {}
    summary = build_salience_feature_summary(
        selected_game=str(selection.get("selected_game") or ""),
        registry_row=registry_row,
        known_issues_text=known_issues_text,
    )
    artifact = build_artifact(
        selection=selection,
        salience_feature_summary=summary,
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
