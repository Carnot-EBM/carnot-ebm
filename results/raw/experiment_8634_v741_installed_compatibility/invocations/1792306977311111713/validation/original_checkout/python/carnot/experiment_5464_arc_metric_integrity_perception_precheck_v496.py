"""Experiment 5464: ARC metric-integrity and perception precheck.

Spec refs: REQ-ARC-FCP-5464, SCENARIO-ARC-FCP-5464.

This module deliberately does not try to solve a level. It checks that the next
ARC live-path attempt starts from a clean metric surface: duplicate-depth solves
are rejected, source/off-path provenance is rejected, one-step null-coordinate
contamination is detected, and the connected-component salience path can emit
receipts from live-agent reachable code.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
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
EXPERIMENT_ID = 5464
EXPERIMENT = "experiment_5464_arc_metric_integrity_perception_precheck_v496"
MILESTONE = "2026.07.496"
RESULT_RELATIVE_PATH = "results/experiment_5464_arc_metric_integrity_perception_precheck_v496.json"
PERCEPTION_RECEIPTS_RELATIVE_PATH = (
    "results/experiment_5464_arc_perception_feature_receipts_v496.json"
)
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5464", "SCENARIO-ARC-FCP-5464"]
INFERENCE_SUBSTRATE = "live_path_precheck_no_solve_claim"
SOLVE_PROVENANCE = "live_agent_self_discovery"
DEFAULT_RECENT_NO_BANK_TARGETS = ("re86:L3", "lf52:L3", "cn04:L4", "ka59:L2")
DEFAULT_TARGET_PRIORITY = (
    "re86",
    "lf52",
    "cn04",
    "ka59",
    "bp35",
    "sb26",
    "g50t",
    "dc22",
    "sp80",
    "su15",
    "m0r0",
    "sk48",
    "vc33",
    "r11l",
    "s5i5",
    "cd82",
    "ar25",
    "ft09",
    "ls20",
    "wa30",
    "lp85",
    "sc25",
    "tn36",
    "tr87",
    "tu93",
)
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5464_arc_metric_integrity_perception_precheck_v496.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5464_arc_metric_integrity_perception_precheck_v496.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5464_arc_metric_integrity_perception_precheck_v496.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "registry_precheck_performed": {
        "principle": "bare bool proving registry/public-game precheck ran before Exp5465 target selection."
    },
    "reproduced_total_levels_before": {
        "principle": "authoritative registry total before this no-solve precheck."
    },
    "duplicate_solve_rejected": {
        "principle": "bare bool: duplicate-depth solve claims fail closed and do not increment metrics."
    },
    "off_path_solve_rejected": {
        "principle": "bare bool: source-derived, replay-only, or outer-loop provenance cannot receive live solve credit."
    },
    "null_coordinate_exploit_valid": {
        "principle": "bare bool: true only if a banked reproduced level is validly explained by the null-coordinate exploit; metric-ready requires false."
    },
    "perception_feature_receipts_path": {
        "principle": "path to JSON receipts for connected components, color blobs, changed pixels, salience tiers, and action-effect observations."
    },
    "target_shortlist": {
        "principle": "Exp5465 candidates selected after avoiding already reached levels and recent duplicate no-bank lanes."
    },
    "recent_no_bank_targets_avoided_or_justified": {
        "principle": "auditable list of recent no-bank targets avoided or any explicit justification for including one."
    },
    "arc_metric_integrity_ready": {
        "principle": "true only when duplicate, provenance, null-coordinate, perception, and target-rotation gates are all clean."
    },
    "inference_substrate": {"principle": "must equal live_path_precheck_no_solve_claim."},
    "honest_verdict": {
        "principle": "one-line verdict starting complete:, honest_null:, or blocked: with no level-solve claim."
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


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


def _target_marker(game: str, level: int) -> str:
    return f"{game}:{_level_label(level)}"


def load_registry(root: Path = REPO) -> dict[str, Any]:
    path = root / REGISTRY_RELATIVE_PATH
    if not path.exists():  # pragma: no cover - defensive missing-repo path
        return {"reproducible_total_levels": 0, "games": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_arc_loop_artifacts(root: Path = REPO) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "results").glob("arc_loop_solve_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt fixture guard
            continue
        game = str(data.get("game") or path.stem.replace("arc_loop_solve_", ""))
        rows[game] = data
    return rows


def registry_precheck(registry: Mapping[str, Any]) -> dict[str, Any]:
    """REQ-ARC-FCP-5464: summarize the authoritative public-game metric surface."""

    rows = _registry_rows(registry)
    games = []
    for game in sorted(rows):
        row = rows[game]
        games.append(
            {
                "game": game,
                "levels_reproduced": _as_int(row.get("levels_reproduced")),
                "reproducibility": str(row.get("reproducibility") or ""),
            }
        )
    return {
        "performed": True,
        "reproduced_total_levels_before": _as_int(
            registry.get("reproducible_total_levels")
        ),
        "public_games_checked": len(games),
        "games": games,
    }


def audit_solve_claim(
    claim: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-5464: reject duplicate-depth and off-path solve credit."""

    game = str(claim.get("game") or "")
    target_level = _as_int(claim.get("target_level") or claim.get("reproduced_levels"))
    row = _registry_rows(registry).get(game) or {}
    registry_depth = _as_int(row.get("levels_reproduced"))
    runtime_trace = list(claim.get("runtime_trace") or [])
    reasons: list[str] = []
    duplicate_rejected = target_level <= registry_depth
    source_path = any(
        claim.get(flag) is True
        for flag in (
            "used_env_source",
            "read_game_source",
            "offline_ground_truth_bfs",
            "source_derived",
        )
    )
    adapter_path = claim.get("per_game_adapter_used") is True or claim.get(
        "hand_built_adapter"
    ) is True
    replay_only = claim.get("replay_only_artifact") is True and not runtime_trace
    bad_provenance = claim.get("solve_provenance") != SOLVE_PROVENANCE

    if duplicate_rejected:
        reasons.append("duplicate_depth")
    if bad_provenance:
        reasons.append("non_live_agent_provenance")
    if source_path:
        reasons.append("source_or_ground_truth_path")
    if adapter_path:
        reasons.append("per_game_adapter_path")
    if replay_only:
        reasons.append("replay_only_without_runtime_trace")
    if claim.get("offline_reproduced") is not True:
        reasons.append("not_offline_reproduced")

    off_path_rejected = bool(bad_provenance or source_path or adapter_path or replay_only)
    return {
        "game": game,
        "target_level": target_level,
        "registry_depth": registry_depth,
        "accepted": not reasons,
        "duplicate_rejected": duplicate_rejected,
        "off_path_rejected": off_path_rejected,
        "rejection_reasons": reasons,
    }


def _step_action(step: Any) -> int | None:
    if isinstance(step, Mapping):
        value = step.get("action", step.get("action_id"))
        if value is None and ("x" in step or "y" in step):
            return 6
        return _as_int(value) if value is not None else None
    return None


def _step_data(step: Any) -> Mapping[str, Any] | None:
    if not isinstance(step, Mapping):
        return None
    data = step.get("data")
    if isinstance(data, Mapping):
        return data
    if "x" in step or "y" in step:
        return step
    return None


def _solution_steps(artifact: Mapping[str, Any]) -> list[Any]:
    for key in ("solution", "actions", "plan"):
        value = artifact.get(key)
        if isinstance(value, list):
            return value
    labels = artifact.get("solution_labels")
    if not isinstance(labels, list):
        return []
    steps: list[Any] = []
    for label in labels:
        if isinstance(label, str):
            try:
                steps.append(json.loads(label))
            except json.JSONDecodeError:
                steps.append({"label": label})
        else:
            steps.append(label)
    return steps


def _is_null_coordinate_step(step: Any) -> bool:
    if _step_action(step) != 6:
        return False
    data = _step_data(step)
    return not (
        isinstance(data, Mapping)
        and data.get("x") is not None
        and data.get("y") is not None
    )


def audit_null_coordinate_exploit(
    loop_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-5464: detect one-step null-coordinate solve contamination."""

    contaminated: list[dict[str, Any]] = []
    checked = 0
    for game, artifact in sorted(loop_artifacts.items()):
        if artifact.get("offline_reproduced") is not True:
            continue
        if _as_int(artifact.get("reproduced_levels")) < 1:
            continue
        checked += 1
        steps = _solution_steps(artifact)
        if len(steps) == 1 and _is_null_coordinate_step(steps[0]):
            contaminated.append(
                {
                    "game": str(artifact.get("game") or game),
                    "reason": "one_step_action6_without_xy_coordinates",
                    "step": steps[0],
                }
            )
    return {
        "checked_reproduced_loop_artifacts": checked,
        "null_coordinate_exploit_valid": bool(contaminated),
        "contaminated_artifacts": contaminated,
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
        for y, x in zip(ys.tolist(), xs.tolist())
    ]


def build_perception_feature_receipts() -> dict[str, Any]:
    """SCENARIO-ARC-FCP-5464: exercise live-reachable perception diagnostics."""

    before, after = _diagnostic_frames()
    prior = ColorBlobSaliencePrior()
    blobs = connected_color_blobs(before, max_component_fraction=1.0)
    component_rows = [
        {
            "color": int(blob.color),
            "pixel_count": int(blob.pixel_count),
            "bbox": [int(value) for value in blob.bbox],
            "centroid_y": float(blob.centroid[0]),
            "centroid_x": float(blob.centroid[1]),
        }
        for blob in blobs
    ]
    candidates = [
        {"action": 6, "data": {"x": int(x), "y": int(y)}, "source": "color_blob_prior"}
        for x, y in prior.click_points(before, max_points=4)
    ]
    action_rows = prior.action_tier_rows(before, candidates)
    live_salience = {
        "connected_component_salience_enabled": True,
        "salience_tiers_emitted": True,
        "generation_stage_action_prioritization": True,
        "tier_rows": prior.tier_rows(before),
        "action_tier_rows": action_rows,
    }
    agent_source = REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py"
    graph_source = REPO / "python" / "carnot" / "agentic" / "arc_graph_explore.py"
    agent_text = agent_source.read_text(encoding="utf-8")
    graph_text = graph_source.read_text(encoding="utf-8")
    changed_pixels = _changed_pixel_rows(before, after)
    return {
        "spec_refs": list(SPEC_REFS),
        "live_agent_policy_reachable": (
            "ColorBlobSaliencePrior" in agent_text
            and "action_prior = ColorBlobSaliencePrior()" in agent_text
            and "prior.click_points" in graph_text
        ),
        "perception_frame_shape": [20, 20],
        "connected_component_rows": component_rows,
        "color_blob_rows": prior.tier_rows(before),
        "changed_pixel_rows": changed_pixels,
        "salience_tier_rows": live_salience.get("tier_rows", []),
        "action_tier_rows": live_salience.get("action_tier_rows", []),
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
        "first_live_candidate": dict(candidates[0]) if candidates else None,
        "live_salience_diagnostics": dict(live_salience),
    }


def write_perception_receipts(root: Path, receipts: Mapping[str, Any]) -> Path:
    path = root / PERCEPTION_RECEIPTS_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_target_shortlist(
    registry: Mapping[str, Any],
    *,
    recent_no_bank_targets: Sequence[str] = DEFAULT_RECENT_NO_BANK_TARGETS,
    priority: Sequence[str] = DEFAULT_TARGET_PRIORITY,
    limit: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """REQ-ARC-FCP-5464: select Exp5465 targets after duplicate/no-bank avoidance."""

    rows = _registry_rows(registry)
    recent = {str(item) for item in recent_no_bank_targets}
    shortlist: list[dict[str, Any]] = []
    avoided: list[dict[str, str]] = []
    for game in priority:
        row = rows.get(str(game))
        if row is None or str(row.get("reproducibility")) != "reproduced":
            continue
        current = _as_int(row.get("levels_reproduced"))
        if current < 1:
            continue
        target_level = current + 1
        marker = _target_marker(str(game), target_level)
        if marker in recent:
            avoided.append(
                {
                    "target": marker,
                    "decision": "avoided",
                    "justification": "recent bounded-budget no-bank lane; Exp5465 should avoid duplicate rerun unless a new mechanism is named.",
                }
            )
            continue
        shortlist.append(
            {
                "game": str(game),
                "target": marker,
                "target_level": target_level,
                "current_reproduced_levels": current,
                "selection_reason": "next unreached reproduced-registry frontier after recent no-bank avoidance",
            }
        )
        if len(shortlist) >= int(limit):
            break
    return shortlist, avoided


def _integrity_probes(registry: Mapping[str, Any]) -> dict[str, Any]:
    rows = _registry_rows(registry)
    first_game = next(iter(rows), "unknown")
    first_depth = _as_int((rows.get(first_game) or {}).get("levels_reproduced"))
    duplicate = audit_solve_claim(
        {
            "game": first_game,
            "target_level": first_depth,
            "offline_reproduced": True,
            "solve_provenance": SOLVE_PROVENANCE,
            "runtime_trace": [{"action": 1}],
        },
        registry,
    )
    off_path = audit_solve_claim(
        {
            "game": first_game,
            "target_level": first_depth + 1,
            "offline_reproduced": True,
            "solve_provenance": "outer_loop_re",
            "used_env_source": True,
            "offline_ground_truth_bfs": True,
            "replay_only_artifact": True,
            "runtime_trace": [],
        },
        registry,
    )
    return {
        "duplicate_probe": duplicate,
        "off_path_probe": off_path,
        "duplicate_solve_rejected": duplicate["accepted"] is False
        and duplicate["duplicate_rejected"] is True,
        "off_path_solve_rejected": off_path["accepted"] is False
        and off_path["off_path_rejected"] is True,
    }


def build_artifact(
    *,
    registry_summary: Mapping[str, Any],
    integrity_probes: Mapping[str, Any],
    null_coordinate_audit: Mapping[str, Any],
    perception_receipts_path: str,
    target_shortlist: Sequence[Mapping[str, Any]],
    recent_no_bank_targets_avoided_or_justified: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    ready = bool(
        registry_summary.get("performed") is True
        and integrity_probes.get("duplicate_solve_rejected") is True
        and integrity_probes.get("off_path_solve_rejected") is True
        and null_coordinate_audit.get("null_coordinate_exploit_valid") is False
        and perception_receipts_path
        and target_shortlist
    )
    status = "complete" if ready else "honest_null"
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5464_arc_metric_integrity_perception_precheck_v496.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "registry_precheck_performed": registry_summary.get("performed") is True,
        "reproduced_total_levels_before": _as_int(
            registry_summary.get("reproduced_total_levels_before")
        ),
        "duplicate_solve_rejected": integrity_probes.get("duplicate_solve_rejected") is True,
        "off_path_solve_rejected": integrity_probes.get("off_path_solve_rejected") is True,
        "null_coordinate_exploit_valid": null_coordinate_audit.get(
            "null_coordinate_exploit_valid"
        )
        is True,
        "perception_feature_receipts_path": perception_receipts_path,
        "target_shortlist": [dict(row) for row in target_shortlist],
        "recent_no_bank_targets_avoided_or_justified": [
            dict(row) for row in recent_no_bank_targets_avoided_or_justified
        ],
        "arc_metric_integrity_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            "complete: ARC live-path metric integrity and perception precheck ready for Exp5465; no solve claimed"
            if ready
            else "honest_null: ARC metric integrity precheck found blockers; no solve claimed"
        ),
        "registry_precheck": dict(registry_summary),
        "metric_integrity_probe_receipts": dict(integrity_probes),
        "null_coordinate_audit": dict(null_coordinate_audit),
        "preconditions_checked": dict(preconditions_checked),
        "tests_run": list(tests_run),
        "duration_s": float(duration_s),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any], *, root: Path = REPO) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    for field in (
        "registry_precheck_performed",
        "duplicate_solve_rejected",
        "off_path_solve_rejected",
        "null_coordinate_exploit_valid",
        "arc_metric_integrity_ready",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    if type(artifact.get("reproduced_total_levels_before")) is not int:
        errors.append("reproduced_total_levels_before must be bare int")
    if artifact.get("registry_precheck_performed") is not True:
        errors.append("registry_precheck_performed must be true")
    if artifact.get("duplicate_solve_rejected") is not True:
        errors.append("duplicate_solve_rejected must be true")
    if artifact.get("off_path_solve_rejected") is not True:
        errors.append("off_path_solve_rejected must be true")
    if artifact.get("arc_metric_integrity_ready") is True and artifact.get(
        "null_coordinate_exploit_valid"
    ) is not False:
        errors.append("arc_metric_integrity_ready requires null_coordinate_exploit_valid false")
    path_value = artifact.get("perception_feature_receipts_path")
    if not isinstance(path_value, str) or not path_value:
        errors.append("perception_feature_receipts_path must be non-empty string")
    elif not (root / path_value).exists():
        errors.append("perception_feature_receipts_path must exist")
    if not isinstance(artifact.get("target_shortlist"), list) or not artifact.get(
        "target_shortlist"
    ):
        errors.append("target_shortlist must be a non-empty list")
    if not isinstance(
        artifact.get("recent_no_bank_targets_avoided_or_justified"), list
    ) or not artifact.get("recent_no_bank_targets_avoided_or_justified"):
        errors.append("recent_no_bank_targets_avoided_or_justified must be a non-empty list")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "honest_null:", "blocked:")):
        errors.append("honest_verdict must start with complete:, honest_null:, or blocked:")
    return errors


def validate_artifact(artifact: Mapping[str, Any], *, root: Path = REPO) -> None:
    errors = artifact_schema_errors(artifact, root=root)
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
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "spec_has_req_5464": (
            "REQ-ARC-FCP-5464" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "no_registry_write": True,
        "no_solve_claim": True,
    }
    registry = load_registry(root)
    loop_artifacts = load_arc_loop_artifacts(root)
    summary = registry_precheck(registry)
    probes = _integrity_probes(registry)
    null_audit = audit_null_coordinate_exploit(loop_artifacts)
    receipts = build_perception_feature_receipts()
    receipts_path = write_perception_receipts(root, receipts)
    shortlist, avoided = build_target_shortlist(registry)
    artifact = build_artifact(
        registry_summary=summary,
        integrity_probes=probes,
        null_coordinate_audit=null_audit,
        perception_receipts_path=str(receipts_path.relative_to(root)),
        target_shortlist=shortlist,
        recent_no_bank_targets_avoided_or_justified=avoided,
        preconditions_checked=preconditions,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(artifact, root=root)
    _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
