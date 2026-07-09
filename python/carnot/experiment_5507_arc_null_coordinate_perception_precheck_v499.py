"""Exp5507: ARC null-coordinate and perception-grounding target precheck.

Spec refs: REQ-ARC-FCP-5507, SCENARIO-ARC-FCP-5507.

This module is intentionally an aggregation step, not a solver. It reads the
live ARC registry and recent ARC artifacts, then decides whether a later live
attempt has a non-duplicate target and a materially changed mechanism. The
important distinction is why the prior dc22 null happened: zero-change clicks
at recorded coordinates are valid observations to learn from, while missing or
fabricated coordinates would make the metric surface untrustworthy.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5507
EXPERIMENT = "experiment_5507_arc_null_coordinate_perception_precheck_v499"
MILESTONE = "2026.07.499"
RESULT_RELATIVE_PATH = "results/experiment_5507_arc_null_coordinate_perception_precheck_v499.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
KNOWN_ISSUES_RELATIVE_PATH = "ops/known-issues.md"
LEVERS_NOTE_RELATIVE_PATH = "docs/research-notes/arc-agi3-levers-tried-x-verdict-2026-06-25.md"
EXP5464_RELATIVE_PATH = "results/experiment_5464_arc_metric_integrity_perception_precheck_v496.json"
EXP5465_RELATIVE_PATH = "results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json"
EXP5480_RELATIVE_PATH = "results/experiment_5480_arc_live_salience_levelup_v497.json"
EXP5493_RELATIVE_PATH = "results/experiment_5493_arc_trajectory_target_precheck_v498.json"
EXP5494_RELATIVE_PATH = "results/experiment_5494_arc_live_trajectory_levelup_v498.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5507", "SCENARIO-ARC-FCP-5507"]
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CHANGED_MECHANISM = (
    "perception-grounded connected-component/color-blob segmentation plus "
    "salience-tiered action generation with trajectory taxonomy receipts"
)
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5507_arc_null_coordinate_perception_precheck_v499.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5507_arc_null_coordinate_perception_precheck_v499.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5507_arc_null_coordinate_perception_precheck_v499.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "registry_path": "must equal ops/arc_solve_registry.yaml.",
    "reproducible_total_levels_before": "authoritative registry total before this no-solve target precheck.",
    "duplicate_targets_rejected": "auditable list of requested target levels rejected because the registry already reproduces them.",
    "recent_no_bank_targets_rejected": "auditable list of recent same-target/same-mechanism no-bank reruns rejected before selection.",
    "null_coordinate_audit": "dict classifying null/missing/no-op coordinate evidence as valid game action, metric artifact, or contamination.",
    "perception_grounding_findings": "list of upstream live-observation findings exposed by connected components, color blobs, salience tiers, changed pixels, or action-effect asymmetry.",
    "selected_game": "selected game id, or empty string when blocked.",
    "selected_level": "selected level label such as L3, or empty string when blocked.",
    "selected_mechanism": "changed live-path perception-generation mechanism, not a same-mechanism rerun.",
    "levelup_attempt_ready": "bare bool true only when the selected level is non-duplicate, same-mechanism reruns are rejected, null-coordinate audit is clean, and perception findings are present.",
    "solve_claimed": "must be false; this artifact is a precheck and cannot bank a level.",
    "inference_substrate": "must equal aggregation_from_upstream_artifacts.",
    "honest_verdict": "terminal status starts with complete: or blocked: and makes no solve claim.",
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class UpstreamEvidence:
    """Container for already-produced artifacts used by the no-solve precheck."""

    registry: Mapping[str, Any]
    exp5493: Mapping[str, Any]
    exp5494: Mapping[str, Any]
    exp5464: Mapping[str, Any]
    prior_salience: Sequence[Mapping[str, Any]]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
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


def _registry_depth(registry: Mapping[str, Any], game: str) -> int:
    return _as_int((_registry_rows(registry).get(game) or {}).get("levels_reproduced"))


def _registry_total(registry: Mapping[str, Any]) -> int:
    return _as_int(registry.get("reproducible_total_levels"))


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_upstream_evidence(root: Path = REPO) -> UpstreamEvidence:
    root = Path(root)
    return UpstreamEvidence(
        registry=_read_yaml(root / REGISTRY_RELATIVE_PATH),
        exp5493=_read_json(root / EXP5493_RELATIVE_PATH),
        exp5494=_read_json(root / EXP5494_RELATIVE_PATH),
        exp5464=_read_json(root / EXP5464_RELATIVE_PATH),
        prior_salience=[
            _read_json(root / EXP5465_RELATIVE_PATH),
            _read_json(root / EXP5480_RELATIVE_PATH),
        ],
    )


def _receipt_coordinate(receipt: Mapping[str, Any]) -> tuple[int, int] | None:
    data = receipt.get("data")
    if not isinstance(data, Mapping):
        return None
    if data.get("x") is None or data.get("y") is None:
        return None
    return (_as_int(data.get("x")), _as_int(data.get("y")))


def audit_null_coordinates(
    exp5494: Mapping[str, Any],
    exp5464: Mapping[str, Any],
) -> JsonDict:
    """REQ-ARC-FCP-5507: classify no-op coordinate evidence before reuse."""

    receipts = list((exp5494.get("attempt") or {}).get("measurement_access_receipts") or [])
    coordinate_summary: dict[str, JsonDict] = {}
    null_or_missing: list[str] = []
    valid_count = 0
    zero_change_count = 0

    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, Mapping):
            continue
        coord = _receipt_coordinate(receipt)
        receipt_id = str(receipt.get("receipt_id") or f"receipt_{index}")
        changed_cells = _as_int(receipt.get("changed_cells"))
        if coord is None:
            null_or_missing.append(receipt_id)
            continue
        valid_count += 1
        if changed_cells == 0:
            zero_change_count += 1
        key = f"{coord[0]},{coord[1]}"
        row = coordinate_summary.setdefault(
            key,
            {
                "x": coord[0],
                "y": coord[1],
                "receipts": 0,
                "changed_receipts": 0,
                "zero_change_receipts": 0,
                "changed_cells_total": 0,
            },
        )
        row["receipts"] += 1
        row["changed_cells_total"] += changed_cells
        if changed_cells:
            row["changed_receipts"] += 1
        else:
            row["zero_change_receipts"] += 1

    prior_audit = exp5464.get("null_coordinate_audit") or {}
    contaminated = list(prior_audit.get("contaminated_artifacts") or [])
    exploit_valid = bool(prior_audit.get("null_coordinate_exploit_valid")) or bool(contaminated)
    metric_artifact = bool(null_or_missing)
    valid_noop_actions = valid_count > 0 and zero_change_count > 0 and not metric_artifact

    return {
        "source_artifacts": [EXP5464_RELATIVE_PATH, EXP5494_RELATIVE_PATH],
        "receipts_checked": len(receipts),
        "valid_coordinate_receipts": valid_count,
        "null_or_missing_coordinate_receipts": len(null_or_missing),
        "null_or_missing_receipt_ids": null_or_missing,
        "zero_change_receipts": zero_change_count,
        "coordinate_effect_summary": coordinate_summary,
        "checked_reproduced_loop_artifacts": _as_int(
            prior_audit.get("checked_reproduced_loop_artifacts")
        ),
        "contaminated_artifacts": contaminated,
        "null_coordinate_exploit_valid": exploit_valid,
        "metric_artifact_detected": metric_artifact,
        "prior_noop_behaviors_valid_game_actions": valid_noop_actions,
        "verdict": (
            "valid_recorded_actions_with_noop_effects"
            if valid_noop_actions and not exploit_valid
            else "blocked_null_coordinate_metric_artifact"
        ),
    }


def _duplicate_rejections(evidence: UpstreamEvidence, selected_game: str, selected_level: int) -> list[JsonDict]:
    registry = evidence.registry
    rejections: list[JsonDict] = []
    duplicate_probe = (
        (evidence.exp5464.get("metric_integrity_probe_receipts") or {}).get("duplicate_probe")
        or {}
    )
    if duplicate_probe.get("duplicate_rejected") is True:
        game = str(duplicate_probe.get("game") or "")
        target_level = _as_int(duplicate_probe.get("target_level"))
        rejections.append(
            {
                "target": _target_marker(game, target_level),
                "registry_depth": _as_int(duplicate_probe.get("registry_depth")),
                "reason": "upstream_duplicate_depth_probe",
                "source_artifact": EXP5464_RELATIVE_PATH,
            }
        )
    selected_depth = _registry_depth(registry, selected_game)
    if selected_game and selected_level <= selected_depth:
        rejections.append(
            {
                "target": _target_marker(selected_game, selected_level),
                "registry_depth": selected_depth,
                "reason": "requested_level_already_reproducible",
                "source_artifact": REGISTRY_RELATIVE_PATH,
            }
        )
    return rejections


def _prior_salience_target(artifact: Mapping[str, Any]) -> tuple[str, int]:
    game = str(artifact.get("target_game") or artifact.get("game") or "")
    level = _as_int(artifact.get("target_level_attempted") or artifact.get("target_level"))
    return game, level


def _recent_no_bank_rejections(evidence: UpstreamEvidence) -> list[JsonDict]:
    rows: list[JsonDict] = []
    seen: set[tuple[str, str]] = set()

    def add(row: JsonDict) -> None:
        key = (str(row.get("target") or ""), str(row.get("prior_mechanism") or ""))
        if key not in seen:
            seen.add(key)
            rows.append(row)

    for marker in evidence.exp5493.get("excluded_recent_no_bank_targets") or []:
        add(
            {
                "target": str(marker),
                "prior_mechanism": "recent_live_no_bank_target",
                "decision": "rejected_recent_no_bank",
                "source_artifact": EXP5493_RELATIVE_PATH,
            }
        )

    for path, artifact in (
        (EXP5465_RELATIVE_PATH, evidence.prior_salience[0] if evidence.prior_salience else {}),
        (EXP5480_RELATIVE_PATH, evidence.prior_salience[1] if len(evidence.prior_salience) > 1 else {}),
    ):
        game, level = _prior_salience_target(artifact)
        if game and level > 0 and artifact.get("new_level_banked") is not True:
            add(
                {
                    "target": _target_marker(game, level),
                    "prior_mechanism": "connected_component_salience_only",
                    "decision": "rejected_same_target_same_mechanism",
                    "source_artifact": path,
                }
            )

    exp5494_game = str(evidence.exp5494.get("selected_game") or "")
    exp5494_level = _as_int(evidence.exp5494.get("target_level"))
    if exp5494_game and exp5494_level > 0 and evidence.exp5494.get("new_level_banked") is not True:
        add(
            {
                "target": _target_marker(exp5494_game, exp5494_level),
                "prior_mechanism": "trajectory_option_induction",
                "decision": "rejected_same_target_same_mechanism",
                "replacement_mechanism": CHANGED_MECHANISM,
                "source_artifact": EXP5494_RELATIVE_PATH,
            }
        )
    return rows


def _feature_counts(receipts: Mapping[str, Any]) -> JsonDict:
    return {
        "connected_components": len(receipts.get("connected_component_rows") or []),
        "color_blobs": len(receipts.get("color_blob_rows") or []),
        "changed_pixels": len(receipts.get("changed_pixel_rows") or []),
        "salience_tiers": len(receipts.get("salience_tier_rows") or []),
        "action_effect_observations": len(receipts.get("action_effect_observations") or []),
    }


def _perception_grounding_findings(evidence: UpstreamEvidence, null_audit: Mapping[str, Any]) -> list[JsonDict]:
    findings: list[JsonDict] = []
    if null_audit.get("prior_noop_behaviors_valid_game_actions") is True:
        findings.append(
            {
                "finding": "dc22_zero_change_clicks_are_valid_observations",
                "source_artifact": EXP5494_RELATIVE_PATH,
                "evidence": "Exp5494 recorded ACTION6 data coordinates for zero-change receipts.",
                "classical_pass": "action_effect_delta_filter",
            }
        )

    verifier_checks = [
        row
        for row in evidence.exp5494.get("verifier_checks") or []
        if isinstance(row, Mapping) and "effect_rate" in row
    ]
    if verifier_checks:
        rates = [_as_int(float(row.get("effect_rate", 0.0)) * 1000) / 1000 for row in verifier_checks]
        findings.append(
            {
                "finding": "dc22_action_effect_asymmetry_between_blob_candidates",
                "source_artifact": EXP5494_RELATIVE_PATH,
                "effect_rate_min": min(rates),
                "effect_rate_max": max(rates),
                "routes": sorted({str(row.get("salience_route") or "") for row in verifier_checks}),
                "classical_pass": "connected_component_color_blob_plus_effect_rate",
            }
        )

    for path, artifact in (
        (EXP5465_RELATIVE_PATH, evidence.prior_salience[0] if evidence.prior_salience else {}),
        (EXP5480_RELATIVE_PATH, evidence.prior_salience[1] if len(evidence.prior_salience) > 1 else {}),
    ):
        receipts = artifact.get("feature_receipts") or artifact.get("salience_feature_summary") or {}
        counts = _feature_counts(receipts if isinstance(receipts, Mapping) else {})
        game, level = _prior_salience_target(artifact)
        if any(counts.values()):
            findings.append(
                {
                    "finding": "prior_salience_no_bank_needs_generation_not_ranking_only",
                    "target": _target_marker(game, level),
                    "source_artifact": path,
                    "feature_counts": counts,
                    "classical_pass": "connected_component_color_blob_salience_receipts",
                }
            )
    return findings


def _selected_target(evidence: UpstreamEvidence) -> tuple[str, int]:
    game = str(evidence.exp5494.get("selected_game") or evidence.exp5493.get("selected_game") or "")
    level = _as_int(evidence.exp5494.get("target_level") or evidence.exp5493.get("selected_target_level"))
    return game, level


def build_precheck(
    evidence: UpstreamEvidence,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
    tests_run: Sequence[str] = (),
    duration_s: float = 0.0,
) -> JsonDict:
    """SCENARIO-ARC-FCP-5507: build the no-solve target-selection artifact."""

    selected_game, selected_target_level = _selected_target(evidence)
    null_audit = audit_null_coordinates(evidence.exp5494, evidence.exp5464)
    findings = _perception_grounding_findings(evidence, null_audit)
    duplicate_rejections = _duplicate_rejections(evidence, selected_game, selected_target_level)
    recent_rejections = _recent_no_bank_rejections(evidence)
    selected_depth = _registry_depth(evidence.registry, selected_game)
    duplicate_selected = bool(selected_game and selected_target_level <= selected_depth)
    clean_null_audit = (
        null_audit.get("metric_artifact_detected") is False
        and null_audit.get("null_coordinate_exploit_valid") is False
    )
    ready = bool(
        selected_game
        and selected_target_level > 0
        and not duplicate_selected
        and clean_null_audit
        and findings
        and recent_rejections
    )
    blockers: list[str] = []
    if not selected_game or selected_target_level <= 0:
        blockers.append("missing_selected_target")
    if duplicate_selected:
        blockers.append(f"{_target_marker(selected_game, selected_target_level)} already reproduced")
    if not clean_null_audit:
        blockers.append("null_coordinate_audit_not_clean")
    if not findings:
        blockers.append("no_perception_grounding_findings")
    if not recent_rejections:
        blockers.append("recent_no_bank_audit_missing")

    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5507_arc_null_coordinate_perception_precheck_v499.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "complete" if ready else "blocked",
        "registry_path": REGISTRY_RELATIVE_PATH,
        "reproducible_total_levels_before": _registry_total(evidence.registry),
        "duplicate_targets_rejected": duplicate_rejections,
        "recent_no_bank_targets_rejected": recent_rejections,
        "null_coordinate_audit": null_audit,
        "perception_grounding_findings": findings,
        "selected_game": selected_game if ready else "",
        "selected_level": _level_label(selected_target_level) if ready else "",
        "selected_mechanism": CHANGED_MECHANISM if ready else "",
        "levelup_attempt_ready": ready,
        "solve_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {selected_game} {_level_label(selected_target_level)} perception-generation precheck ready; no solve claimed"
            if ready
            else "blocked: " + ", ".join(blockers)
        ),
        "input_artifacts": [
            EXP5464_RELATIVE_PATH,
            EXP5465_RELATIVE_PATH,
            EXP5480_RELATIVE_PATH,
            EXP5493_RELATIVE_PATH,
            EXP5494_RELATIVE_PATH,
        ],
        "preconditions_checked": dict(preconditions_checked or {}),
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
    if artifact.get("registry_path") != REGISTRY_RELATIVE_PATH:
        errors.append(f"registry_path must be {REGISTRY_RELATIVE_PATH}")
    if type(artifact.get("reproducible_total_levels_before")) is not int:
        errors.append("reproducible_total_levels_before must be bare int")
    for field in ("duplicate_targets_rejected", "recent_no_bank_targets_rejected"):
        if not isinstance(artifact.get(field), list):
            errors.append(f"{field} must be a list")
    if not isinstance(artifact.get("null_coordinate_audit"), dict):
        errors.append("null_coordinate_audit must be a dict")
    if not isinstance(artifact.get("perception_grounding_findings"), list):
        errors.append("perception_grounding_findings must be a list")
    for field in ("selected_game", "selected_level", "selected_mechanism"):
        if not isinstance(artifact.get(field), str):
            errors.append(f"{field} must be a string")
    if type(artifact.get("levelup_attempt_ready")) is not bool:
        errors.append("levelup_attempt_ready must be bare bool")
    if artifact.get("solve_claimed") is not False:
        errors.append("solve_claimed must be false")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "blocked:")):
        errors.append("honest_verdict must start with complete: or blocked:")
    if _verdict_claims_solve(verdict):
        errors.append("honest_verdict must not claim a solve")
    if artifact.get("levelup_attempt_ready") is True:
        if not artifact.get("selected_game"):
            errors.append("levelup_attempt_ready requires selected_game")
        if not artifact.get("selected_level"):
            errors.append("levelup_attempt_ready requires selected_level")
        if not artifact.get("selected_mechanism"):
            errors.append("levelup_attempt_ready requires selected_mechanism")
        if not artifact.get("perception_grounding_findings"):
            errors.append("levelup_attempt_ready requires perception_grounding_findings")
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
) -> JsonDict:
    started = time.monotonic()
    root = Path(root)
    spec_text = _read_text(root / SPEC_RELATIVE_PATH)
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "CLAUDE.md": (root / "CLAUDE.md").exists(),
        "registry_present": (root / REGISTRY_RELATIVE_PATH).exists(),
        "known_issues_present": (root / KNOWN_ISSUES_RELATIVE_PATH).exists(),
        "levers_note_present": (root / LEVERS_NOTE_RELATIVE_PATH).exists(),
        "exp5493_present": (root / EXP5493_RELATIVE_PATH).exists(),
        "exp5494_present": (root / EXP5494_RELATIVE_PATH).exists(),
        "spec_has_req_5507": "REQ-ARC-FCP-5507" in spec_text,
        "game_source_read": False,
        "offline_bfs_used": False,
        "per_game_adapter_used": False,
        "levers_note_updated": False,
    }
    evidence = load_upstream_evidence(root)
    artifact = build_precheck(
        evidence,
        preconditions_checked=preconditions,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(artifact)
    _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
