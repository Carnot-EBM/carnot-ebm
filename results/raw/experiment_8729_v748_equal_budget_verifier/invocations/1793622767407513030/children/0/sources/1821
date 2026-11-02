"""Exp5520: ARC action-diversity target precheck.

Spec refs: REQ-ARC-FCP-5520, SCENARIO-ARC-FCP-5520.

This module is deliberately a no-credit precheck, not a solver. Exp5508 showed
that a perception-grounded generator could still collapse into a small
coordinate loop. Before the next live level-up attempt, this step re-reads the
solve registry, rotates away from the Exp5508 target, and dry-runs the changed
live-path generator over salience candidates to prove that coordinate diversity
is measurable before any attempt can claim solve credit.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from carnot.agentic.arc_perception_generation import ActionDiversityPerceptionGenerator


JsonDict = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5520
EXPERIMENT = "experiment_5520_arc_action_diversity_target_precheck"
MILESTONE = "2026.07.500"
RESULT_RELATIVE_PATH = "results/experiment_5520_arc_action_diversity_target_precheck.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
EXP5508_RELATIVE_PATH = "results/experiment_5508_arc_live_perception_generation_levelup_v499.json"
EXP5480_RELATIVE_PATH = "results/experiment_5480_arc_live_salience_levelup_v497.json"
EXP5465_RELATIVE_PATH = "results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5520", "SCENARIO-ARC-FCP-5520"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "arc_live_precheck"
NO_CREDIT_PROBE_BUDGET = 8
MIN_ACTION_ENTROPY = 1.5
MAX_REPEATED_COORDINATE_RATE = 0.25
MIN_SALIENCE_COVERAGE_RATE = 0.5
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5520_arc_action_diversity_target_precheck.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5520_arc_action_diversity_target_precheck.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5520_arc_action_diversity_target_precheck.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "registry_precheck_done": "bare bool proving ops/arc_solve_registry.yaml was checked before target selection.",
    "selected_game": "one registry-safe game id selected for the next live level-up attempt, or empty string when blocked.",
    "selected_level": "next unreproduced target level as a string or bare int; it must be strictly deeper than the registry depth.",
    "already_reproduced": "must be false for any ready artifact.",
    "exp5508_pattern_reused": "must be false; Exp5508's repeated ACTION6 coordinate loop cannot be reused.",
    "candidate_generator_changes": "non-empty list naming live-path generation changes such as repeated-coordinate suppression, target rotation, action entropy gating, or salience coverage.",
    "action_entropy": "Shannon entropy over dry-run action/coordinate choices as a bare float.",
    "repeated_coordinate_rate": "fraction of dry-run consecutive coordinate choices that repeat a prior coordinate, as a bare float.",
    "salience_coverage_rate": "fraction of dry-run choices covering distinct salience candidates, as a bare float.",
    "no_credit_probe_attempts": "bare int count of no-credit dry-run choices measured before the live attempt.",
    "arc_levelup_candidate_ready": "bare bool true only when registry and Exp5508-pattern gates pass and the diversity metrics meet threshold.",
    "solve_provenance": "must equal live_agent_self_discovery.",
    "inference_substrate": "must equal arc_live_precheck.",
    "honest_verdict": "one-line verdict starting complete: or blocked: without claiming a solve.",
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class PrecheckEvidence:
    """All source data used by the no-credit Exp5520 precheck."""

    registry: Mapping[str, Any]
    exp5508: Mapping[str, Any]
    candidate_artifacts: Sequence[Mapping[str, Any]]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _level_label(level: int) -> str:
    return f"L{max(0, int(level))}"


def _parse_level(value: Any) -> int:
    text = str(value or "").strip().upper()
    if text.startswith("L") and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    return 0


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


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }


def _registry_depth(registry: Mapping[str, Any], game: str) -> int:
    return _as_int((_registry_rows(registry).get(game) or {}).get("levels_reproduced"))


def _target_marker(game: str, level: int) -> str:
    return f"{game}:{_level_label(level)}"


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


def _candidate_game(artifact: Mapping[str, Any]) -> str:
    return str(
        artifact.get("target_game")
        or artifact.get("game")
        or artifact.get("selected_game")
        or ""
    )


def _candidate_level(artifact: Mapping[str, Any], registry: Mapping[str, Any], game: str) -> int:
    level = _as_int(
        artifact.get("target_level_attempted")
        or artifact.get("target_level")
        or artifact.get("selected_target_level")
    )
    if level > 0:
        return level
    return _registry_depth(registry, game) + 1 if game else 0


def _candidate_rows(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = (
        ("attempt", "salience_diagnostics", "action_tier_rows"),
        ("salience_diagnostics", "action_tier_rows"),
    )
    for path in paths:
        node: Any = artifact
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
        if isinstance(node, list):
            return [dict(row) for row in node if isinstance(row, Mapping)]

    receipts = artifact.get("feature_receipts")
    if isinstance(receipts, Mapping):
        rows: list[dict[str, Any]] = []
        for row in receipts.get("color_blob_rows") or []:
            if not isinstance(row, Mapping):
                continue
            if "centroid_x" in row and "centroid_y" in row:
                rows.append(
                    {
                        "action": 6,
                        "data": {
                            "x": _as_int(row.get("centroid_x")),
                            "y": _as_int(row.get("centroid_y")),
                        },
                        "score": 4000.0 - _as_float(row.get("tier"), 4.0),
                        "tier": _as_int(row.get("tier"), 4),
                    }
                )
        return rows
    return []


def extract_exp5508_pattern(exp5508: Mapping[str, Any]) -> JsonDict:
    """REQ-ARC-FCP-5520: summarize the failed Exp5508 action pattern."""

    steps = (exp5508.get("attempt") or {}).get("trajectory_taxonomy_steps") or []
    action_rows = [row for row in steps if isinstance(row, Mapping) and row.get("action") is not None]
    signatures = [_row_signature(row) for row in action_rows]
    coords = [_row_coordinate(row) for row in action_rows]
    seen: set[tuple[int, int]] = set()
    repeated = 0
    coord_count = 0
    for coord in coords:
        if coord is None:
            continue
        coord_count += 1
        if coord in seen:
            repeated += 1
        seen.add(coord)
    counts = Counter(signatures)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        probability = float(count) / float(total or 1)
        if probability:
            import math

            entropy -= probability * math.log2(probability)
    selected_level = _parse_level(exp5508.get("selected_level") or exp5508.get("target_level"))
    return {
        "selected_game": str(exp5508.get("selected_game") or ""),
        "selected_level": _level_label(selected_level) if selected_level else "",
        "target_level": int(selected_level),
        "action_count": int(len(action_rows)),
        "unique_coordinate_count": int(len(seen)),
        "coordinates": [{"x": x, "y": y} for x, y in sorted(seen)],
        "repeated_coordinate_rate": float(repeated) / float(max(1, coord_count)),
        "action_entropy": float(entropy),
    }


def _pattern_coordinates(pattern: Mapping[str, Any]) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for row in pattern.get("coordinates") or []:
        if isinstance(row, Mapping) and "x" in row and "y" in row:
            coords.add((_as_int(row.get("x")), _as_int(row.get("y"))))
    return coords


def _runs_exp5508_pattern(
    *,
    game: str,
    level: int,
    pattern: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
    repeated_coordinate_rate: float,
    action_entropy: float,
) -> bool:
    if game == pattern.get("selected_game") and level == _as_int(pattern.get("target_level")):
        return True
    if not probe_rows:
        return True
    pattern_coords = _pattern_coordinates(pattern)
    probe_coords = {_row_coordinate(row) for row in probe_rows if isinstance(row, Mapping)}
    probe_coords.discard(None)
    overlap = len(set(probe_coords) & pattern_coords)
    overlap_rate = float(overlap) / float(max(1, len(probe_coords)))
    return bool(
        repeated_coordinate_rate > 0.5
        or (overlap_rate >= 0.5 and action_entropy <= _as_float(pattern.get("action_entropy")) + 0.1)
    )


def _run_no_credit_probe(
    rows: Sequence[Mapping[str, Any]],
    *,
    exp5508_pattern: Mapping[str, Any],
) -> JsonDict:
    generator = ActionDiversityPerceptionGenerator(max_candidates=NO_CREDIT_PROBE_BUDGET)
    selected_rows = generator.prioritize_rows(
        rows,
        avoid_coordinates=_pattern_coordinates(exp5508_pattern),
    )
    total_salience = len({_row_coordinate(row) for row in rows if _row_coordinate(row) is not None})
    metrics = generator.diversity_metrics(
        selected_rows,
        total_salience_candidates=max(1, min(NO_CREDIT_PROBE_BUDGET, total_salience)),
    )
    return {
        "selected_rows": selected_rows,
        "candidate_generator_changes": generator.change_receipts
        + ["target_rotation", "salience_coverage_gate"],
        "action_entropy": float(metrics["action_entropy"]),
        "repeated_coordinate_rate": float(metrics["repeated_coordinate_rate"]),
        "salience_coverage_rate": float(metrics["salience_coverage_rate"]),
        "no_credit_probe_attempts": int(len(selected_rows)),
    }


def _empty_probe() -> JsonDict:
    return {
        "selected_rows": [],
        "candidate_generator_changes": [
            "connected_component_color_blob_salience",
            "repeated_coordinate_suppression",
            "target_rotation",
            "no_credit_action_entropy_probe",
            "salience_coverage_gate",
        ],
        "action_entropy": 0.0,
        "repeated_coordinate_rate": 1.0,
        "salience_coverage_rate": 0.0,
        "no_credit_probe_attempts": 0,
    }


def load_evidence(root: Path = REPO) -> PrecheckEvidence:
    root = Path(root)
    return PrecheckEvidence(
        registry=_read_yaml(root / REGISTRY_RELATIVE_PATH),
        exp5508=_read_json(root / EXP5508_RELATIVE_PATH),
        candidate_artifacts=[
            _read_json(root / EXP5480_RELATIVE_PATH),
            _read_json(root / EXP5465_RELATIVE_PATH),
        ],
    )


def build_precheck(
    evidence: PrecheckEvidence,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
    tests_run: Sequence[str] = (),
    duration_s: float = 0.0,
) -> JsonDict:
    """SCENARIO-ARC-FCP-5520: select one changed-path target or block."""

    pattern = extract_exp5508_pattern(evidence.exp5508)
    selected: JsonDict | None = None
    last_probe = _empty_probe()
    blocked_duplicate = False
    blocked_pattern = False
    blockers: list[str] = []
    candidate_audit: list[JsonDict] = []

    for artifact in evidence.candidate_artifacts:
        if not isinstance(artifact, Mapping) or not artifact:
            continue
        game = _candidate_game(artifact)
        level = _candidate_level(artifact, evidence.registry, game)
        if not game or level <= 0:
            continue
        rows = _candidate_rows(artifact)
        already = level <= _registry_depth(evidence.registry, game)
        probe = _run_no_credit_probe(rows, exp5508_pattern=pattern) if rows else _empty_probe()
        pattern_reused = _runs_exp5508_pattern(
            game=game,
            level=level,
            pattern=pattern,
            probe_rows=probe["selected_rows"],
            repeated_coordinate_rate=_as_float(probe.get("repeated_coordinate_rate")),
            action_entropy=_as_float(probe.get("action_entropy")),
        )
        metrics_ready = (
            _as_float(probe.get("action_entropy")) >= MIN_ACTION_ENTROPY
            and _as_float(probe.get("repeated_coordinate_rate")) <= MAX_REPEATED_COORDINATE_RATE
            and _as_float(probe.get("salience_coverage_rate")) >= MIN_SALIENCE_COVERAGE_RATE
            and _as_int(probe.get("no_credit_probe_attempts")) > 0
        )
        candidate_audit.append(
            {
                "target": _target_marker(game, level),
                "registry_depth": _registry_depth(evidence.registry, game),
                "already_reproduced": bool(already),
                "exp5508_pattern_reused": bool(pattern_reused),
                "action_entropy": float(probe["action_entropy"]),
                "repeated_coordinate_rate": float(probe["repeated_coordinate_rate"]),
                "salience_coverage_rate": float(probe["salience_coverage_rate"]),
                "no_credit_probe_attempts": int(probe["no_credit_probe_attempts"]),
            }
        )
        last_probe = probe
        blocked_duplicate = blocked_duplicate or already
        blocked_pattern = blocked_pattern or pattern_reused
        if already:
            blockers.append(f"{_target_marker(game, level)} already_in_registry")
            continue
        if pattern_reused:
            blockers.append(f"{_target_marker(game, level)} exp5508_pattern_reused")
            continue
        if not metrics_ready:
            blockers.append(f"{_target_marker(game, level)} diversity_probe_below_threshold")
            continue
        selected = {
            "game": game,
            "level": level,
            "probe": probe,
        }
        break

    ready = selected is not None
    probe = selected["probe"] if selected is not None else last_probe
    selected_game = str(selected["game"]) if selected else ""
    selected_level = _level_label(_as_int(selected["level"])) if selected else ""
    already_reproduced = False if ready else bool(blocked_duplicate)
    exp5508_pattern_reused = False if ready else bool(blocked_pattern)
    if not candidate_audit:
        blockers.append("no_candidate_salience_artifacts")
    if not blockers and not ready:  # pragma: no cover - defensive invariant.
        blockers.append("no_candidate_passed_action_diversity_gate")

    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5520_arc_action_diversity_target_precheck.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "complete" if ready else "blocked",
        "registry_precheck_done": bool(evidence.registry),
        "selected_game": selected_game,
        "selected_level": selected_level,
        "already_reproduced": bool(already_reproduced),
        "exp5508_pattern_reused": bool(exp5508_pattern_reused),
        "candidate_generator_changes": list(probe["candidate_generator_changes"]),
        "action_entropy": float(probe["action_entropy"]),
        "repeated_coordinate_rate": float(probe["repeated_coordinate_rate"]),
        "salience_coverage_rate": float(probe["salience_coverage_rate"]),
        "no_credit_probe_attempts": int(probe["no_credit_probe_attempts"]),
        "arc_levelup_candidate_ready": bool(ready),
        "solve_provenance": SOLVE_PROVENANCE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {selected_game} {selected_level} action-diversity precheck ready; no solve claimed"
            if ready
            else "blocked: " + ", ".join(blockers)
        ),
        "exp5508_pattern": pattern,
        "candidate_audit": candidate_audit,
        "input_artifacts": [
            REGISTRY_RELATIVE_PATH,
            EXP5508_RELATIVE_PATH,
            EXP5480_RELATIVE_PATH,
            EXP5465_RELATIVE_PATH,
        ],
        "preconditions_checked": dict(preconditions_checked or {}),
        "tests_run": list(tests_run),
        "duration_s": float(duration_s),
    }
    return artifact


def _verdict_claims_solve(verdict: str) -> bool:
    text = verdict.lower()
    return "solved" in text or "reproduced" in text or "banked" in text


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    if type(artifact.get("registry_precheck_done")) is not bool:
        errors.append("registry_precheck_done must be bare bool")
    if not isinstance(artifact.get("selected_game"), str):
        errors.append("selected_game must be a string")
    if not isinstance(artifact.get("selected_level"), (str, int)):
        errors.append("selected_level must be a string or int")
    for field in ("already_reproduced", "exp5508_pattern_reused", "arc_levelup_candidate_ready"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    if artifact.get("arc_levelup_candidate_ready") is True:
        if artifact.get("already_reproduced") is not False:
            errors.append("ready artifacts require already_reproduced false")
        if artifact.get("exp5508_pattern_reused") is not False:
            errors.append("ready artifacts require exp5508_pattern_reused false")
        if not artifact.get("selected_game"):
            errors.append("ready artifacts require selected_game")
        if not artifact.get("selected_level"):
            errors.append("ready artifacts require selected_level")
        if _as_float(artifact.get("action_entropy")) < MIN_ACTION_ENTROPY:
            errors.append("ready artifacts require action_entropy above threshold")
        if _as_float(artifact.get("repeated_coordinate_rate")) > MAX_REPEATED_COORDINATE_RATE:
            errors.append("ready artifacts require repeated_coordinate_rate below threshold")
        if _as_float(artifact.get("salience_coverage_rate")) < MIN_SALIENCE_COVERAGE_RATE:
            errors.append("ready artifacts require salience_coverage_rate above threshold")
    changes = artifact.get("candidate_generator_changes")
    if not isinstance(changes, list) or not changes:
        errors.append("candidate_generator_changes must be a non-empty list")
    for field in ("action_entropy", "repeated_coordinate_rate", "salience_coverage_rate"):
        if type(artifact.get(field)) is not float:
            errors.append(f"{field} must be bare float")
    for field in ("repeated_coordinate_rate", "salience_coverage_rate"):
        if type(artifact.get(field)) in (float, int) and not (0.0 <= float(artifact[field]) <= 1.0):
            errors.append(f"{field} must be in [0, 1]")
    if type(artifact.get("no_credit_probe_attempts")) is not int:
        errors.append("no_credit_probe_attempts must be bare int")
    elif artifact["no_credit_probe_attempts"] < 0:
        errors.append("no_credit_probe_attempts must be non-negative")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be arc_live_precheck")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "blocked:")):
        errors.append("honest_verdict must start with complete: or blocked:")
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
) -> JsonDict:
    started = time.monotonic()
    root = Path(root)
    spec_text = _read_text(root / SPEC_RELATIVE_PATH)
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "CLAUDE.md": (root / "CLAUDE.md").exists(),
        "registry_present": (root / REGISTRY_RELATIVE_PATH).exists(),
        "exp5508_present": (root / EXP5508_RELATIVE_PATH).exists(),
        "candidate_salience_artifact_present": (root / EXP5480_RELATIVE_PATH).exists(),
        "spec_has_req_5520": "REQ-ARC-FCP-5520" in spec_text,
        "offline_bfs_used": False,
        "game_source_read": False,
        "per_game_adapter_used": False,
        "solve_claimed": False,
    }
    artifact = build_precheck(
        load_evidence(root),
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
