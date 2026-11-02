"""Experiment 5493: ARC trajectory/option target precheck.

Spec refs: REQ-ARC-FCP-5493, SCENARIO-ARC-FCP-5493.

This module deliberately stops before a solve attempt. It reads the ARC solve
registry and the retired-scope manifest, filters out duplicate or stale
no-bank targets, and emits the one target that a later Exp5494 live agent can
attempt using runtime trajectory or option induction from its own observations.
The code never reads game source, never runs offline BFS, and never builds a
per-game adapter because those paths would make the hidden-game live-agent
claim weaker rather than stronger.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5493
EXPERIMENT = "experiment_5493_arc_trajectory_target_precheck_v498"
MILESTONE = "2026.07.498"
RESULT_RELATIVE_PATH = "results/experiment_5493_arc_trajectory_target_precheck_v498.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
EXCLUSION_MANIFEST_RELATIVE_PATH = "ops/exclusion_manifest.yaml"
KNOWN_ISSUES_RELATIVE_PATH = "ops/known-issues.md"
EXP5479_RELATIVE_PATH = "results/experiment_5479_arc_target_rotation_precheck_v497.json"
EXP5480_RELATIVE_PATH = "results/experiment_5480_arc_live_salience_levelup_v497.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5493", "SCENARIO-ARC-FCP-5493"]
INFERENCE_SUBSTRATE = "registry_precheck_no_solve"
RECENT_NO_BANK_TARGETS = ("sb26:L3", "bp35:L3", "ka59:L2", "cn04:L4", "re86:L3")
RETIRED_SCOPE_TOKENS = (
    "novelty-only",
    "novelty-bonus",
    "curiosity-only",
    "curiosity",
    "energy-as-fitness",
    "quality-diversity",
    "archive-granularity",
    "archive granularity",
)
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5493_arc_trajectory_target_precheck_v498.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5493_arc_trajectory_target_precheck_v498.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5493_arc_trajectory_target_precheck_v498.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "registry_path": {"principle": "must equal ops/arc_solve_registry.yaml."},
    "excluded_recent_no_bank_targets": {
        "principle": "auditable list containing sb26:L3, bp35:L3, ka59:L2, cn04:L4, and re86:L3 unless the selected target proves a different level and mechanism."
    },
    "duplicate_solve_avoided": {
        "principle": "bare bool proving the selected target is strictly deeper than the registry depth."
    },
    "selected_game": {
        "principle": "selected game id, or empty string when no eligible target exists."
    },
    "selected_target_level": {
        "principle": "selected next target level as a bare int, or 0 when blocked."
    },
    "prior_levels_reproduced": {
        "principle": "authoritative registry depth for the selected game before Exp5493."
    },
    "proposed_live_mechanism": {
        "principle": "one-line live-path mechanism to hand to Exp5494, not a retired exploration-signal rerun."
    },
    "trajectory_induction_preconditions": {
        "principle": "list of live-observation prerequisites that must hold before Exp5494 attempts the target."
    },
    "offline_bfs_used": {
        "principle": "must be false; this precheck is registry-only and no offline ground-truth BFS is run."
    },
    "per_game_adapter_used": {
        "principle": "must be false; target eligibility is not credited to a hand adapter."
    },
    "arc_trajectory_precheck_ready": {
        "principle": "true only when a non-duplicate, non-recent-no-bank, non-retired-scope target is selected."
    },
    "inference_substrate": {"principle": "must equal registry_precheck_no_solve."},
    "honest_verdict": {
        "principle": "one-line verdict starting complete: or blocked: without a solve claim."
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class TrajectoryCandidate:
    """Candidate metadata used by the registry-only selector.

    Each row is a proposed future Exp5494 target, not an implementation of that
    target. The preconditions describe what the live agent must be able to see
    from runtime frames and action effects before it spends a bounded attempt.
    """

    game: str
    target_level: int
    proposed_live_mechanism: str
    trajectory_induction_preconditions: tuple[str, ...]
    priority: int
    live_mechanism_hooks: tuple[str, ...] = ()
    evidence: str = ""


DEFAULT_TRAJECTORY_CANDIDATES: tuple[TrajectoryCandidate, ...] = (
    TrajectoryCandidate(
        game="sb26",
        target_level=3,
        proposed_live_mechanism="connected-component salience sequence induction",
        trajectory_induction_preconditions=("runtime_action_effect_observations",),
        priority=10,
    ),
    TrajectoryCandidate(
        game="bp35",
        target_level=3,
        proposed_live_mechanism="platformer trajectory option induction from visible falls and blocker clears",
        trajectory_induction_preconditions=("runtime_action_effect_observations",),
        priority=20,
    ),
    TrajectoryCandidate(
        game="ka59",
        target_level=2,
        proposed_live_mechanism="object-motion trajectory induction from pushed block effects",
        trajectory_induction_preconditions=("runtime_action_effect_observations",),
        priority=30,
    ),
    TrajectoryCandidate(
        game="cn04",
        target_level=4,
        proposed_live_mechanism="marker-pair option reinduction from observed sprite moves",
        trajectory_induction_preconditions=("runtime_action_effect_observations",),
        priority=40,
    ),
    TrajectoryCandidate(
        game="re86",
        target_level=3,
        proposed_live_mechanism="sprite-overlay trajectory reinduction from observed source cycling",
        trajectory_induction_preconditions=("runtime_action_effect_observations",),
        priority=50,
    ),
    TrajectoryCandidate(
        game="dc22",
        target_level=3,
        proposed_live_mechanism=(
            "E3AgentPolicy + LiveCoExLandmarkFrontierGenerator option induction "
            "over visible toggle-navigation action effects"
        ),
        trajectory_induction_preconditions=(
            "runtime_action_effect_observations",
            "visible_toggle_or_navigation_state_changes",
            "level_counter_delta_read_from_frames",
            "frontier_prefixes_grouped_into_options",
        ),
        priority=60,
        live_mechanism_hooks=(
            "python/carnot/agentic/arc_competition_agent.py:E3AgentPolicy",
            "python/carnot/agentic/arc_live_trajectory_frontier.py:LiveCoExLandmarkFrontierGenerator",
        ),
        evidence="registry records dc22 L2 config-toggle navigation with visible movement and ACTION6 toggles",
    ),
    TrajectoryCandidate(
        game="g50t",
        target_level=3,
        proposed_live_mechanism="target-offset trajectory induction from clone-support action effects",
        trajectory_induction_preconditions=("runtime_action_effect_observations",),
        priority=70,
    ),
)


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


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():  # pragma: no cover - defensive missing-repo path
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():  # pragma: no cover - defensive missing-repo path
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_text(path: Path) -> str:
    if not path.exists():  # pragma: no cover - defensive missing-repo path
        return ""
    return path.read_text(encoding="utf-8")


def load_registry(root: Path = REPO) -> dict[str, Any]:
    registry = _load_yaml(root / REGISTRY_RELATIVE_PATH)
    if registry:
        return registry
    return {"reproducible_total_levels": 0, "games": []}


def _retired_scope_patterns(exclusion_manifest: Mapping[str, Any]) -> list[str]:
    patterns = set(RETIRED_SCOPE_TOKENS)
    for section in ("retired_extras", "retired_experiments", "retired"):
        for item in exclusion_manifest.get(section) or []:
            if not isinstance(item, Mapping):
                continue
            text = " ".join(
                str(item.get(key) or "")
                for key in ("id", "experiment_scope", "reason")
            ).lower()
            if "exp5154" in text or "generation_axis_exploration_signal" in text:
                for pattern in item.get("blocked_patterns") or []:
                    patterns.add(str(pattern).lower())
    return sorted(patterns)


def _matches_retired_scope(mechanism: str, patterns: Sequence[str]) -> bool:
    text = str(mechanism).lower()
    return any(pattern and pattern in text for pattern in patterns)


def _levels_by_candidate_game(
    registry: Mapping[str, Any],
    candidates: Sequence[TrajectoryCandidate],
) -> dict[str, int]:
    rows = _registry_rows(registry)
    return {
        candidate.game: _as_int((rows.get(candidate.game) or {}).get("levels_reproduced"))
        for candidate in candidates
    }


def _exp5494_command(selection: Mapping[str, Any]) -> str:
    game = str(selection.get("selected_game") or "")
    target = _as_int(selection.get("selected_target_level"))
    prior = _as_int(selection.get("prior_levels_reproduced"))
    if not game or target <= 0:
        return ""
    return (
        ".venv/bin/python -m "
        "carnot.experiment_5494_arc_live_trajectory_option_induction_v498 "
        f"--game {game} --target-level {target} --prior-levels {prior} "
        "--mechanism live-coex-landmark-frontier --no-offline-bfs --no-per-game-adapter"
    )


def select_trajectory_target(
    registry: Mapping[str, Any],
    exclusion_manifest: Mapping[str, Any],
    *,
    candidates: Sequence[TrajectoryCandidate] = DEFAULT_TRAJECTORY_CANDIDATES,
    recent_no_bank_targets: Sequence[str] = RECENT_NO_BANK_TARGETS,
) -> dict[str, Any]:
    """REQ-ARC-FCP-5493: choose one non-duplicate trajectory target, or block."""

    rows = _registry_rows(registry)
    recent = [str(item) for item in recent_no_bank_targets]
    recent_set = set(recent)
    retired_patterns = _retired_scope_patterns(exclusion_manifest)
    audit: dict[str, dict[str, Any]] = {}
    sorted_candidates = sorted(candidates, key=lambda item: item.priority)
    levels_by_game = _levels_by_candidate_game(registry, sorted_candidates)

    for candidate in sorted_candidates:
        row = rows.get(candidate.game)
        prior = _as_int((row or {}).get("levels_reproduced"))
        target = _as_int(candidate.target_level)
        marker = _target_marker(candidate.game, target)
        base = {
            "game": candidate.game,
            "target_level": target,
            "prior_levels_reproduced": prior,
            "proposed_live_mechanism": candidate.proposed_live_mechanism,
        }
        if row is None or str(row.get("reproducibility") or "") != "reproduced":
            audit[marker] = {**base, "decision": "rejected_missing_reproduced_registry_row"}
            continue
        if target <= prior:
            audit[marker] = {**base, "decision": "rejected_duplicate"}
            continue
        if marker in recent_set:
            audit[marker] = {**base, "decision": "rejected_recent_no_bank"}
            continue
        if _matches_retired_scope(candidate.proposed_live_mechanism, retired_patterns):
            audit[marker] = {**base, "decision": "rejected_retired_scope"}
            continue
        if not candidate.trajectory_induction_preconditions or not candidate.live_mechanism_hooks:
            audit[marker] = {**base, "decision": "rejected_missing_live_trajectory_hooks"}
            continue

        audit[marker] = {
            **base,
            "decision": "selected",
            "live_mechanism_hooks": list(candidate.live_mechanism_hooks),
            "evidence": candidate.evidence,
        }
        selection = {
            "blocked": False,
            "selected_game": candidate.game,
            "selected_target_level": target,
            "prior_levels_reproduced": prior,
            "proposed_live_mechanism": candidate.proposed_live_mechanism,
            "trajectory_induction_preconditions": list(
                candidate.trajectory_induction_preconditions
            ),
            "duplicate_solve_avoided": target > prior,
            "candidate_audit": audit,
            "levels_reproduced_by_candidate_game": levels_by_game,
            "excluded_recent_no_bank_targets": recent,
            "excluded_retired_scope_patterns": retired_patterns,
            "selection_reason": f"{marker} is the first eligible trajectory target after registry filters",
        }
        selection["exp5494_command"] = _exp5494_command(selection)
        return selection

    return {
        "blocked": True,
        "blocker": "no_eligible_trajectory_target",
        "selected_game": "",
        "selected_target_level": 0,
        "prior_levels_reproduced": 0,
        "proposed_live_mechanism": "",
        "trajectory_induction_preconditions": [],
        "duplicate_solve_avoided": True,
        "candidate_audit": audit,
        "levels_reproduced_by_candidate_game": levels_by_game,
        "excluded_recent_no_bank_targets": recent,
        "excluded_retired_scope_patterns": retired_patterns,
        "selection_reason": "no eligible target survived duplicate, recent no-bank, and retired-scope filters",
        "exp5494_command": "",
    }


def build_artifact(
    *,
    selection: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    tests_run: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    ready = bool(
        not selection.get("blocked")
        and selection.get("selected_game")
        and _as_int(selection.get("selected_target_level"))
        > _as_int(selection.get("prior_levels_reproduced"))
        and selection.get("duplicate_solve_avoided") is True
    )
    selected_game = str(selection.get("selected_game") or "")
    selected_target_level = _as_int(selection.get("selected_target_level"))
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5493_arc_trajectory_target_precheck_v498.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "complete" if ready else "blocked",
        "registry_path": REGISTRY_RELATIVE_PATH,
        "excluded_recent_no_bank_targets": list(
            selection.get("excluded_recent_no_bank_targets") or RECENT_NO_BANK_TARGETS
        ),
        "duplicate_solve_avoided": selection.get("duplicate_solve_avoided") is True,
        "selected_game": selected_game,
        "selected_target_level": int(selected_target_level),
        "prior_levels_reproduced": _as_int(selection.get("prior_levels_reproduced")),
        "proposed_live_mechanism": str(selection.get("proposed_live_mechanism") or ""),
        "trajectory_induction_preconditions": list(
            selection.get("trajectory_induction_preconditions") or []
        ),
        "offline_bfs_used": False,
        "per_game_adapter_used": False,
        "arc_trajectory_precheck_ready": bool(ready),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {selected_game} {_level_label(selected_target_level)} trajectory precheck ready for Exp5494"
            if ready
            else f"blocked: {selection.get('blocker') or 'no_eligible_trajectory_target'}"
        ),
        "candidate_audit": dict(selection.get("candidate_audit") or {}),
        "levels_reproduced_by_candidate_game": dict(
            selection.get("levels_reproduced_by_candidate_game") or {}
        ),
        "excluded_retired_scope_patterns": list(
            selection.get("excluded_retired_scope_patterns") or []
        ),
        "selection_reason": str(selection.get("selection_reason") or ""),
        "exp5494_command": str(selection.get("exp5494_command") or ""),
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
    if artifact.get("registry_path") != REGISTRY_RELATIVE_PATH:
        errors.append(f"registry_path must be {REGISTRY_RELATIVE_PATH}")
    recent = artifact.get("excluded_recent_no_bank_targets")
    if not isinstance(recent, list):
        errors.append("excluded_recent_no_bank_targets must be a list")
        recent = []
    for marker in RECENT_NO_BANK_TARGETS:
        if marker not in recent:
            errors.append(f"excluded_recent_no_bank_targets missing {marker}")
    for field in ("duplicate_solve_avoided", "offline_bfs_used", "per_game_adapter_used", "arc_trajectory_precheck_ready"):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    if not isinstance(artifact.get("selected_game"), str):
        errors.append("selected_game must be a string")
    for field in ("selected_target_level", "prior_levels_reproduced"):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    mechanism = artifact.get("proposed_live_mechanism")
    if not isinstance(mechanism, str):
        errors.append("proposed_live_mechanism must be a string")
        mechanism = ""
    if _matches_retired_scope(str(mechanism), RETIRED_SCOPE_TOKENS):
        errors.append("proposed_live_mechanism must not match retired exploration-signal scope")
    if not isinstance(artifact.get("trajectory_induction_preconditions"), list):
        errors.append("trajectory_induction_preconditions must be a list")
    for field in ("offline_bfs_used", "per_game_adapter_used"):
        if artifact.get(field) is True:
            errors.append(f"{field} must be false")
    if artifact.get("arc_trajectory_precheck_ready") is True:
        selected_level = _as_int(artifact.get("selected_target_level"))
        prior = _as_int(artifact.get("prior_levels_reproduced"))
        if not artifact.get("selected_game"):
            errors.append("arc_trajectory_precheck_ready requires selected_game")
        if selected_level <= prior:
            errors.append("arc_trajectory_precheck_ready requires selected_target_level > prior_levels_reproduced")
        if not artifact.get("trajectory_induction_preconditions"):
            errors.append("arc_trajectory_precheck_ready requires trajectory_induction_preconditions")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
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
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    manifest_path = root / EXCLUSION_MANIFEST_RELATIVE_PATH
    known_issues_path = root / KNOWN_ISSUES_RELATIVE_PATH
    exp5479_path = root / EXP5479_RELATIVE_PATH
    exp5480_path = root / EXP5480_RELATIVE_PATH
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "spec_has_req_5493": (
            "REQ-ARC-FCP-5493" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "exclusion_manifest_present": manifest_path.exists(),
        "known_issues_present": known_issues_path.exists(),
        "exp5479_present": exp5479_path.exists(),
        "exp5480_present": exp5480_path.exists(),
        "offline_bfs_used": False,
        "per_game_adapter_used": False,
    }
    registry = load_registry(root)
    exclusion_manifest = _load_yaml(manifest_path)
    known_issues_text = _load_text(known_issues_path)
    exp5479 = _load_json(exp5479_path)
    exp5480 = _load_json(exp5480_path)
    recent = list(RECENT_NO_BANK_TARGETS)
    for row in (exp5479, exp5480):
        game = str(row.get("selected_game") or row.get("game") or "")
        level = _as_int(row.get("selected_target_level") or row.get("target_level"))
        if game and level > 0 and row.get("new_level_banked") is not True:
            marker = _target_marker(game, level)
            if marker not in recent:
                recent.append(marker)
    if "archive granularity" in known_issues_text.lower():
        preconditions["retired_archive_granularity_scope_seen"] = True
    selection = select_trajectory_target(
        registry,
        exclusion_manifest,
        recent_no_bank_targets=recent,
    )
    artifact = build_artifact(
        selection=selection,
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
