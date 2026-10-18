"""Exp5561: ARC FSM target rotation precheck.

Spec refs: REQ-ARC-FCP-5561, SCENARIO-ARC-FCP-5561.

This module is a no-credit gate before the next live ARC attempt. It rotates
away from Exp5548's clean no-bank target, checks the registry for duplicate
levels, and dry-runs a tiny finite-state action abstraction through the same
candidate-router shape the live agent can accept at runtime. The abstraction is
intentionally simple: it groups candidate actions into observe, transition,
verify, and reset phases so the later live attempt can diversify actions
without relying on game source, offline BFS, or a per-game calibration solver.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from carnot.agentic.arc_bounded_strategy_router import BoundedStrategyCandidateRouter


JsonDict = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5561
EXPERIMENT = "experiment_5561_arc_fsm_target_rotation_precheck"
MILESTONE = "2026.07.503"
RESULT_RELATIVE_PATH = "results/experiment_5561_arc_fsm_target_rotation_precheck.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
EXP5548_RELATIVE_PATH = "results/experiment_5548_arc_clean_live_levelup.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-human-replay-frame-change/spec.md"
SPEC_REFS = ["REQ-ARC-FCP-5561", "SCENARIO-ARC-FCP-5561"]
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "arc_live_path_precheck_no_llm"
MIN_ACTION_ENTROPY = 1.5
DEFAULT_RANDOM_SEED = 5561
DEFAULT_TARGET_CANDIDATES: tuple[tuple[str, int], ...] = (
    ("g50t", 3),
    ("r11l", 3),
    ("ls20", 3),
    ("re86", 3),
    ("lf52", 3),
    ("sb26", 3),
    ("dc22", 3),
    ("bp35", 3),
)
DEFAULT_TESTS_RUN = [
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5561_arc_fsm_target_rotation_precheck.py "
        "-q --no-cov"
    ),
    (
        ".venv/bin/coverage erase && .venv/bin/coverage run -m pytest "
        "tests/python/test_experiment_5561_arc_fsm_target_rotation_precheck.py "
        "-q -n0 -o addopts= && .venv/bin/coverage report --fail-under=100 -m "
        "python/carnot/experiment_5561_arc_fsm_target_rotation_precheck.py"
    ),
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "llm_invoked": "bare bool false proving the FSM target rotation precheck did not invoke any LLM.",
    "no_model_specs_required": "bare bool true because the declared no-LLM precheck substrate has no model invocation to name.",
    "selected_game": "registry-safe game id selected for the next FSM-guided live attempt, or empty string when blocked.",
    "selected_level": "adjacent unreproduced frontier level label selected after registry and recent no-bank target checks.",
    "registry_precheck_passed": "bare bool proving the registry was read and the selected level is not already reproduced.",
    "already_reproduced": "must remain false because duplicate live levels cannot satisfy the ARC standing progress floor.",
    "recent_no_bank_targets_avoided": "list of recent no-bank target markers rejected before selection unless an explicit retry reason exists.",
    "fsm_action_abstraction_ready": "bare bool proving the FSM action abstraction is live-router reachable and emits bounded action phases.",
    "repeated_coordinate_suppression_enabled": "bare bool proving repeated-coordinate suppression is active before action entropy is trusted.",
    "action_entropy_precheck": "bare float Shannon entropy over suppressed FSM action/coordinate choices before the live attempt.",
    "solve_provenance": "must equal live_agent_self_discovery even though this artifact claims no solve.",
    "arc_fsm_precheck_ready": "bare bool true only when registry, rotation, FSM abstraction, suppression, entropy, and no-LLM gates pass.",
    "tests_added_or_reused": "list of focused tests covering the Exp5561 schema, target rotation, FSM reachability, suppression, and artifact write.",
    "field_principles": "mapping of one-line principle annotations for each headline and gate field.",
    "inference_substrate": "must equal arc_live_path_precheck_no_llm.",
    "honest_verdict": "one-line verdict starting complete: or blocked: without claiming a solve.",
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class FsmPrecheckEvidence:
    """File-backed inputs for the no-credit FSM target rotation gate."""

    registry: Mapping[str, Any]
    exp5548: Mapping[str, Any]


class FSMActionAbstraction:
    """Live-compatible finite-state wrapper for bounded candidate routing."""

    phase_fields: tuple[tuple[str, str], ...] = (
        ("observe_select", "salience_score"),
        ("effect_transition", "effect_score"),
        ("verify_candidate", "verifier_score"),
        ("reset_reinduce", "reset_score"),
    )

    def __init__(
        self,
        *,
        max_candidates: int = 4,
        suppress_repeated_coordinates: bool = True,
    ) -> None:
        self.max_candidates = max(1, int(max_candidates))
        self.suppress_repeated_coordinates = bool(suppress_repeated_coordinates)
        self.last_diagnostics: dict[str, Any] = {}

    def abstract_candidate(self, candidate: Mapping[str, Any]) -> JsonDict:
        """Annotate one live candidate with the FSM phase it currently supports."""

        row = dict(candidate)
        scores = [
            (_as_float(row.get(field)), index, phase)
            for index, (phase, field) in enumerate(self.phase_fields)
        ]
        _, phase_index, phase = max(scores, key=lambda item: (item[0], -item[1]))
        row["fsm_phase"] = phase
        row["fsm_state"] = phase_index
        row["fsm_action"] = f"{phase}:{_candidate_signature(row)}"
        return row

    def rank(
        self,
        frame: Any,
        candidates: Sequence[Mapping[str, Any]],
        *,
        previous_frame: Any | None = None,
    ) -> list[JsonDict]:
        """Rank FSM-annotated candidates through the live candidate-router API."""

        annotated = [self.abstract_candidate(row) for row in candidates]
        router = BoundedStrategyCandidateRouter(
            max_candidates=self.max_candidates,
            per_strategy_limit=1,
            suppress_repeated_coordinates=self.suppress_repeated_coordinates,
        )
        ranked = router.rank(frame, annotated, previous_frame=previous_frame)
        phase_counts = Counter(str(row.get("fsm_phase")) for row in ranked)
        diagnostics = dict(router.last_diagnostics)
        diagnostics["fsm_phase_counts"] = dict(sorted(phase_counts.items()))
        diagnostics["fsm_phase_order"] = [phase for phase, _field in self.phase_fields]
        self.last_diagnostics = diagnostics
        return [dict(row) for row in ranked if isinstance(row, Mapping)]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive parsing.
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):  # pragma: no cover - defensive parsing.
        return float(default)


def _level_label(level: int) -> str:
    return f"L{max(0, int(level))}"


def _read_json(path: Path) -> JsonDict:
    if not path.exists():  # pragma: no cover - missing-file closeout path.
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():  # pragma: no cover - missing-file closeout path.
        return ""
    return path.read_text(encoding="utf-8")


def _read_yaml(path: Path) -> JsonDict:
    if not path.exists():  # pragma: no cover - missing-file closeout path.
        return {"reproducible_total_levels": 0, "games": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {
        "reproducible_total_levels": 0,
        "games": [],
    }


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


def _target_marker(game: str, level: int) -> str:
    return f"{game}:{_level_label(level)}"


def _candidate_coordinate(candidate: Mapping[str, Any]) -> tuple[int, int] | None:
    data = candidate.get("data")
    if isinstance(data, Mapping) and "x" in data and "y" in data:
        return _as_int(data["x"]), _as_int(data["y"])
    if "x" in candidate and "y" in candidate:
        return _as_int(candidate["x"]), _as_int(candidate["y"])
    return None


def _candidate_signature(candidate: Mapping[str, Any]) -> str:
    coord = _candidate_coordinate(candidate)
    action = _as_int(candidate.get("action"))
    if coord is not None:
        return f"A{action}@{coord[0]},{coord[1]}"
    return f"A{action}"


def _selection_entropy(rows: Sequence[Mapping[str, Any]]) -> float:
    counts = Counter(_candidate_signature(row) for row in rows)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        probability = float(count) / float(total or 1)
        if probability:
            entropy -= probability * math.log2(probability)
    return float(entropy)


def recent_no_bank_targets(exp5548: Mapping[str, Any]) -> list[str]:
    """Extract recent no-bank target markers from Exp5548's honest-null artifact."""

    game = str(exp5548.get("selected_game") or "")
    level = str(exp5548.get("selected_level") or "")
    no_bank = bool(
        exp5548
        and game
        and level
        and _as_int(exp5548.get("registry_delta")) == 0
        and _as_int(exp5548.get("reproduced_levels")) == 0
        and exp5548.get("offline_reproduced") is not True
    )
    return [f"{game}:{level}"] if no_bank else []


def select_target(
    evidence: FsmPrecheckEvidence,
    *,
    target_candidates: Sequence[tuple[str, int]] = DEFAULT_TARGET_CANDIDATES,
) -> JsonDict:
    """REQ-ARC-FCP-5561: select a registry-safe target that avoids Exp5548's null."""

    registry = evidence.registry
    rows = _registry_rows(registry)
    recent_targets = set(recent_no_bank_targets(evidence.exp5548))
    avoided: list[str] = []
    audit: dict[str, JsonDict] = {}
    for game, target_level in target_candidates:
        target_level = int(target_level)
        marker = _target_marker(game, target_level)
        depth = _registry_depth(registry, game)
        already = bool(depth >= target_level)
        adjacent = bool(depth + 1 == target_level)
        if game not in rows:
            decision = "rejected_game_missing_from_registry"
        elif already:
            decision = "rejected_already_reproduced"
        elif not adjacent:
            decision = "rejected_not_adjacent_frontier"
        elif marker in recent_targets:
            decision = "rejected_recent_no_bank_target"
            avoided.append(marker)
        else:
            decision = "selected"
        audit[marker] = {
            "game": game,
            "registry_depth": int(depth),
            "target_level": int(target_level),
            "already_reproduced": already,
            "adjacent_frontier": adjacent,
            "recent_no_bank_target": marker in recent_targets,
            "decision": decision,
        }
        if decision == "selected":
            return {
                "blocked": False,
                "selected_game": game,
                "selected_level": _level_label(target_level),
                "target_level": int(target_level),
                "prior_levels_reproduced": int(depth),
                "registry_total_levels": _registry_total(registry),
                "registry_precheck_passed": True,
                "already_reproduced": False,
                "recent_no_bank_targets_avoided": list(dict.fromkeys(avoided)),
                "target_audit": audit,
                "selection_reason": "first_adjacent_target_after_recent_no_bank_rotation",
            }

    return {
        "blocked": True,
        "selected_game": "",
        "selected_level": "",
        "target_level": 0,
        "prior_levels_reproduced": 0,
        "registry_total_levels": _registry_total(registry),
        "registry_precheck_passed": False,
        "already_reproduced": False,
        "recent_no_bank_targets_avoided": list(dict.fromkeys(avoided)),
        "target_audit": audit,
        "blockers": ["no_registry_safe_adjacent_target_after_recent_no_bank_rotation"]
        if rows
        else ["registry_missing"],
        "selection_reason": "no_registry_safe_adjacent_target_after_recent_no_bank_rotation",
    }


def _fsm_probe_candidates(game: str, level: str, random_seed: int) -> list[JsonDict]:
    offset = (sum(ord(ch) for ch in f"{game}:{level}") + int(random_seed)) % 5
    x0 = 24 + (offset * 4)
    y0 = 20 + (offset * 3)
    return [
        {
            "label": "observe-top",
            "action": 6,
            "data": {"x": x0, "y": y0},
            "salience_score": 10.0,
            "effect_score": 1.0,
            "verifier_score": 1.0,
            "reset_score": 0.0,
        },
        {
            "label": "effect-top-same-coordinate",
            "action": 6,
            "data": {"x": x0, "y": y0},
            "salience_score": 9.0,
            "effect_score": 10.0,
            "verifier_score": 2.0,
            "reset_score": 0.0,
        },
        {
            "label": "verify-top-same-coordinate",
            "action": 6,
            "data": {"x": x0, "y": y0},
            "salience_score": 8.0,
            "effect_score": 2.0,
            "verifier_score": 10.0,
            "reset_score": 0.0,
        },
        {
            "label": "effect-transition",
            "action": 6,
            "data": {"x": x0 + 6, "y": y0},
            "salience_score": 7.0,
            "effect_score": 8.0,
            "verifier_score": 3.0,
            "reset_score": 1.0,
        },
        {
            "label": "verify-candidate",
            "action": 6,
            "data": {"x": x0 + 12, "y": y0},
            "salience_score": 6.0,
            "effect_score": 3.0,
            "verifier_score": 8.0,
            "reset_score": 1.0,
        },
        {
            "label": "reset-reinduce",
            "action": 6,
            "data": {"x": x0 + 18, "y": y0},
            "salience_score": 5.0,
            "effect_score": 4.0,
            "verifier_score": 4.0,
            "reset_score": 9.0,
        },
    ]


def run_fsm_precheck_probe(
    selected_game: str,
    selected_level: str,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> JsonDict:
    """SCENARIO-ARC-FCP-5561: dry-run FSM routing with coordinate suppression."""

    candidates = _fsm_probe_candidates(selected_game, selected_level, random_seed)
    unsuppressed = FSMActionAbstraction(
        max_candidates=4,
        suppress_repeated_coordinates=False,
    )
    unsuppressed_rows = unsuppressed.rank(None, candidates)
    abstraction = FSMActionAbstraction(
        max_candidates=4,
        suppress_repeated_coordinates=True,
    )
    selected_rows = abstraction.rank(None, candidates)
    diagnostics = dict(abstraction.last_diagnostics)
    suppressed = _as_int(diagnostics.get("suppressed_coordinate_count"))
    suppression_enabled = bool(
        diagnostics.get("selection_changed_by_suppression") and suppressed > 0
    )
    entropy = _selection_entropy(selected_rows)
    return {
        "selected_rows": selected_rows,
        "unsuppressed_selected_rows": unsuppressed_rows,
        "diagnostics": diagnostics,
        "suppressed_coordinate_count": int(suppressed),
        "repeated_coordinate_suppression_enabled": bool(suppression_enabled),
        "action_entropy_precheck": float(entropy),
        "fsm_action_abstraction_ready": bool(
            callable(getattr(abstraction, "rank", None))
            and len({row.get("fsm_phase") for row in selected_rows}) >= 3
        ),
    }


def fsm_live_path_reachability() -> JsonDict:  # pragma: no cover - live import-boundary probe.
    """Verify the FSM router can be supplied through the live candidate-router hook."""

    checks: dict[str, bool] = {}
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy, StepwiseExplorer
        from carnot.agentic.arc_graph_explore import rich_action_candidates

        router = FSMActionAbstraction()
        checks = {
            "fsm_router_has_rank_method": callable(getattr(router, "rank", None)),
            "stepwise_explorer_accepts_candidate_router": (
                "candidate_router" in inspect.signature(StepwiseExplorer.__init__).parameters
            ),
            "e3_policy_accepts_candidate_router": (
                "candidate_router" in inspect.signature(E3AgentPolicy.__init__).parameters
            ),
            "rich_action_candidates_accepts_candidate_router": (
                "candidate_router" in inspect.signature(rich_action_candidates).parameters
            ),
        }
        return {"ok": all(checks.values()), "checks": checks}
    except Exception as exc:
        return {
            "ok": False,
            "checks": checks,
            "error": f"{type(exc).__name__}: {exc}",
        }


def compute_reproducibility_checksum(
    *,
    selected_game: str,
    selected_level: str,
    registry_total: int,
    registry_depth: int,
    recent_no_bank_targets_avoided: Sequence[str],
    action_entropy_precheck: float,
) -> str:
    """Build a stable checksum for the inputs that define this precheck."""

    payload = {
        "action_entropy_precheck": float(action_entropy_precheck),
        "fsm_action_abstraction_ready": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "llm_invoked": False,
        "recent_no_bank_targets_avoided": list(recent_no_bank_targets_avoided),
        "registry_depth": int(registry_depth),
        "registry_total": int(registry_total),
        "repeated_coordinate_suppression_enabled": True,
        "selected_game": str(selected_game),
        "selected_level": str(selected_level),
        "solve_provenance": SOLVE_PROVENANCE,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_evidence(root: Path = REPO) -> FsmPrecheckEvidence:
    root = Path(root)
    return FsmPrecheckEvidence(
        registry=_read_yaml(root / REGISTRY_RELATIVE_PATH),
        exp5548=_read_json(root / EXP5548_RELATIVE_PATH),
    )


def build_precheck(
    evidence: FsmPrecheckEvidence,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
    tests_run: Sequence[str] = (),
    duration_s: float = 0.0,
    live_path_reachability: Mapping[str, Any] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> JsonDict:
    """SCENARIO-ARC-FCP-5561: build the no-credit FSM target precheck artifact."""

    selection = select_target(evidence)
    live_reachability = dict(live_path_reachability or fsm_live_path_reachability())
    selected_game = str(selection.get("selected_game") or "")
    selected_level = str(selection.get("selected_level") or "")
    if selected_game and selected_level:
        probe = run_fsm_precheck_probe(selected_game, selected_level, random_seed=random_seed)
    else:
        probe = {
            "selected_rows": [],
            "unsuppressed_selected_rows": [],
            "diagnostics": {},
            "suppressed_coordinate_count": 0,
            "repeated_coordinate_suppression_enabled": False,
            "action_entropy_precheck": 0.0,
            "fsm_action_abstraction_ready": False,
        }
    action_entropy = float(probe["action_entropy_precheck"])
    fsm_ready = bool(
        probe.get("fsm_action_abstraction_ready") is True
        and live_reachability.get("ok") is True
    )
    suppression_enabled = bool(probe.get("repeated_coordinate_suppression_enabled") is True)
    registry_passed = bool(selection.get("registry_precheck_passed") is True)
    blockers = list(selection.get("blockers") or [])
    if live_reachability.get("ok") is not True:
        blockers.append("fsm_action_abstraction_not_live_reachable")
    if not suppression_enabled:
        blockers.append("repeated_coordinate_suppression_not_enabled")
    if action_entropy < MIN_ACTION_ENTROPY:
        blockers.append("action_entropy_precheck_below_threshold")
    ready = bool(
        not selection.get("blocked")
        and registry_passed
        and fsm_ready
        and suppression_enabled
        and action_entropy >= MIN_ACTION_ENTROPY
    )
    checksum = compute_reproducibility_checksum(
        selected_game=selected_game,
        selected_level=selected_level,
        registry_total=_as_int(selection.get("registry_total_levels")),
        registry_depth=_as_int(selection.get("prior_levels_reproduced")),
        recent_no_bank_targets_avoided=list(selection.get("recent_no_bank_targets_avoided") or []),
        action_entropy_precheck=action_entropy,
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "carnot.experiment_5561_arc_fsm_target_rotation_precheck.v1",
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "status": "complete" if ready else "blocked",
        "llm_invoked": False,
        "no_model_specs_required": True,
        "selected_game": selected_game,
        "selected_level": selected_level,
        "registry_precheck_passed": registry_passed,
        "already_reproduced": bool(selection.get("already_reproduced", False)),
        "recent_no_bank_targets_avoided": list(
            selection.get("recent_no_bank_targets_avoided") or []
        ),
        "fsm_action_abstraction_ready": fsm_ready,
        "repeated_coordinate_suppression_enabled": suppression_enabled,
        "action_entropy_precheck": action_entropy,
        "solve_provenance": SOLVE_PROVENANCE,
        "arc_fsm_precheck_ready": ready,
        "tests_added_or_reused": list(tests_run or DEFAULT_TESTS_RUN),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"complete: {selected_game} {selected_level} FSM ARC precheck ready; no solve claimed"
            if ready
            else "blocked: " + ", ".join(blockers or ["arc_fsm_precheck_not_ready"])
        ),
        "reproducibility_checksum": checksum,
        "random_seed": int(random_seed),
        "registry_total_levels": _as_int(selection.get("registry_total_levels")),
        "prior_levels_reproduced": _as_int(selection.get("prior_levels_reproduced")),
        "target_selection": dict(selection),
        "target_audit": dict(selection.get("target_audit") or {}),
        "fsm_probe": {
            "selected_rows": list(probe.get("selected_rows") or []),
            "unsuppressed_selected_rows": list(probe.get("unsuppressed_selected_rows") or []),
            "diagnostics": dict(probe.get("diagnostics") or {}),
            "suppressed_coordinate_count": _as_int(probe.get("suppressed_coordinate_count")),
        },
        "live_path_reachability": live_reachability,
        "input_artifacts": [REGISTRY_RELATIVE_PATH, EXP5548_RELATIVE_PATH],
        "preconditions_checked": dict(preconditions_checked or {}),
        "duration_s": float(duration_s),
    }


def _checksum_looks_valid(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest)


def _verdict_claims_solve(verdict: str) -> bool:
    text = verdict.lower()
    return "solved" in text or "reproduced" in text or "banked" in text


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field: {field}" for field in REQUIRED_FIELDS if field not in artifact
    ]
    if artifact.get("llm_invoked") is not False:
        errors.append("llm_invoked must be false")
    if artifact.get("no_model_specs_required") is not True:
        errors.append("no_model_specs_required must be true")
    if not isinstance(artifact.get("selected_game"), str):
        errors.append("selected_game must be a string")
    if not isinstance(artifact.get("selected_level"), str):
        errors.append("selected_level must be a string")
    for field in (
        "registry_precheck_passed",
        "already_reproduced",
        "fsm_action_abstraction_ready",
        "repeated_coordinate_suppression_enabled",
        "arc_fsm_precheck_ready",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be bare bool")
    if not isinstance(artifact.get("recent_no_bank_targets_avoided"), list):
        errors.append("recent_no_bank_targets_avoided must be a list")
    if type(artifact.get("action_entropy_precheck")) is not float:
        errors.append("action_entropy_precheck must be bare float")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    tests = artifact.get("tests_added_or_reused")
    if not isinstance(tests, list) or not tests:
        errors.append("tests_added_or_reused must be a non-empty list")
    if not isinstance(artifact.get("field_principles"), Mapping):
        errors.append("field_principles must be a mapping")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be arc_live_path_precheck_no_llm")
    if "model_specs" in artifact:
        errors.append("model_specs must be omitted for no-LLM substrate")
    if "target_model" in artifact:
        errors.append("target_model must be omitted for no-LLM substrate")
    if "reproducibility_checksum" in artifact and not _checksum_looks_valid(
        artifact.get("reproducibility_checksum")
    ):
        errors.append("reproducibility_checksum must be a sha256 string")
    if artifact.get("arc_fsm_precheck_ready") is True:
        if artifact.get("already_reproduced") is not False:
            errors.append("ready artifacts require already_reproduced false")
        if artifact.get("registry_precheck_passed") is not True:
            errors.append("ready artifacts require registry_precheck_passed true")
        if artifact.get("fsm_action_abstraction_ready") is not True:
            errors.append("ready artifacts require fsm_action_abstraction_ready true")
        if artifact.get("repeated_coordinate_suppression_enabled") is not True:
            errors.append("ready artifacts require repeated_coordinate_suppression_enabled true")
        if not artifact.get("selected_game"):
            errors.append("ready artifacts require selected_game")
        if not artifact.get("selected_level"):
            errors.append("ready artifacts require selected_level")
        if (
            type(artifact.get("action_entropy_precheck")) is float
            and artifact["action_entropy_precheck"] < MIN_ACTION_ENTROPY
        ):
            errors.append("ready artifacts require action_entropy_precheck above threshold")
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
    live_path_reachability: Mapping[str, Any] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> JsonDict:
    started = time.monotonic()
    root = Path(root)
    spec_text = _read_text(root / SPEC_RELATIVE_PATH)
    preconditions = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "CLAUDE.md": (root / "CLAUDE.md").exists(),
        "registry_present": (root / REGISTRY_RELATIVE_PATH).exists(),
        "exp5548_present": (root / EXP5548_RELATIVE_PATH).exists(),
        "spec_has_req_5561": "REQ-ARC-FCP-5561" in spec_text,
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "llm_invoked": False,
        "model_specs_present": False,
        "solve_claimed": False,
    }
    artifact = build_precheck(
        load_evidence(root),
        preconditions_checked=preconditions,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
        live_path_reachability=live_path_reachability,
        random_seed=random_seed,
    )
    validate_artifact(artifact)
    _write_artifact(root, artifact)
    return artifact

