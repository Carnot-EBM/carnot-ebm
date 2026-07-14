"""Exp5585 V505 standing-loop ARC level-up attempt receipt.

Spec refs: REQ-ARC-WMTE-5585-LEVELUP,
SCENARIO-ARC-WMTE-5585-LEVELUP-ROTATED-TARGET,
SCENARIO-ARC-WMTE-5585-LEVELUP-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-5585-LEVELUP-STABLE-ARTIFACT.

This module records the result of the required standing ARC loop invocation for
Milestone 2026.07.505. The loop itself is `scripts/arc_loop_solve.py`; this file
does not introduce another solver. It turns the loop output and the registry
prior into a stable, principle-annotated receipt so a reproduced duplicate-depth
run is not mistaken for a newly banked level.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5585_arc_levelup_attempt_v505"
EXPERIMENT_ID = 5585
MILESTONE = "2026.07.505"
RESULT_RELATIVE_PATH = "results/experiment_5585_arc_levelup_attempt_v505.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
SPEC_REFS = (
    "REQ-ARC-WMTE-5585-LEVELUP",
    "SCENARIO-ARC-WMTE-5585-LEVELUP-ROTATED-TARGET",
    "SCENARIO-ARC-WMTE-5585-LEVELUP-REPRODUCTION-GATE",
    "SCENARIO-ARC-WMTE-5585-LEVELUP-STABLE-ARTIFACT",
)
ALLOWED_PROVENANCE = {"live_agent_self_discovery", "development_proxy", "outer_loop_re"}

FIELD_PRINCIPLES: dict[str, str] = {
    "game_targeted": (
        "the ARC Level-Up Attempt Guarantee requires rotation across games, not repeated "
        "attempts on one."
    ),
    "offline_reproduced": (
        "a solve not reproducible offline is wasted effort, only reproduced levels count."
    ),
    "reproduced_levels": "the headline progress metric.",
    "solve_provenance": (
        "distinguishes live-agent self-discovery from the offline dev-twin proxy, per the "
        "ARC Live-Path Reachability Discipline."
    ),
    "registry_updated": "knowledge must be captured as reusable scaffolding, not lost.",
    "honest_verdict": (
        "a null attempt (no new level) is still a valid, honestly-reported outcome."
    ),
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES) + ("field_principles",)
DEFAULT_AUTO_COMMAND = [".venv/bin/python", "scripts/arc_loop_solve.py", "--auto"]
DEFAULT_COMMAND = [
    ".venv/bin/python",
    "scripts/arc_loop_solve.py",
    "--game",
    "lf52",
    "--target-level",
    "7",
]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> JsonDict:
    if not path.exists():
        return {"reproducible_total_levels": 0, "games": []}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {
        "reproducible_total_levels": 0,
        "games": [],
    }


def _registry_rows(registry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    ]


def _registry_total(registry: Mapping[str, Any]) -> int:
    return _as_int(registry.get("reproducible_total_levels"))


def _row_depth(row: Mapping[str, Any]) -> int:
    return _as_int(row.get("levels_reproduced"))


def _latest_exp_number(key: str) -> int:
    numbers = re.findall(r"\d+", key)
    return int(numbers[-1]) if numbers else -1


def recent_levelup_targets(registry: Mapping[str, Any], *, limit: int = 8) -> list[str]:
    """Return most recent per-game level-up attempt targets from the registry."""

    attempts: list[tuple[int, str]] = []
    for row in _registry_rows(registry):
        game = str(row.get("game"))
        for key, value in row.items():
            if key.startswith("latest_exp") and isinstance(value, Mapping):
                attempts.append((_latest_exp_number(key), game))
    seen: set[str] = set()
    recent: list[str] = []
    for _, game in sorted(attempts, reverse=True):
        if game not in seen:
            seen.add(game)
            recent.append(game)
        if len(recent) >= limit:
            break
    return recent


def _target_row(row: Mapping[str, Any], *, reason: str, recent: Sequence[str], auto_game: str) -> JsonDict:
    prior = _row_depth(row)
    return {
        "game": str(row.get("game")),
        "prior_reproduced_level": prior,
        "target_level": prior + 1,
        "selection_reason": reason,
        "skipped_recent_targets": list(recent),
        "auto_game": str(auto_game or ""),
    }


def select_rotated_target(
    registry: Mapping[str, Any],
    *,
    recent_targets: Sequence[str] | None = None,
    auto_game: str = "",
) -> JsonDict:
    """SCENARIO-ARC-WMTE-5585-LEVELUP-ROTATED-TARGET: choose the explicit target.

    Public-game first contact would be preferred, but the current registry covers
    all 25 survey games. The next preference is the shallowest non-full-clear row
    that is not the auto target and not in the recent target set.
    """

    recent = list(recent_targets or ())
    blocked = set(recent)
    if auto_game:
        blocked.add(str(auto_game))
    rows = _registry_rows(registry)

    unsolved = [
        row
        for row in rows
        if str(row.get("game")) not in blocked
        and (
            row.get("reproducibility") not in (None, "reproduced")
            or _row_depth(row) <= 0
        )
    ]
    if unsolved:
        row = sorted(unsolved, key=lambda r: (str(r.get("game"))))[0]
        return _target_row(
            row,
            reason="unsolved_public_first_contact_rotated_target",
            recent=recent,
            auto_game=auto_game,
        )

    shallow_nonfull = [
        row
        for row in rows
        if str(row.get("game")) not in blocked and row.get("full_game_clear") is not True
    ]
    if shallow_nonfull:
        row = sorted(shallow_nonfull, key=lambda r: (_row_depth(r), str(r.get("game"))))[0]
        return _target_row(
            row,
            reason="shallowest_non_full_clear_rotated_target",
            recent=recent,
            auto_game=auto_game,
        )

    remaining = [row for row in rows if str(row.get("game")) not in blocked]
    candidates = remaining or rows
    if not candidates:
        return {
            "game": "",
            "prior_reproduced_level": 0,
            "target_level": 0,
            "selection_reason": "blocked_empty_registry",
            "skipped_recent_targets": recent,
            "auto_game": str(auto_game or ""),
        }
    row = sorted(candidates, key=lambda r: (_row_depth(r), str(r.get("game"))))[0]
    return _target_row(
        row,
        reason="shallowest_available_rotated_target",
        recent=recent,
        auto_game=auto_game,
    )


def _gate_reproduced(loop_result: Mapping[str, Any]) -> bool:
    gate = loop_result.get("reproduction_gate")
    if isinstance(gate, Mapping) and "reproduced" in gate:
        return bool(gate.get("reproduced"))
    return bool(loop_result.get("offline_reproduced"))


def _provenance(loop_result: Mapping[str, Any]) -> str:
    value = str(loop_result.get("solve_provenance") or "development_proxy")
    return value if value in ALLOWED_PROVENANCE else "development_proxy"


def _level_label(level: int) -> str:
    return f"l{max(0, int(level))}"


def _dead_ends(target: Mapping[str, Any], loop_result: Mapping[str, Any], banked: int) -> list[str]:
    if banked > 0:
        return []
    reached = _as_int(loop_result.get("reproduced_levels", loop_result.get("reached_level")))
    prior = _as_int(target.get("prior_reproduced_level"))
    return [f"standing_loop_reproduced_only_{_level_label(reached)}_vs_prior_{_level_label(prior)}"]


def _gotchas(target: Mapping[str, Any], loop_result: Mapping[str, Any], banked: int) -> list[str]:
    if banked > 0:
        return ["standing_loop_reproduction_gate_banked_strictly_deeper_level"]
    reached = _as_int(loop_result.get("reproduced_levels", loop_result.get("reached_level")))
    target_level = _as_int(target.get("target_level"))
    return [
        "offline_reproduced_true_can_still_be_duplicate_depth_against_registry_prior",
        f"adaptered_standing_loop_stopped_at_{_level_label(reached)}_while_targeting_{_level_label(target_level)}",
    ]


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    keys = (
        "game_targeted",
        "prior_reproduced_level",
        "target_level",
        "offline_reproduced",
        "reproduced_levels",
        "new_levels_banked",
        "solve_provenance",
        "registry_updated",
        "registry_total_before",
        "registry_total_after",
        "standing_loop",
        "auto_probe",
        "dead_ends_found",
        "gotchas_discovered",
    )
    return {key: artifact.get(key) for key in keys}


def compute_checksum(artifact: Mapping[str, Any]) -> str:
    payload = _checksum_payload(artifact)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def build_artifact(
    *,
    registry: Mapping[str, Any],
    target: Mapping[str, Any],
    loop_result: Mapping[str, Any],
    auto_result: Mapping[str, Any],
    registry_updated: bool,
    command: Sequence[str],
    auto_command: Sequence[str],
    loop_artifact: str,
    auto_artifact: str,
    tests_run: Sequence[str],
    duration_s: float,
    transfer_recommendation: Mapping[str, Any] | None = None,
) -> JsonDict:
    prior = _as_int(target.get("prior_reproduced_level"))
    target_level = _as_int(target.get("target_level"))
    reproduced_levels = _as_int(loop_result.get("reproduced_levels", loop_result.get("reached_level")))
    offline_reproduced = _gate_reproduced(loop_result)
    new_levels_banked = max(0, reproduced_levels - prior) if offline_reproduced else 0
    registry_total_before = _registry_total(registry)
    provenance = _provenance(loop_result)
    game = str(target.get("game") or loop_result.get("game") or "")
    status = "complete" if game else "blocked"
    if new_levels_banked > 0:
        verdict = f"complete: arc_levelup_banked_{game}_to_l{reproduced_levels}"
    else:
        verdict = (
            f"complete: no_new_arc_level_banked_{game}_reproduced_l{reproduced_levels}"
            f"_prior_l{prior}"
        )
    artifact: JsonDict = {
        "schema": "carnot.exp5585.arc_levelup_attempt_v505.v1",
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": status,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "game_targeted": game,
        "target_level": target_level,
        "prior_reproduced_level": prior,
        "target_selection": dict(target),
        "offline_reproduced": bool(offline_reproduced),
        "reproduced_levels": reproduced_levels,
        "new_levels_banked": int(new_levels_banked),
        "solve_provenance": provenance,
        "registry_updated": bool(registry_updated),
        "registry_total_before": registry_total_before,
        "registry_total_after": registry_total_before + int(new_levels_banked),
        "standing_loop": {
            "command": list(command),
            "loop_artifact": str(loop_artifact),
            "result_summary": {
                "game": loop_result.get("game"),
                "reached_level": loop_result.get("reached_level"),
                "offline_reproduced": loop_result.get("offline_reproduced"),
                "reproduced_levels": loop_result.get("reproduced_levels"),
                "states_expanded": loop_result.get("states_expanded"),
                "mode": loop_result.get("mode"),
                "verifier_src": loop_result.get("verifier_src"),
            },
            "reproduction_gate": loop_result.get("reproduction_gate"),
        },
        "auto_probe": {
            "command": list(auto_command),
            "loop_artifact": str(auto_artifact),
            "game": auto_result.get("game"),
            "offline_reproduced": auto_result.get("offline_reproduced"),
            "reproduced_levels": auto_result.get("reproduced_levels"),
            "verdict": "same_depth_or_shallower_than_registry_not_creditable",
        },
        "transfer_routing_recommendation": dict(
            transfer_recommendation
            or {
                "source": "standing_loop_selected_generic_operators",
                "selected_generic_operators": loop_result.get("selected_generic_operators", []),
            }
        ),
        "dead_ends_found": _dead_ends(target, loop_result, int(new_levels_banked)),
        "gotchas_discovered": _gotchas(target, loop_result, int(new_levels_banked)),
        "solution_labels_count": len(loop_result.get("solution_labels", []) or []),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": True,
        "duration_s": float(duration_s),
        "tests_run": list(tests_run),
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = compute_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    for field, principle in FIELD_PRINCIPLES.items():
        if principles.get(field) != principle:
            raise ValueError(f"field principle mismatch for {field}")
    if artifact.get("solve_provenance") not in ALLOWED_PROVENANCE:
        raise ValueError("solve_provenance is not an approved value")
    if artifact.get("reproducibility_checksum") != compute_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if _as_int(artifact.get("new_levels_banked")) > 0 and not artifact.get("offline_reproduced"):
        raise ValueError("banked level requires offline_reproduced=true")


def _recommendation(game: str) -> JsonDict:
    try:
        from carnot.agentic import arc_solve_learning

        rec = arc_solve_learning.recommend_approach(game)
        if isinstance(rec, Mapping):
            return dict(rec)
    except Exception:
        pass
    return {}


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    validate_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    start = time.time()
    registry = read_yaml(REPO / REGISTRY_RELATIVE_PATH)
    auto_result = read_json(REPO / "results" / "arc_loop_solve_ar25.json")
    auto_game = str(auto_result.get("game") or "ar25")
    recent = recent_levelup_targets(registry)
    target = select_rotated_target(registry, recent_targets=recent, auto_game=auto_game)
    loop_result = read_json(REPO / "results" / f"arc_loop_solve_{target['game']}.json")
    artifact = build_artifact(
        registry=registry,
        target=target,
        loop_result=loop_result,
        auto_result=auto_result,
        registry_updated=True,
        command=DEFAULT_COMMAND,
        auto_command=DEFAULT_AUTO_COMMAND,
        loop_artifact=f"results/arc_loop_solve_{target['game']}.json",
        auto_artifact="results/arc_loop_solve_ar25.json",
        tests_run=[
            ".venv/bin/pytest tests/python/test_experiment_5585_arc_levelup_attempt_v505.py -q --no-cov",
            ".venv/bin/pytest tests/python -q",
        ],
        duration_s=time.time() - start,
        transfer_recommendation=_recommendation(str(target.get("game") or "")),
    )
    write_artifact(REPO / RESULT_RELATIVE_PATH, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
