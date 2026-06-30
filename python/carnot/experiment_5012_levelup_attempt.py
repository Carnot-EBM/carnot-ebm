"""Experiment 5012: opportunistic ARC level-up attempt.

Spec refs: REQ-ARC-WMTE-5012,
SCENARIO-ARC-WMTE-5012-ROTATED-TARGET-PRECHECK,
SCENARIO-ARC-WMTE-5012-NO-GROUNDED-DELTA,
SCENARIO-ARC-WMTE-5012-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-5012-STABLE-ARTIFACT.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO / "python"))

EXPERIMENT = "experiment_5012_levelup_attempt"
EXPERIMENT_ID = 5012
SCHEMA = "carnot.arc_levelup_attempt_5012.v1"
RESULT_RELATIVE_PATH = "results/experiment_5012_levelup_attempt.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 5012
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

LEVELUP_CANDIDATES = (
    ("sc25", "l5_to_l6", 5),
    ("cn04", "l3_to_l4", 3),
    ("lp85", "l5_to_l6", 5),
)
RECENT_ROTATION_TARGETS = (
    "tn36",
    "g50t",
    "tu93",
    "bp35",
    "tr87",
    "s5i5",
    "ar25",
    "vc33",
    "lf52",
    "sb26",
    "sp80",
    "su15",
    "m0r0",
    "dc22",
)
HIDDEN_STATE_TARGETS = ("ka59", "wa30")
E2_TARGET = "r11l"
HARD_EXCLUDED_TARGETS = tuple(
    sorted(set(RECENT_ROTATION_TARGETS) | set(HIDDEN_STATE_TARGETS) | {E2_TARGET})
)
TARGET_ROTATION = "sc25_recorded_l6_dead_end_then_cn04_l4_before_lp85_alt"
SPEC_REFS = [
    "REQ-ARC-WMTE-5012",
    "SCENARIO-ARC-WMTE-5012-ROTATED-TARGET-PRECHECK",
    "SCENARIO-ARC-WMTE-5012-NO-GROUNDED-DELTA",
    "SCENARIO-ARC-WMTE-5012-REPRODUCTION-GATE",
    "SCENARIO-ARC-WMTE-5012-STABLE-ARTIFACT",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; banked is success_<game>_levelup_banked, no-bank is "
            "complete_<game>_no_new_level_residual_<cause>."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- the agent advanced via its own attempts/runtime "
            "RE; NOT outer_loop_re (CRITICAL) and NOT a re-solve of an already-banked "
            "level (duplicate CRITICAL)."
        )
    },
    "target_game": {
        "principle": (
            "the rotated FRESH grounded deepen target (differs from E2's r11l and "
            "the recently-attempted lanes)."
        )
    },
    "offline_reproduced": {
        "principle": "only reproduced levels count toward reproducible_total_levels."
    },
    "reproduced_levels": {
        "principle": "the new reproducible depth; the monotonic ARC progress metric."
    },
    "new_levels_banked": {
        "principle": (
            ">=1 for a PASS; 0 records the honest rotation dead-end (the deepen well "
            "is dry across all regimes)."
        )
    },
    "live_path_reachable": {
        "principle": (
            "true -- the deepen runs through arc_loop_solve + a live GameAdapter "
            "(arc_orphan_solver_lint passes)."
        )
    },
    "verifier_is_oracle": {
        "principle": "the reproduction gate is the executable oracle (circularity discipline)."
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates for an offline search/gate run "
            "(1s floor); live_llm_inference only if induction ran >=60s."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/env/generator checks; a missing resource emits blocked_, never "
            "a fabricated solve."
        )
    },
    "random_seed": {"principle": "determinism for the offline search."},
    "reproducibility_checksum": {
        "principle": ("content hash of (game, plan, claimed level) so a replication catches drift.")
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "candidate_selection",
    "approach_recommendation",
    "dead_ends_consulted",
    "delta_status",
    "registry_update",
    "standing_loop_command",
    "standing_loop_result_path",
    "standing_loop_ran",
    "reproduction_gate",
    "solution_labels",
    "retire_if_same_verdict",
    "schema_errors",
)


def standing_loop_command(game: str, target_level: int) -> str:
    return f".venv/bin/python scripts/arc_loop_solve.py --game {game} --target-level {int(target_level)}"


def standing_loop_result_path(game: str) -> str:
    return f"results/arc_loop_solve_{game}.json"


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


def _nested_dead_end_notes(value: Any) -> list[str]:
    notes: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key != "dead_ends" and "dead_end" in str(key):
                notes.append(str(item))
            notes.extend(_nested_dead_end_notes(item))
    elif isinstance(value, list):
        for item in value:
            notes.extend(_nested_dead_end_notes(item))
    return notes


def _dead_ends(row: Mapping[str, Any]) -> list[str]:
    values = row.get("dead_ends") or []
    if not isinstance(values, list):
        values = [values]
    rendered = [str(value) for value in values]
    for note in _nested_dead_end_notes(row):
        if note not in rendered:
            rendered.append(note)
    return rendered


def _has_recorded_next_level_dead_end(
    game: str,
    row: Mapping[str, Any],
    target_level: int,
) -> bool:
    tokens = (
        f"no_grounded_l{int(target_level)}_delta",
        f"no grounded l{int(target_level)} delta",
        f"target-level {int(target_level)} replays to l{int(target_level) - 1}",
        "no grounded next-level",
        "no_grounded_next_level",
        "duplicate_depth",
        "same-depth",
        "same depth",
        "no-bank",
    )
    for item in _dead_ends(row):
        lowered = item.lower()
        if lowered.startswith(f"exp{EXPERIMENT_ID} "):
            continue
        if "retired" in lowered or "filled" in lowered:
            continue
        if game not in lowered:
            continue
        if any(token in lowered for token in tokens):
            return True
    return False


def adapter_for(game: str):  # pragma: no cover - wrapper kept injectable for tests
    from carnot.agentic import arc_game_adapters

    return arc_game_adapters.get_adapter(game)


def recommend_approach(
    game: str,
) -> dict[str, Any]:  # pragma: no cover - wrapper kept injectable for tests
    from carnot.agentic import arc_solve_learning

    return dict(arc_solve_learning.recommend_approach(game))


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
    except Exception:
        return False
    return True


def run_standing_loop(
    root: Path, game: str, target_level: int
) -> dict[str, Any]:  # pragma: no cover - subprocess boundary
    subprocess.run(
        [
            ".venv/bin/python",
            "scripts/arc_loop_solve.py",
            "--game",
            game,
            "--target-level",
            str(target_level),
        ],
        cwd=root,
        check=True,
    )
    return json.loads((root / standing_loop_result_path(game)).read_text(encoding="utf-8"))


def grounded_delta_status(game: str, *, prior_level: int, adapter: Any | None) -> dict[str, Any]:
    target_level = int(prior_level) + 1
    if adapter is None:
        return {
            "game": game,
            "prior_level": int(prior_level),
            "target_level": target_level,
            "grounded_next_level_delta": False,
            "reason": f"no_grounded_l{target_level}_delta",
            "live_path_reachable": False,
            "adapter_registered": False,
            "adapter_level_tails": [],
        }
    tails = getattr(adapter, "level_tails", {}) or {}
    tail_keys: list[int] = []
    for key in tails:
        try:
            tail_keys.append(int(key))
        except (TypeError, ValueError):
            continue
    has_tail = bool(tails.get(target_level) or tails.get(str(target_level)))
    return {
        "game": game,
        "prior_level": int(prior_level),
        "target_level": target_level,
        "grounded_next_level_delta": has_tail,
        "reason": "grounded_delta_available" if has_tail else f"no_grounded_l{target_level}_delta",
        "live_path_reachable": True,
        "adapter_registered": True,
        "adapter_game": getattr(adapter, "game", game),
        "adapter_level_tails": sorted(tail_keys),
    }


def select_target(
    registry: Mapping[str, Any],
    *,
    adapter_lookup: Callable[[str], Any | None] | None = None,
    recommend_fn: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = _game_rows(registry)
    lookup = adapter_lookup or adapter_for
    recommender = recommend_fn or recommend_approach
    audit: list[dict[str, Any]] = []
    selected_index: int | None = None
    selected_status = "no_candidate"
    selected_reason = "no_fresh_grounded_candidate"

    for index, (game, lane, required_prior) in enumerate(LEVELUP_CANDIDATES):
        row = rows.get(game, {})
        prior = int(row.get("levels_reproduced") or 0)
        target_level = prior + 1 if prior else required_prior + 1
        recorded_dead_end = _has_recorded_next_level_dead_end(game, row, target_level)
        adapter = None
        delta = {"grounded_next_level_delta": False, "reason": "not_checked"}
        status = "candidate_unselected"
        reason = "lower_priority_candidate"

        if not row:
            status, reason = "skip_missing_registry_row", "missing_registry_row"
        elif prior != required_prior:
            status, reason = "skip_wrong_prior_depth", f"requires_l{required_prior}_prior"
        elif recorded_dead_end:
            status, reason = "skip_recorded_dead_end", f"recorded_l{target_level}_dead_end"
        elif selected_index is not None:
            status, reason = "alternate_not_selected", "higher_priority_candidate_already_selected"
        else:
            adapter = lookup(game)
            delta = grounded_delta_status(game, prior_level=prior, adapter=adapter)
            if delta.get("grounded_next_level_delta") is True:
                status, reason = "candidate_grounded_delta", "fresh_rotation_grounded_delta"
                selected_index = index
                selected_status = "selected"
                selected_reason = reason
            else:
                status = "candidate_no_grounded_delta"
                reason = str(delta.get("reason") or f"no_grounded_l{target_level}_delta")
                selected_index = index
                selected_status = "selected_no_grounded_delta"
                selected_reason = "fresh_rotation_" + reason

        audit.append(
            {
                "game": game,
                "lane": lane,
                "prior_level": prior,
                "target_level": target_level,
                "status": status,
                "reason": reason,
                "has_recorded_next_level_dead_end": recorded_dead_end,
                "dead_ends_consulted": _dead_ends(row),
                "delta_status": dict(delta),
                "adapter_registered": adapter is not None,
            }
        )

    base = {
        "candidate_audit": audit,
        "candidate_order": [game for game, _lane, _prior in LEVELUP_CANDIDATES],
        "excluded_recent_targets": list(RECENT_ROTATION_TARGETS),
        "hidden_state_targets_avoided": list(HIDDEN_STATE_TARGETS),
        "e2_target_avoided": E2_TARGET,
    }
    if selected_index is None:
        return {
            "game": "none",
            "lane": "none",
            "prior_level": 0,
            "target_level": 0,
            "status": selected_status,
            "reason": selected_reason,
            "dead_ends_consulted": [],
            "delta_status": {},
            "adapter_registered": False,
            "approach_recommendation": {},
            **base,
        }

    selected = dict(audit[selected_index])
    game = str(selected["game"])
    selected.update(
        {
            "status": selected_status,
            "reason": selected_reason,
            "approach_recommendation": dict(recommender(game)),
            **base,
        }
    )
    return selected


def _loop_reached_level(loop_result: Mapping[str, Any] | None) -> int:
    if not loop_result:
        return 0
    gate = loop_result.get("reproduction_gate")
    gate = gate if isinstance(gate, Mapping) else {}
    return int(gate.get("reached_level") or loop_result.get("reached_level") or 0)


def _loop_reproduced(loop_result: Mapping[str, Any] | None) -> bool:
    if not loop_result:
        return False
    gate = loop_result.get("reproduction_gate")
    gate = gate if isinstance(gate, Mapping) else {}
    if "reproduced" in gate:
        return bool(
            loop_result.get("offline_reproduced") is True and gate.get("reproduced") is True
        )
    return bool(loop_result.get("offline_reproduced") is True)


def _loop_live_path(loop_result: Mapping[str, Any] | None) -> bool:
    if not loop_result:
        return False
    if loop_result.get("status") == "needs_per_game_RE":
        return False
    return str(loop_result.get("mode") or "").startswith("standing_arc_loop")


def _residual_reason(
    *,
    prior_level: int,
    delta_status: Mapping[str, Any],
    loop_result: Mapping[str, Any] | None,
) -> str:
    if delta_status.get("grounded_next_level_delta") is not True:
        return str(delta_status.get("reason") or f"no_grounded_l{int(prior_level) + 1}_delta")
    if _loop_reproduced(loop_result) and _loop_reached_level(loop_result) <= int(prior_level):
        return "duplicate_depth"
    if not _loop_reproduced(loop_result):
        return "offline_reproduction_failed"
    if not _loop_live_path(loop_result):
        return "live_path_unreachable"
    return "unknown"


def _registry_update_payload(
    *,
    target_game: str,
    prior_level: int,
    prior_total_levels: int,
    reproduced_levels: int,
    new_levels_banked: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "updated": True,
        "path": REGISTRY_RELATIVE_PATH,
        "target_game": target_game,
        "prior_game_levels": int(prior_level),
        "new_game_levels": int(reproduced_levels if new_levels_banked else prior_level),
        "banked_levels": int(new_levels_banked),
        "prior_total_declared": int(prior_total_levels),
        "new_total_declared": int(prior_total_levels) + int(new_levels_banked),
        "reason": "banked_offline_reproduced_level" if new_levels_banked else reason,
    }


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    checksum_payload = {
        "experiment": payload.get("experiment"),
        "target_game": payload.get("target_game"),
        "prior_reproduced_level": payload.get("prior_reproduced_level"),
        "target_level": payload.get("target_level"),
        "reproduced_levels": payload.get("reproduced_levels"),
        "new_levels_banked": payload.get("new_levels_banked"),
        "offline_reproduced": payload.get("offline_reproduced"),
        "candidate_selection": payload.get("candidate_selection"),
        "approach_recommendation": payload.get("approach_recommendation"),
        "dead_ends_consulted": payload.get("dead_ends_consulted"),
        "delta_status": payload.get("delta_status"),
        "solution_labels": payload.get("solution_labels"),
        "reproduction_gate": payload.get("reproduction_gate"),
        "registry_update": payload.get("registry_update"),
        "random_seed": payload.get("random_seed"),
    }
    raw = json.dumps(checksum_payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_artifact(
    *,
    selection: Mapping[str, Any],
    prior_total_levels: int,
    preconditions_checked: Mapping[str, Any],
    loop_result: Mapping[str, Any] | None,
    duration_s: float,
) -> dict[str, Any]:
    target_game = str(selection.get("game") or "none")
    prior_level = int(selection.get("prior_level") or 0)
    target_level = int(selection.get("target_level") or prior_level + 1)
    delta_status = dict(selection.get("delta_status") or {})
    reached = _loop_reached_level(loop_result)
    reproduced = _loop_reproduced(loop_result)
    banked = max(0, reached - prior_level) if reproduced else 0
    success = bool(
        banked >= 1
        and delta_status.get("grounded_next_level_delta") is True
        and _loop_live_path(loop_result)
    )
    reason = (
        "banked_offline_reproduced_level"
        if success
        else _residual_reason(
            prior_level=prior_level,
            delta_status=delta_status,
            loop_result=loop_result,
        )
    )
    reproduced_levels = reached if success else prior_level
    new_levels_banked = banked if success else 0
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": (
            f"success_{target_game}_levelup_banked"
            if success
            else f"complete_{target_game}_no_new_level_residual_{reason}"
        ),
        "solve_provenance": SOLVE_PROVENANCE,
        "target_game": target_game,
        "offline_reproduced": bool(success),
        "reproduced_levels": int(reproduced_levels),
        "new_levels_banked": int(new_levels_banked),
        "live_path_reachable": True if success else bool(delta_status.get("live_path_reachable")),
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "prior_reproduced_level": prior_level,
        "target_level": target_level,
        "reproducible_total_levels_before": int(prior_total_levels),
        "reproducible_total_levels_after": int(prior_total_levels) + int(new_levels_banked),
        "candidate_selection": dict(selection),
        "approach_recommendation": dict(selection.get("approach_recommendation") or {}),
        "dead_ends_consulted": list(selection.get("dead_ends_consulted") or []),
        "delta_status": delta_status,
        "registry_update": _registry_update_payload(
            target_game=target_game,
            prior_level=prior_level,
            prior_total_levels=prior_total_levels,
            reproduced_levels=reproduced_levels,
            new_levels_banked=new_levels_banked,
            reason=reason,
        ),
        "standing_loop_command": standing_loop_command(target_game, target_level),
        "standing_loop_result_path": standing_loop_result_path(target_game),
        "standing_loop_ran": loop_result is not None,
        "reproduction_gate": dict((loop_result or {}).get("reproduction_gate") or {}),
        "solution_labels": list((loop_result or {}).get("solution_labels") or []),
        "solution": list((loop_result or {}).get("solution") or []),
        "states_expanded": int((loop_result or {}).get("states_expanded") or 0),
        "retire_if_same_verdict": not success,
        "duration_s": float(duration_s),
        "reproducibility_checksum": "",
        "schema_errors": [],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def blocked_artifact(
    *,
    target_game: str,
    target_level: int,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    selection: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    prior_level = int(selection.get("prior_level") or 0)
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": f"blocked_{target_game}_{reason}",
        "solve_provenance": SOLVE_PROVENANCE,
        "target_game": target_game,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "new_levels_banked": 0,
        "live_path_reachable": False,
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "prior_reproduced_level": prior_level,
        "target_level": int(target_level),
        "reproducible_total_levels_before": 0,
        "reproducible_total_levels_after": 0,
        "candidate_selection": dict(selection),
        "approach_recommendation": dict(selection.get("approach_recommendation") or {}),
        "dead_ends_consulted": list(selection.get("dead_ends_consulted") or []),
        "delta_status": dict(selection.get("delta_status") or {}),
        "registry_update": {"updated": False, "banked_levels": 0, "reason": reason},
        "standing_loop_command": standing_loop_command(target_game, target_level),
        "standing_loop_result_path": standing_loop_result_path(target_game),
        "standing_loop_ran": False,
        "reproduction_gate": {},
        "solution_labels": [],
        "solution": [],
        "states_expanded": 0,
        "retire_if_same_verdict": True,
        "duration_s": float(duration_s),
        "reproducibility_checksum": "",
        "schema_errors": [],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


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
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if artifact.get("target_game") in HARD_EXCLUDED_TARGETS:
        errors.append("target_game violates hard exclusions")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if type(artifact.get("new_levels_banked")) is not int:
        errors.append("new_levels_banked must be bare int")
    if type(artifact.get("live_path_reachable")) is not bool:
        errors.append("live_path_reachable must be bare bool")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
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
    if verdict.startswith("success_"):
        if artifact.get("offline_reproduced") is not True:
            errors.append("success requires offline_reproduced true")
        if int(artifact.get("new_levels_banked") or 0) < 1:
            errors.append("success requires new_levels_banked >= 1")
        if int(artifact.get("reproduced_levels") or 0) <= int(
            artifact.get("prior_reproduced_level") or 0
        ):
            errors.append("success requires reproduced_levels > prior_reproduced_level")
        if artifact.get("live_path_reachable") is not True:
            errors.append("success requires live_path_reachable true")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        errors.append("reproducibility_checksum must be 64 hex chars")
    elif checksum != reproducibility_checksum(artifact):
        errors.append("checksum mismatch")
    return errors


def _game_block_bounds(registry_text: str, game: str) -> tuple[int, int]:
    marker = f"- game: {game}\n"
    start = registry_text.index(marker)
    next_match = re.search(
        r"\n(?=(?:- game:|[A-Za-z0-9_]+:))",
        registry_text[start + len(marker) :],
    )
    end = start + len(marker) + next_match.start() + 1 if next_match else len(registry_text)
    return start, end


def _quote_yaml_scalar(value: str) -> str:
    return yaml.safe_dump([value], sort_keys=False, width=1000).strip()[2:]


def _artifact_residual_reason(artifact: Mapping[str, Any]) -> str:
    verdict = str(artifact.get("honest_verdict") or "")
    marker = "_no_new_level_residual_"
    if marker in verdict:
        return verdict.split(marker, 1)[1]
    return str((artifact.get("registry_update") or {}).get("reason") or "unknown")


def _latest_registry_block(
    artifact: Mapping[str, Any], reproduced_levels: int, banked: int
) -> list[str]:
    game = str(artifact["target_game"])
    payload = {
        "latest_exp5012_levelup_attempt": {
            "artifact": RESULT_RELATIVE_PATH,
            "loop_artifact": standing_loop_result_path(game),
            "offline_reproduced": bool(artifact.get("offline_reproduced")),
            "reproduced_levels": int(reproduced_levels),
            "new_levels_banked": int(banked),
            "residual_cause": _artifact_residual_reason(artifact) if not banked else None,
            "solve_provenance": SOLVE_PROVENANCE,
            "target_rotation": TARGET_ROTATION,
            "reproducibility_checksum": artifact.get("reproducibility_checksum"),
        }
    }
    return [
        "  " + line for line in yaml.safe_dump(payload, sort_keys=False, width=1000).splitlines()
    ]


def _remove_existing_latest_block(lines: list[str]) -> list[str]:
    try:
        start = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("  latest_exp5012_levelup_attempt:")
        )
    except StopIteration:
        return lines
    end = start + 1
    while end < len(lines) and (lines[end].startswith("    ") or not lines[end].strip()):
        end += 1
    return lines[:start] + lines[end:]


def _append_dead_end(lines: list[str], note: str) -> list[str]:
    rendered_note = "  - " + _quote_yaml_scalar(note)
    for i, line in enumerate(lines):
        if line.strip() == "dead_ends: []":
            return lines[:i] + ["  dead_ends:", rendered_note] + lines[i + 1 :]
        if line.startswith("  dead_ends:"):
            insert_at = i + 1
            while insert_at < len(lines) and not re.match(r"^  [A-Za-z0-9_]+:", lines[insert_at]):
                insert_at += 1
            if rendered_note not in lines[i:insert_at]:
                return lines[:insert_at] + [rendered_note] + lines[insert_at:]
            return lines
    return [*lines, "  dead_ends:", rendered_note]


def _replace_game_row(
    registry_text: str,
    game: str,
    *,
    total_levels: int,
    reproduced_levels: int,
    banked: int,
    dead_end_note: str | None,
    artifact: Mapping[str, Any],
) -> str:
    start, end = _game_block_bounds(registry_text, game)
    lines = registry_text[start:end].rstrip("\n").splitlines()
    if banked:
        lines = [
            re.sub(r"^(  levels_reproduced:\s*)\d+\s*$", rf"\g<1>{int(reproduced_levels)}", line)
            for line in lines
        ]
    elif dead_end_note is not None:
        lines = _append_dead_end(lines, dead_end_note)
    lines = _remove_existing_latest_block(lines)
    lines.extend(_latest_registry_block(artifact, reproduced_levels, banked))
    block = "\n".join(lines) + "\n"
    updated = registry_text[:start] + block + registry_text[end:]
    updated = re.sub(
        r"(?m)^(reproducible_total_levels:\s*)\d+\s*$",
        rf"\g<1>{int(total_levels)}",
        updated,
        count=1,
    )
    return re.sub(r"(?m)^(updated:\s*).*$", r"\g<1>'2026-06-30'", updated, count=1)


def apply_registry_result(
    registry_text: str, *, artifact: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    registry = _load_registry(registry_text)
    rows = _game_rows(registry)
    game = str(artifact["target_game"])
    row = rows[game]
    prior_total = int(
        artifact.get("reproducible_total_levels_before")
        or registry.get("reproducible_total_levels")
        or 0
    )
    banked = int(artifact.get("new_levels_banked") or 0)
    reproduced_levels = int(artifact.get("reproduced_levels") or row.get("levels_reproduced") or 0)
    dead_end_note = None
    if not banked:
        residual = _artifact_residual_reason(artifact)
        dead_end_note = f"Exp5012 {game} no-bank {residual}: {artifact['honest_verdict']}."
    new_total = prior_total + banked
    updated_text = _replace_game_row(
        registry_text,
        game,
        total_levels=new_total,
        reproduced_levels=reproduced_levels,
        banked=banked,
        dead_end_note=dead_end_note,
        artifact=artifact,
    )
    update = dict(artifact.get("registry_update") or {})
    update.update({"updated": True, "banked_levels": banked, "new_total_declared": new_total})
    return updated_text, update


def _write_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _blocked_and_write(
    *,
    root: Path,
    target_game: str,
    target_level: int,
    reason: str,
    preconditions: Mapping[str, Any],
    selection: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    artifact = blocked_artifact(
        target_game=target_game,
        target_level=target_level,
        reason=reason,
        preconditions_checked=preconditions,
        selection=selection,
        duration_s=time.monotonic() - started,
    )
    _write_artifact(root, artifact)
    return artifact


def run_experiment(root: Path = REPO, duration_s: float | None = None) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    empty_selection: dict[str, Any] = {"game": "none", "prior_level": 0, "target_level": 0}
    preconditions: dict[str, Any] = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "arc_world_model_trust_energy_spec_has_req_5012": (
            "REQ-ARC-WMTE-5012" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "registry_loadable": False,
        "offline_arcade_exits_0": False,
        "target_env_present": False,
        "adapter_registered": False,
        "generator_required": False,
        "generator_backend": "not_required_offline_no_induction",
        "gpu_policy": {
            "cuda_gpu0_allowed": True,
            "igpu_hip_allowed": True,
            "igpu_pin_required": False,
        },
    }
    if not preconditions["arc_world_model_trust_energy_spec_has_req_5012"]:
        return _blocked_and_write(
            root=root,
            target_game="none",
            target_level=0,
            reason="spec_missing",
            preconditions=preconditions,
            selection=empty_selection,
            started=started,
        )
    try:
        registry_text = _load_registry_text(root)
        registry = _load_registry(registry_text)
        preconditions["registry_loadable"] = bool(registry)
    except (OSError, yaml.YAMLError):
        return _blocked_and_write(
            root=root,
            target_game="none",
            target_level=0,
            reason="arc_solve_registry_unreadable",
            preconditions=preconditions,
            selection=empty_selection,
            started=started,
        )
    if not preconditions["registry_loadable"]:
        return _blocked_and_write(
            root=root,
            target_game="none",
            target_level=0,
            reason="arc_solve_registry_unreadable",
            preconditions=preconditions,
            selection=empty_selection,
            started=started,
        )

    selection = select_target(registry)
    target_game = str(selection["game"])
    target_level = int(selection.get("target_level") or 0)
    if target_game == "none":
        return _blocked_and_write(
            root=root,
            target_game="none",
            target_level=0,
            reason="no_candidate",
            preconditions=preconditions,
            selection=selection,
            started=started,
        )

    preconditions["offline_arcade_exits_0"] = offline_arcade_available()
    if not preconditions["offline_arcade_exits_0"]:
        return _blocked_and_write(
            root=root,
            target_game=target_game,
            target_level=target_level,
            reason="offline_env_missing",
            preconditions=preconditions,
            selection=selection,
            started=started,
        )
    preconditions["target_env_present"] = (root / "environment_files" / target_game).exists()
    preconditions["adapter_registered"] = bool(selection.get("adapter_registered"))
    if not preconditions["target_env_present"]:
        return _blocked_and_write(
            root=root,
            target_game=target_game,
            target_level=target_level,
            reason="offline_env_missing",
            preconditions=preconditions,
            selection=selection,
            started=started,
        )

    delta_status = dict(selection.get("delta_status") or {})
    loop_result = (
        run_standing_loop(root, target_game, target_level)
        if delta_status.get("grounded_next_level_delta") is True
        else None
    )
    artifact = build_artifact(
        selection=selection,
        prior_total_levels=int(registry.get("reproducible_total_levels") or 0),
        preconditions_checked=preconditions,
        loop_result=loop_result,
        duration_s=duration_s if duration_s is not None else time.monotonic() - started,
    )
    _write_artifact(root, artifact)
    updated_registry_text, _update = apply_registry_result(registry_text, artifact=artifact)
    registry_path.write_text(updated_registry_text, encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - script entrypoint
    artifact = run_experiment()
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "target_game": artifact["target_game"],
                "offline_reproduced": artifact["offline_reproduced"],
                "reproduced_levels": artifact["reproduced_levels"],
                "new_levels_banked": artifact["new_levels_banked"],
            },
            sort_keys=True,
        )
    )
    return 0 if not artifact.get("schema_errors") else 1


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())
