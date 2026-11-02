"""Experiment 4948: fresh L2->L3 ARC level-up attempt.

Spec refs: REQ-CAPSTONE-4948, SCENARIO-CAPSTONE-4948,
SCENARIO-CAPSTONE-4948-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4948-FIELD-PRINCIPLES.
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
EXPERIMENT = "experiment_4948_levelup_attempt"
EXPERIMENT_ID = 4948
SCHEMA = "carnot.arc_levelup_attempt_4948.v1"
RESULT_RELATIVE_PATH = "results/experiment_4948_levelup_attempt.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
CAPSTONE_SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
RANDOM_SEED = 4948
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

L2_CANDIDATES = ("cd82", "vc33", "bp35", "re86")
RECENT_EXCLUDED_TARGETS = ("lf52", "sb26", "sp80", "su15", "cn04", "m0r0", "dc22")
PEER_EXCLUDED_TARGETS = ("ar25", "bp35")
HIDDEN_STATE_TARGETS = ("ka59", "wa30")
SPEC_REFS = [
    "REQ-CAPSTONE-4948",
    "SCENARIO-CAPSTONE-4948",
    "SCENARIO-CAPSTONE-4948-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4948-FIELD-PRINCIPLES",
]
REQUIRED_FIELDS = (
    "honest_verdict",
    "solve_provenance",
    "target_game",
    "offline_reproduced",
    "reproduced_levels",
    "new_levels_banked",
    "live_path_reachable",
    "verifier_is_oracle",
    "inference_substrate",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; banked is success_<game>_levelup_banked, no-bank is "
            "complete_<game>_no_new_level_residual_<cause>."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- NOT outer_loop_re (CRITICAL); NOT a "
            "duplicate re-solve (CRITICAL)."
        )
    },
    "target_game": {
        "principle": (
            "the rotated FRESH L2->L3 target, different from Exp4947/A1 ar25, "
            "A3 self-play bp35, and the recent level-up rotation."
        )
    },
    "offline_reproduced": {
        "principle": "only reproduced levels count toward reproducible_total_levels."
    },
    "reproduced_levels": {
        "principle": "the new reproducible depth; the monotonic ARC progress metric."
    },
    "new_levels_banked": {
        "principle": ">=1 for a PASS; 0 records the honest rotation dead-end for the next planner."
    },
    "live_path_reachable": {
        "principle": (
            "true for complete/success artifacts -- arc_loop_solve plus a live "
            "GameAdapter remains reachable even when no grounded next-level tail exists."
        )
    },
    "verifier_is_oracle": {"principle": "the reproduction gate is the executable oracle."},
    "inference_substrate": {
        "principle": (
            "live_llm_inference if induction runs (60s floor); else "
            "verifier_ensemble_against_cached_candidates / the honest offline arcade substrate."
        )
    },
    "preconditions_checked": {
        "principle": "records arcade/env/generator checks; a missing resource emits blocked_, never a fabricated solve."
    },
    "random_seed": {"principle": "determinism for the offline search."},
    "reproducibility_checksum": {
        "principle": "content hash of (game, plan, claimed level) so a replication catches drift."
    },
}


def standing_loop_command(game: str) -> str:
    return f".venv/bin/python scripts/arc_loop_solve.py --game {game}"


def standing_loop_result_path(game: str) -> str:
    return f"results/arc_loop_solve_{game}.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _game_row(registry: Mapping[str, Any], game: str) -> Mapping[str, Any]:
    rows = _game_rows(registry)
    if game not in rows:
        raise ValueError(f"registry missing game row: {game}")
    return rows[game]


def _dead_ends(row: Mapping[str, Any]) -> list[str]:
    values = row.get("dead_ends") or []
    if isinstance(values, list):
        return [str(value) for value in values]
    return [str(values)]


def _has_recorded_next_level_dead_end(game: str, row: Mapping[str, Any]) -> bool:
    blocking_phrases = (
        "no grounded l3",
        "no_grounded_l3",
        "no grounded next-level",
        "no_grounded_next_level",
        "duplicate_depth",
        "no-bank",
        "replays to l2 only",
        "repeated prior",
    )
    for item in _dead_ends(row):
        lowered = item.lower()
        if "retired" in lowered:
            continue
        if not any(phrase in lowered for phrase in blocking_phrases):
            continue
        if (
            lowered.startswith(f"{game}:")
            or (lowered.startswith("exp") and game in lowered)
            or f"--game {game}" in lowered
            or "current adapter" in lowered
        ):
            return True
    return False


def adapter_for(game: str):  # pragma: no cover - wrapper kept injectable for tests
    from carnot.agentic import arc_game_adapters

    return arc_game_adapters.get_adapter(game)


def grounded_delta_status(game: str, *, prior_level: int, adapter: Any | None) -> dict[str, Any]:
    target_level = int(prior_level) + 1
    if adapter is None:
        return {
            "game": game,
            "prior_level": int(prior_level),
            "target_level": target_level,
            "grounded_next_level_delta": False,
            "reason": "adapter_missing",
            "live_path_reachable": False,
            "adapter_level_tails": [],
        }
    tails = getattr(adapter, "level_tails", {}) or {}
    tail_keys: list[int] = []
    for key in tails.keys():
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
        "adapter_game": getattr(adapter, "game", game),
        "adapter_level_tails": sorted(tail_keys),
    }


def select_target(
    registry: Mapping[str, Any],
    *,
    adapter_lookup: Callable[[str], Any | None] | None = None,
) -> dict[str, Any]:
    rows = _game_rows(registry)
    lookup = adapter_lookup or adapter_for
    audit: list[dict[str, Any]] = []
    grounded_selection: dict[str, Any] | None = None
    no_delta_selection: dict[str, Any] | None = None
    for game in L2_CANDIDATES:
        row = rows.get(game, {})
        prior = int(row.get("levels_reproduced") or 0)
        target_level = prior + 1 if prior else 3
        recorded_dead_end = _has_recorded_next_level_dead_end(game, row)
        delta = {"grounded_next_level_delta": False, "reason": "not_checked"}
        status = "candidate_unselected"
        reason = "lower_priority_candidate"
        if game in RECENT_EXCLUDED_TARGETS:
            status, reason = "skip_recent_target", "recent_levelup_rotation"
        elif game in PEER_EXCLUDED_TARGETS:
            status, reason = "skip_peer_target", "a1_or_a3_peer_target"
        elif game in HIDDEN_STATE_TARGETS:
            status, reason = "skip_hidden_state_bound", "hidden_state_bound"
        elif not row:
            status, reason = "skip_missing_registry_row", "missing_registry_row"
        elif prior != 2:
            status, reason = "skip_wrong_prior_depth", "requires_l2_prior"
        elif recorded_dead_end:
            status, reason = "skip_recorded_dead_end", "recorded_next_level_dead_end"
        else:
            delta = grounded_delta_status(game, prior_level=prior, adapter=lookup(game))
            if delta.get("grounded_next_level_delta") is True:
                status = "selected" if grounded_selection is None else "candidate_grounded_unselected"
                reason = "fresh_l2_to_l3_grounded_delta"
            else:
                status, reason = "candidate_no_grounded_delta", str(delta.get("reason"))
        audit_row = {
            "game": game,
            "lane": "l2_to_l3",
            "prior_level": prior,
            "target_level": target_level,
            "status": status,
            "reason": reason,
            "has_recorded_next_level_dead_end": recorded_dead_end,
            "dead_ends_consulted": _dead_ends(row),
            "delta_status": dict(delta),
        }
        audit.append(audit_row)
        if status == "selected" and grounded_selection is None:
            grounded_selection = audit_row
        if status == "candidate_no_grounded_delta" and no_delta_selection is None:
            no_delta_selection = audit_row

    selected = grounded_selection
    if selected is None and no_delta_selection is not None:
        selected = dict(no_delta_selection)
        selected["status"] = "selected_no_grounded_delta"
        selected["reason"] = "fresh_l2_to_l3_no_grounded_delta"

    base = {
        "candidate_audit": audit,
        "excluded_recent_targets": list(RECENT_EXCLUDED_TARGETS),
        "excluded_peer_targets": list(PEER_EXCLUDED_TARGETS),
        "hidden_state_targets_avoided": list(HIDDEN_STATE_TARGETS),
    }
    if selected is None:
        return {
            "game": "none",
            "prior_level": 0,
            "target_level": 0,
            "status": "no_candidate",
            "reason": "no_fresh_l2_to_l3_rotation_candidate",
            "dead_ends_consulted": [],
            **base,
        }
    return {**selected, **base}


def recommend_approach(game: str) -> dict[str, Any]:  # pragma: no cover - wrapper kept injectable for tests
    from carnot.agentic import arc_solve_learning

    return dict(arc_solve_learning.recommend_approach(game))


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
    except Exception:
        return False
    return True


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
        return bool(loop_result.get("offline_reproduced") is True and gate.get("reproduced") is True)
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
        return str(delta_status.get("reason") or "no_grounded_l3_delta")
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
    target_game: str,
    prior_level: int,
    prior_total_levels: int,
    candidate_selection: Mapping[str, Any],
    approach_recommendation: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    delta_status: Mapping[str, Any],
    loop_result: Mapping[str, Any] | None,
    duration_s: float,
) -> dict[str, Any]:
    reached = _loop_reached_level(loop_result)
    reproduced = _loop_reproduced(loop_result)
    banked = max(0, reached - int(prior_level)) if reproduced else 0
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
    reproduced_levels = reached if success else int(prior_level)
    new_levels_banked = banked if success else 0
    verdict = (
        f"success_{target_game}_levelup_banked"
        if success
        else f"complete_{target_game}_no_new_level_residual_{reason}"
    )
    registry_update = _registry_update_payload(
        target_game=target_game,
        prior_level=prior_level,
        prior_total_levels=prior_total_levels,
        reproduced_levels=reproduced_levels,
        new_levels_banked=new_levels_banked,
        reason=reason,
    )
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": verdict,
        "solve_provenance": SOLVE_PROVENANCE,
        "target_game": target_game,
        "offline_reproduced": bool(success),
        "reproduced_levels": int(reproduced_levels),
        "new_levels_banked": int(new_levels_banked),
        "live_path_reachable": bool(delta_status.get("live_path_reachable")) if not success else True,
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "prior_reproduced_level": int(prior_level),
        "reproducible_total_levels_before": int(prior_total_levels),
        "reproducible_total_levels_after": int(prior_total_levels) + int(new_levels_banked),
        "candidate_selection": dict(candidate_selection),
        "approach_recommendation": dict(approach_recommendation),
        "dead_ends_consulted": list(candidate_selection.get("dead_ends_consulted") or []),
        "delta_status": dict(delta_status),
        "registry_update": registry_update,
        "standing_loop_command": standing_loop_command(target_game),
        "standing_loop_result_path": standing_loop_result_path(target_game),
        "standing_loop_ran": loop_result is not None,
        "reproduction_gate": dict((loop_result or {}).get("reproduction_gate") or {}),
        "solution_labels": list((loop_result or {}).get("solution_labels") or []),
        "solution": list((loop_result or {}).get("solution") or []),
        "states_expanded": int((loop_result or {}).get("states_expanded") or 0),
        "retire_if_same_verdict": not success,
        "duration_s": float(duration_s),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def blocked_artifact(
    *,
    target_game: str,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
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
        "prior_reproduced_level": 0,
        "reproducible_total_levels_before": 0,
        "reproducible_total_levels_after": 0,
        "candidate_selection": {},
        "approach_recommendation": {},
        "dead_ends_consulted": [],
        "delta_status": {},
        "registry_update": {"updated": False, "banked_levels": 0, "reason": reason},
        "standing_loop_command": standing_loop_command(target_game),
        "standing_loop_result_path": standing_loop_result_path(target_game),
        "standing_loop_ran": False,
        "reproduction_gate": {},
        "solution_labels": [],
        "solution": [],
        "states_expanded": 0,
        "retire_if_same_verdict": True,
        "duration_s": float(duration_s),
        "reproducibility_checksum": "",
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
    if artifact.get("target_game") in (
        set(RECENT_EXCLUDED_TARGETS) | set(PEER_EXCLUDED_TARGETS) | set(HIDDEN_STATE_TARGETS)
    ):
        errors.append("target_game violates rotation exclusions")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if type(artifact.get("new_levels_banked")) is not int:
        errors.append("new_levels_banked must be bare int")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if type(artifact.get("live_path_reachable")) is not bool:
        errors.append("live_path_reachable must be bare bool")
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
    if verdict.startswith(("success_", "complete_")) and artifact.get("live_path_reachable") is not True:
        errors.append("success/complete requires live_path_reachable true")
    if verdict.startswith("success_"):
        if artifact.get("offline_reproduced") is not True:
            errors.append("success requires offline_reproduced true")
        if int(artifact.get("new_levels_banked") or 0) < 1:
            errors.append("success requires new_levels_banked >= 1")
        if int(artifact.get("reproduced_levels") or 0) <= int(artifact.get("prior_reproduced_level") or 0):
            errors.append("success requires reproduced_levels > prior_reproduced_level")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        errors.append("reproducibility_checksum must be 64 hex chars")
    elif checksum != reproducibility_checksum(artifact):
        errors.append("checksum mismatch")
    return errors


def _game_block_bounds(registry_text: str, game: str) -> tuple[int, int]:
    marker = f"- game: {game}\n"
    start = registry_text.index(marker)
    next_match = re.search(r"\n- game: ", registry_text[start + len(marker) :])
    end = start + len(marker) + next_match.start() + 1 if next_match else len(registry_text)
    return start, end


def _replace_game_row(registry_text: str, game: str, row: Mapping[str, Any], total_levels: int) -> str:
    start, end = _game_block_bounds(registry_text, game)
    block = yaml.safe_dump([dict(row)], sort_keys=False, width=1000)
    updated = registry_text[:start] + block + registry_text[end:]
    updated = re.sub(
        r"(?m)^(reproducible_total_levels:\s*)\d+\s*$",
        rf"\g<1>{int(total_levels)}",
        updated,
        count=1,
    )
    return re.sub(r"(?m)^(updated:\s*).*$", r"\g<1>'2026-06-28'", updated, count=1)


def _artifact_residual_reason(artifact: Mapping[str, Any]) -> str:
    verdict = str(artifact.get("honest_verdict") or "")
    marker = "_no_new_level_residual_"
    if marker in verdict:
        return verdict.split(marker, 1)[1]
    return str((artifact.get("registry_update") or {}).get("reason") or "unknown")


def apply_registry_result(
    registry_text: str,
    *,
    artifact: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    registry = _load_registry(registry_text)
    game = str(artifact.get("target_game"))
    row = dict(_game_row(registry, game))
    prior = int(row.get("levels_reproduced") or 0)
    prior_total = int(registry.get("reproducible_total_levels") or 0)
    banked = int(artifact.get("new_levels_banked") or 0)
    reached = int(artifact.get("reproduced_levels") or prior)
    update = _registry_update_payload(
        target_game=game,
        prior_level=prior,
        prior_total_levels=prior_total,
        reproduced_levels=reached,
        new_levels_banked=banked,
        reason=_artifact_residual_reason(artifact),
    )
    if banked >= 1:
        row["reproducibility"] = "reproduced"
        row["levels_reproduced"] = reached
        row["reproduce"] = (
            f"Exp4948 {RESULT_RELATIVE_PATH} re-gated {standing_loop_result_path(game)} "
            f"offline_reproduced=True, reached_level={reached}, banked +{banked}, "
            f"checksum {artifact.get('reproducibility_checksum')}."
        )
        row["latest_exp4948_levelup_attempt"] = {
            "artifact": RESULT_RELATIVE_PATH,
            "loop_artifact": standing_loop_result_path(game),
            "offline_reproduced": True,
            "reproduced_levels": reached,
            "new_levels_banked": banked,
            "solve_provenance": SOLVE_PROVENANCE,
            "reproducibility_checksum": artifact.get("reproducibility_checksum"),
        }
    else:
        reason = _artifact_residual_reason(artifact)
        dead_ends = list(row.get("dead_ends") or [])
        note = f"Exp4948 {game} no-bank {reason}: {artifact.get('honest_verdict')}."
        if note not in dead_ends:
            dead_ends.append(note)
        row["dead_ends"] = dead_ends
        row["latest_exp4948_levelup_attempt"] = {
            "artifact": RESULT_RELATIVE_PATH,
            "loop_artifact": standing_loop_result_path(game),
            "offline_reproduced": False,
            "reproduced_levels": prior,
            "new_levels_banked": 0,
            "residual_cause": reason,
            "solve_provenance": SOLVE_PROVENANCE,
            "reproducibility_checksum": artifact.get("reproducibility_checksum"),
        }
    return _replace_game_row(registry_text, game, row, prior_total + banked), update


def precondition_probe(root: Path, target_game: str, adapter: Any | None) -> dict[str, Any]:
    root = Path(root)
    spec_path = root / CAPSTONE_SPEC_RELATIVE_PATH
    if not spec_path.exists():
        spec_path = REPO / CAPSTONE_SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    registry_loadable = False
    if registry_path.exists():
        try:
            _load_registry(registry_path.read_text(encoding="utf-8"))
            registry_loadable = True
        except yaml.YAMLError:
            registry_loadable = False
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4948" in spec_path.read_text(encoding="utf-8")
    return {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "capstone_spec_has_req_4948": spec_has_req,
        "registry_present": registry_path.exists(),
        "registry_loadable": registry_loadable,
        "offline_arcade_exits_0": offline_arcade_available(),
        "target_env_present": (root / "environment_files" / target_game).exists(),
        "adapter_registered": adapter is not None,
        "generator_required": False,
        "generator_backend": "not_required_offline_no_induction",
        "accepted_generator_backends": ["gpu0_cuda", "igpu_hip", "not_required_offline_no_induction"],
    }


def _first_blocker(preconditions: Mapping[str, Any]) -> str | None:
    ordered = (
        ("offline_arcade_exits_0", "offline_arcade_missing"),
        ("registry_present", "registry_missing"),
        ("registry_loadable", "registry_unloadable"),
        ("target_env_present", "offline_env_missing"),
        ("capstone_spec_has_req_4948", "capstone_spec_missing"),
        ("adapter_registered", "adapter_missing"),
    )
    for key, reason in ordered:
        if preconditions.get(key) is not True:
            return reason
    return None


def run_standing_loop(root: Path, game: str) -> dict[str, Any]:  # pragma: no cover
    cmd = [str(root / ".venv/bin/python"), "scripts/arc_loop_solve.py", "--game", game]
    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        return {
            "game": game,
            "offline_reproduced": False,
            "reached_level": 0,
            "mode": "standing_arc_loop_failed",
            "status": "standing_loop_failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-2000:],
        }
    return _read_json(root / standing_loop_result_path(game))


def _write_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    *,
    root: Path = REPO,
    loop_result: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root)
    fallback_target = "vc33"
    registry_path = root / REGISTRY_RELATIVE_PATH
    try:
        registry_text = _load_registry_text(root)
        registry = _load_registry(registry_text)
    except (OSError, yaml.YAMLError):
        reason = "registry_missing" if not registry_path.exists() else "registry_unloadable"
        preconditions = {
            "registry_present": registry_path.exists(),
            "registry_loadable": False,
            "offline_arcade_exits_0": offline_arcade_available(),
        }
        artifact = blocked_artifact(
            target_game=fallback_target,
            reason=reason,
            preconditions_checked=preconditions,
            duration_s=duration_s if duration_s is not None else time.monotonic() - started,
        )
        _write_artifact(root, artifact)
        return artifact

    selection = select_target(registry, adapter_lookup=adapter_for)
    target_game = str(selection.get("game") or fallback_target)
    if target_game == "none":
        target_game = fallback_target
    prior_level = int(selection.get("prior_level") or 0)
    prior_total = int(registry.get("reproducible_total_levels") or 0)
    adapter = adapter_for(target_game)
    preconditions = precondition_probe(root, target_game, adapter)
    blocker = _first_blocker(preconditions)
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    if blocker is not None:
        artifact = blocked_artifact(
            target_game=target_game,
            reason=blocker,
            preconditions_checked=preconditions,
            duration_s=elapsed,
        )
        _write_artifact(root, artifact)
        return artifact

    approach = recommend_approach(target_game)
    delta = grounded_delta_status(target_game, prior_level=prior_level, adapter=adapter)
    loop = dict(loop_result) if loop_result is not None else None
    if delta.get("grounded_next_level_delta") is True and loop is None:
        loop = run_standing_loop(root, target_game)
    artifact = build_artifact(
        target_game=target_game,
        prior_level=prior_level,
        prior_total_levels=prior_total,
        candidate_selection=selection,
        approach_recommendation=approach,
        preconditions_checked=preconditions,
        delta_status=delta,
        loop_result=loop,
        duration_s=duration_s if duration_s is not None else time.monotonic() - started,
    )
    updated_text, _update = apply_registry_result(registry_text, artifact=artifact)
    (root / REGISTRY_RELATIVE_PATH).write_text(updated_text, encoding="utf-8")
    _write_artifact(root, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    del argv
    artifact = run_experiment(root=REPO)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "target_game": artifact["target_game"],
                "offline_reproduced": artifact["offline_reproduced"],
                "reproduced_levels": artifact["reproduced_levels"],
                "new_levels_banked": artifact["new_levels_banked"],
                "live_path_reachable": artifact["live_path_reachable"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
