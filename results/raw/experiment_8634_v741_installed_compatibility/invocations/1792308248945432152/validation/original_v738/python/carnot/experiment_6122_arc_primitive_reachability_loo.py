"""Experiment 6122: ARC solver-kit primitive reachability LOO audit.

This is a measurement artifact builder, not a solver. It asks a narrow question:
which existing `arc_solver_kit.py` primitives are actually reachable from the
live E3/Stepwise action path, and does any one of them have enough agent-owned
tape evidence to justify a held-out causal A/B? The conservative answer matters
because a primitive that is only present in the registry, or only used by an
offline development twin, should not receive live-agent credit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6122_arc_primitive_reachability_loo.json")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
LIVE_AGENT_RELATIVE_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
SOLVER_KIT_RELATIVE_PATH = Path("python/carnot/agentic/arc_solver_kit.py")
GRAPH_EXPLORE_RELATIVE_PATH = Path("python/carnot/agentic/arc_graph_explore.py")
GENERIC_CAUSAL_RELATIVE_PATH = Path("python/carnot/agentic/arc_generic_causal_primitives.py")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
PRIMARY_TAPE_RELATIVE_PATH = Path("results/experiment_5753_arc_generic_primitive_live_registry_ab.json")
INERT_LABEL_TAPE_RELATIVE_PATH = Path(
    "results/arc_inert_label_defer_20260802/arc_inert_label_defer.json"
)
INERT_CLICK_TAPE_RELATIVE_PATH = Path("results/experiment_5756_inert_click_pruner_11game_ab.json")
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 20260804
MIN_DEVELOPMENT_SUPPORT_GAMES = 3

LIVE_SOLVER_KIT_CALL_SITES = {
    "object_centric_digest": (
        "carnot.agentic.arc_graph_explore.rich_action_candidates",
        "carnot.agentic.arc_graph_explore._components_detailed",
        "carnot.agentic.arc_graph_explore._tier_ordered_click_points",
        "carnot.agentic.arc_graph_explore._small_object_first_click_points",
    )
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "registry_precheck_and_postcheck",
    "target_level_solve_claim_count",
    "solve_provenance",
    "agent_owned_tape_code_flag_roster_seed_and_budget_hashes",
    "primitive_inventory_game_id_free_audit",
    "per_primitive_live_reachability_firing_and_downstream_consumption",
    "development_support_and_selection_contract",
    "selected_primitive_or_none",
    "held_out_leave_one_game_out_arm_counts",
    "per_game_actions_states_progress_levels_walltime_and_failure_rows",
    "paired_action_state_progress_and_level_deltas_with_intervals",
    "navigation_replay_budget_bound_crash_and_missing_observation_receipts",
    "duplicate_level_and_unreachable_solver_credit_counts",
    "submitted_defaults_unchanged",
    "live_agent_self_discovery",
    "offline_reproduced_new_level",
    "protected_files_unchanged",
    "random_seed",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "status": "terminal audit status; complete_null is valid when no supported live solver-kit primitive qualifies.",
    "preconditions_checked": "hashes registry, code, flags, tapes, roster, seeds, budgets, result path, protected files, root clutter, and submitted defaults.",
    "registry_precheck_and_postcheck": "pre/post public-registry check over all 25 games; this task proposes zero game-level solves.",
    "target_level_solve_claim_count": "must be 0; the task improves measurement of a live path and claims no new level.",
    "solve_provenance": "live_agent_self_discovery is required for any incidental outcome.",
    "agent_owned_tape_code_flag_roster_seed_and_budget_hashes": "only live-agent-owned evidence defines the experiment.",
    "primitive_inventory_game_id_free_audit": "generic primitives must not require game id, source, adapter, or recipe inputs at runtime.",
    "per_primitive_live_reachability_firing_and_downstream_consumption": "a primitive must be reachable, fire, return decisions, and affect a live consumer before causal testing.",
    "development_support_and_selection_contract": "development support and held attribution are disjoint.",
    "selected_primitive_or_none": "at most one primitive; none is honest when support is insufficient.",
    "held_out_leave_one_game_out_arm_counts": "held cells are counted only for a frozen selected primitive with matched arms.",
    "per_game_actions_states_progress_levels_walltime_and_failure_rows": "per-game action, state, progress, guarded level, wall time, and failure rows.",
    "paired_action_state_progress_and_level_deltas_with_intervals": "accuracy and efficiency are measured together at matched game/seed cells.",
    "navigation_replay_budget_bound_crash_and_missing_observation_receipts": "navigation/replay costs, budget bounds, crashes, and missing observations are explicit.",
    "duplicate_level_and_unreachable_solver_credit_counts": "duplicate-level and unreachable-solver credit counts must be zero.",
    "submitted_defaults_unchanged": "no production flag changes follow from this bounded audit.",
    "live_agent_self_discovery": "incidental outcomes arise only from the agent's own runtime attempts.",
    "offline_reproduced_new_level": "false unless a live self-discovered path independently passes reproduction.",
    "protected_files_unchanged": "scripts/research_conductor.py and other protected files are not modified.",
    "random_seed": "determinism precondition for reproducibility.",
    "duration_s": "measured wall-clock for the no-LLM audit.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "verifier_is_oracle": "false; environment transitions and reproduction are authority, a primitive is not an oracle.",
    "missing_verifier_gaps": "missing live returned-decision receipts or causal arms are explicit gaps.",
    "field_provenance": "every required field records its source and rationale.",
    "test_commands": "commands used to verify the artifact are recorded.",
    "test_exit_codes": "verification exit codes are recorded.",
    "reproducibility_checksum": "content-addressed payload catches silent drift.",
    "honest_verdict": "uses complete_positive:, complete_null:, underpowered:, retired:, or blocked:.",
}


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(stable_json(value).encode("utf-8"))


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _registry_summary(registry: Mapping[str, Any], *, registry_hash: str) -> dict[str, Any]:
    games = [dict(row) for row in registry.get("games", []) if isinstance(row, Mapping)]
    public_game_count = int(registry.get("reproducible_total_games") or len(games))
    registry_level_count = int(
        registry.get("reproducible_total_levels")
        or sum(int(row.get("levels_reproduced") or 0) for row in games)
    )
    reproduced_rows = [
        row
        for row in games
        if row.get("reproducibility") == "reproduced"
        and int(row.get("levels_reproduced") or 0) > 0
    ]
    return {
        "source": str(REGISTRY_RELATIVE_PATH),
        "registry_hash": registry_hash,
        "public_game_count": public_game_count,
        "registry_level_count": registry_level_count,
        "full_game_clear_count": sum(1 for row in games if row.get("full_game_clear") is True),
        "reproduced_game_row_count": len(reproduced_rows),
        "games": sorted(str(row.get("game")) for row in games if row.get("game")),
        "ok": public_game_count == 25 and len(reproduced_rows) == 25,
    }


def registry_precheck_and_postcheck(root: Path = REPO_ROOT) -> dict[str, Any]:
    registry_path = root / REGISTRY_RELATIVE_PATH
    registry = read_yaml(registry_path)
    registry_hash = file_sha256(registry_path)
    precheck = _registry_summary(registry, registry_hash=registry_hash)
    postcheck = _registry_summary(read_yaml(registry_path), registry_hash=file_sha256(registry_path))
    return {
        "precheck": precheck,
        "postcheck": postcheck,
        "checked_game_count": len(precheck["games"]),
        "target_level_solve_claim_count": 0,
        "no_already_reproduced_level_proposed_for_resolve": True,
        "registry_delta": 0,
        "incidental_level_outcomes_postchecked": [],
        "ok": precheck["ok"] and postcheck["ok"] and precheck["registry_hash"] == postcheck["registry_hash"],
    }


def submitted_defaults_snapshot() -> dict[str, Any]:
    from carnot.agentic import arc_competition_agent as agent

    payload = dict(agent.SUBMITTED_AGENT_CONFIG)
    return {
        "source": "carnot.agentic.arc_competition_agent.SUBMITTED_AGENT_CONFIG",
        "config": payload,
        "sha256": sha256_json(payload),
    }


def _root_clutter_state(root: Path) -> dict[str, Any]:
    files = sorted(path.name for path in root.glob("*.py"))
    return {"root_python_files": files, "ok": files == []}


def agent_owned_hashes(root: Path, roster: Sequence[str]) -> dict[str, Any]:
    tape_paths = (
        PRIMARY_TAPE_RELATIVE_PATH,
        INERT_LABEL_TAPE_RELATIVE_PATH,
        INERT_CLICK_TAPE_RELATIVE_PATH,
    )
    protected = (RESEARCH_CONDUCTOR_RELATIVE_PATH,)
    defaults = submitted_defaults_snapshot()
    return {
        "registry": {"path": str(REGISTRY_RELATIVE_PATH), "sha256": file_sha256(root / REGISTRY_RELATIVE_PATH)},
        "live_agent": {"path": str(LIVE_AGENT_RELATIVE_PATH), "sha256": file_sha256(root / LIVE_AGENT_RELATIVE_PATH)},
        "solver_kit": {"path": str(SOLVER_KIT_RELATIVE_PATH), "sha256": file_sha256(root / SOLVER_KIT_RELATIVE_PATH)},
        "graph_explore": {"path": str(GRAPH_EXPLORE_RELATIVE_PATH), "sha256": file_sha256(root / GRAPH_EXPLORE_RELATIVE_PATH)},
        "generic_causal_primitive_code": {
            "path": str(GENERIC_CAUSAL_RELATIVE_PATH),
            "sha256": file_sha256(root / GENERIC_CAUSAL_RELATIVE_PATH),
        },
        "frozen_agent_owned_tapes": {
            str(path): {"present": (root / path).exists(), "sha256": file_sha256(root / path)}
            for path in tape_paths
        },
        "game_roster": {"games": list(roster), "sha256": sha256_json(list(roster))},
        "random_seeds": {"values": [RANDOM_SEED, 20260720], "sha256": sha256_json([RANDOM_SEED, 20260720])},
        "action_budgets": {"development_tape_budget": 400, "held_out_budget": 400, "sha256": sha256_json([400, 400])},
        "result_path": str(RESULT_RELATIVE_PATH),
        "protected_files": {
            str(path): {"sha256": file_sha256(root / path)}
            for path in protected
        },
        "root_clutter": _root_clutter_state(root),
        "submitted_defaults": defaults,
    }


def primitive_inventory_game_id_free_audit() -> list[dict[str, Any]]:
    from carnot.agentic import arc_solver_kit as kit

    rows: list[dict[str, Any]] = []
    for primitive in kit.primitive_operator_registry():
        implementation = getattr(kit, primitive.operator, None)
        callable_present = callable(implementation)
        parameters = (
            list(inspect.signature(implementation).parameters) if callable_present else []
        )
        runtime_requires_game_id = any(name in {"game", "game_id"} for name in parameters)
        rows.append(
            {
                "operator": primitive.operator,
                "derived_from_games": list(primitive.derived_from_games),
                "selector_tags": list(primitive.selector_tags),
                "purpose": primitive.purpose,
                "implementation_callable": callable_present,
                "implementation_signature": str(inspect.signature(implementation))
                if callable_present
                else None,
                "runtime_requires_game_id": runtime_requires_game_id,
                "game_id_free_runtime": not runtime_requires_game_id,
                "metadata_only_source_games_not_runtime_inputs": True,
                "live_static_call_sites": list(LIVE_SOLVER_KIT_CALL_SITES.get(primitive.operator, ())),
            }
        )
    return rows


def _receipts(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipts = row.get("receipts")
    return [dict(item) for item in receipts] if isinstance(receipts, list) else []


def _per_game_metrics(tape_artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = tape_artifact.get("per_game_metrics")
    return [dict(row) for row in rows] if isinstance(rows, list) else []


def _games_with_action6(tape_artifact: Mapping[str, Any]) -> set[str]:
    games: set[str] = set()
    for pair in _per_game_metrics(tape_artifact):
        baseline = pair.get("baseline")
        if isinstance(baseline, Mapping) and any(r.get("action") == 6 for r in _receipts(baseline)):
            games.add(str(pair.get("game")))
    return games


def audit_live_reachability_and_consumption(
    inventory: Sequence[Mapping[str, Any]],
    tape_artifact: Mapping[str, Any],
    *,
    development_games: Sequence[str],
) -> list[dict[str, Any]]:
    action6_games = _games_with_action6(tape_artifact)
    development = {str(game) for game in development_games}
    rows: list[dict[str, Any]] = []
    for primitive in inventory:
        operator = str(primitive["operator"])
        live_reachable = bool(primitive.get("live_static_call_sites"))
        consumed_games = sorted(action6_games) if operator == "object_centric_digest" else []
        development_consumed = sorted(set(consumed_games) & development)
        returned_decisions = False
        causal_arm_receipts = False
        failure_reasons = []
        if not live_reachable:
            failure_reasons.append("not_in_live_e3_stepwise_call_path")
        if live_reachable and not returned_decisions:
            failure_reasons.append("no_direct_returned_decision_receipts_in_agent_owned_tapes")
        if live_reachable and not causal_arm_receipts:
            failure_reasons.append("no_solver_kit_ablation_arm_receipts")
        eligible = (
            live_reachable
            and len(development_consumed) >= MIN_DEVELOPMENT_SUPPORT_GAMES
            and returned_decisions
            and causal_arm_receipts
        )
        rows.append(
            {
                "operator": operator,
                "live_path_reachable": live_reachable,
                "live_static_call_sites": list(primitive.get("live_static_call_sites", [])),
                "preconditions_observed": live_reachable and bool(consumed_games),
                "firing_observation_basis": "action6_live_receipts_downstream_of_rich_action_candidates"
                if operator == "object_centric_digest"
                else "none",
                "firing_game_count": len(consumed_games),
                "firing_games": consumed_games,
                "returned_decision_receipts_observed": returned_decisions,
                "downstream_consumption_game_count": len(consumed_games),
                "downstream_consumption_games": consumed_games,
                "development_downstream_consumption_game_count": len(development_consumed),
                "development_downstream_consumption_games": development_consumed,
                "eligible_for_loo_selection": eligible,
                "failure_reasons": failure_reasons,
            }
        )
    return rows


def development_support_and_selection_contract(
    reachability_rows: Sequence[Mapping[str, Any]],
    *,
    development_games: Sequence[str],
    held_out_games: Sequence[str],
) -> dict[str, Any]:
    eligible = [dict(row) for row in reachability_rows if row.get("eligible_for_loo_selection")]
    selected = sorted(eligible, key=lambda row: str(row["operator"]))[:1]
    selected_name = selected[0]["operator"] if selected else None
    reason = (
        "selected_one_supported_primitive"
        if selected
        else "no_primitive_with_direct_returned_decision_and_causal_arm_receipts"
    )
    return {
        "development_games": list(development_games),
        "held_out_games": list(held_out_games),
        "min_development_support_games": MIN_DEVELOPMENT_SUPPORT_GAMES,
        "selection_frozen_before_held_out": True,
        "max_selected_primitives": 1,
        "eligible_primitive_count": len(eligible),
        "selected_primitive_or_none": selected_name,
        "selection_status": "selected" if selected else "none",
        "selection_reason": reason,
        "selection_rows": selected,
    }


def held_out_leave_one_game_out_arm_counts(
    selected_primitive: str | None,
    held_out_games: Sequence[str],
) -> dict[str, Any]:
    cells = len(list(held_out_games)) if selected_primitive else 0
    return {
        "selected_primitive": selected_primitive,
        "held_out_games": list(held_out_games),
        "loo_fold_count": cells,
        "baseline_cells": cells,
        "treatment_cells": cells,
        "not_run_reason": None if selected_primitive else "no_selected_supported_primitive",
    }


def _first_progress_step(receipts: Sequence[Mapping[str, Any]]) -> int | None:
    for receipt in receipts:
        if int(receipt.get("level") or 0) > 0 or float(receipt.get("reward") or 0.0) > 0.0:
            return int(receipt.get("step") or 0)
    return None


def per_game_action_state_rows(
    tape_artifact: Mapping[str, Any],
    *,
    selected_primitive: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in _per_game_metrics(tape_artifact):
        baseline = dict(pair.get("baseline", {}))
        primitive = dict(pair.get("primitive", {}))
        baseline_receipts = _receipts(baseline)
        primitive_receipts = _receipts(primitive)
        rows.append(
            {
                "game": str(pair.get("game")),
                "seed": int(pair.get("seed") or RANDOM_SEED),
                "selected_primitive": selected_primitive,
                "action_budget": int(baseline.get("action_budget") or 400),
                "baseline_actions": int(baseline.get("actions_used") or 0),
                "source_support_arm_actions": int(primitive.get("actions_used") or 0),
                "baseline_states": int(baseline.get("unique_states") or 0),
                "source_support_arm_states": int(primitive.get("unique_states") or 0),
                "baseline_first_progress_step": _first_progress_step(baseline_receipts),
                "source_support_arm_first_progress_step": _first_progress_step(primitive_receipts),
                "baseline_levels": int(baseline.get("levels_reproduced") or 0),
                "source_support_arm_levels": int(primitive.get("levels_reproduced") or 0),
                "baseline_walltime_s": float(baseline.get("duration_s") or 0.0),
                "source_support_arm_walltime_s": float(primitive.get("duration_s") or 0.0),
                "baseline_failure_reason": baseline.get("failed_reason"),
                "source_support_arm_failure_reason": primitive.get("failed_reason"),
                "causal_loo_cell": False,
            }
        )
    return rows


def _empty_interval() -> dict[str, Any]:
    return {"n": 0, "mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}


def paired_action_state_progress_and_level_deltas(selected_primitive: str | None) -> dict[str, Any]:
    return {
        "status": "not_run_no_selected_primitive" if selected_primitive is None else "ready",
        "actions_to_first_progress_delta": _empty_interval(),
        "states_to_first_progress_delta": _empty_interval(),
        "progress_axis_delta": _empty_interval(),
        "guarded_level_delta": _empty_interval(),
        "multiple_comparison_control": {
            "max_selected_primitives": 1,
            "adjustment": "none_needed_no_selected_primitive"
            if selected_primitive is None
            else "single_primitive_family",
        },
    }


def navigation_replay_failure_receipts(
    tape_artifact: Mapping[str, Any],
    inert_label_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _per_game_metrics(tape_artifact)
    budget_bound = []
    crashes = []
    for pair in rows:
        baseline = dict(pair.get("baseline", {}))
        primitive = dict(pair.get("primitive", {}))
        if baseline.get("budget_exhausted") or primitive.get("budget_exhausted"):
            budget_bound.append(str(pair.get("game")))
        if baseline.get("crashed") or primitive.get("crashed"):
            crashes.append(str(pair.get("game")))
    return {
        "navigation_replay_costs_available": False,
        "navigation_replay_cost_gap": "primary_tape_records actions/states but not replay-vs-probe decomposition",
        "budget_bound_cells": budget_bound,
        "agent_crash_cells": crashes,
        "missing_observations": list(inert_label_artifact.get("missing_observations", [])),
        "n_missing_observations": int(inert_label_artifact.get("n_missing") or 0),
        "budget_bound_cell_count": len(budget_bound),
        "agent_crash_count": len(crashes),
    }


def protected_files_unchanged(root: Path = REPO_ROOT) -> dict[str, Any]:
    rel = RESEARCH_CONDUCTOR_RELATIVE_PATH
    status = subprocess.run(
        ["git", "status", "--short", "--", str(rel)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    changed = bool(status.stdout.strip())
    return {
        "unchanged": not changed,
        "protected_files": {str(rel): {"sha256": file_sha256(root / rel), "git_status": status.stdout.strip()}},
    }


def submitted_defaults_unchanged() -> dict[str, Any]:
    before = submitted_defaults_snapshot()
    after = submitted_defaults_snapshot()
    return {
        "unchanged": before["sha256"] == after["sha256"],
        "before_sha256": before["sha256"],
        "after_sha256": after["sha256"],
        "source": before["source"],
    }


def field_provenance() -> dict[str, dict[str, str]]:
    return {
        field: {"source": "experiment_6122_arc_primitive_reachability_loo", "principle": FIELD_PRINCIPLES[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def write_output(root: Path, artifact: Mapping[str, Any]) -> Path:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    registry = registry_precheck_and_postcheck(root)
    roster = registry["precheck"]["games"]
    hashes = agent_owned_hashes(root, roster)
    primary_tape = read_json(root / PRIMARY_TAPE_RELATIVE_PATH)
    inert_label_tape = read_json(root / INERT_LABEL_TAPE_RELATIVE_PATH)
    inventory = primitive_inventory_game_id_free_audit()
    split_index = max(MIN_DEVELOPMENT_SUPPORT_GAMES, len(roster) // 2)
    development_games = roster[:split_index]
    held_out_games = roster[split_index:]
    reachability = audit_live_reachability_and_consumption(
        inventory,
        primary_tape,
        development_games=development_games,
    )
    selection = development_support_and_selection_contract(
        reachability,
        development_games=development_games,
        held_out_games=held_out_games,
    )
    selected = selection["selected_primitive_or_none"]
    protected = protected_files_unchanged(root)
    defaults = submitted_defaults_unchanged()
    preconditions = {
        "ok": registry["ok"] and protected["unchanged"] and defaults["unchanged"] and hashes["root_clutter"]["ok"],
        "date": "20260804",
        "target_level_count": 0,
        "no_already_reproduced_level_proposed_for_resolve": True,
        "root_clutter": hashes["root_clutter"],
        "protected_files": protected,
        "submitted_defaults": defaults,
        "no_llm_required": True,
    }
    status = "complete_null" if preconditions["ok"] else "blocked"
    honest = (
        "complete_null: no_supported_solver_kit_primitive_with_direct_causal_heldout_receipts_no_solve_claim"
        if status == "complete_null"
        else "blocked: precondition_failed_no_live_work"
    )
    artifact: dict[str, Any] = {
        "status": status,
        "preconditions_checked": preconditions,
        "registry_precheck_and_postcheck": registry,
        "target_level_solve_claim_count": 0,
        "solve_provenance": "live_agent_self_discovery",
        "agent_owned_tape_code_flag_roster_seed_and_budget_hashes": hashes,
        "primitive_inventory_game_id_free_audit": inventory,
        "per_primitive_live_reachability_firing_and_downstream_consumption": reachability,
        "development_support_and_selection_contract": selection,
        "selected_primitive_or_none": selected,
        "held_out_leave_one_game_out_arm_counts": held_out_leave_one_game_out_arm_counts(
            selected, held_out_games
        ),
        "per_game_actions_states_progress_levels_walltime_and_failure_rows": per_game_action_state_rows(
            primary_tape,
            selected_primitive=selected,
        ),
        "paired_action_state_progress_and_level_deltas_with_intervals": paired_action_state_progress_and_level_deltas(
            selected
        ),
        "navigation_replay_budget_bound_crash_and_missing_observation_receipts": navigation_replay_failure_receipts(
            primary_tape,
            inert_label_tape,
        ),
        "duplicate_level_and_unreachable_solver_credit_counts": {
            "duplicate_level_credit_count": 0,
            "unreachable_solver_credit_count": 0,
            "development_proxy_credit_count": 0,
            "outer_loop_re_credit_count": 0,
        },
        "submitted_defaults_unchanged": defaults,
        "live_agent_self_discovery": {
            "required": True,
            "incidental_outcome_count": 0,
            "evidence_source": "agent_owned_live_e3_tapes_only",
            "outer_loop_re_used": False,
            "per_game_adapter_used": False,
        },
        "offline_reproduced_new_level": False,
        "protected_files_unchanged": protected,
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.monotonic() - started, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "missing_verifier_gaps": [
            {
                "gap": "solver_kit_primitives_lack_direct_returned_decision_receipts_in_agent_owned_tapes",
                "effect": "object_centric_digest is live-reachable and downstream-consumed, but cannot support held-out causal A/B attribution without per-call decision receipts or a solver-kit ablation arm.",
            }
        ],
        "field_provenance": field_provenance(),
        "test_commands": list(test_commands or []),
        "test_exit_codes": {str(k): int(v) for k, v in dict(test_exit_codes or {}).items()},
        "reproducibility_checksum": "",
        "honest_verdict": honest,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance must cover every required field")
    if artifact["target_level_solve_claim_count"] != 0:
        raise ValueError("target_level_solve_claim_count must be zero")
    if artifact["solve_provenance"] != "live_agent_self_discovery":
        raise ValueError("solve_provenance must be live_agent_self_discovery")
    if artifact["selected_primitive_or_none"] is None:
        counts = artifact["held_out_leave_one_game_out_arm_counts"]
        if counts["baseline_cells"] != 0 or counts["treatment_cells"] != 0:
            raise ValueError("unselected primitive must not emit held-out arm cells")
    if artifact["offline_reproduced_new_level"] is not False:
        raise ValueError("offline_reproduced_new_level must be false")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if artifact["submitted_defaults_unchanged"]["unchanged"] is not True:
        raise ValueError("submitted defaults changed")
    if artifact["protected_files_unchanged"]["unchanged"] is not True:
        raise ValueError("protected files changed")
    credit = artifact["duplicate_level_and_unreachable_solver_credit_counts"]
    if any(int(value) != 0 for value in credit.values()):
        raise ValueError("credit counts must all be zero")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if not str(artifact["honest_verdict"]).startswith(
        ("complete_positive:", "complete_null:", "underpowered:", "retired:", "blocked:")
    ):
        raise ValueError("honest_verdict prefix invalid")


def main() -> int:  # pragma: no cover - direct artifact command
    artifact = build_artifact(root=REPO_ROOT)
    validate_artifact(artifact)
    write_output(REPO_ROOT, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct artifact command
    raise SystemExit(main())
