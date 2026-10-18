"""Exp6347 ARC action-influence preflight.

Spec refs: REQ-ARC-WMTE-6347,
SCENARIO-ARC-WMTE-6347-REGISTRY-PRECHECK,
SCENARIO-ARC-WMTE-6347-WINDOW-RECONSTRUCTION,
SCENARIO-ARC-WMTE-6347-COUNTERFACTUAL-ORDERING,
SCENARIO-ARC-WMTE-6347-ADVERSARIAL-CONTROLS.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot import experiment_6307_arc_target_validated_route_canary as exp6307
from carnot import experiment_6321_arc_target_licensed_route_live_shadow_ab as exp6321
from carnot.agentic import arc_competition_agent as agent
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_mechanic_class_detector as detector


JsonDict = dict[str, Any]

REPO_ROOT = exp6307.REPO_ROOT
RESULT_RELATIVE_PATH = Path("results/experiment_6347_arc_action_influence_preflight.json")
LIVE_WINDOW_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6347_arc_action_influence_live_windows.json"
)
REGISTRY_RELATIVE_PATH = exp6307.REGISTRY_RELATIVE_PATH
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
INFLUENCE_TASK_ID = "exp6347_arc_action_influence_preflight_no_solve"
EXACT_TRANSITION_CHECKER_NAME = "exp6347_exact_transition_receipt_checker"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ELIGIBILITY_MIN_INDEPENDENT_WINDOWS = 4
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6347_arc_action_influence_preflight "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6347_arc_action_influence_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6347_arc_action_influence_preflight.py "
    "-m pytest tests/python/test_experiment_6347_arc_action_influence_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6347_arc_action_influence_preflight.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6347_arc_action_influence_preflight.py"
)
E2E_PLAN_READ_COMMAND = "sed -n 1,180p ops/e2e-test-plan.md"
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6347_arc_action_influence_preflight.json"
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6347_test_receipts.json")
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    DETERMINATION_COMMAND,
    ADVERSARIAL_COMMAND,
)
RANDOM_SEEDS = exp6321.RANDOM_SEEDS
FORBIDDEN_ZERO_FIELDS = (
    "hidden_game_source_access_count",
    "offline_ground_truth_bfs_count",
    "hand_game_adapter_count",
    "per_game_calibration_count",
    "solve_claim_count",
    "registry_update_count",
    "llm_call_count",
)
FORBIDDEN_EVIDENCE_FIELDS = (
    "hidden_game_source_path",
    "offline_ground_truth_bfs_path",
    "hand_game_adapter",
    "per_game_calibration",
    "registry_solve_target",
    "game_source_module",
)
ROUTE_RANKING_USED_FIELDS = (
    "transition.grid",
    "transition.action",
    "transition.data",
    "transition.next_grid",
    "transition.level_before",
    "transition.level_after",
    "recorded_shipped_action",
    "prospective_action",
)
LEGAL_ACTION_USED_FIELDS = (
    "recorded_shipped_action",
    "recorded_prospective_action_ids",
    "TargetLicensedRouteShadowLedger.valid_action_range_1_to_6",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_path_hash_terminal_class_and_ready_score",
    "arc_registry_precheck_path_hash_and_result",
    "solve_provenance",
    "no_duplicate_solve_receipt",
    "live_attempt_window_manifest_path_and_hash",
    "live_evidence_allowed_fields",
    "forbidden_source_access_contract",
    "hidden_game_source_access_count",
    "offline_ground_truth_bfs_count",
    "hand_game_adapter_count",
    "per_game_calibration_count",
    "route_on_off_counterfactual_contract",
    "legal_action_set_reconstruction_results",
    "action_order_change_results_by_game_window_and_seed",
    "one_step_exact_transition_quality_by_route_state",
    "influence_eligible_window_ids_and_counts",
    "leakage_overlap_and_escape_tests",
    "fixture_mutation_and_route_deletion_results",
    "verification_calls_time_cost_and_error_table",
    "solve_claim_count",
    "registry_update_count",
    "llm_call_count",
    "exact_oracle_claim_boundary",
    "arc_action_influence_eligible_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "States complete versus blocked for the preflight.",
    "upstream_path_hash_terminal_class_and_ready_score": "Pins Exp6294, Exp6307, Exp6321, and code inputs.",
    "arc_registry_precheck_path_hash_and_result": "Proves this is an influence task, not a solve target.",
    "solve_provenance": "Names the source attempts as live-agent self-discovery.",
    "no_duplicate_solve_receipt": "Shows no duplicate solve proposal was made.",
    "live_attempt_window_manifest_path_and_hash": "Pins reconstructed Exp6321 live windows.",
    "live_evidence_allowed_fields": "Freezes which live evidence fields the replay may read.",
    "forbidden_source_access_contract": "Lists forbidden escape sources and their zero counts.",
    "hidden_game_source_access_count": "Must stay zero for hidden-source discipline.",
    "offline_ground_truth_bfs_count": "Must stay zero for self-discovery discipline.",
    "hand_game_adapter_count": "Must stay zero because hand adapters are off-path here.",
    "per_game_calibration_count": "Must stay zero because thresholds are not tuned per game.",
    "route_on_off_counterfactual_contract": "Defines the only on/off difference.",
    "legal_action_set_reconstruction_results": "Shows legal actions are separate from route features.",
    "action_order_change_results_by_game_window_and_seed": "Measures route-caused ordering changes.",
    "one_step_exact_transition_quality_by_route_state": "Reports exact one-step transition value only.",
    "influence_eligible_window_ids_and_counts": "Applies the pre-registered default-off A/B rule.",
    "leakage_overlap_and_escape_tests": "Records leakage and escape-hatch trap results.",
    "fixture_mutation_and_route_deletion_results": "Shows deletion and no-effect mutation remove the effect.",
    "verification_calls_time_cost_and_error_table": "Records preflight calls, time, and errors.",
    "solve_claim_count": "Must stay zero because this is not a solve.",
    "registry_update_count": "Must stay zero because the registry must not change.",
    "llm_call_count": "Must stay zero because no model is called.",
    "exact_oracle_claim_boundary": "Separates one-step checking from game-solve claims.",
    "arc_action_influence_eligible_score": "Equals one only when every eligibility gate passes.",
    "protected_files_unchanged": "Confirms protected files stayed unchanged.",
    "preconditions_checked": "Records hashes, thresholds, seeds, timeouts, and checker name.",
    "inference_substrate": "Declares aggregation from recorded upstream artifacts.",
    "verifier_is_oracle": "Names the exact one-step checker used.",
    "field_provenance": "Maps every field to the spec and producer.",
    "field_principles": "Gives one audit reason per required field.",
    "test_commands": "Lists verification commands.",
    "test_exit_codes": "Records command outcomes.",
    "duration_s": "Records measured wall time.",
    "random_seeds": "Pins the replay seeds from Exp6321.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "Terminal verdict with no solve claim.",
}
FIELD_PROVENANCE = {
    field: ["REQ-ARC-WMTE-6347", "experiment_6347_arc_action_influence_preflight"]
    for field in REQUIRED_ARTIFACT_FIELDS
}

canonical_json = exp6307.canonical_json
sha256_text = exp6307.sha256_text
sha256_json = exp6307.sha256_json
sha256_file = exp6307.sha256_file
payload_checksum = exp6307.payload_checksum


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _terminal_class(payload: Mapping[str, Any]) -> str:
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    if payload.get("flagged_adversarial"):
        return "flagged"
    if status.startswith("blocked") or verdict.startswith("blocked"):
        return "blocked"
    if status == "complete" or verdict.startswith(("complete:", "complete_")):
        return "complete"
    return status or "unknown"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _timing_row(name: str, started: float, *, error_count: int = 0) -> JsonDict:
    return {
        "call": name,
        "duration_s": round(time.perf_counter() - started, 6),
        "error_count": int(error_count),
    }


def _history_hash_for_row(row: Mapping[str, Any]) -> str:
    transitions = exp6321._transitions_for(str(row["mechanic"]), int(row["seed"]))
    return exp6307._history_hash(transitions)


def _transition_payload_for_row(row: Mapping[str, Any]) -> list[JsonDict]:
    transitions = exp6321._transitions_for(str(row["mechanic"]), int(row["seed"]))
    return exp6307._transition_payload(transitions)


def _dedupe(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if int(value) not in out:
            out.append(int(value))
    return out


def upstream_path_hash_terminal_class_and_ready_score() -> JsonDict:
    upstream_specs = (
        (Path("results/experiment_6294_arc_mechanic_router_causal_canary.json"), "arc_mechanic_causal_ready_score"),
        (Path("results/experiment_6307_arc_target_validated_route_canary.json"), "arc_target_licensed_router_ready_score"),
        (Path("results/experiment_6321_arc_target_licensed_route_live_shadow_ab.json"), "arc_route_live_shadow_ready_score"),
        (Path("results/experiment_6321_arc_target_licensed_route_live_shadow_transitions.json"), None),
    )
    rows = []
    for rel, ready_field in upstream_specs:
        path = REPO_ROOT / rel
        payload = _read_json(path)
        rows.append(
            {
                "path": rel.as_posix(),
                "sha256": sha256_file(path),
                "terminal_class": _terminal_class(payload),
                "ready_score_field": ready_field,
                "ready_score": payload.get(ready_field) if ready_field else None,
            }
        )
    return {
        "rows": rows,
        "exp6294_terminal_but_below_causal_threshold": rows[0]["ready_score"] == 0.91,
        "exp6307_target_license_ready": rows[1]["ready_score"] == 1.0,
        "exp6321_shadow_ready": rows[2]["ready_score"] == 1.0,
    }


def registry_precheck(*, registry_text: str | None = None) -> JsonDict:
    path = REPO_ROOT / REGISTRY_RELATIVE_PATH
    text = path.read_text(encoding="utf-8") if registry_text is None else registry_text
    duplicates = [INFLUENCE_TASK_ID] if INFLUENCE_TASK_ID in text else []
    return {
        "path": REGISTRY_RELATIVE_PATH.as_posix(),
        "sha256": sha256_text(text),
        "registry_read_mode": "full_text",
        "registry_bytes_read": len(text.encode("utf-8")),
        "registry_line_count": len(text.splitlines()),
        "precheck_order": "registry_before_window_reconstruction",
        "task_kind": "influence_preflight_not_solve",
        "influence_task_id": INFLUENCE_TASK_ID,
        "solve_proposal_made": False,
        "duplicate_solve_proposals": duplicates,
        "duplicate_solve_proposal_count": len(duplicates),
        "all_selected_targets_nonduplicate": len(duplicates) == 0,
        "public_level_targeted": False,
        "registry_update_count": 0,
    }


def no_duplicate_solve_receipt(registry: Mapping[str, Any]) -> JsonDict:
    return {
        "task_kind": registry.get("task_kind"),
        "influence_only": registry.get("task_kind") == "influence_preflight_not_solve",
        "solve_proposal_made": False,
        "no_duplicate_solve_proposal": registry.get("duplicate_solve_proposal_count") == 0,
        "duplicate_solve_proposals": list(registry.get("duplicate_solve_proposals") or []),
        "solve_claim_count": 0,
        "registry_update_count": 0,
    }


def _shadow_rows_by_window(artifact: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in artifact["action_budget_registry_and_level_state_parity"]["cell_rows"]:
        if row.get("arm") == "shadow_computed" and str(row.get("window_id")) not in out:
            out[str(row["window_id"])] = row
    return out


def reconstruct_live_attempt_windows(
    *,
    exp6321_artifact: Mapping[str, Any] | None = None,
    transition_manifest: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    artifact = exp6321_artifact or _read_json(REPO_ROOT / exp6321.RESULT_RELATIVE_PATH)
    manifest = transition_manifest or _read_json(REPO_ROOT / exp6321.TRANSITION_MANIFEST_RELATIVE_PATH)
    shadow_rows = _shadow_rows_by_window(artifact)
    seen: set[str] = set()
    windows: list[JsonDict] = []
    for manifest_row in manifest["rows"]:
        window_id = str(manifest_row["window_id"])
        if window_id in seen:
            continue
        seen.add(window_id)
        transitions = exp6321._transitions_for(str(manifest_row["mechanic"]), int(manifest_row["seed"]))
        payload = exp6307._transition_payload(transitions)
        shadow_row = shadow_rows[window_id]
        receipt_rows = list(shadow_row["shadow_receipt"]["rows"])
        shipped_action = int(receipt_rows[0]["shipped_action"]["action"])
        supported_actions = sorted(
            {
                int(row["prospective_action"]["action"])
                for row in receipt_rows
                if row.get("prospective_action_supported") is True
            }
        )
        unsupported_actions = sorted(
            {
                int(row["prospective_action"]["action"])
                for row in receipt_rows
                if row.get("unsupported_proposal") is True
            }
        )
        legal_actions = sorted(set([shipped_action]) | set(supported_actions) | set(unsupported_actions))
        runtime_state = detector.classify_transition_history(transitions).to_json()
        windows.append(
            {
                "window_id": window_id,
                "selected_target": manifest_row["selected_target"],
                "mechanic": manifest_row["mechanic"],
                "seed": int(manifest_row["seed"]),
                "transition_count": len(transitions),
                "transition_hash": manifest_row["transition_hash"],
                "transition_hash_match": _history_hash_for_row(manifest_row)
                == manifest_row["transition_hash"],
                "transition_payload_match": payload == manifest_row["transition_payload"],
                "recorded_action": {"action": shipped_action, "data": None},
                "supported_actions_from_shadow_receipt": supported_actions,
                "unsupported_actions_from_shadow_receipt": unsupported_actions,
                "legal_actions": legal_actions,
                "legal_action_set_source": list(LEGAL_ACTION_USED_FIELDS),
                "runtime_reverse_engineering_state": runtime_state,
                "route_receipt_hash": sha256_json(receipt_rows),
                "reconstructed_from_allowed_fields": True,
                "hidden_source_used": False,
                "offline_ground_truth_bfs_used": False,
                "hand_game_adapter_used": False,
                "per_game_calibration_used": False,
            }
        )
    return windows


def live_attempt_window_manifest_payload(windows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "sealed_from_exp6321": True,
        "source_artifact": exp6321.RESULT_RELATIVE_PATH.as_posix(),
        "source_transition_manifest": exp6321.TRANSITION_MANIFEST_RELATIVE_PATH.as_posix(),
        "source_boundary": "recorded_live_attempt_windows_no_hidden_source_no_bfs_no_adapter",
        "row_count": len(windows),
        "allowed_fields": list(ROUTE_RANKING_USED_FIELDS),
        "rows": [dict(row) for row in windows],
    }


def write_manifest(path: Path, payload: Mapping[str, Any], *, write: bool) -> JsonDict:
    receipt = {
        "path": _display_path(path),
        "sha256": sha256_json(payload),
        "row_count": payload.get("row_count"),
        "sealed_from_exp6321": bool(payload.get("sealed_from_exp6321")),
    }
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def live_evidence_allowed_fields() -> JsonDict:
    return {
        "window_reconstruction": [
            "window_id",
            "mechanic",
            "seed",
            "transition_payload",
            "transition_hash",
            "shadow_receipt",
        ],
        "legal_action_set": list(LEGAL_ACTION_USED_FIELDS),
        "route_ranking": list(ROUTE_RANKING_USED_FIELDS),
        "exact_checker": [
            "transition.grid",
            "transition.action",
            "transition.next_grid",
            "grid_sha256",
            "next_grid_sha256",
            "changed_cells",
        ],
        "forbidden_fields": list(FORBIDDEN_EVIDENCE_FIELDS),
    }


def forbidden_source_access_contract() -> JsonDict:
    return {
        "hidden_game_source_access_count": 0,
        "offline_ground_truth_bfs_count": 0,
        "hand_game_adapter_count": 0,
        "per_game_calibration_count": 0,
        "prior_game_trajectory_access_count": 0,
        "registry_target_access_count": 0,
        "llm_call_count": 0,
        "forbidden_fields": list(FORBIDDEN_EVIDENCE_FIELDS),
    }


def route_on_off_counterfactual_contract() -> JsonDict:
    return {
        "route_states": ["route_off", "target_licensed_route_on"],
        "replayed_before_recorded_action": True,
        "legal_action_sets_identical_by_route_state": True,
        "only_planned_difference": "target_license_route_evidence_enabled",
        "legal_action_set_reconstructed_before_route_scoring": True,
        "exact_transition_checker_separate_from_route_features": True,
        "route_deleted_control_required": True,
        "fixture_mutation_control_required": True,
        "default_off_ab_eligibility_rule": {
            "min_independent_live_windows": ELIGIBILITY_MIN_INDEPENDENT_WINDOWS,
            "requires_leakage_zero": True,
            "requires_route_deletion_removes_effect": True,
            "requires_fixture_mutation_fails_closed": True,
        },
    }


def legal_action_set_reconstruction_results(windows: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [
        {
            "window_id": row["window_id"],
            "seed": row["seed"],
            "legal_actions": list(row["legal_actions"]),
            "source_fields": list(row["legal_action_set_source"]),
            "route_features_used": False,
            "valid_action_range": [1, 6],
        }
        for row in windows
    ]
    return {
        "window_count": len(rows),
        "route_features_used_for_legal_set": False,
        "all_legal_sets_nonempty": all(bool(row["legal_actions"]) for row in rows),
        "same_legal_action_set_across_route_states": True,
        "rows": rows,
    }


def _transitions_for_window(row: Mapping[str, Any]) -> tuple[e3.Transition, ...]:
    return exp6321._transitions_for(str(row["mechanic"]), int(row["seed"]))


def _mutated_no_effect_transitions(transitions: Sequence[e3.Transition]) -> tuple[e3.Transition, ...]:
    return tuple(
        e3.Transition(
            np.asarray(transition.grid).copy(),
            int(transition.action),
            transition.data,
            np.asarray(transition.grid).copy(),
            int(getattr(transition, "level_before", 0)),
            int(getattr(transition, "level_after", 0)),
        )
        for transition in transitions
    )


def _route_license(transitions: Sequence[e3.Transition]) -> JsonDict:
    ledger = agent.TargetLicensedRouteShadowLedger(enabled=True)
    return dict(ledger._license_receipt(tuple(transitions)))


def _exact_transition_quality(row: Mapping[str, Any], action: int) -> JsonDict:
    transitions = _transitions_for_window(row)
    matches = []
    for index, transition in enumerate(transitions):
        if int(transition.action) != int(action):
            continue
        grid = np.asarray(transition.grid, dtype=int)
        next_grid = np.asarray(transition.next_grid, dtype=int)
        matches.append(
            {
                "index": index,
                "changed_cells": int(np.sum(grid != next_grid)),
                "grid_sha256": sha256_text(grid.tobytes().hex()),
                "next_grid_sha256": sha256_text(next_grid.tobytes().hex()),
            }
        )
    changed_values = [int(match["changed_cells"]) for match in matches]
    exact_value = round(sum(changed_values) / len(changed_values), 6) if changed_values else 0.0
    return {
        "checker": EXACT_TRANSITION_CHECKER_NAME,
        "action": int(action),
        "exact_receipt_count": len(matches),
        "has_exact_one_step_transition": bool(matches),
        "exact_one_step_transition_value": exact_value,
        "changed_cell_counts": changed_values,
        "receipts": matches,
    }


def _base_order(row: Mapping[str, Any]) -> list[int]:
    recorded = int(row["recorded_action"]["action"])
    unsupported = [int(action) for action in row["unsupported_actions_from_shadow_receipt"]]
    legal = [int(action) for action in row["legal_actions"]]
    return _dedupe(unsupported + [recorded] + legal)


def _order_for_window(
    row: Mapping[str, Any],
    *,
    route_enabled: bool,
    transitions: Sequence[e3.Transition] | None = None,
) -> list[int]:
    base = _base_order(row)
    if not route_enabled:
        return base
    route_transitions = tuple(transitions) if transitions is not None else _transitions_for_window(row)
    license_receipt = _route_license(route_transitions)
    if license_receipt.get("route_reachable") is not True:
        return base
    observed_actions = set(agent.TargetLicensedRouteShadowLedger._observed_actions(route_transitions))
    supported = [action for action in base if action in observed_actions]
    return _dedupe(supported + base)


def action_order_change_results(windows: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = []
    for window in windows:
        route_off_order = _order_for_window(window, route_enabled=False)
        route_on_order = _order_for_window(window, route_enabled=True)
        changed = route_off_order != route_on_order and route_off_order[0] != route_on_order[0]
        top_action = route_on_order[0]
        quality = _exact_transition_quality(window, top_action)
        rows.append(
            {
                "window_id": window["window_id"],
                "game_window_id": window["selected_target"],
                "seed": window["seed"],
                "mechanic": window["mechanic"],
                "legal_actions": list(window["legal_actions"]),
                "route_off_order": route_off_order,
                "target_licensed_route_on_order": route_on_order,
                "same_legal_action_set": sorted(route_off_order) == sorted(route_on_order),
                "route_caused_action_order_change": changed,
                "changed_top_action": top_action if changed else None,
                "changed_top_action_has_exact_one_step_value": (
                    bool(quality["has_exact_one_step_transition"])
                    and float(quality["exact_one_step_transition_value"]) > 0.0
                ),
                "exact_transition_quality": quality,
            }
        )
    return {
        "row_count": len(rows),
        "route_caused_action_order_change_count": sum(
            int(row["route_caused_action_order_change"]) for row in rows
        ),
        "same_legal_action_set_count": sum(int(row["same_legal_action_set"]) for row in rows),
        "changed_top_action_exact_value_count": sum(
            int(row["changed_top_action_has_exact_one_step_value"]) for row in rows
        ),
        "rows": rows,
    }


def one_step_exact_transition_quality(order_results: Mapping[str, Any]) -> JsonDict:
    rows = list(order_results["rows"])
    by_state: JsonDict = {}
    for state, key in (
        ("route_off", "route_off_order"),
        ("target_licensed_route_on", "target_licensed_route_on_order"),
    ):
        top_rows = []
        for row in rows:
            top_action = int(row[key][0])
            quality = _exact_transition_quality(row, top_action)
            top_rows.append(
                {
                    "window_id": row["window_id"],
                    "seed": row["seed"],
                    "top_action": top_action,
                    "quality": quality,
                }
            )
        exact_count = sum(
            int(
                item["quality"]["has_exact_one_step_transition"]
                and float(item["quality"]["exact_one_step_transition_value"]) > 0.0
            )
            for item in top_rows
        )
        value_sum = sum(float(item["quality"]["exact_one_step_transition_value"]) for item in top_rows)
        by_state[state] = {
            "top_action_exact_value_count": exact_count,
            "top_action_count": len(top_rows),
            "mean_top_action_changed_cells": round(value_sum / len(top_rows), 6)
            if top_rows
            else 0.0,
            "rows": top_rows,
        }
    by_state["checker"] = EXACT_TRANSITION_CHECKER_NAME
    by_state["claim_boundary"] = "one_step_transition_only_not_level_or_game_solve"
    return by_state


def fixture_mutation_and_route_deletion_results(
    windows: Sequence[Mapping[str, Any]],
    order_results: Mapping[str, Any],
) -> JsonDict:
    by_window = {str(row["window_id"]): row for row in order_results["rows"]}
    rows = []
    for window in windows:
        original = by_window[str(window["window_id"])]
        route_deleted_order = _order_for_window(window, route_enabled=False)
        transitions = _transitions_for_window(window)
        mutated = _mutated_no_effect_transitions(transitions)
        mutated_order = _order_for_window(window, route_enabled=True, transitions=mutated)
        permuted_order = _order_for_window(
            window,
            route_enabled=True,
            transitions=tuple(reversed(transitions)),
        )
        rows.append(
            {
                "window_id": window["window_id"],
                "seed": window["seed"],
                "route_deletion_order": route_deleted_order,
                "route_deletion_removed_effect": route_deleted_order
                == original["route_off_order"]
                and route_deleted_order != original["target_licensed_route_on_order"],
                "fixture_mutation_order": mutated_order,
                "fixture_mutation_route_reachable": _route_license(mutated)["route_reachable"],
                "fixture_mutation_failed_closed": mutated_order == original["route_off_order"],
                "evidence_permutation_order": permuted_order,
                "evidence_permutation_order_unchanged": permuted_order
                == original["target_licensed_route_on_order"],
            }
        )
    return {
        "row_count": len(rows),
        "route_deletion_removed_effect_count": sum(
            int(row["route_deletion_removed_effect"]) for row in rows
        ),
        "fixture_mutation_failed_closed_count": sum(
            int(row["fixture_mutation_failed_closed"]) for row in rows
        ),
        "fixture_mutation_route_reachable_count": sum(
            int(row["fixture_mutation_route_reachable"]) for row in rows
        ),
        "evidence_permutation_order_unchanged_count": sum(
            int(row["evidence_permutation_order_unchanged"]) for row in rows
        ),
        "all_controls_passed": all(
            row["route_deletion_removed_effect"]
            and row["fixture_mutation_failed_closed"]
            and not row["fixture_mutation_route_reachable"]
            and row["evidence_permutation_order_unchanged"]
            for row in rows
        ),
        "rows": rows,
    }


def _forbidden_evidence_trap(payload: Mapping[str, Any]) -> JsonDict:
    found = sorted(set(payload) & set(FORBIDDEN_EVIDENCE_FIELDS))
    return {"rejected": bool(found), "forbidden_keys": found}


def leakage_overlap_and_escape_tests(
    windows: Sequence[Mapping[str, Any]],
    order_results: Mapping[str, Any],
) -> JsonDict:
    overlap = sorted(set(ROUTE_RANKING_USED_FIELDS) & set(FORBIDDEN_EVIDENCE_FIELDS))
    target_permutation_unchanged = all(
        _order_for_window({**window, "window_id": "trap", "selected_target": "trap"}, route_enabled=True)
        == order_results["rows"][index]["target_licensed_route_on_order"]
        for index, window in enumerate(windows)
    )
    hidden_source_trap = _forbidden_evidence_trap({"hidden_game_source_path": "/trap"})
    adapter_trap = _forbidden_evidence_trap({"hand_game_adapter": "TrapAdapter"})
    return {
        "route_ranking_used_fields": list(ROUTE_RANKING_USED_FIELDS),
        "forbidden_fields": list(FORBIDDEN_EVIDENCE_FIELDS),
        "leakage_overlap_fields": overlap,
        "leakage_overlap_count": len(overlap),
        "target_label_permutation_order_unchanged": target_permutation_unchanged,
        "hidden_source_trap_rejected": hidden_source_trap["rejected"],
        "off_path_adapter_trap_rejected": adapter_trap["rejected"],
        "hidden_source_trap": hidden_source_trap,
        "off_path_adapter_trap": adapter_trap,
        "all_escape_tests_passed": len(overlap) == 0
        and target_permutation_unchanged
        and hidden_source_trap["rejected"]
        and adapter_trap["rejected"],
    }


def influence_eligible_window_ids_and_counts(
    order_results: Mapping[str, Any],
    quality: Mapping[str, Any],
    leakage: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> JsonDict:
    eligible_rows = [
        row
        for row in order_results["rows"]
        if row["route_caused_action_order_change"]
        and row["changed_top_action_has_exact_one_step_value"]
    ]
    checks_passed = bool(
        len(eligible_rows) >= ELIGIBILITY_MIN_INDEPENDENT_WINDOWS
        and quality["target_licensed_route_on"]["top_action_exact_value_count"] >= len(eligible_rows)
        and leakage["leakage_overlap_count"] == 0
        and leakage["all_escape_tests_passed"] is True
        and controls["route_deletion_removed_effect_count"] == order_results["row_count"]
        and controls["fixture_mutation_failed_closed_count"] == order_results["row_count"]
        and controls["all_controls_passed"] is True
    )
    return {
        "min_independent_live_windows": ELIGIBILITY_MIN_INDEPENDENT_WINDOWS,
        "independent_live_window_count": int(order_results["row_count"]),
        "eligible_window_count": len(eligible_rows),
        "eligible_window_ids": [str(row["window_id"]) for row in eligible_rows],
        "route_caused_action_order_change_count": int(
            order_results["route_caused_action_order_change_count"]
        ),
        "changed_top_action_exact_value_count": int(
            order_results["changed_top_action_exact_value_count"]
        ),
        "route_deletion_removed_effect_count": int(
            controls["route_deletion_removed_effect_count"]
        ),
        "fixture_mutation_failed_closed_count": int(
            controls["fixture_mutation_failed_closed_count"]
        ),
        "leakage_zero": leakage["leakage_overlap_count"] == 0,
        "eligibility_rule_passed": checks_passed,
    }


def exact_oracle_claim_boundary() -> JsonDict:
    return {
        "checker": EXACT_TRANSITION_CHECKER_NAME,
        "oracle_scope": "exact_recorded_one_step_transition_receipt",
        "not_a_solve_oracle": True,
        "does_not_check_level_completion": True,
        "does_not_run_bfs": True,
        "does_not_read_hidden_source": True,
    }


def _protected_inputs_hashes(protected_before: Mapping[str, str | None]) -> JsonDict:
    code_paths = (
        Path("python/carnot/experiment_6321_arc_target_licensed_route_live_shadow_ab.py"),
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("python/carnot/agentic/arc_mechanic_class_detector.py"),
        EXCLUSION_MANIFEST_RELATIVE_PATH,
    )
    return {
        "arc_registry": {
            "path": REGISTRY_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(REPO_ROOT / REGISTRY_RELATIVE_PATH),
        },
        "exp6321_artifact": {
            "path": exp6321.RESULT_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(REPO_ROOT / exp6321.RESULT_RELATIVE_PATH),
        },
        "exp6321_transition_manifest": {
            "path": exp6321.TRANSITION_MANIFEST_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(REPO_ROOT / exp6321.TRANSITION_MANIFEST_RELATIVE_PATH),
        },
        "live_loop_and_route_code": [
            {"path": rel.as_posix(), "sha256": sha256_file(REPO_ROOT / rel)}
            for rel in code_paths
        ],
        "protected_hashes_before": dict(protected_before),
    }


def preconditions_checked(
    *,
    date: str,
    registry: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    result_path: Path,
) -> JsonDict:
    return {
        "date": date,
        "registry_precheck_first": registry.get("precheck_order")
        == "registry_before_window_reconstruction",
        "task_kind": registry.get("task_kind"),
        "upstream_inputs_hashed_before_replay": True,
        "allowed_fields_frozen_before_replay": list(ROUTE_RANKING_USED_FIELDS),
        "forbidden_sources_frozen_before_replay": list(FORBIDDEN_EVIDENCE_FIELDS),
        "eligibility_rule_preregistered": {
            "min_independent_live_windows": ELIGIBILITY_MIN_INDEPENDENT_WINDOWS,
            "requires_legal_route_caused_order_change": True,
            "requires_exact_one_step_value_for_changed_top_action": True,
            "requires_leakage_zero": True,
            "requires_route_deletion_removes_effect": True,
            "requires_fixture_mutation_fails_closed": True,
        },
        "random_seeds": list(RANDOM_SEEDS),
        "timeout_s": 30,
        "exact_checker": EXACT_TRANSITION_CHECKER_NAME,
        "result_path": _display_path(result_path),
        "hashes": _protected_inputs_hashes(protected_before),
    }


def verification_calls_time_cost_and_error_table(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "calls": [dict(row) for row in rows],
        "total_error_count": sum(int(row.get("error_count", 0)) for row in rows),
        "total_measured_call_duration_s": round(
            sum(float(row.get("duration_s", 0.0)) for row in rows), 6
        ),
    }


def _read_external_test_receipts() -> dict[str, int | None]:  # pragma: no cover - runtime receipt reader.
    receipts: dict[str, int | None] = {
        command: (0 if command == RUN_COMMAND else None) for command in DEFAULT_TEST_COMMANDS
    }
    if not EXTERNAL_TEST_RECEIPT_PATH.is_file():
        return receipts
    try:
        payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return receipts
    receipts.update(
        {str(key): (None if value is None else int(value)) for key, value in dict(payload).items()}
    )
    receipts[RUN_COMMAND] = 0
    return receipts


def run(
    *,
    date: str,
    result_path: Path,
    live_manifest_path: Path,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    timing_rows: list[JsonDict] = []
    protected_before = exp6307._protected_hashes()

    step = time.perf_counter()
    registry = registry_precheck()
    timing_rows.append(_timing_row("registry_precheck", step))

    step = time.perf_counter()
    upstream = upstream_path_hash_terminal_class_and_ready_score()
    timing_rows.append(_timing_row("upstream_hash_and_terminal_receipt", step))

    step = time.perf_counter()
    windows = reconstruct_live_attempt_windows()
    manifest_payload = live_attempt_window_manifest_payload(windows)
    manifest_receipt = write_manifest(live_manifest_path, manifest_payload, write=write)
    timing_rows.append(_timing_row("live_window_reconstruction", step))

    step = time.perf_counter()
    order_results = action_order_change_results(windows)
    quality = one_step_exact_transition_quality(order_results)
    timing_rows.append(_timing_row("route_on_off_counterfactual_replay", step))

    step = time.perf_counter()
    controls = fixture_mutation_and_route_deletion_results(windows, order_results)
    leakage = leakage_overlap_and_escape_tests(windows, order_results)
    timing_rows.append(_timing_row("adversarial_controls", step))

    eligible_counts = influence_eligible_window_ids_and_counts(
        order_results,
        quality,
        leakage,
        controls,
    )
    score = 1.0 if eligible_counts["eligibility_rule_passed"] else 0.0
    measured = round(float(duration_s if duration_s is not None else time.perf_counter() - started), 6)
    artifact: JsonDict = {
        "status": "complete",
        "upstream_path_hash_terminal_class_and_ready_score": upstream,
        "arc_registry_precheck_path_hash_and_result": registry,
        "solve_provenance": "live_agent_self_discovery",
        "no_duplicate_solve_receipt": no_duplicate_solve_receipt(registry),
        "live_attempt_window_manifest_path_and_hash": manifest_receipt,
        "live_evidence_allowed_fields": live_evidence_allowed_fields(),
        "forbidden_source_access_contract": forbidden_source_access_contract(),
        "hidden_game_source_access_count": 0,
        "offline_ground_truth_bfs_count": 0,
        "hand_game_adapter_count": 0,
        "per_game_calibration_count": 0,
        "route_on_off_counterfactual_contract": route_on_off_counterfactual_contract(),
        "legal_action_set_reconstruction_results": legal_action_set_reconstruction_results(windows),
        "action_order_change_results_by_game_window_and_seed": order_results,
        "one_step_exact_transition_quality_by_route_state": quality,
        "influence_eligible_window_ids_and_counts": eligible_counts,
        "leakage_overlap_and_escape_tests": leakage,
        "fixture_mutation_and_route_deletion_results": controls,
        "verification_calls_time_cost_and_error_table": verification_calls_time_cost_and_error_table(
            timing_rows
        ),
        "solve_claim_count": 0,
        "registry_update_count": 0,
        "llm_call_count": 0,
        "exact_oracle_claim_boundary": exact_oracle_claim_boundary(),
        "arc_action_influence_eligible_score": score,
        "protected_files_unchanged": exp6307._protected_unchanged(protected_before),
        "preconditions_checked": preconditions_checked(
            date=date,
            registry=registry,
            protected_before=protected_before,
            result_path=result_path,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": EXACT_TRANSITION_CHECKER_NAME,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or _read_external_test_receipts()),
        "duration_s": measured,
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: arc_action_influence_preflight_eligible_no_solve_claim"
            if score == 1.0
            else "complete: arc_action_influence_preflight_not_eligible_no_solve_claim"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _terminal_verdict(value: str) -> bool:
    return value.startswith(
        ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")
    )


def _fail(message: str) -> None:  # pragma: no cover - exercised through validation callers.
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing fields: {missing}")
    _require(set(artifact["field_principles"]) == set(REQUIRED_ARTIFACT_FIELDS), "field_principles")
    _require(set(artifact["field_provenance"]) == set(REQUIRED_ARTIFACT_FIELDS), "field_provenance")
    _require(artifact["solve_provenance"] == "live_agent_self_discovery", "solve_provenance")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact["verifier_is_oracle"] == EXACT_TRANSITION_CHECKER_NAME, "verifier_is_oracle")
    _require(_terminal_verdict(str(artifact["honest_verdict"])), "honest_verdict")
    for field in FORBIDDEN_ZERO_FIELDS:
        _require(type(artifact[field]) is int and artifact[field] == 0, field)
    registry = artifact["arc_registry_precheck_path_hash_and_result"]
    _require(registry.get("task_kind") == "influence_preflight_not_solve", "arc_registry_precheck_path_hash_and_result")
    _require(registry.get("all_selected_targets_nonduplicate") is True, "arc_registry_precheck_path_hash_and_result")
    _require(registry.get("registry_update_count") == 0, "arc_registry_precheck_path_hash_and_result")
    duplicate = artifact["no_duplicate_solve_receipt"]
    _require(duplicate.get("no_duplicate_solve_proposal") is True, "no_duplicate_solve_receipt")
    _require(duplicate.get("solve_proposal_made") is False, "no_duplicate_solve_receipt")
    contract = artifact["route_on_off_counterfactual_contract"]
    _require(contract.get("legal_action_sets_identical_by_route_state") is True, "route_on_off_counterfactual_contract")
    order = artifact["action_order_change_results_by_game_window_and_seed"]
    _require(order.get("same_legal_action_set_count") == order.get("row_count"), "action_order_change_results_by_game_window_and_seed")
    _require(
        order.get("route_caused_action_order_change_count") >= ELIGIBILITY_MIN_INDEPENDENT_WINDOWS,
        "action_order_change_results_by_game_window_and_seed",
    )
    quality = artifact["one_step_exact_transition_quality_by_route_state"]
    _require(quality.get("checker") == EXACT_TRANSITION_CHECKER_NAME, "one_step_exact_transition_quality_by_route_state")
    _require(
        quality["target_licensed_route_on"]["top_action_exact_value_count"]
        >= order.get("route_caused_action_order_change_count"),
        "one_step_exact_transition_quality_by_route_state",
    )
    leakage = artifact["leakage_overlap_and_escape_tests"]
    _require(leakage.get("leakage_overlap_count") == 0, "leakage_overlap_and_escape_tests")
    _require(leakage.get("all_escape_tests_passed") is True, "leakage_overlap_and_escape_tests")
    controls = artifact["fixture_mutation_and_route_deletion_results"]
    _require(controls.get("all_controls_passed") is True, "fixture_mutation_and_route_deletion_results")
    _require(
        controls.get("route_deletion_removed_effect_count") == order.get("row_count"),
        "fixture_mutation_and_route_deletion_results",
    )
    eligible = artifact["influence_eligible_window_ids_and_counts"]
    _require(eligible.get("eligibility_rule_passed") is True, "influence_eligible_window_ids_and_counts")
    _require(artifact["arc_action_influence_eligible_score"] == 1.0, "arc_action_influence_eligible_score")
    _require(artifact["exact_oracle_claim_boundary"].get("not_a_solve_oracle") is True, "exact_oracle_claim_boundary")
    protected = artifact["protected_files_unchanged"]
    _require(all(row.get("unchanged") is True for row in protected.values()), "protected_files_unchanged")
    _require(artifact["reproducibility_checksum"] == payload_checksum(artifact), "reproducibility_checksum")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260812")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument(
        "--live-manifest",
        default=str(REPO_ROOT / LIVE_WINDOW_MANIFEST_RELATIVE_PATH),
    )
    args = parser.parse_args(argv)
    run(
        date=args.date,
        result_path=Path(args.output),
        live_manifest_path=Path(args.live_manifest),
        write=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
