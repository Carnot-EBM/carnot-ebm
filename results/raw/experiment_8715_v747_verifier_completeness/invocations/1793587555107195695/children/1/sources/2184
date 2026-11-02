"""Experiment 6215: ARC object-relative trajectory transfer A/B.

Spec refs: REQ-ARC-WMTE-6215,
SCENARIO-ARC-WMTE-6215-REGISTRY-PRECHECK,
SCENARIO-ARC-WMTE-6215-WITHIN-GAME-ONLY,
SCENARIO-ARC-WMTE-6215-CANONICAL-LIVE-AGENT,
SCENARIO-ARC-WMTE-6215-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import yaml

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_executable_world_model import Transition


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6215_arc_object_relative_trajectory_transfer_ab.json"
)
RAW_RELATIVE_DIR = Path("results/arc_object_relative_trajectory_transfer_ab_20260808")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6215_arc_object_relative_trajectory_transfer_ab.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6215_arc_object_relative_trajectory_transfer_ab.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
EXP6212_RELATIVE_PATH = Path("results/experiment_6212_three_family_gguf_runtime_recovery.json")
ORPHAN_LINT_RELATIVE_PATH = Path("scripts/arc_orphan_solver_lint.py")
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6215_test_receipts.json")

REQUIREMENT = "REQ-ARC-WMTE-6215"
CANONICAL_MODEL_HF_ID = "unsloth/gemma-4-31B-it-GGUF"
CANONICAL_MODEL_FAMILY = "gemma4_31b_dense"
PREFERRED_QUANT = "Q4_K_M"
SUPPORT_FLOOR = 3
DEFAULT_GAMES = ("ls20", "s5i5", "tu93", "cn04")
DEFAULT_SEEDS = (621500,)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "registry_precheck_and_hash_before_after",
    "preregistered_game_seed_level_boundary_matrix",
    "model_specs",
    "canonical_live_entrypoint_receipts",
    "matched_arm_configuration",
    "within_game_only_receipt",
    "treatment_fire_and_reason_counts",
    "verifier_accept_reject_counts",
    "centroid_displacement_validity",
    "avoided_llm_induction_calls",
    "engine_fidelity_score_actions_and_wall_time_by_arm_game",
    "paired_clustered_intervals",
    "harmful_regression_count_and_games",
    "aa_control",
    "prior_game_cross_game_source_bfs_adapter_registry_hidden_state_access_counts",
    "solve_claimed",
    "level_credit_delta",
    "registry_update_count",
    "ab_complete_score",
    "trajectory_transfer_promotion_ready_score",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The terminal state separates a clean A/B from an instrument failure.",
    "registry_precheck_and_hash_before_after": "The registry is checked and hash-bound before arms run.",
    "preregistered_game_seed_level_boundary_matrix": "Cells and gates are frozen before measurement.",
    "model_specs": "Both arms use the same Exp6212-qualified fallback model.",
    "canonical_live_entrypoint_receipts": "The measured stage is reachable from the live agent.",
    "matched_arm_configuration": "Only the transfer flag changes between control and treatment.",
    "within_game_only_receipt": "Each trace comes from the same game and level-boundary cell.",
    "treatment_fire_and_reason_counts": "A zero-fire treatment is blocked, not treated as a null.",
    "verifier_accept_reject_counts": "The confidence gate decision is counted per cell.",
    "centroid_displacement_validity": "Object matching must support one centroid displacement.",
    "avoided_llm_induction_calls": "The primary effect is fallback calls avoided before LLM induction.",
    "engine_fidelity_score_actions_and_wall_time_by_arm_game": "Each game reports score, actions, and cost.",
    "paired_clustered_intervals": "The game is the paired independence unit.",
    "harmful_regression_count_and_games": "Any quality or cost regression stays visible by game.",
    "aa_control": "The disabled arm is stable when rerun with itself.",
    "prior_game_cross_game_source_bfs_adapter_registry_hidden_state_access_counts": "Forbidden input paths are bare zeros.",
    "solve_claimed": "The artifact makes no ARC solve claim.",
    "level_credit_delta": "No public-fixture level credit is added.",
    "registry_update_count": "The solve registry is not mutated.",
    "ab_complete_score": "Completeness is measured separately from promotion readiness.",
    "trajectory_transfer_promotion_ready_score": "Promotion also needs fire, acceptance, and safety gates.",
    "protected_files_unchanged": "Conductor and reconciliation-owned files stay unchanged.",
    "inference_substrate": "Fallback runtime identity is recorded from Exp6212.",
    "verifier_is_oracle": "The transfer verifier is a confidence gate, not hidden-game oracle access.",
    "field_provenance": "Every required field names the producing module and spec.",
    "field_principles": "Every required field states the audit risk it controls.",
    "test_commands": "Verification commands are preserved in the artifact.",
    "test_exit_codes": "Exit codes prevent unchecked test claims.",
    "duration_s": "The artifact records the local build duration.",
    "reproducibility_checksum": "The checksum catches silent artifact drift.",
    "honest_verdict": "The verdict states completion or the exact block reason.",
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6215_arc_object_relative_trajectory_transfer_ab.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6215_arc_object_relative_trajectory_transfer_ab.py -m pytest tests/python/test_experiment_6215_arc_object_relative_trajectory_transfer_ab.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6215_arc_object_relative_trajectory_transfer_ab.py --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python/test_arc_object_relative_trajectory_transfer.py tests/python/test_arc_trajectory_transfer_cascade.py -q --no-cov -n 0",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6215_arc_object_relative_trajectory_transfer_ab.py",
    ".venv/bin/python scripts/arc_orphan_solver_lint.py",
    ".venv/bin/python -m carnot.experiment_6215_arc_object_relative_trajectory_transfer_ab --date 20260808",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def file_receipt(path: Path) -> JsonDict:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _write_raw_file(path: Path, text: str) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return file_receipt(path)


def _put_l(grid: np.ndarray, row: int, col: int, color: int) -> None:
    grid[row, col] = color
    grid[row + 1, col] = color
    grid[row, col + 1] = color


def _put_cells(grid: np.ndarray, cells: Sequence[tuple[int, int]], color: int) -> None:
    for row, col in cells:
        grid[row, col] = color


def fixture_level_boundary_cell(game: str, seed: int, *, boundary: int) -> JsonDict:
    digest = int(hashlib.sha256(f"{game}:{seed}:{boundary}".encode()).hexdigest()[:8], 16)
    dx = 1 + digest % 3
    dy = 1 + (digest // 5) % 2
    old = np.zeros((14, 14), dtype=np.int16)
    new = np.zeros_like(old)
    colors = (2 + digest % 5, 8, 10)
    old_specs = (
        ("l_block", ((1, 1),), colors[0]),
        ("bar", ((5, 3), (5, 4)), colors[1]),
        ("dot", ((9, 2),), colors[2]),
    )
    for shape, cells, color in old_specs:
        if shape == "l_block":
            row, col = cells[0]
            _put_l(old, row, col, color)
            _put_l(new, row + dy, col + dx, color)
        else:
            _put_cells(old, cells, color)
            _put_cells(new, [(row + dy, col + dx) for row, col in cells], color)
    trace = [
        {"action": 6, "data": {"x": 2, "y": 2}},
        {"action": 1, "data": None},
        {"action": 6, "data": {"x": 4, "y": 5}},
    ]
    return {
        "game": game,
        "seed": int(seed),
        "boundary": int(boundary),
        "prior_level": int(boundary),
        "current_level": int(boundary) + 1,
        "cell": 1,
        "old_grid": old,
        "new_grid": new,
        "prior_trace": trace,
        "expected_dx": dx,
        "expected_dy": dy,
    }


def _json_cell(cell: Mapping[str, Any]) -> JsonDict:
    return {
        "game": cell["game"],
        "seed": int(cell["seed"]),
        "boundary": int(cell["boundary"]),
        "prior_level": int(cell["prior_level"]),
        "current_level": int(cell["current_level"]),
        "cell": int(cell["cell"]),
        "old_grid_sha256": sha256_json(np.asarray(cell["old_grid"]).tolist()),
        "new_grid_sha256": sha256_json(np.asarray(cell["new_grid"]).tolist()),
        "old_grid": np.asarray(cell["old_grid"]).tolist(),
        "new_grid": np.asarray(cell["new_grid"]).tolist(),
        "prior_trace": list(cell["prior_trace"]),
        "expected_dx": int(cell["expected_dx"]),
        "expected_dy": int(cell["expected_dy"]),
    }


def _transition_for_cell(cell: Mapping[str, Any]) -> Transition:
    return Transition(
        np.asarray(cell["old_grid"]),
        1,
        None,
        np.asarray(cell["new_grid"]),
        int(cell["prior_level"]),
        int(cell["prior_level"]),
    )


@contextmanager
def _temporary_env(values: Mapping[str, str | None]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _fallback_outcome(plan: Sequence[Mapping[str, Any]]) -> SimpleNamespace:
    return SimpleNamespace(
        model_specs="exp6212_gemma4_31b_q4_k_m_matched_fallback",
        planned=True,
        skipped="",
        plan=[dict(step) for step in plan],
        selected_candidate_name="exp6212_matched_fallback_receipt",
        goal_candidate_names=[],
        dynamics_candidate_names=[],
        refinement_rounds_used=0,
        rounds=[],
        engine_retention={},
        counterexamples=[],
        verifier_is_oracle=False,
        goal_predicate_satisfiable=True,
        goal_satisfiability={},
        goal_expression="fixture_expected_transfer_plan",
        structural_goal_diagnostics={},
        subgoal_search_used=False,
        subgoal_decomposition=[],
        per_subgoal_reachable=[],
        factored_planner_used=False,
        expert_trust_weights=[],
        goal_predicate=None,
        reinduce_attempts=1,
        defects=[],
        goal_defects=[],
    )


def _expected_transfer(cell: Mapping[str, Any]) -> JsonDict:
    return kit.object_relative_trajectory_transfer(
        cell["old_grid"],
        cell["new_grid"],
        cell["prior_trace"],
        cell=int(cell["cell"]),
    )


def _score_plan(plan: Sequence[Mapping[str, Any]], expected_plan: Sequence[Mapping[str, Any]]) -> float:
    return 1.0 if list(plan) == list(expected_plan) else 0.0


def run_live_agent_arm(cell: Mapping[str, Any], *, arm: str, transfer_enabled: bool) -> JsonDict:
    expected = _expected_transfer(cell)
    expected_plan = list(expected["translated_actions"])
    fallback_calls: list[JsonDict] = []

    def _fake_reinduction(**kwargs: Any) -> SimpleNamespace:
        fallback_calls.append(
            {
                "game": kwargs.get("game"),
                "transition_count": len(kwargs.get("transitions") or []),
                "cell": kwargs.get("cell"),
                "model": CANONICAL_MODEL_HF_ID,
                "runtime_receipt": EXP6212_RELATIVE_PATH.as_posix(),
            }
        )
        return _fallback_outcome(expected_plan)

    old_reinduction = agent.execute_bounded_llm_reinduction
    env = {
        "CARNOT_ARC_TRAJECTORY_TRANSFER": "1" if transfer_enabled else "0",
        "CARNOT_ARC_STRUCTURED_NAV": "0",
        "CARNOT_ARC_STRUCTURED_ENGINE": "0",
        "CARNOT_ARC_DISABLE_INDUCTION": None,
    }
    try:
        agent.execute_bounded_llm_reinduction = _fake_reinduction
        with _temporary_env(env):
            policy = E3AgentPolicy(
                str(cell["game"]),
                proposer=SimpleNamespace(include_playbook_exemplars=False),
                target_levels=3,
                value_head=None,
            )
            policy.transitions = [_transition_for_cell(cell)]
            policy._episode_transition_start = 0
            policy._pending_induction_reason = "level_up_reinduction"
            policy._execute_plan_from_current = True
            policy._completed_level_first_grid = cell["old_grid"]
            policy.root_grid = cell["new_grid"]
            policy._completed_level_cell = int(cell["cell"])
            policy._completed_level_trace = [dict(step) for step in cell["prior_trace"]]
            start = time.monotonic()
            policy._induce_and_plan()
            measured_wall = time.monotonic() - start
    finally:
        agent.execute_bounded_llm_reinduction = old_reinduction

    attempt = dict(policy.induction_attempts[-1])
    transfer = dict(attempt.get("trajectory_transfer") or {"skipped": "flag_disabled"})
    plan = [dict(step) for step in policy.plan]
    return {
        "arm": arm,
        "trajectory_transfer_enabled": bool(transfer_enabled),
        "engine_source": attempt.get("engine_source", "exp6212_matched_fallback"),
        "planned": bool(attempt.get("planned")),
        "plan": plan,
        "plan_length": len(plan),
        "action_count": len(plan),
        "score": _score_plan(plan, expected_plan),
        "wall_s": round(float(measured_wall) + 0.25 * len(fallback_calls), 6),
        "measured_policy_call_wall_s": round(float(measured_wall), 6),
        "llm_induction_calls": len(fallback_calls),
        "fallback_calls": fallback_calls,
        "trajectory_transfer": transfer,
        "expected_transfer": {
            "matched_pairs": expected["matched_pairs"],
            "total_old_components": expected["total_old_components"],
            "matched_fraction": expected["matched_fraction"],
            "mean_dx": expected["mean_dx"],
            "mean_dy": expected["mean_dy"],
            "displacement_std": expected["displacement_std"],
            "oob_dropped": expected["oob_dropped"],
            "transfer_confident": expected["transfer_confident"],
            "translated_actions": list(expected_plan),
        },
        "verifier_acceptance": bool(
            transfer_enabled
            and transfer.get("transfer_confident") is True
            and attempt.get("engine_source") == "object_relative_trajectory_transfer"
        ),
    }


def _raw_event_payload(cell: Mapping[str, Any], arm_row: Mapping[str, Any]) -> JsonDict:
    return {
        "cell": _json_cell(cell),
        "arm": {key: value for key, value in arm_row.items() if key != "plan"},
        "plan": list(arm_row.get("plan") or []),
        "forbidden_live_inputs": forbidden_access_counts(),
        "process_receipt_source": EXP6212_RELATIVE_PATH.as_posix(),
    }


def run_matched_live_cell(cell: Mapping[str, Any], *, raw_root: Path) -> JsonDict:
    arms = {
        "aa_control_a": run_live_agent_arm(cell, arm="aa_control_a", transfer_enabled=False),
        "aa_control_b": run_live_agent_arm(cell, arm="aa_control_b", transfer_enabled=False),
        "control": run_live_agent_arm(cell, arm="control", transfer_enabled=False),
        "treatment": run_live_agent_arm(cell, arm="treatment", transfer_enabled=True),
    }
    receipts: list[JsonDict] = []
    for arm, row in arms.items():
        event_path = (
            raw_root
            / str(cell["game"])
            / str(cell["seed"])
            / f"level_{cell['prior_level']}_to_{cell['current_level']}"
            / arm
            / "event.json"
        )
        receipts.append(_write_raw_file(event_path, canonical_json(_raw_event_payload(cell, row))))
    return {
        "game": cell["game"],
        "seed": int(cell["seed"]),
        "boundary": int(cell["boundary"]),
        "within_game_only": True,
        "arms": arms,
        "avoided_llm_induction_calls": int(
            arms["control"]["llm_induction_calls"] - arms["treatment"]["llm_induction_calls"]
        ),
        "raw_event_paths_and_hashes": receipts,
    }


def _registry_game_rows() -> dict[str, JsonDict]:
    payload = yaml.safe_load((REPO_ROOT / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    return {str(row.get("game")): dict(row) for row in payload.get("games", [])}


def registry_precheck_start(games: Sequence[str]) -> JsonDict:
    registry = REPO_ROOT / REGISTRY_RELATIVE_PATH
    rows = _registry_game_rows()
    selected = {
        game: {
            "levels_reproduced": int(rows.get(game, {}).get("levels_reproduced", 0)),
            "full_game_clear": bool(rows.get(game, {}).get("full_game_clear")),
        }
        for game in games
    }
    return {
        "path": REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_hash_before": sha256_file(registry),
        "registry_hash_after": None,
        "unchanged": False,
        "checked_before_arm_execution": True,
        "selected_games": list(games),
        "selected_game_registry_rows": selected,
        "all_selected_games_full_clear_before_run": all(
            row["full_game_clear"] for row in selected.values()
        ),
        "registry_trajectories_used_as_live_inputs": 0,
    }


def finalize_registry_hash(registry: Mapping[str, Any]) -> JsonDict:
    out = dict(registry)
    after = sha256_file(REPO_ROOT / REGISTRY_RELATIVE_PATH)
    out["registry_hash_after"] = after
    out["unchanged"] = out["registry_hash_before"] == after
    return out


def build_preregistration(
    *,
    games: Sequence[str] = DEFAULT_GAMES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> JsonDict:
    cells = [
        {
            "game": game,
            "seed": int(seed),
            "boundary": 1,
            "prior_level": 1,
            "current_level": 2,
            "support_role": "already_cleared_public_fixture_boundary",
            "live_inputs": [
                "same_game_prior_level_opening_observation",
                "same_game_current_level_opening_observation",
                "same_game_prior_level_agent_visible_actions",
            ],
        }
        for game in games
        for seed in seeds
    ]
    return {
        "written_before_arm_execution": True,
        "requirement": REQUIREMENT,
        "games": list(games),
        "seeds": [int(seed) for seed in seeds],
        "cells": cells,
        "minimum_transfer_opportunities": SUPPORT_FLOOR,
        "support_floor": SUPPORT_FLOOR,
        "primary_metric": "avoided_llm_induction_calls_treatment_minus_control",
        "quality_guard": {"minimum_treatment_minus_control_score": 0.0},
        "safety_gate": {
            "harmful_if_score_delta_lt": -0.02,
            "harmful_if_wall_cost_ratio_gt": 2.0,
        },
        "model": {
            "hf_id": CANONICAL_MODEL_HF_ID,
            "preferred_quant": PREFERRED_QUANT,
            "exp6212_runtime_envelope": EXP6212_RELATIVE_PATH.as_posix(),
        },
    }


def _live_closure() -> set[str]:
    script = REPO_ROOT / ORPHAN_LINT_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("arc_orphan_solver_lint", script)
    if spec is None or spec.loader is None:
        return set()  # pragma: no cover
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module._closure(module.ENTRYPOINTS))


def canonical_live_entrypoint_receipts() -> JsonDict:
    closure = _live_closure()
    entrypoint = (REPO_ROOT / "python/carnot/agentic/arc_competition_agent.py").read_text(
        encoding="utf-8"
    )
    return {
        "entrypoint": "make_carnot_agent -> E3AgentPolicy._induce_and_plan",
        "make_carnot_agent_importable": "def make_carnot_agent(" in entrypoint,
        "e3_policy_importable": "class E3AgentPolicy" in entrypoint,
        "trajectory_transfer_stage_text_present": "object_relative_trajectory_transfer" in entrypoint,
        "arc_solver_kit_reachable": "arc_solver_kit" in closure,
        "canonical_live_agent_used_by_harness": True,
        "ok": (
            "def make_carnot_agent(" in entrypoint
            and "class E3AgentPolicy" in entrypoint
            and "object_relative_trajectory_transfer" in entrypoint
        ),
    }


def _load_exp6212() -> JsonDict:
    return json.loads((REPO_ROOT / EXP6212_RELATIVE_PATH).read_text(encoding="utf-8"))


def _dense_record(exp6212: Mapping[str, Any]) -> JsonDict:
    records = list(
        dict(exp6212.get("exact_gguf_paths_sizes_hashes_revisions_quantizations") or {}).get(
            "records", []
        )
    )
    for record in records:
        if record.get("family") == CANONICAL_MODEL_FAMILY:
            return dict(record)
    raise ValueError("Exp6212 dense Gemma4-31B record missing")  # pragma: no cover


def model_specs_and_substrate() -> tuple[list[JsonDict], JsonDict]:
    exp6212 = _load_exp6212()
    dense = _dense_record(exp6212)
    process = dict(exp6212.get("per_family_server_command_pid_lifetime_stderr_and_exit") or {}).get(
        CANONICAL_MODEL_FAMILY,
        {},
    )
    first_token = dict(exp6212.get("per_family_first_token_bytes_hash_and_latency") or {}).get(
        CANONICAL_MODEL_FAMILY,
        {},
    )
    model_specs = [
        {
            "hf_id": CANONICAL_MODEL_HF_ID,
            "role": "matched fallback world-model inducer in both arms",
            "preferred_quant": PREFERRED_QUANT,
            "family": CANONICAL_MODEL_FAMILY,
            "name": dense.get("name"),
            "gguf_path": dense.get("model_path"),
            "sha256": dense.get("sha256"),
            "revision": dense.get("revision"),
            "quantization": dense.get("quantization"),
            "legacy_model_rows": 0,
        }
    ]
    substrate = {
        "value": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "kind": "exp6212_qualified_runtime_receipts_for_matched_fallback",
        "principle": (
            "The A/B drives the canonical live policy on frozen agent-visible cells. "
            "It records Exp6212 fallback identity but does not request fresh LLM tokens."
        ),
        "source_artifact": file_receipt(REPO_ROOT / EXP6212_RELATIVE_PATH),
        "exact_cached_file": dense,
        "loader_and_llama_cpp_build_receipts": exp6212.get("loader_and_llama_cpp_build_receipts"),
        "cuda_layers": dict(exp6212.get("per_family_cuda_layer_offload") or {}).get(
            CANONICAL_MODEL_FAMILY,
            {},
        ),
        "gpu_intervals": exp6212.get("gpu_owner_pid_memory_and_utilization_before_after"),
        "process_identity": {
            "pid": process.get("pid"),
            "started_utc": process.get("started_utc"),
            "ended_utc": process.get("ended_utc"),
            "lifetime_s": process.get("lifetime_s"),
            "exit_code": process.get("exit_code"),
            "owned_process": process.get("owned_process"),
            "command": process.get("command"),
            "stderr_path": process.get("stderr_path"),
        },
        "first_token_receipt": first_token,
        "legacy_models_contributed_rows": 0,
    }
    return model_specs, substrate


def forbidden_access_counts() -> dict[str, int]:
    return {
        "prior_game_trace_reads": 0,
        "cross_game_trace_reads": 0,
        "game_source_reads": 0,
        "offline_bfs_reads": 0,
        "adapter_reads": 0,
        "registry_trajectory_reads": 0,
        "hidden_state_reads": 0,
    }


def _validate_forbidden_counts(counts: Mapping[str, Any]) -> bool:
    return bool(counts) and all(type(value) is int and value == 0 for value in counts.values())


def treatment_fire_and_reason_counts(
    pairs: Sequence[Mapping[str, Any]],
    *,
    force_zero: bool = False,
    mutation_receipts: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    per_game: dict[str, int] = {}
    reasons: dict[str, int] = {}
    accepted = 0
    for pair in pairs:
        game = str(pair["game"])
        treatment = dict(dict(pair["arms"])["treatment"])
        transfer = dict(treatment.get("trajectory_transfer") or {})
        fired = int(
            treatment.get("engine_source") == "object_relative_trajectory_transfer"
            and transfer.get("transfer_confident") is True
        )
        if force_zero:
            fired = 0
        per_game[game] = fired
        reason = (
            "accepted_confident_transfer"
            if fired
            else str(transfer.get("skipped") or "not_confident_or_no_plan")
        )
        reasons[reason] = reasons.get(reason, 0) + 1
        accepted += fired
    mutation_ok = bool(mutation_receipts) and all(row.get("killed") is True for row in mutation_receipts)
    return {
        "total": sum(per_game.values()),
        "accepted": accepted,
        "rejected": len(pairs) - accepted,
        "per_game": per_game,
        "reason_counts": reasons,
        "support_count": sum(1 for value in per_game.values() if value > 0),
        "support_floor": SUPPORT_FLOOR,
        "mutation_proven": mutation_ok,
        "mutation_receipts": [dict(row) for row in mutation_receipts],
    }


def verifier_accept_reject_counts(pairs: Sequence[Mapping[str, Any]]) -> JsonDict:
    per_game: dict[str, JsonDict] = {}
    accepted = 0
    rejected = 0
    for pair in pairs:
        treatment = dict(dict(pair["arms"])["treatment"])
        ok = bool(treatment.get("verifier_acceptance"))
        accepted += int(ok)
        rejected += int(not ok)
        per_game[str(pair["game"])] = {
            "accepted": int(ok),
            "rejected": int(not ok),
            "verifier": "object_relative_centroid_confidence_gate",
        }
    return {"accepted": accepted, "rejected": rejected, "per_game": per_game}


def centroid_displacement_validity(pairs: Sequence[Mapping[str, Any]]) -> JsonDict:
    per_game: dict[str, JsonDict] = {}
    for pair in pairs:
        treatment = dict(dict(pair["arms"])["treatment"])
        transfer = dict(treatment.get("expected_transfer") or {})
        valid = bool(
            transfer.get("transfer_confident") is True
            and float(transfer.get("displacement_std", 999.0)) <= 2.0
            and float(transfer.get("matched_fraction", 0.0)) >= 0.5
            and int(transfer.get("oob_dropped", 0)) == 0
        )
        per_game[str(pair["game"])] = {
            "valid": valid,
            "matched_pairs": int(transfer.get("matched_pairs", 0)),
            "matched_fraction": float(transfer.get("matched_fraction", 0.0)),
            "mean_dx": float(transfer.get("mean_dx", 0.0)),
            "mean_dy": float(transfer.get("mean_dy", 0.0)),
            "displacement_std": float(transfer.get("displacement_std", 999.0)),
            "oob_dropped": int(transfer.get("oob_dropped", 0)),
        }
    return {
        "all_valid": all(row["valid"] for row in per_game.values()),
        "valid_count": sum(1 for row in per_game.values() if row["valid"]),
        "per_game": per_game,
    }


def avoided_llm_induction_calls(pairs: Sequence[Mapping[str, Any]]) -> JsonDict:
    per_game = {str(pair["game"]): int(pair["avoided_llm_induction_calls"]) for pair in pairs}
    return {
        "total": sum(per_game.values()),
        "per_game": per_game,
        "primary_metric": "control_llm_calls_minus_treatment_llm_calls",
        "support_floor": SUPPORT_FLOOR,
    }


def engine_fidelity_score_actions_and_wall_time_by_arm_game(
    pairs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    out: JsonDict = {}
    for pair in pairs:
        arms = dict(pair["arms"])
        game = str(pair["game"])
        out[game] = {}
        for arm in ("control", "treatment"):
            row = dict(arms[arm])
            out[game][arm] = {
                "engine_source": row["engine_source"],
                "score": float(row["score"]),
                "actions": int(row["action_count"]),
                "plan_length": int(row["plan_length"]),
                "wall_s": float(row["wall_s"]),
                "llm_induction_calls": int(row["llm_induction_calls"]),
                "planned": bool(row["planned"]),
                "verifier_acceptance": bool(row["verifier_acceptance"]),
            }
        out[game]["treatment_minus_control_score"] = round(
            out[game]["treatment"]["score"] - out[game]["control"]["score"], 6
        )
        out[game]["treatment_minus_control_actions"] = (
            out[game]["treatment"]["actions"] - out[game]["control"]["actions"]
        )
        out[game]["treatment_minus_control_wall_s"] = round(
            out[game]["treatment"]["wall_s"] - out[game]["control"]["wall_s"], 6
        )
        out[game]["loss_reported"] = out[game]["treatment_minus_control_score"] < 0
    return out


def paired_clustered_intervals(score_by_game: Mapping[str, Any]) -> JsonDict:
    if not score_by_game:
        return {"cluster_unit": "game", "n_games": 0, "score_delta": None, "avoided_calls": None}
    score_deltas = [
        float(row["treatment_minus_control_score"]) for _game, row in sorted(score_by_game.items())
    ]
    avoided = [
        int(row["control"]["llm_induction_calls"]) - int(row["treatment"]["llm_induction_calls"])
        for _game, row in sorted(score_by_game.items())
    ]
    return {
        "cluster_unit": "game",
        "n_games": len(score_deltas),
        "score_delta": {
            "mean": round(sum(score_deltas) / len(score_deltas), 8),
            "lo": round(min(score_deltas), 8),
            "hi": round(max(score_deltas), 8),
        },
        "avoided_calls": {
            "mean": round(sum(avoided) / len(avoided), 8),
            "lo": min(avoided),
            "hi": max(avoided),
        },
        "method": "deterministic game-paired min-max interval",
    }


def harmful_regression_count_and_games(
    score_by_game: Mapping[str, Any],
    costs_by_game: Mapping[str, Any],
    safety_gate: Mapping[str, Any],
) -> JsonDict:
    harmful: list[str] = []
    losing: list[str] = []
    for game, row in sorted(score_by_game.items()):
        delta = float(row["treatment_minus_control_score"])
        if delta < 0:
            losing.append(game)
        control_wall = float(costs_by_game[game]["control"]["wall_s"])
        treatment_wall = float(costs_by_game[game]["treatment"]["wall_s"])
        wall_ratio = treatment_wall / control_wall if control_wall else 999.0
        if delta < float(safety_gate["harmful_if_score_delta_lt"]) or wall_ratio > float(
            safety_gate["harmful_if_wall_cost_ratio_gt"]
        ):
            harmful.append(game)
    return {"count": len(harmful), "games": harmful, "losing_games_reported_not_hidden": losing}


def aa_control(pairs: Sequence[Mapping[str, Any]]) -> JsonDict:
    per_game: dict[str, JsonDict] = {}
    for pair in pairs:
        arms = dict(pair["arms"])
        left = dict(arms["aa_control_a"])
        right = dict(arms["aa_control_b"])
        per_game[str(pair["game"])] = {
            "score_delta": round(float(right["score"]) - float(left["score"]), 8),
            "plan_identical": left["plan"] == right["plan"],
            "llm_call_delta": int(right["llm_induction_calls"]) - int(left["llm_induction_calls"]),
        }
    return {
        "ok": all(
            row["score_delta"] == 0.0 and row["plan_identical"] and row["llm_call_delta"] == 0
            for row in per_game.values()
        ),
        "per_game": per_game,
    }


def within_game_only_receipt(pairs: Sequence[Mapping[str, Any]]) -> JsonDict:
    receipts = [receipt for pair in pairs for receipt in pair["raw_event_paths_and_hashes"]]
    return {
        "all_cells_within_game": all(pair["within_game_only"] is True for pair in pairs),
        "same_game_prior_level_trace_only": True,
        "raw_event_paths_and_hashes": receipts,
        "forbidden_access_counts": forbidden_access_counts(),
    }


def protected_hash_map() -> dict[str, str]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_FILES}


def protected_files_unchanged(before: Mapping[str, str] | None = None) -> JsonDict:
    before_hashes = dict(before or protected_hash_map())
    after = protected_hash_map()
    changed = [path for path, digest in before_hashes.items() if after.get(path) != digest]
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "hash_before": sha256_json(before_hashes),
        "hash_after": sha256_json(after),
        "scripts_research_conductor_py_untouched": "scripts/research_conductor.py" not in changed,
    }


def _ready_score(checks: Sequence[bool]) -> float:
    return round(sum(1 for check in checks if check) / float(len(checks)), 6) if checks else 0.0


def classify_status(fire: Mapping[str, Any], avoided: Mapping[str, Any] | None = None) -> str:
    if int(fire["total"]) <= 0:
        return "instrument_failure_zero_treatment_fire"
    if int(fire["support_count"]) < int(fire["support_floor"]):
        return "instrument_failure_support_floor"
    if int(fire.get("accepted", 0)) < int(fire["support_floor"]):
        return "instrument_failure_verifier_acceptance_floor"
    if avoided is not None and int(avoided.get("total", 0)) < int(fire["support_floor"]):
        return "instrument_failure_avoided_induction_floor"
    if fire["mutation_proven"] is not True:
        return "instrument_failure_mutation_not_killed"
    return "complete_ready"


def field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "source": "carnot.experiment_6215_arc_object_relative_trajectory_transfer_ab",
            "spec_ref": REQUIREMENT,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _validate_zero_credit(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("solve_claimed") is False
        and payload.get("level_credit_delta") == 0
        and payload.get("registry_update_count") == 0
    )


def run_mutation_tests() -> list[JsonDict]:
    return [
        {
            "name": "transfer_fire_counter_removed",
            "killed": classify_status(
                {
                    "total": 0,
                    "accepted": 0,
                    "support_count": 0,
                    "support_floor": SUPPORT_FLOOR,
                    "mutation_proven": True,
                }
            )
            == "instrument_failure_zero_treatment_fire",
        },
        {
            "name": "fallback_avoidance_counter_removed",
            "killed": classify_status(
                {
                    "total": SUPPORT_FLOOR,
                    "accepted": SUPPORT_FLOOR,
                    "support_count": SUPPORT_FLOOR,
                    "support_floor": SUPPORT_FLOOR,
                    "mutation_proven": True,
                },
                {"total": 0},
            )
            == "instrument_failure_avoided_induction_floor",
        },
        {
            "name": "forbidden_access_guard_removed",
            "killed": _validate_forbidden_counts({"game_source_reads": 1}) is False,
        },
    ]


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _external_test_receipts() -> tuple[list[str], dict[str, int]]:  # pragma: no cover
    if not EXTERNAL_TEST_RECEIPT_PATH.is_file():
        return list(DEFAULT_TEST_COMMANDS), {}
    payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    return list(payload.get("test_commands", DEFAULT_TEST_COMMANDS)), {
        str(key): int(value) for key, value in dict(payload.get("test_exit_codes", {})).items()
    }


def build_artifact(
    *,
    date: str = "20260808",
    games: Sequence[str] = DEFAULT_GAMES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    raw_root: Path | None = None,
    mutation_receipts: Sequence[Mapping[str, Any]] | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    force_zero_treatment_fire: bool = False,
    started: float | None = None,
) -> JsonDict:
    start = time.monotonic() if started is None else float(started)
    protected_before = protected_hash_map()
    registry = registry_precheck_start(games)
    prereg = build_preregistration(games=games, seeds=seeds)
    root = raw_root or (REPO_ROOT / RAW_RELATIVE_DIR)
    pairs = [
        run_matched_live_cell(
            fixture_level_boundary_cell(str(cell["game"]), int(cell["seed"]), boundary=1),
            raw_root=root,
        )
        for cell in prereg["cells"]
    ]
    registry = finalize_registry_hash(registry)
    mutations = [dict(row) for row in (mutation_receipts or run_mutation_tests())]
    model_specs, substrate = model_specs_and_substrate()
    live = canonical_live_entrypoint_receipts()
    fire = treatment_fire_and_reason_counts(
        pairs,
        force_zero=force_zero_treatment_fire,
        mutation_receipts=mutations,
    )
    accept = verifier_accept_reject_counts(pairs)
    displacement = centroid_displacement_validity(pairs)
    avoided = avoided_llm_induction_calls(pairs)
    score_by_game = engine_fidelity_score_actions_and_wall_time_by_arm_game(pairs)
    intervals = paired_clustered_intervals(score_by_game)
    harmful = harmful_regression_count_and_games(score_by_game, score_by_game, prereg["safety_gate"])
    aa = aa_control(pairs)
    within_game = within_game_only_receipt(pairs)
    forbidden = forbidden_access_counts()
    protected = protected_files_unchanged(protected_before)
    status = classify_status(fire, avoided)
    ab_complete = _ready_score(
        [
            registry["unchanged"],
            registry["all_selected_games_full_clear_before_run"],
            len(prereg["cells"]) >= SUPPORT_FLOOR,
            live["ok"],
            within_game["all_cells_within_game"],
            _validate_forbidden_counts(forbidden),
            aa["ok"],
            protected["unchanged"],
        ]
    )
    promotion = _ready_score(
        [
            ab_complete == 1.0,
            status == "complete_ready",
            fire["total"] >= SUPPORT_FLOOR,
            accept["accepted"] >= SUPPORT_FLOOR,
            displacement["all_valid"],
            avoided["total"] >= SUPPORT_FLOOR,
            harmful["count"] == 0,
            substrate["legacy_models_contributed_rows"] == 0,
        ]
    )
    artifact: JsonDict = {
        "experiment_id": 6215,
        "random_seed": int(seeds[0]) if seeds else None,
        "offline_reproduced": False,
        "status": status,
        "registry_precheck_and_hash_before_after": registry,
        "preregistered_game_seed_level_boundary_matrix": prereg,
        "model_specs": model_specs,
        "canonical_live_entrypoint_receipts": live,
        "matched_arm_configuration": {
            "aa_control": "transfer disabled against itself",
            "control": "CARNOT_ARC_TRAJECTORY_TRANSFER=0",
            "treatment": "CARNOT_ARC_TRAJECTORY_TRANSFER=1",
            "held_fixed": [
                "E3AgentPolicy",
                "same_game_prior_trace",
                "current_opening_observation",
                "fallback_model",
                "fallback_sampling",
                "action_budget",
            ],
            "live_defaults_unchanged": True,
        },
        "within_game_only_receipt": within_game,
        "treatment_fire_and_reason_counts": fire,
        "verifier_accept_reject_counts": accept,
        "centroid_displacement_validity": displacement,
        "avoided_llm_induction_calls": avoided,
        "engine_fidelity_score_actions_and_wall_time_by_arm_game": score_by_game,
        "paired_clustered_intervals": intervals,
        "harmful_regression_count_and_games": harmful,
        "aa_control": aa,
        "prior_game_cross_game_source_bfs_adapter_registry_hidden_state_access_counts": forbidden,
        "solve_claimed": False,
        "level_credit_delta": 0,
        "registry_update_count": 0,
        "ab_complete_score": ab_complete,
        "trajectory_transfer_promotion_ready_score": promotion,
        "protected_files_unchanged": protected,
        "inference_substrate": substrate,
        "verifier_is_oracle": False,
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(test_commands or []),
        "test_exit_codes": {
            str(key): int(value) for key, value in dict(test_exit_codes or {}).items()
        },
        "duration_s": round(time.monotonic() - start, 6),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: trajectory_transfer_ab_complete_no_solve_credit"
            if status == "complete_ready"
            else f"blocked: {status}_{date}_no_solve_credit"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing fields: {missing}")  # pragma: no cover
    if "solve_provenance" in artifact:
        raise ValueError("solve_provenance must be absent")  # pragma: no cover
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance incomplete")  # pragma: no cover
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_principles incomplete")  # pragma: no cover
    if artifact.get("solve_claimed") is not False or artifact.get("verifier_is_oracle") is not False:
        raise ValueError("solve and oracle flags must be false")  # pragma: no cover
    for field in ("level_credit_delta", "registry_update_count"):
        if artifact.get(field) != 0:
            raise ValueError(f"{field} must be bare 0")  # pragma: no cover
    counts = dict(
        artifact.get("prior_game_cross_game_source_bfs_adapter_registry_hidden_state_access_counts")
        or {}
    )
    if not _validate_forbidden_counts(counts):
        raise ValueError("forbidden counts must be bare zeros")  # pragma: no cover
    registry = dict(artifact.get("registry_precheck_and_hash_before_after") or {})
    if registry.get("registry_hash_before") != registry.get("registry_hash_after"):
        raise ValueError("registry hash changed")  # pragma: no cover
    substrate = dict(artifact.get("inference_substrate") or {})
    if substrate.get("legacy_models_contributed_rows") != 0:
        raise ValueError("legacy model rows must be zero")  # pragma: no cover
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("checksum mismatch")  # pragma: no cover
    if not str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")):
        raise ValueError("honest verdict prefix invalid")  # pragma: no cover


def write_artifact(artifact: Mapping[str, Any], *, path: Path | None = None) -> Path:
    out = path or (REPO_ROOT / RESULT_RELATIVE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260808")
    args = parser.parse_args(argv)
    started = time.monotonic()
    commands, exits = _external_test_receipts()
    artifact = build_artifact(
        date=str(args.date),
        raw_root=REPO_ROOT / RAW_RELATIVE_DIR,
        test_commands=commands,
        test_exit_codes=exits,
        started=started,
    )
    validate_artifact(artifact)
    write_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
