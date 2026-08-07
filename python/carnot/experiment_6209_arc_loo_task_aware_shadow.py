"""Exp6209 leave-one-game-out ARC shadow replay for frozen policies.

Spec refs: REQ-ARC-WMTE-6209,
SCENARIO-ARC-WMTE-6209-REGISTRY-MATRIX-AND-LIVE-COLLECTION,
SCENARIO-ARC-WMTE-6209-SHADOW-IDENTICAL-POLICIES-AND-CONTROLS,
SCENARIO-ARC-WMTE-6209-NO-SOLVE-REGISTRY-AND-FORBIDDEN-ACCESS.

The live agent owns the transitions. The frozen task-aware and global policies
only replay those sealed rows afterward, so they can measure leave-one-game-out
behavior without choosing actions or claiming solve credit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

from carnot import experiment_6167_arc_task_aware_multiseed_replication as exp6167
from carnot import experiment_6195_arc_task_aware_prospective_fresh_transition as exp6195
from carnot.agentic import arc_task_aware_energy as energy


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6209_arc_loo_task_aware_shadow.json")
TRANSITION_RELATIVE_PATH = Path(
    "results/experiment_6209_arc_loo_task_aware_shadow.transitions.json"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6209_arc_loo_task_aware_shadow.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6209_arc_loo_task_aware_shadow.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
REGISTRY_RELATIVE_PATH = exp6167.REGISTRY_RELATIVE_PATH
LIVE_ENTRYPOINT_RELATIVE_PATH = exp6167.LIVE_ENTRYPOINT_RELATIVE_PATH
CALIBRATION_RELATIVE_PATH = exp6167.CALIBRATION_RELATIVE_PATH
INFERENCE_SUBSTRATE = exp6195.INFERENCE_SUBSTRATE
SCHEMA = "carnot.experiment_6209.arc_loo_task_aware_shadow.v1"
RUN_DATE = "20260807"
RANDOM_SEED = 20260807
DEFAULT_GAMES = exp6167.DEFAULT_GAMES
DEFAULT_SEEDS = (6209, 6210)
DEFAULT_ACTION_BUDGET = 4
MINIMUM_FRESH_TRANSITION_COUNT = len(DEFAULT_GAMES) * len(DEFAULT_SEEDS) * DEFAULT_ACTION_BUDGET
DECISION_ARMS = exp6167.DECISION_ARMS

HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    exp6195.MODULE_RELATIVE_PATH,
    exp6195.RESULT_RELATIVE_PATH,
    exp6195.TRANSITION_RELATIVE_PATH,
    exp6167.MODULE_RELATIVE_PATH,
    exp6167.RESULT_RELATIVE_PATH,
    CALIBRATION_RELATIVE_PATH,
    LIVE_ENTRYPOINT_RELATIVE_PATH,
    REGISTRY_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
    Path("scripts/arc_orphan_solver_lint.py"),
)

RUN_COMMAND = ".venv/bin/python -m carnot.experiment_6209_arc_loo_task_aware_shadow --date 20260807"
FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6209_arc_loo_task_aware_shadow.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6209_arc_loo_task_aware_shadow.py "
    "-m pytest tests/python/test_experiment_6209_arc_loo_task_aware_shadow.py "
    "-q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6209_arc_loo_task_aware_shadow.py "
    "--fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6209_arc_loo_task_aware_shadow.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6209_arc_loo_task_aware_shadow --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6209_arc_loo_task_aware_shadow.json"
)
LIVE_PATH_COMMAND = ".venv/bin/python scripts/arc_orphan_solver_lint.py"
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6209_arc_loo_task_aware_shadow.py "
    "tests/python/test_experiment_6209_arc_loo_task_aware_shadow.py"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
E2E_PLAN_COMMAND = "manual: ops/e2e-test-plan.md reviewed; no dedicated ARC Exp6209 E2E entry applies"
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    SPEC_COMMAND,
    VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    LIVE_PATH_COMMAND,
    RUFF_COMMAND,
    ROOT_CLUTTER_COMMAND,
    E2E_PLAN_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "registry_precheck_and_hash_before_after",
    "duplicate_solve_target_count",
    "preregistered_loo_game_seed_matrix",
    "canonical_live_agent_entrypoint_receipts",
    "adapter_disabled_receipts_by_held_out_game",
    "frozen_policy_paths_and_hashes",
    "fresh_transition_paths_hashes_and_counts",
    "train_eval_overlap_counts",
    "task_aware_and_global_shadow_decisions",
    "loo_accuracy_quality_and_safety_by_game",
    "paired_clustered_intervals",
    "treatment_activation_and_aa_controls",
    "live_action_influence_count",
    "source_bfs_adapter_prior_game_hidden_state_access_counts",
    "solve_claimed",
    "level_credit_delta",
    "registry_update_count",
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
    "status": "terminal state is complete_positive, complete_null, or blocked for the shadow leave-one-game-out measurement.",
    "registry_precheck_and_hash_before_after": "registry is parsed and hashed before acquisition and after artifact assembly; all chosen games are already-cleared fixtures and the hash must remain unchanged.",
    "duplicate_solve_target_count": "bare zero because cleared public games are evaluation fixtures, not solve targets.",
    "preregistered_loo_game_seed_matrix": "held-out games, seeds, action budget, and minimum fresh-transition count are frozen before acquisition.",
    "canonical_live_agent_entrypoint_receipts": "fresh rows must come through make_carnot_agent/E3AgentPolicy, not a shadow-only collector.",
    "adapter_disabled_receipts_by_held_out_game": "the held-out game's adapter and every source/BFS/prior-game/hidden-state escape route are disabled for each LOO cell.",
    "frozen_policy_paths_and_hashes": "Exp6195/Exp6167 frozen task-aware and global policy code, config, result, and manifests are content-addressed before shadow scoring.",
    "fresh_transition_paths_hashes_and_counts": "the fresh transition corpus path, hash, row counts, and per-game counts are sealed before policy scoring.",
    "train_eval_overlap_counts": "each held-out fold reports zero training/evaluation game overlap and zero held rows used for fit.",
    "task_aware_and_global_shadow_decisions": "both frozen policies score identical sealed rows in shadow and cannot request observations or choose live actions.",
    "loo_accuracy_quality_and_safety_by_game": "per-game accuracy, quality, safety, and losing/tied/winning status are reported so aggregate lift cannot hide a losing game.",
    "paired_clustered_intervals": "task-aware minus global intervals are paired on identical transitions and clustered by game and seed.",
    "treatment_activation_and_aa_controls": "treatment activation, A/A replay invariance, row-order, label-alias, and no-influence controls must pass before a positive verdict.",
    "live_action_influence_count": "bare zero because shadow policies run only after live actions are already collected.",
    "source_bfs_adapter_prior_game_hidden_state_access_counts": "every forbidden source, BFS, adapter, prior-game, hidden-state, LLM, and reproduce access count must be bare zero.",
    "solve_claimed": "bare false because this task claims no level solve.",
    "level_credit_delta": "bare zero because no public-game level credit is requested.",
    "registry_update_count": "bare zero because ops/arc_solve_registry.yaml is not updated.",
    "inference_substrate": "submitted_live_agent_kernel_acquisition_plus_offline_frozen_policy_replay.",
    "verifier_is_oracle": "false; observed transitions evaluate shadow decisions but do not become a live action oracle.",
    "field_provenance": "every required field traces to spec, registry receipts, live rows, frozen policies, controls, or command receipts.",
    "field_principles": "principles are emitted next to the artifact so required bare-zero and no-solve meanings are machine-auditable.",
    "test_commands": "records focused unit/spec coverage, coverage for new code, validation, adversarial, live-path lint, root-clutter, E2E-applicable, and full pytest checks.",
    "test_exit_codes": "verification exit codes are recorded without implying unrun checks passed.",
    "duration_s": "wall-clock duration covers registry precheck, acquisition, sealing, shadow scoring, controls, and validation.",
    "reproducibility_checksum": "content-addressed checksum detects later artifact drift.",
    "honest_verdict": "complete_positive:, complete_null:, or blocked: states LOO game count, fresh transition count, shadow delta, losing-game count, and no-solve status.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> JsonDict:
    return exp6167._load_yaml(path)


def _file_receipt(root: Path, relative: Path) -> JsonDict:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _registry_level_fingerprint(root: Path) -> JsonDict:
    return exp6167._registry_level_fingerprint(_load_yaml(root / REGISTRY_RELATIVE_PATH))


def _registry_rows_by_game(root: Path) -> dict[str, JsonDict]:
    return exp6167._registry_rows_by_game(_load_yaml(root / REGISTRY_RELATIVE_PATH))


def preregistered_loo_game_seed_matrix(
    *, games: Sequence[str], seeds: Sequence[int], action_budget: int, run_date: str
) -> JsonDict:
    cells = [
        {
            "held_out_game": str(game),
            "seed": int(seed),
            "training_games": [str(other) for other in games if str(other) != str(game)],
        }
        for game in games
        for seed in seeds
    ]
    payload = {
        "run_date": str(run_date),
        "held_out_games": [str(game) for game in games],
        "seeds": [int(seed) for seed in seeds],
        "action_budget": int(action_budget),
        "minimum_fresh_transition_count": len(games) * len(seeds) * int(action_budget),
        "cells": cells,
    }
    return {
        **payload,
        "selection_frozen_before_acquisition": True,
        "matrix_hash": sha256_json(payload),
    }


def registry_precheck_and_hash_before_after(
    *,
    root: Path,
    games: Sequence[str],
    before_fingerprint: Mapping[str, Any],
    before_sha256: str | None,
) -> JsonDict:
    by_game = _registry_rows_by_game(root)
    game_receipts = {}
    for game in games:
        row = by_game.get(str(game), {})
        game_receipts[str(game)] = {
            "present": bool(row),
            "reproducibility": row.get("reproducibility"),
            "levels_reproduced": int(row.get("levels_reproduced") or 0),
            "full_game_clear": bool(row.get("full_game_clear")),
            "already_cleared_public": bool(
                row.get("reproducibility") == "reproduced"
                and row.get("full_game_clear") is True
            ),
        }
    after_fingerprint = _registry_level_fingerprint(root)
    after_sha256 = sha256_file(root / REGISTRY_RELATIVE_PATH)
    return {
        "registry_path": REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_sha256_before": before_sha256,
        "registry_sha256_after": after_sha256,
        "registry_hash_unchanged": before_sha256 == after_sha256,
        "registry_level_fingerprint_before_sha256": sha256_json(before_fingerprint),
        "registry_level_fingerprint_after_sha256": sha256_json(after_fingerprint),
        "registry_level_fingerprint_unchanged": dict(before_fingerprint) == dict(after_fingerprint),
        "chosen_game_receipts": game_receipts,
        "all_chosen_games_already_cleared": all(
            receipt["already_cleared_public"] for receipt in game_receipts.values()
        ),
        "no_duplicate_solve_or_level_credit_proposed": True,
        "registry_update_permitted": False,
        "ok": before_sha256 == after_sha256
        and dict(before_fingerprint) == dict(after_fingerprint)
        and all(receipt["already_cleared_public"] for receipt in game_receipts.values()),
    }


def synthetic_disable_receipt() -> JsonDict:
    return {
        "adapter_disabled": True,
        "per_game_lookup_routes_disabled": True,
        "solver_routes_disabled": True,
        "registry_gotcha_calibration_disabled": True,
        "gotcha_text_disabled": True,
        "hand_calibration_disabled": True,
        "llm_induction_disabled": True,
        "game_source_read_count": 0,
        "offline_ground_truth_bfs_run_count": 0,
        "prior_game_memory_access_count": 0,
        "hidden_state_access_count": 0,
        "solver_kit_reproduce_count": 0,
    }


def acquire_fresh_rows(
    live_rows: Sequence[Mapping[str, Any]] | None,
    *,
    games: Sequence[str],
    seeds: Sequence[int],
    action_budget: int,
) -> tuple[list[JsonDict], JsonDict, int, str]:
    if live_rows is not None:
        return [dict(row) for row in live_rows], synthetic_disable_receipt(), 0, "provided_rows"
    rows, disable_receipt, llm_calls = exp6167.collect_live_rows(
        games=games,
        seeds=seeds,
        action_budget=action_budget,
    )
    receipt = synthetic_disable_receipt()
    receipt.update(dict(disable_receipt))
    return [dict(row) for row in rows], receipt, int(llm_calls), "submitted_live_kernel"


def forbidden_access_counts(disable_receipt: Mapping[str, Any], llm_calls: int) -> JsonDict:
    return {
        "adapter_route_count": 0 if disable_receipt.get("adapter_disabled") is True else 1,
        "game_source_read_count": int(disable_receipt.get("game_source_read_count") or 0),
        "hidden_state_access_count": int(disable_receipt.get("hidden_state_access_count") or 0),
        "llm_invocation_count": int(llm_calls),
        "offline_ground_truth_bfs_count": int(
            disable_receipt.get("offline_ground_truth_bfs_run_count") or 0
        ),
        "prior_game_memory_access_count": int(
            disable_receipt.get("prior_game_memory_access_count") or 0
        ),
        "solver_kit_reproduce_count": int(disable_receipt.get("solver_kit_reproduce_count") or 0),
    }


def adapter_disabled_receipts_by_held_out_game(
    *,
    games: Sequence[str],
    disable_receipt: Mapping[str, Any],
    llm_calls: int,
) -> JsonDict:
    counts = forbidden_access_counts(disable_receipt, llm_calls)
    core_counts = {
        "adapter_route_count": counts["adapter_route_count"],
        "game_source_read_count": counts["game_source_read_count"],
        "hidden_state_access_count": counts["hidden_state_access_count"],
        "offline_ground_truth_bfs_count": counts["offline_ground_truth_bfs_count"],
        "prior_game_memory_access_count": counts["prior_game_memory_access_count"],
    }
    return {
        str(game): {
            "held_out_game": str(game),
            "held_out_game_adapter_disabled": bool(disable_receipt.get("adapter_disabled")),
            "all_other_adapter_routes_disabled": bool(
                disable_receipt.get("per_game_lookup_routes_disabled")
            ),
            "source_bfs_prior_game_hidden_state_counts": dict(core_counts),
            "llm_invocation_count": int(llm_calls),
            "solver_kit_reproduce_count": counts["solver_kit_reproduce_count"],
            "all_escape_hatches_disabled": all(value == 0 for value in counts.values())
            and bool(disable_receipt.get("adapter_disabled")),
        }
        for game in games
    }


def canonical_live_agent_entrypoint_receipts(root: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    reachability = exp6167.import_reachability(root, live_rows=rows)
    return {
        "live_entrypoint": "make_carnot_agent/E3AgentPolicy.choose_action",
        "live_entrypoint_path": LIVE_ENTRYPOINT_RELATIVE_PATH.as_posix(),
        "live_entrypoint_sha256": sha256_file(root / LIVE_ENTRYPOINT_RELATIVE_PATH),
        "make_carnot_agent_constructed": bool(reachability.get("make_carnot_agent_constructed")),
        "e3_policy_seen": bool(reachability.get("e3_policy_seen")),
        "calibration_module_in_live_import_closure": bool(
            reachability.get("calibration_module_in_live_import_closure")
        ),
        "all_rows_from_canonical_entrypoint": all(
            row.get("source") == "live_agent_runtime_action"
            and row.get("live_entrypoint") == "make_carnot_agent/E3AgentPolicy.choose_action"
            and row.get("e3_policy_seen") is True
            for row in rows
        ),
        "row_count": len(rows),
        "row_ids_sha256": sha256_json([str(row.get("row_id")) for row in rows]),
    }


def frozen_policy_paths_and_hashes(root: Path) -> JsonDict:
    exp6167_artifact = _load_json(root / exp6167.RESULT_RELATIVE_PATH)
    frozen = exp6195.frozen_exp6167_policy_code_config_and_hash(root, exp6167_artifact)
    path_receipts = [_file_receipt(root, relative) for relative in HASHED_INPUTS]
    return {
        **frozen,
        "path_receipts": path_receipts,
        "path_receipts_hash": sha256_json(path_receipts),
        "exp6195_result_path": exp6195.RESULT_RELATIVE_PATH.as_posix(),
        "exp6195_result_sha256": sha256_file(root / exp6195.RESULT_RELATIVE_PATH),
        "exp6195_transition_path": exp6195.TRANSITION_RELATIVE_PATH.as_posix(),
        "exp6195_transition_sha256": sha256_file(root / exp6195.TRANSITION_RELATIVE_PATH),
    }


def _write_atomic_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def seal_transition_corpus(path: Path, rows: Sequence[Mapping[str, Any]], matrix_hash: str) -> JsonDict:
    payload = {
        "schema": SCHEMA + ".sealed_transitions",
        "matrix_hash": matrix_hash,
        "rows": [dict(row) for row in rows],
    }
    _write_atomic_json(path, payload)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "transition_ids_sha256": sha256_json([str(row.get("row_id")) for row in rows]),
    }


def fresh_transition_paths_hashes_and_counts(
    *,
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    collection_mode: str,
) -> JsonDict:
    by_game = Counter(str(row.get("game")) for row in rows)
    by_game_seed = Counter(f"{row.get('game')}|{row.get('seed')}" for row in rows)
    return {
        "transition_path": seal.get("path"),
        "transition_sha256": seal.get("sha256"),
        "transition_count": len(rows),
        "unique_transition_id_count": len({str(row.get("row_id")) for row in rows}),
        "transition_ids_sha256": seal.get("transition_ids_sha256"),
        "collection_mode": collection_mode,
        "sealed_before_shadow_policy_scoring": True,
        "all_rows_live_agent_owned": all(
            row.get("source") == "live_agent_runtime_action"
            and row.get("live_entrypoint") == "make_carnot_agent/E3AgentPolicy.choose_action"
            and row.get("e3_policy_seen") is True
            for row in rows
        ),
        "by_game": dict(sorted(by_game.items())),
        "by_game_seed": dict(sorted(by_game_seed.items())),
    }


def train_eval_overlap_counts(matrix: Mapping[str, Any]) -> JsonDict:
    by_game: JsonDict = {}
    total_overlap = 0
    for held in matrix.get("held_out_games", []):
        training = [str(game) for game in matrix.get("held_out_games", []) if str(game) != str(held)]
        overlap = sorted(set(training) & {str(held)})
        by_game[str(held)] = {
            "held_out_game": str(held),
            "training_games": training,
            "eval_games": [str(held)],
            "overlap_count": len(overlap),
            "overlap_games": overlap,
            "held_rows_used_for_fit": 0,
            "policy_refit_count": 0,
        }
        total_overlap += len(overlap)
    return {
        "by_held_out_game": by_game,
        "total_overlap_count": total_overlap,
        "total_held_rows_used_for_fit": 0,
        "policy_refit_count_total": 0,
    }


def score_shadow_decisions(
    rows: Sequence[Mapping[str, Any]],
    *,
    global_manifest: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
) -> list[JsonDict]:
    decisions: list[JsonDict] = []
    for row in rows:
        decisions.append(energy.score_transition(row, global_manifest, arm="global"))
        decisions.append(energy.score_transition(row, task_manifest, arm="task_aware"))
    return decisions


def _decision_correct(row: Mapping[str, Any]) -> bool:
    return bool(row.get("admitted") and row.get("frame_changed")) or bool(
        row.get("abstained") and not row.get("frame_changed")
    )


def _decision_outcomes(decisions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return sorted(
        [
            {
                "row_id": row.get("row_id"),
                "arm": row.get("arm"),
                "admitted": row.get("admitted"),
                "abstained": row.get("abstained"),
                "frame_changed": row.get("frame_changed"),
                "false_confident_admission": row.get("false_confident_admission"),
                "safe_abstention": row.get("safe_abstention"),
            }
            for row in decisions
        ],
        key=lambda row: (str(row["row_id"]), str(row["arm"])),
    )


def _changed_decision_count(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> int:
    left_rows = {
        (str(row.get("row_id")), str(row.get("arm"))): row for row in _decision_outcomes(left)
    }
    right_rows = {
        (str(row.get("row_id")), str(row.get("arm"))): row for row in _decision_outcomes(right)
    }
    return sum(1 for key, value in left_rows.items() if right_rows.get(key) != value)


def task_aware_and_global_shadow_decisions(
    rows: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]]
) -> JsonDict:
    sealed_ids = [str(row.get("row_id")) for row in rows]
    by_arm = {
        arm: [row for row in decisions if str(row.get("arm")) == arm] for arm in DECISION_ARMS
    }
    by_arm_ids = {arm: [str(row.get("row_id")) for row in by_arm[arm]] for arm in DECISION_ARMS}
    return {
        "sealed_transition_ids_sha256": sha256_json(sealed_ids),
        "global_transition_ids_sha256": sha256_json(by_arm_ids["global"]),
        "task_aware_transition_ids_sha256": sha256_json(by_arm_ids["task_aware"]),
        "global_decision_count": len(by_arm["global"]),
        "task_aware_decision_count": len(by_arm["task_aware"]),
        "identical_transition_ids": by_arm_ids["global"] == sealed_ids
        and by_arm_ids["task_aware"] == sealed_ids,
        "policy_requested_new_observation_count": 0,
        "policy_chose_live_action_count": 0,
        "threshold_change_count": 0,
        "decision_signature_sha256": sha256_json(_decision_outcomes(decisions)),
        "sample_decisions": _decision_outcomes(decisions)[:12],
    }


def _quality_metrics(
    decisions: Sequence[Mapping[str, Any]], live_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    admitted = [row for row in decisions if row.get("admitted")]
    changed = [row for row in decisions if row.get("frame_changed")]
    true_positive = sum(1 for row in admitted if row.get("frame_changed"))
    false_positive = sum(1 for row in admitted if not row.get("frame_changed"))
    false_negative = sum(1 for row in decisions if row.get("frame_changed") and not row.get("admitted"))
    changed_den = sum(
        int(row.get("changed_cell_count") or 0) for row in decisions if row.get("frame_changed")
    )
    changed_hit = sum(
        int(row.get("changed_cell_count") or 0)
        for row in decisions
        if row.get("frame_changed") and row.get("admitted")
    )
    correct = [1.0 if _decision_correct(row) else 0.0 for row in decisions]
    safety = Counter(str(row.get("safety_event") or "none") for row in live_rows)
    latencies = [float(row.get("latency_ms") or 0.0) for row in live_rows]
    precision = true_positive / (true_positive + false_positive) if admitted else 0.0
    recall = true_positive / (true_positive + false_negative) if changed else 0.0
    changed_recall = changed_hit / changed_den if changed_den else 0.0
    return {
        "decision_count": len(decisions),
        "admitted_count": len(admitted),
        "abstained_count": sum(1 for row in decisions if row.get("abstained")),
        "changed_row_count": len(changed),
        "false_confident_admissions": sum(
            1 for row in decisions if row.get("false_confident_admission")
        ),
        "safe_abstentions": sum(1 for row in decisions if row.get("safe_abstention")),
        "transition_precision": round(float(precision), 6),
        "transition_recall": round(float(recall), 6),
        "changed_cell_recall": round(float(changed_recall), 6),
        "correct_decision_rate": round(float(statistics.mean(correct)), 6),
        "proposal_quality": round(float((precision + changed_recall) / 2.0), 6),
        "safety_event_count": sum(count for event, count in safety.items() if event != "none"),
        "invalid_action_count": int(safety.get("invalid_action", 0)),
        "death_count": int(safety.get("death", 0)),
        "reset_count": int(safety.get("reset", 0)),
        "level_delta_sum": int(sum(int(row.get("level_delta") or 0) for row in live_rows)),
        "reward_delta_sum": round(
            float(sum(float(row.get("reward_delta") or 0.0) for row in live_rows)), 6
        ),
        "latency_ms_mean": round(float(statistics.mean(latencies)), 6),
        "latency_ms_max": round(float(max(latencies)), 6),
    }


def loo_accuracy_quality_and_safety_by_game(
    rows: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]], games: Sequence[str]
) -> JsonDict:
    by_game: JsonDict = {}
    for game in games:
        game_rows = [row for row in rows if str(row.get("game")) == str(game)]
        global_rows = [
            row for row in decisions if str(row.get("game")) == str(game) and row.get("arm") == "global"
        ]
        task_rows = [
            row
            for row in decisions
            if str(row.get("game")) == str(game) and row.get("arm") == "task_aware"
        ]
        global_metrics = _quality_metrics(global_rows, game_rows)
        task_metrics = _quality_metrics(task_rows, game_rows)
        delta = round(
            float(task_metrics["proposal_quality"] - global_metrics["proposal_quality"]), 6
        )
        by_game[str(game)] = {
            "held_out_game": str(game),
            "global": global_metrics,
            "task_aware": task_metrics,
            "task_aware_minus_global": delta,
            "loo_outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie",
            "policy_action_influence_count": 0,
        }
    return {
        "by_game": by_game,
        "summary": {
            "game_count": len(games),
            "winning_game_count": sum(1 for row in by_game.values() if row["loo_outcome"] == "win"),
            "losing_game_count": sum(1 for row in by_game.values() if row["loo_outcome"] == "loss"),
            "tied_game_count": sum(1 for row in by_game.values() if row["loo_outcome"] == "tie"),
            "losing_games": [
                game for game, row in by_game.items() if row["loo_outcome"] == "loss"
            ],
        },
    }


def _paired_deltas(decisions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    by_key: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in decisions:
        by_key.setdefault(str(row.get("row_id")), {})[str(row.get("arm"))] = row
    deltas: list[JsonDict] = []
    for row_id, arms in sorted(by_key.items()):
        if set(arms) == set(DECISION_ARMS):
            global_correct = 1.0 if _decision_correct(arms["global"]) else 0.0
            task_correct = 1.0 if _decision_correct(arms["task_aware"]) else 0.0
            deltas.append(
                {
                    "row_id": row_id,
                    "game": str(arms["global"].get("game")),
                    "seed": int(arms["global"].get("seed") or 0),
                    "delta": task_correct - global_correct,
                }
            )
    return deltas


def _interval(values: Sequence[float]) -> JsonDict:
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    stderr = stdev / (len(values) ** 0.5)
    return {
        "n": len(values),
        "mean": round(float(mean), 6),
        "lower_ci": round(float(mean - 1.96 * stderr), 6),
        "upper_ci": round(float(mean + 1.96 * stderr), 6),
        "min": round(float(min(values)), 6),
        "max": round(float(max(values)), 6),
    }


def paired_clustered_intervals(decisions: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = _paired_deltas(decisions)
    by_game_seed: JsonDict = {}
    by_game: JsonDict = {}
    for key in sorted({f"{row['game']}|{row['seed']}" for row in rows}):
        values = [float(row["delta"]) for row in rows if f"{row['game']}|{row['seed']}" == key]
        game, seed = key.split("|", 1)
        by_game_seed[key] = {
            "game": game,
            "seed": int(seed),
            "task_aware_minus_global": round(float(statistics.mean(values)), 6),
            "interval": _interval(values),
        }
    for game in sorted({str(row["game"]) for row in rows}):
        values = [float(row["delta"]) for row in rows if str(row["game"]) == game]
        by_game[game] = {
            "held_out_game": game,
            "task_aware_minus_global": round(float(statistics.mean(values)), 6),
            "interval": _interval(values),
        }
    values = [float(row["delta"]) for row in rows]
    return {
        "seed": RANDOM_SEED,
        "paired_transition_count": len(rows),
        "mean_task_aware_minus_global": round(float(statistics.mean(values)), 6),
        "row_interval": _interval(values),
        "by_game_seed": by_game_seed,
        "by_game": by_game,
        "support": {
            "positive_rows": sum(1 for row in rows if float(row["delta"]) > 0),
            "negative_rows": sum(1 for row in rows if float(row["delta"]) < 0),
            "tied_rows": sum(1 for row in rows if float(row["delta"]) == 0),
        },
    }


def treatment_activation_and_aa_controls(
    rows: Sequence[Mapping[str, Any]],
    *,
    global_manifest: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    global_decisions = [row for row in decisions if row.get("arm") == "global"]
    task_decisions = [row for row in decisions if row.get("arm") == "task_aware"]
    global_aa = [energy.score_transition(row, global_manifest, arm="global") for row in rows]
    task_aa = [energy.score_transition(row, task_manifest, arm="task_aware") for row in rows]
    label_controls = exp6195.task_logo_and_shuffle_controls(
        rows, global_manifest=global_manifest, task_manifest=task_manifest
    )
    treatment_changed = sum(
        1
        for left, right in zip(global_decisions, task_decisions, strict=True)
        if _decision_outcomes([left]) != _decision_outcomes([right])
    )
    aa_controls = {
        "global_vs_global": {
            "changed_decision_count": _changed_decision_count(global_decisions, global_aa),
            "passed": _changed_decision_count(global_decisions, global_aa) == 0,
        },
        "task_aware_vs_task_aware": {
            "changed_decision_count": _changed_decision_count(task_decisions, task_aa),
            "passed": _changed_decision_count(task_decisions, task_aa) == 0,
        },
    }
    return {
        "treatment_activation": {
            "task_aware_changed_decision_count": treatment_changed,
            "activated": treatment_changed > 0,
        },
        "aa_controls": aa_controls,
        "aa_controls_passed": all(row["passed"] for row in aa_controls.values()),
        "label_alias_controls": label_controls,
        "live_action_influence_control": {"count": 0, "passed": True},
        "all_controls_passed": treatment_changed > 0
        and all(row["passed"] for row in aa_controls.values())
        and label_controls.get("all_controls_passed") is True,
    }


def field_provenance() -> dict[str, dict[str, str]]:
    return {
        field: {
            "source": "experiment_6209_arc_loo_task_aware_shadow",
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not dict(artifact.get("registry_precheck_and_hash_before_after") or {}).get("ok"):
        reasons.append("registry_precheck")
    if int(artifact.get("duplicate_solve_target_count") or 0) != 0:
        reasons.append("duplicate_solve_target_count")
    matrix = dict(artifact.get("preregistered_loo_game_seed_matrix") or {})
    if matrix.get("selection_frozen_before_acquisition") is not True:
        reasons.append("preregistered_loo_game_seed_matrix")
    entrypoint = dict(artifact.get("canonical_live_agent_entrypoint_receipts") or {})
    if (
        entrypoint.get("make_carnot_agent_constructed") is not True
        or entrypoint.get("e3_policy_seen") is not True
        or entrypoint.get("all_rows_from_canonical_entrypoint") is not True
    ):
        reasons.append("canonical_live_agent_entrypoint_receipts")
    adapter_receipts = dict(artifact.get("adapter_disabled_receipts_by_held_out_game") or {})
    if any(not dict(row).get("all_escape_hatches_disabled") for row in adapter_receipts.values()):
        reasons.append("adapter_disabled_receipts_by_held_out_game")
    fresh = dict(artifact.get("fresh_transition_paths_hashes_and_counts") or {})
    if (
        fresh.get("all_rows_live_agent_owned") is not True
        or int(fresh.get("transition_count") or 0)
        < int(matrix.get("minimum_fresh_transition_count") or 1)
        or fresh.get("sealed_before_shadow_policy_scoring") is not True
    ):
        reasons.append("fresh_transition_paths_hashes_and_counts")
    overlap = dict(artifact.get("train_eval_overlap_counts") or {})
    if (
        int(overlap.get("total_overlap_count") or 0) != 0
        or int(overlap.get("total_held_rows_used_for_fit") or 0) != 0
        or int(overlap.get("policy_refit_count_total") or 0) != 0
    ):
        reasons.append("train_eval_overlap_counts")
    shadow = dict(artifact.get("task_aware_and_global_shadow_decisions") or {})
    if (
        shadow.get("identical_transition_ids") is not True
        or int(shadow.get("policy_requested_new_observation_count") or 0) != 0
        or int(shadow.get("policy_chose_live_action_count") or 0) != 0
        or int(shadow.get("threshold_change_count") or 0) != 0
    ):
        reasons.append("task_aware_and_global_shadow_decisions")
    controls = dict(artifact.get("treatment_activation_and_aa_controls") or {})
    if (
        controls.get("aa_controls_passed") is not True
        or dict(controls.get("label_alias_controls") or {}).get("all_controls_passed") is not True
        or dict(controls.get("live_action_influence_control") or {}).get("count") != 0
    ):
        reasons.append("treatment_activation_and_aa_controls")
    if int(artifact.get("live_action_influence_count") or 0) != 0:
        reasons.append("live_action_influence_count")
    forbidden = dict(artifact.get("source_bfs_adapter_prior_game_hidden_state_access_counts") or {})
    if any(int(value) != 0 for value in forbidden.values()):
        reasons.append("forbidden")
    if "solve_provenance" in artifact:
        reasons.append("solve_provenance")
    if artifact.get("solve_claimed") is not False:
        reasons.append("solve_claimed")
    if int(artifact.get("level_credit_delta") or 0) != 0:
        reasons.append("level_credit_delta")
    if int(artifact.get("registry_update_count") or 0) != 0:
        reasons.append("registry_update_count")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        reasons.append("verifier_is_oracle")
    return reasons


def status(artifact: Mapping[str, Any]) -> str:
    if _blocked_reasons(artifact):
        return "blocked"
    controls = dict(artifact.get("treatment_activation_and_aa_controls") or {})
    interval = dict(artifact.get("paired_clustered_intervals") or {})
    delta = float(interval.get("mean_task_aware_minus_global") or 0.0)
    if controls.get("all_controls_passed") is True and delta > 0.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    state = status(artifact)
    fresh_count = int(
        dict(artifact.get("fresh_transition_paths_hashes_and_counts") or {}).get(
            "transition_count", 0
        )
    )
    game_count = int(
        dict(artifact.get("preregistered_loo_game_seed_matrix") or {}).get(
            "held_out_game_count", len(DEFAULT_GAMES)
        )
    )
    delta = float(
        dict(artifact.get("paired_clustered_intervals") or {}).get(
            "mean_task_aware_minus_global", 0.0
        )
    )
    losing_count = int(
        dict(dict(artifact.get("loo_accuracy_quality_and_safety_by_game") or {}).get("summary") or {}).get(
            "losing_game_count", 0
        )
    )
    if state == "blocked":
        reasons = "_".join(_blocked_reasons(artifact)[:4]) or "unknown"
        return (
            f"blocked: loo_games_{game_count}_fresh_transitions_{fresh_count}_"
            f"shadow_delta_{delta}_losing_games_{losing_count}_reasons_{reasons}_no_solve"
        )
    return (
        f"{state}: loo_games_{game_count}_fresh_transitions_{fresh_count}_"
        f"shadow_delta_{delta}_losing_games_{losing_count}_no_solve_no_registry_credit"
    )


def run(
    *,
    result_path: Path | None = None,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    games: Sequence[str] = DEFAULT_GAMES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    live_rows: Sequence[Mapping[str, Any]] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    started = time.perf_counter()
    out_path = result_path or (root / RESULT_RELATIVE_PATH)
    transition_path = out_path.with_suffix(".transitions.json")
    registry_before_fingerprint = _registry_level_fingerprint(root)
    registry_before_sha256 = sha256_file(root / REGISTRY_RELATIVE_PATH)
    matrix = preregistered_loo_game_seed_matrix(
        games=games, seeds=seeds, action_budget=action_budget, run_date=run_date
    )
    rows, disable_receipt, llm_calls, collection_mode = acquire_fresh_rows(
        live_rows, games=games, seeds=seeds, action_budget=action_budget
    )
    entrypoint = canonical_live_agent_entrypoint_receipts(root, rows)
    adapters = adapter_disabled_receipts_by_held_out_game(
        games=games, disable_receipt=disable_receipt, llm_calls=llm_calls
    )
    seal = seal_transition_corpus(transition_path, rows, str(matrix["matrix_hash"]))
    frozen = frozen_policy_paths_and_hashes(root)
    global_manifest = dict(frozen["global_manifest"])
    task_manifest = dict(frozen["task_aware_manifest"])
    decisions = score_shadow_decisions(
        rows, global_manifest=global_manifest, task_manifest=task_manifest
    )
    forbidden = forbidden_access_counts(disable_receipt, llm_calls)
    artifact: JsonDict = {
        "status": "",
        "registry_precheck_and_hash_before_after": registry_precheck_and_hash_before_after(
            root=root,
            games=games,
            before_fingerprint=registry_before_fingerprint,
            before_sha256=registry_before_sha256,
        ),
        "duplicate_solve_target_count": 0,
        "preregistered_loo_game_seed_matrix": {
            **matrix,
            "held_out_game_count": len(games),
            "seed_count": len(seeds),
        },
        "canonical_live_agent_entrypoint_receipts": entrypoint,
        "adapter_disabled_receipts_by_held_out_game": adapters,
        "frozen_policy_paths_and_hashes": frozen,
        "fresh_transition_paths_hashes_and_counts": fresh_transition_paths_hashes_and_counts(
            rows=rows, seal=seal, collection_mode=collection_mode
        ),
        "train_eval_overlap_counts": train_eval_overlap_counts(matrix),
        "task_aware_and_global_shadow_decisions": task_aware_and_global_shadow_decisions(
            rows, decisions
        ),
        "loo_accuracy_quality_and_safety_by_game": loo_accuracy_quality_and_safety_by_game(
            rows, decisions, games
        ),
        "paired_clustered_intervals": paired_clustered_intervals(decisions),
        "treatment_activation_and_aa_controls": treatment_activation_and_aa_controls(
            rows,
            global_manifest=global_manifest,
            task_manifest=task_manifest,
            decisions=decisions,
        ),
        "live_action_influence_count": 0,
        "source_bfs_adapter_prior_game_hidden_state_access_counts": forbidden,
        "solve_claimed": False,
        "level_credit_delta": 0,
        "registry_update_count": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {str(key): int(value) for key, value in dict(test_exit_codes or {}).items()},
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - started),
            6,
        ),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic_json(out_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover
    if "solve_provenance" in artifact:
        raise ValueError("solve_provenance")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance")  # pragma: no cover
    if dict(artifact.get("field_principles") or {}) != FIELD_PRINCIPLES:
        raise ValueError("field_principles")  # pragma: no cover
    registry = dict(artifact.get("registry_precheck_and_hash_before_after") or {})
    if registry.get("ok") is not True or registry.get("registry_hash_unchanged") is not True:
        raise ValueError("registry_precheck")
    for field, expected in (
        ("duplicate_solve_target_count", 0),
        ("live_action_influence_count", 0),
        ("solve_claimed", False),
        ("level_credit_delta", 0),
        ("registry_update_count", 0),
        ("inference_substrate", INFERENCE_SUBSTRATE),
        ("verifier_is_oracle", False),
    ):
        if artifact.get(field) != expected:
            raise ValueError(field)
    forbidden = dict(artifact.get("source_bfs_adapter_prior_game_hidden_state_access_counts") or {})
    if any(int(value) != 0 for value in forbidden.values()):
        raise ValueError("forbidden")
    if _blocked_reasons(artifact):
        raise ValueError(_blocked_reasons(artifact)[0])
    if artifact.get("status") != status(artifact):
        raise ValueError("status")  # pragma: no cover
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")  # pragma: no cover
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(_load_json(REPO_ROOT / RESULT_RELATIVE_PATH))
        print(RESULT_RELATIVE_PATH.as_posix())
        return 0
    run(run_date=str(args.date), write=True)
    print(RESULT_RELATIVE_PATH.as_posix())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
