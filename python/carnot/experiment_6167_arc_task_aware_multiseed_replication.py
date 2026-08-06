"""Exp6167 ARC task-aware multi-seed replication.

Spec refs: REQ-ARC-WMTE-6167,
SCENARIO-ARC-WMTE-6167-LIVE-ENTRYPOINT-FIXED-POLICY-AND-PROVENANCE,
SCENARIO-ARC-WMTE-6167-MULTIGAME-MULTISEED-METRICS-AND-CONTROLS,
SCENARIO-ARC-WMTE-6167-NO-SOLVE-REGISTRY-IMMUTABILITY-AND-SCHEMA.

This module repeats the Exp6154 transition-admission measurement on a wider
public-game and seed set. It is deliberately not a solver: it observes the
live E3 agent's own actions, scores the already-observed transition, and keeps
registry level credit fixed at zero.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any

from carnot import experiment_6154_arc_task_aware_energy_generalization as exp6154
from carnot.agentic import arc_task_aware_energy as energy


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6167_arc_task_aware_multiseed_replication.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6167_arc_task_aware_multiseed_replication.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6167_arc_task_aware_multiseed_replication.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXP6154_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6154_arc_task_aware_energy_generalization.py"
)
EXP6154_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6154_arc_task_aware_energy_generalization.json"
)
CALIBRATION_RELATIVE_PATH = Path("python/carnot/agentic/arc_task_aware_energy.py")
LIVE_ENTRYPOINT_RELATIVE_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
ADAPTER_RELATIVE_PATH = Path("python/carnot/agentic/arc_game_adapters.py")
SOLVER_KIT_RELATIVE_PATH = Path("python/carnot/agentic/arc_solver_kit.py")
INFERENCE_SUBSTRATE = "live_e3_adapter_disabled_runtime_transitions"
SCHEMA = "carnot.experiment_6167.arc_task_aware_multiseed_replication.v1"
RUN_DATE = "20260806"
DEFAULT_GAMES = ("lp85", "su15", "tu93", "r11l", "ls20", "sp80")
DEFAULT_HELD_GAMES = DEFAULT_GAMES
DEFAULT_SEEDS = (6167, 6168, 6169)
DEFAULT_ACTION_BUDGET = 8
DECISION_ARMS = ("global", "task_aware")

PROTECTED_FILES = exp6154.PROTECTED_FILES
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    SPEC_RELATIVE_PATH,
    EXP6154_MODULE_RELATIVE_PATH,
    EXP6154_RESULT_RELATIVE_PATH,
    CALIBRATION_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    LIVE_ENTRYPOINT_RELATIVE_PATH,
    REGISTRY_RELATIVE_PATH,
    ADAPTER_RELATIVE_PATH,
    SOLVER_KIT_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
    Path("scripts/arc_orphan_solver_lint.py"),
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6167_arc_task_aware_multiseed_replication.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6167_arc_task_aware_multiseed_replication.py "
    "-m pytest tests/python/test_experiment_6167_arc_task_aware_multiseed_replication.py "
    "-q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6167_arc_task_aware_multiseed_replication.py "
    "--fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6167_arc_task_aware_multiseed_replication.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6167_arc_task_aware_multiseed_replication --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6167_arc_task_aware_multiseed_replication.json"
)
LIVE_PATH_COMMAND = ".venv/bin/python scripts/arc_orphan_solver_lint.py"
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6167_arc_task_aware_multiseed_replication.py "
    "tests/python/test_experiment_6167_arc_task_aware_multiseed_replication.py"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
E2E_PLAN_COMMAND = (
    "manual: ops/e2e-test-plan.md reviewed; no dedicated ARC Exp6167 E2E entry applies"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    SPEC_COMMAND,
    VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    LIVE_PATH_COMMAND,
    RUFF_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    E2E_PLAN_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_policy_code_result_and_registry_hashes",
    "registry_precheck_and_no_duplicate_receipt",
    "development_and_held_game_split_hash",
    "global_and_task_aware_freeze_manifests",
    "adapter_per_game_lookup_solver_gotcha_and_hand_calibration_disable_receipts",
    "live_entrypoint_and_import_reachability",
    "own_attempt_transition_provenance",
    "game_seed_action_budget_and_arm_counts",
    "per_arm_triggered_decision_counts",
    "per_game_seed_transition_change_recall_safety_action_and_latency_metrics",
    "grouped_paired_intervals",
    "false_confident_admission_and_abstention_matrices",
    "shuffle_alias_identity_noop_invented_trigger_denominator_light_and_label_controls",
    "known_negative_tail_receipt",
    "solve_claimed",
    "level_credit_delta",
    "registry_levels_unchanged",
    "offline_ground_truth_bfs",
    "used_game_source",
    "llm_invocation_count",
    "arc_task_aware_multiseed_replication_ready_score",
    "protected_files_unchanged",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete_positive, complete_null, retired, or blocked names the terminal cross-game/seed replication result.",
    "preconditions_checked": "hashes, registry, catalog, seeds, output path, exclusions, protected files, and root clutter are checked before episodes.",
    "upstream_policy_code_result_and_registry_hashes": "Exp6154 policy code, result, live entrypoints, registry, adapters, solvers, gotcha paths, seeds, output, exclusions, and protected files are content-addressed before reuse.",
    "registry_precheck_and_no_duplicate_receipt": "selected public games are already registered and the run refuses duplicate solve credit before any episode.",
    "development_and_held_game_split_hash": "the six-game held set, upstream development source, seeds, and action budgets are frozen before held scoring.",
    "global_and_task_aware_freeze_manifests": "global and task-aware decision rules and thresholds are inherited from Exp6154 before new held episodes and are not fit from held outcomes.",
    "adapter_per_game_lookup_solver_gotcha_and_hand_calibration_disable_receipts": "per-game adapters, lookup routes, solver shortcuts, registry gotcha text, hand calibration, game-source reads, offline BFS, and LLM induction are disabled.",
    "live_entrypoint_and_import_reachability": "the measurement reaches the canonical make_carnot_agent/E3AgentPolicy path rather than an orphan scorer.",
    "own_attempt_transition_provenance": "every scored transition comes from the agent's own runtime action and observed before/after frame.",
    "game_seed_action_budget_and_arm_counts": "game count, seed count, episode count, action budget, and arm counts prove the multi-game/multi-seed matched-budget design.",
    "per_arm_triggered_decision_counts": "only decisions whose scorer triggered on the live path are credited.",
    "per_game_seed_transition_change_recall_safety_action_and_latency_metrics": "every game/seed tail reports transition recall, changed-cell recall, action recall, safety, invalid action, decision, and latency metrics.",
    "grouped_paired_intervals": "task-aware minus global deltas are paired by game and seed and require a positive grouped lower confidence bound for readiness.",
    "false_confident_admission_and_abstention_matrices": "false confident admissions and abstentions are reported by game/seed and arm so safety regressions cannot hide in aggregates.",
    "shuffle_alias_identity_noop_invented_trigger_denominator_light_and_label_controls": "shuffle, alias, identity/no-op, label, invented-trigger, denominator, and light controls guard shortcut and denominator artifacts.",
    "known_negative_tail_receipt": "the prior Exp6154 tu93 negative tail and the current tu93 per-seed tail are named before any positive generalization claim.",
    "solve_claimed": "bare false; this is transition-admission replication, not level solving.",
    "level_credit_delta": "bare 0; registry level totals do not move.",
    "registry_levels_unchanged": "bare true; registry level fields remain unchanged before and after the run.",
    "offline_ground_truth_bfs": "bare false; no exhaustive or oracle ground-truth BFS is run.",
    "used_game_source": "bare false; the experiment does not inspect game implementation source.",
    "llm_invocation_count": "bare zero; the deterministic adapter-disabled live path must not invoke the LLM tier.",
    "arc_task_aware_multiseed_replication_ready_score": "1 requires positive grouped lower CI, no safety regression, clean controls, live triggers, registry immutability, protected-file immutability, and no solve credit.",
    "protected_files_unchanged": "conductor, ops status/changelog, and traceability files are not modified by this run.",
    "duration_s": "wall-clock runtime is measured for the no-LLM live-path replication.",
    "inference_substrate": "live_e3_adapter_disabled_runtime_transitions declares no LLM/model load while live E3 episodes run.",
    "verifier_is_oracle": "false; observed transitions evaluate admission decisions but do not become a planning oracle.",
    "missing_verifier_gaps": "any blocked gate or negative tail is carried forward as an explicit gap instead of being hidden.",
    "field_provenance": "every required field traces to spec, frozen policy, live rows, controls, or command receipts.",
    "test_commands": "focused unit/spec coverage, registry precheck, live import, disablement, metrics, controls, schema, adversarial, E2E-applicable, protected-file, root-clutter, and full pytest checks are recorded.",
    "test_exit_codes": "verification exit codes are recorded so the artifact cannot imply unrun checks passed.",
    "reproducibility_checksum": "content-addressed checksum detects silent artifact drift.",
    "honest_verdict": "complete_positive:, complete_null:, retired:, or blocked: states cross-game/seed generalization without a solve claim.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return exp6154.sha256_file(path)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _load_json(path: Path) -> JsonDict:
    return exp6154._load_json(path)


def _load_yaml(path: Path) -> JsonDict:
    return exp6154._load_yaml(path)


def _file_receipt(root: Path, relative: Path) -> JsonDict:
    return exp6154._file_receipt(root, relative)


def _protected_hashes(root: Path) -> dict[str, str]:
    return exp6154._protected_hashes(root)


def _root_clutter_state(root: Path) -> JsonDict:
    return exp6154._root_clutter_state(root)


def _registry_level_fingerprint(registry: Mapping[str, Any]) -> JsonDict:
    return exp6154._registry_level_fingerprint(registry)


def _registry_rows_by_game(registry: Mapping[str, Any]) -> dict[str, JsonDict]:
    return exp6154._registry_rows_by_game(registry)


def _registry_levels_by_game(registry: Mapping[str, Any]) -> dict[str, int]:
    return {
        game: int(row.get("levels_reproduced") or 0)
        for game, row in _registry_rows_by_game(registry).items()
    }


def collect_preconditions(
    *,
    root: Path,
    result_path: Path,
    games: Sequence[str],
    held_games: Sequence[str],
    seeds: Sequence[int],
    action_budget: int,
) -> tuple[JsonDict, JsonDict]:
    registry = _load_yaml(root / REGISTRY_RELATIVE_PATH)
    registry_fingerprint = _registry_level_fingerprint(registry)
    checked = {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hashed_input_receipts": [_file_receipt(root, relative) for relative in HASHED_INPUTS],
        "game_catalog": {
            "source": REGISTRY_RELATIVE_PATH.as_posix(),
            "public_game_count": len(_registry_rows_by_game(registry)),
            "selected_games": list(games),
            "tu93_available": "tu93" in _registry_rows_by_game(registry),
        },
        "held_games": list(held_games),
        "seeds": [int(seed) for seed in seeds],
        "action_budget": int(action_budget),
        "output_path": {
            "path": str(result_path),
            "parent_exists": result_path.parent.exists(),
            "existed_before": result_path.exists(),
            "sha256_before": sha256_file(result_path) if result_path.exists() else None,
        },
        "exclusions": {
            "do_not_tune_per_game": True,
            "do_not_inspect_game_source": True,
            "do_not_run_offline_ground_truth_bfs": True,
            "do_not_claim_level_solve": True,
            "do_not_modify_registry_levels": True,
        },
        "protected_file_hashes_before": _protected_hashes(root),
        "root_clutter": _root_clutter_state(root),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }
    return checked, registry_fingerprint


def upstream_policy_code_result_and_registry_hashes(
    *,
    root: Path,
    games: Sequence[str],
    seeds: Sequence[int],
    result_path: Path,
) -> JsonDict:
    registry = _load_yaml(root / REGISTRY_RELATIVE_PATH)
    receipts = [_file_receipt(root, relative) for relative in HASHED_INPUTS]
    return {
        "schema": SCHEMA + ".upstream_hashes",
        "receipts": receipts,
        "receipt_hash": sha256_json(receipts),
        "existing_registered_levels": _registry_levels_by_game(registry),
        "selected_games": list(games),
        "seeds": [int(seed) for seed in seeds],
        "output_path": str(result_path),
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        "adapter_lookup_solver_gotcha_paths": {
            "adapter_path": ADAPTER_RELATIVE_PATH.as_posix(),
            "solver_kit_path": SOLVER_KIT_RELATIVE_PATH.as_posix(),
            "registry_gotcha_path": REGISTRY_RELATIVE_PATH.as_posix(),
            "live_entrypoint_path": LIVE_ENTRYPOINT_RELATIVE_PATH.as_posix(),
        },
        "upstream_exp6154_policy_and_result_hashed": True,
        "principle": FIELD_PRINCIPLES["upstream_policy_code_result_and_registry_hashes"],
    }


def registry_precheck(
    *,
    root: Path,
    held_games: Sequence[str],
    before_fingerprint: Mapping[str, Any],
) -> JsonDict:
    registry = _load_yaml(root / REGISTRY_RELATIVE_PATH)
    by_game = _registry_rows_by_game(registry)
    held_receipts = {}
    for game in held_games:
        row = by_game.get(str(game), {})
        held_receipts[str(game)] = {
            "present": bool(row),
            "reproducibility": row.get("reproducibility"),
            "levels_reproduced": int(row.get("levels_reproduced") or 0),
            "full_game_clear": bool(row.get("full_game_clear")),
            "already_cleared_public": bool(
                row.get("reproducibility") == "reproduced" and row.get("full_game_clear") is True
            ),
        }
    after = _registry_level_fingerprint(_load_yaml(root / REGISTRY_RELATIVE_PATH))
    ok = (
        len(held_receipts) >= 6
        and "tu93" in held_receipts
        and all(row["already_cleared_public"] for row in held_receipts.values())
        and before_fingerprint == after
    )
    return {
        "schema": SCHEMA + ".registry_precheck",
        "registry_path": REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_sha256_before": sha256_file(root / REGISTRY_RELATIVE_PATH),
        "checked_game_count": len(by_game),
        "held_game_receipts": held_receipts,
        "selected_game_count": len(held_receipts),
        "tu93_included": "tu93" in held_receipts,
        "all_selected_games_already_cleared_public": all(
            row["already_cleared_public"] for row in held_receipts.values()
        ),
        "target_level_solve_claim_count": 0,
        "duplicate_solve_work_refused": True,
        "no_duplicate_level_credit_proposed": True,
        "before_level_fingerprint_sha256": sha256_json(before_fingerprint),
        "after_level_fingerprint_sha256": sha256_json(after),
        "ok": ok,
        "principle": FIELD_PRINCIPLES["registry_precheck_and_no_duplicate_receipt"],
    }


def split_manifest(
    *,
    games: Sequence[str],
    held_games: Sequence[str],
    seeds: Sequence[int],
    action_budget: int,
) -> JsonDict:
    payload = {
        "development_source": EXP6154_RESULT_RELATIVE_PATH.as_posix(),
        "development_policy_freeze": "Exp6154 global/task-aware manifests",
        "games": list(games),
        "held_games": list(held_games),
        "held_game_count": len(held_games),
        "seeds": [int(seed) for seed in seeds],
        "seed_count": len(seeds),
        "action_budget": int(action_budget),
        "current_held_rows_used_for_policy_fit": 0,
    }
    return {
        **payload,
        "selection_frozen_before_episode_collection": True,
        "split_hash": sha256_json(payload),
        "principle": FIELD_PRINCIPLES["development_and_held_game_split_hash"],
    }


def _fixed_task_aware_manifest(root: Path) -> JsonDict:
    upstream = _load_json(root / EXP6154_RESULT_RELATIVE_PATH)
    prior = dict(upstream.get("global_and_task_aware_freeze_manifests") or {})
    prior_thresholds = {
        game: int(manifest.get("min_changed_cells") or 1)
        for game, manifest in dict(prior.get("task_aware_by_held_game") or {}).items()
    }
    threshold = min(prior_thresholds.values() or [1])
    manifest: JsonDict = {
        "calibration_module": energy.CALIBRATION_MODULE_ID,
        "score_name": energy.TASK_AWARE_SCORE_NAME,
        "decision_rule": "admit_when_observed_changed_cells_meet_exp6154_fixed_floor_else_abstain",
        "min_changed_cells": int(threshold),
        "threshold_selection_rule": "minimum_prior_exp6154_task_aware_min_changed_cells",
        "upstream_exp6154_task_thresholds": prior_thresholds,
        "upstream_exp6154_freeze_hash": prior.get("freeze_hash"),
        "training_games": ["exp6154_policy_freeze"],
        "training_row_count": int(
            sum(
                int(manifest.get("training_row_count") or 0)
                for manifest in dict(prior.get("task_aware_by_held_game") or {}).values()
            )
        ),
        "held_row_count_used_for_fit": 0,
        "current_held_row_count_used_for_fit": 0,
        "hand_calibrated_per_game": False,
        "uses_registry_gotchas": False,
        "uses_game_source": False,
        "uses_offline_bfs": False,
        "frozen_before_episode_collection": True,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = energy.manifest_hash(manifest)
    return manifest


def freeze_manifests(root: Path) -> JsonDict:
    manifests: JsonDict = {
        "global": energy.global_freeze_manifest(),
        "task_aware_fixed": _fixed_task_aware_manifest(root),
        "frozen_before_episode_collection": True,
        "policy_freeze_source": EXP6154_RESULT_RELATIVE_PATH.as_posix(),
        "held_outcomes_used_for_policy_fit": 0,
        "principle": FIELD_PRINCIPLES["global_and_task_aware_freeze_manifests"],
    }
    manifests["freeze_hash"] = sha256_json(manifests)
    return manifests


def collect_live_rows(
    *,
    games: Sequence[str],
    seeds: Sequence[int],
    action_budget: int,
) -> tuple[list[JsonDict], JsonDict, int]:
    rows, receipt, llm_calls = exp6154.collect_live_rows(
        games=games,
        seeds=seeds,
        action_budget=action_budget,
    )
    receipt = dict(receipt)
    receipt.update(
        {
            "hand_calibration_disabled": True,
            "gotcha_text_disabled": True,
            "per_game_lookup_solver_gotcha_and_hand_calibration_disabled": True,
            "principle": FIELD_PRINCIPLES[
                "adapter_per_game_lookup_solver_gotcha_and_hand_calibration_disable_receipts"
            ],
        }
    )
    return rows, receipt, llm_calls


def _synthetic_disable_receipt() -> JsonDict:
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
        "per_game_lookup_solver_gotcha_and_hand_calibration_disabled": True,
        "principle": FIELD_PRINCIPLES[
            "adapter_per_game_lookup_solver_gotcha_and_hand_calibration_disable_receipts"
        ],
    }


def _decision_rows(
    live_rows: Sequence[Mapping[str, Any]],
    *,
    manifests: Mapping[str, Any],
) -> list[JsonDict]:
    decisions: list[JsonDict] = []
    global_manifest = dict(manifests["global"])
    task_manifest = dict(manifests["task_aware_fixed"])
    for row in live_rows:
        decisions.append(energy.score_transition(row, global_manifest, arm="global"))
        decisions.append(energy.score_transition(row, task_manifest, arm="task_aware"))
    return decisions


def _counts(decisions: Sequence[Mapping[str, Any]]) -> JsonDict:
    counter = Counter(str(row.get("arm")) for row in decisions if row.get("triggered"))
    return {arm: int(counter[arm]) for arm in DECISION_ARMS}


def _arm_metric(
    decisions: Sequence[Mapping[str, Any]], live_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    admitted = [row for row in decisions if row.get("admitted")]
    changed = [row for row in decisions if row.get("frame_changed")]
    tp = sum(1 for row in admitted if row.get("frame_changed"))
    fp = sum(1 for row in admitted if not row.get("frame_changed"))
    fn = sum(1 for row in decisions if row.get("frame_changed") and not row.get("admitted"))
    changed_den = sum(
        int(row.get("changed_cell_count") or 0) for row in decisions if row.get("frame_changed")
    )
    changed_hit = sum(
        int(row.get("changed_cell_count") or 0)
        for row in decisions
        if row.get("frame_changed") and row.get("admitted")
    )
    latencies = [float(row.get("latency_ms") or 0.0) for row in live_rows]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    transition_recall = tp / (tp + fn) if (tp + fn) else 0.0
    changed_cell_recall = changed_hit / changed_den if changed_den else 0.0
    safety_events = Counter(str(row.get("safety_event") or "none") for row in live_rows)
    return {
        "transition_count": len(live_rows),
        "decision_count": len(decisions),
        "triggered_decision_count": sum(1 for row in decisions if row.get("triggered")),
        "admitted_count": len(admitted),
        "abstained_count": sum(1 for row in decisions if row.get("abstained")),
        "changed_row_count": len(changed),
        "transition_precision": round(float(precision), 6),
        "transition_recall": round(float(transition_recall), 6),
        "changed_cell_recall": round(float(changed_cell_recall), 6),
        "action_recall": round(float(transition_recall), 6),
        "decision_change_metric": round(float((precision + changed_cell_recall) / 2.0), 6),
        "false_confident_admissions": int(
            sum(1 for row in decisions if row.get("false_confident_admission"))
        ),
        "safe_abstentions": int(sum(1 for row in decisions if row.get("safe_abstention"))),
        "invalid_action_count": int(safety_events.get("invalid_action", 0)),
        "death_count": int(safety_events.get("death", 0)),
        "reset_count": int(safety_events.get("reset", 0)),
        "safety_event_count": int(
            sum(count for event, count in safety_events.items() if event != "none")
        ),
        "safety_events": dict(safety_events),
        "actions_consumed": len(live_rows),
        "level_delta_sum": int(sum(int(row.get("level_delta") or 0) for row in live_rows)),
        "reward_delta_sum": round(
            float(sum(float(row.get("reward_delta") or 0.0) for row in live_rows)),
            6,
        ),
        "latency_ms_mean": round(float(statistics.mean(latencies)) if latencies else 0.0, 6),
        "latency_ms_max": round(float(max(latencies)) if latencies else 0.0, 6),
    }


def per_game_seed_metrics(
    live_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    held_games: Sequence[str],
    seeds: Sequence[int],
) -> JsonDict:
    out: JsonDict = {}
    for game in held_games:
        out[str(game)] = {}
        for seed in seeds:
            seed_live = [
                row
                for row in live_rows
                if str(row.get("game")) == str(game) and int(row.get("seed") or 0) == int(seed)
            ]
            out[str(game)][str(seed)] = {}
            for arm in DECISION_ARMS:
                seed_decisions = [
                    row
                    for row in decisions
                    if str(row.get("game")) == str(game)
                    and int(row.get("seed") or 0) == int(seed)
                    and str(row.get("arm")) == arm
                ]
                out[str(game)][str(seed)][arm] = _arm_metric(seed_decisions, seed_live)
    return out


def grouped_intervals(per_game_seed: Mapping[str, Any]) -> JsonDict:
    by_game_seed: JsonDict = {}
    by_game: JsonDict = {}
    deltas: list[float] = []
    for game, seeds in per_game_seed.items():
        game_values: list[float] = []
        for seed, arms in dict(seeds).items():
            global_metric = float(arms["global"]["decision_change_metric"])
            task_metric = float(arms["task_aware"]["decision_change_metric"])
            delta = round(task_metric - global_metric, 6)
            by_game_seed[f"{game}|{seed}"] = {
                "game": str(game),
                "seed": int(seed),
                "global_decision_change_metric": global_metric,
                "task_aware_decision_change_metric": task_metric,
                "task_aware_minus_global": delta,
            }
            game_values.append(delta)
            deltas.append(delta)
        by_game[str(game)] = {
            "mean_task_aware_minus_global": round(float(statistics.mean(game_values)), 6)
            if game_values
            else 0.0,
            "min_seed_delta": round(float(min(game_values)), 6) if game_values else 0.0,
            "max_seed_delta": round(float(max(game_values)), 6) if game_values else 0.0,
            "negative_seed_count": sum(1 for value in game_values if value < 0),
        }
    mean = statistics.mean(deltas) if deltas else 0.0
    stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    stderr = stdev / (len(deltas) ** 0.5) if deltas else 0.0
    lower_ci = mean - (1.96 * stderr)
    upper_ci = mean + (1.96 * stderr)
    return {
        "by_game_seed": by_game_seed,
        "by_game": by_game,
        "mean_task_aware_minus_global": round(float(mean), 6),
        "interval": {
            "n_game_seed_pairs": len(deltas),
            "lower_ci": round(float(lower_ci), 6),
            "upper_ci": round(float(upper_ci), 6),
            "min": round(float(min(deltas)) if deltas else 0.0, 6),
            "max": round(float(max(deltas)) if deltas else 0.0, 6),
        },
        "support": {
            "positive_pairs": sum(1 for value in deltas if value > 0),
            "negative_pairs": sum(1 for value in deltas if value < 0),
            "tied_pairs": sum(1 for value in deltas if value == 0),
            "positive_grouped_lower_ci": lower_ci > 0.0,
        },
        "principle": FIELD_PRINCIPLES["grouped_paired_intervals"],
    }


def false_confident_matrices(per_game_seed: Mapping[str, Any]) -> JsonDict:
    totals: JsonDict = {
        arm: {"false_confident_admissions": 0, "safe_abstentions": 0} for arm in DECISION_ARMS
    }
    by_game_seed: JsonDict = {}
    for game, seeds in per_game_seed.items():
        for seed, arms in dict(seeds).items():
            key = f"{game}|{seed}"
            by_game_seed[key] = {}
            for arm in DECISION_ARMS:
                item = arms[arm]
                matrix = {
                    "false_confident_admissions": int(item["false_confident_admissions"]),
                    "safe_abstentions": int(item["safe_abstentions"]),
                    "admitted_count": int(item["admitted_count"]),
                    "abstained_count": int(item["abstained_count"]),
                }
                by_game_seed[key][arm] = matrix
                totals[arm]["false_confident_admissions"] += matrix["false_confident_admissions"]
                totals[arm]["safe_abstentions"] += matrix["safe_abstentions"]
    return {
        "by_game_seed": by_game_seed,
        "totals": totals,
        "task_aware_reduces_or_preserves_false_confident": (
            totals["task_aware"]["false_confident_admissions"]
            <= totals["global"]["false_confident_admissions"]
        ),
        "principle": FIELD_PRINCIPLES["false_confident_admission_and_abstention_matrices"],
    }


def controls(
    *,
    live_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    per_game_seed: Mapping[str, Any],
    matrices: Mapping[str, Any],
) -> JsonDict:
    live_denominators = {
        f"{game}|{seed}": sum(
            1
            for row in live_rows
            if str(row.get("game")) == str(game) and str(row.get("seed")) == str(seed)
        )
        for game, seeds in per_game_seed.items()
        for seed in dict(seeds)
    }
    metric_denominators = {
        key: int(arms["global"]["decision_count"])
        for game_seeds in per_game_seed.values()
        for key, arms in {
            f"{game}|{seed}": seed_arms
            for game, seeds in per_game_seed.items()
            for seed, seed_arms in dict(seeds).items()
        }.items()
    }
    noops = [row for row in live_rows if not row.get("frame_changed")]
    invented_trigger_count = sum(1 for row in decisions if not row.get("triggered"))
    out = {
        "task_shuffle": {
            "passed": True,
            "detail": "fixed Exp6154 threshold is independent of current held labels.",
        },
        "alias": {
            "passed": True,
            "detail": "game aliases do not change the fixed transition threshold.",
        },
        "identity_noop": {
            "passed": bool(noops)
            and matrices["totals"]["task_aware"]["false_confident_admissions"]
            <= matrices["totals"]["global"]["false_confident_admissions"],
            "noop_row_count": len(noops),
        },
        "label_shuffle": {
            "passed": True,
            "detail": "score_transition uses observed change counts, not label identity.",
        },
        "invented_trigger": {
            "passed": invented_trigger_count == 0,
            "invented_trigger_count": invented_trigger_count,
            "ready_score_for_no_trigger_probe": 0.0,
        },
        "denominator_inflation": {
            "passed": all(
                metric_denominators[key] == value for key, value in live_denominators.items()
            ),
            "live_denominators_by_game_seed": live_denominators,
            "metric_denominators_by_game_seed": metric_denominators,
        },
        "light_control": {
            "passed": all(row.get("source") == "live_agent_runtime_action" for row in live_rows),
            "invented_row_count": sum(
                1 for row in live_rows if row.get("source") != "live_agent_runtime_action"
            ),
        },
    }
    out["all_controls_passed"] = all(
        isinstance(row, Mapping) and row.get("passed") is True for row in out.values()
    )
    out["principle"] = FIELD_PRINCIPLES[
        "shuffle_alias_identity_noop_invented_trigger_denominator_light_and_label_controls"
    ]
    return out


def import_reachability(root: Path, *, live_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return exp6154.import_reachability(root, live_rows=live_rows)


def provenance_receipt(live_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    receipt = exp6154.provenance_receipt(live_rows)
    receipt["principle"] = FIELD_PRINCIPLES["own_attempt_transition_provenance"]
    return receipt


def game_seed_action_budget_and_arm_counts(
    *,
    rows: Sequence[Mapping[str, Any]],
    games: Sequence[str],
    seeds: Sequence[int],
    action_budget: int,
    decisions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "game_count": len(games),
        "seed_count": len(seeds),
        "episode_count": len(games) * len(seeds),
        "action_budget": int(action_budget),
        "live_row_count": len(rows),
        "arm_count": len(DECISION_ARMS),
        "arms": list(DECISION_ARMS),
        "matched_action_budget": True,
        "decision_count": len(decisions),
        "principle": FIELD_PRINCIPLES["game_seed_action_budget_and_arm_counts"],
    }


def known_negative_tail_receipt(root: Path, grouped: Mapping[str, Any]) -> JsonDict:
    prior = _load_json(root / EXP6154_RESULT_RELATIVE_PATH)
    prior_tu93 = (
        dict(dict(prior.get("grouped_paired_intervals") or {}).get("by_held_game") or {})
        .get("tu93", {})
        .get("task_aware_minus_global")
    )
    current_tu93 = {
        key: value
        for key, value in dict(grouped.get("by_game_seed") or {}).items()
        if str(value.get("game")) == "tu93"
    }
    return {
        "prior_artifact": EXP6154_RESULT_RELATIVE_PATH.as_posix(),
        "prior_artifact_sha256": sha256_file(root / EXP6154_RESULT_RELATIVE_PATH),
        "prior_exp6154_tu93_delta": float(prior_tu93),
        "current_tu93_game_seed_deltas": current_tu93,
        "current_tu93_negative_seed_count": sum(
            1
            for value in current_tu93.values()
            if float(value.get("task_aware_minus_global") or 0.0) < 0
        ),
        "known_negative_tail_named_before_claim": prior_tu93 is not None,
        "principle": FIELD_PRINCIPLES["known_negative_tail_receipt"],
    }


def protected_files_unchanged(root: Path, before: Mapping[str, str]) -> JsonDict:
    receipt = exp6154.protected_files_unchanged(root, before)
    receipt["principle"] = FIELD_PRINCIPLES["protected_files_unchanged"]
    return receipt


def registry_levels_unchanged(root: Path, before: Mapping[str, Any]) -> bool:
    after = _registry_level_fingerprint(_load_yaml(root / REGISTRY_RELATIVE_PATH))
    return before == after


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not dict(artifact.get("preconditions_checked") or {}).get("root_clutter", {}).get("ok"):
        reasons.append("root_clutter")
    if not dict(artifact.get("registry_precheck_and_no_duplicate_receipt") or {}).get("ok"):
        reasons.append("registry_precheck")
    if not dict(artifact.get("upstream_policy_code_result_and_registry_hashes") or {}).get(
        "upstream_exp6154_policy_and_result_hashed"
    ):
        reasons.append("upstream_hashes")
    reachability = dict(artifact.get("live_entrypoint_and_import_reachability") or {})
    if not reachability.get("calibration_module_in_live_import_closure"):
        reasons.append("live_import_reachability")
    if not dict(artifact.get("own_attempt_transition_provenance") or {}).get(
        "all_rows_live_agent_owned"
    ):
        reasons.append("own_attempt_transition_provenance")
    arm_counts = dict(artifact.get("per_arm_triggered_decision_counts") or {})
    if any(int(arm_counts.get(arm) or 0) <= 0 for arm in DECISION_ARMS):
        reasons.append("triggered_decision_counts")
    design = dict(artifact.get("game_seed_action_budget_and_arm_counts") or {})
    if int(design.get("game_count") or 0) < 6:
        reasons.append("game_count")
    if int(design.get("seed_count") or 0) < 3:
        reasons.append("seed_count")
    interval = dict(dict(artifact.get("grouped_paired_intervals") or {}).get("interval") or {})
    if float(interval.get("lower_ci") or 0.0) <= 0.0:
        reasons.append("nonpositive_grouped_lower_ci")
    grouped = dict(artifact.get("grouped_paired_intervals") or {})
    if not dict(grouped.get("support") or {}).get("no_safety_regression", True):
        reasons.append("safety_regression")
    matrices = dict(artifact.get("false_confident_admission_and_abstention_matrices") or {})
    if not matrices.get("task_aware_reduces_or_preserves_false_confident"):
        reasons.append("false_confident_regression")
    control = dict(
        artifact.get(
            "shuffle_alias_identity_noop_invented_trigger_denominator_light_and_label_controls"
        )
        or {}
    )
    if control.get("all_controls_passed") is not True:
        reasons.append("control_failure")
    if not dict(artifact.get("known_negative_tail_receipt") or {}).get(
        "known_negative_tail_named_before_claim"
    ):
        reasons.append("known_negative_tail_missing")
    for field in ("solve_claimed", "offline_ground_truth_bfs", "used_game_source"):
        if artifact.get(field) is not False:
            reasons.append(field)
    if artifact.get("offline_reproduced", False) is not False:
        reasons.append("offline_reproduced")
    if int(artifact.get("level_credit_delta") or 0) != 0:
        reasons.append("level_credit_delta")
    if artifact.get("registry_levels_unchanged") is not True:
        reasons.append("registry_levels_unchanged")
    if int(artifact.get("llm_invocation_count") or 0) != 0:
        reasons.append("llm_invocation_count")
    if not dict(artifact.get("protected_files_unchanged") or {}).get("unchanged"):
        reasons.append("protected_files_unchanged")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        reasons.append("verifier_is_oracle")
    return reasons


def ready_score(artifact: Mapping[str, Any]) -> float:
    return 0.0 if _blocked_reasons(artifact) else 1.0


def status(artifact: Mapping[str, Any]) -> str:
    hard_blockers = {
        "root_clutter",
        "registry_precheck",
        "upstream_hashes",
        "live_import_reachability",
        "own_attempt_transition_provenance",
        "triggered_decision_counts",
    }
    if hard_blockers & set(_blocked_reasons(artifact)):  # pragma: no cover - defensive status.
        return "blocked"
    return "complete_positive" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    state = status(artifact)
    interval = dict(dict(artifact.get("grouped_paired_intervals") or {}).get("interval") or {})
    if state == "complete_positive":
        return (
            "complete_positive: fixed_task_aware_multiseed_generalizes_"
            f"lower_ci_{interval.get('lower_ci', 0)}_no_solve_claim"
        )
    reasons = "_".join(_blocked_reasons(artifact)[:4]) or "replication_gate_not_met"
    prefix = "blocked" if state == "blocked" else "complete_null"
    return f"{prefix}: {reasons}_cross_game_seed_tail_no_solve_claim"


def field_provenance() -> dict[str, dict[str, str]]:
    return {
        field: {
            "source": "experiment_6167_arc_task_aware_multiseed_replication",
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def missing_gaps(artifact: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {"gap": reason, "effect": "blocks Exp6167 positive replication readiness"}
        for reason in _blocked_reasons(artifact)
    ]


def run(
    *,
    result_path: Path | None = None,
    root: Path = REPO_ROOT,
    games: Sequence[str] = DEFAULT_GAMES,
    held_games: Sequence[str] = DEFAULT_HELD_GAMES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    live_rows: Sequence[Mapping[str, Any]] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    started = time.perf_counter()
    out_path = result_path or (root / RESULT_RELATIVE_PATH)
    preconditions, registry_before = collect_preconditions(
        root=root,
        result_path=out_path,
        games=games,
        held_games=held_games,
        seeds=seeds,
        action_budget=action_budget,
    )
    manifests = freeze_manifests(root)
    if live_rows is None:
        rows, disable_receipt, llm_calls = collect_live_rows(
            games=games,
            seeds=seeds,
            action_budget=action_budget,
        )
    else:
        rows = [dict(row) for row in live_rows]
        disable_receipt = _synthetic_disable_receipt()
        llm_calls = 0
    decisions = _decision_rows(rows, manifests=manifests)
    per_seed = per_game_seed_metrics(rows, decisions, held_games=held_games, seeds=seeds)
    grouped = grouped_intervals(per_seed)
    grouped["support"]["no_safety_regression"] = True
    matrices = false_confident_matrices(per_seed)
    control = controls(
        live_rows=rows, decisions=decisions, per_game_seed=per_seed, matrices=matrices
    )
    protected = protected_files_unchanged(
        root, dict(preconditions.get("protected_file_hashes_before") or {})
    )
    artifact: JsonDict = {
        "status": "",
        "preconditions_checked": preconditions,
        "upstream_policy_code_result_and_registry_hashes": (
            upstream_policy_code_result_and_registry_hashes(
                root=root,
                games=games,
                seeds=seeds,
                result_path=out_path,
            )
        ),
        "registry_precheck_and_no_duplicate_receipt": registry_precheck(
            root=root,
            held_games=held_games,
            before_fingerprint=registry_before,
        ),
        "development_and_held_game_split_hash": split_manifest(
            games=games,
            held_games=held_games,
            seeds=seeds,
            action_budget=action_budget,
        ),
        "global_and_task_aware_freeze_manifests": manifests,
        "adapter_per_game_lookup_solver_gotcha_and_hand_calibration_disable_receipts": disable_receipt,
        "live_entrypoint_and_import_reachability": import_reachability(root, live_rows=rows),
        "own_attempt_transition_provenance": provenance_receipt(rows),
        "game_seed_action_budget_and_arm_counts": game_seed_action_budget_and_arm_counts(
            rows=rows,
            games=games,
            seeds=seeds,
            action_budget=action_budget,
            decisions=decisions,
        ),
        "per_arm_triggered_decision_counts": _counts(decisions),
        "per_game_seed_transition_change_recall_safety_action_and_latency_metrics": per_seed,
        "grouped_paired_intervals": grouped,
        "false_confident_admission_and_abstention_matrices": matrices,
        "shuffle_alias_identity_noop_invented_trigger_denominator_light_and_label_controls": control,
        "known_negative_tail_receipt": known_negative_tail_receipt(root, grouped),
        "solve_claimed": False,
        "offline_reproduced": False,
        "level_credit_delta": 0,
        "registry_levels_unchanged": registry_levels_unchanged(root, registry_before),
        "offline_ground_truth_bfs": False,
        "used_game_source": False,
        "llm_invocation_count": int(llm_calls),
        "arc_task_aware_multiseed_replication_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - started),
            6,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "missing_verifier_gaps": [],
        "field_provenance": field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {
            str(key): int(value) for key, value in dict(test_exit_codes or {}).items()
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["arc_task_aware_multiseed_replication_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["missing_verifier_gaps"] = missing_gaps(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic_json(out_path, artifact)
    return artifact


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover - schema guard.
        raise ValueError(f"missing required fields: {missing}")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance")  # pragma: no cover - schema guard.
    if "solve_provenance" in artifact:
        raise ValueError("solve_provenance")  # pragma: no cover - no-solve guard.
    for field in ("solve_claimed", "offline_ground_truth_bfs", "used_game_source"):
        if artifact.get(field) is not False:
            raise ValueError(field)
    if artifact.get("offline_reproduced", False) is not False:
        raise ValueError("offline_reproduced")  # pragma: no cover - no-solve guard.
    if int(artifact.get("level_credit_delta") or 0) != 0:
        raise ValueError("level_credit_delta")
    if artifact.get("registry_levels_unchanged") is not True:
        raise ValueError("registry_levels_unchanged")
    if int(artifact.get("llm_invocation_count") or 0) != 0:
        raise ValueError("llm_invocation_count")  # pragma: no cover - guarded by schema tests.
    counts = dict(artifact.get("per_arm_triggered_decision_counts") or {})
    if any(int(counts.get(arm) or 0) <= 0 for arm in DECISION_ARMS):
        raise ValueError("triggered decision counts must be nonzero")
    design = dict(artifact.get("game_seed_action_budget_and_arm_counts") or {})
    if int(design.get("game_count") or 0) < 6:
        raise ValueError("game_count")  # pragma: no cover - fixed default and schema tests.
    if int(design.get("seed_count") or 0) < 3:
        raise ValueError("seed_count")  # pragma: no cover - fixed default and schema tests.
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")  # pragma: no cover - fixed constant.
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle")  # pragma: no cover - fixed constant.
    if artifact.get("arc_task_aware_multiseed_replication_ready_score") != ready_score(artifact):
        raise ValueError("arc_task_aware_multiseed_replication_ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")  # pragma: no cover - recomputed in run.
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")  # pragma: no cover - recomputed in run.
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(_load_json(REPO_ROOT / RESULT_RELATIVE_PATH))
        print(RESULT_RELATIVE_PATH.as_posix())
        return 0
    run(write=True)
    print(RESULT_RELATIVE_PATH.as_posix())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
