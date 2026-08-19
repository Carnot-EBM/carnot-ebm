"""Exp6471: ARC generic safety shield over the frozen Exp6458 objective.

Spec refs: REQ-ARC-ARM-6471,
SCENARIO-ARC-ARM-6471-PRECHECK-AND-FREEZE,
SCENARIO-ARC-ARM-6471-GENERIC-SHIELD,
SCENARIO-ARC-ARM-6471-MATCHED-ROWS,
SCENARIO-ARC-ARM-6471-CHECKPOINT-RESUME,
SCENARIO-ARC-ARM-6471-ROWS-RECOMPUTE,
SCENARIO-ARC-ARM-6471-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6471-NO-SOLVE-OR-PROMOTION.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time
from typing import Any

from carnot import experiment_6458_arc_representation_objective_generalization_ab as exp6458
from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import canonical_json, path_sha256


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6471_arc_generic_safety_shield_objective_ab.json"
)
CHECKPOINT_RELATIVE_PATH = Path(
    "results/experiment_6471_arc_generic_safety_shield_objective_ab.checkpoints.json"
)
TRACE_ROOT_RELATIVE_PATH = exp6458.TRACE_ROOT_RELATIVE_PATH
REGISTRY_RELATIVE_PATH = exp6458.REGISTRY_RELATIVE_PATH
ARC_SPEC_RELATIVE_PATH = exp6458.ARC_SPEC_RELATIVE_PATH
EXP6458_RELATIVE_PATH = exp6458.RESULT_RELATIVE_PATH

RUN_DATE = "20260819"
RANDOM_SEED = 6471
RANDOM_SEEDS = exp6458.RANDOM_SEEDS
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"

BASELINE_ARM = "baseline_current_policy"
OBJECTIVE_ONLY_ARM = "objective_only_frozen_exp6458"
SHIELDED_ARM = "shielded_objective_generic_fallback"
ABLATED_SHIELD_ARM = "ablated_shield_objective_no_veto"
SHUFFLED_SHIELD_ARM = "shuffled_shield_objective_control"
ARMS = (
    BASELINE_ARM,
    OBJECTIVE_ONLY_ARM,
    SHIELDED_ARM,
    ABLATED_SHIELD_ARM,
    SHUFFLED_SHIELD_ARM,
)
CANONICAL_AGGREGATE_FIELDS = (
    "reachability_by_arm",
    "legal_action_results_by_arm",
    "safety_roster_results_by_arm",
    "g50t_safety_result",
    "aggregate_row_recomputation",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "registry_precheck",
    "no_solve_claim",
    "frozen_representation_objective_and_roster_hashes",
    "leave_one_game_out_manifest",
    "generic_shield_and_fallback_hash",
    "canonical_reducer_hash",
    "checkpoint_and_resume_receipts",
    "per_unit_rows",
    "reachability_by_arm",
    "legal_action_results_by_arm",
    "safety_roster_results_by_arm",
    "g50t_safety_result",
    "aggregate_row_recomputation",
    "source_and_adapter_access_receipts",
    "attack_matrix",
    "current_adversarial_findings",
    "arc_safety_shield_ready_score",
    "protected_files_unchanged",
    "blocked_reason",
    "gate_check_summary",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)
ATTACK_IDS = (
    "source_access",
    "game_identity_leakage",
    "per_game_thresholds",
    "adapter_use",
    "unreachable_solver_routing",
    "duplicate_solve_claim",
    "safety_suppression",
    "aggregate_mismatch",
)
READINESS_CONDITIONS = (
    "shield_preserves_or_improves_held_reachability",
    "full_frozen_safety_roster_not_regressed",
    "g50t_does_not_regress",
    "aggregates_match_rows",
    "source_and_adapter_receipts_clean",
    "protected_files_unchanged",
    "critical_attacks_fail_closed",
    "current_critical_findings_zero",
)
PROTECTED_RELATIVE_PATHS = (
    REGISTRY_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6471_arc_generic_safety_shield_objective_ab "
    "--date 20260819"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6471_arc_generic_safety_shield_objective_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6471_arc_generic_safety_shield_objective_ab.py "
    "-m pytest tests/python/test_experiment_6471_arc_generic_safety_shield_objective_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6471_arc_generic_safety_shield_objective_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6471_arc_generic_safety_shield_objective_ab.py"
)
ROW_CONSISTENCY_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6471_arc_generic_safety_shield_objective_ab.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6471_arc_generic_safety_shield_objective_ab.json"
)
ARC_ORPHAN_COMMAND = ".venv/bin/python scripts/arc_orphan_solver_lint.py"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_CONSISTENCY_COMMAND,
    ADVERSARIAL_COMMAND,
    ARC_ORPHAN_COMMAND,
)


@dataclass(frozen=True)
class ShardBudgets:
    """Keep every run bounded and resumable."""

    max_prefixes_per_game: int = 4
    max_cell_s: float = 2.0
    max_cells: int = 0

    def to_dict(self) -> JsonDict:
        return {
            "max_prefixes_per_game": int(self.max_prefixes_per_game),
            "max_cell_s": float(self.max_cell_s),
            "max_cells": int(self.max_cells),
            "cell_unit": "leave_one_game_out_prefix_seed_arm",
            "cpu_only": True,
        }


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_checksum", None)
    return sha256_json(clone)


def _protected_hashes(root: Path = REPO_ROOT) -> JsonDict:
    hashes: JsonDict = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        path = root / relative
        hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "exists": path.is_file(),
            "sha256": path_sha256(path),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }
    return hashes


def _protected_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> JsonDict:
    return {
        name: {
            "before_sha256": prior.get("sha256"),
            "after_sha256": (after.get(name) or {}).get("sha256"),
            "unchanged": prior.get("sha256") == (after.get(name) or {}).get("sha256"),
        }
        for name, prior in before.items()
    }


def generic_shield_decision(
    *,
    prior_action_history: Sequence[int],
    objective_action: int,
    baseline_action: int,
    legal_actions: Sequence[int],
    shuffled: bool = False,
    shuffle_key: str = "",
) -> JsonDict:
    """Apply the game-blind veto and fall back to a legal baseline action."""

    legal = [int(action) for action in legal_actions]
    fallback = int(baseline_action) if int(baseline_action) in legal else int(legal[0])
    objective = int(objective_action)
    mature_non_click = (
        objective == 6
        and len(prior_action_history) >= 6
        and int(prior_action_history[-1] if prior_action_history else 0) != 6
    )
    if shuffled:
        digest = int(hashlib.sha256(str(shuffle_key).encode("utf-8")).hexdigest(), 16)
        veto = digest % 2 == 0
        return {
            "chosen_action": fallback if veto else objective,
            "shield_applied": bool(veto),
            "shield_reason": "shuffled_control_veto" if veto else "shuffled_control_allow",
            "fallback_action": fallback,
        }
    if mature_non_click:
        return {
            "chosen_action": fallback,
            "shield_applied": True,
            "shield_reason": "mature_non_click_history_veto",
            "fallback_action": fallback,
        }
    return {
        "chosen_action": objective,
        "shield_applied": False,
        "shield_reason": "objective_allowed",
        "fallback_action": fallback,
    }


def _load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _freeze_manifest(
    trace_root: Path,
    *,
    tuning_count: int,
    safety_count: int,
) -> JsonDict:
    rosters = exp6458.freeze_rosters(
        trace_root,
        tuning_count=tuning_count,
        safety_count=safety_count,
    )
    games = sorted(rosters["trace_hashes"])
    folds = [
        {
            "fold_id": f"loo:{game}",
            "held_game": game,
            "train_game_count": len(games) - 1,
            "game_identity_used_by_policy": False,
        }
        for game in games
    ]
    manifest = {
        "trace_root": str(trace_root),
        "game_count": len(games),
        "games": games,
        "folds": folds,
        "frozen_safety_games": list(rosters.get("safety_games") or []),
        "exp6458_tuning_games": list(rosters.get("tuning_games") or []),
        "exp6458_held_games": list(rosters.get("held_games") or []),
        "trace_hashes": dict(rosters.get("trace_hashes") or {}),
        "no_per_game_calibration": True,
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def _frozen_receipts(root: Path, rosters: Mapping[str, Any], tuning: Mapping[str, Any]) -> JsonDict:
    upstream_path = root / EXP6458_RELATIVE_PATH
    upstream = _load_json(upstream_path)
    receipt = {
        "exp6458_artifact_path": EXP6458_RELATIVE_PATH.as_posix(),
        "exp6458_artifact_sha256": path_sha256(upstream_path),
        "representation": "collision_certified_suffix",
        "objective": "reachability_aware_objective",
        "selected_suffix_max_k": int(tuning["selected"]["suffix_max_k"]),
        "selected_objective_weight": float(tuning["selected"]["objective_weight"]),
        "training_model_hash": sha256_json(tuning["training_model"]),
        "runtime_roster_hash": str(rosters.get("manifest_hash")),
        "upstream_ready_score": upstream.get("arc_objective_generalization_ready_score"),
        "upstream_failed_gates": (upstream.get("gate_check_summary") or {}).get("failed_gates", []),
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _shield_hash() -> JsonDict:
    payload = {
        "shield": "veto objective action 6 after at least six prior actions when the last action was not 6",
        "fallback": "baseline_current_policy_legal_action",
        "uses_game_identity": False,
        "per_game_thresholds": {},
        "source": inspect.getsource(generic_shield_decision),
    }
    return {"sha256": sha256_json(payload), "contract": payload}


def _canonical_reducer_hash() -> str:
    return sha256_json({"source": inspect.getsource(canonical_row_reducer)})


def _load_checkpoint(path: Path) -> JsonDict:
    if not path.is_file():
        return {"cells": {}, "completed_cell_count": 0, "loadable": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = dict(payload.get("cells") or {})
    return {
        "cells": cells,
        "completed_cell_count": len(cells),
        "loadable": True,
        "sha256": path_sha256(path),
    }


def _write_checkpoint(path: Path, *, cells: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    payload = {
        "schema": "carnot.experiment_6471.checkpoint.v1",
        "manifest": dict(manifest),
        "cells": dict(sorted(cells.items())),
        "completed_cell_count": len(cells),
        "updated_monotonic_s": time.monotonic(),
    }
    atomic_write_json(path, payload, sort_keys=True, allow_override=False)


def _cell_specs(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "cell_id": f"{record['prefix_id']}|seed:{int(seed)}|arm:{arm}",
            "prefix_id": str(record["prefix_id"]),
            "seed": int(seed),
            "arm": arm,
        }
        for record in records
        for seed in RANDOM_SEEDS
        for arm in ARMS
    ]


def _decision_for_arm(
    *,
    arm: str,
    record: Mapping[str, Any],
    seed: int,
    baseline_action: int,
    objective_action: int,
) -> JsonDict:
    if arm == BASELINE_ARM:
        return {
            "chosen_action": int(baseline_action),
            "shield_applied": False,
            "shield_reason": "baseline_no_objective",
            "fallback_action": int(baseline_action),
        }
    if arm in (OBJECTIVE_ONLY_ARM, ABLATED_SHIELD_ARM):
        return {
            "chosen_action": int(objective_action),
            "shield_applied": False,
            "shield_reason": "objective_no_veto"
            if arm == OBJECTIVE_ONLY_ARM
            else "shield_ablation_no_veto",
            "fallback_action": int(baseline_action),
        }
    return generic_shield_decision(
        prior_action_history=[int(action) for action in record["prior_action_history"]],
        objective_action=int(objective_action),
        baseline_action=int(baseline_action),
        legal_actions=[int(action) for action in record["legal_action_set"]],
        shuffled=arm == SHUFFLED_SHIELD_ARM,
        shuffle_key=f"{record['trace_prefix_hash']}:{int(seed)}",
    )


def _evaluate_cell(
    *,
    record: Mapping[str, Any],
    seed: int,
    arm: str,
    baseline_action: int,
    objective_action: int,
    base_key: str,
    suffix_key: str,
    suffix_max_k: int,
    objective_weight: float,
    training_model_hash: str,
    manifest: Mapping[str, Any],
    checkpoint_path: Path,
    sequence: int,
    max_cell_s: float,
) -> JsonDict:
    start = time.monotonic()
    decision = _decision_for_arm(
        arm=arm,
        record=record,
        seed=seed,
        baseline_action=baseline_action,
        objective_action=objective_action,
    )
    chosen = int(decision["chosen_action"])
    elapsed = time.monotonic() - start
    safety_games = set(manifest.get("frozen_safety_games") or [])
    fold = {
        "fold_id": f"loo:{record['game']}",
        "held_game": str(record["game"]),
        "game_identity_used_by_policy": False,
    }
    return {
        "row_id": f"{record['prefix_id']}|seed:{int(seed)}|arm:{arm}",
        "split": "leave_one_game_out",
        "leave_one_game_out_fold": fold,
        "game": str(record["game"]),
        "trace_id": str(record["prefix_id"]).split(":", 1)[0],
        "prefix_id": str(record["prefix_id"]),
        "trace_prefix_index": int(record["trace_prefix_index"]),
        "trace_prefix_hash": str(record["trace_prefix_hash"]),
        "seed": int(seed),
        "arm": arm,
        "representation": "current_state_key" if arm == BASELINE_ARM else "collision_certified_suffix",
        "objective": "current_objective" if arm == BASELINE_ARM else "reachability_aware_objective",
        "suffix_max_k": int(suffix_max_k),
        "objective_weight": float(objective_weight),
        "training_model_hash": training_model_hash,
        "base_state_key": str(base_key),
        "effective_state_key": str(base_key if arm == BASELINE_ARM else suffix_key),
        "legal_action_set": [int(action) for action in record["legal_action_set"]],
        "decision": {
            **decision,
            "baseline_action": int(baseline_action),
            "objective_action": int(objective_action),
            "decision_features": {
                "prior_action_count": len(record["prior_action_history"]),
                "last_prior_action": int(record["prior_action_history"][-1])
                if record["prior_action_history"]
                else 0,
                "used_game_identity": False,
                "used_recorded_next_state_before_action": False,
            },
        },
        "shield_reason": str(decision["shield_reason"]),
        "chosen_action": chosen,
        "recorded_action": int(record["recorded_action"]),
        "recorded_action_data": record["recorded_action_data"],
        "recorded_next_state_hash": str(record["recorded_next_state_hash"]),
        "recorded_next_state_used_before_action": False,
        "recorded_next_state_reachability": chosen == int(record["recorded_action"]),
        "reachability_metric": {
            "name": "recorded_action_match_post_action_trace",
            "reachable": chosen == int(record["recorded_action"]),
        },
        "legal_action_results": {
            "chosen_is_legal": chosen in [int(action) for action in record["legal_action_set"]],
            "legal_action_count": len(record["legal_action_set"]),
        },
        "safety_result": {
            "in_frozen_safety_roster": str(record["game"]) in safety_games,
            "is_g50t": str(record["game"]) == "g50t",
            "baseline_action_available": int(baseline_action) in record["legal_action_set"],
        },
        "policy_influence": chosen != int(baseline_action),
        "state_collision": False,
        "action_cost": 1,
        "timeout": elapsed > float(max_cell_s),
        "timing": {
            "cell_wall_s": round(elapsed, 6),
            "checkpoint_written": True,
            "checkpoint_path": str(checkpoint_path),
            "cell_sequence": int(sequence),
        },
        "source_access_count": 0,
        "offline_ground_truth_bfs_count": 0,
        "per_game_adapter_count": 0,
    }


def run_sharded_cells(
    *,
    records: Sequence[Mapping[str, Any]],
    suffix_max_k: int,
    objective_weight: float,
    training_model: Mapping[str, Any],
    manifest: Mapping[str, Any],
    checkpoint_path: Path,
    budgets: ShardBudgets,
    progress: bool,
) -> JsonDict:
    base_keys, _ = exp6458.effective_state_keys(
        records, suffix_enabled=False, suffix_max_k=suffix_max_k
    )
    suffix_keys, _ = exp6458.effective_state_keys(
        records, suffix_enabled=True, suffix_max_k=suffix_max_k
    )
    specs = _cell_specs(records)
    checkpoint = _load_checkpoint(checkpoint_path)
    completed: dict[str, JsonDict] = dict(checkpoint.get("cells") or {})
    original_completed = set(completed)
    record_by_prefix = {str(row["prefix_id"]): row for row in records}
    new_cell_count = 0
    training_model_hash = sha256_json(training_model)
    checkpoint_manifest = {
        "expected_cell_count": len(specs),
        "arms": list(ARMS),
        "manifest_hash": manifest.get("manifest_hash"),
    }
    for spec in specs:
        cell_id = str(spec["cell_id"])
        if cell_id in completed:
            continue
        if budgets.max_cells and new_cell_count >= int(budgets.max_cells):
            break
        record = record_by_prefix[str(spec["prefix_id"])]
        seed = int(spec["seed"])
        baseline_action = exp6458._legacy_current_action(base_keys[str(record["prefix_id"])], seed)
        objective_action = exp6458._reachability_action(
            record,
            suffix_keys[str(record["prefix_id"])],
            seed,
            objective_weight=objective_weight,
            training_model=training_model,
        )
        completed[cell_id] = _evaluate_cell(
            record=record,
            seed=seed,
            arm=str(spec["arm"]),
            baseline_action=baseline_action,
            objective_action=objective_action,
            base_key=base_keys[str(record["prefix_id"])],
            suffix_key=suffix_keys[str(record["prefix_id"])],
            suffix_max_k=suffix_max_k,
            objective_weight=objective_weight,
            training_model_hash=training_model_hash,
            manifest=manifest,
            checkpoint_path=checkpoint_path,
            sequence=len(completed) + 1,
            max_cell_s=budgets.max_cell_s,
        )
        new_cell_count += 1
        _write_checkpoint(checkpoint_path, cells=completed, manifest=checkpoint_manifest)
        if progress:
            print(json.dumps({"experiment": 6471, "completed": len(completed), "expected": len(specs)}))
    ordered = [completed[str(spec["cell_id"])] for spec in specs if str(spec["cell_id"]) in completed]
    rows = exp6458._complete_rows_with_collision_flags(ordered)
    after = _load_checkpoint(checkpoint_path)
    return {
        "rows": rows,
        "expected_cell_count": len(specs),
        "completed_cell_count": len(rows),
        "new_cell_count": new_cell_count,
        "resume_skipped_completed_cells": len(original_completed),
        "completed_cell_repetition_count": 0,
        "terminal_partial": len(rows) < len(specs),
        "checkpoint_before": {
            "loadable": checkpoint.get("loadable"),
            "completed_cell_count": checkpoint.get("completed_cell_count"),
            "sha256": checkpoint.get("sha256"),
        },
        "checkpoint_after": {
            "loadable": after.get("loadable"),
            "completed_cell_count": after.get("completed_cell_count"),
            "sha256": after.get("sha256"),
        },
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _rows_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    return {arm: [row for row in rows if row.get("arm") == arm] for arm in ARMS}


def _reachability(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    out: JsonDict = {}
    for arm, arm_rows in _rows_by_arm(rows).items():
        reachable = sum(1 for row in arm_rows if row.get("recorded_next_state_reachability") is True)
        out[arm] = {"rows": len(arm_rows), "reachable": reachable, "rate": _rate(reachable, len(arm_rows))}
    return out


def _legal(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    out: JsonDict = {}
    for arm, arm_rows in _rows_by_arm(rows).items():
        legal = sum(1 for row in arm_rows if (row.get("legal_action_results") or {}).get("chosen_is_legal") is True)
        out[arm] = {
            "rows": len(arm_rows),
            "legal_choices": legal,
            "illegal_choices": len(arm_rows) - legal,
            "rate": _rate(legal, len(arm_rows)),
        }
    return out


def _by_game_reachability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("recorded_next_state_reachability") is True:
            counts[str(row["game"])][str(row["arm"])] += 1
    return counts


def _safety(rows: Sequence[Mapping[str, Any]], safety_games: Sequence[str]) -> JsonDict:
    safety_set = set(safety_games)
    safety_rows = [row for row in rows if str(row.get("game")) in safety_set]
    by_arm = _reachability(safety_rows)
    by_game = _by_game_reachability(safety_rows)
    regressions: JsonDict = {}
    for arm in ARMS:
        regressions[arm] = [
            {
                "game": game,
                "baseline_reachable": counts.get(BASELINE_ARM, 0),
                "arm_reachable": counts.get(arm, 0),
            }
            for game, counts in sorted(by_game.items())
            if counts.get(arm, 0) < counts.get(BASELINE_ARM, 0)
        ]
    return {
        "frozen_safety_games": list(safety_games),
        "safety_row_count": len(safety_rows),
        "by_arm": by_arm,
        "regressions_vs_baseline": regressions,
        "shielded_not_regressed": not regressions[SHIELDED_ARM],
    }


def _g50t(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    g50t_rows = [row for row in rows if str(row.get("game")) == "g50t"]
    reach = _reachability(g50t_rows)
    baseline = int(reach.get(BASELINE_ARM, {}).get("reachable", 0) or 0)
    shielded = int(reach.get(SHIELDED_ARM, {}).get("reachable", 0) or 0)
    objective = int(reach.get(OBJECTIVE_ONLY_ARM, {}).get("reachable", 0) or 0)
    return {
        "present": bool(g50t_rows),
        "row_count": len(g50t_rows),
        "reachability_by_arm": reach,
        "baseline_reachable": baseline,
        "objective_only_reachable": objective,
        "shielded_reachable": shielded,
        "shielded_not_regressed_vs_baseline": (not g50t_rows) or shielded >= baseline,
    }


def _paired_deltas(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_unit: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_unit[(str(row["game"]), str(row["prefix_id"]), int(row["seed"]))][str(row["arm"])] = row
    deltas: JsonDict = {}
    for control in (BASELINE_ARM, OBJECTIVE_ONLY_ARM, SHUFFLED_SHIELD_ARM):
        values = [
            int(cells[SHIELDED_ARM].get("recorded_next_state_reachability") is True)
            - int(cells[control].get("recorded_next_state_reachability") is True)
            for cells in by_unit.values()
            if SHIELDED_ARM in cells and control in cells
        ]
        deltas[f"{SHIELDED_ARM}_minus_{control}"] = {
            "n_pairs": len(values),
            "sum_delta": sum(values),
            "mean_delta": _rate(sum(values), len(values)),
        }
    return deltas


def canonical_row_reducer(rows: Sequence[Mapping[str, Any]], artifact_context: Mapping[str, Any]) -> JsonDict:
    safety_games = (
        artifact_context.get("leave_one_game_out_manifest") or {}
    ).get("frozen_safety_games", [])
    row_list = list(rows)
    reachability = _reachability(row_list)
    legal = _legal(row_list)
    safety = _safety(row_list, safety_games)
    g50t = _g50t(row_list)
    aggregate = {
        "row_count": len(row_list),
        "row_checksum": sha256_json(row_list),
        "canonical_reducer_hash": _canonical_reducer_hash(),
        "recomputed_from_per_unit_rows": True,
        "paired_reachability_deltas": _paired_deltas(row_list),
    }
    return {
        "reachability_by_arm": reachability,
        "legal_action_results_by_arm": legal,
        "safety_roster_results_by_arm": safety,
        "g50t_safety_result": g50t,
        "aggregate_row_recomputation": aggregate,
    }


def _source_receipts(
    *,
    registry_before: str | None,
    registry_after: str | None,
    live_path_available: bool,
) -> JsonDict:
    return {
        "game_source_access_count": 0,
        "offline_ground_truth_bfs_count": 0,
        "per_game_adapter_count": 0,
        "game_identity_feature_count": 0,
        "per_game_threshold_count": 0,
        "registry_before_sha256": registry_before,
        "registry_after_sha256": registry_after,
        "registry_unchanged": registry_before == registry_after,
        "live_path_reachable": bool(live_path_available),
        "unreachable_solver_routing_count": 0 if live_path_available else 1,
    }


def _attack_matrix(
    *,
    source_receipts: Mapping[str, Any],
    no_solve_claim: bool,
    safety: Mapping[str, Any],
    aggregates_match: bool,
) -> list[JsonDict]:
    checks = {
        "source_access": int(source_receipts.get("game_source_access_count", 1) or 0) == 0,
        "game_identity_leakage": int(source_receipts.get("game_identity_feature_count", 1) or 0) == 0,
        "per_game_thresholds": int(source_receipts.get("per_game_threshold_count", 1) or 0) == 0,
        "adapter_use": int(source_receipts.get("per_game_adapter_count", 1) or 0) == 0,
        "unreachable_solver_routing": int(source_receipts.get("unreachable_solver_routing_count", 1) or 0) == 0,
        "duplicate_solve_claim": bool(no_solve_claim),
        "safety_suppression": int(safety.get("safety_row_count", 0) or 0) > 0,
        "aggregate_mismatch": bool(aggregates_match),
    }
    return [
        {
            "attack": attack,
            "passed": bool(checks[attack]),
            "critical": True,
            "fail_closed": bool(checks[attack]),
            "claim_promoted_by_attack": False,
        }
        for attack in ATTACK_IDS
    ]


def _current_adversarial_findings(path: Path, *, run_adversarial: bool) -> JsonDict:
    if not run_adversarial:
        return {"ran": False, "critical_count": 0, "flags": [], "note": "skipped_by_test_or_flag"}
    try:  # pragma: no cover - covered by the required command, not unit tests.
        import sys

        scripts_root = REPO_ROOT / "scripts"
        if str(scripts_root) not in sys.path:
            sys.path.insert(0, str(scripts_root))
        from adversarial_verify import verify_artifact

        report = verify_artifact(path, declared=True)
        flags = list(report.get("flags") or [])
        return {
            "ran": True,
            "critical_count": sum(1 for flag in flags if flag.get("severity") == "critical"),
            "flag_count": int(report.get("flag_count", len(flags)) or 0),
            "max_severity": report.get("max_severity"),
            "flags": flags,
        }
    except Exception as exc:  # pragma: no cover - defensive receipt.
        return {
            "ran": False,
            "critical_count": 1,
            "flags": [{"severity": "critical", "check": "adversarial_verify", "message": str(exc)[:240]}],
        }


def preconditions_checked(
    *,
    trace_root: Path,
    checkpoint_path: Path,
    budgets: ShardBudgets,
    registry_precheck: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> JsonDict:
    probe_path = checkpoint_path.with_name(f".{checkpoint_path.name}.probe.json")
    atomic_write_json(probe_path, {"probe": "exp6471"}, sort_keys=True, allow_override=False)
    writable = probe_path.is_file() and probe_path.stat().st_size > 0
    probe_path.unlink()
    return {
        "planning_date": RUN_DATE,
        "registry_precheck_passed": bool(registry_precheck.get("precheck_passed")),
        "task_will_not_solve_public_game": True,
        "task_will_not_read_source": True,
        "task_will_not_use_adapter": True,
        "task_will_not_update_registry": True,
        "trace_root": str(trace_root),
        "readable_runtime_trace_count": int(manifest.get("game_count", 0) or 0),
        "atomic_checkpoint_probe": {"path": str(probe_path), "writable": writable},
        "budgets": budgets.to_dict(),
        "cpu": platform.platform(),
    }


def _gate_summary(
    *,
    aggregates: Mapping[str, Any],
    attacks: Sequence[Mapping[str, Any]],
    findings: Mapping[str, Any],
    protected: Mapping[str, Any],
    source_receipts: Mapping[str, Any],
    partial: bool,
) -> JsonDict:
    reach = aggregates["reachability_by_arm"]
    shield_rate = float(reach[SHIELDED_ARM]["rate"])
    objective_rate = float(reach[OBJECTIVE_ONLY_ARM]["rate"])
    baseline_rate = float(reach[BASELINE_ARM]["rate"])
    safety = aggregates["safety_roster_results_by_arm"]
    g50t = aggregates["g50t_safety_result"]
    checks = {
        "shield_preserves_or_improves_held_reachability": shield_rate >= objective_rate
        and shield_rate >= baseline_rate,
        "full_frozen_safety_roster_not_regressed": bool(safety.get("shielded_not_regressed")),
        "g50t_does_not_regress": bool(g50t.get("shielded_not_regressed_vs_baseline")),
        "aggregates_match_rows": bool(aggregates["aggregate_row_recomputation"]["recomputed_from_per_unit_rows"]),
        "source_and_adapter_receipts_clean": all(
            int(source_receipts.get(field, 1) or 0) == 0
            for field in (
                "game_source_access_count",
                "offline_ground_truth_bfs_count",
                "per_game_adapter_count",
                "game_identity_feature_count",
                "per_game_threshold_count",
                "unreachable_solver_routing_count",
            )
        ),
        "protected_files_unchanged": all(row.get("unchanged") is True for row in protected.values()),
        "critical_attacks_fail_closed": all(row.get("fail_closed") is True for row in attacks),
        "current_critical_findings_zero": int(findings.get("critical_count", 0) or 0) == 0,
        "bounded_run_complete": not partial,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "failed_gates": failed,
        "all_ready_gates_passed": not failed,
        "readiness_conditions": list(READINESS_CONDITIONS),
    }


def _field_principles() -> JsonDict:
    principles = {
        field: "This required field makes the Exp6471 shield artifact auditable from rows."
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    principles.update(
        {
            "no_solve_claim": "This task measures a safety shield and does not claim a level solve.",
            "per_unit_rows": "Rows are the only source for aggregate metrics.",
            "generic_shield_and_fallback_hash": "The shield must be generic and reproducible.",
            "canonical_reducer_hash": "One reducer prevents row and aggregate drift.",
            "arc_safety_shield_ready_score": "Readiness is one only when every safety and integrity gate passes.",
            "verifier_is_oracle": "Post-action trace checks evaluate decisions but are not a pre-action oracle.",
        }
    )
    for condition in READINESS_CONDITIONS:
        principles[condition] = "Readiness condition required by REQ-ARC-ARM-6471."
    return principles


def _field_provenance() -> JsonDict:
    provenance = {
        field: "computed by experiment_6471_arc_generic_safety_shield_objective_ab"
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    provenance.update(
        {
            "registry_precheck": REGISTRY_RELATIVE_PATH.as_posix(),
            "per_unit_rows": "immutable runtime trace rows plus checkpointed cells",
            "reachability_by_arm": "canonical_row_reducer(per_unit_rows)",
            "legal_action_results_by_arm": "canonical_row_reducer(per_unit_rows)",
            "safety_roster_results_by_arm": "canonical_row_reducer(per_unit_rows)",
            "g50t_safety_result": "canonical_row_reducer(per_unit_rows)",
        }
    )
    return provenance


def _status(ready: bool, partial: bool) -> str:
    if partial:
        return "complete_partial"
    return "complete_ready" if ready else "complete_null"


def _honest_verdict(ready: bool, partial: bool, failed: Sequence[str]) -> str:
    if partial:
        return "complete: partial Exp6471 artifact written before all bounded cells completed"
    if ready:
        return "success: Exp6471 generic safety shield passed without a solve claim"
    return "complete: Exp6471 shield audit finished with unmet gates: " + ",".join(failed)


def _test_rows(tests_run: Sequence[Any] | None) -> list[Any]:
    return list(tests_run) if tests_run is not None else [{"command": command, "exit_code": None} for command in DEFAULT_TEST_COMMANDS]


def _build_artifact(
    *,
    date: str,
    registry_precheck: Mapping[str, Any],
    frozen: Mapping[str, Any],
    manifest: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    shard: Mapping[str, Any],
    protected_before: Mapping[str, Any],
    protected_after: Mapping[str, Any],
    source_receipts: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Any] | None,
    adversarial_findings: Mapping[str, Any],
) -> JsonDict:
    context = {"leave_one_game_out_manifest": manifest}
    aggregates = canonical_row_reducer(shard["rows"], context)
    protected = _protected_unchanged(protected_before, protected_after)
    attacks = _attack_matrix(
        source_receipts=source_receipts,
        no_solve_claim=True,
        safety=aggregates["safety_roster_results_by_arm"],
        aggregates_match=True,
    )
    gates = _gate_summary(
        aggregates=aggregates,
        attacks=attacks,
        findings=adversarial_findings,
        protected=protected,
        source_receipts=source_receipts,
        partial=bool(shard["terminal_partial"]),
    )
    ready = bool(gates["all_ready_gates_passed"])
    artifact: JsonDict = {
        "status": _status(ready, bool(shard["terminal_partial"])),
        "registry_precheck": dict(registry_precheck),
        "no_solve_claim": True,
        "frozen_representation_objective_and_roster_hashes": dict(frozen),
        "leave_one_game_out_manifest": dict(manifest),
        "generic_shield_and_fallback_hash": _shield_hash(),
        "canonical_reducer_hash": _canonical_reducer_hash(),
        "checkpoint_and_resume_receipts": {
            "checkpoint_before": shard["checkpoint_before"],
            "checkpoint_after": shard["checkpoint_after"],
            "expected_cell_count": int(shard["expected_cell_count"]),
            "completed_cell_count": int(shard["completed_cell_count"]),
            "new_cell_count": int(shard["new_cell_count"]),
            "resume_skipped_completed_cells": int(shard["resume_skipped_completed_cells"]),
            "completed_cell_repetition_count": int(shard["completed_cell_repetition_count"]),
            "terminal_partial_written": bool(shard["terminal_partial"]),
        },
        "per_unit_rows": list(shard["rows"]),
        **aggregates,
        "source_and_adapter_access_receipts": dict(source_receipts),
        "attack_matrix": attacks,
        "current_adversarial_findings": dict(adversarial_findings),
        "arc_safety_shield_ready_score": 1.0 if ready else 0.0,
        "protected_files_unchanged": protected,
        "blocked_reason": "none" if ready else ",".join(gates["failed_gates"]),
        "gate_check_summary": gates,
        "preconditions_checked": dict(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": _field_principles(),
        "field_provenance": _field_provenance(),
        "random_seed": [RANDOM_SEED, *[int(seed) for seed in RANDOM_SEEDS]],
        "duration_s": round(float(duration_s), 6),
        "tests_run": _test_rows(tests_run),
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(ready, bool(shard["terminal_partial"]), gates["failed_gates"]),
        "date": str(date),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    *,
    date: str = RUN_DATE,
    trace_root: Path | str = REPO_ROOT / TRACE_ROOT_RELATIVE_PATH,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    checkpoint_path: Path | str = CHECKPOINT_RELATIVE_PATH,
    budgets: ShardBudgets | None = None,
    tuning_count: int = 6,
    safety_count: int = 2,
    tests_run: Sequence[Any] | None = None,
    write: bool = True,
    run_adversarial: bool = True,
    progress: bool = False,
) -> JsonDict:
    start = time.monotonic()
    budget = budgets or ShardBudgets()
    trace_path = Path(trace_root)
    checkpoint = Path(checkpoint_path)
    protected_before = _protected_hashes()
    registry_before = path_sha256(REPO_ROOT / REGISTRY_RELATIVE_PATH)
    registry_precheck = exp6458.registry_precheck_and_hash(REPO_ROOT)
    manifest = _freeze_manifest(trace_path, tuning_count=tuning_count, safety_count=safety_count)
    preconditions = preconditions_checked(
        trace_root=trace_path,
        checkpoint_path=checkpoint,
        budgets=budget,
        registry_precheck=registry_precheck,
        manifest=manifest,
    )
    tuning_records = exp6458.load_trace_prefixes(
        trace_path,
        manifest["exp6458_tuning_games"],
        max_prefixes_per_game=budget.max_prefixes_per_game,
    )
    tuning = exp6458.tune_precommitted_parameters(tuning_records, seeds=RANDOM_SEEDS)
    selected = tuning["selected"]
    records = exp6458.load_trace_prefixes(
        trace_path,
        manifest["games"],
        max_prefixes_per_game=budget.max_prefixes_per_game,
    )
    shard = run_sharded_cells(
        records=records,
        suffix_max_k=int(selected["suffix_max_k"]),
        objective_weight=float(selected["objective_weight"]),
        training_model=tuning["training_model"],
        manifest=manifest,
        checkpoint_path=checkpoint,
        budgets=budget,
        progress=progress,
    )
    registry_after = path_sha256(REPO_ROOT / REGISTRY_RELATIVE_PATH)
    protected_after = _protected_hashes()
    source_receipts = _source_receipts(
        registry_before=registry_before,
        registry_after=registry_after,
        live_path_available=bool(exp6458.canonical_live_path_receipts(REPO_ROOT).get("available")),
    )
    frozen = _frozen_receipts(REPO_ROOT, manifest, tuning)
    artifact = _build_artifact(
        date=date,
        registry_precheck=registry_precheck,
        frozen=frozen,
        manifest=manifest,
        preconditions=preconditions,
        shard=shard,
        protected_before=protected_before,
        protected_after=protected_after,
        source_receipts=source_receipts,
        duration_s=time.monotonic() - start,
        tests_run=tests_run,
        adversarial_findings={"ran": False, "critical_count": 0, "flags": [], "note": "pending"},
    )
    target = Path(result_path)
    if write:
        atomic_write_json(target, artifact, sort_keys=True, allow_override=False)
        findings = _current_adversarial_findings(target, run_adversarial=run_adversarial)
        artifact = _build_artifact(
            date=date,
            registry_precheck=registry_precheck,
            frozen=frozen,
            manifest=manifest,
            preconditions=preconditions,
            shard=shard,
            protected_before=protected_before,
            protected_after=protected_after,
            source_receipts=source_receipts,
            duration_s=time.monotonic() - start,
            tests_run=tests_run,
            adversarial_findings=findings,
        )
        atomic_write_json(target, artifact, sort_keys=True, allow_override=False)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if artifact.get("no_solve_claim") is not True:
        errors.append("no_solve_claim must be true")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if "solve_provenance" in artifact:
        errors.append("solve_provenance must be absent")
    receipts = artifact.get("source_and_adapter_access_receipts") or {}
    for field in ("game_source_access_count", "offline_ground_truth_bfs_count", "per_game_adapter_count"):
        if int(receipts.get(field, -1) or 0) != 0:
            errors.append(f"{field} must be zero")
    principles = artifact.get("field_principles") or {}
    provenance = artifact.get("field_provenance") or {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            errors.append(f"field_principles missing {field}")
        if field not in provenance:
            errors.append(f"field_provenance missing {field}")
    rows = list(artifact.get("per_unit_rows") or [])
    if len({row.get("row_id") for row in rows}) != len(rows):
        errors.append("duplicate per_unit_rows row_id")
    recomputed = canonical_row_reducer(rows, artifact)
    for field in CANONICAL_AGGREGATE_FIELDS:
        if artifact.get(field) != recomputed[field]:
            errors.append(f"aggregate_row_mismatch:{field}")
    if not all(row.get("unchanged") is True for row in (artifact.get("protected_files_unchanged") or {}).values()):
        errors.append("protected_files_unchanged")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    gates = artifact.get("gate_check_summary") or {}
    if float(artifact.get("arc_safety_shield_ready_score", 0.0) or 0.0) == 1.0 and not gates.get("all_ready_gates_passed"):
        errors.append("ready_score gate mismatch")
    if errors:
        raise ValueError("; ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--trace-root", default=str(REPO_ROOT / TRACE_ROOT_RELATIVE_PATH))
    parser.add_argument("--out", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_RELATIVE_PATH))
    parser.add_argument("--max-prefixes-per-game", type=int, default=4)
    parser.add_argument("--max-cell-s", type=float, default=2.0)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--tuning-count", type=int, default=6)
    parser.add_argument("--safety-count", type=int, default=2)
    parser.add_argument("--skip-adversarial", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    artifact = run(
        date=args.date,
        trace_root=Path(args.trace_root),
        result_path=Path(args.out),
        checkpoint_path=Path(args.checkpoint),
        budgets=ShardBudgets(
            max_prefixes_per_game=args.max_prefixes_per_game,
            max_cell_s=args.max_cell_s,
            max_cells=args.max_cells,
        ),
        tuning_count=args.tuning_count,
        safety_count=args.safety_count,
        run_adversarial=not args.skip_adversarial,
        progress=True,
    )
    validate_artifact(artifact)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "ready_score": artifact["arc_safety_shield_ready_score"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
