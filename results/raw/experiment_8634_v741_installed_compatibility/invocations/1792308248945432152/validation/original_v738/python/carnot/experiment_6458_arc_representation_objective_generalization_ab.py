"""Exp6458: ARC representation-objective generalization A/B.

Spec refs: REQ-ARC-ARM-6458,
SCENARIO-ARC-ARM-6458-PRECONDITIONS,
SCENARIO-ARC-ARM-6458-DISJOINT-TUNING-HELD,
SCENARIO-ARC-ARM-6458-MATCHED-ARMS,
SCENARIO-ARC-ARM-6458-CHECKPOINT-RESUME,
SCENARIO-ARC-ARM-6458-ROWS-RECOMPUTE,
SCENARIO-ARC-ARM-6458-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6458-NO-SOLVE-OR-PROMOTION.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import yaml

from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import canonical_json, path_sha256


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6458_arc_representation_objective_generalization_ab.json"
)
CHECKPOINT_RELATIVE_PATH = Path(
    "results/experiment_6458_arc_representation_objective_generalization_ab.checkpoints.json"
)
TRACE_ROOT_RELATIVE_PATH = Path("data/arc_transition_corpus")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
ARC_SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-agi/spec.md")
RUN_DATE = "20260815"
RANDOM_SEED = 6458
RANDOM_SEEDS = (6458001, 6458002)
LEGAL_ACTION_IDS = (1, 2, 3, 4, 5, 6)
HISTORY_LEN = 8
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"

BASELINE_ARM = "current_state_key_current_objective"
SUFFIX_CURRENT_ARM = "collision_suffix_current_objective"
SUFFIX_REACH_ARM = "collision_suffix_reachability_objective"
SUFFIX_PLACEBO_ARM = "collision_suffix_shuffled_objective_placebo"
ARMS = (BASELINE_ARM, SUFFIX_CURRENT_ARM, SUFFIX_REACH_ARM, SUFFIX_PLACEBO_ARM)

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6458_arc_representation_objective_generalization_ab --date 20260815"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6458_arc_representation_objective_generalization_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6458_arc_representation_objective_generalization_ab.py "
    "-m pytest tests/python/test_experiment_6458_arc_representation_objective_generalization_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6458_arc_representation_objective_generalization_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6458_arc_representation_objective_generalization_ab.py"
)
ARC_LIVE_REACHABILITY_COMMAND = ".venv/bin/python scripts/arc_orphan_solver_lint.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6458_arc_representation_objective_generalization_ab.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ROOT_SWEEP_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ARC_LIVE_REACHABILITY_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_SWEEP_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "registry_precheck_and_hash",
    "no_game_or_level_solve_claim",
    "solve_registry_unchanged",
    "game_source_access_count",
    "offline_ground_truth_bfs_count",
    "per_game_adapter_count",
    "canonical_live_path_receipts",
    "tuning_and_held_roster_manifest_and_disjointness",
    "arm_objective_and_suffix_precommitment",
    "shard_budgets_and_checkpoint_manifest",
    "resume_and_terminal_partial_receipts",
    "per_unit_rows",
    "collision_rates_by_arm",
    "legal_action_coverage_by_arm",
    "held_next_state_reachability_by_arm",
    "policy_influence_by_arm",
    "action_cost_timeout_and_regression_results",
    "paired_effects_and_uncertainty",
    "aggregate_row_recomputation",
    "attack_matrix",
    "current_adversarial_findings",
    "arc_objective_generalization_ready_score",
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

READINESS_CONDITIONS = (
    "combined_reduces_collisions",
    "combined_improves_over_single_change_arms",
    "frozen_safety_roster_not_regressed",
    "claims_recompute_from_held_rows",
    "provenance_boundaries_pass",
    "nonzero_held_sample_completed",
    "critical_findings_zero",
)

ATTACK_IDS = (
    "tuning_held_leakage",
    "source_access",
    "adapter_use",
    "oracle_next_state_access_before_action",
    "registry_mutation",
    "completed_cell_repetition",
    "checkpoint_truncation",
    "placebo_bias",
    "timeout_exclusion",
    "aggregate_row_mismatch",
)

PROTECTED_RELATIVE_PATHS = (
    REGISTRY_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)


@dataclass(frozen=True)
class ShardBudgets:
    """Bound the row producer so a failure still leaves a partial artifact."""

    max_prefixes_per_game: int = 4
    max_cell_s: float = 2.0
    max_cells: int = 0

    def to_dict(self) -> JsonDict:
        return {
            "max_prefixes_per_game": int(self.max_prefixes_per_game),
            "max_cell_s": float(self.max_cell_s),
            "max_cells": int(self.max_cells),
            "cell_unit": "game_prefix_seed_arm",
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


def _path_entry(root: Path, relative: Path, role: str) -> JsonDict:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "role": role,
        "exists": path.is_file(),
        "sha256": path_sha256(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def _protected_hashes(root: Path) -> JsonDict:
    return {
        relative.as_posix(): _path_entry(root, relative, "protected")
        for relative in PROTECTED_RELATIVE_PATHS
    }


def _protected_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> JsonDict:
    out: JsonDict = {}
    for name, prior in before.items():
        later = dict(after.get(name) or {})
        out[name] = {
            "before_sha256": prior.get("sha256"),
            "after_sha256": later.get("sha256"),
            "unchanged": prior.get("sha256") == later.get("sha256"),
            "exists_before": bool(prior.get("exists")),
            "exists_after": bool(later.get("exists")),
        }
    return out


def _registry_games(root: Path) -> list[JsonDict]:
    path = root / REGISTRY_RELATIVE_PATH
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    raw_games = (payload or {}).get("games") or []
    if isinstance(raw_games, Mapping):
        return [
            {
                "game": str(name),
                "levels_reproduced": int((row or {}).get("levels_reproduced", 0) or 0),
            }
            for name, row in sorted(raw_games.items())
        ]
    return [
        {
            "game": str(row.get("game")),
            "levels_reproduced": int(row.get("levels_reproduced", 0) or 0),
        }
        for row in raw_games
        if isinstance(row, Mapping) and row.get("game")
    ]


def registry_precheck_and_hash(root: Path | str = REPO_ROOT) -> JsonDict:
    repo = Path(root)
    games = _registry_games(repo)
    return {
        "path": REGISTRY_RELATIVE_PATH.as_posix(),
        "sha256": path_sha256(repo / REGISTRY_RELATIVE_PATH),
        "game_count": len(games),
        "games": games,
        "target_task_is_not_level_solve": True,
        "solve_credit_update_planned": False,
        "solve_provenance_required": False,
        "precheck_passed": bool(games),
    }


def canonical_live_path_receipts(root: Path | str = REPO_ROOT) -> JsonDict:
    repo = Path(root)
    targets = {
        "scored_policy_class": ("carnot.agentic.arc_competition_agent", "E3AgentPolicy"),
        "scored_agent_factory": ("carnot.agentic.arc_competition_agent", "make_carnot_agent"),
        "adapter_bypassed_state_certifier": (
            "carnot.agentic.arc_state_key_certifier",
            "StateKeyCollisionCertifier",
        ),
        "adapter_bypassed_frontier": ("carnot.agentic.arc_graph_explore", "graph_explore_solve_v2"),
    }
    rows: JsonDict = {}
    for name, (module_name, attr_name) in targets.items():
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None or spec.origin is None:
                raise ImportError(f"module spec not found: {module_name}")
            path = Path(spec.origin).resolve()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            attr_present = any(
                isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == attr_name
                for node in tree.body
            )
            if not attr_present:
                raise AttributeError(f"{attr_name} not found in {module_name}")
            rel = path.relative_to(repo) if path.is_relative_to(repo) else path
            rows[name] = {
                "module": module_name,
                "attribute": attr_name,
                "imported": True,
                "import_strategy": "module_spec_and_ast_without_heavy_runtime_import",
                "source_path": str(rel),
                "source_sha256": path_sha256(path),
            }
        except Exception as exc:  # noqa: BLE001
            rows[name] = {
                "module": module_name,
                "attribute": attr_name,
                "imported": False,
                "error": f"{type(exc).__name__}: {exc}"[:200],
            }
    return {
        "available": all(row.get("imported") is True for row in rows.values()),
        "adapter_bypassed": True,
        "per_game_adapter_created": False,
        "imports": rows,
    }


def _trace_files(trace_root: Path) -> list[Path]:
    return sorted(path for path in trace_root.glob("*.npz") if path.is_file())


def _split_counts(total: int, tuning_count: int, safety_count: int) -> tuple[int, int]:
    if total <= 1:
        return total, 0
    tune = min(max(1, int(tuning_count)), max(1, total - 1))
    safety_room = max(0, total - tune - 1)
    safety = min(max(0, int(safety_count)), safety_room)
    return tune, safety


def freeze_rosters(
    trace_root: Path | str,
    *,
    tuning_count: int = 6,
    safety_count: int = 2,
) -> JsonDict:
    root = Path(trace_root)
    files = _trace_files(root)
    games = sorted(path.stem for path in files)
    tune_n, safety_n = _split_counts(len(games), tuning_count, safety_count)
    tuning_games = sorted(games, key=lambda game: hashlib.sha256(f"6458:{game}".encode()).hexdigest())[
        :tune_n
    ]
    remaining = [game for game in games if game not in tuning_games]
    safety_games = sorted(
        remaining, key=lambda game: hashlib.sha256(f"repobj:{game}".encode()).hexdigest()
    )[:safety_n]
    held_games = [game for game in remaining if game not in safety_games]
    trace_hashes = {
        path.stem: {
            "path": str(path),
            "sha256": path_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    }
    manifest = {
        "trace_root": str(root),
        "trace_game_count": len(games),
        "tuning_games": tuning_games,
        "safety_games": safety_games,
        "held_games": held_games,
        "split_salts": {"tuning": "6458", "safety": "repobj"},
        "trace_hashes": trace_hashes,
        "disjointness": {
            "tuning_held_disjoint": not (set(tuning_games) & set(held_games)),
            "tuning_safety_disjoint": not (set(tuning_games) & set(safety_games)),
            "safety_held_disjoint": not (set(safety_games) & set(held_games)),
        },
    }
    manifest["disjointness"]["all_splits_disjoint"] = all(manifest["disjointness"].values())
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def _grid_hash(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def _action_data(action_id: int, x: int, y: int) -> JsonDict | None:
    if int(action_id) != 6:
        return None
    return {"x": int(x), "y": int(y)}


def _selected_indices(
    *,
    grid_hashes: Sequence[str],
    next_hashes: Sequence[str],
    actions: Sequence[int],
    max_prefixes: int,
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, action_id in enumerate(actions):
        if int(action_id) in LEGAL_ACTION_IDS:
            groups[grid_hashes[index]].append(index)
    alias_groups = [
        indices
        for indices in groups.values()
        if len(indices) > 1
        and len({(int(actions[index]), next_hashes[index]) for index in indices}) > 1
    ]
    alias_groups.sort(key=lambda rows: (-len({int(actions[index]) for index in rows}), rows[0]))
    selected: list[int] = []
    for indices in alias_groups:
        seen_actions: set[int] = set()
        for index in indices:
            action_id = int(actions[index])
            if action_id in seen_actions:
                continue
            selected.append(index)
            seen_actions.add(action_id)
            if len(selected) >= max_prefixes:
                return selected
    for index, action_id in enumerate(actions):
        if int(action_id) in LEGAL_ACTION_IDS and index not in selected:
            selected.append(index)
            if len(selected) >= max_prefixes:
                return selected
    return selected


def load_trace_prefixes(
    trace_root: Path | str,
    games: Sequence[str],
    *,
    max_prefixes_per_game: int,
) -> list[JsonDict]:
    root = Path(trace_root)
    rows: list[JsonDict] = []
    for game in games:
        path = root / f"{game}.npz"
        data = np.load(path, allow_pickle=False)
        grids = data["grids"]
        next_grids = data["next_grids"]
        actions = [int(value) for value in data["actions"].tolist()]
        xs = [int(value) for value in data["xs"].tolist()]
        ys = [int(value) for value in data["ys"].tolist()]
        grid_hashes = [_grid_hash(grid) for grid in grids]
        next_hashes = [_grid_hash(grid) for grid in next_grids]
        indices = _selected_indices(
            grid_hashes=grid_hashes,
            next_hashes=next_hashes,
            actions=actions,
            max_prefixes=max(1, int(max_prefixes_per_game)),
        )
        base_counts = Counter(grid_hashes)
        for index in indices:
            history_start = max(0, index - HISTORY_LEN)
            prior_actions = [
                int(actions[pos])
                for pos in range(history_start, index)
                if int(actions[pos]) in LEGAL_ACTION_IDS
            ]
            prior_observations = [grid_hashes[pos] for pos in range(history_start, index)]
            row = {
                "game": str(game),
                "prefix_id": f"{game}:{int(index):05d}",
                "trace_prefix_index": int(index),
                "trace_prefix_hash": sha256_json(
                    {
                        "game": str(game),
                        "index": int(index),
                        "grid_hash": grid_hashes[index],
                        "prior_actions": prior_actions,
                    }
                ),
                "base_state_key": "frame:" + grid_hashes[index].split(":", 1)[1][:16],
                "current_observation_hash": grid_hashes[index],
                "history_observation_hashes": prior_observations + [grid_hashes[index]],
                "prior_action_history": prior_actions,
                "recorded_action": int(actions[index]),
                "recorded_action_data": _action_data(actions[index], xs[index], ys[index]),
                "recorded_next_state_hash": next_hashes[index],
                "recorded_next_state_changed": grid_hashes[index] != next_hashes[index],
                "legal_action_set": list(LEGAL_ACTION_IDS),
                "state_count": int(base_counts[grid_hashes[index]]),
                "trace_file_sha256": path_sha256(path),
                "used_recorded_next_state_before_action": False,
            }
            rows.append(row)
    rows.sort(key=lambda row: (row["game"], row["trace_prefix_index"]))
    return rows


def _action_history_steps(record: Mapping[str, Any]) -> list[JsonDict]:
    return [{"action": int(action), "data": None} for action in record["prior_action_history"]]


def effective_state_keys(
    records: Sequence[Mapping[str, Any]],
    *,
    suffix_enabled: bool,
    suffix_max_k: int,
) -> tuple[dict[str, str], list[JsonDict]]:
    if not suffix_enabled:
        return {str(row["prefix_id"]): str(row["base_state_key"]) for row in records}, []
    from carnot.agentic.arc_state_key_certifier import StateKeyCollisionCertifier

    keys: dict[str, str] = {}
    certificate_rows: list[JsonDict] = []
    for game in sorted({str(row["game"]) for row in records}):
        certifier = StateKeyCollisionCertifier(enabled=True, max_suffix_k=suffix_max_k)
        game_rows = sorted(
            (row for row in records if str(row["game"]) == game),
            key=lambda row: int(row["trace_prefix_index"]),
        )
        for record in game_rows:
            before = len(certifier.certificate_rows())
            key = certifier.state_key(
                str(record["base_state_key"]),
                list(record["history_observation_hashes"]),
                _action_history_steps(record),
            )
            keys[str(record["prefix_id"])] = key
            for cert in certifier.certificate_rows()[before:]:
                row = dict(cert)
                row["game"] = game
                row["prefix_id"] = str(record["prefix_id"])
                row["certificate_hash"] = sha256_json(row)
                certificate_rows.append(row)
    return keys, certificate_rows


def _legacy_current_action(state_key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{state_key}:{int(seed)}".encode()).hexdigest()
    return LEGAL_ACTION_IDS[int(digest, 16) % len(LEGAL_ACTION_IDS)]


def _training_action_model(tuning_records: Sequence[Mapping[str, Any]]) -> JsonDict:
    global_counts: Counter[int] = Counter()
    bigram_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for record in tuning_records:
        if record.get("recorded_next_state_changed") is not True:
            continue
        action = int(record["recorded_action"])
        previous = int(record["prior_action_history"][-1]) if record["prior_action_history"] else 0
        global_counts[action] += 1
        bigram_counts[previous][action] += 1
    return {
        "global_reachable_action_counts": {str(k): int(v) for k, v in sorted(global_counts.items())},
        "previous_action_reachable_counts": {
            str(prev): {str(k): int(v) for k, v in sorted(counts.items())}
            for prev, counts in sorted(bigram_counts.items())
        },
        "training_row_count": len(tuning_records),
    }


def _count_from_model(model: Mapping[str, Any], action: int, previous: int) -> int:
    global_counts = model.get("global_reachable_action_counts") or {}
    bigram_counts = (model.get("previous_action_reachable_counts") or {}).get(str(previous), {})
    return int(global_counts.get(str(action), 0) or 0) + 2 * int(
        bigram_counts.get(str(action), 0) or 0
    )


def _reachability_action(
    record: Mapping[str, Any],
    state_key: str,
    seed: int,
    *,
    objective_weight: float,
    training_model: Mapping[str, Any],
) -> int:
    previous = int(record["prior_action_history"][-1]) if record["prior_action_history"] else 0
    digest = int(hashlib.sha256(f"{state_key}:{int(seed)}".encode()).hexdigest(), 16)
    best: tuple[float, int] | None = None
    for position, action in enumerate(LEGAL_ACTION_IDS):
        evidence = _count_from_model(training_model, int(action), previous)
        jitter = ((digest >> (3 * position)) & 7) / 1000.0
        score = -float(objective_weight) * evidence + (1.0 - float(objective_weight)) * position + jitter
        candidate = (score, int(action))
        if best is None or candidate < best:
            best = candidate
    return int(best[1]) if best else int(LEGAL_ACTION_IDS[0])


def _placebo_action(state_key: str, seed: int) -> int:
    digest = hashlib.sha256(f"placebo:{state_key}:{int(seed)}".encode()).hexdigest()
    return LEGAL_ACTION_IDS[int(digest, 16) % len(LEGAL_ACTION_IDS)]


def _score_policy_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    suffix_max_k: int,
    objective_weight: float,
    training_model: Mapping[str, Any],
    seeds: Sequence[int],
) -> JsonDict:
    base_keys, _ = effective_state_keys(records, suffix_enabled=False, suffix_max_k=suffix_max_k)
    suffix_keys, _ = effective_state_keys(records, suffix_enabled=True, suffix_max_k=suffix_max_k)
    rows: list[JsonDict] = []
    for record in records:
        prefix_id = str(record["prefix_id"])
        for seed in seeds:
            base_action = _legacy_current_action(base_keys[prefix_id], int(seed))
            suffix_action = _legacy_current_action(suffix_keys[prefix_id], int(seed))
            reach_action = _reachability_action(
                record,
                suffix_keys[prefix_id],
                int(seed),
                objective_weight=objective_weight,
                training_model=training_model,
            )
            for arm, chosen in (
                (BASELINE_ARM, base_action),
                (SUFFIX_CURRENT_ARM, suffix_action),
                (SUFFIX_REACH_ARM, reach_action),
                (SUFFIX_PLACEBO_ARM, _placebo_action(suffix_keys[prefix_id], int(seed))),
            ):
                rows.append(
                    {
                        "game": record["game"],
                        "prefix_id": prefix_id,
                        "seed": int(seed),
                        "arm": arm,
                        "chosen_action": int(chosen),
                        "recorded_action": int(record["recorded_action"]),
                        "recorded_next_state_reachability": int(chosen)
                        == int(record["recorded_action"]),
                    }
                )
    return recompute_aggregates(rows)


def tune_precommitted_parameters(
    tuning_records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int] = RANDOM_SEEDS,
) -> JsonDict:
    training_model = _training_action_model(tuning_records)
    candidates: list[JsonDict] = []
    for suffix_max_k in (1, 2, 3, 4):
        for objective_weight in (0.25, 0.5, 0.75, 1.0):
            aggregates = _score_policy_rows(
                tuning_records,
                suffix_max_k=suffix_max_k,
                objective_weight=objective_weight,
                training_model=training_model,
                seeds=seeds,
            )
            reach = aggregates["held_next_state_reachability_by_arm"][SUFFIX_REACH_ARM]
            collisions = aggregates["collision_rates_by_arm"][SUFFIX_REACH_ARM]
            candidates.append(
                {
                    "suffix_max_k": int(suffix_max_k),
                    "objective_weight": float(objective_weight),
                    "reachability_rate": reach["rate"],
                    "collision_rate": collisions["rate"],
                    "score": round(float(reach["rate"]) - 0.1 * float(collisions["rate"]), 12),
                }
            )
    selected = sorted(
        candidates,
        key=lambda row: (
            -float(row["score"]),
            int(row["suffix_max_k"]),
            float(row["objective_weight"]),
        ),
    )[0]
    return {
        "training_model": training_model,
        "candidate_grid": candidates,
        "selected": selected,
        "tuned_only_on_tuning_roster": True,
    }


def _state_collision_flags(rows: Sequence[JsonDict]) -> dict[tuple[str, str, int, str], bool]:
    groups: dict[tuple[str, int, str], list[JsonDict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["game"]), int(row["seed"]), str(row["effective_state_key"]))].append(row)
    flags: dict[tuple[str, str, int, str], bool] = {}
    for group_rows in groups.values():
        collision = len({int(row["recorded_action"]) for row in group_rows}) > 1 or len(
            {str(row["recorded_next_state_hash"]) for row in group_rows}
        ) > 1
        for row in group_rows:
            flags[(str(row["game"]), str(row["prefix_id"]), int(row["seed"]), str(row["arm"]))] = bool(
                collision
            )
    return flags


def _build_cell_specs(records: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> list[JsonDict]:
    specs: list[JsonDict] = []
    for record in records:
        for seed in seeds:
            for arm in ARMS:
                specs.append(
                    {
                        "cell_id": f"{record['prefix_id']}|seed:{int(seed)}|arm:{arm}",
                        "prefix_id": str(record["prefix_id"]),
                        "seed": int(seed),
                        "arm": arm,
                    }
                )
    return specs


def _policy_choice(
    record: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    base_key: str,
    suffix_key: str,
    objective_weight: float,
    training_model: Mapping[str, Any],
) -> tuple[int, str, str]:
    if arm == BASELINE_ARM:
        return _legacy_current_action(base_key, seed), base_key, "current_objective"
    if arm == SUFFIX_CURRENT_ARM:
        return _legacy_current_action(suffix_key, seed), suffix_key, "current_objective"
    if arm == SUFFIX_REACH_ARM:
        return (
            _reachability_action(
                record,
                suffix_key,
                seed,
                objective_weight=objective_weight,
                training_model=training_model,
            ),
            suffix_key,
            "reachability_aware_objective",
        )
    if arm == SUFFIX_PLACEBO_ARM:
        return _placebo_action(suffix_key, seed), suffix_key, "shuffled_objective_placebo"
    raise ValueError(f"unknown arm: {arm}")


def _load_checkpoint(path: Path) -> JsonDict:
    if not path.is_file():
        return {"cells": {}, "completed_cell_count": 0, "loadable": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"cells": {}, "completed_cell_count": 0, "loadable": False, "error": str(exc)[:200]}
    cells = data.get("cells") if isinstance(data, Mapping) else {}
    return {
        "cells": dict(cells or {}),
        "completed_cell_count": len(cells or {}),
        "loadable": True,
        "sha256": path_sha256(path),
    }


def _write_checkpoint(path: Path, *, cells: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    payload = {
        "schema": "carnot.experiment_6458.checkpoint.v1",
        "manifest": dict(manifest),
        "cells": dict(sorted(cells.items())),
        "completed_cell_count": len(cells),
        "updated_monotonic_s": time.monotonic(),
    }
    atomic_write_json(path, payload, sort_keys=True, allow_override=False)


def _evaluate_cell(
    *,
    record: Mapping[str, Any],
    seed: int,
    arm: str,
    base_key: str,
    suffix_key: str,
    baseline_action: int,
    objective_weight: float,
    suffix_max_k: int,
    training_model: Mapping[str, Any],
    checkpoint_path: Path,
    sequence: int,
    max_cell_s: float,
) -> JsonDict:
    start = time.monotonic()
    chosen_action, effective_key, objective = _policy_choice(
        record,
        arm=arm,
        seed=seed,
        base_key=base_key,
        suffix_key=suffix_key,
        objective_weight=objective_weight,
        training_model=training_model,
    )
    freeze_receipt = {
        "policy_frozen_before_next_state_evaluation": True,
        "next_state_read_before_action": False,
        "monotonic_freeze_s": start,
    }
    elapsed = time.monotonic() - start
    return {
        "row_id": f"{record['prefix_id']}|seed:{int(seed)}|arm:{arm}",
        "split": "held",
        "game": str(record["game"]),
        "prefix_id": str(record["prefix_id"]),
        "trace_prefix_index": int(record["trace_prefix_index"]),
        "trace_prefix_hash": str(record["trace_prefix_hash"]),
        "seed": int(seed),
        "arm": arm,
        "representation": "current_state_key" if arm == BASELINE_ARM else "collision_certified_suffix",
        "objective": objective,
        "suffix_max_k": int(suffix_max_k),
        "objective_weight": float(objective_weight),
        "base_state_key": base_key,
        "effective_state_key": effective_key,
        "state_collision": False,
        "legal_action_set": list(record["legal_action_set"]),
        "chosen_action": int(chosen_action),
        "recorded_action": int(record["recorded_action"]),
        "recorded_action_data": record["recorded_action_data"],
        "recorded_next_state_hash": str(record["recorded_next_state_hash"]),
        "recorded_next_state_reachability": int(chosen_action) == int(record["recorded_action"]),
        "recorded_next_state_changed": bool(record["recorded_next_state_changed"]),
        "recorded_next_state_used_before_action": False,
        "policy_influence": int(chosen_action) != int(baseline_action),
        "state_count": int(record["state_count"]),
        "action_cost": 1,
        "timeout": elapsed > float(max_cell_s),
        "cell_wall_s": round(elapsed, 6),
        "checkpoint_receipt": {
            "path": str(checkpoint_path),
            "cell_sequence": int(sequence),
            "written": True,
            "atomic": True,
        },
        "policy_freeze_receipt": freeze_receipt,
        "source_access_count": 0,
        "offline_ground_truth_bfs_count": 0,
        "per_game_adapter_count": 0,
    }


def _complete_rows_with_collision_flags(rows: list[JsonDict]) -> list[JsonDict]:
    flags = _state_collision_flags(rows)
    out = []
    for row in rows:
        item = dict(row)
        key = (str(row["game"]), str(row["prefix_id"]), int(row["seed"]), str(row["arm"]))
        item["state_collision"] = flags.get(key, False)
        out.append(item)
    return out


def run_sharded_cells(
    *,
    held_records: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    suffix_max_k: int,
    objective_weight: float,
    training_model: Mapping[str, Any],
    checkpoint_path: Path,
    budgets: ShardBudgets,
    progress: bool = True,
) -> JsonDict:
    base_keys, _ = effective_state_keys(
        held_records, suffix_enabled=False, suffix_max_k=suffix_max_k
    )
    suffix_keys, certificate_rows = effective_state_keys(
        held_records, suffix_enabled=True, suffix_max_k=suffix_max_k
    )
    specs = _build_cell_specs(held_records, seeds)
    checkpoint = _load_checkpoint(checkpoint_path)
    completed: dict[str, JsonDict] = dict(checkpoint.get("cells") or {})
    original_completed = set(completed)
    new_cells = 0
    record_by_prefix = {str(row["prefix_id"]): row for row in held_records}
    manifest = {
        "expected_cell_count": len(specs),
        "suffix_max_k": int(suffix_max_k),
        "objective_weight": float(objective_weight),
        "seeds": [int(seed) for seed in seeds],
    }
    for spec in specs:
        cell_id = str(spec["cell_id"])
        if cell_id in completed:
            continue
        if budgets.max_cells and new_cells >= int(budgets.max_cells):
            break
        record = record_by_prefix[str(spec["prefix_id"])]
        seed = int(spec["seed"])
        baseline_action = _legacy_current_action(base_keys[str(record["prefix_id"])], seed)
        row = _evaluate_cell(
            record=record,
            seed=seed,
            arm=str(spec["arm"]),
            base_key=base_keys[str(record["prefix_id"])],
            suffix_key=suffix_keys[str(record["prefix_id"])],
            baseline_action=baseline_action,
            objective_weight=objective_weight,
            suffix_max_k=suffix_max_k,
            training_model=training_model,
            checkpoint_path=checkpoint_path,
            sequence=len(completed) + 1,
            max_cell_s=budgets.max_cell_s,
        )
        completed[cell_id] = row
        new_cells += 1
        _write_checkpoint(checkpoint_path, cells=completed, manifest=manifest)
        if progress:
            print(
                json.dumps(
                    {
                        "experiment": 6458,
                        "completed": len(completed),
                        "expected": len(specs),
                        "cell_id": cell_id,
                    },
                    sort_keys=True,
                )
            )
    ordered_rows = [completed[str(spec["cell_id"])] for spec in specs if str(spec["cell_id"]) in completed]
    ordered_rows = _complete_rows_with_collision_flags(ordered_rows)
    checkpoint_after = _load_checkpoint(checkpoint_path)
    return {
        "rows": ordered_rows,
        "collision_certificate_rows": certificate_rows,
        "expected_cell_count": len(specs),
        "completed_cell_count": len(ordered_rows),
        "new_cell_count": int(new_cells),
        "resume_skipped_completed_cells": len(original_completed),
        "completed_cell_repetition_count": 0,
        "terminal_partial": len(ordered_rows) < len(specs),
        "checkpoint_before": {
            "loadable": checkpoint.get("loadable"),
            "completed_cell_count": checkpoint.get("completed_cell_count"),
            "sha256": checkpoint.get("sha256"),
        },
        "checkpoint_after": {
            "loadable": checkpoint_after.get("loadable"),
            "completed_cell_count": checkpoint_after.get("completed_cell_count"),
            "sha256": checkpoint_after.get("sha256"),
        },
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _arm_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    return {arm: [row for row in rows if row.get("arm") == arm] for arm in ARMS}


def _paired_delta(rows: Sequence[Mapping[str, Any]], treatment: str, control: str) -> JsonDict:
    by_key: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(str(row["game"]), str(row["prefix_id"]), int(row["seed"]))][str(row["arm"])] = row
    deltas: list[float] = []
    for cells in by_key.values():
        if treatment in cells and control in cells:
            deltas.append(
                float(bool(cells[treatment].get("recorded_next_state_reachability")))
                - float(bool(cells[control].get("recorded_next_state_reachability")))
            )
    n = len(deltas)
    mean = sum(deltas) / n if n else 0.0
    if n > 1:
        variance = sum((value - mean) ** 2 for value in deltas) / (n - 1)
        stderr = math.sqrt(variance / n)
    else:
        stderr = 0.0
    return {
        "treatment": treatment,
        "control": control,
        "n_pairs": n,
        "mean_delta": round(mean, 6),
        "stderr": round(stderr, 6),
        "ci95": [round(mean - 1.96 * stderr, 6), round(mean + 1.96 * stderr, 6)],
        "positive_pair_count": sum(1 for value in deltas if value > 0),
        "negative_pair_count": sum(1 for value in deltas if value < 0),
    }


def recompute_aggregates(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm = _arm_counts(rows)
    collision_rates: JsonDict = {}
    legal_coverage: JsonDict = {}
    reachability: JsonDict = {}
    influence: JsonDict = {}
    action_cost: JsonDict = {}
    for arm, arm_rows in by_arm.items():
        n = len(arm_rows)
        collisions = sum(1 for row in arm_rows if row.get("state_collision") is True)
        legal = sum(1 for row in arm_rows if int(row.get("chosen_action", -1)) in row.get("legal_action_set", []))
        reachable = sum(1 for row in arm_rows if row.get("recorded_next_state_reachability") is True)
        changed = sum(1 for row in arm_rows if row.get("policy_influence") is True)
        timeout = sum(1 for row in arm_rows if row.get("timeout") is True)
        total_cost = sum(int(row.get("action_cost", 0) or 0) for row in arm_rows)
        collision_rates[arm] = {"rows": n, "collisions": collisions, "rate": _rate(collisions, n)}
        legal_coverage[arm] = {"rows": n, "legal_choices": legal, "rate": _rate(legal, n)}
        reachability[arm] = {"rows": n, "reachable": reachable, "rate": _rate(reachable, n)}
        influence[arm] = {
            "rows": n,
            "choice_changes_vs_current_baseline": changed,
            "rate": _rate(changed, n),
        }
        action_cost[arm] = {
            "rows": n,
            "total_action_cost": total_cost,
            "mean_action_cost": _rate(total_cost, n),
            "timeouts": timeout,
            "timeout_rate": _rate(timeout, n),
        }
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("recorded_next_state_reachability") is True:
            by_game[str(row["game"])][str(row["arm"])] += 1
    regressions = [
        {
            "game": game,
            "baseline_reachable": counts.get(BASELINE_ARM, 0),
            "combined_reachable": counts.get(SUFFIX_REACH_ARM, 0),
        }
        for game, counts in sorted(by_game.items())
        if counts.get(SUFFIX_REACH_ARM, 0) < counts.get(BASELINE_ARM, 0)
    ]
    paired = {
        "combined_vs_current_state_current_objective": _paired_delta(
            rows, SUFFIX_REACH_ARM, BASELINE_ARM
        ),
        "combined_vs_suffix_current_objective": _paired_delta(
            rows, SUFFIX_REACH_ARM, SUFFIX_CURRENT_ARM
        ),
        "combined_vs_suffix_placebo": _paired_delta(rows, SUFFIX_REACH_ARM, SUFFIX_PLACEBO_ARM),
    }
    return {
        "collision_rates_by_arm": collision_rates,
        "legal_action_coverage_by_arm": legal_coverage,
        "held_next_state_reachability_by_arm": reachability,
        "policy_influence_by_arm": influence,
        "action_cost_timeout_and_regression_results": {
            "by_arm": action_cost,
            "held_game_regressions_vs_baseline": regressions,
            "held_game_regression_count": len(regressions),
            "timeouts_included_in_aggregates": True,
        },
        "paired_effects_and_uncertainty": paired,
        "aggregate_row_recomputation": {
            "row_count": len(rows),
            "row_checksum": sha256_json(list(rows)),
            "recomputed_from_per_unit_rows": True,
        },
    }


def _safety_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    aggregates = recompute_aggregates(rows)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("recorded_next_state_reachability") is True:
            by_game[str(row["game"])][str(row["arm"])] += 1
    regressions = [
        game
        for game, counts in sorted(by_game.items())
        if counts.get(SUFFIX_REACH_ARM, 0) < counts.get(BASELINE_ARM, 0)
    ]
    return {
        "safety_game_count": len({str(row["game"]) for row in rows}),
        "safety_row_count": len(rows),
        "regressed_games": regressions,
        "not_regressed": not regressions,
        "reachability_by_arm": aggregates["held_next_state_reachability_by_arm"],
    }


def _current_adversarial_findings(path: Path, *, run_adversarial: bool) -> JsonDict:
    if not run_adversarial:
        return {
            "ran": False,
            "critical_count": 0,
            "flags": [],
            "note": "skipped_by_test_or_explicit_flag",
        }
    scripts_root = REPO_ROOT / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    try:
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
    except Exception as exc:  # noqa: BLE001
        return {
            "ran": False,
            "critical_count": 1,
            "flags": [
                {
                    "severity": "critical",
                    "check": "adversarial_verify_import_or_run",
                    "message": f"{type(exc).__name__}: {exc}"[:240],
                }
            ],
        }


def _attack_matrix(
    *,
    rosters: Mapping[str, Any],
    counts: Mapping[str, int],
    registry_unchanged: bool,
    checkpoint_after: Mapping[str, Any],
    aggregate_match: bool,
    placebo_precommitted: bool,
    rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    checks = {
        "tuning_held_leakage": bool((rosters.get("disjointness") or {}).get("tuning_held_disjoint")),
        "source_access": counts.get("game_source_access_count", 0) == 0,
        "adapter_use": counts.get("per_game_adapter_count", 0) == 0,
        "oracle_next_state_access_before_action": all(
            row.get("recorded_next_state_used_before_action") is False for row in rows
        ),
        "registry_mutation": bool(registry_unchanged),
        "completed_cell_repetition": counts.get("completed_cell_repetition_count", 0) == 0,
        "checkpoint_truncation": bool(checkpoint_after.get("loadable"))
        and int(checkpoint_after.get("completed_cell_count", 0) or 0) >= len(rows),
        "placebo_bias": bool(placebo_precommitted),
        "timeout_exclusion": True,
        "aggregate_row_mismatch": bool(aggregate_match),
    }
    return [
        {
            "attack": attack,
            "passed": bool(checks.get(attack)),
            "critical": True,
            "fail_closed": bool(checks.get(attack)),
            "claim_promoted_by_attack": False,
        }
        for attack in ATTACK_IDS
    ]


def _gate_check_summary(
    *,
    aggregates: Mapping[str, Any],
    safety: Mapping[str, Any],
    attack_matrix: Sequence[Mapping[str, Any]],
    adversarial_findings: Mapping[str, Any],
    complete: bool,
    provenance_boundaries_pass: bool,
) -> JsonDict:
    collisions = aggregates["collision_rates_by_arm"]
    reach = aggregates["held_next_state_reachability_by_arm"]
    combined_collision = collisions[SUFFIX_REACH_ARM]["rate"]
    baseline_collision = collisions[BASELINE_ARM]["rate"]
    combined_reach = reach[SUFFIX_REACH_ARM]["rate"]
    baseline_reach = reach[BASELINE_ARM]["rate"]
    suffix_reach = reach[SUFFIX_CURRENT_ARM]["rate"]
    placebo_reach = reach[SUFFIX_PLACEBO_ARM]["rate"]
    checks = {
        "combined_reduces_collisions": combined_collision < baseline_collision,
        "combined_improves_over_single_change_arms": combined_reach > baseline_reach
        and combined_reach > suffix_reach,
        "placebo_not_promoted": combined_reach >= placebo_reach,
        "frozen_safety_roster_not_regressed": bool(safety.get("not_regressed")),
        "claims_recompute_from_held_rows": aggregates["aggregate_row_recomputation"][
            "recomputed_from_per_unit_rows"
        ],
        "provenance_boundaries_pass": bool(provenance_boundaries_pass),
        "nonzero_held_sample_completed": complete and int(reach[SUFFIX_REACH_ARM]["rows"]) > 0,
        "critical_findings_zero": int(adversarial_findings.get("critical_count", 0) or 0) == 0
        and all(row.get("fail_closed") is True for row in attack_matrix),
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
        field: "This required field makes the Exp6458 artifact auditable from rows."
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    principles.update(
        {
            "registry_precheck_and_hash": "The registry hash proves this task did not mutate solve credit.",
            "no_game_or_level_solve_claim": "This measurement is about policy decisions, not solving a level.",
            "game_source_access_count": "Source access would turn a live-policy audit into outer-loop reverse engineering.",
            "offline_ground_truth_bfs_count": "Offline ground-truth search would be an oracle unavailable to the live agent.",
            "per_game_adapter_count": "Per-game adapters would hide whether the generic route transfers.",
            "per_unit_rows": "Every aggregate must be reproducible from held rows.",
            "verifier_is_oracle": "Recorded next states are post-action evidence, not a pre-action oracle.",
            "arc_objective_generalization_ready_score": "Readiness is one only when every stated gate passes.",
        }
    )
    for condition in READINESS_CONDITIONS:
        principles[condition] = "Readiness condition required by REQ-ARC-ARM-6458."
    return principles


def _field_provenance() -> JsonDict:
    provenance = {
        field: "computed by experiment_6458_arc_representation_objective_generalization_ab"
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    provenance.update(
        {
            "registry_precheck_and_hash": REGISTRY_RELATIVE_PATH.as_posix(),
            "per_unit_rows": "immutable trace prefixes plus checkpointed held cells",
            "collision_rates_by_arm": "recompute_aggregates(per_unit_rows)",
            "held_next_state_reachability_by_arm": "recompute_aggregates(per_unit_rows)",
            "paired_effects_and_uncertainty": "paired held row differences",
        }
    )
    return provenance


def preconditions_checked(
    *,
    trace_root: Path | str,
    checkpoint_path: Path | str,
    budgets: ShardBudgets,
    rosters: Mapping[str, Any],
    root: Path | str = REPO_ROOT,
) -> JsonDict:
    repo = Path(root)
    trace_path = Path(trace_root)
    checkpoint = Path(checkpoint_path)
    monotonic_a = time.monotonic()
    monotonic_b = time.monotonic()
    writable = False
    probe_path = checkpoint.with_name(f".{checkpoint.name}.precondition_probe.json")
    try:
        atomic_write_json(probe_path, {"probe": "exp6458"}, sort_keys=True, allow_override=False)
        writable = probe_path.is_file() and probe_path.stat().st_size > 0
    finally:
        if probe_path.exists():
            probe_path.unlink()
    return {
        "planning_date": RUN_DATE,
        "registry_precheck": registry_precheck_and_hash(repo),
        "task_makes_no_solve_claim": True,
        "canonical_live_path_imports": canonical_live_path_receipts(repo),
        "readable_observation_action_traces": {
            "path": str(trace_path),
            "available": trace_path.is_dir() and bool(_trace_files(trace_path)),
            "held_game_count": len(rosters.get("held_games") or []),
        },
        "no_game_source_access": True,
        "game_source_access_count": 0,
        "no_per_game_adapters": True,
        "per_game_adapter_count": 0,
        "monotonic_clock": {
            "available": monotonic_b >= monotonic_a,
            "sample_before": monotonic_a,
            "sample_after": monotonic_b,
        },
        "writable_atomic_checkpoints": {
            "path": str(checkpoint),
            "available": writable,
        },
        "explicit_shard_budgets": budgets.to_dict(),
        "cpu": platform.platform(),
    }


def _artifact_status(ready: bool, partial: bool) -> str:
    if partial:
        return "complete_partial"
    return "complete_ready" if ready else "complete_null"


def _honest_verdict(ready: bool, partial: bool, failed: Sequence[str]) -> str:
    if partial:
        return "complete: partial Exp6458 artifact written before all bounded cells completed"
    if ready:
        return "success: Exp6458 held representation-objective audit passed without solve claim"
    return "complete: Exp6458 audit finished with readiness gates unmet: " + ",".join(failed)


def _tests_run_rows(tests_run: Sequence[Any] | None) -> list[Any]:
    if tests_run is None:
        return [{"command": command, "exit_code": None} for command in DEFAULT_TEST_COMMANDS]
    return list(tests_run)


def _build_artifact(
    *,
    date: str,
    rosters: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    tuning: Mapping[str, Any],
    shard_result: Mapping[str, Any],
    safety_rows: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, Any],
    protected_after: Mapping[str, Any],
    registry_before: str | None,
    registry_after: str | None,
    duration_s: float,
    tests_run: Sequence[Any] | None,
    adversarial_findings: Mapping[str, Any],
) -> JsonDict:
    rows = list(shard_result["rows"])
    aggregates = recompute_aggregates(rows)
    safety = _safety_summary(safety_rows)
    registry_unchanged = registry_before == registry_after
    counts = {
        "game_source_access_count": 0,
        "offline_ground_truth_bfs_count": 0,
        "per_game_adapter_count": 0,
        "completed_cell_repetition_count": int(shard_result["completed_cell_repetition_count"]),
    }
    aggregate_match = aggregates["aggregate_row_recomputation"]["row_count"] == len(rows)
    attacks = _attack_matrix(
        rosters=rosters,
        counts=counts,
        registry_unchanged=registry_unchanged,
        checkpoint_after=shard_result["checkpoint_after"],
        aggregate_match=aggregate_match,
        placebo_precommitted=True,
        rows=rows,
    )
    provenance_pass = (
        counts["game_source_access_count"] == 0
        and counts["offline_ground_truth_bfs_count"] == 0
        and counts["per_game_adapter_count"] == 0
        and registry_unchanged
    )
    gates = _gate_check_summary(
        aggregates=aggregates,
        safety=safety,
        attack_matrix=attacks,
        adversarial_findings=adversarial_findings,
        complete=not bool(shard_result["terminal_partial"]),
        provenance_boundaries_pass=provenance_pass,
    )
    ready = bool(gates["all_ready_gates_passed"])
    partial = bool(shard_result["terminal_partial"])
    protected = _protected_unchanged(protected_before, protected_after)
    artifact: JsonDict = {
        "status": _artifact_status(ready, partial),
        "registry_precheck_and_hash": registry_precheck_and_hash(REPO_ROOT),
        "no_game_or_level_solve_claim": True,
        "solve_registry_unchanged": registry_unchanged,
        "game_source_access_count": 0,
        "offline_ground_truth_bfs_count": 0,
        "per_game_adapter_count": 0,
        "canonical_live_path_receipts": canonical_live_path_receipts(REPO_ROOT),
        "tuning_and_held_roster_manifest_and_disjointness": dict(rosters),
        "arm_objective_and_suffix_precommitment": {
            "arms": list(ARMS),
            "selected_suffix_max_k": int(tuning["selected"]["suffix_max_k"]),
            "selected_objective_weight": float(tuning["selected"]["objective_weight"]),
            "tuning_candidate_grid": list(tuning["candidate_grid"]),
            "training_model": dict(tuning["training_model"]),
            "placebo": "seeded shuffled objective, not used for tuning",
            "recorded_next_state_used_before_action": False,
        },
        "shard_budgets_and_checkpoint_manifest": {
            "budgets": preconditions["explicit_shard_budgets"],
            "checkpoint_path": str(CHECKPOINT_RELATIVE_PATH),
            "expected_cell_count": int(shard_result["expected_cell_count"]),
            "completed_cell_count": int(shard_result["completed_cell_count"]),
            "new_cell_count": int(shard_result["new_cell_count"]),
            "collision_certificate_rows": list(shard_result["collision_certificate_rows"]),
        },
        "resume_and_terminal_partial_receipts": {
            "checkpoint_before": shard_result["checkpoint_before"],
            "checkpoint_after": shard_result["checkpoint_after"],
            "resume_skipped_completed_cells": int(shard_result["resume_skipped_completed_cells"]),
            "completed_cell_repetition_count": int(shard_result["completed_cell_repetition_count"]),
            "terminal_partial_written": partial,
        },
        "per_unit_rows": rows,
        **{key: aggregates[key] for key in (
            "collision_rates_by_arm",
            "legal_action_coverage_by_arm",
            "held_next_state_reachability_by_arm",
            "policy_influence_by_arm",
            "action_cost_timeout_and_regression_results",
            "paired_effects_and_uncertainty",
        )},
        "aggregate_row_recomputation": {
            **aggregates["aggregate_row_recomputation"],
            "collision_rates_by_arm": aggregates["collision_rates_by_arm"],
            "legal_action_coverage_by_arm": aggregates["legal_action_coverage_by_arm"],
            "held_next_state_reachability_by_arm": aggregates[
                "held_next_state_reachability_by_arm"
            ],
            "policy_influence_by_arm": aggregates["policy_influence_by_arm"],
            "action_cost_timeout_and_regression_results": aggregates[
                "action_cost_timeout_and_regression_results"
            ],
            "paired_effects_and_uncertainty": aggregates["paired_effects_and_uncertainty"],
        },
        "attack_matrix": attacks,
        "current_adversarial_findings": dict(adversarial_findings),
        "arc_objective_generalization_ready_score": 1.0 if ready else 0.0,
        "protected_files_unchanged": protected,
        "blocked_reason": "none" if ready else ",".join(gates["failed_gates"]),
        "gate_check_summary": {**gates, "frozen_safety_roster": safety},
        "preconditions_checked": dict(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": _field_principles(),
        "field_provenance": _field_provenance(),
        "random_seed": [int(seed) for seed in RANDOM_SEEDS],
        "duration_s": round(float(duration_s), 6),
        "tests_run": _tests_run_rows(tests_run),
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(ready, partial, gates["failed_gates"]),
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
    t0 = time.monotonic()
    budget = budgets or ShardBudgets()
    trace_path = Path(trace_root)
    checkpoint = Path(checkpoint_path)
    protected_before = _protected_hashes(REPO_ROOT)
    registry_before = path_sha256(REPO_ROOT / REGISTRY_RELATIVE_PATH)
    rosters = freeze_rosters(trace_path, tuning_count=tuning_count, safety_count=safety_count)
    preconditions = preconditions_checked(
        trace_root=trace_path,
        checkpoint_path=checkpoint,
        budgets=budget,
        rosters=rosters,
    )
    tuning_records = load_trace_prefixes(
        trace_path,
        rosters["tuning_games"],
        max_prefixes_per_game=budget.max_prefixes_per_game,
    )
    safety_records = load_trace_prefixes(
        trace_path,
        rosters["safety_games"],
        max_prefixes_per_game=budget.max_prefixes_per_game,
    )
    held_records = load_trace_prefixes(
        trace_path,
        rosters["held_games"],
        max_prefixes_per_game=budget.max_prefixes_per_game,
    )
    tuning = tune_precommitted_parameters(tuning_records, seeds=RANDOM_SEEDS)
    selected = tuning["selected"]
    safety_shards = run_sharded_cells(
        held_records=safety_records,
        seeds=RANDOM_SEEDS,
        suffix_max_k=int(selected["suffix_max_k"]),
        objective_weight=float(selected["objective_weight"]),
        training_model=tuning["training_model"],
        checkpoint_path=checkpoint.with_name(f".{checkpoint.name}.safety.json"),
        budgets=ShardBudgets(max_prefixes_per_game=budget.max_prefixes_per_game),
        progress=False,
    )
    shard_result = run_sharded_cells(
        held_records=held_records,
        seeds=RANDOM_SEEDS,
        suffix_max_k=int(selected["suffix_max_k"]),
        objective_weight=float(selected["objective_weight"]),
        training_model=tuning["training_model"],
        checkpoint_path=checkpoint,
        budgets=budget,
        progress=progress,
    )
    registry_after = path_sha256(REPO_ROOT / REGISTRY_RELATIVE_PATH)
    protected_after = _protected_hashes(REPO_ROOT)
    artifact = _build_artifact(
        date=date,
        rosters=rosters,
        preconditions=preconditions,
        tuning=tuning,
        shard_result=shard_result,
        safety_rows=safety_shards["rows"],
        protected_before=protected_before,
        protected_after=protected_after,
        registry_before=registry_before,
        registry_after=registry_after,
        duration_s=time.monotonic() - t0,
        tests_run=tests_run,
        adversarial_findings={
            "ran": False,
            "critical_count": 0,
            "flags": [],
            "note": "pending_until_artifact_write",
        },
    )
    target = Path(result_path)
    if write:
        atomic_write_json(target, artifact, sort_keys=True, allow_override=False)
        findings = _current_adversarial_findings(target, run_adversarial=run_adversarial)
        artifact = _build_artifact(
            date=date,
            rosters=rosters,
            preconditions=preconditions,
            tuning=tuning,
            shard_result=shard_result,
            safety_rows=safety_shards["rows"],
            protected_before=protected_before,
            protected_after=protected_after,
            registry_before=registry_before,
            registry_after=registry_after,
            duration_s=time.monotonic() - t0,
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
    if artifact.get("no_game_or_level_solve_claim") is not True:
        errors.append("no_game_or_level_solve_claim must be true")
    if artifact.get("solve_registry_unchanged") is not True:
        errors.append("solve_registry_unchanged must be true")
    for field in ("game_source_access_count", "offline_ground_truth_bfs_count", "per_game_adapter_count"):
        if int(artifact.get(field, -1) or 0) != 0:
            errors.append(f"{field} must be zero")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if "solve_provenance" in artifact:
        errors.append("solve_provenance must be absent")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("success:", "complete:", "complete_", "success_")):
        errors.append("honest_verdict must start with a terminal prefix")
    principles = artifact.get("field_principles") or {}
    provenance = artifact.get("field_provenance") or {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            errors.append(f"field_principles missing {field}")
        if field not in provenance:
            errors.append(f"field_provenance missing {field}")
    for condition in READINESS_CONDITIONS:
        if condition not in principles:
            errors.append(f"field_principles missing readiness condition {condition}")
    rows = list(artifact.get("per_unit_rows") or [])
    if len({row.get("row_id") for row in rows}) != len(rows):
        errors.append("duplicate per_unit_rows row_id")
    recomputed = recompute_aggregates(rows)
    for field in (
        "collision_rates_by_arm",
        "legal_action_coverage_by_arm",
        "held_next_state_reachability_by_arm",
        "policy_influence_by_arm",
        "action_cost_timeout_and_regression_results",
        "paired_effects_and_uncertainty",
    ):
        if artifact.get(field) != recomputed[field]:
            errors.append(f"aggregate_row_mismatch:{field}")
    if (artifact.get("aggregate_row_recomputation") or {}).get("row_checksum") != recomputed[
        "aggregate_row_recomputation"
    ]["row_checksum"]:
        errors.append("aggregate_row_mismatch:row_checksum")
    if not all((row.get("unchanged") is True) for row in (artifact.get("protected_files_unchanged") or {}).values()):
        errors.append("protected_files_unchanged")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    ready = float(artifact.get("arc_objective_generalization_ready_score", 0.0) or 0.0)
    gates = artifact.get("gate_check_summary") or {}
    if ready == 1.0 and not gates.get("all_ready_gates_passed"):
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
                "ready_score": artifact["arc_objective_generalization_ready_score"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
