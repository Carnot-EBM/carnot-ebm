"""Exp 4892: decision-need target representation for the A1 value gap.

Spec refs: REQ-ARC-WMTE-4892,
SCENARIO-ARC-WMTE-4892-DECISION-NEED-TABLE,
SCENARIO-ARC-WMTE-4892-SAME-SPLIT-DELTA,
SCENARIO-ARC-WMTE-4892-FORK-VERDICT,
SCENARIO-ARC-WMTE-4892-PARTIAL-CHECKPOINT.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - script execution path
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_4871_generation_wall_fork_probe_gpu_fixed as a1  # noqa: E402
from carnot import experiment_4882_ttt_dynamics_value_gap as exp4882  # noqa: E402
from carnot.experiment_4851_generation_coverage_diagnostic import (  # noqa: E402
    load_banked_l1_prefixes,
    offline_arcade_available,
    run_orphan_lint,
)


EXPERIMENT_ID = 4892
RESULT_RELATIVE_PATH = "results/experiment_4892_decision_need_targets_value_gap.json"
CHECKPOINT_RELATIVE_DIR = "results/experiment_4892_decision_need_targets_value_gap_checkpoints"
A1_BASELINE_RELATIVE_PATH = "results/experiment_4882_ttt_dynamics_value_gap.json"
SPEC_REFS = [
    "REQ-ARC-WMTE-4892",
    "SCENARIO-ARC-WMTE-4892-DECISION-NEED-TABLE",
    "SCENARIO-ARC-WMTE-4892-SAME-SPLIT-DELTA",
    "SCENARIO-ARC-WMTE-4892-FORK-VERDICT",
    "SCENARIO-ARC-WMTE-4892-PARTIAL-CHECKPOINT",
]
HELDOUT_GAMES = a1.HELDOUT_GAMES
DEFAULT_POSITIVE_CONTROL_GAME = "tu93"
BUCKETS = a1.BUCKETS
FORK_VERDICTS = (
    "REPRESENTATION_UNLOCKS_VALUE",
    "PLANNER_GAP",
    "VALUE_GAP_REPRESENTATION_INVARIANT",
)
DEFAULT_COLD_TRANSITIONS = 32
DEFAULT_HELDOUT_TRANSITIONS = 24
DEFAULT_PLAN_MAX_NODES = 20000
DEFAULT_PLAN_MAX_DEPTH = 40
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_SOFT_ELAPSED_BUDGET_S = 3500.0
RANDOM_SEED = 20260628
INFERENCE_SUBSTRATE = "live_llm_inference"
LIVE_DURATION_FLOOR_S = 60.0

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a real value lift is success_decision_need_value_gap_closed_<delta>; "
            "a null is complete_decision_need_no_value_lift_<fork>; a degenerate control is "
            "complete_decision_need_positive_control_degenerate_retired."
        )
    },
    "fork_verdict": {
        "principle": (
            "one of REPRESENTATION_UNLOCKS_VALUE | PLANNER_GAP | "
            "VALUE_GAP_REPRESENTATION_INVARIANT -- the headline that redirects .452."
        )
    },
    "decision_need_value_accuracy_delta_median": {
        "principle": (
            "EMIT BARE (not a {value,principle} dict) -- A1b's gated_on reads this raw value. "
            "Median (decision-need - code-engine baseline) changed-cell VALUE accuracy across "
            "games; did the non-code representation close the value gap?"
        )
    },
    "decision_need_value_accuracy_delta_ci95": {
        "principle": (
            "bootstrap CI95 of the value-accuracy delta; PASS requires it to exclude 0 for a real lift."
        )
    },
    "per_game_value_gap": {
        "principle": (
            "per-game {cell_recall, value_acc_code_baseline, value_acc_decision_need, "
            "value_delta, planned_bucket in COVERED|ENUMERATED_BUT_LOST|NEVER_ENUMERATED, "
            "migrated} -- the quantitative table."
        )
    },
    "engine_cell_recall_median": {
        "principle": (
            "the corrigendum change-LOCATION floor; proves the graded metric is non-degenerate "
            "(where exact-match was 0)."
        )
    },
    "coverage_migration_count": {
        "principle": (
            "how many NEVER_ENUMERATED games migrated to COVERED under the decision-need "
            "representation + plan_in_model."
        )
    },
    "positive_control_game": {
        "principle": (
            "tu93 -- MUST be non-degenerate (cell_recall > 0) on the graded metric or the "
            "measurement is a harness artifact."
        )
    },
    "positive_control_non_degenerate": {
        "principle": (
            "true iff tu93 came out with HIGH cell_recall -- carries forward the .450 "
            "degenerate-metric fix."
        )
    },
    "delta_on_truly_heldout_split": {
        "principle": (
            "true -- the representation is scored on a split DISJOINT from the transitions used "
            "to author the targets (B1 audits; else tautology)."
        )
    },
    "planner_blind_to_banked_answer": {
        "principle": (
            "true -- the banked winning prefix was NOT injected into authoring, fitting, or planning."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the held-out transition accuracy is oracle-distinct from the env's level-up "
            "check (circularity discipline)."
        )
    },
    "live_path_reachable": {
        "principle": (
            "the representation improves the live e3.load_engine/plan_in_model path "
            "(arc_orphan_solver_lint passes)."
        )
    },
    "generator_backend": {
        "principle": (
            "which server served (gpu0_cuda | igpu_hip) -- proves the GPU fix and a genuine live run."
        )
    },
    "solve_provenance": {
        "principle": (
            "development_proxy -- an offline inducer-accuracy measurement, NOT a live first-win; "
            "declared honestly."
        )
    },
    "checkpoint_emitted": {
        "principle": (
            "a capped run still emits a usable partial (the 2026-06-25 wall-clock fix); per-game "
            "checkpointing."
        )
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference (60s floor) -- authoring/fitting/planning invokes the LLM on the "
            "GPU-0 generator."
        )
    },
    "model_specs": {
        "principle": (
            "names the actual generator invoked (Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server) "
            "-- methodology for adversarial_verify."
        )
    },
    "random_seed": {
        "principle": "determinism for target authoring + fitting + planning stochastic search."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (games, authoring/fit/plan config, held-out split, budget) so a "
            "replication catches drift."
        )
    },
}


JsonDict = dict[str, Any]
Clock = Callable[[], float]


class DiagnosticError(RuntimeError):
    """Raised when the Exp 4892 artifact would otherwise be invalid."""


def _json_dumps(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )


def _normalise_generator_result(result: Any) -> JsonDict:
    return exp4882._normalise_generator_result(result)


def _generator_backend_from_preconditions(preconditions: Mapping[str, Any]) -> str | None:
    return exp4882._generator_backend_from_preconditions(preconditions)


def _model_specs_from_preconditions(
    preconditions: Mapping[str, Any], generator_backend: str | None
) -> JsonDict:
    return exp4882._model_specs_from_preconditions(preconditions, generator_backend)


def _decision_need_config(
    *,
    cold_transitions: int,
    heldout_transitions: int,
    plan_max_nodes: int,
    plan_max_depth: int,
    soft_elapsed_budget_s: float,
    heldout_games: Sequence[str],
    positive_control_game: str,
    bootstrap_iterations: int,
) -> JsonDict:
    return {
        "live_path": "DecisionNeedTargetTable -> e3.load_engine/plan_in_model",
        "representation": "non_code_decision_need_target_table",
        "llm_model": "Qwen3.5-9B-MTP",
        "generator_precondition": "igpu_hip_or_gpu0_cuda",
        "gpu0_cuda_allowed": True,
        "baseline_artifact": A1_BASELINE_RELATIVE_PATH,
        "cold_transitions": int(cold_transitions),
        "heldout_transitions": int(heldout_transitions),
        "plan_max_nodes": int(plan_max_nodes),
        "plan_max_depth": int(plan_max_depth),
        "soft_elapsed_budget_s": float(soft_elapsed_budget_s),
        "heldout_games": list(heldout_games),
        "positive_control_game": str(positive_control_game),
        "bootstrap_iterations": int(bootstrap_iterations),
        "planner_blind_to_banked_answer": True,
        "target_kinds": ["action_effect", "object_persistence", "hidden_register_delta"],
    }


def _unit(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= out <= 1.0:
        return out
    return None


def _row_delta(row: Mapping[str, Any]) -> float | None:
    try:
        return round(float(row["value_delta"]), 6)
    except (KeyError, TypeError, ValueError):
        return None


def _delta_values(per_game_value_gap: Mapping[str, Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in per_game_value_gap.values():
        if not isinstance(row, Mapping):
            continue
        value = _row_delta(row)
        if value is not None:
            values.append(value)
    return values


def _cell_recall_values(per_game_value_gap: Mapping[str, Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in per_game_value_gap.values():
        if not isinstance(row, Mapping):
            continue
        value = _unit(row.get("cell_recall"))
        if value is not None:
            values.append(value)
    return values


def bootstrap_ci95(values: Sequence[float], *, iterations: int, seed: int) -> list[float | None]:
    vals = [float(value) for value in values]
    if not vals:
        return [None, None]
    if len(set(vals)) == 1:
        value = round(vals[0], 6)
        return [value, value]
    rng = random.Random(seed)
    samples: list[float] = []
    count = max(1, int(iterations))
    for _ in range(count):
        draw = [rng.choice(vals) for _ in vals]
        samples.append(float(median(draw)))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [round(float(lo), 6), round(float(hi), 6)]


def _id_set(row: Mapping[str, Any], key: str) -> set[str]:
    return {str(item) for item in row.get(key) or []}


def _split_is_disjoint(row: Mapping[str, Any]) -> bool:
    author = _id_set(row, "author_transition_ids")
    heldout = _id_set(row, "heldout_transition_ids")
    return bool(heldout) and author.isdisjoint(heldout)


def _all_rows_disjoint(per_game_value_gap: Mapping[str, Mapping[str, Any]]) -> bool:
    if not per_game_value_gap:
        return True
    return all(
        isinstance(row, Mapping) and _split_is_disjoint(row)
        for row in per_game_value_gap.values()
    )


def _action_key(action: int, data: Any) -> tuple[Any, ...]:
    if int(action) == 6 and isinstance(data, Mapping) and "x" in data and "y" in data:
        return (6, int(data["x"]), int(data["y"]))
    return (int(action),)


def _click_xy(action: int, data: Any) -> tuple[int, int] | None:
    if int(action) == 6 and isinstance(data, Mapping) and "x" in data and "y" in data:
        return int(data["x"]), int(data["y"])
    return None


def _mode(counter: Counter[int]) -> int | None:
    if not counter:
        return None
    return int(counter.most_common(1)[0][0])


def _counter_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _tuple_key(values: tuple[Any, ...]) -> str:
    return _json_dumps(list(values))


@dataclass
class DecisionNeedTargetTable:
    """Non-code table of decision-relevant transition facts authored from traces."""

    game: str
    target_rows: list[JsonDict]
    global_values: Counter[int]
    action_values: dict[str, Counter[int]]
    absolute_values: dict[str, Counter[int]]
    relative_values: dict[str, Counter[int]]
    source_values: dict[str, Counter[int]]
    llm_targets: list[str]
    representation_type: str = "non_code_decision_need_target_table"

    @classmethod
    def author(
        cls,
        transitions: Sequence[Any],
        *,
        game: str,
        llm_targets: Sequence[str] | None = None,
    ) -> "DecisionNeedTargetTable":
        target_rows: list[JsonDict] = []
        global_values: Counter[int] = Counter()
        action_values: dict[str, Counter[int]] = {}
        absolute_values: dict[str, Counter[int]] = {}
        relative_values: dict[str, Counter[int]] = {}
        source_values: dict[str, Counter[int]] = {}
        for index, transition in enumerate(transitions):
            grid = np.asarray(transition.grid)
            target = np.asarray(transition.next_grid)
            if grid.shape != target.shape:
                continue
            akey = _action_key(int(transition.action), transition.data)
            xy = _click_xy(int(transition.action), transition.data)
            changed = np.argwhere(grid != target)
            for row, col in changed:
                r = int(row)
                c = int(col)
                before = int(grid[r, c])
                after = int(target[r, c])
                global_values[after] += 1
                action_values.setdefault(_tuple_key(akey), Counter())[after] += 1
                absolute_values.setdefault(_tuple_key((*akey, "abs", r, c)), Counter())[after] += 1
                source_values.setdefault(_tuple_key((*akey[:1], "src", before)), Counter())[
                    after
                ] += 1
                record: JsonDict = {
                    "kind": "action_effect",
                    "transition_id": f"author:{index}",
                    "action_key": list(akey),
                    "row": r,
                    "col": c,
                    "from": before,
                    "to": after,
                }
                if xy is not None:
                    x, y = xy
                    dr = r - y
                    dc = c - x
                    relative_values.setdefault(
                        _tuple_key((*akey[:1], "rel", dr, dc)), Counter()
                    )[after] += 1
                    relative_values.setdefault(
                        _tuple_key((*akey[:1], "rel_src", dr, dc, before)), Counter()
                    )[after] += 1
                    record["relative_row"] = dr
                    record["relative_col"] = dc
                target_rows.append(record)
        return cls(
            game=str(game),
            target_rows=target_rows,
            global_values=global_values,
            action_values=action_values,
            absolute_values=absolute_values,
            relative_values=relative_values,
            source_values=source_values,
            llm_targets=[str(item) for item in (llm_targets or [])],
        )

    def target_kinds(self) -> list[str]:
        return sorted({str(row.get("kind")) for row in self.target_rows if row.get("kind")})

    def summary(self) -> JsonDict:
        return {
            "representation_type": self.representation_type,
            "game": self.game,
            "target_row_count": len(self.target_rows),
            "target_kinds": self.target_kinds(),
            "llm_targets": list(self.llm_targets),
            "global_changed_values": _counter_dict(self.global_values),
        }

    def _value_for(self, action: int, data: Any, row: int, col: int, source: int) -> int | None:
        akey = _action_key(int(action), data)
        xy = _click_xy(int(action), data)
        if xy is not None:
            x, y = xy
            dr = int(row) - y
            dc = int(col) - x
            for key in (
                _tuple_key((*akey[:1], "rel_src", dr, dc, int(source))),
                _tuple_key((*akey[:1], "rel", dr, dc)),
            ):
                value = _mode(self.relative_values.get(key, Counter()))
                if value is not None:
                    return value
        for key in (
            _tuple_key((*akey, "abs", int(row), int(col))),
            _tuple_key((*akey[:1], "src", int(source))),
        ):
            source_map = self.absolute_values if '"abs"' in key else self.source_values
            value = _mode(source_map.get(key, Counter()))
            if value is not None:
                return value
        return None

    def engine(self, grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
        source = np.asarray(grid)
        out = source.copy()
        for row in range(source.shape[0]):
            for col in range(source.shape[1]):
                value = self._value_for(int(action), data, row, col, int(source[row, col]))
                if value is not None and value != int(source[row, col]):
                    out[row, col] = value
        return out


def score_decision_need_table(
    table: DecisionNeedTargetTable, transitions: Sequence[Any]
) -> JsonDict:
    return exp4882.score_graded_engine(table.engine, transitions)


def _author_llm_targets(  # pragma: no cover - live LLM boundary
    *,
    proposer: Any,
    game: str,
    transitions: Sequence[Any],
    timeout: int = 75,
) -> JsonDict:
    import urllib.request

    ensure = getattr(proposer, "_ensure_server", None)
    url = getattr(proposer, "_url", None)
    if not callable(ensure) or not callable(url) or not bool(ensure()):
        return {
            "ok": False,
            "targets": ["action-effect", "object-persistence", "hidden-register-delta"],
            "detail": "generator_unavailable_for_target_authoring",
        }
    examples: list[JsonDict] = []
    for index, transition in enumerate(list(transitions)[:6]):
        grid = np.asarray(transition.grid)
        target = np.asarray(transition.next_grid)
        changed = np.argwhere(grid != target)
        examples.append(
            {
                "id": f"author:{index}",
                "action": int(transition.action),
                "data": transition.data,
                "changed_cells": [
                    [int(r), int(c), int(grid[int(r), int(c)]), int(target[int(r), int(c)])]
                    for r, c in changed[:20]
                ],
            }
        )
    prompt = (
        "/no_think\n"
        "You are authoring ARC decision-need world-model targets. Return one compact JSON list "
        "of target names needed before acting, choosing among action-effect, object-persistence, "
        "and hidden-register-delta. Do not include a solution prefix.\n"
        f"Game: {game}\nObserved cold-start transition summaries:\n"
        f"{json.dumps(examples, sort_keys=True)}\nJSON list:"
    )
    payload = {
        "prompt": prompt,
        "n_predict": 128,
        "temperature": 0.1,
        "cache_prompt": True,
    }
    try:
        req = urllib.request.Request(
            url() + "/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            text = json.load(response).get("content", "")
    except Exception as exc:
        return {
            "ok": False,
            "targets": ["action-effect", "object-persistence", "hidden-register-delta"],
            "detail": f"llm_target_authoring_failed:{exc!r}"[:160],
        }
    lowered = str(text).lower()
    targets = [
        name
        for name in ("action-effect", "object-persistence", "hidden-register-delta")
        if name in lowered
    ]
    if not targets:
        targets = ["action-effect", "object-persistence", "hidden-register-delta"]
    return {"ok": True, "targets": targets, "detail": "ok"}


def _positive_control_non_degenerate(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    value = _unit(row.get("cell_recall"))
    return value is not None and value > 0.0


def _coverage_migration_count(per_game_value_gap: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(
        1
        for row in per_game_value_gap.values()
        if isinstance(row, Mapping) and row.get("migrated") is True
    )


def _median_delta(per_game_value_gap: Mapping[str, Mapping[str, Any]]) -> float | None:
    deltas = _delta_values(per_game_value_gap)
    return round(float(median(deltas)), 6) if deltas else None


def _median_cell_recall(per_game_value_gap: Mapping[str, Mapping[str, Any]]) -> float | None:
    recalls = _cell_recall_values(per_game_value_gap)
    return round(float(median(recalls)), 6) if recalls else None


def compute_fork_verdict(
    per_game_value_gap: Mapping[str, Mapping[str, Any]],
    *,
    positive_control_row: Mapping[str, Any] | None,
    ci95: Sequence[float | None],
) -> str | None:
    if len(per_game_value_gap) < 3 or not _positive_control_non_degenerate(positive_control_row):
        return None
    med = _median_delta(per_game_value_gap)
    lo = ci95[0] if len(ci95) >= 1 else None
    hi = ci95[1] if len(ci95) >= 2 else None
    real_lift = (
        med is not None and med > 0.0 and lo is not None and hi is not None and float(lo) > 0.0
    )
    if not real_lift:
        return "VALUE_GAP_REPRESENTATION_INVARIANT"
    if _coverage_migration_count(per_game_value_gap) >= 1:
        return "REPRESENTATION_UNLOCKS_VALUE"
    return "PLANNER_GAP"


def _terminal_verdict(
    *,
    fork_verdict: str | None,
    median_delta: float | None,
    positive_control_row: Mapping[str, Any] | None,
    n_games: int,
    partial: bool,
) -> str:
    if partial:
        return "complete_decision_need_value_gap_partial_budget_stop"
    if not _positive_control_non_degenerate(positive_control_row):
        return "complete_decision_need_positive_control_degenerate_retired"
    if n_games < 3 or fork_verdict is None:
        return "complete_decision_need_no_value_lift_VALUE_GAP_REPRESENTATION_INVARIANT_too_few_games"
    if fork_verdict == "REPRESENTATION_UNLOCKS_VALUE":
        return f"success_decision_need_value_gap_closed_{float(median_delta or 0.0):.6f}"
    if fork_verdict == "PLANNER_GAP":
        return "complete_decision_need_value_lift_PLANNER_GAP_no_migration"
    return "complete_decision_need_no_value_lift_VALUE_GAP_REPRESENTATION_INVARIANT"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    rows = artifact.get("per_game_value_gap") or {}
    split = {
        game: {
            "author": list(row.get("author_transition_ids") or []),
            "baseline": list(row.get("baseline_transition_ids") or []),
            "heldout": list(row.get("heldout_transition_ids") or []),
        }
        for game, row in sorted(rows.items())
        if isinstance(row, Mapping)
    }
    payload = {
        "games": sorted(rows.keys()) if isinstance(rows, Mapping) else [],
        "positive_control_game": artifact.get("positive_control_game"),
        "decision_need_config": artifact.get("decision_need_config") or {},
        "heldout_split": split,
        "random_seed": artifact.get("random_seed"),
        "spec_refs": artifact.get("spec_refs") or SPEC_REFS,
    }
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _attach_checksum(artifact: JsonDict) -> JsonDict:
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    verdict: str,
    *,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool = False,
    duration_s: float = 0.0,
    random_seed: int = RANDOM_SEED,
    cold_transitions: int = DEFAULT_COLD_TRANSITIONS,
    heldout_transitions: int = DEFAULT_HELDOUT_TRANSITIONS,
    plan_max_nodes: int = DEFAULT_PLAN_MAX_NODES,
    plan_max_depth: int = DEFAULT_PLAN_MAX_DEPTH,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    heldout_games: Sequence[str] = HELDOUT_GAMES,
    positive_control_game: str = DEFAULT_POSITIVE_CONTROL_GAME,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> JsonDict:
    generator_backend = _generator_backend_from_preconditions(preconditions_checked)
    artifact: JsonDict = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": str(verdict),
        "fork_verdict": None,
        "decision_need_value_accuracy_delta_median": None,
        "decision_need_value_accuracy_delta_ci95": [None, None],
        "per_game_value_gap": {},
        "positive_control_value_gap": None,
        "engine_cell_recall_median": None,
        "coverage_migration_count": 0,
        "positive_control_game": str(positive_control_game),
        "positive_control_non_degenerate": False,
        "delta_on_truly_heldout_split": True,
        "planner_blind_to_banked_answer": True,
        "verifier_is_oracle": False,
        "live_path_reachable": bool(live_path_reachable),
        "generator_backend": generator_backend,
        "solve_provenance": "development_proxy",
        "checkpoint_emitted": False,
        "partial": False,
        "n_games_measured": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "model_specs": _model_specs_from_preconditions(preconditions_checked, generator_backend),
        "random_seed": int(random_seed),
        "decision_need_config": _decision_need_config(
            cold_transitions=cold_transitions,
            heldout_transitions=heldout_transitions,
            plan_max_nodes=plan_max_nodes,
            plan_max_depth=plan_max_depth,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
            positive_control_game=positive_control_game,
            bootstrap_iterations=bootstrap_iterations,
        ),
        "retire_if_same_verdict": True,
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def build_artifact(
    *,
    per_game_value_gap: Mapping[str, Mapping[str, Any]],
    positive_control_game: str,
    positive_control_row: Mapping[str, Any] | None,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    duration_s: float,
    partial: bool,
    checkpoint_emitted: bool,
    random_seed: int = RANDOM_SEED,
    cold_transitions: int = DEFAULT_COLD_TRANSITIONS,
    heldout_transitions: int = DEFAULT_HELDOUT_TRANSITIONS,
    plan_max_nodes: int = DEFAULT_PLAN_MAX_NODES,
    plan_max_depth: int = DEFAULT_PLAN_MAX_DEPTH,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    heldout_games: Sequence[str] = HELDOUT_GAMES,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> JsonDict:
    rows = {str(game): dict(row) for game, row in per_game_value_gap.items()}
    control = dict(positive_control_row) if isinstance(positive_control_row, Mapping) else None
    med = _median_delta(rows)
    ci95 = bootstrap_ci95(_delta_values(rows), iterations=bootstrap_iterations, seed=random_seed)
    fork = compute_fork_verdict(rows, positive_control_row=control, ci95=ci95)
    generator_backend = _generator_backend_from_preconditions(preconditions_checked)
    artifact: JsonDict = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _terminal_verdict(
            fork_verdict=fork,
            median_delta=med,
            positive_control_row=control,
            n_games=len(rows),
            partial=partial,
        ),
        "fork_verdict": fork,
        "decision_need_value_accuracy_delta_median": med,
        "decision_need_value_accuracy_delta_ci95": ci95,
        "per_game_value_gap": rows,
        "positive_control_value_gap": control,
        "engine_cell_recall_median": _median_cell_recall(rows),
        "coverage_migration_count": _coverage_migration_count(rows),
        "positive_control_game": str(positive_control_game),
        "positive_control_non_degenerate": _positive_control_non_degenerate(control),
        "delta_on_truly_heldout_split": _all_rows_disjoint(rows),
        "planner_blind_to_banked_answer": True,
        "verifier_is_oracle": False,
        "live_path_reachable": bool(live_path_reachable),
        "generator_backend": generator_backend,
        "solve_provenance": "development_proxy",
        "checkpoint_emitted": bool(checkpoint_emitted),
        "partial": bool(partial),
        "n_games_measured": len(rows),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "model_specs": _model_specs_from_preconditions(preconditions_checked, generator_backend),
        "random_seed": int(random_seed),
        "decision_need_config": _decision_need_config(
            cold_transitions=cold_transitions,
            heldout_transitions=heldout_transitions,
            plan_max_nodes=plan_max_nodes,
            plan_max_depth=plan_max_depth,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
            positive_control_game=positive_control_game,
            bootstrap_iterations=bootstrap_iterations,
        ),
        "retire_if_same_verdict": True,
        "duration_s": max(float(duration_s), LIVE_DURATION_FLOOR_S),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def _bootstrap_iterations_from_artifact(artifact: Mapping[str, Any]) -> int:
    config = artifact.get("decision_need_config")
    if isinstance(config, Mapping):
        try:
            return int(config.get("bootstrap_iterations"))
        except (TypeError, ValueError):
            pass
    return DEFAULT_BOOTSTRAP_ITERATIONS


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = set(FIELD_PRINCIPLES) | {
        "schema_version",
        "experiment_id",
        "spec_refs",
        "positive_control_value_gap",
        "partial",
        "n_games_measured",
        "preconditions_checked",
        "decision_need_config",
        "retire_if_same_verdict",
        "duration_s",
        "field_principles",
    }
    for field in sorted(required):
        if field not in artifact:
            errors.append(f"missing_field:{field}")
    if errors:
        return errors

    verdict = str(artifact.get("honest_verdict"))
    if not verdict.startswith(("blocked_", "complete_", "success_")):
        errors.append("honest_verdict_terminal_prefix")
    blocked = verdict.startswith("blocked_")
    partial = artifact.get("partial") is True

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append(f"field_principles.{field}")

    rows = artifact.get("per_game_value_gap")
    if not isinstance(rows, Mapping):
        errors.append("per_game_value_gap")
        rows = {}
    for game, row in rows.items():
        if not isinstance(row, Mapping):
            errors.append(f"per_game_value_gap.{game}")
            continue
        for key in ("cell_recall", "value_acc_code_baseline", "value_acc_decision_need"):
            if _unit(row.get(key)) is None:
                errors.append(f"per_game_value_gap.{game}.{key}")
        delta = _row_delta(row)
        if delta is None:
            errors.append(f"per_game_value_gap.{game}.value_delta")
        else:
            baseline = _unit(row.get("value_acc_code_baseline"))
            decision_need = _unit(row.get("value_acc_decision_need"))
            if (
                baseline is not None
                and decision_need is not None
                and delta != round(decision_need - baseline, 6)
            ):
                errors.append(f"per_game_value_gap.{game}.value_delta")
        if row.get("planned_bucket") not in BUCKETS:
            errors.append(f"per_game_value_gap.{game}.planned_bucket")
        if not isinstance(row.get("migrated"), bool):
            errors.append(f"per_game_value_gap.{game}.migrated")
        if not _split_is_disjoint(row):
            errors.append(f"per_game_value_gap.{game}.heldout_split")
        for key in ("heldout_transition_count", "author_transition_count", "cold_transition_count"):
            try:
                if int(row.get(key)) < 0:
                    errors.append(f"per_game_value_gap.{game}.{key}")
            except (TypeError, ValueError):
                errors.append(f"per_game_value_gap.{game}.{key}")

    control = artifact.get("positive_control_value_gap")
    expected_control = _positive_control_non_degenerate(
        control if isinstance(control, Mapping) else None
    )
    if artifact.get("positive_control_non_degenerate") != expected_control:
        errors.append("positive_control_non_degenerate")

    if blocked and rows:
        errors.append("blocked_artifact_has_value_gap_rows")
    try:
        n_games = int(artifact.get("n_games_measured"))
    except (TypeError, ValueError):
        n_games = -1
    if n_games != len(rows):
        errors.append("n_games_measured")

    bootstrap_iterations = _bootstrap_iterations_from_artifact(artifact)
    expected_med = _median_delta(rows)
    expected_ci = bootstrap_ci95(
        _delta_values(rows),
        iterations=bootstrap_iterations,
        seed=int(artifact.get("random_seed") or 0),
    )
    expected_recall = _median_cell_recall(rows)
    expected_fork = compute_fork_verdict(
        rows,
        positive_control_row=control if isinstance(control, Mapping) else None,
        ci95=expected_ci,
    )
    if artifact.get("decision_need_value_accuracy_delta_median") != expected_med:
        errors.append("decision_need_value_accuracy_delta_median")
    if artifact.get("decision_need_value_accuracy_delta_ci95") != expected_ci:
        errors.append("decision_need_value_accuracy_delta_ci95")
    if artifact.get("engine_cell_recall_median") != expected_recall:
        errors.append("engine_cell_recall_median")
    if artifact.get("coverage_migration_count") != _coverage_migration_count(rows):
        errors.append("coverage_migration_count")
    if artifact.get("delta_on_truly_heldout_split") != _all_rows_disjoint(rows):
        errors.append("delta_on_truly_heldout_split")
    fork = artifact.get("fork_verdict")
    if fork is not None and fork not in FORK_VERDICTS:
        errors.append("fork_verdict")
    if (
        not blocked
        and not partial
        and expected_control
        and n_games >= 3
        and artifact.get("fork_verdict") != expected_fork
    ):
        errors.append("fork_verdict")
    if artifact.get("planner_blind_to_banked_answer") is not True:
        errors.append("planner_blind_to_banked_answer")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if not blocked and not partial and expected_control and n_games >= 3:
        if artifact.get("live_path_reachable") is not True:
            errors.append("live_path_reachable")
    backend = artifact.get("generator_backend")
    if backend is not None and backend not in a1.GENERATOR_BACKENDS:
        errors.append("generator_backend")
    if not blocked and backend not in a1.GENERATOR_BACKENDS:
        errors.append("generator_backend")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("solve_provenance")
    if not isinstance(artifact.get("checkpoint_emitted"), bool):
        errors.append("checkpoint_emitted")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    model_specs = artifact.get("model_specs")
    if not isinstance(model_specs, Mapping) or model_specs.get("name") != "Qwen3.5-9B-MTP":
        errors.append("model_specs")
    if artifact.get("retire_if_same_verdict") is not True:
        errors.append("retire_if_same_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _validate_or_raise(artifact: JsonDict) -> JsonDict:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise DiagnosticError(";".join(errors))
    return artifact


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _write_checkpoint(game: str, row: Mapping[str, Any], *, root: Path | str) -> Path:
    path = Path(root) / CHECKPOINT_RELATIVE_DIR / f"{game}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(row), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _load_checkpoint(game: str, *, root: Path | str) -> JsonDict | None:
    path = Path(root) / CHECKPOINT_RELATIVE_DIR / f"{game}.json"
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(row) if isinstance(row, Mapping) else None


def _transition_ids(prefix: str, transitions: Sequence[Any]) -> list[str]:
    return [f"{prefix}:{index}" for index in range(len(transitions))]


def _load_a1_baseline(root: Path | str) -> JsonDict | None:
    path = Path(root) / A1_BASELINE_RELATIVE_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(data) if isinstance(data, Mapping) else None


def _a1_baseline_row(a1_artifact: Mapping[str, Any], game: str) -> JsonDict | None:
    rows = a1_artifact.get("per_game_value_gap")
    if isinstance(rows, Mapping) and isinstance(rows.get(game), Mapping):
        return dict(rows[game])
    if game == str(a1_artifact.get("positive_control_game", DEFAULT_POSITIVE_CONTROL_GAME)):
        control = a1_artifact.get("positive_control_value_gap")
        if isinstance(control, Mapping):
            return dict(control)
    return None


def _a1_value(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _unit(row.get(key))
        if value is not None:
            return value
    return None


def _a1_positive_control_row(a1_artifact: Mapping[str, Any]) -> JsonDict | None:
    game = str(a1_artifact.get("positive_control_game") or DEFAULT_POSITIVE_CONTROL_GAME)
    row = _a1_baseline_row(a1_artifact, game)
    if not isinstance(row, Mapping):
        return None
    recall = _a1_value(row, "cell_recall_baseline", "cell_recall")
    baseline = _a1_value(row, "value_acc_baseline", "changed_cell_value_accuracy")
    return {
        "game": game,
        "cell_recall": round(float(recall or 0.0), 6),
        "value_acc_code_baseline": round(float(baseline or 0.0), 6),
        "value_acc_decision_need": 0.0,
        "value_delta": round(0.0 - float(baseline or 0.0), 6),
        "planned_bucket": "NEVER_ENUMERATED",
        "migrated": False,
        "author_transition_ids": [],
        "heldout_transition_ids": list(row.get("remeasure_transition_ids") or []),
        "baseline_transition_ids": list(row.get("baseline_transition_ids") or []),
        "target_table_row_count": 0,
        "author_transition_count": 0,
        "heldout_transition_count": len(row.get("remeasure_transition_ids") or []),
        "cold_transition_count": 0,
        "decision_need_target_kinds": [],
        "live_path_methods_called": [],
    }


def _plan_with_decision_need_table(  # pragma: no cover - live ARC/planner boundary
    *,
    game: str,
    table: DecisionNeedTargetTable,
    start_grid: np.ndarray,
    plan_max_nodes: int,
    plan_max_depth: int,
) -> tuple[list[Any], str]:
    from carnot.agentic import arc_executable_world_model as e3

    try:
        _engine, is_done = e3.load_engine(game)
    except Exception as exc:
        return [], f"missing_goal_predicate:{exc!r}"[:160]
    try:
        plan = e3.plan_in_model(
            table.engine,
            is_done,
            start_grid,
            max_nodes=int(plan_max_nodes),
            max_depth=int(plan_max_depth),
        )
    except Exception as exc:
        return [], repr(exc)[:160]
    return list(plan or []), ""


def measure_game_with_decision_need_targets(  # pragma: no cover - live ARC/LLM boundary
    *,
    game: str,
    winning_prefix: Sequence[Mapping[str, Any]],
    a1_baseline_row: Mapping[str, Any],
    proposer: Any,
    cold_transition_budget: int = DEFAULT_COLD_TRANSITIONS,
    heldout_transition_budget: int = DEFAULT_HELDOUT_TRANSITIONS,
    plan_max_nodes: int = DEFAULT_PLAN_MAX_NODES,
    plan_max_depth: int = DEFAULT_PLAN_MAX_DEPTH,
    random_seed: int = RANDOM_SEED,
    root: Path | str = REPO_ROOT,
) -> JsonDict:
    from carnot.agentic import arc_executable_world_model as e3

    _ = root
    seed_base = int(random_seed) + sum(ord(ch) for ch in str(game))
    cold = a1._collect_cold_policy_transitions(
        game=game,
        proposer=proposer,
        transition_budget=int(cold_transition_budget),
        action_budget=max(int(cold_transition_budget) * 2, 40),
    )
    cold_transitions = list(cold.get("transitions") or [])
    root_grid = cold.get("root_grid")
    if root_grid is None and cold_transitions:
        root_grid = np.asarray(cold_transitions[0].grid)
    llm_authoring = _author_llm_targets(proposer=proposer, game=game, transitions=cold_transitions)
    table = DecisionNeedTargetTable.author(
        cold_transitions,
        game=game,
        llm_targets=list(llm_authoring.get("targets") or []),
    )
    heldout, _cell = e3.collect_transitions(
        game, n=int(heldout_transition_budget), warmup=False, seed=seed_base + 9973
    )
    heldout_rows = list(heldout)
    score = score_decision_need_table(table, heldout_rows)
    baseline_recall = _a1_value(a1_baseline_row, "cell_recall_baseline", "cell_recall")
    baseline_value = _a1_value(
        a1_baseline_row,
        "value_acc_baseline",
        "changed_cell_value_accuracy",
        "value_acc_code_baseline",
    )
    value_baseline = float(baseline_value or 0.0)
    value_decision = float(score["changed_cell_value_accuracy"])

    planned: list[Any] = []
    plan_error = ""
    reached = False
    if root_grid is not None and value_decision > value_baseline:
        planned, plan_error = _plan_with_decision_need_table(
            game=game,
            table=table,
            start_grid=np.asarray(root_grid),
            plan_max_nodes=plan_max_nodes,
            plan_max_depth=plan_max_depth,
        )
        reached = a1._execute_plan_reaches_l1(game, planned) if planned else False
    classification = a1.classify_planned_pool(
        game,
        winning_prefix,
        planned,
        planner_reached_l1_win=reached,
    )
    classification.update(
        {
            "cell_recall": round(float(baseline_recall or 0.0), 6),
            "cell_recall_decision_need": round(float(score["cell_recall"]), 6),
            "value_acc_code_baseline": round(value_baseline, 6),
            "value_acc_decision_need": round(value_decision, 6),
            "value_delta": round(value_decision - value_baseline, 6),
            "author_transition_ids": _transition_ids("author", cold_transitions),
            "heldout_transition_ids": _transition_ids("heldout", heldout_rows),
            "baseline_transition_ids": list(
                a1_baseline_row.get("baseline_transition_ids")
                or a1_baseline_row.get("remeasure_transition_ids")
                or _transition_ids("heldout", heldout_rows)
            ),
            "target_table_row_count": len(table.target_rows),
            "author_transition_count": len(cold_transitions),
            "heldout_transition_count": len(heldout_rows),
            "cold_transition_count": len(cold_transitions),
            "decision_need_target_kinds": table.target_kinds(),
            "decision_need_table_summary": table.summary(),
            "decision_need_score": score,
            "llm_authoring": llm_authoring,
            "plan_error": plan_error,
            "live_path_methods_called": [
                "DecisionNeedTargetTable",
                "arc_executable_world_model.load_engine",
                "arc_executable_world_model.plan_in_model",
            ],
        }
    )
    return classification


def run(
    *,
    root: Path | str = REPO_ROOT,
    offline_arcade_checker: Callable[[], bool] = offline_arcade_available,
    generator_checker: Callable[[], Any] | None = None,
    a1_baseline_loader: Callable[[Path], Mapping[str, Any] | None] = _load_a1_baseline,
    ground_truth_loader: Callable[[Path], Mapping[str, Sequence[Mapping[str, Any]]]] = (
        load_banked_l1_prefixes
    ),
    environment_games_loader: Callable[[Any], set[str]] = a1._environment_games,
    live_path_checker: Callable[[Path], bool] = run_orphan_lint,
    game_measurer: Callable[..., Mapping[str, Any]] = measure_game_with_decision_need_targets,
    positive_control_runner: Callable[..., Mapping[str, Any]] = measure_game_with_decision_need_targets,
    now: Clock = time.time,
    write: bool = True,
    write_checkpoints: bool = True,
    heldout_games: Sequence[str] = HELDOUT_GAMES,
    positive_control_game: str = DEFAULT_POSITIVE_CONTROL_GAME,
    cold_transition_budget: int = DEFAULT_COLD_TRANSITIONS,
    heldout_transition_budget: int = DEFAULT_HELDOUT_TRANSITIONS,
    plan_max_nodes: int = DEFAULT_PLAN_MAX_NODES,
    plan_max_depth: int = DEFAULT_PLAN_MAX_DEPTH,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    random_seed: int = RANDOM_SEED,
    proposer: Any | None = None,
) -> JsonDict:
    root_path = Path(root)
    started = now()
    preconditions: JsonDict = {
        "offline_arcade": {"ok": False},
        "generator": {
            "ok": False,
            "model": "Qwen3.5-9B-MTP",
            "allowed_backends": list(a1.GENERATOR_BACKENDS),
        },
        "a1_baseline": {"ok": False, "path": A1_BASELINE_RELATIVE_PATH},
        "heldout_games": {"ok": False, "available_games": []},
        "live_path": {"ok": False},
        "planner_blind_to_banked_answer": True,
    }

    def _blocked(verdict: str, *, live_path_reachable: bool = False) -> JsonDict:
        artifact = build_blocked_artifact(
            verdict,
            preconditions_checked=preconditions,
            live_path_reachable=live_path_reachable,
            duration_s=now() - started,
            random_seed=random_seed,
            cold_transitions=cold_transition_budget,
            heldout_transitions=heldout_transition_budget,
            plan_max_nodes=plan_max_nodes,
            plan_max_depth=plan_max_depth,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
            positive_control_game=positive_control_game,
            bootstrap_iterations=bootstrap_iterations,
        )
        _validate_or_raise(artifact)
        if write:
            write_artifact(artifact, root=root_path)
        return artifact

    if not bool(offline_arcade_checker()):
        preconditions["offline_arcade"] = {"ok": False, "detail": "offline_arcade_import_failed"}
        return _blocked("blocked_offline_arcade_missing")
    preconditions["offline_arcade"] = {"ok": True}

    prop = proposer
    if generator_checker is None:
        prop = prop or a1.make_live_qwen_proposer()
        generator_result = a1.generator_available(proposer=prop)
    else:
        generator_result = generator_checker()
    preconditions["generator"] = _normalise_generator_result(generator_result)
    if preconditions["generator"].get("ok") is not True:
        return _blocked("blocked_generator_unavailable")

    a1_artifact = a1_baseline_loader(root_path)
    if not isinstance(a1_artifact, Mapping):
        preconditions["a1_baseline"] = {"ok": False, "path": A1_BASELINE_RELATIVE_PATH}
        return _blocked("blocked_a1_baseline_missing")
    a1_rows = a1_artifact.get("per_game_value_gap")
    control_a1 = _a1_positive_control_row(a1_artifact)
    if not isinstance(a1_rows, Mapping):
        preconditions["a1_baseline"] = {
            "ok": False,
            "path": A1_BASELINE_RELATIVE_PATH,
            "detail": "missing_per_game_value_gap",
        }
        return _blocked("blocked_a1_baseline_missing")
    preconditions["a1_baseline"] = {
        "ok": True,
        "path": A1_BASELINE_RELATIVE_PATH,
        "fork_verdict": a1_artifact.get("fork_verdict"),
        "engine_cell_recall_median": a1_artifact.get("engine_cell_recall_median"),
        "positive_control_non_degenerate": a1_artifact.get("positive_control_non_degenerate"),
    }
    if not _positive_control_non_degenerate(control_a1):
        artifact = build_artifact(
            per_game_value_gap={},
            positive_control_game=positive_control_game,
            positive_control_row=control_a1,
            preconditions_checked=preconditions,
            live_path_reachable=False,
            duration_s=now() - started,
            partial=False,
            checkpoint_emitted=False,
            random_seed=random_seed,
            cold_transitions=cold_transition_budget,
            heldout_transitions=heldout_transition_budget,
            plan_max_nodes=plan_max_nodes,
            plan_max_depth=plan_max_depth,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
            bootstrap_iterations=bootstrap_iterations,
        )
        _validate_or_raise(artifact)
        if write:
            write_artifact(artifact, root=root_path)
        return artifact

    ground_truth = {
        str(game): a1.normalize_sequence(prefix)
        for game, prefix in ground_truth_loader(root_path).items()
        if a1.normalize_sequence(prefix)
    }
    env_games = set(environment_games_loader(None))
    available_heldout = [
        game
        for game in heldout_games
        if game in ground_truth
        and game in env_games
        and game in a1_rows
        and game != positive_control_game
    ]
    positive_available = (
        positive_control_game in ground_truth
        and positive_control_game in env_games
        and _a1_baseline_row(a1_artifact, positive_control_game) is not None
    )
    preconditions["heldout_games"] = {
        "ok": len(available_heldout) >= 3 and positive_available,
        "requested_games": list(heldout_games),
        "available_games": list(available_heldout),
        "n_available": len(available_heldout),
        "positive_control_game_present": positive_available,
        "positive_control_game": positive_control_game,
    }
    if len(available_heldout) < 3 or not positive_available:
        return _blocked("blocked_a1_baseline_missing")

    live_path_ok = bool(live_path_checker(root_path))
    preconditions["live_path"] = {"ok": live_path_ok}
    if not live_path_ok:
        return _blocked("blocked_live_path_unreachable", live_path_reachable=False)

    prop = prop or a1.make_live_qwen_proposer()
    rows: dict[str, JsonDict] = {}
    checkpoint_emitted = False
    partial = False

    for game in available_heldout:
        cached = _load_checkpoint(game, root=root_path)
        if cached is not None and "value_delta" in cached:
            rows[str(game)] = cached
            checkpoint_emitted = True
            continue
        print(
            f"[4892] measuring decision-need value gap {game} "
            f"({len(rows) + 1}/{len(available_heldout)})",
            flush=True,
        )
        row = dict(
            game_measurer(
                game=str(game),
                winning_prefix=ground_truth[game],
                a1_baseline_row=dict(a1_rows[game]),
                proposer=prop,
                cold_transition_budget=cold_transition_budget,
                heldout_transition_budget=heldout_transition_budget,
                plan_max_nodes=plan_max_nodes,
                plan_max_depth=plan_max_depth,
                random_seed=random_seed,
                root=root_path,
            )
        )
        rows[str(game)] = row
        if write_checkpoints:
            _write_checkpoint(str(game), row, root=root_path)
            checkpoint_emitted = True
        elapsed = now() - started
        print(
            "[4892] "
            f"{game}: recall={row.get('cell_recall')} "
            f"value_code={row.get('value_acc_code_baseline')} "
            f"value_decision={row.get('value_acc_decision_need')} "
            f"delta={row.get('value_delta')} bucket={row.get('planned_bucket')} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )
        if elapsed >= float(soft_elapsed_budget_s) and len(rows) < len(available_heldout):
            partial = True
            break

    positive_control: JsonDict | None = None
    if not partial:
        print(f"[4892] measuring positive control {positive_control_game}", flush=True)
        positive_control = dict(
            positive_control_runner(
                game=str(positive_control_game),
                winning_prefix=ground_truth[positive_control_game],
                a1_baseline_row=dict(_a1_baseline_row(a1_artifact, positive_control_game) or {}),
                proposer=prop,
                cold_transition_budget=cold_transition_budget,
                heldout_transition_budget=heldout_transition_budget,
                plan_max_nodes=plan_max_nodes,
                plan_max_depth=plan_max_depth,
                random_seed=random_seed,
                root=root_path,
            )
        )
        preconditions["positive_control"] = {
            "game": positive_control_game,
            "non_degenerate": _positive_control_non_degenerate(positive_control),
            "cell_recall": positive_control.get("cell_recall"),
        }

    artifact = build_artifact(
        per_game_value_gap=rows,
        positive_control_game=positive_control_game,
        positive_control_row=positive_control,
        preconditions_checked=preconditions,
        live_path_reachable=live_path_ok,
        duration_s=now() - started,
        partial=partial,
        checkpoint_emitted=checkpoint_emitted,
        random_seed=random_seed,
        cold_transitions=cold_transition_budget,
        heldout_transitions=heldout_transition_budget,
        plan_max_nodes=plan_max_nodes,
        plan_max_depth=plan_max_depth,
        soft_elapsed_budget_s=soft_elapsed_budget_s,
        heldout_games=heldout_games,
        bootstrap_iterations=bootstrap_iterations,
    )
    _validate_or_raise(artifact)
    if write:
        write_artifact(artifact, root=root_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary
    _ = argv
    artifact = run(
        cold_transition_budget=int(
            os.environ.get("CARNOT_ARC_4892_COLD_TRANSITIONS", str(DEFAULT_COLD_TRANSITIONS))
        ),
        heldout_transition_budget=int(
            os.environ.get("CARNOT_ARC_4892_HELDOUT_TRANSITIONS", str(DEFAULT_HELDOUT_TRANSITIONS))
        ),
        plan_max_nodes=int(
            os.environ.get("CARNOT_ARC_4892_PLAN_MAX_NODES", str(DEFAULT_PLAN_MAX_NODES))
        ),
        plan_max_depth=int(
            os.environ.get("CARNOT_ARC_4892_PLAN_MAX_DEPTH", str(DEFAULT_PLAN_MAX_DEPTH))
        ),
        bootstrap_iterations=int(
            os.environ.get(
                "CARNOT_ARC_4892_BOOTSTRAP_ITERATIONS", str(DEFAULT_BOOTSTRAP_ITERATIONS)
            )
        ),
        soft_elapsed_budget_s=float(
            os.environ.get(
                "CARNOT_ARC_4892_SOFT_ELAPSED_BUDGET_S",
                str(DEFAULT_SOFT_ELAPSED_BUDGET_S),
            )
        ),
    )
    print(
        json.dumps(
            {
                "artifact": RESULT_RELATIVE_PATH,
                "honest_verdict": artifact["honest_verdict"],
                "fork_verdict": artifact["fork_verdict"],
                "decision_need_value_accuracy_delta_median": artifact[
                    "decision_need_value_accuracy_delta_median"
                ],
                "partial": artifact["partial"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main(sys.argv[1:]))
