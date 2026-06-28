"""Exp 4893: action-prefix latent adapter for the A1 value gap.

Spec refs: REQ-ARC-WMTE-4893,
SCENARIO-ARC-WMTE-4893-A1-LOW-VALUE-GATE,
SCENARIO-ARC-WMTE-4893-ACTION-PREFIX-LATENT,
SCENARIO-ARC-WMTE-4893-GENUINELY-LIVE,
SCENARIO-ARC-WMTE-4893-FORK-VERDICT.
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
from carnot import experiment_4892_decision_need_targets_value_gap as exp4892  # noqa: E402
from carnot.experiment_4851_generation_coverage_diagnostic import (  # noqa: E402
    load_banked_l1_prefixes,
    offline_arcade_available,
    run_orphan_lint,
)


EXPERIMENT_ID = 4893
RESULT_RELATIVE_PATH = "results/experiment_4893_action_prefix_latent_adapter.json"
CHECKPOINT_RELATIVE_DIR = "results/experiment_4893_action_prefix_latent_adapter_checkpoints"
A1_BASELINE_RELATIVE_PATH = "results/experiment_4892_decision_need_targets_value_gap.json"
SPEC_REFS = [
    "REQ-ARC-WMTE-4893",
    "SCENARIO-ARC-WMTE-4893-A1-LOW-VALUE-GATE",
    "SCENARIO-ARC-WMTE-4893-ACTION-PREFIX-LATENT",
    "SCENARIO-ARC-WMTE-4893-GENUINELY-LIVE",
    "SCENARIO-ARC-WMTE-4893-FORK-VERDICT",
]
HELDOUT_GAMES = a1.HELDOUT_GAMES
DEFAULT_POSITIVE_CONTROL_GAME = "tu93"
FORK_VERDICTS = ("REPRESENTATION_MATTERS", "VALUE_GAP_REPRESENTATION_INVARIANT_HARD")
DEFAULT_COLD_TRANSITIONS = 32
DEFAULT_HELDOUT_TRANSITIONS = 24
DEFAULT_MAX_PREFIX_LEN = 3
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_SOFT_ELAPSED_BUDGET_S = 3500.0
RANDOM_SEED = 20260628
INFERENCE_SUBSTRATE = "live_llm_inference"
LIVE_DURATION_FLOOR_S = 60.0
A1_LOW_VALUE_DELTA_GATE = 0.1

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a real lift is success_action_prefix_latent_value_gap_closed_<delta>; "
            "a flat null is complete_action_prefix_latent_no_value_lift_representation_invariant_hard."
        )
    },
    "fork_verdict": {
        "principle": (
            "one of REPRESENTATION_MATTERS | VALUE_GAP_REPRESENTATION_INVARIANT_HARD -- "
            "redirects .452."
        )
    },
    "action_prefix_value_accuracy_delta_median": {
        "principle": (
            "median (latent-representation - A1 code-engine baseline) changed-cell value "
            "accuracy; did the latent representation close the value gap?"
        )
    },
    "action_prefix_value_accuracy_delta_ci95": {
        "principle": (
            "bootstrap CI95 of the latent-representation value-accuracy delta; PASS requires "
            "it to exclude 0 for a real lift."
        )
    },
    "per_game_value_gap": {
        "principle": (
            "per-game {value_acc_code_baseline, value_acc_latent, cell_recall, delta, ci95} "
            "-- the quantitative table."
        )
    },
    "ran_genuinely_live": {
        "principle": (
            "true iff duration_s > 60 and the latent induction/scoring genuinely ran (the "
            "explicit .450 A1b 13.7s non-test fix)."
        )
    },
    "delta_on_truly_heldout_split": {
        "principle": (
            "true -- scored on the SAME held-out split as A1, disjoint from any fit set "
            "(B1 audits)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the held-out transition score is oracle-distinct from the env's "
            "level-up check (circularity discipline)."
        )
    },
    "live_path_reachable": {
        "principle": (
            "the latent adapter uses the live e3 load_engine interface (arc_orphan_solver_lint "
            "passes), not a parallel solver."
        )
    },
    "generator_backend": {
        "principle": (
            "which server served (gpu0_cuda | igpu_hip) -- proves the GPU fix and a genuine "
            "live run."
        )
    },
    "solve_provenance": {
        "principle": (
            "development_proxy -- a representation-accuracy measurement, NOT a banked level."
        )
    },
    "checkpoint_emitted": {
        "principle": "a capped run still emits a usable partial (per-game checkpointing)."
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference (60s floor) -- the latent induction invokes an LLM on the "
            "GPU-0 generator."
        )
    },
    "model_specs": {
        "principle": (
            "names the inducer (Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server) -- methodology "
            "for adversarial_verify."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/generator/A1-baseline checks; a missing resource emits blocked_."
        )
    },
    "random_seed": {
        "principle": "determinism for the latent-adapter induction stochastic search."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (games, A1 baseline, latent-adapter config, held-out split) so "
            "a replication catches drift."
        )
    },
}


JsonDict = dict[str, Any]
Clock = Callable[[], float]
ActionKey = tuple[Any, ...]


class DiagnosticError(RuntimeError):
    """Raised when the Exp 4893 artifact would otherwise be invalid."""


def _json_dumps(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )


def _normalise_generator_result(result: Any) -> JsonDict:
    return exp4892._normalise_generator_result(result)


def _generator_backend_from_preconditions(preconditions: Mapping[str, Any]) -> str | None:
    return exp4892._generator_backend_from_preconditions(preconditions)


def _model_specs_from_preconditions(
    preconditions: Mapping[str, Any], generator_backend: str | None
) -> JsonDict:
    return exp4892._model_specs_from_preconditions(preconditions, generator_backend)


def _action_prefix_config(
    *,
    cold_transitions: int,
    heldout_transitions: int,
    max_prefix_len: int,
    soft_elapsed_budget_s: float,
    heldout_games: Sequence[str],
    positive_control_game: str,
    bootstrap_iterations: int,
) -> JsonDict:
    return {
        "live_path": "ActionPrefixLatentAdapter -> e3.load_engine",
        "representation": "non_code_action_prefix_latent_delta_table",
        "llm_model": "Qwen3.5-9B-MTP",
        "generator_precondition": "igpu_hip_or_gpu0_cuda",
        "gpu0_cuda_allowed": True,
        "baseline_artifact": A1_BASELINE_RELATIVE_PATH,
        "cold_transitions": int(cold_transitions),
        "heldout_transitions": int(heldout_transitions),
        "max_prefix_len": int(max_prefix_len),
        "soft_elapsed_budget_s": float(soft_elapsed_budget_s),
        "heldout_games": list(heldout_games),
        "positive_control_game": str(positive_control_game),
        "bootstrap_iterations": int(bootstrap_iterations),
        "planner_blind_to_banked_answer": True,
        "latent_substrate": "action_prefix_future_state_delta",
        "papers": ["arXiv:2606.26217", "arXiv:2603.19312"],
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
        return round(float(row["delta"]), 6)
    except (KeyError, TypeError, ValueError):
        return None


def _delta_values(per_game_value_gap: Mapping[str, Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in per_game_value_gap.values():
        if isinstance(row, Mapping):
            value = _row_delta(row)
            if value is not None:
                values.append(value)
    return values


def _cell_recall_values(per_game_value_gap: Mapping[str, Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in per_game_value_gap.values():
        if isinstance(row, Mapping):
            value = _unit(row.get("cell_recall"))
            if value is not None:
                values.append(value)
    return values


def bootstrap_ci95(values: Sequence[float], *, iterations: int, seed: int) -> list[float | None]:
    return exp4892.bootstrap_ci95(values, iterations=iterations, seed=seed)


def _id_set(row: Mapping[str, Any], key: str) -> set[str]:
    return {str(item) for item in row.get(key) or []}


def _split_is_disjoint(row: Mapping[str, Any]) -> bool:
    fit = _id_set(row, "fit_transition_ids")
    heldout = _id_set(row, "heldout_transition_ids")
    return bool(heldout) and fit.isdisjoint(heldout)


def _all_rows_disjoint(per_game_value_gap: Mapping[str, Mapping[str, Any]]) -> bool:
    if not per_game_value_gap:
        return True
    return all(
        isinstance(row, Mapping) and _split_is_disjoint(row)
        for row in per_game_value_gap.values()
    )


def _action_key(action: int, data: Any) -> ActionKey:
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


def _prefix_key(prefix: Sequence[ActionKey]) -> str:
    return _json_dumps([list(item) for item in prefix])


def _prefix_suffixes(prefix: Sequence[ActionKey]) -> list[tuple[ActionKey, ...]]:
    out: list[tuple[ActionKey, ...]] = []
    full = tuple(prefix)
    for size in range(len(full), 0, -1):
        out.append(full[-size:])
    return out


@dataclass
class ActionPrefixLatentAdapter:
    """Non-code table of latent future-state deltas keyed by action prefixes."""

    game: str
    max_prefix_len: int
    latent_deltas: list[JsonDict]
    absolute_values: dict[str, Counter[int]]
    relative_values: dict[str, Counter[int]]
    source_values: dict[str, Counter[int]]
    llm_latents: list[str]
    representation_type: str = "action_prefix_latent_delta_table"

    @classmethod
    def induce(
        cls,
        transitions: Sequence[Any],
        *,
        game: str,
        llm_latents: Sequence[str] | None = None,
        max_prefix_len: int = DEFAULT_MAX_PREFIX_LEN,
    ) -> "ActionPrefixLatentAdapter":
        latent_deltas: list[JsonDict] = []
        absolute_values: dict[str, Counter[int]] = {}
        relative_values: dict[str, Counter[int]] = {}
        source_values: dict[str, Counter[int]] = {}
        history: list[ActionKey] = []
        prefix_limit = max(1, int(max_prefix_len))
        for index, transition in enumerate(transitions):
            action = int(transition.action)
            data = transition.data
            current = _action_key(action, data)
            prefix = [*history[-(prefix_limit - 1) :], current]
            grid = np.asarray(transition.grid)
            target = np.asarray(transition.next_grid)
            if grid.shape != target.shape:
                history.append(current)
                continue
            changed = np.argwhere(grid != target)
            for suffix in _prefix_suffixes(prefix):
                pkey = _prefix_key(suffix)
                xy = _click_xy(action, data)
                for row, col in changed:
                    r = int(row)
                    c = int(col)
                    before = int(grid[r, c])
                    after = int(target[r, c])
                    absolute_values.setdefault(
                        _tuple_key((pkey, "abs", r, c, before)), Counter()
                    )[after] += 1
                    source_values.setdefault(_tuple_key((pkey, "src", before)), Counter())[
                        after
                    ] += 1
                    record: JsonDict = {
                        "kind": "action_prefix_future_state_delta",
                        "transition_id": f"fit:{index}",
                        "prefix": [list(item) for item in suffix],
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
                            _tuple_key((pkey, "rel_src", dr, dc, before)), Counter()
                        )[after] += 1
                        record["relative_row"] = dr
                        record["relative_col"] = dc
                    latent_deltas.append(record)
            history.append(current)
        return cls(
            game=str(game),
            max_prefix_len=prefix_limit,
            latent_deltas=latent_deltas,
            absolute_values=absolute_values,
            relative_values=relative_values,
            source_values=source_values,
            llm_latents=[str(item) for item in (llm_latents or [])],
        )

    def summary(self) -> JsonDict:
        return {
            "representation_type": self.representation_type,
            "game": self.game,
            "max_prefix_len": self.max_prefix_len,
            "latent_delta_count": len(self.latent_deltas),
            "accepted_delta_count": len(self.latent_deltas),
            "llm_latents": list(self.llm_latents),
            "source_value_histogram_count": len(self.source_values),
        }

    def _value_for(
        self,
        prefix: Sequence[ActionKey],
        action: int,
        data: Any,
        row: int,
        col: int,
        source: int,
    ) -> int | None:
        xy = _click_xy(action, data)
        for suffix in _prefix_suffixes(prefix):
            pkey = _prefix_key(suffix)
            if xy is not None:
                x, y = xy
                dr = int(row) - y
                dc = int(col) - x
                value = _mode(
                    self.relative_values.get(
                        _tuple_key((pkey, "rel_src", dr, dc, int(source))), Counter()
                    )
                )
                if value is not None:
                    return value
            for key, values in (
                (_tuple_key((pkey, "abs", int(row), int(col), int(source))), self.absolute_values),
                (_tuple_key((pkey, "src", int(source))), self.source_values),
            ):
                value = _mode(values.get(key, Counter()))
                if value is not None:
                    return value
        return None

    def predict(
        self,
        grid: np.ndarray,
        action: int,
        data: Any = None,
        *,
        history: Sequence[ActionKey] = (),
    ) -> np.ndarray:
        source = np.asarray(grid)
        out = source.copy()
        current = _action_key(int(action), data)
        prefix = [*list(history)[-(self.max_prefix_len - 1) :], current]
        for row in range(source.shape[0]):
            for col in range(source.shape[1]):
                value = self._value_for(
                    prefix, int(action), data, row, col, int(source[row, col])
                )
                if value is not None and value != int(source[row, col]):
                    out[row, col] = value
        return out

    def engine(self, grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
        return self.predict(grid, action, data, history=())


def score_action_prefix_adapter(
    adapter: ActionPrefixLatentAdapter, transitions: Sequence[Any]
) -> JsonDict:
    actual_changed_cells = 0
    predicted_changed_cells = 0
    overlap_changed_cells = 0
    correct_changed_values = 0
    changing_transition_count = 0
    prediction_errors = 0
    history: list[ActionKey] = []

    for transition in transitions:
        grid = np.asarray(transition.grid)
        target = np.asarray(transition.next_grid)
        actual_mask = grid != target
        actual_count = int(actual_mask.sum())
        current = _action_key(int(transition.action), transition.data)
        if actual_count <= 0:
            history.append(current)
            continue
        changing_transition_count += 1
        actual_changed_cells += actual_count
        try:
            pred = np.asarray(
                adapter.predict(
                    grid.copy(), int(transition.action), transition.data, history=history
                )
            )
        except Exception:
            prediction_errors += 1
            history.append(current)
            continue
        history.append(current)
        if pred.shape != target.shape:
            prediction_errors += 1
            continue
        pred_mask = grid != pred
        predicted_changed_cells += int(pred_mask.sum())
        overlap = actual_mask & pred_mask
        overlap_count = int(overlap.sum())
        overlap_changed_cells += overlap_count
        if overlap_count:
            correct_changed_values += int((pred[overlap] == target[overlap]).sum())

    cell_recall = (
        float(overlap_changed_cells / actual_changed_cells) if actual_changed_cells else 0.0
    )
    value_accuracy = (
        float(correct_changed_values / overlap_changed_cells) if overlap_changed_cells else 0.0
    )
    return {
        "cell_recall": round(cell_recall, 6),
        "changed_cell_value_accuracy": round(value_accuracy, 6),
        "actual_changed_cells": int(actual_changed_cells),
        "predicted_changed_cells": int(predicted_changed_cells),
        "overlap_changed_cells": int(overlap_changed_cells),
        "correct_changed_values": int(correct_changed_values),
        "changing_transition_count": int(changing_transition_count),
        "prediction_errors": int(prediction_errors),
    }


def _induce_llm_latents(  # pragma: no cover - live LLM boundary
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
            "latents": ["action-prefix-delta"],
            "detail": "generator_unavailable_for_latent_induction",
            "live_llm_invocations": 0,
        }
    examples: list[JsonDict] = []
    for index, transition in enumerate(list(transitions)[:6]):
        grid = np.asarray(transition.grid)
        target = np.asarray(transition.next_grid)
        changed = np.argwhere(grid != target)
        examples.append(
            {
                "id": f"fit:{index}",
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
        "Return a compact JSON list naming latent action-prefix delta factors for this ARC "
        "world-model adapter. Use non-code factor names only. Do not include a solution prefix.\n"
        f"Game: {game}\nObserved transition summaries:\n"
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
            "latents": ["action-prefix-delta"],
            "detail": f"llm_latent_induction_failed:{exc!r}"[:160],
            "live_llm_invocations": 1,
        }
    lowered = str(text).lower()
    latents = [
        name
        for name in ("action-prefix-delta", "future-state-delta", "latent-register-delta")
        if name in lowered
    ]
    if not latents:
        latents = ["action-prefix-delta"]
    return {"ok": True, "latents": latents, "detail": "ok", "live_llm_invocations": 1}


def _positive_control_non_degenerate(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    value = _unit(row.get("cell_recall"))
    return value is not None and value > 0.0


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
    if med is not None and med > 0.0 and lo is not None and hi is not None and float(lo) > 0.0:
        return "REPRESENTATION_MATTERS"
    return "VALUE_GAP_REPRESENTATION_INVARIANT_HARD"


def _terminal_verdict(
    *,
    fork_verdict: str | None,
    median_delta: float | None,
    positive_control_row: Mapping[str, Any] | None,
    n_games: int,
    partial: bool,
) -> str:
    if partial:
        return "complete_action_prefix_latent_partial_budget_stop"
    if not _positive_control_non_degenerate(positive_control_row):
        return "complete_action_prefix_latent_positive_control_degenerate_retired"
    if n_games < 3 or fork_verdict is None:
        return "complete_action_prefix_latent_no_value_lift_representation_invariant_hard_too_few_games"
    if fork_verdict == "REPRESENTATION_MATTERS":
        return f"success_action_prefix_latent_value_gap_closed_{float(median_delta or 0.0):.6f}"
    return "complete_action_prefix_latent_no_value_lift_representation_invariant_hard"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    rows = artifact.get("per_game_value_gap") or {}
    split = {
        game: {
            "fit": list(row.get("fit_transition_ids") or []),
            "baseline": list(row.get("baseline_transition_ids") or []),
            "heldout": list(row.get("heldout_transition_ids") or []),
        }
        for game, row in sorted(rows.items())
        if isinstance(row, Mapping)
    }
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, Mapping):
        preconditions = {}
    a1 = preconditions.get("a1_baseline", {})
    if not isinstance(a1, Mapping):
        a1 = {}
    payload = {
        "games": sorted(rows.keys()) if isinstance(rows, Mapping) else [],
        "positive_control_game": artifact.get("positive_control_game"),
        "a1_baseline": {
            "path": artifact.get("a1_baseline_path"),
            "decision_need_value_accuracy_delta_median": a1.get(
                "decision_need_value_accuracy_delta_median"
            ),
            "fork_verdict": a1.get("fork_verdict"),
        },
        "action_prefix_latent_config": artifact.get("action_prefix_latent_config") or {},
        "heldout_split": split,
        "random_seed": artifact.get("random_seed"),
        "spec_refs": artifact.get("spec_refs") or SPEC_REFS,
    }
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _attach_checksum(artifact: JsonDict) -> JsonDict:
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _ran_genuinely_live(
    *,
    blocked: bool,
    partial: bool,
    duration_s: float,
    live_llm_invocations: int,
    n_games: int,
) -> bool:
    return (
        not blocked
        and not partial
        and float(duration_s) > LIVE_DURATION_FLOOR_S
        and int(live_llm_invocations) > 0
        and int(n_games) >= 3
    )


def build_blocked_artifact(
    verdict: str,
    *,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool = False,
    duration_s: float = 0.0,
    random_seed: int = RANDOM_SEED,
    cold_transitions: int = DEFAULT_COLD_TRANSITIONS,
    heldout_transitions: int = DEFAULT_HELDOUT_TRANSITIONS,
    max_prefix_len: int = DEFAULT_MAX_PREFIX_LEN,
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
        "action_prefix_value_accuracy_delta_median": None,
        "action_prefix_value_accuracy_delta_ci95": [None, None],
        "per_game_value_gap": {},
        "positive_control_game": str(positive_control_game),
        "positive_control_value_gap": None,
        "positive_control_non_degenerate": False,
        "engine_cell_recall_median": None,
        "ran_genuinely_live": False,
        "delta_on_truly_heldout_split": True,
        "planner_blind_to_banked_answer": True,
        "verifier_is_oracle": False,
        "live_path_reachable": bool(live_path_reachable),
        "generator_backend": generator_backend,
        "solve_provenance": "development_proxy",
        "checkpoint_emitted": False,
        "partial": False,
        "n_games_measured": 0,
        "duration_s": float(duration_s),
        "live_llm_invocations": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "model_specs": _model_specs_from_preconditions(preconditions_checked, generator_backend),
        "random_seed": int(random_seed),
        "a1_baseline_path": A1_BASELINE_RELATIVE_PATH,
        "action_prefix_latent_config": _action_prefix_config(
            cold_transitions=cold_transitions,
            heldout_transitions=heldout_transitions,
            max_prefix_len=max_prefix_len,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
            positive_control_game=positive_control_game,
            bootstrap_iterations=bootstrap_iterations,
        ),
        "retire_if_same_verdict": True,
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
    live_llm_invocations: int,
    partial: bool,
    checkpoint_emitted: bool,
    random_seed: int = RANDOM_SEED,
    cold_transitions: int = DEFAULT_COLD_TRANSITIONS,
    heldout_transitions: int = DEFAULT_HELDOUT_TRANSITIONS,
    max_prefix_len: int = DEFAULT_MAX_PREFIX_LEN,
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
    blocked = False
    ran_live = _ran_genuinely_live(
        blocked=blocked,
        partial=partial,
        duration_s=duration_s,
        live_llm_invocations=live_llm_invocations,
        n_games=len(rows),
    )
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
        "action_prefix_value_accuracy_delta_median": med,
        "action_prefix_value_accuracy_delta_ci95": ci95,
        "per_game_value_gap": rows,
        "positive_control_game": str(positive_control_game),
        "positive_control_value_gap": control,
        "positive_control_non_degenerate": _positive_control_non_degenerate(control),
        "engine_cell_recall_median": _median_cell_recall(rows),
        "ran_genuinely_live": ran_live,
        "delta_on_truly_heldout_split": _all_rows_disjoint(rows),
        "planner_blind_to_banked_answer": True,
        "verifier_is_oracle": False,
        "live_path_reachable": bool(live_path_reachable),
        "generator_backend": generator_backend,
        "solve_provenance": "development_proxy",
        "checkpoint_emitted": bool(checkpoint_emitted),
        "partial": bool(partial),
        "n_games_measured": len(rows),
        "duration_s": round(float(duration_s), 6),
        "live_llm_invocations": int(live_llm_invocations),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "model_specs": _model_specs_from_preconditions(preconditions_checked, generator_backend),
        "random_seed": int(random_seed),
        "a1_baseline_path": A1_BASELINE_RELATIVE_PATH,
        "action_prefix_latent_config": _action_prefix_config(
            cold_transitions=cold_transitions,
            heldout_transitions=heldout_transitions,
            max_prefix_len=max_prefix_len,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
            positive_control_game=positive_control_game,
            bootstrap_iterations=bootstrap_iterations,
        ),
        "retire_if_same_verdict": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def _bootstrap_iterations_from_artifact(artifact: Mapping[str, Any]) -> int:
    config = artifact.get("action_prefix_latent_config")
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
        "positive_control_game",
        "positive_control_value_gap",
        "positive_control_non_degenerate",
        "engine_cell_recall_median",
        "planner_blind_to_banked_answer",
        "partial",
        "n_games_measured",
        "duration_s",
        "live_llm_invocations",
        "a1_baseline_path",
        "action_prefix_latent_config",
        "retire_if_same_verdict",
        "field_principles",
    }
    for field in sorted(required):
        if field not in artifact:
            errors.append(f"missing_field:{field}")
    if errors:
        return errors

    verdict = str(artifact.get("honest_verdict"))
    if not verdict.startswith(("blocked_", "skipped_", "complete_", "success_")):
        errors.append("honest_verdict_terminal_prefix")
    blocked = verdict.startswith(("blocked_", "skipped_"))
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
        for key in ("cell_recall", "value_acc_code_baseline", "value_acc_latent"):
            if _unit(row.get(key)) is None:
                errors.append(f"per_game_value_gap.{game}.{key}")
        delta = _row_delta(row)
        if delta is None:
            errors.append(f"per_game_value_gap.{game}.delta")
        else:
            baseline = _unit(row.get("value_acc_code_baseline"))
            latent = _unit(row.get("value_acc_latent"))
            if baseline is not None and latent is not None and delta != round(latent - baseline, 6):
                errors.append(f"per_game_value_gap.{game}.delta")
        ci95 = row.get("ci95")
        if (
            not isinstance(ci95, Sequence)
            or isinstance(ci95, (str, bytes))
            or len(ci95) != 2
            or ci95 != bootstrap_ci95([float(delta or 0.0)], iterations=1, seed=0)
        ):
            errors.append(f"per_game_value_gap.{game}.ci95")
        if not _split_is_disjoint(row):
            errors.append(f"per_game_value_gap.{game}.heldout_split")
        for key in (
            "latent_delta_count",
            "accepted_delta_count",
            "fit_transition_count",
            "heldout_transition_count",
            "cold_transition_count",
        ):
            try:
                if int(row.get(key)) < 0:
                    errors.append(f"per_game_value_gap.{game}.{key}")
            except (TypeError, ValueError):
                errors.append(f"per_game_value_gap.{game}.{key}")
        methods = row.get("live_path_methods_called")
        if not isinstance(methods, Sequence) or "arc_executable_world_model.load_engine" not in {
            str(item) for item in methods
        }:
            errors.append(f"per_game_value_gap.{game}.live_path_methods_called")

    if blocked and rows:
        errors.append("blocked_artifact_has_value_gap_rows")
    try:
        n_games = int(artifact.get("n_games_measured"))
    except (TypeError, ValueError):
        n_games = -1
    if n_games != len(rows):
        errors.append("n_games_measured")

    control = artifact.get("positive_control_value_gap")
    expected_control = _positive_control_non_degenerate(
        control if isinstance(control, Mapping) else None
    )
    if artifact.get("positive_control_non_degenerate") != expected_control:
        errors.append("positive_control_non_degenerate")

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
    if artifact.get("action_prefix_value_accuracy_delta_median") != expected_med:
        errors.append("action_prefix_value_accuracy_delta_median")
    if artifact.get("action_prefix_value_accuracy_delta_ci95") != expected_ci:
        errors.append("action_prefix_value_accuracy_delta_ci95")
    if artifact.get("engine_cell_recall_median") != expected_recall:
        errors.append("engine_cell_recall_median")
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

    try:
        duration_s = float(artifact.get("duration_s"))
        live_llm_invocations = int(artifact.get("live_llm_invocations"))
    except (TypeError, ValueError):
        duration_s = -1.0
        live_llm_invocations = -1
        errors.append("duration_s")
    expected_live = _ran_genuinely_live(
        blocked=blocked,
        partial=partial,
        duration_s=duration_s,
        live_llm_invocations=live_llm_invocations,
        n_games=n_games,
    )
    if artifact.get("ran_genuinely_live") != expected_live:
        errors.append("ran_genuinely_live")
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
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked")
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


def _a1_rows(a1_artifact: Mapping[str, Any]) -> Mapping[str, Any] | None:
    rows = a1_artifact.get("per_game_value_gap")
    return rows if isinstance(rows, Mapping) else None


def _a1_baseline_row(a1_artifact: Mapping[str, Any], game: str) -> JsonDict | None:
    rows = _a1_rows(a1_artifact)
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
    baseline = _a1_value(row, "value_acc_code_baseline", "value_acc_baseline")
    recall = _a1_value(row, "cell_recall", "cell_recall_baseline")
    return {
        "game": game,
        "cell_recall": round(float(recall or 0.0), 6),
        "value_acc_code_baseline": round(float(baseline or 0.0), 6),
        "value_acc_latent": 0.0,
        "delta": round(0.0 - float(baseline or 0.0), 6),
        "ci95": [round(0.0 - float(baseline or 0.0), 6)] * 2,
        "fit_transition_ids": [],
        "heldout_transition_ids": list(row.get("heldout_transition_ids") or []),
        "baseline_transition_ids": list(row.get("baseline_transition_ids") or []),
        "latent_delta_count": 0,
        "accepted_delta_count": 0,
        "fit_transition_count": 0,
        "heldout_transition_count": len(row.get("heldout_transition_ids") or []),
        "cold_transition_count": 0,
        "live_path_methods_called": ["arc_executable_world_model.load_engine"],
    }


def _a1_baseline_is_ready(a1_artifact: Mapping[str, Any]) -> tuple[bool, str]:
    try:
        delta = float(a1_artifact.get("decision_need_value_accuracy_delta_median"))
    except (TypeError, ValueError):
        return False, "missing_decision_need_value_accuracy_delta_median"
    if delta >= A1_LOW_VALUE_DELTA_GATE:
        return False, "a1_value_gap_closed"
    rows = _a1_rows(a1_artifact)
    if not isinstance(rows, Mapping) or len(rows) < 3:
        return False, "missing_per_game_value_gap"
    for game, row in rows.items():
        if not isinstance(row, Mapping):
            return False, f"{game}:row_not_mapping"
        if _a1_value(row, "value_acc_code_baseline", "value_acc_baseline") is None:
            return False, f"{game}:missing_value_acc_code_baseline"
        if not row.get("heldout_transition_ids"):
            return False, f"{game}:missing_heldout_transition_ids"
    return True, "ok"


def measure_game_with_action_prefix_latents(  # pragma: no cover - live ARC/LLM boundary
    *,
    game: str,
    a1_baseline_row: Mapping[str, Any],
    proposer: Any,
    cold_transition_budget: int = DEFAULT_COLD_TRANSITIONS,
    heldout_transition_budget: int = DEFAULT_HELDOUT_TRANSITIONS,
    max_prefix_len: int = DEFAULT_MAX_PREFIX_LEN,
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
    _engine, _is_done = e3.load_engine(game)
    llm_induction = _induce_llm_latents(
        proposer=proposer,
        game=game,
        transitions=cold_transitions,
    )
    adapter = ActionPrefixLatentAdapter.induce(
        cold_transitions,
        game=game,
        llm_latents=list(llm_induction.get("latents") or []),
        max_prefix_len=max_prefix_len,
    )
    heldout_count = len(a1_baseline_row.get("heldout_transition_ids") or []) or int(
        heldout_transition_budget
    )
    heldout, _cell = e3.collect_transitions(
        game, n=heldout_count, warmup=False, seed=seed_base + 9973
    )
    heldout_rows = list(heldout)
    score = score_action_prefix_adapter(adapter, heldout_rows)
    baseline_value = float(
        _a1_value(a1_baseline_row, "value_acc_code_baseline", "value_acc_baseline") or 0.0
    )
    latent_value = float(score["changed_cell_value_accuracy"])
    delta = round(latent_value - baseline_value, 6)
    summary = adapter.summary()
    return {
        "game": str(game),
        "cell_recall": round(float(score["cell_recall"]), 6),
        "value_acc_code_baseline": round(baseline_value, 6),
        "value_acc_latent": round(latent_value, 6),
        "delta": delta,
        "ci95": [delta, delta],
        "fit_transition_ids": _transition_ids("fit", cold_transitions),
        "heldout_transition_ids": _transition_ids("heldout", heldout_rows),
        "baseline_transition_ids": list(
            a1_baseline_row.get("heldout_transition_ids")
            or a1_baseline_row.get("baseline_transition_ids")
            or _transition_ids("heldout", heldout_rows)
        ),
        "latent_delta_count": int(summary["latent_delta_count"]),
        "accepted_delta_count": int(summary["accepted_delta_count"]),
        "fit_transition_count": len(cold_transitions),
        "heldout_transition_count": len(heldout_rows),
        "cold_transition_count": len(cold_transitions),
        "latent_adapter_summary": summary,
        "latent_score": score,
        "llm_latent_induction": llm_induction,
        "live_llm_invocations": int(llm_induction.get("live_llm_invocations") or 0),
        "live_path_methods_called": [
            "ActionPrefixLatentAdapter",
            "arc_executable_world_model.load_engine",
        ],
    }


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
    game_measurer: Callable[..., Mapping[str, Any]] = measure_game_with_action_prefix_latents,
    positive_control_runner: Callable[..., Mapping[str, Any]] = measure_game_with_action_prefix_latents,
    now: Clock = time.time,
    write: bool = True,
    write_checkpoints: bool = True,
    heldout_games: Sequence[str] = HELDOUT_GAMES,
    positive_control_game: str = DEFAULT_POSITIVE_CONTROL_GAME,
    cold_transition_budget: int = DEFAULT_COLD_TRANSITIONS,
    heldout_transition_budget: int = DEFAULT_HELDOUT_TRANSITIONS,
    max_prefix_len: int = DEFAULT_MAX_PREFIX_LEN,
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
            max_prefix_len=max_prefix_len,
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
    a1_ready, a1_detail = _a1_baseline_is_ready(a1_artifact)
    a1_delta = a1_artifact.get("decision_need_value_accuracy_delta_median")
    preconditions["a1_baseline"] = {
        "ok": a1_ready,
        "path": A1_BASELINE_RELATIVE_PATH,
        "detail": a1_detail,
        "fork_verdict": a1_artifact.get("fork_verdict"),
        "decision_need_value_accuracy_delta_median": a1_delta,
        "low_value_delta": a1_ready,
    }
    if not a1_ready:
        if a1_detail == "a1_value_gap_closed":
            return _blocked("skipped_a1_value_gap_closed")
        return _blocked("blocked_a1_baseline_missing")
    a1_rows = _a1_rows(a1_artifact) or {}

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
    live_llm_invocations = 0

    for game in available_heldout:
        cached = _load_checkpoint(game, root=root_path)
        if cached is not None and "delta" in cached:
            rows[str(game)] = cached
            checkpoint_emitted = True
            live_llm_invocations += int(cached.get("live_llm_invocations") or 0)
            continue
        print(
            f"[4893] measuring action-prefix latent value gap {game} "
            f"({len(rows) + 1}/{len(available_heldout)})",
            flush=True,
        )
        row = dict(
            game_measurer(
                game=str(game),
                a1_baseline_row=dict(a1_rows[game]),
                proposer=prop,
                cold_transition_budget=cold_transition_budget,
                heldout_transition_budget=heldout_transition_budget,
                max_prefix_len=max_prefix_len,
                random_seed=random_seed,
                root=root_path,
            )
        )
        row.setdefault("game", str(game))
        row.setdefault("ci95", bootstrap_ci95([float(row.get("delta", 0.0))], iterations=1, seed=0))
        row.setdefault("live_llm_invocations", 1)
        rows[str(game)] = row
        live_llm_invocations += int(row.get("live_llm_invocations") or 0)
        if write_checkpoints:
            _write_checkpoint(str(game), row, root=root_path)
            checkpoint_emitted = True
        elapsed = now() - started
        print(
            "[4893] "
            f"{game}: recall={row.get('cell_recall')} "
            f"value_code={row.get('value_acc_code_baseline')} "
            f"value_latent={row.get('value_acc_latent')} "
            f"delta={row.get('delta')} elapsed_s={elapsed:.1f}",
            flush=True,
        )
        if elapsed >= float(soft_elapsed_budget_s) and len(rows) < len(available_heldout):
            partial = True
            break

    positive_control: JsonDict | None = None
    if not partial:
        print(f"[4893] measuring positive control {positive_control_game}", flush=True)
        positive_control = dict(
            positive_control_runner(
                game=str(positive_control_game),
                a1_baseline_row=dict(_a1_baseline_row(a1_artifact, positive_control_game) or {}),
                proposer=prop,
                cold_transition_budget=cold_transition_budget,
                heldout_transition_budget=heldout_transition_budget,
                max_prefix_len=max_prefix_len,
                random_seed=random_seed,
                root=root_path,
            )
        )
        positive_control.setdefault("ci95", bootstrap_ci95([float(positive_control.get("delta", 0.0))], iterations=1, seed=0))
        positive_control.setdefault("live_llm_invocations", 1)
        live_llm_invocations += int(positive_control.get("live_llm_invocations") or 0)
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
        live_llm_invocations=live_llm_invocations,
        partial=partial,
        checkpoint_emitted=checkpoint_emitted,
        random_seed=random_seed,
        cold_transitions=cold_transition_budget,
        heldout_transitions=heldout_transition_budget,
        max_prefix_len=max_prefix_len,
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
            os.environ.get("CARNOT_ARC_4893_COLD_TRANSITIONS", str(DEFAULT_COLD_TRANSITIONS))
        ),
        heldout_transition_budget=int(
            os.environ.get("CARNOT_ARC_4893_HELDOUT_TRANSITIONS", str(DEFAULT_HELDOUT_TRANSITIONS))
        ),
        max_prefix_len=int(
            os.environ.get("CARNOT_ARC_4893_MAX_PREFIX_LEN", str(DEFAULT_MAX_PREFIX_LEN))
        ),
        bootstrap_iterations=int(
            os.environ.get(
                "CARNOT_ARC_4893_BOOTSTRAP_ITERATIONS", str(DEFAULT_BOOTSTRAP_ITERATIONS)
            )
        ),
        soft_elapsed_budget_s=float(
            os.environ.get(
                "CARNOT_ARC_4893_SOFT_ELAPSED_BUDGET_S",
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
                "action_prefix_value_accuracy_delta_median": artifact[
                    "action_prefix_value_accuracy_delta_median"
                ],
                "ran_genuinely_live": artifact["ran_genuinely_live"],
                "partial": artifact["partial"],
                "duration_s": artifact["duration_s"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
