"""Experiment 10012: do cached world-model gates predict real simulator usefulness?

REQ-ARC-WMTE-10012. This CPU-only harness never changes the live gate and never
calls a model. It scores frozen candidates, plans in each cached model, executes
the plan prefix in the public offline simulator, and reduces fixed gate tables.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from carnot import experiment_10010_b2_think_on_pilot as exp10
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_agi3_world_model import grid_of

EXPERIMENT_ID = 10012
REQUIREMENT_ID = "REQ-ARC-WMTE-10012"
SCHEMA = "carnot.experiment_10012_gate_usefulness.v2"
RANDOM_SEED = 10012
WINDOWS = exp10.PILOT_WINDOWS
CODEONLY_SEEDS = ("7491001", "7491002", "7491003")
PLAN_MAX_NODES = 20_000
PLAN_MAX_DEPTH = 80
BUDGET_MAX_NODES = 150_000
BUDGET_MAX_DEPTH = 200
GOAL_GUIDANCE_LAMBDA = 1.0
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SOLVE_PROVENANCE = "development_proxy"
RAW_REL = Path("results/raw/experiment_10012_gate_usefulness")
ARTIFACT_REL = Path("results/experiment_10012_gate_usefulness.json")
V1_ARTIFACT_REL = RAW_REL / "v1_offline_twin_halt_artifact.json"
LIVE_SESSION_REL = Path(
    "results/raw/experiment_10009_b2_induction_gate_measurement_v3/live_session.json"
)
TELEMETRY_REL = Path(
    "results/raw/experiment_10009_b2_induction_gate_measurement_v3/induction_gate_telemetry.jsonl"
)
THINK_SHARD_REL = Path("results/raw/experiment_10010_b2_think_on_pilot/shard.jsonl")

GATE_NAMES = (
    "live_exact_1.0",
    "masked_exact_1.0",
    "masked_exact_0.875",
    "masked_exact_0.75",
    "change_fidelity_1.0_noop_0",
    "change_fidelity_1.0_noop_0.25",
    "change_fidelity_0.9_noop_0",
    "change_fidelity_0.9_noop_0.25",
    "change_fidelity_0.8_noop_0",
    "change_fidelity_0.8_noop_0.25",
    "change_fidelity_0.7_noop_0",
    "change_fidelity_0.7_noop_0.25",
    "cell_recall_0.9",
    "cell_recall_0.8",
)


def progress(message: str) -> None:
    print(f"[exp{EXPERIMENT_ID} {time.strftime('%H:%M:%S')}] {message}", flush=True)


class Heartbeat:
    def __init__(self, label: str, interval_s: float = 45.0) -> None:
        self.label = label
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(self.interval_s):
            progress(f"heartbeat: {self.label}")

    def __enter__(self) -> "Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_checksum(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload["reproducibility_checksum"] = ""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256_text(encoded)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class IdentityUsefulHarnessBug(RuntimeError):
    """IDENTITY reached a real level-up, invalidating the measurement harness."""


@dataclass
class RebuiltState:
    recoverable: bool
    reason: Optional[str]
    grid: np.ndarray
    level: int
    frame: Any
    env: Any
    actions_replayed: int
    expected_sha256: str
    observed_sha256: str


@dataclass(frozen=True)
class ExecutionArm:
    name: str
    max_nodes: int
    max_depth: int
    halt_on_divergence: bool
    start_state: str
    live: bool


LIVE_SCORED = ExecutionArm(
    "LIVE_SCORED", PLAN_MAX_NODES, PLAN_MAX_DEPTH, False, "scored_agent_rule", True
)
OFFLINE_TWIN_HALT = ExecutionArm(
    "OFFLINE_TWIN_HALT", PLAN_MAX_NODES, PLAN_MAX_DEPTH, True, "induction_state", False
)
BUDGET_150K = ExecutionArm(
    "BUDGET_150K", BUDGET_MAX_NODES, BUDGET_MAX_DEPTH, False, "scored_agent_rule", False
)
EXECUTION_ARMS = (LIVE_SCORED, OFFLINE_TWIN_HALT, BUDGET_150K)


@dataclass(frozen=True)
class Candidate:
    pair_id: str
    game: str
    engine_family: str
    variant: str
    source: Optional[str]
    source_sha256: Optional[str]
    source_path: Optional[str]
    source_status: str


def execution_labels(execution: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "USEFUL": bool(execution.get("real_level_up")),
        "FAITHFUL": bool(
            int(execution.get("matched_steps_before_divergence") or 0) >= 3
            and execution.get("real_state_changed")
        ),
    }


def classify_window_labels(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    if any(
        row.get("engine_family") == "IDENTITY" and bool((row.get("labels") or {}).get("USEFUL"))
        for row in rows
    ):
        raise IdentityUsefulHarnessBug("IDENTITY produced a real level-up")
    expert = next((row for row in rows if row.get("engine_family") == "EXPERT"), None)
    if expert is None or not bool((expert.get("labels") or {}).get("USEFUL")):
        return {
            "status": "label_uninformative",
            "reason": "expert_no_real_level_up_within_live_planning_budget",
        }
    return {"status": "label_informative", "reason": "expert_real_level_up"}


def classify_window_arms(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    for row in rows:
        if row.get("engine_family") != "IDENTITY":
            continue
        for arm_name, labels in (row.get("labels_by_arm") or {}).items():
            if bool((labels or {}).get("USEFUL")):
                raise IdentityUsefulHarnessBug(f"IDENTITY produced a real level-up in {arm_name}")
    expert = next((row for row in rows if row.get("engine_family") == "EXPERT"), None)
    if expert is None:
        raise ValueError("window has no EXPERT row")
    predicate = str(expert.get("goal_predicate_status") or "")
    useful = {
        arm.name: bool(((expert.get("labels_by_arm") or {}).get(arm.name) or {}).get("USEFUL"))
        for arm in EXECUTION_ARMS
    }
    out: dict[str, dict[str, str]] = {}
    for arm in EXECUTION_ARMS:
        if useful[arm.name]:
            out[arm.name] = {
                "status": "label_informative",
                "reason": "expert_real_level_up",
            }
        elif predicate == "missing":
            out[arm.name] = {
                "status": "label_uninformative",
                "reason": "expert_goal_predicate_missing",
            }
        elif predicate == "constant_false":
            out[arm.name] = {
                "status": "label_uninformative",
                "reason": "expert_goal_predicate_always_false",
            }
        elif arm.name == OFFLINE_TWIN_HALT.name and useful[LIVE_SCORED.name]:
            out[arm.name] = {
                "status": "label_uninformative",
                "reason": "harness_halt_rule",
            }
        elif arm.name != BUDGET_150K.name and useful[BUDGET_150K.name]:
            out[arm.name] = {
                "status": "label_uninformative",
                "reason": "planner_budget",
            }
        else:
            out[arm.name] = {"status": "label_uninformative", "reason": "unresolved"}
    return out


def build_gate_tables(
    rows: Sequence[Mapping[str, Any]],
    gate_names: Sequence[str],
    *,
    label: str,
    arm_name: Optional[str] = None,
    candidate_only: bool = False,
) -> dict[str, dict[str, int]]:
    kept = []
    for row in rows:
        if candidate_only and row.get("engine_family") in {"EXPERT", "IDENTITY"}:
            continue
        status = row.get("window_status")
        if arm_name is not None:
            status = ((row.get("arm_status") or {}).get(arm_name) or {}).get("status")
        if status == "label_informative":
            kept.append(row)
    tables: dict[str, dict[str, int]] = {}
    for gate in gate_names:
        counts = {
            "accepted_and_positive": 0,
            "accepted_and_negative": 0,
            "rejected_positive": 0,
            "rejected_negative": 0,
            "n_pairs": len(kept),
        }
        for row in kept:
            accepted = bool((row.get("gate_decisions") or {}).get(gate))
            labels = row.get("labels") or {}
            if arm_name is not None:
                labels = (row.get("labels_by_arm") or {}).get(arm_name) or {}
            positive = bool(labels.get(label))
            key = (
                "accepted_and_positive"
                if accepted and positive
                else "accepted_and_negative"
                if accepted
                else "rejected_positive"
                if positive
                else "rejected_negative"
            )
            counts[key] += 1
        tables[gate] = counts
    return tables


def _grid_sha(grid: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(grid, dtype=np.int16))
    return sha256_bytes(arr.tobytes() + json.dumps(list(arr.shape)).encode())


def rebuild_state_from_actions(
    *,
    env: Any,
    action_rows: Sequence[Mapping[str, Any]],
    induction_action_index: int,
    expected_grid: np.ndarray,
    action_resolver: Callable[[str], Any],
    grid_reader: Callable[[Any], np.ndarray],
    evidence_path: Optional[Path] = None,
) -> RebuiltState:
    frame = None
    replayed = 0
    for row in action_rows:
        action_index = int(row["action_index"])
        if action_index >= int(induction_action_index):
            break
        name = str(row["action"])
        data = row.get("data") or None
        if name == "RESET":
            frame = env.reset()
        else:
            frame = env.step(action_resolver(name), data=data)
        replayed += 1
    if frame is None:
        observed = np.empty((0, 0), dtype=np.int16)
        level = -1
        reason = "no_frame_before_induction"
    else:
        observed = np.asarray(grid_reader(frame))
        level = int(getattr(frame, "levels_completed", 0) or 0)
        reason = None if np.array_equal(observed, expected_grid) else "rebuilt_grid_mismatch"
    result = RebuiltState(
        recoverable=reason is None,
        reason=reason,
        grid=observed,
        level=level,
        frame=frame,
        env=env,
        actions_replayed=replayed,
        expected_sha256=_grid_sha(np.asarray(expected_grid)),
        observed_sha256=_grid_sha(observed),
    )
    if evidence_path is not None:
        atomic_json(
            evidence_path,
            {
                "recoverable": result.recoverable,
                "reason": result.reason,
                "actions_replayed": replayed,
                "induction_action_index": int(induction_action_index),
                "level_from_frame": level,
                "expected_sha256": result.expected_sha256,
                "observed_sha256": result.observed_sha256,
            },
        )
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_episode_actions(repo_root: Path, game: str) -> list[dict[str, Any]]:
    session = json.loads((repo_root / LIVE_SESSION_REL).read_text())
    matches = [
        episode
        for episode in session.get("episodes", [])
        if episode.get("game") == game and int(episode.get("seed", -1)) == 7_491_001
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one seed-7491001 episode for {game}, got {len(matches)}")
    return list(matches[0].get("action_rows") or [])


def induction_alignment(repo_root: Path, game: str, actions: Sequence[Mapping[str, Any]]) -> dict:
    attempts = [
        row
        for row in _read_jsonl(repo_root / TELEMETRY_REL)
        if row.get("record_type") == "induction_attempt"
        and row.get("game_id") == game
        and row.get("episode_id") == f"{game}:seed-7491001"
    ]
    if not attempts:
        raise ValueError(f"no induction attempt for {game}:seed-7491001")
    attempt = attempts[0]
    duration = float(attempt.get("induction_wall_time_s") or attempt.get("seconds") or 0.0)
    end = float(attempt["monotonic_timestamp_s"])
    start = end - duration
    enclosing = [
        row
        for row in actions
        if float(row["interval_start_monotonic_ns"]) / 1e9
        <= start
        <= float(row["interval_end_monotonic_ns"]) / 1e9
    ]
    if len(enclosing) != 1:
        raise ValueError(f"induction start for {game} lies in {len(enclosing)} action intervals")
    return {
        "induction_reason": str(attempt.get("reason") or ""),
        "level_before": int(attempt.get("level_before") or 0),
        "telemetry_step_index": int(attempt["step_index"]),
        "induction_duration_s": duration,
        "induction_end_monotonic_s": end,
        "induction_start_monotonic_s": start,
        "enclosing_action_index": int(enclosing[0]["action_index"]),
        "enclosing_action": str(enclosing[0]["action"]),
    }


def execute_plan_from_current(induction_reason: str) -> bool:
    return induction_reason == "level_up_reinduction"


def plan_start_state(induction_reason: str) -> str:
    return "induction_state" if execute_plan_from_current(induction_reason) else "root_after_reset"


def _source_candidate(
    game: str,
    family: str,
    variant: str,
    source: Optional[str],
    source_path: Optional[Path],
    status: str,
) -> Candidate:
    pair_id = f"{game}__{family.lower()}__{variant}"
    return Candidate(
        pair_id=pair_id,
        game=game,
        engine_family=family,
        variant=variant,
        source=source,
        source_sha256=sha256_text(source) if source is not None else None,
        source_path=str(source_path) if source_path is not None else None,
        source_status=status,
    )


def load_candidates(repo_root: Path, paths: exp10.EvidencePaths) -> dict[str, list[Candidate]]:
    think_rows = {row["game"]: row for row in _read_jsonl(repo_root / THINK_SHARD_REL)}
    out: dict[str, list[Candidate]] = {}
    for game in WINDOWS:
        candidates: list[Candidate] = []
        think = (think_rows.get(game, {}).get("generation") or {}).get("engine_source")
        candidates.append(
            _source_candidate(
                game,
                "THINK",
                "pilot",
                think if isinstance(think, str) else None,
                repo_root / THINK_SHARD_REL,
                "cached" if isinstance(think, str) else "missing_no_completed_pilot_engine",
            )
        )
        for seed in CODEONLY_SEEDS:
            response_path = paths.response(game, seed)
            try:
                source, _ = exp10.recorded_first_shot_source(paths, game, seed)
                status = "cached"
            except Exception as exc:
                source, status = None, f"unavailable:{type(exc).__name__}:{exc}"[:200]
            candidates.append(
                _source_candidate(game, "CODEONLY", f"seed-{seed}", source, response_path, status)
            )
        expert_path = paths.expert_engine(game)
        expert_source = expert_path.read_text()
        candidates.append(
            _source_candidate(
                game, "EXPERT", "positive-control", expert_source, expert_path, "cached"
            )
        )
        identity_source = identity_source_from_expert(expert_source)
        candidates.append(
            _source_candidate(
                game,
                "IDENTITY",
                "negative-control",
                identity_source,
                expert_path,
                "derived_identity_dynamics_with_constant_false_goal",
            )
        )
        out[game] = candidates
    return out


def identity_source_from_expert(expert_source: str) -> str:
    return (
        expert_source
        + "\n\ndef engine(grid, action, data=None):\n    return grid\n"
        + "\ndef is_level_complete(grid):\n    return False\n"
    )


def goal_predicate_status(candidate: Candidate) -> str:
    if candidate.source is None:
        return "source_missing"
    try:
        tree = ast.parse(candidate.source)
    except SyntaxError:
        return "source_invalid"
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "is_level_complete"
    ]
    if not functions:
        return "missing"
    body = functions[-1].body
    statements = [
        node
        for node in body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    if len(statements) == 1 and isinstance(statements[0], ast.Return):
        value = statements[0].value
        if isinstance(value, ast.Constant) and value.value is False:
            return "constant_false"
    return "callable"


def _levelup(row: Any) -> bool:
    return int(row.level_after) > int(row.level_before)


def score_gate_metrics(candidate: Candidate, spec: exp10.WindowSpec) -> dict[str, Any]:
    mask = exp10.build_mask(spec)
    indices = [i for i in spec.heldout_indices if not _levelup(spec.rows[i])]
    compile_error: Optional[str] = None
    if candidate.source is None:
        compile_error = candidate.source_status
    else:
        try:
            compile(candidate.source, f"<{candidate.pair_id}>", "exec")
        except BaseException as exc:
            compile_error = f"{type(exc).__name__}: {exc}"[:200]
    predictions: dict[int, Any] = {}
    run_meta: dict[str, Any] = {}
    if compile_error is None and indices:
        predicted, run_meta = exp10.predict_rows_in_child(
            str(candidate.source),
            [exp10._row_input(spec.rows[i]) for i in indices],
            candidate.pair_id,
        )
        predictions = dict(zip(indices, predicted))

    live_n = live_exact = masked_n = masked_exact = 0
    changing_rows = noop_rows = noop_hallucinated = 0
    raised = raised_changing = raised_noop = 0
    true_changed_cells = correct_changed_cells = spurious_changed_cells = 0
    fidelities: list[float] = []
    per_row: list[dict[str, Any]] = []
    for index in spec.heldout_indices:
        row = spec.rows[index]
        entry: dict[str, Any] = {"row": int(index)}
        if _levelup(row):
            entry["status"] = "levelup_excluded"
            per_row.append(entry)
            continue
        prediction = predictions.get(index)
        pred = None if prediction is None else prediction.grid
        error = "no_engine" if prediction is None else prediction.error
        nxt = np.asarray(row.next_grid)
        valid = pred is not None and np.asarray(pred).shape == nxt.shape
        live_n += 1
        live_match = bool(valid and np.array_equal(pred, nxt))
        live_exact += int(live_match)
        entry["live_exact"] = live_match
        if index in spec.excluded_indices:
            entry["status"] = "excluded_preregistered"
            per_row.append(entry)
            continue

        masked_n += 1
        before = e3.apply_hud_mask(np.asarray(row.grid), mask)
        after = e3.apply_hud_mask(nxt, mask)
        masked_pred = e3.apply_hud_mask(np.asarray(pred), mask) if valid else None
        changing = bool(np.any(before != after))
        exact = bool(masked_pred is not None and np.array_equal(masked_pred, after))
        masked_exact += int(exact)
        entry.update(changing=changing, masked_exact=exact)
        if error:
            raised += 1
            entry["raised"] = str(error)[:200]
        if changing:
            changing_rows += 1
            changed = before != after
            true_changed_cells += int(changed.sum())
            raised_changing += int(bool(error))
            fidelity = 0.0
            if masked_pred is not None:
                written = masked_pred != before
                union = changed | written
                fidelity = float(((masked_pred == after) & union).sum() / union.sum())
                correct_changed_cells += int(((masked_pred == after) & changed).sum())
                spurious_changed_cells += int((written & ~changed).sum())
            fidelities.append(fidelity)
            entry["change_fidelity"] = fidelity
        else:
            noop_rows += 1
            hallucinated = not exact
            noop_hallucinated += int(hallucinated)
            raised_noop += int(bool(error))
            if masked_pred is not None:
                spurious_changed_cells += int((masked_pred != before).sum())
            entry["noop_hallucinated"] = hallucinated
        per_row.append(entry)

    live_accuracy = live_exact / live_n if live_n else None
    masked_accuracy = masked_exact / masked_n if masked_n else None
    change_fidelity = float(np.mean(fidelities)) if fidelities else None
    noop_rate = noop_hallucinated / noop_rows if noop_rows else None
    cell_recall = correct_changed_cells / true_changed_cells if true_changed_cells else None
    effective_noop = 0.0 if noop_rate is None else float(noop_rate)
    existing_change_pass = bool(
        changing_rows > 0
        and correct_changed_cells >= 1
        and change_fidelity is not None
        and change_fidelity >= 0.5
        and effective_noop <= 0.25
    )
    return {
        "engine_status": compile_error or "compiled",
        "live_unmasked_exact_accuracy": live_accuracy,
        "live_unmasked_n": live_n,
        "masked_exact_accuracy": masked_accuracy,
        "masked_n": masked_n,
        "masked_change_fidelity": change_fidelity,
        "n_changing_rows": changing_rows,
        "no_op_hallucination_rate": noop_rate,
        "no_op_rate_for_gate": effective_noop,
        "n_noop_rows": noop_rows,
        "cell_recall": cell_recall,
        "true_changed_cells": true_changed_cells,
        "correct_changed_cells": correct_changed_cells,
        "spurious_changed_cells": spurious_changed_cells,
        "n_raised_rows": raised,
        "n_raised_changing_rows": raised_changing,
        "n_raised_noop_rows": raised_noop,
        "live_selector": {
            "accepted": bool(live_accuracy is not None and live_accuracy >= 1.0),
            "metric": "heldout_unmasked_exact_accuracy",
            "threshold": 1.0,
            "shipped_defaults": True,
        },
        "existing_change_gate": {
            "counterfactual_enabled_decision": existing_change_pass,
            "shipped_enabled": False,
            "fidelity_threshold": 0.5,
            "max_noop_hallucination_rate": 0.25,
            "min_correct_changed_cells": 1,
        },
        "engine_run": run_meta,
        "per_row": per_row,
    }


def candidate_gate_decisions(metrics: Mapping[str, Any]) -> dict[str, bool]:
    live = metrics.get("live_unmasked_exact_accuracy")
    exact = metrics.get("masked_exact_accuracy")
    fidelity = metrics.get("masked_change_fidelity")
    recall = metrics.get("cell_recall")
    noop = float(metrics.get("no_op_rate_for_gate") or 0.0)
    decisions = {
        "live_exact_1.0": bool(live is not None and float(live) >= 1.0),
        "masked_exact_1.0": bool(exact is not None and float(exact) >= 1.0),
        "masked_exact_0.875": bool(exact is not None and float(exact) >= 0.875),
        "masked_exact_0.75": bool(exact is not None and float(exact) >= 0.75),
        "cell_recall_0.9": bool(recall is not None and float(recall) >= 0.9),
        "cell_recall_0.8": bool(recall is not None and float(recall) >= 0.8),
    }
    for threshold in (1.0, 0.9, 0.8, 0.7):
        for noop_limit in (0.0, 0.25):
            name = f"change_fidelity_{threshold}_noop_{noop_limit:g}"
            decisions[name] = bool(
                fidelity is not None
                and int(metrics.get("n_changing_rows") or 0) > 0
                and float(fidelity) >= threshold
                and noop <= noop_limit
            )
    if set(decisions) != set(GATE_NAMES):
        raise AssertionError(f"gate decision mismatch: {sorted(set(GATE_NAMES) ^ set(decisions))}")
    return decisions


def load_engine_namespace(candidate: Candidate) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if candidate.source is None:
        return None, candidate.source_status
    namespace: dict[str, Any] = {"__name__": candidate.pair_id}
    try:
        exec(compile(candidate.source, f"<{candidate.pair_id}>", "exec"), namespace)
    except BaseException as exc:
        return None, f"{type(exc).__name__}: {exc}"[:200]
    if not callable(namespace.get("engine")):
        return None, "missing_callable_engine"
    if not callable(namespace.get("is_level_complete")):
        return None, "missing_callable_is_level_complete"
    return namespace, None


def resolve_environment_files(repo_root: Path) -> Path:
    candidates = [repo_root / "environment_files"]
    if len(repo_root.parents) >= 3:
        candidates.append(repo_root.parents[2] / "environment_files")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0]


def rebuild_real_state(
    repo_root: Path,
    game: str,
    spec: exp10.WindowSpec,
    actions: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
    evidence_path: Optional[Path] = None,
) -> RebuiltState:
    from arcengine import GameAction

    kit.ENV_DIR = resolve_environment_files(repo_root)
    arcade = kit.offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    return rebuild_state_from_actions(
        env=env,
        action_rows=actions,
        induction_action_index=int(alignment["enclosing_action_index"]),
        expected_grid=np.asarray(spec.rows[-1].next_grid),
        action_resolver=lambda name: GameAction[name],
        grid_reader=lambda frame: e3.to_logical(grid_of(frame), spec.cell),
        evidence_path=evidence_path,
    )


def rebuild_reset_state_from_env(
    *,
    env: Any,
    expected_level: int,
    grid_reader: Callable[[Any], np.ndarray],
    expected_grid: Optional[np.ndarray] = None,
    evidence_path: Optional[Path] = None,
) -> RebuiltState:
    frame = env.reset()
    if frame is None:
        observed = np.empty((0, 0), dtype=np.int16)
        level = -1
        reason = "reset_returned_none"
    else:
        observed = np.asarray(grid_reader(frame))
        level = int(getattr(frame, "levels_completed", 0) or 0)
        reason = None if level == int(expected_level) else "reset_level_mismatch"
        if (
            reason is None
            and expected_grid is not None
            and not np.array_equal(observed, np.asarray(expected_grid))
        ):
            reason = "reset_grid_mismatch"
    observed_sha = _grid_sha(observed)
    expected_sha = _grid_sha(observed if expected_grid is None else np.asarray(expected_grid))
    result = RebuiltState(
        recoverable=reason is None,
        reason=reason,
        grid=observed,
        level=level,
        frame=frame,
        env=env,
        actions_replayed=1,
        expected_sha256=expected_sha,
        observed_sha256=observed_sha,
    )
    if evidence_path is not None:
        atomic_json(
            evidence_path,
            {
                "recoverable": result.recoverable,
                "reason": result.reason,
                "reset_sent": True,
                "expected_level": int(expected_level),
                "level_from_frame": level,
                "expected_sha256": expected_sha,
                "observed_sha256": observed_sha,
            },
        )
    return result


def rebuild_reset_state(
    repo_root: Path,
    game: str,
    spec: exp10.WindowSpec,
    expected_level: int,
    evidence_path: Optional[Path] = None,
) -> RebuiltState:
    kit.ENV_DIR = resolve_environment_files(repo_root)
    arcade = kit.offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    return rebuild_reset_state_from_env(
        env=env,
        expected_level=expected_level,
        grid_reader=lambda frame: e3.to_logical(grid_of(frame), spec.cell),
        expected_grid=np.asarray(spec.rows[0].grid),
        evidence_path=evidence_path,
    )


def plan_and_execute_from_state(
    candidate: Candidate,
    rebuilt: RebuiltState,
    *,
    arm: ExecutionArm = LIVE_SCORED,
) -> dict[str, Any]:
    from arcengine import GameAction
    from carnot.agentic.arc_engine_call_guard import EngineCallGuardError, guarded_call

    namespace, load_error = load_engine_namespace(candidate)
    base = {
        "plan_found": False,
        "plan_length": 0,
        "arm": arm.name,
        "arm_live_status": "LIVE" if arm.live else "NOT LIVE",
        "start_state": arm.start_state,
        "steps_executed_before_divergence": 0,
        "matched_steps_before_divergence": 0,
        "matched_steps": 0,
        "real_level_up": False,
        "real_actions_used": 0,
        "real_actions_spent_after_first_divergence": 0,
        "real_state_changed": False,
        "executed_actions": [],
        "diverged": False,
        "first_divergence_index": None,
        "divergence_reason": None,
        "halted_on_divergence": False,
        "execution_termination_reason": "no_plan",
        "planner": {
            "max_nodes": arm.max_nodes,
            "max_depth": arm.max_depth,
            "goal_energy_source": "binary_first_contact",
            "goal_guidance_lambda": GOAL_GUIDANCE_LAMBDA,
        },
    }
    if namespace is None:
        base["planner_error"] = load_error
        return base
    engine = namespace["engine"]
    is_done = namespace["is_level_complete"]

    def binary_goal_energy(grid: np.ndarray) -> float:
        try:
            return 0.0 if bool(is_done(grid)) else GOAL_GUIDANCE_LAMBDA
        except Exception:
            return GOAL_GUIDANCE_LAMBDA

    binary_goal_energy.energy_source = "binary"
    diagnostics: dict[str, Any] = {}
    try:
        with Heartbeat(f"planning {candidate.pair_id}"):
            plan = e3.plan_in_model(
                engine,
                is_done,
                rebuilt.grid.copy(),
                max_nodes=arm.max_nodes,
                max_depth=arm.max_depth,
                goal_energy=binary_goal_energy,
                diagnostics=diagnostics,
            )
    except BaseException as exc:
        base["planner_error"] = f"{type(exc).__name__}: {exc}"[:300]
        base["planner_diagnostics"] = diagnostics
        return base
    base["planner_diagnostics"] = diagnostics
    if not plan:
        return base
    base["plan_found"] = True
    base["plan_length"] = len(plan)
    base["execution_termination_reason"] = "plan_exhausted"
    current = rebuilt.grid.copy()
    start = rebuilt.grid.copy()
    for plan_index, step in enumerate(plan):
        action = int(step["action"])
        data = step.get("data")
        predicted: Optional[np.ndarray] = None
        prediction_error: Optional[str] = None
        try:
            predicted = np.asarray(guarded_call(engine, current.copy(), action, data))
        except EngineCallGuardError as exc:
            prediction_error = f"engine_guard:{type(exc).__name__}:{exc}"[:240]
        except BaseException as exc:
            prediction_error = f"engine_raised:{type(exc).__name__}:{exc}"[:240]
        if prediction_error is not None and arm.halt_on_divergence:
            base["diverged"] = True
            base["first_divergence_index"] = plan_index
            base["divergence_reason"] = prediction_error
            base["halted_on_divergence"] = True
            base["execution_termination_reason"] = "divergence_halt"
            break
        frame = rebuilt.env.step(_game_action(GameAction, action), data=data)
        if frame is None:
            base["diverged"] = True
            base["first_divergence_index"] = plan_index
            base["divergence_reason"] = "environment_returned_none"
            base["execution_termination_reason"] = "environment_returned_none"
            break
        observed = e3.to_logical(grid_of(frame), max(1, 64 // int(current.shape[1])))
        base["real_actions_used"] += 1
        base["executed_actions"].append({"plan_index": plan_index, "action": action, "data": data})
        if not np.array_equal(observed, start):
            base["real_state_changed"] = True
        level_up = _levels_completed(frame) > rebuilt.level
        if level_up and arm.halt_on_divergence:
            base["real_level_up"] = True
            base["level_after"] = _levels_completed(frame)
            base["execution_termination_reason"] = "real_level_up_boundary"
            break
        matched = bool(
            predicted is not None
            and predicted.shape == observed.shape
            and np.array_equal(predicted, observed)
        )
        if matched:
            base["matched_steps"] += 1
            if base["first_divergence_index"] is None:
                base["matched_steps_before_divergence"] += 1
                base["steps_executed_before_divergence"] += 1
        elif base["first_divergence_index"] is None:
            base["diverged"] = True
            base["first_divergence_index"] = plan_index
            if prediction_error is not None:
                base["divergence_reason"] = prediction_error
            elif predicted is not None and predicted.shape != observed.shape:
                base["divergence_reason"] = "prediction_shape_mismatch"
            else:
                base["divergence_reason"] = "prediction_cell_mismatch"
            base["divergence_plan_index"] = plan_index
            if arm.halt_on_divergence:
                base["halted_on_divergence"] = True
                base["execution_termination_reason"] = "divergence_halt"
                break
        current = observed
        if level_up:
            base["real_level_up"] = True
            base["level_after"] = _levels_completed(frame)
            base["execution_termination_reason"] = "real_level_up_boundary"
            break
    first = base["first_divergence_index"]
    if first is not None:
        base["real_actions_spent_after_first_divergence"] = max(
            0, int(base["real_actions_used"]) - int(first) - 1
        )
    return base


def compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"per_row", "engine_run"}}


def _state_evidence(rebuilt: RebuiltState, *, mode: str, reset_sent: bool) -> dict[str, Any]:
    return {
        "recoverable": rebuilt.recoverable,
        "reason": rebuilt.reason,
        "mode": mode,
        "reset_sent": reset_sent,
        "actions_replayed": rebuilt.actions_replayed,
        "level_from_frame": rebuilt.level,
        "expected_sha256": rebuilt.expected_sha256,
        "observed_sha256": rebuilt.observed_sha256,
    }


def rebuild_arm_start_state(
    repo_root: Path,
    game: str,
    spec: exp10.WindowSpec,
    actions: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
    arm: ExecutionArm,
) -> tuple[RebuiltState, dict[str, Any]]:
    induction = rebuild_real_state(repo_root, game, spec, actions, alignment)
    reason = str(alignment.get("induction_reason") or "")
    from_current = execute_plan_from_current(reason)
    if arm.start_state == "induction_state" or from_current:
        mode = "induction_state"
        return induction, _state_evidence(induction, mode=mode, reset_sent=False)
    root = rebuild_reset_state_from_env(
        env=induction.env,
        expected_level=induction.level,
        grid_reader=lambda frame: e3.to_logical(grid_of(frame), spec.cell),
        expected_grid=np.asarray(spec.rows[0].grid),
    )
    root.actions_replayed += induction.actions_replayed
    evidence = _state_evidence(root, mode="root_after_reset", reset_sent=True)
    evidence["pre_reset_induction_state_recoverable"] = induction.recoverable
    evidence["pre_reset_actions_replayed"] = induction.actions_replayed
    return root, evidence


def run_pair(
    repo_root: Path,
    raw_dir: Path,
    candidate: Candidate,
    spec: exp10.WindowSpec,
    actions: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
    window_recoverable: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = score_gate_metrics(candidate, spec)
    decisions = candidate_gate_decisions(metrics)
    executions: dict[str, dict[str, Any]] = {}
    labels_by_arm: dict[str, dict[str, bool]] = {}
    state_evidence: dict[str, dict[str, Any]] = {}
    for arm in EXECUTION_ARMS:
        if window_recoverable:
            rebuilt, evidence = rebuild_arm_start_state(
                repo_root, candidate.game, spec, actions, alignment, arm
            )
            state_evidence[arm.name] = evidence
            if rebuilt.recoverable:
                execution = plan_and_execute_from_state(candidate, rebuilt, arm=arm)
                execution["start_state"] = evidence["mode"]
            else:
                execution = {
                    "arm": arm.name,
                    "plan_found": False,
                    "real_level_up": False,
                    "matched_steps_before_divergence": 0,
                    "real_state_changed": False,
                    "excluded_reason": evidence["reason"] or "state_unrecoverable",
                }
        else:
            execution = {
                "arm": arm.name,
                "plan_found": False,
                "real_level_up": False,
                "matched_steps_before_divergence": 0,
                "real_state_changed": False,
                "excluded_reason": "state_unrecoverable",
            }
        executions[arm.name] = execution
        labels_by_arm[arm.name] = execution_labels(execution)
    execution = executions[LIVE_SCORED.name]
    labels = labels_by_arm[LIVE_SCORED.name]
    pair = {
        "pair_id": candidate.pair_id,
        "game": candidate.game,
        "engine_family": candidate.engine_family,
        "variant": candidate.variant,
        "source_status": candidate.source_status,
        "source_path": candidate.source_path,
        "source_sha256": candidate.source_sha256,
        "gate_metrics": compact_metrics(metrics),
        "gate_decisions": decisions,
        "goal_predicate_status": goal_predicate_status(candidate),
        "executions": executions,
        "labels_by_arm": labels_by_arm,
        "execution": execution,
        "labels": labels,
        "window_status": "pending",
        "arm_status": {},
    }
    raw = {
        **pair,
        "requirement_id": REQUIREMENT_ID,
        "alignment": dict(alignment),
        "state_rebuild_by_arm": state_evidence,
        "per_heldout_row": metrics["per_row"],
        "engine_run": metrics["engine_run"],
    }
    return pair, raw


def _preconditions(repo_root: Path) -> list[dict[str, Any]]:
    resources = [
        repo_root / "docs/research-notes/b2-think-on-pilot-2026-09-24.md",
        repo_root / "docs/research-notes/b2-positive-control-2026-09-23.md",
        repo_root / "results/raw/b2_positive_control_2026_09_23/workflow_result.json",
        repo_root / THINK_SHARD_REL,
        repo_root / LIVE_SESSION_REL,
        repo_root / TELEMETRY_REL,
        repo_root / V1_ARTIFACT_REL,
        repo_root / "python/carnot/experiment_10010_engine_child.py",
    ]
    checks = [
        {"resource": str(path.relative_to(repo_root)), "available": path.is_file()}
        for path in resources
    ]
    environment_dir = resolve_environment_files(repo_root)
    for game in WINDOWS:
        checks.append(
            {
                "resource": str(environment_dir / game),
                "available": (environment_dir / game).is_dir(),
            }
        )
    checks.extend(
        [
            {
                "resource": "CUDA_VISIBLE_DEVICES is empty",
                "available": os.environ.get("CUDA_VISIBLE_DEVICES", "") == "",
            },
            {
                "resource": "JAX_PLATFORMS is cpu",
                "available": os.environ.get("JAX_PLATFORMS") == "cpu",
            },
            {
                "resource": "live planner overrides absent",
                "available": not any(
                    os.environ.get(key)
                    for key in ("CARNOT_ARC_PLAN_MAX_NODES", "CARNOT_ARC_PLAN_MAX_DEPTH")
                ),
            },
        ]
    )
    return checks


def _blocked_artifact(
    *,
    reason: str,
    started: float,
    checks: Sequence[Mapping[str, Any]],
    deviations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "requirement_id": REQUIREMENT_ID,
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "solve_provenance": SOLVE_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "preconditions_checked": list(checks),
        "duration_s": round(time.monotonic() - started, 6),
        "per_pair_rows": [],
        "gate_tables_useful": {},
        "gate_tables_faithful": {},
        "deviations": list(deviations),
        "limits": ["No complete measurement is claimed after a failed precondition."],
    }
    artifact["reproducibility_checksum"] = canonical_checksum(artifact)
    return artifact


def run_experiment(repo_root: Path, output_path: Path, raw_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    deviations: list[dict[str, Any]] = []
    deviations.append(
        {
            "id": "AMENDMENT-1-SCORED-AGENT-PARITY",
            "reason": (
                "The v1 design incorrectly attributed the offline twin's divergence halt and "
                "induction-state start to the scored E3AgentPolicy."
            ),
            "change": (
                "LIVE_SCORED now follows scored start/reset and full-plan replay semantics; "
                "OFFLINE_TWIN_HALT preserves v1; BUDGET_150K is a NOT LIVE diagnostic."
            ),
            "preserved_artifact": str(V1_ARTIFACT_REL),
            "effect_on_design": "Execution measurement only; frozen gates and candidates unchanged.",
        }
    )
    environment_dir = resolve_environment_files(repo_root)
    if environment_dir != (repo_root / "environment_files").resolve():
        deviations.append(
            {
                "id": "DEV-10012-ENVIRONMENT-FILES-LOCATION",
                "reason": "The worktree has no environment_files directory; it is gitignored input data.",
                "change": (
                    "Read the public simulator files read-only from the parent checkout while all "
                    "code, evidence, and outputs remain in this worktree."
                ),
                "path": str(environment_dir),
                "effect_on_design": "Location-only; game bytes and replay method are unchanged.",
            }
        )
    progress("phase preconditions: begin")
    checks = _preconditions(repo_root)
    failed = [row["resource"] for row in checks if not row["available"]]
    if failed:
        artifact = _blocked_artifact(
            reason="precondition_failed", started=started, checks=checks, deviations=deviations
        )
        artifact["failed_preconditions"] = failed
        artifact["reproducibility_checksum"] = canonical_checksum(artifact)
        atomic_json(output_path, artifact)
        progress(f"phase preconditions: blocked {failed}")
        return artifact
    progress("phase preconditions: passed")

    paths = exp10.EvidencePaths.under(repo_root)
    reports = exp10.load_control_reports(paths)
    progress("phase windows: loading frozen reports and splits")
    specs = {
        game: exp10.load_window(game, index, reports[game], paths)
        for index, game in enumerate(WINDOWS)
    }
    candidates = load_candidates(repo_root, paths)
    raw_dir.mkdir(parents=True, exist_ok=True)

    per_pair: list[dict[str, Any]] = []
    per_window: list[dict[str, Any]] = []
    try:
        for game in WINDOWS:
            progress(f"phase window {game}: rebuild pre-induction state")
            spec = specs[game]
            actions = load_episode_actions(repo_root, game)
            alignment = induction_alignment(repo_root, game, actions)
            precheck = rebuild_real_state(
                repo_root,
                game,
                spec,
                actions,
                alignment,
                raw_dir / f"{game}__state_rebuild.json",
            )
            rows: list[dict[str, Any]] = []
            raw_rows: list[tuple[Path, dict[str, Any]]] = []
            for index, candidate in enumerate(candidates[game], start=1):
                progress(
                    f"phase pair {game} {index}/{len(candidates[game])}: "
                    f"{candidate.engine_family}/{candidate.variant}"
                )
                pair, raw = run_pair(
                    repo_root,
                    raw_dir,
                    candidate,
                    spec,
                    actions,
                    alignment,
                    precheck.recoverable,
                )
                rows.append(pair)
                raw_rows.append((raw_dir / f"{candidate.pair_id}.json", raw))
                outcomes = ",".join(
                    f"{arm.name}={pair['labels_by_arm'][arm.name]['USEFUL']}"
                    for arm in EXECUTION_ARMS
                )
                progress(f"phase pair {candidate.pair_id}: {outcomes}")
            if precheck.recoverable:
                arm_status = classify_window_arms(rows)
            else:
                arm_status = {
                    arm.name: {
                        "status": "state_unrecoverable",
                        "reason": precheck.reason or "state_rebuild_failed",
                    }
                    for arm in EXECUTION_ARMS
                }
            for pair, (path, raw) in zip(rows, raw_rows):
                pair["arm_status"] = arm_status
                pair["window_status"] = arm_status[LIVE_SCORED.name]["status"]
                pair["window_status_reason"] = arm_status[LIVE_SCORED.name]["reason"]
                raw["arm_status"] = arm_status
                raw["window_status"] = pair["window_status"]
                raw["window_status_reason"] = pair["window_status_reason"]
                atomic_json(path, raw)
            per_pair.extend(rows)
            per_window.append(
                {
                    "game": game,
                    "status": arm_status[LIVE_SCORED.name]["status"],
                    "reason": arm_status[LIVE_SCORED.name]["reason"],
                    "arm_status": arm_status,
                    "induction_reason": alignment["induction_reason"],
                    "execute_plan_from_current": execute_plan_from_current(
                        alignment["induction_reason"]
                    ),
                    "live_start_state": plan_start_state(alignment["induction_reason"]),
                    "state_recoverable": precheck.recoverable,
                    "state_rebuild_reason": precheck.reason,
                    "alignment": alignment,
                    "actions_replayed": precheck.actions_replayed,
                    "expected_grid_sha256": precheck.expected_sha256,
                    "observed_grid_sha256": precheck.observed_sha256,
                }
            )
    except IdentityUsefulHarnessBug as exc:
        artifact = _blocked_artifact(
            reason="identity_useful_harness_bug",
            started=started,
            checks=checks,
            deviations=deviations,
        )
        artifact["error"] = str(exc)
        artifact["per_pair_rows"] = per_pair
        artifact["per_window"] = per_window
        artifact["reproducibility_checksum"] = canonical_checksum(artifact)
        atomic_json(output_path, artifact)
        progress("phase labels: blocked because IDENTITY was useful")
        return artifact

    progress("phase reduction: per-arm fixed USEFUL and FAITHFUL gate tables")
    tables_by_arm: dict[str, Any] = {}
    for arm in EXECUTION_ARMS:
        tables_by_arm[arm.name] = {
            "with_experts": {
                "USEFUL": build_gate_tables(
                    per_pair, GATE_NAMES, label="USEFUL", arm_name=arm.name
                ),
                "FAITHFUL": build_gate_tables(
                    per_pair, GATE_NAMES, label="FAITHFUL", arm_name=arm.name
                ),
            },
            "without_experts": {
                "USEFUL": build_gate_tables(
                    per_pair,
                    GATE_NAMES,
                    label="USEFUL",
                    arm_name=arm.name,
                    candidate_only=True,
                ),
                "FAITHFUL": build_gate_tables(
                    per_pair,
                    GATE_NAMES,
                    label="FAITHFUL",
                    arm_name=arm.name,
                    candidate_only=True,
                ),
            },
        }
    primary_tables = tables_by_arm[LIVE_SCORED.name]
    useful_tables = primary_tables["with_experts"]["USEFUL"]
    faithful_tables = primary_tables["with_experts"]["FAITHFUL"]
    informative = sum(row["status"] == "label_informative" for row in per_window)
    unrecoverable = sum(row["status"] == "state_unrecoverable" for row in per_window)
    artifact = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "requirement_id": REQUIREMENT_ID,
        "honest_verdict": (
            f"complete_gate_usefulness_measured_{informative}_informative_windows_"
            f"{unrecoverable}_unrecoverable"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "solve_provenance": SOLVE_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "preconditions_checked": checks,
        "duration_s": round(time.monotonic() - started, 6),
        "windows": list(WINDOWS),
        "per_window": per_window,
        "per_pair_rows": per_pair,
        "candidate_gates": list(GATE_NAMES),
        "execution_arms": {
            arm.name: {
                "max_nodes": arm.max_nodes,
                "max_depth": arm.max_depth,
                "halt_on_divergence": arm.halt_on_divergence,
                "start_state": arm.start_state,
                "live_status": "LIVE" if arm.live else "NOT LIVE",
            }
            for arm in EXECUTION_ARMS
        },
        "gate_tables_by_arm": tables_by_arm,
        "gate_table_cohorts": {
            "with_experts": "THINK, CODEONLY, EXPERT, and IDENTITY rows",
            "without_experts": (
                "Real candidates only: THINK and CODEONLY; both EXPERT and IDENTITY controls removed"
            ),
        },
        "gate_tables_useful": useful_tables,
        "gate_tables_faithful": faithful_tables,
        "gate_tables_useful_without_experts": primary_tables["without_experts"]["USEFUL"],
        "gate_tables_faithful_without_experts": primary_tables["without_experts"]["FAITHFUL"],
        "deviations": deviations,
        "environment_files": str(environment_dir),
        "live_code_citations": {
            "planner_defaults_and_search": "python/carnot/agentic/arc_executable_world_model.py:9269",
            "scored_execute_replays_without_prediction_check": (
                "python/carnot/agentic/arc_competition_agent.py:7681-7690"
            ),
            "scored_plan_start_choice": "python/carnot/agentic/arc_competition_agent.py:8110-8124",
            "scored_reset_before_plan": "python/carnot/agentic/arc_competition_agent.py:7669-7678",
            "offline_twin_halt_order": (
                "python/carnot/agentic/arc_executable_world_model.py:9527-9607"
            ),
            "selector": "python/carnot/agentic/arc_world_model_trust_energy.py:731",
        },
        "limits": [
            "Public offline environments are a development proxy, not hidden-game efficacy.",
            "Each cached induction response is one draw from one prompt.",
            "Expert engines are source-derived and level-specific positive controls.",
            "Candidate goal predicates are evaluated with their cached dynamics; goal quality is not isolated.",
            "BUDGET_150K is a diagnostic arm and is NOT LIVE.",
            "No p-values were computed and no threshold was selected from these outcomes.",
            "The optional live-explorer context comparison was omitted as non-essential extra simulator work.",
            "No live gate or agent default was changed.",
        ],
    }
    artifact["reproducibility_checksum"] = canonical_checksum(artifact)
    atomic_json(output_path, artifact)
    progress(f"phase artifact: wrote {output_path}")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output = args.output or repo_root / ARTIFACT_REL
    raw_dir = args.raw_dir or repo_root / RAW_REL
    artifact = run_experiment(repo_root, output, raw_dir)
    print(json.dumps({"honest_verdict": artifact["honest_verdict"], "output": str(output)}))
    return 0 if str(artifact["honest_verdict"]).startswith("complete_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
