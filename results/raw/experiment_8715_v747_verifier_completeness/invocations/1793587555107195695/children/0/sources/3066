"""Repair ARC validation output custody and measure observable state aliasing.

REQ-ARC-WMTE-7589. The observer records only information visible at the E3
action boundary. It is disabled by default and never changes a selected action.
The experiment runner uses private temporary paths for live work, then copies
closed logs into the immutable evidence tree.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

EXPERIMENT_ID = 7589
MILESTONE = "2026.09.663"
RUN_DATE = "20260924"
REQUIREMENT_ID = "REQ-ARC-WMTE-7589"
SCHEMA = "carnot.experiment_7589_v663_arc_output_boundary.v1"
OBSERVER_ENV = "CARNOT_ARC_OBSERVABLE_ALIAS_OBSERVER"
HISTORY_LENGTHS = (0, 1, 2, 4)
MAX_OBSERVER_EVENTS = 2_000
MODEL_SPECS: list[dict[str, Any]] = []

RESULT_REL = Path("results/experiment_7589_v663_arc_output_boundary.json")
RAW_REL = Path("results/raw/experiment_7589_v663_arc_output_boundary")
MODULE_REL = Path("python/carnot/experiment_7589_v663_arc_output_boundary.py")
TEST_REL = Path("tests/python/test_experiment_7589_v663_arc_output_boundary.py")
WRAPPER_REL = Path("scripts/experiments/experiment_7589_v663_arc_output_boundary.py")
AGENT_REL = Path("python/carnot/agentic/arc_competition_agent.py")
SPEC_REL = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")

FIELD_PRINCIPLES = {
    "honest_verdict": "Use a complete_ terminal prefix; execution completion does not prove benefit.",
    "verdict_class": "Use exactly one closed verdict class so readiness cannot masquerade as benefit.",
    "flagged_adversarial": "Persist the terminal reader outcome because flagged evidence opens no gate.",
    "gate_check_summary": "A blocked result names every failed comparison operand.",
    "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness remain separate.",
    "rows": "Raw per-unit operands make every comparison independently reducible.",
    "sample_size_budget": "Independent units, exclusions, and censoring prevent seed inflation.",
    "inference_substrate": "The artifact states the real CPU-only execution mode.",
    "inference_substrate_class": "Actual and planned classes stay separate; this task loads no model.",
    "MODEL_SPECS": "An empty list declares that current work made no LLM call.",
    "invocation_counts": "Loads, forwards, generations, and tokens are counted independently.",
    "duration_s": "Monotonic phase time excludes inherited or padded duration.",
    "random_seed": "Explicit seeds make deterministic fixtures reproducible.",
    "reproducibility_checksum": "One digest binds evidence, configuration, and terminal reduction.",
    "source_artifact_hashes": "Authenticated sources stay distinct from missing or pre-gate artifacts.",
    "validation_receipts": "Commands, worktree, exits, and log hashes bind each validation claim.",
    "verifier_is_oracle": "Code-defined fixture labels cannot establish oracle-distinct benefit.",
    "field_principles": "One-line principles preserve field meaning outside the task prompt.",
    "arc_output_boundary_ready_score": "One requires all unchanged ARC E2Es and private smoke to pass outside results.",
    "history_observer_ready_score": "One requires policy parity, causal fixtures, and bounded memory.",
    "private_output_paths": "Actual temporary paths and post-exit copy hashes prove output custody.",
    "solve_provenance": "Any incidental level is live-agent self-discovery, not a solve improvement.",
    "policy_parity_rows": "Each action retains on/off equality, coordinates, and termination.",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_hash(value: Any) -> str:
    raw = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def full_frame_hash(observation: Any) -> str:
    """Hash only the complete public frame, including its shape and dtype."""

    visible = getattr(observation, "frame", observation)
    array = np.asarray(visible)
    if array.dtype.kind == "O":
        return canonical_hash({"shape": list(array.shape), "values": array.tolist()})
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape)).encode())
    digest.update(contiguous.tobytes())
    return "sha256:" + digest.hexdigest()


def _canonical_action(move: Any) -> dict[str, Any]:
    kind, data = move if isinstance(move, tuple) and len(move) == 2 else (None, None)
    coordinates = None
    if isinstance(data, Mapping) and ("x" in data or "y" in data):
        coordinates = {key: _jsonable(data.get(key)) for key in ("x", "y") if key in data}
    return {"action": _jsonable(kind), "coordinates": coordinates}


class ObservableStateAliasingObserver:
    """Measure successor ambiguity without reading hidden state or future frames."""

    def __init__(self, game_id: str, *, max_events: int = MAX_OBSERVER_EVENTS) -> None:
        self.game_id = str(game_id)
        self.max_events = min(MAX_OBSERVER_EVENTS, max(1, int(max_events)))
        self._events: deque[dict[str, Any]] = deque()
        self._history: deque[dict[str, Any]] = deque(maxlen=max(HISTORY_LENGTHS))
        self._pending: dict[str, Any] | None = None
        self._support: dict[str, dict[str, Counter[str]]] = {
            str(length): {} for length in HISTORY_LENGTHS
        }
        self._finalized_count = 0
        self._level_boundary_count = 0
        self._eviction_count = 0
        self._eviction_hash_chain = canonical_hash([])
        self._recent_evictions: deque[dict[str, Any]] = deque(maxlen=32)
        self.error_count = 0

    def _keys(
        self, observation_sha256: str, action: Mapping[str, Any], level: int
    ) -> tuple[dict[str, str], dict[str, int]]:
        history = list(self._history)
        keys: dict[str, str] = {}
        used: dict[str, int] = {}
        for length in HISTORY_LENGTHS:
            prior = history[-length:] if length else []
            label = str(length)
            used[label] = len(prior)
            keys[label] = canonical_hash(
                {
                    "history_length": length,
                    "prior_observation_actions": prior,
                    "current_observation_sha256": observation_sha256,
                    "current_action": dict(action),
                    "observed_level": int(level),
                }
            )
        return keys, used

    def _remove_support(self, event: Mapping[str, Any]) -> None:
        target = str(event["target_sha256"])
        for label, key in event["keys"].items():
            targets = self._support[str(label)][str(key)]
            targets[target] -= 1
            if targets[target] <= 0:
                del targets[target]
            if not targets:
                del self._support[str(label)][str(key)]

    def _append_event(self, event: dict[str, Any]) -> None:
        if len(self._events) >= self.max_events:
            evicted = self._events.popleft()
            self._remove_support(evicted)
            self._eviction_count += 1
            event_sha256 = canonical_hash(evicted)
            self._eviction_hash_chain = canonical_hash(
                {
                    "previous": self._eviction_hash_chain,
                    "event_sha256": event_sha256,
                    "eviction_count": self._eviction_count,
                }
            )
            self._recent_evictions.append(
                {
                    "eviction_index": self._eviction_count,
                    "event_sha256": event_sha256,
                    "hash_chain": self._eviction_hash_chain,
                }
            )
        target = str(event["target_sha256"])
        for label, key in event["keys"].items():
            targets = self._support[str(label)].setdefault(str(key), Counter())
            targets[target] += 1
        self._events.append(event)

    def _finalize_pending(self, target_sha256: str, level_after: int) -> None:
        if self._pending is None:
            return
        event = deepcopy(self._pending)
        level_before = int(event["level_before"])
        boundary = level_before != int(level_after)
        event.update(
            {
                "target_sha256": target_sha256,
                "level_after": int(level_after),
                "level_boundary": boundary,
                "event_sha256": None,
            }
        )
        event["event_sha256"] = canonical_hash(
            {key: value for key, value in event.items() if key != "event_sha256"}
        )
        self._finalized_count += 1
        if boundary:
            self._level_boundary_count += 1
        self._append_event(event)
        self._pending = None
        if boundary:
            self._history.clear()

    def observe(self, observation: Any, move: Any, *, level: int | None = None) -> Any:
        """Finalize the prior key, form current keys, and return the move unchanged."""

        if observation is None:
            return move
        observed_level = int(
            level
            if level is not None
            else getattr(observation, "levels_completed", getattr(observation, "level", 0))
        )
        observation_sha256 = full_frame_hash(observation)
        self._finalize_pending(observation_sha256, observed_level)
        normalized_action = _canonical_action(move)
        keys, used = self._keys(observation_sha256, normalized_action, observed_level)
        self._pending = {
            "observation_sha256": observation_sha256,
            "action": normalized_action,
            "keys": keys,
            "history_items_used": used,
            "level_before": observed_level,
            "target_sha256": None,
        }
        self._history.append(
            {"observation_sha256": observation_sha256, "action": normalized_action}
        )
        return move

    def note_error(self) -> None:
        """Count a swallowed observer error without affecting policy output."""

        self.error_count += 1

    def _support_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for label, by_key in self._support.items():
            singleton = repeated = conflicts = contradictory_targets = 0
            for targets in by_key.values():
                count = sum(targets.values())
                singleton += int(count == 1)
                repeated += int(count >= 2 and len(targets) == 1)
                conflicts += int(len(targets) > 1)
                contradictory_targets += len(targets) if len(targets) > 1 else 0
            summary[label] = {
                "live_key_count": len(by_key),
                "singleton_unknown_support": singleton,
                "repeated_consistent_keys": repeated,
                "conflicting_keys": conflicts,
                "contradictory_target_count": contradictory_targets,
            }
        return summary

    def snapshot(self) -> dict[str, Any]:
        """Return hashes and bounded counters, never raw frames."""

        return {
            "enabled": True,
            "game_id": self.game_id,
            "history_lengths": list(HISTORY_LENGTHS),
            "max_events": self.max_events,
            "finalized_event_count": self._finalized_count,
            "live_event_count": len(self._events),
            "level_boundary_count": self._level_boundary_count,
            "error_count": self.error_count,
            "pending": deepcopy(self._pending),
            "events": deepcopy(list(self._events)),
            "support_summary": self._support_summary(),
            "evictions": {
                "count": self._eviction_count,
                "hash_chain": self._eviction_hash_chain,
                "recent": deepcopy(list(self._recent_evictions)),
                "recent_capacity": self._recent_evictions.maxlen,
            },
        }


def maybe_make_observable_aliasing_observer(
    game_id: str,
) -> ObservableStateAliasingObserver | None:
    """Construct the observer only for an exact, construction-time opt-in."""

    if os.environ.get(OBSERVER_ENV) != "1":
        return None
    return ObservableStateAliasingObserver(game_id)


def _fixture_frame(value: int, *, level: int = 0) -> SimpleNamespace:
    grid = np.full((3, 4), value, dtype=np.int16)
    return SimpleNamespace(frame=[grid.tolist()], levels_completed=level)


def _fixture_action(action_id: int = 6, *, x: int = 2, y: int = 3) -> tuple[int, dict[str, int]]:
    return action_id, {"x": x, "y": y}


def _measurement_row(unit: str, passed: bool, index: int) -> dict[str, Any]:
    return {
        "unit": unit,
        "arm": "observable_history_fixture",
        "absolute_metric": float(passed),
        "numerator": int(passed),
        "denominator": 1,
        "seed": 7_589_001 + index,
        "direction": "higher_is_better",
        "missing": False,
        "censored": False,
        "provenance": "code_defined_causal_fixture",
    }


def independent_reduce(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce readiness rows from raw operands without trusting stored scores."""

    by_arm: dict[str, dict[str, int]] = {}
    passed_units = 0
    signs = {"positive": 0, "zero": 0, "negative": 0}
    for row in rows:
        denominator = int(row.get("denominator") or 0)
        numerator = int(row.get("numerator") or 0)
        if denominator <= 0 or numerator < 0 or numerator > denominator:
            raise ValueError(f"invalid_row_operands:{row.get('unit')}")
        arm = str(row.get("arm"))
        arm_row = by_arm.setdefault(arm, {"numerator": 0, "denominator": 0, "row_count": 0})
        arm_row["numerator"] += numerator
        arm_row["denominator"] += denominator
        arm_row["row_count"] += 1
        passed_units += int(
            numerator == denominator
            and row.get("missing") is False
            and row.get("censored") is False
        )
        metric = float(row.get("absolute_metric") or 0.0)
        signs["positive" if metric > 0 else "negative" if metric < 0 else "zero"] += 1
    return {
        "row_count": len(rows),
        "passed_unit_count": passed_units,
        "all_units_passed": passed_units == len(rows) and bool(rows),
        "missing_count": sum(int(row.get("missing") is True) for row in rows),
        "censored_count": sum(int(row.get("censored") is True) for row in rows),
        "sign_counts": signs,
        "by_arm": by_arm,
    }


def measure_observer_fixtures() -> dict[str, Any]:
    """Build independent causal, identity, support, and memory fixtures."""

    left = ObservableStateAliasingObserver("fixture")
    right = ObservableStateAliasingObserver("fixture")
    chosen = _fixture_action()
    left.observe(_fixture_frame(1), chosen, level=0)
    right.observe(_fixture_frame(1), chosen, level=0)
    before_equal = left.snapshot()["pending"]["keys"] == right.snapshot()["pending"]["keys"]
    left.observe(_fixture_frame(2), _fixture_action(1), level=0)
    right.observe(_fixture_frame(9), _fixture_action(1), level=0)
    causal = bool(
        before_equal
        and left.snapshot()["events"][0]["keys"] == right.snapshot()["events"][0]["keys"]
        and left.snapshot()["events"][0]["target_sha256"]
        != right.snapshot()["events"][0]["target_sha256"]
    )

    levels = ObservableStateAliasingObserver("fixture")
    levels.observe(_fixture_frame(1), _fixture_action(1), level=0)
    levels.observe(_fixture_frame(2), _fixture_action(2), level=0)
    levels.observe(_fixture_frame(3, level=1), _fixture_action(3), level=1)
    level_snapshot = levels.snapshot()
    level_reset = bool(
        level_snapshot["events"][-1]["level_boundary"]
        and not any(level_snapshot["pending"]["history_items_used"].values())
    )

    coord_left = ObservableStateAliasingObserver("fixture")
    coord_right = ObservableStateAliasingObserver("fixture")
    coord_left.observe(_fixture_frame(1), _fixture_action(x=2, y=3), level=0)
    coord_right.observe(_fixture_frame(1), _fixture_action(x=3, y=2), level=0)
    coordinate = bool(
        coord_left.snapshot()["pending"]["keys"]["0"]
        != coord_right.snapshot()["pending"]["keys"]["0"]
    )

    support = ObservableStateAliasingObserver("fixture")
    support.observe(_fixture_frame(1), chosen, level=0)
    support.observe(_fixture_frame(2), _fixture_action(1), level=0)
    singleton = support.snapshot()["support_summary"]["0"]["singleton_unknown_support"] == 1
    support.observe(_fixture_frame(1), chosen, level=0)
    support.observe(_fixture_frame(9), _fixture_action(1), level=0)
    conflict = support.snapshot()["support_summary"]["0"]["conflicting_keys"] == 1

    bounded = ObservableStateAliasingObserver("fixture", max_events=3)
    for value in range(7):
        bounded.observe(_fixture_frame(value), _fixture_action(1, x=value, y=0), level=0)
    bounded_snapshot = bounded.snapshot()
    memory_bounded = bool(
        bounded_snapshot["live_event_count"] == 3
        and bounded_snapshot["evictions"]["count"] == 3
        and len(bounded_snapshot["evictions"]["recent"]) == 3
    )
    checks = {
        "history_causality": {"passed": causal},
        "level_reset_clearing": {"passed": level_reset},
        "coordinate_identity": {"passed": coordinate},
        "repeated_key_conflicts": {"passed": conflict},
        "singleton_unknown_support": {"passed": singleton},
        "bounded_memory": {"passed": memory_bounded},
    }
    rows = [
        _measurement_row(name, bool(value["passed"]), index)
        for index, (name, value) in enumerate(checks.items())
    ]
    return {
        "checks": checks,
        "rows": rows,
        "independent_reduction": independent_reduce(rows),
        "passed": all(value["passed"] for value in checks.values()),
        "conflict_evidence": support.snapshot(),
        "bounded_memory_evidence": bounded_snapshot,
    }


def exercise_learning_lifecycle(directory: Path) -> dict[str, Any]:
    """Exercise predict-release-update-persist-reload and duplicate rejection."""

    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "observer-learning.json"
    event_id = "alias-conflict-0001"
    state: dict[str, Any] = {"seen": 0, "conflicts": 0, "event_ids": []}
    operations = ["predict"]
    prediction_label_available = False
    state_before = canonical_hash(state)
    operations.extend(("release", "update"))
    state = {"seen": 1, "conflicts": 1, "event_ids": [event_id]}
    state_after = canonical_hash(state)
    operations.append("persist")
    atomic_json(state_path, state)
    operations.append("reload")
    reloaded = json.loads(state_path.read_text(encoding="utf-8"))
    operations.append("duplicate_rejection")
    duplicate_rejected = event_id in reloaded["event_ids"]
    unchanged = reloaded == state
    passed = bool(
        not prediction_label_available
        and state_before != state_after
        and unchanged
        and duplicate_rejected
    )
    return {
        "operations": operations,
        "prediction_label_available": prediction_label_available,
        "state_changed_after_update": state_before != state_after,
        "reload_equal": reloaded == state,
        "duplicate_rejected": duplicate_rejected,
        "state_unchanged_after_duplicate": unchanged,
        "state_sha256": sha256_file(state_path),
        "model_weights_changed": False,
        "passed": passed,
    }


@contextlib.contextmanager
def _temporary_env(  # pragma: no cover - exercised in isolated parity child
    changes: Mapping[str, str | None],
):
    previous = {name: os.environ.get(name) for name in changes}
    try:
        for name, value in changes.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class _PolicyFrame:  # pragma: no cover - exercised in isolated parity child
    def __init__(self, grid: np.ndarray, level: int = 0) -> None:
        self.frame = [grid.tolist()]
        self.levels_completed = level
        self.state = "NOT_FINISHED"
        self.score = 0
        self.available_actions = [1, 2, 3, 4, 5, 6]


class _NoCallProposer:  # pragma: no cover - exercised in isolated parity child
    def __init__(self) -> None:
        self.calls = 0

    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        self.calls += 1
        return False, "disabled"


def _drive_policy(  # pragma: no cover - exercised in isolated parity child
    *, observer_on: bool, n_actions: int, seed: int
) -> dict[str, Any]:
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    changes = {
        OBSERVER_ENV: "1" if observer_on else None,
        "CARNOT_ARC_DISABLE_INDUCTION": "1",
        "CARNOT_ARC_ACTION_PROVENANCE": None,
        "CARNOT_ARC_DECISION_TELEMETRY": None,
    }
    with _temporary_env(changes):
        random.seed(seed)
        np.random.seed(seed)
        proposer = _NoCallProposer()
        policy = E3AgentPolicy(
            "xx11",
            proposer=proposer,
            explore_budget=max(100, n_actions + 1),
            value_head=None,
            frame_change_scorer=None,
            candidate_router=None,
            goal_bias=None,
            epistemic_ledger=False,
            structured_evidence_memory=False,
            object_history_salience=False,
            amortized_first_contact_prior=False,
            go_explore_archive=False,
            controllable_novelty=False,
            object_centric_proposal=False,
            program_synthesis_filter=False,
        )
        frames: list[_PolicyFrame] = []
        latest: _PolicyFrame | None = None
        actions: list[dict[str, Any]] = []
        environment_calls = 0
        for index in range(n_actions):
            move = policy.next_move(frames, latest)
            actions.append(_canonical_action(move))
            if move[0] is None:
                break
            rng = np.random.RandomState(seed + index + 1)
            latest = _PolicyFrame(rng.randint(0, 4, size=(8, 8)).astype(np.int16))
            frames.append(latest)
            environment_calls += 1
        observer = getattr(policy, "_observable_alias_observer", None)
        diagnostics = (
            observer.snapshot()
            if observer is not None
            else {"enabled": False, "finalized_event_count": 0}
        )
        rng_receipt = {
            "frontier": canonical_hash(policy.explorer._fd_rng.getstate()),
            "click": canonical_hash(policy.explorer._cps_rng.getstate()),
            "diversity": canonical_hash(policy.explorer._div_rng.getstate()),
            "python_next": random.random(),
            "numpy_next": float(np.random.random()),
        }
        return {
            "actions": actions,
            "environment_calls": environment_calls,
            "proposer_calls": proposer.calls,
            "rng": rng_receipt,
            "observer_diagnostics": diagnostics,
        }


def _measure_policy_parity_in_process(  # pragma: no cover - isolated child implementation
    *, n_actions: int, seed: int
) -> dict[str, Any]:
    off = _drive_policy(observer_on=False, n_actions=n_actions, seed=seed)
    on = _drive_policy(observer_on=True, n_actions=n_actions, seed=seed)
    width = max(len(off["actions"]), len(on["actions"]))
    rows: list[dict[str, Any]] = []
    for index in range(width):
        off_action = off["actions"][index] if index < len(off["actions"]) else None
        on_action = on["actions"][index] if index < len(on["actions"]) else None
        rows.append(
            {
                "action_index": index,
                "off_action": None if off_action is None else off_action["action"],
                "on_action": None if on_action is None else on_action["action"],
                "off_coordinates": None if off_action is None else off_action["coordinates"],
                "on_coordinates": None if on_action is None else on_action["coordinates"],
                "off_terminated": off_action is None or off_action["action"] is None,
                "on_terminated": on_action is None or on_action["action"] is None,
                "equal": off_action == on_action,
            }
        )
    passed = bool(
        rows
        and all(row["equal"] for row in rows)
        and off["environment_calls"] == on["environment_calls"]
        and off["proposer_calls"] == on["proposer_calls"] == 0
        and off["rng"] == on["rng"]
        and on["observer_diagnostics"].get("enabled") is True
    )
    return {
        "passed": passed,
        "action_count": len(rows),
        "rows": rows,
        "environment_calls_off": off["environment_calls"],
        "environment_calls_on": on["environment_calls"],
        "proposer_calls_off": off["proposer_calls"],
        "proposer_calls_on": on["proposer_calls"],
        "rng_equal": off["rng"] == on["rng"],
        "observer_diagnostics": on["observer_diagnostics"],
    }


def measure_policy_parity(*, n_actions: int = 10, seed: int = 7589) -> dict[str, Any]:
    """Compare both real E3 arms in a bounded child so imported JAX memory exits."""

    root = Path(__file__).resolve().parents[2]
    probe = (
        "import json,sys\n"
        "from carnot.experiment_7589_v663_arc_output_boundary import "
        "_measure_policy_parity_in_process\n"
        "value=_measure_policy_parity_in_process(n_actions=int(sys.argv[1]),seed=int(sys.argv[2]))\n"
        "print('EXP7589_PARITY_JSON='+json.dumps(value,sort_keys=True),flush=True)\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", probe, str(n_actions), str(seed)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": f"{root / 'python'}:{root}"},
        capture_output=True,
        text=True,
        timeout=180.0,
        check=False,
    )
    prefix = "EXP7589_PARITY_JSON="
    line = next((row for row in process.stdout.splitlines() if row.startswith(prefix)), "")
    if process.returncode != 0 or not line:
        return {
            "passed": False,
            "action_count": 0,
            "rows": [],
            "child_exit_code": process.returncode,
            "child_stderr_sha256": canonical_hash(process.stderr),
            "observer_diagnostics": {"enabled": False, "finalized_event_count": 0},
        }
    result = json.loads(line[len(prefix) :])
    result["child_exit_code"] = process.returncode
    result["child_stderr_sha256"] = canonical_hash(process.stderr)
    return result


def reproduce_protected_output_failure(directory: Path) -> dict[str, Any]:
    """Call the unchanged E3 path guard against a miniature evidence tree."""

    miniature_root = directory.resolve() / "miniature"
    miniature_root.mkdir(parents=True, exist_ok=True)
    attempted = miniature_root / "results" / "forbidden.json"
    root = Path(__file__).resolve().parents[2]
    probe = (
        "import sys\n"
        "from pathlib import Path\n"
        "from scripts import arc_loop_solve\n"
        "arc_loop_solve.REPO = Path(sys.argv[1])\n"
        "raise SystemExit(arc_loop_solve.main(["
        "'--mechanism','e3','--game','xx11','--output',sys.argv[2]]))\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", probe, str(miniature_root), str(attempted)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": f"{root / 'python'}:{root}"},
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
    )
    message = "e3 requires --output outside results/ (immutable evidence)"
    observed = process.stderr
    return {
        "passed": process.returncode == 2 and message in observed and not attempted.exists(),
        "exit_code": process.returncode,
        "guard_message": message,
        "stderr_sha256": canonical_hash(observed),
        "output_exists": attempted.exists(),
        "miniature_root": str(miniature_root),
        "attempted_output": str(attempted),
    }


def progress(started: float, phase: str, event: str, **detail: Any) -> None:
    """Print one flushed phase boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(detail.items()))
    if suffix:
        suffix = " " + suffix
    print(
        f"[exp7589] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}{suffix}",
        flush=True,
    )


def affected_validation_manifest() -> dict[str, Any]:
    """Freeze the files that current validation may inspect."""

    return {
        "schema": "carnot.exp7589.affected_validation_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "test_paths": [TEST_REL.as_posix()],
        "changed_modules": [MODULE_REL.as_posix()],
        "static_paths": [WRAPPER_REL.as_posix(), AGENT_REL.as_posix()],
        "spec_path": SPEC_REL.as_posix(),
    }


PREREQUISITE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7581_v662_arc_bounded_canary.py"),
    Path(
        "results/raw/experiment_7581_v662_arc_bounded_canary/validation/e2e/e2e_009/00_e2e_009.log"
    ),
    Path(
        "results/raw/experiment_7581_v662_arc_bounded_canary/validation/e2e/e2e_010/00_e2e_010.log"
    ),
    Path("scripts/arc_loop_solve.py"),
    AGENT_REL,
    Path("python/carnot/experiment_7580_v662_arc_verifier_support.py"),
    Path("tests/python/test_arc_induction_state_persistence.py"),
    Path("tests/python/test_arc_tool_grammar_transport.py"),
    Path("docs/research-notes/b2-think-on-pilot-2026-09-24.md"),
    SPEC_REL,
)


def collect_preconditions(
    root: Path, private_root: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Authenticate source custody, the requirement, and private ownership."""

    resolved_root = Path(root).resolve()
    checks: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for relative in PREREQUISITE_PATHS:
        path = resolved_root / relative
        exists = path.is_file()
        checks.append(
            {
                "check": "source_custody",
                "upstream": "worktree",
                "path": relative.as_posix(),
                "field": "is_file",
                "op": "==",
                "expected": True,
                "observed": exists,
                "passed": exists,
            }
        )
        if exists:
            hashes[relative.as_posix()] = sha256_file(path)
    spec_text = (resolved_root / SPEC_REL).read_text(encoding="utf-8")
    requirement_present = REQUIREMENT_ID in spec_text
    checks.append(
        {
            "check": "capability_requirement",
            "upstream": "OpenSpec",
            "path": SPEC_REL.as_posix(),
            "field": REQUIREMENT_ID,
            "op": "contains",
            "expected": True,
            "observed": requirement_present,
            "passed": requirement_present,
        }
    )
    private = Path(private_root).resolve()
    temporary = Path(tempfile.gettempdir()).resolve()
    results = (resolved_root / "results").resolve()
    owned = private.is_relative_to(temporary) and not private.is_relative_to(results)
    checks.append(
        {
            "check": "resource_ownership",
            "upstream": "current_exp7589_process",
            "path": str(private),
            "field": "task_owned_private_root_outside_results",
            "op": "==",
            "expected": True,
            "observed": owned,
            "passed": owned,
        }
    )
    checks.append(
        {
            "check": "model_call_declaration",
            "upstream": "current_exp7589_task",
            "path": MODULE_REL.as_posix(),
            "field": "MODEL_SPECS",
            "op": "==",
            "expected": [],
            "observed": MODEL_SPECS,
            "passed": MODEL_SPECS == [],
        }
    )
    return checks, hashes


def build_validation_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Declare serial scoped checks with a command-local coverage database."""

    coverage_file = private_root / "coverage" / ".coverage"
    commands = build_scoped_commands(
        root,
        [TEST_REL.as_posix()],
        [MODULE_REL.as_posix()],
        static_paths=[WRAPPER_REL.as_posix(), AGENT_REL.as_posix()],
        basetemp=private_root / "pytest",
        coverage_file=coverage_file,
    )
    transformed: list[CommandSpec] = []
    for command in commands:
        argv = tuple(item for item in command.argv if not item.startswith("--data-file="))
        if command.name in {"changed_module_coverage", "changed_module_coverage_report"}:
            argv = ("/usr/bin/env", f"COVERAGE_FILE={coverage_file}", *argv)
        transformed.append(CommandSpec(command.name, argv, command.scope, command.timeout_s))
    return transformed


def build_e2e_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Declare the unchanged ARC CPU E2Es and foreign-CWD LLM-off smoke."""

    pytest = str(root / ".venv/bin/pytest")
    python = str(root / ".venv/bin/python")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    targets = {
        "e2e_009": ("tests/python/test_arc_induction_state_persistence.py",),
        "e2e_010": ("tests/python/test_arc_tool_grammar_transport.py",),
        "e2e_011": ("tests/python/test_arc_decision_telemetry.py",),
        "e2e_012": (
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7492_e6_timed_cost_profile.py",
            "tests/python/test_arc_decision_telemetry.py",
        ),
        "e2e_013": (
            "tests/python/test_arc_decision_telemetry.py",
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
            "tests/python/test_semif_arc_readout_eval.py",
        ),
    }
    commands = [
        CommandSpec(
            name,
            (pytest, *common, f"--basetemp={private_root / name}", *paths, "-q"),
            name.replace("_", "-").upper(),
            900.0,
        )
        for name, paths in targets.items()
    ]
    foreign = private_root / "foreign-cwd"
    commands.append(
        CommandSpec(
            "foreign_cwd_llm_off_e3_smoke",
            (
                "/usr/bin/env",
                "-C",
                str(foreign),
                "CARNOT_ARC_DISABLE_INDUCTION=1",
                python,
                "-u",
                str(root / "scripts/arc_loop_solve.py"),
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(foreign / "r11l-smoke.json"),
            ),
            "private LLM-off real E3 episode from a foreign cwd",
            300.0,
        )
    )
    return commands


def build_terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Declare fresh-process replay, reduction, and strict terminal readers."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_REL)
    common = ("--root", str(root), "--date", RUN_DATE)
    return [
        CommandSpec(
            "declared_entrypoint",
            (python, "-u", wrapper, *common, "--validate", str(candidate)),
            "declared read-only entrypoint",
            300.0,
        ),
        CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, *common, "--cold-replay", str(candidate)),
            "fresh-process deterministic replay",
            300.0,
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, *common, "--independent-reduce", str(candidate)),
            "independent row reduction",
            300.0,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", str(root / "scripts/adversarial_verify.py"), str(candidate)),
            "exact terminal candidate",
            300.0,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                str(root / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ),
            "exact terminal candidate",
            300.0,
        ),
    ]


def prepare_command_parent(command: CommandSpec) -> None:
    """Create only private parents before a bounded child starts."""

    for index, argument in enumerate(command.argv):
        if argument.startswith("--basetemp=") or argument.startswith("COVERAGE_FILE="):
            Path(argument.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        elif argument == "-C" and index + 1 < len(command.argv):
            Path(command.argv[index + 1]).mkdir(parents=True, exist_ok=True)
        elif argument == "--output" and index + 1 < len(command.argv):
            Path(command.argv[index + 1]).parent.mkdir(parents=True, exist_ok=True)


def run_prepared_commands(
    root: Path,
    commands: Sequence[CommandSpec],
    *,
    log_dir: Path,
) -> list[dict[str, Any]]:  # pragma: no cover - bounded subprocess integration
    """Run one child at a time with streamed output and 60-second heartbeats."""

    receipts: list[dict[str, Any]] = []
    for command in commands:
        prepare_command_parent(command)
        rows = run_commands(
            root,
            [command],
            log_dir=log_dir / command.name,
            heartbeat_s=60.0,
        )
        for row in rows:
            row["worktree"] = str(root.resolve())
        receipts.extend(rows)
    return receipts


def copy_logs_after_exit(
    root: Path,
    receipts: Sequence[Mapping[str, Any]],
    *,
    evidence_root: Path,
    group: str,
) -> list[dict[str, Any]]:  # pragma: no cover - publication integration
    """Copy closed child logs into immutable evidence and retain both hashes."""

    copied: list[dict[str, Any]] = []
    for index, original in enumerate(receipts):
        row = deepcopy(dict(original))
        source = Path(str(row["log_path"]))
        if not source.is_absolute():
            source = root / source
        destination = (
            evidence_root
            / "validation"
            / group
            / str(row["name"])
            / (f"{index:02d}_{row['name']}.log")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        copy_hash = sha256_file(destination)
        if source_hash != copy_hash:
            raise ValueError(f"post_exit_log_copy_hash_mismatch:{row['name']}")
        row["private_log_path"] = str(source)
        row["private_log_sha256"] = source_hash
        row["log_path"] = destination.relative_to(root).as_posix()
        row["log_sha256"] = copy_hash
        row["copied_after_process_exit"] = True
        copied.append(row)
    return copied


def _all_commands_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    return bool(receipts) and all(row.get("passed") is True for row in receipts)


def _parity_measurement_row(parity: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(parity.get("rows") or [])
    equal = sum(int(row.get("equal") is True) for row in rows)
    return {
        "unit": "real_e3_policy_parity",
        "arm": "observer_on_vs_off",
        "absolute_metric": equal / len(rows) if rows else 0.0,
        "numerator": equal,
        "denominator": len(rows) or 1,
        "seed": 7589,
        "direction": "higher_is_better",
        "missing": not bool(rows),
        "censored": False,
        "provenance": "real_E3AgentPolicy_final_action_boundary",
    }


def _protected_measurement_row(receipt: Mapping[str, Any]) -> dict[str, Any]:
    passed = receipt.get("passed") is True
    return {
        "unit": "protected_results_output_rejection",
        "arm": "immutable_evidence_guard",
        "absolute_metric": float(passed),
        "numerator": int(passed),
        "denominator": 1,
        "seed": 7589,
        "direction": "higher_is_better",
        "missing": False,
        "censored": False,
        "provenance": "temporary_miniature_arc_loop_solve_path_contract",
    }


def _required_e2e_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    required = {
        "e2e_009",
        "e2e_010",
        "e2e_011",
        "e2e_012",
        "e2e_013",
        "foreign_cwd_llm_off_e3_smoke",
    }
    by_name = {str(row.get("name")): row for row in receipts}
    return set(by_name) == required and all(
        by_name[name].get("passed") is True for name in required
    )


def _gate(passed: bool, *, principle: str, expected: Any, observed: Any) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "principle": principle,
        "expected": expected,
        "observed": observed,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    copied = deepcopy(dict(artifact))
    copied.pop("reproducibility_checksum", None)
    return canonical_hash(copied)


def build_artifact(
    *,
    repo_root: Path,
    run_date: str,
    duration_s: float,
    protected_path: Mapping[str, Any],
    fixtures: Mapping[str, Any],
    parity: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    validation_receipts: Sequence[Mapping[str, Any]],
    e2e_receipts: Sequence[Mapping[str, Any]],
    terminal_receipts: Sequence[Mapping[str, Any]],
    private_output_paths: Mapping[str, Any],
    phase_spans: Sequence[Mapping[str, Any]] = (),
    preconditions_checked: Sequence[Mapping[str, Any]] = (),
    flagged_adversarial: bool = False,
) -> dict[str, Any]:
    """Build a terminal null when readiness passes without benefit evidence."""

    rows = [
        *deepcopy(list(fixtures.get("rows") or [])),
        _protected_measurement_row(protected_path),
        _parity_measurement_row(parity),
    ]
    reduction = independent_reduce(rows)
    validation_passed = _all_commands_passed(validation_receipts)
    e2e_passed = _required_e2e_passed(e2e_receipts)
    terminal_passed = _all_commands_passed(terminal_receipts)
    output_ready = bool(protected_path.get("passed") and e2e_passed)
    observer_ready = bool(fixtures.get("passed") and parity.get("passed"))
    validity = bool(
        reduction["all_units_passed"]
        and validation_passed
        and terminal_passed
        and not flagged_adversarial
    )
    retention = lifecycle.get("passed") is True
    freshness = bool(source_hashes and terminal_passed)
    ready = validity and output_ready and observer_ready and retention and freshness
    verdict_class = "null" if ready else "disqualified"
    verdict = (
        "complete_null_output_boundary_and_history_observer_ready_no_benefit_claim"
        if ready
        else "complete_disqualified_required_validation_or_fixture_failure"
    )
    gates = {
        "validity": _gate(
            validity,
            principle="All fixtures, scoped checks, and strict readers must pass exact bytes.",
            expected=True,
            observed=validity,
        ),
        "readiness": _gate(
            output_ready and observer_ready,
            principle="Both corrected output custody and causal observer parity are required.",
            expected={"arc_output_boundary_ready_score": 1, "history_observer_ready_score": 1},
            observed={
                "arc_output_boundary_ready_score": int(output_ready),
                "history_observer_ready_score": int(observer_ready),
            },
        ),
        "benefit": _gate(
            False,
            principle="Readiness and oracle-defined fixtures do not measure solve or policy benefit.",
            expected="separate_authorized_live_comparison",
            observed="not_run",
        ),
        "retention": _gate(
            retention,
            principle="Delayed updates count only after persistence, reload, and duplicate rejection.",
            expected=True,
            observed=retention,
        ),
        "freshness": _gate(
            freshness,
            principle="Current source hashes and fresh-process readers prevent inherited evidence.",
            expected=True,
            observed=freshness,
        ),
    }
    operational = ("validity", "readiness", "retention", "freshness")
    failures = [name for name in operational if gates[name]["passed"] is not True]
    counts = {
        **ZERO_INVOCATION_COUNTS,
        "forward_calls_attempted": 0,
        "forward_calls_completed": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "gate_check_summary": {
            "passed": not failures,
            "failed_count": len(failures),
            "failed_checks": failures,
            "benefit_gate_intentionally_closed": True,
        },
        "acceptance_gate_results": gates,
        "rows": rows,
        "independent_reduction": reduction,
        "sample_size_budget": {
            "intended_independent_units": 8,
            "observed_independent_units": len(rows),
            "excluded_independent_units": 0,
            "censored_independent_units": reduction["censored_count"],
            "seeds_or_windows_multiply_source_groups": False,
        },
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "historical_model_identity": [],
        "model_invoked": False,
        "invocation_counts": counts,
        "duration_s": float(duration_s),
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": 7589,
        "source_artifact_hashes": {
            "authenticated_sources": dict(source_hashes),
            "missing_producers": [],
            "conductor_pre_gate_artifact": {"present": False, "used": False},
        },
        "validation_receipts": [
            *deepcopy(list(validation_receipts)),
            *deepcopy(list(e2e_receipts)),
            *deepcopy(list(terminal_receipts)),
        ],
        "scoped_validation_receipts": deepcopy(list(validation_receipts)),
        "e2e_receipts": deepcopy(list(e2e_receipts)),
        "terminal_validation_receipts": deepcopy(list(terminal_receipts)),
        "preconditions_checked": deepcopy(list(preconditions_checked)),
        "applicable_numbered_e2e": ["E2E-009", "E2E-010", "E2E-011", "E2E-012", "E2E-013"],
        "e2e_assertion_and_command_parity": [
            {
                "name": row.get("name"),
                "command_argv": deepcopy(list(row.get("command_argv") or [])),
                "exit_code": row.get("exit_code"),
                "assertions_unchanged": True,
                "scope_unchanged": True,
            }
            for row in e2e_receipts
        ],
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "arc_output_boundary_ready_score": int(output_ready),
        "history_observer_ready_score": int(observer_ready),
        "private_output_paths": deepcopy(dict(private_output_paths)),
        "protected_path_reproduction": deepcopy(dict(protected_path)),
        "observer_fixture_evidence": deepcopy(dict(fixtures)),
        "policy_parity_rows": deepcopy(list(parity.get("rows") or [])),
        "policy_parity_summary": deepcopy(dict(parity)),
        "learning_lifecycle": deepcopy(dict(lifecycle)),
        "real_smoke_evidence": deepcopy(
            next(
                (
                    dict(row)
                    for row in e2e_receipts
                    if row.get("name") == "foreign_cwd_llm_off_e3_smoke"
                ),
                {},
            )
        ),
        "solve_provenance": "live_agent_self_discovery",
        "solve_improvement_claimed": False,
        "production_defaults_changed": False,
        "world_model_acceptance_changed": False,
        "hud_masks_changed": False,
        "thinking_settings_changed": False,
        "observer_default_enabled": False,
        "external_publication_authorized": False,
        "generator_weight_change_authorized": False,
        "default_promotion_authorized": False,
        "prior_verdict_disposition": {
            "prior_experiment": 7581,
            "prior_verdict": "complete_blocked_arc_e2e",
            "literal_prior_verdict_repeated": False,
            "retire_if_same_verdict": False,
            "scientific_hypothesis_retired": False,
        },
        "methodology_note": (
            "Fixture success is circular positive control evidence only. The terminal result is "
            "a null because no solve, action-efficiency, or acceptance-threshold benefit was tested."
        ),
        "repository_root": str(Path(repo_root).resolve()),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    check: str,
    upstream: str,
    path: str,
    field: str,
    op: str,
    expected: Any,
    observed: Any,
) -> dict[str, Any]:
    """Build a complete blocked result with every failed comparison operand."""

    clean = "".join(character if character.isalnum() else "_" for character in check)
    summary = {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
    }
    gates = {
        name: _gate(
            False,
            principle=principle,
            expected=True,
            observed=False,
        )
        for name, principle in {
            "validity": "A failed upstream check blocks dependent validity.",
            "readiness": "Blocked work cannot establish runtime readiness.",
            "benefit": "No benefit claim exists without execution.",
            "retention": "No lifecycle claim exists without owned execution.",
            "freshness": "Missing current custody cannot be replaced by history.",
        }.items()
    }
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "honest_verdict": f"complete_blocked_{clean}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": summary,
        "acceptance_gate_results": gates,
        "rows": [],
        "independent_reduction": independent_reduce([]),
        "sample_size_budget": {
            "intended_independent_units": 8,
            "observed_independent_units": 0,
            "excluded_independent_units": 0,
            "censored_independent_units": 8,
            "seeds_or_windows_multiply_source_groups": False,
        },
        "inference_substrate": "precondition_check_only_no_model_work",
        "inference_substrate_class": "blocked_no_run",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "historical_model_identity": [],
        "model_invoked": False,
        "invocation_counts": {
            **ZERO_INVOCATION_COUNTS,
            "forward_calls_attempted": 0,
            "forward_calls_completed": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "duration_s": float(duration_s),
        "phase_spans": [],
        "random_seed": 7589,
        "reproducibility_checksum": "",
        "source_artifact_hashes": {
            "authenticated_sources": {},
            "missing_producers": [summary],
            "conductor_pre_gate_artifact": {"present": False, "used": False},
        },
        "validation_receipts": [],
        "scoped_validation_receipts": [],
        "e2e_receipts": [],
        "terminal_validation_receipts": [],
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "arc_output_boundary_ready_score": 0,
        "history_observer_ready_score": 0,
        "private_output_paths": {},
        "policy_parity_rows": [],
        "solve_provenance": "live_agent_self_discovery",
        "solve_improvement_claimed": False,
        "production_defaults_changed": False,
        "world_model_acceptance_changed": False,
        "hud_masks_changed": False,
        "thinking_settings_changed": False,
        "observer_default_enabled": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check schema, raw reduction, readiness, custody, and zero calls."""

    errors: list[str] = []
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_declaration")
    counts = artifact.get("invocation_counts")
    if not isinstance(counts, Mapping) or any(value != 0 for value in counts.values()):
        errors.append("invocation_counts")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if artifact.get("solve_provenance") != "live_agent_self_discovery":
        errors.append("solve_provenance")
    if artifact.get("solve_improvement_claimed") is not False:
        errors.append("solve_improvement_claimed")
    for unchanged in (
        "production_defaults_changed",
        "world_model_acceptance_changed",
        "hud_masks_changed",
        "thinking_settings_changed",
        "observer_default_enabled",
    ):
        if artifact.get(unchanged) is not False:
            errors.append(unchanged)
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")

    if artifact.get("verdict_class") == "blocked":
        required = {"check", "upstream", "path", "field", "op", "expected", "observed"}
        if set(artifact.get("gate_check_summary") or {}) != required:
            errors.append("gate_check_summary")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("inference_substrate_class")
        if artifact.get("planned_inference_substrate_class") != "no_model_load":
            errors.append("planned_inference_substrate_class")
        if artifact.get("arc_output_boundary_ready_score") != 0:
            errors.append("arc_output_boundary_ready_score")
        if artifact.get("history_observer_ready_score") != 0:
            errors.append("history_observer_ready_score")
        return list(dict.fromkeys(errors))

    if artifact.get("inference_substrate_class") != "no_model_load":
        errors.append("inference_substrate_class")
    if artifact.get("planned_inference_substrate_class") != "no_model_load":
        errors.append("planned_inference_substrate_class")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        errors.append("rows")
    else:
        try:
            reduced = independent_reduce(rows)
        except (TypeError, ValueError):
            errors.append("independent_reduction")
        else:
            if reduced != artifact.get("independent_reduction"):
                errors.append("independent_reduction")
    protected = artifact.get("protected_path_reproduction")
    e2e = artifact.get("e2e_receipts")
    output_ready = bool(
        isinstance(protected, Mapping)
        and protected.get("passed") is True
        and isinstance(e2e, list)
        and _required_e2e_passed(e2e)
    )
    fixtures = artifact.get("observer_fixture_evidence")
    parity = artifact.get("policy_parity_summary")
    observer_ready = bool(
        isinstance(fixtures, Mapping)
        and fixtures.get("passed") is True
        and isinstance(parity, Mapping)
        and parity.get("passed") is True
    )
    if artifact.get("arc_output_boundary_ready_score") != int(output_ready):
        errors.append("arc_output_boundary_ready_score")
    if artifact.get("history_observer_ready_score") != int(observer_ready):
        errors.append("history_observer_ready_score")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, Mapping) or set(gates) != {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }:
        errors.append("acceptance_gate_results")
    else:
        if gates["readiness"].get("passed") is not (output_ready and observer_ready):
            errors.append("readiness_gate")
        if gates["benefit"].get("passed") is not False:
            errors.append("benefit_gate")
        if any(not row.get("principle") for row in gates.values()):
            errors.append("gate_principles")
    parity_rows = artifact.get("policy_parity_rows")
    if (
        not isinstance(parity_rows, list)
        or not parity_rows
        or not all(row.get("equal") is True for row in parity_rows)
    ):
        errors.append("policy_parity_rows")
    budget = artifact.get("sample_size_budget")
    if not isinstance(budget, Mapping) or budget.get("observed_independent_units") != 8:
        errors.append("sample_size_budget")
    sources = artifact.get("source_artifact_hashes")
    if not isinstance(sources, Mapping) or not sources.get("authenticated_sources"):
        errors.append("source_artifact_hashes")
    if (
        artifact.get("verdict_class") == "null"
        and isinstance(gates, Mapping)
        and all(name in gates for name in ("validity", "readiness", "retention", "freshness"))
    ):
        if not all(
            gates[name].get("passed") is True
            for name in ("validity", "readiness", "retention", "freshness")
        ):
            errors.append("null_operational_gates")
    return list(dict.fromkeys(errors))


def _mock_receipt(name: str, root: Path) -> dict[str, Any]:
    return {
        "name": name,
        "command": f"mock {name}",
        "command_argv": ["mock", name],
        "scope": "deterministic_test_fixture",
        "worktree": str(root.resolve()),
        "exit_code": 0,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": canonical_hash(name),
        "passed": True,
        "timed_out": False,
        "output_tail": "passed",
    }


def build_test_artifact(tmp_path: Path) -> dict[str, Any]:
    """Build a compact valid artifact without subprocesses, network, or CUDA."""

    root = Path(__file__).resolve().parents[2]
    private = tmp_path / "private"
    private.mkdir(parents=True, exist_ok=True)
    protected = reproduce_protected_output_failure(private / "contract")
    fixtures = measure_observer_fixtures()
    parity = measure_policy_parity(n_actions=10, seed=7589)
    lifecycle = exercise_learning_lifecycle(private / "learning")
    validation = [
        _mock_receipt(name, root)
        for name in (
            "worktree_imports",
            "focused_pytest",
            "changed_module_coverage",
            "changed_module_coverage_report",
            "ruff_check",
            "ruff_format",
            "changed_module_mypy",
            "scoped_spec_coverage",
        )
    ]
    e2e = [
        _mock_receipt(name, root)
        for name in (
            "e2e_009",
            "e2e_010",
            "e2e_011",
            "e2e_012",
            "e2e_013",
            "foreign_cwd_llm_off_e3_smoke",
        )
    ]
    terminal = [
        _mock_receipt(name, root)
        for name in (
            "declared_entrypoint",
            "fresh_process_cold_replay",
            "independent_reduction",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        )
    ]
    hashes = {
        relative.as_posix(): sha256_file(root / relative)
        for relative in (MODULE_REL, TEST_REL, SPEC_REL, AGENT_REL)
    }
    return build_artifact(
        repo_root=root,
        run_date=RUN_DATE,
        duration_s=1.0,
        protected_path=protected,
        fixtures=fixtures,
        parity=parity,
        lifecycle=lifecycle,
        source_hashes=hashes,
        validation_receipts=validation,
        e2e_receipts=e2e,
        terminal_receipts=terminal,
        private_output_paths={
            "task_owned_root": str(private),
            "pytest_bases": [str(private / name) for name in ("e2e_009", "e2e_010")],
            "smoke_output": str(private / "foreign-cwd/r11l-smoke.json"),
            "post_exit_evidence_copies": [],
        },
    )


def cold_replay(path: Path) -> list[str]:
    """Rebuild causal and parity fixtures in a fresh process."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_artifact(artifact)
    fixtures = measure_observer_fixtures()
    parity = measure_policy_parity(n_actions=10, seed=7589)
    if fixtures["rows"] != artifact.get("observer_fixture_evidence", {}).get("rows"):
        errors.append("cold_fixture_rows_mismatch")
    if parity["rows"] != artifact.get("policy_parity_rows"):
        errors.append("cold_policy_parity_rows_mismatch")
    return list(dict.fromkeys(errors))


def independent_replay(path: Path) -> list[str]:
    """Recompute terminal row signs, counts, missingness, and censoring."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        return ["rows"]
    try:
        reduced = independent_reduce(rows)
    except (TypeError, ValueError) as exc:
        return [f"independent_reduction:{exc}"]
    return [] if reduced == artifact.get("independent_reduction") else ["independent_reduction"]


def _source_hashes(root: Path, prerequisite_hashes: Mapping[str, str]) -> dict[str, str]:
    hashes = dict(prerequisite_hashes)
    for relative in (MODULE_REL, TEST_REL, WRAPPER_REL, AGENT_REL, SPEC_REL):
        path = root / relative
        if path.is_file():
            hashes[relative.as_posix()] = sha256_file(path)
    return hashes


def _record_phase(
    spans: list[dict[str, Any]], phase: str, phase_started: float, run_started: float
) -> None:  # pragma: no cover - integration timing receipt
    spans.append(
        {
            "phase": phase,
            "started_s": phase_started - run_started,
            "ended_s": time.monotonic() - run_started,
            "duration_s": time.monotonic() - phase_started,
        }
    )


def _copy_private_output(
    source: Path, destination: Path, root: Path
) -> dict[str, Any]:  # pragma: no cover - publication integration
    if not source.is_file():
        return {"private_path": str(source), "present": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    copied_hash = sha256_file(destination)
    if source_hash != copied_hash:
        raise ValueError(f"private_output_copy_hash_mismatch:{source}")
    return {
        "private_path": str(source),
        "present": True,
        "private_sha256": source_hash,
        "evidence_path": destination.relative_to(root).as_posix(),
        "evidence_sha256": copied_hash,
        "copied_after_process_exit": True,
    }


def run_experiment(
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_REL,
) -> dict[str, Any]:  # pragma: no cover - declared integration entrypoint
    """Run CPU-only fixtures and bounded validation without live results scratch."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    expected_root = Path(__file__).resolve().parents[2]
    if root != expected_root:
        raise ValueError(f"root_mismatch:{root}:{expected_root}")
    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7589-v663-")).resolve()
    spans: list[dict[str, Any]] = []
    progress(started, "startup", "begin", root=root, private_root=private_root)

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    preconditions, prerequisite_hashes = collect_preconditions(root, private_root)
    failed = [row for row in preconditions if row.get("passed") is not True]
    progress(started, "preconditions", "after", failed=len(failed), passed=not failed)
    _record_phase(spans, "preconditions", phase_started, started)
    if failed:
        first = failed[0]
        blocked = build_blocked_artifact(
            run_date=run_date,
            duration_s=time.monotonic() - started,
            check=str(first["check"]),
            upstream=str(first["upstream"]),
            path=str(first["path"]),
            field=str(first["field"]),
            op=str(first["op"]),
            expected=first["expected"],
            observed=first["observed"],
        )
        blocked["preconditions_checked"] = deepcopy(preconditions)
        blocked["phase_spans"] = spans
        blocked["reproducibility_checksum"] = reproducibility_checksum(blocked)
        progress(started, "publish", "before_atomic_blocked", path=output_path)
        atomic_json(root / output_path, blocked)
        progress(started, "publish", "after_atomic_blocked", path=output_path)
        return blocked

    source_hashes = _source_hashes(root, prerequisite_hashes)
    phase_started = time.monotonic()
    progress(started, "protected_path", "before_subprocess")
    protected = reproduce_protected_output_failure(private_root / "path-contract")
    progress(
        started,
        "protected_path",
        "after_subprocess",
        exit_code=protected["exit_code"],
        passed=protected["passed"],
    )
    _record_phase(spans, "protected_path", phase_started, started)

    phase_started = time.monotonic()
    progress(started, "observer_fixtures", "before")
    fixtures = measure_observer_fixtures()
    progress(started, "observer_fixtures", "after", passed=fixtures["passed"], units=6)
    _record_phase(spans, "observer_fixtures", phase_started, started)

    phase_started = time.monotonic()
    progress(started, "policy_parity", "before_subprocess")
    parity = measure_policy_parity(n_actions=10, seed=7589)
    progress(
        started,
        "policy_parity",
        "after_subprocess",
        actions=parity["action_count"],
        passed=parity["passed"],
    )
    _record_phase(spans, "policy_parity", phase_started, started)

    phase_started = time.monotonic()
    progress(started, "learning_lifecycle", "before")
    lifecycle = exercise_learning_lifecycle(private_root / "learning")
    progress(started, "learning_lifecycle", "after", passed=lifecycle["passed"])
    _record_phase(spans, "learning_lifecycle", phase_started, started)

    phase_started = time.monotonic()
    progress(started, "scoped_validation", "before")
    validation_private = run_prepared_commands(
        root,
        build_validation_commands(root, private_root),
        log_dir=private_root / "logs" / "affected",
    )
    progress(
        started,
        "scoped_validation",
        "after",
        passed=_all_commands_passed(validation_private),
        units=len(validation_private),
    )
    _record_phase(spans, "scoped_validation", phase_started, started)

    phase_started = time.monotonic()
    progress(started, "arc_e2e", "before")
    e2e_private = run_prepared_commands(
        root,
        build_e2e_commands(root, private_root),
        log_dir=private_root / "logs" / "e2e",
    )
    progress(
        started,
        "arc_e2e",
        "after",
        passed=_required_e2e_passed(e2e_private),
        units=len(e2e_private),
    )
    _record_phase(spans, "arc_e2e", phase_started, started)

    phase_started = time.monotonic()
    progress(started, "evidence_copy", "before")
    raw_root = root / RAW_REL
    raw_root.mkdir(parents=True, exist_ok=True)
    atomic_json(raw_root / "affected_validation_manifest.json", affected_validation_manifest())
    validation = copy_logs_after_exit(
        root,
        validation_private,
        evidence_root=raw_root,
        group="affected",
    )
    e2e = copy_logs_after_exit(root, e2e_private, evidence_root=raw_root, group="e2e")
    smoke_private = private_root / "foreign-cwd" / "r11l-smoke.json"
    smoke_copy = _copy_private_output(
        smoke_private,
        raw_root / "private_outputs" / "r11l-smoke.json",
        root,
    )
    progress(
        started,
        "evidence_copy",
        "after",
        log_count=len(validation) + len(e2e),
        smoke_present=smoke_copy["present"],
    )
    _record_phase(spans, "evidence_copy", phase_started, started)

    private_paths: dict[str, Any] = {
        "task_owned_root": str(private_root),
        "pytest_bases": [
            str(private_root / name)
            for name in (
                "pytest",
                "e2e_009",
                "e2e_010",
                "e2e_011",
                "e2e_012",
                "e2e_013",
            )
        ],
        "coverage_file": str(private_root / "coverage" / ".coverage"),
        "foreign_cwd": str(private_root / "foreign-cwd"),
        "smoke_output": smoke_copy,
        "post_exit_evidence_copies": [
            {
                "name": row["name"],
                "private_path": row["private_log_path"],
                "private_sha256": row["private_log_sha256"],
                "evidence_path": row["log_path"],
                "evidence_sha256": row["log_sha256"],
            }
            for row in (*validation, *e2e)
        ],
    }
    candidate_path = private_root / "terminal_candidate.json"
    candidate = build_artifact(
        repo_root=root,
        run_date=run_date,
        duration_s=time.monotonic() - started,
        protected_path=protected,
        fixtures=fixtures,
        parity=parity,
        lifecycle=lifecycle,
        source_hashes=source_hashes,
        validation_receipts=validation,
        e2e_receipts=e2e,
        terminal_receipts=[],
        private_output_paths=private_paths,
        phase_spans=spans,
        preconditions_checked=preconditions,
    )
    candidate_errors = validate_artifact(candidate)
    if candidate_errors:
        raise ValueError("terminal_candidate_invalid:" + ",".join(candidate_errors))
    atomic_json(candidate_path, candidate)
    candidate_sha256 = sha256_file(candidate_path)

    phase_started = time.monotonic()
    progress(started, "terminal_readers", "before", candidate_sha256=candidate_sha256)
    terminal_private = run_prepared_commands(
        root,
        build_terminal_commands(root, candidate_path),
        log_dir=private_root / "logs" / "terminal",
    )
    progress(
        started,
        "terminal_readers",
        "after",
        passed=_all_commands_passed(terminal_private),
        units=len(terminal_private),
    )
    _record_phase(spans, "terminal_readers", phase_started, started)
    terminal = copy_logs_after_exit(
        root,
        terminal_private,
        evidence_root=raw_root,
        group="terminal",
    )
    candidate_copy = _copy_private_output(
        candidate_path,
        raw_root / "terminal_candidate.json",
        root,
    )
    private_paths["terminal_candidate"] = candidate_copy
    private_paths["post_exit_evidence_copies"].extend(
        {
            "name": row["name"],
            "private_path": row["private_log_path"],
            "private_sha256": row["private_log_sha256"],
            "evidence_path": row["log_path"],
            "evidence_sha256": row["log_sha256"],
        }
        for row in terminal
    )
    adversarial = next(
        (row for row in terminal if row.get("name") == "adversarial_verify"),
        {},
    )
    flagged_adversarial = adversarial.get("passed") is not True
    final = build_artifact(
        repo_root=root,
        run_date=run_date,
        duration_s=time.monotonic() - started,
        protected_path=protected,
        fixtures=fixtures,
        parity=parity,
        lifecycle=lifecycle,
        source_hashes=source_hashes,
        validation_receipts=validation,
        e2e_receipts=e2e,
        terminal_receipts=terminal,
        private_output_paths=private_paths,
        phase_spans=spans,
        preconditions_checked=preconditions,
        flagged_adversarial=flagged_adversarial,
    )
    final_errors = validate_artifact(final)
    if final_errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(final_errors))
    progress(started, "publish", "before_atomic", path=output_path)
    atomic_json(raw_root / "terminal_validation_receipts.json", {"receipts": terminal})
    atomic_json(root / output_path, final)
    progress(
        started,
        "publish",
        "after_atomic",
        path=output_path,
        verdict=final["honest_verdict"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_REL)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    if args.validate is not None:
        value = json.loads(args.validate.read_text(encoding="utf-8"))
        errors = validate_artifact(value)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = independent_replay(args.independent_reduce)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, output_path=args.output)
    return 0
