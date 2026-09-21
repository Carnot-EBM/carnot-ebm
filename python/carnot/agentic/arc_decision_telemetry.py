"""Bounded shadow telemetry for the live ARC decision seams.

The candidate seam is `arc_competition_agent.py:2812`. The induction seam is
`arc_competition_agent.py:6134`. The supervisor seam is
`arc_competition_agent.py:6013`. The main world-model seams are
`arc_competition_agent.py:8664` and `arc_competition_agent.py:8760`. Bounded
reinduction uses `arc_llm_reinduction.py:1862`. Episode rows start at
`arc_competition_agent.py:6882` and end at `arc_competition_agent.py:9820`.
These hooks only observe values the policy already computed.

Spec: REQ-ARC-WMTE-7465.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, TypeVar


TELEMETRY_ENV_FLAG = "CARNOT_ARC_DECISION_TELEMETRY"
TELEMETRY_PATH_ENV = "CARNOT_ARC_DECISION_TELEMETRY_PATH"
DEFAULT_OUTPUT_DIR_ENVS = (
    "CARNOT_ARC_ACTION_PROVENANCE_DIR",
    "CARNOT_ARC_LIVENESS_DIR",
)
SCHEMA_VERSION = 1
MAX_OPTIONS_PER_RECORD = 15
MAX_STATE_TEXT_CHARS = 512
MAX_RECORD_BYTES = 16 * 1024
MAX_RECORDS_PER_EPISODE = 4096
MAX_TOTAL_BYTES_PER_RUN = 32 * 1024 * 1024
FLUSH_EVERY_RECORDS = 64
FORBIDDEN_KEYS = frozenset(
    {
        "game_source",
        "game_source_code",
        "source_code",
        "hidden_information",
        "hidden_state",
        "future_frame",
        "future_frames",
        "adapter_data",
        "game_adapter",
        "per_game_adapter",
    }
)

_TRUNCATION_RESERVE_BYTES = 1024
_LOGGER = logging.getLogger(__name__)
_WARNING_LOCK = threading.Lock()
_MISSING_PATH_WARNING_EMITTED = False
_PATH_STATE_LOCK = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}
_RUN_BYTES: dict[str, int] = {}
_F = TypeVar("_F", bound=Callable[..., Any])


def telemetry_enabled() -> bool:
    """Return true only for the exact operator opt-in value."""

    return os.environ.get(TELEMETRY_ENV_FLAG) == "1"


def reset_warning_state_for_tests() -> None:
    """Reset the process warning latch for one isolated unit test."""

    global _MISSING_PATH_WARNING_EMITTED
    with _WARNING_LOCK:
        _MISSING_PATH_WARNING_EMITTED = False


def _warn_missing_path_once() -> None:
    global _MISSING_PATH_WARNING_EMITTED
    with _WARNING_LOCK:
        if _MISSING_PATH_WARNING_EMITTED:
            return
        _MISSING_PATH_WARNING_EMITTED = True
    _LOGGER.warning(
        "ARC decision telemetry refused activation because no safe output directory exists."
    )


def _safe_default_path() -> Path | None:
    for name in DEFAULT_OUTPUT_DIR_ENVS:
        raw = os.environ.get(name)
        if not raw:
            continue
        candidate = Path(raw)
        if candidate.is_dir():
            return candidate / "arc_decision_telemetry.jsonl"
    return None


def _forbidden_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return normalized in FORBIDDEN_KEYS or any(
        marker in normalized
        for marker in ("game_source", "hidden_information", "future_frame", "adapter")
    )


def _clip_text(value: str) -> str:
    return value[:MAX_STATE_TEXT_CHARS]


def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip_text(value)
    if isinstance(value, Path):
        return _clip_text(str(value))
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, child in list(value.items())[:64]:
            if _forbidden_key(key):
                continue
            out[_clip_text(str(key))] = _sanitize(child)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize(child) for child in list(value)[:64]]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _sanitize(item())
        except Exception:
            return type(value).__name__
    return type(value).__name__


def _path_lock(path: Path) -> threading.RLock:
    key = os.path.abspath(os.fspath(path))
    with _PATH_STATE_LOCK:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


class NoOpDecisionTelemetryRecorder:
    """Disabled recorder. Every public hook is a cheap no-op."""

    enabled = False
    error_count = 0

    def count_error(self) -> None:
        return None

    def begin_step(self, *, level_before: int | None, phase: str) -> None:
        return None

    def begin_policy_step(self, policy: Any, latest: Any) -> None:
        return None

    def record_event(self, seam: str, payload: Mapping[str, Any], wall_time_s: float) -> None:
        return None

    def record_candidate_action(
        self,
        options: Sequence[Mapping[str, Any]],
        *,
        ranking_changed: bool,
        wall_time_s: float,
        state: Mapping[str, Any] | None = None,
        state_summary: str = "",
    ) -> None:
        return None

    def record_induction_decision(
        self,
        policy: Any,
        *,
        stalled: bool,
        won: bool,
        decision: tuple[bool, str | None],
        wall_time_s: float,
    ) -> None:
        return None

    def time_supervisor_selection(self, supervisor: Any, snapshot: Any) -> Any:
        return supervisor.observe(snapshot)

    def time_world_model_selection(
        self,
        selector: Callable[..., Any],
        transitions: Sequence[Any],
        candidates: Sequence[Any],
        *,
        acceptance_threshold: float | None = None,
        **kwargs: Any,
    ) -> Any:
        del acceptance_threshold
        return selector(transitions, candidates, **kwargs)

    def time_world_model_verification(
        self,
        verifier: Any,
        engine: Any,
        *,
        candidates: Sequence[Any] = (),
        candidate_source: str = "loaded_engine",
    ) -> Any:
        del candidates, candidate_source
        return verifier.score(engine)

    def complete_induction(
        self, policy: Any, attempt: Mapping[str, Any], wall_time_s: float
    ) -> None:
        return None

    def finish_episode(self, *, level_end: int | None, actions_used: int | None = None) -> None:
        return None


NOOP_RECORDER = NoOpDecisionTelemetryRecorder()


class DecisionTelemetryRecorder:
    """Write one bounded JSON object per line. Public methods never raise."""

    enabled = True

    def __init__(
        self,
        game_id: str,
        *,
        path: Path | str,
        episode_id: str | None = None,
        max_records: int = MAX_RECORDS_PER_EPISODE,
        max_bytes: int = MAX_TOTAL_BYTES_PER_RUN,
        flush_every: int = FLUSH_EVERY_RECORDS,
    ) -> None:
        self.game_id = str(game_id)
        self.episode_id = str(episode_id or game_id)
        self.path = Path(path)
        self._path_key = os.path.abspath(os.fspath(self.path))
        self.run_id = f"{self.path.stem}:{os.getpid()}"
        self.max_records = max(2, int(max_records))
        self.max_bytes = max(2 * _TRUNCATION_RESERVE_BYTES, int(max_bytes))
        self.flush_every = max(1, int(flush_every))
        self._error_count = 0
        self._file: Any = None
        self._lock = _path_lock(self.path)
        self._record_count = 0
        self._records_since_flush = 0
        self._current_step = -1
        self._level_before: int | None = None
        self._level_start: int | None = None
        self._phase = "init"
        self._episode_started = False
        self._episode_finished = False
        self._truncated = False
        self._pending_world_gates: list[dict[str, Any]] = []
        try:
            if not self.path.parent.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
            with _PATH_STATE_LOCK:
                _RUN_BYTES.setdefault(self._path_key, 0)
            self._file = self.path.open("a", encoding="utf-8", buffering=64 * 1024)
        except Exception:
            self._error_count += 1
            self._file = None

    @property
    def error_count(self) -> int:
        return self._error_count

    def count_error(self) -> None:
        try:
            self._error_count += 1
        except Exception:
            return None

    def _common(self, *, record_type: str, seam: str | None, wall_time_s: float) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": record_type,
            "seam": seam,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "game_id": self.game_id,
            "step_index": max(0, self._current_step),
            "level_before": self._level_before,
            "phase": self._phase,
            "monotonic_timestamp_s": float(time.monotonic()),
            "wall_time_s": round(max(0.0, float(wall_time_s)), 9),
        }

    def _encode(self, row: Mapping[str, Any]) -> bytes:
        safe = _sanitize(row)
        return (json.dumps(safe, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    def _run_bytes(self) -> int:
        with _PATH_STATE_LOCK:
            return _RUN_BYTES.get(self._path_key, 0)

    def _reserve_run_bytes(self, size: int, *, reserve: int = 0) -> bool:
        with _PATH_STATE_LOCK:
            current = _RUN_BYTES.get(self._path_key, 0)
            if current + size + reserve > self.max_bytes:
                return False
            _RUN_BYTES[self._path_key] = current + size
        return True

    def _write_bytes(self, encoded: bytes) -> bool:
        if self._file is None:
            self._error_count += 1
            return False
        try:
            with self._lock:
                self._file.write(encoded.decode("utf-8"))
                self._record_count += 1
                self._records_since_flush += 1
                if self._records_since_flush >= self.flush_every:
                    self._file.flush()
                    self._records_since_flush = 0
            return True
        except Exception:
            self._error_count += 1
            return False

    def _truncate(self, reason: str) -> None:
        if self._truncated or self._episode_finished:
            return
        self._truncated = True
        try:
            row = self._common(record_type="telemetry_truncated", seam=None, wall_time_s=0.0)
            row.update(
                {
                    "reason": str(reason),
                    "records_written": self._record_count,
                    "bytes_written_for_run": self._run_bytes(),
                }
            )
            encoded = self._encode(row)
            if self._reserve_run_bytes(len(encoded)):
                self._write_bytes(encoded)
        except Exception:
            self._error_count += 1

    def _append(self, row: Mapping[str, Any]) -> None:
        if self._truncated or self._episode_finished:
            return
        try:
            encoded = self._encode(row)
            if len(encoded) > MAX_RECORD_BYTES:
                self._truncate("record_byte_cap")
                return
            if self._record_count >= self.max_records - 1:
                self._truncate("episode_record_cap")
                return
            if not self._reserve_run_bytes(
                len(encoded),
                reserve=_TRUNCATION_RESERVE_BYTES,
            ):
                self._truncate("run_byte_cap")
                return
            self._write_bytes(encoded)
        except Exception:
            self._error_count += 1

    def begin_step(self, *, level_before: int | None, phase: str) -> None:
        try:
            if self._truncated or self._episode_finished:
                return
            self._current_step += 1
            self._level_before = None if level_before is None else int(level_before)
            self._phase = str(phase)
            if self._episode_started:
                return
            self._episode_started = True
            self._level_start = self._level_before
            row = self._common(record_type="episode_start", seam=None, wall_time_s=0.0)
            row.update({"level_start": self._level_start, "actions_used": 0})
            self._append(row)
        except Exception:
            self._error_count += 1

    def begin_policy_step(self, policy: Any, latest: Any) -> None:
        try:
            level = None
            if latest is not None:
                raw_level = getattr(latest, "levels_completed", None)
                if raw_level is None:
                    raw_level = getattr(latest, "level", None)
                if raw_level is not None:
                    level = int(raw_level)
            elif getattr(policy, "_prev_level", None) is not None:
                level = int(policy._prev_level)
            self.begin_step(level_before=level, phase=str(getattr(policy, "phase", "unknown")))
        except Exception:
            self._error_count += 1

    def record_event(self, seam: str, payload: Mapping[str, Any], wall_time_s: float) -> None:
        try:
            if self._truncated or self._episode_finished:
                return
            safe_payload = _sanitize(payload)
            row = dict(safe_payload) if isinstance(safe_payload, dict) else {}
            row.update(
                self._common(record_type="decision", seam=str(seam), wall_time_s=wall_time_s)
            )
            self._append(row)
        except Exception:
            self._error_count += 1

    @staticmethod
    def _option_id(row: Mapping[str, Any], rank: int) -> str:
        kind = row.get("action", row.get("action_id", row.get("kind", "unknown")))
        data = row.get("data")
        if isinstance(data, Mapping) and "x" in data and "y" in data:
            return f"action:{kind}:x:{data['x']}:y:{data['y']}"
        return f"action:{kind}" if kind != "unknown" else f"option:{rank}"

    def record_candidate_action(
        self,
        options: Sequence[Mapping[str, Any]],
        *,
        ranking_changed: bool,
        wall_time_s: float,
        state: Mapping[str, Any] | None = None,
        state_summary: str = "",
    ) -> None:
        try:
            option_rows: list[dict[str, Any]] = []
            all_options = list(options)
            for rank, raw in enumerate(all_options[:MAX_OPTIONS_PER_RECORD], start=1):
                row = dict(raw)
                data = row.get("data") if isinstance(row.get("data"), Mapping) else {}
                kind = row.get("action", row.get("action_id", row.get("kind")))
                features = {
                    str(key): value
                    for key, value in row.items()
                    if key not in {"action", "action_id", "kind", "data", "game_source"}
                    and not any(
                        marker in str(key).lower()
                        for marker in (
                            "adapter",
                            "frame",
                            "game",
                            "grid",
                            "hidden",
                            "source",
                            "state",
                        )
                    )
                }
                option_rows.append(
                    {
                        "option_id": self._option_id(row, rank),
                        "kind": kind,
                        "coordinates": (
                            {"x": data.get("x"), "y": data.get("y")}
                            if "x" in data and "y" in data
                            else None
                        ),
                        "incumbent_rank": rank,
                        "features": features,
                    }
                )
            chosen = option_rows[0]["option_id"] if option_rows else None
            self.record_event(
                "candidate_action_selection",
                {
                    "option_count_total": len(all_options),
                    "options": option_rows,
                    "options_omitted": max(0, len(all_options) - len(option_rows)),
                    "chosen_option_id": chosen,
                    "ranking_changed": bool(ranking_changed),
                    "state": dict(state or {}),
                    "state_summary": state_summary,
                },
                wall_time_s,
            )
        except Exception:
            self._error_count += 1

    def record_induction_decision(
        self,
        policy: Any,
        *,
        stalled: bool,
        won: bool,
        decision: tuple[bool, str | None],
        wall_time_s: float,
    ) -> None:
        try:
            should_induce, reason = decision
            explorer = getattr(policy, "explorer", None)
            transitions = len(getattr(policy, "transitions", ()) or ())
            last_attempts = getattr(policy, "induction_attempts", ()) or ()
            last_attempt = last_attempts[-1] if last_attempts else {}
            prior_outcome = {
                key: last_attempt.get(key)
                for key in (
                    "skipped",
                    "planned",
                    "verify_accuracy",
                    "verify_cell_recall",
                    "verify_change_accuracy",
                    "trust_energy",
                )
                if key in last_attempt
            }
            if not should_induce:
                action = "continue_exploring"
            elif reason == "renewed_stall_reinduction":
                action = "reinduce"
            else:
                action = "induce"
            new_transitions = transitions - int(
                getattr(policy, "_transitions_at_last_induction_attempt", 0)
            )
            state = {
                "level": self._level_before,
                "goal_level": getattr(policy, "_current_goal_level", None),
                "transition_count": transitions,
                "explored_out": bool(getattr(explorer, "explored_out", False)),
                "stall_state": "stalled" if stalled else "not_stalled",
                "won": bool(won),
                "induced": bool(getattr(policy, "induced", False)),
                "prior_induction_count": int(getattr(policy, "_induction_attempt_count", 0)),
                "new_transitions_since_induction": max(0, new_transitions),
                "last_verifier_outcome": prior_outcome,
            }
            self.record_event(
                "induction_timing",
                {
                    "decision": action,
                    "reason": reason,
                    "state": state,
                    "state_summary": (
                        f"level={state['level']} transitions={transitions} "
                        f"explored_out={state['explored_out']} stall={state['stall_state']}"
                    ),
                },
                wall_time_s,
            )
        except Exception:
            self._error_count += 1

    def time_supervisor_selection(self, supervisor: Any, snapshot: Any) -> Any:
        span = _SupervisorSelectionSpan(self, snapshot, supervisor)
        span.__enter__()
        try:
            return supervisor.observe(snapshot)
        finally:
            span.__exit__(None, None, None)

    @staticmethod
    def _selection_metrics(selection: Any) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        for score in list(getattr(selection, "rows", ()) or ())[:MAX_OPTIONS_PER_RECORD]:
            change_gate = dict(getattr(score, "change_gate", {}) or {})
            candidates.append(
                {
                    "candidate_id": str(getattr(getattr(score, "candidate", None), "name", "")),
                    "exact_accuracy": change_gate.get(
                        "legacy_accuracy", getattr(score, "heldout_accuracy", None)
                    ),
                    "cell_recall": change_gate.get("cell_recall"),
                    "change_accuracy": change_gate.get("change_accuracy"),
                    "trust_energy": getattr(score, "trust_energy", None),
                    "off_path_checks": {
                        "baseline_clears": getattr(score, "baseline_clears", None),
                        "heldout_best": getattr(score, "heldout_best", None),
                        "nondegenerate": getattr(score, "nondegenerate", None),
                        "trust_pass": getattr(score, "trust_pass", None),
                        "binary_gate_pass": getattr(score, "binary_gate_pass", None),
                        "change_gate": change_gate,
                    },
                }
            )
        selected = getattr(getattr(selection, "selected", None), "name", None)
        return {
            "candidate_ids": [row["candidate_id"] for row in candidates],
            "candidate_count_total": len(getattr(selection, "rows", ()) or ()),
            "candidates": candidates,
            "candidates_omitted": max(
                0,
                len(getattr(selection, "rows", ()) or ()) - len(candidates),
            ),
            "selected_candidate_id": selected,
        }

    def time_world_model_selection(
        self,
        selector: Callable[..., Any],
        transitions: Sequence[Any],
        candidates: Sequence[Any],
        *,
        acceptance_threshold: float | None = None,
        **kwargs: Any,
    ) -> Any:
        started = time.perf_counter()
        result = selector(transitions, candidates, **kwargs)
        elapsed = time.perf_counter() - started
        try:
            metrics = self._selection_metrics(result)
            selected_score = getattr(result, "selected_score", None)
            heldout = getattr(selected_score, "heldout_accuracy", None)
            accepted = (
                None
                if acceptance_threshold is None or heldout is None
                else float(heldout) >= float(acceptance_threshold)
            )
            self._pending_world_gates.append(
                {
                    **metrics,
                    "accepted_by_threshold": accepted,
                    "wall_time_s": elapsed,
                }
            )
        except Exception:
            self._error_count += 1
        return result

    def time_world_model_verification(
        self,
        verifier: Any,
        engine: Any,
        *,
        candidates: Sequence[Any] = (),
        candidate_source: str = "loaded_engine",
    ) -> Any:
        started = time.perf_counter()
        result = verifier.score(engine)
        elapsed = time.perf_counter() - started
        try:
            candidate_ids = [
                str(getattr(candidate, "name", f"candidate:{index}"))
                for index, candidate in enumerate(candidates)
            ]
            if not candidate_ids:
                candidate_ids = [str(candidate_source)]
            self._pending_world_gates.append(
                {
                    "candidate_ids": [str(value) for value in candidate_ids],
                    "candidate_count_total": len(candidate_ids),
                    "selected_candidate_id": str(candidate_ids[0]) if candidate_ids else None,
                    "candidates": [
                        {
                            "candidate_id": str(candidate_ids[0])
                            if candidate_ids
                            else "loaded_engine",
                            "exact_accuracy": getattr(result, "accuracy", None),
                            "cell_recall": getattr(result, "cell_recall", None),
                            "change_accuracy": getattr(result, "change_accuracy", None),
                            "trust_energy": None,
                            "off_path_checks": {
                                "change_fidelity": getattr(result, "change_fidelity", None),
                                "correct_changed_cells": getattr(
                                    result, "correct_changed_cells", None
                                ),
                                "spurious_changed_cells": getattr(
                                    result, "spurious_changed_cells", None
                                ),
                                "noop_hallucination_rate": getattr(
                                    result, "noop_hallucination_rate", None
                                ),
                            },
                        }
                    ],
                    "accepted_by_threshold": None,
                    "wall_time_s": elapsed,
                }
            )
        except Exception:
            self._error_count += 1
        return result

    @staticmethod
    def _gate_outcome(
        gate: Mapping[str, Any],
        attempt: Mapping[str, Any],
        *,
        has_later_gate: bool,
    ) -> str:
        accepted = gate.get("accepted_by_threshold")
        if accepted is True:
            return "accept"
        if accepted is False:
            return "escalate" if has_later_gate else "reject"
        skipped = str(attempt.get("skipped") or "")
        if any(token in skipped for token in ("below_threshold", "change_gate", "verification")):
            return "reject"
        return "accept"

    def complete_induction(
        self, policy: Any, attempt: Mapping[str, Any], wall_time_s: float
    ) -> None:
        try:
            proposer = getattr(policy, "proposer", None)
            generated = getattr(proposer, "last_generated_tokens", None)
            prompt = getattr(proposer, "last_prompt_tokens", None)
            self.record_event(
                "induction_timing",
                {
                    "decision": "induction_call",
                    "reason": attempt.get("reason"),
                    "generated_tokens": generated
                    if isinstance(generated, int) and generated >= 0
                    else None,
                    "prompt_tokens": prompt if isinstance(prompt, int) and prompt >= 0 else None,
                    "seconds": round(max(0.0, float(wall_time_s)), 9),
                    "transition_count": attempt.get("transition_count"),
                },
                wall_time_s,
            )
            pending = list(self._pending_world_gates)
            self._pending_world_gates.clear()
            for index, gate in enumerate(pending):
                payload = dict(gate)
                gate_wall = float(payload.pop("wall_time_s", 0.0) or 0.0)
                payload.update(
                    {
                        "outcome": self._gate_outcome(
                            gate,
                            attempt,
                            has_later_gate=index + 1 < len(pending),
                        ),
                        "plan_found": bool(attempt.get("planned")) and index + 1 == len(pending),
                    }
                )
                self.record_event("world_model_hypothesis_gate", payload, gate_wall)
        except Exception:
            self._error_count += 1

    def finish_episode(self, *, level_end: int | None, actions_used: int | None = None) -> None:
        try:
            if self._episode_finished:
                return
            if not self._episode_started and not self._truncated:
                self.begin_step(level_before=level_end, phase="end")
            if not self._truncated:
                end = None if level_end is None else int(level_end)
                start = self._level_start
                row = self._common(record_type="episode_end", seam=None, wall_time_s=0.0)
                row.update(
                    {
                        "level_start": start,
                        "level_end": end,
                        "levels_gained": (
                            None if start is None or end is None else max(0, int(end) - int(start))
                        ),
                        "actions_used": (
                            max(0, self._current_step + 1)
                            if actions_used is None
                            else max(0, int(actions_used))
                        ),
                    }
                )
                self._append(row)
            self._episode_finished = True
            if self._file is not None:
                with self._lock:
                    self._file.flush()
                    self._file.close()
                self._file = None
        except Exception:
            self._error_count += 1


class _SupervisorSelectionSpan:
    def __init__(self, recorder: DecisionTelemetryRecorder, snapshot: Any, supervisor: Any) -> None:
        self.recorder = recorder
        self.snapshot = snapshot
        self.supervisor = supervisor
        self.started = 0.0
        self.before: dict[str, Any] = {}

    def __enter__(self) -> None:
        try:
            self.before = dict(self.supervisor.receipt())
            self.started = time.perf_counter()
        except Exception:
            self.recorder._error_count += 1
        return None

    def _eligible(self, snapshot: Any, before: Mapping[str, Any]) -> list[str]:
        used = set(before.get("arms_used") or ())
        enabled = set(before.get("arms_enabled") or ())
        attempts = int(getattr(snapshot, "induction_attempts", 0))
        new_transitions = int(getattr(snapshot, "new_transitions_since_induction", 0))
        evidence_floor = int(getattr(self.supervisor, "reinduction_evidence_floor", 200))
        attempt_cap = int(getattr(self.supervisor, "reinduction_attempt_cap", 3))
        arms: list[str] = []
        if "drop_goal_bias" not in used and bool(getattr(snapshot, "goal_bias_installed", False)):
            arms.append("drop_goal_bias")
        if (
            "allow_reinduction" not in used
            and bool(getattr(snapshot, "induced", False))
            and new_transitions >= evidence_floor
            and attempts < attempt_cap
        ):
            arms.append("allow_reinduction")
        if (
            "tool_loop_reinduction" in enabled
            and "tool_loop_reinduction" not in used
            and "allow_reinduction" in used
            and attempts < attempt_cap
        ):
            arms.append("tool_loop_reinduction")
        if "force_exploration_diversity" not in used and not bool(
            getattr(snapshot, "diversity_active", False)
        ):
            arms.append("force_exploration_diversity")
        return arms

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        elapsed = time.perf_counter() - self.started if self.started else 0.0
        if exc_type is not None:
            return False
        try:
            after = dict(self.supervisor.receipt())
            before_redirects = len(self.before.get("redirects") or ())
            after_redirects = list(after.get("redirects") or ())
            before_empty = int(self.before.get("stagnations_unredirected") or 0)
            after_empty = int(after.get("stagnations_unredirected") or 0)
            selected = len(after_redirects) > before_redirects
            no_redirect = after_empty > before_empty
            if not selected and not no_redirect:
                return False
            chosen = after_redirects[-1] if selected else {}
            snapshot_row = {
                "level": int(getattr(self.snapshot, "level", 0)),
                "goal_bias_installed": bool(getattr(self.snapshot, "goal_bias_installed", False)),
                "induced": bool(getattr(self.snapshot, "induced", False)),
                "induction_attempts": int(getattr(self.snapshot, "induction_attempts", 0)),
                "new_transitions_since_induction": int(
                    getattr(self.snapshot, "new_transitions_since_induction", 0)
                ),
                "diversity_active": bool(getattr(self.snapshot, "diversity_active", False)),
            }
            self.recorder.record_event(
                "supervisor_arm_selection",
                {
                    "trajectory_snapshot": snapshot_row,
                    "eligible_arms": self._eligible(self.snapshot, self.before),
                    "chosen_arm": chosen.get("arm") if selected else "no_redirect",
                    "diagnosis": chosen.get("diagnosis", ""),
                },
                elapsed,
            )
        except Exception:
            self.recorder._error_count += 1
        return False


def maybe_make_recorder(
    game_id: str,
    *,
    episode_id: str | None = None,
    max_records: int = MAX_RECORDS_PER_EPISODE,
    max_bytes: int = MAX_TOTAL_BYTES_PER_RUN,
    flush_every: int = FLUSH_EVERY_RECORDS,
) -> DecisionTelemetryRecorder | NoOpDecisionTelemetryRecorder:
    """Construct the recorder only for the exact operator opt-in."""

    if not telemetry_enabled():
        return NOOP_RECORDER
    try:
        raw_path = os.environ.get(TELEMETRY_PATH_ENV)
        path = Path(raw_path) if raw_path else _safe_default_path()
        if path is None:
            _warn_missing_path_once()
            return NOOP_RECORDER
        return DecisionTelemetryRecorder(
            game_id,
            path=path,
            episode_id=episode_id,
            max_records=max_records,
            max_bytes=max_bytes,
            flush_every=flush_every,
        )
    except Exception:
        return NOOP_RECORDER


def capture_candidate_decision(function: _F) -> _F:
    """Time the existing candidate function without changing its return."""

    @wraps(function)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        recorder = getattr(self, "_decision_telemetry", NOOP_RECORDER)
        if not recorder.enabled:
            return function(self, *args, **kwargs)
        started = time.perf_counter()
        result = function(self, *args, **kwargs)
        elapsed = time.perf_counter() - started
        try:
            selector = getattr(self, "belief_candidate_selector", None)
            decision = getattr(selector, "last_decision", {}) or {}
            path = kwargs.get("path")
            if path is None and len(args) > 1:
                path = args[1]
            path_depth = len(path or ())
            state = {
                "path_depth": path_depth,
                "best_level": getattr(self, "best_level", None),
                "explored_out": bool(getattr(self, "explored_out", False)),
                "steps_since_progress": int(getattr(self, "_steps_since_progress", 0)),
            }
            recorder.record_candidate_action(
                result,
                ranking_changed=bool(decision.get("ranking_changed", False)),
                wall_time_s=elapsed,
                state=state,
                state_summary=(
                    f"path_depth={path_depth} best_level={state['best_level']} "
                    f"explored_out={state['explored_out']} "
                    f"steps_since_progress={state['steps_since_progress']}"
                ),
            )
        except Exception:
            recorder.count_error()
        return result

    return wrapped  # type: ignore[return-value]


def capture_induction_decision(function: _F) -> _F:
    """Time the existing induction admission function without changing it."""

    @wraps(function)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        recorder = getattr(self, "_decision_telemetry", NOOP_RECORDER)
        if not recorder.enabled:
            return function(self, *args, **kwargs)
        started = time.perf_counter()
        result = function(self, *args, **kwargs)
        elapsed = time.perf_counter() - started
        try:
            recorder.record_induction_decision(
                self,
                stalled=bool(kwargs.get("stalled", False)),
                won=bool(kwargs.get("won", False)),
                decision=result,
                wall_time_s=elapsed,
            )
        except Exception:
            recorder.count_error()
        return result

    return wrapped  # type: ignore[return-value]


def load_telemetry(path: Path | str) -> list[dict[str, Any]]:
    """Load valid JSON objects from one telemetry file."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def summarize_telemetry(path: Path | str) -> dict[str, Any]:
    """Summarize seam counts, option widths, and episode wall time."""

    rows = load_telemetry(path)
    seam_counts: Counter[str] = Counter()
    option_counts: Counter[int] = Counter()
    episodes: dict[str, dict[str, Any]] = {}
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        episode = episodes.setdefault(
            episode_id,
            {
                "records_per_seam": Counter(),
                "option_count_distribution": Counter(),
                "total_wall_time_s": 0.0,
                "wall_time_s_by_seam": Counter(),
            },
        )
        seam = row.get("seam")
        if not seam:
            continue
        seam_name = str(seam)
        wall = float(row.get("wall_time_s") or 0.0)
        seam_counts[seam_name] += 1
        episode["records_per_seam"][seam_name] += 1
        episode["total_wall_time_s"] += wall
        episode["wall_time_s_by_seam"][seam_name] += wall
        if seam_name == "candidate_action_selection":
            count = int(row.get("option_count_total") or 0)
            option_counts[count] += 1
            episode["option_count_distribution"][count] += 1
    clean_episodes: dict[str, Any] = {}
    for episode_id, episode in episodes.items():
        clean_episodes[episode_id] = {
            "records_per_seam": dict(sorted(episode["records_per_seam"].items())),
            "option_count_distribution": {
                str(key): value
                for key, value in sorted(episode["option_count_distribution"].items())
            },
            "total_wall_time_s": round(float(episode["total_wall_time_s"]), 9),
            "wall_time_s_by_seam": {
                key: round(float(value), 9)
                for key, value in sorted(episode["wall_time_s_by_seam"].items())
            },
        }
    return {
        "records_per_seam": dict(sorted(seam_counts.items())),
        "option_count_distribution": {
            str(key): value for key, value in sorted(option_counts.items())
        },
        "episodes": clean_episodes,
    }
