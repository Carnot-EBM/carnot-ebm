"""Explicit typed client for the durable recalibration process.

The client is opt-in. Importing this module starts no process and creates no
state. Service failures produce an unavailable escalation, not verification.

Spec: REQ-CL-7598, REQ-VERIFY-7598, SCENARIO-CL-7598-LIFECYCLE/FAILURE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import select
import subprocess
import time
from typing import Any, Literal

from carnot import experiment_7585_v662_portable_service as portable


DecisionAction = Literal["accept", "reject", "escalate"]
DURABILITY_POLICY = "atomic_file_fsync_rename_directory_fsync_reload_ack"
_ACTIONS = {"accept", "reject", "escalate"}


@dataclass(frozen=True)
class CalibratedDecision:
    """One policy decision. ``verified`` stays false by contract."""

    event_id: str
    error_probability: float
    action: DecisionAction
    available: bool
    verified: bool = False
    error: str | None = None
    costs: dict[str, float] = field(default_factory=dict)
    service_ns: int = 0
    stage_ns: dict[str, int] = field(default_factory=dict)
    exclusive_stage_ns: dict[str, int] = field(default_factory=dict)
    caller_stage_ns: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackAcknowledgment:
    """One feedback result whose durable flag requires a reloaded state."""

    event_id: str
    available: bool
    acknowledged: bool
    durable: bool
    error: str | None = None
    durability_policy: str | None = None
    service_ns: int = 0
    kernel_ns: int = 0
    stage_ns: dict[str, int] = field(default_factory=dict)
    exclusive_stage_ns: dict[str, int] = field(default_factory=dict)
    caller_stage_ns: dict[str, int] = field(default_factory=dict)


def frozen_decision_costs(error_probability: float) -> dict[str, float]:
    """Return the registered costs for one finite probability."""

    if not math.isfinite(error_probability) or not 0.0 <= error_probability <= 1.0:
        raise ValueError("finite_probability_required")
    return {
        "escalate": 0.2,
        "accept": 5.0 * error_probability,
        "reject": 1.0 - error_probability,
    }


def _fallback_probability(value: float) -> float:
    if math.isfinite(value):
        return min(1.0, max(0.0, value))
    return 1.0


class CalibratedDecisionService:
    """Own one bounded JSON-lines worker and one durable session state.

    Callers must construct this class explicitly. The default command is the
    existing Rust service. ``process_command`` exists for matched service tests
    and does not change the production default.
    """

    def __init__(
        self,
        *,
        state_path: str | os.PathLike[str],
        binary_path: str | os.PathLike[str],
        response_timeout_s: float = 5.0,
        process_command: Sequence[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        extra_env: Mapping[str, str] | None = None,
        telemetry_enabled: bool = False,
    ) -> None:
        if not math.isfinite(response_timeout_s) or response_timeout_s <= 0.0:
            raise ValueError("positive_response_timeout_required")
        self.state_path = Path(state_path)
        self.binary_path = Path(binary_path)
        self.response_timeout_s = float(response_timeout_s)
        self.telemetry_enabled = bool(telemetry_enabled)
        if process_command is None and not self.binary_path.is_file():
            raise FileNotFoundError(self.binary_path)
        if not self.state_path.exists():
            portable.initialize_state(self.state_path)
        self._pending: dict[str, float] = {}
        self._released: set[str] = set()
        try:
            self._released.update(portable._load_state(self.state_path).processed_event_ids)
        except (OSError, TypeError, ValueError):
            # The worker owns schema errors. Construction still succeeds so the
            # public call can return a typed unavailable escalation.
            pass
        command = tuple(process_command or (str(self.binary_path),))
        environment = dict(os.environ)
        environment.update(dict(extra_env or {}))
        environment["CARNOT_SERVICE_TIMING"] = "1" if self.telemetry_enabled else "0"
        startup_started = time.perf_counter_ns()
        self._owned_process = subprocess.Popen(  # noqa: S603 - explicit caller command.
            command,
            cwd=Path(cwd) if cwd is not None else None,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.startup_ns = time.perf_counter_ns() - startup_started
        self.owned_pid = self._owned_process.pid
        self._closed = False

    def __enter__(self) -> CalibratedDecisionService:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _request(
        self, request: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None, int, dict[str, int]]:
        started = time.perf_counter_ns()
        caller_stages: dict[str, int] = {}
        process = self._owned_process
        if self._closed or process.poll() is not None:
            return None, "process_exited", time.perf_counter_ns() - started, caller_stages
        if process.stdin is None or process.stdout is None:
            return None, "worker_pipe_missing", time.perf_counter_ns() - started, caller_stages
        try:
            encode_started = time.perf_counter_ns()
            encoded = json.dumps(dict(request), sort_keys=True, separators=(",", ":")) + "\n"
            if self.telemetry_enabled:
                caller_stages["caller_encode"] = time.perf_counter_ns() - encode_started
            send_started = time.perf_counter_ns()
            process.stdin.write(encoded)
            process.stdin.flush()
            if self.telemetry_enabled:
                caller_stages["caller_send"] = time.perf_counter_ns() - send_started
        except (BrokenPipeError, OSError, ValueError):
            return None, "process_exited", time.perf_counter_ns() - started, caller_stages
        wait_started = time.perf_counter_ns()
        ready, _writable, _errors = select.select([process.stdout], [], [], self.response_timeout_s)
        if not ready:
            return None, "response_timeout", time.perf_counter_ns() - started, caller_stages
        line = process.stdout.readline()
        elapsed = time.perf_counter_ns() - started
        if self.telemetry_enabled:
            caller_stages["caller_wait_decode"] = time.perf_counter_ns() - wait_started
        if not line:
            return None, "process_exited", elapsed, caller_stages
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return None, "response_json_invalid", elapsed, caller_stages
        if not isinstance(value, Mapping):
            return None, "response_not_object", elapsed, caller_stages
        return dict(value), None, elapsed, caller_stages

    def _unavailable_decision(
        self, event_id: str, probability: float, error: str, elapsed_ns: int = 0
    ) -> CalibratedDecision:
        safe_probability = _fallback_probability(probability)
        return CalibratedDecision(
            event_id=event_id,
            error_probability=safe_probability,
            action="escalate",
            available=False,
            verified=False,
            error=error,
            costs=frozen_decision_costs(safe_probability),
            service_ns=elapsed_ns,
        )

    def predict(self, event_id: str, error_probability: float) -> CalibratedDecision:
        """Request one typed decision without revealing an outcome label."""

        identifier = str(event_id)
        try:
            costs = frozen_decision_costs(float(error_probability))
        except (TypeError, ValueError):
            return self._unavailable_decision(
                identifier, float(error_probability), "finite_probability_required"
            )
        if not identifier:
            return self._unavailable_decision(identifier, error_probability, "event_id_required")
        if identifier in self._released:
            return self._unavailable_decision(
                identifier, error_probability, f"duplicate_feedback:{identifier}"
            )
        if identifier in self._pending:
            return self._unavailable_decision(
                identifier, error_probability, f"duplicate_prediction:{identifier}"
            )
        response, transport_error, elapsed, caller_stages = self._request(
            {
                "operation": "predict",
                "state_path": str(self.state_path),
                "queries": [{"event_id": identifier, "probability": error_probability}],
            }
        )
        if transport_error is not None or response is None:
            return self._unavailable_decision(
                identifier, error_probability, transport_error or "response_missing", elapsed
            )
        if response.get("ok") is not True:
            return self._unavailable_decision(
                identifier,
                error_probability,
                str(response.get("error") or "service_failed"),
                elapsed,
            )
        predictions = response.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != 1:
            return self._unavailable_decision(
                identifier, error_probability, "prediction_count_invalid", elapsed
            )
        prediction = predictions[0]
        if not isinstance(prediction, Mapping):
            return self._unavailable_decision(
                identifier, error_probability, "prediction_not_object", elapsed
            )
        try:
            calibrated = float(prediction["probability"])
            output_costs = frozen_decision_costs(calibrated)
        except (KeyError, TypeError, ValueError):
            return self._unavailable_decision(
                identifier, error_probability, "prediction_probability_invalid", elapsed
            )
        action = prediction.get("action")
        expected_action = min(output_costs, key=output_costs.__getitem__)
        if (
            prediction.get("event_id") != identifier
            or action not in _ACTIONS
            or action != expected_action
        ):
            return self._unavailable_decision(
                identifier, error_probability, "prediction_contract_invalid", elapsed
            )
        self._pending[identifier] = float(error_probability)
        stage_ns = response.get("stage_ns")
        exclusive = response.get("exclusive_stage_ns")
        return CalibratedDecision(
            event_id=identifier,
            error_probability=calibrated,
            action=action,
            available=True,
            verified=False,
            costs=output_costs,
            service_ns=elapsed,
            stage_ns={str(key): int(value) for key, value in dict(stage_ns or {}).items()},
            exclusive_stage_ns={
                str(key): int(value) for key, value in dict(exclusive or {}).items()
            },
            caller_stage_ns=caller_stages,
        )

    def _unavailable_ack(
        self, event_id: str, error: str, elapsed_ns: int = 0
    ) -> FeedbackAcknowledgment:
        return FeedbackAcknowledgment(
            event_id=event_id,
            available=False,
            acknowledged=False,
            durable=False,
            error=error,
            service_ns=elapsed_ns,
        )

    def release_feedback(self, event_id: str, label: int) -> FeedbackAcknowledgment:
        """Release one binary outcome and require a durable reloaded acknowledgment."""

        identifier = str(event_id)
        if label not in (0, 1) or isinstance(label, bool):
            return self._unavailable_ack(identifier, "binary_label_required")
        if identifier in self._released:
            return self._unavailable_ack(identifier, f"duplicate_feedback:{identifier}")
        if identifier not in self._pending:
            return self._unavailable_ack(identifier, f"unknown_prediction:{identifier}")
        response, transport_error, elapsed, caller_stages = self._request(
            {
                "operation": "trace",
                "state_path": str(self.state_path),
                "events": [
                    {
                        "event_id": identifier,
                        "probability": self._pending[identifier],
                        "label": label,
                    }
                ],
            }
        )
        if transport_error is not None or response is None:
            return self._unavailable_ack(identifier, transport_error or "response_missing", elapsed)
        if response.get("ok") is not True:
            return self._unavailable_ack(
                identifier, str(response.get("error") or "service_failed"), elapsed
            )
        acknowledgment_started = time.perf_counter_ns()
        state = response.get("state")
        processed = state.get("processed_event_ids") if isinstance(state, Mapping) else None
        durable = bool(
            response.get("acknowledgments") == [0]
            and response.get("acknowledged_release_count") == 1
            and response.get("processed_event_count") == 1
            and response.get("reloaded_state_matches") is True
            and response.get("durability_policy") == DURABILITY_POLICY
            and isinstance(processed, list)
            and identifier in processed
        )
        if not durable:
            return self._unavailable_ack(identifier, "durable_acknowledgment_invalid", elapsed)
        if self.telemetry_enabled:
            caller_stages["caller_acknowledgement"] = (
                time.perf_counter_ns() - acknowledgment_started
            )
        self._pending.pop(identifier)
        self._released.add(identifier)
        stage_ns = response.get("stage_ns")
        exclusive = response.get("exclusive_stage_ns")
        return FeedbackAcknowledgment(
            event_id=identifier,
            available=True,
            acknowledged=True,
            durable=True,
            durability_policy=DURABILITY_POLICY,
            service_ns=elapsed,
            kernel_ns=int(response.get("kernel_ns") or 0),
            stage_ns={str(key): int(value) for key, value in dict(stage_ns or {}).items()},
            exclusive_stage_ns={
                str(key): int(value) for key, value in dict(exclusive or {}).items()
            },
            caller_stage_ns=caller_stages,
        )

    def close(self) -> None:
        """Close and, when needed, terminate only this client's child process."""

        if self._closed:
            return
        self._closed = True
        process = self._owned_process
        if process.stdin is not None:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=self.response_timeout_s)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=self.response_timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.response_timeout_s)
