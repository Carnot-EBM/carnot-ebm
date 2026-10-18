"""Explicit opt-in client for the qualified native recalibration service.

Importing this module does not load native code or create state. A caller must
construct ``NativeServiceClient`` and name either a binding or extension path.

Spec: REQ-PIPELINE-7641 and SCENARIO-PIPELINE-7641-*.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, cast

from carnot.pipeline.calibrated_decision_service import (
    DURABILITY_POLICY,
    CalibratedDecision,
    DecisionAction,
    FeedbackAcknowledgment,
    frozen_decision_costs,
)


_ACTIONS = {"accept", "reject", "escalate"}


def load_native_extension(extension_path: str | Path | None = None) -> ModuleType:
    """Load an installed binding or one exact private extension path."""

    if extension_path is None:
        module = importlib.import_module("carnot._rust")
    else:
        resolved = Path(extension_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        sys.modules.pop("carnot._rust", None)
        spec = importlib.util.spec_from_file_location("carnot._rust", resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"native_loader_unavailable:{resolved}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    if not hasattr(module, "RustPortableRecalibrationService"):
        raise ImportError("native_service_class_missing")
    return module


def _safe_probability(value: object) -> float:
    try:
        probability = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return 1.0
    return probability


class NativeServiceClient:
    """Adapt the direct Rust service to the existing typed policy results.

    Native failures become unavailable values. The client never computes a
    calibrated result in Python when the extension cannot answer.
    """

    def __init__(
        self,
        binding: Any | None,
        state_path: str | Path,
        *,
        unavailable_error: str | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self._inner: Any | None = None
        self._unavailable_error = unavailable_error
        self._closed = False
        self._pending: set[str] = set()
        self._released: set[str] = set()
        if unavailable_error is not None:
            return
        try:
            if binding is None or not hasattr(binding, "RustPortableRecalibrationService"):
                raise TypeError("native_service_class_missing")
            self._inner = binding.RustPortableRecalibrationService(str(self.state_path))
            _count, event_ids, _schema = self._inner.state_summary()
            self._released.update(str(value) for value in event_ids)
        except Exception as error:  # noqa: BLE001 - native boundary must return typed failure.
            self._inner = None
            self._unavailable_error = f"native_service_unavailable:{error}"

    @classmethod
    def from_extension(
        cls,
        *,
        state_path: str | Path,
        extension_path: str | Path | None = None,
    ) -> NativeServiceClient:
        """Construct explicitly and retain extension failures as typed state."""

        try:
            binding = load_native_extension(extension_path)
        except Exception as error:  # noqa: BLE001 - loader errors share one public contract.
            return cls(
                None,
                state_path,
                unavailable_error=f"native_extension_unavailable:{error}",
            )
        return cls(binding, state_path)

    def _decision_unavailable(
        self, event_id: str, probability: object, error: str
    ) -> CalibratedDecision:
        safe = _safe_probability(probability)
        return CalibratedDecision(
            event_id=event_id,
            error_probability=safe,
            action="escalate",
            available=False,
            verified=False,
            error=error,
            costs=frozen_decision_costs(safe),
        )

    @staticmethod
    def _ack_unavailable(event_id: str, error: str) -> FeedbackAcknowledgment:
        return FeedbackAcknowledgment(
            event_id=event_id,
            available=False,
            acknowledged=False,
            durable=False,
            error=error,
        )

    def _availability_error(self) -> str | None:
        if self._closed:
            return "native_client_closed"
        return self._unavailable_error

    def predict(self, event_id: str, error_probability: object) -> CalibratedDecision:
        """Return one typed native decision or an unavailable escalation."""

        identifier = str(event_id)
        error = self._availability_error()
        if error is not None:
            return self._decision_unavailable(identifier, error_probability, error)
        try:
            probability = float(error_probability)  # type: ignore[arg-type]
            costs = frozen_decision_costs(probability)
        except (TypeError, ValueError):
            return self._decision_unavailable(
                identifier, error_probability, "finite_probability_required"
            )
        if not identifier:
            return self._decision_unavailable(identifier, probability, "event_id_required")
        if identifier in self._released:
            return self._decision_unavailable(
                identifier, probability, f"duplicate_feedback:{identifier}"
            )
        if identifier in self._pending:
            return self._decision_unavailable(
                identifier, probability, f"duplicate_prediction:{identifier}"
            )
        try:
            returned_id, calibrated_raw, action = self._inner.predict(identifier, probability)
            calibrated = float(calibrated_raw)
            output_costs = frozen_decision_costs(calibrated)
        except Exception as native_error:  # noqa: BLE001 - native exceptions become unavailable.
            return self._decision_unavailable(identifier, probability, str(native_error))
        expected_action = min(output_costs, key=output_costs.__getitem__)
        if returned_id != identifier or action not in _ACTIONS or action != expected_action:
            return self._decision_unavailable(
                identifier, probability, "prediction_contract_invalid"
            )
        self._pending.add(identifier)
        return CalibratedDecision(
            event_id=identifier,
            error_probability=calibrated,
            action=cast(DecisionAction, action),
            available=True,
            verified=False,
            costs=output_costs,
        )

    def release_feedback(self, event_id: str, label: int) -> FeedbackAcknowledgment:
        """Release one label and require the Rust core's durable acknowledgment."""

        identifier = str(event_id)
        error = self._availability_error()
        if error is not None:
            return self._ack_unavailable(identifier, error)
        if label not in (0, 1) or isinstance(label, bool):
            return self._ack_unavailable(identifier, "binary_label_required")
        if identifier in self._released:
            return self._ack_unavailable(identifier, f"duplicate_feedback:{identifier}")
        if identifier not in self._pending:
            return self._ack_unavailable(identifier, f"unknown_prediction:{identifier}")
        try:
            returned_id, acknowledged, durable, native_error = self._inner.release_feedback(
                identifier, label
            )
        except Exception as error_value:  # noqa: BLE001 - native exceptions become unavailable.
            return self._ack_unavailable(identifier, str(error_value))
        valid = (
            returned_id == identifier
            and acknowledged is True
            and durable is True
            and native_error is None
        )
        if not valid:
            return self._ack_unavailable(
                identifier, str(native_error or "durable_acknowledgment_invalid")
            )
        self._pending.remove(identifier)
        self._released.add(identifier)
        return FeedbackAcknowledgment(
            event_id=identifier,
            available=True,
            acknowledged=True,
            durable=True,
            durability_policy=DURABILITY_POLICY,
        )

    def state_summary(self) -> dict[str, object]:
        """Return native state metadata without exposing mutable internal state."""

        error = self._availability_error()
        if error is not None:
            return {
                "available": False,
                "sample_count": 0,
                "processed_event_ids": [],
                "schema": None,
                "error": error,
            }
        try:
            sample_count, event_ids, schema = self._inner.state_summary()
        except Exception as native_error:  # noqa: BLE001 - summary failure stays explicit.
            return {
                "available": False,
                "sample_count": 0,
                "processed_event_ids": [],
                "schema": None,
                "error": str(native_error),
            }
        return {
            "available": True,
            "sample_count": int(sample_count),
            "processed_event_ids": [str(value) for value in event_ids],
            "schema": str(schema),
            "error": None,
        }

    def close(self) -> None:
        """Close this lightweight client without changing its durable state."""

        self._closed = True


__all__ = [
    "CalibratedDecision",
    "FeedbackAcknowledgment",
    "NativeServiceClient",
    "load_native_extension",
]
