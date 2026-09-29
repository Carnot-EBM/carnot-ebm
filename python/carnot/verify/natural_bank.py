"""Small restartable advisory bank for REQ-VERIFY-7853.

Forecasts and delayed labels are durable. Their coefficients never replace an
exact verifier or change generator weights.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify.natural_predicates import NAMES

SCHEMA = "carnot.natural_bank.v1"
DELAY_TICKS = 1
LEARNING_RATE = 0.01
BLOCK_SIZE = 8


class NaturalBank:
    """Store an integrity-checked queue and clipped prequential coefficients."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.state: dict[str, Any] = json.loads(self.path.read_text())
            digest = self.state.pop("checksum", None)
            if digest != canonical_hash(self.state) or self.state.get("schema") != SCHEMA:
                raise ValueError("corrupt bank state")
        else:
            self.state = {
                "schema": SCHEMA,
                "names": list(NAMES),
                "coefficients": [0.0] * len(NAMES),
                "pending": {},
                "released": {},
                "admitted": {},
                "last_tick": -1,
                "ledger": [],
            }
            self._save("init", {})
        if self.state["names"] != list(NAMES):
            raise ValueError("corrupt bank grammar")

    @property
    def pending(self) -> list[str]:
        """Expose the queue so a restart can verify all unreleased forecasts."""
        return list(self.state["pending"])

    def _save(self, kind: str, detail: dict[str, Any]) -> None:
        prior = self.state["ledger"][-1]["hash"] if self.state["ledger"] else "genesis"
        event = {"kind": kind, "detail": detail, "previous": prior}
        event["hash"] = canonical_hash(event)
        self.state["ledger"].append(event)
        atomic_json(self.path, {**self.state, "checksum": canonical_hash(self.state)})

    def predict(
        self,
        event_id: str,
        tick: int,
        features: dict[str, float],
        base_probability: float,
        *,
        read_only: bool = False,
    ) -> dict[str, Any]:
        """Seal the feature and coefficient snapshot before later feedback."""
        if not event_id or event_id in self.state["pending"] or event_id in self.state["released"]:
            raise ValueError("duplicate prediction")
        if tick <= self.state["last_tick"]:
            raise ValueError("prediction order")
        if set(features) != set(NAMES) or any(
            value not in (0.0, 1.0) for value in features.values()
        ):
            raise ValueError("invalid predicate features")
        if not math.isfinite(base_probability) or not 0 <= base_probability <= 1:
            raise ValueError("invalid base probability")
        probability = min(
            1.0,
            max(
                0.0,
                base_probability
                + sum(
                    features[name] * coefficient
                    for name, coefficient in zip(NAMES, self.state["coefficients"], strict=True)
                ),
            ),
        )
        receipt = {
            "event_id": event_id,
            "tick": tick,
            "features": features,
            "base_probability": base_probability,
            "probability": probability,
            "bank_hash_before": canonical_hash(self.state["coefficients"]),
        }
        if not read_only:
            self.state["pending"][event_id] = receipt
            self.state["last_tick"] = tick
            self._save("predict", {"event_id": event_id, "tick": tick})
        return receipt

    def release(self, event_id: str, tick: int, label: int) -> None:
        """Update only after a later tick, using the prediction made beforehand."""
        row = self.state["pending"].get(event_id)
        if row is None:
            raise ValueError("unknown pending prediction")
        if tick < row["tick"] + DELAY_TICKS:
            raise ValueError("early feedback")
        if label not in (0, 1):
            raise ValueError("invalid feedback label")
        residual = label - row["probability"]
        for index, name in enumerate(NAMES):
            value = (
                self.state["coefficients"][index] + LEARNING_RATE * residual * row["features"][name]
            )
            self.state["coefficients"][index] = min(1.0, max(-1.0, value))
        self.state["released"][event_id] = {**row, "release_tick": tick, "label": label}
        del self.state["pending"][event_id]
        self._save("release", {"event_id": event_id, "tick": tick, "label": label})

    def admit(self, event_id: str, name: str) -> None:
        """Admit at most one active predicate per block from positive feedback."""
        row = self.state["released"].get(event_id)
        if row is None or row["label"] != 1 or name not in NAMES or row["features"][name] != 1:
            raise ValueError("admission requires released positive evidence")
        block = str(row["tick"] // BLOCK_SIZE)
        if block in self.state["admitted"]:
            raise ValueError("admission block already used")
        self.state["admitted"][block] = {"event_id": event_id, "name": name}
        self._save("admit", {"event_id": event_id, "name": name, "block": block})
