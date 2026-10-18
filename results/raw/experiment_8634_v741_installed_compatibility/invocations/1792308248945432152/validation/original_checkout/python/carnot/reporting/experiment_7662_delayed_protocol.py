"""Bounded delayed count/residual update service (REQ-REPORT-7662).

Only released update labels affect the candidate statistics. Admission labels
are read once for a held-out decision and never enter a gradient or count.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


PARTITIONS = {"update", "admission"}
STATE_LIMIT_BYTES = 100_000


def source_stratum(feature: dict[str, Any]) -> int:
    """Map the measured partial source scope to one of eight fixed bins."""

    checked = int(feature.get("checked_structural_propositions", 0)) > 0
    unknown = int(feature.get("unknown_claims", 0)) > 0
    contradiction = int(feature.get("scoped_contradictions", 0)) > 0
    return int(checked) + 2 * int(unknown) + 4 * int(contradiction)


def _numerical() -> dict[str, list[float]]:
    return {"count": [0.0] * 8, "residual": [0.0] * 8}


def _probability(base: float, stratum: int, numerical: dict[str, list[float]]) -> float:
    count = numerical["count"][stratum]
    residual = numerical["residual"][stratum]
    return min(1 - 1e-6, max(1e-6, base + residual / (count + 8.0)))


def _brier(probability: float, label: int) -> float:
    return (probability - label) ** 2


class DelayedUpdateService:
    """Persist every acknowledged event and reload exactly that JSON state."""

    def __init__(self, state_path: Path, *, arm: str = "source") -> None:
        if arm not in {"source", "scalar", "frozen"}:
            raise ValueError("arm_invalid")
        self.state_path = Path(state_path)
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state["arm"] != arm:
                raise ValueError("arm_state_mismatch")
        else:
            self.state = {
                "arm": arm,
                "numerical": _numerical(),
                "predictions": {},
                "feedback": {},
                "used_updates": [],
                "used_admissions": [],
                "proposal": None,
                "last_origin": -1,
                "last_release_ordinal": -1,
                "acknowledgments": 0,
            }
            self._persist()

    @property
    def state_hash(self) -> str:
        return canonical_hash(self.state)

    @property
    def numerical_hash(self) -> str:
        return canonical_hash(self.state["numerical"])

    @property
    def state_bytes(self) -> int:
        return self.state_path.stat().st_size

    def _persist(self) -> None:
        encoded = json.dumps(self.state, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode()) > STATE_LIMIT_BYTES:
            raise ValueError("state_bytes_exceeded")
        atomic_json(self.state_path, self.state)
        if json.loads(self.state_path.read_text(encoding="utf-8")) != self.state:
            raise OSError("durable_reload_mismatch")

    def _ack(self) -> dict[str, Any]:
        self.state["acknowledgments"] += 1
        self._persist()
        return {"acknowledged": True, "durable": True, "state_hash": self.state_hash}

    def predict(
        self, event_id: str, origin: int, base_probability: float, stratum: int, partition: str
    ) -> dict[str, Any]:
        """Record a label-free forecast before its source origin can release."""

        if not event_id or event_id in self.state["predictions"]:
            raise ValueError("duplicate_prediction")
        if origin <= self.state["last_origin"]:
            raise ValueError("origin_order")
        if partition not in PARTITIONS:
            raise ValueError("partition_invalid")
        if not 0 <= stratum < 8:
            raise ValueError("stratum_invalid")
        if not math.isfinite(base_probability) or not 0 <= base_probability <= 1:
            raise ValueError("probability_invalid")
        index = stratum if self.state["arm"] == "source" else 0
        probability = _probability(base_probability, index, self.state["numerical"])
        self.state["predictions"][event_id] = {
            "origin": origin,
            "base": base_probability,
            "stratum": index,
            "partition": partition,
            "probability": probability,
            "state_hash": self.numerical_hash,
        }
        self.state["last_origin"] = origin
        return {"event_id": event_id, "probability": probability, **self._ack()}

    def feedback_status(self, event_id: str) -> str:
        return self.state["feedback"].get(event_id, {}).get("status", "unreleased")

    def _release_check(self, event_id: str, release_ordinal: int) -> dict[str, Any]:
        prediction = self.state["predictions"].get(event_id)
        if prediction is None:
            raise ValueError("unknown_prediction")
        if event_id in self.state["feedback"]:
            raise ValueError("duplicate_release")
        if release_ordinal < prediction["origin"] + 8:
            raise ValueError("feedback_too_early")
        if release_ordinal < self.state["last_release_ordinal"]:
            raise ValueError("release_order")
        pending_origins = [
            row["origin"]
            for name, row in self.state["predictions"].items()
            if name not in self.state["feedback"]
        ]
        if prediction["origin"] != min(pending_origins):
            raise ValueError("release_order")
        return prediction

    def release(self, event_id: str, label: int, release_ordinal: int) -> dict[str, Any]:
        """Acknowledge only the real binary label for the oldest eligible event."""

        prediction = self._release_check(event_id, release_ordinal)
        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("binary_label_required")
        self.state["feedback"][event_id] = {
            "status": "released",
            "label": label,
            "release_ordinal": release_ordinal,
            "partition": prediction["partition"],
        }
        self.state["last_release_ordinal"] = release_ordinal
        return self._ack()

    def mark_missing(self, event_id: str, release_ordinal: int) -> dict[str, Any]:
        """Advance the legal release cursor without fabricating feedback."""

        prediction = self._release_check(event_id, release_ordinal)
        self.state["feedback"][event_id] = {
            "status": "missing",
            "label": None,
            "release_ordinal": release_ordinal,
            "partition": prediction["partition"],
        }
        self.state["last_release_ordinal"] = release_ordinal
        return self._ack()

    def propose(self, update_ids: list[str]) -> dict[str, Any]:
        """Propose one pure count/residual update from five released updates."""

        if self.state["proposal"] is not None:
            raise ValueError("proposal_pending")
        if len(update_ids) != 5 or len(set(update_ids)) != 5:
            raise ValueError("five_distinct_updates_required")
        if any(event_id in self.state["used_updates"] for event_id in update_ids):
            raise ValueError("update_used")
        for event_id in update_ids:
            prediction = self.state["predictions"].get(event_id, {})
            feedback = self.state["feedback"].get(event_id, {})
            if prediction.get("partition") != "update" or feedback.get("status") != "released":
                raise ValueError("unreleased_update")
        prior = self.state["numerical"]
        candidate = deepcopy(prior)
        started = time.perf_counter_ns()
        for event_id in update_ids:
            prediction = self.state["predictions"][event_id]
            feedback = self.state["feedback"][event_id]
            stratum = prediction["stratum"]
            # The stored pre-release forecast supplies the proper-loss residual.
            candidate["count"][stratum] += 1.0
            candidate["residual"][stratum] += feedback["label"] - prediction["probability"]
        elapsed = max(1, time.perf_counter_ns() - started)
        proposal = {
            "update_ids": list(update_ids),
            "prior_hash": canonical_hash(prior),
            "candidate_hash": canonical_hash(candidate),
            "candidate": candidate,
            "update_ns": elapsed,
        }
        self.state["proposal"] = proposal
        self._ack()
        return deepcopy(proposal)

    def admit(self, admission_ids: list[str]) -> dict[str, Any]:
        """Consume five held-out labels once and keep or roll back the proposal."""

        if any(event_id in self.state["used_admissions"] for event_id in admission_ids):
            raise ValueError("admission_used")
        proposal = self.state["proposal"]
        if proposal is None:
            raise ValueError("proposal_missing")
        if len(admission_ids) != 5 or len(set(admission_ids)) != 5:
            raise ValueError("five_distinct_admissions_required")
        for event_id in admission_ids:
            prediction = self.state["predictions"].get(event_id, {})
            feedback = self.state["feedback"].get(event_id, {})
            if prediction.get("partition") != "admission" or feedback.get("status") != "released":
                raise ValueError("admission_not_released")
        if proposal["prior_hash"] != self.numerical_hash:
            raise ValueError("stale_proposal")
        prior_loss = 0.0
        candidate_loss = 0.0
        for event_id in admission_ids:
            prediction = self.state["predictions"][event_id]
            label = self.state["feedback"][event_id]["label"]
            stratum = prediction["stratum"]
            prior_loss += _brier(
                _probability(prediction["base"], stratum, self.state["numerical"]), label
            )
            candidate_loss += _brier(
                _probability(prediction["base"], stratum, proposal["candidate"]), label
            )
        accepted = self.state["arm"] != "frozen" and candidate_loss < prior_loss
        if accepted:
            self.state["numerical"] = deepcopy(proposal["candidate"])
        self.state["used_updates"].extend(proposal["update_ids"])
        self.state["used_admissions"].extend(admission_ids)
        self.state["proposal"] = None
        acknowledgment = self._ack()
        return {
            "accepted": accepted,
            "prior_hash": proposal["prior_hash"],
            "candidate_hash": proposal["candidate_hash"],
            "state_hash_after": self.numerical_hash,
            "prior_admission_brier": prior_loss / 5,
            "candidate_admission_brier": candidate_loss / 5,
            "admission_count": 5,
            "update_ns": proposal["update_ns"],
            **acknowledgment,
        }

    def simulate_crash_before_ack(self) -> None:
        """Drop a hypothetical unacknowledged operation without touching disk."""

        _uncommitted = deepcopy(self.state)
        _uncommitted["acknowledgments"] += 1
