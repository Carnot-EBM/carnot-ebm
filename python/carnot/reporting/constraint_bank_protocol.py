"""Durable advisory constraint bank for REQ-CL-7705-BOUNDED-BANK.

The bank learns a small predictor adjustment from released feedback. It does
not change logical truth: the existing exact verifier keeps that authority.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.typed_decision_energy import FEATURE_ORDER


MAX_PENDING = 16
MAX_TEMPLATES = 36
MAX_PROPOSALS = 6
MAX_STEPS = 50
DELAY = 8
PERIODS = (2, 4, 6, 8, 10, 12)


def grammar() -> dict[str, Any]:
    """Return the already frozen feature names and all unordered pairs."""
    names = list(FEATURE_ORDER)
    return {
        "primitives": names,
        "pairs": [[names[i], names[j]] for i in range(8) for j in range(i + 1, 8)],
    }


def advisory_status(exact_status: str, candidate_active: bool) -> str:
    """Keep exact decisions even when a learned check fires."""
    if exact_status not in {"supported", "contradicted", "unknown"}:
        raise ValueError("exact_status_invalid")
    return exact_status


def _snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key != "ledger"}


def replay_ledger(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Cold-apply recorded state deltas and reject a broken event chain."""
    state: dict[str, Any] = {}
    previous = "genesis"
    for index, event in enumerate(ledger):
        if event["sequence"] != index or event["previous"] != previous:
            raise ValueError("ledger_mismatch")
        digest = canonical_hash(
            {key: event[key] for key in ("sequence", "previous", "kind", "detail", "patch")}
        )
        if digest != event["event_hash"]:
            raise ValueError("ledger_mismatch")
        state.update(deepcopy(event["patch"]))
        if canonical_hash(state) != event["state_hash"]:
            raise ValueError("ledger_mismatch")
        previous = digest
    return {"state": state, "state_hash": canonical_hash(state), "events": len(ledger)}


class Bank:
    """Persist every forecast and feedback before acknowledging the caller."""

    def __init__(
        self,
        path: Path,
        frozen_grammar: dict[str, Any],
        scheduler: str,
        threshold: float,
        period: int,
    ) -> None:
        if frozen_grammar != grammar():
            raise ValueError("grammar_invalid")
        if scheduler not in {"priority", "fixed", "read_only", "weight_only", "static"}:
            raise ValueError("scheduler_invalid")
        if scheduler == "fixed" and period not in PERIODS:
            raise ValueError("period_invalid")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold_invalid")
        self.path = Path(path)
        self.config = {
            "grammar": frozen_grammar,
            "scheduler": scheduler,
            "threshold": threshold,
            "period": period,
        }
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state["config"] != self.config:
                raise ValueError("state_config_mismatch")
            self.replay_ledger()
        else:
            self.state = {
                "config": self.config,
                "predictions": {},
                "feedback": {},
                "templates": [],
                "proposal": None,
                "scalar_count": 0,
                "scalar_residual": 0.0,
                "used_counterexamples": [],
                "used_admissions": [],
                "budget": {
                    "proposal_credits_spent": 0,
                    "admission_credits_spent": 0,
                    "gradient_steps_spent": 0,
                    "overflow_rejections": 0,
                },
                "last_tick": -1,
                "last_release_tick": -1,
                "ledger": [],
            }
            self._save("init", {}, {})

    @property
    def state_hash(self) -> str:
        return canonical_hash(_snapshot(self.state))

    @property
    def budget(self) -> dict[str, int]:
        return self.state["budget"]

    def replay_ledger(self) -> dict[str, Any]:
        result = replay_ledger(self.state["ledger"])
        if result["state"] != _snapshot(self.state):
            raise ValueError("ledger_mismatch")
        return result

    def _save(self, kind: str, detail: dict[str, Any], before: dict[str, Any]) -> None:
        current = _snapshot(self.state)
        patch = {
            key: deepcopy(value)
            for key, value in current.items()
            if key not in before or before[key] != value
        }
        ledger = self.state["ledger"]
        event = {
            "sequence": len(ledger),
            "previous": ledger[-1]["event_hash"] if ledger else "genesis",
            "kind": kind,
            "detail": detail,
            "patch": patch,
            "state_hash": canonical_hash(current),
        }
        event["event_hash"] = canonical_hash(
            {key: event[key] for key in ("sequence", "previous", "kind", "detail", "patch")}
        )
        ledger.append(event)
        atomic_json(self.path, self.state)

    def _before(self) -> dict[str, Any]:
        return deepcopy(_snapshot(self.state))

    def predict(
        self,
        event_id: str,
        tick: int,
        features: dict[str, float],
        base_probability: float,
        exact_status: str,
        partition: str,
        source_id: str,
    ) -> dict[str, Any]:
        """Freeze a label-free forecast and keep its Unicode source identity."""
        if not event_id or event_id in self.state["predictions"]:
            raise ValueError("duplicate_prediction")
        if tick <= self.state["last_tick"]:
            raise ValueError("prediction_order")
        if partition not in {"update", "admission", "retention"}:
            raise ValueError("partition_invalid")
        if set(features) != set(self.config["grammar"]["primitives"]) or any(
            not math.isfinite(value) or value < 0 for value in features.values()
        ):
            raise ValueError("features_invalid")
        if not math.isfinite(base_probability) or not 0 <= base_probability <= 1:
            raise ValueError("base_probability_invalid")
        if not source_id:
            raise ValueError("source_identity_missing")
        advisory_status(exact_status, False)
        pending = sum(name not in self.state["feedback"] for name in self.state["predictions"])
        if pending >= MAX_PENDING:
            before = self._before()
            self.budget["overflow_rejections"] += 1
            self._save("overflow", {"event_id": event_id, "tick": tick}, before)
            raise ValueError("pending_overflow")
        adjustment = sum(
            item["weight"]
            for item in self.state["templates"]
            if all(features[name] > 0 for name in item["pair"])
        )
        if self.config["scheduler"] == "weight_only":
            adjustment += self.state["scalar_residual"] / (self.state["scalar_count"] + 8)
        probability = min(1 - 1e-6, max(1e-6, base_probability + adjustment))
        before = self._before()
        self.state["predictions"][event_id] = {
            "tick": tick,
            "features": features,
            "base_probability": base_probability,
            "probability": probability,
            "exact_status": exact_status,
            "advisory_status": advisory_status(exact_status, adjustment != 0),
            "partition": partition,
            "source_id": source_id,
            "bank_hash_before": canonical_hash(self.state["templates"]),
        }
        self.state["last_tick"] = tick
        self._save("predict", {"event_id": event_id, "tick": tick, "partition": partition}, before)
        return {"event_id": event_id, "probability": probability, "state_hash": self.state_hash}

    def _check_feedback(self, event_id: str, tick: int) -> dict[str, Any]:
        row = self.state["predictions"].get(event_id)
        if row is None:
            raise ValueError("unknown_prediction")
        if event_id in self.state["feedback"]:
            raise ValueError("duplicate_feedback")
        if tick < row["tick"] + DELAY:
            raise ValueError("future_feedback")
        pending = [
            (value["tick"], key)
            for key, value in self.state["predictions"].items()
            if key not in self.state["feedback"]
        ]
        if tick < self.state["last_release_tick"] or event_id != min(pending)[1]:
            raise ValueError("feedback_order")
        return row

    def release(self, event_id: str, label: int, tick: int) -> dict[str, Any]:
        """Apply a real binary label once, after its original prediction."""
        row = self._check_feedback(event_id, tick)
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("label_invalid")
        before = self._before()
        self.state["feedback"][event_id] = {
            "status": "released",
            "label": label,
            "release_tick": tick,
            "loss": (row["probability"] - label) ** 2,
        }
        if self.config["scheduler"] == "weight_only" and row["partition"] == "update":
            self.state["scalar_count"] += 1
            self.state["scalar_residual"] += label - row["probability"]
        self.state["last_release_tick"] = tick
        self._save("release", {"event_id": event_id, "tick": tick, "label": label}, before)
        return {"event_id": event_id, "state_hash": self.state_hash}

    def mark_missing(self, event_id: str, tick: int) -> None:
        """Close a lost handle without inventing an outcome."""
        self._check_feedback(event_id, tick)
        before = self._before()
        self.state["feedback"][event_id] = {
            "status": "missing",
            "label": None,
            "release_tick": tick,
        }
        self.state["last_release_tick"] = tick
        self._save("missing", {"event_id": event_id, "tick": tick}, before)

    def propose(self) -> dict[str, Any] | None:
        """Freeze one conjunction using only already released update mistakes."""
        if any(feedback["status"] == "missing" for feedback in self.state["feedback"].values()):
            raise ValueError("missing_feedback")
        if self.config["scheduler"] in {"read_only", "weight_only", "static"}:
            return None
        if self.state["proposal"] is not None:
            raise ValueError("proposal_pending")
        if self.budget["proposal_credits_spent"] >= MAX_PROPOSALS:
            return None
        released = [
            (name, self.state["predictions"][name], feedback)
            for name, feedback in self.state["feedback"].items()
            if feedback["status"] == "released"
            and self.state["predictions"][name]["partition"] == "update"
        ]
        available = [
            (name, row, feedback)
            for name, row, feedback in released
            if name not in self.state["used_counterexamples"]
            and feedback["label"] == 1
            and row["probability"] < 0.5
            and any(
                all(row["features"][key] > 0 for key in pair)
                for pair in self.config["grammar"]["pairs"]
            )
        ]
        if not available:
            return None
        name, row, feedback = max(available, key=lambda item: (item[2]["loss"], item[0]))
        if self.config["scheduler"] == "priority" and feedback["loss"] < self.config["threshold"]:
            return None
        if self.config["scheduler"] == "fixed" and len(released) % self.config["period"]:
            return None
        pair = next(
            pair
            for pair in self.config["grammar"]["pairs"]
            if all(row["features"][key] > 0 for key in pair)
        )
        examples = [
            (example["probability"], item["label"])
            for _, example, item in released
            if all(example["features"][key] > 0 for key in pair)
        ]
        weight = 0.0
        for _ in range(MAX_STEPS):
            gradient = sum(probability + weight - label for probability, label in examples) / len(
                examples
            )
            weight = max(-0.4, min(0.4, weight - 0.02 * gradient))
        before = self._before()
        self.budget["proposal_credits_spent"] += 1
        self.budget["gradient_steps_spent"] += MAX_STEPS
        self.state["used_counterexamples"].append(name)
        proposal = {
            "pair": pair,
            "weight": weight,
            "gradient_steps": MAX_STEPS,
            "counterexample_id": name,
            "released_loss": feedback["loss"],
            "freeze_tick": feedback["release_tick"],
            "scheduler": self.config["scheduler"],
            "admission_decision": None,
        }
        self.state["proposal"] = proposal
        self._save(
            "propose",
            {"counterexample_id": name, "pair": pair, "freeze_tick": feedback["release_tick"]},
            before,
        )
        return deepcopy(proposal)

    def admit(self, accepted: bool, admission_id: str | None = None) -> dict[str, Any]:
        """Spend admission credit and either commit once or roll back."""
        proposal = self.state["proposal"]
        if proposal is None:
            raise ValueError("no_proposal")
        if self.budget["admission_credits_spent"] >= MAX_PROPOSALS:
            raise ValueError("admission_budget_exhausted")
        if admission_id is not None:
            feedback = self.state["feedback"].get(admission_id)
            row = self.state["predictions"].get(admission_id)
            if (
                row is None
                or feedback is None
                or row["partition"] != "admission"
                or feedback["status"] != "released"
            ):
                raise ValueError("admission_feedback_invalid")
            if admission_id in self.state["used_admissions"]:
                raise ValueError("admission_used")
        before = self._before()
        self.budget["admission_credits_spent"] += 1
        if admission_id is not None:
            self.state["used_admissions"].append(admission_id)
        proposal["admission_decision"] = "commit" if accepted else "rollback"
        if accepted:
            if len(self.state["templates"]) >= MAX_TEMPLATES:
                raise ValueError("template_overflow")
            self.state["templates"].append(deepcopy(proposal))
        self.state["proposal"] = None
        self._save(
            "commit" if accepted else "rollback",
            {"admission_id": admission_id, "pair": proposal["pair"]},
            before,
        )
        return {"accepted": accepted, "state_hash": self.state_hash}

    def reject(self, reason: str) -> dict[str, Any]:
        """Record a failed admission without refunding either credit."""
        result = self.admit(False)
        result["reason"] = reason
        return result

    def simulate_crash_before_commit(self) -> None:
        """A pre-commit crash has no acknowledged state transition."""
        if self.state["proposal"] is None:
            raise ValueError("no_proposal")
