"""Reusable causal replay and paired reduction for REQ-REPORT-7663."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7660_atom_energy import score
from carnot.reporting.experiment_7662_delayed_protocol import (
    DelayedUpdateService,
    _brier,
    _probability,
    source_stratum,
)


def decision(probability: float, label: int) -> tuple[str, float]:
    """Apply the registered accept, reject, or escalation loss."""
    costs = {"accept": 5 * probability, "reject": 1 - probability, "escalate": 0.2}
    action = min(costs, key=costs.__getitem__)
    actual = {"accept": 5.0 * label, "reject": 1.0 - label, "escalate": 0.2}[action]
    return action, actual


class CostGuardedService(DelayedUpdateService):
    """Keep a proposed update only when both held-out outcomes allow it."""

    def admit(self, admission_ids: list[str]) -> dict[str, Any]:
        if any(event_id in self.state["used_admissions"] for event_id in admission_ids):
            raise ValueError("admission_used")
        proposal = self.state["proposal"]
        if proposal is None:
            raise ValueError("proposal_missing")
        if len(admission_ids) != 5 or len(set(admission_ids)) != 5:
            raise ValueError("five_distinct_admissions_required")
        if proposal["prior_hash"] != self.numerical_hash:
            raise ValueError("stale_proposal")
        for event_id in admission_ids:
            forecast = self.state["predictions"].get(event_id, {})
            feedback = self.state["feedback"].get(event_id, {})
            if forecast.get("partition") != "admission" or feedback.get("status") != "released":
                raise ValueError("admission_not_released")
        prior_loss = candidate_loss = prior_cost = candidate_cost = 0.0
        for event_id in admission_ids:
            forecast = self.state["predictions"][event_id]
            label = self.state["feedback"][event_id]["label"]
            stratum = forecast["stratum"]
            prior = _probability(forecast["base"], stratum, self.state["numerical"])
            candidate = _probability(forecast["base"], stratum, proposal["candidate"])
            prior_loss += _brier(prior, label)
            candidate_loss += _brier(candidate, label)
            prior_cost += decision(prior, label)[1]
            candidate_cost += decision(candidate, label)[1]
        accepted = (
            self.state["arm"] != "frozen"
            and candidate_loss < prior_loss
            and candidate_cost <= prior_cost
        )
        reason = (
            None
            if accepted
            else "frozen_control"
            if self.state["arm"] == "frozen"
            else "brier_not_improved"
            if candidate_loss >= prior_loss
            else "cost_increased"
        )
        if accepted:
            self.state["numerical"] = deepcopy(proposal["candidate"])
        self.state["used_updates"].extend(proposal["update_ids"])
        self.state["used_admissions"].extend(admission_ids)
        self.state["proposal"] = None
        ack = self._ack()
        return {
            "accepted": accepted,
            "rollback_reason": reason,
            "prior_hash": proposal["prior_hash"],
            "candidate_hash": proposal["candidate_hash"],
            "state_hash_after": self.numerical_hash,
            "prior_admission_brier": prior_loss / 5,
            "candidate_admission_brier": candidate_loss / 5,
            "prior_admission_cost": prior_cost / 5,
            "candidate_admission_cost": candidate_cost / 5,
            "admission_count": 5,
            "update_ns": proposal["update_ns"],
            **ack,
        }


def paired_block_interval(
    paired: list[tuple[float, float]], seed: int, *, draws: int = 10000
) -> dict:
    """Resample contiguous blocks of eight, preserving every paired source ID."""
    if not paired or len(paired) % 8:
        raise ValueError("block_roster")
    rng = random.Random(seed)
    blocks = [paired[index : index + 8] for index in range(0, len(paired), 8)]
    count = len(blocks)
    values = []
    for _ in range(draws):
        selected = [blocks[rng.randrange(count)] for _ in range(count)]
        values.append(
            sum(base - candidate for block in selected for candidate, base in block) / len(paired)
        )
    values.sort()
    return {
        "estimate": sum(base - candidate for candidate, base in paired) / len(paired),
        "lower_ci95": values[int(0.025 * draws)],
        "upper_ci95": values[min(draws - 1, int(0.975 * draws))],
        "effective_blocks": count,
        "block_size": 8,
        "draws": draws,
    }


def replay_arm(root: Path, inputs: list[dict], heads: dict, arm: str, *, restart: bool) -> dict:
    """Replay one frozen roster through the durable service and one-use guard."""
    if len(inputs) != 80 or len({item["unit_id"] for item in inputs}) != 80:
        raise ValueError("online_roster_invalid")
    path = root / f"{arm}_state.json"
    service_arm = arm if arm in {"source", "scalar", "frozen"} else "source"
    service = CostGuardedService(path, arm=service_arm)
    head = heads["heads"][heads["selected"]]
    rows: list[dict] = []
    feedback_rows: list[dict] = []
    decisions: list[dict] = []
    updates: list[str] = []
    admissions: list[str] = []
    released_past: list[int] = []
    predictions: dict[str, float] = {}
    prediction_times: dict[str, float] = {}
    started = time.monotonic()

    def advance() -> None:
        if (
            service.state["proposal"] is None
            and len(updates) >= 5
            and len(service.state["used_admissions"]) < 40
        ):
            chosen = updates[:5]
            del updates[:5]
            service.propose(chosen)
        if service.state["proposal"] is not None and len(admissions) >= 5:
            chosen = admissions[:5]
            del admissions[:5]
            update_ids = list(service.state["proposal"]["update_ids"])
            outcome = service.admit(chosen)
            decisions.append(
                {"arm": arm, "update_ids": update_ids, "admission_ids": chosen, **outcome}
            )
            advance()

    def release(origin: int, ordinal: int) -> None:
        item = inputs[origin]
        unit = item["unit_id"]
        missing = arm == "omission" and (origin + 1) % 4 == 0
        released_label: int | None = item["label"]
        label_from: str | None = unit
        if arm == "permuted" and item["partition"] == "update":
            if released_past:
                released_label = released_past[(len(released_past) // 2) % len(released_past)]
                label_from = "eligible_past_update_history"
            else:
                missing = True
        if missing:
            released_label = None
            label_from = None
            ack = service.mark_missing(unit, ordinal)
        else:
            assert released_label is not None
            ack = service.release(unit, released_label, ordinal)
            (updates if item["partition"] == "update" else admissions).append(unit)
        if item["partition"] == "update":
            released_past.append(item["label"])
        probability = predictions[unit]
        action, cost = decision(probability, item["label"])
        feature = item["feature"]
        rows.append(
            {
                "unit_id": unit,
                "arm": arm,
                "origin_ordinal": origin,
                "partition": item["partition"],
                "label": item["label"],
                "probability": probability,
                "brier": _brier(probability, item["label"]),
                "typed_action": action,
                "decision_cost": cost,
                "feedback_status": "missing" if missing else "released",
                "released_label": released_label,
                "label_from": label_from,
                "prediction_time_s": prediction_times[unit],
                "label_release_ordinal": ordinal,
                "excluded": feature["excluded"],
                "censored": feature["censored"],
                "counts": {"independent_group": 1, "paired_view": 1},
                "raw_metrics": {
                    "checked_structural_propositions": feature["checked_structural_propositions"],
                    "unknown_claims": feature["unknown_claims"],
                },
                "provenance": {
                    "source_sha256": feature["source_sha256"],
                    "label_sidecar": "Exp7602",
                },
            }
        )
        feedback_rows.append(
            {
                "unit_id": unit,
                "arm": arm,
                "prediction_time_s": prediction_times[unit],
                "label_release_ordinal": ordinal,
                "origin_ordinal": origin,
                "released_label": released_label,
                "label_from": label_from,
                "admission_identity": unit if item["partition"] == "admission" else None,
                "accepted_version": service.numerical_hash,
                "rollback_reason": decisions[-1]["rollback_reason"] if decisions else None,
                "acknowledgment": ack,
            }
        )
        advance()

    for origin, item in enumerate(inputs):
        unit = item["unit_id"]
        base = score(item["feature"], item["raw_probability"], head)
        prediction_times[unit] = time.monotonic() - started
        prediction = service.predict(
            unit, origin, base, source_stratum(item["feature"]), item["partition"]
        )
        predictions[unit] = prediction["probability"]
        atomic_json(
            root / f"{arm}_checkpoint.json",
            {
                "completed_units": origin + 1,
                "state_hash": service.state_hash,
            },
        )
        if origin >= 8:
            release(origin - 8, origin)
        if origin == 39 and restart:
            service = CostGuardedService(path, arm=service_arm)
        if origin % 10 == 9:
            print(
                f"[exp7663] replay arm={arm} completed={origin + 1}/80 elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    for origin in range(72, 80):
        release(origin, origin + 8)
    return {
        "rows": rows,
        "causal_feedback_rows": feedback_rows,
        "admission_decisions": decisions,
        "numerical_hash": service.numerical_hash,
        "state_hash": service.state_hash,
        "state_bytes": service.state_bytes,
        "numerical_state": deepcopy(service.state["numerical"]),
        "wall_time_per_event_s": (time.monotonic() - started) / 80,
    }
