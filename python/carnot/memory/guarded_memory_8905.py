"""REQ-REPORT-8905: test signed-head candidates before atomic deployment.

Repeated guards are an operational heuristic. They provide no independent risk
certificate, and the store has no interface for research retention examples.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from carnot.memory import signed_memory_8892 as signed
from carnot.memory import causal_memory_8802 as primitive
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting import topology_memory_methods_8881 as old
from carnot.reporting.causal_learning_audit_8810 import require, score
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
SCHEMA = "signed-candidate-committed-v1"
ARMS = ("frozen_verified", "ordinary_true", "guarded_true", "guarded_permuted")
SCHEDULE = (16, 24, 32, 40, 48)
SEED = 7628905


def split(group_id: str) -> str:
    """Assign the whole group without inspecting its label or observations."""
    return "guard" if int(hashlib.sha256(group_id.encode()).hexdigest(), 16) % 4 == 0 else "train"


def metrics(params: Json, rows: list[Json]) -> tuple[float, float]:
    """Use the unchanged three-class Brier score and typed decision error."""
    predictions = [signed.probabilities(params, r["counts"]) for r in rows]
    brier = (
        sum(score(p, r["label"], 0.5)["metric"] for p, r in zip(predictions, rows, strict=True))
        / len(rows)
        if rows
        else 0.0
    )
    error = (
        sum(
            primitive.LABELS[max(range(3), key=p.__getitem__)] != r["label"]
            for p, r in zip(predictions, rows, strict=True)
        )
        / len(rows)
        if rows
        else 0.0
    )
    return brier, error


class Store:
    """Keep both head versions and the replay boundary in one durable transaction."""

    def __init__(self, path: Path, params: Json, arm: str, anchors: list[Json]) -> None:
        self.path = path
        require(arm in ARMS and len(anchors) == 16, "store_configuration")
        self.state: Json = (
            json.loads(path.read_bytes())
            if path.exists()
            else dict(
                schema=SCHEMA,
                arm=arm,
                anchors=deepcopy(anchors),
                committed=deepcopy(params),
                candidate=deepcopy(params),
                committed_version=0,
                candidate_version=0,
                splits={},
                pending={},
                released=[],
                blocks=[],
                templates=[],
                cursor=0,
                prediction_cursor=0,
                promotions=[],
                training_calls=0,
                guard_scores=0,
            )
        )
        require(
            self.state["schema"] == SCHEMA
            and self.state["arm"] == arm
            and self.state["anchors"] == anchors,
            "state_schema",
        )
        for name in ("committed", "candidate"):
            signed.vector(self.state[name], [0.0] * 8)
        if params:
            require(
                all(
                    params[k] == self.state["committed"][k]
                    for k in ("feature_schema", "feature_arm", "scale")
                ),
                "state_schema",
            )
        if not path.exists():
            self.commit(self.state)

    def commit(self, state: Json) -> None:
        """Publish the file before memory so a failed write cannot deploy weights."""
        atomic_json(self.path, state)
        self.state = deepcopy(state)

    def predict(self, group: str, payload: Json, ordinal: int, due: int) -> Json:
        """Seal a label-free prediction before any feedback can change its state."""
        require(
            ordinal == self.state["prediction_cursor"] + 1
            and due > ordinal
            and group not in self.state["splits"]
            and group not in {r["group_id"] for r in self.state["anchors"]},
            "prediction_boundary",
        )
        observation = signed.observe(payload, self.state["templates"])
        sealed = dict(
            observation,
            group_id=group,
            ordinal=ordinal,
            due=due,
            split=split(group),
            evidence_hash=canonical_hash(payload),
            parameter_hash=canonical_hash(self.state["committed"]),
            template_hash=canonical_hash(self.state["templates"]),
            prediction=signed.probabilities(self.state["committed"], observation["counts"]),
        )
        state = deepcopy(self.state)
        state["prediction_cursor"] = ordinal
        state["splits"][group] = sealed["split"]
        state["pending"][group] = sealed
        self.commit(state)
        return deepcopy(sealed)

    def release(self, batch: list[tuple[str, Json]], tick: int) -> None:
        """Validate a due, ordered block before opening its independent true labels."""
        state = deepcopy(self.state)
        pending = [state["pending"].get(group) for group, _ in batch]
        require(bool(batch) and all(p is not None for p in pending), "release_unknown")
        for index, ((group, value), sealed) in enumerate(zip(batch, pending, strict=True)):
            require(
                sealed["ordinal"] == state["cursor"] + index + 1
                and tick >= sealed["due"]
                and canonical_hash(value) == sealed["evidence_hash"]
                and state["splits"][group] == sealed["split"],
                "release_boundary",
            )
        labels = [str(symbolic.decide(value)["decision"]) for _, value in batch]
        targets = (
            random.Random(SEED + state["cursor"]).sample(labels, len(labels))
            if state["arm"] == "guarded_permuted"
            else labels
        )
        state["blocks"].append(
            dict(groups=[g for g, _ in batch], labels=labels, targets=targets, tick=tick)
        )
        for (group, value), label, target in zip(batch, labels, targets, strict=True):
            sealed = state["pending"].pop(group)
            state["released"].append(dict(sealed, label=target, true_label=label))
            key = old.topology(value["query"])
            if key not in state["templates"]:
                state["templates"].append(key)
            state["cursor"] += 1
            if state["cursor"] in SCHEDULE:
                self.fit_candidate(state)
        self.commit(state)

    def fit_candidate(self, state: Json) -> None:
        """Train only on released training groups; score identical work in every arm."""
        training = [r for r in state["released"] if r["split"] == "train"][-32:]
        guards = [r for r in state["released"] if r["split"] == "guard"][-8:]
        candidate = deepcopy(state["committed"])
        gradients = []
        for row in training:
            candidate, gradient = signed.update(candidate, row["counts"], row["label"])
            gradients.append(gradient)
        recent_old, recent_new = metrics(state["committed"], guards), metrics(candidate, guards)
        anchor_old = metrics(state["committed"], state["anchors"])
        anchor_new = metrics(candidate, state["anchors"])
        eligible = bool(training) and len(guards) >= 4
        safe = (
            recent_old[0] - recent_new[0] >= 0.001
            and anchor_new[0] - anchor_old[0] <= 0.005
            and anchor_new[1] <= anchor_old[1]
        )
        promoted = eligible and (
            state["arm"] == "ordinary_true" or (state["arm"].startswith("guarded_") and safe)
        )
        state["candidate"] = candidate
        state["candidate_version"] += 1
        state["training_calls"] += len(training)
        state["guard_scores"] += 2 * (len(guards) + len(state["anchors"]))
        row = dict(
            feedback_count=state["cursor"],
            eligible=eligible,
            promoted=promoted,
            training_groups=[r["group_id"] for r in training],
            guard_groups=[r["group_id"] for r in guards],
            gradients=gradients,
            recent_improvement=recent_old[0] - recent_new[0],
            anchor_brier_degradation=anchor_new[0] - anchor_old[0],
            anchor_error_increase=anchor_new[1] - anchor_old[1],
            before=canonical_hash(state["committed"]),
            candidate=canonical_hash(candidate),
            candidate_version=state["candidate_version"],
        )
        if promoted:
            state["committed"] = deepcopy(candidate)
            state["committed_version"] = state["candidate_version"]
        row.update(
            after=canonical_hash(state["committed"]), committed_version=state["committed_version"]
        )
        state["promotions"].append(row)
