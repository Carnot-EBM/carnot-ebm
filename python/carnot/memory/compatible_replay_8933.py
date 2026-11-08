"""REQ-REPORT-8933: replay only feedback that was already released for training.

The signed energy head and operational guard stay fixed. Distance reads visible
features only; oracle labels train the head after the delayed release boundary.
"""

from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from carnot.memory import guarded_memory_8905 as g
from carnot.memory import signed_memory_8892 as signed
from carnot.memory import causal_memory_8802 as primitive
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting import topology_memory_methods_8881 as old
from carnot.reporting.causal_learning_audit_8810 import require
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
SCHEMA = "signed-compatible-replay-v1"
ARMS = (
    "frozen_verified",
    "guarded_current",
    "uniform_replay",
    "compatible_replay",
    "compatible_permuted",
)


def nearest(params: Json, visible: list[float], buffer: list[Json]) -> str:
    """Reuse label-blind signature matching with squared signed-feature distance.

    Equal signatures are legitimate collisions. Stable ids resolve their ties;
    duplicate ids are rejected so copies cannot receive extra replay weight.
    """
    require(len({r["group_id"] for r in buffer}) == len(buffer), "duplicate_replay_id")
    x = np.asarray(signed.vector(params, visible))
    return str(
        min(
            buffer,
            key=lambda r: (
                float(np.sum((np.asarray(signed.vector(params, r["counts"])) - x) ** 2)),
                r["group_id"],
            ),
        )["group_id"]
    )


def selector_controls(params: Json) -> list[Json]:
    """Execute the old collision/permutation controls without borrowing its scores."""
    a: Json = dict(group_id="a", counts=[1.0] * 8, label="support")
    b: Json = dict(group_id="b", counts=[2.0] * 8, label="refute")
    collision = dict(b, counts=a["counts"])
    return [
        dict(
            check="signature_permutation",
            passed=nearest(params, a["counts"], [a, b])
            != nearest(
                params, a["counts"], [dict(a, counts=b["counts"]), dict(b, counts=a["counts"])]
            ),
        ),
        dict(
            check="collision_injection", passed=nearest(params, a["counts"], [collision, a]) == "a"
        ),
        dict(check="all_compatible", passed=nearest(params, a["counts"], [a, collision]) == "a"),
        dict(check="none_exactly_compatible", passed=nearest(params, [3.0] * 8, [a, b]) == "b"),
        dict(
            check="label_permutation_blind",
            passed=nearest(params, a["counts"], [dict(a, label="refute"), dict(b, label="support")])
            == "a",
        ),
    ]


class Store(g.Store):
    """Persist the buffer, replay choices and RNG with the existing atomic transaction."""

    def __init__(
        self, path: Path, params: Json, arm: str, anchors: list[Json], *, seed: int = 7648933
    ) -> None:
        require(arm in ARMS and len(anchors) == 16, "store_configuration")
        self.path, self.optimizer = path, "sgd"
        state: Json = (
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
                buffer=[],
                selection_rows=[],
                gradient_ledger=[],
                visible_counts=None,
                rng_state=np.random.default_rng(seed).bit_generator.state,
            )
        )
        require(
            state["schema"] == SCHEMA and state["arm"] == arm and state["anchors"] == anchors,
            "state_schema",
        )
        if params:
            require(
                all(
                    params[k] == state["committed"][k]
                    for k in ("feature_schema", "feature_arm", "scale")
                ),
                "state_schema",
            )
        self.commit(state)

    def commit(self, state: Json) -> None:
        """Reject corrupt replay state before either the file or serving head can change."""
        for key in ("committed", "candidate"):
            signed.vector(state[key], [0.0] * 8)
        buffer = state["buffer"]
        require(
            len(buffer) <= 32 and len({r["group_id"] for r in buffer}) == len(buffer),
            "buffer_capacity_or_duplicate",
        )
        require(
            buffer == [r for r in state["released"] if r["split"] == "train"][-32:],
            "buffer_release_boundary",
        )
        for row in buffer:
            require(
                row["split"] == "train" and g.split(row["group_id"]) == "train", "buffer_isolation"
            )
            signed.vector(state["candidate"], row["counts"])
            require(row["label"] in primitive.LABELS, "buffer_label")
        if state["pending"]:
            state["visible_counts"] = max(state["pending"].values(), key=lambda r: r["ordinal"])[
                "counts"
            ]
        if state["visible_counts"] is not None:
            signed.vector(state["candidate"], state["visible_counts"])
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        require(state["training_calls"] == len(state["gradient_ledger"]), "gradient_accounting")
        super().commit(state)

    def release(self, batch: list[tuple[str, Json]], tick: int) -> None:
        """Open ordered due labels only after every row has a durable prediction."""
        state = deepcopy(self.state)
        pending = [state["pending"].get(group) for group, _ in batch]
        require(bool(batch) and all(p is not None for p in pending), "release_unknown")
        for i, ((group, value), sealed) in enumerate(zip(batch, pending, strict=True)):
            require(
                sealed["ordinal"] == state["cursor"] + i + 1
                and tick >= sealed["due"]
                and canonical_hash(value) == sealed["evidence_hash"]
                and state["splits"][group] == sealed["split"],
                "release_boundary",
            )
        labels = [str(symbolic.decide(value)["decision"]) for _, value in batch]
        targets = list(labels)
        if state["arm"] == "compatible_permuted":
            indices = [i for i, row in enumerate(pending) if row["split"] == "train"]
            permuted = random.Random(g.SEED + state["cursor"]).sample(
                [labels[i] for i in indices], len(indices)
            )
            for i, target in zip(indices, permuted, strict=True):
                targets[i] = target
        state["blocks"].append(
            dict(groups=[group for group, _ in batch], labels=labels, targets=targets, tick=tick)
        )
        for (group, value), label, target in zip(batch, labels, targets, strict=True):
            row = dict(state["pending"].pop(group), label=target, true_label=label)
            if row["split"] == "train":
                self.train_release(state, row)
                state["buffer"] = (state["buffer"] + [row])[-32:]
            state["released"].append(row)
            key = old.topology(value["query"])
            if key not in state["templates"]:
                state["templates"].append(key)
            state["cursor"] += 1
            if state["cursor"] in g.SCHEDULE:
                self.fit_candidate(state)
        self.commit(state)

    def train_release(self, state: Json, row: Json) -> None:
        """Take exactly two gradients, including the empty-buffer repeated-row case."""
        if state["arm"] == "frozen_verified":
            return
        buffer = state["buffer"]
        selected = row
        if buffer and state["arm"] == "uniform_replay":
            rng = np.random.default_rng()
            rng.bit_generator.state = state["rng_state"]
            selected = buffer[int(rng.integers(len(buffer)))]
            state["rng_state"] = rng.bit_generator.state
        elif buffer and state["arm"] in ("compatible_replay", "compatible_permuted"):
            group = nearest(state["candidate"], state["visible_counts"], buffer)
            selected = next(r for r in buffer if r["group_id"] == group)
        state["selection_rows"].append(
            dict(
                released_id=row["group_id"],
                selected_id=selected["group_id"],
                visible_counts=state["visible_counts"],
                buffer_ids=[r["group_id"] for r in buffer],
                buffer_hash=canonical_hash(buffer),
                empty_buffer=not buffer,
                distance=float(
                    np.sum(
                        (
                            np.asarray(signed.vector(state["candidate"], selected["counts"]))
                            - np.asarray(signed.vector(state["candidate"], state["visible_counts"]))
                        )
                        ** 2
                    )
                ),
            )
        )
        for role, example in (("current", row), ("replay", selected)):
            before = canonical_hash(state["candidate"])
            state["candidate"], receipt = signed.update(
                state["candidate"], example["counts"], example["label"]
            )
            state["gradient_ledger"].append(
                dict(
                    receipt,
                    role=role,
                    released_id=row["group_id"],
                    training_id=example["group_id"],
                    before=before,
                    after=canonical_hash(state["candidate"]),
                )
            )
            state["training_calls"] += 1

    def fit_candidate(self, state: Json) -> None:
        """Apply the original guard thresholds and schedule without extra training."""
        guards = [r for r in state["released"] if r["split"] == "guard"][-8:]
        candidate = state["candidate"]
        recent_old, recent_new = g.metrics(state["committed"], guards), g.metrics(candidate, guards)
        anchor_old, anchor_new = (
            g.metrics(state["committed"], state["anchors"]),
            g.metrics(candidate, state["anchors"]),
        )
        eligible = bool(state["buffer"]) and len(guards) >= 4
        safe = (
            recent_old[0] - recent_new[0] >= 0.001
            and anchor_new[0] - anchor_old[0] <= 0.005
            and anchor_new[1] <= anchor_old[1]
        )
        promoted = eligible and safe and state["arm"] != "frozen_verified"
        state["candidate_version"] += 1
        state["guard_scores"] += 2 * (len(guards) + len(state["anchors"]))
        row = dict(
            feedback_count=state["cursor"],
            eligible=eligible,
            promoted=promoted,
            training_groups=[r["group_id"] for r in state["buffer"]],
            guard_groups=[r["group_id"] for r in guards],
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
