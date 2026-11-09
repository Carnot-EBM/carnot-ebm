"""REQ-REPORT-8947: choose replay using only already released TRAIN gradients.

The original store owns prediction, guard and durable state boundaries. This
variant changes replay selection and records the scan that every comparator pays.
"""

from copy import deepcopy
from pathlib import Path
import random
from typing import Any

import jax.numpy as jnp
import numpy as np

from carnot.memory import compatible_replay_8933 as previous
from carnot.memory import signed_memory_8892 as signed
from carnot.memory import guarded_memory_8905 as g
from carnot.memory import causal_memory_8802 as primitive
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting import topology_memory_methods_8881 as old
from carnot.reporting.causal_learning_audit_8810 import require
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
SCHEMA = "signed-gradient-replay-v1"
ARMS = (
    "frozen_verified",
    "guarded_current",
    "uniform_replay",
    "signature_replay",
    "gradient_replay",
    "gradient_permuted",
)
SEED = 7658947
base = namespace(previous, "gradient8947_store")
base.SCHEMA, base.ARMS = SCHEMA, ARMS


def cosine(left: Any, right: Any) -> float:
    """A zero gradient has no direction, so it receives the declared zero score."""
    a, b = np.asarray(left, dtype=float).ravel(), np.asarray(right, dtype=float).ravel()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm else 0.0


def scan(params: Json, current: Json, buffer: list[Json]) -> list[Json]:
    """Evaluate each candidate at the same W1 without applying its proposed update."""
    rows = [current] + buffer
    require(
        len({r["group_id"] for r in rows}) == len(rows)
        and all(r["split"] == "train" and r["label"] in primitive.LABELS for r in rows),
        "candidate_boundary",
    )
    candidates = []
    for i, row in enumerate(rows):
        x = jnp.asarray(signed.vector(params, row["counts"]), dtype=jnp.float32)
        updated, loss, grad = signed.step(
            jnp.asarray(params["weights"][0], dtype=jnp.float32),
            x,
            jnp.asarray(primitive.LABELS.index(row["label"])),
        )
        candidates.append(
            dict(
                unit_id=row["group_id"],
                released_id=current["group_id"],
                role="current_W1" if i == 0 else "prior_W1",
                split="train",
                label=row["label"],
                counts=row["counts"],
                gradient=np.asarray(grad).tolist(),
                proposed_weights=np.asarray(updated).tolist(),
                gradient_hash=canonical_hash(np.asarray(grad).tolist()),
                gradient_norm=float(jnp.linalg.norm(grad)),
                loss=float(loss),
                at_weights=canonical_hash(params),
            )
        )
    for row in candidates:
        row["cosine"] = cosine(candidates[0]["gradient"], row["gradient"])
    return candidates


def select(candidates: list[Json]) -> str:
    """Exclude the current row unless no prior row exists; stable IDs break ties."""
    return str(
        min(candidates[1:] or candidates, key=lambda r: (-r["cosine"], r["unit_id"]))["unit_id"]
    )


def selector_controls(params: Json) -> list[Json]:
    """Opposite targets expose information that a feature-distance selector lacks."""
    current = dict(group_id="current", counts=[1.0] * 8, label="support", split="train")
    a = dict(current, group_id="a", label="refute")
    b = dict(current, group_id="b")
    return previous.selector_controls(params) + [
        dict(
            check="opposite_gradient_same_feature",
            passed=select(scan(params, current, [a, b])) == "b",
        ),
        dict(
            check="label_sensitive",
            passed=select(
                scan(params, current, [dict(a, label="support"), dict(b, label="refute")])
            )
            == "a",
        ),
        dict(check="zero_norm", passed=cosine(np.zeros((3, 9)), np.ones((3, 9))) == 0),
    ]


class Store(previous.Store):
    """Keep existing guard and checkpoint behavior in an isolated opt-in schema."""

    def __init__(
        self, path: Path, params: Json, arm: str, anchors: list[Json], *, seed: int = SEED
    ) -> None:
        base.Store.__init__(self, path, params, arm, anchors, seed=seed)

    def commit(self, state: Json) -> None:
        """Persist evaluated work separately so discarded scans cannot look free."""
        state.setdefault("candidate_gradient_rows", [])
        state.setdefault("evaluated_gradients", 0)
        require(
            state["evaluated_gradients"]
            == len(state["candidate_gradient_rows"]) + state["training_calls"] // 2,
            "evaluated_gradient_accounting",
        )
        super().commit(state)

    def release(self, batch: list[tuple[str, Json]], tick: int) -> None:
        """Validate every due seal before opening truth; permute TRAIN targets only."""
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
        if state["arm"] == "gradient_permuted":
            indices = [i for i, row in enumerate(pending) if row["split"] == "train"]
            permuted = random.Random(SEED + state["cursor"]).sample(
                [labels[i] for i in indices], len(indices)
            )
            for i, target in zip(indices, permuted, strict=True):
                targets[i] = target
        state["blocks"].append(
            dict(groups=[group for group, _ in batch], labels=labels, targets=targets, tick=tick)
        )
        for (group, value), label, target in zip(batch, labels, targets, strict=True):
            row = dict(
                state["pending"].pop(group), label=target, true_label=label, release_tick=tick
            )
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
        """Apply current once, scan at W1, then apply exactly the chosen scan gradient."""
        if state["arm"] == "frozen_verified":
            return
        buffer = state["buffer"]
        before = canonical_hash(state["candidate"])
        state["candidate"], receipt = signed.update(state["candidate"], row["counts"], row["label"])
        w1 = canonical_hash(state["candidate"])
        state["gradient_ledger"].append(
            dict(
                receipt,
                role="current",
                released_id=row["group_id"],
                training_id=row["group_id"],
                before=before,
                after=w1,
            )
        )
        candidates = scan(state["candidate"], row, buffer)
        selected = row["group_id"]
        if buffer and state["arm"] == "uniform_replay":
            rng = np.random.default_rng()
            rng.bit_generator.state = state["rng_state"]
            selected = buffer[int(rng.integers(len(buffer)))]["group_id"]
            state["rng_state"] = rng.bit_generator.state
        elif buffer and state["arm"] == "signature_replay":
            selected = previous.nearest(state["candidate"], state["visible_counts"], buffer)
        elif state["arm"] in ("gradient_replay", "gradient_permuted"):
            selected = select(candidates)
        chosen = next(r for r in candidates if r["unit_id"] == selected)
        state["selection_rows"].append(
            dict(
                released_id=row["group_id"],
                selected_id=selected,
                signature_id=previous.nearest(state["candidate"], state["visible_counts"], buffer)
                if buffer
                else row["group_id"],
                gradient_id=select(candidates),
                buffer_ids=[r["group_id"] for r in buffer],
                buffer_hash=canonical_hash(buffer),
                empty_buffer=not buffer,
                scan_count=len(candidates),
                selected_cosine=chosen["cosine"],
                at_weights=w1,
            )
        )
        state["candidate_gradient_rows"] += [
            {k: v for k, v in r.items() if k not in ("gradient", "proposed_weights")}
            for r in candidates
        ]
        state["evaluated_gradients"] += 1 + len(candidates)
        weights = jnp.asarray(state["candidate"]["weights"][0], dtype=jnp.float32)
        updated = jnp.asarray(chosen["proposed_weights"], dtype=jnp.float32)
        state["candidate"] = dict(state["candidate"], weights=[np.asarray(updated).tolist()])
        state["gradient_ledger"].append(
            dict(
                role="replay",
                released_id=row["group_id"],
                training_id=selected,
                before=w1,
                after=canonical_hash(state["candidate"]),
                loss=chosen["loss"],
                gradient_norm=chosen["gradient_norm"],
                gradient_hash=chosen["gradient_hash"],
                parameter_delta=float(jnp.linalg.norm(updated - weights)),
                optimizer_steps=1,
            )
        )
        state["training_calls"] += 2
