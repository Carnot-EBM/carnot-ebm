"""V676 fixture adapter for REQ-VERIFY-7769.

This layer chooses registered fixture arms and turns complete source bytes into
the same numeric batches consumed by the Exp7755 runtime. It has no labels
other than those supplied by the private fixture caller.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from carnot.experiment_7742_v674_bank_qualification import sentence_features
from carnot.verify import evidence_views
from carnot.verify import training_runtime as runtime

ARMS = tuple(evidence_views.ARMS)


def fixture_records() -> list[dict[str, Any]]:
    """Return four small, separable synthetic responses with known local labels."""
    return [
        {
            "id": "fixture-0",
            "source": "Alpha is 12.",
            "answer": "Alpha is 12.",
            "label": 0,
            "known": [1],
        },
        {
            "id": "fixture-1",
            "source": "Alpha is 12.",
            "answer": "Alpha is 13.",
            "label": 1,
            "known": [0],
        },
        {
            "id": "fixture-2",
            "source": "Beta is 30.",
            "answer": "Beta is 30.",
            "label": 0,
            "known": [1],
        },
        {
            "id": "fixture-3",
            "source": "Beta is 30.",
            "answer": "Beta is 31.",
            "label": 1,
            "known": [0],
        },
    ]


def make_batch(
    records: list[dict[str, Any]], arm: str, names: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build both label-free views and retain every excluded record."""
    if arm not in ARMS:
        raise ValueError("unknown qualification arm")
    if arm == "complete_static_constrained_set" and (len(names) != 16 or len(set(names)) != 16):
        raise ValueError("sixteen predicates required")
    eligible, excluded = [], []
    for index, record in enumerate(records):
        source = record["source"].encode()
        answer = record["answer"].encode()
        if arm == "source_erased_constrained_set":
            source = b""
        pair = evidence_views.prepare_views(source, answer)
        reason = pair["a"]["abstention"]
        if reason:
            excluded.append({"id": record["id"], "reason": reason, "index": index})
            continue
        eligible.append(
            {
                "view_a": pair["a"],
                "view_b": pair["b"],
                "label": record["label"],
                "known": record["known"],
                "id": record["id"],
            }
        )
    batch = runtime.prepare_batch(eligible)
    if arm == "complete_static_constrained_set":
        for view in ("a", "b"):
            values = np.asarray(
                [
                    list(sentence_features({"unit_id": row["id"]}, names).values())
                    for row in eligible
                ]
            )
            x = np.asarray(batch[view]["x"])
            extra = np.broadcast_to(values[:, None, None, :], (*x.shape[:3], 16))
            batch[view] = {**batch[view], "x": jnp.asarray(np.concatenate((x, extra), axis=-1))}
    return batch, excluded


def averaged_logistic_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Pool within each view and average features before the shared head."""
    pooled = []
    for key in ("a", "b"):
        view = batch[key]
        pooled.append(jnp.einsum("buld,bl->bud", view["x"], view["prior"]))
    features = ((pooled[0] + pooled[1]) / 2)[:, :, None, :]
    merged = {
        **batch["a"],
        "x": features,
        "prior": jnp.ones((features.shape[0], 1)),
        "place_mask": jnp.ones((features.shape[0], 1)),
    }
    return {**batch, "a": merged, "b": merged}


def aggregate_temperature(a: float, b: float, temperature: float) -> float:
    """Apply temperature once to the arithmetic mean of raw view risks."""
    return runtime.temperature_risk((a + b) / 2, temperature)


def fit_arm(
    arm: str, batch: dict[str, Any], *, epochs: int = 2, static_names: list[str] | None = None
) -> dict[str, Any]:
    """Fit one registered arm with the existing normalized optimizer."""
    config = evidence_views.ARMS[arm]
    head_arm = "response_set" if arm == "response_set" else config["head"]
    initial = runtime.init_params(head_arm, runtime.SEEDS[0])
    static_names = static_names or []
    if arm == "complete_static_constrained_set":
        static_names = static_names or [f"predicate_{index}" for index in range(16)]
        initial = {**initial, "w": jnp.pad(initial["w"], ((0, 16), (0, 0)))}
    result = runtime.fit(
        head_arm,
        batch,
        batch,
        runtime.SEEDS[0],
        runtime.LEARNING_RATES[1],
        epochs,
        config["mode"],
        initial=initial,
    )
    result["temperature"] = runtime.calibrate(result["params"], batch, head_arm, config["paired"])
    result["head_arm"] = head_arm
    result["paired"] = config["paired"]
    result["static_predicates"] = static_names
    result["static_coefficients"] = {
        name: float(result["params"]["w"][132 + index, 1] - result["params"]["w"][132 + index, 0])
        for index, name in enumerate(static_names)
    }
    result["parameter_count"] = runtime.parameter_count(
        result["params"], 2 if config["mode"] == "constrained" else 0
    )
    result["gradient_error"] = runtime.gradient_error(initial, batch, head_arm)
    return result


class NaturalHeadAdapter:
    """Use the fixture runtime's aggregate, temperature and action functions."""

    def __init__(self, head: dict[str, Any]) -> None:
        self.head = head

    def decide(self, batch: dict[str, Any], index: int) -> dict[str, Any]:
        """Return the same typed decision used by training and cold replay."""
        paired = bool(self.head["paired"])
        if self.head["arm"] == "logistic_local" and paired:
            batch = averaged_logistic_batch(batch)
        return runtime.decide(self.head, batch, index, paired)
