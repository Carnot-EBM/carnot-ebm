"""Disjoint public-byte adapter over the existing numerical head.

REQ-VERIFY-7853 keeps role and group checks outside the fixture adapter.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from carnot.verify import evidence_views
from carnot.verify import natural_predicates
from carnot.verify import training_runtime as runtime

ARMS = tuple(evidence_views.ARMS)
SEEDS = (67801, 67802, 67803)
LEARNING_RATE = 0.01
MAX_EPOCHS = 16


def _bytes(value: str | bytes) -> bytes:
    return value.encode("utf-8") if isinstance(value, str) else value


def prepare(records: list[dict[str, Any]], arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build real views before labels are copied into the numeric target arrays."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record in records:
        source = _bytes(record["source"])
        answer = _bytes(record["answer"])
        if arm == "source_erased_constrained_set":
            source = b""
        pair = evidence_views.prepare_views(source, answer)
        reason = pair["a"]["abstention"]
        if reason:
            excluded.append({"id": record["id"], "reason": reason})
            continue
        eligible.append(
            {
                "view_a": pair["a"],
                "view_b": pair["b"],
                "label": record.get("label", 0),
                "known": record.get("known", [-1] * len(pair["a"]["answer_units"])),
                "source": source,
                "answer": answer,
            }
        )
    if not eligible:
        raise ValueError("no eligible records")
    batch = runtime.prepare_batch(eligible)
    if arm == "complete_static_constrained_set":
        values = np.asarray(
            [
                list(natural_predicates.features(row["source"], row["answer"]).values())
                for row in eligible
            ]
        )
        for key in ("a", "b"):
            x = np.asarray(batch[key]["x"])
            extra = np.broadcast_to(values[:, None, None, :], (*x.shape[:3], 16))
            batch[key] = {**batch[key], "x": jnp.asarray(np.concatenate((x, extra), axis=-1))}
    return batch, excluded


def fit(
    train: list[dict[str, Any]],
    tune: list[dict[str, Any]],
    arm: str,
    seed: int,
    learning_rate: float,
    epochs: int,
) -> dict[str, Any]:
    """Select temperature only on disjoint tune groups after fitting train rows."""
    if seed not in SEEDS or learning_rate != LEARNING_RATE or not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError("unregistered natural budget")
    if {row["group"] for row in train} & {row["group"] for row in tune}:
        raise ValueError("fit tune group overlap")
    if any(row.get("role") != "fit" for row in train) or any(
        row.get("role") != "tune" for row in tune
    ):
        raise ValueError("fit tune role mismatch")
    if any("label" not in row for row in [*train, *tune]):
        raise ValueError("fit tune labels required")
    train_batch, train_excluded = prepare(train, arm)
    tune_batch, tune_excluded = prepare(tune, arm)
    if train_excluded or tune_excluded:
        raise ValueError("ineligible fit tune row")
    config = evidence_views.ARMS[arm]
    head_arm = "response_set" if arm == "response_set" else config["head"]
    initial = runtime.init_params(head_arm, seed)
    if arm == "complete_static_constrained_set":
        initial = {**initial, "w": jnp.pad(initial["w"], ((0, 16), (0, 0)))}
    result = runtime.fit(
        head_arm,
        train_batch,
        tune_batch,
        seed,
        learning_rate,
        epochs,
        config["mode"],
        initial=initial,
    )
    result["temperature"] = runtime.calibrate(
        result["params"], tune_batch, head_arm, config["paired"]
    )
    result["paired"] = config["paired"]
    result["view_arm"] = arm
    result["gradient_error"] = runtime.gradient_error(initial, train_batch, head_arm)
    result["parameter_count"] = runtime.parameter_count(
        result["params"], 2 if config["mode"] == "constrained" else 0
    )
    return result


def predict(head: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predict from public bytes; caller labels are not used by the head."""
    batch, excluded = prepare(records, head["view_arm"])
    if excluded:
        raise ValueError("ineligible prediction row")
    return [runtime.decide(head, batch, index, head["paired"]) for index in range(len(records))]
