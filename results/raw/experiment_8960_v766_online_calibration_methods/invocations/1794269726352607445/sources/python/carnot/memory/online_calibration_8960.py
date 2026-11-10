"""REQ-REPORT-8960: released numbers can change two scalars, never the frozen scorer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
ARMS = ("frozen", "bias_only", "adaptive", "permuted")
PERMUTATION = (1, 0, 3, 2)


def logit(p: float) -> float:
    """Clipping is only a finite logit conversion, never a fitted probability floor."""
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("base_probability")
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def probability(params: list[float], z: float) -> float:
    """Stable sigmoid avoids overflow without clipping the adapted prediction."""
    x = params[0] * z + params[1]
    e = math.exp(-abs(x))
    return 1 / (1 + e) if x >= 0 else e / (1 + e)


def step(params: list[float], zs: list[float], ys: list[int], arm: str) -> list[float]:
    """One mean-loss gradient per released block keeps feedback exposure equal."""
    if (
        arm not in ARMS
        or len(zs) != 4
        or len(ys) != 4
        or any(type(y) is not int or y not in (0, 1) for y in ys)
        or any(not math.isfinite(z) for z in zs)
        or len(params) != 2
        or not 0.25 <= params[0] <= 4
        or not -2 <= params[1] <= 2
    ):
        raise ValueError("released_block_or_parameters")
    if arm == "frozen":
        return [1.0, 0.0]
    residual = [probability(params, z) - y for z, y in zip(zs, ys, strict=True)]
    da = sum(e * z for e, z in zip(residual, zs, strict=True)) / 4 + 0.02 * (params[0] - 1)
    db = sum(residual) / 4 + 0.02 * params[1]
    return [
        1.0 if arm == "bias_only" else min(4, max(0.25, params[0] - 0.05 * da)),
        min(2, max(-2, params[1] - 0.05 * db)),
    ]


def guard(params: list[float], rows: list[tuple[float, int]]) -> Json:
    """Compare every proposal with the initial adapter, rather than a drifting baseline."""
    if len(rows) != 64 or any(type(y) is not int or y not in (0, 1) for _, y in rows):
        raise ValueError("guard_roster")
    logits = [logit(p) for p, _ in rows]
    base = [probability([1, 0], z) for z in logits]
    candidate = [probability(params, z) for z in logits]
    brier = sum((q - y) ** 2 - (p - y) ** 2 for q, p, (_, y) in zip(candidate, base, rows)) / 64
    error = (
        sum(
            int((q >= 0.5) != y) - int((p >= 0.5) != y)
            for q, p, (_, y) in zip(candidate, base, rows)
        )
        / 64
    )
    return dict(brier_increase=brier, error_increase=error, passed=brier <= 0.005 and error <= 0)


class LabelBroker:
    """Only this runtime boundary reads sealed online truth after a durable prediction."""

    def __init__(self, path: Path):
        self.path = path

    def release(self, query: int, seal: Path, *, role: str = "online") -> list[int]:
        """The final eight labels remain pending because the stream stops at query32."""
        if role != "online":
            raise PermissionError("retention_label_access_refused")
        if query < 8 or query > 32 or query % 4:
            return []
        try:
            state = json.loads(seal.read_text())
            if state["query"] != query or len(state["rows"]) != query * 4:
                raise ValueError("prediction_seal")
            if state["prediction_hash"] != canonical_hash(state["rows"]):
                raise ValueError("prediction_seal")
        except (OSError, KeyError) as error:
            raise ValueError("prediction_seal") from error
        values = json.loads(self.path.read_text())["labels"]
        if len(values) != 32 or any(type(y) is not int or y not in (0, 1) for y in values):
            raise ValueError("online_label_schema")
        return list(values[query - 8 : query - 4])


def trajectory(
    raw: Path,
    probabilities: list[float],
    label_path: Path,
    guard_rows: list[tuple[float, int]],
    *,
    restart: bool = False,
) -> Json:
    """A durable journal contains all state needed to resume exactly after query16."""
    if len(probabilities) != 32:
        raise ValueError("stream_roster")
    zs = [logit(p) for p in probabilities]
    state: Json = dict(
        query=0,
        params={arm: [1.0, 0.0] for arm in ARMS},
        rows=[],
        feedback_rows=[],
        update_rows=[],
        pending=[],
        released_through=0,
        numerical_training_calls=0,
    )
    broker, seal = LabelBroker(label_path), raw / "state.json"
    for query, z in enumerate(zs, 1):
        for arm in ARMS:
            p = probability(state["params"][arm], z)
            state["rows"].append(
                dict(
                    unit_id=f"private-{query:02d}",
                    cluster_id="private-stream",
                    arm=arm,
                    ordinal=query,
                    metric="p_error",
                    value=p,
                    numerator=p,
                    denominator=1,
                    condition="private_mechanics",
                    status="completed",
                    censor_reason=None,
                )
            )
        state.update(query=query, pending=list(range(state["released_through"] + 1, query + 1)))
        state["prediction_hash"] = canonical_hash(state["rows"])
        atomic_json(seal, state)
        released = broker.release(query, seal)
        if released:
            ordinals = list(range(query - 7, query - 3))
            permuted = [released[i] for i in PERMUTATION]
            state["feedback_rows"].append(
                dict(
                    query=query,
                    ordinals=ordinals,
                    labels=released,
                    permuted_labels=permuted,
                    prediction_hash=state["prediction_hash"],
                )
            )
            for arm in ARMS[1:]:
                before = list(state["params"][arm])
                candidate = step(
                    before,
                    zs[query - 8 : query - 4],
                    permuted if arm == "permuted" else released,
                    arm,
                )
                checked = guard(candidate, guard_rows)
                state["numerical_training_calls"] += 1
                if checked["passed"]:
                    state["params"][arm] = candidate
                state["update_rows"].append(
                    dict(
                        query=query,
                        arm=arm,
                        before=before,
                        candidate=candidate,
                        after=list(state["params"][arm]),
                        **checked,
                    )
                )
            state["pending"] = list(range(query - 3, query + 1))
            state["released_through"] = query - 4
        atomic_json(seal, state)
        if restart and query == 16:
            state = json.loads(seal.read_text())
            broker = LabelBroker(label_path)
    state["pending"] = list(range(25, 33))
    atomic_json(seal, state)
    return state
