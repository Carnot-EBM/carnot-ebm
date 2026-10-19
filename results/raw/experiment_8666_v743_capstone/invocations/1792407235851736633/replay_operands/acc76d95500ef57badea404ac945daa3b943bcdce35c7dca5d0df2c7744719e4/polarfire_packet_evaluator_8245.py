"""REQ-VERIFY-8245: a standard-library receiver preserves the probability tree.

The receiver hashes complete saved state, including fields unused for scoring.
It applies each correction in order because clipping changes the final answer.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

Json = dict[str, Any]
SCHEMA = "carnot.polarfire.packet.v1"


def canonical(value: Any) -> bytes:
    """Stable finite JSON lets two machines compare the same bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(payload: bytes) -> str:
    """A byte hash binds the actual transport rather than a displayed summary."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def clip(p: float) -> float:
    """Reject invalid probabilities before preserving the original clipping bounds."""
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability")
    return min(1 - 1e-6, max(1e-6, p))


def predict(model: Json, row: Json) -> float | None:
    """Membership uses the frozen baseline, so corrections cannot move their own bins."""
    if row["p"] is None:
        return None
    kind = model["kind"]
    if kind == "input":
        return clip(float(row["p"]))
    if kind == "global":
        baseline = clip(float(row["p"]))
        z = model["scale"] * math.log(baseline / (1 - baseline)) + model["intercept"]
        return clip(1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z)))
    p = predict(model["base"], row)
    assert p is not None
    if kind == "mixture":
        q = predict(model["candidate"], row)
        assert q is not None
        return float((1 - model["step"]) * p + model["step"] * q)
    if kind != "patch":
        raise ValueError("model_kind")
    for op in model["patches"]:
        bounds, fp = op["group"]["interval"], row["baseline_p"]
        member = bool(
            (
                bounds is None
                or fp is not None
                and bounds[0] <= fp
                and (fp < bounds[1] or bounds[1] == fp == 1)
            )
            and (not op["group"]["reject_only"] or row["baseline_action"] == "reject")
        )
        p = min(1 - 1e-6, max(1e-6, p + op["delta"] * member))
    return p


def action(p: float | None, permission: str) -> str:
    """Baseline permission restricts acceptance without asserting correctness."""
    if permission not in {"accept", "reject", "escalate"}:
        raise ValueError("baseline_action")
    if p is None:
        return "escalate"
    clip(p)
    costs = {"reject": 1 - p, "escalate": 0.5}
    if permission == "accept":
        costs["accept"] = 5 * p
    winners = [a for a, cost in costs.items() if cost == min(costs.values())]
    return winners[0] if len(winners) == 1 else "escalate"


def evaluate(packet: Json) -> Json:
    """Reject transport drift before decoding or using the saved model."""
    if packet["schema"] != SCHEMA:
        raise ValueError("packet_schema")
    for name in ["state", "query"]:
        if digest(packet[name + "_bytes"].encode()) != packet[name + "_sha256"]:
            raise ValueError(name + "_hash")
    envelope, queries = json.loads(packet["state_bytes"]), json.loads(packet["query_bytes"])
    state = envelope["payload"]
    if envelope["version"] != 1 or state["schema_version"] != 1:
        raise ValueError("state_version")
    if digest(canonical(state)) != envelope["payload_sha256"]:
        raise ValueError("payload_hash")
    predictions = []
    for row in queries:
        p = predict(state["model"], row)
        predictions.append(
            dict(
                unit_id=row["unit_id"],
                p=p,
                action=action(p, row["baseline_action"]),
                baseline_action=row["baseline_action"],
            )
        )
    result = dict(
        state_sha256=packet["state_sha256"],
        query_sha256=packet["query_sha256"],
        output_sha256=digest(canonical(predictions)),
        predictions=predictions,
    )
    if result["output_sha256"] != packet["expected_output_sha256"]:
        raise ValueError("output_hash")
    return result


def main(argv: list[str]) -> int:
    """The same small receiver runs on host and board without third-party packages."""
    try:
        packet = json.loads(Path(argv[0]).read_bytes())
        if digest(Path(__file__).read_bytes()) != packet["evaluator_sha256"]:
            raise ValueError("evaluator_hash")
        print(json.dumps(evaluate(packet), sort_keys=True), flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(error=str(error))), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
