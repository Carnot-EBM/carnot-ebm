"""REQ-REPORT-8968: one durable service arm pays only for its actual operations."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import resource
from typing import Any

from carnot.memory import online_calibration_8960 as adapter
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
ARMS = ("confidence_logistic", "frozen", "bias_only", "adaptive")
FEATURES = ("mean_token_nll", "first_answer_top2_margin")


def confidence(service: Json, features: Json) -> float:
    """The frozen linear ensemble needs two numbers and no graph or JAX runtime."""
    if set(features) != set(FEATURES) or any(
        type(features[k]) not in (int, float) or not math.isfinite(features[k]) or features[k] < 0
        for k in FEATURES
    ):
        raise ValueError("confidence_features")
    scores = []
    for state in service["states"]:
        scale = state["scaling"]
        x = [(features[k] - scale["mean"][i]) / scale["scale"][i] for i, k in enumerate(FEATURES)]
        weights, bias = state["params"]["head"]
        z = (sum(a * b for a, b in zip(x, weights, strict=True)) + bias) / service["temperature"]
        scores.append(adapter.probability([1, 0], z))
    return float(sum(scores) / len(scores))


def score(service: Json, rows: list[Json], arm: str, counts: Json) -> list[float]:
    """Reusing the original graph scorer keeps the measured neural path unchanged."""
    counts["scored_rows"] += len(rows)
    if arm == "confidence_logistic":
        return [confidence(service, row) for row in rows]
    from carnot.autoresearch import relational_graph_encoder as g

    counts["graph_pack_calls"] += 3
    counts["graph_encoded_rows"] += 3 * len(rows)
    counts["message_rounds"] += 9 * len(rows)
    return [float(p) for p in g.ensemble(service["states"], rows, service["temperature"])]


def commit(path: Path, state: Json, storage: Json) -> None:
    """The acknowledgement includes the directory barrier after atomic file replacement."""
    atomic_json(path, state)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    storage["file_fsync"] += 1
    storage["directory_fsync"] += 1
    storage["durable_bytes"] += path.stat().st_size


def release(request: Json, seal: Path, query: int, drain: bool = False) -> list[int]:
    """A single-arm broker preserves the existing delay and checks its durable seal."""
    if not drain and (query < 8 or query > 32 or query % 4):
        return []
    state = json.loads(seal.read_bytes())
    if (
        state["query"] != query
        or len(state["rows"]) != query
        or state["prediction_hash"] != canonical_hash(state["rows"])
    ):
        raise ValueError("prediction_seal")
    labels = json.loads(Path(request["label_path"]).read_bytes())["labels"]
    if len(labels) != 32 or any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("online_label_schema")
    return list(labels[28:32] if drain else labels[query - 8 : query - 4])


def update(state: Json, labels: list[int], query: int, arm: str) -> None:
    """Only released four-label blocks can propose the existing bounded scalar update."""
    first = state["released_through"]
    zs = [r["base_logit"] for r in state["rows"][first : first + 4]]
    candidate = adapter.step(state["params"], zs, labels, arm)
    guard = adapter.guard(candidate, state["guards"])
    before = state["params"]
    if guard["passed"]:
        state["params"] = candidate
    state["feedback_rows"].append(
        dict(
            released_after=query,
            ordinals=list(range(first + 1, first + 5)),
            labels=labels,
            before=before,
            candidate=candidate,
            after=state["params"],
            **guard,
        )
    )
    state["released_through"] += 4
    state["pending"] = list(range(state["released_through"] + 1, state["query"] + 1))
    state["counts"]["updates"] += 1
    state["counts"]["guard_evaluations"] += 128


def run(request_path: Path, stop: int) -> Json:
    """Cold startup and resume execute the same scorer; counters are read after work."""
    request = json.loads(request_path.read_bytes())
    arm = request["arm"]
    if (
        arm not in ARMS
        or stop not in (16, 32)
        or len(request["rows"]) != 32
        or len(set(request["unit_ids"])) != 32
    ):
        raise ValueError("request_roster")
    print(f"[exp8968-child] before_frozen_state_load arm={arm} stop={stop}", flush=True)
    service = json.loads(Path(request["service_path"]).read_bytes())
    path = request_path.parent / "state.json"
    online = arm in ("bias_only", "adaptive")
    storage = dict(file_fsync=0, directory_fsync=0, durable_bytes=0)
    if path.exists():
        state = json.loads(path.read_bytes())
        if state["query"] != 16 or state["request_hash"] != canonical_hash(request):
            raise ValueError("resume_state")
    else:
        state = dict(
            query=0,
            request_hash=canonical_hash(request),
            params=[1.0, 0.0],
            rows=[],
            feedback_rows=[],
            pending=[],
            released_through=0,
            guards=[],
            counts=dict(
                scored_rows=0,
                graph_pack_calls=0,
                graph_encoded_rows=0,
                message_rounds=0,
                updates=0,
                guard_evaluations=0,
                broker_calls=0,
                request_parses=0,
                sampling_call_count=0,
            ),
        )
        if online:
            guards = score(service, request["guard_rows"], arm, state["counts"])
            state["guards"] = [list(r) for r in zip(guards, request["guard_labels"], strict=True)]
    initial = deepcopy(state)
    print(f"[exp8968-child] after_frozen_state_load arm={arm}", flush=True)
    for query in range(state["query"] + 1, stop + 1):
        row = json.loads(request["rows"][query - 1])
        state["counts"]["request_parses"] += 1
        base = score(service, [row], arm, state["counts"])[0]
        z = adapter.logit(base)
        p = adapter.probability(state["params"], z) if online else base
        state["rows"].append(
            dict(
                unit_id=request["unit_ids"][query - 1],
                query=query,
                base_logit=z,
                p_error=p,
                decision="accept" if p <= 0.1 else "reject" if p >= 0.9 else "escalate",
            )
        )
        state.update(
            query=query,
            pending=list(range(state["released_through"] + 1, query + 1)) if online else [],
        )
        state["prediction_hash"] = canonical_hash(state["rows"])
        commit(path, state, storage)
        if online:
            state["counts"]["broker_calls"] += 1
            labels = release(request, path, query)
            if labels:
                update(state, labels, query, arm)
            commit(path, state, storage)
    if stop == 32 and online:
        state["counts"]["broker_calls"] += 1
        update(state, release(request, path, 32, drain=True), 40, arm)
        commit(path, state, storage)
    result = dict(
        pid=os.getpid(),
        initial_state=initial,
        state=state,
        storage=storage,
        peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    atomic_json(request_path.parent / f"segment-{stop}.json", result)
    print(f"[exp8968-child] completed={stop} pending={32 - stop} arm={arm}", flush=True)
    return result
