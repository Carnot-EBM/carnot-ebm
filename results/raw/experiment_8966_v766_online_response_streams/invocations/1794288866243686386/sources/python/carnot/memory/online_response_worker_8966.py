"""REQ-REPORT-8966: persist predictions before the broker opens delayed truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from carnot.memory import online_calibration_8960 as adapter
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]


def retention(state: Json, request: Json, query: int) -> None:
    """The adapter receives frozen probabilities, never retention truth or a label path."""
    for arm in adapter.ARMS:
        for unit, base in zip(
            request["retention_ids"], request["retention_probabilities"], strict=True
        ):
            p = adapter.probability(state["params"][arm], adapter.logit(base))
            state["retention_rows"].append(dict(unit_id=unit, arm=arm, query=query, p_error=p))


def trajectory(request: Json, raw: Path, *, resume: bool = False, stop: int = 32) -> Json:
    """Every saved commit contains the complete queue, parameters and prediction history."""
    if (
        len(request["unit_ids"]) != 32
        or len(set(request["unit_ids"])) != 32
        or len(request["probabilities"]) != 32
        or len(request["retention_ids"]) != 128
        or len(set(request["retention_ids"])) != 128
        or stop not in (16, 32)
    ):
        raise ValueError("stream_roster")
    zs = [adapter.logit(p) for p in request["probabilities"]]
    journal = raw / "state.json"
    if resume:
        state = json.loads(journal.read_text())
        if state["query"] != 16 or state["request_hash"] != canonical_hash(request):
            raise ValueError("resume_state")
    else:
        state = dict(
            query=0,
            request_hash=canonical_hash(request),
            params={a: [1.0, 0.0] for a in adapter.ARMS},
            rows=[],
            retention_rows=[],
            feedback_rows=[],
            promotion_rows=[],
            pending=[],
            pending_ids=[],
            released_through=0,
            seals=[],
            adapter_state_hashes=[],
        )
        retention(state, request, 0)
    broker = adapter.LabelBroker(Path(request["label_path"]))
    for query in range(state["query"] + 1, stop + 1):
        for arm in adapter.ARMS:
            state["rows"].append(
                dict(
                    unit_id=request["unit_ids"][query - 1],
                    arm=arm,
                    query=query,
                    p_error=adapter.probability(state["params"][arm], zs[query - 1]),
                )
            )
        state.update(query=query, pending=list(range(state["released_through"] + 1, query + 1)))
        state["pending_ids"] = [request["unit_ids"][i - 1] for i in state["pending"]]
        state["prediction_hash"] = canonical_hash(state["rows"])
        seal = raw / "seals" / f"{query:02d}.json"
        atomic_json(seal, state)
        released = broker.release(query, seal)
        state["seals"].append(dict(query=query, prediction_hash=state["prediction_hash"]))
        if released:
            ordinals = list(range(query - 7, query - 3))
            ids = [request["unit_ids"][i - 1] for i in ordinals]
            permuted = [released[i] for i in adapter.PERMUTATION]
            state["feedback_rows"].append(
                dict(
                    query=query,
                    ordinals=ordinals,
                    released_ids=ids,
                    labels=released,
                    permuted_labels=permuted,
                    prediction_hash=state["prediction_hash"],
                )
            )
            for arm in adapter.ARMS[1:]:
                before = list(state["params"][arm])
                labels = permuted if arm == "permuted" else released
                logits = zs[query - 8 : query - 4]
                candidate = adapter.step(before, logits, labels, arm)
                checked = adapter.guard(candidate, request["guard_rows"])
                if checked["passed"]:
                    state["params"][arm] = candidate
                state["promotion_rows"].append(
                    dict(
                        query=query,
                        arm=arm,
                        before=before,
                        candidate=candidate,
                        after=list(state["params"][arm]),
                        logits=logits,
                        labels=labels,
                        released_ids=ids,
                        accepted=checked["passed"],
                        **checked,
                    )
                )
            state.update(released_through=query - 4, pending=list(range(query - 3, query + 1)))
            state["pending_ids"] = [request["unit_ids"][i - 1] for i in state["pending"]]
        if query in (16, 32):
            retention(state, request, query)
        state["adapter_state_hashes"].append(
            dict(
                query=query,
                sha256=canonical_hash(
                    {k: v for k, v in state.items() if k != "adapter_state_hashes"}
                ),
            )
        )
        atomic_json(journal, state)
    return dict(state)


def main(argv: list[str] | None = None) -> int:
    """A parent terminates only the child that it owns after the durable midpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--stop16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print("[exp8966-child] before_trajectory", flush=True)
    state = trajectory(
        json.loads(args.request.read_text()),
        args.raw,
        resume=args.resume,
        stop=16 if args.stop16 else 32,
    )
    print(f"[exp8966-child] durable_query={state['query']}", flush=True)
    if args.stop16:
        print("READY16", flush=True)
        sys.stdin.readline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
