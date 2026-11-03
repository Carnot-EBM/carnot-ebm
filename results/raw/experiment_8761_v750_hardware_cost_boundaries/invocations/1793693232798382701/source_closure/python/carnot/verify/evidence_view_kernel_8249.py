"""REQ-VERIFY-8249: qualify byte-preserving views and delayed soft counters.

These counters adjust predictions on cached evidence; neither an address nor a
private fixture label proves a natural claim. The journal reuses qualified fsync
boundaries so a restart cannot observe feedback before its persisted prediction.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import evidence_intervention_8248 as methods
from carnot.verify import sentence_transport_8179 as transport
from carnot.verify.calibrated_memory_trajectory_8211 import append, journal

Json = dict[str, Any]
ARMS = ["global", "group", "shuffled"]
GROUPS = [f"{i:03b}" for i in range(8)]
bind_roles = methods.validate_roles


def construct_view(row: Json, removed: list[int]) -> Json:
    """Keep original offsets alongside view offsets so deletions remain auditable.

    Only a complete source sentence can be removed. All answer bytes, including
    negation and qualifiers, pass through unchanged; empty source is diagnostic.
    """
    source = bytes.fromhex(row["source_bytes"])
    segments = transport.partition(source)
    if any(
        type(i) is not int or not 0 <= i < len(segments) or not segments[i]["complete"]
        for i in removed
    ):
        raise ValueError("sentence_address")
    pieces, mapping, offset, index = [], [], 0, 0
    for segment in segments:
        start, end = segment["byte_start"], segment["byte_end"]
        deleted = segment["sentence_index"] in removed
        mapping.append(
            dict(
                original_index=segment["sentence_index"],
                original_byte_start=start,
                original_byte_end=end,
                view_index=None if deleted else index,
                view_byte_start=None if deleted else offset,
                view_byte_end=None if deleted else offset + end - start,
            )
        )
        if not deleted:
            pieces.append(source[start:end])
            offset += end - start
            index += 1
    return dict(
        source_bytes=b"".join(pieces).hex(),
        answer_bytes=row["answer_bytes"],
        original_source_bytes=row["source_bytes"],
        sentence_map=mapping,
        removed_indices=sorted(set(removed)),
    )


def capture_request(view: Json) -> Json:
    """Reuse the qualified full-input protocol without introducing acquisition machinery."""
    return transport.requests(view)


def accept_capture(view: Json, transcript: str, request_hash: str) -> Json:
    """Bind responses to exact requests, then check complete records, never semantic truth."""
    request = capture_request(view)
    if canonical_hash(request) != request_hash:
        raise ValueError("hash_drift")
    indices = [s["sentence_index"] for s in request["sentences"]]
    return transport.parse(transcript, indices, len(request["source_segments"]))


def protocol_views(row: Json, cached: list[Json]) -> Json:
    """Reuse the frozen public selector; unavailable length controls remain unavailable."""
    plan = methods.view(row, cached, lambda text: len(text.encode()))
    if plan["status"] == "completed":
        plan["byte_views"] = {
            name: construct_view(row, removed)
            for name, removed in [
                ("original", []),
                ("selected_evidence_deleted", [plan["selected_sentence_index"]]),
                ("nonselected_deletion", [plan["control_sentence_index"]]),
            ]
        }
    return plan


def group_key(features: Json | None) -> str | None:
    """Freeze eight public groups without consulting human labels or source identities."""
    if features is None:
        return None
    if features["relation"] not in "ECB" or any(
        not math.isfinite(features[k]) for k in ["selected_delta", "control_delta"]
    ):
        raise ValueError("features")
    return f"{int(features['relation'] == 'E')}{int(features['selected_delta'] > 0.1)}{int(abs(features['control_delta']) > 0.1)}"


def initial() -> Json:
    """Beta priors are stored explicitly; zero observations keep the static prediction."""
    return dict(
        global_counts=dict(n=0, unsupported=0),
        groups={g: dict(n=0, unsupported=0) for g in GROUPS},
        shuffled={g: dict(n=0, unsupported=0) for g in GROUPS},
        issued=[],
        released=[],
        events=[],
    )


def shuffled_group(slot: int) -> str:
    """Shuffle each public block of eight to preserve the control's exact group sizes.

    The assignment uses only the issue slot and frozen seed, so no released or
    future human label can change a control group or its parameter budget.
    """
    groups = list(GROUPS)
    random.Random(101 + slot // 8).shuffle(groups)
    return groups[slot % 8]


def mix(p: float, counts: Json) -> float:
    """Shrink small counter estimates so each admitted group is a soft energy term."""
    n = counts["n"]
    weight = n / (n + 16)
    return float((1 - weight) * p + weight * (counts["unsupported"] + 1) / (n + 2))


def probability(state: Json, p: float | None, group: str | None, arm: str) -> float | None:
    """Global-only ends before group mixing; missing features never fabricate probabilities."""
    if p is None:
        return None
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability")
    result = mix(p, state["global_counts"])
    groups = state["shuffled" if arm == "shuffled" else "groups"]
    return (
        mix(result, groups[group])
        if arm != "global" and group is not None and groups[group]["n"] >= 8
        else result
    )


def transition(state: Json, event: Json) -> None:
    """Recompute every prediction and due release so rehashed edits still fail replay."""
    if event["kind"] not in {"issue", "release"}:
        raise ValueError("event_kind")
    record = event["record"]
    if event["kind"] == "issue":
        if record["slot"] != len(state["issued"]) or record["role"] not in {"stream", "retention"}:
            raise ValueError("issue_order")
        group = group_key(record["features"]) if record["p_static"] is not None else None
        shuffled = shuffled_group(record["slot"]) if group is not None else None
        expected = {
            a: probability(state, record["p_static"], shuffled if a == "shuffled" else group, a)
            for a in ARMS
        }
        if (
            record["predictions"] != expected
            or record["group"] != group
            or record["shuffled_group"] != shuffled
        ):
            raise ValueError("ledger_drift")
        state["issued"].append(record)
    else:
        origin, now = record["origin"], record["now"]
        if not 0 <= origin < len(state["issued"]):
            raise ValueError("unissued")
        if now != origin + 8 or not now < len(state["issued"]):
            raise ValueError("future_feedback")
        issued = state["issued"][origin]
        if issued["role"] == "retention":
            raise ValueError("retention")
        if issued["source_cluster_id"] in state["released"]:
            raise ValueError("duplicate_source")
        if type(record["label"]) is not int or record["label"] not in [0, 1]:
            raise ValueError("label")
        for counts in [state["global_counts"]] + (
            []
            if issued["group"] is None
            else [state["groups"][issued["group"]], state["shuffled"][issued["shuffled_group"]]]
        ):
            counts["n"] += 1
            counts["unsupported"] += record["label"]
        state["released"].append(issued["source_cluster_id"])
    state["events"].append(event)


def load_state(path: Path) -> Json:
    """Rebuild exact state from the existing durable issue/release journal interface."""
    if path.is_file() and path.read_bytes() and not path.read_bytes().endswith(b"\n"):
        raise ValueError("partial_record")
    state = initial()
    for event in journal(path):
        transition(state, event)
    return state


def issue(path: Path, row: Json) -> Json:
    """Persist all arms' predictions together before any label at this clock is released."""
    state = load_state(path)
    group = group_key(row["features"]) if row["p_static"] is not None else None
    shuffled = shuffled_group(row["slot"]) if group is not None else None
    record = {
        key: row[key] for key in ["slot", "source_cluster_id", "p_static", "features", "role"]
    }
    record.update(
        group=group,
        shuffled_group=shuffled,
        predictions={
            a: probability(state, row["p_static"], shuffled if a == "shuffled" else group, a)
            for a in ARMS
        },
    )
    event = dict(kind="issue", record=record)
    transition(state, event)
    append(path, event)
    return record


def release(path: Path, origin: int, now: int, label: int) -> None:
    """Admit only due labels after a durably issued prediction, rejecting source reuse."""
    state = load_state(path)
    event = dict(kind="release", record=dict(origin=origin, now=now, label=label))
    transition(state, event)
    append(path, event)


def summary(events: list[Json], panel: list[Json]) -> Json:
    """Replay actual admission predictions and score a separate private oracle fixture.

    The cost is a unit charge for the wrong binary decision at threshold .5.
    Retention is evaluated without releasing its labels into the counter state.
    """
    state = initial()
    for event in events:
        transition(state, event)
    labels = {e["record"]["origin"]: e["record"]["label"] for e in events if e["kind"] == "release"}
    costs = {}
    for arm in ARMS:
        later = [r for r in state["issued"] if r["slot"] >= 192 and r["slot"] in labels]
        cost = sum(int(r["predictions"][arm] >= 0.5) != labels[r["slot"]] for r in later)
        retained = sum(
            int(
                probability(
                    state,
                    r["p_static"],
                    r["shuffled_group"] if arm == "shuffled" else r["group"],
                    arm,
                )
                >= 0.5
            )
            != r["label"]
            for r in panel
        )
        earlier = sum(int(r["p_static"] >= 0.5) != r["label"] for r in panel)
        costs[arm] = dict(
            later_cost=cost,
            later_denominator=len(later),
            retention_cost=retained,
            retention_denominator=len(panel),
            earlier_cost=earlier,
        )
    return dict(
        state=state,
        fixture_costs=costs,
        admission_ready=costs["group"]["later_cost"]
        < min(costs[a]["later_cost"] for a in ["global", "shuffled"])
        and costs["group"]["retention_cost"] <= costs["group"]["earlier_cost"],
    )


def reconstruct(evidence: Json) -> Json:
    """Derive maps and counter results again instead of trusting stored readiness headlines."""
    result = summary(evidence["events"], evidence["panel"])
    mutations = [
        dict(row=r["row"], removed=r["removed"], view=construct_view(r["row"], r["removed"]))
        for r in evidence["mutations"]
    ]
    return dict(
        result,
        events=evidence["events"],
        panel=evidence["panel"],
        mutations=mutations,
        view_ready=all(m["view"]["answer_bytes"] == m["row"]["answer_bytes"] for m in mutations),
        fixture_only=True,
    )


def qualify(private: Path) -> Json:
    """Use a private learnable stream; oracle gains qualify mechanics without natural credit."""
    private.mkdir(parents=True, exist_ok=True)
    path = private / "ledger.jsonl"
    rng = random.Random(101)
    panel = []
    for index, group in enumerate(GROUPS):
        panel.append(
            dict(
                group=group,
                shuffled_group=GROUPS[index],
                p_static=0 if group[0] == "1" else 1,
                label=0 if group[0] == "1" else 1,
            )
        )
    labels = []
    state = initial()
    for t in range(384):
        group = GROUPS[t % 8]
        labels.append(int(rng.random() < (0.05 if group[0] == "1" else 0.95)))
        row = dict(
            slot=t,
            source_cluster_id=f"private-{t}",
            p_static=0.5,
            features=dict(
                relation="E" if group[0] == "1" else "B",
                selected_delta=0.2 if group[1] == "1" else 0,
                control_delta=0.2 if group[2] == "1" else 0,
            ),
            role="stream",
        )
        # The same journal transition used live is exercised here, without repeated disk scans.
        group_id = group_key(row["features"])
        shuffled = shuffled_group(t)
        record = dict(
            row,
            group=group_id,
            shuffled_group=shuffled,
            predictions={
                a: probability(state, 0.5, shuffled if a == "shuffled" else group_id, a)
                for a in ARMS
            },
        )
        event = dict(kind="issue", record=record)
        transition(state, event)
        append(path, event)
        if t >= 8:
            event = dict(kind="release", record=dict(origin=t - 8, now=t, label=labels[t - 8]))
            transition(state, event)
            append(path, event)
        if t % 64 == 0:
            print(f"[exp8249] phase=private_stream completed={t + 1} pending={383 - t}", flush=True)
    source = "Café is open. Café is open. It is not closed."
    row = dict(
        source_bytes=source.encode().hex(),
        answer_bytes="Café is open, but only today.".encode().hex(),
    )
    evidence = dict(
        events=journal(path),
        panel=panel,
        mutations=[dict(row=row, removed=removed) for removed in [[], [0], [1], [0, 1, 2]]],
    )
    return reconstruct(evidence)
