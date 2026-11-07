"""REQ-REPORT-8884: measure causal learning under the authenticated topology protocol.

Exact signed-record truth is an oracle. The experiment measures a small energy
head and durable memory, without training or loading a language model.
"""

from collections import Counter
from copy import deepcopy
from functools import partial
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.memory import categorical_memory_8830 as c
from carnot.memory import causal_memory_8802 as primitive
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting.causal_learning_audit_8810 import require
from carnot.reporting import topology_memory_methods_8881 as base

__all__ = ["base", "require"]
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.linear_causal_streams_8856 import semantic as prior_semantic

Json = dict[str, Any]
ROOT = base.ROOT
NAME = "experiment_8884_v760_topology_causal_streams"
TASK = "exp8884-topology-causal-streams"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/topology_causal_streams_8884.py",
    "python/carnot/reporting/topology_causal_runner_8884.py",
    CLI,
]
TESTS = ["tests/python/test_topology_causal_streams_8884.py"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report real finite work so supervision cannot mistake silence for progress."""
    print(f"[exp8884] phase={phase} completed={completed} pending={pending}", flush=True)


def semantic(value: Any) -> Any:
    """Independent executions must match decisions and states, while clocks may differ."""
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in prior_semantic(value).items() if k != "timing"}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


class Store(base.Store):
    """Keep the original atomic state format and retain every completed transaction."""

    def __init__(self, path: Path, params: Json, transactions: list[Json]) -> None:
        self.transactions = transactions
        super().__init__(path, params)

    def save(self) -> None:
        """Initial creation and exact reopen use the same fsync path as feedback."""
        self.commit(self.state)

    def commit(self, state: Json) -> None:
        """Measure real file/directory fsyncs and keep failed writes out of live state."""
        before = canonical_hash(self.state)
        fsync = os.fsync
        costs: list[float] = []

        def measured_fsync(fd: int) -> None:
            began = time.monotonic_ns()
            fsync(fd)
            costs.append((time.monotonic_ns() - began) / 1e9)

        began = time.monotonic_ns()
        with patch.object(os, "fsync", measured_fsync):
            super().commit(state)
        row = dict(
            arm=self.path.stem,
            sequence=len(self.transactions),
            before=before,
            after=canonical_hash(state),
            state=deepcopy(state),
            timing=dict(
                duration_s=(time.monotonic_ns() - began) / 1e9,
                fsync_calls=len(costs),
                fsync_duration_s=sum(costs),
            ),
        )
        atomic_json(self.path.parent / "transactions" / f"{row['sequence']:05d}.json", row)
        self.transactions.append(row)

    def predict(self, event: str, payload: Json, threshold: float) -> Json:
        """Seal all serving outputs before any verifier can reveal delayed truth."""
        if self.active is not None:
            raise RuntimeError("active query")
        require(event not in self.state["pending"] and event not in self.state["acks"], "duplicate")
        require(len(self.state["pending"]) < 32, "pending capacity")
        snapshot = deepcopy(self.state)
        observation = base.observe(payload, snapshot["templates"])
        prediction = c.probabilities(snapshot["parameters"], observation["counts"])
        key = base.topology(payload["query"])
        sealed = dict(
            observation,
            ordinal=int(event[1:]),
            evidence_hash=canonical_hash(payload),
            query_digest=canonical_hash(payload["query"]),
            prediction=prediction,
            confidence=max(prediction),
            typed_decision=primitive.LABELS[prediction.index(max(prediction))],
            action=primitive.action(prediction, threshold),
            state_hash=canonical_hash(snapshot),
            template_active=key in snapshot["templates"],
            template_uses=[key] if key in snapshot["templates"] else [],
        )
        state = deepcopy(snapshot)
        state["pending"][event] = sealed
        self.commit(state)
        self.active = snapshot
        return deepcopy(sealed)


def run_stream(raw: Path, plan: Json, heads: Json, protocol: Json) -> Json:
    """Instrument the original four-arm loop without changing its releases or features."""
    transactions: list[Json] = []
    labels: list[Json] = []
    counts: Counter[str] = Counter()
    original = symbolic.decide
    digests = {
        canonical_hash(
            base.world(
                protocol["topology_manifest"]["development" if plan["pilot"] else split][
                    r["template"]
                ],
                r["unit_id"] + "/",
                r["error_type"],
                r["world_seed"],
            )
        ): split
        for split in ("online", "retention")
        for r in plan[split]
    }

    def decide(value: Json) -> Json:
        result = dict(original(value))
        if len(value["query"]) == 1:
            counts["primitive"] += 1
        else:
            digest = canonical_hash(value)
            split = digests[digest]
            counts[split + "_whole_query"] += 1
            labels.append(
                dict(
                    split=split,
                    evidence_hash=digest,
                    truth=result["decision"],
                    after_transaction_count=len(transactions),
                )
            )
        return result

    progress("stream_before_" + plan["stream_id"], 0, 48)
    with (
        patch.object(base, "Store", partial(Store, transactions=transactions)),
        patch.object(symbolic, "decide", decide),
    ):
        result = base.run_stream(raw, plan, heads, protocol)
    result.update(
        transaction_rows=transactions, label_call_rows=labels, label_call_counts=dict(counts)
    )
    result.update(evidence(result))
    verify_stream(result)
    progress("stream_after_" + plan["stream_id"], 48, 0)
    return result


def evidence(result: Json) -> Json:
    """Reduce delay, movement and admissions from actual rows and durable transactions."""
    identity = dict(cluster_id=result["stream"]["stream_id"], seed=result["stream"]["seed"])
    online = [
        dict(
            r,
            prediction_seal=canonical_hash(r["sealed"]),
            template_uses=r["sealed"]["template_uses"],
        )
        for r in result["rows"]
        if r["split"] == "online"
    ]
    releases, admissions = [], []
    for transition in result["transitions"]:
        batch = next(b for b in result["batches"] if b["release_ordinal"] == transition["ordinal"])
        gradients = {r["event"]: r for r in transition["release_rows"]}
        for i in batch["events"]:
            row = next(r for r in online if r["arm"] == transition["arm"] and r["ordinal"] == i)
            releases.append(
                dict(
                    identity,
                    arm=row["arm"],
                    unit_id=row["unit_id"],
                    event=i,
                    prediction_seal=row["prediction_seal"],
                    release_ordinal=transition["ordinal"],
                    actual_delay=transition["ordinal"] - i,
                    configured_delay=result["stream"]["delays"][i - 1],
                    **{
                        k: v
                        for k, v in gradients.get(i, dict(optimizer_steps=0)).items()
                        if k not in {"event", "release_ordinal"}
                    },
                )
            )
    previous: dict[str, Json] = {}
    for tx in result["transaction_rows"]:
        prior = previous.get(tx["arm"], tx["state"])
        inserted = [t for t in tx["state"]["templates"] if t not in prior["templates"]]
        if inserted:
            admissions.append(
                dict(
                    identity,
                    arm=tx["arm"],
                    sequence=tx["sequence"],
                    before=tx["before"],
                    after=tx["after"],
                    inserted_templates=inserted,
                    admission="syntax_after_prediction"
                    if tx["arm"] == "frozen_query_only"
                    else "verified_release",
                )
            )
        previous[tx["arm"]] = tx["state"]
    return dict(
        prediction_rows=online,
        release_rows=releases,
        admission_rows=admissions,
        gradient_rows=[r for r in releases if r["optimizer_steps"]],
        retention_rows=[r for r in result["rows"] if r["split"] == "retention"],
        state_hashes=[
            dict(arm=t["arm"], sequence=t["sequence"], before=t["before"], after=t["after"])
            for t in result["transaction_rows"]
        ],
        delay_totals=[
            dict(
                identity,
                arm=a,
                configured_cumulative_delay=sum(
                    r["configured_delay"] for r in releases if r["arm"] == a
                ),
                actual_cumulative_delay=sum(r["actual_delay"] for r in releases if r["arm"] == a),
            )
            for a in base.ARMS
        ],
        maximum_pending=max(len(t["state"]["pending"]) for t in result["transaction_rows"]),
        tail_drain_rows=[
            dict(
                identity,
                arm=a,
                final_ordinal=max(b["release_ordinal"] for b in result["batches"]),
                pending_final=len(state["pending"]),
                acks_final=len(state["acks"]),
                tail_events=sum(r["release_ordinal"] > 48 for r in releases if r["arm"] == a),
                state_hash=canonical_hash(state),
            )
            for a, state in result["final_states"].items()
        ],
    )


def verify_stream(result: Json) -> None:
    """Reject self-consistent rehashed summaries when durable causal operands disagree."""
    expected = evidence(result)
    require(
        all(canonical_hash(result[k]) == canonical_hash(v) for k, v in expected.items()),
        "evidence_reduction",
    )
    require(
        len(result["rows"]) == 768 and all(r["passed"] for r in result["controls"]), "stream_census"
    )
    previous: dict[str, str] = {}
    for tx in result["transaction_rows"]:
        require(
            tx["after"] == canonical_hash(tx["state"])
            and tx["before"] == previous.get(tx["arm"], tx["before"]),
            "transaction_chain",
        )
        previous[tx["arm"]] = tx["after"]
    for row in result["prediction_rows"]:
        sealed = row["sealed"]
        p = sealed["prediction"]
        require(
            sealed["confidence"] == max(p)
            and sealed["typed_decision"] == primitive.LABELS[p.index(max(p))]
            and sealed["action"] == primitive.action(p, 0.5),
            "typed_seal",
        )
        value = result["worlds"]["online"][row["ordinal"] - 1]
        require(
            sealed["evidence_hash"] == canonical_hash(value)
            and sealed["query_digest"] == canonical_hash(value["query"]),
            "query_digest",
        )
        require(
            any(
                t["arm"] == row["arm"] and t["state"]["pending"].get(f"e{row['ordinal']}") == sealed
                for t in result["transaction_rows"]
            ),
            "durable_seal",
        )
    require(
        all(r["pending_final"] == 0 and r["acks_final"] == 48 for r in result["tail_drain_rows"]),
        "tail_drain",
    )
    require(
        result["label_call_counts"]["online_whole_query"] == 192
        and result["label_call_counts"]["retention_whole_query"] == 576,
        "label_census",
    )


def reduce_streams(results: list[Json]) -> Json:
    """Count paired streams once; preliminary effects never serve as benefit acceptance."""
    fields = [
        "prediction_rows",
        "release_rows",
        "admission_rows",
        "state_hashes",
        "gradient_rows",
        "retention_rows",
        "delay_totals",
        "tail_drain_rows",
    ]
    rows = [r for s in results for r in s["rows"]]
    return dict(
        rows=rows,
        **{k: [r for s in results for r in s[k]] for k in fields},
        maximum_pending=max((s["maximum_pending"] for s in results), default=0),
        label_call_counts={
            k: sum(s["label_call_counts"].get(k, 0) for s in results)
            for k in ("primitive", "online_whole_query", "retention_whole_query")
        },
        descriptive_effects={
            k: v for k, v in base.summarize(rows).items() if k != "true_label_benefit"
        }
        if results
        else {},
        transaction_costs=dict(
            transactions=sum(len(s["transaction_rows"]) for s in results),
            fsync_calls=sum(
                t["timing"]["fsync_calls"] for s in results for t in s["transaction_rows"]
            ),
            duration_s=sum(
                t["timing"]["duration_s"] for s in results for t in s["transaction_rows"]
            ),
            fsync_duration_s=sum(
                t["timing"]["fsync_duration_s"] for s in results for t in s["transaction_rows"]
            ),
        ),
    )
