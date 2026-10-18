"""REQ-VERIFY-8159: frozen scoring with a bounded durable request-batch API.

Rust computes the historical Gaussian features. The existing host file store
commits a complete ordered batch before returning any of its decisions. This
prototype has one writer; its speed applies to this batch acknowledgement API.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from typing import Any

import numpy as np

from carnot import experiment_8145_v704_natural_service_cost as old
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8159_v705_durable_batch_service"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_durable_batch_8159.py"
OWNED = [
    "python/carnot/verify/durable_batch_8159.py",
    "python/carnot/reporting/durable_batch_execution_8159.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
ARMS = ("python_scalar_serial", "python_batch_serial", "native_batch_serial", "native_atomic_batch")
CONFIG: Json = dict(
    seed=7058159,
    batches=[1, 8, 32],
    repetitions=30,
    conditions=["cold", "warm", "restart"],
    arrivals=["all_at_once", "cadence_4ms"],
    cadence_ns=4_000_000,
    maximum_wait_ns=32_000_000,
    bootstraps=10000,
)
atomic_json = old.atomic_json
canonical_hash = old.canonical_hash
sha256_file = old.sha256_file
reference = old.reference
checksum = old.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose actual work boundaries so slow storage cannot look like a dead task."""
    print(f"[exp8159] {phase} completed={completed} pending={pending}", flush=True)


def inputs(root: Path, raw: Path) -> Json:
    """Authenticate original head bytes, not a new learning outcome or mutable methods."""
    data = old.inputs(root, raw)
    path = root / "results/experiment_8145_v704_natural_service_cost.json"
    old.gate(data, path, "natural_cost_resource_exists", True, path.is_file())
    if path.is_file():
        try:
            value = json.loads(path.read_text())
            for field, expected in [
                ("natural_service_ready_score", 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                old.gate(data, path, field, expected, value.get(field))
            terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
            bound = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
            old.gate(data, path, "natural_cost_terminal_passed", True, bound["report"]["passed"])
            for ref in [
                reference(path),
                value["primitive_rows"],
                reference(
                    ROOT / "openspec/change-proposals/research-roadmap-v704-preserved-20261005.md"
                ),
            ]:
                old.gate(
                    data, Path(ref["path"]), "sha256", ref["sha256"], sha256_file(Path(ref["path"]))
                )
                data["refs"].append(ref)
        except (OSError, ValueError, KeyError) as exc:
            old.gate(data, path, "natural_cost_authentication", True, str(exc))
    data["ready"] = all(r["passed"] for r in data["checks"])
    atomic_json(raw / "input_data.json", data)
    return data


class Store:
    """Keep a single ordered journal so one rename commits every request together.

    The directory fsync makes the rename durable. A duplicate ID with unchanged
    input returns its original decision; changed content is an error, not an update.
    """

    def __init__(self, path: Path, head_hash: str):
        self.path = path
        self.state: Json = dict(head_hash=head_hash, records=[], causal_hash=head_hash)
        if path.exists():
            envelope = json.loads(path.read_text())
            if (
                envelope.get("sha256") != canonical_hash(envelope.get("state"))
                or envelope["state"]["head_hash"] != head_hash
            ):
                raise ValueError("store_custody")
            self.state = envelope["state"]
            previous = head_hash
            identities: set[str] = set()
            for index, record in enumerate(self.state["records"]):
                if (
                    record["sequence"] != index + 1
                    or record["previous_hash"] != previous
                    or record["request_id"] in identities
                ):
                    raise ValueError("store_custody")
                identities.add(record["request_id"])
                previous = canonical_hash(record)
            if self.state["causal_hash"] != previous:
                raise ValueError("store_custody")
        self.boundaries: Json = {}

    def commit(self, rows: list[Json], crash: str | None = None) -> list[Json]:
        """Install all new records only after file and directory writes reach storage."""
        start = time.monotonic_ns()
        if len({r["request_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate_batch_id")
        candidate = deepcopy(self.state)
        known = {r["request_id"]: r for r in candidate["records"]}
        results = []
        for row in rows:
            request_id = row["request_id"]
            if request_id in known:
                if known[request_id]["input_hash"] != row["input_hash"]:
                    raise ValueError("request_id_input_hash")
                results.append(known[request_id])
                continue
            record = dict(
                row, sequence=len(candidate["records"]) + 1, previous_hash=candidate["causal_hash"]
            )
            candidate["causal_hash"] = canonical_hash(record)
            candidate["records"].append(record)
            results.append(record)
        envelope = dict(state=candidate, sha256=canonical_hash(candidate))
        serialized = time.monotonic_ns()
        if crash == "before_commit":
            os._exit(71)
        commit_start = time.monotonic_ns()
        if candidate != self.state or not self.path.exists():
            atomic_json(self.path, envelope)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        commit_end = time.monotonic_ns()
        self.state = candidate
        self.boundaries = dict(
            serialization_start_ns=start,
            serialization_end_ns=serialized,
            commit_start_ns=commit_start,
            commit_end_ns=commit_end,
        )
        if crash == "after_commit":
            os._exit(72)
        return results


def transaction(
    data: Json,
    native: Any,
    slot: Json,
    arm: str,
    raw: Path,
    config: Json = CONFIG,
    crash: str | None = None,
) -> Json:
    """Measure one complete request group including real arrivals and durable responses."""
    raw.mkdir(parents=True, exist_ok=True)
    panel = [r for r in data["public"] if r["values"] is not None]
    selected = [
        panel[(slot["repetition"] * slot["batch"] + i) % len(panel)] for i in range(slot["batch"])
    ]
    head_hash = canonical_hash(data["head"])
    store_path = raw / "store.json"
    store = Store(store_path, head_hash)
    if slot["condition"] in {"warm", "restart"}:
        store.commit([])
    if slot["condition"] == "warm":
        old.score(
            data["head"],
            data["geometry"],
            [selected[0]["values"]],
            native,
            "native_batch" if arm.startswith("native") else "python_batch",
        )
    queue: Queue[Json] = Queue()

    def arrivals() -> None:
        previous = 0
        for index, row in enumerate(selected):
            if index and slot["arrival"] == "cadence_4ms":
                remaining = previous + config["cadence_ns"] - time.monotonic_ns()
                if remaining > 0:
                    time.sleep(remaining / 1e9)
            previous = time.monotonic_ns()
            queue.put(
                dict(
                    row,
                    request_id=f"{slot['unit_id']}-{index}",
                    input_hash=canonical_hash(row),
                    enqueue_ns=previous,
                )
            )

    began = time.monotonic_ns()
    producer = threading.Thread(target=arrivals)
    producer.start()
    if slot["condition"] in {"cold", "restart"}:
        store = Store(store_path, head_hash)
    lifecycle = time.monotonic_ns() - began
    costs: Json = dict(
        lifecycle_ns=lifecycle, arithmetic_ns=0, serialization_ns=0, commit_write_fsync_ns=0
    )
    requests: list[Json] = []
    commits = 0
    while len(requests) < len(selected):
        group = [queue.get()]
        limit = 1 if arm == ARMS[0] else slot["batch"]
        deadline = group[0]["enqueue_ns"] + config["maximum_wait_ns"]
        while len(group) < min(limit, len(selected) - len(requests)):
            remaining = deadline - time.monotonic_ns()
            if remaining <= 0:
                break
            try:
                group.append(queue.get(timeout=remaining / 1e9))
            except Empty:
                break
        execution_start = time.monotonic_ns()
        probabilities = old.score(
            data["head"],
            data["geometry"],
            [r["values"] for r in group],
            native,
            "native_batch" if arm.startswith("native") else "python_batch",
        )
        execution_end = time.monotonic_ns()
        costs["arithmetic_ns"] += execution_end - execution_start
        records = [
            dict(
                request_id=r["request_id"],
                input_hash=r["input_hash"],
                source_cluster_id=r["source_cluster_id"],
                source_unit_id=r["unit_id"],
                values=r["values"],
                probability=float(p),
                action=old.engine.historical.radial.action(float(p)),
                head_hash=head_hash,
            )
            for r, p in zip(group, probabilities, strict=True)
        ]
        chunks = [records] if arm == ARMS[-1] else [[r] for r in records]
        offset = 0
        for chunk in chunks:
            durable = store.commit(chunk, crash)
            commits += 1
            boundary = dict(store.boundaries)
            response = time.monotonic_ns()
            costs["serialization_ns"] += (
                boundary["serialization_end_ns"] - boundary["serialization_start_ns"]
            )
            costs["commit_write_fsync_ns"] += (
                boundary["commit_end_ns"] - boundary["commit_start_ns"]
            )
            for r, record in zip(group[offset : offset + len(chunk)], durable, strict=True):
                requests.append(
                    dict(
                        record,
                        enqueue_ns=r["enqueue_ns"],
                        execution_start_ns=execution_start,
                        execution_end_ns=execution_end,
                        response_ns=response,
                        latency_ns=response - r["enqueue_ns"],
                        queue_ns=execution_start - r["enqueue_ns"],
                        **boundary,
                    )
                )
            offset += len(chunk)
    producer.join()
    duration = time.monotonic_ns() - began
    costs["residual_and_queue_ns"] = duration - sum(costs.values())
    return dict(
        arm=arm,
        requests=requests,
        duration_ns=duration,
        components=costs,
        commit_count=commits,
        store_path=str(store_path),
        store_sha256=sha256_file(store_path),
        head_hash=head_hash,
        arrival=slot["arrival"],
        condition=slot["condition"],
    )


def parity(arms: list[Json]) -> bool:
    """Require identical inputs, provenance, order and typed decisions across the four arms."""
    keys = [
        "request_id",
        "input_hash",
        "source_cluster_id",
        "values",
        "action",
        "sequence",
        "head_hash",
    ]
    return len(arms) == 4 and all(
        [[r[k] for k in keys] for r in a["requests"]]
        == [[r[k] for k in keys] for r in arms[0]["requests"]]
        and np.allclose(
            [r["probability"] for r in a["requests"]],
            [r["probability"] for r in arms[0]["requests"]],
            atol=1e-10,
            rtol=0,
        )
        for a in arms[1:]
    )


def measure(data: Json, native: Any, raw: Path, config: Json = CONFIG) -> Json:
    """Complete the frozen paired schedule; repetitions retain their original source identities."""
    atomic_json(raw / "input_data.json", data)
    began = time.monotonic()
    plan = [
        dict(unit_id=f"b{b}-{c}-{a}-r{r}", batch=b, condition=c, arrival=a, repetition=r)
        for b in config["batches"]
        for c in config["conditions"]
        for a in config["arrivals"]
        for r in range(config["repetitions"])
    ]
    work: Json = dict(config=config, pairs=[], durability=[])
    for index, slot in enumerate(plan):
        order = list(ARMS) if slot["repetition"] % 2 == 0 else list(reversed(ARMS))
        arms = []
        for arm in order:
            progress("benchmark_before_" + slot["unit_id"] + "_" + arm, index, len(plan) - index)
            arms.append(
                transaction(data, native, slot, arm, raw / "stores" / slot["unit_id"] / arm, config)
            )
            progress(
                "benchmark_after_" + slot["unit_id"] + "_" + arm, index + 1, len(plan) - index - 1
            )
        work["pairs"].append(
            dict(slot, order=order, arms=arms, status="completed", exclusion_reason=None)
        )
    work["durability"] = durability(data, native, raw / "crashes")
    work["measurement_duration_s"] = time.monotonic() - began
    atomic_json(raw / "primitive_rows.json", work)
    return work


def durability(data: Json, native: Any, raw: Path) -> list[Json]:
    """Crash real child processes on both sides of commit, then retry stable IDs."""
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec

    results = []
    for point, expected in [("before_commit", 71), ("after_commit", 72)]:
        directory = raw / point
        directory.mkdir(parents=True, exist_ok=True)
        slot = dict(unit_id=point, batch=8, repetition=0, condition="cold", arrival="all_at_once")
        store = Store(directory / "store.json", canonical_hash(data["head"]))
        acknowledged = dict(
            request_id="already_acknowledged",
            input_hash="sealed_input",
            head_hash=canonical_hash(data["head"]),
            source_cluster_id="private_fixture",
        )
        original = store.commit([acknowledged])
        manifest = directory / "worker.json"
        atomic_json(manifest, dict(data=data, slot=slot, raw=str(directory)))
        command = CommandSpec(
            "crash_" + point,
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--worker",
                str(manifest),
                "--crash",
                point,
            ),
            "private_crash",
            60,
        )
        receipt = old.prior.execute([command], directory, expected)[0]
        reopened = Store(directory / "store.json", canonical_hash(data["head"]))
        present = len(reopened.state["records"]) - 1
        transaction(data, native, slot, ARMS[-1], directory)
        restored = Store(directory / "store.json", canonical_hash(data["head"]))
        records = restored.state["records"]
        count = len(records)
        transaction(data, native, slot, ARMS[-1], directory)
        final = Store(directory / "store.json", canonical_hash(data["head"]))
        previous = canonical_hash(data["head"])
        causal = True
        for index, record in enumerate(records):
            causal = (
                causal and record["sequence"] == index + 1 and record["previous_hash"] == previous
            )
            previous = canonical_hash(record)
        exact = (
            count == 9
            and len({r["request_id"] for r in records}) == count
            and final.state == restored.state
        )
        ack = records[0] == original[0]
        results.append(
            dict(
                fixture=True,
                verdict_class="circular_positive",
                crash_point=point,
                crash_exit=receipt["actual_exit"],
                validation_receipt=receipt,
                acknowledged_preserved=ack,
                exactly_once=exact,
                causal_order_preserved=causal,
                unacknowledged_present_before_retry=present,
                record_count=count,
                passed=receipt["passed"]
                and ack
                and exact
                and causal
                and present == (0 if expected == 71 else 8),
                store=reference(directory / "store.json"),
            )
        )
    return results


def reduce_rows(work: Json) -> Json:
    """Rebuild throughput, quantiles and paired batch intervals from primitive clocks."""
    config = work["config"]
    intended = (
        len(config["batches"])
        * len(config["conditions"])
        * len(config["arrivals"])
        * config["repetitions"]
    )
    rows, summaries, intervals = [], [], []
    passed = (
        len(work["pairs"]) == intended
        and len(work["durability"]) == 2
        and all(r["passed"] for r in work["durability"])
    )
    for pair in work["pairs"]:
        passed = passed and parity(pair["arms"])
        for arm in pair["arms"]:
            requests = arm["requests"]
            passed = passed and len(requests) == pair["batch"] and arm["duration_ns"] > 0
            for r in requests:
                clocks = [
                    r[k]
                    for k in [
                        "enqueue_ns",
                        "execution_start_ns",
                        "execution_end_ns",
                        "serialization_start_ns",
                        "serialization_end_ns",
                        "commit_start_ns",
                        "commit_end_ns",
                        "response_ns",
                    ]
                ]
                passed = (
                    passed
                    and clocks == sorted(clocks)
                    and r["latency_ns"] == clocks[-1] - clocks[0]
                )
            rows.append(
                dict(
                    unit_id=pair["unit_id"],
                    source_cluster_id=canonical_hash([r["source_cluster_id"] for r in requests]),
                    arm=arm["arm"],
                    condition=pair["condition"],
                    arrival=pair["arrival"],
                    batch=pair["batch"],
                    metric="complete_batch_throughput_requests_per_second",
                    numerator=len(requests),
                    denominator=arm["duration_ns"] / 1e9,
                    status=pair["status"],
                    exclusion_reason=pair["exclusion_reason"],
                )
            )
    rng = np.random.default_rng(config["seed"])
    for batch in config["batches"]:
        for condition in config["conditions"]:
            for arrival in config["arrivals"]:
                pairs = [
                    p
                    for p in work["pairs"]
                    if (p["batch"], p["condition"], p["arrival"]) == (batch, condition, arrival)
                ]
                for name in ARMS:
                    arms = [next(a for a in p["arms"] if a["arm"] == name) for p in pairs]
                    latencies = [r["latency_ns"] for a in arms for r in a["requests"]]
                    if not arms:
                        continue
                    summaries.append(
                        dict(
                            batch=batch,
                            condition=condition,
                            arrival=arrival,
                            arm=name,
                            complete_batch_throughput=sum(len(a["requests"]) for a in arms)
                            * 1e9
                            / sum(a["duration_ns"] for a in arms),
                            p50_request_latency_ns=float(np.quantile(latencies, 0.5)),
                            p95_request_latency_ns=float(np.quantile(latencies, 0.95)),
                        )
                    )
                for baseline in ARMS[:-1]:
                    ratios = [
                        next(a["duration_ns"] for a in p["arms"] if a["arm"] == baseline)
                        / next(a["duration_ns"] for a in p["arms"] if a["arm"] == ARMS[-1])
                        for p in pairs
                    ]
                    if not ratios:
                        continue
                    logs = np.log(ratios)
                    draws = np.exp(
                        np.mean(
                            logs[
                                rng.integers(0, len(logs), size=(config["bootstraps"], len(logs)))
                            ],
                            axis=1,
                        )
                    )
                    lower = float(np.quantile(draws, 0.05))
                    intervals.append(
                        dict(
                            batch=batch,
                            condition=condition,
                            arrival=arrival,
                            baseline=baseline,
                            treatment=ARMS[-1],
                            ratios=ratios,
                            draws=config["bootstraps"],
                            geometric_mean=float(np.exp(np.mean(logs))),
                            lower95=lower,
                            upper95=float(np.quantile(draws, 0.95)),
                            speed_supported=lower > 1,
                            nfr01_met=lower >= 10,
                        )
                    )
    return dict(
        passed=bool(passed),
        rows=rows,
        summaries=summaries,
        intervals=intervals,
        intended_count=intended,
        completed_count=len(work["pairs"]),
    )


def build(
    data: Json,
    work: Json,
    raw: Path,
    receipts: list[Json],
    date: str,
    duration: float,
    fixture: bool,
) -> Json:
    """Grant readiness only to complete, equivalent and durable measured batches."""
    reduction = (
        reduce_rows(work)
        if data["ready"]
        else dict(
            passed=True, rows=[], summaries=[], intervals=[], intended_count=0, completed_count=0
        )
    )
    checked = all(r.get("passed") and r.get("normal_exit") for r in receipts)
    ready = data["ready"] and reduction["passed"] and checked and not fixture
    verdict = "positive" if any(r["speed_supported"] for r in reduction["intervals"]) else "null"
    honest = "complete_" + verdict + "_durable_batch_service_measured"
    if not data["ready"]:
        verdict, honest = (
            "blocked",
            "complete_blocked_" + next(r["check"] for r in data["checks"] if not r["passed"]),
        )
    if fixture and data["ready"]:
        verdict, honest = "circular_positive", "complete_circular_positive_private_batch_fixture"
    if not checked or not reduction["passed"]:
        verdict, honest = "disqualified", "complete_disqualified_owned_validation"
    components, latencies = [], []
    for pair in work["pairs"]:
        for arm in pair["arms"]:
            components.extend(
                dict(unit_id=pair["unit_id"], arm=arm["arm"], component=k, duration_ns=v)
                for k, v in arm["components"].items()
            )
            latencies.extend(
                dict(
                    unit_id=pair["unit_id"],
                    arm=arm["arm"],
                    condition=pair["condition"],
                    arrival=pair["arrival"],
                    batch=pair["batch"],
                    **r,
                )
                for r in arm["requests"]
            )
    atomic_json(raw / "request_latency_rows.json", dict(rows=latencies))
    paths = [
        *OWNED,
        TEST,
        "python/carnot/experiment_8145_v704_natural_service_cost.py",
        "python/carnot/verify/learning_protocol_8138.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "crates/carnot-python/src/radial_8105.rs",
        "Cargo.lock",
    ]
    value: Json = dict(
        experiment_id=8159,
        task_id="exp8159-durable-batch-service",
        schema="carnot.durable_batch_service.v1",
        honest_verdict=honest,
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        claim_scope="Frozen cached natural seed101 head, bounded single-writer atomic-batch API; loaded Rust arithmetic and host durable commits; no independent single-request speed or new learning claim",
        exposure_scope="exposed_historical_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(checked and reduction["passed"]),
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        gate_check_summary=data["checks"],
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if data["ready"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        trained_head_specs=data.get("trained_head_specs", []),
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        call_ledger=[],
        historical_model_provenance=data.get("historical_model_provenance", []),
        rows=reduction["rows"],
        intended_count=reduction["intended_count"],
        eligible_count=reduction["intended_count"],
        independent_count=len({r["source_cluster_id"] for r in latencies}),
        completed_count=reduction["completed_count"],
        excluded_count=0,
        censored_count=0,
        failed_count=int(not reduction["passed"]),
        sample_size_budget=dict(
            work["config"],
            original_source_slots=len(data["public"]),
            missing_original_slots=sum(r["values"] is None for r in data["public"]),
            repeats_are_not_sources=True,
        ),
        original_source_mask=[
            dict(
                unit_id=r["unit_id"],
                source_cluster_id=r["source_cluster_id"],
                status=r["status"],
                exclusion_reason=r["exclusion_reason"],
            )
            for r in data["public"]
        ],
        run_date=date,
        duration_s=duration,
        random_seed=work["config"]["seed"],
        source_artifact_hashes=data["refs"],
        raw_shard_hashes=[
            reference(p) for p in raw.rglob("*.json") if p.name != "terminal_validation.json"
        ],
        code_config_hashes={p: sha256_file(ROOT / p) for p in paths},
        measurement_config=work["config"],
        phase_spans=[dict(phase="measurement", duration_s=work.get("measurement_duration_s", 0))],
        acceptance_gates=dict(
            readiness="completeness, parity1e-10, durable acknowledgement and retry deduplication",
            speed="10000 paired complete-batch log-ratio draws; lower95>1; NFR-01 lower95>=10",
        ),
        host_batch_ready_score=int(ready),
        natural_batch_rows=reduction["summaries"],
        component_cost_rows=components,
        request_latency_rows=reference(raw / "request_latency_rows.json"),
        durability_rows=work["durability"],
        paired_speed_intervals=reduction["intervals"],
        native_extension_hash=data["library"].get("sha256"),
        rejected_update_cost=None,
        fixture_mode=fixture,
        primitive_rows=reference(raw / "primitive_rows.json"),
        input_data=reference(raw / "input_data.json"),
        methodology="Real monotonic arrivals; maximum32ms batch wait; queue included in request latency; complete group throughput includes arrival, lifecycle, scoring, serialization and durable file/directory fsync. Frozen scoring only. Bootstrap batches are paired repetitions, not independent sources.",
        field_principles=dict(
            host_batch_ready_score="Completeness, parity and durability only; speed reported separately.",
            paired_speed_intervals="Descriptive exposed host performance with unchanged paired inputs.",
            rejected_update_cost="Historical lack of rejected admissions remains unknown.",
            model_invocation_counts="Current calls are zero; imported provenance is separate.",
            request_latency_rows="Every request includes actual enqueue wait and returns after commit.",
            verdict_class="Only unfinished owned work may be partial.",
        ),
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Read all custody again and independently score original vectors before trusting totals."""
    try:
        value = json.loads(path.read_text())
        if value["reproducibility_checksum"] != checksum(value):
            return False
        refs = [
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            value["primitive_rows"],
            value["input_data"],
            value["request_latency_rows"],
        ]
        if any(sha256_file(Path(r["path"])) != r["sha256"] for r in refs):
            return False
        if any(sha256_file(ROOT / p) != h for p, h in value["code_config_hashes"].items()):
            return False
        for receipt in value["validation_receipts"]:
            if (
                "log_path" in receipt
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        data = json.loads(Path(value["input_data"]["path"]).read_text())
        work = json.loads(Path(value["primitive_rows"]["path"]).read_text())
        if work["config"] != value["measurement_config"]:
            return False
        if not data["ready"]:
            return value["verdict_class"] == "blocked" and value["host_batch_ready_score"] == 0
        reduced = reduce_rows(work)
        if not reduced["passed"] or any(
            value[k] != reduced[v]
            for k, v in [
                ("rows", "rows"),
                ("natural_batch_rows", "summaries"),
                ("paired_speed_intervals", "intervals"),
            ]
        ):
            return False
        latencies = []
        panel = [r for r in data["public"] if r["values"] is not None]
        for pair in work["pairs"]:
            for arm in pair["arms"]:
                store = Store(Path(arm["store_path"]), canonical_hash(data["head"]))
                if sha256_file(Path(arm["store_path"])) != arm["store_sha256"]:
                    return False
                for index, (r, committed) in enumerate(
                    zip(arm["requests"], store.state["records"], strict=True)
                ):
                    source = panel[(pair["repetition"] * pair["batch"] + index) % len(panel)]
                    if (
                        r["input_hash"] != canonical_hash(source)
                        or r["values"] != source["values"]
                        or r["source_cluster_id"] != source["source_cluster_id"]
                        or r["request_id"] != f"{pair['unit_id']}-{index}"
                    ):
                        return False
                    probability = old.engine.scalar_probability(
                        data["head"], data["geometry"], r["values"]
                    )
                    if abs(probability - r["probability"]) > 1e-10 or r[
                        "action"
                    ] != old.engine.historical.radial.action(probability):
                        return False
                    if any(r[k] != v for k, v in committed.items()):
                        return False
                    latencies.append(
                        dict(
                            unit_id=pair["unit_id"],
                            arm=arm["arm"],
                            condition=pair["condition"],
                            arrival=pair["arrival"],
                            batch=pair["batch"],
                            **r,
                        )
                    )
        if latencies != json.loads(Path(value["request_latency_rows"]["path"]).read_text())["rows"]:
            return False
        # A separate array reduction checks the throughput and latency headlines.
        for summary in value["natural_batch_rows"]:
            arms = [
                a
                for p in work["pairs"]
                for a in p["arms"]
                if (p["batch"], p["condition"], p["arrival"], a["arm"])
                == (summary["batch"], summary["condition"], summary["arrival"], summary["arm"])
            ]
            rate = np.sum([len(a["requests"]) for a in arms]) / (
                np.sum([a["duration_ns"] for a in arms]) / 1e9
            )
            ordered = sorted(r["latency_ns"] for a in arms for r in a["requests"])
            for quantile, field in [
                (0.5, "p50_request_latency_ns"),
                (0.95, "p95_request_latency_ns"),
            ]:
                position = (len(ordered) - 1) * quantile
                lower, upper = int(np.floor(position)), int(np.ceil(position))
                independent = ordered[lower] + (ordered[upper] - ordered[lower]) * (
                    position - lower
                )
                if not np.isclose(independent, summary[field], atol=1e-6, rtol=1e-12):
                    return False
            if not np.isclose(rate, summary["complete_batch_throughput"], rtol=1e-12):
                return False
        expected_ready = int(not value["fixture_mode"] and value["required_checks_passed"])
        return (
            value["host_batch_ready_score"] == expected_ready
            and value["rejected_update_cost"] is None
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
