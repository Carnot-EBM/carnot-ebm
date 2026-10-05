"""REQ-VERIFY-8174: each arm owns a new acquisition and durable response.

The original source and deployed head stay fixed. Separate generations can
change decisions, so timing equivalence must be measured instead of assumed.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
import tempfile
from typing import Any

import numpy as np

from carnot.verify import shared_acquisition_8160 as shared
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES

Json = dict[str, Any]
ROOT = shared.ROOT
NAME = "experiment_8174_v706_complete_request_cost"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/verify/complete_request_8174.py"
RUNNER = "python/carnot/reporting/complete_request_execution_8174.py"
TEST = "tests/python/test_complete_request_8174.py"
OWNED = [MODULE, RUNNER, CLI]
UPSTREAM = "results/experiment_8173_v706_service_validation.json"
PIN = "sha256:6a86cbe73289436cde2bb13446688208f27f26d7fbe8b6c746c3642efed45430"
ARMS = ("python_durable_batch", "native_atomic_batch")
MODEL_SPECS = ["unsloth/Qwen3.8-27B-GGUF"]
CONFIG = dict(
    seed=7068174,
    sources=24,
    batch_size=8,
    main_calls=48,
    warmups=2,
    max_tokens=128,
    load_timeout_s=300,
    call_timeout_s=120,
    latest_launch_s=3000,
    closure_s=4800,
    bootstrap_draws=10000,
    startup_amortization_requests=48,
)
atomic_json, reference, sha256_file = shared.atomic_json, shared.reference, shared.sha256_file
checksum, canonical_hash = shared.checksum, shared.canonical_hash
progress = shared.progress


def inputs(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Reopen hash-bound deployment inputs; scientific hypotheses are not gates."""
    data: Json = dict(ready=False, checks=[], refs=[], slots=[])
    path = root / UPSTREAM
    gate = shared.host.old.gate
    gate(data, path, "resource_exists", True, path.is_file())
    if not path.is_file():
        atomic_json(raw / "input_data.json", data)
        return data
    try:
        upstream = json.loads(path.read_text())
        gate(
            data,
            path,
            "service_protocol_ready_score",
            1,
            upstream.get("service_protocol_ready_score"),
        )
        gate(data, path, "sha256", PIN, sha256_file(path))
        snapshot = next(
            r for r in upstream["raw_shard_hashes"] if Path(r["path"]).name == "input_data.json"
        )
        frozen = Path(snapshot["path"])
        gate(data, frozen, "sha256", snapshot["sha256"], sha256_file(frozen))
        original = json.loads(frozen.read_text())["sources"][0]["input"]
        data.update(
            {
                k: deepcopy(original[k])
                for k in ["head", "geometry", "library", "protocol", "trained_head_specs"]
            }
        )
        data["refs"] = [reference(path), snapshot, *original["refs"]]
        for ref in original["refs"]:
            operand = Path(ref["path"])
            gate(
                data,
                operand,
                "input_sha256",
                ref["sha256"],
                sha256_file(operand) if operand.is_file() else None,
            )
        if not all(c["passed"] for c in data["checks"]):
            atomic_json(raw / "input_data.json", data)
            return data
        identity, counts = (
            (dict(tokenizer="private_fixture"), [1] * len(original["slots"]))
            if fixture
            else shared.tokenizer_counts(original)
        )
        selected: list[Json] = []
        seen: set[str] = set()
        for slot, count in zip(original["slots"], counts, strict=True):
            source = slot["source_cluster_id"]
            if (
                slot["public_eligible"]
                and count <= 6000
                and source not in seen
                and len(selected) < 24
            ):
                seen.add(source)
                selected.append(deepcopy(slot))
        gate(data, frozen, "eligible_source_count", 24, len(selected))
        data.update(
            slots=selected,
            runtime_freeze=identity,
            upstream_scope=dict(
                host_8159="host-only acquisition-excluded durable batch costs",
                preserved_host_cost_rows=upstream["imported_cost_rows"][1],
                composition_8173=upstream["composition_replay_ready_score"],
                preserved_composition_cost_rows=upstream["composed_cost_rows"],
                composition_missing="No measured acquisition was recovered by validation repair",
            ),
        )
        data["ready"] = all(c["passed"] for c in data["checks"])
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        gate(data, path, "authenticated_external_inputs", True, f"{type(error).__name__}:{error}")
    atomic_json(raw / "input_data.json", data)
    return data


def acquire(slot: Json, arm: str, runtime: Any, raw: Path, ledger: Any, call_id: str) -> Json:
    """Save prompts and outputs separately for every actual full request attempt."""
    raw.mkdir(parents=True, exist_ok=True)
    arrival = time.monotonic_ns()
    row: Json = dict(
        unit_id=call_id,
        source_unit_id=slot["unit_id"],
        source_cluster_id=slot["source_cluster_id"],
        arm=arm,
        condition="independent_full_request",
        metric="complete_latency_ns",
        numerator=None,
        denominator=1,
        status="failed",
        exclusion_reason=None,
        arrival_ns=arrival,
        queue_start_ns=arrival,
        request=deepcopy(slot["request"]),
        visible_ids=slot["visible_ids"],
        family_id=slot["family_id"],
    )
    atomic_json(raw / "prompt.json", row["request"])
    row["generation_start_ns"] = time.monotonic_ns()
    ledger.start("generation", call_id, row["request"])
    progress("8174_generation_before_" + call_id, 0, 1)
    try:
        response = runtime.generate(row["request"])
        row["generation_end_ns"] = time.monotonic_ns()
        ledger.finish(call_id, "completed", response)
        row.update(
            raw_response=response,
            input_tokens=response["usage"]["prompt_tokens"],
            output_tokens=response["usage"]["completion_tokens"],
            server_prefill_generation_timings=response.get("timings", {}),
            prefill_start_monotonic_ns=None,
            prefill_end_monotonic_ns=None,
            prefill_clock_observation="Native server reports durations; no absolute prefill clock is exposed",
        )
        atomic_json(raw / "response.json", response)
        row["generated_text"] = response["choices"][0]["message"]["content"]
        row["parsing_start_ns"] = time.monotonic_ns()
        parsed = shared.prior.stream.risk.transport.parse_response(response, slot["visible_ids"])
        row["parsing_end_ns"] = time.monotonic_ns()
        row["feature_start_ns"] = time.monotonic_ns()
        body = json.loads(row["request"]["messages"][1]["content"])
        lexical = shared.prior.stream.lexical.extract(
            dict(
                family_id=slot["family_id"],
                source_bytes=body["complete_source"].encode().hex(),
                answer_bytes=body["original_answer"].encode().hex(),
            )
        )
        if (
            not parsed["completed"]
            or lexical["values"] is None
            or row["input_tokens"] > 6000
            or row["output_tokens"] > 128
        ):
            raise ValueError("unusable_parse_or_token_budget")
        p = min(1 - 1e-6, max(1e-6, parsed["probability"]))
        row.update(
            values=[math.log(p / (1 - p)), *lexical["values"]],
            parsed=parsed,
            feature_end_ns=time.monotonic_ns(),
            status="acquired",
        )
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        row["exclusion_reason"] = f"{type(error).__name__}:{error}"
        if ledger.rows[-1]["status"] == "running":
            ledger.finish(call_id, "failed", {})
    row["acquisition_end_ns"] = time.monotonic_ns()
    row["evidence"] = [reference(p) for p in raw.glob("*.json")]
    progress("8174_generation_after_" + call_id, 1, 0)
    return row


def commit_group(data: Json, group: list[Json], native: Any, raw: Path) -> None:
    """Score a real batch; Python commits serially and native commits atomically.

    The response clock is taken after fsync. Acquisition is never shared or
    added again to an inclusive latency; waiting for the batch stays in queue.
    """
    arm = group[0]["arm"]
    progress("8174_benchmark_before_" + arm, 0, len(group))
    head_hash = canonical_hash(data["head"])
    raw.mkdir(parents=True, exist_ok=True)
    start = time.monotonic_ns()
    store = shared.host.Store(raw / "store.json", head_hash)
    crossing = time.monotonic_ns()
    probabilities = shared.host.old.score(
        data["head"],
        data["geometry"],
        [r["values"] for r in group],
        native,
        "native_batch" if arm == ARMS[1] else "python_batch",
    )
    scored = time.monotonic_ns()
    records = [
        dict(
            request_id=r["unit_id"],
            input_hash=canonical_hash(r["request"]),
            source_cluster_id=r["source_cluster_id"],
            values=r["values"],
            probability=float(p),
            action=shared.host.old.engine.historical.radial.action(float(p)),
            head_hash=head_hash,
        )
        for r, p in zip(group, probabilities, strict=True)
    ]
    chunks = [records] if arm == ARMS[1] else [[r] for r in records]
    offset = 0
    for chunk in chunks:
        committed = store.commit(chunk)
        response = time.monotonic_ns()
        for row, record in zip(group[offset : offset + len(chunk)], committed, strict=True):
            row.update(
                probability=record["probability"],
                action=record["action"],
                status="completed",
                batch_start_ns=start,
                boundary_crossing_start_ns=crossing,
                arithmetic_start_ns=crossing,
                arithmetic_end_ns=scored,
                boundary_crossing_end_ns=scored,
                response_ns=response,
                latency_ns=response - row["arrival_ns"],
                numerator=response - row["arrival_ns"],
                **store.boundaries,
            )
        offset += len(chunk)
    for row in group:
        row["store"] = reference(store.path)
    progress("8174_benchmark_after_" + arm, len(group), 0)


def measure(
    data: Json,
    runtime: Any,
    native: Any,
    raw: Path,
    ledger: Any,
    *,
    started: float | None = None,
    **kwargs: Any,
) -> Json:
    """Counterbalance independent calls by source hash before any output is read."""
    began = time.monotonic() if started is None else started
    work: Json = dict(requests=[], warmups=[], config=CONFIG)
    slots = data["slots"]
    for i in range(2 if slots else 0):
        if time.monotonic() - began < CONFIG["latest_launch_s"]:
            work["warmups"].append(
                acquire(
                    slots[i % len(slots)],
                    "warmup",
                    runtime,
                    raw / f"warmup-{i}",
                    ledger,
                    f"warmup-{i}",
                )
            )
    for batch in range(0, len(slots), 8):
        selected = slots[batch : batch + 8]
        queues: Json = {a: [] for a in ARMS}
        for repeat in range(2):
            for slot in selected:
                order = (
                    ARMS
                    if int(slot["source_cluster_id"].split(":")[-1], 16) % 2 == 0
                    else ARMS[::-1]
                )
                arm = order[repeat]
                call_id = f"source-{slot['slot']}-{arm}"
                if time.monotonic() - began >= CONFIG["latest_launch_s"]:
                    row = dict(
                        unit_id=call_id,
                        source_cluster_id=slot["source_cluster_id"],
                        arm=arm,
                        condition="independent_full_request",
                        metric="complete_latency_ns",
                        numerator=None,
                        denominator=1,
                        status="censored",
                        exclusion_reason="launch_cutoff",
                    )
                else:
                    row = acquire(slot, arm, runtime, raw / call_id, ledger, call_id)
                    if row["status"] == "acquired":
                        queues[arm].append(row)
                work["requests"].append(row)
                if len(queues[arm]) == len(selected):
                    commit_group(data, queues[arm], native, raw / f"batch-{batch}" / arm)
                    queues[arm] = []
                progress(
                    "8174_completed_request_calls",
                    len(work["requests"]),
                    2 * len(slots) - len(work["requests"]),
                )
        for arm, group in queues.items():
            if group:
                commit_group(data, group, native, raw / f"batch-{batch}" / arm)
        atomic_json(raw / "primitive_rows.json", work)
    return work


def reduce(work: Json) -> Json:
    """Join by original source and bootstrap sources, retaining output confounders."""
    rows = work.get("requests", [])
    by_source: Json = {}
    for row in rows:
        by_source.setdefault(row["source_cluster_id"], {})[row["arm"]] = row
    pairs, differences, components = [], [], []
    for source, arms in by_source.items():
        if set(arms) != set(ARMS) or any(r["status"] != "completed" for r in arms.values()):
            continue
        a, b = [arms[arm] for arm in ARMS]
        pairs.append((a["response_ns"] - a["arrival_ns"], b["response_ns"] - b["arrival_ns"]))
        differences.append(
            dict(
                unit_id=source,
                source_cluster_id=source,
                generated_output_changed=a["generated_text"] != b["generated_text"],
                typed_decision_changed=a["action"] != b["action"],
                probability_changed=abs(a["probability"] - b["probability"]) > 1e-10,
            )
        )
        for row in (a, b):
            components.append(
                dict(
                    unit_id=row["unit_id"],
                    source_cluster_id=source,
                    arm=row["arm"],
                    acquisition_ns=row["acquisition_end_ns"] - row["arrival_ns"],
                    queue_after_acquisition_ns=row["batch_start_ns"] - row["acquisition_end_ns"],
                    arithmetic_inclusive_boundary_ns=row["arithmetic_end_ns"]
                    - row["arithmetic_start_ns"],
                    persistence_ns=row["commit_end_ns"] - row["commit_start_ns"],
                    complete_latency_ns=row["response_ns"] - row["arrival_ns"],
                    overlapping_timers_added=False,
                )
            )
    equivalent = bool(pairs) and not any(
        any(
            r[k]
            for k in ["generated_output_changed", "typed_decision_changed", "probability_changed"]
        )
        for r in differences
    )
    intervals: list[Json] = []
    if pairs:
        ratios = np.log(np.array(pairs)[:, 0] / np.array(pairs)[:, 1])
        draws = (
            np.random.default_rng(CONFIG["seed"]).choice(ratios, (10000, len(ratios))).mean(axis=1)
        )
        intervals = [
            dict(
                estimate=float(np.exp(ratios.mean())),
                lower95=float(np.exp(np.quantile(draws, 0.05))),
                upper95=float(np.exp(np.quantile(draws, 0.95))),
                sources=len(pairs),
                resamples=10000,
                descriptive_only=not equivalent,
            )
        ]
    ready = len(pairs) == 24 and len(rows) == 48
    return dict(
        rows=[
            dict(
                **{
                    k: r[k]
                    for k in [
                        "unit_id",
                        "source_cluster_id",
                        "arm",
                        "condition",
                        "metric",
                        "numerator",
                        "denominator",
                        "status",
                        "exclusion_reason",
                    ]
                },
                complete_latency_score_ns=r["numerator"],
            )
            for r in rows
        ],
        per_source_request_rows=rows,
        request_latency_rows=[
            dict(
                unit_id=r["unit_id"],
                source_cluster_id=r["source_cluster_id"],
                arm=r["arm"],
                latency_ns=r["response_ns"] - r["arrival_ns"],
            )
            for r in rows
            if r["status"] == "completed"
        ],
        component_cost_rows=components,
        output_difference_rows=differences,
        paired_speed_intervals=intervals,
        request_distributions={
            a: dict(
                p50_ns=float(np.quantile([p[i] for p in pairs], 0.5)),
                p95_ns=float(np.quantile([p[i] for p in pairs], 0.95)),
            )
            for i, a in enumerate(ARMS)
        }
        if pairs
        else {},
        intended_count=48,
        eligible_count=len(rows),
        independent_count=len(pairs),
        completed_count=sum(r["status"] == "completed" for r in rows),
        excluded_count=48 - len(rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        complete_service_ready_score=int(ready),
        equivalent_behavior_score=int(ready and equivalent),
        nfr01_met=bool(ready and equivalent and intervals[0]["lower95"] >= 10),
    )


def validate_work(data: Json, work: Json) -> bool:
    """Reparse independent outputs and recompute decisions before trusting clocks."""
    seen: set[tuple[str, str]] = set()
    for row in work.get("requests", []):
        key = row["source_cluster_id"], row["arm"]
        if key in seen or row["arm"] not in ARMS:
            return False
        seen.add(key)
        if row["status"] != "completed":
            continue
        clocks = [
            row[k]
            for k in [
                "arrival_ns",
                "generation_start_ns",
                "generation_end_ns",
                "parsing_start_ns",
                "parsing_end_ns",
                "feature_start_ns",
                "feature_end_ns",
                "acquisition_end_ns",
                "batch_start_ns",
                "arithmetic_start_ns",
                "arithmetic_end_ns",
                "serialization_start_ns",
                "serialization_end_ns",
                "commit_start_ns",
                "commit_end_ns",
                "response_ns",
            ]
        ]
        if (
            clocks != sorted(clocks)
            or row["latency_ns"] != clocks[-1] - clocks[0]
            or row["numerator"] != row["latency_ns"]
        ):
            return False
        if any(
            sha256_file(Path(r["path"])) != r["sha256"] for r in row["evidence"] + [row["store"]]
        ):
            return False
        slot = next(s for s in data["slots"] if s["source_cluster_id"] == row["source_cluster_id"])
        evidence = {Path(r["path"]).name: Path(r["path"]) for r in row["evidence"]}
        if (
            row["request"] != slot["request"]
            or row["visible_ids"] != slot["visible_ids"]
            or json.loads(evidence["prompt.json"].read_text()) != row["request"]
            or json.loads(evidence["response.json"].read_text()) != row["raw_response"]
        ):
            return False
        text = row["raw_response"]["choices"][0]["message"]["content"]
        parsed = shared.prior.stream.risk.transport.parse_response(
            row["raw_response"], row["visible_ids"]
        )
        body = json.loads(row["request"]["messages"][1]["content"])
        lexical = shared.prior.stream.lexical.extract(
            dict(
                family_id=row["family_id"],
                source_bytes=body["complete_source"].encode().hex(),
                answer_bytes=body["original_answer"].encode().hex(),
            )
        )
        p = min(1 - 1e-6, max(1e-6, parsed["probability"]))
        values = [math.log(p / (1 - p)), *lexical["values"]]
        probability = float(
            shared.host.old.score(data["head"], data["geometry"], [values], None, "python_batch")[0]
        )
        store = shared.host.Store(Path(row["store"]["path"]), canonical_hash(data["head"]))
        durable = next(r for r in store.state["records"] if r["request_id"] == row["unit_id"])
        if (
            text != row["generated_text"]
            or values != row["values"]
            or abs(probability - row["probability"]) > 1e-10
            or row["action"] != shared.host.old.engine.historical.radial.action(probability)
            or durable["action"] != row["action"]
            or durable["input_hash"] != canonical_hash(row["request"])
        ):
            return False
    return True


def live(data: Json, raw: Path, scratch: Path) -> Json:
    """Reuse the qualified worker with this task's lease identity and call schedule."""
    from unittest.mock import patch

    if os.environ.get("CARNOT_FORCE_LIVE") != "1":
        shared.host.old.gate(
            data, raw / "environment", "CARNOT_FORCE_LIVE", "1", os.environ.get("CARNOT_FORCE_LIVE")
        )
        return dict(work={}, ledger=[], checks=[])
    legacy = shared.prior.qualified.legacy
    original_capture, original_load = legacy.live_capture, legacy.QwenRuntime.load

    def owned_capture(*args: Any, **kwargs: Any) -> Json:
        """The shared loader may set an older task name; the lease must name this run."""
        with patch.object(legacy, "TASK", "exp8174-complete-request-cost"):
            return dict(original_capture(*args, **kwargs))

    def deadline_load(runtime: Any) -> Json:
        """Bound hash and worker startup together, keeping the original300s budget."""
        return dict(legacy.bounded(lambda: original_load(runtime), 300))

    def pulse(phase: str, started: float, units: int = 0) -> None:
        """Report persisted completed and pending calls during the owned child wait."""
        counts = shared.Ledger(raw / "ledger.json").counts()
        progress(
            "8174_" + phase,
            counts["generation_calls_completed"] + counts["model_loads_completed"],
            counts["generation_calls_in_flight"] + counts["model_loads_in_flight"],
        )

    with (
        patch.object(shared.prior, "capture", measure),
        patch.object(legacy, "live_capture", owned_capture),
        patch.object(legacy.QwenRuntime, "load", deadline_load),
        patch.object(legacy, "progress", pulse),
    ):
        return dict(shared.prior.live(data, raw, scratch))


def build(
    data: Json,
    result: Json,
    raw: Path,
    receipts: list[Json],
    date: str,
    duration: float,
    fixture: bool,
) -> Json:
    """Distinguish missing external inputs, owned failure and measured timing limits."""
    work = result.get("work") or dict(requests=[], warmups=[], config=CONFIG)
    atomic_json(raw / "primitive_rows.json", work)
    atomic_json(raw / "input_data.json", data)
    reduction = reduce(work)
    ledger = shared.Ledger(raw / "build-ledger.json")
    ledger.rows = result.get("ledger", [])
    counts = dict(shared.prior.ZERO_INVOCATION_COUNTS) if fixture else ledger.counts()
    checked = (
        bool(receipts)
        and (fixture or set(REQUIRED_CHECK_NAMES) <= {r.get("name") for r in receipts})
        and all(
            r.get("passed") and r.get("normal_exit") and r.get("actual_exit", 0) == 0
            for r in receipts
        )
    )
    checks = data["checks"] + result.get("checks", [])
    blocked = next((r["check"] for r in checks if not r["passed"]), None)
    ready = bool(
        reduction["complete_service_ready_score"]
        and checked
        and not blocked
        and not fixture
        and counts["generation_calls_completed"] >= 48
        and counts["model_loads_completed"] == 1
        and result.get("gpu_lease_receipt")
        and duration >= 10
    )
    verdict = "circular_positive" if fixture else "null"
    if (
        ready
        and reduction["equivalent_behavior_score"]
        and reduction["paired_speed_intervals"][0]["lower95"] > 1
    ):
        verdict = "positive"
    if blocked:
        verdict = "blocked"
    if not checked:
        verdict = "disqualified"
    value: Json = dict(
        reduction,
        experiment_id=8174,
        task_id="exp8174-complete-request-cost",
        schema="carnot.complete_request_cost.v1",
        honest_verdict="complete_blocked_" + str(blocked)
        if verdict == "blocked"
        else "complete_" + verdict + "_complete_request_cost",
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        fixture_mode=fixture,
        claim_scope="Independent complete durable requests; descriptive pipeline timing when outputs or decisions differ; no model quality or learning claim",
        exposure_scope="exposed_historical_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        gate_check_summary=checks,
        inference_substrate="live_llm_inference"
        if counts["generation_calls_completed"]
        else "private_transport_fixture"
        if fixture
        else "no_model_execution",
        inference_substrate_class="model_bounded_generation"
        if counts["model_loads_attempted"]
        else "no_model_load",
        declared_inference_substrate_class="model_bounded_generation",
        inference_mode="live_gpu" if counts["generation_calls_completed"] else "no_live_inference",
        MODEL_SPECS=MODEL_SPECS,
        trained_head_specs=data.get("trained_head_specs", []),
        model_invocation_counts=counts,
        call_ledger=[] if fixture else ledger.rows,
        private_transport_ledger=ledger.rows if fixture else [],
        cited_upstream_artifacts=[
            dict(
                experiment_id=8173,
                fields_imported=[
                    "service_protocol_ready_score",
                    "immutable_input_snapshot",
                    "next_service_protocol",
                ],
                sha256=PIN,
            )
        ],
        source_artifact_hashes=data["refs"],
        raw_shard_hashes=[
            reference(raw / "primitive_rows.json"),
            reference(raw / "input_data.json"),
        ],
        code_config_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
        config_sha256=canonical_hash(CONFIG),
        random_seed=CONFIG["seed"],
        run_date=date,
        duration_s=duration,
        sample_size_budget=CONFIG,
        phase_spans=[
            dict(phase=r.get("name"), duration_s=r.get("duration_s", 0)) for r in receipts
        ],
        acceptance_gates=dict(
            sources=24,
            equivalent_behavior_required_for_speed_claim=True,
            nfr01_lower95=10,
            duration_floor_s=10,
        ),
        field_principles=dict(
            rows="Each arm pays its own acquisition exactly once in inclusive latency",
            counts="Original sources define independence",
            readiness="Normal owned validation is required",
            provenance="Imported calls do not count as current calls",
        ),
        methodology="48 distinct bounded judgments joined by original source; source bootstrap with10000 draws; batch8 durable response clocks; separate cold model load",
        gpu_receipts={k: v for k, v in result.items() if k not in {"work", "ledger", "checks"}},
        startup_costs=dict(
            cold_load_ns=sum(
                r["ended_monotonic_ns"] - r["started_monotonic_ns"]
                for r in ledger.rows
                if r["operation"] == "model_load" and r["ended_monotonic_ns"]
            ),
            startup_total_s=result.get("startup_total_s"),
            amortization_requests=48,
            charged_to_warm_latencies=False,
        ),
        comparison_scopes=data.get("upstream_scope", {}),
    )
    value.update(
        complete_service_ready_score=int(ready),
        equivalent_behavior_score=int(ready and reduction["equivalent_behavior_score"]),
        nfr01_met=bool(ready and reduction["nfr01_met"]),
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Authenticate evidence and recompute headlines in a fresh process without a model."""
    try:
        value = json.loads(path.read_text())
        if (
            checksum(value) != value["reproducibility_checksum"]
            or canonical_hash(CONFIG) != value["config_sha256"]
        ):
            return False
        refs = value["raw_shard_hashes"] + value["source_artifact_hashes"]
        refs += [dict(path=str(ROOT / p), sha256=h) for p, h in value["code_config_hashes"].items()]
        refs += [
            dict(path=r["log_path"], sha256=r["log_sha256"])
            for r in value["validation_receipts"]
            if "log_path" in r
        ]
        if any(sha256_file(Path(r["path"])) != r["sha256"] for r in refs):
            return False
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_text())
        data = json.loads(Path(value["raw_shard_hashes"][1]["path"]).read_text())
        if not validate_work(data, work):
            return False
        reduction = reduce(work)
        if any(
            value[k] != reduction[k]
            for k in reduction
            if k not in {"complete_service_ready_score", "equivalent_behavior_score", "nfr01_met"}
        ):
            return False
        with tempfile.TemporaryDirectory(prefix="carnot-8174-replay-") as private:
            rebuilt = build(
                data,
                dict(
                    work=work,
                    ledger=value.get("private_transport_ledger")
                    if value["fixture_mode"]
                    else value["call_ledger"],
                    checks=value["gate_check_summary"][len(data["checks"]) :],
                    **value["gpu_receipts"],
                ),
                Path(private),
                value["validation_receipts"],
                value["run_date"],
                value["duration_s"],
                value["fixture_mode"],
            )
        return all(
            value[k] == rebuilt[k]
            for k in [
                "verdict_class",
                "required_checks_passed",
                "complete_service_ready_score",
                "equivalent_behavior_score",
                "nfr01_met",
                "model_invocation_counts",
            ]
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
