"""REQ-REPORT-9093-V775: preserve work that incomplete service clocks cannot price.

An inventory can be complete while the scientific service remains unavailable.
Frozen byte receipts let another process repeat the accounting without inference.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9093_v775_service_cost", "exp9093-service-cost"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/service_cost_9093.py",
    "python/carnot/reporting/service_cost_runner_9093.py",
    CLI,
]
TESTS = ["tests/python/test_service_cost_9093.py"]
SOURCES = {
    n: f"results/experiment_{n}_v775_{suffix}.json"
    for n, suffix in zip(
        range(9082, 9093),
        (
            "contract",
            "prospective_custody",
            "human_methods",
            "collect_a",
            "collect_b",
            "sealed_admission",
            "human_embeddings",
            "static_seal",
            "online_seal",
            "graph_decision",
            "human_decision",
        ),
        strict=True,
    )
}
SOURCES[9079] = "results/experiment_9079_v774_hardware_boundary.json"
OPERATIONS = (
    "roster_construction",
    "custody",
    "GGUF_load",
    "generation",
    "embedding_extraction",
    "feature_transformation",
    "spline_fit",
    "linear_fit",
    "prediction",
    "sparse_update",
    "dense_update",
    "guard_scan",
    "rollback",
    "checkpoint_fsync",
    "independent_reduction",
)


def progress(phase: str) -> None:
    """Flush every boundary so supervisors can distinguish progress from a stall."""
    print(f"[exp9093] phase={phase}", flush=True)


def ref(path: Path) -> Json:
    """Exact byte hashes distinguish absent evidence from observed zero."""
    return dict(path=str(path), sha256=sha256_file(path) if path.is_file() else None)


def gate(
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    check: str,
    task: Any,
    digest: str | None = None,
) -> Json:
    """Keep the failed operand inspectable without changing an accounting gate."""
    return dict(
        check=check,
        upstream=str(task),
        path=path,
        sha256=digest,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=type(expected) is type(observed) and expected == observed,
    )


def freeze(root: Path, raw: Path) -> Json:
    """Snapshot declared sources and original producer bindings before reduction."""
    progress("before_source_freeze")
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    entries: list[Json] = []
    checks = [gate(str(raw), "mode", 0o700, raw.stat().st_mode & 0o777, "private_storage", TASK)]

    def save(path: Path, kind: str, task: int, expected: str | None = None) -> Json:
        actual = ref(path)
        row = dict(
            source_path=str(path),
            kind=kind,
            task=task,
            producer_bound=expected is not None,
            authenticated=expected is None or actual["sha256"] == expected,
            **actual,
        )
        checks.append(
            gate(
                str(path),
                "sha256" if expected else "exists",
                expected or True,
                actual["sha256"] if expected else True if actual["sha256"] else "missing_file",
                "producer_hash" if expected else "source_presence",
                task,
                actual["sha256"],
            )
        )
        if actual["sha256"]:
            target = raw / "objects" / actual["sha256"].split(":")[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                target.write_bytes(path.read_bytes())
            target.chmod(0o400)
            row["path"] = str(target)
        else:
            row["path"] = None
        entries.append(row)
        return row

    for task, label in sorted(SOURCES.items()):
        row = save(root / label, "artifact", task)
        if row["path"]:
            value = json.loads(Path(row["path"]).read_bytes())
            checks.append(
                gate(
                    str(root / label),
                    "experiment_id",
                    task,
                    value.get("experiment_id", "missing_field"),
                    "identity",
                    task,
                    row["sha256"],
                )
            )
            bindings = value.get("code_config_hashes", []) + value.get("raw_shard_hashes", [])
            receipts = value.get("current_work_receipts", [])
            if isinstance(receipts, list):
                bindings += [r for r in receipts if "path" in r]
            for binding in bindings:
                path = Path(binding["path"])
                path = path if path.is_absolute() else root / path
                kind = (
                    "work"
                    if path.name == "work.json"
                    else "receipt"
                    if path.name == "current_work.json"
                    else "binding"
                )
                save(path, kind, task, binding["sha256"])
        if task in (9085, 9086, 9088):
            suffix = {9085: "collect_a", 9086: "collect_b", 9088: "human_embeddings"}[task]
            save(root / f"results/experiment_{task}_{suffix}.json", "diagnostic", task)
        for path in sorted(
            (root / "results/raw").glob(f"experiment_{task}_*/invocations/*/work.json")
        ):
            if str(path) not in {r["source_path"] for r in entries}:
                save(path, "work", task)
        progress(f"source_{task}_frozen")
    result = dict(
        entries=entries,
        gate_check_summary=checks,
        scope="V775 task invocations and V774 boundary context; fixtures separate",
    )
    atomic_json(raw / "inventory_manifest.json", result)
    (raw / "inventory_manifest.json").chmod(0o400)
    return result


def interval_union(intervals: list[tuple[int, int]]) -> int:
    """Nested parent and child observations describe one physical time interval."""
    end, total = 0, 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(rows: list[Json]) -> Json:
    """Reduce raw clocks by domain while withholding incomplete service ratios."""
    seen: set[str] = set()
    groups: dict[tuple[str, str], list[Json]] = defaultdict(list)
    unavailable = []
    for row in rows:
        if row["unit_id"] in seen:
            raise ValueError("duplicate_operation")
        seen.add(row["unit_id"])
        start, end = row.get("start_ns"), row.get("end_ns")
        if start is None or end is None:
            unavailable.append(
                dict(
                    unit_id=row["unit_id"],
                    operand="raw_start_end",
                    reason=row.get("censor_reason", "missing_field"),
                )
            )
            continue
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError("clock_order_or_type")
        groups[(row["scope"], row["clock_domain"])].append(row)
    latency = []
    shares = []
    for (scope, domain), observed in sorted(groups.items()):
        intervals = [(r["start_ns"], r["end_ns"]) for r in observed]
        charged = interval_union(intervals) / 1e9
        samples = [(b - a) / 1e9 for a, b in intervals]
        parents = [r for r in observed if not r.get("parent_id")]
        parent_samples = [(r["end_ns"] - r["start_ns"]) / 1e9 for r in parents]
        latency.append(
            dict(
                scope=scope,
                clock_domain=domain,
                raw_samples_s=samples,
                parent_samples_s=parent_samples,
                charged_s=charged,
                median_s=statistics.median(samples),
                p95_s=sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
                throughput_completed_operations_per_s=sum(
                    r["status"] == "completed" for r in parents
                )
                / charged
                if charged
                else None,
                uncertainty="Descriptive observed intervals only; no scientific request population or confidence interval.",
            )
        )
        if charged and len(parents) == 1:
            shares.append(
                dict(
                    scope=scope,
                    clock_domain=domain,
                    operation=parents[0]["operation"],
                    share=1.0,
                    boundary="measured inclusive parent only; component shares unavailable",
                )
            )
    return dict(
        observed_latency_summary=latency,
        complete_service_ratios=[],
        unavailable_operands=unavailable,
        bottleneck_operation_shares=shares,
        energy_sensor_coverage=dict(
            status="unmeasured",
            joules=None,
            measured_operations=0,
            intended_operations=len(rows),
            boundary="No authenticated power integral; TDP is not a measurement.",
        ),
    )


def cost_row(uid: str, task: int, operation: str, status: str, scope: str, **extra: Any) -> Json:
    """Use one primitive row shape for measured, censored and unstarted work."""
    row: Json = dict(
        unit_id=uid,
        task=task,
        arm="administrative",
        condition=scope,
        family="V775",
        group=uid,
        seed=None,
        operation=operation,
        status=status,
        scope=scope,
        start_ns=None,
        end_ns=None,
        clock_domain=uid,
        parent_id=None,
        latency_s=None,
        absolute_metric=None,
        numerator=None,
        denominator=None,
        model=None,
        device=None,
        joules=None,
        peak_memory_bytes=None,
        censor_reason=None,
        exclusion_reason=None,
        measurement_boundary="unavailable",
    )
    row.update(extra)
    if row["start_ns"] is not None and row["end_ns"] is not None:
        row["latency_s"] = (row["end_ns"] - row["start_ns"]) / 1e9
        row.update(
            absolute_metric=row["latency_s"],
            numerator=row["end_ns"] - row["start_ns"],
            denominator=1e9,
        )
    return row


def inventory(manifest: Json) -> Json:
    """Reconstruct every retained invocation without using producer headlines as clocks."""
    checks = list(manifest["gate_check_summary"])
    values: list[tuple[Json, Json]] = []
    for entry in manifest["entries"]:
        if entry["path"]:
            if ref(Path(entry["path"]))["sha256"] != entry["sha256"]:
                raise ValueError("retained_hash_drift")
            if entry["kind"] != "binding":
                values.append((entry, json.loads(Path(entry["path"]).read_bytes())))
    rows: list[Json] = []
    dispositions = []
    historical = []
    seen: set[str] = set()
    for task in range(9082, 9093):
        selected = [(e, v) for e, v in values if e["task"] == task]
        primary = next((v for e, v in selected if e["kind"] == "artifact"), {})
        diagnostic = next((v for e, v in selected if e["kind"] == "diagnostic"), {})
        checks.extend(primary.get("gate_check_summary", []))
        disposition = "gate_skipped" if diagnostic else primary.get("verdict_class", "unstarted")
        dispositions.append(
            dict(
                task=task,
                disposition=disposition,
                intended=1,
                attempted=int(bool(primary)),
                completed=int(bool(primary) and disposition not in ("blocked", "disqualified")),
                failed=int(disposition == "disqualified"),
                censored=int(disposition in ("blocked", "gate_skipped")),
                original_honest_verdict=primary.get(
                    "honest_verdict", diagnostic.get("honest_verdict")
                ),
                source_rows=primary.get("rows", []),
                boundary="Task accounting counts, not scientific request counts.",
            )
        )
        historical.append(
            dict(
                task=task,
                MODEL_SPECS=primary.get("MODEL_SPECS"),
                invocation_counts=primary.get("model_invocation_counts"),
                scope="Historical producer provenance only; never current invocation counters.",
            )
        )
        if diagnostic:
            path = str(diagnostic.get("failed_evidence_path", "missing_field"))
            operand = next((v for e, v in values if e["source_path"] == path), {})
            field = diagnostic.get("failed_field", "missing_field")
            checks.append(
                gate(
                    path,
                    field,
                    diagnostic.get("failed_expected", 1),
                    operand.get(field, "missing_field"),
                    "upstream_gate_observation",
                    task,
                    diagnostic.get("failed_evidence_sha256"),
                )
            )
            checks.append(
                gate(
                    path,
                    "sha256",
                    diagnostic.get("failed_evidence_sha256"),
                    next(
                        (e["sha256"] for e, _ in values if e["source_path"] == path), "missing_file"
                    ),
                    "diagnostic_operand_hash",
                    task,
                )
            )
        for entry, value in selected:
            receipts = value.get("current_work_receipts", [])
            if isinstance(receipts, dict):
                receipts = [receipts]
            if entry["kind"] == "receipt":
                receipts = [value]
            if entry["kind"] == "work" and "start_ns" in value:
                receipts = [
                    dict(
                        started_monotonic_ns=value["start_ns"],
                        ended_monotonic_ns=value["end_ns"],
                        current_owner_pid=value["owner_pid"],
                        current_run_id=str(value["start_ns"]),
                        duration_s=(value["end_ns"] - value["start_ns"]) / 1e9,
                        execution_venue="host",
                        inference_substrate_class="no_model_load",
                    )
                ]
            for receipt in receipts:
                if "started_monotonic_ns" not in receipt:
                    continue
                start, end = receipt["started_monotonic_ns"], receipt["ended_monotonic_ns"]
                uid = f"{task}/{receipt['current_owner_pid']}/{start}"
                if uid in seen:
                    continue
                seen.add(uid)
                domain = receipt.get("clock_domain", uid)
                clock_check = gate(
                    entry["source_path"],
                    "duration_s",
                    (end - start) / 1e9,
                    receipt.get("duration_s", "missing_field"),
                    "raw_clock_duration",
                    task,
                    entry["sha256"],
                )
                checks.append(clock_check)
                rows.append(
                    cost_row(
                        uid,
                        task,
                        "qualification",
                        "completed"
                        if primary and (entry["kind"] != "work" or entry["producer_bound"])
                        else "censored",
                        "administrative",
                        start_ns=start
                        if clock_check["passed"] and entry["authenticated"]
                        else None,
                        end_ns=end if clock_check["passed"] and entry["authenticated"] else None,
                        raw_start_ns=start,
                        raw_end_ns=end,
                        clock_domain=domain,
                        model=receipt.get("MODEL_SPECS", []),
                        device=receipt.get("execution_venue"),
                        source_reference=dict(path=entry["path"], sha256=entry["sha256"]),
                        censor_reason=None
                        if primary
                        else "partial validation or unpublished invocation",
                        measurement_boundary="retained parent monotonic interval; excludes costs outside declared receipt; cross-run boot identity unavailable",
                    )
                )
                for index, validation in enumerate(value.get("validation_receipts", [])):
                    rows.append(
                        cost_row(
                            f"{uid}/validation/{index}",
                            task,
                            "validation",
                            "completed" if validation.get("passed") else "failed",
                            "administrative",
                            start_ns=validation.get("started_monotonic_ns"),
                            end_ns=validation.get("ended_monotonic_ns"),
                            parent_id=uid,
                            clock_domain=domain,
                            device="host",
                            source_reference=dict(path=entry["path"], sha256=entry["sha256"]),
                            raw_validation_receipt=validation,
                            censor_reason="missing raw validation boundaries",
                        )
                    )
        for arm in ("static", "online", "exposed_human"):
            for operation in OPERATIONS:
                rows.append(
                    cost_row(
                        f"{task}/{arm}/{operation}",
                        task,
                        operation,
                        "unstarted" if disposition == "unstarted" else "censored",
                        "science",
                        arm=arm,
                        family="human_LPHB" if arm == "exposed_human" else "graph",
                        censor_reason=f"{disposition}; no authenticated per-request operation clocks",
                        exclusion_reason="No complete scientific service operand; administrative and fixture clocks cannot substitute.",
                    )
                )
    reduced = summarize(rows)
    return dict(
        reduced,
        rows=rows,
        operation_cost_rows=rows,
        invocation_dispositions=dispositions,
        historical_model_provenance=historical,
        gate_check_summary=checks,
        intended_count=len(dispositions),
        attempted_count=sum(r["attempted"] for r in dispositions),
        completed_count=sum(r["completed"] for r in dispositions),
        failed_count=sum(r["failed"] for r in dispositions),
        censored_count=sum(r["censored"] for r in dispositions),
        excluded_count=0,
        unstarted_count=sum(r["disposition"] == "unstarted" for r in dispositions),
        independent_count=0,
        effective_independent_groups=0,
        scientific_request_counts=dict(
            intended=None,
            attempted=None,
            completed=None,
            failed=None,
            censored=None,
            reason="Authenticated roster unavailable; task counts and seeds are not independent requests.",
        ),
        cost_per_request=dict(
            intended=None,
            attempted=None,
            completed=None,
            uncertainty="Neither complete scientific costs nor request denominators are authenticated.",
        ),
        censored_cost_rows=[r for r in rows if r["status"] in ("censored", "unstarted", "failed")],
        memory_rows=[
            dict(
                scope="science",
                peak_memory_bytes=None,
                boundary="No authenticated scientific process high-water mark.",
            )
        ],
    )


def fixture_replay(raw: Path) -> list[Json]:
    """Exercise raw clocks and fsync on tiny synthetic rows without pricing a service."""
    progress("before_bounded_cpu_fixture")
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(3):
        start = time.monotonic_ns()
        payload = [dict(unit_id=str(i), numerator=i, denominator=8) for i in range(8)]
        path = raw / f"fixture-{index}.json"
        atomic_json(path, dict(rows=payload))
        restored = json.loads(path.read_bytes())
        assert sum(r["numerator"] for r in restored["rows"]) == 28
        end = time.monotonic_ns()
        rows.append(
            cost_row(
                f"fixture/{index}",
                9093,
                "checkpoint_fsync_and_reduction",
                "completed",
                "fixture",
                start_ns=start,
                end_ns=end,
                clock_domain="current-host",
                device=platform.machine(),
                group="synthetic-mechanics",
                seed=775093,
                peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                measurement_boundary="Synthetic eight-row JSON write/fsync/read/reduction; interpreter high-water RSS; not production performance",
                source_reference=ref(path),
            )
        )
        progress(f"fixture_{index}_complete")
    return rows
