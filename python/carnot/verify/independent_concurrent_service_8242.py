"""REQ-VERIFY-8242: measure fresh requests with one fixed Rust durable service.

Each workload owns a fresh server. Repeated source sweeps improve descriptive
accounting but do not turn this designed workload into a population sample.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.inference import concurrency_runtime_8227 as runtime
from carnot.verify import concurrency_canary_8227 as e
from carnot.verify import prospective_service_8214 as service

Json = dict[str, Any]
CAP_S = 3000


def persist(data: Json, row: Json, response: Json, native: Any, raw: Path) -> Json:
    """Keep normalization and durable scoring inside the current request boundary."""
    began = time.monotonic_ns()
    family = next(
        r["family_id"] for r in data["roster"] if r["source_cluster_id"] == row["source_cluster_id"]
    )
    branch: Json = dict(
        unit_id=row["request_id"],
        source_cluster_id=row["source_cluster_id"],
        arm=service.ARMS[1],
        request=row["envelope"]["payload"],
        arrival_ns=began,
    )
    branch["values"] = service.convert(row, response, family)
    converted = time.monotonic_ns()
    e.atomic_json(raw / "pending.json", dict(request_id=row["request_id"], values=branch["values"]))
    pending = time.monotonic_ns()
    service.prior.commit_group(data, [branch], native, raw)
    read_start = time.monotonic_ns()
    branch["state"] = service.prior.shared.host.Store(raw / "store.json", e.key(data["head"])).state
    branch.update(
        start_ns=began,
        conversion_end_ns=converted,
        pending_end_ns=pending,
        readout_start_ns=read_start,
        end_ns=time.monotonic_ns(),
    )
    branch["stages"] = service.stages(branch)
    return branch


def measure(data: Json, resources: Json, raw: Path, *, cap_s: float = CAP_S) -> Json:
    """Own bounded loads and shutdowns, retaining every original request obligation."""
    protocol = data["protocol"]
    rows = [
        dict(
            deepcopy(r),
            request_id="8242-" + r["request_id"],
            response_id=None,
            status="censored",
            result={},
        )
        for r in protocol["benchmark"]
    ]
    work: Json = dict(rows=rows, workloads=[], loads=[], checks=[], phase_spans=[], telemetry=[])
    began = time.monotonic()
    gpu, model = resources["gpu"], Path(resources["model"]["model_path"])
    try:
        lease = runtime.GpuLease.acquire(
            runtime_dir="/tmp/carnot-gpu-leases",
            task_id="exp8242-independent-concurrent-service",
            device_uuid=gpu["uuid"],
            expected_model=str(model),
            vram_before_mb=gpu["used_mb"],
            ttl_s=3300,
        )
    except runtime.LeaseError as error:
        work["checks"].append(runtime.operand("gpu_lease", model, "free owned lease", str(error)))
        return work
    lease.transition("admitted")
    lease.transition("loading")
    try:
        for name in protocol["launch_order"]:
            if time.monotonic() - began >= cap_s:
                break
            selected = [r for r in rows if r["workload"] == name]
            folder = raw / name
            folder.mkdir(parents=True, exist_ok=True)
            owned = runtime.QwenRuntime(model, folder, gpu["index"])
            owned.command = runtime.command(model, folder, gpu["index"])
            owned.port = int(owned.command[owned.command.index("--port") + 1])
            e.atomic_json(folder / "command.json", dict(argv=owned.command))
            cold = time.monotonic_ns()
            load: Json = dict(workload=name, attempted=True, completed=False)
            work["loads"].append(load)
            warm_start = warm_end = None
            e.progress(
                "8242_model_load_before_" + name, len(work["workloads"]), 4 - len(work["workloads"])
            )
            try:
                load["receipt"] = runtime.bounded(
                    owned.load, min(300, max(0.001, cap_s - (time.monotonic() - began)))
                )
                load["completed"] = True
                props = load["receipt"]["props"]
                if (
                    props["total_slots"] != 2
                    or e.key(props["chat_template"]) != protocol["identity"]["chat_template_sha256"]
                ):
                    raise ValueError("served_slots_or_template")
                native, binding = service.prior.shared.host.old.prior.old.host.load_binding(data)
                load["service_binding"] = binding
                e.progress("8242_model_load_after_" + name, 1, 0)
                if lease.document["phase"] == "loading":
                    lease.transition("resident", vram_mb=model.stat().st_size // 2**20)
                    lease.transition("inferencing")
                concurrency = 1 if name.endswith("serial") else 2
                cursors = [iter(selected[s::concurrency]) for s in range(concurrency)]

                def generate(payload: Json, slot: int) -> Json:
                    row = next(cursors[slot])
                    if row["envelope"]["payload"] != payload:
                        raise ValueError("queued_request_payload")
                    response = runtime.bounded(
                        lambda: runtime.generate(owned, payload, slot),
                        min(120, max(0.001, cap_s - (time.monotonic() - began))),
                    )
                    generated = time.monotonic_ns()
                    e.atomic_json(folder / row["request_id"] / "generation.json", response)
                    normalized = time.monotonic_ns()
                    try:
                        response["service"] = persist(
                            data, row, response, native, folder / row["request_id"]
                        )
                    except (ValueError, KeyError, TypeError) as error:
                        response["service"] = dict(
                            status="failed",
                            error=str(error),
                            start_ns=normalized,
                            end_ns=time.monotonic_ns(),
                        )
                    response["generation_end_ns"] = generated
                    return response

                telemetry = runtime.execute(
                    [
                        runtime.CommandSpec(
                            name + "_active_gpu",
                            (
                                "nvidia-smi",
                                f"--id={gpu['index']}",
                                "--query-gpu=uuid,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits",
                            ),
                            "active_measured_server",
                            15,
                        )
                    ],
                    folder / "telemetry",
                )
                work["telemetry"].extend(telemetry)
                warm_start = time.monotonic_ns()
                e.progress("8242_benchmark_before_" + name, 0, 24)
                observed = e.acquire(
                    selected,
                    generate,
                    folder / "requests",
                    1 if name.endswith("serial") else 2,
                    cap_s=max(0, cap_s - (time.monotonic() - began)),
                )
                warm_end = time.monotonic_ns()
                for row in observed:
                    row["journal_path"] = str(folder / "requests/events.jsonl")
                    row["response_id"] = row["result"].get("id")
                    clocks = row["clocks"]
                    row["latency_s"] = (clocks["durability"] - clocks["issue"]) / 1e9
                    row["issue_fsync_s"] = (clocks["queue"] - clocks["issue"]) / 1e9
                    row["terminal_fsync_s"] = (clocks["durability"] - clocks["end"]) / 1e9
                    row["active_request_s"] = (
                        (clocks["durability"] - clocks["start"]) / 1e9 if clocks["start"] else 0
                    )
                    row["queue_s"] = (
                        (clocks["start"] - clocks["queue"]) / 1e9 if clocks["start"] else None
                    )
                    row["acquisition_status"] = row["status"]
                    response = row["result"].get("response", row["result"])
                    branch = response.get("service", {})
                    if branch.get("status") == "failed":
                        row["status"] = "failed"
                    row["acquisition_s"] = (
                        (response.get("generation_end_ns", clocks["end"]) - clocks["start"]) / 1e9
                        if clocks["start"]
                        else 0
                    )
                    row["service_s"] = (branch.get("end_ns", 0) - branch.get("start_ns", 0)) / 1e9
                    row["scoring_s"] = branch.get("stages", {}).get("scoring_ns", 0) / 1e9
                    row["prefill_decode"] = response.get("native_events", [{}])[-1].get("timings")
                    rows[
                        rows.index(next(r for r in rows if r["request_id"] == row["request_id"]))
                    ] = row
                e.progress("8242_benchmark_after_" + name, 24, 0)
            except (RuntimeError, OSError, ValueError, KeyError, TimeoutError) as error:
                load["error"] = str(error)
                work["checks"].append(
                    runtime.operand(
                        name + "_startup_or_service", model, "complete owned workload", str(error)
                    )
                )
            finally:
                shutdown = time.monotonic_ns()
                e.progress("8242_shutdown_before_" + name, 0, 1)
                cleanup = owned.close()
                ended = time.monotonic_ns()
                work["checks"].append(
                    runtime.operand(name + "_cleanup", owned.log, True, cleanup["leak_free"])
                )
                work["workloads"].append(
                    dict(
                        workload=name,
                        startup_s=((warm_start or shutdown) - cold) / 1e9,
                        warm_s=(warm_end - warm_start) / 1e9 if warm_start and warm_end else None,
                        shutdown_s=(ended - shutdown) / 1e9,
                        cold_s=(ended - cold) / 1e9,
                        cleanup=cleanup,
                    )
                )
                work["phase_spans"].append(dict(phase=name, start_ns=cold, end_ns=ended))
                work["phase_spans"].extend(
                    [
                        dict(
                            phase="model_load",
                            workload=name,
                            start_ns=cold,
                            end_ns=warm_start or shutdown,
                        ),
                        dict(
                            phase="generation",
                            workload=name,
                            start_ns=warm_start or shutdown,
                            end_ns=warm_end or shutdown,
                        ),
                        dict(phase="shutdown", workload=name, start_ns=shutdown, end_ns=ended),
                    ]
                )
                e.atomic_json(raw / "checkpoint.json", work)
                e.progress("8242_shutdown_after_" + name, 1, 0)
    finally:
        if lease.document["phase"] == "inferencing":
            lease.transition("unloading")
            lease.transition(
                "validating",
                vram_mb=gpu["used_mb"],
                exit_code=0,
                unload_observed=all(w["cleanup"]["leak_free"] for w in work["workloads"]),
            )
        lease.transition(
            "terminal_complete" if all(c["passed"] for c in work["checks"]) else "terminal_blocked"
        )
        work["lease"] = lease.release()
    work["measurement_duration_s"] = time.monotonic() - began
    return work


def reduce(work: Json) -> Json:
    """Recompute completion-safe costs, using sources rather than repeat counts."""
    rows, workloads = work.get("rows", []), work.get("workloads", [])
    valid = [r for r in rows if r["status"] == "completed"]
    arms = ["serial", "concurrent"]
    counts = {a: sum(r["arm"] == a for r in valid) for a in arms}
    sources = sorted({r["source_cluster_id"] for r in rows})
    pairs = []
    differences = []
    for source in sources:
        selected = [r for r in rows if r["source_cluster_id"] == source]
        if len(selected) != 4 or any(r["status"] != "completed" for r in selected):
            continue
        means = {
            a: float(np.mean([r["latency_s"] for r in selected if r["arm"] == a])) for a in arms
        }
        pairs.append(means["serial"] - means["concurrent"])
        for sweep in ["s1", "s2"]:
            pair = [next(r for r in selected if r["workload"] == sweep + "_" + a) for a in arms]
            differences.append(
                dict(
                    source_cluster_id=source,
                    sweep=sweep,
                    output_equal=pair[0]["result"]["choices"] == pair[1]["result"]["choices"],
                    token_delta=pair[1]["result"]["usage"]["completion_tokens"]
                    - pair[0]["result"]["usage"]["completion_tokens"],
                )
            )
    intervals = None
    if len(pairs) >= 20:
        rng = np.random.default_rng(e.SEED)
        draws = np.asarray(pairs)[rng.integers(0, len(pairs), size=(2000, len(pairs)))].mean(axis=1)
        intervals = dict(
            supported_sources=len(pairs),
            mean_paired_delta_s=float(np.mean(pairs)),
            descriptive_source_bootstrap_95_percent=[
                float(v) for v in np.quantile(draws, [0.025, 0.975])
            ],
            scope="designed development sources; two sweeps averaged within source",
        )
    acquisition = sum(r.get("acquisition_s", 0) for r in rows)
    downstream = sum(r.get("service_s", 0) for r in rows)
    scoring = sum(r.get("scoring_s", 0) for r in rows)
    total = sum(
        r.get("active_request_s", r.get("acquisition_s", 0) + r.get("service_s", 0))
        + r.get("issue_fsync_s", 0)
        for r in rows
    )
    totals = {a: sum(w["cold_s"] for w in workloads if w["workload"].endswith(a)) for a in arms}
    sufficient = (
        len(rows) == 96
        and len(pairs) >= 20
        and len(workloads) == 4
        and all(r["status"] in {"completed", "error", "failed"} for r in rows)
    )
    return dict(
        rows=rows,
        intended_count=96,
        completed_count=len(valid),
        failed_count=sum(r["status"] in {"error", "failed"} for r in rows),
        censored_count=sum(r["status"] in {"blocked", "censored"} for r in rows),
        excluded_count=0,
        independent_count=len({r["source_cluster_id"] for r in valid}),
        paired_source_count=len(pairs),
        valid_completion_counts=counts,
        token_totals={
            a: {
                k: sum(
                    r["result"].get("response", r["result"]).get("usage", {}).get(k, 0)
                    for r in rows
                    if r["arm"] == a
                )
                for k in ["prompt_tokens", "completion_tokens"]
            }
            for a in arms
        },
        output_token_differences=differences,
        sweep_makespans=workloads,
        warm_throughput={
            w["workload"]: sum(r["workload"] == w["workload"] for r in valid) / w["warm_s"]
            if w.get("warm_s")
            else None
            for w in workloads
        },
        cold_inclusive_makespans=totals if workloads else None,
        cold_costs=[{k: w[k] for k in ["workload", "startup_s", "shutdown_s"]} for w in workloads],
        latency_intervals=intervals,
        throughput_population_interval=None,
        acquisition_fraction=acquisition / total if total else None,
        scoring_fraction=scoring / total if total else None,
        scoring_speedup_upper_bound=1 / (1 - scoring / total)
        if total and scoring < total
        else None,
        amdahl_scope="sum of observed request service times, excludes cold costs and queue; Rust fixed",
        nfr01_met=None,
        nfr01_status="untested_Rust_fixed",
        evidence_complete=sufficient,
        improvement_qualified=bool(
            sufficient
            and counts["concurrent"] >= counts["serial"]
            and totals["concurrent"] < totals["serial"]
        ),
        active_gpu_telemetry=work.get("telemetry", []),
        phase_spans=work.get("phase_spans", []),
    )
