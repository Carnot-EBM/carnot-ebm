"""REQ-REPORT-8968: complete process costs remain valid when quality gains are null."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import t

from carnot.memory import complete_service_worker_8968 as w
from carnot.reporting import online_response_streams_8966 as upstream
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.causal_learning_audit_8810 import require as check
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT, SEED = upstream.ROOT, 7668968
NAME, TASK = "experiment_8968_v766_complete_service_cost", "exp8968-complete-service-cost"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/memory/complete_service_worker_8968.py",
    "python/carnot/reporting/complete_service_cost_8968.py",
    "python/carnot/reporting/complete_service_cost_runner_8968.py",
    CLI,
]
TESTS = ["tests/python/test_complete_service_cost_8968.py"]
INPUTS = upstream.INPUTS + [
    (
        "results/experiment_8967_v766_online_benefit_audit.json",
        "sha256:dbb48d10ce064fb08fe6d0efcd6671111863641c7591dcfc4bf6df68c5dc3f21",
        "online_audit_ready_score",
        "exp8967-online-benefit-audit",
    )
]
auth = namespace(upstream, "service8968_auth")
auth.INPUTS = INPUTS
authenticate, ref = auth.authenticate, upstream.ref
__all__ = ["authenticate", "ref", "check"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so long process work is visible to the supervisor."""
    print(f"[exp8968] phase={phase} completed={completed} pending={pending}", flush=True)


def roster() -> list[Json]:
    """Randomized balanced arm order pairs24 fresh process seeds on the fixed corpus."""
    order = np.random.default_rng(SEED).permutation(w.ARMS).tolist()
    return [
        dict(
            unit_id=f"cost8968-{i:02d}/{arm}",
            cluster_id=f"cost8968-{i:02d}",
            stream_index=i,
            seed=SEED + i,
            arm=arm,
            order=position,
        )
        for i in range(24)
        for position, arm in enumerate(order[i % 4 :] + order[: i % 4])
    ]


def prepare(evidence: Json | None, raw: Path) -> Json:
    """Frozen request operands retain original producers; no generation is repeated."""
    path = raw / "plan.json"
    if evidence is None:
        return dict(json.loads(path.read_bytes()))
    producers = evidence["producers"]
    corpus = producers["exp8963-corpus-admission"]
    detector = producers["exp8964-relational-decisions"]
    service = upstream.prior.read_reference(detector["calibrated_state_reference"])
    views = {
        role: upstream.prior.read_reference(corpus["inference_views"][role])
        for role in ("online", "guard")
    }
    by_id = dict(zip(views["online"]["unit_ids"], views["online"]["rows"], strict=True))
    streams = [
        dict(s, rows=[json.dumps(by_id[u], sort_keys=True) for u in s["unit_ids"]])
        for s in corpus["broker_manifest"]["online"]["streams"]
    ]
    check(
        len(streams) == 24 and all(len(s["rows"]) == 32 for s in streams) and len(by_id) == 768,
        "corpus_roster",
    )
    services = {}
    for arm in w.ARMS:
        model = "confidence_logistic" if arm == "confidence_logistic" else "relational_energy"
        target = raw / "services" / (arm + ".json")
        atomic_json(
            target,
            dict(states=service["states"][model], temperature=service["temperatures"][model]),
        )
        services[arm] = str(target)
    plan = dict(
        roster=roster(),
        streams=streams,
        services=services,
        service_reference=detector["calibrated_state_reference"],
        guard_rows=views["guard"]["rows"],
        guard_labels=views["guard"]["labels"],
        quality=dict(
            detector_benefit_score=detector.get("detector_benefit_score"),
            relational_controls_ready_score=producers["exp8965-relational-controls"][
                "relational_controls_ready_score"
            ],
            online_benefit_score=producers["exp8967-online-benefit-audit"].get(
                "online_benefit_score"
            ),
        ),
        upstream_generation=dict(
            scope="retained Qwen responses; zero current generation calls; these durations are not remeasured",
            producers=[
                dict(
                    reference=ref(ROOT / f"results/experiment_{i}_v766_graph_corpus_{a}.json"),
                    retained_costs={
                        k: json.loads(
                            (
                                ROOT / f"results/experiment_{i}_v766_graph_corpus_{a}.json"
                            ).read_bytes()
                        )[k]
                        for k in ("duration_s", "phase_spans", "model_invocation_counts")
                    },
                )
                for i, a in ((8961, "a"), (8962, "b"))
            ],
        ),
    )
    atomic_json(path, plan)
    return plan


def request_for(plan: Json, spec: Json) -> Json:
    """Ordinary service parses only its confidence features and receives no guard or labels."""
    stream, arm = plan["streams"][spec["stream_index"]], spec["arm"]
    online = arm in ("bias_only", "adaptive")
    rows = (
        [json.dumps(json.loads(r)["features"]) for r in stream["rows"]]
        if arm == "confidence_logistic"
        else stream["rows"]
    )
    return dict(
        arm=arm,
        seed=spec["seed"],
        unit_ids=stream["unit_ids"],
        rows=rows,
        service_path=plan["services"][arm],
        label_path=stream["label_reference"]["path"] if online else None,
        guard_rows=plan["guard_rows"] if online else [],
        guard_labels=plan["guard_labels"] if online else [],
    )


def child(argv: list[str], log: Path, timeout: float = 120) -> Json:
    """wait4 supplies parent-observed RSS and exit status without component timers."""
    log.parent.mkdir(parents=True, exist_ok=True)
    started, heartbeat, timed_out = time.monotonic_ns(), time.monotonic(), False
    env = dict(
        os.environ,
        PYTHONPATH=f"{ROOT / 'python'}:{ROOT}",
        JAX_PLATFORMS="cpu",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    with (
        log.open("wb") as output,
        subprocess.Popen(
            argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
        ) as process,
    ):
        assert process.stdout is not None
        while True:
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                data = os.read(process.stdout.fileno(), 65536)
                output.write(data)
                print(data.decode(errors="replace"), end="", flush=True)
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                tail = process.stdout.read()
                output.write(tail)
                print(tail.decode(errors="replace"), end="", flush=True)
                break
            if (time.monotonic_ns() - started) / 1e9 > timeout:
                process.kill()
                timed_out = True
            if time.monotonic() - heartbeat >= 15:
                progress("child_pending", 0, 1)
                heartbeat = time.monotonic()
    return dict(
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        pid=process.pid,
        exit_code=process.returncode,
        timed_out=timed_out,
        passed=process.returncode == 0 and not timed_out,
        peak_memory_bytes=usage.ru_maxrss * 1024,
        user_cpu_s=usage.ru_utime,
        system_cpu_s=usage.ru_stime,
        log=ref(log),
    )


def execute(raw: Path, spec: Json, plan: Json) -> Json:
    """Two fresh children include startup, exit and the durable query16 handoff."""
    atomic_json(raw / "request.json", request_for(plan, spec))
    commands = []
    for stop in (16, 32):
        progress("before_service_child", 0, 1)
        receipt = child(
            [
                sys.executable,
                "-u",
                str(ROOT / CLI),
                "--segment",
                str(raw / "request.json"),
                "--stop",
                str(stop),
            ],
            raw / f"child-{stop}.log",
        )
        commands.append(receipt)
        check(receipt["passed"], "service_child_failure")
        progress("after_service_child", 1, 0)
    segments = [json.loads((raw / f"segment-{stop}.json").read_bytes()) for stop in (16, 32)]
    cell = dict(spec=spec, commands=commands, segments=segments)
    verify(cell)
    atomic_json(raw / "unit.json", cell)
    return cell


def verify(cell: Json) -> None:
    """Retained low-level states bind restart, operation counts and complete responses."""
    first, last = cell["segments"]
    state, arm = last["state"], cell["spec"]["arm"]
    online = arm in ("bias_only", "adaptive")
    check(
        first["state"] == last["initial_state"]
        and first["state"]["query"] == 16
        and first["pid"] != last["pid"],
        "restart_parity",
    )
    check(state["query"] == len(state["rows"]) == 32 and not state["pending"], "complete_responses")
    check(
        state["counts"]["updates"] == (8 if online else 0)
        and state["counts"]["guard_evaluations"] == (1024 if online else 0)
        and state["counts"]["graph_encoded_rows"]
        == (288 if online else 0 if arm == "confidence_logistic" else 96),
        "operation_counts",
    )
    check(
        all(
            c["passed"]
            and c["ended_monotonic_ns"] > c["started_monotonic_ns"]
            and c["peak_memory_bytes"] > 0
            and c["pid"] == s["pid"]
            for c, s in zip(cell["commands"], cell["segments"], strict=True)
        ),
        "parent_process_receipts",
    )
    check(
        all(
            s["storage"]["file_fsync"] == s["storage"]["directory_fsync"] > 0
            for s in cell["segments"]
        ),
        "durability",
    )


def hardware() -> list[Json]:
    """Candidate maps require measured transfers, parity and a complete-service crossover."""
    return [
        dict(
            operation=op,
            cpu_today=cpu,
            rust_simd=rust,
            gpu_npu_batch=gpu,
            break_even_evidence=needed,
            estimated_speedup=None,
        )
        for op, cpu, rust, gpu, needed in (
            (
                "graph_json_parse_and_pack",
                "JSON and anonymous60-node adjacency on CPU",
                "bounded parse and vector packing",
                "host parse; packed adjacency transfer",
                "measure parse plus copy versus batch size; include padding and allocation",
            ),
            (
                "graph_encoder_and_energy_score",
                "original three-seed JAX CPU scorer; three width8 rounds",
                "float64 matrix kernels after parity qualification",
                "batch graph message matrices and energy heads; weights resident",
                "same precision and probabilities; transfer adjacency/features/weights in, scores out; measure kernel launch and CPU crossover with batching delay",
            ),
            (
                "confidence_logistic",
                "two standardized scalar features, three frozen heads",
                "scalar dot products or SIMD",
                "only if an existing resident batch amortizes launches",
                "include two-feature upload and probability download; compare complete request latency",
            ),
            (
                "scalar_calibration_and_guard",
                "four-label bounded SGD;64 guard rows and baseline comparisons",
                "scalar arithmetic and vector guard probabilities",
                "batch independent adapters and guard vectors",
                "include state/feedback transfers and synchronization; preserve causal delays and exact promotion decisions",
            ),
            (
                "feedback_serialization_fsync_restart",
                "sealed broker and file/directory barriers on host",
                "CPU service boundary; durable semantics unchanged",
                "host work; cannot offload fsync to neural kernels",
                "measure unchanged crash recovery, bytes and barriers inside full process costs",
            ),
        )
    ] + [
        dict(
            operation="hardware_limit",
            device=device,
            qualification=status,
            sampling_call_count=0,
            measured_speedup=None,
        )
        for device, status in (
            (
                "KV260",
                "qualified quadratic sampling only; k_max<=5; no neural-service implementation",
            ),
            ("PolarFire", "CPU dispatch only"),
            ("GateMate", "blocked at0xffffffff"),
            ("TSU", "unqualified locally"),
            ("NPU", "unqualified locally"),
        )
    ]


def reduce(raw: Path) -> Json:
    """Reconstruct all96 scheduled units from parent clocks, never from headline values."""
    rows, processes, durability = [], [], []
    totals: Json = {}
    for spec in roster():
        directory = raw / spec["cluster_id"] / spec["arm"]
        path, failed = directory / "unit.json", directory / "failure.json"
        cell = json.loads(path.read_bytes()) if path.is_file() else None
        status = "completed" if cell else "failed" if failed.is_file() else "censored"
        row = dict(
            **spec,
            metric="complete_process_s",
            value=None,
            numerator=None,
            denominator=1,
            request_denominator=32,
            condition="cached_Qwen_response_service",
            status=status,
            censor_reason=None
            if cell
            else json.loads(failed.read_bytes())["reason"]
            if failed.is_file()
            else "external_or_unfinished",
        )
        if cell:
            check(cell["spec"] == spec, "cell_identity")
            verify(cell)
            duration = (
                cell["commands"][-1]["ended_monotonic_ns"]
                - cell["commands"][0]["started_monotonic_ns"]
            ) / 1e9
            storage = {
                k: sum(s["storage"][k] for s in cell["segments"])
                for k in ("durable_bytes", "file_fsync", "directory_fsync")
            }
            op = dict(cell["segments"][-1]["state"]["counts"], **storage)
            row.update(
                value=duration,
                numerator=duration,
                peak_memory_bytes=max(c["peak_memory_bytes"] for c in cell["commands"]),
                durable_bytes=storage["durable_bytes"],
                response_count=len(cell["segments"][-1]["state"]["rows"]),
            )
            processes.append(
                dict(
                    **spec,
                    commands=cell["commands"],
                    operations=op,
                    response_count=32,
                    peak_memory_bytes=row["peak_memory_bytes"],
                    parent_bounds_scope="first startup through second exit, including restart handoff; validation excluded",
                )
            )
            durability.append(
                dict(
                    **spec,
                    passed=True,
                    restart_query=16,
                    final_pending=[],
                    state_hash=canonical_hash(cell["segments"][-1]["state"]),
                    **storage,
                )
            )
            for k, v in op.items():
                totals[k] = totals.get(k, 0) + v
        rows.append(row)
    summaries, intervals = [], []
    for arm in w.ARMS:
        for metric in ("complete_process_s", "peak_memory_bytes", "durable_bytes"):
            samples = [
                r["value" if metric == "complete_process_s" else metric]
                for r in rows
                if r["arm"] == arm and r["status"] == "completed"
            ]
            summaries.append(
                dict(
                    arm=arm,
                    metric=metric,
                    independent_count=len(samples),
                    p50=float(np.median(samples)) if samples else None,
                    p95=float(np.quantile(samples, 0.95)) if samples else None,
                )
            )
    lookup = {(r["cluster_id"], r["arm"]): r["value"] for r in rows if r["status"] == "completed"}
    for arm in w.ARMS[1:]:
        deltas = [
            lookup[c, arm] - lookup[c, w.ARMS[0]]
            for c in sorted({r["cluster_id"] for r in rows})
            if (c, arm) in lookup and (c, w.ARMS[0]) in lookup
        ]
        n = len(deltas)
        mean = float(np.mean(deltas)) if n else None
        margin = (
            float(t.ppf(1 - 0.05 / 6, n - 1) * np.std(deltas, ddof=1) / n**0.5) if n > 1 else None
        )
        intervals.append(
            dict(
                arm=arm,
                against=w.ARMS[0],
                metric="complete_process_s",
                independent_count=n,
                paired_differences=deltas,
                mean=mean,
                lower=mean - margin if margin is not None and mean is not None else None,
                upper=mean + margin if margin is not None and mean is not None else None,
                method="paired Student t;95-percent family interval; Bonferroni three comparisons; finite cached workloads",
            )
        )
    return dict(
        rows=rows,
        service_cost_rows=summaries,
        paired_cost_intervals=intervals,
        process_seed_rows=processes,
        durability_rows=durability,
        operation_counts=totals,
        numerical_training_calls=totals.get("updates", 0),
        intended_count=96,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=sum(
            sum(p["cluster_id"] == c for p in processes) == 4
            for c in {s["cluster_id"] for s in roster()}
        ),
        sampling_call_count=0,
        removable_fraction=None,
        service_100x_projection=None,
        hardware_compatibility_rows=hardware(),
    )


def empty_result() -> Json:
    """Missing external evidence keeps every scheduled process visible without timings."""
    return reduce(Path("/nonexistent/carnot8968"))


def measure(evidence: Json, raw: Path) -> Json:
    """Every scheduled failure remains retained; a failed owned process disqualifies."""
    plan = prepare(evidence, raw)
    started = time.monotonic()
    for i, spec in enumerate(plan["roster"]):
        check(time.monotonic() - started < 2400, "measurement_budget")
        progress("before_paired_cell", i, 96 - i)
        directory = raw / spec["cluster_id"] / spec["arm"]
        try:
            execute(directory, spec, plan)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            atomic_json(directory / "failure.json", dict(reason=str(error), spec=spec))
        progress("after_paired_cell", i + 1, 95 - i)
    return reduce(raw)


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Cost readiness needs complete valid measurements; benefit gates only deployment."""
    result, failures = work["result"], work["evidence"]["failures"]
    owned = upstream.prior.admission.shard.qualified(
        dict(
            validation,
            receipts=[
                r for r in validation["receipts"] if r["name"] in upstream.calibration.CHECKS
            ],
        ),
        OWNED,
    ) and all(r["passed"] for r in validation["receipts"])
    complete = (
        result["completed_count"] == 96
        and result["independent_count"] == 24
        and len(result["durability_rows"]) == 96
    )
    cls = (
        "disqualified"
        if not fixture and (not owned or work["owned_failure"] or not failures and not complete)
        else "blocked"
        if failures
        else "null"
    )
    ready = int(owned and complete and not failures and not fixture)
    raw = Path(work["raw"])
    plan = prepare(None, raw) if (raw / "plan.json").exists() else {}
    quality = plan.get("quality", {})
    comparison = dict(
        arm="adaptive",
        against="confidence_logistic",
        favorable=any(
            r["arm"] == "adaptive"
            and r["independent_count"] == 24
            and r["upper"] is not None
            and r["upper"] < 0
            for r in result["paired_cost_intervals"]
        ),
        criterion="simultaneous paired complete-process upper<0 plus all three authenticated quality gates=1; no other tradeoff authorized",
    )
    recommend = bool(
        ready
        and comparison["favorable"]
        and all(
            type(quality.get(k)) is int and quality[k] == 1
            for k in (
                "detector_benefit_score",
                "relational_controls_ready_score",
                "online_benefit_score",
            )
        )
    )
    verdict = f"complete_{cls}_cached_relational_complete_service_cost"
    previous = work.get("prior_terminal", {})
    if (
        cls in ("blocked", "disqualified")
        and previous.get("verdict_class") == cls
        and previous.get("gate_check_summary") == failures
    ):
        verdict = previous["honest_verdict"]
    value = dict(
        result,
        experiment_id=8968,
        task_id=TASK,
        milestone="2026.11.766",
        run_date="20261110",
        random_seed=SEED,
        honest_verdict=verdict,
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if failures else "success",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        duration_s=(work["ended"] - work["started"]) / 1e9,
        phase_spans=work["phase_spans"],
        preconditions_checked=dict(historical_tasks_are_retained_inputs=True, failures=failures),
        verifier_is_oracle=False,
        exposure_scope="cached real Qwen graph responses; executable-oracle labels enter delayed feedback and exposed development guards only",
        independent_generalization_score=0,
        gate_check_summary=failures,
        source_artifact_hashes=work["evidence"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[ref(raw / "work.json"), *work["raw_references"]],
        required_checks_passed=owned,
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        cost_rows_ready_score=ready,
        quality_reference=quality,
        complete_cost_quality_comparison=comparison,
        deployment_recommendation=recommend,
        upstream_generation_cost=plan.get("upstream_generation", {}),
        fixture_only=fixture,
        flagged_adversarial=False,
        clock_contract=dict(
            clock="parent time.monotonic_ns before Popen through wait4 exit; includes both cold interpreter starts and restart handoff",
            memory="parent wait4 ru_maxrss; max across both segments",
            observer_subtraction=False,
            component_timing=False,
            observer_scope="state counters, output forwarding and segment diagnostic serialization are charged; no component saving inferred",
        ),
        methodology_note="24 fresh paired process seeds with randomized balanced arm order;32 requests per service. Frozen real-response corpus and three-seed heads. Confidence logistic parses only two features; no graph work, updates or guards. Frozen relational has no feedback or guard. Online services pay initial64-graph guard encoding, seven delayed updates plus final four-label drain, exact query16 restart and file/directory barriers. Parent startup-through-exit clocks and wait4 memory. Generation and historical training are retained evidence only. No sampler or board operation. Shared corpus bounds independent generalization.",
        later_hardware_experiment_evidence=[
            "Reproduce probabilities, promotions and durable restart semantics at the proposed precision.",
            "Measure batch-size crossover including graph packing, host/device transfers, synchronization and queueing latency.",
            "Retain complete paired CPU/device service clocks, memory, energy and quality at the same request load.",
            "Qualify a neural kernel on the actual device before any removable fraction; existing sampler qualification is insufficient.",
        ],
    )
    value["acceptance_gates"] = {
        k: dict(
            passed=v,
            principle="Authenticate bytes and complete process units; owned failures disqualify. Cost readiness is independent of quality gain.",
        )
        for k, v in dict(
            authenticated_inputs=not failures, complete_costs=complete, owned_checks=owned
        ).items()
    }
    value["field_principles"] = {
        k: "Actual scoped receipt or authenticated retained operand; never an inferred hardware saving."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    for fields, principle in (
        (
            "experiment_id task_id milestone run_date random_seed reproducibility_checksum",
            "Bind the result to the scheduled task, input bytes and reproducible randomness.",
        ),
        (
            "honest_verdict verdict_class",
            "An honest terminal disposition avoids wasted retries; failed gates and oracle inference forbid positive.",
        ),
        (
            "inference_substrate inference_substrate_class execution_venue MODEL_SPECS model_specs model_invocation_counts numerical_training_calls",
            "Separate small scalar training, retained Qwen outputs and zero actual generator loads or calls; host is the venue.",
        ),
        (
            "duration_s phase_spans preconditions_checked",
            "Actual clocks and authenticated resources prevent fabricated execution or duration padding.",
        ),
        (
            "verifier_is_oracle exposure_scope independent_generalization_score",
            "Exact labels are feedback and evaluation only; graph service costs establish no oracle-free generalization.",
        ),
        (
            "intended_count completed_count failed_count censored_count excluded_count independent_count rows",
            "Retain96 scheduled process units and typed operands;32 requests and three scoring seeds never enlarge independent n=24.",
        ),
        (
            "gate_check_summary",
            "Name upstream, path/hash, check, exact field, expected and observed values for every external refusal.",
        ),
        (
            "source_artifact_hashes code_config_hashes raw_shard_hashes field_principles",
            "Exact input, code and raw byte custody explains and binds each field.",
        ),
        (
            "acceptance_gates required_checks_passed owned_statement_coverage validation_receipts global_test_debt",
            "Require every owned gate and100-percent statements with zero exclusions; global debt cannot waive owned failures.",
        ),
        (
            "cost_rows_ready_score",
            "Complete process timings can qualify independently of scientific benefit.",
        ),
        (
            "service_cost_rows paired_cost_intervals operation_counts durability_rows hardware_compatibility_rows",
            "Hardware feasibility applies only to real counted service work; paired units and durable bytes support complete costs.",
        ),
        (
            "sampling_call_count removable_fraction service_100x_projection deployment_recommendation",
            "Zero sampling leaves removable fraction and100x projection null; deployment requires all three benefits and favorable complete costs.",
        ),
    ):
        value["field_principles"].update({k: principle for k in fields.split()})
    value["reproducibility_checksum"] = upstream.calibration.checksum(value)
    return value
