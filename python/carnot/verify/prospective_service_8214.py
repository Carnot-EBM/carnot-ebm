"""REQ-VERIFY-8214: measure one designed schedule through qualified services.

The generator runs once. Charging its observed work to two service arms is an
accounting comparison, and does not measure two independent deployment runs.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from carnot.verify import complete_request_8174 as prior
from carnot.verify import request_recorder_8213 as recorder
from carnot.reporting import recorder_execution_8213 as qualified
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.request_trace_inventory_8200 import copy_bytes, operand

Json = dict[str, Any]
ROOT = recorder.ROOT
NAME = "experiment_8214_v709_prospective_service_measurement"
TASK = "exp8214-prospective-service-measurement"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_prospective_service_measurement_8214.py"
MODULE = "python/carnot/verify/prospective_service_8214.py"
RUNNER = "python/carnot/reporting/prospective_service_execution_8214.py"
OWNED = [MODULE, RUNNER, CLI]
ARMS = prior.ARMS
SEED = recorder.SEED
MODEL_SPECS = [
    dict(
        hf_id="unsloth/Qwen3.8-27B-GGUF",
        quantization="Q4_K_M",
        role="headline",
        backend="llama.cpp CUDA",
    )
]
CONFIG = dict(
    seed=SEED,
    requests=24,
    max_tokens=128,
    call_timeout_s=120,
    deadline_s=3600,
    duration_floor_s=10,
    cache_enabled=False,
)
progress, atomic_json, reference = recorder.progress, prior.atomic_json, prior.reference
sha256_file, key, checksum = prior.sha256_file, recorder.key, qualified.checksum


def inputs(root: Path, raw: Path) -> Json:
    """Bind recorder readiness and the exact schedule instead of selecting new slots."""
    data = qualified.inputs(root, raw)
    path = root / "results/experiment_8213_v709_prospective_request_recorder.json"
    data["checks"].append(operand("recorder_exists", path, True, path.is_file()))
    try:
        if path.is_file():
            upstream = json.loads(path.read_text())
            data["refs"].append(copy_bytes(path, raw))
            data["checks"].append(
                operand(
                    "request_recorder_ready_score",
                    path,
                    1,
                    upstream.get("request_recorder_ready_score"),
                )
            )
            ref = upstream["frozen_exp8214_configuration"]
            configuration = Path(ref["path"])
            data["checks"].append(
                operand(
                    "configuration_sha256", configuration, ref["sha256"], sha256_file(configuration)
                )
            )
            data["refs"].append(copy_bytes(configuration, raw))
            frozen = json.loads(configuration.read_text())
            schedule = Path(frozen["schedule"]["path"])
            data["checks"].append(
                operand(
                    "schedule_sha256", schedule, frozen["schedule"]["sha256"], sha256_file(schedule)
                )
            )
            data["refs"].append(copy_bytes(schedule, raw))
            data["checks"].append(
                operand(
                    "schedule_equality",
                    schedule,
                    True,
                    json.loads(schedule.read_text()) == data["schedule"],
                )
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        data["checks"].append(
            operand(
                "recorder_operand_structure",
                path,
                "authenticated schedule/configuration",
                str(error),
            )
        )
    data["ready"] = data["ready"] and all(c["passed"] for c in data["checks"])
    atomic_json(Path(data["input_path"]), data)
    return data


class FixtureRuntime:
    """Private scripted responses test storage and errors with zero model credit."""

    def __init__(self, mode: str = "valid"):
        self.mode = mode

    def generate(self, payload: Json) -> Json:
        """Use the requested sentence address; fixtures never claim semantic truth."""
        if self.mode == "error":
            raise RuntimeError("scripted_generation_error")
        index = json.loads(payload["messages"][-1]["content"])["answer_sentence_indices"][0]
        text = "malformed" if self.mode == "malformed" else f"{index}|E|0.10|[0]"
        return dict(
            choices=[
                dict(
                    message=dict(content=text),
                    finish_reason="length" if self.mode == "truncated" else "stop",
                )
            ],
            usage=dict(prompt_tokens=1, completion_tokens=12),
            fixture=True,
        )


def convert(slot: Json, response: Json, family: str) -> list[float]:
    """Reuse the sentence parser and lexical features; invalid bytes stay invalid."""
    body = json.loads(slot["envelope"]["payload"]["messages"][-1]["content"])
    parsed = recorder.transport.parse(
        response["choices"][0]["message"]["content"].strip(),
        body["answer_sentence_indices"],
        len(body["source_segments"]),
    )
    if (
        parsed["status"] != "completed"
        or response["choices"][0]["finish_reason"] != "stop"
        or response["usage"]["completion_tokens"] > 128
    ):
        raise ValueError("invalid_or_truncated_completion")
    lexical = prior.shared.prior.stream.lexical.extract(
        dict(
            family_id=family,
            source_bytes=slot["envelope"]["source_bytes"],
            answer_bytes=slot["envelope"]["answer_bytes"],
        )
    )
    p = min(1 - 1e-6, max(1e-6, parsed["rows"][0]["p_unsupported"]))
    return [math.log(p / (1 - p)), *lexical["values"]]


def measure(data: Json, runtime: Any, raw: Path, *, deadline_s: float = 3600) -> Json:
    """Measure all original slots once and counterbalance matched fresh stores."""
    raw.mkdir(parents=True, exist_ok=True)
    began = time.monotonic_ns()
    native, binding = prior.shared.host.old.prior.old.host.load_binding(data)
    startup = (time.monotonic_ns() - began) / 1e9
    (raw / "events.jsonl").touch()
    journal = recorder.Journal(raw / "events.jsonl")
    rng = random.Random(SEED)
    families = {r["source_cluster_id"]: r["family_id"] for r in data["roster"]}
    rows = []
    for i, slot in enumerate(data["schedule"]["rows"]):
        progress("8214_request", i, 24 - i)
        row = dict(deepcopy(slot), arms=[], status="censored", acquisition_ns=0)
        if (time.monotonic_ns() - began) / 1e9 >= deadline_s:
            rows.append(row)
            continue
        acquisition_start = time.monotonic_ns()
        terminal = recorder.invoke(journal, runtime, slot["request_id"], slot["envelope"])
        row.update(
            terminal=terminal,
            acquisition_start_ns=acquisition_start,
            acquisition_end_ns=time.monotonic_ns(),
            status="failed",
        )
        row["acquisition_ns"] = row["acquisition_end_ns"] - acquisition_start
        order = list(ARMS)
        rng.shuffle(order)
        row["branch_order"] = order
        row["native_parity_failed"] = False
        if terminal["status"] == "completed":
            for arm in order:
                branch = dict(
                    unit_id=slot["request_id"],
                    source_cluster_id=slot["source_cluster_id"],
                    arm=arm,
                    request=slot["envelope"]["payload"],
                )
                start = time.monotonic_ns()
                try:
                    branch["values"] = convert(
                        slot, terminal["result"], families[slot["source_cluster_id"]]
                    )
                    converted = time.monotonic_ns()
                    path = raw / slot["request_id"] / arm
                    atomic_json(
                        path / "pending.json",
                        dict(
                            request_id=slot["request_id"],
                            semantic_key=terminal["semantic_key"],
                            values=branch["values"],
                        ),
                    )
                    pending = time.monotonic_ns()
                    branch["arrival_ns"] = start
                    prior.commit_group(data, [branch], native, path)
                    read_start = time.monotonic_ns()
                    state = prior.shared.host.Store(path / "store.json", key(data["head"])).state
                    end = time.monotonic_ns()
                    branch.update(
                        state=state,
                        start_ns=start,
                        conversion_end_ns=converted,
                        pending_end_ns=pending,
                        readout_start_ns=read_start,
                        end_ns=end,
                    )
                except (ValueError, KeyError, TypeError) as error:
                    branch.update(
                        status="failed",
                        error=str(error),
                        start_ns=start,
                        end_ns=time.monotonic_ns(),
                    )
                row["arms"].append(branch)
            if pair_ok(row):
                row["status"] = "completed"
            elif all(b["status"] == "completed" for b in row["arms"]):
                row["native_parity_failed"] = True
        rows.append(row)
        atomic_json(raw / "checkpoint.json", dict(completed=i + 1, pending=23 - i))
        progress("8214_request_after", i + 1, 23 - i)
    return dict(
        requests=rows,
        startup_s=startup,
        binding=binding,
        journal=reference(journal.path),
        start_ns=began,
        end_ns=time.monotonic_ns(),
    )


def pair_ok(row: Json) -> bool:
    """Compare reopened typed state with the established probability tolerance."""
    arms = row["arms"]
    if len(arms) != 2 or any(a["status"] != "completed" for a in arms):
        return False
    left, right = arms
    records = [deepcopy(a["state"]["records"][0]) for a in arms]
    probabilities = [r.pop("probability") for r in records]
    return bool(
        abs(left["probability"] - right["probability"]) <= 1e-10
        and abs(probabilities[0] - probabilities[1]) <= 1e-10
        and records[0] == records[1]
    )


def stages(branch: Json) -> Json:
    """Partition observed downstream spans without adding nested clocks twice."""
    fields = dict(
        conversion=("start_ns", "conversion_end_ns"),
        pending_state=("conversion_end_ns", "pending_end_ns"),
        scoring=("boundary_crossing_start_ns", "boundary_crossing_end_ns"),
        serialization=("serialization_start_ns", "serialization_end_ns"),
        commit_fsync=("commit_start_ns", "commit_end_ns"),
        readout=("readout_start_ns", "end_ns"),
    )
    costs = {k + "_ns": branch[b] - branch[a] for k, (a, b) in fields.items()}
    costs["other_host_ns"] = branch["end_ns"] - branch["start_ns"] - sum(costs.values())
    return dict(request_id=branch["unit_id"], arm=branch["arm"], **costs)


def reduce(work: Json) -> Json:
    """Bootstrap distinct matched sources and retain every failed acquisition cost."""
    rows = work.get("requests", [])
    pairs = [r for r in rows if r["status"] == "completed" and pair_ok(r)]
    acquisition = sum(r["acquisition_ns"] for r in rows) / 1e9
    startup = work.get("startup_s", 0)
    wall = (work.get("end_ns", 0) - work.get("start_ns", 0)) / 1e9
    downstream = sum((b["end_ns"] - b["start_ns"]) / 1e9 for r in rows for b in r["arms"])
    overhead = max(0, wall - work.get("service_startup_s", startup) - acquisition - downstream)
    totals = {
        a: startup
        + acquisition
        + overhead
        + sum((b["end_ns"] - b["start_ns"]) / 1e9 for r in rows for b in r["arms"] if b["arm"] == a)
        for a in ARMS
    }
    ci = None
    if len(pairs) >= 20:
        logs = []
        for row in pairs:
            arms = {b["arm"]: b for b in row["arms"]}
            costs = [row["acquisition_ns"] + arms[a]["end_ns"] - arms[a]["start_ns"] for a in ARMS]
            logs.append(math.log(costs[0] / costs[1]))
        draws = np.random.default_rng(SEED).choice(logs, (10000, len(logs))).mean(axis=1)
        ci = dict(
            estimate=float(np.exp(np.mean(logs))),
            lower95=float(np.exp(np.quantile(draws, 0.025))),
            upper95=float(np.exp(np.quantile(draws, 0.975))),
            sources=len(pairs),
            resamples=10000,
            scope="matched shared acquisition; excludes once-per-workload startup",
        )
    cost_rows = [stages(b) for r in pairs for b in r["arms"]]
    scoring = sum(c["scoring_ns"] for c in cost_rows if c["arm"] == ARMS[0]) / 1e9
    baseline = totals[ARMS[0]]
    return dict(
        completed_count=len(pairs),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=24 - len(rows) + sum(r["status"] == "excluded" for r in rows),
        independent_count=len({r["source_cluster_id"] for r in pairs}),
        paired_ratio_ci95=ci,
        complete_workload_totals=totals,
        stage_cost_rows=cost_rows,
        cold_start_cost_s=startup,
        measured_acquisition_s=acquisition,
        all24_slot_wall_s=wall,
        shared_recording_and_checkpoint_overhead_s=overhead,
        amdahl_ceiling=dict(
            scoring_share=scoring / baseline,
            maximum_speedup=baseline / (baseline - scoring),
            scope="remove measured Python scoring only; retain acquisition and storage",
        )
        if baseline > scoring
        else None,
    )


def validate_work(data: Json, work: Json) -> bool:
    """Reparse current responses and recompute native decisions without inference."""
    journal = recorder.Journal(Path(work["journal"]["path"]))
    native, _ = prior.shared.host.old.prior.old.host.load_binding(data)
    families = {r["source_cluster_id"]: r["family_id"] for r in data["roster"]}
    terminals = {r["request_id"]: r for r in journal.events if r["event"] == "terminal"}
    for row, slot in zip(work["requests"], data["schedule"]["rows"], strict=True):
        if row["envelope"] != slot["envelope"] or row["request_id"] != slot["request_id"]:
            return False
        if row["status"] == "censored":
            continue
        if row["acquisition_ns"] != row["acquisition_end_ns"] - row["acquisition_start_ns"]:
            return False
        if row["terminal"] != terminals[row["request_id"]]:
            return False
        if row["status"] == "completed":
            values = convert(slot, row["terminal"]["result"], families[row["source_cluster_id"]])
            for branch in row["arms"]:
                expected = prior.shared.host.old.score(
                    data["head"],
                    data["geometry"],
                    [values],
                    native,
                    "native_batch" if branch["arm"] == ARMS[1] else "python_batch",
                )[0]
                if (
                    abs(float(expected) - branch["probability"]) > 1e-10
                    or branch["values"] != values
                    or prior.shared.host.Store(
                        Path(branch["store"]["path"]), key(data["head"])
                    ).state
                    != branch["state"]
                    or sha256_file(Path(branch["store"]["path"])) != branch["store"]["sha256"]
                    or any(v < 0 for k, v in stages(branch).items() if k.endswith("_ns"))
                ):
                    return False
            if not pair_ok(row):
                return False
    return not journal.pending


def replay(path: Path) -> bool:
    """Cold replay authenticates frozen bytes before reconstructing headline fields."""
    try:
        value = json.loads(path.read_text())
        if value["reproducibility_checksum"] != checksum(value):
            return False
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            if sha256_file(Path(ref.get("frozen_path", ref["path"]))) != ref["sha256"]:
                return False
        data = json.loads(Path(value["input_path"]).read_text())
        result = json.loads(Path(value["result_path"]).read_text())
        if result.get("work") and not validate_work(data, result["work"]):
            return False
        expected = build(
            data,
            result,
            Path(value["raw_path"]),
            value["validation_receipts"],
            value["duration_s"],
            value["fixture"],
        )
        return all(
            value[k] == expected[k]
            for k in [
                "rows",
                "completed_count",
                "failed_count",
                "independent_count",
                "censored_count",
                "complete_workload_totals",
                "stage_cost_rows",
                "paired_ratio_ci95",
                "amdahl_ceiling",
                "verdict_class",
                "service_measurement_ready_score",
            ]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def build(
    data: Json, result: Json, raw: Path, receipts: list[Json], duration: float, fixture: bool
) -> Json:
    """Readiness describes current evidence completeness, never a deployment benefit."""
    work = result.get("work", {})
    if not work:
        slots = data["schedule"].get("rows", []) or [
            dict(request_id=f"prospective-{i + 1:02}", source_cluster_id=None) for i in range(24)
        ]
        work = dict(
            requests=[
                dict(
                    deepcopy(s),
                    status="excluded",
                    arms=[],
                    acquisition_ns=0,
                    exclusion_reason="not_started_required_operand_or_owned_check",
                )
                for s in slots
            ]
        )
    reduced = reduce(work)
    checks = [*data["checks"], *result.get("checks", [])]
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    resource = data["ready"] and all(c["passed"] for c in checks)
    live = not fixture and result.get("model_loads_completed", 0) == 1
    generation_s = sum(
        (r["ended_monotonic_ns"] - r["started_monotonic_ns"]) / 1e9
        for r in result.get("runtime_receipts", [])
    )
    support = reduced["independent_count"] >= 20
    checks.extend(
        [
            operand(
                "independent_count", raw / "result.json", 20, reduced["independent_count"], op=">="
            ),
            operand("measured_generation_s", raw / "result.json", 10, generation_s, op=">="),
        ]
    )
    parity = not any(r.get("native_parity_failed") for r in work.get("requests", []))
    ready = resource and owned and live and support and generation_s >= 10 and parity
    cls, verdict = "null", "complete_null_shared_acquisition_accounting"
    if not resource:
        cls, verdict = "blocked", "complete_blocked_required_operand"
    elif not support or (not fixture and generation_s < 10):
        cls, verdict = "blocked", "complete_blocked_support"
    if not owned or not parity:
        cls, verdict = "disqualified", "complete_disqualified_owned_validation"
    requests = work.get("requests", [])
    rows = [
        dict(
            source_cluster_id=s["source_cluster_id"],
            request_id=s["request_id"],
            seed=SEED,
            condition="original",
            status=s["status"],
            metric="matched_complete_latency_ns",
            numerator=sum(b["end_ns"] - b["start_ns"] for b in s["arms"]),
            denominator=1,
        )
        for s in requests
    ]
    counts = dict(ZERO_INVOCATION_COUNTS)
    if not fixture:
        counts.update(
            model_loads_attempted=result.get("model_loads_attempted", 0),
            model_loads_completed=result.get("model_loads_completed", 0),
            generation_calls_attempted=sum("terminal" in r for r in requests),
            generation_calls_completed=sum(
                r.get("terminal", {}).get("status") == "completed" for r in requests
            ),
            generation_calls_failed=sum(
                r.get("terminal", {}).get("status") == "error" for r in requests
            ),
        )
    value = dict(
        experiment_id=8214,
        experiment=8214,
        task_id=TASK,
        milestone="2026.10.709",
        run_date="20261006",
        title="Prospective bounded Qwen and matched durable service costs",
        honest_verdict=verdict,
        verdict_class=cls,
        status="completed",
        fixture=fixture,
        schema="carnot.prospective_service_measurement.v1",
        inference_substrate="live_llm_inference" if live else "aggregation_from_upstream_artifacts",
        inference_substrate_class="model_bounded_generation" if live else "no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_specs=MODEL_SPECS,
        model_invocation_counts=counts,
        preconditions_checked=True,
        duration_s=duration,
        random_seed=SEED,
        source_artifact_hashes=data["refs"],
        cited_upstream_artifacts=data["refs"],
        rows=rows,
        intended_count=24,
        **reduced,
        verifier_is_oracle=False,
        exposure_scope="exposed fit development; designed research requests",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        gate_check_summary=checks,
        acceptance_gates=dict(
            required_inputs=resource,
            owned_validation=owned,
            native_parity=parity,
            current_cuda_work=live,
            independent_pairs=support,
            bounded_generation_floor=generation_s >= 10,
        ),
        service_measurement_ready_score=int(ready),
        validation_receipts=receipts,
        required_checks_passed=owned and parity,
        raw_shard_hashes=[],
        code_config_hashes=[],
        request_rows=requests,
        native_extension_sha256=work.get("binding", {}).get("sha256"),
        measured_gpu_provenance={
            k: result.get(k)
            for k in [
                "gpu_lease_receipt",
                "model_identity_receipt",
                "resident_gpu_receipt",
                "resolved_library",
                "cleanup",
                "server_log",
            ]
        },
        shared_acquisition_accounting=dict(
            scope="matched shared-acquisition counterfactual accounting",
            independent_generative_executions=False,
            end_to_end_deployment_speedup=False,
            startup_charged_once_per_arm=True,
            same_actual_acquisition_charged_to_each_arm=True,
        ),
        workload_scope="designed_research_requests",
        deployment_demand_observed=False,
        measured_repeat_frequency=None,
        designed_exact_repetitions=sum("envelope" in r for r in requests)
        - len({key(r["envelope"]) for r in requests if "envelope" in r}),
        nfr01_met=False,
        nfr01_threshold=10,
        cache_enabled=False,
        measured_generation_s=generation_s,
        config=CONFIG,
        input_path=data["input_path"],
        raw_path=str(raw),
        result_path=str(raw / "result.json"),
        benchmark_performed=bool(requests),
        flagged_adversarial=False,
        methodology="24 fixed sources once; fsynced issue before generation; seeded branch order; "
        "fresh stores; paired shared acquisition accounting; startup once per workload; "
        "all attempted slots retained; no natural repeat-frequency or deployment estimate.",
    )
    value["field_principles"] = {
        k: "Bind reported state to current primitive bytes; historical provenance is not current work."
        for k in value
    }
    return value
