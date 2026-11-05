"""REQ-REPORT-8851: map paid CPU operations without inventing FPGA acceleration."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import hardware_operation_budget_8838 as historical
from carnot.reporting import hardware_operation_map_8825 as board_history
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8851_v757_hardware_operation_budget"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/hardware_operation_budget_8851.py",
    "python/carnot/reporting/hardware_budget_validation_8851.py",
    CLI,
]
TEST = "tests/python/test_hardware_operation_budget_8851.py"
DEFAULT_OUTPUT = ROOT / "results" / (NAME + ".json")
AUTHORITIES = ROOT / "results/raw" / NAME / "authorities"
EXTERNAL = {
    "anechoic": (
        "https://github.com/skethz/Anechoic",
        "1f20dde5bedbe9919b95f210d53c3cc2aa6f60411ca48aa9f4bc91a1b2abe855",
    ),
    "extropic": (
        "https://extropic.ai/writing/baby-thermo-rsi/",
        "0620d779bda83b2c5824391a2ab87cae80a51f9d1b6f76755eb288f1f2833d06",
    ),
    "torx": (
        "https://arxiv.org/abs/2608.01612v2",
        "e62ce2bdc70d3d478c87bc69fd89fde40707d0a3c2e1e518957d644f2c2bb7dc",
    ),
    "thermalizers": (
        "https://arxiv.org/abs/2608.01615v2",
        "7f88f442f84adfee3600ed89befcd4172d6b3d8057da049fed04de1ec88ee311",
    ),
}
# Separate globals preserve every historical consumer's original bindings.
engine = namespace(historical, "hardware8851")
engine.OWNED, engine.TEST, engine.CLI, engine.NAME = OWNED, TEST, CLI, NAME
engine.PRODUCERS = {
    "witness": (
        "results/experiment_8847_v757_indexed_cost.json",
        "sha256:47548df6e4e4437f12f353298461b28be99129ee7759a55da779fb93a55f9a3e",
        "indexed_cost_ready_score",
        8847,
    ),
    "whole": (
        "results/experiment_8848_v757_memory_size_cost.json",
        "sha256:62ddef9b5966312650bf33fecb9b02308beffab6f58929e041b31a977c2994d3",
        "whole_cost_ready_score",
        8848,
    ),
}
engine.HISTORY = {
    "results/experiment_8838_v756_hardware_operation_budget.json": "sha256:9bcc71f3f09db9d895858660a563c065b484dfdb5f45948093e9285cb82463e4",
    **{p: h for p, h in historical.HISTORY.items() if p.endswith(".md")},
    **{
        p: h
        for p, h in board_history.HISTORY.items()
        if any(n in p for n in ["8230", "8259", "8302"])
    },
    **{f"results/raw/{NAME}/authorities/{h}": "sha256:" + h for _, h in EXTERNAL.values()},
}
engine.SOURCES = (
    [
        p
        for p in historical.SOURCES
        if p not in historical.OWNED + [historical.TEST]
        and not p.startswith("docs/research-notes/")
    ]
    + OWNED
    + [
        TEST,
        "python/carnot/reporting/hardware_operation_budget_8838.py",
        "python/carnot/reporting/hardware_operation_runner_8825.py",
        "python/carnot/reporting/v748_collection_contract.py",
    ]
)
base_build = engine.build
base_collect = engine.collect
replay = engine.replay


def progress(phase: str) -> None:
    """Flush phase boundaries so a quiet reader cannot conceal a stopped run."""
    print(f"[exp8851] phase={phase}", flush=True)


def retain(path: Path, raw: Path, expected: str | None = None) -> Json:
    """A changed document may use preserved bytes; executable drift is rejected."""
    source = path
    if expected is not None and sha256_file(path) != expected:
        if not ("openspec/capabilities/" in str(path) and path.name == "spec.md"):
            raise ValueError("operand_hash_drift:" + str(path))
        source = AUTHORITIES / expected.split(":")[-1]
    digest = sha256_file(source)
    if expected is not None and digest != expected:
        raise ValueError("operand_hash_drift:" + str(path))
    target = raw / "operands" / digest.split(":")[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    if sha256_file(target) != digest:
        raise ValueError("operand_hash_drift:" + str(path))
    return dict(path=str(target.absolute()), sha256=digest, original_path=str(path))


def collect(raw: Path) -> Json:
    """Check the actual interpreter and resources before independently reading costs."""
    work: Json = base_collect(raw)
    work["checks"].update(
        running_repository_virtualenv=Path(sys.prefix) == ROOT / ".venv",
        installed_dependencies={
            n: importlib.util.find_spec(n) is not None
            for n in ["pytest", "coverage", "ruff", "mypy"]
        },
        time_budget_s=4200,
        preflight_elapsed_s=(time.monotonic_ns() - work["started_ns"]) / 1e9,
        physical_probes_performed=0,
    )
    progress("operand_admission_after")
    return work


def reduce(evidence: Json) -> Json:
    """Keep complete clocks and measured state bytes; counts never imply time shares."""
    costs, memory, prep = [], [], []
    for branch in sorted(evidence):
        producer = evidence[branch]
        for row in producer["window_rows" if branch == "whole" else "rows"]:
            cost = {
                k: row.get(k)
                for k in [
                    "unit_id",
                    "cluster_id",
                    "arm",
                    "seed",
                    "status",
                    "denominator",
                    "missing_reason",
                ]
            }
            if branch == "whole":
                predictions = [p for s in row["segments"] for p in s["predictions"]]
                releases = [r for s in row["segments"] for r in s["releases"]]
                counts = dict(
                    counter_updates=len(predictions),
                    template_matches=len(predictions) + len(releases),
                    energy_evaluation=len(predictions) + len(releases),
                    gradients=row["attempted_updates"],
                    applied_updates=row["applied_updates"],
                    feedback=sum(r["feedback_verifier_calls"] for r in releases),
                    template_additions=row["template_additions"],
                    **row["fsync_counts"],
                )
                counts["serialization"] = len(predictions) + len(releases) + len(row["segments"])
                seconds = row["complete_ns"] / 1e9
                memory.append(
                    dict(
                        unit_id=row["unit_id"],
                        occupancy=row["occupancy"],
                        final_occupancy=row["final_occupancy"],
                        capacity=32,
                        state_bytes=row["serialized_state_bytes"],
                        response_bytes=row["response_bytes"],
                        template_hits=row["template_hits"],
                        wire_bytes=None,
                    )
                )
                cost.update(
                    occupancy=row["occupancy"],
                    fsync_counts=row["fsync_counts"],
                    operation_count_units=dict(
                        counter_updates="six-feature observation batches; internal arithmetic uncounted",
                        template_matches="prediction and release membership lookups; string comparisons uncounted",
                        energy_evaluation="three-label normalization batches, including gradient loss",
                        gradients="attempted numerical update calls",
                        serialization="durable store saves; other serialization calls uncounted",
                        file="file fsync calls",
                        directory="directory fsync calls",
                    ),
                )
            else:
                counts, seconds = row["operation_counts"], row["wall_time_s"]
                cost.update(
                    cpu_time_s=row["cpu_time_s"],
                    primitive_operations=row["primitive_operations"],
                    condition=row["condition"],
                    split=row["split"],
                    size=row["size"],
                )
            cost.update(
                metric=seconds / row["denominator"],
                complete_operation_s=seconds,
                operation_counts=counts,
                source_branch=branch,
            )
            costs.append(cost)
        prep.extend(producer.get("preparation_cost_rows", []))
    operations = {
        "index_build": "exact predicate index allocation and construction",
        "predicate_lookup": "exact symbolic predicate matching",
        "binding_joins": "content-dependent string bindings and conflict checks",
        "counter_updates": "integer observations and six feature counters",
        "template_matching": "exact normalized templates; capacity32",
        "energy_evaluation": "float32 three labels, 9x4 then4x1 SiLU;45 parameters/180 bytes",
        "gradients": "float32 dense autodiff; attempted and committed updates differ",
        "serialization": "complete JSON request, response and checkpoint bytes",
        "feedback": "delayed label verification and durable acknowledgements",
        "fsync": "file and directory durability across restart and tail drain",
    }
    boards = [
        dict(
            unit_id=board + "/" + op,
            cluster_id=board,
            arm=op,
            seed=0,
            denominator=1,
            metric=0,
            status="completed",
            missing_reason=None,
            operation_status="requires_new_kernel" if op in list(operations)[:7] else "host_only",
            requirements=description,
            boundary=boundary["boundary"],
            physical_block="0xffffffff" if board == "GateMate" else None,
            fabric_supported=False,
            measured_board_time_s=None,
            transfer_bytes=None,
        )
        for board, boundary in engine.BOARDS.items()
        for op, description in operations.items()
    ]
    count_keys = dict(
        index_build=["index_build"],
        predicate_lookup=["index_lookup"],
        binding_joins=["join_attempts"],
        counter_updates=["counter_updates"],
        template_matching=["template_matches"],
        energy_evaluation=["energy_evaluation"],
        gradients=["gradients"],
        serialization=["request_serialization", "serialization"],
        feedback=["feedback"],
        fsync=["file", "directory"],
    )
    for row in boards:
        row["operation_counts_by_branch"] = {
            branch: {
                key: sum(c["operation_counts"][key] for c in costs if c["source_branch"] == branch)
                for key in count_keys[row["arm"]]
                if all(key in c["operation_counts"] for c in costs if c["source_branch"] == branch)
            }
            for branch in sorted(evidence)
        }
        row["count_scope"] = (
            "paid CPU inventory; absent keys are unmeasured; no timing fractions or fabric work"
        )
    return dict(
        rows=costs + boards,
        board_operation_rows=boards,
        complete_cost_refs=costs,
        preparation_cost_refs=prep,
        memory_size_requirements=memory,
        indexed_build_refs=evidence.get("witness", {}).get("index_build_rows", []),
        occupancy_cost_curve=evidence.get("whole", {}).get("absolute_cost_rows", []),
        unsupported_operations=[
            r["unit_id"] for r in boards if r["operation_status"] == "requires_new_kernel"
        ],
        unknown_costs=[
            "component times and compatible fractions",
            "board startup and transfer",
            "wire format and buffers",
            "NPU and TSU kernels",
        ],
        sampling_call_count=0,
        physical_evidence_state=engine.BOARDS,
    )


def build(work: Json, validation: Json) -> Json:
    """Passing owned checks qualifies a map, while physical gain remains unproved."""
    engine.verify_validation(validation)
    adjusted = dict(work)
    adjusted["failures"] = [
        r for r in work["failures"] if r["check"] != "producer_admission" or not work["evidence"]
    ]
    checks = work["checks"]
    adjusted["internal_failure"] = work["internal_failure"] or not (
        checks["running_repository_virtualenv"]
        and all(checks["installed_dependencies"].values())
        and checks["free_disk_bytes"] >= checks["minimum_free_bytes"]
        and checks["preflight_elapsed_s"] < checks["time_budget_s"]
    )
    value: Json = base_build(adjusted, validation)
    verdict = value["verdict_class"]
    if verdict != "disqualified" and not work["evidence"]:
        verdict = "blocked"
    ready = int(verdict == "null")
    start, end = work["started_ns"], work["ended_ns"]
    receipt = build_current_work_receipt(
        run_id=str(start),
        owner_pid=work["owner_pid"],
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load" if work["evidence"] else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(operation="immutable CPU evidence reader"),
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[dict(phase="operand_admission", start_s=0, end_s=(end - start) / 1e9)],
    )
    value.update(
        {
            k: receipt[k]
            for k in [
                "MODEL_SPECS",
                "inference_substrate",
                "inference_substrate_class",
                "inference_substrate_details",
                "duration_s",
                "phase_spans",
            ]
        }
    )
    value.update(
        experiment_id=8851,
        task_id="exp8851-hardware-operation-budget",
        milestone="2026.11.757",
        random_seed=7578851,
        run_date="20261105",
        honest_verdict="complete_" + verdict + "_hardware_operation_budget",
        verdict_class=verdict,
        hardware_budget_ready_score=ready,
        required_checks_passed=bool(ready),
        execution_venue="host",
        current_work_receipt=receipt,
        phase_work=work,
        gate_check_summary=work["failures"],
    )
    value["acceptance_gates"]["audit"]["passed"] = bool(ready)
    value["hardware_learning_path"] = dict(
        counter_updates="host exact integer observations; fabric needs a new validated kernel",
        template_matching="host exact string/binding lookup with capacity32; indexed verifier is separate",
        small_head_fitting="host float32 dense categorical gradients; no qualified fabric or TSU path",
        durable_boundary="complete_cost_refs includes imports, two startups, restart, feedback tail, serialization and file/directory fsync; preparation_cost_refs remains separate",
        research_100x="operation-specific future target; no measured gain",
        rust_service_10x="PRD target unproved",
    )
    value["acquisition_priority"] = dict(
        recommend_acquisition=False,
        reason="no measured complete supported physical workload benefit",
        priority="defer",
    )
    value["external_compatibility"] = dict(
        authorities={n: dict(url=u, sha256="sha256:" + h) for n, (u, h) in EXTERNAL.items()},
        anechoic="synchronous dense binary Max-Cut annealing with Onsager correction on Alveo V80; optimization quality does not certify Gibbs sampling or KV260 compatibility",
        torx="typed local stochastic kernels over binary, categorical and continuous wires; software types do not qualify this categorical head on physical TSUs",
        thermalizers="approximate factor-to-EBM compilation; factor error can accumulate across trajectories",
        engineering_evidence_needed=[
            "executable kernel mapping, topology and precision parity",
            "target-distribution and temperature calibration, mixing, bias and effective sample size",
            "authenticated simulator/device access and reproducible compiler/runtime",
            "complete host, transfer, programming, readout, fitting and durable costs",
        ],
        local_simulator_access=False,
        local_physical_deployment=False,
        npu_qualified=False,
        tsu_qualified=False,
    )
    history = {}
    for board, number in [("KV260", "8230"), ("PolarFire", "8259"), ("GateMate", "8302")]:
        ref = next((r for r in work["refs"] if number in r["original_path"]), None)
        if ref is None:
            history[board] = dict(
                **engine.BOARDS[board], receipt=None, run_date=None, age_days=None
            )
            continue
        original = json.loads(Path(ref["path"]).read_bytes())
        date = original["run_date"]
        age = (datetime.strptime("20261105", "%Y%m%d") - datetime.strptime(date, "%Y%m%d")).days
        history[board] = dict(
            **engine.BOARDS[board],
            receipt=ref,
            run_date=date,
            age_days=age,
            access="ssh kria" if board == "KV260" else "historical receipt only",
        )
    value["physical_evidence_state"] = history
    value["methodology_note"] = (
        "Independent immutable indexed and occupancy evidence; current reader owns zero training or model calls. Operation counts are not timing fractions. Historical receipts are not fresh availability probes. No physical acceleration, hidden-game or natural-language generalization is established."
    )
    value["field_principles"].update(
        {
            k: "Retain measured complete CPU boundaries separately from unknown physical cost and operation-specific future targets."
            for k in value
            if k not in value["field_principles"]
        }
    )
    return normalize_artifact_for_template_write(value)


engine.progress, engine.retain, engine.reduce, engine.build = progress, retain, reduce, build


def main(argv: list[str] | None = None) -> int:
    """Private runs cannot bypass validation to publish a ready primary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261105"], default="20261105")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--replay", "--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError) as exc:
            print(json.dumps(dict(passed=False, reason=str(exc))), flush=True)
            return 1
    from carnot.reporting.hardware_budget_validation_8851 import qualify, terminal
    from carnot.reporting.primary_publication import publish_primary

    output = (args.private_output or DEFAULT_OUTPUT).absolute()
    raw = output.parent / "raw" / NAME / "invocations" / str(time.monotonic_ns())
    progress("admission_before")
    work = collect(raw)
    atomic_json(raw / "work.json", work)
    progress("validation_before")
    validation = (
        dict(passed=False, receipts=[], coverage=dict(complete=False))
        if args.private_output
        else qualify(raw)
    )
    progress("validation_after")
    value = build(work, validation)
    progress("publication_before")
    report = publish_primary(
        output, value, lambda p: replay(p) if args.private_output else terminal(p, raw)
    )
    atomic_json(raw / "publication.json", report)
    progress("complete")
    return 0
