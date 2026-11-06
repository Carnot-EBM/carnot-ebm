"""REQ-REPORT-8864: map installed and durable costs without inferring hardware speed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting import hardware_operation_budget_8851 as historical
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT = historical.ROOT
NAME = "experiment_8864_v758_hardware_budget"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/hardware_budget_8864.py",
    "python/carnot/reporting/hardware_budget_validation_8864.py",
    CLI,
]
TEST = "tests/python/test_hardware_budget_8864.py"
DEFAULT_OUTPUT = ROOT / "results" / (NAME + ".json")
# Separate module globals preserve historical producer and replay bindings.
previous = namespace(historical, "hardware8864")
engine = previous.engine
old_owned = list(engine.OWNED)
previous.NAME, previous.OWNED = NAME, OWNED
previous.AUTHORITIES = ROOT / "results/raw" / NAME / "authorities"
engine.NAME, engine.OWNED, engine.CLI, engine.TEST = NAME, OWNED, CLI, TEST
engine.PRODUCERS = {
    "witness": (
        "results/experiment_8861_v758_installed_index_cost.json",
        "sha256:7d3e94e552dac90e0a6c54ee0ab0fcbf8135b7337b1ed5a0a7a911ceac511d31",
        "installed_cost_ready_score",
        8861,
    ),
    "whole": (
        "results/experiment_8862_v758_linear_durable_cost.json",
        "sha256:6fd5d69c40f4fc3884ef5c3cce6ee0de5f98778b969b9850bfd9df89c03d2931",
        "whole_cost_ready_score",
        8862,
    ),
}
engine.HISTORY = {
    **engine.HISTORY,
    "results/raw/experiment_8864_v758_hardware_budget/authorities/9c876ba8da91fa5c1a081fa3efd9348ccb75d7880f72ff865a7970a527654b95": "sha256:9c876ba8da91fa5c1a081fa3efd9348ccb75d7880f72ff865a7970a527654b95",
    "results/experiment_8851_v757_hardware_operation_budget.json": "sha256:78a125b933907c5e1065edc7649adc19bbe7806c6313cd78f573ab1eeebe2fcb",
}
engine.SOURCES = list(
    dict.fromkeys(
        [
            *[p for p in engine.SOURCES if p not in old_owned + [historical.TEST]],
            *old_owned,
            historical.TEST,
            *OWNED,
            TEST,
        ]
    )
)
collect, replay = previous.collect, engine.replay


def progress(phase: str) -> None:
    """Flush phase boundaries so a source reader cannot conceal a stopped run."""
    print(f"[exp8864] phase={phase} completed=0 pending=0", flush=True)


def admission(value: Json, branch: str) -> str | None:
    """Censored installed requests are measured costs, so retain their original status."""
    field = engine.PRODUCERS[branch][2]
    if value.get(field) is None:
        return "missing_field"
    if value[field] == 0:
        return "observed_zero"
    rows = value.get("window_rows" if branch == "whole" else "rows", [])
    if (
        value[field] != 1
        or value.get("experiment_id") != engine.PRODUCERS[branch][3]
        or value.get("required_checks_passed") is not True
        or value.get("flagged_adversarial", False) is not False
        or value.get("verdict_class") not in {"positive", "circular_positive", "null"}
        or not str(value.get("honest_verdict", "")).startswith("complete_")
        or not rows
        or any(
            r.get("status") not in {"completed", "censored"} or r.get("denominator", 0) <= 0
            for r in rows
        )
    ):
        return "rejected_producer"
    return None


def reduce(evidence: Json) -> Json:
    """Keep complete parent clocks separate from inner calls and untimed operation counts."""
    costs = []
    for branch, producer in sorted(evidence.items()):
        for row in producer["window_rows" if branch == "whole" else "rows"]:
            cost = {
                k: row.get(k)
                for k in (
                    "unit_id",
                    "cluster_id",
                    "arm",
                    "seed",
                    "status",
                    "denominator",
                    "missing_reason",
                )
            }
            if branch == "whole":
                predictions = [p for s in row["segments"] for p in s["predictions"]]
                releases = [r for s in row["segments"] for r in s["releases"]]
                counts = dict(
                    feature_gathering=len(predictions),
                    primitive_checks=sum(p["primitive_checks"] for p in predictions),
                    linear_energy=len(predictions),
                    gradients=sum(r["optimizer_steps"] for r in releases),
                    attempted_updates=row["attempted_updates"],
                    applied_updates=row["applied_updates"],
                    serialization=len(predictions) + len(releases) + len(row["segments"]),
                    **row["fsync_counts"],
                )
                seconds = row["complete_ns"] / 1e9
                cost.update(
                    serialized_bytes=row["serialized_bytes"],
                    state_bytes=row["serialized_state_bytes"],
                    response_bytes=row["response_bytes"],
                    occupancy=row["occupancy"],
                )
            else:
                counts, seconds = row["operation_counts"], row["wall_time_s"]
                cost.update(
                    boundary=row["boundary"],
                    size=row["size"],
                    world_wall_time_s=row["world_wall_time_s"],
                    cpu_time_s=row["cpu_time_s"],
                    primitive_operations=row["primitive_operations"],
                )
            cost.update(
                metric=seconds / row["denominator"],
                complete_operation_s=seconds,
                source_branch=branch,
                operation_counts=counts,
            )
            costs.append(cost)
    operations = {
        "index_build": ["index_build", "index_allocation"],
        "predicate_lookup": ["index_lookup"],
        "binding_joins": ["join_attempts", "binding_copy", "conflict_checks"],
        "feature_gathering": ["feature_gathering", "primitive_checks"],
        "template_matching": [],
        "linear_energy": ["linear_energy"],
        "gradients": ["gradients", "attempted_updates", "applied_updates"],
        "serialization": ["serialization", "request_serialization"],
        "fsync": ["file", "directory"],
    }
    boards = [
        dict(
            unit_id=board + "/" + operation,
            cluster_id=board,
            arm=operation,
            seed=0,
            metric=0,
            denominator=1,
            status="completed",
            missing_reason=None,
            operation_status="host-only"
            if operation in {"serialization", "fsync"}
            else "new-kernel-required",
            boundary=boundary["boundary"],
            fabric_supported=False,
            measured_board_time_s=None,
            transfer_bytes=None,
            operation_counts_by_branch={
                branch: {
                    key: sum(
                        c["operation_counts"][key] for c in costs if c["source_branch"] == branch
                    )
                    for key in keys
                    if all(
                        key in c["operation_counts"] for c in costs if c["source_branch"] == branch
                    )
                }
                for branch in sorted(evidence)
            },
            count_scope="CPU inventory only; absent counts and component times are unknown",
        )
        for board, boundary in engine.BOARDS.items()
        for operation, keys in operations.items()
    ]
    return dict(
        rows=costs + boards,
        complete_cost_refs=costs,
        board_operation_rows=boards,
        preparation_cost_refs=evidence.get("whole", {}).get("preparation_cost_rows", []),
        installed_parent_cost_refs=evidence.get("witness", {}).get("cold_process_rows", []),
        installed_resident_cost_refs=evidence.get("witness", {}).get("resident_rows", []),
        installation_accounting={
            k: evidence.get("witness", {}).get(k)
            for k in ("installation_s", "installation_accounting")
        },
        linear_head=dict(
            parameter_scalars=21,
            parameter_float32_bytes=84,
            labels=3,
            features_with_bias=7,
            arithmetic="float32 3x7 linear logits and stable three-way normalization",
        ),
        operation_count_units=dict(
            linear_energy="prediction batches only; gradient loss counted separately",
            gradients="executed optimizer steps; attempted and applied updates retained",
            serialization="durable store saves; other JSON calls uncounted",
            request_serialization="metered units; not elapsed time",
            fsync="store file and directory barriers; parent includes other barriers",
        ),
        unsupported_operations=[
            r["unit_id"] for r in boards if r["operation_status"] == "new-kernel-required"
        ],
        unknown_costs=[
            "compatible component timing fraction",
            "host-board transfers and wire bytes",
            "index/joins/features/normalization component times",
            "serialization/fsync component times",
            "board startup and compiler/runtime costs",
            "NPU and TSU kernels",
        ],
        end_to_end_gain_bound=None,
        sampling_call_count=0,
        physical_evidence_state=engine.BOARDS,
    )


def build(work: Json, validation: Json) -> Json:
    """Mapping readiness requires owned controls and never implies physical acceleration."""
    value: Json = previous.build(work, validation)
    value.update(
        experiment_id=8864,
        task_id="exp8864-hardware-budget",
        milestone="2026.11.758",
        run_date="20261106",
        random_seed=7588864,
        honest_verdict="complete_" + value["verdict_class"] + "_hardware_budget",
    )
    for status, field in [
        ("completed", "completed_count"),
        ("censored", "censored_count"),
        ("failed", "failed_count"),
    ]:
        value[field] = sum(r["status"] == status for r in value["rows"])
    for board in value["physical_evidence_state"].values():
        if board["age_days"] is not None:
            board["age_days"] += 1
    value["hardware_learning_path"] = dict(
        fr11="CPU integer counters and exact feature gathering; batched small-head fitting and accelerated template matching need separate precision and full-boundary measurements",
        small_head="three labels, six counts plus bias,21 float32 parameters; linear fitting is not binary Gibbs sampling",
        durable_boundary="retain installation, process imports/startups, restart, delayed feedback tail, serialization and file/directory fsync; preparation stays separate",
        research_100x="unknown end-to-end gain; first measure a disjoint compatible time fraction, then kernel parity, transfers and full durable latency",
        rust_service_10x="PRD native target unproved; compare identical complete installed/durable workloads including errors and censors",
        minimum_evidence=[
            "exact index, binding join, feature and template kernel parity",
            "float32 linear logits, normalization and gradient parity",
            "measured startup, host transfers, device buffers and durable tail",
            "paired complete-workload timings with a supported measured fraction",
        ],
    )
    value["external_compatibility"]["fpga_asic"] = dict(
        url="https://arxiv.org/abs/2602.15985v2",
        sha256="sha256:9c876ba8da91fa5c1a081fa3efd9348ccb75d7880f72ff865a7970a527654b95",
        assumptions="capacity-constrained Ising decomposition requires host orchestration and boundary traffic; it supplies no exact relational join or normalized three-way score kernel",
        compatibility="An Ising energy representation alone does not establish board compatibility; no local physical acceleration or purchase case follows.",
    )
    value["methodology_note"] = (
        "Independent installed-request and whole durable linear costs. Censored requests retain paid work. Counts are not timing fractions. Exact relational joins and normalized three-way scores are not binary Gibbs sampling. Current reader owns zero training/model calls and performs no SSH, JTAG or availability probe. No physical gain or independent generalization is established."
    )
    value["field_principles"].update(
        {
            k: "Retain full measured CPU boundaries; counts and unknown transfers cannot establish physical gain."
            for k in value
            if k not in value["field_principles"]
        }
    )
    return value


engine.progress, engine.admission, engine.reduce, engine.build = progress, admission, reduce, build
previous.progress = progress


def main(argv: list[str] | None = None) -> int:
    """Private fixtures stay disqualified; normal publication runs all frozen checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261106"], default="20261106")
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
    from carnot.reporting.hardware_budget_validation_8864 import qualify, terminal
    from carnot.reporting.primary_publication import publish_primary

    output = (args.private_output or DEFAULT_OUTPUT).absolute()
    raw = output.parent / "raw" / NAME / "invocations" / str(time.monotonic_ns())
    progress("operand_admission_before")
    work = collect(raw)
    atomic_json(raw / "work.json", work)
    progress("owned_validation_before")
    validation = (
        dict(passed=False, receipts=[], coverage=dict(complete=False))
        if args.private_output
        else qualify(raw)
    )
    progress("owned_validation_after")
    value = build(work, validation)
    progress("publication_before")
    report = publish_primary(
        output, value, lambda p: replay(p) if args.private_output else terminal(p, raw)
    )
    atomic_json(raw / "publication.json", report)
    progress("complete")
    return 0
