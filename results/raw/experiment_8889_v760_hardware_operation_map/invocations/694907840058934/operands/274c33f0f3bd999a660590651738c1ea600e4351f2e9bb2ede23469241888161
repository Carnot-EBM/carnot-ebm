"""REQ-REPORT-8876: map measured V759 CPU work without predicting device speed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import hardware_budget_8864 as historical
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT = historical.ROOT
NAME = "experiment_8876_v759_hardware_operation_budget"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/hardware_operation_budget_8876.py",
    "python/carnot/reporting/hardware_budget_validation_8876.py",
    CLI,
]
TEST = "tests/python/test_hardware_operation_budget_8876.py"
DEFAULT_OUTPUT = ROOT / "results" / (NAME + ".json")
previous = namespace(historical, "hardware8876")
engine = previous.engine
engine.NAME, engine.OWNED, engine.CLI, engine.TEST = NAME, OWNED, CLI, TEST
engine.PRODUCERS = {
    "witness": (
        "results/experiment_8874_v759_size_aware_complete_cost.json",
        "sha256:2aaaa2de3a2f2d4f4873c51623d9031f4bcdbd63a7eb83b136102170393b5f8c",
        "size_cost_ready_score",
        8874,
    ),
    "learning": (
        "results/experiment_8872_v759_matched_causal_streams.json",
        "unavailable_original_execution_authority",
        "matched_stream_rows_ready_score",
        8872,
    ),
    "label_audit": (
        "results/experiment_8873_v759_label_value_audit.json",
        "unavailable_original_execution_authority",
        "label_value_audit_ready_score",
        8873,
    ),
}
engine.HISTORY = {
    **engine.HISTORY,
    f"results/raw/{NAME}/authorities/d74c2cff9d61e53cbb80de9893420388796398e90c6702aed46622fa212cb22b": "sha256:d74c2cff9d61e53cbb80de9893420388796398e90c6702aed46622fa212cb22b",
    f"results/raw/{NAME}/authorities/0b83813e58e4d351a662df3104ada7ed2e40526477841028579750b5cd229cbc": "sha256:0b83813e58e4d351a662df3104ada7ed2e40526477841028579750b5cd229cbc",
    "results/experiment_8864_v758_hardware_budget.json": "sha256:78ed5aa61b01d376f7a1a3053d30f77a108da9a14337dcef9bdf202528623d0b",
    "results/experiment_8862_v758_linear_durable_cost.json": "sha256:6fd5d69c40f4fc3884ef5c3cce6ee0de5f98778b969b9850bfd9df89c03d2931",
    "results/experiment_8866_v759_branch_contract.json": "sha256:0909c9352854d04febb5e6a2e5dd9a2d67bed228bc2b916399f8fb692f23635a",
}
engine.SOURCES = list(dict.fromkeys([*engine.SOURCES, *OWNED, TEST, "ops/north-star.md"]))
collect, replay = previous.collect, engine.replay


def progress(phase: str) -> None:
    """Flush boundaries so the fixed execution deadline remains observable."""
    print(f"[exp8876] phase={phase} completed=0 pending=0", flush=True)


def admission(value: Json, branch: str) -> str | None:
    """Missing original learning authority cannot be replaced by later mutable bytes."""
    if branch != "witness":
        return "rejected_producer"
    return previous.admission(value, branch)


def reduce(evidence: Json) -> Json:
    """Reduce actual request counters; parent clocks include process and durable overhead."""
    installed = evidence.get("witness", {})
    rows = [
        dict(r, wall_time_s=r["complete_wall_time_s"], world_wall_time_s=r["complete_wall_time_s"])
        for r in installed.get("rows", [])
    ]
    reduced: Json = previous.reduce({"witness": dict(installed, rows=rows)} if rows else {})
    for cost, original in zip(
        reduced["complete_cost_refs"], installed.get("rows", []), strict=True
    ):
        cost.update(
            {
                k: original[k]
                for k in ["repeat", "population", "primitive_limit", "budget_exhausted"]
            }
        )
        cost["kernel_wall_time_s"] = original["wall_time_s"]
        cost["operation_counts"] = dict(original["operation_counts"])
    reduced.update(
        learning_operation_rows=[],
        label_call_savings=dict(
            status="blocked",
            saved_calls=None,
            baseline_calls=None,
            query_only_calls=None,
            missing_reason="missing_original_learning_rows",
        ),
        lookup_operation_totals=[
            dict(
                boundary=b,
                arm=a,
                observation_count=len(sample),
                independent_world_count=len({r["cluster_id"] for r in sample}),
                counts={
                    key: sum(r["operation_counts"].get(key, 0) for r in sample)
                    for key in sorted({k for r in sample for k in r["operation_counts"]})
                },
            )
            for b in ("cold", "resident")
            for a in ("scan", "indexed", "size_aware")
            if (sample := [r for r in rows if r["boundary"] == b and r["arm"] == a])
        ],
        sampling_call_count=0,
        operation_map=dict(
            tier1="CPU integer checks and feature counters",
            tier2="CPU/RAM normalized-template matching; future FPGA index requires a new kernel",
            small_linear_updates="batchable GPU/native float32 arithmetic; no new V759 update measurements",
            joins="CPU exact binding joins",
            categorical_normalization="CPU exact three-class enumeration",
            durable_acknowledgements="host serialization and file/directory fsync",
        ),
    )
    reduced["operation_count_units"].update(
        lookup_operation_totals="sum of metered primitive counts by boundary and arm; three repeats remain observations",
        label_checks="unknown without original V759 release rows",
        template_matches="unknown without original V759 template lookup rows",
        fsync="unknown for V759 learning; installed request durability is inside parent clocks",
        sampling_call_count="zero calls in this audit and installed exact-witness path; unavailable learning is unknown",
    )
    reduced["lookup_operation_savings"] = [
        dict(
            boundary=b,
            comparison="scan_minus_size_aware",
            observation_count=len(sample) // 3,
            **{
                "saved_" + key + "_units": sum(
                    r["operation_counts"].get(key, 0)
                    * (1 if r["arm"] == "scan" else -1 if r["arm"] == "size_aware" else 0)
                    for r in sample
                )
                for key in ["index_build", "index_lookup", "fact_comparisons", "join_attempts"]
            },
        )
        for b in ("cold", "resident")
        if (sample := [r for r in rows if r["boundary"] == b])
    ]
    reduced["unknown_costs"].extend(
        [
            "V759 label-check/template-match/small-head-update/fsync operands",
            "V759 learning and label audit original execution authorities",
        ]
    )
    return reduced


def build(work: Json, validation: Json) -> Json:
    """Qualify owned accounting separately from blocked external learning evidence."""
    value: Json = previous.build(work, validation)
    verdict = value["verdict_class"]
    if verdict != "disqualified" and work["failures"]:
        verdict = "blocked"
    failures = [r for r in work["failures"] if r["upstream"] in {"learning", "label_audit"}]
    learning = [
        dict(
            unit_id=r["upstream"],
            cluster_id=r["upstream"],
            arm=r["upstream"],
            seed=7598876,
            metric=None,
            denominator=1,
            status="blocked",
            missing_reason=r["missing_reason"],
            label_checks=None,
            template_matches=None,
            small_head_updates=None,
            fsync_counts=None,
            sampling_call_count=None,
            source_gate=r,
        )
        for r in failures
    ]
    value["rows"].extend(learning)
    value.update(
        experiment_id=8876,
        task_id="exp8876-hardware-operation-budget",
        milestone="2026.11.759",
        run_date="20261106",
        random_seed=7598876,
        verdict_class=verdict,
        honest_verdict="complete_" + verdict + "_hardware_operation_budget",
        hardware_budget_ready_score=int(verdict == "null"),
        required_checks_passed=bool(validation["passed"] and verdict != "disqualified"),
        learning_operation_rows=learning,
        gate_check_summary=work["failures"],
        independent_count=len({r["cluster_id"] for r in value["complete_cost_refs"]}),
        intended_count=len(value["rows"]),
        blocked_count=len(learning),
        invocation=dict(
            argv=work.get("invocation_argv", []),
            owner_pid=work["owner_pid"],
            run_id=str(work["started_ns"]),
        ),
        historical_complete_cost_refs=[
            r
            for r in work["refs"]
            if r["original_path"]
            in {str(ROOT / p) for p in engine.HISTORY if "8862_" in p or "8864_" in p}
        ],
    )
    for status, field in [
        ("completed", "completed_count"),
        ("censored", "censored_count"),
        ("failed", "failed_count"),
    ]:
        value[field] = sum(r["status"] == status for r in value["rows"])
    value["acceptance_gates"]["audit"]["passed"] = value["required_checks_passed"]
    value["acceptance_gates"]["upstream_complete"] = dict(
        passed=not work["failures"],
        threshold="all original V759 producer authorities available",
        principle="Missing learning inputs remain terminal blocked while installed work is retained",
    )
    value["external_compatibility"].update(
        higher_order_ising=dict(
            url="https://arxiv.org/abs/2610.40522v1",
            comparison="Versal VP1802 PUBO spin-group engine; not a KV260 kernel or measured Carnot acceleration",
            local_speed_inherited=False,
        ),
        z1t=dict(
            url="https://extropic.ai/writing/z1t",
            comparison="Future heterogeneous sparse probabilistic hardware/software lead; vendor estimates remain separate from local work",
            local_speed_inherited=False,
        ),
    )
    value["hardware_learning_path"].update(value["operation_map"])
    value["methodology_note"] = (
        "Independently authenticated installed scan/index/size-aware null evidence. Label savings and learning updates require absent original V759 rows. Current CPU reader loads no model, trains no head, invokes no sampler and performs no physical probe. Operation counts cannot give timing fractions, natural-language generalization or FPGA/TSU acceleration. The100x ambition remains a target."
    )
    value["field_principles"].update(
        {
            k: "Measured CPU counts retain full costs and repeats; missing learning work and transfer fractions remain unknown, never physical speed evidence."
            for k in value
            if k not in value["field_principles"]
        }
    )
    return value


engine.progress, engine.admission, engine.reduce, engine.build = progress, admission, reduce, build
previous.previous.progress = progress


def main(argv: list[str] | None = None) -> int:
    """Publish checked terminal bytes; private fixtures cannot certify owned readiness."""
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
    from carnot.reporting.hardware_budget_validation_8876 import qualify, terminal
    from carnot.reporting.primary_publication import publish_primary

    output = (args.private_output or DEFAULT_OUTPUT).absolute()
    raw = output.parent / "raw" / NAME / "invocations" / str(time.monotonic_ns())
    progress("operand_admission_before")
    work = collect(raw)
    work["invocation_argv"] = [
        sys.executable,
        str(ROOT / CLI),
        *(sys.argv[1:] if argv is None else argv),
    ]
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
