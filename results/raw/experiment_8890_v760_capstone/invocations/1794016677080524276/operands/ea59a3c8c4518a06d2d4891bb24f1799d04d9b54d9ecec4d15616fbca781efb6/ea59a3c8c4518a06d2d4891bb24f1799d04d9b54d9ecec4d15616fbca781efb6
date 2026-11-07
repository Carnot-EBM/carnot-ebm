"""REQ-REPORT-8889: map paid CPU operations without inventing device fractions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import hardware_operation_budget_8838 as shared
from carnot.reporting import hardware_operation_map_8825 as board_history
from carnot.reporting import hardware_operation_runner_8825 as validation_runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
ROOT = shared.ROOT
NAME = "experiment_8889_v760_hardware_operation_map"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/hardware_operation_map_8889.py", CLI]
TEST = "tests/python/test_hardware_operation_map_8889.py"
DEFAULT_OUTPUT = ROOT / "results" / (NAME + ".json")
PRODUCERS = {
    "attribution": (
        "results/experiment_8882_v760_request_cost_attribution.json",
        "sha256:cbb9f79adfe6fbe6c974169366efb5603005f4abb7ea0e50b7d6415753612a93",
        "cost_attribution_ready_score",
        8882,
    ),
    "labels": (
        "results/experiment_8885_v760_sequential_label_cost.json",
        "sha256:4372f5a0a50918da1fe848350a1386208aa0697208a9a907b7f90e99b0b56c53",
        "sequential_rows_ready_score",
        8885,
    ),
    "adaptation": (
        "results/experiment_8888_v760_matched_adaptation_cost.json",
        "sha256:9f5517cbd96ed2456a5b6ee222bf74ae1bacf9be49d06c7b1a29465a3a7867a1",
        "adaptation_cost_ready_score",
        8888,
    ),
}
engine = namespace(shared, "hardware8889")
engine.NAME, engine.CLI, engine.OWNED, engine.TEST = NAME, CLI, OWNED, TEST
engine.PRODUCERS = PRODUCERS
EXTERNAL = {
    "anechoic": (
        "https://arxiv.org/abs/2610.13368v2",
        "ebd7a754731cae482901aeb5d0b0ef54628d67bc4373eaa29e817e46f678da75",
    ),
    "torx": (
        "https://github.com/extropic-ai/torx",
        "a625bf81a167417cb79cef62dcb7b04c791c84efccedcafc2ae6f4dcf0f1ceba",
    ),
    "thermalizers_z1": (
        "https://extropic.ai/writing/from-one-to-one-billion",
        "4767b95ac6ad9c584875678c34af3d232c4b1ff6020586e2d1f2caddb3bc2359",
    ),
    "z1t": (
        "https://extropic.ai/writing/z1t",
        "0b83813e58e4d351a662df3104ada7ed2e40526477841028579750b5cd229cbc",
    ),
}
engine.HISTORY = {
    "results/experiment_8876_v759_hardware_operation_budget.json": "sha256:25498e6fc2eea0829be36337a084dfc0c1e4933779abd85aa7e6aca8a90d9f2b",
    "openspec/change-proposals/research-methods-v760-20261106.json": "sha256:c99296bd441c10200bad04e77643649bd65f62fb2199029302b24473af6da3d7",
    **{
        p: h
        for p, h in board_history.HISTORY.items()
        if any(n in p for n in ("8230", "8259", "8302"))
    },
    **{f"results/raw/{NAME}/authorities/{h}": "sha256:" + h for _, h in EXTERNAL.values()},
}
engine.SOURCES = OWNED + [
    TEST,
    "research-hardware-wishlist.md",
    "research-references.md",
    "ops/exclusion_manifest.yaml",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/hardware_operation_budget_8838.py",
    "python/carnot/reporting/hardware_operation_runner_8825.py",
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
]
base_collect, base_build = engine.collect, engine.build
checker = namespace(validation_runner, "hardware8889_validation")
checker.m = sys.modules[__name__]


def progress(phase: str) -> None:
    """Flush boundaries so evidence reading remains visible under the fixed deadline."""
    print(f"[exp8889] phase={phase} completed=0 pending=0", flush=True)


def admission(value: Json, branch: str) -> str | None:
    """A null attribution gate retains diagnostic clocks without qualifying fractions."""
    field = PRODUCERS[branch][2]
    if value.get(field) is None:
        return "missing_field"
    if value[field] == 0 and branch != "attribution":
        return "observed_zero"
    if (
        value[field] not in (0, 1)
        or value.get("experiment_id") != PRODUCERS[branch][3]
        or value.get("required_checks_passed") is not True
        or value.get("flagged_adversarial", False)
        or value.get("verdict_class") not in {"positive", "circular_positive", "null"}
        or not str(value.get("honest_verdict", "")).startswith("complete_")
        or not value.get("rows")
        or any(
            r.get("status") not in {"completed", "complete", "censored", "failed"}
            or r.get("denominator", 0) <= 0
            for r in value["rows"]
        )
    ):
        return "rejected_producer"
    return None


def collect(raw: Path) -> Json:
    """Authenticate every original branch and raw operand before using its counters."""
    work: Json = base_collect(raw)
    work["checks"].update(
        installed_dependencies={
            n: importlib.util.find_spec(n) is not None
            for n in ("pytest", "coverage", "ruff", "mypy")
        },
        time_budget_s=4200,
        hard_cap_s=4800,
        physical_probes_performed=0,
        running_repository_virtualenv=Path(sys.prefix) == ROOT / ".venv",
    )
    return work


def amdahl(fraction: float | None) -> Json:
    """A component must exceed 99 percent of total work to permit a finite 100x target."""
    if fraction is not None and not 0 <= fraction <= 1:
        raise ValueError("fraction_out_of_range")
    host = None if fraction is None else 1 - fraction
    return dict(
        qualified_fraction=fraction,
        unaccelerated_host_storage_fraction=host,
        infinite_component_speed_bound=None if host is None or host == 0 else 1 / host,
        required_component_speed_for_100x=None
        if host is None or host >= 0.01
        else fraction / (0.01 - host),
        device_transfer_cost_s=None,
        target_speedup=100,
        missing_reason="missing_qualified_nonoverlapping_fraction" if fraction is None else None,
    )


def reduce(evidence: Json) -> Json:
    """Retain original costs and statuses; overlapping phase totals never become fractions."""
    rows = []
    for branch, producer in sorted(evidence.items()):
        for original in producer["rows"]:
            row = {
                k: original.get(k)
                for k in (
                    "unit_id",
                    "cluster_id",
                    "arm",
                    "seed",
                    "metric",
                    "denominator",
                    "missing_reason",
                )
            }
            row.update(
                source_branch=branch,
                status="completed" if original["status"] == "complete" else original["status"],
                source_status=original["status"],
                operation_counts=original.get("operation_counts"),
                complete_wall_time_s=original.get(
                    "complete_wall_time_s", original.get("complete_ns", 0) / 1e9
                )
                if branch != "labels"
                else None,
                components_ns=original.get("components_ns"),
                repeat=original.get("repeat"),
                boundary=original.get("boundary"),
                total_labels=original.get("total_labels"),
            )
            rows.append(row)
    operations = {
        "parsing": "CPU validation and syntax admission",
        "exact_joins": "CPU integer binding joins; new fabric kernel needed",
        "label_checking": "CPU exact verifier checks; label count is not elapsed time",
        "small_head_arithmetic": "CPU three-way linear logits, normalization and gradient updates; no sampler",
        "template_lookup": "CPU exact normalized-template matching; no qualified device kernel",
        "serialization": "host JSON encoding; separate from storage barriers",
        "fsync": "host file and directory storage barriers",
        "sampling": "zero sampler invocations on these measured paths",
    }
    boards = [
        dict(
            unit_id=f"{board}/{op}",
            cluster_id=board,
            arm=op,
            seed=0,
            metric=None,
            denominator=1,
            status="completed",
            missing_reason=None,
            board=board,
            operation=op,
            operation_status="host-only"
            if op in {"serialization", "fsync"}
            else "unsupported_workload",
            fabric_supported=False,
            measured_board_time_s=None,
            transfer_bytes=None,
            boundary=boundary["boundary"],
            required_operation=description,
        )
        for board, boundary in engine.BOARDS.items()
        for op, description in operations.items()
    ]
    labels, adaptation = evidence.get("labels", {}), evidence.get("adaptation", {})
    standalone = labels.get("standalone_costs", [])
    savings = [dict(r, saved_calls=r["fixed"] - r["sequential"]) for r in standalone]
    fractions = [
        dict(
            source_branch=b,
            operation=op,
            qualified_fraction=None,
            missing_reason="observed_zero_attribution_gate"
            if b == "attribution"
            else "no_disjoint_operation_timer",
            observed_gate=evidence[b][PRODUCERS[b][2]],
        )
        for b in sorted(evidence)
        for op in operations
    ]
    return dict(
        rows=rows,
        board_operation_rows=boards,
        operation_map=operations,
        compatible_fraction_rows=fractions,
        unsupported_operations=[r["unit_id"] for r in boards],
        amdahl_bounds=[dict(source_branch=b, **amdahl(None)) for b in PRODUCERS],
        sampling_call_count=0 if evidence else None,
        sampling_scope="deterministic certificate/linear-update paths; candidate draws are retained label replay, not Ising calls",
        upstream_numerical_training_calls=adaptation.get("numerical_training_calls"),
        label_call_savings=dict(
            status="completed" if labels else "blocked",
            saved_calls=sum(
                r["fixed_total_labels"] - r["sequential_total_labels"]
                for r in labels["shared_union_costs"]
            )
            if labels
            else None,
            standalone_rows=savings,
            shared_union_costs=labels.get("shared_union_costs"),
            actual_label_ledger=labels.get("actual_label_ledger"),
            wall_time_saved_s=None,
            joules_saved=None,
            missing_reason=None if labels else "missing_label_branch",
        ),
        adaptation_phase_costs=adaptation.get("phase_costs"),
        adaptation_durable_commits=adaptation.get("durable_commit_rows"),
        attribution_gates=evidence.get("attribution", {}).get("attribution_gates"),
        unknown_costs=[
            "qualified non-overlapping accelerator-compatible fractions",
            "unaccelerated host/storage timing ceiling",
            "host-device transfer and device runtime",
            "separate small-head/template/serialization/fsync times",
            "joules per operation",
            "NPU/TSU local kernels",
        ],
        end_to_end_gain_bound=None,
    )


def build(work: Json, validation: Json) -> Json:
    """Complete accounting may be ready even when a missing external branch blocks science."""
    value: Json = base_build(work, validation)
    ready = value.pop("hardware_budget_ready_score")
    verdict = value["verdict_class"]
    previous = [
        r
        for r in work["refs"]
        if r["original_path"].endswith("experiment_8876_v759_hardware_operation_budget.json")
    ]
    physical = (
        json.loads(Path(previous[0]["path"]).read_bytes())["physical_evidence_state"]
        if previous
        else engine.BOARDS
    )
    diagnostics = [
        dict(
            check="compatible_fraction_admission",
            upstream=b,
            artifact_field=PRODUCERS[b][2],
            expected=1,
            observed=0,
            path=ref["path"],
            sha256=ref["sha256"],
            missing_reason="observed_zero",
        )
        for b, ref in sorted(work["evidence"].items())
        if json.loads(Path(ref["path"]).read_bytes())[PRODUCERS[b][2]] == 0
    ]
    blocked = [
        dict(
            unit_id=r["upstream"],
            cluster_id=r["upstream"],
            arm=r["upstream"],
            seed=7608889,
            metric=None,
            denominator=1,
            status="blocked",
            missing_reason=r["missing_reason"],
            source_gate=r,
        )
        for r in work["failures"]
    ]
    value["rows"].extend(blocked)
    value.update(
        experiment_id=8889,
        task_id="exp8889-hardware-operation-map",
        milestone="2026.11.760",
        run_date="20261107",
        random_seed=7608889,
        honest_verdict="complete_" + verdict + "_hardware_operation_map",
        hardware_map_ready_score=ready,
        independent_count=0,
        intended_count=len(value["rows"]),
        blocked_count=len(blocked),
        invocation=work.get("invocation", {}),
        cited_upstream_artifacts=[ref for _, ref in sorted(work["evidence"].items())],
        physical_evidence_state=physical,
        gate_check_summary=work["failures"] + diagnostics,
        external_source_receipts={
            k: dict(
                url=url,
                sha256="sha256:" + h,
                checked_date="20261107",
                scope="vendor announcement; no local performance credit",
            )
            for k, (url, h) in EXTERNAL.items()
        },
        hardware_learning_path=dict(
            parsing="CPU parser; no FPGA parser measured",
            exact_joins="CPU exact binding checks",
            labels="retained oracle label replay; savings do not imply joules or time savings",
            small_head="21 linear parameters; batching hypothesis requires exact parity and full durable costs",
            templates="CPU normalized-string lookups; no installed accelerator kernel",
            durability="host serialization and file/directory fsync remain paid",
            sampling="zero Ising calls; quadratic k_max<=5 support cannot accelerate absent sampler work",
            research_100x="target only; missing qualified fractions leave ceiling and required component speed null",
        ),
        acquisition_priority=[
            dict(
                rank=1,
                candidate="existing CPU/native batching",
                status="future_hypothesis",
                measured_workload="Exp8888 48 durable adaptation streams and Exp8882 installed requests",
                condition="measure disjoint arithmetic versus host/storage time before acquisition",
                purchase_authorized=False,
            ),
            dict(
                rank=2,
                candidate="Anechoic Alveo V80",
                status="future_hypothesis",
                measured_workload="zero Max-Cut or Ising calls in Exp8882/8885/8888",
                condition="requires actual dense Max-Cut demand, exact quality and complete transfer/host measurements",
                purchase_authorized=False,
            ),
            dict(
                rank=3,
                candidate="Extropic Torx/Thermalizers/Z1",
                status="future_hypothesis",
                measured_workload="deterministic label checks and linear gradients; zero thermodynamic sampler calls",
                condition="requires stochastic reformulation, approximation/error parity and local device access",
                purchase_authorized=False,
            ),
        ],
        external_compatibility=dict(
            anechoic=dict(
                url="https://arxiv.org/abs/2610.13368",
                operation="synchronous Onsager-corrected dense Max-Cut on Alveo V80",
                local_support=False,
            ),
            torx=dict(
                url="https://github.com/extropic-ai/torx",
                operation="stochastic transition-kernel programming",
                local_support=False,
            ),
            thermalizers_z1=dict(
                url="https://extropic.ai/writing/from-one-to-one-billion",
                operation="variational compilation to thermodynamic kernels; distinct from exact joins and fsync",
                local_support=False,
            ),
            z1t=dict(
                url="https://extropic.ai/writing/z1t",
                operation="heterogeneous sparse probabilistic model; no local deterministic head kernel",
                local_support=False,
            ),
        ),
        methodology_note="Independent byte-authenticated CPU producer evidence. Attribution closure and observer gates fail, so raw clocks are diagnostic only. Adaptation phase clocks include durable work and do not isolate device-compatible arithmetic. Labels saved, seconds saved and joules saved are distinct. No model loading, current numerical training, board probes, purchase, milestone activation or independent generalization occurs.",
    )
    for status, field in [
        ("completed", "completed_count"),
        ("censored", "censored_count"),
        ("failed", "failed_count"),
    ]:
        value[field] = sum(r["status"] == status for r in value["rows"])
    value["field_principles"].update(
        {
            k: "Retain exact producer scope; complete accounting is separate from scientific benefit and unmeasured device speed."
            for k in value
            if k not in value["field_principles"]
        }
    )
    return value


engine.progress, engine.admission, engine.reduce, engine.build = progress, admission, reduce, build
replay = engine.replay


def main(argv: list[str] | None = None) -> int:
    """Publish checked terminal bytes after private validation and cold replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261107"], default="20261107")
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
    output = (args.private_output or DEFAULT_OUTPUT).absolute()
    raw = output.parent / "raw" / NAME / "invocations" / str(time.monotonic_ns())
    template = ExperimentTemplate(
        8889, "Hardware operation map", output.name, repo_root=output.parent, seed=7608889
    )
    progress("operand_admission_before")
    with template.phase("operand_admission"):
        work = collect(raw)
    work["invocation"] = dict(
        argv=[sys.executable, str(ROOT / CLI), *(sys.argv[1:] if argv is None else argv)],
        owner_pid=work["owner_pid"],
        run_id=str(work["started_ns"]),
    )
    atomic_json(raw / "work.json", work)
    progress("owned_validation_before")
    validation = (
        dict(passed=False, receipts=[], coverage=dict(complete=False))
        if args.private_output
        else checker.validate(raw)
    )
    progress("owned_validation_after")
    value = build(work, validation)
    progress("publication_before")
    report = publish_primary(
        output, value, lambda p: replay(p) if args.private_output else checker.terminal(p, raw)
    )
    atomic_json(raw / "publication.json", report)
    progress("complete")
    return 0
