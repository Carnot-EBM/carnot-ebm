"""REQ-REPORT-8928: qualify hardware evidence without inventing removable work.

Retained current operation counts and historical clocks have different scopes.
The existing custody and publication helpers enforce the same bytes on replay.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import hardware_operation_budget_8838 as shared
from carnot.reporting import hardware_operation_runner_8825 as validation_runner
from carnot.reporting.hardware_operation_budget_8915 import amdahl
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
ROOT = shared.ROOT
NAME, TASK = "experiment_8928_v763_hardware_operation_budget", "exp8928-hardware-operation-budget"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = ["python/carnot/reporting/hardware_operation_budget_8928.py", CLI]
TEST = "tests/python/test_hardware_operation_budget_8928.py"
DEFAULT_OUTPUT = ROOT / "results" / (NAME + ".json")
PRODUCERS = {
    "request": (
        "results/experiment_8927_v763_inline_request_cost.json",
        "unavailable_at_method_freeze",
        "cost_attribution_ready_score",
        8927,
    ),
    "durable": (
        "results/experiment_8926_v763_durable_cost_qualification.json",
        "sha256:1c5f1969924341ce5416db9c649265dac03985946d722e111792b03372cb5a24",
        "adaptation_cost_ready_score",
        8926,
    ),
    "benefit": (
        "results/experiment_8925_v763_interpolated_benefit_audit.json",
        "sha256:88f6d845ac54d3ff1b553e135b4d8bf416cf714bbbbd1e73bcc0acb1635c7269",
        "interpolated_benefit_score",
        8925,
    ),
    "causal": (
        "results/experiment_8922_v763_interpolated_causal_streams.json",
        "sha256:53a63467e2cbad29541ba7f3b3bb3b8aec5fa091ca54a1df26aed63e106c710b",
        "causal_rows_ready_score",
        8922,
    ),
    "clock": (
        "results/experiment_8921_v763_inline_clock_methods.json",
        "sha256:93d1af0f09146a17914d9b3730ff84e3d6289d30c38dec9bb49470900e97fc19",
        "inline_clock_ready_score",
        8921,
    ),
    "labels": (
        "results/experiment_8920_v763_label_demand_methods.json",
        "sha256:6fa9b66aa01b265ec5097f0ab8910e0281b42a3f4eda686aec3b17493d4691a3",
        "label_demand_ready_score",
        8920,
    ),
    "gpu": (
        "results/experiment_8918_v763_bfs_error_pilot.json",
        "sha256:b2d62a2d78535ad3bf5ca9aa83c330f2490ed7d40941a7a42b18a489aafd800e",
        "qwen_runtime_ready_score",
        8918,
    ),
    "request_skip": (
        "results/experiment_8927_inline_request_cost.json",
        "sha256:353ea1de2edc7f3a85e6ceebdc7598434d0ae4cfea9703c00307b6cfddac1399",
        "failed_observed",
        8927,
    ),
    "label_skip": (
        "results/experiment_8923_label_demand_evaluation.json",
        "sha256:7a86cdde9ed000881c9ee292f3223e2c529fb8177f8ffbfcd013ca581574b99b",
        "failed_observed",
        8923,
    ),
}
TASKS = dict(
    request="exp8927-inline-request-cost",
    durable="exp8926-durable-cost-qualification",
    benefit="exp8925-interpolated-benefit-audit",
    causal="exp8922-interpolated-causal-streams",
    clock="exp8921-inline-clock-methods",
    labels="exp8920-label-demand-methods",
    gpu="exp8918-bfs-error-pilot",
    request_skip="exp8927-inline-request-cost",
    label_skip="exp8923-label-demand-evaluation",
)
BOUNDARIES = {
    "CPU_Tier_1": "Counters, symbolic joins, vector updates, guards and durable state remain CPU work; optimize only after qualified disjoint clocks.",
    "GPU_0": "Actual Qwen BFS workload receipts exist; small-head batching needs its own equivalent workload, transfer and complete-service receipts.",
    "GPU_1": "Detached; no workload or transfer qualification.",
    "NPU": "Unqualified. Wishlist condition: working mlir-aie or VitisAI SDK plus equivalent batched vector kernel, parity and complete-service benchmark.",
    "KV260": "Supported quadratic Ising only, k_max<=5. Wishlist condition: an equivalent sampling kernel with parity, host/transfer/durability receipts; current deterministic work has no such kernel.",
    "PolarFire": "Linux dispatch executes on CPU. Wishlist condition: separately authenticated fabric kernel and matched complete workload; Linux dispatch cannot qualify fabric.",
    "GateMate": "Unchanged JTAG 0xffffffff physical blocker. Wishlist condition: operator cable/port/power change then authenticated supported kernel; no retry here.",
    "Extropic_TSU": "Unqualified. Wishlist condition: authenticated hardware/SDK access, supported sampling kernel, sample quality and complete-service receipts; simulation/vendor claims do not qualify.",
}
engine = namespace(shared, "hardware8928")
engine.NAME, engine.CLI, engine.OWNED, engine.TEST = NAME, CLI, OWNED, TEST
engine.PRODUCERS, engine.HISTORY = PRODUCERS, {}
engine.SOURCES = [
    *OWNED,
    TEST,
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/hardware_operation_budget_8838.py",
    "python/carnot/reporting/hardware_operation_budget_8915.py",
    "python/carnot/reporting/hardware_operation_runner_8825.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "python/carnot/reporting/v748_collection_contract.py",
    "python/carnot/memory/interpolated_memory_8919.py",
    "research-program.md",
    "research-hardware-wishlist.md",
    "ops/north-star.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "ops/e2e-test-plan.md",
]
base_build = engine.build
checker = namespace(validation_runner, "hardware8928_validation")
checker.m = sys.modules[__name__]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries and counts so the supervisor can observe progress."""
    print(f"[exp8928] phase={phase} completed={completed} pending={pending}", flush=True)


def admission(value: Json, branch: str) -> str | None:
    """A valid zero supplies diagnostics; failed owned controls cannot qualify costs."""
    _, _, field, identity = PRODUCERS[branch]
    if value.get(field) is None:
        return "missing_field"
    if branch.endswith("skip"):
        source = "clock" if branch == "request_skip" else "labels"
        return (
            None
            if (
                value.get("schema") == "blocked_gate_check_v1"
                and value.get("experiment") == identity
                and value.get("blocked_at_layer") == "conductor_pre_gate"
                and value.get("failed_upstream") == TASKS[source]
                and value.get("failed_field") == PRODUCERS[source][2]
                and value.get("failed_evidence_sha256") == PRODUCERS[source][1]
                and value.get("failed_evidence_path") == str(ROOT / PRODUCERS[source][0])
                and type(value[field]) is int
                and value[field] == 0
                and type(value.get("failed_expected")) is int
                and value["failed_expected"] == 1
            )
            else "rejected_producer"
        )
    if (
        type(value[field]) is not int
        or value[field] not in (0, 1)
        or value.get("experiment_id") != identity
        or value.get("task_id") != TASKS[branch]
        or value.get("milestone") != "2026.11.763"
        or value.get("run_date") != "20261108"
        or value.get("required_checks_passed") is not True
        or value.get("flagged_adversarial", False) is not False
        or value.get("verdict_class") not in {"null", "positive", "circular_positive"}
        or (value.get("verifier_is_oracle") is True and value["verdict_class"] == "positive")
        or not str(value.get("honest_verdict", "")).startswith("complete_")
        or not value.get("rows")
        or any(
            r.get("status") not in {"complete", "completed", "failed", "censored"}
            or r.get("denominator", 0) <= 0
            for r in value["rows"]
        )
    ):
        return "rejected_producer"
    return None


def collect(raw: Path) -> Json:
    """Freeze all available evidence, including rejected producers and their operands."""
    started = time.monotonic_ns()
    originals, candidates, hashes = {}, {}, {}
    for branch, (name, pin, _, _) in PRODUCERS.items():
        path = ROOT / name
        if path.is_file() and sha256_file(path) == pin:
            value = json.loads(path.read_bytes())
            originals[branch] = value
            for key in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"):
                refs = value.get(key, [])
                for ref in refs.values() if isinstance(refs, dict) else refs:
                    operand = Path(ref["path"])
                    if operand not in hashes:
                        hashes[operand] = sha256_file(operand) if operand.is_file() else None
                    if hashes[operand] == ref["sha256"]:
                        candidates[ref["sha256"]] = operand
            progress("index_original_operands", len(originals), len(PRODUCERS) - len(originals))

    def retain(path: Path, destination: Path, expected: str | None = None) -> Json:
        """Reuse authenticated originals when an append-only document has changed."""
        observed = hashes.get(path)
        if observed is None and path.is_file():
            observed = hashes[path] = sha256_file(path)
        source = candidates.get(expected, path) if expected and observed != expected else path
        digest = sha256_file(source)
        if expected and digest != expected:
            raise ValueError("operand_hash_drift:" + str(path))
        target = destination / "operands" / digest.split(":")[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
        if sha256_file(target) != digest:
            raise ValueError("immutable_operand_drift")
        return dict(path=str(target.absolute()), sha256=digest, original_path=str(path))

    engine.retain = retain
    work: Json = engine.collect(raw)
    work["diagnostic_evidence"] = {}
    for branch, value in originals.items():
        name, pin, _, _ = PRODUCERS[branch]
        ref = retain(ROOT / name, raw, pin)
        work["diagnostic_evidence"][branch] = ref
        work["refs"].append(ref)
        for key in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"):
            refs = value.get(key, [])
            for operand in refs.values() if isinstance(refs, dict) else refs:
                try:
                    work["refs"].append(retain(Path(operand["path"]), raw, operand["sha256"]))
                except (OSError, ValueError) as error:
                    work["failures"].append(
                        dict(
                            check="diagnostic_operand",
                            upstream=TASKS[branch],
                            artifact_field=key,
                            expected=operand["sha256"],
                            observed=str(error),
                            path=operand["path"],
                            sha256=None,
                            missing_reason="rejected_producer",
                        )
                    )
        progress("retained_original", len(work["refs"]), 0)
    work["refs"] = list({(r["original_path"], r["sha256"]): r for r in work["refs"]}.values())
    work["checks"].update(physical_probes_performed=0, hard_cap_s=4800, pilot_budget_s=4200)
    work.update(started_ns=started, ended_ns=time.monotonic_ns())
    work["phase_spans"] = [
        dict(phase="operand_admission", start_s=0, end_s=(work["ended_ns"] - started) / 1e9)
    ]
    return work


def reduce(evidence: Json) -> Json:
    """Count current work without treating counts or inclusive clocks as fractions."""
    causal = evidence.get("causal", {})
    ledger = causal.get("gradient_call_ledger", [])
    operations = []
    for arm in sorted({r["arm"] for r in ledger}):
        selected = [r for r in ledger if r["arm"] == arm]
        counts = dict(
            gradient=sum(r["calls"] for r in selected),
            fast_slow_interpolation=sum(r["update_count"] for r in selected),
            guard_scoring=sum(r["guard_scores"] for r in selected),
            persistence=sum(r["arm"] == arm for r in causal["optimizer_sequences"]),
            feature_checks=sum(
                r["primitive_checks"] for r in causal["prediction_rows"] if r["arm"] == arm
            ),
            label_oracle=sum(r["arm"] == arm for r in causal["label_call_rows"]),
            evaluator_oracle=sum(r["arm"] == arm for r in causal["scoring_rows"]),
            sampling=0,
        )
        for operation, count in counts.items():
            operations.append(
                dict(
                    unit_id=arm + "/" + operation,
                    cluster_id="current_operation_inventory",
                    arm=arm,
                    seed=7638928,
                    metric=count,
                    denominator=1,
                    status="completed",
                    missing_reason=None,
                    operation=operation,
                    source_branch="causal",
                    count_scope="retained current calls/transactions; no timing or independent n",
                )
            )
    operations.append(
        dict(
            unit_id="policy_mask",
            cluster_id="current_operation_inventory",
            arm="policy_mask",
            seed=7638928,
            metric=None,
            denominator=1,
            status="completed",
            missing_reason="disqualified_label_methods_and_skipped_evaluation",
            operation="policy_mask",
            source_branch="labels",
            count_scope="no qualified policy-mask count; label-oracle acquisitions counted separately",
        )
    )
    compatibility = [
        dict(
            unit_id=board,
            cluster_id=board,
            arm="supported_scope",
            seed=7638928,
            metric=None,
            denominator=1,
            status="completed",
            missing_reason="no_measured_compatible_kernel",
            board=board,
            boundary=boundary,
            fabric_supported=False,
            wishlist_condition=boundary,
        )
        for board, boundary in BOUNDARIES.items()
    ]
    durable = evidence.get("durable", {})
    costs = [
        dict(
            r,
            unit_id="historical/" + r["unit_id"],
            source_branch="durable",
            cost_scope="historical_guarded_only",
            interpolated_optimizer_cost_measured=False,
        )
        for r in durable.get("service_cost_rows", [])
    ]
    request = evidence.get("request", {})
    qualified = (
        bool(request)
        and admission(request, "request") is None
        and request["cost_attribution_ready_score"] == 1
    )
    return dict(
        rows=operations + costs + compatibility,
        operation_rows=operations,
        compatibility_rows=compatibility,
        whole_service_rows=costs,
        inclusive_timer_rows=[
            dict(r, additive_partition=False, cost_scope="historical_guarded_only")
            for r in durable.get("inclusive_timer_rows", [])
        ],
        qualified_component_rows=request.get("component_rows", []) if qualified else [],
        measured_fraction_rows=[],
        unknown_fraction_rows=[
            dict(
                board=b,
                fraction=None,
                missing_reason="no_measured_compatible_fraction",
                denominator="complete service including host, transfers, startup and durability",
                required_evidence=boundary,
            )
            for b, boundary in BOUNDARIES.items()
        ],
        amdahl_rows=[
            dict(
                board=b,
                compatible_fraction=None,
                kernel_speedup=100,
                ceiling_100x=amdahl(None),
                feasible_10x=None,
                host_fraction=None,
                transfer_fraction=None,
                durability_fraction=None,
                claim="hypothetical ceiling unknown; no measured acceleration",
            )
            for b in BOUNDARIES
        ],
        sampling_call_count=0 if ledger else None,
        physical_acceleration_demonstrated=False,
        nfr_01=dict(
            whole_service_target_speedup=10,
            proved=False,
            feasibility="unknown_without_measured_compatible_fraction",
            required_compatible_fraction_for_100x_kernel=10 / 11,
            condition="For measured compatible f only: 1/((1-f)+f/100) >= 10; host, transfer and durability remain paid. Zero sampler calls supply no removable fraction.",
        ),
    )


def build(work: Json, validation: Json) -> Json:
    """Separate complete accounting from upstream benefit and physical acceleration."""
    value: Json = base_build(work, validation)
    receipt = build_current_work_receipt(
        run_id=str(work["started_ns"]),
        owner_pid=work["owner_pid"],
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(operation="CPU evidence accounting and owned validation"),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["started_ns"],
        ended_monotonic_ns=work["ended_ns"],
        phase_spans=work["phase_spans"],
    )
    value.update(
        current_work_receipt=receipt,
        duration_s=receipt["duration_s"],
        phase_spans=receipt["phase_spans"],
    )
    refs = work["diagnostic_evidence"]
    evidence = {b: json.loads(Path(r["path"]).read_bytes()) for b, r in refs.items()}
    diagnostics = list(work["failures"])
    for branch, producer in sorted(evidence.items()):
        _, _, field, _ = PRODUCERS[branch]
        if producer.get(field) == 0:
            source = (
                "clock"
                if branch == "request_skip"
                else "labels"
                if branch == "label_skip"
                else branch
            )
            diagnostics.append(
                dict(
                    check="scheduler_skip" if branch.endswith("skip") else "producer_gate",
                    upstream=TASKS[source],
                    artifact_field=PRODUCERS[source][2],
                    expected=1,
                    observed=0,
                    path=refs[branch]["original_path"],
                    sha256=refs[branch]["sha256"],
                    missing_reason="observed_zero",
                )
            )
    verdict = (
        "disqualified"
        if value["verdict_class"] == "disqualified"
        else "blocked"
        if diagnostics
        else "null"
    )
    value.update(
        experiment_id=8928,
        task_id=TASK,
        milestone="2026.11.763",
        run_date="20261108",
        honest_verdict="complete_" + verdict + "_hardware_operation_budget",
        verdict_class=verdict,
        random_seed=7638928,
        execution_venue="host",
        hardware_map_ready_score=value.pop("hardware_budget_ready_score"),
        gate_check_summary=diagnostics,
        invocation=work.get("invocation", {}),
        physical_evidence_state={
            b: dict(
                boundary=boundary,
                actual_workload_receipt=work["evidence"].get("gpu") if b == "GPU_0" else None,
                current_device_probe=False,
                optimizer_batching_qualified=False,
            )
            for b, boundary in BOUNDARIES.items()
        },
        preserved_upstream_assertions={
            b: {
                k: p.get(k)
                for k in (
                    "honest_verdict",
                    "verdict_class",
                    "required_checks_passed",
                    "acceptance_gates",
                    "gate_check_summary",
                    "owned_statement_coverage",
                    "intended_count",
                    "completed_count",
                    "independent_count",
                )
            }
            for b, p in evidence.items()
        },
        cited_upstream_artifacts=[
            dict(
                r,
                producer_task_id=TASKS[b],
                source_admitted=b in work["evidence"],
                fields_imported=[PRODUCERS[b][2], "operation_counts_or_historical_costs_only"],
            )
            for b, r in sorted(refs.items())
        ],
        methodology_note="Unconditional CPU evidence map. Qualified historical guarded costs never time the new fast/slow optimizer. New counts are retained CPU Tier 1 work. Disqualified label methods and scheduler skips remain distinct. GPU0 BFS receipts qualify only that actual workload; GPU/NPU batching needs separate parity and complete-service receipts. FPGA sampling remains a separate supported-kernel path. No model load, numerical training, device probe, flash, purchase or timing rerun occurs. No physical speedup or independent natural-language generalization is established.",
    )
    value["reproducibility_checksum"] = canonical_hash(dict(rows=value["rows"], refs=work["refs"]))
    value["field_principles"].update(
        {
            k: "Preserve exact producer scope; unknown compatible fractions stay null and current inventory establishes no acceleration."
            for k in value
            if k not in value["field_principles"]
        }
    )
    return value


engine.progress, engine.admission, engine.reduce, engine.build = progress, admission, reduce, build
replay = engine.replay


def main(argv: list[str] | None = None) -> int:
    """Run actual private checks before atomically publishing the terminal map."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261108"], default="20261108")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--replay", "--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError) as error:
            print(json.dumps(dict(passed=False, reason=str(error))), flush=True)
            return 1
    output = (args.private_output or DEFAULT_OUTPUT).absolute()
    raw = output.parent / "raw" / NAME / "invocations" / str(time.monotonic_ns())
    template = ExperimentTemplate(
        8928, "Hardware operation budget", output.name, repo_root=output.parent, seed=7638928
    )
    progress("authentication_before")
    with template.phase("operand_admission"):
        work = collect(raw)
    work["invocation"] = dict(
        argv=sys.orig_argv if argv is None else [sys.executable, str(ROOT / CLI), *argv],
        cwd=str(Path.cwd()),
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
    validation_end = time.monotonic_ns()
    work["phase_spans"].append(
        dict(
            phase="owned_validation",
            start_s=(work["ended_ns"] - work["started_ns"]) / 1e9,
            end_s=(validation_end - work["started_ns"]) / 1e9,
        )
    )
    work["ended_ns"] = validation_end
    atomic_json(raw / "work.json", work)
    value = build(work, validation)
    progress("publication_before")
    report = publish_primary(
        output, value, lambda p: replay(p) if args.private_output else checker.terminal(p, raw)
    )
    atomic_json(raw / "publication.json", report)
    progress("complete")
    return 0
