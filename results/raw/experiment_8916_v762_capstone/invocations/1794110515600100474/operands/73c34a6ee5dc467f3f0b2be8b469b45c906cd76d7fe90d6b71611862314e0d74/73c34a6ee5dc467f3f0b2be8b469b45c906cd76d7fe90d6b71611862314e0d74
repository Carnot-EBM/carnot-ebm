"""REQ-REPORT-8915: preserve unavailable evidence without inventing device timing."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.reporting import hardware_operation_budget_8838 as shared
from carnot.reporting import hardware_operation_runner_8825 as validation_runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
ROOT = shared.ROOT
NAME = "experiment_8915_v762_hardware_operation_budget"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/hardware_operation_budget_8915.py", CLI]
TEST = "tests/python/test_hardware_operation_budget_8915.py"
DEFAULT_OUTPUT = ROOT / "results" / (NAME + ".json")
TASKS = dict(
    request="exp8912-complete-request-cost",
    durable="exp8914-guarded-durable-cost",
    benefit="exp8911-guarded-benefit-audit",
)
PRODUCERS = {
    "request": (
        "results/experiment_8912_v762_complete_request_cost.json",
        "sha256:0a468088fa1dd6d95fd88199fc1276ff83bb6fd1f0a43572e1bf58310d53a57f",
        "cost_attribution_ready_score",
        8912,
    ),
    "durable": (
        "results/experiment_8914_v762_guarded_durable_cost.json",
        "sha256:e4bc58031745b08524ed21e5c3d7ddf84329d7d88aaf0be2ad9ceac14034d479",
        "adaptation_cost_ready_score",
        8914,
    ),
    "benefit": (
        "results/experiment_8911_v762_guarded_benefit_audit.json",
        "sha256:28f12afb613a933f778248172532fa84c71f474b631b3c9aaa9fd4d296048d74",
        "guarded_benefit_score",
        8911,
    ),
}
HISTORY = {
    "results/experiment_8902_v761_hardware_operation_map.json": "sha256:8be6cb316af9d9711f62e4eda7bbfd63b0402e3e38c1f4243cf714566431a0f2",
    "results/experiment_8904_v762_branch_contract.json": "sha256:6da560933050f3f9a35c0230dc0cc7151c6b4edcf19f272db7202f41c8dace18",
}
OPERATIONS = {
    "typed_source_parsing": "CPU syntax/schema checks and decoded objects",
    "relational_joins_witness_search": "CPU exact signed atom and binding intersections; no quadratic Ising kernel",
    "memory_lookup": "CPU content-addressed templates and stored state",
    "signed_features": "CPU integer/string comparisons and 27-coordinate signed features",
    "three_class_normalization": "CPU three-logit exact softmax; no sampling",
    "candidate_gradients": "CPU float32 small-head gradient updates; rejected work still paid",
    "guard_scoring": "CPU held-out candidate/deployed predictions and promotion decisions",
    "atomic_persistence": "Host JSON, file fsync, atomic rename and directory fsync",
    "sampling": "No Ising sampling in these deterministic request and learner paths",
}
BOUNDARIES = {
    "KV260": "Quadratic Ising kernels only, k_max<=5; current operations have no supported mapping",
    "PolarFire": "Linux dispatch executes on CPU; fabric work unqualified",
    "GateMate": "Physical JTAG0xffffffff block; changed cable/port/power evidence required before retry",
    "GPU": "External ops/cuda-runtime-repair-receipt.json with causal changes and actual workload/offload required",
    "Extropic": "Hardware access unavailable; vendor and simulator performance cannot substitute",
}
engine = namespace(shared, "hardware8915")
engine.NAME, engine.CLI, engine.OWNED, engine.TEST = NAME, CLI, OWNED, TEST
engine.PRODUCERS, engine.HISTORY = PRODUCERS, HISTORY
engine.SOURCES = [
    *OWNED,
    TEST,
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/hardware_operation_budget_8838.py",
    "python/carnot/reporting/hardware_operation_runner_8825.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "python/carnot/reporting/v748_collection_contract.py",
    "research-program.md",
    "research-hardware-wishlist.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "ops/e2e-test-plan.md",
]
base_build = engine.build
checker = namespace(validation_runner, "hardware8915_validation")
checker.m = sys.modules[__name__]


def progress(phase: str) -> None:
    """Show real boundaries while the original deadline continues to run."""
    print(f"[exp8915] phase={phase} completed=0 pending=0", flush=True)


def admission(value: Json, branch: str) -> str | None:
    """Valid zero gates supply diagnostics; failed controls cannot qualify costs."""
    field = PRODUCERS[branch][2]
    if value.get(field) is None:
        return "missing_field"
    if (
        type(value[field]) is not int
        or value[field] not in (0, 1)
        or value.get("experiment_id") != PRODUCERS[branch][3]
        or value.get("task_id") != TASKS[branch]
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
    """Freeze original bytes even for rejected producers; never rerun their work."""
    candidates = {}
    for name, pin, _, _ in PRODUCERS.values():
        path = ROOT / name
        if path.is_file() and sha256_file(path) == pin:
            value = json.loads(path.read_bytes())
            for key in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"):
                refs = value.get(key, [])
                for ref in refs.values() if isinstance(refs, dict) else refs:
                    candidate = Path(ref["path"])
                    if candidate.is_file() and sha256_file(candidate) == ref["sha256"]:
                        candidates[ref["sha256"]] = candidate

    def retain(path: Path, destination: Path, expected: str | None = None) -> Json:
        """Use retained originals when an append-only document has changed."""
        source = path
        if expected and (not path.is_file() or sha256_file(path) != expected):
            source = candidates.get(expected, path)
        data = source.read_bytes()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        if (
            expected == "sha256:1580bedce5dafc151968211bbbbe61965176b9454e98c259fddc4e3b13fae21d"
            and digest != expected
        ):
            progress("original_spec_git_before")
            data = subprocess.check_output(
                [
                    "git",
                    "show",
                    "da52fd97bdddf596899a84134f5a9f66b47470ab:openspec/capabilities/research-reporting/spec.md",
                ],
                cwd=ROOT,
                timeout=15,
            )
            progress("original_spec_git_after")
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
        if expected and digest != expected:
            raise ValueError("operand_hash_drift:" + str(path))
        target = destination / "operands" / digest.split(":")[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError("immutable_operand_drift")
        if not target.exists():
            target.write_bytes(data)
            target.chmod(0o444)
        return dict(path=str(target.absolute()), sha256=digest, original_path=str(path))

    engine.retain = retain
    work: Json = engine.collect(raw)
    work["diagnostic_evidence"] = {}
    for branch, (name, pin, _, _) in PRODUCERS.items():
        path = ROOT / name
        if branch not in work["evidence"] and path.is_file() and sha256_file(path) == pin:
            ref = retain(path, raw, pin)
            work["diagnostic_evidence"][branch] = ref
            work["refs"].append(ref)
            value = json.loads(Path(ref["path"]).read_bytes())
            for key in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"):
                refs = value.get(key, [])
                for i, operand in enumerate(refs.values() if isinstance(refs, dict) else refs):
                    try:
                        work["refs"].append(retain(Path(operand["path"]), raw, operand["sha256"]))
                    except (OSError, ValueError) as exc:
                        work["failures"].append(
                            dict(
                                check="diagnostic_operand",
                                upstream=branch,
                                artifact_field=key,
                                expected=operand["sha256"],
                                observed=str(exc),
                                path=operand["path"],
                                sha256=None,
                                missing_reason="rejected_producer",
                            )
                        )
                    if i % 100 == 0:
                        progress(f"diagnostic_operands_{branch}_{i}")
    work["checks"].update(physical_probes_performed=0, hard_cap_s=4800, pilot_budget_s=4200)
    return work


def amdahl(fraction: float | None) -> float | None:
    """A hypothetical kernel improvement needs a measured share of whole service."""
    if fraction is None:
        return None
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("invalid_compatible_fraction")
    return 1 / ((1 - fraction) + fraction / 100)


def reduce(evidence: Json) -> Json:
    """Separate operation inventory, whole clocks, inclusive clocks and unknown fractions."""
    operations = [
        dict(
            unit_id=op,
            cluster_id="operation_inventory",
            arm=op,
            seed=7628915,
            metric=None,
            denominator=1,
            status="completed",
            missing_reason=None,
            description=description,
        )
        for op, description in OPERATIONS.items()
    ]
    compatibility = [
        dict(
            unit_id=f"{board}/{op}",
            cluster_id=board,
            arm=op,
            seed=7628915,
            metric=None,
            denominator=1,
            status="completed",
            missing_reason="no_measured_compatible_kernel",
            operation=op,
            board=board,
            boundary=boundary,
            fabric_supported=False,
            changed_evidence_required="Authenticate a supported equivalent kernel and measure full host/transfer/durability workload with parity; "
            + boundary,
        )
        for board, boundary in BOUNDARIES.items()
        for op in OPERATIONS
    ]
    costs = []
    for branch, producer in sorted(evidence.items()):
        source_rows = (
            producer.get("rows", [])
            if branch == "request"
            else producer.get("service_cost_rows", [])
            if branch == "durable"
            else []
        )
        for r in source_rows:
            ns = r.get("parent_elapsed_ns") if branch == "request" else r.get("complete_ns")
            costs.append(
                dict(
                    unit_id=branch + "/" + r["unit_id"],
                    upstream_unit_id=r["unit_id"],
                    cluster_id=r["cluster_id"],
                    arm=r["arm"],
                    seed=r["seed"],
                    denominator=r["denominator"],
                    status="completed" if r["status"] == "complete" else r["status"],
                    missing_reason=r.get("missing_reason"),
                    metric=None if ns is None else ns / 1e9,
                    whole_service_ns=ns,
                    source_branch=branch,
                    source_admitted=admission(producer, branch) is None,
                    upstream_status=r["status"],
                    scope="whole service including paid host/startup/transport/durability; never a compatible fraction",
                )
            )
    qualified = (
        evidence.get("request", {}).get("cost_attribution_ready_score") == 1
        and admission(evidence["request"], "request") is None
    )
    inclusive = [
        dict(
            r,
            source_admitted=admission(evidence["durable"], "durable") is None,
            additive_partition=False,
        )
        for r in evidence.get("durable", {}).get("inclusive_timer_rows", [])
    ]
    return dict(
        rows=costs + operations + compatibility,
        operation_rows=operations,
        compatibility_rows=compatibility,
        whole_service_rows=costs,
        inclusive_timer_rows=inclusive,
        qualified_component_rows=evidence.get("request", {}).get("component_rows", [])
        if qualified
        else [],
        measured_fraction_rows=[],
        unknown_fraction_rows=[
            dict(
                board=r["board"],
                operation=r["operation"],
                fraction=None,
                missing_reason="no_measured_compatible_fraction",
                denominator="complete parent/service wall time; host, transfers and durability included",
                required_evidence=r["changed_evidence_required"],
            )
            for r in compatibility
        ],
        amdahl_rows=[
            dict(
                board=board,
                compatible_fraction=None,
                kernel_speedup=100,
                ceiling_100x=amdahl(None),
                host_fraction=None,
                transfer_fraction=None,
                durability_fraction=None,
                claim="analytical ceiling unknown; no measured device speed or service gain",
            )
            for board in BOUNDARIES
        ],
        sampling_call_count=0 if evidence else None,
        physical_acceleration_demonstrated=False,
        nfr_01=dict(
            whole_service_target_speedup=10,
            proved=False,
            required_compatible_fraction_for_100x_kernel=10 / 11,
            next_condition="Qualify disjoint component clocks and an equivalent supported kernel; measure matched full-service latency including host, transfers and durable commits. GPU also needs external changed-runtime receipt and actual workload; benefit requires authentic guarded_benefit_score=1.",
        ),
    )


def build(work: Json, validation: Json) -> Json:
    """A complete map may be blocked; failed current controls remain disqualified."""
    value: Json = base_build(work, validation)
    ready = value.pop("hardware_budget_ready_score")
    refs = {**work.get("diagnostic_evidence", {}), **work["evidence"]}
    evidence = {b: json.loads(Path(ref["path"]).read_bytes()) for b, ref in refs.items()}
    value.update(reduce(evidence))
    diagnostics = deepcopy(work["failures"])
    for branch, producer in sorted(evidence.items()):
        field = PRODUCERS[branch][2]
        if producer.get(field) == 0:
            diagnostics.append(
                dict(
                    check="producer_gate",
                    upstream=TASKS[branch],
                    artifact_field=field,
                    expected=1,
                    observed=0,
                    path=refs[branch]["original_path"],
                    sha256=refs[branch]["sha256"],
                    missing_reason="observed_zero",
                )
            )
        diagnostics.extend(
            dict(r, missing_reason=r.get("missing_reason", r["check"]))
            for r in producer.get("gate_check_summary", [])
        )
    for ref in work["refs"]:
        if ref["original_path"].endswith("experiment_8904_v762_branch_contract.json"):
            producer = json.loads(Path(ref["path"]).read_bytes())
            diagnostics.extend(
                dict(r, missing_reason=r.get("observed", "rejected_producer"))
                for r in producer["gate_check_summary"]
            )
    blocked = [
        dict(
            unit_id=f"gate/{i}",
            cluster_id=r["upstream"],
            arm="prerequisite",
            seed=7628915,
            denominator=1,
            metric=None,
            status="blocked",
            missing_reason=r["missing_reason"],
            source_gate=r,
        )
        for i, r in enumerate(diagnostics)
    ]
    value["rows"].extend(blocked)
    verdict = "disqualified" if value["verdict_class"] == "disqualified" else "blocked"
    value.update(
        experiment_id=8915,
        task_id="exp8915-hardware-operation-budget",
        milestone="2026.11.762",
        run_date="20261108",
        random_seed=7628915,
        execution_venue="host",
        honest_verdict="complete_" + verdict + "_hardware_operation_budget",
        verdict_class=verdict,
        hardware_map_ready_score=ready,
        gate_check_summary=diagnostics,
        intended_count=len(value["rows"]),
        blocked_count=len(blocked),
        independent_count=0,
        invocation=work.get("invocation", {}),
        cited_upstream_artifacts=[
            dict(
                ref,
                producer_task_id=evidence[b].get("task_id"),
                producer_verdict=evidence[b].get("verdict_class"),
                source_admitted=b in work["evidence"],
                fields=[PRODUCERS[b][2], "rows", "code_config_hashes", "raw_shard_hashes"],
            )
            for b, ref in sorted(refs.items())
        ],
        upstream_scheduled_counts={
            b: {
                k: p.get(k)
                for k in (
                    "intended_count",
                    "completed_count",
                    "failed_count",
                    "censored_count",
                    "excluded_count",
                    "independent_count",
                )
            }
            for b, p in evidence.items()
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
                )
            }
            for b, p in evidence.items()
        },
        physical_evidence_state=BOUNDARIES,
        methodology_note="Unconditional CPU evidence accounting. Valid null costs remain useful; Exp8912 fractions require cost_attribution_ready_score=1. Exp8914 inclusive timers are nonadditive and rejected producer clocks are diagnostic only. Unknown fractions never become zero or 100x benefit. No model loading, training, device probe, purchase, vendor substitution or external publication. PRD NFR-01 whole-service10x remains unproved.",
    )
    for status in ("completed", "failed", "censored"):
        value[status + "_count"] = sum(r["status"] == status for r in value["rows"])
    value["reproducibility_checksum"] = canonical_hash(dict(rows=value["rows"], refs=work["refs"]))
    value["field_principles"].update(
        {
            k: "Preserve original producer scope; inventory and analytical limits establish neither device speed nor independent generalization."
            for k in value
            if k not in value["field_principles"]
        }
    )
    return value


engine.progress, engine.admission, engine.reduce, engine.build = progress, admission, reduce, build
replay = engine.replay


def main(argv: list[str] | None = None) -> int:
    """Validate private prospective bytes before replacing the terminal primary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261108"], default="20261108")
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
        8915, "Hardware operation budget", output.name, repo_root=output.parent, seed=7628915
    )
    progress("operand_admission_before")
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
    value = build(work, validation)
    progress("publication_before")
    report = publish_primary(
        output, value, lambda p: replay(p) if args.private_output else checker.terminal(p, raw)
    )
    atomic_json(raw / "publication.json", report)
    progress("complete")
    return 0
