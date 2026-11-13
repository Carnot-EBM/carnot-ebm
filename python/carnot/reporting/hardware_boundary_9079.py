"""REQ-REPORT-9079-V774: map exact work while preserving unavailable evidence.

Sparse spline evaluation does not remove dense anchors or host durability.
Historical device evidence establishes boundaries, not current service speed.
"""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting import hardware_operation_runner_8825 as historical_runner
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.encoder_head_diagnosis_8972 import checksum, reference as ref, retain
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.spline_basis_methods_8986 import gate
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9079_v774_hardware_boundary", "exp9079-hardware-boundary"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = ["python/carnot/reporting/hardware_boundary_9079.py", CLI]
TEST = "tests/python/test_hardware_boundary_9079.py"
PROTOCOL = "openspec/change-proposals/v774-hardware-boundary-9079.json"
OPERATIONS = {
    "basis_scoring": (
        "float64",
        "27 inputs; linear28 or cubic spline217 coordinates per encoder; three encoders",
        "input and design buffers; local basis support; dense NumPy design",
    ),
    "coefficient_updates": (
        "float64",
        "3x217 coefficients and AdaGrad accumulators; 3x28 linear alternative",
        "candidate, gradient and accumulator reads/writes; sqrt and norm clipping",
    ),
    "dense_anchors": (
        "float64",
        "initial and current coefficients across every coordinate; bias excluded",
        "dense 0.01*(theta-initial), even where local spline support is zero",
    ),
    "guard_replay": (
        "float64/Boolean",
        "all64 original guard graphs, three encoders; initial and candidate predictions",
        "full design/state reads; sigmoid, averaging and rollback",
    ),
    "durable_recovery": (
        "JSON bytes/exact IDs",
        "coefficients, accumulator, pending predictions, acknowledgments and guard state",
        "host serialization, flush/fsync, atomic rename, restart and feedback drain",
    ),
    "qwen_embeddings": (
        "Q4_K_M weights; activation/output dtype unmeasured",
        "Qwen3.8-27B; final-layer mean pooling; human feature receipt blocked",
        "model startup, tokenization, device weights and embedding host transfer",
    ),
    "quadratic_sampling": (
        "signed fixed-point spins/couplings",
        "historical Ising state only; k_max<=5",
        "register/state and coupling transfer; no demonstrated current spline-to-Ising lowering",
    ),
}
BOARDS = {
    "CPU": "NumPy/SciPy deterministic spline and linear mechanics; current complete service unmeasured",
    "leased_RTX_3090": "Qwen CUDA is a possible encoder path; lease and exact embedding workload unqualified; small-head offload unmeasured",
    "KV260": "Historical quadratic Ising fabric only, k_max<=5; no spline update kernel; SSH-only future access",
    "PolarFire": "Historical authenticated Linux CPU-only dispatch; fabric kernel unqualified",
    "GateMate": "Unchanged physical/JTAG0xffffffff block; no qualified current execution",
}
CONDITIONS = {
    "CPU": "Complete paired service clocks and correctness, including recovery; no acquisition required.",
    "leased_RTX_3090": "Authenticated lease/runtime and exact Qwen embedding receipt; matched batching, precision, startup, transfers and service costs before continued lease or purchase.",
    "KV260": "A supported equivalent kernel, fixed-point parity and matched actual-device complete service costs; deterministic spline locality alone is insufficient.",
    "PolarFire": "Exact hash-bound current CPU workload and complete costs; fabric acceleration requires a separate authenticated kernel and parity receipt.",
    "GateMate": "Operator-dated cable, port or power change, then valid GM1Ax IDCODE and ordered n16 smoke; no unchanged bring-up scheduled.",
    "NPU": "Working supported SDK, equivalent operators, precision parity and actual-device complete service evidence.",
    "Extropic_TSU": "Authenticated device/SDK access, supported stochastic lowering, sample quality and whole-service evidence including host and transfers; vendor energy claims are insufficient.",
    "larger_FPGA_or_GPU": "Measured current compatible bottleneck, capacity need and matched actual-device service benefit before procurement.",
}
checker = namespace(historical_runner, "hardware9079_validation")
checker.m = sys.modules[__name__]


def progress(phase: str) -> None:
    """Flush boundaries so even an evidence-only refusal remains observable."""
    print(f"[exp9079] phase={phase}", flush=True)


def bounds(clocks: Json | None) -> Json:
    """Complete disjoint clocks pay host, startup and transfers before ideal offload."""
    result: Json = dict(
        nonaccelerable_fraction=None,
        acceleration_bound=None,
        denominator="parent complete service seconds including startup, host/durability, transfers and accelerated work",
        missing_operands=["qualified disjoint complete service and compatible kernel timing"],
        uncertainty="No qualified current timing uncertainty; ideal arithmetic is conditional, not a device measurement.",
    )
    fields = ("total_s", "host_s", "startup_s", "transfer_s", "accelerated_s")
    if not clocks or any(type(clocks.get(k)) not in (float, int) for k in fields):
        return result
    if any(not math.isfinite(clocks[k]) or clocks[k] < 0 for k in fields):
        return result
    total = clocks["total_s"]
    if total <= 0 or not math.isclose(sum(clocks[k] for k in fields[1:]), total, rel_tol=1e-9):
        return result
    f = sum(clocks[k] for k in ("host_s", "startup_s", "transfer_s")) / total
    result.update(
        nonaccelerable_fraction=f,
        acceleration_bound=1 / f if f else None,
        strictly_more_than_100x_possible=f < 0.01,
        missing_operands=[],
        unbounded_ideal_limit=f == 0,
        uncertainty=clocks.get("uncertainty", "Conditional point estimate; interval unavailable."),
    )
    return result


def collect(root: Path, raw: Path) -> Json:
    """Check identity and freeze original bytes before interpreting any primitive rows."""
    progress("preconditions_before")
    start = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    contract = json.loads((root / PROTOCOL).read_bytes())
    checks = [
        gate(raw, "mode", 0o700, raw.stat().st_mode & 0o777, "private_storage"),
        gate(raw, "free_disk", True, shutil.disk_usage(raw).free > 10_000_000, "private_disk"),
    ]
    references, values = [], {}
    for item in contract["inputs"] + contract["vendor_claim_sources"]:
        path = root / item["path"]
        observed = ref(path)["sha256"] or "missing_file"
        row = gate(
            path,
            "sha256",
            item["sha256"],
            observed,
            "source_identity",
            str(item.get("experiment_id") or item.get("url", "source")),
        )
        checks.append(row)
        if row["passed"]:
            saved = dict(retain(ref(path), raw), experiment_id=item.get("experiment_id"))
            references.append(saved)
            if item.get("experiment_id") == 9073:
                values["feature"] = json.loads(Path(saved["path"]).read_bytes())
    for name, version in contract["dependencies"].items():
        checks.append(
            gate(
                root / "pyproject.toml",
                name,
                version,
                importlib.metadata.version(name),
                "dependency",
            )
        )
    cost_path = root / contract["upstream_cost_path"]
    checks.append(
        gate(
            cost_path,
            "exists",
            True,
            True if cost_path.is_file() else "missing_file",
            "current_cost",
            "exp9078",
        )
    )
    if cost_path.is_file():
        saved = retain(ref(cost_path), raw)
        references.append(dict(saved, experiment_id=9078))
        values["cost"] = json.loads(Path(saved["path"]).read_bytes())
        checks.append(
            gate(
                cost_path,
                "experiment_id",
                9078,
                values["cost"].get("experiment_id", "missing_field"),
                "cost_identity",
                "exp9078",
            )
        )
    feature = values.get("feature", {})
    feature_path = root / "results/experiment_9073_human_features.json"
    if feature.get("schema") == "blocked_gate_check_v1":
        for row in feature["gates_evaluated"]:
            checks.append(
                gate(
                    Path(row["artifact_path"]),
                    row["artifact_field"],
                    row["expected"],
                    row["actual"],
                    "upstream_feature_block",
                    row["upstream"],
                )
            )
    else:
        checks.append(
            gate(
                feature_path,
                "rows",
                "present",
                "present" if feature.get("rows") else "missing_field",
                "current_features",
                "exp9073",
            )
        )
    sources = [
        retain(ref(ROOT / p), raw)
        for p in [
            *OWNED,
            TEST,
            PROTOCOL,
            "tests/python/test_primary_publication_7928.py",
            "tests/python/test_current_work_receipt.py",
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
            "tests/python/test_hard_exit_learning_qualification_8206.py",
        ]
    ]
    elapsed = time.monotonic() - start
    progress("preconditions_after_science_stopped_if_missing")
    work = dict(
        **values,
        checks=checks,
        failures=[r for r in checks if not r["passed"]],
        references=references,
        sources=sources,
        contract=contract,
        duration_s=elapsed,
        phase_spans=[dict(phase="source_authentication", start_s=0, end_s=elapsed)],
    )
    atomic_json(raw / "work.json", work)
    return dict(work, work_reference=ref(raw / "work.json"))


def reduce(work: Json) -> Json:
    """Keep current primitive rows separate from historical implementation feasibility."""
    operations = []
    for board in BOARDS:
        for operation, (dtype, state, movement) in OPERATIONS.items():
            implemented = board == "CPU" and operation not in {
                "qwen_embeddings",
                "quadratic_sampling",
            }
            status = "implemented_host_source" if implemented else "unsupported_current_workload"
            if board == "leased_RTX_3090" and operation == "qwen_embeddings":
                status = "candidate_runtime_missing_receipt"
            if board == "KV260" and operation == "quadratic_sampling":
                status = "historical_quadratic_only"
            operations.append(
                dict(
                    unit_id=board + "/" + operation,
                    group_id=board,
                    arm=board,
                    condition=operation,
                    operation=operation,
                    dtype=dtype,
                    state=state,
                    movement=movement,
                    status=status,
                    metric="current_duration_s",
                    value=None,
                    numerator=None,
                    denominator=None,
                    exclusion_reason="current_complete_timing_unavailable",
                    measured_here=False,
                    current_device_implementation_measured=False,
                )
            )
    boards = [
        dict(
            board=b,
            boundary=s,
            current_measured_operations=[],
            procurement_condition=CONDITIONS[b],
            physical_probe_performed=False,
        )
        for b, s in BOARDS.items()
    ]
    for board, identity in [("KV260", 8230), ("PolarFire", 8259), ("GateMate", 8302)]:
        selected = next(b for b in boards if b["board"] == board)
        selected["historical_artifact_reference"] = next(
            (r for r in work.get("references", []) if r.get("experiment_id") == identity), None
        )
    missing = [
        dict(
            upstream="exp9078",
            missing="complete disjoint operation clocks, parent denominator, host, startup, transfers, recovery and uncertainty",
        ),
        dict(
            upstream="exp9073",
            missing="qualified embedding acquisition, dtype/state bytes, model load and transfer clocks",
        ),
        dict(
            upstream="actual_devices",
            missing="equivalent supported current kernels, correctness and complete service receipts",
        ),
    ]
    return dict(
        rows=operations,
        operation_support_rows=operations,
        per_board_rows=boards,
        upstream_operation_rows=work.get("cost", {}).get("operation_rows", []),
        upstream_feature_rows=work.get("feature", {}).get("rows", []),
        missing_evidence_rows=missing,
        procurement_conditions=CONDITIONS,
        workload_change="Basis scoring and coefficient/accumulator updates add dense anchors, full guard replay, durable recovery and Qwen embeddings; a local spline update is not an Ising sampler.",
        **bounds(None),
    )


def build(work: Json, validation: Json, date: str, *, fixture: bool) -> Json:
    """External failure blocks science; failed owned checks disqualify the producer."""
    reduced = reduce(work)
    owned = validation["passed"] is True
    cls = "disqualified" if not owned else "blocked" if work["failures"] else "null"
    reason = (
        "owned_validation"
        if not owned
        else work["failures"][0]["check"]
        if work["failures"]
        else "hardware_boundary"
    )
    value: Json = dict(
        reduced,
        experiment_id=9079,
        task_id=TASK,
        milestone="2026.11.774",
        run_date=date,
        honest_verdict=f"complete_{cls}_{reason}",
        verdict_class=cls,
        status="blocked" if cls == "blocked" else "error" if cls == "disqualified" else "success",
        hardware_map_ready_score=int(owned and cls == "null" and not fixture),
        required_checks_passed=owned,
        flagged_adversarial=False,
        fixture_only=fixture,
        gate_check_summary=work["checks"],
        preconditions_checked=dict(
            authenticated_before_computation=True,
            science_stopped=bool(work["failures"]),
            private_storage=True,
            dependencies=work["contract"]["dependencies"],
            no_required_device_or_model_access=True,
            scientific_budget_s=3300,
            validation_reserve_s=1200,
            hard_cap_s=4800,
        ),
        verifier_is_oracle=True,
        independent_generalization_score=0,
        claim_scope="Workload feasibility only. Graph truth is executable; graph wins are circular_positive. No verifier benefit, purchase, 100x or actual-device speedup claim.",
        exposure_scope="Historical graph metadata and public hardware/literature evidence. Human agreement is neither medical correctness nor a general verifier moat. Protected labels remain sealed.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_context=dict(
            experiment_id=9058,
            model="unsloth/Qwen3.8-27B-GGUF",
            quantization="Q4_K_M",
            scope="cited generation only; not a new load or embedding receipt",
        ),
        intended_count=len(reduced["rows"]),
        completed_count=0,
        failed_count=0,
        censored_count=len(reduced["rows"]),
        excluded_count=0,
        independent_count=0,
        duration_s=work["duration_s"],
        random_seed=774079,
        phase_spans=work["phase_spans"],
        raw_shard_hashes=work["references"],
        code_config_hashes=work["sources"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=r["experiment_id"],
                sha256=r["sha256"],
                fields_imported=["schema", "gates_evaluated"]
                if r["experiment_id"] == 9073
                else ["operation_rows"]
                if r["experiment_id"] == 9078
                else ["honest_verdict", "verdict_class"],
            )
            for r in work["references"]
            if r.get("experiment_id")
        ],
        vendor_claim_sources=work["contract"]["vendor_claim_sources"],
        validation_scope_manifest=dict(
            owned=OWNED,
            tests=[TEST],
            direct_consumers=[
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_current_work_receipt.py",
            ],
            coverage_exclusions=[],
            e2e=["E2E-018", "E2E-020", "E2E-021"],
        ),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation.get("coverage", {}),
        repository_health=[
            r for r in validation["receipts"] if r.get("scope") == "repository_health"
        ],
        owned_qualification=dict(validation_passed=owned, science_available=not work["failures"]),
        work_reference=work["work_reference"],
        physical_acceleration_demonstrated=False,
        amdahl_condition="For measured nonaccelerable fraction f over full service, ideal speedup<=1/f. f>=0.01 prevents strictly more than100x; arithmetic-only timing cannot establish f.",
    )
    value["field_principles"] = {
        k: "Bind retained bytes and scope; absence is unknown, not measured zero or independent scientific benefit."
        for k in value
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rehash retained operands and recompute the reduction instead of trusting summaries."""
    progress("cold_replay_before")
    try:
        value = json.loads(path.read_bytes())
        if checksum(value) != value["reproducibility_checksum"]:
            raise ValueError("checksum_drift")
        for r in [
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
            value["work_reference"],
        ]:
            if ref(Path(r["path"]))["sha256"] != r["sha256"]:
                raise ValueError("operand_drift")
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        work["work_reference"] = value["work_reference"]
        for branch, identity in [("feature", 9073), ("cost", 9078)]:
            original = next(
                (r for r in work["references"] if r.get("experiment_id") == identity), None
            )
            if original and work.get(branch) != json.loads(Path(original["path"]).read_bytes()):
                raise ValueError("original_producer_drift")
        for receipt in value["validation_receipts"]:
            log = ROOT / receipt["log_path"]
            if ref(log)["sha256"] != receipt["log_sha256"]:
                raise ValueError("validation_log_drift")
        expected = build(
            work,
            dict(
                passed=value["required_checks_passed"],
                receipts=value["validation_receipts"],
                coverage=value["owned_statement_coverage"],
            ),
            value["run_date"],
            fixture=value["fixture_only"],
        )
        if expected != value:
            raise ValueError("reduction_drift")
        progress("cold_replay_after")
        return dict(passed=True)
    except (OSError, ValueError, KeyError) as error:
        return dict(passed=False, reason=str(error))


def validate(raw: Path) -> Json:
    """Reuse measured owned coverage and add affected consumers and real recovery checks."""
    result: Json = checker.validate(raw)
    py = str(ROOT / ".venv/bin/python")
    direct = run_commands(
        ROOT,
        [
            CommandSpec(
                "direct_consumers_recovery",
                (
                    py,
                    "-m",
                    "pytest",
                    "--noconftest",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    "tests/python/test_primary_publication_7928.py",
                    "tests/python/test_current_work_receipt.py",
                    "tests/python/test_hard_exit_learning_qualification_8206.py",
                ),
                "owned",
            )
        ],
        log_dir=raw / "direct",
        heartbeat_s=30,
    )
    result["receipts"].extend(direct)
    result["passed"] = result["passed"] and all(r["passed"] for r in direct)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the evidence audit or its cold replay without device or model operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261113"], default="20261113")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        result = replay(args.replay)
        print(json.dumps(result), flush=True)
        return 0 if result["passed"] else 1
    if args.fixture and args.output.resolve().is_relative_to(ROOT):
        raise ValueError("fixture_output_must_be_private")
    raw = args.output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    work = collect(ROOT, raw)
    validation = dict(passed=False, receipts=[], coverage={}) if args.fixture else validate(raw)
    progress("publication_before")
    value = build(work, validation, args.date, fixture=args.fixture)
    publish_primary(
        args.output, value, lambda p: replay(p) if args.fixture else checker.terminal(p, raw)
    )
    progress("publication_after")
    return 0
