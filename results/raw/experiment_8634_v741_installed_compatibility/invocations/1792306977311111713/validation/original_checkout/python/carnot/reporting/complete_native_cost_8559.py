"""REQ-REPORT-8559: immutable operands qualify complete costs independently of speed."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np

from carnot.reporting import nine_input_qualification_8558 as qualified
from carnot.reporting import v735_contract_methods as methods
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v709_execution import child, progress

__all__ = ["qualified"]
Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8559_v735_complete_native_cost"
TASK = "exp8559-complete-native-cost"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_complete_native_cost_8559.py"
OWNED = [
    f"python/carnot/reporting/complete_native_{s}_8559.py"
    for s in ["cost", "worker", "runner", "validation"]
] + [CLI]
UPSTREAM = "results/experiment_8558_v735_nine_input_native_qualification.json"
PIN = "sha256:efbf55b9e5cbdbd84c3e26040d53b7c76226fee6667595b1176c63d73538310d"
MODEL_SPECS: list[Json] = []
reference, read = qualified.inputs.reference, qualified.inputs.read
REQUIRED = {
    "exact_disjoint_node_union",
    "owned_python_coverage_100",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "cold",
    "missing",
    "owned_failure",
    "rehashed_tamper",
    "adversarial",
    "strict_rows",
}


class GateFailure(ValueError):
    """A gate retains both operands so an absent resource cannot look like zero work."""

    def __init__(self, path: Path, field: str, expected: Any, observed: Any):
        self.row = dict(
            check=field,
            upstream=qualified.TASK,
            path=str(path),
            sha256=sha256_file(path) if path.is_file() else None,
            field=field,
            operator="==",
            expected=expected,
            observed=observed,
            passed=False,
        )
        super().__init__(field)


def require(path: Path, field: str, expected: Any, observed: Any) -> None:
    """A missing prerequisite blocks measurement before any clocks are interpreted."""
    if expected != observed:
        raise GateFailure(path, field, expected, observed)


def admit(root: Path, raw: Path) -> Json:
    """Authenticate the original policy and actual consumer without rewriting its history."""
    path = root / UPSTREAM
    require(path, "sha256", PIN, sha256_file(path) if path.is_file() else None)
    value = read(dict(path=str(path), sha256=PIN))
    for field, expected in dict(
        nine_input_native_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        run_date="20261015",
    ).items():
        require(path, field, expected, value.get(field))
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    side = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
    require(path, "bound_publication", True, side["report"]["passed"])
    protocol = methods.protocol()["nine_input_native_inference"]
    original = json.loads(
        (ROOT / "openspec/change-proposals/v733-methods-protocol.json").read_bytes()
    )
    require(path, "original_timing_protocol", original["nine_input_native_inference"], protocol)
    authority = methods.authority(root, raw / "authority")
    require(path, "activated_authority", True, authority["activated"])
    task = next(t for t in methods.protocol()["task_map_v735"] if t["id"] == TASK)
    require(path, "deliverable", "results/" + NAME + ".json", task["deliverable"])
    receipt = child(
        "qualified_native_consumer",
        [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / qualified.CLI), "--consume", str(path)],
        raw / "admission",
        deadline=180,
    )
    require(path, "original_consumer", True, receipt["passed"])
    tape, head = read(value["request_tape_reference"]), read(value["head_reference"])
    rows = [r for r in tape["rows"] if r["unit_id"] in tape["timed_ids"]]
    require(path, "first64_test_ids", tape["timed_ids"], [r["unit_id"] for r in rows])
    require(path, "test_role_count", 64, sum(r["role"] == "test" for r in rows))
    refs = [
        reference(path),
        reference(Path(value["extension_reference"]["path"])),
        value["head_reference"],
        value["request_tape_reference"],
        value["measurement_kernel_reference"],
        reference(ROOT / methods.PROTOCOL),
        reference(ROOT / "openspec/change-proposals/v733-methods-protocol.json"),
        reference(Path(terminal["publication"]["sidecar_path"])),
        *value["extension_reference"]["sources"],
    ]
    for ref in refs:
        require(Path(ref["path"]), "operand_hash", ref["sha256"], sha256_file(Path(ref["path"])))
    extension = value["extension_reference"]
    require(
        Path(extension["path"]),
        "extension_hash",
        extension["sha256"],
        sha256_file(Path(extension["path"])),
    )
    return dict(
        head=head,
        requests=rows,
        timed_ids=tape["timed_ids"],
        extension=extension,
        historical_dates=tape["historical_dates"],
        protocol=protocol,
        refs=refs,
        historical_model_provenance=value["historical_model_provenance"],
        receipt=receipt,
        authority=authority,
    )


def roster() -> list[tuple[int, int, str, bool]]:
    """Warmups stay visible and measured pairs keep the original alternating order."""
    return [
        (b, r, a, r == -1)
        for b in [1, 8]
        for r in range(-1, 10)
        for a in (("python", "native") if r % 2 == 0 or r == -1 else ("native", "python"))
    ]


def measure(root: Path, raw: Path) -> Json:
    """Each arm is a fresh bounded child; no cached feature enters its timing path."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(raw=str(raw), started_ns=start, rows=[], gate_check_summary=[], bundle=None)
    try:
        with TemporaryDirectory(prefix="exp8559-disk-", dir="/var/tmp") as folder:
            scratch = Path(folder)
            probe = scratch / "probe"
            probe.write_bytes(b"disk-backed writable scratch")
            filesystem = child(
                "scratch_filesystem",
                ["findmnt", "-n", "-o", "FSTYPE", "-T", folder],
                raw / "preflight",
                deadline=10,
            )
            require(
                scratch,
                "disk_backed_scratch",
                True,
                filesystem["passed"]
                and Path(filesystem["stdout_path"]).read_text().strip() not in {"tmpfs", "ramfs"},
            )
            memory = (
                int(
                    next(
                        s.split()[1]
                        for s in Path("/proc/meminfo").read_text().splitlines()
                        if s.startswith("MemAvailable:")
                    )
                )
                * 1024
            )
            require(scratch, "memory_at_least_256MiB", True, memory >= 256 * 1024**2)
            require(
                scratch,
                "disk_at_least_64MiB",
                True,
                shutil.disk_usage(scratch).free >= 64 * 1024**2,
            )
            for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
                require(
                    ROOT / ".venv/bin" / tool,
                    "tool_" + tool,
                    True,
                    (ROOT / ".venv/bin" / tool).is_file(),
                )
            work["preconditions"] = dict(
                memory_available_bytes=memory,
                filesystem=filesystem,
                scratch=str(scratch),
                measurement_cap_s=3600,
                validation_reserve_s=900,
                total_cap_s=4800,
            )
        bundle = admit(root, raw)
        work["bundle"] = bundle
        manifest = raw / "bundle.json"
        atomic_json(manifest, bundle)
        for index, (size, repeat, arm, warmup) in enumerate(roster()):
            progress("8559_benchmark", index, 44 - index)
            target = raw / f"batch{size}-repeat{repeat}-{arm}.json"
            receipt = child(
                target.stem,
                [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--worker",
                    str(manifest),
                    "--worker-output",
                    str(target),
                    "--arm",
                    arm,
                    "--batch",
                    str(size),
                ],
                raw / "children",
                deadline=min(120, max(0.01, 3600 - (time.monotonic_ns() - start) / 1e9)),
            )
            work["rows"].append(
                dict(
                    unit_id=target.stem,
                    batch=size,
                    repeat=repeat,
                    arm=arm,
                    warmup=warmup,
                    receipt=receipt,
                    primitive_reference=reference(target) if receipt["passed"] else None,
                    missing_reason=None if receipt["passed"] else "timeout_or_child_failure",
                )
            )
    except (GateFailure, OSError, ValueError, KeyError) as error:
        work["gate_check_summary"].append(
            error.row
            if isinstance(error, GateFailure)
            else GateFailure(root, "available_original_inputs", True, str(error)).row
        )
    work["ended_ns"] = time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    return work


def checksum(value: Json) -> str:
    """Exclude only the checksum field so repaired summaries still need full replay."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def interval(pairs: list[Json]) -> list[float]:
    """Resample paired repeat indices; requests never masquerade as independent trials."""
    py, native = (np.asarray([p[k] for p in pairs]) for k in ["python_s", "native_s"])
    indices = np.random.default_rng(7338534).integers(0, len(pairs), size=(10000, len(pairs)))
    ratios = py[indices].sum(axis=1) / native[indices].sum(axis=1)
    return [float(x) for x in np.quantile(ratios, [0.025, 0.975])]


def latency_interval(rows: list[Json], size: int) -> list[float]:
    """Paired repeat resampling preserves all64 within-repeat request dependencies."""
    indices = np.random.default_rng(7338534).integers(0, 10, size=(10000, 10))
    quantiles = {}
    for arm in ["python", "native"]:
        values = np.asarray(
            [
                [
                    r["complete_latency_s"]
                    for r in rows
                    if r["batch"] == size and r["repeat"] == repeat and r["arm"] == arm
                ]
                for repeat in range(10)
            ]
        )
        quantiles[arm] = np.quantile(values[indices].reshape(10000, -1), 0.95, axis=1)
    return [
        float(x) for x in np.quantile(quantiles["native"] / quantiles["python"], [0.025, 0.975])
    ]


def speed(ci: list[float] | None, p95: float | None) -> tuple[int, int]:
    """Readiness and speed are separate; the complete primary boundary decides benefit."""
    clean = ci is not None and p95 is not None and p95 <= 1.05
    return int(clean and ci is not None and ci[0] > 1), int(
        clean and ci is not None and ci[0] >= 10
    )


def reduce(work: Json) -> Json:
    """Reduce adjacent primitive spans and serialized responses without fitting any model."""
    result: Json = dict(
        per_request_rows=[],
        component_span_rows=[],
        paired_repeat_rows=[],
        operation_shares=[],
        warmup_rows=[],
        startup_cost=[],
        throughput_ratio_ci95=None,
        p95_latency_ratio=None,
        p95_latency_ratio_ci95=None,
        complete_throughput_ratio=None,
        fallback_count=0,
        fallback_share=None,
        typed_decision_mismatch_count=0,
        max_probability_error=0.0,
        completed_count=0,
        failed_count=0,
        censored_count=0,
        measurement_clean=False,
        cell_statistics=[],
        peak_memory_kib={},
    )
    bundle = work["bundle"]
    if bundle is None:
        return result
    expected_ids = bundle["timed_ids"]
    references = {
        r["unit_id"]: json.loads(qualified.legacy.numeric.request(bundle["head"], r))
        for r in bundle["requests"]
    }
    observed = {}
    for row, slot in zip(work["rows"], roster(), strict=True):
        if (row["batch"], row["repeat"], row["arm"], row["warmup"]) != slot:
            raise ValueError("roster_drift")
        receipt = row["receipt"]
        if not receipt["passed"]:
            result["censored_count" if receipt["timed_out"] else "failed_count"] += 1
            continue
        p = read(row["primitive_reference"])
        if (p["arm"], p["batch"]) != (row["arm"], row["batch"]) or not (
            receipt["started_monotonic_ns"]
            <= p["start_ns"]
            <= p["ready_ns"]
            <= p["end_ns"]
            <= receipt["ended_monotonic_ns"]
        ):
            raise ValueError("worker_clock_or_identity")
        if p["arm"] == "native" and p["loaded_extension"] != {
            k: bundle["extension"][k] for k in ["path", "sha256"]
        }:
            raise ValueError("loaded_extension_drift")
        if [r["unit_id"] for r in p["requests"]] != expected_ids:
            raise ValueError("request_roster")
        wall = (p["end_ns"] - receipt["started_monotonic_ns"]) / 1e9
        startup = (p["ready_ns"] - receipt["started_monotonic_ns"]) / 1e9
        costs: Json = dict(startup=startup)
        last = p["ready_ns"]
        for span in p["spans"]:
            if not last <= span["start_ns"] <= span["end_ns"] <= p["end_ns"]:
                raise ValueError("overlapping_or_invalid_span")
            costs[span["operation"]] = (
                costs.get(span["operation"], 0.0) + (span["end_ns"] - span["start_ns"]) / 1e9
            )
            last = span["end_ns"]
            if not row["warmup"]:
                result["component_span_rows"].append(
                    dict(span, repeat=row["repeat"], arm=row["arm"], batch=row["batch"])
                )
        costs["unattributed_orchestration"] = wall - sum(costs.values())
        if costs["unattributed_orchestration"] < -1e-12 or not np.isfinite(wall) or wall <= 0:
            raise ValueError("wall_reconciliation")
        summary = dict(
            unit_id=row["unit_id"],
            batch=row["batch"],
            repeat=row["repeat"],
            arm=row["arm"],
            wall_s=wall,
            startup_s=startup,
            throughput_per_s=64 / wall,
            warmup=row["warmup"],
            peak_rss_kib=p["peak_rss_kib"],
        )
        for request in p["requests"]:
            response = json.loads(request["response"])
            ref = references[request["unit_id"]]
            error = abs(response["probability"] - ref["probability"])
            mismatch = response["decision"] != ref["decision"]
            if (
                response["unit_id"] != request["unit_id"]
                or not np.isfinite(error)
                or request["latency_ns"] != request["end_ns"] - request["start_ns"]
                or not p["ready_ns"] <= request["start_ns"] <= request["end_ns"] <= p["end_ns"]
            ):
                raise ValueError("response_identity_or_clock")
            result["max_probability_error"] = max(result["max_probability_error"], error)
            result["typed_decision_mismatch_count"] += int(mismatch)
            if not row["warmup"]:
                result["per_request_rows"].append(
                    dict(
                        request,
                        batch=row["batch"],
                        repeat=row["repeat"],
                        arm=row["arm"],
                        probability_error=error,
                        decision_mismatch=mismatch,
                        boundary_fallback=response["boundary_fallback"],
                        complete_latency_s=request["latency_ns"] / 1e9 + startup / 64,
                        passed=error <= 1e-12 * (1 + abs(ref["probability"])) and not mismatch,
                    )
                )
        if row["warmup"]:
            result["warmup_rows"].append(
                dict(summary, request_rows=p["requests"], operation_seconds=costs)
            )
        else:
            result["completed_count"] += 1
            observed[(row["batch"], row["repeat"], row["arm"])] = summary
            result["startup_cost"].append(summary)
            result["operation_shares"].append(
                dict(
                    summary,
                    seconds=costs,
                    fractions={k: v / wall for k, v in costs.items()},
                    overlap_s=0.0,
                    unattributed_s=costs["unattributed_orchestration"],
                )
            )
            result["peak_memory_kib"][row["arm"]] = max(
                result["peak_memory_kib"].get(row["arm"], 0), p["peak_rss_kib"]
            )
    for size in [1, 8]:
        pairs = []
        for repeat in range(10):
            if all((size, repeat, arm) in observed for arm in ["python", "native"]):
                pair = dict(
                    batch=size,
                    repeat=repeat,
                    python_s=observed[(size, repeat, "python")]["wall_s"],
                    native_s=observed[(size, repeat, "native")]["wall_s"],
                )
                pairs.append(pair)
        result["paired_repeat_rows"].extend(pairs)
        if len(pairs) == 10:
            latency = {
                arm: float(
                    np.quantile(
                        [
                            r["complete_latency_s"]
                            for r in result["per_request_rows"]
                            if r["batch"] == size and r["arm"] == arm
                        ],
                        0.95,
                    )
                )
                for arm in ["python", "native"]
            }
            stats = dict(
                batch=size,
                throughput_ratio_ci95=interval(pairs),
                complete_throughput_ratio=sum(p["python_s"] for p in pairs)
                / sum(p["native_s"] for p in pairs),
                p95_latency_ratio=latency["native"] / latency["python"],
                p95_latency_ratio_ci95=latency_interval(result["per_request_rows"], size),
                p95_latency_s=latency,
            )
            result["cell_statistics"].append(stats)
            if size == 1:
                result.update(
                    {
                        k: stats[k]
                        for k in [
                            "throughput_ratio_ci95",
                            "complete_throughput_ratio",
                            "p95_latency_ratio",
                            "p95_latency_ratio_ci95",
                        ]
                    }
                )
    result["fallback_count"] = sum(r["boundary_fallback"] for r in result["per_request_rows"])
    result["fallback_share"] = (
        result["fallback_count"] / len(result["per_request_rows"])
        if result["per_request_rows"]
        else None
    )
    result["measurement_clean"] = (
        result["completed_count"] == 40
        and len(result["warmup_rows"]) == 4
        and not result["failed_count"]
        and not result["censored_count"]
        and result["typed_decision_mismatch_count"] == 0
        and all(r["passed"] for r in result["per_request_rows"])
    )
    return result


def artifact(work: Json, receipts: list[Json], output: Path) -> Json:
    """Publish a clean null when complete costs fail the prespecified speed threshold."""
    reduced = reduce(work)
    checks = {r["name"]: r["passed"] for r in receipts}
    required = REQUIRED <= checks.keys() and all(checks.values())
    ready = reduced["measurement_clean"] and required
    benefit, nfr = speed(reduced["throughput_ratio_ci95"], reduced["p95_latency_ratio"])
    verdict = (
        "disqualified"
        if any(not r["passed"] for r in receipts)
        else "blocked"
        if work["gate_check_summary"]
        else "circular_positive"
        if ready and benefit
        else "null"
        if ready
        else "disqualified"
    )
    bundle = work["bundle"] or {}
    refs = bundle.get("refs", [])
    raw = Path(work["raw"])
    value: Json = dict(
        reduced,
        experiment_id=8559,
        task_id=TASK,
        milestone="2026.10.735",
        run_date="20261016",
        honest_verdict=f"complete_{verdict}_v735_complete_native_cost",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"]
        + [
            dict(
                check=r["name"],
                upstream=TASK,
                path=r.get("stdout_path"),
                sha256=r.get("stdout_sha256"),
                field="passed",
                operator="==",
                expected=True,
                observed=False,
            )
            for r in receipts
            if not r["passed"]
        ],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        current_llm_invocations=0,
        historical_model_provenance=bundle.get("historical_model_provenance", []),
        rows=work["rows"],
        intended_count=40,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            requests=64,
            paired_repeats=10,
            batches=[1, 8],
            machines=1,
            independent_workloads=0,
            warmup_children=4,
            measured_children=40,
            bootstrap_draws=10000,
            bootstrap_unit="paired repeat indices on one machine",
        ),
        verifier_is_oracle=True,
        exposure_scope="original exposed development history; first64 test-role records; no labels used",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(ready),
        flagged_adversarial=bool(work.get("adversarial_findings", [])),
        adversarial_findings=work.get("adversarial_findings", []),
        acceptance_gates=dict(
            original_inputs=bool(bundle),
            complete_measurement=reduced["measurement_clean"],
            required_validation=required,
            numerical_parity=reduced["max_probability_error"] <= 1e-12
            and reduced["typed_decision_mismatch_count"] == 0,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        preconditions_checked=work.get("preconditions", {}),
        duration_s=work.get("duration_s", (work["ended_ns"] - work["started_ns"]) / 1e9),
        phase_spans=[
            dict(
                phase="measurement",
                started_monotonic_ns=work["started_ns"],
                ended_monotonic_ns=work["ended_ns"],
            )
        ],
        random_seed=7338534,
        source_artifact_hashes=refs,
        code_config_hashes=[reference(ROOT / p) for p in OWNED + [TEST]],
        raw_shard_hashes=[
            r["primitive_reference"] for r in work["rows"] if r["primitive_reference"]
        ],
        cited_upstream_artifacts=refs,
        work_reference=reference(raw / "measurement.json"),
        publication_output=str(output.absolute()),
        complete_native_cost_ready_score=int(ready),
        scoped_native_speed_benefit_score=benefit if ready else 0,
        nfr01_score=nfr if ready else 0,
        complete_boundary=bundle.get("protocol", {}).get("complete_boundary", []),
        excluded_operations=["model acquisition", "feedback update", "durable storage"],
        preserved_quarantines=[8530, 8533, 8537, 8545],
        historical_source_null="Exp8529 remains a qualified exposed source-information null",
        historical_append_ratio=dict(
            value=2.16268,
            eligible_here=False,
            scope="original five-input append-service boundary only",
        ),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8560, 8565)
        ],
        methodology_note="First64 original test-role cached answer texts; fresh interpreter per arm/repeat. One explicit warmup per arm/cell is retained separately. Complete throughput includes interpreter/import/head/extension startup and every request operation through serialization; evidence-file persistence and interpreter teardown are separately outside that boundary. Request latency charges startup/64 plus observed batch-ready-to-serialized-response latency. No fitting, model calls, feedback or durable-store operation. Exposed repeated clocks on one machine cannot establish generalization.",
        original_authority=bundle.get("authority"),
        historical_dates=bundle.get("historical_dates"),
        python_coverage=work.get("python_coverage"),
        exact_union_receipt=work.get("exact_union_receipt"),
    )
    value["field_principles"] = {
        k: "Bind immutable invocation operands and complete measured work; missing evidence cannot grant readiness or independent benefit."
        for k in value
    }
    value["field_principles"].update(
        operation_shares="Adjacent non-overlapping measured spans plus explicit unattributed orchestration sum to complete wall. Overlap is zero; no overlapping subtraction or measured hardware ceiling.",
        throughput_ratio_ci95="10000 paired repeat-index draws, seed7338534; sum Python complete walls / sum native complete walls; 95 percent interval; one machine.",
        p95_latency_ratio="Native/Python pooled request p95; each request includes startup/64 plus actual batch response latency; batch1 is primary.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Recompute all summaries and responses; repaired envelope hashes grant no authority."""
    try:
        value = json.loads(path.read_bytes())
        if checksum(value) != value["reproducibility_checksum"]:
            return False
        work = read(value["work_reference"])
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if work["bundle"]:
            bundle = work["bundle"]
            upstream = read(dict(path=str(ROOT / UPSTREAM), sha256=PIN))
            if (
                bundle["timed_ids"] != read(upstream["request_tape_reference"])["timed_ids"]
                or bundle["head"] != read(upstream["head_reference"])
                or bundle["requests"]
                != [
                    r
                    for r in read(upstream["request_tape_reference"])["rows"]
                    if r["unit_id"] in bundle["timed_ids"]
                ]
                or bundle["extension"] != upstream["extension_reference"]
                or bundle["protocol"] != methods.protocol()["nine_input_native_inference"]
            ):
                return False
            for row in work["rows"]:
                receipt = row["receipt"]
                for stream in ["stdout", "stderr"]:
                    if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                        return False
        return bool(
            artifact(work, value["validation_receipts"], Path(value["publication_output"])) == value
        )
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False


def consume(path: Path) -> Json:
    """The hardware reader accepts only reconciled complete-operation cost evidence."""
    value = json.loads(path.read_bytes())
    if value["complete_native_cost_ready_score"] != 1 or not replay(path):
        raise ValueError("complete_native_cost_not_ready")
    return dict(
        ready=True,
        cost_reference=reference(path),
        operation_shares=value["operation_shares"],
        complete_boundary=value["complete_boundary"],
        excluded_operations=value["excluded_operations"],
    )
