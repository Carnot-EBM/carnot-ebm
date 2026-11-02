"""REQ-REPORT-8622: complete resident clocks keep lifecycle and arithmetic apart."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np

from carnot.reporting import resident_complete_cost_8584 as base
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.lean_resident_8608 import authenticate, checksum, reference
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v709_execution import progress

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8622_v740_lean_resident_cost"
TASK = "exp8622-lean-resident-cost"
CLI = f"scripts/experiments/{NAME}.py"
WORKER = "python/carnot/reporting/lean_cost_worker_8622.py"
OWNED = [
    f"python/carnot/reporting/lean_cost{s}_8622.py"
    for s in ["", "_session", "_worker", "_runner", "_validation"]
] + [CLI]
TESTS = ["tests/python/test_lean_cost_8622.py", "tests/python/test_lean_cost_runner_8622.py"]
ARMS = ["legacy_python", "legacy_native", "lean_python", "lean_native"]
MODEL_SPECS: list[Json] = []
PINS = {
    "results/experiment_8615_v740_contract_methods.json": "sha256:6f7ea729a61997543357bc926f558510d0dbfdb95cabcd42aecbce936267209e",
    "results/experiment_8621_v740_lean_resident_qualification.json": "sha256:ac23baa78afed19caca35030b1b717ab1b2d1a1096b1c3646596364be98c8fea",
    "results/experiment_8596_v738_resident_complete_cost.json": "sha256:5dd00385cb4f330b900f842af1a43b580f06c5948a658c36ce3b01d35b46ac6c",
}
REQUIRED = {
    "owned_tests",
    "exact_node_union",
    "owned_coverage",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "e2e018",
    "e2e021",
    "primary_publication",
    "cold",
    "missing",
    "rehashed_tamper",
    "owned_failure",
    "private_readers",
}


def load(ref: Json) -> Json:
    """Only authenticated bytes may provide operands or immutable measurements."""
    return dict(json.loads(authenticate(ref).read_bytes()))


def admit(root: Path) -> Json:
    """Pinned original producers supply permission, not clocks for this invocation."""
    values, refs = [], []
    for name, digest in PINS.items():
        ref = dict(path=str(root / name), sha256=digest)
        value = load(ref)
        side = root / "results/raw" / Path(name).stem / "validators" / (digest[7:] + ".json")
        if (
            value["required_checks_passed"] is not True
            or value["flagged_adversarial"] is not False
            or read_bound_sidecar(root / name, side)["report"]["passed"] is not True
        ):
            raise ValueError("unqualified_original_producer")
        values.append(value)
        refs.extend([ref, reference(side)])
    methods, qualified, historical = values
    protocol = load(methods["methods_protocol_reference"])
    qwork = load(qualified["work_reference"])
    prior = load(historical["work_reference"])["bundle"]
    config = load(qwork["config_reference"])
    if (
        methods["methods_ready_score"] != 1
        or qualified["lean_measurement_ready_score"] != 1
        or protocol["producer_map_v740"][TASK] != f"results/{NAME}.json"
        or methods["lean_worker_2x2"] != protocol["lean_worker_2x2"]
        or qwork["registered_lean_methods"] != protocol["lean_worker_2x2"]
        or config["head"] != prior["head"]
        or any(config["extension"][k] != prior["extension"][k] for k in ["path", "sha256"])
        or qualified["target_model"] != historical["target_model"]
        or config["rows"][:64] != prior["requests"]
    ):
        raise ValueError("registered_operands_or_readiness")
    refs.extend(
        [
            methods["methods_protocol_reference"],
            qualified["work_reference"],
            historical["work_reference"],
            qwork["config_reference"],
        ]
    )
    refs.extend(qualified["code_config_hashes"] + qwork["sources"])
    refs.extend([qualified["target_model"]["checkpoint"], config["extension"]])
    for ref in refs:
        authenticate(ref)
    requests = [{k: r[k] for k in ["unit_id", "public", "q"]} for r in prior["requests"]]
    from carnot.reporting.complete_native_worker_8559 import batch

    answers, _ = batch(config["head"], requests)
    return dict(
        head=config["head"],
        extension=config["extension"],
        requests=requests,
        references=[json.loads(r["response"]) for r in answers],
        target_model=qualified["target_model"],
        historical_model_provenance=qwork["historical_model_provenance"],
        protocol=protocol["lean_worker_2x2"],
        refs=refs,
        historical_cost_null=prior["historical_cost_null"],
        preserved_quarantines=qwork["preserved_quarantines"],
    )


def roster() -> list[Json]:
    """Rotate lifecycle pairs and alternate arithmetic order before seeing timings."""
    return [dict(s, arm=ARMS[base.ARMS.index(s["arm"])]) for s in base.roster()]


def preconditions(raw: Path) -> Json:
    """CPU work needs writable disk and memory, independently of optional devices."""
    with TemporaryDirectory(prefix="carnot8622-", dir="/var/tmp") as scratch:
        probe = Path(scratch) / "probe"
        probe.write_bytes(b"disk-backed")
        mount = subprocess.check_output(
            ["findmnt", "-T", scratch, "-n", "-o", "FSTYPE"], text=True, timeout=5
        ).strip()
        memory = Path("/proc/meminfo").read_text()
        available = (
            int(next(s.split()[1] for s in memory.splitlines() if s.startswith("MemAvailable:")))
            * 1024
        )
        tools = {
            n: str(ROOT / ".venv/bin" / n) for n in ["python", "pytest", "ruff", "mypy", "coverage"]
        }
        if (
            mount in {"tmpfs", "ramfs"}
            or probe.read_bytes() != b"disk-backed"
            or available < 256 * 1024**2
            or shutil.disk_usage(scratch).free < 64 * 1024**2
            or not all(Path(p).is_file() for p in tools.values())
        ):
            raise ValueError("cpu_resource_precondition")
        return dict(
            scratch=scratch,
            mount=mount,
            memory=memory,
            disk_free_bytes=shutil.disk_usage(scratch).free,
            tools=tools,
            python=sys.version,
            cpu_threads=1,
            host_load=os.getloadavg(),
            optional_gpu_required=False,
            optional_board_required=False,
            measurement_stop_s=3600,
            validation_reserve_s=900,
            hard_cap_s=4800,
        )


def measure(root: Path, raw: Path, run_date: str) -> Json:
    """Retain every planned cell even if admission or the fixed budget prevents clocks."""
    from carnot.reporting.lean_cost_session_8622 import session

    progress("8622_preconditions_no_model_load_MODEL_SPECS=[]_llm_calls=0")
    raw.mkdir(parents=True, exist_ok=True)
    started = time.monotonic_ns()
    work: Json = dict(
        raw=str(raw),
        input_root=str(root),
        run_date=run_date,
        started_ns=started,
        bundle=None,
        rows=[],
        gate_check_summary=[],
        preconditions_checked={},
    )
    try:
        work["preconditions_checked"] = preconditions(raw)
        progress("8622_authenticate")
        work["bundle"] = admit(root)
    except (OSError, ValueError, KeyError) as error:
        work["gate_check_summary"].append(
            dict(
                check="original_inputs_and_cpu_resources",
                upstream="exp8615-contract-methods/exp8621-lean-resident-qualification",
                path=str(root / next(iter(PINS))),
                sha256=PINS[next(iter(PINS))],
                field="methods_ready_score/lean_measurement_ready_score",
                operator="==",
                expected=1,
                observed="missing" if isinstance(error, OSError) else str(error),
            )
        )
    for i, slot in enumerate(roster()):
        row = dict(
            slot,
            unit_id=f"L{slot['length']}-B{slot['batch']}-R{slot['repeat']}-{slot['arm']}",
            status="censored",
            primitive_reference=None,
            missing_reason="upstream_gate",
            intended_count=64 * slot["length"],
            completed_count=0,
            failed_count=0,
            censored_count=64 * slot["length"],
        )
        if work["bundle"]:
            row["missing_reason"] = "measurement_deadline"
            if time.monotonic_ns() + 40_000_000_000 < started + 3600_000_000_000:
                progress("8622_before_benchmark", i, 240 - i)
                try:
                    result = session(
                        work["bundle"],
                        slot,
                        raw / "sessions" / row["unit_id"],
                        deadline_ns=started + 3600_000_000_000,
                    )
                    path = raw / "primitives" / (row["unit_id"] + ".json")
                    atomic_json(path, result)
                    row.update(
                        status="completed",
                        primitive_reference=reference(path),
                        missing_reason=None,
                        completed_count=64 * slot["length"],
                        censored_count=0,
                    )
                    if i == 0:
                        work["first_cell_forecast"] = dict(
                            observed_s=result["duration_s"],
                            projected_s=result["duration_s"] * 240 * 7,
                            policy="diagnostic only; retain fixed roster and stop at3600s",
                        )
                except (OSError, ValueError, KeyError, TimeoutError) as error:
                    row.update(
                        status="failed",
                        missing_reason=str(error),
                        failed_count=64 * slot["length"],
                        censored_count=0,
                    )
                progress("8622_after_benchmark", i + 1, 239 - i)
        work["rows"].append(row)
    work["ended_ns"] = time.monotonic_ns()
    work["phase_spans"] = [
        dict(phase="preconditions_and_fresh_measurement", start_ns=started, end_ns=work["ended_ns"])
    ]
    work["measurement_code_config_hashes"] = []
    for name in OWNED:
        frozen = raw / "code" / Path(name).name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, frozen)
        work["measurement_code_config_hashes"].append(
            dict(reference(frozen), source_path=str(ROOT / name))
        )
    atomic_json(raw / "work.json", work)
    return work


def summarize_session(result: Json, bundle: Json) -> Json:
    """Frame endpoints reconstruct clocks and answers instead of trusting summary scalars."""
    spans = result["operations"]
    if (
        spans[0]["start_ns"] != result["start_ns"]
        or spans[-1]["end_ns"] != result["end_ns"]
        or any(s["start_ns"] > s["end_ns"] for s in spans)
        or any(a["end_ns"] != b["start_ns"] for a, b in zip(spans, spans[1:]))
        or result["duration_s"] != (result["end_ns"] - result["start_ns"]) / 1e9
        or result["exit_code"] != 0
    ):
        raise ValueError("clock_partition_or_exit")
    config = load(result["config_reference"])
    authenticate(result["stderr_reference"])
    if config != dict(head=bundle["head"], extension=bundle["extension"]):
        raise ValueError("session_configuration")
    rebuilt = []
    operation_rebuild = [
        dict(
            operation="startup",
            start_ns=result["start_ns"],
            end_ns=result["ready"]["imports_start_ns"],
        ),
        dict(
            operation="imports",
            start_ns=result["ready"]["imports_start_ns"],
            end_ns=result["ready"]["imports_end_ns"],
        ),
        dict(
            operation="startup",
            start_ns=result["ready"]["imports_end_ns"],
            end_ns=result["ready_ns"],
        ),
    ]
    for i, observed in enumerate(result["observations"]):
        response = observed["response"]
        panel, index = map(int, observed["request"]["request_id"].split("-"))
        operands = bundle["requests"][index : index + result["slot"]["batch"]]
        if (
            observed["request"]["row"]["rows"] != operands
            or observed["request"]["config_hash"] != result["config_reference"]["sha256"]
            or response["status"] != "ok"
            or response["request_id"] != observed["request"]["request_id"]
        ):
            raise ValueError("frame_operands")
        child_spans = response["spans"]
        operation_rebuild.extend(
            [
                dict(
                    operation="parent_residual",
                    start_ns=operation_rebuild[-1]["end_ns"],
                    end_ns=observed["start_ns"],
                ),
                dict(
                    operation="frame_transport",
                    start_ns=observed["start_ns"],
                    end_ns=child_spans[0]["start_ns"],
                ),
                *child_spans,
                dict(
                    operation="frame_transport",
                    start_ns=child_spans[-1]["end_ns"],
                    end_ns=observed["end_ns"],
                ),
            ]
        )
        for operand, answer in zip(operands, response["responses"], strict=True):
            rebuilt.append(
                dict(
                    unit_id=operand["unit_id"],
                    panel=panel,
                    request_index=index,
                    response=json.loads(answer["response"]),
                    latency_s=(
                        observed["end_ns"] - (observed["start_ns"] if i else result["start_ns"])
                    )
                    / 1e9,
                )
            )
    operation_rebuild.append(
        dict(
            operation="shutdown", start_ns=operation_rebuild[-1]["end_ns"], end_ns=result["end_ns"]
        )
    )
    if (
        rebuilt != result["requests"]
        or operation_rebuild != spans
        or len(rebuilt) != 64 * result["slot"]["length"]
        or [(o["request"]["request_id"]) for o in result["observations"]]
        != [
            f"{p}-{i}"
            for p in range(result["slot"]["length"])
            for i in range(0, 64, result["slot"]["batch"])
        ]
    ):
        raise ValueError("primitive_reconstruction")
    references = {r["unit_id"]: r for r in bundle["references"]}
    mismatches = sum(
        not math.isfinite(r["response"]["probability"])
        or r["response"]["unit_id"] != r["unit_id"]
        or r["response"]["decision"] != references[r["unit_id"]]["decision"]
        or abs(r["response"]["probability"] - references[r["unit_id"]]["probability"])
        > 1e-12 * (1 + abs(references[r["unit_id"]]["probability"]))
        for r in rebuilt
    )
    return dict(
        result["slot"],
        intended_count=64 * result["slot"]["length"],
        completed_count=len(rebuilt),
        failed_count=0,
        censored_count=0,
        total_session_wall_s=result["duration_s"],
        throughput_answers_s=len(rebuilt) / result["duration_s"],
        p95_request_latency_s=float(np.quantile([r["latency_s"] for r in rebuilt], 0.95)),
        decision_mismatches=mismatches,
        peak_rss_kib=result["peak_rss_kib"],
        growth_rss_kib=result["growth_rss_kib"],
    )


def reduce(work: Json) -> Json:
    """Bootstrap only complete paired-repeat cells and keep all missing cells visible."""
    pairs, requests, operations, censored = [], [], [], []
    for row in work["rows"]:
        if row["status"] != "completed":
            censored.append(row)
            continue
        primitive = load(row["primitive_reference"])
        if primitive["slot"] != {k: row[k] for k in ["length", "batch", "repeat", "arm"]}:
            raise ValueError("slot_identity")
        pairs.append(
            dict(
                row,
                **{
                    k: v
                    for k, v in summarize_session(primitive, work["bundle"]).items()
                    if k not in {"length", "batch", "repeat", "arm"}
                },
            )
        )
        requests.extend(
            dict(
                r,
                cell_id=row["unit_id"],
                arm=row["arm"],
                length=row["length"],
                batch=row["batch"],
                repeat=row["repeat"],
            )
            for r in primitive["requests"]
        )
        operations.extend(
            dict(
                cell_id=row["unit_id"],
                operation=name,
                duration_s=sum(
                    s["end_ns"] - s["start_ns"]
                    for s in primitive["operations"]
                    if s["operation"] == name
                )
                / 1e9,
            )
            for name in sorted({s["operation"] for s in primitive["operations"]})
        )
    contrasts = []
    for length in [1, 4, 16]:
        for size in [1, 8]:
            for name, a, b in [
                ("primary", "legacy_native", "lean_native"),
                ("lifecycle", "legacy_python", "lean_python"),
                ("arithmetic", "lean_python", "lean_native"),
            ]:
                left = sorted(
                    [r for r in pairs if (r["length"], r["batch"], r["arm"]) == (length, size, a)],
                    key=lambda r: r["repeat"],
                )
                right = sorted(
                    [r for r in pairs if (r["length"], r["batch"], r["arm"]) == (length, size, b)],
                    key=lambda r: r["repeat"],
                )
                full = (
                    [r["repeat"] for r in left] == [r["repeat"] for r in right] == list(range(10))
                )
                ci = (
                    base.interval(
                        [r["total_session_wall_s"] for r in left],
                        [r["total_session_wall_s"] for r in right],
                    )
                    if full
                    else None
                )
                latency = (
                    float(
                        np.quantile(
                            [
                                r["latency_s"]
                                for r in requests
                                if (r["length"], r["batch"], r["arm"]) == (length, size, b)
                            ],
                            0.95,
                        )
                        / np.quantile(
                            [
                                r["latency_s"]
                                for r in requests
                                if (r["length"], r["batch"], r["arm"]) == (length, size, a)
                            ],
                            0.95,
                        )
                    )
                    if full
                    else None
                )
                contrasts.append(
                    dict(
                        contrast=name,
                        length=length,
                        batch=size,
                        baseline=a,
                        treatment=b,
                        throughput_ratio_ci95=ci,
                        p95_request_latency_ratio=latency,
                        paired_repeats=10 if full else 0,
                        full_parity=full
                        and all(r["decision_mismatches"] == 0 for r in left + right),
                        descriptive=(length, size) != (16, 1),
                    )
                )
    return dict(
        paired_session_rows=pairs,
        per_request_rows=requests,
        operation_cost_rows=operations,
        censoring_rows=censored,
        throughput_intervals=contrasts,
    )


def artifact(work: Json, receipts: list[Json], output: Path) -> Json:
    """Readiness qualifies a full table even when both scientific benefit gates are zero."""
    reduced = reduce(work)
    request_path = Path(work["raw"]) / "per_request_rows.json"
    request_rows = reduced.pop("per_request_rows")
    if request_path.exists():
        if json.loads(request_path.read_bytes()) != dict(rows=request_rows):
            raise ValueError("per_request_reduction")
    else:
        atomic_json(request_path, dict(rows=request_rows))
    checks = {r["name"]: r["passed"] for r in receipts}
    for receipt in receipts:
        if "argv" in receipt:
            checks[receipt["name"]] = (
                checks[receipt["name"]]
                and receipt["normal_exit"]
                and not receipt["timed_out"]
                and receipt["exit_code"] == receipt["expected_exit"]
            )
            for stream in ["stdout", "stderr"]:
                authenticate(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
    required = REQUIRED <= checks.keys() and all(checks.values())
    clean = len(reduced["paired_session_rows"]) == 240 and all(
        r["decision_mismatches"] == 0 for r in reduced["paired_session_rows"]
    )
    ready = int(required and clean and not work["gate_check_summary"])
    primary, arithmetic, lifecycle = [
        next(
            r
            for r in reduced["throughput_intervals"]
            if (r["contrast"], r["length"], r["batch"]) == (name, 16, 1)
        )
        for name in ["primary", "arithmetic", "lifecycle"]
    ]
    service, _ = base.benefit(
        primary["throughput_ratio_ci95"],
        primary["p95_request_latency_ratio"],
        bool(ready and primary["full_parity"]),
    )
    _, nfr = base.benefit(
        arithmetic["throughput_ratio_ci95"],
        arithmetic["p95_request_latency_ratio"],
        bool(ready and arithmetic["full_parity"]),
    )
    verdict = (
        "disqualified"
        if any(not r["passed"] for r in receipts)
        or any(r["status"] == "failed" for r in work["rows"])
        else "blocked"
        if work["gate_check_summary"]
        else "partial"
        if not clean
        else "disqualified"
        if not required
        else "circular_positive"
        if service
        else "null"
    )
    bundle = work["bundle"] or {}
    value: Json = dict(
        experiment_id=8622,
        task_id=TASK,
        milestone="2026.10.740",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v740_lean_resident_cost",
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
        inference_substrate="CPU numerical spline energy inference with complete resident clocks",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        current_llm_invocations=0,
        historical_model_provenance=bundle.get("historical_model_provenance", []),
        target_model=bundle.get("target_model"),
        rows=reduced["paired_session_rows"] + reduced["censoring_rows"],
        intended_count=240,
        completed_count=len(reduced["paired_session_rows"]),
        failed_count=sum(r["status"] == "failed" for r in work["rows"]),
        censored_count=len(reduced["censoring_rows"])
        - sum(r["status"] == "failed" for r in work["rows"]),
        excluded_count=0,
        independent_count=int(bool(reduced["paired_session_rows"])),
        sample_size_budget=dict(
            hosts=1,
            arms=4,
            paired_repeats=10,
            lengths=[1, 4, 16],
            batches=[1, 8],
            requests_per_panel=64,
            intended_cells=240,
            intended_answer_observations=107520,
            bootstrap_draws=10000,
            bootstrap_seed=7368573,
            bootstrap_unit="paired repeat index; one host",
        ),
        verifier_is_oracle=True,
        exposure_scope="original exposed64 answers, repeated on one host",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=required,
        flagged_adversarial=verdict == "disqualified",
        acceptance_gates=dict(
            original_inputs=bool(bundle), complete_measurement=clean, required_validation=required
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=(work["ended_ns"] - work["started_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=7368573,
        source_artifact_hashes=bundle.get("refs", []),
        code_config_hashes=[reference(ROOT / p) for p in OWNED + TESTS],
        raw_shard_hashes=[
            r["primitive_reference"] for r in work["rows"] if r["primitive_reference"]
        ],
        cited_upstream_artifacts=bundle.get("refs", []),
        lean_cost_ready_score=ready,
        lean_service_benefit_score=service,
        nfr01_score=nfr,
        branch_gate_values=dict(
            lean_cost_ready_score=ready, lean_service_benefit_score=service, nfr01_score=nfr
        ),
        per_request_rows_reference=reference(request_path),
        latency_reconstruction="Disjoint spans reconstruct integer launch-through-clean-exit clocks. First-frame request latency includes startup; later frames start at send. Batch members share frame latency.",
        primary_contrast=primary,
        arithmetic_contrast=arithmetic,
        lifecycle_contrast=lifecycle,
        work_reference=reference(Path(work["raw"]) / "work.json"),
        publication_output=str(output),
        first_cell_forecast=work.get("first_cell_forecast"),
        historical_cost_null=bundle.get("historical_cost_null"),
        preserved_quarantines=bundle.get("preserved_quarantines", []),
        measurement_code_config_hashes=work["measurement_code_config_hashes"],
        methodology_note="Ten timings on one host are not ten independent deployments. No cached features, results or language-model calls. Old cold and native findings remain unchanged.",
        **reduced,
    )
    value["field_principles"] = {
        k: "Authenticated invocation evidence; missing cells retain reasons and no population inference follows."
        for k in value
    }
    value["field_principles"].update(
        lean_cost_ready_score="One only for full240 cells, parity and all required checks; a qualified null remains ready.",
        lean_service_benefit_score="Primary lean_native/legacy_native throughput lower95>1, p95<=1.05 and exact decisions.",
        nfr01_score="Unchanged native/Python gate within lean lifecycle: throughput lower95>=10, p95<=1.05 and parity.",
        operation_cost_rows="Nonoverlapping integer endpoints sum exactly to each complete process wall.",
        throughput_intervals="10000 paired-repeat resamples seed7368573; other cells descriptive.",
        target_model="Actual Exp8431 numerical checkpoint; no language-model load does not mean no numerical inference.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> None:
    """Fresh reduction rejects rehashed summaries and unbound original authority."""
    value = json.loads(path.read_bytes())
    if checksum(value) != value["reproducibility_checksum"]:
        raise ValueError("checksum")
    work = load(value["work_reference"])
    if work["bundle"] and admit(Path(work["input_root"])) != work["bundle"]:
        raise ValueError("original_authority")
    for ref in (
        value["code_config_hashes"]
        + value["raw_shard_hashes"]
        + value["measurement_code_config_hashes"]
    ):
        authenticate(ref)
    for receipt in value["validation_receipts"]:
        if receipt.get("data_reference"):
            authenticate(receipt["data_reference"])
            authenticate(receipt["report"])
        for stream in ["stdout", "stderr"]:
            if stream + "_path" in receipt:
                authenticate(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
    if artifact(work, value["validation_receipts"], Path(value["publication_output"])) != value:
        raise ValueError("reduction_drift")
    progress("8622_cold_replay_passed")
