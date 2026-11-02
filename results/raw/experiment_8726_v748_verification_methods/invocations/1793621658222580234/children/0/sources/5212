"""REQ-REPORT-8572: qualification binds original operands and immutable snapshots."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot.reporting import resident_inference_8571 as upstream
from carnot.reporting import resident_measure_8572 as m
from carnot.reporting import resident_worker_8572 as w
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json, sha256_file
from carnot.verify import nine_input_8533 as numeric
from carnot.verify.native_nine_input_8533 import load

Json = dict[str, Any]
ROOT = upstream.ROOT
NAME = "experiment_8572_v736_resident_inference_qualification"
TASK = "exp8572-resident-inference-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_resident_qualification_8572.py"
OWNED = [
    f"python/carnot/reporting/resident_{n}_8572.py"
    for n in ["worker", "measure", "qualification", "runner", "validation"]
] + [CLI]
GateFailure, require, reference, read, checksum = (
    upstream.GateFailure,
    upstream.require,
    upstream.reference,
    upstream.read,
    upstream.checksum,
)
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


def admit(root: Path) -> Json:
    """Reuse original admission and separately authenticate the ready prototype and protocol."""
    bundle = upstream.admit(root)
    prototype_ref = dict(
        path=str(root / "results" / (upstream.NAME + ".json")),
        sha256="sha256:0fc4bc1e2a5f65960292af3f6ecbea9255d7387946e9047d41b554df980336f3",
    )
    prototype = read(prototype_ref)
    for key, expected in dict(
        prototype_ready_score=1, required_checks_passed=True, flagged_adversarial=False
    ).items():
        require(Path(prototype_ref["path"]), key, expected, prototype.get(key))
    parity_ref = bundle["refs"][0]
    parity = read(parity_ref)
    tape = read(parity["request_tape_reference"])
    answers = tape["rows"]
    controls = [
        dict(r, kind="numerical")
        for r in parity["parity_rows"]
        if r["kind"] in {"knot", "zero", "decay"}
    ]
    require(root, "answer_count", 900, len(answers))
    require(root, "numerical_control_count", 274, len(controls))
    protocol_ref = reference(ROOT / "openspec/change-proposals/v736-methods-protocol.json")
    protocol = read(protocol_ref)["resident_session"]
    contract_ref = dict(
        path=str(root / "results/experiment_8565_v736_contract_methods.json"),
        sha256="sha256:66843de62020d69730b2796f0966ca2038d36ea49e02c337c6dd0071f69a0dd4",
    )
    contract = read(contract_ref)
    return dict(
        bundle,
        answers=answers,
        controls=controls,
        protocol=protocol,
        prototype_ready_score=prototype["prototype_ready_score"],
        contract_primary_verdict=contract["honest_verdict"],
        refs=bundle["refs"] + [prototype_ref, contract_ref, protocol_ref],
    )


def summarize(primitive: Json, bundle: Json) -> Json:
    """Re-evaluate immutable rows and retain every arm rather than trusting success counters."""
    groups = m.roster(bundle)
    require(
        ROOT,
        "session_roster",
        [g["name"] for g in groups],
        [s["name"] for s in primitive["sessions"]],
    )
    native = load(Path(bundle["extension"]["path"]))
    cells: dict[str, list[Json]] = {"answers": [], "controls": [], "isolation": []}
    clocks = []
    expected_cache: dict[str, Json] = {}
    reference_cache: dict[str, Json] = {}
    for group, session in zip(groups, primitive["sessions"], strict=True):
        require(ROOT, "session_exit", 0, session["exit_code"])
        require(ROOT, "configuration_loads", 1, session["ready"]["config_loads"])
        expected_loaded = (
            {k: bundle["extension"][k] for k in ["path", "sha256"]}
            if group["arm"] == "native"
            else None
        )
        require(
            ROOT, "actual_loaded_extension", expected_loaded, session["ready"]["loaded_extension"]
        )
        require(ROOT, "response_count", len(group["rows"]), len(session["observations"]))
        config = read(dict(path=session["config_path"], sha256=session["config_hash"]))
        require(
            ROOT,
            "immutable_configuration",
            dict(head=group["head"], extension=bundle["extension"], rows=group["rows"]),
            config,
        )
        for i, (row, observed) in enumerate(
            zip(group["rows"], session["observations"], strict=True)
        ):
            if i % 100 == 0:
                m.progress("8572_reconstruct_" + group["name"], i, len(group["rows"]) - i)
            actual = observed["response"]["response"]
            operand_key = json.dumps([group["head"], row], sort_keys=True)
            expected_key = group["arm"] + operand_key
            if expected_key not in expected_cache:
                expected_cache[expected_key] = w.evaluate(
                    group["head"], row, native if group["arm"] == "native" else None
                )[0]
            expected = expected_cache[expected_key]
            require(ROOT, "recomputed_response", expected, actual)
            if group["category"] == "answers":
                if operand_key not in reference_cache:
                    reference_cache[operand_key] = json.loads(numeric.request(group["head"], row))
                ref = reference_cache[operand_key]
            elif group["category"] == "controls":
                ref = dict(
                    probability=row["reference_probability"],
                    decision=numeric.action(row["reference_probability"]),
                    residual=row["reference_residual"],
                )
            else:
                ref, _ = w.evaluate(group["head"], row, None)
            error = abs(actual["probability"] - ref["probability"])
            passed = (
                actual == expected
                and error <= 1e-12 * (1 + abs(ref["probability"]))
                and actual["decision"] == ref["decision"]
            )
            if "residual" in ref:
                passed = passed and abs(actual["residual"] - ref["residual"]) <= 1e-12 * (
                    1 + abs(ref["residual"])
                )
            if group["kind"] == "resident":
                request = observed["request"]
                passed = (
                    passed and request["row"] == row and request["request_id"] == f"request-{i}"
                )
                passed = (
                    passed
                    and observed["response"]["request_id"] == request["request_id"]
                    and observed["response"]["config_hash"] == session["config_hash"]
                )
            cells[group["category"]].append(
                dict(
                    unit_id=row["unit_id"],
                    group=group["name"],
                    arm=group["arm"],
                    passed=bool(passed),
                    probability_error=error,
                    decision_mismatch=actual["decision"] != ref["decision"],
                    boundary_fallback=actual["boundary_fallback"],
                    response=actual,
                    missing_reason=None if passed else "parity_failure",
                )
            )
        clocks.append(dict(name=group["name"], **m.partition(session, group["kind"])))
    answers = [
        dict(
            unit_id=bundle["answers"][i]["unit_id"],
            passed=all(cells["answers"][i + 900 * j]["passed"] for j in range(4)),
            comparisons=[cells["answers"][i + 900 * j] for j in range(4)],
            missing_reason=None,
        )
        for i in range(900)
    ]
    controls = [
        dict(
            unit_id=bundle["controls"][i]["unit_id"],
            passed=all(cells["controls"][i + 274 * j]["passed"] for j in range(4)),
            comparisons=[cells["controls"][i + 274 * j] for j in range(4)],
            missing_reason=None,
        )
        for i in range(274)
    ]
    rows = answers + controls + cells["isolation"] + primitive["failure_control_rows"]
    return dict(
        parity_rows=answers,
        numerical_control_rows=controls,
        request_isolation_rows=cells["isolation"],
        rows=rows,
        clock_partition_receipt=dict(
            passed=all(c["passed"] for c in clocks),
            sessions=clocks,
            no_overlapping_intervals=True,
            timing_benefit_inferred=False,
        ),
        loaded_extension_hash=bundle["extension"]["sha256"],
    )


execute, failure_controls = m.execute, m.failure_controls


@lru_cache(maxsize=4)
def _reconstructed_summary(primitive_json: str, bundle_json: str) -> bool:
    """A process-local cache saves repeated reductions; every cold child starts empty."""
    primitive, bundle = json.loads(primitive_json), json.loads(bundle_json)
    summary = summarize(primitive, bundle)
    return all(primitive[k] == value for k, value in summary.items())


def replay_primitive(primitive: Json, bundle: Json) -> bool:
    """Cold recomputation checks observations, transport outcomes and exact log bytes."""
    try:
        require(
            ROOT,
            "extension_bytes",
            bundle["extension"]["sha256"],
            sha256_file(Path(bundle["extension"]["path"])),
        )
        for session in primitive["sessions"]:
            require(
                ROOT,
                "configuration_bytes",
                session["config_hash"],
                sha256_file(Path(session["config_path"])),
            )
        if not _reconstructed_summary(
            json.dumps(primitive, sort_keys=True), json.dumps(bundle, sort_keys=True)
        ):
            return False
        require(
            ROOT,
            "failure_control_roster",
            [
                "malformed",
                "oversize",
                "eof_mid_frame",
                "crash",
                "deadline",
                "full_queue",
                "duplicate_id",
                "shutdown",
                "configuration_change",
                "idle_deadline",
            ],
            [r["kind"] for r in primitive["failure_control_rows"]],
        )
        for row in primitive["failure_control_rows"]:
            lifecycle = row["lifecycle"]
            expected_exit = (
                0
                if row["kind"] in {"malformed", "deadline", "duplicate_id", "shutdown"}
                else -15
                if row["kind"] in {"crash", "full_queue"}
                else 1
            )
            expected_status = (
                "error"
                if row["kind"] in {"malformed", "deadline", "duplicate_id"}
                else "shutdown"
                if row["kind"] == "shutdown"
                else "censored"
            )
            require(ROOT, "failure_exit", expected_exit, lifecycle["exit_code"])
            require(ROOT, "failure_status", expected_status, row["response"]["status"])
            require(ROOT, "failure_control_passed", True, row["passed"])
            require(
                ROOT,
                "failure_outcome",
                True,
                row["response"]["status"] in {"error", "censored", "shutdown"},
            )
            require(ROOT, "no_stale_response", False, "response" in row["response"])
            require(
                ROOT,
                "failure_log",
                lifecycle["stderr_sha256"],
                sha256_file(Path(lifecycle["stderr_path"])),
            )
        for session in primitive["sessions"]:
            require(
                ROOT,
                "session_log",
                session["stderr_sha256"],
                sha256_file(Path(session["stderr_path"])),
            )
        return True
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False


def measure(root: Path, raw: Path) -> Json:
    """Reuse CPU resource checks without invoking any prior measurement controller."""
    from tempfile import TemporaryDirectory
    import shutil

    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    os.environ["LLVM_PROFILE_FILE"] = str(raw / "native-%p-%m.profraw")
    work: Json = dict(
        raw=str(raw), bundle=None, primitive=None, gate_check_summary=[], phase_spans=[]
    )
    with TemporaryDirectory(prefix="exp8572-", dir="/var/tmp") as scratch:
        probe = Path(scratch) / "probe"
        probe.write_bytes(b"disk-backed")
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
        mounted = max(
            (m for m in mounts if Path(scratch).is_relative_to(m[4])), key=lambda m: len(m[4])
        )
        fs = mounted[mounted.index("-") + 1]
        checks: Json = dict(
            scratch=scratch,
            writable=probe.read_bytes() == b"disk-backed",
            scratch_filesystem=fs,
            free_disk_bytes=shutil.disk_usage(scratch).free,
            memory_available_kib=int(memory["MemAvailable"].split()[0]),
            tools={
                n: shutil.which(str(ROOT / ".venv/bin" / n))
                for n in ["python", "pytest", "ruff", "mypy", "coverage"]
            },
            measurement_deadline_s=3600,
            validation_reserve_s=900,
            task_cap_s=4800,
            optional_cuda="not_required_for_CPU",
            optional_board="not_required_for_CPU",
        )
        work["preconditions_checked"] = checks
        try:
            for key, passed in dict(
                writable_disk=checks["writable"],
                disk_backed=fs not in {"tmpfs", "ramfs"},
                memory=checks["memory_available_kib"] >= 512 * 1024,
                disk=checks["free_disk_bytes"] >= 64 * 1024 * 1024,
                tools=all(checks["tools"].values()),
            ).items():
                require(Path(scratch), key, True, passed)
            work["bundle"] = admit(root)
            admitted = time.monotonic_ns()
            work["primitive"] = execute(work["bundle"], raw / "measurements")
            end = time.monotonic_ns()
            work["phase_spans"] = [
                dict(phase="admission", start_ns=start, end_ns=admitted),
                dict(phase="qualification", start_ns=admitted, end_ns=end),
            ]
            require(raw, "measurement_deadline", True, (end - start) / 1e9 <= 3600)
        except GateFailure as error:
            work["gate_check_summary"].append(error.row)
            if work["bundle"] is not None:
                work["owned_failure"] = str(error)
        except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
            work["owned_failure"] = str(error)
        work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    return work


def seal(work: Json, raw: Path, phase: str) -> Json:
    """Each candidate gets a separate read-only primitive snapshot before reduction."""
    directory = raw / "snapshots" / phase
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "primitive.json"
    atomic_json(path, work)
    path.chmod(0o444)
    return dict(work=deepcopy(work), reference=reference(path))


def artifact(sealed: Json, receipts: list[Json], output: Path) -> Json:
    """A null qualification grants readiness only after every required check passes."""
    work = sealed["work"]
    if read(sealed["reference"]) != work:
        raise ValueError("snapshot_drift")
    bundle, primitive = work["bundle"], work["primitive"]
    rows = primitive["rows"] if primitive else []
    required = REQUIRED <= {r["name"] for r in receipts} and all(r["passed"] for r in receipts)
    clean = (
        bool(primitive)
        and all(r["passed"] for r in rows)
        and primitive["clock_partition_receipt"]["passed"]
    )
    ready = clean and required and not work["gate_check_summary"] and not work.get("owned_failure")
    verdict = (
        "disqualified"
        if work.get("owned_failure") or any(not r["passed"] for r in receipts)
        else "blocked"
        if work["gate_check_summary"]
        else "null"
        if ready
        else "disqualified"
    )
    value: Json = dict(
        experiment_id=8572,
        task_id=TASK,
        milestone="2026.10.736",
        run_date="20261016",
        honest_verdict=f"complete_{verdict}_v736_resident_inference_qualification",
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
        inference_substrate="CPU transformation of frozen authentic outputs, no LLM; resident Python/native CPU arithmetic",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        current_llm_invocations=0,
        historical_model_provenance=bundle["historical_model_provenance"] if bundle else [],
        rows=rows,
        intended_count=1214,
        completed_count=len(rows),
        failed_count=sum(not r["passed"] for r in rows),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            original_answers=900,
            numerical_controls=274,
            isolation_requests=30,
            failure_controls=10,
            independent_workloads=0,
            comparative_timing_trials=0,
        ),
        verifier_is_oracle=True,
        exposure_scope="original exposed answer inputs and private numerical/lifecycle fixtures",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        resident_inference_ready_score=int(ready),
        required_checks_passed=required,
        flagged_adversarial=bool(work.get("adversarial_findings")) or verdict == "disqualified",
        acceptance_gates=dict(
            parity_and_isolation=clean,
            required_validation=required,
            clock_accounting=bool(primitive) and primitive["clock_partition_receipt"]["passed"],
            speed_benefit_measured=False,
            harmful_decision_changes_allowed=0,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        development_diagnostics=work.get("development_diagnostics", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7368572,
        source_artifact_hashes=bundle["refs"] if bundle else [],
        code_config_hashes=[reference(ROOT / p) for p in [*OWNED, TEST]],
        raw_shard_hashes=[sealed["reference"], *work.get("development_diagnostics", [])],
        primitive_snapshots=[*work.get("snapshot_history", []), sealed["reference"]],
        cited_upstream_artifacts=[8571, 8565, 8558, 8559],
        primitive_reference=sealed["reference"],
        publication_output=str(output.absolute()),
        cost_consumer_receipt=work.get("cost_consumer_receipt"),
        original_authority=bundle["original_authority"] if bundle else None,
        historical_cost_null=bundle["historical_cost_null"] if bundle else None,
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8573, 8579)
        ],
        python_coverage=work.get("python_coverage"),
        exact_union_receipt=work.get("exact_union_receipt"),
        preserved_quarantines="All historical findings and original result bytes remain authoritative.",
        methodology_note="Qualification only. No timing benefit is inferred from fixtures or launch-through-exit clock accounting.",
    )
    for key in [
        "parity_rows",
        "numerical_control_rows",
        "request_isolation_rows",
        "failure_control_rows",
        "clock_partition_receipt",
        "loaded_extension_hash",
    ]:
        value[key] = primitive[key] if primitive else None
    value["field_principles"] = {
        k: "Bind original immutable evidence to this invocation; qualification grants no benefit or independence."
        for k in value
    }
    value["field_principles"].update(
        resident_inference_ready_score="One requires all four-path parity checks, transport controls, owned coverage and complete disjoint clocks; it confers no timing benefit.",
        parity_rows="Exactly900 original exposed answer units; four execution paths are dependent comparisons of each unit.",
        numerical_control_rows="Exactly274 immutable knot/endpoint/zero/decay operands; each retains reference probabilities and residuals.",
        request_isolation_rows="Thirty requests cover duplicate text with distinct IDs, reversed order, restarts, changed temperature and six native boundary fallbacks.",
        failure_control_rows="Ten private controls retain actual process exits, errors or censored outcomes; intentional censoring is a completed control rather than an unmeasured planned unit.",
        clock_partition_receipt="All intervals lie within parent launch-through-exit time, are disjoint, and leave every remaining nanosecond visible as unattributed overhead.",
        loaded_extension_hash="SHA256 of the actual original Rust/PyO3 module imported in every native child.",
        primitive_snapshots="Each read-only snapshot precedes its candidate and cannot be overwritten by a later validation phase.",
        cost_consumer_receipt="The actual complete_native_worker_8559 consumes all64 original request inputs after replay; no fixture speed claim follows.",
        python_coverage="Only newly owned statements count, combining genuine Python and native worker-process Python coverage shards.",
        exact_union_receipt="All collected original node IDs equal the disjoint union of terminal passes under original suite authority.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rehashing cannot alter roster, response arithmetic, acceptance or immutable operands."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != checksum(value):
            return False
        work = read(value["primitive_reference"])
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            require(Path(ref["path"]), "replay_hash", ref["sha256"], sha256_file(Path(ref["path"])))
        reconstructed = artifact(
            dict(work=work, reference=value["primitive_reference"]),
            value["validation_receipts"],
            Path(value["publication_output"]),
        )
        if value != reconstructed:
            return False
        return not work["bundle"] or (
            work["bundle"] == admit(ROOT)
            and (work["primitive"] is None or replay_primitive(work["primitive"], work["bundle"]))
        )
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False


def consume(path: Path) -> Json:
    """The resident cost adapter admits complete qualification without asserting speed."""
    value = json.loads(path.read_bytes())
    if value["resident_inference_ready_score"] != 1 or not replay(path):
        raise ValueError("not_ready")
    from carnot.reporting.complete_native_worker_8559 import execute as cost_execute

    bundle = read(value["primitive_reference"])["bundle"]
    result = cost_execute(bundle, "native", 1)
    require(path, "actual_cost_consumer_requests", 64, len(result["requests"]))
    errors = []
    for actual, row in zip(result["requests"], bundle["requests"], strict=True):
        response, expected = (
            json.loads(actual["response"]),
            json.loads(numeric.request(bundle["head"], row)),
        )
        error = abs(response["probability"] - expected["probability"])
        require(path, "actual_cost_consumer_decision", expected["decision"], response["decision"])
        require(
            path,
            "actual_cost_consumer_probability",
            True,
            error <= 1e-12 * (1 + abs(expected["probability"])),
        )
        errors.append(error)
    return dict(
        passed=True,
        consumer="complete_native_worker_8559.execute via resident_cost_input_adapter_8572",
        actual_consumer_requests=len(result["requests"]),
        max_probability_error=max(errors),
        actual_consumer_loaded_extension=result["loaded_extension"],
        primary=reference(path),
        protocol=value["original_authority"],
        speed_benefit_inferred=False,
    )
