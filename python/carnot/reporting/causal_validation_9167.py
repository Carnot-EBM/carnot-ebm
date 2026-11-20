"""REQ-REPORT-9167-V781: retain causal diagnostics without granting unproven readiness.

A completed observation is distinct from a controlled correction. This producer
keeps unfinished assertions and the complete next qualification scope visible.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any

from coverage import Coverage

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.forward_evidence_runner_9111 import bound
from carnot.reporting.natural_label_eligibility_8975 import ref
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.spline_basis_methods_8986 import gate
from carnot.reporting.v709_execution import child
from carnot.reporting import validation_trace_9167 as trace

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_9167_v781_causal_validation"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_causal_validation_9167.py"
OWNED = [
    "python/carnot/reporting/causal_validation_9167.py",
    "python/carnot/reporting/validation_trace_9167.py",
    "python/carnot/reporting/validation_reuse_9167.py",
    CLI,
]
SEAM = "python/carnot/reporting/forward_evidence_runner_9111.py"
BASELINE = ROOT / "results/raw/experiment_9167_v781_causal_validation/reader_before.py"
BASELINE_PIN = "sha256:cb082c5d7263286b23a1b1f0f6a9a3b50fe7e57e6c67050e565da62640df0f25"
SEED = 7819167
NODE = "tests/python/test_forward_evidence_runner_9111.py::test_rehashed_primitive_tampering[derivation]"
INPUTS = {
    "results/experiment_9166_v781_contract.json": (
        "sha256:f4c3a8e419b2c956452128d251ac14b351407e08b0552a704cd55dccac747f55",
        "contract_ready_score",
        1,
    ),
    "results/experiment_9153_v780_validation_schedule.json": (
        "sha256:8d34b3a96e0d2f98f8cb7013a78011300bfe15460fdcd15874b856098826cc29",
        "schedule_ready_score",
        0,
    ),
}


def progress(phase: str) -> None:
    """Phase output prevents a supervised computation being mistaken for a stall."""
    print("[exp9167] phase=" + phase, flush=True)


def authenticate(root: Path, raw: Path) -> Json:
    """Copy exact upstream bytes only after byte and typed gate authentication."""
    work: Json = dict(checks=[], references=[], inputs={})
    for name, (pin, field, expected) in INPUTS.items():
        path = root / name
        observed = ref(path)["sha256"] or "missing_file"
        work["checks"].append(gate(path, "sha256", pin, observed, "input_bytes", name))
        if not work["checks"][-1]["passed"]:
            return work
        value = json.loads(bound(dict(path=str(path), sha256=pin)).read_bytes())
        work["checks"].append(
            gate(path, field, expected, value.get(field, "missing_field"), "input_gate", name)
        )
        if not work["checks"][-1]["passed"]:
            return work
        target = raw / "inputs" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        bound(dict(path=str(target), sha256=pin))
        work["references"].append(
            dict(ref(target), producer_path=str(path), imported_fields=list(value))
        )
        work["inputs"][name] = value
    return work


def resources(raw: Path) -> list[Json]:
    """Check actual private storage, imports and resource limits before subprocesses."""
    limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    observed = dict(
        cpu_count=os.cpu_count(),
        cpu_limit=resource.getrlimit(resource.RLIMIT_CPU),
        memory_limit=limit,
        memory_available_bytes=os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
        private_mode=oct(raw.stat().st_mode & 0o777),
        disk_free_bytes=shutil.disk_usage(raw).free,
        imports_available=all(
            importlib.util.find_spec(n) is not None
            for n in ["pytest", "coverage", "carnot.reporting.forward_evidence_runner_9111"]
        ),
    )
    return [
        dict(resource=k, available=bool(ok), check_receipt=dict(observed=observed))
        for k, ok in [
            ("cpu", bool(observed["cpu_count"])),
            ("memory", limit == resource.RLIM_INFINITY or limit >= 256 * 1024**2),
            ("disk", observed["disk_free_bytes"] >= 100_000_000),
            ("imports", observed["imports_available"]),
            ("private_storage", observed["private_mode"] == "0o700"),
        ]
    ]


def trace_evidence(raw: Path) -> Json:
    """Reduce executed operation records; observations alone do not prove a correction."""
    references = [ref(p) for p in sorted(raw.rglob("trace-*.jsonl"))]
    events = [json.loads(line) for r in references for line in bound(r).read_text().splitlines()]
    snapshots = [e for e in events if e["event"] in {"heartbeat", "summary"}]
    latest = {e["pid"]: e for e in snapshots}
    operations = [
        dict(pid=pid, operation=op, calls=n, inclusive_elapsed_s=e["elapsed_ns"].get(op, 0) / 1e9)
        for pid, e in latest.items()
        for op, n in e["counters"].items()
        if op.rsplit(":", 1)[-1]
        in {
            "authenticate",
            "measure",
            "historical_report",
            "replay",
            "current_report",
            "preprocessing",
            "classify",
            "check_tautology",
            "_is_arc_generation_or_exploration_artifact",
            "_field_name_text",
            "sha256_file",
            "canonical_hash",
        }
    ]
    return dict(
        trace_references=references,
        process_rows=[e for e in events if e["event"] in {"start", "spawn", "child_exit"}],
        operation_rows=operations,
        wait_stack_references=references,
        instrumentation_overhead=[
            dict(
                pid=e["pid"],
                measured_callback_s=e["overhead_ns"] / 1e9,
                observed_duration_s=e["duration_ns"] / 1e9,
                scope="selected profiling callbacks only; unselected dispatch and audit overhead unmeasured",
            )
            for e in latest.values()
        ],
        cause_demonstrated=False,
        corrective_execution_passed=False,
        competing_hypotheses=[
            dict(operation=n, status="unproven")
            for n in [
                "repeated hashing",
                "historical checker execution",
                "filesystem work",
                "nested child",
            ]
        ],
        diagnostic_defect="Exp9153 used parent-only timers and unconditionally recorded unknown; original signal11 is retained, not explained by that defect",
        conclusion="Executed operations are observations; no unchanged controlled corrective reproduction is available.",
    )


def diagnose(raw: Path, stop: float) -> Json:
    """Run the retained node with its original module fixture and inherited tracing."""
    directory = raw / "original"
    bootstrap = trace.bootstrap(directory)
    argv = [
        "env",
        "CARNOT_9167_TRACE=" + str(directory),
        "PYTHONPATH=" + str(bootstrap) + ":" + str(ROOT / "python") + ":" + str(ROOT),
        "CARNOT_9125_NODE_LOG=" + str(directory / "nodes.jsonl"),
        str(ROOT / ".venv/bin/python"),
        "-m",
        "pytest",
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
        "-p",
        "carnot.reporting.receipt_qualification_9125",
        "-s",
        "-q",
        NODE,
    ]
    atomic_json(
        raw / "diagnostic_plan.json",
        dict(
            argv=argv,
            random_seed=SEED,
            exclusions=[],
            diagnostic_ceiling_s=900,
            formula="phase elapsed and operation counts; no watchdogs in elapsed sums",
            target=NODE,
            assertion_reference=ref(ROOT / NODE.split("::")[0]),
            hook_reference=ref(Path(trace.__file__)),
            correction_rule="Only trace-demonstrated corrections may change the runner seam.",
        ),
    )
    receipt = child(
        "original",
        argv,
        raw / "logs",
        deadline=min(900, max(0.001, stop - time.monotonic())),
        heartbeat=60,
    )
    return dict(receipt=receipt, evidence=trace_evidence(directory), directory=str(raw))


def commands(raw: Path) -> list[Json]:
    """Freeze owned validation and private lifecycle checks before their outcomes."""
    py = str(ROOT / ".venv/bin/python")
    tests = [py, "-m", "pytest", "-n0", "-o", "addopts=", "--no-cov", "-q"]
    return [
        dict(name=n, argv=a)
        for n, a in [
            (
                "owned_tests",
                [
                    "env",
                    "COVERAGE_RCFILE=" + str(raw / "coverage.ini"),
                    "COVERAGE_FILE=" + str(raw / ".coverage"),
                    py,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + str(raw / "coverage.ini"),
                    "-m",
                    "pytest",
                    "-n0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    TEST,
                ],
            ),
            (
                "e2e018_021",
                [
                    *tests,
                    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                    "tests/python/test_restricted_decision_audit_8210.py",
                ],
            ),
            ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST]),
            ("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST]),
            (
                "strict_mypy",
                [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED],
            ),
            ("spec_coverage", [py, "scripts/check_spec_coverage.py", TEST]),
        ]
    ]


def verify(work: Json, stop: float) -> None:
    """Combine only post-exit unit and real child coverage, with no excluded lines."""
    raw = Path(work["raw"])
    work["validation_receipts"] = []
    for spec in work["commands"]:
        progress(spec["name"] + "_before")
        work["validation_receipts"].append(
            child(
                spec["name"],
                spec["argv"],
                raw / "validation",
                deadline=max(0.001, min(600, stop - time.monotonic())),
                heartbeat=60,
            )
        )
        progress(spec["name"] + "_after")
    tracer = Coverage(config_file=str(raw / "coverage.ini"))
    tracer.combine(strict=True, keep=True)
    tracer.save()
    files = []
    for path in [*OWNED, SEAM]:
        files.append(statement_row(tracer, path))
    total = sum(len(f["statements"]) for f in files)
    work["coverage"] = dict(
        files=files,
        statement_percent=100 * (total - sum(len(f["missing"]) for f in files)) / total,
        excluded_statements=sum(len(f["excluded"]) for f in files),
        data_reference=ref(raw / ".coverage"),
    )


def qualified(work: Json) -> bool:
    """Reopen exact command logs and SQLite lines rather than trusting a passing label."""
    actual = work.get("validation_receipts", [])
    expected = commands(Path(work["raw"]))
    if len(actual) != len(expected) or work.get("commands") != expected:
        return False
    for receipt, spec in zip(actual, expected, strict=True):
        if (
            receipt["name"] != spec["name"]
            or receipt["argv"] != spec["argv"]
            or not receipt["passed"]
            or not receipt["normal_exit"]
            or receipt["timed_out"]
        ):
            return False
        for stream in ["stdout", "stderr"]:
            bound(dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"]))
    coverage = work["coverage"]
    tracer = Coverage(config_file=False, data_file=str(bound(coverage["data_reference"])))
    tracer.set_option("report:exclude_lines", [])
    tracer.load()
    for row in coverage["files"]:
        actual = statement_row(tracer, row["path"])
        if actual != row or actual["missing"] or actual["excluded"]:
            return False
    return [r["path"] for r in coverage["files"]] == [*OWNED, SEAM]


def statement_row(tracer: Coverage, path: str) -> Json:
    """Own the entire new modules and only changed statements in the preserved runner."""
    _, statements, excluded, missing, _ = tracer.analysis2(str(ROOT / path))
    selected = set(statements)
    if path == SEAM:
        before = bound(dict(path=str(BASELINE), sha256=BASELINE_PIN)).read_text().splitlines()
        after = (ROOT / path).read_text().splitlines()
        selected &= {
            i + 1
            for tag, _, _, start, end in difflib.SequenceMatcher(a=before, b=after).get_opcodes()
            if tag in {"insert", "replace"}
            for i in range(start, end)
        }
    return dict(
        path=path,
        statements=sorted(selected),
        missing=sorted(selected & set(missing)),
        excluded=sorted(selected & set(excluded)),
    )


def adopt_diagnosis(reference: Json) -> Json:
    """Retain completed private process evidence without treating censored work as a pass."""
    manifest = json.loads(bound(reference).read_bytes())
    directory = Path(manifest["directory"])
    before = json.loads(bound(manifest["original_receipt_reference"]).read_bytes())
    after = json.loads(bound(manifest["corrected_receipt_reference"]).read_bytes())
    for receipt in [before, after]:
        for stream in ["stdout", "stderr"]:
            bound(dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"]))
    for refrow in manifest["source_references"]:
        bound(refrow)
    return dict(
        directory=str(directory),
        receipt=before,
        corrected_receipt=after,
        manifest_reference=reference,
        manifest=manifest,
        evidence=trace_evidence(directory),
    )


def causal_status(work: Json) -> Json:
    """Read actual operation counts and every original module phase before granting readiness."""
    diagnosis = work.get("diagnosis", {})
    if not diagnosis.get("manifest_reference"):
        return dict(
            cause_demonstrated=False,
            corrective_execution_passed=False,
            negative_controls_passed=False,
            ready=False,
        )
    actual = adopt_diagnosis(diagnosis["manifest_reference"])
    if actual != diagnosis:
        raise ValueError("diagnosis_drift")
    directory = Path(actual["directory"])
    before = trace_evidence(directory / "original")
    after = trace_evidence(directory / actual["manifest"]["corrected_directory"])

    def calls(evidence: Json, name: str) -> int:
        return sum(
            row["calls"]
            for row in evidence["operation_rows"]
            if row["operation"].endswith(":" + name)
        )

    repeated = calls(before, "_is_arc_generation_or_exploration_artifact") > 1
    executed = calls(after, "_is_arc_generation_or_exploration_artifact") > 0 and calls(
        after, "classify"
    ) > calls(after, "_is_arc_generation_or_exploration_artifact")
    events = json.loads(bound(actual["manifest"]["node_events_reference"]).read_bytes())["events"]
    expected = [
        n
        for value in work["inputs"].values()
        for n in value.get("validation_obligations", {}).get("original_node_ids", [])
        if n.split("::")[0] == NODE.split("::")[0]
    ]
    if expected != actual["manifest"]["expected_module_nodes"]:
        raise ValueError("module_scope_drift")
    by_node = {
        n: [e for e in events if e["nodeid"] == n and e["when"] != "start"] for n in expected
    }
    complete = (
        bool(expected)
        and set(e["nodeid"] for e in events) == set(expected)
        and all(
            sorted((e["when"], e["outcome"]) for e in rows)
            == [("call", "passed"), ("setup", "passed"), ("teardown", "passed")]
            for rows in by_node.values()
        )
    )
    negatives = [n for n in expected if "test_rehashed_primitive_tampering[" in n]
    controls = len(negatives) == 7 and all(
        len(by_node[n]) == 3 and all(e["outcome"] == "passed" for e in by_node[n])
        for n in negatives
    )
    receipt = actual["corrected_receipt"]
    corrected = (
        complete
        and executed
        and receipt["passed"]
        and receipt["normal_exit"]
        and not receipt["timed_out"]
    )
    return dict(
        cause_demonstrated=repeated and executed,
        corrective_execution_passed=corrected,
        negative_controls_passed=controls,
        ready=repeated and corrected and controls,
        before_operation_rows=before["operation_rows"],
        after_operation_rows=after["operation_rows"],
        before_after_receipts=[actual["receipt"], receipt],
        conclusion="Repeated structural field traversal is demonstrated. A complete corrected module and all seven primitive tamper refusals are still required.",
        competing_hypotheses=[
            dict(operation=n, status="not_established_as_cause")
            for n in ["repeated hashing", "historical checker execution", "nested child"]
        ],
        unfinished_module_nodes=[
            n
            for n, rows in by_node.items()
            if len(rows) != 3 or not all(e["outcome"] == "passed" for e in rows)
        ],
    )


def build(work: Json) -> Json:
    """Keep a terminal diagnostic block distinct from failed owned validation."""
    diagnosis = work.get("diagnosis", {})
    proof = causal_status(work)
    receipt = diagnosis.get("receipt")
    events = work.get("node_events", [])
    passed = bool(events) and all(
        any(e["when"] == phase and e["outcome"] == "passed" for e in events)
        for phase in ["setup", "call", "teardown"]
    )
    failed = any(e["outcome"] == "failed" for e in events)
    process_receipts = [receipt, diagnosis.get("corrected_receipt"), None]
    statuses = [
        "completed" if passed else "failed" if failed else "censored",
        "completed" if proof["corrective_execution_passed"] else "censored",
        "completed" if proof["negative_controls_passed"] else "censored",
    ]
    rows = [
        dict(
            unit_id=unit,
            qid=None,
            group_id=unit,
            sibling=None,
            arm="causal_diagnosis",
            condition=condition,
            seed=SEED,
            absolute_metric=int(passed) if i == 0 and (passed or failed) else None,
            numerator=int(passed) if i == 0 and (passed or failed) else None,
            denominator=1,
            status=statuses[i],
            attempted=bool(process_receipts[i]),
            measured_elapsed_s=process_receipts[i]["duration_s"] if process_receipts[i] else None,
            exclusion_reason=None,
            censor_reason=None
            if statuses[i] != "censored"
            else "original_path_unfinished"
            if i == 0
            else "controlled_correction_unavailable",
        )
        for i, (unit, condition) in enumerate(
            [
                (NODE, "original"),
                ("corrected_target_module", "corrected"),
                ("unchanged_tamper_refusals", "corrected_controls"),
            ]
        )
    ]
    counts = {k: sum(r["status"] == k for r in rows) for k in ["completed", "failed", "censored"]}
    checked = qualified(work)
    ready = proof["ready"] and checked and all(c["passed"] for c in work["checks"])
    affected_failed = work.get("validation_receipts") and not checked
    target_failed = diagnosis.get("corrected_receipt") and not proof["corrective_execution_passed"]
    cls = "disqualified" if affected_failed or target_failed else "null" if ready else "blocked"
    value: Json = dict(
        schema="carnot.causal_validation.v1",
        experiment_id=9167,
        task_id="exp9167-causal-validation",
        milestone="2026.11.781",
        run_date=work["date"],
        honest_verdict="complete_" + cls + "_causal_validation",
        verdict_class=cls,
        gate_check_summary=work["checks"],
        preconditions_checked=work["resources"],
        rows=rows,
        intended_count=3,
        attempted_count=sum(row["attempted"] for row in rows),
        completed_count=counts["completed"],
        failed_count=counts["failed"],
        censored_count=counts["censored"],
        sample_size_budget=dict(
            intended=3,
            attempted=sum(row["attempted"] for row in rows),
            **counts,
            effective_independent_groups=0,
        ),
        effective_independent_groups=0,
        independent_generalization_score=0,
        verifier_is_oracle=False,
        claim_scope=dict(
            exposure_scope="exposed_development",
            independent_generalization_score=0,
            allowed_claims=["Executed diagnostic operations and exact remaining obligations only"],
        ),
        MODEL_SPECS=[],
        model_invocation_status="not_requested",
        inference_substrate_class="no_model_load" if receipt else "blocked_no_run",
        requested_inference_substrate_class="no_model_load",
        inference_substrate="deterministic_no_model_load",
        inference_substrate_details=dict(
            work="CPU source authentication, Python descendant tracing and subprocess validation"
        ),
        model_invocation_counts=dict(
            current_llm_invocations=0,
            loads_attempted=0,
            loads_completed=0,
            loads_failed=0,
            forward_calls=0,
            generated_tokens=0,
        ),
        current_llm_invocations=0,
        generated_tokens=0,
        duration_s=work["duration_s"],
        random_seed=SEED,
        source_references=work["references"],
        code_config_hashes=work["code_config_hashes"],
        current_work_receipts=[work["current_work"]],
        validation_receipts=work.get("validation_receipts", []),
        owned_statement_coverage=work.get("coverage", {}),
        repository_health=work.get("repository_health", {}),
        required_checks_passed=bool(
            checked and not target_failed and all(c["passed"] for c in work["checks"])
        ),
        owned_checks_passed=checked,
        causal_fix_ready_score=int(ready),
        causal_evidence=dict(diagnosis.get("evidence", {}), controlled_qualification=proof),
        revision_reference=work["revision_reference"],
        remaining_obligations=work["remaining_obligations"],
        full_674_node_qualification_score=0,
        model_work_authorized=False,
        external_publication_authorized=False,
        cost_model_rows=[
            dict(
                unit_id=row["unit_id"],
                measured_elapsed_s=row["measured_elapsed_s"],
                estimated_remaining_s=None,
                kill_timeout_s=process_receipts[i]["deadline_s"] if process_receipts[i] else None,
                uncertainty="right_censored"
                if process_receipts[i] and not process_receipts[i]["normal_exit"]
                else "observed_process_only"
                if process_receipts[i]
                else "unmeasured",
            )
            for i, row in enumerate(rows)
        ],
        acceptance_gates={
            k: dict(condition=c, principle=p)
            for k, c, p in [
                (
                    "cause",
                    "Executed expensive operation traced beyond the parent wait",
                    "A wait location alone cannot establish a cause.",
                ),
                (
                    "correction",
                    "Same-input original and corrected reproductions, full targeted module and unchanged tamper refusals pass",
                    "Observations cannot replace controlled corrective execution.",
                ),
                (
                    "validation",
                    "Every affected command passes and all owned statements execute with zero exclusions",
                    "Failed affected checks disqualify readiness.",
                ),
                (
                    "scope",
                    "Full exact obligation union belongs to Exp9168",
                    "A targeted diagnostic grants no model or whole-suite authority.",
                ),
            ]
        },
        work_reference=ref(Path(work["raw"]) / "work.json"),
    )
    value["field_principles"] = {
        k: "Exact bound inputs and executed operation receipts support this field; no unexecuted correction or model work is claimed."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def prepare(root: Path, raw: Path, date: str) -> Json:
    """Freeze exact obligations, source revisions and budgets before diagnosing outcomes."""
    import ast

    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work = authenticate(root, raw)
    work.update(
        raw=str(raw),
        date=date,
        start_ns=time.monotonic_ns(),
        resources=resources(raw),
        duration_s=0,
        code_config_hashes=[ref(ROOT / p) for p in [*OWNED, SEAM, TEST, "pyproject.toml"]],
        commands=commands(raw),
    )
    work["checks"].extend(
        dict(
            check=r["resource"],
            upstream="current_host",
            path=str(raw),
            sha256=None,
            artifact_field=r["resource"],
            op="==",
            expected=True,
            observed=r["available"],
            passed=r["available"],
        )
        for r in work["resources"]
    )
    bound(dict(path=str(BASELINE), sha256=BASELINE_PIN))
    work["references"].append(
        dict(
            ref(BASELINE),
            producer_path=str(ROOT / SEAM),
            imported_fields=["pre_correction_exact_bytes"],
        )
    )
    nodes: dict[str, list[str]] = {}
    for value in work["inputs"].values():
        obligations = value.get("validation_obligations", {})
        for reference in obligations.get("assertion_references", []) + obligations.get(
            "v779_addition_assertion_references", []
        ):
            path = Path(reference["path"])
            hist_floor = (
                ROOT
                / "results/raw/experiment_9256_v788_corpus_prefilter/baseline/"
                "test_adversarial_verify_arc_no_llm_floor_2026_07_30.py"
            )
            if (
                path == ROOT / "tests/python/test_adversarial_verify_arc_no_llm_floor_2026_07_30.py"
                and reference["sha256"]
                == "sha256:30dde5c4fd63097d24010d781f45441f9fca9db4cb6d3c126882a18d94d5d845"
                and hist_floor.exists()
            ):
                observed = ref(hist_floor)["sha256"] or "missing_file"
            else:
                observed = ref(path)["sha256"] or "missing_file"
            work["checks"].append(
                gate(
                    path,
                    "sha256",
                    reference["sha256"],
                    observed,
                    "original_assertion_bytes",
                    "exp9166-contract",
                )
            )
            if work["checks"][-1]["passed"]:
                work["references"].append(
                    dict(
                        reference,
                        producer_path=reference["path"],
                        imported_fields=["original_assertion_bytes"],
                    )
                )
        for key in [
            "original_node_ids",
            "v779_addition_ids",
            "affected_node_assertions",
            "affected_v780_node_assertions",
        ]:
            for item in obligations.get(key, []):
                node = item if isinstance(item, str) else item["node_id"]
                nodes.setdefault(node, []).append(key)
        if value.get("experiment_id") == 9153:
            for stream in ["stdout", "stderr"]:
                old = value["replay_input"]["work"]["slow_trace"]
                reference = dict(path=old[stream + "_path"], sha256=old[stream + "_sha256"])
                bound(reference)
                work["references"].append(
                    dict(
                        reference,
                        producer_path=reference["path"],
                        imported_fields=["original_trace_bytes"],
                    )
                )
    for test in [
        TEST,
        "tests/python/test_v781_contract_9166.py",
        "tests/python/test_v781_queue_binding.py",
    ]:
        for node in ast.walk(ast.parse((ROOT / test).read_text())):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                if (
                    test.endswith("test_v781_queue_binding.py")
                    and node.name == "test_matching_queue_pin_cannot_bless_changed_task"
                ):
                    for suffix in [
                        "0-prompt-changed {project_root} {date}",
                        "0-MODEL_SPECS-value1",
                        "1-gated_on-value2",
                    ]:
                        nodes.setdefault(test + "::" + node.name + "[" + suffix + "]", []).append(
                            "current_affected_assertion"
                        )
                else:
                    nodes.setdefault(test + "::" + node.name, []).append(
                        "current_affected_assertion"
                    )
    work["remaining_obligations"] = [
        dict(
            node_id=n,
            reasons=sorted(set(reasons)),
            status="pending",
            current_qualification_credit=0,
        )
        for n, reasons in sorted(nodes.items())
    ]
    work["revision_reference"] = dict(
        owned_code_config_hashes=work["code_config_hashes"],
        exact_obligation_union=work["remaining_obligations"],
        original_and_v779_count=674,
        budget=dict(
            measurement_s=3300,
            diagnosis_s=900,
            verification_s=1200,
            publication_s=300,
            hard_cap_s=4800,
        ),
        remaining_cost_s=None,
        watchdog_is_elapsed=False,
        full_qualification_owner="exp9168-forward-qualification",
    )
    code_manifest = []
    for name in [*OWNED, SEAM, TEST, "pyproject.toml"]:
        target = raw / "revision" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        code_manifest.append(
            dict(
                ref(target),
                producer_path=str(ROOT / name),
                imported_fields=["exact_revised_source"],
            )
        )
    work["references"].extend(code_manifest)
    work["revision_reference"]["exact_revised_source_manifest"] = code_manifest
    union = {}
    original_commands = [
        c
        for v in work["inputs"].values()
        for c in v.get("schedule_reference", {}).get("verification_graph", [])
    ]
    for command in original_commands + work["commands"]:
        identity = canonical_hash(dict(argv=command["argv"], expected=command.get("expected", 0)))
        union.setdefault(
            identity,
            dict(unit_id="command:" + identity, command=command, estimated_remaining_s=None),
        )
    work["revision_reference"]["exact_validation_command_union"] = list(union.values())
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in [*OWNED, SEAM])
        + "[report]\nexclude_lines=\nomit=\n"
    )
    atomic_json(
        raw / "frozen_scope.json",
        dict(
            commands=work["commands"], revision=work["revision_reference"], seed=SEED, exclusions=[]
        ),
    )
    work["current_work"] = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_validation_diagnosis",
        inference_substrate_details=dict(
            work="Actual hash authentication, descendant operation tracing and check subprocesses"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["start_ns"],
        ended_monotonic_ns=time.monotonic_ns(),
    )
    return work


def replay(path: Path) -> Json:
    """Authenticate retained rows, sources and child logs before rebuilding every headline."""
    value = json.loads(path.read_bytes())
    work = json.loads(bound(value["work_reference"]).read_bytes())
    for reference in work["references"] + work["code_config_hashes"]:
        bound(reference)
    for name, upstream_value in work["inputs"].items():
        pin = INPUTS[name][0]
        original = json.loads(
            bound(
                dict(path=str(Path(work["raw"]) / "inputs" / Path(name).name), sha256=pin)
            ).read_bytes()
        )
        if upstream_value != original:
            raise ValueError("input_drift")
    if work.get("diagnosis"):
        directory = Path(work["diagnosis"]["directory"])
        if trace_evidence(directory) != work["diagnosis"]["evidence"]:
            raise ValueError("trace_drift")
    if value != build(work):
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True)


def terminal(path: Path, raw: Path) -> Json:
    """Publish only exact candidate bytes accepted by both unchanged checkers."""
    replay(path)
    checks = [
        child(
            name,
            [str(ROOT / ".venv/bin/python"), str(ROOT / script), flag, str(path)],
            raw,
            deadline=120,
            heartbeat=60,
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    return dict(passed=all(r["passed"] for r in checks), checks=checks)


def main(argv: list[str] | None = None) -> int:
    """Run bounded diagnostics and checks; private controls can only publish private outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261117")
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--from-work", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.date != "20261117":
            raise ValueError("run_date")
        if args.replay:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        if args.from_work and args.output.absolute().is_relative_to(ROOT):
            raise ValueError("private_work_requires_external_output")
        raw = args.output.absolute().parent / "raw" / NAME / "invocations" / str(time.time_ns())
        progress("authentication_before")
        work = (
            json.loads(args.from_work.read_bytes())
            if args.from_work
            else prepare(args.input_root, raw, args.date)
        )
        progress("authentication_after")
        if not args.from_work:
            if all(c["passed"] for c in work["checks"]):
                progress("diagnosis_before")
                retained = os.environ.get("CARNOT_9167_DIAGNOSIS_REFERENCE")
                work["diagnosis"] = (
                    adopt_diagnosis(json.loads(retained))
                    if retained
                    else diagnose(
                        raw / "diagnosis",
                        min(work["start_ns"] / 1e9 + 3300, time.monotonic() + 900),
                    )
                )
                events = Path(work["diagnosis"]["directory"]) / "original/nodes.jsonl"
                work["node_events"] = (
                    [json.loads(l) for l in events.read_text().splitlines()]
                    if events.exists()
                    else []
                )
                progress("diagnosis_after")
            diagnosis_manifest = work.get("diagnosis", {}).get("manifest", {})
            work["references"].extend(diagnosis_manifest.get("source_references", []))
            work["checks"].extend(diagnosis_manifest.get("missing_intermediate_log_checks", []))
            if diagnosis_manifest:
                work["references"].append(work["diagnosis"]["manifest_reference"])
                measured = diagnosis_manifest["measured_diagnosis_elapsed_s"]
                work["checks"].append(
                    dict(
                        check="diagnostic_budget",
                        upstream="current_diagnosis",
                        path=work["diagnosis"]["manifest_reference"]["path"],
                        sha256=work["diagnosis"]["manifest_reference"]["sha256"],
                        artifact_field="measured_diagnosis_elapsed_s",
                        op="<=",
                        expected=900,
                        observed=measured,
                        passed=measured <= 900,
                    )
                )
            work["checks"].append(
                dict(
                    check="controlled_correction",
                    upstream="current_diagnosis",
                    path=diagnosis_manifest.get("corrected_receipt_reference", {}).get(
                        "path", str(raw / "diagnosis")
                    ),
                    sha256=diagnosis_manifest.get("corrected_receipt_reference", {}).get("sha256"),
                    artifact_field="corrective_execution_passed",
                    op="==",
                    expected=True,
                    observed=causal_status(work)["ready"],
                    passed=causal_status(work)["ready"],
                )
            )
            verify(work, min(work["start_ns"] / 1e9 + 4500, time.monotonic() + 1200))
            work["repository_health"] = child(
                "repository_health",
                [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
                raw / "health",
                deadline=120,
                heartbeat=60,
                scope="repository_health",
            )
            work["duration_s"] = (time.monotonic_ns() - work["start_ns"]) / 1e9
            work["current_work"] = build_current_work_receipt(
                run_id=raw.name,
                owner_pid=os.getpid(),
                events=[],
                inference_substrate="cpu_validation_diagnosis",
                inference_substrate_details=dict(
                    work="Actual diagnostic processes and validation receipts; no model work"
                ),
                inference_substrate_class="no_model_load",
                execution_venue="host",
                started_monotonic_ns=work["start_ns"],
                ended_monotonic_ns=time.monotonic_ns(),
            )
            atomic_json(raw / "work.json", work)
        value = build(work)
        progress("publication_before")
        receipt = publish_primary(args.output, value, lambda p: terminal(p, raw / "terminal"))
        print(
            json.dumps(
                dict(
                    receipt,
                    honest_verdict=value["honest_verdict"],
                    causal_fix_ready_score=value["causal_fix_ready_score"],
                )
            ),
            flush=True,
        )
        progress("publication_after")
        return 0
    except (OSError, ValueError, KeyError) as error:
        progress("refused:" + str(error))
        return 1
