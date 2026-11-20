"""REQ-REPORT-9125-V778: finish bounded receipt checks without rewriting old science.

Node identities and source bytes are frozen before measurement. Unfinished checks
remain visible and cannot be replaced by a successful fixture or a longer deadline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import forward_evidence_runner_9111 as old
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    build_current_work_receipt,
)
from carnot.reporting.natural_label_eligibility_8975 import ref
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.publication_probe_8764 import supervise
from carnot.reporting.spline_basis_methods_8986 import gate
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK = "experiment_9125_v778_receipt_qualification", "exp9125-receipt-qualification"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_receipt_qualification_9125.py"
OWNED = ["python/carnot/reporting/receipt_qualification_9125.py", CLI]
CONTRACT = "results/experiment_9124_v778_contract.json"
PREVIOUS = "results/experiment_9111_v777_forward_evidence.json"
PINS = {
    CONTRACT: "sha256:a93fa7d34b6b6f151bcf5a7bc8c3fe9fcc1813395d68f9c825fba6b40ed0a4e2",
    PREVIOUS: "sha256:d62ba29c76582a20981d8643d9c61d00b984d1c7530476bfd8702df780087db7",
}
MODEL_SPECS: list[Json] = []
MEASUREMENT_BUDGET_S = 2100
REUSED = [
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/publication_probe_8764.py",
    "python/carnot/reporting/deadline_cost_8738.py",
]


def progress(phase: str) -> None:
    """Flush progress while the shared supervisor supplies bounded child heartbeats."""
    print("[exp9125] phase=" + phase, flush=True)


def authenticate(root: Path, raw: Path) -> Json:
    """Retain exact upstream bytes before reading any dependent historical evidence."""
    work: Json = dict(
        checks=[],
        references=[],
        nodes=[],
        partition_node_manifests=[],
        events=[],
        receipts=[],
        controls=[],
        coverage={},
        real_success_path_receipt={},
    )
    for name in [CONTRACT, PREVIOUS]:
        path = root / name
        observed = ref(path)["sha256"] or "missing_file"
        work["checks"].append(gate(path, "sha256", PINS[name], observed, "immutable_bytes", name))
        if not work["checks"][-1]["passed"]:
            return work
        target = raw / "inputs" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        old.bound(dict(path=str(target), sha256=PINS[name]))
        work["references"].append(ref(target))
        value = json.loads(target.read_bytes())
        if name == CONTRACT:
            observed = value.get("contract_ready_score", "missing_field")
            work["checks"].append(
                gate(
                    path,
                    "contract_ready_score",
                    1,
                    observed,
                    "contract_ready_score",
                    "exp9124-contract",
                )
            )
            if not work["checks"][-1]["passed"]:
                return work
        else:
            work["previous"] = value
    previous = work["previous"]
    references = previous["code_config_hashes"] + previous["source_references"]
    references += [previous["owned_statement_coverage"]["data_reference"]]
    for receipt in previous["validation_receipts"]:
        references += [
            dict(path=receipt[s + "_path"], sha256=receipt[s + "_sha256"])
            for s in ["stdout", "stderr"]
        ]
    for reference in references:
        path = Path(reference["path"])
        observed = ref(path)["sha256"] or "missing_file"
        work["checks"].append(
            gate(
                path,
                "sha256",
                reference["sha256"],
                observed,
                "original_evidence_bytes",
                "exp9111-forward-evidence",
            )
        )
    if all(c["passed"] for c in work["checks"]):
        work["references"].extend(references)
        work["evidence"] = old.authenticate(root, raw / "historical")
        work["checks"].extend(work["evidence"]["checks"])
    return work


def partitions(nodes: list[str]) -> list[list[str]]:
    """Keep every node once and keep each module fixture within its own partition."""
    if len(set(nodes)) != len(nodes):
        raise ValueError("duplicate_node")
    by_file: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        by_file[node.split("::")[0]].append(node)
    return list(by_file.values())


def node_rows(nodes: list[str], events: list[Json]) -> list[Json]:
    """Require completed setup, assertion and teardown; retain the exact censored node."""
    if any(e["nodeid"] not in nodes for e in events):
        raise ValueError("foreign_node")
    rows = []
    for node in nodes:
        phases = [e for e in events if e["nodeid"] == node]
        failed = any(e["outcome"] in {"failed", "skipped"} for e in phases)
        done = all(
            any(e["when"] == p and e["outcome"] == "passed" for e in phases)
            for p in ["setup", "call", "teardown"]
        )
        status = "failed" if failed else "completed" if done else "censored"
        rows.append(
            dict(
                unit_id=node,
                qid=None,
                group_id=node,
                sibling=None,
                arm="original_assertions",
                condition="unchanged",
                seed=7789125,
                absolute_metric=int(done and not failed),
                numerator=int(done and not failed),
                denominator=1,
                status=status,
                exclusion_reason=None,
                censor_reason=None
                if status != "censored"
                else "unfinished_node"
                if phases
                else "not_started_before_bound",
                duration_s=sum(e["duration"] for e in phases),
                phases=phases,
            )
        )
    return rows


def _event(row: Json) -> None:
    """Append each phase immediately so a killed pytest still leaves node evidence."""
    path = os.environ.get("CARNOT_9125_NODE_LOG")
    if path:
        with Path(path).open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()


def pytest_runtest_logstart(nodeid: str, location: Any) -> None:
    """Record the pending node before its fixture or assertion can block."""
    _event(dict(nodeid=nodeid, when="start", outcome="started", duration=0))


def pytest_runtest_logreport(report: Any) -> None:
    """Keep actual phase outcomes and elapsed time rather than inferring from dots."""
    _event(
        dict(
            nodeid=report.nodeid, when=report.when, outcome=report.outcome, duration=report.duration
        )
    )


def foreign_argv(*args: str) -> list[str]:
    """Private CLI commands must import correctly outside the checkout."""
    return [
        "env",
        "-u",
        "PYTHONPATH",
        "-C",
        "/tmp",
        str(ROOT / ".venv/bin/python"),
        str(ROOT / CLI),
        *args,
    ]


def run_check(
    name: str, argv: list[str], raw: Path, deadline: float = 120, expected: int = 0
) -> Json:
    """Use the existing process-group supervisor and preserve full stream hashes."""
    return supervise(name, argv, raw / name, deadline, expected=expected)


def measure(work: Json, raw: Path, stop: float) -> Json:
    """Freeze the original union, then time finite partitions and the real normal CLI."""
    previous = work["previous"]
    scope = previous["replay_input"]["work"]["scope"]
    commands = scope["commands"]
    if commands != old.validation_commands(Path(scope["coverage_configuration"]["path"]).parent):
        raise ValueError("original_command_scope_drift")
    files = []
    for spec in commands[:2]:
        files.extend(a for a in spec["argv"] if a.startswith("tests/python/"))
    work["original_assertion_references"] = [ref(ROOT / p) for p in files]
    work["references"].extend(work["original_assertion_references"])
    py = str(ROOT / ".venv/bin/python")
    base = [py, "-m", "pytest", "-n0", "-o", "addopts=", "--no-cov"]
    collection = run_check(
        "collect_original",
        [
            "env",
            "-u",
            "COVERAGE_PROCESS_CONFIG",
            "-u",
            "COVERAGE_RCFILE",
            "-u",
            "COVERAGE_PROCESS_START",
            *base,
            "--collect-only",
            "-q",
            *files,
        ],
        raw,
    )
    work["receipts"].append(collection)
    work["collection_receipt"] = collection
    work["nodes"] = [
        s
        for s in Path(collection["stdout_path"]).read_text().splitlines()
        if s.startswith("tests/python/") and "::" in s
    ]
    groups = partitions(work["nodes"])
    work["partition_node_manifests"] = [
        dict(partition_id=i, node_ids=part) for i, part in enumerate(groups)
    ]
    work["required_command_union"] = deepcopy(commands + scope.get("cli_commands", []))
    work["frozen_protocol"] = dict(
        node_ids=work["nodes"],
        partitions=groups,
        assertion_references=work["original_assertion_references"],
        exclusions=[],
        phase_requirements=["setup", "call", "teardown"],
        per_partition_deadline_s=240,
        original_control_unit_ids=[
            r["unit_id"]
            for r in previous.get("semantic_control_rows", [])
            + previous.get("preserved_detection_rows", [])
        ],
        count_formulas=dict(
            completed="count completed node rows",
            failed="count failed node rows",
            censored="count unfinished or unstarted original nodes",
            independent_groups=0,
        ),
        measurement_budget_s=MEASUREMENT_BUDGET_S,
        validation_reserve_s=1200,
        required_command_union=work["required_command_union"],
        original_cli_requires_diagnostic_only=False,
    )
    atomic_json(raw / "frozen_protocol.json", work["frozen_protocol"])
    work["references"].append(ref(raw / "frozen_protocol.json"))
    measured = old.measure(work["evidence"], raw / "semantic")
    work["controls"] = measured["semantic_control_rows"] + measured["preserved_detection_rows"]
    work["original_semantic_controls"] = measured["semantic_control_rows"]
    work["adversarial_control_rows"] = measured["preserved_detection_rows"]
    work["historical_work"] = measured
    # Each file owns one fixture lifetime; small files share a child to avoid repeated import cost.
    batches: list[list[str]] = []
    for group in groups:
        if not batches or len(batches[-1]) + len(group) > 100:
            batches.append([])
        batches[-1].extend(group)
    work["execution_partition_manifests"] = [
        dict(partition_id=i, node_ids=b) for i, b in enumerate(batches)
    ]
    atomic_json(raw / "execution_partitions.json", work["execution_partition_manifests"])
    work["references"].append(ref(raw / "execution_partitions.json"))
    for index, batch in enumerate(batches):
        progress(f"partition_{index}_before")
        if time.monotonic() >= stop - 1800:
            break
        log = raw / f"nodes_{index}.jsonl"
        argv = [
            "env",
            "CARNOT_9125_NODE_LOG=" + str(log),
            *base,
            "-p",
            "carnot.reporting.receipt_qualification_9125",
            "-vv",
            "--durations=0",
            "-o",
            "faulthandler_timeout=90",
            *batch,
        ]
        receipt = run_check(f"partition_{index}", argv, raw, min(240, stop - time.monotonic()))
        work["receipts"].append(receipt)
        if log.exists():
            work["events"].extend(json.loads(line) for line in log.read_text().splitlines())
            work["references"].append(ref(log))
            work.setdefault("event_references", []).append(ref(log))
        atomic_json(raw / "measurement_checkpoint.json", work)
        progress(f"partition_{index}_after")
    output = raw / "normal_cli" / (old.NAME + ".json")
    remaining = stop - time.monotonic()
    if remaining > 30:
        argv = [
            "env",
            "-u",
            "PYTHONPATH",
            "-C",
            "/tmp",
            py,
            str(ROOT / old.CLI),
            "--output",
            str(output),
        ]
        receipt = run_check("original_nondiagnostic_cli", argv, raw, remaining)
        work["receipts"].append(receipt)
        work["real_success_path_receipt"] = dict(
            command_receipt=receipt,
            artifact_reference=ref(output),
            diagnostic_only=False,
            readiness_verified=False,
        )
        health_base = output.parent / "raw" / old.NAME / "invocations"
        for path in health_base.glob("*/health/repository_health.receipt.json"):
            work["repository_health_reference"] = ref(path)
            work["repository_health"] = dict(
                classification="repository_health_only", receipt=json.loads(path.read_bytes())
            )
            work["references"].append(ref(path))
        if output.is_file():
            value = json.loads(output.read_bytes())
            old.replay(output)
            work["real_success_path_receipt"]["readiness_verified"] = (
                value["forward_evidence_ready_score"] == 1 and receipt["passed"]
            )
            work["original_validation_receipts"] = value["validation_receipts"]
            work["original_coverage"] = value["owned_statement_coverage"]
            work["references"].append(ref(output))
    return work


def coverage_report(raw: Path, work: Json) -> Json:
    """Measure only owned statements; retain original missing lines separately."""
    tracer = Coverage(config_file=str(raw / "coverage.ini"))
    tracer.combine(strict=True, keep=True)
    original_data = work.get("original_coverage", {}).get("data_reference")
    if original_data:
        original = Coverage(config_file=False, data_file=str(old.bound(original_data)))
        original.load()
        tracer.get_data().update(original.get_data())
    tracer.save()
    files = []
    for name in OWNED:
        _, statements, excluded, missing, _ = tracer.analysis2(str(ROOT / name))
        files.append(dict(path=name, statements=statements, excluded=excluded, missing=missing))
    total = sum(len(r["statements"]) for r in files)
    missing = sum(len(r["missing"]) for r in files)
    if "evidence" in work:
        work["original_coverage"] = dict(files=old.coverage_rows(tracer, work["evidence"]))
    return dict(
        percent=100 * (total - missing) / total,
        excluded_statements=sum(len(r["excluded"]) for r in files),
        files=files,
        data_reference=ref(raw / ".coverage"),
    )


def verification_commands(raw: Path) -> list[tuple[str, list[str]]]:
    """Recompute the fixed verification roster so rehashing cannot remove a failed check."""
    py = str(ROOT / ".venv/bin/python")
    specs = [
        (
            "owned_tests",
            [
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
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST]),
        ("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST]),
        (
            "strict_mypy",
            [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED],
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", TEST]),
    ]
    return specs


def verify(work: Json, raw: Path) -> None:
    """Run each frozen owned command once and measure actual statement coverage."""
    specs = verification_commands(raw)
    work["verification_root"] = str(raw)
    work["owned_validation_manifest"] = specs
    atomic_json(raw / "owned_validation_manifest.json", specs)
    for name, argv in specs:
        work["receipts"].append(run_check(name, argv, raw / "verification", 180))
    work["coverage"] = coverage_report(raw, work)


def qualified(work: Json, rows: list[Json]) -> bool:
    """Original normal-mode readiness is necessary; fixtures cannot authorize reuse."""
    union = [n for p in work["partition_node_manifests"] for n in p["node_ids"]]
    summary = bool(
        work["nodes"]
        and union == work["nodes"]
        and all(r["status"] == "completed" for r in rows)
        and work["controls"]
        and all(r["passed"] for r in work["controls"])
        and work["receipts"]
        and all(r["passed"] for r in work["receipts"])
        and work["coverage"].get("percent") == 100
        and work["coverage"].get("files")
        and not any(r["missing"] or r["excluded"] for r in work["coverage"]["files"])
    )
    if not summary:
        return False
    try:
        raw = Path(work["verification_root"])
        commands = verification_commands(raw)
        actual = work["receipts"][-len(commands) :]
        if len(actual) != len(commands) or any(
            r["name"] != name
            or r["argv"] != argv
            or r["passed"] is not True
            or r["normal_exit"] is not True
            or r["timed_out"] is not False
            for r, (name, argv) in zip(actual, commands, strict=True)
        ):
            return False
        data = old.bound(work["coverage"]["data_reference"])
        tracer = Coverage(config_file=False, data_file=str(data))
        tracer.set_option("report:exclude_lines", [])
        tracer.load()
        files = []
        for name in OWNED:
            _, statements, excluded, missing, _ = tracer.analysis2(str(ROOT / name))
            files.append(dict(path=name, statements=statements, excluded=excluded, missing=missing))
        if files != work["coverage"]["files"]:
            return False
        proof = work["real_success_path_receipt"]
        path = old.bound(proof["artifact_reference"])
        original = json.loads(path.read_bytes())
        old.replay(path)
        saved = original["replay_input"]
        return bool(
            [r["name"] for r in work["receipts"]]
            == [
                "collect_original",
                *[
                    "partition_" + str(p["partition_id"])
                    for p in work.get("execution_partition_manifests", [])
                ],
                "original_nondiagnostic_cli",
                *[n for n, _ in commands],
            ]
            and proof["diagnostic_only"] is False
            and proof["command_receipt"]["passed"] is True
            and "--diagnostic-only" not in proof["command_receipt"]["argv"]
            and original["forward_evidence_ready_score"] == 1
            and old.validation_qualified(saved["work"], saved["validation"], saved["coverage"])
        )
    except (OSError, ValueError, KeyError):
        return False


def build(work: Json, date: str, start: int, end: int) -> Json:
    """Reduce recorded units without treating archived model activity as current work."""
    rows = node_rows(work["nodes"], work["events"])
    failed = [c for c in work["checks"] if not c["passed"]]
    ready = not failed and qualified(work, rows)
    receipt = build_current_work_receipt(
        run_id="exp9125",
        owner_pid=work.get("owner_pid", os.getpid()),
        events=[],
        inference_substrate="deterministic_no_model_load",
        inference_substrate_details=dict(
            work="Bounded CPU receipt qualification",
            historical_model_work="Opaque upstream provenance only",
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
    union = [n for p in work["partition_node_manifests"] for n in p["node_ids"]]
    missing = [
        dict(path=r["path"], statement=line, reason="not_observed_in_retained_coverage")
        for r in work.get("original_coverage", {}).get("files", [])
        for line in r["missing"]
    ]
    value = dict(
        experiment_id=9125,
        task_id=TASK,
        milestone="2026.11.778",
        run_date=date,
        schema="carnot.receipt_qualification.v1",
        receipt_ready_score=int(ready),
        honest_verdict="complete_blocked_upstream"
        if failed
        else "complete_null_receipt_qualified"
        if ready
        else "complete_disqualified_bounded_receipt_qualification",
        verdict_class="blocked" if failed else "null" if ready else "disqualified",
        required_checks_passed=bool(ready),
        gate_check_summary=work["checks"],
        rows=rows,
        original_node_manifest=work["nodes"],
        partition_node_manifests=work["partition_node_manifests"],
        union_equality=union == work["nodes"],
        node_timing_rows=rows,
        missing_statement_rows=missing,
        original_semantic_controls=work.get("original_semantic_controls", []),
        adversarial_control_rows=work.get("adversarial_control_rows", []),
        real_success_path_receipt=work["real_success_path_receipt"],
        original_quarantine_preserved=True,
        owned_statement_coverage=work["coverage"],
        original_statement_coverage=work.get("original_coverage", {}),
        intended_count=len(rows),
        attempted_count=sum(bool(r["phases"]) for r in rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        sample_size_budget=dict(
            original_test_nodes=len(rows), semantic_controls=14, structural_controls=5
        ),
        effective_independent_groups=0,
        independent_generalization_score=0,
        verifier_is_oracle=True,
        claim_scope="Administrative receipt qualification only; no human benefit or ARC level solve.",
        exposure_scope="Exposed human development panel; GAP-ORACLE-DISTINCT and GAP-DETECTOR-AUROC-4208 stay open.",
        MODEL_SPECS=[],
        requested_inference_substrate_class="no_model_load",
        inference_substrate_class="no_model_load",
        inference_substrate="deterministic_no_model_load",
        current_llm_invocations=0,
        model_invocation_counts=receipt["invocation_counts"],
        generated_tokens=0,
        execution_venue="host",
        duration_s=(end - start) / 1e9,
        random_seed=7789125,
        current_work_receipts=[receipt],
        validation_receipts=work["receipts"],
        source_references=work["references"],
        code_config_hashes=work.get("code_config_hashes", []),
        acceptance_gates=dict(
            authenticated_inputs=not failed, complete_bounded_validation=bool(ready)
        ),
        bounded_scope_retired=not ready,
        downstream_rule="Require Exp9125 receipt_ready_score=1; Exp9111 or fixture passes never substitute.",
        repository_health_reference=work.get("repository_health_reference"),
        repository_health=work.get("repository_health", {}),
        required_command_union=work.get("required_command_union", []),
        original_validation_receipts=work.get("original_validation_receipts", []),
        execution_partition_manifests=work.get("execution_partition_manifests", []),
        methodology_note="Original assertions and immutable sources; measured node phases and real finite child commands. No semantic repair or model load.",
        preconditions_checked=work.get("resource_checks", []),
        field_principles={
            **old.principles(),
            "receipt_ready_score": "Complete affected work, real normal orchestration and coverage are conjunctive.",
            "original_node_manifest": "Frozen original owned and direct-consumer identities prevent dropped assertions.",
            "partition_node_manifests": "Partitioning changes scheduling while preserving exact assertion scope.",
            "union_equality": "Exact union equality prevents silently shrinking the denominator.",
            "node_timing_rows": "Phase timing identifies cost and distinguishes failure from censoring.",
            "missing_statement_rows": "Unexecuted original paths stay explicit instead of being excluded.",
            "original_semantic_controls": "Replay original positive and adverse operational meanings unchanged.",
            "adversarial_control_rows": "Unsupported counts and copied outcomes must retain their rejecting controls.",
            "real_success_path_receipt": "A diagnostic or fixture cannot masquerade as normal validation readiness.",
            "original_quarantine_preserved": "Administrative checks never rehabilitate the disqualified judge.",
            "bounded_scope_retired": "An exhausted finite qualification cannot become an endless deadline retry.",
        },
    )
    saved = dict(work=work, date=date, start=start, end=end)
    value["replay_input"] = saved
    value["reproducibility_checksum"] = canonical_hash(saved)
    return value


def replay(path: Path) -> Json:
    """Authenticate complete logs, sources and assertions before rebuilding headlines."""
    value = json.loads(path.read_bytes())
    saved = value["replay_input"]
    if value["reproducibility_checksum"] != canonical_hash(saved):
        raise ValueError("replay_hash")
    for reference in value["source_references"] + value["code_config_hashes"]:
        old.bound(reference)
    for receipt in value["validation_receipts"]:
        for stream in ["stdout", "stderr"]:
            old.bound(dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"]))
    work = saved["work"]
    if "collection_receipt" in work:
        collected = [
            s
            for s in Path(work["collection_receipt"]["stdout_path"]).read_text().splitlines()
            if s.startswith("tests/python/") and "::" in s
        ]
        events = [
            json.loads(line)
            for r in work.get("event_references", [])
            for line in old.bound(r).read_text().splitlines()
        ]
        expected_parts = [
            dict(partition_id=i, node_ids=part) for i, part in enumerate(partitions(collected))
        ]
        if (
            work["nodes"] != collected
            or work["events"] != events
            or work["partition_node_manifests"] != expected_parts
        ):
            raise ValueError("node_evidence_drift")
    if "historical_work" in work:
        historical = work["historical_work"]
        diagnostic = json.loads(old.bound(historical["diagnostic_input_reference"]).read_bytes())
        judge = json.loads(old.bound(work["evidence"]["snapshots"]["judge"]).read_bytes())
        semantic = old.semantic_control_rows()
        adverse = old.structural_controls(diagnostic)
        if (
            diagnostic != dict(judge, structural_count_evidence="qid-panel-v1")
            or work["controls"] != semantic + adverse
            or work["original_semantic_controls"] != semantic
            or work["adversarial_control_rows"] != adverse
        ):
            raise ValueError("semantic_control_drift")
    if value != build(saved["work"], saved["date"], saved["start"], saved["end"]):
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True)


def terminal(path: Path, raw: Path) -> Json:
    """Both current checkers must inspect exactly the candidate that gets published."""
    replay(path)
    receipts = [
        run_check(name, [str(ROOT / ".venv/bin/python"), str(ROOT / script), flag, str(path)], raw)
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    return dict(passed=all(r["passed"] for r in receipts), checks=receipts)


def main(argv: list[str] | None = None) -> int:
    """Bound measurement, reserve verification time and publish one checked terminal result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261116")
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.replay:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        if args.date != "20261116":
            raise ValueError("run_date")
        start = time.monotonic_ns()
        raw = args.output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        progress("authentication_before")
        work = authenticate(args.input_root, raw)
        raw.mkdir(parents=True, exist_ok=True)
        work["owner_pid"] = os.getpid()
        progress("authentication_after")
        work["code_config_hashes"] = [ref(ROOT / p) for p in [*OWNED, TEST, *REUSED]]
        resources = dict(
            disk_free_bytes=shutil.disk_usage(raw).free,
            open_files_limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0],
            imports_available=all(
                importlib.util.find_spec(p) is not None for p in ["pytest", "coverage"]
            ),
        )
        work["resource_checks"] = resources
        work["checks"].append(
            gate(
                raw,
                "imports_available",
                True,
                resources["imports_available"],
                "required_imports",
                "current_host",
            )
        )
        work["checks"].append(
            dict(
                path=str(raw),
                sha256=None,
                upstream="current_host",
                check="disk_free_bytes",
                artifact_field="disk_free_bytes",
                op=">=",
                expected=100_000_000,
                observed=resources["disk_free_bytes"],
                passed=resources["disk_free_bytes"] >= 100_000_000,
            )
        )
        if all(c["passed"] for c in work["checks"]):
            config = raw / "coverage.ini"
            config.write_text(
                "[run]\nparallel=true\npatch=subprocess\ndata_file="
                + str(raw / ".coverage")
                + "\ninclude=\n"
                + "".join("    " + str(ROOT / p) + "\n" for p in OWNED + old.OWNED)
                + "[report]\nexclude_lines=\nomit=\n"
            )
            atomic_json(raw / "frozen_owned_commands.json", verification_commands(raw))
            work["references"].append(ref(raw / "frozen_owned_commands.json"))
            os.environ["COVERAGE_RCFILE"] = str(config)
            tracer = Coverage(config_file=str(config), data_suffix=True)
            tracer.start()
            private = raw / "private"
            private.mkdir(parents=True, exist_ok=True)
            work = measure(work, private, start / 1e9 + MEASUREMENT_BUDGET_S)
            tracer.stop()
            tracer.save()
            verify(work, raw)
            value = build(work, args.date, start, time.monotonic_ns())
            progress("publication_before")
            publish_primary(args.output, value, lambda p: terminal(p, raw / "terminal"))
            progress("publication_after")
            return 0
        value = build(work, args.date, start, time.monotonic_ns())
        publish_primary(args.output, value, lambda p: terminal(p, raw / "terminal"))
        progress("terminal_blocked_published")
        return 0
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print("[exp9125] refused=" + str(error), flush=True)
        return 1
