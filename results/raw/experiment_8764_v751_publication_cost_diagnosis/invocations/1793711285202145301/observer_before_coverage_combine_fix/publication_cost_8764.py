"""REQ-REPORT-8764: localize publication work without another deadline benchmark.

A completed diagnostic null is useful evidence. It cannot authorize a deadline
rerun until a safe correction and the complete original union are qualified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import deadline_cost_8738 as old
from carnot.reporting import learning_custody_8641 as custody
from carnot.reporting import learning_core_8654 as core
from carnot.reporting import publication_probe_8764 as p
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK = "experiment_8764_v751_publication_cost_diagnosis", "exp8764-publication-cost-diagnosis"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/publication_cost_8764.py",
    "python/carnot/reporting/publication_probe_8764.py",
    CLI,
]
TEST = "tests/python/test_publication_cost_8764.py"
CHECK_TESTS = [
    TEST,
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_deadline_edge_8603.py::test_clock_cannot_replace_or_feed_kernel",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]
MODEL_SPECS: list[Json] = []
STAGES = ["precondition", "copy", "materialize", "scan", "row_lint", "cold_replay"]


def empty(raw: Path, run_date: str) -> Json:
    """Give null and rejection paths the same honest invocation ownership."""
    work = old.empty_work(raw, run_date)
    work.update(
        inputs=[],
        stage_rows=[],
        historical_evidence={},
        paired_terminal_rows=[],
        reuse_invalidation_rows=[],
        process_cleanup_rows=[],
        original_union_projected_seconds=None,
        correction_source_refs=[],
        repository_health_reference={},
        ended_monotonic_ns=work["started_monotonic_ns"],
    )
    return work


def accepted(work: Json) -> bool:
    """Owned checks and consistent clocks gate acceptance independently of diagnosis."""
    return bool(
        work["checks"]
        and all(work["checks"].values())
        and work["duration_s"] >= 0
        and work["ended_monotonic_ns"] >= work["started_monotonic_ns"]
        and abs(
            work["duration_s"] - (work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9
        )
        <= 1e-9
        and all(r["passed"] and r["normal_exit"] for r in work["receipts"])
    )


def seal(work: Json) -> None:
    """Keep an independent primitive seal so summary-only repairs cannot pass replay."""
    old.seal(work)


def build(work: Json) -> Json:
    """Release a measured null without confusing execution acceptance with readiness."""
    value = old.build(work)
    owned = accepted(work)
    verdict = "null" if owned else "disqualified"
    observed = {row["unit"]: row for row in work["stage_rows"]}
    rows = [
        observed.get(
            name,
            dict(
                unit=name,
                arm="original",
                status="censored",
                passed=False,
                duration_s=None,
                denominator=1,
                missing_reason="stage_not_executed",
            ),
        )
        for name in STAGES
    ]
    value.update(
        experiment_id=8764,
        task_id=TASK,
        milestone="2026.11.751",
        random_seed=7518764,
        honest_verdict=f"complete_{verdict}_v751_publication_cost_diagnosis",
        verdict_class=verdict,
        inference_substrate="deterministic_cpu_publication_boundary_diagnosis",
        rows=rows,
        stage_rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "completed" and not r["passed"] for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        publication_path_ready_score=0,
        required_checks_passed=owned,
        flagged_adversarial=bool(work["adversarial_findings"]) or not owned,
        acceptance_gates=dict(work["checks"], scoped_execution=owned, publication_path=False),
        source_artifact_hashes=work["inputs"],
        cited_upstream_artifacts=work["inputs"],
        historical_evidence=work["historical_evidence"],
        observed_call_graph=[
            "deadline_case -> runner.run",
            "run -> build -> publish_primary",
            "publish_primary -> terminal -> v709_execution.child(adversarial)",
            "terminal -> child(strict rows) -> child(cold CLI) -> stream.replay",
            "deadline_case -> stream.replay (independent parent check)",
            "stream.replay -> checked_step -> check_step; shipped.retention; reductions; build",
        ],
        paired_terminal_rows=work["paired_terminal_rows"],
        reuse_invalidation_rows=work["reuse_invalidation_rows"],
        process_cleanup_rows=work["process_cleanup_rows"],
        original_union_projected_seconds=work["original_union_projected_seconds"],
        correction_source_refs=work["correction_source_refs"],
        repository_health_reference=work["repository_health_reference"],
        scoped_validation_scope="new module and real CLI statements; private E2E-018/021; publication and clock consumers; global health separate",
        diagnosed_cause=work["diagnosed_cause"],
        gate_check_summary=[
            dict(
                upstream=TASK,
                path=str(Path(work["raw"]) / "work.json"),
                path_hash=p.ref(Path(work["raw"]) / "work.json")["sha256"],
                failed_check="no_demonstrated_safe_correction",
                field="publication_path_ready_score",
                operator="==",
                expected=1,
                observed=0,
            )
        ],
        preserved_historical_limits=[
            "Exp8750 remains disqualified, including late output and bad observer.",
            "Independent cold replay remains mandatory. Exp8765 alone qualifies original deadline semantics and E2E-020.",
            "No correction, original union timing or independent scientific generalization is claimed.",
        ],
    )
    value["raw_shard_hashes"] += [
        r for r in work["references"] if Path(r["path"]).is_relative_to(Path(work["raw"]))
    ]
    value["field_principles"].update(
        {
            key: "Authenticate this invocation; diagnostic timing is not numerical learning or deadline qualification."
            for key in value
        }
    )
    value["reproducibility_checksum"] = old.digest(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild exact fields only after inputs, code, logs and clocks authenticate."""
    try:
        value = json.loads(path.read_bytes())
        work = custody.read(value["primitive_reference"])
        if custody.read(work["invocation_seal_reference"]) != {
            k: v for k, v in work.items() if k != "invocation_seal_reference"
        }:
            return False
        p.authenticate(work["inputs"] + work["references"])
        for row in [*work["receipts"], *[r["receipt"] for r in work["stage_rows"]]]:
            if (
                row["ended_monotonic_ns"] < row["started_monotonic_ns"]
                or abs(
                    row["duration_s"]
                    - (row["ended_monotonic_ns"] - row["started_monotonic_ns"]) / 1e9
                )
                > 1e-9
            ):
                return False
            for name in ("stdout", "stderr"):
                if p.ref(Path(row[name + "_path"]))["sha256"] != row[name + "_sha256"]:
                    return False
        return bool(build(work) == value)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def diagnostic(work: Json, plan: Path) -> None:
    """Supervise disjoint original stages, stopping after the first overrun or failure."""
    config = json.loads(plan.read_bytes())
    raw, scratch = Path(work["raw"]), Path(config["scratch"])
    start = time.monotonic()
    for index, name in enumerate(STAGES):
        old.progress(f"8764_stage_{index + 1}/{len(STAGES)}_{name}")
        budget = min(300.0, 1800 - (time.monotonic() - start))
        argv = [sys.executable, "-u", str(ROOT / CLI), "--stage", name, "--plan", str(plan)]
        if name == "scan":
            argv = [
                sys.executable,
                "-u",
                "scripts/adversarial_verify.py",
                "--json",
                str(scratch / "candidate.json"),
            ]
        elif name == "row_lint":
            argv = [
                sys.executable,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(scratch / "candidate.json"),
            ]
        receipt = p.supervise(
            name,
            argv,
            raw / "stages" / name,
            budget,
            expected=config.get("expected_scan_exit", 0) if name == "scan" else 0,
        )
        if name == "scan" and receipt["passed"]:
            scan = json.loads(Path(receipt["stdout_path"]).read_bytes())["reports"][0]
            receipt["passed"] = scan["flags"] == config["expected_scan_findings"] and not any(
                flag["severity"] == "critical" for flag in scan["flags"]
            )
            atomic_json(raw / "stages" / name / (name + ".receipt.json"), receipt)
        events = scratch / (name + ".events.jsonl")
        evidence = p.summarize(events)
        if events.exists():
            destination = raw / (name + ".events.jsonl")
            destination.write_bytes(events.read_bytes())
        work["stage_rows"].append(
            dict(
                unit=name,
                arm="original",
                denominator=1,
                status="censored" if receipt["timed_out"] else "completed",
                passed=receipt["passed"],
                duration_s=receipt["duration_s"],
                missing_reason="budget_overrun" if receipt["timed_out"] else None,
                receipt=receipt,
                boundary_evidence=evidence,
            )
        )
        work["process_cleanup_rows"].append(
            dict(unit=name, passed=receipt["cleanup_passed"], pid=receipt["pid"])
        )
        if not receipt["passed"]:
            break
    work["diagnosed_cause"] = (
        "terminal_stages_measured_no_safe_duplicate_correction_demonstrated; original_timeout_cause_unresolved"
    )


def prepare(work: Json) -> Path:
    """Freeze methods and private operands before launching any measured stage."""
    raw = Path(work["raw"])
    upstream = ROOT / "results/experiment_8750_v750_deadline_observer_control.json"
    previous = json.loads(upstream.read_bytes())
    prior = custody.read(previous["primitive_reference"])
    historical = []
    inputs = [p.ref(upstream), previous["primitive_reference"]]
    for arm in previous["observer_comparison_rows"]:
        receipt = arm["receipt"]
        refs = [p.ref(Path(receipt[n + "_path"])) for n in ("stdout", "stderr")]
        p.authenticate(
            [
                dict(path=receipt[n + "_path"], sha256=receipt[n + "_sha256"])
                for n in ("stdout", "stderr")
            ]
        )
        text = Path(receipt["stdout_path"]).read_text()
        historical.append(
            dict(
                arm=arm["arm"],
                eligible=False,
                receipt=receipt,
                cold_replay_cell_messages=text.count("phase=cold_replay"),
                retention_boundary_messages=text.count("phase=retention_boundary"),
                numerical_replay_count=None,
                source_read_count=None,
                qualification="messages are observations, not exact function-call counts",
            )
        )
        inputs.extend(refs)
        if arm["stage_reference"]:
            inputs.append(arm["stage_reference"])
        if "parent_samples" in receipt:
            inputs.append(receipt["parent_samples"])
    aborted = [
        custody.read(p.ref(path)) for path in sorted(raw.parent.glob("*/aborted_parent.json"))
    ]
    inputs += [p.ref(path) for path in sorted(raw.parent.glob("*/aborted_parent.json"))]
    inputs += [ref for row in aborted for ref in row["source_snapshots"]]
    fixture = prior["input_references"][3]
    original = custody.read(fixture)
    current_code = [p.ref(Path(r["path"])) for r in original["code_refs"]]
    work["reuse_invalidation_rows"] = [
        dict(
            upstream="original_fixture",
            path=before["path"],
            expected=before["sha256"],
            observed=after["sha256"],
            field="sha256",
            operator="==",
            passed=False,
            action="historical_execution_receipt_reuse_rejected; existing fixture copy rebinds current source",
        )
        for before, after in zip(original["code_refs"], current_code, strict=True)
        if before != after
    ]
    operands = [*original["source_refs"], *current_code, *original["primitive_files"]]
    work["inputs"] = (
        inputs
        + [fixture]
        + p.closure(operands, opaque={r["path"] for r in original["source_refs"]})
    )
    scratch = raw / "private_fixture"
    scratch.mkdir(parents=True, exist_ok=True)
    probe = scratch / "writable"
    probe.write_bytes(b"private scratch checked")
    work["preconditions"] = dict(
        writable_private_scratch=str(scratch),
        free_disk_bytes=__import__("shutil").disk_usage(scratch).free,
        exclusion_manifest=p.ref(ROOT / "ops/exclusion_manifest.yaml"),
        current_llm_calls=0,
        diagnostic_ceiling_s=1800,
        validation_reserve_s=900,
        clock="real monotonic_ns; no process-wide substitution",
    )
    work["historical_evidence"] = dict(
        verdict_class=previous["verdict_class"],
        arms=historical,
        late_output_eligible=False,
        bad_observer_eligible=False,
        original_timeout_cause="unresolved",
        owned_aborted_diagnostics=aborted,
    )
    work["references"] = core.source_closure(
        [
            ROOT / path
            for path in [
                *OWNED,
                *[name.split("::")[0] for name in CHECK_TESTS],
                "python/carnot/reporting/causal_brier_runner_8582.py",
                "python/carnot/reporting/deadline_edge_8603.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ]
    )
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess,_exit\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / name) + "\n" for name in OWNED)
        + "[report]\nexclude_lines=\n"
    )
    collection = p.supervise(
        "collection",
        [sys.executable, "-m", "pytest", *core.OPTIONS, "--collect-only", *CHECK_TESTS],
        raw / "checks",
        60,
    )
    work["receipts"].append(collection)
    nodes = [
        line
        for line in Path(collection["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    plan = raw / "frozen_method.json"
    atomic_json(
        plan,
        dict(
            inputs=work["inputs"],
            sources=work["references"],
            fixture=fixture,
            scratch=str(scratch),
            nodes=nodes,
            stages=STAGES,
            stage_cap_s=300,
            total_diagnostic_cap_s=1800,
            corrections_authorized=1,
            correction_selected=None,
            expected_scan_exit=int(bool(original["adversarial_findings"])),
            expected_scan_findings=original["adversarial_findings"],
            cache_enabled=False,
            comparison_repeats_if_corrected=3,
            independent_cold_replay_required=True,
            coverage_config=p.ref(config),
            interpreter=p.ref(Path(sys.executable).resolve()),
            environment=dict(JAX_PLATFORMS="cpu", PYTHONPATH="python:.", PYTHONUNBUFFERED="1"),
            dependency_versions={
                n: __import__("importlib.metadata", fromlist=["version"]).version(n)
                for n in ["pytest", "coverage", "ruff", "mypy"]
            },
        ),
    )
    work["references"].extend(
        [p.ref(config), p.ref(plan), work["preconditions"]["exclusion_manifest"]]
    )
    p.authenticate(work["inputs"] + work["references"])
    return plan


def validate(work: Json) -> None:
    """Retain real scoped checks and one separately bounded global health diagnostic."""
    from coverage import Coverage

    raw = Path(work["raw"])
    commands = [
        (
            "owned",
            [
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(raw / "coverage.ini"),
                "-m",
                "pytest",
                *core.OPTIONS,
                "--junitxml=" + str(raw / "owned.xml"),
                TEST,
            ],
        ),
        (
            "consumers",
            [
                "-m",
                "pytest",
                *core.OPTIONS,
                "--junitxml=" + str(raw / "consumers.xml"),
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_deadline_edge_8603.py::test_clock_cannot_replace_or_feed_kernel",
            ],
        ),
        (
            "e2e018",
            [
                "-m",
                "pytest",
                *core.OPTIONS,
                "--junitxml=" + str(raw / "e2e018.xml"),
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            ],
        ),
        (
            "e2e021",
            [
                "-m",
                "pytest",
                *core.OPTIONS,
                "--junitxml=" + str(raw / "e2e021.xml"),
                "tests/python/test_restricted_decision_audit_8210.py",
            ],
        ),
        ("ruff", ["-m", "ruff", "check", *OWNED, TEST]),
        ("format", ["-m", "ruff", "format", "--check", *OWNED, TEST]),
        (
            "mypy",
            [
                "-m",
                "mypy",
                "--config-file=/dev/null",
                "--strict",
                "--ignore-missing-imports",
                "--follow-imports=silent",
                *OWNED,
            ],
        ),
        ("spec", ["scripts/check_spec_coverage.py", "--files", TEST]),
    ]
    private = mkdtemp(prefix="8764-checks-", dir="/var/tmp")
    with patch.dict(os.environ, CARNOT_EXPERIMENT_ARTIFACT_ROOT=private, TMPDIR="/var/tmp"):
        for name, argv in commands:
            work["receipts"].append(p.supervise(name, [sys.executable, *argv], raw / "checks", 180))
        health = next(
            (
                row["repository_health_receipt"]
                for row in work["historical_evidence"].get("owned_aborted_diagnostics", [])
                if "repository_health_receipt" in row
            ),
            None,
        )
        if health is None:
            health = p.supervise(
                "repository_health",
                [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
                raw / "repository_health",
                120,
            )
    work["repository_health_reference"] = dict(
        receipt=health,
        scoped_prerequisite=False,
        full_suite_pass=health["passed"],
        statement="one bounded global diagnostic; no full-suite pass claimed",
    )
    work["checks"] = {
        row["name"]: row["passed"] and row["cleanup_passed"] for row in work["receipts"]
    }
    meter = Coverage(data_file=str(raw / ".coverage.combined"), config_file=False)
    meter.combine(data_paths=[str(raw)], strict=True, keep=True)
    meter.save()
    reports = []
    for name in OWNED:
        filename, statements, excluded, missing, _ = meter.analysis2(str(ROOT / name))
        reports.append(
            dict(path=filename, statements=statements, excluded=excluded, missing_lines=missing)
        )
    atomic_json(raw / "coverage.json", dict(files=reports, data=p.ref(raw / ".coverage.combined")))
    work["checks"]["new_statement_coverage"] = all(
        not r["missing_lines"] and not r["excluded"] for r in reports
    )


def terminal(path: Path, raw: Path) -> Json:
    """Keep the unchanged scanner and row rules and replay through this task's CLI."""
    with patch.object(old, "CLI", CLI):
        return dict(old.terminal(path, raw))


def run(output: Path, run_date: str) -> int:
    """Publish only terminal checked bytes, with validation outside numerical timing."""
    from scripts.adversarial_verify import verify_artifact

    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    work = empty(raw, run_date)
    old.progress("8764_preconditions_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    plan = prepare(work)
    diagnostic(work, plan)
    old.progress("8764_before_validation")
    validate(work)
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["duration_s"] = (work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9
    work["ended_at_utc"] = datetime.now(UTC).isoformat()
    work["phase_spans"] = [
        dict(
            phase=row["unit"],
            started_monotonic_ns=row["receipt"]["started_monotonic_ns"],
            ended_monotonic_ns=row["receipt"]["ended_monotonic_ns"],
            duration_s=row["duration_s"],
            scope="diagnostic_not_numerical_learning",
        )
        for row in work["stage_rows"]
    ]
    work["references"] += [
        p.ref(path)
        for path in raw.rglob("*")
        if path.is_file() and path.name not in {"work.json", "invocation_seal.json"}
    ]
    seal(work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, build(work))
    work["adversarial_findings"] = verify_artifact(candidate)["flags"]
    seal(work)
    value = build(work)
    publication = publish_primary(output, value, lambda path: terminal(path, raw))
    selected = reader_receipt(TASK, output.parent, field="publication_path_ready_score", expected=0)
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(publication=publication, selected_identity=selected),
    )
    old.progress("8764_normal_exit_" + value["verdict_class"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Reject unsafe output paths before creating files and keep child replay explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261103")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--stage", choices=["precondition", "copy", "materialize", "cold_replay"])
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
        if len(args.date) != 8 or (
            args.private_output and args.private_output.resolve().is_relative_to(ROOT)
        ):
            return 2
    except ValueError:
        return 2
    if args.cold_replay:
        return int(not replay(args.cold_replay))
    if args.stage:
        if args.plan is None:
            return 2
        return p.stage(args.stage, args.plan)
    return run(args.private_output or ROOT / "results" / (NAME + ".json"), args.date)
