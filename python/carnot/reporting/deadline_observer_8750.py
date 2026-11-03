"""REQ-REPORT-8750: compare bounded observers without explaining an old timeout.

The unchanged deadline cases retain their finite numerical clocks. Only a pytest
boundary observer and timer dumps are added; authentication is never wrapped.
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import faulthandler
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp
import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import deadline_cost_8738 as old
from carnot.reporting import learning_clock_8592 as clock
from carnot.reporting import learning_core_8654 as core
from carnot.reporting import learning_custody_8641 as c
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK = "experiment_8750_v750_deadline_observer_control", "exp8750-deadline-observer-control"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/deadline_observer_8750.py", CLI]
TEST = "tests/python/test_deadline_observer_8750.py"
MODEL_SPECS: list[Json] = []
SAMPLE_S = 30.0
RETAINED_UNPROFILED: Json | None = dict(
    path=str(
        ROOT
        / "results/raw/experiment_8750_v750_deadline_observer_control/invocations/1793672403459757663/unprofiled/unprofiled.receipt.json"
    ),
    sha256="sha256:9f8b16bb0b6d35d2a5ecbe50e9326586d6d82206c24cea42f40378bccb409293",
)


def lineage() -> tuple[Json, Json]:
    """Authenticate timeout and signal separately so correlation cannot become cause."""
    history = old.historical()
    receipt = c.read(dict(old.RETAINED_DIAGNOSTIC or {}))
    streams = {name: c.ref(Path(receipt[name + "_path"])) for name in ("stdout", "stderr")}
    if any(streams[n]["sha256"] != receipt[n + "_sha256"] for n in streams):
        raise ValueError("historical_signal_stream_hash")
    if receipt["exit_code"] != -11 or receipt["timed_out"]:
        raise ValueError("historical_signal_identity")
    return history, dict(receipt=receipt, **streams, causal_explanation=False)


def snapshot(pid: int) -> Json:
    """Retain OS observations without entering or inspecting a child's Python frames."""
    resources = {}
    for name in ("status", "stat", "wchan", "io"):
        try:
            resources[name] = Path(f"/proc/{pid}/{name}").read_text()
        except OSError:
            pass
    return dict(pid=pid, resources=resources)


def supervise(name: str, argv: list[str], raw: Path, budget: float) -> Json:
    """Reuse the shipped import repair while sampling the parent's owned process tree."""
    raw.mkdir(parents=True, exist_ok=True)
    samples = raw / "parent_samples.jsonl"
    stop = threading.Event()

    def observe() -> None:
        while True:
            tree = old.process_tree(os.getpid())
            row = dict(
                monotonic_ns=time.monotonic_ns(), processes=[snapshot(r["pid"]) | r for r in tree]
            )
            with samples.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            old.progress(f"8750_parent_heartbeat_{name}_processes={len(tree)}")
            if stop.wait(SAMPLE_S):
                return

    sampler = threading.Thread(target=observe)
    sampler.start()
    try:
        receipt = old.supervise(name, argv, raw / "process", budget)
    finally:
        stop.set()
        sampler.join()
    receipt["parent_samples"] = c.ref(samples)
    for argument in argv:
        if (
            argument.startswith("--junitxml=")
            and (junit := Path(argument.split("=", 1)[1])).is_file()
        ):
            receipt["junit_reference"] = c.ref(junit)
    atomic_json(raw / (name + ".receipt.json"), receipt)
    return receipt


def event(stage: str, boundary: str, **values: Any) -> None:
    """Write only observed boundaries; unknown internal spans receive no invented time."""
    if path := os.environ.get("CARNOT_8750_STAGES"):
        with Path(path).open("a") as stream:
            stream.write(
                json.dumps(
                    dict(stage=stage, boundary=boundary, monotonic_ns=time.monotonic_ns(), **values)
                )
                + "\n"
            )
        old.progress(f"8750_{stage}_{boundary}_{values}")


def pytest_configure(config: Any) -> None:
    """Timer dumps run outside Python profiling so they cannot recurse in authentication."""
    if os.environ.get("CARNOT_8750_TIMER") == "1":
        faulthandler.dump_traceback_later(60, repeat=True)
    event("session", "start")


def pytest_unconfigure(config: Any) -> None:
    """Cancel only this observer's timer after a normal pytest exit."""
    if os.environ.get("CARNOT_8750_TIMER") == "1":
        faulthandler.cancel_dump_traceback_later()
    event("session", "end")


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef: Any, request: Any) -> Any:
    """Observe fixture boundaries without changing fixture code or authentication."""
    event("fixture_setup", "start", fixture=fixturedef.argname)
    yield
    event("fixture_setup", "end", fixture=fixturedef.argname)


def pytest_runtest_logreport(report: Any) -> None:
    """Keep actual node results and elapsed phases alongside the original JUnit."""
    event(
        "case_execution" if report.when == "call" else report.when,
        "end",
        node_id=report.nodeid,
        outcome=report.outcome,
        duration_s=report.duration,
    )


def compare(work: Json) -> None:
    """Run exactly one matched pair; unresolved work is terminal and never auto-retried."""
    raw = Path(work["raw"])
    work["observer_comparison_rows"] = []
    config = raw / "diagnostic_coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess,_exit\ncontext=diagnostic\ndata_file="
        + str(raw / ".coverage.diagnostic")
        + "\ninclude=\n"
        + "".join(
            "    " + str(ROOT / p) + "\n"
            for p in [
                *OWNED,
                *old.OWNED,
                "python/carnot/reporting/causal_brier_runner_8582.py",
                "python/carnot/reporting/deadline_edge_8603.py",
                "tests/python/test_deadline_edge_8603.py",
            ]
        )
    )
    common = [
        sys.executable,
        "-u",
        "-m",
        "coverage",
        "run",
        "--rcfile=" + str(config),
        "-m",
        "pytest",
        *core.OPTIONS,
        "-s",
        "-p",
        "carnot.reporting.deadline_observer_8750",
    ]
    plans: list[Json] = []
    retained = c.read(RETAINED_UNPROFILED) if RETAINED_UNPROFILED else None
    if retained:
        method = c.read(retained["frozen_method"])
        if method["nodes"] != old.NODES or method["diagnostic_cap_s"] != 420:
            raise ValueError("retained_arm_method")
        snapshots = Path(retained["frozen_method"]["path"]).parent
        for name, snapshot_name in [
            (OWNED[0], "observer_source_before_correction.py"),
            (TEST, "observer_tests_before_correction.py"),
        ]:
            expected = next(r for r in method["source_references"] if r["path"] == str(ROOT / name))
            snapshot = c.ref(snapshots / snapshot_name)
            if snapshot["sha256"] != expected["sha256"]:
                raise ValueError("retained_observer_source")
            work["references"].append(snapshot)
        work["references"] += [RETAINED_UNPROFILED, retained["frozen_method"]]
        work["changed_mechanism"] = (
            "parent_supervisor_log_namespace_isolated_after_KeyError_event; no unprofiled retry"
        )
    for arm in ("unprofiled", "timer_observed"):
        namespace = raw / arm
        namespace.mkdir()
        env = dict(
            CARNOT_8750_STAGES=str(namespace / "stages.jsonl"),
            CARNOT_8750_TIMER=str(int(arm == "timer_observed")),
            CARNOT_8603_CASE_ROOT=str(namespace / "deadline_cases"),
            CARNOT_8738_TRACE="",
            PYTHONUNBUFFERED="1",
            JAX_PLATFORMS="cpu",
            TMPDIR="/var/tmp",
            CARNOT_EXPERIMENT_ARTIFACT_ROOT=mkdtemp(prefix="8750-" + arm + "-", dir="/var/tmp"),
        )
        plans.append(
            dict(
                arm=arm,
                environment=env,
                argv=[*common, "--junitxml=" + str(namespace / "nodes.xml"), *old.NODES],
            )
        )
    atomic_json(
        raw / "frozen_method.json",
        dict(
            plans=plans,
            nodes=old.NODES,
            source_references=work["references"],
            input_references=work.get("input_references", []),
            preconditions=work.get("preconditions", {}),
            diagnostic_cap_s=420,
            aggregate_cap_s=1800,
            model_operations_authorized=False,
            assertions="unchanged originals",
            corrections_authorized=1,
            retained_unprofiled_reference=RETAINED_UNPROFILED,
        ),
    )
    work["references"] += [c.ref(config), c.ref(raw / "frozen_method.json")]
    for index, plan in enumerate(plans):
        arm, env = plan["arm"], plan["environment"]
        namespace = raw / arm
        old.progress(f"8750_arm_{index + 1}/2_{arm}")
        is_retained = arm == "unprofiled" and retained is not None
        if is_retained and retained is not None:
            receipt = dict(retained)
            namespace = Path(receipt["stdout_path"]).parent
        else:
            with patch.dict(os.environ, env):
                receipt = supervise(arm, plan["argv"], namespace, 420)
        work["receipts"].append(receipt)
        rows = []
        for elapsed, node in zip((3599, 3600, 3601), old.NODES, strict=True):
            path = namespace / "deadline_cases" / f"case-{elapsed}" / "case.json"
            row = (
                json.loads(path.read_bytes())
                if path.exists()
                else dict(predicate_elapsed_s=elapsed, passed=False)
            )
            if is_retained:
                row = dict(
                    predicate_elapsed_s=elapsed,
                    passed=False,
                    excluded_post_budget_row=row if path.exists() else None,
                    missing_reason="supervision_failed; post-budget output ineligible",
                )
            row.update(
                node_id=node,
                arm=arm,
                denominator=1,
                status="completed" if path.exists() and not is_retained else "censored",
                cost_s=None,
            )
            rows.append(row)
        work["original_deadline_rows"] += rows
        work["references"] += [
            c.ref(p) for p in namespace.rglob("*") if p.is_file() and p.suffix != ".pyc"
        ]
        work["observer_comparison_rows"].append(
            dict(
                arm=arm,
                receipt=receipt,
                rows=rows,
                unobserved_spans=["source_authentication", "copy", "publication", "replay"],
                stage_reference=c.ref(namespace / "stages.jsonl")
                if (namespace / "stages.jsonl").exists()
                else None,
            )
        )
    work["diagnosed_cause"] = "original_timeout_unresolved_observer_comparison_not_causal"
    work["deadline_ready"] = False
    work["references"] += [
        c.ref(p)
        for arm in plans
        for p in (raw / arm["arm"]).rglob("*")
        if p.is_file() and p.suffix != ".pyc"
    ]
    atomic_json(raw / "diagnostic_complete.json", work["observer_comparison_rows"])
    old.progress("8750_terminal_diagnostic_no_correction_no_retry")


def build(work: Json) -> Json:
    """Reuse the established schema while keeping diagnostic nulls separate from health."""
    value = old.build(work)
    rows = work["original_deadline_rows"]
    value.update(
        experiment_id=8750,
        task_id=TASK,
        milestone="2026.11.750",
        random_seed=7508750,
        honest_verdict=f"complete_{value['verdict_class']}_v750_deadline_observer_control",
        inference_substrate="aggregation_from_upstream_artifacts",
        intended_count=6,
        censored_count=6 - sum(r["status"] == "completed" for r in rows),
        observer_comparison_rows=work["observer_comparison_rows"],
        historical_signal_observation=work["historical_signal"],
        repository_health_reference=work["repository_health_reference"],
        scoped_validation_scope="new module/CLI; original clock consumers; private E2E-018/020/021; global health separate",
        source_artifact_hashes=work.get("input_references", []),
        cited_upstream_artifacts=work.get("input_references", []),
        preserved_historical_limits=[
            "Exp8724 and Exp8738 remain disqualified; no new learning, cost, ARC or physical result.",
            "Historical timeout cause unresolved; no correction or repeated timeout authorized.",
        ],
    )
    for gate in value["gate_check_summary"]:
        gate["upstream"] = TASK
    value["raw_shard_hashes"] += [
        r for r in work["references"] if Path(r["path"]).is_relative_to(Path(work["raw"]))
    ]
    value["field_principles"].update(
        {
            k: "Bind actual scoped evidence; observer correlation does not establish timeout cause."
            for k in value
        }
    )
    value["reproducibility_checksum"] = old.digest(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct sealed claims and authenticate streams rather than trusting a summary."""
    try:
        value = json.loads(path.read_bytes())
        work = c.read(value["primitive_reference"])
        if c.read(work["invocation_seal_reference"]) != {
            k: v for k, v in work.items() if k != "invocation_seal_reference"
        }:
            return False
        for reference in [*work["references"], *work.get("input_references", [])]:
            if c.ref(Path(reference["path"])) != reference:
                return False
        for receipt in work["receipts"]:
            for stream in ("stdout", "stderr"):
                if c.ref(Path(receipt[stream + "_path"]))["sha256"] != receipt[stream + "_sha256"]:
                    return False
        for manifest in work["children"]:
            c.attest_children([manifest], manifest["sources"])
        if lineage() != (work["history"], work["historical_signal"]):
            return False
        return bool(build(work) == value)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Keep unchanged scanners and strict consumers, with this task's fresh CLI replay."""
    with patch.object(old, "CLI", CLI):
        return dict(old.terminal(path, raw))


def validate(work: Json, driver: Any) -> None:
    """Measure owned coverage and the real E2Es without gating on unrelated health."""
    from coverage import Coverage

    raw = Path(work["raw"])
    work["sources"] = work["references"]
    core.instrument(work, "owned_tests", [TEST], [c.ref(ROOT / p) for p in OWNED], 300, "owned")
    commands = [
        ("ruff_check", ["-m", "ruff", "check", *OWNED, TEST]),
        ("ruff_format", ["-m", "ruff", "format", "--check", *OWNED, TEST]),
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
        ("private_e2e018", ["tests/python/test_experiment_7891_v685_authority_lifecycle.py"]),
        ("private_e2e021", ["tests/python/test_restricted_decision_audit_8210.py"]),
        ("e2e020", ["tests/python/test_hard_exit_learning_qualification_8206.py"]),
        (
            "clock_consumers",
            [
                "tests/python/test_invocation_clock_8724.py",
                "tests/python/test_learning_clock_8592.py",
                "tests/python/test_deadline_cost_8738.py",
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_deadline_edge_8603.py::test_clock_cannot_replace_or_feed_kernel",
            ],
        ),
    ]
    private = mkdtemp(prefix="8750-validation-", dir="/var/tmp")
    for name, args in commands:
        if name.startswith("private_") or name in {"e2e020", "clock_consumers"}:
            args = [
                "-m",
                "pytest",
                *core.OPTIONS,
                "--junitxml=" + str(raw / (name + ".xml")),
                *args,
            ]
        with patch.dict(os.environ, CARNOT_EXPERIMENT_ARTIFACT_ROOT=private, TMPDIR="/var/tmp"):
            work["receipts"].append(
                supervise(name, [sys.executable, *args], raw / "checks" / name, 300)
            )
    work["checks"] = {
        r["name"]: bool(r["passed"] and r["normal_exit"])
        for r in work["receipts"]
        if r["name"] not in {"unprofiled", "timer_observed"}
    }
    with patch.dict(
        os.environ,
        CARNOT_EXPERIMENT_ARTIFACT_ROOT=private,
        TMPDIR="/var/tmp",
        COVERAGE_FILE=str(raw / "repository_health" / ".coverage"),
    ):
        health = supervise(
            "repository_health",
            [
                str(ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "--junitxml=" + str(raw / "repository_health.xml"),
            ],
            raw / "repository_health",
            180,
        )
    work["repository_health_reference"]["current_receipt"] = health
    work["repository_health_reference"]["full_suite_pass"] = health["passed"]
    work["checks"]["observer_supervision"] = not any(
        r.get("supervision_failed") for r in work["receipts"]
    )
    driver.stop()
    driver.save()
    meter = Coverage(data_file=str(raw / ".coverage.combined"), config_file=False)
    meter.combine(
        data_paths=[
            str(raw / ".coverage.driver"),
            *[r["reference"]["path"] for m in work["children"] for r in m["files"]],
        ],
        strict=True,
        keep=True,
    )
    meter.save()
    reports = []
    for name in OWNED:
        filename, statements, excluded, missing, _ = meter.analysis2(str(ROOT / name))
        reports.append(
            dict(path=filename, statements=statements, excluded=excluded, missing_lines=missing)
        )
    atomic_json(
        raw / "new_statement_coverage.json",
        dict(files=reports, data=c.ref(raw / ".coverage.combined")),
    )
    work["checks"]["new_statement_coverage"] = all(
        not r["excluded"] and not r["missing_lines"] for r in reports
    )
    work["references"] += [
        c.ref(raw / "new_statement_coverage.json"),
        c.ref(raw / ".coverage.combined"),
    ]


def run(output: Path, run_date: str, *, private_control: bool = False) -> int:
    """Freeze inputs, compare once and publish checked terminal bytes with current clocks."""
    from coverage import Coverage
    from carnot.reporting.historical_learning_runner_8541 import resources
    from scripts.adversarial_verify import verify_artifact

    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    meter = Coverage(
        data_file=str(raw / ".coverage.driver"),
        config_file=False,
        include=[str(ROOT / p) for p in OWNED],
    )
    if not private_control:
        meter.start()
    work = old.empty_work(raw, run_date)
    work.update(observer_comparison_rows=[], historical_signal={}, repository_health_reference={})
    old.progress("8750_preconditions_no_model_load_MODEL_SPECS=[]_LLM_calls=0")
    work["history"], work["historical_signal"] = lineage()
    work["references"] = core.source_closure(
        [
            ROOT / p
            for p in [
                *OWNED,
                TEST,
                "tests/python/test_deadline_edge_8603.py",
                "pyproject.toml",
                "tests/python/conftest.py",
            ]
        ]
    )
    upstream = c.read(dict(path=str(old.UPSTREAM), sha256=old.PIN))
    primitive = c.read(upstream["primitive_reference"])
    fixture = c.read(primitive["original_mechanics_fixture"])
    inputs = [
        dict(path=str(old.UPSTREAM), sha256=old.PIN),
        upstream["primitive_reference"],
        c.ref(ROOT / "results/experiment_8738_v749_deadline_cost_diagnosis.json"),
        primitive["original_mechanics_fixture"],
        *[
            c.ref(Path(fixture["raw"]) / n)
            for n in (
                "stream_rows.json",
                "released_feedback.json",
                "retention_rows.json",
                "coefficient_transitions.json",
            )
        ],
    ]
    for identity, pin in clock.PINS.items():
        path = next((ROOT / "results").glob(f"experiment_{identity}_*.json"))
        reference = dict(path=str(path), sha256=pin)
        original = c.read(reference)
        inputs += [reference, original["primitive_reference"]]
    work["input_references"] = inputs
    work["preconditions"] = dict(
        resources(Path(mkdtemp(prefix="8750-scratch-", dir="/var/tmp"))),
        exclusion_manifest=c.ref(ROOT / "ops/exclusion_manifest.yaml"),
        diagnostic_cap_s=420,
        measurement_cap_s=1800,
        immutable_inputs=inputs,
    )
    health = c.read(
        json.loads(
            (ROOT / "results/experiment_8738_v749_deadline_cost_diagnosis.json").read_bytes()
        )["primitive_reference"]
    )
    work["repository_health_reference"] = dict(
        upstream=inputs[2],
        receipt=next(r for r in health["receipts"] if r["name"] == "full_python_suite"),
        full_suite_pass=False,
        scoped_prerequisite=False,
    )
    if not private_control:
        compare(work)
        validate(work, meter)
    else:
        work["checks"] = dict(private_control_unqualified=False)
    meter.stop()
    meter.save()
    work["references"] += [
        c.ref(p)
        for p in raw.rglob("*")
        if p.is_file()
        and p.suffix != ".pyc"
        and p.name not in {"work.json", "invocation_seal.json"}
    ]
    work["duration_s"] = (time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9
    work["ended_at_utc"] = datetime.now(UTC).isoformat()
    work["phase_spans"] = [
        dict(phase="observer_comparison_and_validation", duration_s=work["duration_s"])
    ]
    old.seal(work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, build(work))
    work["adversarial_findings"] = verify_artifact(candidate)["flags"]
    old.seal(work)
    value = build(work)
    publication = publish_primary(output, value, lambda p: terminal(p, raw))
    selected = reader_receipt(
        TASK, output.parent, field="deadline_controls_ready_score", expected=0
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(publication=publication, selected_identity=selected),
    )
    old.progress("8750_normal_exit_" + value["verdict_class"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Validate private destinations before creating evidence; keep replay process-fresh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261103")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
        if (
            len(args.date) != 8
            or (args.private_output and args.private_output.resolve().is_relative_to(ROOT))
            or (args.private_control and not args.private_output)
        ):
            return 2
    except ValueError:
        return 2
    if args.cold_replay:
        return int(not replay(args.cold_replay))
    return run(
        args.private_output or ROOT / "results" / (NAME + ".json"),
        args.date,
        private_control=args.private_control,
    )
