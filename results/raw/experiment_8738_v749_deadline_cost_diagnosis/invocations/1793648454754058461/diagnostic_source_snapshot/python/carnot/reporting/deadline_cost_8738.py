"""REQ-REPORT-8738: locate deadline work before authorizing a bounded repair.

The original pytest nodes execute unchanged. A profile records function boundaries
so a timeout can identify active work without guessing from empty historical logs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import faulthandler
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import mkdtemp
import time
from types import FrameType
from typing import Any
from unittest.mock import patch

from carnot.reporting import deadline_edge_8603 as edge
from carnot.reporting import learning_core_8654 as core
from carnot.reporting import learning_custody_8641 as c
from carnot.reporting.count_semantics_runner_8566 import terminal_nodes
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
ROOT = c.ROOT
NAME, TASK = "experiment_8738_v749_deadline_cost_diagnosis", "exp8738-deadline-cost-diagnosis"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/deadline_cost_8738.py", CLI]
TEST = "tests/python/test_deadline_cost_8738.py"
UPSTREAM = ROOT / "results/experiment_8724_v748_invocation_clock_core.json"
PIN = "sha256:53a6baafaa3e23f12758f9f58433c69c5df8dc14b73a525a1a620f29ecaddebb"
MODEL_SPECS: list[Json] = []
NODES = [
    "tests/python/test_deadline_edge_8603.py::test_real_deadline_edges" + suffix
    for suffix in ("[3599.0-2-None]", "[3600.0-3-3600.25]", "[3601.0-3-3601.25]")
]


def progress(phase: str) -> None:
    """Flush boundaries so long original fixture work remains visible to the parent."""
    print(f"[exp8738] phase={phase}", flush=True)


def historical() -> Json:
    """Separate authentic old passes from the failed smoke child's unknown cause."""
    upstream = c.read(dict(path=str(UPSTREAM), sha256=PIN))
    work = c.read(upstream["primitive_reference"])
    receipt = next(r for r in work["receipts"] if r["name"] == "deadline_smoke")
    sizes = {}
    for stream in ("stdout", "stderr"):
        path = Path(receipt[stream + "_path"])
        if c.ref(path)["sha256"] != receipt[stream + "_sha256"]:
            raise ValueError("historical_log_hash")
        sizes[stream + "_bytes"] = len(path.read_bytes())
    if receipt["exit_code"] != -9 or not receipt["timed_out"] or any(sizes.values()):
        raise ValueError("historical_timeout_identity")
    children = [m for m in work["children"] if m["kind"] == "A"]
    invalidations = []
    seal = c.read(work["frozen_source_bundle"])
    for row in seal["sources"]:
        for reference in (row["original"], row["frozen"]):
            path = Path(reference["path"])
            observed = c.ref(path)["sha256"] if path.is_file() else None
            if observed != reference["sha256"]:
                invalidations.append(
                    dict(path=str(path), expected=reference["sha256"], observed=observed)
                )
    sources = list({s["path"]: s for m in children for s in m["sources"]}.values())
    try:
        core.attest(children, sources)
        for manifest in children:
            r = manifest["receipt"]
            for key in ("junit_reference", "coverage_config_reference"):
                if c.ref(Path(r[key]["path"])) != r[key]:
                    raise ValueError("historical_junit_or_configuration")
            outcomes = terminal_nodes(Path(r["junit_reference"]["path"]))
            if sorted(m["node_id"] for m in outcomes) != sorted(manifest["node_ids"]) or any(
                m["outcome"] != "passed" for m in outcomes
            ):
                raise ValueError("historical_node_outcomes")
        for reference in work["input_references"]:
            if c.ref(Path(reference["path"])) != reference:
                raise ValueError("historical_input_identity")
    except (OSError, ValueError) as error:
        invalidations.append(dict(check="retained_child_authentication", observed=str(error)))
    return dict(
        observed_timeout_reference=dict(
            primary=dict(path=str(UPSTREAM), sha256=PIN),
            primitive=upstream["primitive_reference"],
            receipt=receipt,
            inference="timeout_proven_internal_cause_unobserved",
            **sizes,
        ),
        historical_verdict_class=upstream["verdict_class"],
        retained_core_manifest=[
            dict(
                node_ids=m["node_ids"],
                seal_reference=m["seal_reference"],
                sources=m["sources"],
                receipt=m["receipt"],
            )
            for m in children
        ],
        retained_core_eligible_score=int(
            not invalidations
            and sum(len(m["node_ids"]) for m in children) == 39
            and work["original_coverage"]["complete"]
        ),
        reuse_invalidations=invalidations,
    )


def stage_rows(path: Path) -> list[Json]:
    """Keep unfinished stages censored; elapsed values come only from paired real clocks."""
    rows: list[Json] = []
    active = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        event = json.loads(line)
        if event["event"] == "call":
            row = dict(event, completed=False, duration_s=None)
            rows.append(row)
            active[event["frame"]] = row
        elif event["frame"] in active:
            row = active.pop(event["frame"])
            row.update(
                completed=True, duration_s=(event["monotonic_ns"] - row["monotonic_ns"]) / 1e9
            )
    return rows


def process_tree(pid: int) -> list[Json]:
    """Identify descendants by parent identity, including children in separate sessions."""
    snapshot = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,stat=,args="], capture_output=True, text=True, check=True
    )
    entries = [line.strip().split(None, 3) for line in snapshot.stdout.splitlines()]
    selected = {pid}
    for _ in entries:
        before = len(selected)
        selected.update(int(p) for p, parent, *_ in entries if int(parent) in selected)
        if len(selected) == before:
            break
    return [
        dict(pid=int(p), ppid=int(parent), state=state, command=command)
        for p, parent, state, command in entries
        if int(p) in selected
    ]


def supervise(name: str, argv: list[str], raw: Path, budget: float) -> Json:
    """Capture active stages before terminating only this invocation's owned descendants."""
    raw.mkdir(parents=True, exist_ok=True)
    out, err = raw / (name + ".stdout"), raw / (name + ".stderr")
    start = time.monotonic_ns()
    progress("before_" + name)
    tree: list[Json] = []
    timed_out = False
    with out.open("wb") as stdout, err.open("wb") as stderr:
        process = subprocess.Popen(
            argv, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True
        )
        while process.poll() is None:
            try:
                process.wait(
                    timeout=max(0.001, min(30, budget - (time.monotonic_ns() - start) / 1e9))
                )
            except subprocess.TimeoutExpired:
                progress("heartbeat_" + name)
                if (time.monotonic_ns() - start) / 1e9 >= budget:
                    timed_out = True
                    tree = process_tree(process.pid)
                    atomic_json(
                        raw / (name + ".timeout.json"),
                        dict(process_tree=tree, stages=stage_rows(raw / "stages.jsonl")),
                    )
                    for row in reversed(tree):
                        try:
                            os.kill(row["pid"], signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    process.wait()
    receipt = dict(
        name=name,
        argv=argv,
        pid=process.pid,
        exit_code=process.returncode,
        normal_exit=process.returncode == 0 and not timed_out,
        passed=process.returncode == 0 and not timed_out,
        timed_out=timed_out,
        duration_s=(time.monotonic_ns() - start) / 1e9,
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        process_tree=tree,
        cleanup_passed=not process_tree(process.pid),
        stdout_path=str(out),
        stderr_path=str(err),
        stdout_sha256=c.ref(out)["sha256"],
        stderr_sha256=c.ref(err)["sha256"],
    )
    atomic_json(raw / (name + ".receipt.json"), receipt)
    progress("after_" + name)
    return receipt


def profile(frame: FrameType, event: str, arg: Any) -> None:
    """Observe original boundaries without replacing inspection, replay or assertions."""
    labels = {
        ("deadline_edge_8603.py", "prepare"): "fixture_setup",
        ("learning_clock_8592.py", "prepare"): "original_fixture_setup",
        ("causal_brier_stream_8582.py", "inspect"): "inspect_authentication",
        ("deadline_edge_8603.py", "deadline_case"): "deadline_case",
        ("causal_brier_runner_8582.py", "run"): "original_runner",
        ("primary_publication.py", "publish_primary"): "publication",
        ("causal_brier_runner_8582.py", "terminal"): "terminal_checks",
        ("causal_brier_evidence_8582.py", "replay"): "cold_replay",
    }
    name = frame.f_code.co_name
    label = labels.get((Path(frame.f_code.co_filename).name, name))
    if name == "atomic_json" and "fixture" in str(frame.f_locals.get("path", "")):
        label = "fixture_copy:" + Path(frame.f_locals["path"]).name
    if label and event in {"call", "return"}:
        row = dict(
            event=event,
            stage=label,
            frame=id(frame),
            monotonic_ns=time.monotonic_ns(),
            elapsed=frame.f_locals.get("elapsed"),
            pid=os.getpid(),
        )
        with Path(os.environ["CARNOT_8738_TRACE"]).open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        progress(f"{event}_{label}_elapsed={row['elapsed']}")


def pytest_configure(config: Any) -> None:
    """Enable observations only in the task's explicitly configured diagnostic child."""
    if os.environ.get("CARNOT_8738_TRACE"):
        sys.setprofile(profile)
        faulthandler.dump_traceback_later(60, repeat=True)


def pytest_unconfigure(config: Any) -> None:
    """Remove task instrumentation before pytest exits normally."""
    if os.environ.get("CARNOT_8738_TRACE"):
        sys.setprofile(None)
        faulthandler.cancel_dump_traceback_later()


def empty_work(raw: Path, run_date: str) -> Json:
    """Give even an unqualified diagnosis its own real invocation clock and zero claims."""
    return dict(
        raw=str(raw),
        run_date=run_date,
        invocation_id=raw.name,
        started_at_utc=datetime.now(UTC).isoformat(),
        started_monotonic_ns=time.monotonic_ns(),
        ended_at_utc=None,
        duration_s=0.0,
        phase_spans=[],
        references=[],
        receipts=[],
        history={},
        diagnostic_stage_rows=[],
        original_deadline_rows=[],
        children=[],
        diagnosed_cause="unmeasured",
        changed_mechanism=None,
        preconditions={},
        checks={},
        adversarial_findings=[],
        deadline_ready=False,
    )


def digest(value: Json) -> str:
    """Exclude only the checksum so independently reconstructed fields must still match."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def seal(work: Json) -> None:
    """Bind a second primitive copy so repairing only the primary's checksum is ineffective."""
    path = Path(work["raw"]) / "invocation_seal.json"
    atomic_json(path, {k: v for k, v in work.items() if k != "invocation_seal_reference"})
    work["invocation_seal_reference"] = c.ref(path)
    atomic_json(Path(work["raw"]) / "work.json", work)


def build(work: Json) -> Json:
    """Execution readiness cannot imply learning, generalization or physical acceleration."""
    checks = work["checks"]
    owned = bool(checks) and all(checks.values())
    ready = owned and work["deadline_ready"]
    verdict = "null" if owned else "disqualified"
    rows = work["original_deadline_rows"]
    history = work["history"]
    value = dict(
        experiment_id=8738,
        task_id=TASK,
        milestone="2026.11.749",
        run_date=work["run_date"],
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_v749_deadline_cost_diagnosis",
        verdict_class=verdict,
        inference_substrate="deterministic_cpu_deadline_controls_and_upstream_authentication",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            current_llm_calls=0, model_loads=0, generations=0, numerical_training=0
        ),
        rows=rows,
        intended_count=3,
        completed_count=sum(r.get("status") == "completed" for r in rows),
        failed_count=sum(not r["passed"] and r.get("status") == "completed" for r in rows),
        censored_count=3 - sum(r.get("status") == "completed" for r in rows),
        excluded_count=0,
        independent_count=0,
        verifier_is_oracle=True,
        exposure_scope="original exposed deterministic deadline fixture; execution only",
        independent_generalization_score=0,
        acceptance_gates=dict(checks, deadline_controls=bool(ready)),
        required_checks_passed=owned,
        flagged_adversarial=bool(work["adversarial_findings"]) or not owned,
        validation_receipts=work["receipts"],
        adversarial_findings=work["adversarial_findings"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        gate_check_summary=[
            dict(
                upstream=TASK,
                path=str(Path(work["raw"]) / "work.json"),
                path_hash=c.ref(Path(work["raw"]) / "work.json")["sha256"],
                failed_check=k,
                field=k,
                operator="==",
                expected=True,
                observed=v,
            )
            for k, v in dict(checks, deadline_controls=bool(ready)).items()
            if not v
        ],
        preconditions_checked=work["preconditions"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7498738,
        source_artifact_hashes=[dict(path=str(UPSTREAM), sha256=PIN)],
        code_config_hashes=work["references"],
        raw_shard_hashes=[c.ref(Path(work["raw"]) / "work.json")],
        cited_upstream_artifacts=[dict(path=str(UPSTREAM), sha256=PIN)],
        primitive_reference=c.ref(Path(work["raw"]) / "work.json"),
        deadline_controls_ready_score=int(ready),
        retained_core_eligible_score=history.get("retained_core_eligible_score", 0),
        diagnostic_stage_rows=work["diagnostic_stage_rows"],
        observed_timeout_reference=history.get("observed_timeout_reference"),
        diagnosed_cause=work["diagnosed_cause"],
        changed_mechanism=work["changed_mechanism"],
        original_deadline_rows=rows,
        retained_core_manifest=history.get("retained_core_manifest", []),
        reuse_invalidations=history.get("reuse_invalidations", []),
        child_cleanup_rows=[
            dict(name=r["name"], passed=r.get("cleanup_passed", r["normal_exit"]))
            for r in work["receipts"]
        ],
        preserved_historical_limits=[
            "Exp8724 remains disqualified; no new learning, CUDA repair, ARC outcome or physical acceleration.",
            "Exp8739 owns the full learning union and every affected rerun.",
        ],
    )
    value["field_principles"] = {
        k: "Bind current claims to measured original evidence; execution grants no scientific benefit."
        for k in value
    }
    value["reproducibility_checksum"] = digest(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct final fields and authenticate every retained original operand in a fresh process."""
    try:
        value = json.loads(path.read_bytes())
        work = c.read(value["primitive_reference"])
        if c.read(work["invocation_seal_reference"]) != {
            k: v for k, v in work.items() if k != "invocation_seal_reference"
        }:
            return False
        for reference in work["references"]:
            if c.ref(Path(reference["path"])) != reference:
                return False
        for manifest in work["children"]:
            c.attest_children([manifest], manifest["sources"])
        if work["history"] and historical() != work["history"]:
            return False
        return bool(build(work) == value)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def diagnostic(work: Json) -> None:
    """Run the original nodes once under the frozen cap; an unexplained timeout is terminal."""
    raw = Path(work["raw"])
    namespace = raw / "diagnostic"
    namespace.mkdir(parents=True)
    sources = [
        c.ref(ROOT / p)
        for p in [
            *OWNED,
            "python/carnot/reporting/causal_brier_runner_8582.py",
            "python/carnot/reporting/deadline_edge_8603.py",
            "tests/python/test_deadline_edge_8603.py",
        ]
    ]
    config = namespace / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess,_exit\ncontext=diagnostic\ndata_file="
        + str(namespace / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + r["path"] + "\n" for r in sources)
    )
    environment = dict(
        CARNOT_8738_TRACE=str(namespace / "stages.jsonl"),
        CARNOT_8603_CASE_ROOT=str(namespace / "deadline_cases"),
        CARNOT_EXPERIMENT_ARTIFACT_ROOT=mkdtemp(prefix="8738-diagnostic-", dir="/var/tmp"),
        PYTHONUNBUFFERED="1",
        JAX_PLATFORMS="cpu",
        TMPDIR="/var/tmp",
    )
    junit = namespace / "nodes.xml"
    command = [
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
        "carnot.reporting.deadline_cost_8738",
        "--junitxml=" + str(junit),
        *NODES,
    ]
    atomic_json(
        raw / "frozen_method.json",
        dict(
            nodes=NODES,
            assertions="unchanged originals",
            diagnostic_cap_s=420,
            measurement_cap_s=2700,
            environment=environment,
            command=command,
            source_references=work["references"],
            model_operations_authorized=False,
        ),
    )
    with patch.dict(os.environ, environment):
        receipt = supervise("original_deadline_diagnostic", command, namespace, 420)
    work["receipts"].append(receipt)
    work["diagnostic_stage_rows"] = stage_rows(namespace / "stages.jsonl")
    for elapsed, node in zip((3599, 3600, 3601), NODES, strict=True):
        path = namespace / "deadline_cases" / f"case-{elapsed}" / "case.json"
        row = (
            json.loads(path.read_bytes())
            if path.exists()
            else dict(
                unit=f"deadline_{elapsed}",
                predicate_elapsed_s=elapsed,
                passed=False,
                status="censored",
                missing_reason="single_bounded_diagnostic_did_not_finish_this_case",
            )
        )
        row.update(
            node_id=node,
            arm="original_assertions",
            denominator=1,
            status=row.get("status", "completed"),
            cost_s=receipt["duration_s"] if elapsed == 3599 else None,
        )
        work["original_deadline_rows"].append(row)
    if receipt["passed"]:
        work["children"].append(
            c.seal_child(namespace, sources, receipt, NODES, "original_deadline")
        )
    work["diagnosed_cause"] = (
        "current_diagnostic_completed_historical_internal_cause_unobserved"
        if receipt["passed"]
        else "bounded_diagnostic_timeout_no_causal_repair_validated"
    )
    work["deadline_ready"] = receipt["passed"] and all(
        r["passed"] for r in work["original_deadline_rows"]
    )
    work["references"] += [
        c.ref(p) for p in namespace.rglob("*") if p.is_file() and p.suffix not in {".pyc"}
    ]
    progress("diagnostic_terminal_no_automatic_retry")


def run(output: Path, run_date: str) -> int:
    """Measure once, retain owned validation, and expose only terminal validated bytes."""
    from coverage import Coverage
    from carnot.reporting.primary_publication import publish_primary
    from scripts.adversarial_verify import verify_artifact

    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    meter = Coverage(
        data_file=str(raw / ".coverage.driver"), include=[str(ROOT / p) for p in OWNED]
    )
    meter.start()
    work = empty_work(raw, run_date)
    progress("authenticate_original_timeout_before_execution")
    work["history"] = historical()
    work["references"] = core.source_closure(
        [ROOT / p for p in [*OWNED, TEST, "tests/python/test_deadline_edge_8603.py"]]
    )
    from carnot.reporting.historical_learning_runner_8541 import resources

    work["preconditions"] = dict(
        resources(raw / "scratch"),
        exclusion_manifest=c.ref(ROOT / "ops/exclusion_manifest.yaml"),
        diagnostic_cap_s=420,
        measurement_cap_s=2700,
        current_llm_calls=0,
    )
    diagnostic(work)
    progress("before_owned_validation")
    validate(work, meter)
    work["duration_s"] = (time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9
    work["ended_at_utc"] = datetime.now(UTC).isoformat()
    work["phase_spans"] = [dict(phase="diagnosis_and_validation", duration_s=work["duration_s"])]
    seal(work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, build(work))
    work["adversarial_findings"] = verify_artifact(candidate)["flags"]
    seal(work)
    value = build(work)
    publication = publish_primary(output, value, lambda p: terminal(p, raw))
    atomic_json(Path(value["terminal_validation_sidecar_path"]), publication)
    progress("normal_exit_" + value["verdict_class"])
    return 0


def terminal(path: Path, raw: Path) -> Json:
    """Reuse the unchanged scanner, strict row consumer and bounded cold replay."""
    from carnot.reporting import learning_core_runner_8696 as runtime
    from carnot.reporting import learning_core_8696 as original

    with patch.object(original, "CLI", CLI):
        return runtime.terminal(path, raw)


def validate(work: Json, driver: Any) -> None:
    """Retain real owned coverage and private consumer checks without rerunning the core."""
    from coverage import Coverage

    raw = Path(work["raw"])
    work["sources"] = work["references"]
    core.instrument(work, "owned_tests", [TEST], [c.ref(ROOT / p) for p in OWNED], 180, "owned")
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
        (
            "private_e2e018",
            [
                "-m",
                "pytest",
                *core.OPTIONS,
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            ],
        ),
        (
            "private_e2e021",
            ["-m", "pytest", *core.OPTIONS, "tests/python/test_restricted_decision_audit_8210.py"],
        ),
    ]
    private_artifacts = mkdtemp(prefix="8738-validation-", dir="/var/tmp")
    for index, (name, command) in enumerate(commands):
        progress(f"validation_{index + 1}/{len(commands)}_{name}")
        with patch.dict(os.environ, CARNOT_EXPERIMENT_ARTIFACT_ROOT=private_artifacts):
            work["receipts"].append(
                c.supervise(name, [sys.executable, *command], raw / "checks", deadline=180)
            )
    work["receipts"].append(
        supervise(
            "full_python_suite",
            [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
            raw / "full_suite",
            180,
        )
    )
    work["checks"] = {
        r["name"]: r["passed"] and r["normal_exit"]
        for r in work["receipts"]
        if r["name"] != "original_deadline_diagnostic"
    }
    driver.stop()
    driver.save()
    meter = Coverage(data_file=str(raw / ".coverage.combined"), config_file=False)
    paths = [str(raw / ".coverage.driver")] + [
        row["reference"]["path"] for m in work["children"] for row in m["files"]
    ]
    meter.combine(data_paths=paths, strict=True, keep=True)
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
        not r["missing_lines"] and not r["excluded"] for r in reports
    )
    work["references"] += [
        c.ref(raw / "new_statement_coverage.json"),
        c.ref(raw / ".coverage.combined"),
    ]
    progress("after_owned_validation")


def main(argv: list[str] | None = None) -> int:
    """Reject unsafe output locations before the task can create any artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261102")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
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
    return run(args.private_output or ROOT / "results" / (NAME + ".json"), args.date)
