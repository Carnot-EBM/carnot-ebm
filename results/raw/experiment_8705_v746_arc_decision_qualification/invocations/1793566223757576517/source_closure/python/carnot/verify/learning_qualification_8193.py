"""REQ-VERIFY-8193: qualify the unchanged convex method with measured children.

Coverage survives a hard exit only when the active child saves its own data.
Private known targets establish execution readiness, never natural benefit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from types import FrameType
from typing import Any
from unittest.mock import patch

import coverage

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import calibrated_memory_methods_8180 as legacy

Json = dict[str, Any]
ROOT = legacy.ROOT
NAME = "experiment_8193_v708_learning_qualification"
TASK = "exp8193-learning-qualification"
MODULE = "python/carnot/verify/learning_qualification_8193.py"
RUNNER = "python/carnot/reporting/learning_qualification_execution_8193.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_learning_qualification_8193.py"
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []
BASE_AUTH = legacy.authenticate
BASE_MEASURE = legacy.measure
BASE_BUILD = legacy.build
BASE_SEED = legacy.seed_child


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual work counts so a supervisor can distinguish work from silence."""
    print(f"[exp8193] phase={phase} completed={completed} pending={pending}", flush=True)


def seed_child(inputs: Path, output: Path, resume: Path | None, crash: int) -> None:
    """Save after the final traced statement, then perform the actual hard exit.

    The historical child saves before stdout.flush and os._exit. This boundary
    saves those last executed statements too; the real process still exits73.
    """
    original_exit = os._exit

    def flushed_exit(frame: FrameType, event: str, argument: Any) -> None:
        if event == "c_call" and argument is original_exit:
            active = coverage.Coverage.current()
            if active:
                active.save()

    previous_profile = sys.getprofile()
    sys.setprofile(flushed_exit)
    try:
        BASE_SEED(inputs, output, resume, crash)
    finally:
        sys.setprofile(previous_profile)


def restart_specs(raw: Path, private: Path | None = None) -> list[Json]:
    """Start coverage in the real child before importing the unchanged trainer."""
    directory = raw / "child_coverage"
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(directory / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in [legacy.MODULE, MODULE, RUNNER, CLI])
    )
    cli = [
        "/usr/bin/env",
        "-u",
        "PYTHONPATH",
        "-u",
        "COVERAGE_PROCESS_START",
        str(ROOT / ".venv/bin/python"),
        "-u",
        "-m",
        "coverage",
        "run",
        "--rcfile=" + str(config),
        "--data-file=" + str(directory / ".coverage"),
        str(ROOT / CLI),
        "--seed-input",
        str((private or raw) / "restart-input.json"),
        "--seed-output",
        str(raw / "restart"),
    ]
    return [
        dict(
            name="hard_restart_crash",
            argv=cli + ["--crash-slot", "90"],
            expected_exit=73,
            deadline_s=120,
            classification="required",
        ),
        dict(
            name="hard_restart_resume",
            argv=cli + ["--resume-state", str(raw / "restart/crash.json")],
            expected_exit=0,
            deadline_s=120,
            classification="required",
        ),
    ]


def child_coverage(raw: Path) -> Json:
    """Combine saved child data and retain the statements that prove active.save()."""
    directory = raw / "child_coverage"
    shards = sorted(directory.glob(".coverage.*"))
    outer = os.environ.get("COVERAGE_RCFILE")
    if outer:
        for shard in shards:
            shutil.copyfile(shard, Path(outer).parent / (".coverage.child-" + shard.name[10:]))
    cov = coverage.Coverage(
        data_file=str(directory / ".coverage"), config_file=str(directory / "coverage.ini")
    )
    cov.combine(keep=True)
    cov.save()
    report = directory / "coverage.json"
    cov.json_report(outfile=str(report))
    lines = sorted(cov.get_data().lines(str(ROOT / legacy.MODULE)) or [])
    return dict(
        passed={291, 292} <= set(lines),
        executed_lines=lines,
        required_lines=[291, 292],
        data_path=str(directory / ".coverage"),
        data_sha256=sha256_file(directory / ".coverage"),
        report_path=str(report),
        report_sha256=sha256_file(report),
        child_shards=len(shards),
    )


def authenticate(root: Path, binder: Any, fixture_mode: bool, extra: Json) -> Json:
    """Historical failure is evidence; independently qualified stream bytes gate reuse."""
    upstream = BASE_AUTH(root, binder, fixture_mode)
    old_path = ROOT / "results" / (legacy.NAME + ".json")
    old = binder.read(old_path)
    receipt = next(r for r in old["validation_receipts"] if r["name"] == "coverage_report")
    binder.bind(Path(receipt["log_path"]), receipt["log_sha256"])
    extra["prior_failure"] = dict(
        receipt,
        verdict_class=old["verdict_class"],
        honest_verdict=old["honest_verdict"],
        covered_statements=384,
        total_statements=386,
    )
    for name, digest in old["code_config_hashes"].items():
        binder.bind(ROOT / name, digest)
    direct = legacy.schedule.authenticate(root, binder.raw, binder)
    path = root / "results/experiment_8165_v706_learning_qualification.json"
    qualified = binder.read(path)
    for field, expected in [
        ("learning_protocol_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        binder.require(path, field, expected, qualified.get(field))
    terminal = legacy.engine.methods.historical.terminal(path, qualified, binder)
    binder.require(path, "terminal.report.passed", True, terminal["report"].get("passed"))
    for role, total in [("stream", 256), ("retention", 64)]:
        ref = direct[role + "_feature_manifest"]
        rows = legacy.engine.methods.read_ref(binder, ref)["rows"]
        legacy.engine.historical.public_rows(rows, total)
        binder.require(
            path,
            role + "_manifest_agreement",
            ref,
            qualified["input_manifests"][role + "_feature_manifest"],
        )
        label_ref = direct["evaluator_label_manifests"][role]
        labels = legacy.engine.methods.read_ref(binder, label_ref)["rows"]
        binder.require(
            path,
            role + "_label_source_order",
            [r["source_cluster_id"] for r in rows],
            [r["source_cluster_id"] for r in labels],
        )
    extra["independent_stream_custody"] = direct["direct_stream_authentication"]
    extra["stream_custody_ready"] = 1
    extra["cited_qualification"] = dict(
        experiment_id=8165,
        fields_imported=["learning_protocol_ready_score", "input_manifests"],
        sha256=sha256_file(path),
    )
    return upstream


def measure(root: Path, raw: Path, *, fixture: bool = False, **kwargs: Any) -> Json:
    """Reuse qualified fixtures while recording new stream and coverage evidence."""
    progress("preconditions_start")
    extra: Json = dict(stream_custody_ready=0, prior_failure={}, measured_child_coverage={})
    raw.mkdir(parents=True, exist_ok=True)
    probe = raw / ".writable"
    probe.write_text("private writable custody\n")
    probe.unlink()
    with (
        patch.object(legacy, "authenticate", lambda r, b, f: authenticate(r, b, f, extra)),
        patch.object(legacy, "restart_specs", restart_specs),
    ):
        work = BASE_MEASURE(root, raw, fixture=fixture, **kwargs)
    if work["input_ready"]:
        extra["measured_child_coverage"] = child_coverage(raw)
    work.update(extra)
    work["preconditions_checked"].update(
        private_writable_storage=True, source_custody_independent=True, natural_fitting=False
    )
    work["code_config_hashes"].update(
        {p: sha256_file(ROOT / p) for p in [MODULE, RUNNER, CLI, TEST]}
    )
    atomic_json(raw / "qualification_evidence.json", extra)
    for path in [
        raw / "qualification_evidence.json",
        raw / "child_coverage/coverage.json",
        raw / "child_coverage/.coverage",
        raw / "child_coverage/coverage.ini",
    ]:
        if path.is_file():
            work["raw_shard_hashes"].append(dict(path=str(path), sha256=sha256_file(path)))
    work["fixture_rows"] = work["fixture_summaries"]
    work["input_manifest"] = work["input_manifests"]
    work["restart_parity"] = work["restart_fixture"]
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(work["fixture_states"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Only normally passing owned checks and measured child data grant readiness."""
    evidence_ok = not work["input_ready"] or bool(
        work["stream_custody_ready"] and work["measured_child_coverage"].get("passed")
    )
    old = BASE_BUILD(
        work, raw, receipts if evidence_ok else receipts + [dict(passed=False)], fixture=fixture
    )
    legacy_path = raw / "legacy_replay.json"
    atomic_json(legacy_path, old)
    value = dict(old)
    value["cited_upstream_artifacts"] = list(old["cited_upstream_artifacts"])
    value["raw_shard_hashes"] = list(old["raw_shard_hashes"])
    value["field_principles"] = dict(old["field_principles"])
    value.pop("reproducibility_checksum")
    value["cited_upstream_artifacts"].extend(
        [work["cited_qualification"]] if "cited_qualification" in work else []
    )
    value.update(
        experiment_id=8193,
        task_id=TASK,
        milestone="2026.10.708",
        legacy_replay_path=str(legacy_path),
        legacy_replay_sha256=sha256_file(legacy_path),
        claim_scope="Unchanged calibrated-memory mechanics qualified by real covered hard-exit and cold replay; natural calibrated learning remains untested.",
        exposure_scope="private_circular_fixtures_and_exposed_historical_custody",
        coverage_statement_counts={},
        acceptance_gates=dict(
            old["acceptance_gates"],
            measured_child="active.save statements291/292; actual exit73 then exact resumed state",
            owned_coverage="100 percent added statements; legacy hard-exit statements separately measured",
        ),
    )
    coverage_receipt = next((r for r in receipts if r.get("name") == "coverage_json"), None)
    if coverage_receipt and coverage_receipt["passed"]:
        report_path = Path(coverage_receipt["argv"][-2])
        report = json.loads(report_path.read_text())
        sealed = raw / "owned_coverage.json"
        atomic_json(sealed, report)
        value["coverage_statement_counts"] = {p: r["summary"] for p, r in report["files"].items()}
        value["raw_shard_hashes"].append(dict(path=str(sealed), sha256=sha256_file(sealed)))
    value["field_principles"].update(
        {
            k: "Measured new qualification cannot erase historical failure or create independent benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reexecute old mechanics and independently compare the new custody evidence."""
    progress("cold_replay_start")
    try:
        value = json.loads(path.read_text())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value) or value["experiment_id"] != 8193:
            return False
        saved = Path(value["legacy_replay_path"])
        if sha256_file(saved) != value["legacy_replay_sha256"]:
            return False
        old = json.loads(saved.read_text())
        for key in [
            "rows",
            "completed_count",
            "intended_count",
            "verdict_class",
            "honest_verdict",
            "required_checks_passed",
            "calibrated_memory_ready_score",
            "stream_input_ready_score",
            "measured_child_coverage",
            "fixture_rows",
            "restart_parity",
            "input_manifest",
        ]:
            if value[key] != old[key]:
                return False
        for ref in value["raw_shard_hashes"]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if value["input_ready"]:
            child = value["measured_child_coverage"]
            cov = coverage.Coverage(data_file=child["data_path"], config_file=False)
            cov.load()
            if (
                sorted(cov.get_data().lines(str(ROOT / legacy.MODULE)) or [])
                != child["executed_lines"]
            ):
                return False
        passed = legacy.replay(saved)
        progress("cold_replay_complete", len(value["fixture_states"]), 0)
        return passed
    except (OSError, ValueError, KeyError, TypeError):
        return False
