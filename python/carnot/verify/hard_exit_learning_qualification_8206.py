"""REQ-VERIFY-8206: qualify unchanged learning with Coverage.py's real exit patch.

Private fixtures establish that recovery and calibration execute correctly.
They provide no independent evidence of natural learning benefit.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import shutil
from typing import Any
from unittest.mock import patch

import coverage
import coverage.patch

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child
from carnot.verify import learning_qualification_8193 as old

Json = dict[str, Any]
legacy = old.legacy
ROOT = legacy.ROOT
NAME = "experiment_8206_v709_hard_exit_learning_qualification"
TASK = "exp8206-hard-exit-learning-qualification"
MODULE = "python/carnot/verify/hard_exit_learning_qualification_8206.py"
RUNNER = "python/carnot/reporting/hard_exit_learning_execution_8206.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hard_exit_learning_qualification_8206.py"
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so long CPU work remains visible to the supervisor."""
    print(f"[exp8206] phase={phase} completed={completed} pending={pending}", flush=True)


def run_check(root: Path, spec: Json, private: Path, durable: Path, **kwargs: Any) -> Json:
    """Reuse qualified group cleanup while preserving both complete output streams."""
    return child(
        spec["name"],
        spec["argv"],
        durable,
        deadline=spec["deadline_s"],
        expected=spec["expected_exit"],
        heartbeat=kwargs.get("heartbeat_s", 20),
        scope=spec.get("classification", "required"),
    )


def restart_specs(raw: Path, private: Path | None = None) -> list[Json]:
    """Freeze a private configuration before the legacy CLI imports its trainer."""
    directory = raw / "child_coverage"
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / "coverage.ini"
    config.write_text(
        "[run]\npatch = _exit\nparallel = true\ndata_file = "
        + str(directory / ".coverage")
        + "\ninclude = "
        + str(ROOT / legacy.MODULE)
        + "\n[report]\nexclude_lines =\n"
    )
    cli = [
        "/usr/bin/env",
        "-u",
        "PYTHONPATH",
        "-u",
        "COVERAGE_PROCESS_START",
        "-u",
        "COVERAGE_RCFILE",
        "-u",
        "COVERAGE_FILE",
        str(ROOT / ".venv/bin/python"),
        "-u",
        "-m",
        "coverage",
        "run",
        "--rcfile=" + str(config),
        str(ROOT / legacy.CLI),
        "--seed-input",
        str((private or raw) / "restart-input.json"),
    ]
    rows = []
    for name, directory_name, args, expected in [
        ("uninterrupted", "uninterrupted", [], 0),
        ("hard_restart_crash90", "restart90", ["--crash-slot", "90"], 73),
        (
            "hard_restart_resume90",
            "restart90",
            ["--resume-state", str(raw / "restart90/crash.json")],
            0,
        ),
        ("hard_restart_crash170", "restart", ["--crash-slot", "170"], 73),
        (
            "hard_restart_resume170",
            "restart",
            ["--resume-state", str(raw / "restart/crash.json")],
            0,
        ),
    ]:
        rows.append(
            dict(
                name=name,
                argv=cli + ["--seed-output", str(raw / directory_name)] + args,
                expected_exit=expected,
                deadline_s=120,
                classification="required",
            )
        )
    return rows


def child_coverage(raw: Path) -> Json:
    """Measure durable exit statements and compare complete restart arithmetic."""
    directory = raw / "child_coverage"
    shards = sorted(directory.glob(".coverage.*"))
    parent = os.environ.get("CARNOT_8206_COVERAGE_PARENT")
    if parent:
        for shard in shards:
            shutil.copyfile(shard, Path(parent) / (".coverage.child-" + shard.name[10:]))
    cov = coverage.Coverage(
        data_file=str(directory / ".coverage"), config_file=str(directory / "coverage.ini")
    )
    cov.combine(keep=True)
    cov.save()
    report = directory / "coverage.json"
    cov.json_report(outfile=str(report))
    lines = sorted(cov.get_data().lines(str(ROOT / legacy.MODULE)) or [])
    baseline = json.loads((raw / "uninterrupted/final.json").read_text())
    hashes = []
    for slot, name in [(90, "restart90"), (170, "restart")]:
        resumed = json.loads((raw / name / "final.json").read_text())
        saved = json.loads((raw / name / "crash.json").read_text())
        hashes.append(
            dict(
                crash_slot=slot,
                baseline_sha256=canonical_hash(legacy.stable(baseline)),
                resumed_sha256=canonical_hash(legacy.stable(resumed)),
                saved_state_sha256=sha256_file(raw / name / "crash.json"),
                saved_cursor=saved["cursor"],
                pending_count=len(saved["pending"]),
                passed=legacy.stable(baseline) == legacy.stable(resumed)
                and saved["cursor"] == slot
                and bool(saved["pending"]),
            )
        )
    return dict(
        passed=len(shards) == 5
        and {289, 290, 291, 292} <= set(lines)
        and all(r["passed"] for r in hashes),
        executed_lines=lines,
        required_lines=[289, 290, 291, 292],
        child_shards=len(shards),
        data_path=str(directory / ".coverage"),
        data_sha256=sha256_file(directory / ".coverage"),
        report_path=str(report),
        report_sha256=sha256_file(report),
        restart_state_hashes=hashes,
        shards=[dict(path=str(p), sha256=sha256_file(p)) for p in shards],
    )


def authenticate(root: Path, binder: Any, fixture: bool, extra: Json) -> Json:
    """Preserve both old failures while independently binding original stream custody."""
    binder.require(ROOT / ".venv/bin/coverage", "coverage_version", "7.14.1", coverage.__version__)
    implementation = Path(inspect.getfile(coverage.patch))
    binder.bind(implementation)
    extra["coverage_patch_implementation"] = dict(
        path=str(implementation), sha256=sha256_file(implementation)
    )
    upstream = old.authenticate(root, binder, fixture, extra)
    ref = upstream["upstream_primary"]
    original = binder.read(Path(ref["path"]), ref["sha256"])
    for role in [
        "stream_feature_manifest",
        "retention_feature_manifest",
        "evaluator_label_manifests",
    ]:
        binder.require(
            Path(ref["path"]),
            role,
            upstream["input_manifests"][role],
            original["input_manifests"][role],
        )
    path = ROOT / "results" / (old.NAME + ".json")
    historical = binder.read(
        path, "sha256:e01de4e66a5f67aa1f58beb2183f6acc03ba5dfc4c5723f596ed6f13826847ca"
    )
    receipt = next(r for r in historical["validation_receipts"] if r["name"] == "coverage_report")
    binder.bind(Path(receipt["log_path"]), receipt["log_sha256"])
    report_ref = next(
        r for r in historical["raw_shard_hashes"] if r["path"].endswith("owned_coverage.json")
    )
    report = binder.read(Path(report_ref["path"]), report_ref["sha256"])
    extra["prior_8193_failure"] = dict(
        receipt,
        verdict_class=historical["verdict_class"],
        honest_verdict=historical["honest_verdict"],
        missing_lines=report["files"][old.MODULE]["missing_lines"],
    )
    return upstream


def measure(root: Path, raw: Path, *, fixture: bool = False, **kwargs: Any) -> Json:
    """Freeze H2 before fitting, then add measured exits to qualified fixture primitives."""
    progress("preconditions_start")
    raw.mkdir(parents=True, exist_ok=True)
    frozen = raw / "numerical_protocol.json"
    frozen.write_bytes((ROOT / legacy.PROTOCOL).read_bytes())
    extra: Json = dict(
        stream_custody_ready=0, prior_failure={}, prior_8193_failure={}, measured_child_coverage={}
    )

    def counted(phase: str, completed: int = 0, pending: int = 0) -> None:
        progress(phase, completed, max(0, 5 - completed) if "subprocess" in phase else pending)

    with (
        patch.object(legacy, "authenticate", lambda r, b, f: authenticate(r, b, f, extra)),
        patch.object(legacy, "restart_specs", restart_specs),
        patch.object(legacy, "run_check", run_check),
        patch.object(legacy, "progress", counted),
    ):
        work = old.BASE_MEASURE(root, raw, fixture=fixture, **kwargs)
    if work["input_ready"]:
        extra["measured_child_coverage"] = child_coverage(raw)
    work.update(extra)
    work.update(
        fixture_rows=work["fixture_summaries"],
        input_manifest=work["input_manifests"],
        restart_parity=work["restart_fixture"],
    )
    work["preconditions_checked"].update(
        private_writable_storage=True,
        coverage_version=coverage.__version__,
        coverage_patch="_exit",
        numerical_protocol_frozen_before_outcomes=True,
    )
    work["code_config_hashes"].update(
        {p: sha256_file(ROOT / p) for p in [MODULE, RUNNER, CLI, TEST]}
    )
    atomic_json(raw / "qualification_evidence.json", extra)
    for path in [
        raw / "qualification_evidence.json",
        frozen,
        raw / "child_coverage/coverage.ini",
        raw / "child_coverage/coverage.json",
        raw / "child_coverage/.coverage",
    ]:
        if path.is_file():
            work["raw_shard_hashes"].append(dict(path=str(path), sha256=sha256_file(path)))
    work["raw_shard_hashes"].extend(extra["measured_child_coverage"].get("shards", []))
    preflight = os.environ.get("CARNOT_8206_PREFLIGHT_RECEIPT")
    if preflight:
        sealed = raw / "preflight_receipt.json"
        shutil.copyfile(preflight, sealed)
        work["raw_shard_hashes"].append(dict(path=str(sealed), sha256=sha256_file(sealed)))
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(work["fixture_states"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Only complete owned measurements qualify mechanics; custody has its own score."""
    qualified = old.build(work, raw, receipts, fixture=fixture)
    saved = raw / "qualification_replay.json"
    atomic_json(saved, qualified)
    value = dict(qualified)
    value["raw_shard_hashes"] = list(qualified["raw_shard_hashes"])
    for receipt in receipts:
        if receipt.get("name") == "coverage_json":
            for argument in receipt["argv"]:
                if argument.startswith("--rcfile="):
                    config = raw / "owned_coverage.ini"
                    shutil.copyfile(argument.split("=", 1)[1], config)
                    value["raw_shard_hashes"].append(
                        dict(path=str(config), sha256=sha256_file(config))
                    )
    value.pop("reproducibility_checksum")
    value.update(
        experiment_id=8206,
        task_id=TASK,
        milestone="2026.10.709",
        qualification_replay_path=str(saved),
        qualification_replay_sha256=sha256_file(saved),
        numerical_protocol_sha256=work["protocol_sha256"],
        child_exit_rows=work["restart_fixture"].get("receipts", []),
        restart_state_hashes=work["measured_child_coverage"].get("restart_state_hashes", []),
        stream_input_ready_score=int(bool(work["stream_custody_ready"])),
        claim_scope="Invocation-local hard-exit mechanics qualification; natural learning remains untested.",
    )
    value["acceptance_gates"] = dict(
        value["acceptance_gates"],
        measured_child="Coverage.py patch=_exit; real exit73 at slots90/170; exact complete state recovery",
        stream_custody="Original Exp8171/8172 stream, delayed labels, roles and independent retention bytes",
    )
    value["field_principles"] = dict(value["field_principles"])
    value["field_principles"].update(
        {
            k: "Bind current exit and custody evidence without crediting exposed fixtures as generalization."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reject changed exit evidence before the qualified cold trajectory replay."""
    progress("cold_replay_start")
    try:
        value = json.loads(path.read_text())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value) or value["experiment_id"] != 8206:
            return False
        for ref in value["raw_shard_hashes"]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in value["validation_receipts"] + value["child_exit_rows"]:
            if (
                "stderr_path" in receipt
                and sha256_file(Path(receipt["stderr_path"])) != receipt["stderr_sha256"]
            ):
                return False
        saved = Path(value["qualification_replay_path"])
        if sha256_file(saved) != value["qualification_replay_sha256"]:
            return False
        previous = json.loads(saved.read_text())
        for key in [
            "rows",
            "completed_count",
            "intended_count",
            "verdict_class",
            "honest_verdict",
            "required_checks_passed",
            "calibrated_memory_ready_score",
            "measured_child_coverage",
            "fixture_rows",
            "restart_parity",
            "input_manifest",
            "coverage_statement_counts",
            "prior_8193_failure",
        ]:
            if value[key] != previous[key]:
                return False
        if (
            value["numerical_protocol_sha256"] != legacy.PROTOCOL_HASH
            or value["stream_input_ready_score"] != int(bool(previous["stream_custody_ready"]))
            or value["child_exit_rows"] != previous["restart_fixture"].get("receipts", [])
            or value["restart_state_hashes"]
            != previous["measured_child_coverage"].get("restart_state_hashes", [])
        ):
            return False
        passed = old.replay(saved)
        progress("cold_replay_complete", len(value["fixture_states"]), 0)
        return passed
    except (OSError, ValueError, KeyError, TypeError):
        return False
