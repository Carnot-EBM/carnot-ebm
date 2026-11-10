"""REQ-REPORT-8963: retain evidence, qualify owned controls, then publish atomically."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.memory.online_calibration_8960 import LabelBroker
from carnot.reporting import corpus_admission_8963 as d
from carnot.reporting import calibration_methods_8937 as calibration
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import relational_methods_8959 as methods
from carnot.reporting.depth_opportunity_8945 import Rejected
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.corpus_views_8963 import detector_view
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.matched_corpus_runner_8949 import validation_plan
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured

Json = dict[str, Any]


def write_views(
    raw: Path, evidence: Json, units: list[Json], rows: list[Json], result: Json
) -> tuple[Json, Json]:
    """Save role-specific inputs while truth stays outside the encoder boundary."""
    views, brokers = {}, {}
    developer = evidence["history"].get("developer", [])
    dev_labels = {r["unit_id"]: r["error"] for r in evidence["history"].get("labels", [])}
    for role in ("fit", "guard"):
        selected = [r for r in developer if r["development_role"] == role]
        path = raw / "views" / (role + ".json")
        atomic_json(
            path,
            dict(
                unit_ids=[r["unit_id"] for r in selected],
                rows=methods.detector_view(selected),
                labels=[dev_labels[r["unit_id"]] for r in selected],
            ),
        )
        views[role] = calibration.reference(path)
    for role in ("calibration", "test", "online"):
        indexes = [
            i for i, u in enumerate(units) if u["role"] == role and rows[i]["status"] == "completed"
        ]
        path = raw / "views" / (role + ".json")
        atomic_json(
            path,
            dict(
                unit_ids=[units[i]["unit_id"] for i in indexes],
                rows=detector_view([units[i] for i in indexes], [rows[i] for i in indexes]),
            ),
        )
        views[role] = calibration.reference(path)
        labels = [
            dict(unit_id=units[i]["unit_id"], label=result["evaluation_labels"][i]["label"])
            for i in indexes
        ]
        label_path = raw / "sealed_labels" / (role + ".json")
        atomic_json(label_path, dict(rows=labels))
        label_path.chmod(0o600)
        brokers[role] = dict(
            label_reference=calibration.reference(label_path),
            reader="calibration_only"
            if role == "calibration"
            else "corpus_views_8963.release_final_test"
            if role == "test"
            else "online_calibration_8960.LabelBroker",
            released=False,
            label_count=len(labels),
        )
    streams = sorted({u["stream_id"] for u in units if u["role"] == "online"})
    brokers["online"]["streams"] = []
    for stream in streams:
        indexes = sorted(
            (i for i, u in enumerate(units) if u["stream_id"] == stream),
            key=lambda i: units[i]["stream_position"],
        )
        path = raw / "sealed_labels" / (stream + ".json")
        atomic_json(path, dict(labels=[result["evaluation_labels"][i]["label"] for i in indexes]))
        path.chmod(0o600)
        brokers["online"]["streams"].append(
            dict(
                stream_id=stream,
                unit_ids=[units[i]["unit_id"] for i in indexes],
                label_reference=calibration.reference(path),
                delay=8,
                block=4,
                pending_at_end=8,
            )
        )
    brokers["online"]["broker_class"] = LabelBroker.__module__ + ".LabelBroker"
    return views, brokers


def admit(evidence: Json, raw: Path) -> tuple[list[Json], list[Json], Json]:
    """Replay external join failures as the same blocked schedule, without inventing rows."""
    units: list[Json] = []
    rows: list[Json] = []
    if not evidence["failures"]:
        try:
            units, rows = d.join(evidence)
            result = d.reduce(units, rows, evidence["history"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            receipt = (
                error.receipt
                if isinstance(error, Rejected)
                else dict(
                    upstream=d.TASK,
                    path=str(raw / "inputs"),
                    hash=None,
                    check="corpus_custody",
                    artifact_field="first_response_construction_and_labels",
                    expected="1024 independent original responses and frozen roles",
                    observed=str(error),
                    failure_kind="rejected_producer",
                )
            )
            evidence["failures"].append(receipt)
            units, rows = [], []
            result = d.reduce([], [], {})
    else:
        result = d.reduce([], [], {})
    return units, rows, result


def freeze(root: Path, raw: Path, fixture: bool) -> Json:
    """Authenticate immutable source bytes before joining or scoring retained responses."""
    started = time.monotonic()
    source.progress("8963_before_authentication", 0, 1024)
    names = [r[0] for r in d.INPUTS] + [r[0] for r in methods.INPUTS]
    for name in [] if fixture else names:
        path = root / name
        if path.is_file():
            target = raw / "inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    evidence = d.authenticate(raw / "inputs")
    source.progress("8963_after_authentication", 0, 1024)
    units, rows, result = admit(evidence, raw)
    source.progress("8963_after_join", result["completed_count"], result["censored_count"])
    controls = d.isolation(units[:2], rows[:2], raw) if units else []
    views, brokers = write_views(raw, evidence, units, rows, result)
    sources = []
    for name in [
        *d.OWNED,
        *d.TESTS,
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/depth_opportunity_8945.py",
        "python/carnot/reporting/graph_source_8931.py",
        "python/carnot/reporting/relational_methods_8959.py",
        "python/carnot/autoresearch/relational_graph_encoder.py",
        "python/carnot/memory/online_calibration_8960.py",
        "scripts/experiment_template.py",
        "openspec/capabilities/research-reporting/spec.md",
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((d.ROOT / name).read_bytes())
        sources.append(
            dict(
                calibration.reference(target),
                source_path=str(d.ROOT / name),
                executed_source=name.endswith(".py"),
            )
        )
    ended = time.monotonic()
    return dict(
        raw=str(raw),
        fixture_only=fixture,
        evidence=evidence,
        units=units,
        responses=rows,
        result=result,
        controls=controls,
        views=views,
        broker_manifest=brokers,
        sources=sources,
        input_references=[
            calibration.reference(p) for p in sorted((raw / "inputs").rglob("*")) if p.is_file()
        ],
        started=started,
        ended=ended,
        phase_spans=[
            dict(phase="authenticate_join_independent_labels_and_views", duration_s=ended - started)
        ],
        global_test_debt={},
        prior_terminal={},
    )


def replay(path: Path) -> Json:
    """Recompute custody and reductions so rehashed aggregate tampering still fails."""
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            if calibration.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
            if (
                ref.get("executed_source")
                and calibration.reference(Path(ref["source_path"]))["sha256"] != ref["sha256"]
            ):
                raise ValueError("executed_bytes_drift")
        raw = Path(value["raw_shard_hashes"][0]["path"]).parent
        work = json.loads((raw / "work.json").read_text())
        evidence = d.authenticate(raw / "inputs")
        units, rows, result = admit(evidence, raw)
        if evidence != work["evidence"]:
            raise ValueError("authentication_drift")
        if units != work["units"] or rows != work["responses"] or result != work["result"]:
            raise ValueError("corpus_reduction_drift")
        with tempfile.TemporaryDirectory(prefix="carnot-8963-replay-") as folder:
            rebuilt_raw = Path(folder)
            write_views(rebuilt_raw, evidence, units, rows, result)
            controls = d.isolation(units[:2], rows[:2], rebuilt_raw) if units else []
            controls = [
                dict(row, mutated_sidecar=work["controls"][i]["mutated_sidecar"])
                for i, row in enumerate(controls)
            ]
            if controls != work["controls"] or any(
                path.read_bytes() != (raw / path.relative_to(rebuilt_raw)).read_bytes()
                for path in rebuilt_raw.rglob("*.json")
            ):
                raise ValueError("inference_view_or_control_drift")
        validation = json.loads((raw / "validation.json").read_text())
        passed = d.build(work, validation, value["fixture_only"]) == value
        return dict(passed=passed, replay_passed=passed)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False)


def main(argv: list[str] | None = None) -> int:
    """Qualify current code once; fixtures publish only in private scratch without models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261110":
        parser.error("this producer requires --date 20261110")
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or d.ROOT / "results" / (d.NAME + ".json")
    if fixture and output.resolve().is_relative_to(d.ROOT):
        parser.error("fixture writes require private scratch")
    prior = json.loads(output.read_text()) if output.is_file() else {}
    raw = output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    work = freeze(d.ROOT, raw, fixture)
    work["prior_terminal"] = prior
    validation: Json = dict(receipts=[], coverage={})
    validation_started = time.monotonic()
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot-8963-validation-"))
        source.progress("8963_before_owned_checks")
        validation["receipts"] = run_commands(
            d.ROOT,
            [
                replace(c, timeout_s=600) if c.name == "owned_unit_child" else c
                for c in validation_plan(private, producer=d)
            ],
            log_dir=raw / "validation",
            heartbeat_s=15,
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        source.progress("8963_after_owned_checks")
        prior_global = os.environ.get("CARNOT_8963_GLOBAL_RECEIPT")
        global_command = (str(d.ROOT / ".venv/bin/pytest"), "tests/python", "-q")
        if prior_global:
            receipt = json.loads(Path(prior_global).read_text())
            if (
                receipt["command_argv"] != list(global_command)
                or calibration.reference(Path(receipt["log_path"]))["sha256"]
                != receipt["log_sha256"]
            ):
                raise ValueError("global_receipt_custody")
            target = raw / "repository_health" / "full_python_suite.log"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(receipt["log_path"]).read_bytes())
            receipt = dict(receipt, original_log_path=receipt["log_path"], log_path=str(target))
            global_receipts = [receipt]
        else:
            global_receipts = run_commands(
                d.ROOT,
                [
                    CommandSpec(
                        "full_python_suite",
                        global_command,
                        "repository_health",
                        300,
                    )
                ],
                log_dir=raw / "repository_health",
                heartbeat_s=15,
            )
        work["global_test_debt"] = dict(
            receipts=global_receipts,
            affects_owned_readiness=False,
            repeated=False,
        )
    work["ended"] = time.monotonic()
    work["phase_spans"].append(
        dict(
            phase="owned_validation_and_global_health",
            duration_s=work["ended"] - validation_started,
        )
    )
    atomic_json(raw / "validation.json", validation)
    work["raw_references"] = [
        calibration.reference(p)
        for p in sorted(raw.rglob("*.json"))
        if "sources" not in p.relative_to(raw).parts and "inputs" not in p.relative_to(raw).parts
    ]
    atomic_json(raw / "work.json", work)
    value = d.build(work, validation, fixture)
    receipts: list[Json] = []

    def validate(candidate: Path) -> Json:
        report = replay(candidate)
        if not fixture:
            commands = [
                replace(c, argv=tuple(a.replace(calibration.CLI, d.CLI) for a in c.argv))
                for c in calibration.terminal_commands(candidate)
            ]
            receipts.extend(
                run_commands(d.ROOT, commands, log_dir=raw / "terminal_validation", heartbeat_s=15)
            )
            report.update(
                passed=report["passed"] and all(r["passed"] for r in receipts), receipts=receipts
            )
        return report

    source.progress("8963_before_primary_publication")
    try:
        published = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        validation["terminal_rejected"] = True
        validation["receipts"].extend(receipts)
        atomic_json(raw / "validation.json", validation)
        work["raw_references"] = [
            calibration.reference(Path(r["path"])) for r in work["raw_references"]
        ]
        atomic_json(raw / "work.json", work)
        value = d.build(work, validation, fixture)
        published = publish_primary(output, value, replay)
    source.progress("8963_terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                corpus_ready_score=value["corpus_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
