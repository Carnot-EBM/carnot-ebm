"""REQ-REPORT-8958: qualify development custody before atomic publication."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import confidence_diagnosis_8958 as d
from carnot.reporting import calibration_methods_8937 as calibration
from carnot.reporting import matched_corpus_admission_8951 as admission
from carnot.reporting import matched_corpus_runner_8949 as shared
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured

Json = dict[str, Any]


def freeze(root: Path, raw: Path) -> Json:
    """Retained producers stay untouched; new views bind their own exact bytes."""
    started = time.monotonic()
    d.b.progress("8958_before_authentication", 0, 320)
    evidence = d.authenticate(root)
    d.b.progress(
        "8958_after_authentication", len(evidence["corpus"]), 320 - len(evidence["corpus"])
    )
    result = d.empty_result() if evidence["failures"] else d.diagnose(evidence)
    atomic_json(
        raw / "developer_view.json",
        dict(rows=result["developer_rows"], scope="exposed_development_no_truth"),
    )
    label_path = raw / "protected_labels" / "labels.json"
    label_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_json(
        label_path, dict(rows=result["label_rows"], scope="protected_exposed_development_labels")
    )
    label_path.chmod(0o600)
    sources = []
    for name in [
        *d.OWNED,
        *d.TESTS,
        "python/carnot/autoresearch/graph_calibration_benchmark.py",
        "python/carnot/reporting/matched_corpus_admission_8951.py",
        "python/carnot/reporting/matched_calibrated_decisions_8952.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/current_work_receipt.py",
        "openspec/capabilities/research-reporting/spec.md",
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
        sources.append(
            dict(
                calibration.reference(target),
                source_path=str(root / name),
                executed_source=name in d.OWNED,
            )
        )
    ended = time.monotonic()
    work = dict(
        evidence=evidence,
        result=result,
        input_references=evidence["references"],
        sources=sources,
        developer_view=calibration.reference(raw / "developer_view.json"),
        protected_label_view=dict(
            calibration.reference(label_path),
            mode="0600",
            directory_mode="0700",
            scope="exposed_development_labels",
        ),
        raw_references=[],
        duration_s=ended - started,
        phase_spans=[
            dict(phase="authentication_reduction_development_fits", duration_s=ended - started)
        ],
        global_test_debt={},
    )
    atomic_json(raw / "measurement.json", work)
    work["raw_references"] = [
        calibration.reference(raw / "measurement.json"),
        work["developer_view"],
        work["protected_label_view"],
    ]
    return work


def replay(path: Path) -> Json:
    """Rebuild science from pinned inputs, refusing even freshly rehashed aggregates."""
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
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
        validation = json.loads((raw / "validation.json").read_text())
        work["raw_references"].extend(
            [
                calibration.reference(raw / "work.json"),
                calibration.reference(raw / "validation.json"),
            ]
        )
        evidence = d.authenticate(d.ROOT)
        result = d.empty_result() if evidence["failures"] else d.diagnose(evidence)
        if evidence != work["evidence"] or result != work["result"]:
            raise ValueError("measurement_drift")
        if (
            json.loads(Path(value["developer_view"]["path"]).read_text())["rows"]
            != result["developer_rows"]
            or json.loads(Path(value["protected_label_view"]["path"]).read_text())["rows"]
            != result["label_rows"]
            or Path(value["protected_label_view"]["path"]).stat().st_mode & 0o777 != 0o600
        ):
            raise ValueError("view_custody")
        passed = d.build(work, validation, value["fixture_only"]) == value
        return dict(passed=passed, replay_passed=passed)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False)


def main(argv: list[str] | None = None) -> int:
    """Private children exercise publication without recursive self-qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261109":
        parser.error("this producer requires --date 20261109")
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or d.ROOT / "results" / (d.NAME + ".json")
    if fixture and output.resolve().is_relative_to(d.ROOT):
        parser.error("fixture writes require private scratch")
    raw = output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    started = time.monotonic()
    work = freeze(d.ROOT, raw)
    validation: Json = dict(receipts=[], coverage={})
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot-8958-validation-"))
        d.b.progress("8958_before_owned_checks")
        validation["receipts"] = run_commands(
            d.ROOT,
            shared.validation_plan(private, producer=d),
            log_dir=raw / "validation",
            heartbeat_s=15,
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        d.b.progress("8958_after_owned_checks")
        work["global_test_debt"] = dict(
            receipts=run_commands(
                d.ROOT,
                [
                    CommandSpec(
                        "full_python_suite",
                        (str(d.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
                        "repository_health",
                        300,
                    )
                ],
                log_dir=raw / "repository_health",
                heartbeat_s=15,
            ),
            affects_owned_readiness=False,
            repeated=False,
        )
    work["duration_s"] = time.monotonic() - started
    work["phase_spans"].append(
        dict(
            phase="validation", duration_s=work["duration_s"] - work["phase_spans"][0]["duration_s"]
        )
    )
    atomic_json(raw / "validation.json", validation)
    atomic_json(raw / "work.json", work)
    work["raw_references"].extend(
        [
            calibration.reference(raw / "work.json"),
            calibration.reference(raw / "validation.json"),
        ]
    )
    value = d.build(work, validation, fixture)

    def validate(candidate: Path) -> Json:
        report = replay(candidate)
        if not fixture:
            commands = [
                replace(c, argv=tuple(a.replace(calibration.CLI, d.CLI) for a in c.argv))
                for c in calibration.terminal_commands(candidate)
            ]
            receipts = run_commands(
                d.ROOT, commands, log_dir=raw / "terminal_validation", heartbeat_s=15
            )
            report.update(
                passed=report["passed"] and admission.shard_a.pilot.terminal_passed(receipts),
                receipts=receipts,
            )
        return report

    d.b.progress("8958_before_primary_publication")
    published = publish_primary(output, value, validate)
    d.b.progress("8958_terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                diagnosis_ready_score=value["diagnosis_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
