"""REQ-REPORT-8952: preserve evidence and qualify checks before publishing decisions."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, UTC
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import matched_calibrated_decisions_8952 as d
from carnot.reporting import matched_corpus_admission_8951 as admission
from carnot.reporting import calibration_methods_8937 as calibration
from carnot.reporting import matched_corpus_a_8949 as shard
from carnot.reporting import matched_corpus_runner_8949 as shared
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]


def freeze(root: Path, raw: Path, fixture: bool) -> Json:
    """Retain original producers and execution sources before numerical fitting."""
    started, started_at = time.monotonic(), datetime.now(UTC).isoformat()
    d.b.progress("8952_before_authentication", 0, 320)
    branch = admission.matched.branch
    for name in [
        *(r[0] for r in d.INPUTS + admission.INPUTS),
        branch.DESIGN,
        branch.ACTIVE,
        branch.STAGED,
    ]:
        path = root / name
        if path.is_file():
            target = raw / "inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    evidence = d.authenticate(raw / "inputs")
    d.b.progress("8952_after_authentication", 0, len(evidence["failures"]))
    result, development = d.empty_result(), {}
    if not fixture and not evidence["failures"]:
        d.b.progress("8952_before_development_controls")
        dev = admission.development()
        development = {
            key: dev[key]
            for key in ("control_rows", "numerical_training_calls", "feature_access_controls")
        }
        d.b.progress("8952_after_development_controls", len(development["control_rows"]), 0)
        corpus = json.loads((raw / "inputs" / d.INPUTS[1][0]).read_text())["rows"]
        result = d.evaluate(corpus, raw)
    atomic_json(raw / "development.json", development)
    sources = []
    for name in [
        *d.OWNED,
        *d.TESTS,
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        "openspec/capabilities/research-reporting/spec.md",
        "ops/e2e-test-plan.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/models/gibbs/__init__.py",
        *[
            str(Path(module.__file__).relative_to(d.ROOT))
            for module in (admission, calibration, d.b, shared, branch)
        ],
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((d.ROOT / name).read_bytes())
        sources.append(
            dict(
                calibration.reference(target),
                source_path=str(d.ROOT / name),
                executed_source=name in d.OWNED,
            )
        )
    ended = time.monotonic()
    return dict(
        raw=str(raw),
        evidence=evidence,
        result=result,
        development=development,
        sources=sources,
        input_references=[
            calibration.reference(p) for p in sorted((raw / "inputs").rglob("*")) if p.is_file()
        ],
        started=started,
        ended=ended,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        phase_spans=[
            dict(
                phase="authentication_development_and_frozen_evaluation", duration_s=ended - started
            )
        ],
        global_test_debt={},
        template=ExperimentTemplate(
            d.EXPERIMENT_ID, "Matched calibrated decisions", f"results/{d.NAME}.json", seed=d.b.SEED
        ).build_result({}, status="success"),
    )


def replay(path: Path) -> Json:
    """Recompute numerical reductions and reject semantic drift even after rehashing."""
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
        if d.authenticate(raw / "inputs") != work["evidence"]:
            raise ValueError("authentication_drift")
        result = d.empty_result()
        development = {}
        if work["result"]["completed_count"]:
            corpus = json.loads((raw / "inputs" / d.INPUTS[1][0]).read_text())["rows"]
            result = d.evaluate(corpus, raw, replay=True)
            dev = admission.development()
            development = {key: dev[key] for key in work["development"]}
        if (
            result != work["result"]
            or development != work["development"]
            or json.loads((raw / "development.json").read_text()) != work["development"]
        ):
            raise ValueError("measurement_drift")
        validation = json.loads((raw / "validation.json").read_text())
        passed = d.build(work, validation, value["fixture_only"]) == value
        return dict(passed=passed, replay_passed=passed)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False)


def main(argv: list[str] | None = None) -> int:
    """Private CLI fixtures never claim live readiness or recursively run qualification."""
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
    work = freeze(d.ROOT, raw, fixture)
    validation: Json = dict(receipts=[], coverage={})
    validation_started = time.monotonic()
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot-8952-validation-"))
        d.b.progress("8952_before_owned_checks")
        validation["receipts"] = run_commands(
            d.ROOT,
            shared.validation_plan(private, producer=d),
            log_dir=raw / "validation",
            heartbeat_s=15,
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        d.b.progress("8952_after_owned_checks")
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
    work["ended"], work["finished_at"] = time.monotonic(), datetime.now(UTC).isoformat()
    work["phase_spans"].append(
        dict(phase="validation", duration_s=work["ended"] - validation_started)
    )
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
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
                passed=report["passed"] and shard.pilot.terminal_passed(receipts),
                receipts=receipts,
            )
        return report

    d.b.progress("8952_before_terminal_validation")
    published = publish_primary(output, value, validate)
    d.b.progress("8952_terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                detector_rows_ready_score=value["detector_rows_ready_score"],
                detector_benefit_score=value["detector_benefit_score"],
            )
        ),
        flush=True,
    )
    return 0
