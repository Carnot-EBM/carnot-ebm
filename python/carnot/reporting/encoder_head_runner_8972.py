"""REQ-REPORT-8972: qualify saved primitives before one checked atomic publication."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import encoder_head_diagnosis_8972 as d
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]


def freeze(root: Path, raw: Path) -> Json:
    """Retain inputs and code before numerical work; external absence preserves every slot."""
    started = time.monotonic()
    d.progress("start_authentication", 0, 5)
    resource_failures = d.resources(raw)
    evidence = d.authenticate(root, raw)
    evidence["failures"].extend(resource_failures)
    authenticated = time.monotonic()
    d.progress("after_authentication", len(evidence["producers"]), len(evidence["failures"]))
    sources = []
    for name in [
        *d.OWNED,
        *d.TESTS,
        "python/carnot/autoresearch/relational_graph_encoder.py",
        "python/carnot/autoresearch/graph_calibration_benchmark.py",
        "python/carnot/models/gibbs/__init__.py",
        "python/carnot/reporting/relational_decisions_8964.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/experiment_template.py",
        "docs/research-notes/v767-method-ingestion.md",
    ]:
        ref = d.retain(d.reference(d.ROOT / name), raw)
        sources.append(dict(ref, executed_source=name.endswith(".py")))
    result = d.empty_result() if evidence["failures"] else d.evaluate(evidence, raw)
    ended = time.monotonic()
    return dict(
        evidence=evidence,
        result=result,
        sources=sources,
        raw=str(raw),
        preconditions=dict(
            cpu_only=True,
            optional_hardware_required=False,
            historical_tasks_are_retained_inputs=True,
            resource_failures=resource_failures,
            checked_dependencies=["jax", "numpy", "sklearn", "coverage", "pytest", "ruff", "mypy"],
            writable_disk_checked=True,
            memory_checked=True,
        ),
        duration_s=ended - started,
        phase_spans=[
            dict(phase="authentication", duration_s=authenticated - started),
            dict(phase="numerical_measurement", duration_s=ended - authenticated),
        ],
        global_test_debt={},
        raw_references=[],
    )


def replay(path: Path) -> Json:
    """Recompute features, fitted heads, predictions and bounds; rehashing cannot repair drift."""
    d.progress("before_cold_replay")
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if d.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
            if (
                ref.get("executed_source")
                and d.reference(Path(ref["source_path"]))["sha256"] != ref["sha256"]
            ):
                raise ValueError("executed_bytes_drift")
        raw = Path(value["work_reference"]["path"]).parent
        work = json.loads((raw / "work.json").read_text())
        validation = json.loads((raw / "validation.json").read_text())
        evidence = d.authenticate(d.ROOT, raw)
        evidence["failures"].extend(work["preconditions"]["resource_failures"])
        result = (
            d.empty_result() if evidence["failures"] else d.evaluate(evidence, raw, replay=True)
        )
        if evidence != work["evidence"] or result != work["result"]:
            raise ValueError("prediction_or_reduction_drift")
        work["raw_references"] = value["raw_shard_hashes"]
        rebuilt = artifact(work, validation, value["fixture_only"])
        if rebuilt != value:
            raise ValueError("aggregate_drift")
        d.progress("after_cold_replay", 1, 0)
        return dict(
            passed=True,
            replay_passed=True,
            numerical_training_calls=result["numerical_training_calls"],
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        d.progress("cold_replay_refused", 0, 1)
        return dict(passed=False, replay_passed=False, reason=str(error))


def artifact(work: Json, validation: Json, fixture: bool) -> Json:
    """Bind the work ledger after it is sealed, avoiding a recursive self-hash."""
    value = d.build(work, validation, fixture)
    value["work_reference"] = d.reference(Path(work["raw"]) / "work.json")
    value["field_principles"]["work_reference"] = (
        "Bind the immutable primitive work ledger used by fresh replay."
    )
    value["reproducibility_checksum"] = d.checksum(value)
    return value


def main(argv: list[str] | None = None) -> int:
    """Private full CLI runs cannot self-qualify; production records each required check once."""
    d.progress("start")
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
    raw = output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    started = time.monotonic()
    work = freeze(d.ROOT, raw)
    validation: Json = dict(receipts=[], coverage={})
    if not fixture and not work["preconditions"]["resource_failures"]:
        from carnot.reporting.matched_corpus_runner_8949 import validation_plan
        from carnot.reporting.v756_branch_qualification import measured
        import carnot.reporting.encoder_head_diagnosis_8972 as producer

        private = Path(tempfile.mkdtemp(prefix="carnot-8972-checks-"))
        d.progress("before_owned_checks")
        validation["receipts"] = run_commands(
            d.ROOT,
            [
                replace(c, timeout_s=600) if c.name == "owned_unit_child" else c
                for c in validation_plan(private, producer=producer)
            ],
            log_dir=raw / "validation",
            heartbeat_s=15,
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        d.progress("after_owned_checks")
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
            phase="validation",
            duration_s=work["duration_s"] - sum(p["duration_s"] for p in work["phase_spans"]),
        )
    )
    atomic_json(raw / "validation.json", validation)
    atomic_json(raw / "work.json", work)
    work["raw_references"] = [
        d.reference(p) for p in sorted(raw.rglob("*.json")) if "inputs" not in p.parts
    ]
    value = artifact(work, validation, fixture)
    atomic_json(raw / "candidate.json.private", value)

    def validate(candidate: Path) -> Json:
        if fixture or work["preconditions"]["resource_failures"]:
            return replay(candidate)
        d.progress("before_terminal_validators")
        receipts = run_commands(
            d.ROOT,
            [
                CommandSpec(
                    "cold_replay",
                    (sys.executable, "-u", d.CLI, "--cold-replay", str(candidate)),
                    "private_terminal",
                    180,
                ),
                CommandSpec(
                    "adversarial",
                    (
                        sys.executable,
                        "-u",
                        "scripts/adversarial_verify.py",
                        "--json",
                        str(candidate),
                    ),
                    "private_terminal",
                    60,
                ),
                CommandSpec(
                    "strict_rows",
                    (
                        sys.executable,
                        "-u",
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ),
                    "private_terminal",
                    60,
                ),
            ],
            log_dir=raw / "terminal",
            heartbeat_s=15,
        )
        d.progress("after_terminal_validators")
        report = dict(passed=all(r["passed"] for r in receipts), receipts=receipts)
        atomic_json(raw / "terminal_report.json", report)
        return report

    d.progress("before_primary_publication")
    published = publish_primary(output, value, validate)
    d.progress("completed", value["completed_count"], value["censored_count"])
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
