"""REQ-REPORT-8964: qualify owned work before publishing one reproducible result."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import relational_decisions_8964 as d
from carnot.reporting import corpus_admission_8963 as admission
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.depth_opportunity_8945 import Rejected
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.matched_corpus_runner_8949 import validation_plan
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured

Json = dict[str, Any]


def evaluation(evidence: Json, raw: Path, *, replay: bool = False) -> tuple[Json, bool]:
    """Keep external absence distinct from a failing owned inference control."""
    if evidence["failures"]:
        return d.empty_result(), False
    try:
        return d.evaluate(evidence, raw, replay=replay), False
    except (OSError, Rejected) as error:
        evidence["failures"].append(
            error.receipt
            if isinstance(error, Rejected)
            else dict(
                upstream=admission.TASK,
                path=str(raw / "inputs"),
                hash=None,
                check="inference_operand",
                artifact_field="retained_view",
                expected="authenticated readable operand",
                observed=str(error),
                failure_kind="missing_file",
            )
        )
        return d.empty_result(), False
    except (ValueError, KeyError, TypeError) as error:
        evidence["failures"].append(
            dict(
                upstream=d.TASK,
                path=str(raw),
                hash=None,
                check="owned_inference_control",
                artifact_field="frozen_inference_contract",
                expected="complete frozen states and allowed views",
                observed=str(error),
                failure_kind="failed_owned_control",
            )
        )
        return d.empty_result(), True


def freeze(root: Path, raw: Path, fixture: bool) -> Json:
    """Snapshot producer and code bytes before any calibration or held-out scoring."""
    started = time.monotonic()
    d.b.progress("8964_before_authentication", 0, 2)
    for name, _, _, _ in [] if fixture else d.INPUTS:
        path = root / name
        if path.is_file():
            target = raw / "inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    evidence = d.authenticate(raw / "inputs")
    authenticated = time.monotonic()
    d.b.progress("8964_after_authentication", 2, 0)
    result, owned_failure = evaluation(evidence, raw)
    sources = []
    for name in [
        *d.OWNED,
        *d.TESTS,
        "python/carnot/autoresearch/relational_graph_encoder.py",
        "python/carnot/autoresearch/graph_calibration_benchmark.py",
        "python/carnot/reporting/corpus_views_8963.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/relational_methods_8959.py",
        "python/carnot/memory/online_calibration_8960.py",
        "scripts/experiment_template.py",
        "openspec/capabilities/research-reporting/spec.md",
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((d.ROOT / name).read_bytes())
        sources.append(
            dict(
                d.calibration.reference(target),
                source_path=str(d.ROOT / name),
                executed_source=name.endswith(".py"),
            )
        )
    ended = time.monotonic()
    return dict(
        raw=str(raw),
        fixture_only=fixture,
        evidence=evidence,
        result=result,
        owned_failure=owned_failure,
        sources=sources,
        started=started,
        ended=ended,
        phase_spans=[
            dict(phase="authentication", duration_s=authenticated - started),
            dict(phase="calibration_seal_and_test_reduction", duration_s=ended - authenticated),
        ],
        raw_references=[],
        global_test_debt={},
        prior_terminal={},
    )


def replay(path: Path) -> Json:
    """Reconstruct predictions and every reduction; a new checksum cannot repair tampering."""
    d.b.progress("8964_before_cold_replay")
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if d.calibration.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
            if (
                ref.get("executed_source")
                and d.calibration.reference(Path(ref["source_path"]))["sha256"] != ref["sha256"]
            ):
                raise ValueError("executed_bytes_drift")
        raw = Path(value["raw_shard_hashes"][0]["path"]).parent
        work = json.loads((raw / "work.json").read_text())
        evidence = d.authenticate(raw / "inputs")
        result, owned_failure = evaluation(evidence, raw, replay=True)
        if (
            evidence != work["evidence"]
            or result != work["result"]
            or owned_failure != work["owned_failure"]
        ):
            raise ValueError("prediction_or_reduction_drift")
        validation = json.loads((raw / "validation.json").read_text())
        passed = d.build(work, validation, value["fixture_only"]) == value
        d.b.progress("8964_after_cold_replay", int(passed), int(not passed))
        return dict(passed=passed, replay_passed=passed)
    except (OSError, ValueError, KeyError, TypeError):
        d.b.progress("8964_after_cold_replay_refusal", 0, 1)
        return dict(passed=False, replay_passed=False)


def main(argv: list[str] | None = None) -> int:
    """Keep fixtures private, bound subprocesses and publish only through checked candidates."""
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
        private = Path(tempfile.mkdtemp(prefix="carnot-8964-validation-"))
        d.b.progress("8964_before_owned_checks")
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
        d.b.progress("8964_after_owned_checks")
        d.b.progress("8964_before_full_python_suite")
        global_receipts = run_commands(
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
        )
        work["global_test_debt"] = dict(
            receipts=global_receipts, affects_owned_readiness=False, repeated=False
        )
        d.b.progress("8964_after_full_python_suite")
    work["ended"] = time.monotonic()
    work["phase_spans"].append(
        dict(
            phase="owned_validation_and_global_health",
            duration_s=work["ended"] - validation_started,
        )
    )
    atomic_json(raw / "validation.json", validation)
    work["raw_references"] = [
        d.calibration.reference(raw / name)
        for name in (
            "validation.json",
            *(
                ["calibrated_states.json", "fixed_thresholds.json", "test_predictions.json"]
                if work["result"]["completed_count"]
                else []
            ),
        )
    ]
    atomic_json(raw / "work.json", work)
    value = d.build(work, validation, fixture)
    receipts: list[Json] = []

    def validate(candidate: Path) -> Json:
        report = replay(candidate)
        if not fixture:
            receipts.extend(
                run_commands(
                    d.ROOT,
                    [
                        replace(c, argv=tuple(a.replace(d.calibration.CLI, d.CLI) for a in c.argv))
                        for c in d.calibration.terminal_commands(candidate)
                    ],
                    log_dir=raw / "terminal_validation",
                    heartbeat_s=15,
                )
            )
            report.update(
                passed=report["passed"] and all(r["passed"] for r in receipts), receipts=receipts
            )
        return report

    d.b.progress("8964_before_primary_publication")
    try:
        published = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        validation["terminal_rejected"] = True
        validation["receipts"].extend(receipts)
        atomic_json(raw / "validation.json", validation)
        work["raw_references"] = [
            d.calibration.reference(Path(r["path"])) for r in work["raw_references"]
        ]
        atomic_json(raw / "work.json", work)
        value = d.build(work, validation, fixture)
        published = publish_primary(output, value, replay)
    d.b.progress("8964_terminal_published", value["completed_count"], value["censored_count"])
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
