"""REQ-REPORT-8951: qualify the owned admission checks before atomic publication."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, UTC
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import matched_corpus_admission_8951 as d
from carnot.reporting import calibration_methods_8937 as calibration
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import matched_source_methods_8946 as matched
from carnot.reporting import matched_corpus_a_8949 as shard_a
from carnot.reporting import matched_corpus_b_8950 as shard_b
from carnot.reporting import matched_corpus_runner_8949 as shards
from carnot.reporting import pilot_label_audit_8932 as audit
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]


def freeze(root: Path, raw: Path, fixture: bool) -> Json:
    """Retain exact branch, producer and code bytes before reading any corpus outcomes."""
    started, started_at = time.monotonic(), datetime.now(UTC).isoformat()
    source.progress("before_admission_authentication", 0, 320)
    for name in [
        *(r[0] for r in d.INPUTS),
        matched.branch.DESIGN,
        matched.branch.ACTIVE,
        matched.branch.STAGED,
    ]:
        path = root / name
        if path.is_file():
            target = raw / "inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    evidence = d.authenticate(raw / "inputs")
    source.progress("after_admission_authentication", 0, 320)
    if not fixture and not evidence["failures"]:
        source.progress("before_first_response_join", 0, 320)
        try:
            units, rows, old = d.join(raw / "inputs")
            result = d.reduce(units, rows, old)
        except matched.audit.Rejected as error:
            evidence["failures"].append(error.receipt)
            result = d.reduce([], [], [])
        except (OSError, ValueError, KeyError, TypeError) as error:
            evidence["failures"].append(
                dict(
                    upstream=d.TASK,
                    path=str(raw / "inputs"),
                    hash=None,
                    check="corpus_custody",
                    artifact_field="first_responses_and_frozen_roles",
                    expected="320 authenticated independent responses",
                    observed=str(error),
                    failure_kind="rejected_producer",
                )
            )
            result = d.reduce([], [], [])
        source.progress(
            "after_first_response_join", result["completed_count"], result["censored_count"]
        )
    else:
        result = d.reduce([], [], [])
    development = d.development() if not fixture and not evidence["failures"] else {}
    atomic_json(raw / "development.json", development)
    atomic_json(
        raw / "detector_view.json",
        dict(rows=[r["features"] for r in result["rows"] if r["status"] == "completed"]),
    )
    atomic_json(raw / "evaluation_labels.json", dict(rows=result["corpus_manifest"]))
    sources = []
    names = [
        *d.OWNED,
        d.TEST,
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        "openspec/capabilities/research-reporting/spec.md",
        "ops/e2e-test-plan.md",
        "scripts/experiment_template.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/v756_branch_qualification.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        *[
            str(Path(module.__file__).relative_to(d.ROOT))
            for module in (
                source,
                calibration,
                calibration.b,
                audit,
                matched,
                matched.branch,
                shard_a,
                shard_b,
                shards,
            )
        ],
    ]
    for name in names:
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
    source_path = raw / "inputs" / d.INPUTS[0][0]
    history_path = raw / "inputs" / matched.branch.HISTORY
    history = (
        json.loads(history_path.read_text()).get("outcome_rows", [])
        if history_path.is_file() and not evidence["failures"]
        else []
    )
    ended = time.monotonic()
    return dict(
        raw=str(raw),
        fixture_only=fixture,
        evidence=evidence,
        result=result,
        development=development,
        source_protocol_hash=json.loads(source_path.read_text()).get("protocol_hash")
        if source_path.is_file() and not evidence["failures"]
        else None,
        historical_scheduler_skips=[
            r for r in history if r["experiment_id"] in (8935, 8936, 8938, 8939)
        ],
        input_references=[
            calibration.reference(p) for p in sorted((raw / "inputs").rglob("*")) if p.is_file()
        ],
        sources=sources,
        template=ExperimentTemplate(
            d.EXPERIMENT_ID, "Matched corpus admission", f"results/{d.NAME}.json", seed=d.SEED
        ).build_result({}, status="success"),
        started=started,
        ended=ended,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        phase_spans=[dict(phase="authenticate_join_and_development", duration_s=ended - started)],
        global_test_debt={},
    )


def replay(path: Path) -> Json:
    """Rebuild the corpus and methods, refusing semantic tampering even after rehashing."""
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
            raise ValueError("input_authentication_drift")
        result = (
            d.reduce(*d.join(raw / "inputs"))
            if work["result"]["completed_count"]
            else d.reduce([], [], [])
        )
        if result != work["result"]:
            raise ValueError("corpus_reduction_drift")
        dev = d.development() if work["development"] else {}
        if dev != work["development"] or dev != json.loads((raw / "development.json").read_text()):
            raise ValueError("development_drift")
        for name, expected in (
            (
                "detector_view.json",
                [r["features"] for r in result["rows"] if r["status"] == "completed"],
            ),
            ("evaluation_labels.json", result["corpus_manifest"]),
        ):
            if json.loads((raw / name).read_text()) != dict(rows=expected):
                raise ValueError("sidecar_drift")
        validation = json.loads((raw / "validation.json").read_text())
        passed = d.build(work, validation, value["fixture_only"]) == value
        return dict(passed=passed, replay_passed=passed)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return dict(passed=False, replay_passed=False)


def main(argv: list[str] | None = None) -> int:
    """Private fixtures exercise the CLI without recursively qualifying or opening live rows."""
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
    raw = output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    work = freeze(d.ROOT, raw, fixture)
    validation: Json = dict(receipts=[], coverage={})
    validation_started = time.monotonic()
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot-8951-validation-"))
        source.progress("before_admission_owned_checks")
        validation["receipts"] = run_commands(
            d.ROOT,
            shards.validation_plan(private, producer=d),
            log_dir=raw / "validation",
            heartbeat_s=15,
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        source.progress("after_admission_owned_checks")
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
        dict(
            phase="owned_validation_and_repository_health",
            duration_s=work["ended"] - validation_started,
        )
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
                passed=report["passed"] and shard_a.pilot.terminal_passed(receipts),
                receipts=receipts,
            )
        return report

    source.progress("before_admission_terminal_validation")
    try:
        published = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        validation["terminal_rejected"] = True
        validation["receipts"].extend(receipts)
        atomic_json(raw / "validation.json", validation)
        value = d.build(work, validation, fixture)
        published = publish_primary(output, value, replay)
    source.progress(
        "admission_terminal_published", value["completed_count"], value["censored_count"]
    )
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
