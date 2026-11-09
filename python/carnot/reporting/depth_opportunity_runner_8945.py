"""REQ-REPORT-8945: validate the retained-data audit before atomic publication."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone, UTC
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import depth_opportunity_8945 as a
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]


def validation_plan(private: Path) -> list[CommandSpec]:
    """Reuse the scoped checks, measuring only this audit and its direct CLI children."""
    commands = pilot.validation_plan(private)
    substitutions = dict(zip(pilot.OWNED, a.OWNED, strict=True))
    substitutions[pilot.TEST] = a.TEST
    (private / "coverage.ini").write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
        + "".join("    */" + name + "\n" for name in a.OWNED)
        + "[report]\nexclude_lines =\n"
    )
    return [replace(c, argv=tuple(substitutions.get(arg, arg) for arg in c.argv)) for c in commands]


def freeze(root: Path, raw: Path) -> Json:
    """Snapshot producer and code bytes before checks so replay has stable inputs."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    a.progress("before_input_authentication", 0, 100)
    for name in a.INPUTS:
        path = root / name
        if path.is_file():
            target = raw / "inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    evidence = a.authenticate(raw / "inputs")
    sources = []
    for name in a.OWNED + [
        a.TEST,
        "python/carnot/reporting/graph_source_8931.py",
        "python/carnot/reporting/depth_control_8934.py",
        "python/carnot/reporting/bfs_error_pilot_8918.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "python/carnot/reporting/v756_branch_qualification.py",
        "scripts/experiment_template.py",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((a.ROOT / name).read_bytes())
        sources.append(
            dict(
                a.reference(target), source_path=str(a.ROOT / name), executed_source=name in a.OWNED
            )
        )
    ended = time.monotonic()
    a.progress(
        "after_input_authentication",
        100 if not evidence["failures"] else 0,
        0 if not evidence["failures"] else 100,
    )
    return dict(
        raw=str(raw),
        evidence=evidence,
        sources=sources,
        template=ExperimentTemplate(
            8945, "Depth answer opportunity audit", f"results/{a.NAME}.json"
        ).build_result({}, status="success"),
        started=started,
        ended=ended,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        phase_spans=[dict(phase="authenticate_and_snapshot", duration_s=ended - started)],
        global_test_debt={},
    )


def replay(path: Path) -> Json:
    """Recompute every row and readiness field; a rehashed aggregate is still rejected."""
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            if a.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
        raw = Path(value["raw_shard_hashes"][0]["path"]).parent
        work = json.loads((raw / "work.json").read_text())
        validation = json.loads((raw / "validation.json").read_text())
        if a.authenticate(raw / "inputs") != work["evidence"]:
            raise ValueError("input_authentication_drift")
        passed = a.build(work, validation, value["fixture_only"]) == value
        return dict(passed=passed, replay_passed=passed)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False)


def main(argv: list[str] | None = None) -> int:
    """Private fixtures exercise publication without recursively launching validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--input-root", type=Path, default=a.ROOT)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261109":
        parser.error("this producer requires --date 20261109")
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or a.ROOT / "results" / (a.NAME + ".json")
    raw = output.parent / "raw" / a.NAME / "invocations" / str(time.time_ns())
    work = freeze(args.input_root, raw)
    started = time.monotonic()
    validation: Json = dict(receipts=[], coverage={})
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot-8945-validation-"))
        a.progress("before_owned_checks")
        validation["receipts"] = run_commands(
            a.ROOT, validation_plan(private), log_dir=raw / "validation", heartbeat_s=15
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], a.OWNED, raw / "measured")
        a.progress("after_owned_checks")
        work["global_test_debt"] = dict(
            receipts=run_commands(
                a.ROOT,
                [
                    CommandSpec(
                        "full_python_suite",
                        (str(a.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
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
    work.update(ended=time.monotonic(), finished_at=datetime.now(UTC).isoformat())
    work["phase_spans"].append(
        dict(phase="validation_and_repository_health", duration_s=work["ended"] - started)
    )
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    value = a.build(work, validation, fixture)
    a.progress("before_terminal_publication")
    published = publish_primary(output, value, replay)
    a.progress("terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                diagnosis_ready_score=value["diagnosis_ready_score"],
                opportunity_confounded_score=value["opportunity_confounded_score"],
            )
        ),
        flush=True,
    )
    return 0
