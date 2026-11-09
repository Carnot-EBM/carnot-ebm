"""REQ-REPORT-8946: retain, qualify and atomically publish model-free methods."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, UTC
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import calibration_methods_8937 as calibration
from carnot.reporting import matched_source_methods_8946 as m
from carnot.reporting import v765_branch_contract as branch
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]


def validation_plan(private: Path) -> list[CommandSpec]:
    """Keep existing checks and assertions, substituting only the new owned paths."""
    commands = pilot.validation_plan(private)
    substitutions = dict(zip(pilot.OWNED + [pilot.TEST], m.OWNED + [m.TEST], strict=True))
    config = private / "coverage.ini"
    text = config.read_text()
    for before, after in substitutions.items():
        text = text.replace(before, after)
    config.write_text(text)
    return [
        replace(
            c,
            argv=tuple(substitutions.get(a, a) for a in c.argv),
            timeout_s=600 if c.name == "owned_unit_child" else c.timeout_s,
        )
        for c in commands
    ]


def freeze(root: Path, raw: Path) -> Json:
    """Snapshot exact producers and source bytes before graph construction or training."""
    started, started_at = time.monotonic(), datetime.now(UTC).isoformat()
    m.progress("before_input_authentication")
    for name in [*m.INPUTS, branch.DESIGN, branch.ACTIVE, branch.STAGED]:
        path = root / name
        if path.is_file():
            target = raw / "inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    evidence = m.authenticate(raw / "inputs")
    retained = []
    for i, path in enumerate(sorted({r["path"] for r in evidence["references"]})):
        original = Path(path)
        if original.is_file():
            target = raw / "retained" / str(i) / original.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original.read_bytes())
            retained.append(dict(m.reference(target), original_path=path))
    sources = []
    for name in m.OWNED + [
        m.TEST,
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        "openspec/capabilities/research-reporting/spec.md",
        branch.DESIGN,
        "ops/e2e-test-plan.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/graph_source_8931.py",
        "python/carnot/reporting/graph_source_methods_8931.py",
        "python/carnot/reporting/calibration_methods_8937.py",
        "python/carnot/autoresearch/graph_calibration_benchmark.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/inference/qwen_sufficiency_7920.py",
        "python/carnot/reporting/bfs_error_pilot_8918.py",
        "python/carnot/reporting/depth_opportunity_8945.py",
        "python/carnot/reporting/v765_branch_contract.py",
        "python/carnot/reporting/v756_branch_qualification.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((m.ROOT / name).read_bytes())
        sources.append(
            dict(
                m.reference(target), source_path=str(m.ROOT / name), executed_source=name in m.OWNED
            )
        )
    m.progress("after_input_authentication", len(retained), len(evidence["failures"]))
    m.progress("before_source_freeze", 0, 370)
    protocol = m.graph_protocol(evidence["old_graphs"]) if not evidence["failures"] else {}
    atomic_json(raw / "protocol.json", protocol)
    ended = time.monotonic()
    m.progress(
        "after_source_freeze",
        len(protocol.get("graph_manifest", [])),
        370 - len(protocol.get("graph_manifest", [])),
    )
    return dict(
        raw=str(raw),
        evidence=evidence,
        protocol=protocol,
        retained_inputs=retained,
        sources=sources,
        template=ExperimentTemplate(
            8946, "Matched graph source methods", f"results/{m.NAME}.json", seed=m.SEED
        ).build_result({}, status="success"),
        started=started,
        ended=ended,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        phase_spans=[dict(phase="authenticate_and_freeze", duration_s=ended - started)],
        global_test_debt={},
    )


def replay(path: Path) -> Json:
    """Rebuild readiness and graph decisions; changed aggregates fail even after rehashing."""
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            if m.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
        raw = Path(value["raw_shard_hashes"][0]["path"]).parent
        work = json.loads((raw / "work.json").read_text())
        if m.authenticate(raw / "inputs") != work["evidence"]:
            raise ValueError("input_authentication_drift")
        protocol = json.loads((raw / "protocol.json").read_text())
        if protocol != work["protocol"] or (
            protocol and not m.validate_protocol(protocol, work["evidence"]["old_graphs"])
        ):
            raise ValueError("protocol_drift")
        result = m.development(protocol, work["evidence"]["old_graphs"]) if protocol else {}
        if result != json.loads((raw / "development.json").read_text()):
            raise ValueError("development_drift")
        validation = json.loads((raw / "validation.json").read_text())
        passed = m.build(work, result, validation, value["fixture_only"]) == value
        return dict(passed=passed, replay_passed=passed)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False)


def main(argv: list[str] | None = None) -> int:
    """Private CLI cases exercise custody without recursively qualifying themselves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--input-root", type=Path, default=m.ROOT)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261109":
        parser.error("this producer requires --date 20261109")
    if args.fixture_e2e and args.fixture_e2e.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or m.ROOT / "results" / (m.NAME + ".json")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    work = freeze(args.input_root, raw)
    started = time.monotonic()
    result = (
        m.development(work["protocol"], work["evidence"]["old_graphs"]) if work["protocol"] else {}
    )
    atomic_json(raw / "development.json", result)
    m.progress("after_development", len(result.get("control_rows", [])), 0)
    validation: Json = dict(receipts=[], coverage={})
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot-8946-validation-"))
        m.progress("before_owned_checks")
        validation["receipts"] = run_commands(
            m.ROOT, validation_plan(private), log_dir=raw / "validation", heartbeat_s=15
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], m.OWNED, raw / "measured")
        m.progress("after_owned_checks")
        m.progress("before_full_python_suite")
        work["global_test_debt"] = dict(
            receipts=run_commands(
                m.ROOT,
                [
                    CommandSpec(
                        "full_python_suite",
                        (str(m.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
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
        m.progress("after_full_python_suite")
    work.update(ended=time.monotonic(), finished_at=datetime.now(UTC).isoformat())
    work["phase_spans"].append(
        dict(phase="development_and_validation", duration_s=work["ended"] - started)
    )
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    value = m.build(work, result, validation, fixture)
    receipts: list[Json] = []

    def validate(candidate: Path) -> Json:
        report = replay(candidate)
        if not fixture:
            commands = [
                replace(
                    c,
                    argv=tuple(
                        str(m.ROOT / m.CLI) if a == str(m.ROOT / calibration.CLI) else a
                        for a in c.argv
                    ),
                )
                for c in calibration.terminal_commands(candidate)
            ]
            receipts.extend(
                run_commands(m.ROOT, commands, log_dir=raw / "terminal_validation", heartbeat_s=15)
            )
            report["passed"] = report["passed"] and all(r["passed"] for r in receipts)
            report["receipts"] = receipts
        return report

    m.progress("before_terminal_publication")
    try:
        published = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        validation["receipts"].extend(receipts)
        atomic_json(raw / "validation.json", validation)
        published = publish_primary(output, m.build(work, result, validation, fixture), replay)
    m.progress("terminal_published", 1, 0)
    print(json.dumps(published), flush=True)
    return 0
