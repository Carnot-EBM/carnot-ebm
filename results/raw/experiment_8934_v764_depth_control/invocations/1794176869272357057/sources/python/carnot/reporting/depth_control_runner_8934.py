"""REQ-REPORT-8934: validate owned work before publishing a terminal depth result."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.inference import depth_runtime_8934 as runtime
from carnot.reporting import depth_control_8934 as d
from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import graph_source_methods_8931 as methods
from carnot.reporting import graph_source_8931 as source
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured

Json = dict[str, Any]


def validation_plan(private: Path) -> list[CommandSpec]:
    """Measure only new files while keeping the unchanged private CLI acceptance checks."""
    commands = pilot.validation_plan(private)
    substitutions = {
        pilot.OWNED[0]: d.OWNED[:3],
        pilot.OWNED[1]: [],
        pilot.OWNED[2]: [d.CLI],
        pilot.TEST: [d.TEST],
    }
    (private / "coverage.ini").write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
        + "".join("    */" + name + "\n" for name in d.OWNED)
        + "[report]\nexclude_lines =\n"
    )
    return [
        replace(c, argv=tuple(value for arg in c.argv for value in substitutions.get(arg, [arg])))
        for c in commands
    ]


def freeze(root: Path, raw: Path, fixture: bool) -> Json:
    """Seal original inputs, execution order and source bytes before live generation."""
    started = time.monotonic()
    source.progress("before_input_authentication", 0, 100)
    inputs = raw / "inputs"
    for name in [
        *(row[0] for row in d.INPUTS),
        *methods.INPUT_PINS,
        methods.branch.DESIGN,
        methods.branch.ACTIVE,
        methods.branch.STAGED,
    ]:
        path = root / name
        if path.is_file():
            target = inputs / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    evidence = d.authenticate(inputs)
    # The frozen method roster still defines all slots when an external gate is absent.
    protocol = evidence["protocol"] or json.loads((d.ROOT / d.INPUTS[0][0]).read_text())
    units = d.ordered_units(protocol)
    atomic_json(raw / "scheduled_units.json", dict(units=units))
    source.progress("after_input_authentication", 0, 100)
    sources = []
    for name in [
        *d.OWNED,
        d.TEST,
        "python/carnot/reporting/graph_source_8931.py",
        "python/carnot/reporting/graph_source_methods_8931.py",
        "python/carnot/reporting/pilot_label_audit_runner_8932.py",
        "python/carnot/inference/bfs_pilot_runtime_8918.py",
        "python/carnot/inference/qwen_sufficiency_7920.py",
        "python/carnot/inference/llama_cpp_process.py",
        "python/carnot/gpu_lease_phase_journal.py",
        "python/carnot/verify/qwen_development_capture_7995.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/experiment_template.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "openspec/capabilities/research-reporting/spec.md",
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((d.ROOT / name).read_bytes())
        sources.append(
            dict(
                methods.reference(target),
                source_path=str(d.ROOT / name),
                executed_source=name in d.OWNED,
            )
        )
    auth_duration = time.monotonic() - started
    if fixture or evidence["failures"]:
        result = dict(
            rows=[
                source.censored(u, "private_fixture" if fixture else "external_inputs")
                for u in units
            ],
            counts=dict(ZERO_INVOCATION_COUNTS),
            checks=[],
            live=False,
            model_receipt={},
            resource_lease={},
            phase_spans=[],
            settings=dict(source.SETTINGS),
            owned_failure=False,
        )
        atomic_json(raw / "runtime" / "runtime.json", result)
    else:
        result = runtime.run(units, raw / "runtime")
    return dict(
        template=d.ExperimentTemplate(
            8934, "Paired graph depth control", f"results/{d.NAME}.json"
        ).build_result({}, status="success"),
        raw=str(raw),
        started=started,
        ended=time.monotonic(),
        units=units,
        runtime=result,
        authentication=evidence,
        sources=sources,
        input_references=[methods.reference(p) for p in sorted(inputs.rglob("*")) if p.is_file()],
        raw_references=[
            methods.reference(p) for p in sorted((raw / "runtime").rglob("*")) if p.is_file()
        ]
        + [methods.reference(raw / "scheduled_units.json")],
        phase_spans=[dict(phase="authenticate_and_snapshot", duration_s=auth_duration)],
        repository_health={},
    )


def replay(path: Path) -> Json:
    """Rebuild labels, signals and gates from authenticated raw bytes, not headline values."""
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            if methods.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
        raw = Path(value["raw_shard_hashes"][0]["path"]).parent
        work = json.loads((raw / "work.json").read_text())
        validation = json.loads((raw / "validation.json").read_text())
        if (
            d.authenticate(raw / "inputs") != work["authentication"]
            or d.ordered_units(work["authentication"]["protocol"]) != work["units"]
        ):
            raise ValueError("input_or_order_drift")
        for unit, row in zip(work["units"], work["runtime"]["rows"], strict=True):
            response_path = raw / "runtime" / "collection" / unit["unit_id"] / "response.json"
            if response_path.is_file():
                envelope = json.loads(response_path.read_text())
                rebuilt = source.label_response(unit, envelope["raw_response"])
                rebuilt.update(
                    duration_s=envelope["duration_s"],
                    actual_token_usage=envelope["raw_response"].get("usage"),
                )
                if rebuilt != row:
                    raise ValueError("response_label_drift")
        passed = d.build(work, validation, value["fixture_only"]) == value
        return dict(passed=passed, replay_passed=passed)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False)


def main(argv: list[str] | None = None) -> int:
    """Private CLI cases cannot load weights or recursively qualify their own fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261108":
        parser.error("this producer requires --date 20261108")
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or d.ROOT / "results" / (d.NAME + ".json")
    raw = output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    work = freeze(d.ROOT, raw, fixture)
    validation_started = time.monotonic()
    validation: Json = dict(receipts=[], coverage={})
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot-8934-validation-"))
        source.progress("before_owned_checks")
        validation["receipts"] = run_commands(
            d.ROOT, validation_plan(private), log_dir=raw / "validation", heartbeat_s=15
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        source.progress("after_owned_checks")
        work["repository_health"] = dict(
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
    work["ended"] = time.monotonic()
    work["phase_spans"].append(
        dict(
            phase="validation_and_repository_health", duration_s=work["ended"] - validation_started
        )
    )
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    value = d.build(work, validation, fixture)
    receipts: list[Json] = []

    def validate(candidate: Path) -> Json:
        report = replay(candidate)
        if not fixture:
            py = str(d.ROOT / ".venv/bin/python")
            commands = [
                CommandSpec(
                    "cold_replay_cli",
                    (
                        py,
                        "-u",
                        "-c",
                        "import os,subprocess,sys,tempfile; env=dict(os.environ); env.pop('PYTHONPATH',None); raise SystemExit(subprocess.run(sys.argv[1:],cwd=tempfile.gettempdir(),env=env,timeout=60).returncode)",
                        py,
                        "-u",
                        str(d.ROOT / d.CLI),
                        "--cold-replay",
                        str(candidate),
                    ),
                    "terminal",
                    75,
                ),
                CommandSpec(
                    "adversarial",
                    (py, "scripts/adversarial_verify.py", "--json", str(candidate)),
                    "terminal",
                    60,
                ),
                CommandSpec(
                    "strict_rows",
                    (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
                    "terminal",
                    60,
                ),
            ]
            receipts.extend(
                run_commands(d.ROOT, commands, log_dir=raw / "terminal_validation", heartbeat_s=15)
            )
            report.update(
                passed=report["passed"] and pilot.terminal_passed(receipts), receipts=receipts
            )
        return report

    source.progress("before_terminal_validation")
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
    source.progress("terminal_published", 100, 0)
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                depth_source_ready_score=value["depth_source_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
